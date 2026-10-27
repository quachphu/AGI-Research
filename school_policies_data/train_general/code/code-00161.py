        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        ShortTimeFFT: Class this property belongs to.
        """
        return self._fs

    @fs.setter
    def fs(self, v: float):
        """Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.
        """
        if not (v > 0):
            raise ValueError(f"Sampling frequency fs={v} must be positive!")
        self._fs = v

    @property
    def fft_mode(self) -> FFT_MODE_TYPE:
        """Mode of utilized FFT ('twosided', 'centered', 'onesided' or
        'onesided2X').

        It can have the following values:

        'twosided':
            Two-sided FFT, where values for the negative frequencies are in
            upper half of the array. Corresponds to :func:`~scipy.fft.fft()`.
        'centered':
            Two-sided FFT with the values being ordered along monotonically
            increasing frequencies. Corresponds to applying
            :func:`~scipy.fft.fftshift()` to :func:`~scipy.fft.fft()`.
        'onesided':
            Calculates only values for non-negative frequency values.
            Corresponds to :func:`~scipy.fft.rfft()`.
        'onesided2X':
            Like `onesided`, but the non-zero frequencies are doubled if
            `scaling` is set to 'magnitude' or multiplied by ``sqrt(2)`` if
            set to 'psd'. If `scaling` is ``None``, setting `fft_mode` to
            `onesided2X` is not allowed.
            If the FFT length `mfft` is even, the last FFT value is not paired,
            and thus it is not scaled.

        Note that `onesided` and `onesided2X` do not work for complex-valued signals or
        complex-valued windows. Furthermore, the frequency values can be obtained by
        reading the `f` property, and the number of samples by accessing the `f_pts`
        property.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        f_pts: Width of the frequency bins of the STFT.
        onesided_fft: True if a one-sided FFT is used.
        scaling: Normalization applied to the window function
        ShortTimeFFT: Class this property belongs to.
        """
        return self._fft_mode

    @fft_mode.setter
    def fft_mode(self, t: FFT_MODE_TYPE):
        """Set mode of FFT.

        Allowed values are 'twosided', 'centered', 'onesided', 'onesided2X'.
        See the property `fft_mode` for more details.
        """
        if t not in (fft_mode_types := get_args(FFT_MODE_TYPE)):
            raise ValueError(f"fft_mode='{t}' not in {fft_mode_types}!")

        if t in {'onesided', 'onesided2X'} and np.iscomplexobj(self.win):
            raise ValueError(f"One-sided spectra, i.e., fft_mode='{t}', " +
                             "are not allowed for complex-valued windows!")

        if t == 'onesided2X' and self.scaling is None:
            raise ValueError(f"For scaling is None, fft_mode='{t}' is invalid!"
                             "Do scale_to('psd') or scale_to('magnitude')!")
        self._fft_mode = t

    @property
    def mfft(self) -> int:
        """Length of input for the FFT used - may be larger than window
        length `m_num`.

        If not set, `mfft` defaults to the window length `m_num`.

        See Also
        --------
        f_pts: Number of points along the frequency axis.
        f: Frequencies values of the STFT.
        m_num: Number of samples in window `win`.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._mfft

    @mfft.setter
    def mfft(self, n_: int):
        """Setter for the length of FFT utilized.

        See the property `mfft` for further details.
        """
        if not (n_ >= self.m_num):
            raise ValueError(f"Attribute mfft={n_} needs to be at least the " +
                             f"window length m_num={self.m_num}!")
        self._mfft = n_

    @property
    def scaling(self) -> Literal['magnitude', 'psd'] | None:
        """Normalization applied to the window function
        ('magnitude', 'psd' or ``None``).

        If not ``None``, the FFTs can be either interpreted as a magnitude or
        a power spectral density spectrum.

        The window function can be scaled by calling the `scale_to` method,
        or it is set by the initializer parameter ``scale_to``.

        See Also
        --------
        fac_magnitude: Scaling factor for to a magnitude spectrum.
        fac_psd: Scaling factor for to  a power spectral density spectrum.
        fft_mode: Mode of utilized FFT
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._scaling

    def scale_to(self, scaling: Literal['magnitude', 'psd']):
        """Scale window to obtain 'magnitude' or 'psd' scaling for the STFT.

        The window of a 'magnitude' spectrum has an integral of one, i.e., unit
        area for non-negative windows. This ensures that absolute the values of
        spectrum does not change if the length of the window changes (given
        the input signal is stationary).

        To represent the power spectral density ('psd') for varying length
        windows the area of the absolute square of the window needs to be
        unity.

        The `scaling` property shows the current scaling. The properties
        `fac_magnitude` and `fac_psd` show the scaling factors required to
        scale the STFT values to a magnitude or a psd spectrum.

        This method is called, if the initializer parameter `scale_to` is set.

        See Also
        --------
        fac_magnitude: Scaling factor for to  a magnitude spectrum.
        fac_psd: Scaling factor for to  a power spectral density spectrum.
        fft_mode: Mode of utilized FFT
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this method belongs to.
        """
        if scaling not in (scaling_values := {'magnitude', 'psd'}):
            raise ValueError(f"{scaling=} not in {scaling_values}!")
        if self._scaling == scaling:  # do nothing
            return

        s_fac = self.fac_psd if scaling == 'psd' else self.fac_magnitude
        self._win = self._win * s_fac
        if self._dual_win is not None:
            self._dual_win = self._dual_win / s_fac
        self._fac_mag, self._fac_psd = None, None  # reset scaling factors
        self._scaling = scaling

    @property
    def phase_shift(self) -> int | None:
        """If set, add linear phase `phase_shift` / `mfft` * `f` to each FFT
        slice of frequency `f`.

        Shifting (more precisely `rolling`) an `mfft`-point FFT input by
        `phase_shift` samples results in a multiplication of the output by
        ``np.exp(2j*np.pi*q*phase_shift/mfft)`` at the frequency q * `delta_f`.

        The default value 0 ensures that there is no phase shift on the
        zeroth slice (in which t=0 is centered).
        No phase shift (``phase_shift is None``) is equivalent to
        ``phase_shift = -mfft//2``. In this case slices are not shifted
        before calculating the FFT.

        The absolute value of `phase_shift` is limited to be less than `mfft`.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        mfft: Length of input for the FFT used
        ShortTimeFFT: Class this property belongs to.
        """
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, v: int | None):
        """The absolute value of the phase shift needs to be less than mfft
        samples.

        See the `phase_shift` getter method for more details.
        """
        if v is None:
            self._phase_shift = v
            return
        if not isinstance(v, int):
            raise ValueError(f"phase_shift={v} has the unit samples. Hence " +
                             "it needs to be an int or it may be None!")
        if not (-self.mfft < v < self.mfft):
            raise ValueError("-mfft < phase_shift < mfft does not hold " +
                             f"for mfft={self.mfft}, phase_shift={v}!")
        self._phase_shift = v

    def _x_slices(self, x: np.ndarray, k_off: int, p0: int, p1: int,
                  padding: PAD_TYPE) -> Generator[np.ndarray, None, None]:
        """Generate signal slices along last axis of `x`.

        This method is only used by `stft_detrend`. The parameters are
        described in `~ShortTimeFFT.stft`.
        """
        if padding not in (padding_types := get_args(PAD_TYPE)):
            raise ValueError(f"Parameter {padding=} not in {padding_types}!")
        pad_kws: dict[str, dict] = {  # possible keywords to pass to np.pad:
            'zeros': dict(mode='constant', constant_values=(0, 0)),
            'edge': dict(mode='edge'),
            'even': dict(mode='reflect', reflect_type='even'),
            'odd': dict(mode='reflect', reflect_type='odd'),
           }  # typing of pad_kws is needed to make mypy happy

        n, n1 = x.shape[-1], (p1 - p0) * self.hop
        k0 = p0 * self.hop - self.m_num_mid + k_off  # start sample
        k1 = k0 + n1 + self.m_num  # end sample

        i0, i1 = max(k0, 0), min(k1, n)  # indexes to shorten x
        # dimensions for padding x:
        pad_width = [(0, 0)] * (x.ndim-1) + [(-min(k0, 0), max(k1 - n, 0))]

        x1 = np.pad(x[..., i0:i1], pad_width, **pad_kws[padding])
        for k_ in range(0, n1, self.hop):
            yield x1[..., k_:k_ + self.m_num]

    def stft(self, x: np.ndarray, p0: int | None = None,
             p1: int | None = None, *, k_offset: int = 0,
             padding: PAD_TYPE = 'zeros', axis: int = -1) \
            -> np.ndarray:
        """Perform the short-time Fourier transform.

        A two-dimensional matrix with ``p1-p0`` columns is calculated.
        The `f_pts` rows represent value at the frequencies `f`. The q-th
        column of the windowed FFT with the window `win` is centered at t[q].
        The columns represent the values at the frequencies `f`.

        Parameters
        ----------
        x
            The input signal as real or complex valued array. For complex values, the
            property `fft_mode` must be set to 'twosided' or 'centered'.
        p0
            The first element of the range of slices to calculate. If ``None``
            then it is set to :attr:`p_min`, which is the smallest possible
            slice.
        p1
            The end of the array. If ``None`` then `p_max(n)` is used.
        k_offset
            Index of first sample (t = 0) in `x`.
        padding
            Kind of values which are added, when the sliding window sticks out
            on either the lower or upper end of the input `x`. Zeros are added
            if the default 'zeros' is set. For 'edge' either the first or the
            last value of `x` is used. 'even' pads by reflecting the
            signal on the first or last sample and 'odd' additionally
            multiplies it with -1.
        axis
            The axis of `x` over which to compute the STFT.
            If not given, the last axis is used.

        Returns
        -------
        S
            A complex array is returned with the dimension always being larger
            by one than of `x`. The last axis always represent the time slices
            of the STFT. `axis` defines the frequency axis (default second to
            last). E.g., for a one-dimensional `x`, a complex 2d array is
            returned, with axis 0 representing frequency and axis 1 the time
            slices.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        delta_t: Time increment of STFT
        f: Frequencies values of the STFT.
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        p_range: Determine and validate slice index range.
        stft_detrend: STFT with detrended segments.
        t: Times of STFT for an input signal with `n` samples.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        return self.stft_detrend(x, None, p0, p1, k_offset=k_offset,
                                 padding=padding, axis=axis)

    def stft_detrend(self, x: np.ndarray,
                     detr: Callable[[np.ndarray], np.ndarray] | Literal['linear', 'constant'] | None,  # noqa: E501
                     p0: int | None = None, p1: int | None = None, *,
                     k_offset: int = 0, padding: PAD_TYPE = 'zeros',
                     axis: int = -1) \
            -> np.ndarray:
        """Short-time Fourier transform with a trend being subtracted from each
        segment beforehand.

        If `detr` is set to 'constant', the mean is subtracted, if set to
        "linear", the linear trend is removed. This is achieved by calling
        :func:`scipy.signal.detrend`. If `detr` is a function, `detr` is
        applied to each segment.
        All other parameters have the same meaning as in `~ShortTimeFFT.stft`.

        Note that due to the detrending, the original signal cannot be
        reconstructed by the `~ShortTimeFFT.istft`.

        See Also
        --------
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        :meth:`~ShortTimeFFT.stft`: Short-time Fourier transform
                                   (without detrending).
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        if self.onesided_fft and np.iscomplexobj(x):
            raise ValueError(f"Complex-valued `x` not allowed for {self.fft_mode=}'! "
                             "Set property `fft_mode` to 'twosided' or 'centered'.")
        if isinstance(detr, str):
            detr = partial(detrend, type=detr)
        elif not (detr is None or callable(detr)):
            raise ValueError(f"Parameter {detr=} is not a str, function or " +
                             "None!")
        n = x.shape[axis]
        if not (n >= (m2p := self.m_num-self.m_num_mid)):
            e_str = f'{len(x)=}' if x.ndim == 1 else f'of {axis=} of {x.shape}'
            raise ValueError(f"{e_str} must be >= ceil(m_num/2) = {m2p}!")

        if x.ndim > 1:  # motivated by the NumPy broadcasting mechanisms:
            x = np.moveaxis(x, axis, -1)
        # determine slice index range:
        p0, p1 = self.p_range(n, p0, p1)
        S_shape_1d = (self.f_pts, p1 - p0)
        S_shape = x.shape[:-1] + S_shape_1d if x.ndim > 1 else S_shape_1d
        S = np.zeros(S_shape, dtype=complex)
        for p_, x_ in enumerate(self._x_slices(x, k_offset, p0, p1, padding)):
            if detr is not None:
                x_ = detr(x_)
            S[..., :, p_] = self._fft_func(x_ * self.win.conj())
        if x.ndim > 1:
            return np.moveaxis(S, -2, axis if axis >= 0 else axis-1)
        return S

    def spectrogram(self, x: np.ndarray, y: np.ndarray | None = None,
                    detr: Callable[[np.ndarray], np.ndarray] | Literal['linear', 'constant'] | None = None,  # noqa: E501
                    *,
                    p0: int | None = None, p1: int | None = None,
                    k_offset: int = 0, padding: PAD_TYPE = 'zeros',
                    axis: int = -1) \
            -> np.ndarray:
        r"""Calculate spectrogram or cross-spectrogram.

        The spectrogram is the absolute square of the STFT, i.e., it is
        ``abs(S[q,p])**2`` for given ``S[q,p]``  and thus is always
        non-negative.
        For two STFTs ``Sx[q,p], Sy[q,p]``, the cross-spectrogram is defined
        as ``Sx[q,p] * np.conj(Sy[q,p])`` and is complex-valued.
        This is a convenience function for calling `~ShortTimeFFT.stft` /
        `stft_detrend`, hence all parameters are discussed there. If `y` is not
        ``None`` it needs to have the same shape as `x`.

        Examples
        --------
        The following example shows the spectrogram of a square wave with
        varying frequency :math:`f_i(t)` (marked by a green dashed line in the
        plot) sampled with 20 Hz:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.signal import square, ShortTimeFFT
        >>> from scipy.signal.windows import gaussian
        ...
        >>> T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
        >>> t_x = np.arange(N) * T_x  # time indexes for signal
        >>> f_i = 5e-3*(t_x - t_x[N // 3])**2 + 1  # varying frequency
        >>> x = square(2*np.pi*np.cumsum(f_i)*T_x)  # the signal

        The utilized Gaussian window is 50 samples or 2.5 s long. The
        parameter ``mfft=800`` (oversampling factor 16) and the `hop` interval
        of 2 in `ShortTimeFFT` was chosen to produce a sufficient number of
        points:

        >>> g_std = 12  # standard deviation for Gaussian window in samples
        >>> win = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian wind.
        >>> SFT = ShortTimeFFT(win, hop=2, fs=1/T_x, mfft=800, scale_to='psd')
        >>> Sx2 = SFT.spectrogram(x)  # calculate absolute square of STFT

        The plot's colormap is logarithmically scaled as the power spectral
        density is in dB. The time extent of the signal `x` is marked by
        vertical dashed lines and the shaded areas mark the presence of border
        effects:

        >>> fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
        >>> t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        >>> ax1.set_title(rf"Spectrogram ({SFT.m_num*SFT.T:g}$\,s$ Gaussian " +
        ...               rf"window, $\sigma_t={g_std*SFT.T:g}\,$s)")
        >>> ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
        ...                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ...         ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
        ...                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        ...         xlim=(t_lo, t_hi))
        >>> Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # limit range to -40 dB
        >>> im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',
        ...                  extent=SFT.extent(N), cmap='magma')
        >>> ax1.plot(t_x, f_i, 'g--', alpha=.5, label='$f_i(t)$')
        >>> fig1.colorbar(im1, label='Power Spectral Density ' +
        ...                          r"$20\,\log_{10}|S_x(t, f)|$ in dB")
        ...
        >>> # Shade areas where window slices stick out to the side:
        >>> for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
        ...                  (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
        ...     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)
        >>> for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
        ...     ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)
        >>> ax1.legend()
        >>> fig1.tight_layout()
        >>> plt.show()

        The logarithmic scaling reveals the odd harmonics of the square wave,
        which are reflected at the Nyquist frequency of 10 Hz. This aliasing
        is also the main source of the noise artifacts in the plot.


        See Also
        --------
        :meth:`~ShortTimeFFT.stft`: Perform the short-time Fourier transform.
        stft_detrend: STFT with a trend subtracted from each segment.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        Sx = self.stft_detrend(x, detr, p0, p1, k_offset=k_offset,
                               padding=padding, axis=axis)
        if y is None or y is x:  # do spectrogram:
            return Sx.real**2 + Sx.imag**2
        # Cross-spectrogram:
        Sy = self.stft_detrend(y, detr, p0, p1, k_offset=k_offset,
                               padding=padding, axis=axis)
        return Sx * Sy.conj()

    @property
    def dual_win(self) -> np.ndarray:
        """Canonical dual window.

        A STFT can be interpreted as the input signal being expressed as a
        weighted sum of modulated and time-shifted dual windows. Note that for
        a given window there exist many dual windows. The canonical window is
        the one with the minimal energy (i.e., :math:`L_2` norm).

        `dual_win` has same length as `win`, namely `m_num` samples.

        If the dual window cannot be calculated a ``ValueError`` is raised.
        This attribute is read only and calculated lazily.

        See Also
        --------
        dual_win: Canonical dual window.
        m_num: Number of samples in window `win`.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        if self._dual_win is None:
            self._dual_win = _calc_dual_canonical_window(self.win, self.hop)
        return self._dual_win

    @property
    def invertible(self) -> bool:
        """Check if STFT is invertible.

        This is achieved by trying to calculate the canonical dual window.

        See Also
        --------
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        m_num: Number of samples in window `win` and `dual_win`.
        dual_win: Canonical dual window.
        win: Window for STFT.
        ShortTimeFFT: Class this property belongs to.
        """
        try:
            return len(self.dual_win) > 0  # call self.dual_win()
        except ValueError:
            return False

    def istft(self, S: np.ndarray, k0: int = 0, k1: int | None = None, *,
              f_axis: int = -2, t_axis: int = -1) \
            -> np.ndarray:
        """Inverse short-time Fourier transform.

        It returns an array of dimension ``S.ndim - 1``  which is real
        if `onesided_fft` is set, else complex. If the STFT is not
        `invertible`, or the parameters are out of bounds  a ``ValueError`` is
        raised.

        Parameters
        ----------
        S
            A complex valued array where `f_axis` denotes the frequency
            values and the `t-axis` dimension the temporal values of the
            STFT values.
        k0, k1
            The start and the end index of the reconstructed signal. The
            default (``k0 = 0``, ``k1 = None``) assumes that the maximum length
            signal should be reconstructed.
        f_axis, t_axis
            The axes in `S` denoting the frequency and the time dimension.

        Notes
        -----
        It is required that `S` has `f_pts` entries along the `f_axis`. For
        the `t_axis` it is assumed that the first entry corresponds to
        `p_min` * `delta_t` (being <= 0). The length of `t_axis` needs to be
        compatible with `k1`. I.e., ``S.shape[t_axis] >= self.p_max(k1)`` must
        hold, if `k1` is not ``None``. Else `k1` is set to `k_max` with::

            q_max = S.shape[t_range] + self.p_min
            k_max = (q_max - 1) * self.hop + self.m_num - self.m_num_mid

        The :ref:`tutorial_stft` section of the :ref:`user_guide` discussed the
        slicing behavior by means of an example.

        See Also
        --------
        invertible: Check if STFT is invertible.
        :meth:`~ShortTimeFFT.stft`: Perform Short-time Fourier transform.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.
        """
        if f_axis == t_axis:
            raise ValueError(f"{f_axis=} may not be equal to {t_axis=}!")
        if S.shape[f_axis] != self.f_pts:
            raise ValueError(f"{S.shape[f_axis]=} must be equal to " +
                             f"{self.f_pts=} ({S.shape=})!")
        n_min = self.m_num-self.m_num_mid  # minimum signal length
        if not (S.shape[t_axis] >= (q_num := self.p_num(n_min))):
            raise ValueError(f"{S.shape[t_axis]=} needs to have at least " +
                             f"{q_num} slices ({S.shape=})!")
        if t_axis != S.ndim - 1 or f_axis != S.ndim - 2:
            t_axis = S.ndim + t_axis if t_axis < 0 else t_axis
            f_axis = S.ndim + f_axis if f_axis < 0 else f_axis
            S = np.moveaxis(S, (f_axis, t_axis), (-2, -1))

        q_max = S.shape[-1] + self.p_min
        k_max = (q_max - 1) * self.hop + self.m_num - self.m_num_mid

        k1 = k_max if k1 is None else k1
        if not (self.k_min <= k0 < k1 <= k_max):
            raise ValueError(f"({self.k_min=}) <= ({k0=}) < ({k1=}) <= " +
                             f"({k_max=}) is false!")
        if not (num_pts := k1 - k0) >= n_min:
            raise ValueError(f"({k1=}) - ({k0=}) = {num_pts} has to be at " +
                             f"least the half the window length {n_min}!")

        q0 = (k0 // self.hop + self.p_min if k0 >= 0 else  # p_min always <= 0
              k0 // self.hop)
        q1 = min(self.p_max(k1), q_max)
        k_q0, k_q1 = self.nearest_k_p(k0), self.nearest_k_p(k1, left=False)
        n_pts = k_q1 - k_q0 + self.m_num - self.m_num_mid
        x = np.zeros(S.shape[:-2] + (n_pts,),
                     dtype=float if self.onesided_fft else complex)
        for q_ in range(q0, q1):
            xs = self._ifft_func(S[..., :, q_ - self.p_min]) * self.dual_win
            i0 = q_ * self.hop - self.m_num_mid
            i1 = min(i0 + self.m_num, n_pts+k0)
            j0, j1 = 0, i1 - i0
            if i0 < k0:  # xs sticks out to the left on x:
                j0 += k0 - i0
                i0 = k0
            x[..., i0-k0:i1-k0] += xs[..., j0:j1]
        x = x[..., :k1-k0]
        if x.ndim > 1:
            x = np.moveaxis(x, -1, f_axis if f_axis < x.ndim else t_axis)
        return x

    @property
    def fac_magnitude(self) -> float:
        """Factor to multiply the STFT values by to scale each frequency slice
        to a magnitude spectrum.

        It is 1 if attribute ``scaling == 'magnitude'``.
        The window can be scaled to a magnitude spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_psd: Scaling factor for to a power spectral density spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
        if self.scaling == 'magnitude':
            return 1
        if self._fac_mag is None:
            self._fac_mag = 1 / abs(sum(self.win))
        return self._fac_mag

    @property
    def fac_psd(self) -> float:
        """Factor to multiply the STFT values by to scale each frequency slice
        to a power spectral density (PSD).

        It is 1 if attribute ``scaling == 'psd'``.
        The window can be scaled to a psd spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_magnitude: Scaling factor for to a magnitude spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
        if self.scaling == 'psd':
            return 1
        if self._fac_psd is None:
            self._fac_psd = 1 / np.sqrt(
                sum(self.win.real**2+self.win.imag**2) / self.T)
        return self._fac_psd

    @property
    def m_num(self) -> int:
        """Number of samples in window `win`.

        Note that the FFT can be oversampled by zero-padding. This is achieved
        by setting the `mfft` property.

        See Also
        --------
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: Time increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        return len(self.win)

    @property
    def m_num_mid(self) -> int:
        """Center index of window `win`.

        For odd `m_num`, ``(m_num - 1) / 2`` is returned and
        for even `m_num` (per definition) ``m_num / 2`` is returned.

        See Also
        --------
        m_num: Number of samples in window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: ime increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        return self.m_num // 2

    @cache
    def _pre_padding(self) -> tuple[int, int]:
        """Smallest signal index and slice index due to padding.

         Since, per convention, for time t=0, n,q is zero, the returned values
         are negative or zero.
         """
        w2 = self.win.real**2 + self.win.imag**2
        # move window to the left until the overlap with t >= 0 vanishes:
        n0 = -self.m_num_mid
        for q_, n_ in enumerate(range(n0, n0-self.m_num-1, -self.hop)):
            n_next = n_ - self.hop
            if n_next + self.m_num <= 0 or all(w2[n_next:] == 0):
                return n_, -q_
        raise RuntimeError("This is code line should not have been reached!")
        # If this case is reached, it probably means the first slice should be
        # returned, i.e.: return n0, 0

    @property
    def k_min(self) -> int:
        """The smallest possible signal index of the STFT.

        `k_min` is the index of the left-most non-zero value of the lowest
        slice `p_min`. Since the zeroth slice is centered over the zeroth
        sample of the input signal, `k_min` is never positive.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._pre_padding()[0]

    @property
    def p_min(self) -> int:
        """The smallest possible slice index.

        `p_min` is the index of the left-most slice, where the window still
        sticks into the signal, i.e., has non-zero part for t >= 0.
        `k_min` is the smallest index where the window function of the slice
        `p_min` is non-zero.

        Since, per convention the zeroth slice is centered at t=0,
        `p_min` <= 0 always holds.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._pre_padding()[1]

    @lru_cache(maxsize=256)
    def _post_padding(self, n: int) -> tuple[int, int]:
        """Largest signal index and slice index due to padding.

        Parameters
        ----------
        n : int
            Number of samples of input signal (must be ≥ half of the window length).
        """
        if not (n >= (m2p := self.m_num - self.m_num_mid)):
            raise ValueError(f"Parameter n must be >= ceil(m_num/2) = {m2p}!")
        w2 = self.win.real**2 + self.win.imag**2
        # move window to the right until the overlap for t < t[n] vanishes:
        q1 = n // self.hop   # last slice index with t[p1] <= t[n]
        k1 = q1 * self.hop - self.m_num_mid
        for q_, k_ in enumerate(range(k1, n+self.m_num, self.hop), start=q1):
            n_next = k_ + self.hop
            if n_next >= n or all(w2[:n-n_next] == 0):
                return k_ + self.m_num, q_ + 1
        raise RuntimeError("This is code line should not have been reached!")
        # If this case is reached, it probably means the last slice should be
        # returned, i.e.: return k1 + self.m_num - self.m_num_mid, q1 + 1

    def k_max(self, n: int) -> int:
        """First sample index after signal end not touched by a time slice.

        `k_max` - 1 is the largest sample index of the slice `p_max` for a
        given input signal of `n` samples.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        Parameters
        ----------
        n : int
            Number of samples of input signal (must be ≥ half of the window length).

        See Also
        --------
        k_min: The smallest possible signal index.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
        return self._post_padding(n)[0]

    def p_max(self, n: int) -> int:
        """Index of first non-overlapping upper time slice for `n` sample
        input.

        Note that center point t[p_max] = (p_max(n)-1) * `delta_t` is typically
        larger than last time index t[n-1] == (`n`-1) * `T`. The upper border
        of samples indexes covered by the window slices is given by `k_max`.
        Furthermore, `p_max` does not denote the number of slices `p_num` since
        `p_min` is typically less than zero.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        p_min: The smallest possible slice index.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
        return self._post_padding(n)[1]

    def p_num(self, n: int) -> int:
        """Number of time slices for an input signal with `n` samples.

        It is given by `p_num` = `p_max` - `p_min` with `p_min` typically
        being negative.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this method belongs to.
        """
        return self.p_max(n) - self.p_min

    @property
    def lower_border_end(self) -> tuple[int, int]:
        """First signal index and first slice index unaffected by pre-padding.

        Describes the point where the window does not stick out to the left
        of the signal domain.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
        # not using @cache decorator due to MyPy limitations
        if self._lower_border_end is not None:
            return self._lower_border_end

        # first non-zero element in self.win:
        m0 = np.flatnonzero(self.win.real**2 + self.win.imag**2)[0]

        # move window to the right until does not stick out to the left:
        k0 = -self.m_num_mid + m0
        for q_, k_ in enumerate(range(k0, self.hop + 1, self.hop)):
            if k_ + self.hop >= 0:  # next entry does not stick out anymore
                self._lower_border_end = (k_ + self.m_num, q_ + 1)
                return self._lower_border_end
        self._lower_border_end = (0, max(self.p_min, 0))  # ends at first slice
        return self._lower_border_end

    @lru_cache(maxsize=256)
    def upper_border_begin(self, n: int) -> tuple[int, int]:
        """First signal index and first slice index affected by post-padding.

        Describes the point where the window does begin stick out to the right
        of the signal domain.
        A detailed example is given :ref:`tutorial_stft_sliding_win` section
        of the :ref:`user_guide`.

        Parameters
        ----------
        n : int
            Number of samples of input signal (must be ≥ half of the window length).

        Returns
        -------
        k_ub : int
            Lowest signal index, where a touching time slice sticks out past the
            signal end.
        p_ub : int
            Lowest index of time slice of which the end sticks out past the signal end.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
        if not (n >= (m2p := self.m_num - self.m_num_mid)):
            raise ValueError(f"Parameter n must be >= ceil(m_num/2) = {m2p}!")
        w2 = self.win.real**2 + self.win.imag**2
        q2 = n // self.hop + 1  # first t[q] >= t[n]
        q1 = max((n-self.m_num) // self.hop - 1, -1)
        # move window left until does not stick out to the right:
        for q_ in range(q2, q1, -1):
            k_ = q_ * self.hop + (self.m_num - self.m_num_mid)
            if k_ <= n or all(w2[n-k_:] == 0):
                return (q_ + 1) * self.hop - self.m_num_mid, q_ + 1
        return 0, 0  # border starts at first slice

    @property
    def delta_t(self) -> float:
        """Time increment of STFT.

        The time increment `delta_t` = `T` * `hop` represents the sample
        increment `hop` converted to time based on the sampling interval `T`.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        hop: Hop size in signal samples for sliding window.
        t: Times of STFT for an input signal with `n` samples.
        T: Sampling interval of input signal and window `win`.
        ShortTimeFFT: Class this property belongs to
        """
        return self.T * self.hop

    def p_range(self, n: int, p0: int | None = None,
                p1: int | None = None) -> tuple[int, int]:
        """Determine and validate slice index range.

        Parameters
        ----------
        n : int
            Number of samples of input signal, assuming t[0] = 0.
        p0 : int | None
            First slice index. If 0 then the first slice is centered at t = 0.
            If ``None`` then `p_min` is used. Note that p0 may be < 0 if
            slices are left of t = 0.
        p1 : int | None
            End of interval (last value is p1-1).
            If ``None`` then `p_max(n)` is used.


        Returns
        -------
        p0_ : int
            The fist slice index
        p1_ : int
            End of interval (last value is p1-1).

        Notes
        -----
        A ``ValueError`` is raised if ``p_min <= p0 < p1 <= p_max(n)`` does not
        hold.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
        p_max = self.p_max(n)  # shorthand
        p0_ = self.p_min if p0 is None else p0
        p1_ = p_max if p1 is None else p1
        if not (self.p_min <= p0_ < p1_ <= p_max):
            raise ValueError(f"Invalid Parameter {p0=}, {p1=}, i.e., " +
                             f"{self.p_min=} <= p0 < p1 <= {p_max=} " +
                             f"does not hold for signal length {n=}!")
        return p0_, p1_

    @lru_cache(maxsize=1)
    def t(self, n: int, p0: int | None = None, p1: int | None = None,
          k_offset: int = 0) -> np.ndarray:
        """Times of STFT for an input signal with `n` samples.

        Returns a 1d array with times of the `~ShortTimeFFT.stft` values with
        the same  parametrization. Note that the slices are
        ``delta_t = hop * T`` time units apart.

         Parameters
        ----------
        n
            Number of sample of the input signal.
        p0
            The first element of the range of slices to calculate. If ``None``
            then it is set to :attr:`p_min`, which is the smallest possible
            slice.
        p1
            The end of the array. If ``None`` then `p_max(n)` is used.
        k_offset
            Index of first sample (t = 0) in `x`.


        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        nearest_k_p: Nearest sample index k_p for which t[k_p] == t[p] holds.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        ShortTimeFFT: Class this method belongs to.
        """
        p0, p1 = self.p_range(n, p0, p1)
        return np.arange(p0, p1) * self.delta_t + k_offset * self.T

    def nearest_k_p(self, k: int, left: bool = True) -> int:
        """Return nearest sample index k_p for which t[k_p] == t[p] holds.

        The nearest next smaller time sample p (where t[p] is the center
        position of the window of the p-th slice) is p_k = k // `hop`.
        If `hop` is a divisor of `k` than `k` is returned.
        If `left` is set than p_k * `hop` is returned else (p_k+1) * `hop`.

        This method can be used to slice an input signal into chunks for
        calculating the STFT and iSTFT incrementally.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        T: Sampling interval of input signal and of the window (``1/fs``).
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this method belongs to.
        """
        p_q, remainder = divmod(k, self.hop)
        if remainder == 0:
            return k
        return p_q * self.hop if left else (p_q + 1) * self.hop

    @property
    def delta_f(self) -> float:
        """Width of the frequency bins of the STFT.

        Return the frequency interval `delta_f` = 1 / (`mfft` * `T`).

        See Also
        --------
        delta_t: Time increment of STFT.
        f_pts: Number of points along the frequency axis.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        T: Sampling interval.
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this property belongs to.
        """
        return 1 / (self.mfft * self.T)

    @property
    def f_pts(self) -> int:
        """Number of points along the frequency axis.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        ShortTimeFFT: Class this property belongs to.
        """
        return self.mfft // 2 + 1 if self.onesided_fft else self.mfft

    @property
    def onesided_fft(self) -> bool:
        """Return True if a one-sided FFT is used.

        Returns ``True`` if `fft_mode` is either 'onesided' or 'onesided2X'.

        See Also
        --------
        fft_mode: Utilized FFT ('twosided', 'centered', 'onesided' or
                 'onesided2X')
        ShortTimeFFT: Class this property belongs to.
        """
        return self.fft_mode in {'onesided', 'onesided2X'}

    @property
    def f(self) -> np.ndarray:
        """Frequencies values of the STFT.

        A 1d array of length `f_pts` with `delta_f` spaced entries is returned.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        f_pts: Number of points along the frequency axis.
        mfft: Length of the input for FFT used.
        ShortTimeFFT: Class this property belongs to.
        """
        if self.fft_mode in {'onesided', 'onesided2X'}:
            return fft_lib.rfftfreq(self.mfft, self.T)
        elif self.fft_mode == 'twosided':
            return fft_lib.fftfreq(self.mfft, self.T)
        elif self.fft_mode == 'centered':
            return fft_lib.fftshift(fft_lib.fftfreq(self.mfft, self.T))
        # This should never happen but makes the Linters happy:
        fft_modes = get_args(FFT_MODE_TYPE)
        raise RuntimeError(f"{self.fft_mode=} not in {fft_modes}!")

    def _fft_func(self, x: np.ndarray) -> np.ndarray:
        """FFT based on the `fft_mode`, `mfft`, `scaling` and `phase_shift`
        attributes.

        For multidimensional arrays the transformation is carried out on the
        last axis.
        """
        if self.phase_shift is not None:
            if x.shape[-1] < self.mfft:  # zero pad if needed
                z_shape = list(x.shape)
                z_shape[-1] = self.mfft - x.shape[-1]
                x = np.hstack((x, np.zeros(z_shape, dtype=x.dtype)))
            p_s = (self.phase_shift + self.m_num_mid) % self.m_num
            x = np.roll(x, -p_s, axis=-1)

        if self.fft_mode == 'twosided':
            return fft_lib.fft(x, n=self.mfft, axis=-1)
        if self.fft_mode == 'centered':
            return fft_lib.fftshift(fft_lib.fft(x, self.mfft, axis=-1), axes=-1)
        if self.fft_mode == 'onesided':
            return fft_lib.rfft(x, n=self.mfft, axis=-1)
        if self.fft_mode == 'onesided2X':
            X = fft_lib.rfft(x, n=self.mfft, axis=-1)
            # Either squared magnitude (psd) or magnitude is doubled:
            fac = np.sqrt(2) if self.scaling == 'psd' else 2
            # For even input length, the last entry is unpaired:
            X[..., 1: -1 if self.mfft % 2 == 0 else None] *= fac
            return X
        # This should never happen but makes the Linter happy:
        fft_modes = get_args(FFT_MODE_TYPE)
        raise RuntimeError(f"{self.fft_mode=} not in {fft_modes}!")

    def _ifft_func(self, X: np.ndarray) -> np.ndarray:
        """Inverse to `_fft_func`.

        Returned is an array of length `m_num`. If the FFT is `onesided`
        then a float array is returned else a complex array is returned.
        For multidimensional arrays the transformation is carried out on the
        last axis.
        """
        if self.fft_mode == 'twosided':
            x = fft_lib.ifft(X, n=self.mfft, axis=-1)
        elif self.fft_mode == 'centered':
            x = fft_lib.ifft(fft_lib.ifftshift(X, axes=-1), n=self.mfft, axis=-1)
        elif self.fft_mode == 'onesided':
            x = fft_lib.irfft(X, n=self.mfft, axis=-1)
        elif self.fft_mode == 'onesided2X':
            Xc = X.copy()  # we do not want to modify function parameters
            fac = np.sqrt(2) if self.scaling == 'psd' else 2
            # For even length X the last value is not paired with a negative
            # value on the two-sided FFT:
            q1 = -1 if self.mfft % 2 == 0 else None
            Xc[..., 1:q1] /= fac
            x = fft_lib.irfft(Xc, n=self.mfft, axis=-1)
        else:  # This should never happen but makes the Linter happy:
            error_str = f"{self.fft_mode=} not in {get_args(FFT_MODE_TYPE)}!"
            raise RuntimeError(error_str)

        if self.phase_shift is None:
            return x[..., :self.m_num]
        p_s = (self.phase_shift + self.m_num_mid) % self.m_num
        return np.roll(x, p_s, axis=-1)[..., :self.m_num]

    def extent(self, n: int, axes_seq: Literal['tf', 'ft'] = 'tf',
               center_bins: bool = False) -> tuple[float, float, float, float]:
        """Return minimum and maximum values time-frequency values.

        A tuple with four floats  ``(t0, t1, f0, f1)`` for 'tf' and
        ``(f0, f1, t0, t1)`` for 'ft' is returned describing the corners
        of the time-frequency domain of the `~ShortTimeFFT.stft`.
        That tuple can be passed to `matplotlib.pyplot.imshow` as a parameter
        with the same name.

        Parameters
        ----------
        n : int
            Number of samples in input signal.
        axes_seq : {'tf', 'ft'}
            Return time extent first and then frequency extent or vice-versa.
        center_bins: bool
            If set (default ``False``), the values of the time slots and
            frequency bins are moved from the side the middle. This is useful,
            when plotting the `~ShortTimeFFT.stft` values as step functions,
            i.e., with no interpolation.

        See Also
        --------
        :func:`matplotlib.pyplot.imshow`: Display data as an image.
        :class:`scipy.signal.ShortTimeFFT`: Class this method belongs to.

        Examples
        --------
        The following two plots illustrate the effect of the parameter `center_bins`:
        The grid lines represent the three time and the four frequency values of the
        STFT.
        The left plot, where ``(t0, t1, f0, f1) = (0, 3, 0, 4)`` is passed as parameter
        ``extent`` to `~matplotlib.pyplot.imshow`, shows the standard behavior of the
        time and frequency values being at the lower edge of the corrsponding bin.
        The right plot, with ``(t0, t1, f0, f1) = (-0.5, 2.5, -0.5, 3.5)``, shows that
        the bins are centered over the respective values when passing
        ``center_bins=True``.

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.signal import ShortTimeFFT
        ...
        >>> n, m = 12, 6
        >>> SFT = ShortTimeFFT.from_window('hann', fs=m, nperseg=m, noverlap=0)
        >>> Sxx = SFT.stft(np.cos(np.arange(n)))  # produces a colorful plot
        ...
        >>> fig, axx = plt.subplots(1, 2, tight_layout=True, figsize=(6., 4.))
        >>> for ax_, center_bins in zip(axx, (False, True)):
        ...     ax_.imshow(abs(Sxx), origin='lower', interpolation=None, aspect='equal',
        ...                cmap='viridis', extent=SFT.extent(n, 'tf', center_bins))
        ...     ax_.set_title(f"{center_bins=}")
        ...     ax_.set_xlabel(f"Time ({SFT.p_num(n)} points, Δt={SFT.delta_t})")
        ...     ax_.set_ylabel(f"Frequency ({SFT.f_pts} points, Δf={SFT.delta_f})")
        ...     ax_.set_xticks(SFT.t(n))  # vertical grid line are timestamps
        ...     ax_.set_yticks(SFT.f)  # horizontal grid line are frequency values
        ...     ax_.grid(True)
        >>> plt.show()

        Note that the step-like behavior with the constant colors is caused by passing
        ``interpolation=None`` to `~matplotlib.pyplot.imshow`.
        """
        if axes_seq not in ('tf', 'ft'):
            raise ValueError(f"Parameter {axes_seq=} not in ['tf', 'ft']!")

        if self.onesided_fft:
            q0, q1 = 0, self.f_pts
        elif self.fft_mode == 'centered':
            q0 = -(self.mfft // 2)
            q1 = self.mfft // 2 if self.mfft % 2 == 0 else self.mfft // 2 + 1
        else:
            raise ValueError(f"Attribute fft_mode={self.fft_mode} must be " +
                             "in ['centered', 'onesided', 'onesided2X']")

        p0, p1 = self.p_min, self.p_max(n)  # shorthand
        if center_bins:
            t0, t1 = self.delta_t * (p0 - 0.5), self.delta_t * (p1 - 0.5)
            f0, f1 = self.delta_f * (q0 - 0.5), self.delta_f * (q1 - 0.5)
        else:
            t0, t1 = self.delta_t * p0, self.delta_t * p1
            f0, f1 = self.delta_f * q0, self.delta_f * q1
        return (t0, t1, f0, f1) if axes_seq == 'tf' else (f0, f1, t0, t1)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
    'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
    'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
    'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
    'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
    'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
    'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
    'sosfreqz', 'freqz_sos', 'iirnotch', 'iirpeak', 'bilinear_zpk',
    'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
    'gammatone', 'iircomb',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="filter_design",
                                   private_modules=["_filter_design"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
    'unit_impulse',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="waveforms",
                                   private_modules=["_waveforms"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Filter design."""
import math
import operator
import warnings

import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, sqrt, tan, log10,
                   arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, full, array,
                   mintypecode)
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots

from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.signal._arraytools import _validate_fs


__all__ = ['findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
           'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
           'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
           'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
           'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
           'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
           'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
           'sosfreqz', 'freqz_sos', 'iirnotch', 'iirpeak', 'bilinear_zpk',
           'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
           'gammatone', 'iircomb']


class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""
    pass


abs = absolute


def _is_int_type(x):
    """
    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will
    pass, while ``5.0`` and ``array([5])`` will fail.
    """
    if np.ndim(x) != 0:
        # Older versions of NumPy did not raise for np.array([1]).__index__()
        # This is safe to remove when support for those versions is dropped
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True


def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e., zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)

    >>> from scipy import signal
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    if kind == 'ba':
        ep = atleast_1d(roots(den)) + 0j
        tz = atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = atleast_1d(den) + 0j
        tz = atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")

    if len(ep) == 0:
        ep = atleast_1d(-1000) + 0j

    ez = np.r_[ep[ep.imag >= 0], tz[(np.abs(tz) < 1e5) & (tz.imag >= 0)]]

    integ = np.abs(ez) < 1e-10
    hfreq = np.round(np.log10(np.max(3 * np.abs(ez.real + integ) +
                                     1.5 * ez.imag)) + 0.5)
    lfreq = np.round(np.log10(0.1 * np.min(np.abs((ez + integ).real) +
                                           2 * ez.imag)) - 0.5)

    w = np.logspace(lfreq, hfreq, N)
    return w


def freqs(b, a, worN=200, plot=None):
    """
    Compute frequency response of analog filter.

    Given the M-order numerator `b` and N-order denominator `a` of an analog
    filter, compute its frequency response::

             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
     H(w) = ----------------------------------------------
             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqz : Compute the frequency response of a digital filter.

    Notes
    -----
    Using Matplotlib's "plot" function as the callable for `plot` produces
    unexpected results, this plots the real part of the complex transfer
    function, not the magnitude. Try ``lambda w, h: plot(w, abs(h))``.

    Examples
    --------
    >>> from scipy.signal import freqs, iirfilter
    >>> import numpy as np

    >>> b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1')

    >>> w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid(True)
    >>> plt.show()

    """
    if worN is None:
        # For backwards compatibility
        w = findfreqs(b, a, 200)
    elif _is_int_type(worN):
        w = findfreqs(b, a, worN)
    else:
        w = atleast_1d(worN)

    s = 1j * w
    h = polyval(b, s) / polyval(a, s)
    if plot is not None:
        plot(w, h)

    return w, h


def freqs_zpk(z, p, k, worN=200):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqz : Compute the frequency response of a digital filter in TF form
    freqz_zpk : Compute the frequency response of a digital filter in ZPK form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import freqs_zpk, iirfilter

    >>> z, p, k = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1',
    ...                     output='zpk')

    >>> w, h = freqs_zpk(z, p, k, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid(True)
    >>> plt.show()

    """
    k = np.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')

    if worN is None:
        # For backwards compatibility
        w = findfreqs(z, p, 200, kind='zp')
    elif _is_int_type(worN):
        w = findfreqs(z, p, worN, kind='zp')
    else:
        w = worN

    w = atleast_1d(w)
    s = 1j * w
    num = polyvalfromroots(s, z)
    den = polyvalfromroots(s, p)
    h = k * num/den
    return w, h


def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2*pi,
          include_nyquist=False):
    """
    Compute the frequency response of a digital filter.

    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::

                 jw                 -jw              -jwM
        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
     H(e  ) = ------ = -----------------------------------
                 jw                 -jw              -jwN
              A(e  )    a[0] + a[1]e    + ... + a[N]e

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    a : array_like
        Denominator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512). This is a convenient alternative to::

            np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)

        Using a number that is fast for FFT computations can result in
        faster computations (see Notes).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if worN is array_like.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0
    include_nyquist : bool, optional
        If `whole` is False and `worN` is an integer, setting `include_nyquist`
        to True will include the last frequency (Nyquist frequency) and is
        otherwise ignored.

        .. versionadded:: 1.5.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz_zpk
    freqz_sos

    Notes
    -----
    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable
    for `plot` produces unexpected results, as this plots the real part of the
    complex transfer function, not the magnitude.
    Try ``lambda w, h: plot(w, np.abs(h))``.

    A direct computation via (R)FFT is used to compute the frequency response
    when the following conditions are met:

    1. An integer value is given for `worN`.
    2. `worN` is fast to compute via FFT (i.e.,
       `next_fast_len(worN) <scipy.fft.next_fast_len>` equals `worN`).
    3. The denominator coefficients are a single value (``a.shape[0] == 1``).
    4. `worN` is at least as long as the numerator coefficients
       (``worN >= b.shape[0]``).
    5. If ``b.ndim > 1``, then ``b.shape[-1] == 1``.

    For long FIR filters, the FFT approach can have lower error and be much
    faster than the equivalent direct polynomial calculation.

    Examples
    --------
    >>> from scipy import signal
    >>> import numpy as np
    >>> taps, f_c = 80, 1.0  # number of taps and cut-off frequency
    >>> b = signal.firwin(taps, f_c, window=('kaiser', 8), fs=2*np.pi)
    >>> w, h = signal.freqz(b)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax1 = plt.subplots(tight_layout=True)
    >>> ax1.set_title(f"Frequency Response of {taps} tap FIR Filter" +
    ...               f"($f_c={f_c}$ rad/sample)")
    >>> ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
    >>> ax1.set_ylabel("Amplitude in dB", color='C0')
    >>> ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi))

    >>> ax2 = ax1.twinx()
    >>> phase = np.unwrap(np.angle(h))
    >>> ax2.plot(w, phase, 'C1')
    >>> ax2.set_ylabel('Phase [rad]', color='C1')
    >>> ax2.grid(True)
    >>> ax2.axis('tight')
    >>> plt.show()

    Broadcasting Examples

    Suppose we have two FIR filters whose coefficients are stored in the
    rows of an array with shape (2, 25). For this demonstration, we'll
    use random data:

    >>> rng = np.random.default_rng()
    >>> b = rng.random((2, 25))

    To compute the frequency response for these two filters with one call
    to `freqz`, we must pass in ``b.T``, because `freqz` expects the first
    axis to hold the coefficients. We must then extend the shape with a
    trivial dimension of length 1 to allow broadcasting with the array
    of frequencies.  That is, we pass in ``b.T[..., np.newaxis]``, which has
    shape (25, 2, 1):

    >>> w, h = signal.freqz(b.T[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)

    Now, suppose we have two transfer functions, with the same numerator
    coefficients ``b = [0.5, 0.5]``. The coefficients for the two denominators
    are stored in the first dimension of the 2-D array  `a`::

        a = [   1      1  ]
            [ -0.25, -0.5 ]

    >>> b = np.array([0.5, 0.5])
    >>> a = np.array([[1, 1], [-0.25, -0.5]])

    Only `a` is more than 1-D. To make it compatible for
    broadcasting with the frequencies, we extend it with a trivial dimension
    in the call to `freqz`:

    >>> w, h = signal.freqz(b, a[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)

    """
    b = atleast_1d(b)
    a = atleast_1d(a)

    fs = _validate_fs(fs, allow_none=False)

    if worN is None:
        # For backwards compatibility
        worN = 512

    h = None

    if _is_int_type(worN):
        N = operator.index(worN)
        del worN
        if N < 0:
            raise ValueError(f'worN must be nonnegative, got {N}')
        lastpoint = 2 * pi if whole else pi
        # if include_nyquist is true and whole is false, w should
        # include end point
        w = np.linspace(0, lastpoint, N,
                        endpoint=include_nyquist and not whole)
        n_fft = N if whole else 2 * (N - 1) if include_nyquist else 2 * N
        if (a.size == 1 and (b.ndim == 1 or (b.shape[-1] == 1))
                and n_fft >= b.shape[0]
                and n_fft > 0):  # TODO: review threshold acc. to benchmark?
            if np.isrealobj(b) and np.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft
            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a
            if fft_func is sp_fft.rfft and whole:
                # exclude DC and maybe Nyquist (no need to use axis_reverse
                # here because we can build reversal with the truncation)
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = np.concatenate((h, h[h_flip].conj()))
            if b.ndim > 1:
                # Last axis of h has length 1, so drop it.
                h = h[..., 0]
                # Move the first axis of h to the end.
                h = np.moveaxis(h, 0, -1)
    else:
        w = atleast_1d(worN)
        del worN
        w = 2*pi*w/fs

    if h is None:  # still need to compute using freqs w
        zm1 = exp(-1j * w)
        h = (npp_polyval(zm1, b, tensor=False) /
             npp_polyval(zm1, a, tensor=False))

    w = w*(fs/(2*pi))

    if plot is not None:
        plot(w, h)

    return w, h


def freqz_zpk(z, p, k, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response:

    :math:`H(z)=k \prod_i (z - Z[i]) / \prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 4th-order digital Butterworth filter with cut-off of 100 Hz in a
    system with sample rate of 1000 Hz, and plot the frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> z, p, k = signal.butter(4, 100, output='zpk', fs=1000)
    >>> w, h = signal.freqz_zpk(z, p, k, fs=1000)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(1, 1, 1)
    >>> ax1.set_title('Digital filter frequency response')

    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [Hz]')
    >>> ax1.grid(True)

    >>> ax2 = ax1.twinx()
    >>> phase = np.unwrap(np.angle(h))
    >>> ax2.plot(w, phase, 'g')
    >>> ax2.set_ylabel('Phase [rad]', color='g')

    >>> plt.axis('tight')
    >>> plt.show()

    """
    z, p = map(atleast_1d, (z, p))

    fs = _validate_fs(fs, allow_none=False)

    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi

    if worN is None:
        # For backwards compatibility
        w = np.linspace(0, lastpoint, 512, endpoint=False)
    elif _is_int_type(worN):
        w = np.linspace(0, lastpoint, worN, endpoint=False)
    else:
        w = atleast_1d(worN)
        w = 2*pi*w/fs

    zm1 = exp(1j * w)
    h = k * polyvalfromroots(zm1, z) / polyvalfromroots(zm1, p)

    w = w*(fs/(2*pi))

    return w, h


def group_delay(system, w=512, whole=False, fs=2*pi):
    r"""Compute the group delay of a digital filter.

    The group delay measures by how many samples amplitude envelopes of
    various spectral components of a signal are delayed by a filter.
    It is formally defined as the derivative of continuous (unwrapped) phase::

               d        jw
     D(w) = - -- arg H(e)
              dw

    Parameters
    ----------
    system : tuple of array_like (b, a)
        Numerator and denominator coefficients of a filter transfer function.
    w : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the delay at the frequencies given. These
        are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which group delay was computed, in the same units
        as `fs`.  By default, `w` is normalized to the range [0, pi)
        (radians/sample).
    gd : ndarray
        The group delay.

    See Also
    --------
    freqz : Frequency response of a digital filter

    Notes
    -----
    The similar function in MATLAB is called `grpdelay`.

    If the transfer function :math:`H(z)` has zeros or poles on the unit
    circle, the group delay at corresponding frequencies is undefined.
    When such a case arises the warning is raised and the group delay
    is set to 0 at those frequencies.

    For the details of numerical computation of the group delay refer to [1]_ or [2]_.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Richard G. Lyons, "Understanding Digital Signal Processing,
           3rd edition", p. 830.
    .. [2] Julius O. Smith III, "Numerical Computation of Group Delay",
           in "Introduction to Digital Filters with Audio Applications",
           online book, 2007,
           https://ccrma.stanford.edu/~jos/fp/Numerical_Computation_Group_Delay.html

    Examples
    --------
    >>> from scipy import signal
    >>> b, a = signal.iirdesign(0.1, 0.3, 5, 50, ftype='cheby1')
    >>> w, gd = signal.group_delay((b, a))

    >>> import matplotlib.pyplot as plt
    >>> plt.title('Digital filter group delay')
    >>> plt.plot(w, gd)
    >>> plt.ylabel('Group delay [samples]')
    >>> plt.xlabel('Frequency [rad/sample]')
    >>> plt.show()

    """
    if w is None:
        # For backwards compatibility
        w = 512

    fs = _validate_fs(fs, allow_none=False)

    if _is_int_type(w):
        if whole:
            w = np.linspace(0, 2 * pi, w, endpoint=False)
        else:
            w = np.linspace(0, pi, w, endpoint=False)
    else:
        w = np.atleast_1d(w)
        w = 2*pi*w/fs

    b, a = map(np.atleast_1d, system)
    c = np.convolve(b, conjugate(a[::-1]))
    cr = c * np.arange(c.size)
    z = np.exp(-1j * w)
    num = np.polyval(cr[::-1], z)
    den = np.polyval(c[::-1], z)
    gd = np.real(num / den) - a.size + 1
    singular = ~np.isfinite(gd)
    near_singular = np.absolute(den) < 10 * EPSILON

    if np.any(singular):
        gd[singular] = 0
        warnings.warn(
            "The group delay is singular at frequencies "
            f"[{', '.join(f'{ws:.3f}' for ws in w[singular])}], setting to 0",
            stacklevel=2
        )

    elif np.any(near_singular):
        warnings.warn(
            "The filter's denominator is extremely small at frequencies "
            f"[{', '.join(f'{ws:.3f}' for ws in w[near_singular])}], "
            "around which a singularity may be present",
            stacklevel=2
        )

    w = w*(fs/(2*pi))

    return w, gd


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


def freqz_sos(sos, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in SOS format.

    Given `sos`, an array with shape (n, 6) of second order sections of
    a digital filter, compute the frequency response of the system function::

               B0(z)   B1(z)         B{n-1}(z)
        H(z) = ----- * ----- * ... * ---------
               A0(z)   A1(z)         A{n-1}(z)

    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and
    denominator of the transfer function of the k-th second order section.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).  Using a number that is fast for FFT computations can result
        in faster computations (see Notes of `freqz`).

        If an array_like, compute the response at the frequencies given (must
        be 1-D). These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz, sosfilt

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 15th-order bandpass filter in SOS format.

    >>> from scipy import signal
    >>> import numpy as np
    >>> sos = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='sos')

    Compute the frequency response at 1500 points from DC to Nyquist.

    >>> w, h = signal.freqz_sos(sos, worN=1500)

    Plot the response.

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()

    If the same filter is implemented as a single transfer function,
    numerical error corrupts the frequency response:

    >>> b, a = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='ba')
    >>> w, h = signal.freqz(b, a, worN=1500)
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=False)

    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.
    for row in sos:
        w, rowh = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        h *= rowh
    return w, h


def sosfreqz(*args, **kwargs):
    """
    Compute the frequency response of a digital filter in SOS format.

    .. warning:: This function is an alias, provided for backward
                 compatibility. New code should use the function
                 :func:`scipy.signal.freqz_sos`.
    """
    return freqz_sos(*args, **kwargs)


def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (`zc`) and real (`zr`)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxpair

    Examples
    --------
    >>> from scipy.signal._filter_design import _cplxreal
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    z = atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Get tolerance from dtype of input
        tol = 100 * np.finfo((1.0 * z).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[np.lexsort((abs(z.imag), z.real))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr


def _cplxpair(z, tol=None):
    """
    Sort into pairs of complex conjugates.

    Complex conjugates in `z` are sorted by increasing real part. In each
    pair, the number with negative imaginary part appears first.

    If pairs have identical real parts, they are sorted by increasing
    imaginary magnitude.

    Two complex numbers are considered a conjugate pair if their real and
    imaginary parts differ in magnitude by less than ``tol * abs(z)``.  The
    pairs are forced to be exact complex conjugates by averaging the positive
    and negative values.

    Purely real numbers are also sorted, but placed after the complex
    conjugate pairs. A number is considered real if its imaginary part is
    smaller than `tol` times the magnitude of the number.

    Parameters
    ----------
    z : array_like
        1-D input array to be sorted.
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    y : ndarray
        Complex conjugate pairs followed by real numbers.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxreal

    Examples
    --------
    >>> from scipy.signal._filter_design import _cplxpair
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> z = _cplxpair(a)
    >>> print(z)
    [ 1.-1.j  1.+1.j  2.-1.j  2.+1.j  2.-1.j  2.+1.j  2.-2.j  2.+2.j  1.+0.j
      3.+0.j  4.+0.j]
    """

    z = atleast_1d(z)
    if z.size == 0 or np.isrealobj(z):
        return np.sort(z)

    if z.ndim != 1:
        raise ValueError('z must be 1-D')

    zc, zr = _cplxreal(z, tol)

    # Interleave complex values and their conjugates, with negative imaginary
    # parts first in each pair
    zc = np.dstack((zc.conj(), zc)).flatten()
    z = np.append(zc, zr)
    return z


def tf2zpk(b, a):
    r"""Return zero, pole, gain (z, p, k) representation from a numerator,
    denominator representation of a linear filter.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    If some values of `b` are too close to 0, they are removed. In that case,
    a BadCoefficients warning is emitted.

    The `b` and `a` arrays are interpreted as coefficients for positive,
    descending powers of the transfer function variable. So the inputs
    :math:`b = [b_0, b_1, ..., b_M]` and :math:`a =[a_0, a_1, ..., a_N]`
    can represent an analog filter of the form:

    .. math::

        H(s) = \frac
        {b_0 s^M + b_1 s^{(M-1)} + \cdots + b_M}
        {a_0 s^N + a_1 s^{(N-1)} + \cdots + a_N}

    or a discrete-time filter of the form:

    .. math::

        H(z) = \frac
        {b_0 z^M + b_1 z^{(M-1)} + \cdots + b_M}
        {a_0 z^N + a_1 z^{(N-1)} + \cdots + a_N}

    This "positive powers" form is found more commonly in controls
    engineering. If `M` and `N` are equal (which is true for all filters
    generated by the bilinear transform), then this happens to be equivalent
    to the "negative powers" discrete-time form preferred in DSP:

    .. math::

        H(z) = \frac
        {b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}
        {a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}}

    Although this is true for common filters, remember that this is not true
    in the general case. If `M` and `N` are not equal, the discrete-time
    transfer function coefficients must first be converted to the "positive
    powers" form before finding the poles and zeros.

    Examples
    --------
    Find the zeroes, poles and gain of
    a filter with the transfer function

    .. math::

        H(s) = \frac{3s^2}{s^2 + 5s + 13}

    >>> from scipy.signal import tf2zpk
    >>> tf2zpk([3, 0, 0], [1, 5, 13])
    (   array([ 0.               ,  0.              ]),
        array([ -2.5+2.59807621j ,  -2.5-2.59807621j]),
        3.0)
    """
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0]
    b /= b[0]
    z = roots(b)
    p = roots(a)
    return z, p, k


def zpk2tf(z, p, k):
    r"""
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    Examples
    --------
    Find the polynomial representation of a transfer function H(s)
    using its 'zpk' (Zero-Pole-Gain) representation.

    .. math::

        H(z) = 5 \frac
        { (s - 2)(s - 6) }
        { (s - 1)(s - 8) }

    >>> from scipy.signal import zpk2tf
    >>> z   = [2,   6]
    >>> p   = [1,   8]
    >>> k   = 5
    >>> zpk2tf(z, p, k)
    (   array([  5., -40.,  60.]), array([ 1., -9.,  8.]))
    """
    z = atleast_1d(z)
    k = atleast_1d(k)
    if len(z.shape) > 1:
        temp = poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * poly(z)
    a = atleast_1d(poly(p))

    # Use real output if possible. Copied from np.poly, since
    # we can't depend on a specific version of numpy.
    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a


def tf2sos(b, a, pairing=None, *, analog=False):
    r"""
    Return second-order sections from transfer function representation

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See `zpk2sos` for information and restrictions on `pairing` and
        `analog` arguments.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

        .. versionadded:: 1.8.0

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    zpk2sos, sosfilt

    Notes
    -----
    It is generally discouraged to convert from TF to SOS format, since doing
    so usually will not improve numerical precision errors. Instead, consider
    designing filters in ZPK format and converting directly to SOS. TF is
    converted to SOS by first converting to ZPK format, then converting
    ZPK to SOS.

    .. versionadded:: 0.16.0

    Examples
    --------
    Find the 'sos' (second-order sections) of the transfer function H(s)
    using its polynomial representation.

    .. math::

        H(s) = \frac{s^2 - 3.5s - 2}{s^4 + 3s^3 - 15s^2 - 19s + 30}

    >>> from scipy.signal import tf2sos
    >>> tf2sos([1, -3.5, -2], [1, 3, -15, -19, 30], analog=True)
    array([[  0. ,   0. ,   1. ,   1. ,   2. , -15. ],
           [  1. ,  -3.5,  -2. ,   1. ,   1. ,  -2. ]])
    """
    return zpk2sos(*tf2zpk(b, a), pairing=pairing, analog=analog)


def sos2tf(sos):
    r"""
    Return a single transfer function from a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    Notes
    -----
    .. versionadded:: 0.16.0

    Examples
    --------
    Find the polynomial representation of an elliptic filter
    using its 'sos' (second-order sections) format.

    >>> from scipy.signal import sos2tf
    >>> from scipy import signal
    >>> sos = signal.ellip(1, 0.001, 50, 0.1, output='sos')
    >>> sos2tf(sos)
    (   array([0.91256522, 0.91256522, 0.        ]),
        array([1.        , 0.82513043, 0.        ]))
    """
    sos = np.asarray(sos)
    result_type = sos.dtype
    if result_type.kind in 'bui':
        result_type = np.float64

    b = np.array([1], dtype=result_type)
    a = np.array([1], dtype=result_type)
    n_sections = sos.shape[0]
    for section in range(n_sections):
        b = np.polymul(b, sos[section, :3])
        a = np.polymul(a, sos[section, 3:])
    return b, a


def sos2zpk(sos):
    """
    Return zeros, poles, and gain of a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    The number of zeros and poles returned will be ``n_sections * 2``
    even if some of these are (effectively) zero.

    .. versionadded:: 0.16.0
    """
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    z = np.zeros(n_sections*2, np.complex128)
    p = np.zeros(n_sections*2, np.complex128)
    k = 1.
    for section in range(n_sections):
        zpk = tf2zpk(sos[section, :3], sos[section, 3:])
        z[2*section:2*section+len(zpk[0])] = zpk[0]
        p[2*section:2*section+len(zpk[1])] = zpk[1]
        k *= zpk[2]
    return z, p, k


def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex', 'any')
    order = np.argsort(np.abs(fro - to))
    if which == 'any':
        return order[0]
    else:
        mask = np.isreal(fro[order])
        if which == 'complex':
            mask = ~mask
        return order[np.nonzero(mask)[0][0]]


def _single_zpksos(z, p, k):
    """Create one second-order section from up to two zeros and poles"""
    sos = np.zeros(6)
    b, a = zpk2tf(z, p, k)
    sos[3-len(b):3] = b
    sos[6-len(a):6] = a
    return sos


def zpk2sos(z, p, k, pairing=None, *, analog=False):
    """Return second-order sections from zeros, poles, and gain of a system

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        If analog is False and pairing is None, pairing is set to 'nearest';
        if analog is True, pairing must be 'minimal', and is set to that if
        it is None.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

        .. versionadded:: 1.8.0

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt

    Notes
    -----
    The algorithm used to convert ZPK to SOS format is designed to
    minimize errors due to numerical precision issues. The pairing
    algorithm attempts to minimize the peak gain of each biquadratic
    section. This is done by pairing poles with the nearest zeros, starting
    with the poles closest to the unit circle for discrete-time systems, and
    poles closest to the imaginary axis for continuous-time systems.

    ``pairing='minimal'`` outputs may not be suitable for `sosfilt`,
    and ``analog=True`` outputs will never be suitable for `sosfilt`.

    *Algorithms*

    The steps in the ``pairing='nearest'``, ``pairing='keep_odd'``,
    and ``pairing='minimal'`` algorithms are mostly shared. The
    ``'nearest'`` algorithm attempts to minimize the peak gain, while
    ``'keep_odd'`` minimizes peak gain under the constraint that
    odd-order systems should retain one section as first order.
    ``'minimal'`` is similar to ``'keep_odd'``, but no additional
    poles or zeros are introduced

    The algorithm steps are as follows:

    As a pre-processing step for ``pairing='nearest'``,
    ``pairing='keep_odd'``, add poles or zeros to the origin as
    necessary to obtain the same number of poles and zeros for
    pairing.  If ``pairing == 'nearest'`` and there are an odd number
    of poles, add an additional pole and a zero at the origin.

    The following steps are then iterated over until no more poles or
    zeros remain:

    1. Take the (next remaining) pole (complex or real) closest to the
       unit circle (or imaginary axis, for ``analog=True``) to
       begin a new filter section.

    2. If the pole is real and there are no other remaining real poles [#]_,
       add the closest real zero to the section and leave it as a first
       order section. Note that after this step we are guaranteed to be
       left with an even number of real poles, complex poles, real zeros,
       and complex zeros for subsequent pairing iterations.

    3. Else:

        1. If the pole is complex and the zero is the only remaining real
           zero*, then pair the pole with the *next* closest zero
           (guaranteed to be complex). This is necessary to ensure that
           there will be a real zero remaining to eventually create a
           first-order section (thus keeping the odd order).

        2. Else pair the pole with the closest remaining zero (complex or
           real).

        3. Proceed to complete the second-order section by adding another
           pole and zero to the current pole and zero in the section:

            1. If the current pole and zero are both complex, add their
               conjugates.

            2. Else if the pole is complex and the zero is real, add the
               conjugate pole and the next closest real zero.

            3. Else if the pole is real and the zero is complex, add the
               conjugate zero and the real pole closest to those zeros.

            4. Else (we must have a real pole and real zero) add the next
               real pole closest to the unit circle, and then add the real
               zero closest to that pole.

    .. [#] This conditional can only be met for specific odd-order inputs
           with the ``pairing = 'keep_odd'`` or ``'minimal'`` methods.

    .. versionadded:: 0.16.0

    Examples
    --------

    Design a 6th order low-pass elliptic digital filter for a system with a
    sampling rate of 8000 Hz that has a pass-band corner frequency of
    1000 Hz. The ripple in the pass-band should not exceed 0.087 dB, and
    the attenuation in the stop-band should be at least 90 dB.

    In the following call to `ellip`, we could use ``output='sos'``,
    but for this example, we'll use ``output='zpk'``, and then convert
    to SOS format with `zpk2sos`:

    >>> from scipy import signal
    >>> import numpy as np
    >>> z, p, k = signal.ellip(6, 0.087, 90, 1000/(0.5*8000), output='zpk')

    Now convert to SOS format.

    >>> sos = signal.zpk2sos(z, p, k)

    The coefficients of the numerators of the sections:

    >>> sos[:, :3]
    array([[0.0014152 , 0.00248677, 0.0014152 ],
           [1.        , 0.72976874, 1.        ],
           [1.        , 0.17607852, 1.        ]])

    The symmetry in the coefficients occurs because all the zeros are on the
    unit circle.

    The coefficients of the denominators of the sections:

    >>> sos[:, 3:]
    array([[ 1.        , -1.32544025,  0.46989976],
           [ 1.        , -1.26118294,  0.62625924],
           [ 1.        , -1.2570723 ,  0.8619958 ]])

    The next example shows the effect of the `pairing` option.  We have a
    system with three poles and three zeros, so the SOS array will have
    shape (2, 6). The means there is, in effect, an extra pole and an extra
    zero at the origin in the SOS representation.

    >>> z1 = np.array([-1, -0.5-0.5j, -0.5+0.5j])
    >>> p1 = np.array([0.75, 0.8+0.1j, 0.8-0.1j])

    With ``pairing='nearest'`` (the default), we obtain

    >>> signal.zpk2sos(z1, p1, 1)
    array([[ 1.  ,  1.  ,  0.5 ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.  ,  1.  , -1.6 ,  0.65]])

    The first section has the zeros {-0.5-0.05j, -0.5+0.5j} and the poles
    {0, 0.75}, and the second section has the zeros {-1, 0} and poles
    {0.8+0.1j, 0.8-0.1j}. Note that the extra pole and zero at the origin
    have been assigned to different sections.

    With ``pairing='keep_odd'``, we obtain:

    >>> signal.zpk2sos(z1, p1, 1, pairing='keep_odd')
    array([[ 1.  ,  1.  ,  0.  ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.5 ,  1.  , -1.6 ,  0.65]])

    The extra pole and zero at the origin are in the same section.
    The first section is, in effect, a first-order section.

    With ``pairing='minimal'``, the first-order section doesn't have
    the extra pole and zero at the origin:

    >>> signal.zpk2sos(z1, p1, 1, pairing='minimal')
    array([[ 0.  ,  1.  ,  1.  ,  0.  ,  1.  , -0.75],
           [ 1.  ,  1.  ,  0.5 ,  1.  , -1.6 ,  0.65]])

    """
    # TODO in the near future:
    # 1. Add SOS capability to `filtfilt`, `freqz`, etc. somehow (#3259).
    # 2. Make `decimate` use `sosfilt` instead of `lfilter`.
    # 3. Make sosfilt automatically simplify sections to first order
    #    when possible. Note this might make `sosfiltfilt` a bit harder (ICs).
    # 4. Further optimizations of the section ordering / pole-zero pairing.
    # See the wiki for other potential issues.

    if pairing is None:
        pairing = 'minimal' if analog else 'nearest'

    valid_pairings = ['nearest', 'keep_odd', 'minimal']
    if pairing not in valid_pairings:
        raise ValueError(f'pairing must be one of {valid_pairings}, not {pairing}')

    if analog and pairing != 'minimal':
        raise ValueError('for analog zpk2sos conversion, '
                         'pairing must be "minimal"')

    if len(z) == len(p) == 0:
        if not analog:
            return np.array([[k, 0., 0., 1., 0., 0.]])
        else:
            return np.array([[0., 0., k, 0., 0., 1.]])

    if pairing != 'minimal':
        # ensure we have the same number of poles and zeros, and make copies
        p = np.concatenate((p, np.zeros(max(len(z) - len(p), 0))))
        z = np.concatenate((z, np.zeros(max(len(p) - len(z), 0))))
        n_sections = (max(len(p), len(z)) + 1) // 2

        if len(p) % 2 == 1 and pairing == 'nearest':
            p = np.concatenate((p, [0.]))
            z = np.concatenate((z, [0.]))
        assert len(p) == len(z)
    else:
        if len(p) < len(z):
            raise ValueError('for analog zpk2sos conversion, '
                             'must have len(p)>=len(z)')

        n_sections = (len(p) + 1) // 2

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))
    if not np.isreal(k):
        raise ValueError('k must be real')
    k = k.real

    if not analog:
        # digital: "worst" is the closest to the unit circle
        def idx_worst(p):
            return np.argmin(np.abs(1 - np.abs(p)))
    else:
        # analog: "worst" is the closest to the imaginary axis
        def idx_worst(p):
            return np.argmin(np.abs(np.real(p)))

    sos = np.zeros((n_sections, 6))

    # Construct the system, reversing order so the "worst" are last
    for si in range(n_sections-1, -1, -1):
        # Select the next "worst" pole
        p1_idx = idx_worst(p)
        p1 = p[p1_idx]
        p = np.delete(p, p1_idx)

        # Pair that pole with a zero

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            # Special case (1): last remaining real pole
            if pairing != 'minimal':
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)
                sos[si] = _single_zpksos([z1, 0], [p1, 0], 1)
            elif len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)
                sos[si] = _single_zpksos([z1], [p1], 1)
            else:
                sos[si] = _single_zpksos([], [p1], 1)

        elif (len(p) + 1 == len(z)
              and not np.isreal(p1)
              and np.isreal(p).sum() == 1
              and np.isreal(z).sum() == 1):

            # Special case (2): there's one real pole and one real zero
            # left, and an equal number of poles and zeros to pair up.
            # We *must* pair with a complex zero

            z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)
            sos[si] = _single_zpksos([z1, z1.conj()], [p1, p1.conj()], 1)

        else:
            if np.isreal(p1):
                prealidx = np.flatnonzero(np.isreal(p))
                p2_idx = prealidx[idx_worst(p[prealidx])]
                p2 = p[p2_idx]
                p = np.delete(p, p2_idx)
            else:
                p2 = p1.conj()

            # find closest zero
            if len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'any')
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)

                if not np.isreal(z1):
                    sos[si] = _single_zpksos([z1, z1.conj()], [p1, p2], 1)
                else:
                    if len(z) > 0:
                        z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                        z2 = z[z2_idx]
                        assert np.isreal(z2)
                        z = np.delete(z, z2_idx)
                        sos[si] = _single_zpksos([z1, z2], [p1, p2], 1)
                    else:
                        sos[si] = _single_zpksos([z1], [p1, p2], 1)
            else:
                # no more zeros
                sos[si] = _single_zpksos([], [p1, p2], 1)

    assert len(p) == len(z) == 0  # we've consumed all poles and zeros
    del p, z

    # put gain in first sos
    sos[0][:3] *= k
    return sos


def _align_nums(nums):
    """Aligns the shapes of multiple numerators.

    Given an array of numerator coefficient arrays [[a_1, a_2,...,
    a_n],..., [b_1, b_2,..., b_m]], this function pads shorter numerator
    arrays with zero's so that all numerators have the same length. Such
    alignment is necessary for functions like 'tf2ss', which needs the
    alignment when dealing with SIMO transfer functions.

    Parameters
    ----------
    nums: array_like
        Numerator or list of numerators. Not necessarily with same length.

    Returns
    -------
    nums: array
        The numerator. If `nums` input was a list of numerators then a 2-D
        array with padded zeros for shorter numerators is returned. Otherwise
        returns ``np.asarray(nums)``.
    """
    try:
        # The statement can throw a ValueError if one
        # of the numerators is a single digit and another
        # is array-like e.g. if nums = [5, [1, 2, 3]]
        nums = asarray(nums)

        if not np.issubdtype(nums.dtype, np.number):
            raise ValueError("dtype of numerator is non-numeric")

        return nums

    except ValueError:
        nums = [np.atleast_1d(num) for num in nums]
        max_width = max(num.size for num in nums)

        # pre-allocate
        aligned_nums = np.zeros((len(nums), max_width))

        # Create numerators with padded zeros
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size:] = num

        return aligned_nums


def normalize(b, a):
    """Normalize numerator/denominator of a continuous-time transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    Parameters
    ----------
    b: array_like
        Numerator of the transfer function. Can be a 2-D array to normalize
        multiple transfer functions.
    a: array_like
        Denominator of the transfer function. At most 1-D.

    Returns
    -------
    num: array
        The numerator of the normalized transfer function. At least a 1-D
        array. A 2-D array if the input `num` is a 2-D array.
    den: 1-D array
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).

    Examples
    --------
    >>> from scipy.signal import normalize

    Normalize the coefficients of the transfer function
    ``(3*s^2 - 2*s + 5) / (2*s^2 + 3*s + 1)``:

    >>> b = [3, -2, 5]
    >>> a = [2, 3, 1]
    >>> normalize(b, a)
    (array([ 1.5, -1. ,  2.5]), array([1. , 1.5, 0.5]))

    A warning is generated if, for example, the first coefficient of
    `b` is 0.  In the following example, the result is as expected:

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     num, den = normalize([0, 3, 6], [2, -5, 4])

    >>> num
    array([1.5, 3. ])
    >>> den
    array([ 1. , -2.5,  2. ])

    >>> print(w[0].message)
    Badly conditioned filter coefficients (numerator): the results may be meaningless

    """
    num, den = b, a

    den = np.atleast_1d(den)
    num = np.atleast_2d(_align_nums(num))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or"
                         " rank-2 array.")
    if np.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den = np.trim_zeros(den, 'f')

    # Normalize transfer function
    num, den = num / den[0], den / den[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if np.allclose(col, 0, atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless",
                      BadCoefficients, stacklevel=2)
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num[0, :]

    return num, den


def lp2lp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed low-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed low-pass filter.

    See Also
    --------
    lp2hp, lp2bp, lp2bs, bilinear
    lp2lp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    Examples
    --------

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> lp2 = signal.lti(*signal.lp2lp(lp.num, lp.den, 2))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_lp2, p_lp2 = lp2.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_lp2, label='Transformed Lowpass')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.legend()

    """
    a, b = map(atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    M = max((d, n))
    pwo = pow(wo, np.arange(M - 1, -1, -1))
    start1 = max((n - d, 0))
    start2 = max((d - n, 0))
    b = b * pwo[start1] / pwo[start2:]
    a = a * pwo[start1] / pwo[start1:]
    return normalize(b, a)


def lp2hp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed high-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed high-pass filter.

    See Also
    --------
    lp2lp, lp2bp, lp2bs, bilinear
    lp2hp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> hp = signal.lti(*signal.lp2hp(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_hp, p_hp = hp.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_hp, label='Highpass')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.legend()

    """
    a, b = map(atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    if wo != 1:
        pwo = pow(wo, np.arange(max((d, n))))
    else:
        pwo = np.ones(max((d, n)), b.dtype.char)
    if d >= n:
        outa = a[::-1] * pwo
        outb = resize(b, (d,))
        outb[n:] = 0.0
        outb[:n] = b[::-1] * pwo[:n]
    else:
        outb = b[::-1] * pwo
        outa = resize(a, (n,))
        outa[d:] = 0.0
        outa[:d] = a[::-1] * pwo[:d]

    return normalize(outb, outa)


def lp2bp(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-pass filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bs, bilinear
    lp2bp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> bp = signal.lti(*signal.lp2bp(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bp, p_bp = bp.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bp, label='Bandpass')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.legend()
    """

    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = mintypecode((a, b))
    ma = max([N, D])
    Np = N + ma
    Dp = D + ma
    bprime = np.empty(Np + 1, artype)
    aprime = np.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * b[N - i] * (wosq) ** (i - k) / bw ** i
        bprime[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * a[D - i] * (wosq) ** (i - k) / bw ** i
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


def lp2bs(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.5])
    >>> bs = signal.lti(*signal.lp2bs(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bs, p_bs = bs.bode(w)
    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bs, label='Bandstop')
    >>> plt.semilogx()
    >>> plt.grid(True)
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.legend()
    """
    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = mintypecode((a, b))
    M = max([N, D])
    Np = M + M
    Dp = M + M
    bprime = np.empty(Np + 1, artype)
    aprime = np.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * b[N - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        bprime[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * a[D - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


def bilinear(b, a, fs=1.0):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    b : ndarray
        Numerator of the transformed digital filter transfer function.
    a : ndarray
        Denominator of the transformed digital filter transfer function.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, lp2bs
    bilinear_zpk

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass',
    ...                                   analog=True))
    >>> filtz = signal.lti(*signal.bilinear(filts.num, filts.den, fs))
    >>> wz, hz = signal.freqz(filtz.num, filtz.den)
    >>> ws, hs = signal.freqs(filts.num, filts.den, worN=fs*wz)

    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(True)
    """
    fs = _validate_fs(fs, allow_none=False)
    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = float
    M = max([N, D])
    Np = M
    Dp = M
    bprime = np.empty(Np + 1, artype)
    aprime = np.empty(Dp + 1, artype)
    for j in range(Np + 1):
        val = 0.0
        for i in range(N + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * b[N - i] *
                                pow(2 * fs, i) * (-1) ** k)
        bprime[j] = real(val)
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * a[D - i] *
                                pow(2 * fs, i) * (-1) ** k)
        aprime[j] = real(val)

    return normalize(bprime, aprime)


def _validate_gpass_gstop(gpass, gstop):

    if gpass <= 0.0:
        raise ValueError("gpass should be larger than 0.0")
    elif gstop <= 0.0:
        raise ValueError("gstop should be larger than 0.0")
    elif gpass > gstop:
        raise ValueError("gpass should be smaller than gstop")


def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba',
              fs=None):
    """Complete IIR digital and analog filter design.

    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type. Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.

    Parameters
    ----------
    wp, ws : float or array like, shape (2,)
        Passband and stopband edge frequencies. Possible values are scalars
        (for lowpass and highpass filters) or ranges (for bandpass and bandstop
        filters).
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
        Note, that for bandpass and bandstop filters passband must lie strictly
        inside stopband or vice versa. Also note that the cutoff at the band edges 
        for IIR filters is defined as half-power, so -3dB, not half-amplitude (-6dB)
        like for `scipy.signal.fiwin`.
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.ticker

    >>> wp = 0.2
    >>> ws = 0.3
    >>> gpass = 1
    >>> gstop = 40

    >>> system = signal.iirdesign(wp, ws, gpass, gstop)
    >>> w, h = signal.freqz(*system)

    >>> fig, ax1 = plt.subplots()
    >>> ax1.set_title('Digital filter frequency response')
    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [rad/sample]')
    >>> ax1.grid(True)
    >>> ax1.set_ylim([-120, 20])
    >>> ax2 = ax1.twinx()
    >>> phase = np.unwrap(np.angle(h))
    >>> ax2.plot(w, phase, 'g')
    >>> ax2.set_ylabel('Phase [rad]', color='g')
    >>> ax2.grid(True)
    >>> ax2.axis('tight')
    >>> ax2.set_ylim([-6, 1])
    >>> nticks = 8
    >>> ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    >>> ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    """
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError(f"Invalid IIR filter type: {ftype}") from e
    except IndexError as e:
        raise ValueError(f"{ftype} does not have order selection. "
                         "Use iirfilter function.") from e

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)

    fs = _validate_fs(fs, allow_none=True)

    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError("wp and ws must have one or two elements each, and "
                         f"the same shape, got {wp.shape} and {ws.shape}")

    if any(wp <= 0) or any(ws <= 0):
        raise ValueError("Values for wp, ws must be greater than 0")

    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError("Values for wp, ws must be less than 1")
        elif any(wp >= fs/2) or any(ws >= fs/2):
            raise ValueError("Values for wp, ws must be less than fs/2 "
                             f"(fs={fs} -> fs/2={fs/2})")

    if wp.shape[0] == 2:
        if not ((ws[0] < wp[0] and wp[1] < ws[1]) or
               (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError("Passband must lie strictly inside stopband "
                             "or vice versa")

    band_type = 2 * (len(wp) - 1)
    band_type += 1
    if wp[0] >= ws[0]:
        band_type += 1

    btype = {1: 'lowpass', 2: 'highpass',
             3: 'bandstop', 4: 'bandpass'}[band_type]

    N, Wn = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype,
                     ftype=ftype, output=output, fs=fs)


def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).

        When Wn is a length-2 sequence, ``Wn[0]`` must be less than ``Wn[1]``.
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Generate a 17th-order Chebyshev II analog bandpass filter from 50 Hz to
    200 Hz and plot the frequency response:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.iirfilter(17, [2*np.pi*50, 2*np.pi*200], rs=60,
    ...                         btype='band', analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    Create a digital filter with the same properties, in a system with
    sampling rate of 2000 Hz, and plot the frequency response. (Second-order
    sections implementation is required to ensure stability of a filter of
    this order):

    >>> sos = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                        analog=False, ftype='cheby2', fs=2000,
    ...                        output='sos')
    >>> w, h = signal.freqz_sos(sos, 2000, fs=2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=True)
    ftype, btype, output = (x.lower() for x in (ftype, btype, output))
    Wn = asarray(Wn)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = Wn / (fs/2)

    if np.any(Wn <= 0):
        raise ValueError("filter critical frequencies must be greater than 0")

    if Wn.size > 1 and not Wn[0] < Wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")

    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError(f"'{btype}' is an invalid bandtype for filter.") from e

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError(f"'{ftype}' is not a valid basic IIR filter.") from e

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError(f"'{output}' is not a valid output form.")

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError(f"'{ftype}' not implemented in iirfilter.")

    # Pre-warp frequencies for digital filter design
    if not analog:
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if np.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn '
                             'for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for '
                             'bandpass or bandstop filter') from e

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError(f"'{btype}' not implemented in iirfilter.")

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k, analog=analog)


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear

    Notes
    -----
    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass', analog=True,
    ...                                   output='zpk'))
    >>> filtz = signal.lti(*signal.bilinear_zpk(filts.zeros, filts.poles,
    ...                                         filts.gain, fs))
    >>> wz, hz = signal.freqz_zpk(filtz.zeros, filtz.poles, filtz.gain)
    >>> ws, hs = signal.freqs_zpk(filts.zeros, filts.poles, filts.gain,
    ...                           worN=fs*wz)
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(True)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)

    fs = _validate_fs(fs, allow_none=False)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = append(z_z, -ones(degree))

    # Compensate for gain change
    k_z = k * real(prod(fs2 - z) / prod(fs2 - p))

    return z_z, p_z, k_z


def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.

    See Also
    --------
    lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2lp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    .. versionadded:: 1.1.0

    Examples
    --------
    Use the 'zpk' (Zero-Pole-Gain) representation of a lowpass filter to
    transform it to a new 'zpk' representation associated with a cutoff frequency wo.

    >>> from scipy.signal import lp2lp_zpk
    >>> z   = [7,   2]
    >>> p   = [5,   13]
    >>> k   = 0.8
    >>> wo  = 0.4
    >>> lp2lp_zpk(z, p, k, wo)
    (   array([2.8, 0.8]), array([2. , 5.2]), 0.8)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def lp2hp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    .. versionadded:: 1.1.0

    Examples
    --------
    Use the 'zpk' (Zero-Pole-Gain) representation of a lowpass filter to
    transform it to a highpass filter with a cutoff frequency wo.

    >>> from scipy.signal import lp2hp_zpk
    >>> z   = [ -2 + 3j ,  -0.5 - 0.8j ]
    >>> p   = [ -1      ,  -4          ]
    >>> k   = 10
    >>> wo  = 0.6
    >>> lp2hp_zpk(z, p, k, wo)
    (   array([-0.09230769-0.13846154j, -0.33707865+0.53932584j]),
        array([-0.6 , -0.15]),
        8.5)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = append(z_hp, zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * real(prod(-z) / prod(-p))

    return z_hp, p_hp, k_hp


def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear
    lp2bp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    .. versionadded:: 1.1.0

    Examples
    --------
    Use the 'zpk' (Zero-Pole-Gain) representation of a lowpass filter to
    transform it to a bandpass filter with a center frequency wo and
    bandwidth bw.

    >>> from scipy.signal import lp2bp_zpk
    >>> z   = [ 5 + 2j ,  5 - 2j ]
    >>> p   = [ 7      ,  -16    ]
    >>> k   = 0.8
    >>> wo  = 0.62
    >>> bw  = 15
    >>> lp2bp_zpk(z, p, k, wo, bw)
    (   array([7.49955815e+01+3.00017676e+01j, 7.49955815e+01-3.00017676e+01j,
               4.41850748e-03-1.76761126e-03j, 4.41850748e-03+1.76761126e-03j]),
        array([1.04996339e+02+0.j, -1.60167736e-03+0.j,  3.66108003e-03+0.j,
               -2.39998398e+02+0.j]), 0.8)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = concatenate((z_lp + sqrt(z_lp**2 - wo**2),
                        z_lp - sqrt(z_lp**2 - wo**2)))
    p_bp = concatenate((p_lp + sqrt(p_lp**2 - wo**2),
                        p_lp - sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = append(z_bp, zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    .. versionadded:: 1.1.0

    Examples
    --------
    Transform a low-pass filter represented in 'zpk' (Zero-Pole-Gain) form
    into a bandstop filter represented in 'zpk' form, with a center frequency wo and
    bandwidth bw.

    >>> from scipy.signal import lp2bs_zpk
    >>> z   = [             ]
    >>> p   = [ 0.7 ,    -1 ]
    >>> k   = 9
    >>> wo  = 0.5
    >>> bw  = 10
    >>> lp2bs_zpk(z, p, k, wo, bw)
    (   array([0.+0.5j, 0.+0.5j, 0.-0.5j, 0.-0.5j]),
        array([14.2681928 +0.j, -0.02506281+0.j,  0.01752149+0.j, -9.97493719+0.j]),
        -12.857142857142858)
    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = concatenate((z_hp + sqrt(z_hp**2 - wo**2),
                        z_hp - sqrt(z_hp**2 - wo**2)))
    p_bs = concatenate((p_hp + sqrt(p_hp**2 - wo**2),
                        p_hp - sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = append(z_bs, full(degree, +1j*wo))
    z_bs = append(z_bs, full(degree, -1j*wo))

    # Cancel out gain change caused by inversion
    k_bs = k * real(prod(-z) / prod(-p))

    return z_bs, p_bs, k_bs


def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.

        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    buttord, buttap

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    The ``'sos'`` output parameter was added in 0.16.0.

    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.

    .. warning::
        Designing high-order and narrowband IIR filters in TF form can
        result in unstable or incorrect filtering due to floating point
        numerical precision issues. Consider inspecting output filter
        characteristics `freqz` or designing the filters with second-order
        sections via ``output='sos'``.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth filter frequency response')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [s]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)


def cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type I digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type I filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type I filters, this is the point in the transition band at which
        the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    cheb1ord, cheb1ap

    Notes
    -----
    The Chebyshev type I filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the passband and increased ringing in the step response.

    Type I filters roll off faster than Type II (`cheby2`), but Type II
    filters do not have any ripple in the passband.

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type I frequency response (rp=5)')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby1(10, 1, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [s]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='cheby1', fs=fs)


def cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type II digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type II filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type II filters, this is the point in the transition band at which
        the gain first reaches -`rs`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    cheb2ord, cheb2ap

    Notes
    -----
    The Chebyshev type II filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the stopband and increased ringing in the step response.

    Type II filters do not roll off as fast as Type I (`cheby1`).

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type II frequency response (rs=40)')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 17 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby2(12, 20, 17, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 17 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [s]')
    >>> plt.show()
    """
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog,
                     output=output, ftype='cheby2', fs=fs)


def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Elliptic (Cauer) digital and analog filter design.

    Design an Nth-order digital or analog elliptic filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For elliptic filters, this is the point in the transition band at
        which the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter. Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    ellipord, ellipap

    Notes
    -----
    Also known as Cauer or Zolotarev filters, the elliptical filter maximizes
    the rate of transition between the frequency response's passband and
    stopband, at the expense of ripple in both, and increased ringing in the
    step response.

    As `rp` approaches 0, the elliptical filter becomes a Chebyshev
    type II filter (`cheby2`). As `rs` approaches 0, it becomes a Chebyshev
    type I filter (`cheby1`). As both approach 0, it becomes a Butterworth
    filter (`butter`).

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.ellip(4, 5, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Elliptic filter frequency response (rp=5, rs=40)')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 17 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.ellip(8, 1, 100, 17, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 17 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [s]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, rs=rs, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='elliptic', fs=fs)


def bessel(N, Wn, btype='low', analog=False, output='ba', norm='phase',
           fs=None):
    """
    Bessel/Thomson digital and analog filter design.

    Design an Nth-order digital or analog Bessel filter and return the
    filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies (defined
        by the `norm` parameter).
        For analog filters, `Wn` is an angular frequency (e.g., rad/s).

        For digital filters, `Wn` are in the same units as `fs`.  By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned. (See Notes.)
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.
    norm : {'phase', 'delay', 'mag'}, optional
        Critical frequency normalization:

        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at angular (e.g. rad/s) frequency `Wn`. This happens for
            both low-pass and high-pass filters, so this is the
            "phase-matched" case.

            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.

            This is the default, and matches MATLAB's implementation.

        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1/`Wn` (e.g., seconds). This is the "natural" type obtained by
            solving Bessel polynomials.

        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency `Wn`.

        .. versionadded:: 0.18.0
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    Notes
    -----
    Also known as a Thomson filter, the analog Bessel filter has maximally
    flat group delay and maximally linear phase response, with very little
    ringing in the step response. [1]_

    The Bessel is inherently an analog filter. This function generates digital
    Bessel filters using the bilinear transform, which does not preserve the
    phase response of the analog filter. As such, it is only approximately
    correct at frequencies below about fs/4. To get maximally-flat group
    delay at higher frequencies, the analog Bessel filter must be transformed
    using phase-preserving techniques.

    See `besselap` for implementation details and references.

    The ``'sos'`` output parameter was added in 0.16.0.

    References
    ----------
    .. [1] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.

    Examples
    --------
    Plot the phase-normalized frequency response, showing the relationship
    to the Butterworth's cutoff frequency (green):

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)), color='silver', ls='dashed')
    >>> b, a = signal.bessel(4, 100, 'low', analog=True, norm='phase')
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))
    >>> plt.title('Bessel filter magnitude response (with Butterworth)')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green')  # cutoff frequency
    >>> plt.show()

    and the phase midpoint:

    >>> plt.figure()
    >>> plt.semilogx(w, np.unwrap(np.angle(h)))
    >>> plt.axvline(100, color='green')  # cutoff frequency
    >>> plt.axhline(-np.pi, color='red')  # phase midpoint
    >>> plt.title('Bessel filter phase response')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Phase [rad]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    Plot the magnitude-normalized frequency response, showing the -3 dB cutoff:

    >>> b, a = signal.bessel(3, 10, 'low', analog=True, norm='mag')
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))
    >>> plt.axhline(-3, color='red')  # -3 dB magnitude
    >>> plt.axvline(10, color='green')  # cutoff frequency
    >>> plt.title('Amplitude-normalized Bessel filter frequency response')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    Plot the delay-normalized filter, showing the maximally-flat group delay
    at 0.1 seconds:

    >>> b, a = signal.bessel(5, 1/0.1, 'low', analog=True, norm='delay')
    >>> w, h = signal.freqs(b, a)
    >>> plt.figure()
    >>> plt.semilogx(w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w))
    >>> plt.axhline(0.1, color='red')  # 0.1 seconds group delay
    >>> plt.title('Bessel filter group delay')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Group delay [s]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='bessel_'+norm, fs=fs)


def maxflat():
    pass


def yulewalk():
    pass


def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.

    Returns the non-integer order for an analog band stop filter.

    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.

    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).

    Notes
    -----
    Band-stop filters are used in applications where certain frequency
    components need to be blocked while others are allowed; for instance,
    removing noise at specific frequencies while allowing the desired signal
    to pass through. The order of a filter often determines its complexity and
    accuracy. Determining the right order can be a challenge. This function
    aims to provide an appropriate order for an analog band stop filter.
    
    Examples
    --------

    >>> import numpy as np
    >>> from scipy.signal import band_stop_obj
    >>> wp = 2
    >>> ind = 1
    >>> passb = np.array([1, 3])
    >>> stopb = np.array([0.5, 4])
    >>> gstop = 30
    >>> gpass = 3
    >>> filter_type = 'butter'
    >>> band_stop_obj(wp, ind, passb, stopb, gpass, gstop, filter_type)
    np.float64(-2.758504160760643)

    """

    _validate_gpass_gstop(gpass, gstop)

    passbC = passb.copy()
    passbC[ind] = wp
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = (log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat)))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = special.ellipk([arg0 ** 2, 1 - arg0 ** 2])
        d1 = special.ellipk([arg1 ** 2, 1 - arg1 ** 2])
        n = (d0[0] * d1[1] / (d0[1] * d1[0]))
    else:
        raise ValueError(f"Incorrect type: {type}")
    return n


def _pre_warp(wp, ws, analog):
    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = np.tan(pi * wp / 2.0)
        stopb = np.tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0
    return passb, stopb


def _validate_wp_ws(wp, ws, fs, analog):
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2 * wp / fs
        ws = 2 * ws / fs

    filter_type = 2 * (len(wp) - 1) + 1
    if wp[0] >= ws[0]:
        filter_type += 1

    return wp, ws, filter_type


def _find_nat_freq(stopb, passb, gpass, gstop, filter_type, filter_kind):
    if filter_type == 1:            # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:          # stop

       ### breakpoint()

        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop,
                                       filter_kind),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop,
                                       filter_kind),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:          # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))
    else:
        raise ValueError(f"should not happen: {filter_type =}.")

    nat = min(abs(nat))
    return nat, passb


def _postprocess_wn(WN, analog, fs):
    wn = WN if analog else np.arctan(WN) * 2.0 / pi
    if len(wn) == 1:
        wn = wn[0]
    if fs is not None:
        wn = wn * fs / 2
    return wn


def buttord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Butterworth filter order selection.

    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency"). Should
        be used with `butter` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `butter`.

    See Also
    --------
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog bandpass filter with passband within 3 dB from 20 to
    50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
    Plot its frequency response, showing the passband and stopband
    constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
    >>> b, a = signal.butter(N, Wn, 'band', True)
    >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth bandpass filter fit to constraints')
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
    >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
    >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
    >>> plt.axis([10, 100, -60, 3])
    >>> plt.show()

    """
    _validate_gpass_gstop(gpass, gstop)
    fs = _validate_fs(fs, allow_none=True)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'butter')

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))

    # Find the Butterworth natural frequency WN (or the "3dB" frequency")
    # to give exactly gpass at passb.
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        warnings.warn("Order is zero...check input parameters.",
                      RuntimeWarning, stacklevel=2)

    # now convert this frequency back from lowpass prototype
    # to the original analog filter

    if filter_type == 1:  # low
        WN = W0 * passb
    elif filter_type == 2:  # high
        WN = passb / W0
    elif filter_type == 3:  # stop
        WN = np.empty(2, float)
        discr = sqrt((passb[1] - passb[0]) ** 2 +
                     4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = np.sort(abs(WN))
    elif filter_type == 4:  # pass
        W0 = np.array([-W0, W0], float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                   passb[0] * passb[1]))
        WN = np.sort(abs(WN))
    else:
        raise ValueError(f"Bad type: {filter_type}")

    wn = _postprocess_wn(WN, analog, fs)

    return ord, wn


def cheb1ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type I filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type I
    filter that loses no more than `gpass` dB in the passband and has at
    least `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type I filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby1` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby1`.

    See Also
    --------
    cheby1 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital lowpass filter such that the passband is within 3 dB up
    to 0.2*(fs/2), while rejecting at least -40 dB above 0.3*(fs/2). Plot its
    frequency response, showing the passband and stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.cheb1ord(0.2, 0.3, 3, 40)
    >>> b, a = signal.cheby1(N, 3, Wn, 'low')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev I lowpass filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0) # pass
    >>> plt.axis([0.08, 1, -60, 3])
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=True)
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'cheby')

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    v_pass_stop = np.arccosh(np.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(ceil(v_pass_stop / np.arccosh(nat)))

    # Natural frequencies are just the passband edges
    wn = _postprocess_wn(passb, analog, fs)

    return ord, wn


def cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type II filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type II
    filter that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type II filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby2` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby2`.

    See Also
    --------
    cheby2 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital bandstop filter which rejects -60 dB from 0.2*(fs/2) to
    0.5*(fs/2), while staying within 3 dB below 0.1*(fs/2) or above
    0.6*(fs/2). Plot its frequency response, showing the passband and
    stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> N, Wn = signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    >>> b, a = signal.cheby2(N, 60, Wn, 'stop')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev II bandstop filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, .1, .1, .01], [-3,  -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([.2,  .2, .5,  .5], [ 9, -60, -60,   9], '0.9', lw=0) # pass
    >>> plt.fill([.6,  .6,  2,   2], [-99, -3,  -3, -99], '0.9', lw=0) # stop
    >>> plt.axis([0.06, 1, -80, 3])
    >>> plt.show()

    """
    fs = _validate_fs(fs, allow_none=True)
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'cheby')

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    v_pass_stop = np.arccosh(np.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(ceil(v_pass_stop / arccosh(nat)))

    # Find frequency where analog response is -gpass dB.
    # Then convert back from low-pass prototype to the original filter.

    new_freq = cosh(1.0 / ord * v_pass_stop)
    new_freq = 1.0 / new_freq

    if filter_type == 1:
        nat = passb / new_freq
    elif filter_type == 2:
        nat = passb * new_freq
    elif filter_type == 3:
        nat = np.empty(2, float)
        nat[0] = (new_freq / 2.0 * (passb[0] - passb[1]) +
                  sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 +
                       passb[1] * passb[0]))
        nat[1] = passb[1] * passb[0] / nat[0]
    elif filter_type == 4:
        nat = np.empty(2, float)
        nat[0] = (1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) +
                  sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) +
                       passb[1] * passb[0]))
        nat[1] = passb[0] * passb[1] / nat[0]

    wn = _postprocess_wn(nat, analog, fs)

    return ord, wn


_POW10_LOG10 = np.log(10)


def _pow10m1(x):
    """10 ** x - 1 for x near 0"""
    return np.expm1(_POW10_LOG10 * x)


def ellipord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Elliptic (Cauer) filter order selection.

    Return the order of the lowest order digital or analog elliptic filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

