        ``2*m - 2`` where ``m`` is the length of the transformed axis of
        the input. To get an odd number of output points, `n` must be
        specified, for instance as ``2*m - 1`` in the typical case,

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See also
    --------
    rfft : Compute the one-dimensional FFT for real input.
    ihfft : The inverse of `hfft`.

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So here it's `hfft` for
    which you must supply the length of the result if it is to be odd.

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.

    The correct interpretation of the hermitian input depends on the length of
    the original data, as given by `n`. This is because each input shape could
    correspond to either an odd or even length signal. By default, `hfft`
    assumes an even output length which puts the last entry at the Nyquist
    frequency; aliasing with its symmetric counterpart. By Hermitian symmetry,
    the value is thus treated as purely real. To avoid losing information, the
    shape of the full signal **must** be given.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 3, 4, 3, 2])
    >>> np.fft.fft(signal)
    array([15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]) # may vary
    >>> np.fft.hfft(signal[:4]) # Input first half of signal
    array([15.,  -4.,   0.,  -1.,   0.,  -4.])
    >>> np.fft.hfft(signal, 6)  # Input entire signal and truncate
    array([15.,  -4.,   0.,  -1.,   0.,  -4.])


    >>> signal = np.array([[1, 1.j], [-1.j, 2]])
    >>> np.conj(signal.T) - signal   # check Hermitian symmetry
    array([[ 0.-0.j,  -0.+0.j], # may vary
           [ 0.+0.j,  0.-0.j]])
    >>> freq_spectrum = np.fft.hfft(signal)
    >>> freq_spectrum
    array([[ 1.,  1.],
           [ 2., -2.]])

    """
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    new_norm = _swap_direction(norm)
    output = irfft(conjugate(a), n, axis, norm=new_norm, out=None)
    return output


@array_function_dispatch(_fft_dispatcher)
def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped.  If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So here it's `hfft` for
    which you must supply the length of the result if it is to be odd:

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.

    Examples
    --------
    >>> import numpy as np
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> np.fft.ifft(spectrum)
    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary
    >>> np.fft.ihfft(spectrum)
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary

    """
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    new_norm = _swap_direction(norm)
    out = rfft(a, n, axis, norm=new_norm, out=out)
    return conjugate(out, out=out)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if not shapeless:
            msg = ("`axes` should not be `None` if `s` is not `None` "
                   "(Deprecated in NumPy 2.0). In a future version of NumPy, "
                   "this will raise an error and `s[i]` will correspond to "
                   "the size along the transformed axis specified by "
                   "`axes[i]`. To retain current behaviour, pass a sequence "
                   "[0, ..., k-1] to `axes` for an array of dimension k.")
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s:
        msg = ("Passing an array containing `None` values to `s` is "
               "deprecated in NumPy 2.0 and will raise an error in "
               "a future version of NumPy. To use the default behaviour "
               "of the corresponding 1-D transform, pass the value matching "
               "the default for its `n` parameter. To use the default "
               "behaviour for every axis, the `s` argument can be omitted.")
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    # use the whole input array along axis `i` if `s[i] == -1`
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None, out=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm, out=out)
    return a


def _fftn_dispatcher(a, s=None, axes=None, norm=None, out=None):
    return (a, out)


@array_function_dispatch(_fftn_dispatcher)
def fftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    numpy.fft : Overall view of discrete Fourier transforms, with definitions
        and conventions used.
    ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.
    fft : The one-dimensional FFT, with definitions and conventions used.
    rfftn : The *n*-dimensional FFT of real input.
    fft2 : The two-dimensional FFT.
    fftshift : Shifts zero-frequency terms to centre of array

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of all axes, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    See `numpy.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.mgrid[:3, :3, :3][0]
    >>> np.fft.fftn(a, axes=(1, 2))
    array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[ 9.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[18.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]]])
    >>> np.fft.fftn(a, (2, 2), axes=(0, 1))
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    >>> import matplotlib.pyplot as plt
    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
    ...                      2 * np.pi * np.arange(200) / 34)
    >>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
    >>> FS = np.fft.fftn(S)
    >>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


@array_function_dispatch(_fftn_dispatcher)
def ifftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform over any number of axes in an M-dimensional array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(a)) == a`` to within numerical accuracy.
    For a description of the definitions and conventions used, see `numpy.fft`.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e. it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. See notes for issue on `ifft` zero padding.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    numpy.fft : Overall view of discrete Fourier transforms, with definitions
         and conventions used.
    fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.
    ifft : The one-dimensional inverse FFT.
    ifft2 : The two-dimensional inverse FFT.
    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
        of array.

    Notes
    -----
    See `numpy.fft` for definitions and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension.  Although this is the common
    approach, it might lead to surprising results.  If another form of zero
    padding is desired, it must be performed before `ifftn` is called.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.eye(4)
    >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])


    Create and plot an image with band-limited frequency content:

    >>> import matplotlib.pyplot as plt
    >>> n = np.zeros((200,200), dtype=complex)
    >>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
    >>> im = np.fft.ifftn(n).real
    >>> plt.imshow(im)
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)


@array_function_dispatch(_fftn_dispatcher)
def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence only the
        last axis can have ``s`` not equal to the shape at that axis).

        .. versionadded:: 2.0.0

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
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    numpy.fft : Overall view of discrete Fourier transforms, with definitions
         and conventions used.
    ifft2 : The inverse two-dimensional FFT.
    fft : The one-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For two-dimensional input, swaps first and third quadrants, and second
        and fourth quadrants.

    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.

    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    See `fftn` for details and a plotting example, and `numpy.fft` for
    definitions and conventions used.


    Examples
    --------
    >>> import numpy as np
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.fft2(a)
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
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


@array_function_dispatch(_fftn_dispatcher)
def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an M-dimensional array by means of
    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``
    to within numerical accuracy.  By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e. it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

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
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    numpy.fft : Overall view of discrete Fourier transforms, with definitions
         and conventions used.
    fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.
    ifftn : The inverse of the *n*-dimensional FFT.
    fft : The one-dimensional FFT.
    ifft : The one-dimensional inverse FFT.

    Notes
    -----
    `ifft2` is just `ifftn` with a different default for `axes`.

    See `ifftn` for details and a plotting example, and `numpy.fft` for
    definition and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension.  Although this is the common
    approach, it might lead to surprising results.  If another form of zero
    padding is desired, it must be performed before `ifft2` is called.

    Examples
    --------
    >>> import numpy as np
    >>> a = 4 * np.eye(4)
    >>> np.fft.ifft2(a)
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])

    """
    return _raw_fftnd(a, s, axes, ifft, norm, out=None)


@array_function_dispatch(_fftn_dispatcher)
def rfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real array by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    a : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for all axes (and hence is
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT
         of real input.
    fft : The one-dimensional FFT, with definitions and conventions used.
    rfft : The one-dimensional FFT of real input.
    fftn : The n-dimensional FFT.
    rfft2 : The two-dimensional FFT of real input.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`.  The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    See `fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ones((2, 2, 2))
    >>> np.fft.rfftn(a)
    array([[[8.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    >>> np.fft.rfftn(a, axes=(2, 0))
    array([[[4.+0.j,  0.+0.j], # may vary
            [4.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm, out=out)
    for ii in range(len(axes)-2, -1, -1):
        a = fft(a, s[ii], axes[ii], norm, out=out)
    return a


@array_function_dispatch(_fftn_dispatcher)
def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional FFT of a real array.

    Parameters
    ----------
    a : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default: ``(-2, -1)``.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for the last inverse transform.
        incompatible with passing in all but the trivial ``s``).

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    See Also
    --------
    rfftn : Compute the N-dimensional discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.rfft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ]])
    """
    return rfftn(a, s, axes, norm, out=out)


@array_function_dispatch(_fftn_dispatcher)
def irfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Computes the inverse of `rfftn`.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform for real input over any number of axes in an
    M-dimensional array by means of the Fast Fourier Transform (FFT).  In
    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e. as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    a : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped.  If it is larger, the input is padded
        with zeros.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to
        be ``2*(m-1)`` where ``m`` is the length of the input along that axis.

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must be explicitly specified too.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for the last transformation.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
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
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    rfftn : The forward n-dimensional FFT of real input,
            of which `ifftn` is the inverse.
    fft : The one-dimensional FFT, with definitions and conventions used.
    irfft : The inverse of the one-dimensional FFT of real input.
    irfft2 : The inverse of the two-dimensional FFT of real input.

    Notes
    -----
    See `fft` for definitions and conventions used.

    See `rfft` for definitions and conventions used for real input.

    The correct interpretation of the hermitian input depends on the shape of
    the original data, as given by `s`. This is because each input shape could
    correspond to either an odd or even length signal. By default, `irfftn`
    assumes an even output length which puts the last entry at the Nyquist
    frequency; aliasing with its symmetric counterpart. When performing the
    final complex to real transform, the last value is thus treated as purely
    real. To avoid losing information, the correct shape of the real input
    **must** be given.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 2, 2))
    >>> a[0, 0, 0] = 3 * 2 * 2
    >>> np.fft.irfftn(a)
    array([[[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]]])

    """
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = ifft(a, s[ii], axes[ii], norm)
    a = irfft(a, s[-1], axes[-1], norm, out=out)
    return a


@array_function_dispatch(_fftn_dispatcher)
def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Computes the inverse of `rfft2`.

    Parameters
    ----------
    a : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.

        .. versionchanged:: 2.0

            If it is ``-1``, the whole input is used (no padding/trimming).

        .. deprecated:: 2.0

            If `s` is not ``None``, `axes` must not be ``None`` either.

        .. deprecated:: 2.0

            `s` must contain only ``int`` s, not ``None`` values. ``None``
            values currently mean that the default value for ``n`` is used
            in the corresponding 1-D transform, but this behaviour is
            deprecated.

    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default: ``(-2, -1)``, the last two axes.

        .. deprecated:: 2.0

            If `s` is specified, the corresponding `axes` to be transformed
            must not be ``None``.

    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype for the last transformation.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The forward two-dimensional FFT of real input,
            of which `irfft2` is the inverse.
    rfft : The one-dimensional FFT for real input.
    irfft : The inverse of the one-dimensional FFT of real input.
    irfftn : Compute the inverse of the N-dimensional FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.mgrid[:5, :5][0]
    >>> A = np.fft.rfft2(a)
    >>> np.fft.irfft2(A, s=a.shape)
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])
    """
    return irfftn(a, s, axes, norm, out=None)
<|endoftext|>"""
Discrete Fourier Transforms - _helper.py

"""
from numpy._core import integer, empty, arange, asarray, roll
from numpy._core.overrides import array_function_dispatch, set_module

# Created by Pearu Peterson, September 2002

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']

integer_types = (int, integer)


def _fftshift_dispatcher(x, axes=None):
    return (x,)


@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

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
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return roll(x, shift, axes)


@array_function_dispatch(_fftshift_dispatcher, module='numpy.fft')
def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
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
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return roll(x, shift, axes)


@set_module('numpy.fft')
def fftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

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
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    f : ndarray
        Array of length `n` containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = np.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int, device=device)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int, device=device)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int, device=device)
    results[N:] = p2
    return results * val


@set_module('numpy.fft')
def rfftfreq(n, d=1.0, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies
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
    device : str, optional
        The device on which to place the created array. Default: ``None``.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = np.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20., ..., -30., -20., -10.])
    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = arange(0, N, dtype=int, device=device)
    return results * val
<|endoftext|>"""
Build common block mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__
f2py_version = __version__.version

from .auxfuncs import (
    hasbody, hascommon, hasnote, isintent_hide, outmess, getuseblocks
)
from . import capi_maps
from . import func2subr
from .crackfortran import rmbadname


def findcommonblocks(block, top=1):
    ret = []
    if hascommon(block):
        for key, value in block['common'].items():
            vars_ = {v: block['vars'][v] for v in value}
            ret.append((key, value, vars_))
    elif hasbody(block):
        for b in block['body']:
            ret = ret + findcommonblocks(b, 0)
    if top:
        tret = []
        names = []
        for t in ret:
            if t[0] not in names:
                names.append(t[0])
                tret.append(t)
        return tret
    return ret


def buildhooks(m):
    ret = {'commonhooks': [], 'initcommonhooks': [],
           'docs': ['"COMMON blocks:\\n"']}
    fwrap = ['']

    def fadd(line, s=fwrap):
        s[0] = '%s\n      %s' % (s[0], line)
    chooks = ['']

    def cadd(line, s=chooks):
        s[0] = '%s\n%s' % (s[0], line)
    ihooks = ['']

    def iadd(line, s=ihooks):
        s[0] = '%s\n%s' % (s[0], line)
    doc = ['']

    def dadd(line, s=doc):
        s[0] = '%s\n%s' % (s[0], line)
    for (name, vnames, vars) in findcommonblocks(m):
        lower_name = name.lower()
        hnames, inames = [], []
        for n in vnames:
            if isintent_hide(vars[n]):
                hnames.append(n)
            else:
                inames.append(n)
        if hnames:
            outmess('\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n\t\t  Hidden: %s\n' % (
                name, ','.join(inames), ','.join(hnames)))
        else:
            outmess('\t\tConstructing COMMON block support for "%s"...\n\t\t  %s\n' % (
                name, ','.join(inames)))
        fadd('subroutine f2pyinit%s(setupfunc)' % name)
        for usename in getuseblocks(m):
            fadd(f'use {usename}')
        fadd('external setupfunc')
        for n in vnames:
            fadd(func2subr.var2fixfortran(vars, n))
        if name == '_BLNK_':
            fadd('common %s' % (','.join(vnames)))
        else:
            fadd('common /%s/ %s' % (name, ','.join(vnames)))
        fadd('call setupfunc(%s)' % (','.join(inames)))
        fadd('end\n')
        cadd('static FortranDataDef f2py_%s_def[] = {' % (name))
        idims = []
        for n in inames:
            ct = capi_maps.getctype(vars[n])
            elsize = capi_maps.get_elsize(vars[n])
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, vars[n])
            if dm['dims']:
                idims.append('(%s)' % (dm['dims']))
            else:
                idims.append('')
            dms = dm['dims'].strip()
            if not dms:
                dms = '-1'
            cadd('\t{\"%s\",%s,{{%s}},%s, %s},'
                 % (n, dm['rank'], dms, at, elsize))
        cadd('\t{NULL}\n};')
        inames1 = rmbadname(inames)
        inames1_tps = ','.join(['char *' + s for s in inames1])
        cadd('static void f2py_setup_%s(%s) {' % (name, inames1_tps))
        cadd('\tint i_f2py=0;')
        for n in inames1:
            cadd('\tf2py_%s_def[i_f2py++].data = %s;' % (name, n))
        cadd('}')
        if '_' in lower_name:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        cadd('extern void %s(f2pyinit%s,F2PYINIT%s)(void(*)(%s));'
             % (F_FUNC, lower_name, name.upper(),
                ','.join(['char*'] * len(inames1))))
        cadd('static void f2py_init_%s(void) {' % name)
        cadd('\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);'
             % (F_FUNC, lower_name, name.upper(), name))
        cadd('}\n')
        iadd('\ttmp = PyFortranObject_New(f2py_%s_def,f2py_init_%s);' % (name, name))
        iadd('\tif (tmp == NULL) return NULL;')
        iadd('\tif (F2PyDict_SetItemString(d, \"%s\", tmp) == -1) return NULL;'
             % name)
        iadd('\tPy_DECREF(tmp);')
        tname = name.replace('_', '\\_')
        dadd('\\subsection{Common block \\texttt{%s}}\n' % (tname))
        dadd('\\begin{description}')
        for n in inames:
            dadd('\\item[]{{}\\verb@%s@{}}' %
                 (capi_maps.getarrdocsign(n, vars[n])))
            if hasnote(vars[n]):
                note = vars[n]['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd('--- %s' % (note))
        dadd('\\end{description}')
        ret['docs'].append(
            '"\t/%s/ %s\\n"' % (name, ','.join(map(lambda v, d: v + d, inames, idims))))
    ret['commonhooks'] = chooks
    ret['initcommonhooks'] = ihooks
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fwrap[0]
<|endoftext|>"""

f2py2e - Fortran to Python C/API generator. 2nd Edition.
         See __usage__ below.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import sys
import os
import pprint
import re
import argparse

from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
from .cfuncs import errmess
from numpy.f2py._backends import f2py_build_generator

f2py_version = __version__.version
numpy_version = __version__.version

# outmess=sys.stdout.write
show = pprint.pprint
outmess = auxfuncs.outmess
MESON_ONLY_VER = (sys.version_info >= (3, 12))

__usage__ =\
f"""Usage:

1) To construct extension module sources:

      f2py [<options>] <fortran files> [[[only:]||[skip:]] \\
                                        <fortran functions> ] \\
                                       [: <fortran files> ...]

2) To compile fortran files and build extension modules:

      f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>

3) To generate signature files:

      f2py -h <filename.pyf> ...< same options as in (1) >

Description: This program generates a Python C/API file (<modulename>module.c)
             that contains wrappers for given fortran functions so that they
             can be called from Python. With the -c option the corresponding
             extension modules are built.

Options:

  -h <filename>    Write signatures of the fortran routines to file <filename>
                   and exit. You can then edit <filename> and use it instead
                   of <fortran files>. If <filename>==stdout then the
                   signatures are printed to stdout.
  <fortran functions>  Names of fortran routines for which Python C/API
                   functions will be generated. Default is all that are found
                   in <fortran files>.
  <fortran files>  Paths to fortran/signature files that will be scanned for
                   <fortran functions> in order to determine their signatures.
  skip:            Ignore fortran functions that follow until `:'.
  only:            Use only fortran functions that follow until `:'.
  :                Get back to <fortran files> mode.

  -m <modulename>  Name of the module; f2py generates a Python/C API
                   file <modulename>module.c or extension module <modulename>.
                   Default is 'untitled'.

  '-include<header>'  Writes additional headers in the C wrapper, can be passed
                      multiple times, generates #include <header> each time.

  --[no-]lower     Do [not] lower the cases in <fortran files>. By default,
                   --lower is assumed with -h key, and --no-lower without -h key.

  --build-dir <dirname>  All f2py generated files are created in <dirname>.
                   Default is tempfile.mkdtemp().

  --overwrite-signature  Overwrite existing signature file.

  --[no-]latex-doc Create (or not) <modulename>module.tex.
                   Default is --no-latex-doc.
  --short-latex    Create 'incomplete' LaTeX document (without commands
                   \\documentclass, \\tableofcontents, and \\begin{{document}},
                   \\end{{document}}).

  --[no-]rest-doc Create (or not) <modulename>module.rst.
                   Default is --no-rest-doc.

  --debug-capi     Create C/API code that reports the state of the wrappers
                   during runtime. Useful for debugging.

  --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77
                   functions. --wrap-functions is default because it ensures
                   maximum portability/compiler independence.

  --[no-]freethreading-compatible    Create a module that declares it does or
                   doesn't require the GIL. The default is
                   --freethreading-compatible for backward
                   compatibility. Inspect the Fortran code you are wrapping for
                   thread safety issues before passing
                   --no-freethreading-compatible, as f2py does not analyze
                   fortran code for thread safety issues.

  --include-paths <path1>:<path2>:...   Search include files from the given
                   directories.

  --help-link [..] List system resources found by system_info.py. See also
                   --link-<resource> switch below. [..] is optional list
                   of resources names. E.g. try 'f2py --help-link lapack_opt'.

  --f2cmap <filename>  Load Fortran-to-Python KIND specification from the given
                   file. Default: .f2py_f2cmap in current directory.

  --quiet          Run quietly.
  --verbose        Run with extra verbosity.
  --skip-empty-wrappers   Only generate wrapper files when needed.
  -v               Print f2py version ID and exit.


build backend options (only effective with -c)
[NO_MESON] is used to indicate an option not meant to be used
with the meson backend or above Python 3.12:

  --fcompiler=         Specify Fortran compiler type by vendor [NO_MESON]
  --compiler=          Specify distutils C compiler type [NO_MESON]

  --help-fcompiler     List available Fortran compilers and exit [NO_MESON]
  --f77exec=           Specify the path to F77 compiler [NO_MESON]
  --f90exec=           Specify the path to F90 compiler [NO_MESON]
  --f77flags=          Specify F77 compiler flags
  --f90flags=          Specify F90 compiler flags
  --opt=               Specify optimization flags [NO_MESON]
  --arch=              Specify architecture specific optimization flags [NO_MESON]
  --noopt              Compile without optimization [NO_MESON]
  --noarch             Compile without arch-dependent optimization [NO_MESON]
  --debug              Compile with debugging information

  --dep                <dependency>
                       Specify a meson dependency for the module. This may
                       be passed multiple times for multiple dependencies.
                       Dependencies are stored in a list for further processing.

                       Example: --dep lapack --dep scalapack
                       This will identify "lapack" and "scalapack" as dependencies
                       and remove them from argv, leaving a dependencies list
                       containing ["lapack", "scalapack"].

  --backend            <backend_type>
                       Specify the build backend for the compilation process.
                       The supported backends are 'meson' and 'distutils'.
                       If not specified, defaults to 'distutils'. On
                       Python 3.12 or higher, the default is 'meson'.

Extra options (only effective with -c):

  --link-<resource>    Link extension module with <resource> as defined
                       by numpy.distutils/system_info.py. E.g. to link
                       with optimized LAPACK libraries (vecLib on MacOSX,
                       ATLAS elsewhere), use --link-lapack_opt.
                       See also --help-link switch. [NO_MESON]

  -L/path/to/lib/ -l<libname>
  -D<define> -U<name>
  -I/path/to/include/
  <filename>.o <filename>.so <filename>.a

  Using the following macros may be required with non-gcc Fortran
  compilers:
    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN

  When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY
  interface is printed out at exit (platforms: Linux).

  When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is
  sent to stderr whenever F2PY interface makes a copy of an
  array. Integer <int> sets the threshold for array sizes when
  a message should be shown.

Version:     {f2py_version}
numpy Version: {numpy_version}
License:     NumPy license (see LICENSE.txt in the NumPy source code)
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
https://numpy.org/doc/stable/f2py/index.html\n"""


def scaninputline(inputline):
    files, skipfuncs, onlyfuncs, debug = [], [], [], []
    f, f2, f3, f5, f6, f8, f9, f10 = 1, 0, 0, 0, 0, 0, 0, 0
    verbose = 1
    emptygen = True
    dolc = -1
    dolatexdoc = 0
    dorestdoc = 0
    wrapfuncs = 1
    buildpath = '.'
    include_paths, freethreading_compatible, inputline = get_newer_options(inputline)
    signsfile, modulename = None, None
    options = {'buildpath': buildpath,
               'coutput': None,
               'f2py_wrapper_output': None}
    for l in inputline:
        if l == '':
            pass
        elif l == 'only:':
            f = 0
        elif l == 'skip:':
            f = -1
        elif l == ':':
            f = 1
        elif l[:8] == '--debug-':
            debug.append(l[8:])
        elif l == '--lower':
            dolc = 1
        elif l == '--build-dir':
            f6 = 1
        elif l == '--no-lower':
            dolc = 0
        elif l == '--quiet':
            verbose = 0
        elif l == '--verbose':
            verbose += 1
        elif l == '--latex-doc':
            dolatexdoc = 1
        elif l == '--no-latex-doc':
            dolatexdoc = 0
        elif l == '--rest-doc':
            dorestdoc = 1
        elif l == '--no-rest-doc':
            dorestdoc = 0
        elif l == '--wrap-functions':
            wrapfuncs = 1
        elif l == '--no-wrap-functions':
            wrapfuncs = 0
        elif l == '--short-latex':
            options['shortlatex'] = 1
        elif l == '--coutput':
            f8 = 1
        elif l == '--f2py-wrapper-output':
            f9 = 1
        elif l == '--f2cmap':
            f10 = 1
        elif l == '--overwrite-signature':
            options['h-overwrite'] = 1
        elif l == '-h':
            f2 = 1
        elif l == '-m':
            f3 = 1
        elif l[:2] == '-v':
            print(f2py_version)
            sys.exit()
        elif l == '--show-compilers':
            f5 = 1
        elif l[:8] == '-include':
            cfuncs.outneeds['userincludes'].append(l[9:-1])
            cfuncs.userincludes[l[9:-1]] = '#include ' + l[8:]
        elif l == '--skip-empty-wrappers':
            emptygen = False
        elif l[0] == '-':
            errmess('Unknown option %s\n' % repr(l))
            sys.exit()
        elif f2:
            f2 = 0
            signsfile = l
        elif f3:
            f3 = 0
            modulename = l
        elif f6:
            f6 = 0
            buildpath = l
        elif f8:
            f8 = 0
            options["coutput"] = l
        elif f9:
            f9 = 0
            options["f2py_wrapper_output"] = l
        elif f10:
            f10 = 0
            options["f2cmap_file"] = l
        elif f == 1:
            try:
                with open(l):
                    pass
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}. Skipping file "{l!s}".\n')
        elif f == -1:
            skipfuncs.append(l)
        elif f == 0:
            onlyfuncs.append(l)
    if not f5 and not files and not modulename:
        print(__usage__)
        sys.exit()
    if not os.path.isdir(buildpath):
        if not verbose:
            outmess('Creating build directory %s\n' % (buildpath))
        os.mkdir(buildpath)
    if signsfile:
        signsfile = os.path.join(buildpath, signsfile)
    if signsfile and os.path.isfile(signsfile) and 'h-overwrite' not in options:
        errmess(
            'Signature file "%s" exists!!! Use --overwrite-signature to overwrite.\n' % (signsfile))
        sys.exit()

    options['emptygen'] = emptygen
    options['debug'] = debug
    options['verbose'] = verbose
    if dolc == -1 and not signsfile:
        options['do-lower'] = 0
    else:
        options['do-lower'] = dolc
    if modulename:
        options['module'] = modulename
    if signsfile:
        options['signsfile'] = signsfile
    if onlyfuncs:
        options['onlyfuncs'] = onlyfuncs
    if skipfuncs:
        options['skipfuncs'] = skipfuncs
    options['dolatexdoc'] = dolatexdoc
    options['dorestdoc'] = dorestdoc
    options['wrapfuncs'] = wrapfuncs
    options['buildpath'] = buildpath
    options['include_paths'] = include_paths
    options['requires_gil'] = not freethreading_compatible
    options.setdefault('f2cmap_file', None)
    return files, options


def callcrackfortran(files, options):
    rules.options = options
    crackfortran.debug = options['debug']
    crackfortran.verbose = options['verbose']
    if 'module' in options:
        crackfortran.f77modulename = options['module']
    if 'skipfuncs' in options:
        crackfortran.skipfuncs = options['skipfuncs']
    if 'onlyfuncs' in options:
        crackfortran.onlyfuncs = options['onlyfuncs']
    crackfortran.include_paths[:] = options['include_paths']
    crackfortran.dolowercase = options['do-lower']
    postlist = crackfortran.crackfortran(files)
    if 'signsfile' in options:
        outmess('Saving signatures to file "%s"\n' % (options['signsfile']))
        pyf = crackfortran.crack2fortran(postlist)
        if options['signsfile'][-6:] == 'stdout':
            sys.stdout.write(pyf)
        else:
            with open(options['signsfile'], 'w') as f:
                f.write(pyf)
    if options["coutput"] is None:
        for mod in postlist:
            mod["coutput"] = "%smodule.c" % mod["name"]
    else:
        for mod in postlist:
            mod["coutput"] = options["coutput"]
    if options["f2py_wrapper_output"] is None:
        for mod in postlist:
            mod["f2py_wrapper_output"] = "%s-f2pywrappers.f" % mod["name"]
    else:
        for mod in postlist:
            mod["f2py_wrapper_output"] = options["f2py_wrapper_output"]
    for mod in postlist:
        if options["requires_gil"]:
            mod['gil_used'] = 'Py_MOD_GIL_USED'
        else:
            mod['gil_used'] = 'Py_MOD_GIL_NOT_USED'
    return postlist


def buildmodules(lst):
    cfuncs.buildcfuncs()
    outmess('Building modules...\n')
    modules, mnames, isusedby = [], [], {}
    for item in lst:
        if '__user__' in item['name']:
            cb_rules.buildcallbacks(item)
        else:
            if 'use' in item:
                for u in item['use'].keys():
                    if u not in isusedby:
                        isusedby[u] = []
                    isusedby[u].append(item['name'])
            modules.append(item)
            mnames.append(item['name'])
    ret = {}
    for module, name in zip(modules, mnames):
        if name in isusedby:
            outmess('\tSkipping module "%s" which is used by %s.\n' % (
                name, ','.join('"%s"' % s for s in isusedby[name])))
        else:
            um = []
            if 'use' in module:
                for u in module['use'].keys():
                    if u in isusedby and u in mnames:
                        um.append(modules[mnames.index(u)])
                    else:
                        outmess(
                            f'\tModule "{name}" uses nonexisting "{u}" '
                            'which will be ignored.\n')
            ret[name] = {}
            dict_append(ret[name], rules.buildmodule(module, um))
    return ret


def dict_append(d_out, d_in):
    for (k, v) in d_in.items():
        if k not in d_out:
            d_out[k] = []
        if isinstance(v, list):
            d_out[k] = d_out[k] + v
        else:
            d_out[k].append(v)


def run_main(comline_list):
    """
    Equivalent to running::

        f2py <args>

    where ``<args>=string.join(<list>,' ')``, but in Python.  Unless
    ``-h`` is used, this function returns a dictionary containing
    information on generated modules and their dependencies on source
    files.

    You cannot build extension modules with this function, that is,
    using ``-c`` is not allowed. Use the ``compile`` command instead.

    Examples
    --------
    The command ``f2py -m scalar scalar.f`` can be executed from Python as
    follows.

    .. literalinclude:: ../../source/f2py/code/results/run_main_session.dat
        :language: python

    """
    crackfortran.reset_global_f2py_vars()
    f2pydir = os.path.dirname(os.path.abspath(cfuncs.__file__))
    fobjhsrc = os.path.join(f2pydir, 'src', 'fortranobject.h')
    fobjcsrc = os.path.join(f2pydir, 'src', 'fortranobject.c')
    # gh-22819 -- begin
    parser = make_f2py_compile_parser()
    args, comline_list = parser.parse_known_args(comline_list)
    pyf_files, _ = filter_files("", "[.]pyf([.]src|)", comline_list)
    # Checks that no existing modulename is defined in a pyf file
    # TODO: Remove all this when scaninputline is replaced
    if args.module_name:
        if "-h" in comline_list:
            modname = (
                args.module_name
            )  # Directly use from args when -h is present
        else:
            modname = validate_modulename(
                pyf_files, args.module_name
            )  # Validate modname when -h is not present
        comline_list += ['-m', modname]  # needed for the rest of scaninputline
    # gh-22819 -- end
    files, options = scaninputline(comline_list)
    auxfuncs.options = options
    capi_maps.load_f2cmap_file(options['f2cmap_file'])
    postlist = callcrackfortran(files, options)
    isusedby = {}
    for plist in postlist:
        if 'use' in plist:
            for u in plist['use'].keys():
                if u not in isusedby:
                    isusedby[u] = []
                isusedby[u].append(plist['name'])
    for plist in postlist:
        if plist['block'] == 'python module' and '__user__' in plist['name']:
            if plist['name'] in isusedby:
                # if not quiet:
                outmess(
                    f'Skipping Makefile build for module "{plist["name"]}" '
                    'which is used by {}\n'.format(
                        ','.join(f'"{s}"' for s in isusedby[plist['name']])))
    if 'signsfile' in options:
        if options['verbose'] > 1:
            outmess(
                'Stopping. Edit the signature file and then run f2py on the signature file: ')
            outmess('%s %s\n' %
                    (os.path.basename(sys.argv[0]), options['signsfile']))
        return
    for plist in postlist:
        if plist['block'] != 'python module':
            if 'python module' not in options:
                errmess(
                    'Tip: If your original code is Fortran source then you must use -m option.\n')
            raise TypeError('All blocks must be python module blocks but got %s' % (
                repr(plist['block'])))
    auxfuncs.debugoptions = options['debug']
    f90mod_rules.options = options
    auxfuncs.wrapfuncs = options['wrapfuncs']

    ret = buildmodules(postlist)

    for mn in ret.keys():
        dict_append(ret[mn], {'csrc': fobjcsrc, 'h': fobjhsrc})
    return ret


def filter_files(prefix, suffix, files, remove_prefix=None):
    """
    Filter files by prefix and suffix.
    """
    filtered, rest = [], []
    match = re.compile(prefix + r'.*' + suffix + r'\Z').match
    if remove_prefix:
        ind = len(prefix)
    else:
        ind = 0
    for file in [x.strip() for x in files]:
        if match(file):
            filtered.append(file[ind:])
        else:
            rest.append(file)
    return filtered, rest


def get_prefix(module):
    p = os.path.dirname(os.path.dirname(module.__file__))
    return p


class CombineIncludePaths(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        include_paths_set = set(getattr(namespace, 'include_paths', []) or [])
        if option_string == "--include_paths":
            outmess("Use --include-paths or -I instead of --include_paths which will be removed")
        if option_string == "--include-paths" or option_string == "--include_paths":
            include_paths_set.update(values.split(':'))
        else:
            include_paths_set.add(values)
        namespace.include_paths = list(include_paths_set)

def f2py_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-I", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include-paths", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include_paths", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--freethreading-compatible", dest="ftcompat", action=argparse.BooleanOptionalAction)
    return parser

def get_newer_options(iline):
    iline = (' '.join(iline)).split()
    parser = f2py_parser()
    args, remain = parser.parse_known_args(iline)
    ipaths = args.include_paths
    if args.include_paths is None:
        ipaths = []
    return ipaths, args.ftcompat, remain

def make_f2py_compile_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dep", action="append", dest="dependencies")
    parser.add_argument("--backend", choices=['meson', 'distutils'], default='distutils')
    parser.add_argument("-m", dest="module_name")
    return parser

def preparse_sysargv():
    # To keep backwards bug compatibility, newer flags are handled by argparse,
    # and `sys.argv` is passed to the rest of `f2py` as is.
    parser = make_f2py_compile_parser()

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    backend_key = args.backend
    if MESON_ONLY_VER and backend_key == 'distutils':
        outmess("Cannot use distutils backend with Python>=3.12,"
                " using meson backend instead.\n")
        backend_key = "meson"

    return {
        "dependencies": args.dependencies or [],
        "backend": backend_key,
        "modulename": args.module_name,
    }

def run_compile():
    """
    Do it all in one call!
    """
    import tempfile

    # Collect dependency flags, preprocess sys.argv
    argy = preparse_sysargv()
    modulename = argy["modulename"]
    if modulename is None:
        modulename = 'untitled'
    dependencies = argy["dependencies"]
    backend_key = argy["backend"]
    build_backend = f2py_build_generator(backend_key)

    i = sys.argv.index('-c')
    del sys.argv[i]

    remove_build_dir = 0
    try:
        i = sys.argv.index('--build-dir')
    except ValueError:
        i = None
    if i is not None:
        build_dir = sys.argv[i + 1]
        del sys.argv[i + 1]
        del sys.argv[i]
    else:
        remove_build_dir = 1
        build_dir = tempfile.mkdtemp()

    _reg1 = re.compile(r'--link-')
    sysinfo_flags = [_m for _m in sys.argv[1:] if _reg1.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in sysinfo_flags]
    if sysinfo_flags:
        sysinfo_flags = [f[7:] for f in sysinfo_flags]

    _reg2 = re.compile(
        r'--((no-|)(wrap-functions|lower|freethreading-compatible)|debug-capi|quiet|skip-empty-wrappers)|-include')
    f2py_flags = [_m for _m in sys.argv[1:] if _reg2.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags]
    f2py_flags2 = []
    fl = 0
    for a in sys.argv[1:]:
        if a in ['only:', 'skip:']:
            fl = 1
        elif a == ':':
            fl = 0
        if fl or a == ':':
            f2py_flags2.append(a)
    if f2py_flags2 and f2py_flags2[-1] != ':':
        f2py_flags2.append(':')
    f2py_flags.extend(f2py_flags2)
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags2]
    _reg3 = re.compile(
        r'--((f(90)?compiler(-exec|)|compiler)=|help-compiler)')
    flib_flags = [_m for _m in sys.argv[1:] if _reg3.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in flib_flags]
    # TODO: Once distutils is dropped completely, i.e. min_ver >= 3.12, unify into --fflags
    reg_f77_f90_flags = re.compile(r'--f(77|90)flags=')
    reg_distutils_flags = re.compile(r'--((f(77|90)exec|opt|arch)=|(debug|noopt|noarch|help-fcompiler))')
    fc_flags = [_m for _m in sys.argv[1:] if reg_f77_f90_flags.match(_m)]
    distutils_flags = [_m for _m in sys.argv[1:] if reg_distutils_flags.match(_m)]
    if not (MESON_ONLY_VER or backend_key == 'meson'):
        fc_flags.extend(distutils_flags)
    sys.argv = [_m for _m in sys.argv if _m not in (fc_flags + distutils_flags)]

    del_list = []
    for s in flib_flags:
        v = '--fcompiler='
        if s[:len(v)] == v:
            if MESON_ONLY_VER or backend_key == 'meson':
                outmess(
                    "--fcompiler cannot be used with meson,"
                    "set compiler with the FC environment variable\n"
                    )
            else:
                from numpy.distutils import fcompiler
                fcompiler.load_all_fcompiler_classes()
                allowed_keys = list(fcompiler.fcompiler_class.keys())
                nv = ov = s[len(v):].lower()
                if ov not in allowed_keys:
                    vmap = {}  # XXX
                    try:
                        nv = vmap[ov]
                    except KeyError:
                        if ov not in vmap.values():
                            print('Unknown vendor: "%s"' % (s[len(v):]))
                    nv = ov
                i = flib_flags.index(s)
                flib_flags[i] = '--fcompiler=' + nv
                continue
    for s in del_list:
        i = flib_flags.index(s)
        del flib_flags[i]
    assert len(flib_flags) <= 2, repr(flib_flags)

    _reg5 = re.compile(r'--(verbose)')
    setup_flags = [_m for _m in sys.argv[1:] if _reg5.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in setup_flags]

    if '--quiet' in f2py_flags:
        setup_flags.append('--quiet')

    # Ugly filter to remove everything but sources
    sources = sys.argv[1:]
    f2cmapopt = '--f2cmap'
    if f2cmapopt in sys.argv:
        i = sys.argv.index(f2cmapopt)
        f2py_flags.extend(sys.argv[i:i + 2])
        del sys.argv[i + 1], sys.argv[i]
        sources = sys.argv[1:]

    pyf_files, _sources = filter_files("", "[.]pyf([.]src|)", sources)
    sources = pyf_files + _sources
    modulename = validate_modulename(pyf_files, modulename)
    extra_objects, sources = filter_files('', '[.](o|a|so|dylib)', sources)
    library_dirs, sources = filter_files('-L', '', sources, remove_prefix=1)
    libraries, sources = filter_files('-l', '', sources, remove_prefix=1)
    undef_macros, sources = filter_files('-U', '', sources, remove_prefix=1)
    define_macros, sources = filter_files('-D', '', sources, remove_prefix=1)
    for i in range(len(define_macros)):
        name_value = define_macros[i].split('=', 1)
        if len(name_value) == 1:
            name_value.append(None)
        if len(name_value) == 2:
            define_macros[i] = tuple(name_value)
        else:
            print('Invalid use of -D:', name_value)

    # Construct wrappers / signatures / things
    if backend_key == 'meson':
        if not pyf_files:
            outmess('Using meson backend\nWill pass --lower to f2py\nSee https://numpy.org/doc/stable/f2py/buildtools/meson.html\n')
            f2py_flags.append('--lower')
            run_main(f" {' '.join(f2py_flags)} -m {modulename} {' '.join(sources)}".split())
        else:
            run_main(f" {' '.join(f2py_flags)} {' '.join(pyf_files)}".split())

    # Order matters here, includes are needed for run_main above
    include_dirs, _, sources = get_newer_options(sources)
    # Now use the builder
    builder = build_backend(
        modulename,
        sources,
        extra_objects,
        build_dir,
        include_dirs,
        library_dirs,
        libraries,
        define_macros,
        undef_macros,
        f2py_flags,
        sysinfo_flags,
        fc_flags,
        flib_flags,
        setup_flags,
        remove_build_dir,
        {"dependencies": dependencies},
    )

    builder.compile()


def validate_modulename(pyf_files, modulename='untitled'):
    if len(pyf_files) > 1:
        raise ValueError("Only one .pyf file per call")
    if pyf_files:
        pyff = pyf_files[0]
        pyf_modname = auxfuncs.get_f2py_modulename(pyff)
        if modulename != pyf_modname:
            outmess(
                f"Ignoring -m {modulename}.\n"
                f"{pyff} defines {pyf_modname} to be the modulename.\n"
            )
            modulename = pyf_modname
    return modulename

def main():
    if '--help-link' in sys.argv[1:]:
        sys.argv.remove('--help-link')
        if MESON_ONLY_VER:
            outmess("Use --dep for meson builds\n")
        else:
            from numpy.distutils.system_info import show_all
            show_all()
        return

    if '-c' in sys.argv[1:]:
        run_compile()
    else:
        run_main(sys.argv[1:])
<|endoftext|>"""

Rules for building C/API module with f2py2e.

Here is a skeleton of a new wrapper function (13Dec2001):

wrapper_function(args)
  declarations
  get_python_arguments, say, `a' and `b'

  get_a_from_python
  if (successful) {

    get_b_from_python
    if (successful) {

      callfortran
      if (successful) {

        put_a_to_python
        if (successful) {

          put_b_to_python
          if (successful) {

            buildvalue = ...

          }

        }

      }

    }
    cleanup_b

  }
  cleanup_a

  return buildvalue

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import os
import sys
import time
import copy
from pathlib import Path

# __version__.version is now the same as the NumPy version
from . import __version__

from .auxfuncs import (
    applyrules, debugcapi, dictappend, errmess, gentitle, getargs2,
    hascallstatement, hasexternals, hasinitvalue, hasnote,
    hasresultnote, isarray, isarrayofstrings, ischaracter,
    ischaracterarray, ischaracter_or_characterarray, iscomplex,
    iscomplexarray, iscomplexfunction, iscomplexfunction_warn,
    isdummyroutine, isexternal, isfunction, isfunction_wrap, isint1,
    isint1array, isintent_aux, isintent_c, isintent_callback,
    isintent_copy, isintent_hide, isintent_inout, isintent_nothide,
    isintent_out, isintent_overwrite, islogical, islong_complex,
    islong_double, islong_doublefunction, islong_long,
    islong_longfunction, ismoduleroutine, isoptional, isrequired,
    isscalar, issigned_long_longarray, isstring, isstringarray,
    isstringfunction, issubroutine, isattr_value,
    issubroutine_wrap, isthreadsafe, isunsigned, isunsigned_char,
    isunsigned_chararray, isunsigned_long_long,
    isunsigned_long_longarray, isunsigned_short, isunsigned_shortarray,
    l_and, l_not, l_or, outmess, replace, stripcomma, requiresf90wrapper
)

from . import capi_maps
from . import cfuncs
from . import common_rules
from . import use_rules
from . import f90mod_rules
from . import func2subr

f2py_version = __version__.version
numpy_version = __version__.version

options = {}
sepdict = {}
# for k in ['need_cfuncs']: sepdict[k]=','
for k in ['decl',
          'frompyobj',
          'cleanupfrompyobj',
          'topyarr', 'method',
          'pyobjfrom', 'closepyobjfrom',
          'freemem',
          'userincludes',
          'includes0', 'includes', 'typedefs', 'typedefs_generated',
          'cppmacros', 'cfuncs', 'callbacks',
          'latexdoc',
          'restdoc',
          'routine_defs', 'externroutines',
          'initf2pywraphooks',
          'commonhooks', 'initcommonhooks',
          'f90modhooks', 'initf90modhooks']:
    sepdict[k] = '\n'

#################### Rules for C/API module #################

generationtime = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
module_rules = {
    'modulebody': """\
/* File: #modulename#module.c
 * This file is auto-generated with f2py (version:#f2py_version#).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: """ + time.asctime(time.gmtime(generationtime)) + """
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern \"C\" {
#endif

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

/* Unconditionally included */
#include <Python.h>
#include <numpy/npy_os.h>

""" + gentitle("See f2py2e/cfuncs.py: includes") + """
#includes#
#includes0#

""" + gentitle("See f2py2e/rules.py: mod_rules['modulebody']") + """
static PyObject *#modulename#_error;
static PyObject *#modulename#_module;

""" + gentitle("See f2py2e/cfuncs.py: typedefs") + """
#typedefs#

""" + gentitle("See f2py2e/cfuncs.py: typedefs_generated") + """
#typedefs_generated#

""" + gentitle("See f2py2e/cfuncs.py: cppmacros") + """
#cppmacros#

""" + gentitle("See f2py2e/cfuncs.py: cfuncs") + """
#cfuncs#

""" + gentitle("See f2py2e/cfuncs.py: userincludes") + """
#userincludes#

""" + gentitle("See f2py2e/capi_rules.py: usercode") + """
#usercode#

/* See f2py2e/rules.py */
#externroutines#

""" + gentitle("See f2py2e/capi_rules.py: usercode1") + """
#usercode1#

""" + gentitle("See f2py2e/cb_rules.py: buildcallback") + """
#callbacks#

""" + gentitle("See f2py2e/rules.py: buildapi") + """
#body#

""" + gentitle("See f2py2e/f90mod_rules.py: buildhooks") + """
#f90modhooks#

""" + gentitle("See f2py2e/rules.py: module_rules['modulebody']") + """

""" + gentitle("See f2py2e/common_rules.py: buildhooks") + """
#commonhooks#

""" + gentitle("See f2py2e/rules.py") + """

static FortranDataDef f2py_routine_defs[] = {
#routine_defs#
    {NULL}
};

static PyMethodDef f2py_module_methods[] = {
#pymethoddef#
    {NULL,NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "#modulename#",
    NULL,
    -1,
    f2py_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_#modulename#(void) {
    int i;
    PyObject *m,*d, *s, *tmp;
    m = #modulename#_module = PyModule_Create(&moduledef);
    Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
    import_array();
    if (PyErr_Occurred())
        {PyErr_SetString(PyExc_ImportError, \"can't initialize module #modulename# (failed to import numpy)\"); return m;}
    d = PyModule_GetDict(m);
    s = PyUnicode_FromString(\"#f2py_version#\");
    PyDict_SetItemString(d, \"__version__\", s);
    Py_DECREF(s);
    s = PyUnicode_FromString(
        \"This module '#modulename#' is auto-generated with f2py (version:#f2py_version#).\\nFunctions:\\n\"\n#docs#\".\");
    PyDict_SetItemString(d, \"__doc__\", s);
    Py_DECREF(s);
    s = PyUnicode_FromString(\"""" + numpy_version + """\");
    PyDict_SetItemString(d, \"__f2py_numpy_version__\", s);
    Py_DECREF(s);
    #modulename#_error = PyErr_NewException (\"#modulename#.error\", NULL, NULL);
    /*
     * Store the error object inside the dict, so that it could get deallocated.
     * (in practice, this is a module, so it likely will not and cannot.)
     */
    PyDict_SetItemString(d, \"_#modulename#_error\", #modulename#_error);
    Py_DECREF(#modulename#_error);
    for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
        tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
        PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
        Py_DECREF(tmp);
    }
#initf2pywraphooks#
#initf90modhooks#
#initcommonhooks#
#interface_usercode#

#if Py_GIL_DISABLED
    // signal whether this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(m , #gil_used#);
#endif

#ifdef F2PY_REPORT_ATEXIT
    if (! PyErr_Occurred())
        on_exit(f2py_report_on_exit,(void*)\"#modulename#\");
#endif

    if (PyType_Ready(&PyFortran_Type) < 0) {
        return NULL;
    }

    return m;
}
#ifdef __cplusplus
}
#endif
""",
    'separatorsfor': {'latexdoc': '\n\n',
                      'restdoc': '\n\n'},
    'latexdoc': ['\\section{Module \\texttt{#texmodulename#}}\n',
                 '#modnote#\n',
                 '#latexdoc#'],
    'restdoc': ['Module #modulename#\n' + '=' * 80,
                '\n#restdoc#']
}

defmod_rules = [
    {'body': '/*eof body*/',
     'method': '/*eof method*/',
     'externroutines': '/*eof externroutines*/',
     'routine_defs': '/*eof routine_defs*/',
     'initf90modhooks': '/*eof initf90modhooks*/',
     'initf2pywraphooks': '/*eof initf2pywraphooks*/',
     'initcommonhooks': '/*eof initcommonhooks*/',
     'latexdoc': '',
     'restdoc': '',
     'modnote': {hasnote: '#note#', l_not(hasnote): ''},
     }
]

routine_rules = {
    'separatorsfor': sepdict,
    'body': """
#begintitle#
static char doc_#apiname#[] = \"\\\n#docreturn##name#(#docsignatureshort#)\\n\\nWrapper for ``#name#``.\\\n\\n#docstrsigns#\";
/* #declfortranroutine# */
static PyObject *#apiname#(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           #functype# (*f2py_func)(#callprotoargument#)) {
    PyObject * volatile capi_buildvalue = NULL;
    volatile int f2py_success = 1;
#decl#
    static char *capi_kwlist[] = {#kwlist##kwlistopt##kwlistxa#NULL};
#usercode#
#routdebugenter#
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\\
        \"#argformat#|#keyformat##xaformat#:#pyname#\",\\
        capi_kwlist#args_capi##keys_capi##keys_xa#))\n        return NULL;
#frompyobj#
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
#callfortranroutine#
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
        if (f2py_success) {
#pyobjfrom#
/*end of pyobjfrom*/
        CFUNCSMESS(\"Building return value.\\n\");
        capi_buildvalue = Py_BuildValue(\"#returnformat#\"#return#);
/*closepyobjfrom*/
#closepyobjfrom#
        } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
#cleanupfrompyobj#
    if (capi_buildvalue == NULL) {
#routdebugfailure#
    } else {
#routdebugleave#
    }
    CFUNCSMESS(\"Freeing memory.\\n\");
#freemem#
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
    return capi_buildvalue;
}
#endtitle#
""",
    'routine_defs': '#routine_def#',
    'initf2pywraphooks': '#initf2pywraphook#',
    'externroutines': '#declfortranroutine#',
    'doc': '#docreturn##name#(#docsignature#)',
    'docshort': '#docreturn##name#(#docsignatureshort#)',
    'docs': '"    #docreturn##name#(#docsignature#)\\n"\n',
    'need': ['arrayobject.h', 'CFUNCSMESS', 'MINMAX'],
    'cppmacros': {debugcapi: '#define DEBUGCFUNCS'},
    'latexdoc': ['\\subsection{Wrapper function \\texttt{#texname#}}\n',
                 """
\\noindent{{}\\verb@#docreturn##name#@{}}\\texttt{(#latexdocsignatureshort#)}
#routnote#

#latexdocstrsigns#
"""],
    'restdoc': ['Wrapped function ``#name#``\n' + '-' * 80,

                ]
}

################## Rules for C/API function ##############

rout_rules = [
    {  # Init
        'separatorsfor': {'callfortranroutine': '\n', 'routdebugenter': '\n', 'decl': '\n',
                          'routdebugleave': '\n', 'routdebugfailure': '\n',
                          'setjmpbuf': ' || ',
                          'docstrreq': '\n', 'docstropt': '\n', 'docstrout': '\n',
                          'docstrcbs': '\n', 'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        'kwlist': '', 'kwlistopt': '', 'callfortran': '', 'callfortranappend': '',
        'docsign': '', 'docsignopt': '', 'decl': '/*decl*/',
        'freemem': '/*freemem*/',
        'docsignshort': '', 'docsignoptshort': '',
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '\\nParameters\\n----------',
        'docstropt': '\\nOther Parameters\\n----------------',
        'docstrout': '\\nReturns\\n-------',
        'docstrcbs': '\\nNotes\\n-----\\nCall-back functions::\\n',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
        'args_capi': '', 'keys_capi': '', 'functype': '',
        'frompyobj': '/*frompyobj*/',
        # this list will be reversed
        'cleanupfrompyobj': ['/*end of cleanupfrompyobj*/'],
        'pyobjfrom': '/*pyobjfrom*/',
        # this list will be reversed
        'closepyobjfrom': ['/*end of closepyobjfrom*/'],
        'topyarr': '/*topyarr*/', 'routdebugleave': '/*routdebugleave*/',
        'routdebugenter': '/*routdebugenter*/',
        'routdebugfailure': '/*routdebugfailure*/',
        'callfortranroutine': '/*callfortranroutine*/',
        'argformat': '', 'keyformat': '', 'need_cfuncs': '',
        'docreturn': '', 'return': '', 'returnformat': '', 'rformat': '',
        'kwlistxa': '', 'keys_xa': '', 'xaformat': '', 'docsignxa': '', 'docsignxashort': '',
        'initf2pywraphook': '',
        'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
    }, {
        'apiname': 'f2py_rout_#modulename#_#name#',
        'pyname': '#modulename#.#name#',
        'decl': '',
        '_check': l_not(ismoduleroutine)
    }, {
        'apiname': 'f2py_rout_#modulename#_#f90modulename#_#name#',
        'pyname': '#modulename#.#f90modulename#.#name#',
        'decl': '',
        '_check': ismoduleroutine
    }, {  # Subroutine
        'functype': 'void',
        'declfortranroutine': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'extern void #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
                               l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)): 'extern void #fortranname#(#callprotoargument#);',
                               ismoduleroutine: '',
                               isdummyroutine: ''
                               },
        'routine_def': {
            l_not(l_or(ismoduleroutine, isintent_c, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_FUNC#(#fortranname#,#FORTRANNAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            l_and(l_not(ismoduleroutine), isdummyroutine):
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'need': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'F_FUNC'},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `#fortranname#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
        if (#setjmpbuf#) {
            f2py_success = 0;
        } else {"""},
            {isthreadsafe: '            Py_BEGIN_ALLOW_THREADS'},
            {hascallstatement: '''                #callstatement#;
                /*(*f2py_func)(#callfortran#);*/'''},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '                (*f2py_func)(#callfortran#);'},
            {isthreadsafe: '            Py_END_ALLOW_THREADS'},
            {hasexternals: """        }"""}
        ],
        '_check': l_and(issubroutine, l_not(issubroutine_wrap)),
    }, {  # Wrapped function
        'functype': 'void',
        'declfortranroutine': {l_not(l_or(ismoduleroutine, isdummyroutine)): 'extern void #F_WRAPPEDFUNC#(#name_lower#,#NAME#)(#callprotoargument#);',
                               isdummyroutine: '',
                               },

        'routine_def': {
            l_not(l_or(ismoduleroutine, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_WRAPPEDFUNC#(#name_lower#,#NAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'initf2pywraphook': {l_not(l_or(ismoduleroutine, isdummyroutine)): '''
    {
      extern #ctype# #F_FUNC#(#name_lower#,#NAME#)(void);
      PyObject* o = PyDict_GetItemString(d,"#name#");
      tmp = F2PyCapsule_FromVoidPtr((void*)#F_WRAPPEDFUNC#(#name_lower#,#NAME#),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
      s = PyUnicode_FromString("#name#");
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    '''},
        'need': {l_not(l_or(ismoduleroutine, isdummyroutine)): ['F_WRAPPEDFUNC', 'F_FUNC']},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `f2pywrap#name_lower#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    (*f2py_func)(#callfortran#);'},
            {hascallstatement:
                '    #callstatement#;\n    /*(*f2py_func)(#callfortran#);*/'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'}
        ],
        '_check': isfunction_wrap,
    }, {  # Wrapped subroutine
        'functype': 'void',
        'declfortranroutine': {l_not(l_or(ismoduleroutine, isdummyroutine)): 'extern void #F_WRAPPEDFUNC#(#name_lower#,#NAME#)(#callprotoargument#);',
                               isdummyroutine: '',
                               },

        'routine_def': {
            l_not(l_or(ismoduleroutine, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_WRAPPEDFUNC#(#name_lower#,#NAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'initf2pywraphook': {l_not(l_or(ismoduleroutine, isdummyroutine)): '''
    {
      extern void #F_FUNC#(#name_lower#,#NAME#)(void);
      PyObject* o = PyDict_GetItemString(d,"#name#");
      tmp = F2PyCapsule_FromVoidPtr((void*)#F_FUNC#(#name_lower#,#NAME#),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
      s = PyUnicode_FromString("#name#");
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    '''},
        'need': {l_not(l_or(ismoduleroutine, isdummyroutine)): ['F_WRAPPEDFUNC', 'F_FUNC']},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `f2pywrap#name_lower#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    (*f2py_func)(#callfortran#);'},
            {hascallstatement:
                '    #callstatement#;\n    /*(*f2py_func)(#callfortran#);*/'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'}
        ],
        '_check': issubroutine_wrap,
    }, {  # Function
        'functype': '#ctype#',
        'docreturn': {l_not(isintent_hide): '#rname#,'},
        'docstrout': '#pydocsignout#',
        'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                           {hasresultnote: '--- #resultnote#'}],
        'callfortranroutine': [{l_and(debugcapi, isstringfunction): """\
#ifdef USESCOMPAQFORTRAN
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callcompaqfortran#)\\n\");
#else
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callfortran#)\\n\");
#endif
"""},
                               {l_and(debugcapi, l_not(isstringfunction)): """\
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callfortran#)\\n\");
"""}
                               ],
        '_check': l_and(isfunction, l_not(isfunction_wrap))
    }, {  # Scalar function
        'declfortranroutine': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'extern #ctype# #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
                               l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)): 'extern #ctype# #fortranname#(#callprotoargument#);',
                               isdummyroutine: ''
                               },
        'routine_def': {
            l_and(l_not(l_or(ismoduleroutine, isintent_c)),
                  l_not(isdummyroutine)):
            ('    {\"#name#\",-1,{{-1}},0,0,(char *)'
             '  #F_FUNC#(#fortranname#,#FORTRANNAME#),'
             '  (f2py_init_func)#apiname#,doc_#apiname#},'),
            l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)):
            ('    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,'
             '  (f2py_init_func)#apiname#,doc_#apiname#},'),
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '(f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'decl': [{iscomplexfunction_warn: '    #ctype# #name#_return_value={0,0};',
                  l_not(iscomplexfunction): '    #ctype# #name#_return_value=0;'},
                 {iscomplexfunction:
                  '    PyObject *#name#_return_value_capi = Py_None;'}
                 ],
        'callfortranroutine': [
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {hascallstatement: '''    #callstatement#;
/*    #name#_return_value = (*f2py_func)(#callfortran#);*/
'''},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    #name#_return_value = (*f2py_func)(#callfortran#);'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'},
            {l_and(debugcapi, iscomplexfunction)
                   : '    fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value.r,#name#_return_value.i);'},
            {l_and(debugcapi, l_not(iscomplexfunction)): '    fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value);'}],
        'pyobjfrom': {iscomplexfunction: '    #name#_return_value_capi = pyobj_from_#ctype#1(#name#_return_value);'},
        'need': [{l_not(isdummyroutine): 'F_FUNC'},
                 {iscomplexfunction: 'pyobj_from_#ctype#1'},
                 {islong_longfunction: 'long_long'},
                 {islong_doublefunction: 'long_double'}],
        'returnformat': {l_not(isintent_hide): '#rformat#'},
        'return': {iscomplexfunction: ',#name#_return_value_capi',
                   l_not(l_or(iscomplexfunction, isintent_hide)): ',#name#_return_value'},
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(isfunction_wrap))
    }, {  # String function # in use for --no-wrap
        'declfortranroutine': 'extern void #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
        'routine_def': {l_not(l_or(ismoduleroutine, isintent_c)):
                        '    {\"#name#\",-1,{{-1}},0,0,(char *)#F_FUNC#(#fortranname#,#FORTRANNAME#),(f2py_init_func)#apiname#,doc_#apiname#},',
                        l_and(l_not(ismoduleroutine), isintent_c):
                        '    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,(f2py_init_func)#apiname#,doc_#apiname#},'
                        },
        'decl': ['    #ctype# #name#_return_value = NULL;',
                 '    int #name#_return_value_len = 0;'],
        'callfortran':'#name#_return_value,#name#_return_value_len,',
        'callfortranroutine':['    #name#_return_value_len = #rlength#;',
                              '    if ((#name#_return_value = (string)malloc('
                              + '#name#_return_value_len+1) == NULL) {',
                              '        PyErr_SetString(PyExc_MemoryError, \"out of memory\");',
                              '        f2py_success = 0;',
                              '    } else {',
                              "        (#name#_return_value)[#name#_return_value_len] = '\\0';",
                              '    }',
                              '    if (f2py_success) {',
                              {hasexternals: """\
        if (#setjmpbuf#) {
            f2py_success = 0;
        } else {"""},
                              {isthreadsafe: '        Py_BEGIN_ALLOW_THREADS'},
                              """\
#ifdef USESCOMPAQFORTRAN
        (*f2py_func)(#callcompaqfortran#);
#else
        (*f2py_func)(#callfortran#);
#endif
""",
                              {isthreadsafe: '        Py_END_ALLOW_THREADS'},
                              {hasexternals: '        }'},
                              {debugcapi:
                                  '        fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value_len,#name#_return_value);'},
                              '    } /* if (f2py_success) after (string)malloc */',
                              ],
        'returnformat': '#rformat#',
        'return': ',#name#_return_value',
        'freemem': '    STRINGFREE(#name#_return_value);',
        'need': ['F_FUNC', '#ctype#', 'STRINGFREE'],
        '_check':l_and(isstringfunction, l_not(isfunction_wrap))  # ???obsolete
    },
    {  # Debugging
        'routdebugenter': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#(#docsignature#)\\n");',
        'routdebugleave': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#: successful.\\n");',
        'routdebugfailure': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#: failure.\\n");',
        '_check': debugcapi
    }
]

################ Rules for arguments ##################

typedef_need_dict = {islong_long: 'long_long',
                     islong_double: 'long_double',
                     islong_complex: 'complex_long_double',
                     isunsigned_char: 'unsigned_char',
                     isunsigned_short: 'unsigned_short',
                     isunsigned: 'unsigned',
                     isunsigned_long_long: 'unsigned_long_long',
                     isunsigned_chararray: 'unsigned_char',
                     isunsigned_shortarray: 'unsigned_short',
                     isunsigned_long_longarray: 'unsigned_long_long',
                     issigned_long_longarray: 'long_long',
                     isint1: 'signed_char',
                     ischaracter_or_characterarray: 'character',
                     }

aux_rules = [
    {
        'separatorsfor': sepdict
    },
    {  # Common
        'frompyobj': ['    /* Processing auxiliary variable #varname# */',
                      {debugcapi: '    fprintf(stderr,"#vardebuginfo#\\n");'}, ],
        'cleanupfrompyobj': '    /* End of cleaning variable #varname# */',
        'need': typedef_need_dict,
    },
    # Scalars (not complex)
    {  # Common
        'decl': '    #ctype# #varname# = 0;',
        'need': {hasinitvalue: 'math.h'},
        'frompyobj': {hasinitvalue: '    #varname# = #init#;'},
        '_check': l_and(isscalar, l_not(iscomplex)),
    },
    {
        'return': ',#varname#',
        'docstrout': '#pydocsignout#',
        'docreturn': '#outvarname#,',
        'returnformat': '#varrformat#',
        '_check': l_and(isscalar, l_not(iscomplex), isintent_out),
    },
    # Complex scalars
    {  # Common
        'decl': '    #ctype# #varname#;',
        'frompyobj': {hasinitvalue: '    #varname#.r = #init.r#, #varname#.i = #init.i#;'},
        '_check': iscomplex
    },
    # String
    {  # Common
        'decl': ['    #ctype# #varname# = NULL;',
                 '    int slen(#varname#);',
                 ],
        'need':['len..'],
        '_check':isstring
    },
    # Array
    {  # Common
        'decl': ['    #ctype# *#varname# = NULL;',
                 '    npy_intp #varname#_Dims[#rank#] = {#rank*[-1]#};',
                 '    const int #varname#_Rank = #rank#;',
                 ],
        'need':['len..', {hasinitvalue: 'forcomb'}, {hasinitvalue: 'CFUNCSMESS'}],
        '_check': isarray
    },
    # Scalararray
    {  # Common
        '_check': l_and(isarray, l_not(iscomplexarray))
    }, {  # Not hidden
        '_check': l_and(isarray, l_not(iscomplexarray), isintent_nothide)
    },
    # Integer*1 array
    {'need': '#ctype#',
     '_check': isint1array,
     '_depend': ''
     },
    # Integer*-1 array
    {'need': '#ctype#',
     '_check': l_or(isunsigned_chararray, isunsigned_char),
     '_depend': ''
     },
    # Integer*-2 array
    {'need': '#ctype#',
     '_check': isunsigned_shortarray,
     '_depend': ''
     },
    # Integer*-8 array
    {'need': '#ctype#',
     '_check': isunsigned_long_longarray,
     '_depend': ''
     },
    # Complexarray
    {'need': '#ctype#',
     '_check': iscomplexarray,
     '_depend': ''
     },
    # Stringarray
    {
        'callfortranappend': {isarrayofstrings: 'flen(#varname#),'},
        'need': 'string',
        '_check': isstringarray
    }
]

arg_rules = [
    {
        'separatorsfor': sepdict
    },
    {  # Common
        'frompyobj': ['    /* Processing variable #varname# */',
                      {debugcapi: '    fprintf(stderr,"#vardebuginfo#\\n");'}, ],
        'cleanupfrompyobj': '    /* End of cleaning variable #varname# */',
        '_depend': '',
        'need': typedef_need_dict,
    },
    # Doc signatures
    {
        'docstropt': {l_and(isoptional, isintent_nothide): '#pydocsign#'},
        'docstrreq': {l_and(isrequired, isintent_nothide): '#pydocsign#'},
        'docstrout': {isintent_out: '#pydocsignout#'},
        'latexdocstropt': {l_and(isoptional, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrreq': {l_and(isrequired, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrout': {isintent_out: ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                                          {l_and(hasnote, isintent_hide): '--- #note#',
                                           l_and(hasnote, isintent_nothide): '--- See above.'}]},
        'depend': ''
    },
    # Required/Optional arguments
    {
        'kwlist': '"#varname#",',
        'docsign': '#varname#,',
        '_check': l_and(isintent_nothide, l_not(isoptional))
    },
    {
        'kwlistopt': '"#varname#",',
        'docsignopt': '#varname#=#showinit#,',
        'docsignoptshort': '#varname#,',
        '_check': l_and(isintent_nothide, isoptional)
    },
    # Docstring/BuildValue
    {
        'docreturn': '#outvarname#,',
        'returnformat': '#varrformat#',
        '_check': isintent_out
    },
    # Externals (call-back functions)
    {  # Common
        'docsignxa': {isintent_nothide: '#varname#_extra_args=(),'},
        'docsignxashort': {isintent_nothide: '#varname#_extra_args,'},
        'docstropt': {isintent_nothide: '#varname#_extra_args : input tuple, optional\\n    Default: ()'},
        'docstrcbs': '#cbdocstr#',
        'latexdocstrcbs': '\\item[] #cblatexdocstr#',
        'latexdocstropt': {isintent_nothide: '\\item[]{{}\\verb@#varname#_extra_args := () input tuple@{}} --- Extra arguments for call-back function {{}\\verb@#varname#@{}}.'},
        'decl': ['    #cbname#_t #varname#_cb = { Py_None, NULL, 0 };',
                 '    #cbname#_t *#varname#_cb_ptr = &#varname#_cb;',
                 '    PyTupleObject *#varname#_xa_capi = NULL;',
                 {l_not(isintent_callback):
                  '    #cbname#_typedef #varname#_cptr;'}
                 ],
        'kwlistxa': {isintent_nothide: '"#varname#_extra_args",'},
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'xaformat': {isintent_nothide: 'O!'},
        'args_capi': {isrequired: ',&#varname#_cb.capi'},
        'keys_capi': {isoptional: ',&#varname#_cb.capi'},
        'keys_xa': ',&PyTuple_Type,&#varname#_xa_capi',
        'setjmpbuf': '(setjmp(#varname#_cb.jmpbuf))',
        'callfortran': {l_not(isintent_callback): '#varname#_cptr,'},
        'need': ['#cbname#', 'setjmp.h'],
        '_check':isexternal
    },
    {
        'frompyobj': [{l_not(isintent_callback): """\
if(F2PyCapsule_Check(#varname#_cb.capi)) {
  #varname#_cptr = F2PyCapsule_AsVoidPtr(#varname#_cb.capi);
} else {
  #varname#_cptr = #cbname#;
}
"""}, {isintent_callback: """\
if (#varname#_cb.capi==Py_None) {
  #varname#_cb.capi = PyObject_GetAttrString(#modulename#_module,\"#varname#\");
  if (#varname#_cb.capi) {
    if (#varname#_xa_capi==NULL) {
      if (PyObject_HasAttrString(#modulename#_module,\"#varname#_extra_args\")) {
        PyObject* capi_tmp = PyObject_GetAttrString(#modulename#_module,\"#varname#_extra_args\");
        if (capi_tmp) {
          #varname#_xa_capi = (PyTupleObject *)PySequence_Tuple(capi_tmp);
          Py_DECREF(capi_tmp);
        }
        else {
          #varname#_xa_capi = (PyTupleObject *)Py_BuildValue(\"()\");
        }
        if (#varname#_xa_capi==NULL) {
          PyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#varname#_extra_args to tuple.\\n\");
          return NULL;
        }
      }
    }
  }
  if (#varname#_cb.capi==NULL) {
    PyErr_SetString(#modulename#_error,\"Callback #varname# not defined (as an argument or module #modulename# attribute).\\n\");
    return NULL;
  }
}
"""},
            """\
    if (create_cb_arglist(#varname#_cb.capi,#varname#_xa_capi,#maxnofargs#,#nofoptargs#,&#varname#_cb.nofargs,&#varname#_cb.args_capi,\"failed in processing argument list for call-back #varname#.\")) {
""",
            {debugcapi: ["""\
        fprintf(stderr,\"debug-capi:Assuming %d arguments; at most #maxnofargs#(-#nofoptargs#) is expected.\\n\",#varname#_cb.nofargs);
        CFUNCSMESSPY(\"for #varname#=\",#varname#_cb.capi);""",
                         {l_not(isintent_callback): """        fprintf(stderr,\"#vardebugshowvalue# (call-back in C).\\n\",#cbname#);"""}]},
            """\
        CFUNCSMESS(\"Saving callback variables for `#varname#`.\\n\");
        #varname#_cb_ptr = swap_active_#cbname#(#varname#_cb_ptr);""",
        ],
        'cleanupfrompyobj':
        """\
        CFUNCSMESS(\"Restoring callback variables for `#varname#`.\\n\");
        #varname#_cb_ptr = swap_active_#cbname#(#varname#_cb_ptr);
        Py_DECREF(#varname#_cb.args_capi);
    }""",
        'need': ['SWAP', 'create_cb_arglist'],
        '_check':isexternal,
        '_depend':''
    },
    # Scalars (not complex)
    {  # Common
        'decl': '    #ctype# #varname# = 0;',
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#);'},
        'callfortran': {l_or(isintent_c, isattr_value): '#varname#,', l_not(l_or(isintent_c, isattr_value)): '&#varname#,'},
        'return': {isintent_out: ',#varname#'},
        '_check': l_and(isscalar, l_not(iscomplex))
    }, {
        'need': {hasinitvalue: 'math.h'},
        '_check': l_and(isscalar, l_not(iscomplex)),
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'pyobjfrom': {isintent_inout: """\
    f2py_success = try_pyarr_from_#ctype#(#varname#_capi,&#varname#);
    if (f2py_success) {"""},
        'closepyobjfrom': {isintent_inout: "    } /*if (f2py_success) of #varname# pyobjfrom*/"},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#'},
        '_check': l_and(isscalar, l_not(iscomplex), l_not(isstring),
                        isintent_nothide)
    }, {
        'frompyobj': [
            # hasinitvalue...
            #   if pyobj is None:
            #     varname = init
            #   else
            #     from_pyobj(varname)
            #
            # isoptional and noinitvalue...
            #   if pyobj is not None:
            #     from_pyobj(varname)
            #   else:
            #     varname is uninitialized
            #
            # ...
            #   from_pyobj(varname)
            #
            {hasinitvalue: '    if (#varname#_capi == Py_None) #varname# = #init#; else',
             '_depend': ''},
            {l_and(isoptional, l_not(hasinitvalue)): '    if (#varname#_capi != Py_None)',
             '_depend': ''},
            {l_not(islogical): '''\
        f2py_success = #ctype#_from_pyobj(&#varname#,#varname#_capi,"#pyname#() #nth# (#varname#) can\'t be converted to #ctype#");
    if (f2py_success) {'''},
            {islogical: '''\
        #varname# = (#ctype#)PyObject_IsTrue(#varname#_capi);
        f2py_success = 1;
    if (f2py_success) {'''},
        ],
        'cleanupfrompyobj': '    } /*if (f2py_success) of #varname#*/',
        'need': {l_not(islogical): '#ctype#_from_pyobj'},
        '_check': l_and(isscalar, l_not(iscomplex), isintent_nothide),
        '_depend': ''
    }, {  # Hidden
        'frompyobj': {hasinitvalue: '    #varname# = #init#;'},
        'need': typedef_need_dict,
        '_check': l_and(isscalar, l_not(iscomplex), isintent_hide),
        '_depend': ''
    }, {  # Common
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#);'},
        '_check': l_and(isscalar, l_not(iscomplex)),
        '_depend': ''
    },
    # Complex scalars
    {  # Common
        'decl': '    #ctype# #varname#;',
        'callfortran': {isintent_c: '#varname#,', l_not(isintent_c): '&#varname#,'},
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#.r,#varname#.i);'},
        'return': {isintent_out: ',#varname#_capi'},
        '_check': iscomplex
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#'},
        'pyobjfrom': {isintent_inout: """\
        f2py_success = try_pyarr_from_#ctype#(#varname#_capi,&#varname#);
        if (f2py_success) {"""},
        'closepyobjfrom': {isintent_inout: "        } /*if (f2py_success) of #varname# pyobjfrom*/"},
        '_check': l_and(iscomplex, isintent_nothide)
    }, {
        'frompyobj': [{hasinitvalue: '    if (#varname#_capi==Py_None) {#varname#.r = #init.r#, #varname#.i = #init.i#;} else'},
                      {l_and(isoptional, l_not(hasinitvalue))
                             : '    if (#varname#_capi != Py_None)'},
                      '        f2py_success = #ctype#_from_pyobj(&#varname#,#varname#_capi,"#pyname#() #nth# (#varname#) can\'t be converted to #ctype#");'
                      '\n    if (f2py_success) {'],
        'cleanupfrompyobj': '    }  /*if (f2py_success) of #varname# frompyobj*/',
        'need': ['#ctype#_from_pyobj'],
        '_check': l_and(iscomplex, isintent_nothide),
        '_depend': ''
    }, {  # Hidden
        'decl': {isintent_out: '    PyObject *#varname#_capi = Py_None;'},
        '_check': l_and(iscomplex, isintent_hide)
    }, {
        'frompyobj': {hasinitvalue: '    #varname#.r = #init.r#, #varname#.i = #init.i#;'},
        '_check': l_and(iscomplex, isintent_hide),
        '_depend': ''
    }, {  # Common
        'pyobjfrom': {isintent_out: '    #varname#_capi = pyobj_from_#ctype#1(#varname#);'},
        'need': ['pyobj_from_#ctype#1'],
        '_check': iscomplex
    }, {
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#.r,#varname#.i);'},
        '_check': iscomplex,
        '_depend': ''
    },
    # String
    {  # Common
        'decl': ['    #ctype# #varname# = NULL;',
                 '    int slen(#varname#);',
                 '    PyObject *#varname#_capi = Py_None;'],
        'callfortran':'#varname#,',
        'callfortranappend':'slen(#varname#),',
        'pyobjfrom':[
            {debugcapi:
             '    fprintf(stderr,'
             '"#vardebugshowvalue#\\n",slen(#varname#),#varname#);'},
            # The trailing null value for Fortran is blank.
            {l_and(isintent_out, l_not(isintent_c)):
             "        STRINGPADN(#varname#, slen(#varname#), ' ', '\\0');"},
        ],
        'return': {isintent_out: ',#varname#'},
        'need': ['len..',
                 {l_and(isintent_out, l_not(isintent_c)): 'STRINGPADN'}],
        '_check': isstring
    }, {  # Common
        'frompyobj': [
            """\
    slen(#varname#) = #elsize#;
    f2py_success = #ctype#_from_pyobj(&#varname#,&slen(#varname#),#init#,"""
"""#varname#_capi,\"#ctype#_from_pyobj failed in converting #nth#"""
"""`#varname#\' of #pyname# to C #ctype#\");
    if (f2py_success) {""",
            # The trailing null value for Fortran is blank.
            {l_not(isintent_c):
             "        STRINGPADN(#varname#, slen(#varname#), '\\0', ' ');"},
        ],
        'cleanupfrompyobj': """\
        STRINGFREE(#varname#);
    }  /*if (f2py_success) of #varname#*/""",
        'need': ['#ctype#_from_pyobj', 'len..', 'STRINGFREE',
                 {l_not(isintent_c): 'STRINGPADN'}],
        '_check':isstring,
        '_depend':''
    }, {  # Not hidden
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'pyobjfrom': [
            {l_and(isintent_inout, l_not(isintent_c)):
             "        STRINGPADN(#varname#, slen(#varname#), ' ', '\\0');"},
            {isintent_inout: '''\
    f2py_success = try_pyarr_from_#ctype#(#varname#_capi, #varname#,
                                          slen(#varname#));
    if (f2py_success) {'''}],
        'closepyobjfrom': {isintent_inout: '    } /*if (f2py_success) of #varname# pyobjfrom*/'},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#',
                 l_and(isintent_inout, l_not(isintent_c)): 'STRINGPADN'},
        '_check': l_and(isstring, isintent_nothide)
    }, {  # Hidden
        '_check': l_and(isstring, isintent_hide)
    }, {
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",slen(#varname#),#varname#);'},
        '_check': isstring,
        '_depend': ''
    },
    # Array
    {  # Common
        'decl': ['    #ctype# *#varname# = NULL;',
                 '    npy_intp #varname#_Dims[#rank#] = {#rank*[-1]#};',
                 '    const int #varname#_Rank = #rank#;',
                 '    PyArrayObject *capi_#varname#_as_array = NULL;',
                 '    int capi_#varname#_intent = 0;',
                 {isstringarray: '    int slen(#varname#) = 0;'},
                 ],
        'callfortran':'#varname#,',
        'callfortranappend': {isstringarray: 'slen(#varname#),'},
        'return': {isintent_out: ',capi_#varname#_as_array'},
        'need': 'len..',
        '_check': isarray
    }, {  # intent(overwrite) array
        'decl': '    int capi_overwrite_#varname# = 1;',
        'kwlistxa': '"overwrite_#varname#",',
        'xaformat': 'i',
        'keys_xa': ',&capi_overwrite_#varname#',
        'docsignxa': 'overwrite_#varname#=1,',
        'docsignxashort': 'overwrite_#varname#,',
        'docstropt': 'overwrite_#varname# : input int, optional\\n    Default: 1',
        '_check': l_and(isarray, isintent_overwrite),
    }, {
        'frompyobj': '    capi_#varname#_intent |= (capi_overwrite_#varname#?0:F2PY_INTENT_COPY);',
        '_check': l_and(isarray, isintent_overwrite),
        '_depend': '',
    },
    {  # intent(copy) array
        'decl': '    int capi_overwrite_#varname# = 0;',
        'kwlistxa': '"overwrite_#varname#",',
        'xaformat': 'i',
        'keys_xa': ',&capi_overwrite_#varname#',
        'docsignxa': 'overwrite_#varname#=0,',
        'docsignxashort': 'overwrite_#varname#,',
        'docstropt': 'overwrite_#varname# : input int, optional\\n    Default: 0',
        '_check': l_and(isarray, isintent_copy),
    }, {
        'frompyobj': '    capi_#varname#_intent |= (capi_overwrite_#varname#?0:F2PY_INTENT_COPY);',
        '_check': l_and(isarray, isintent_copy),
        '_depend': '',
    }, {
        'need': [{hasinitvalue: 'forcomb'}, {hasinitvalue: 'CFUNCSMESS'}],
        '_check': isarray,
        '_depend': ''
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        '_check': l_and(isarray, isintent_nothide)
    }, {
        'frompyobj': [
            '    #setdims#;',
            '    capi_#varname#_intent |= #intent#;',
            ('    const char * capi_errmess = "#modulename#.#pyname#:'
             ' failed to create array from the #nth# `#varname#`";'),
            {isintent_hide:
             '    capi_#varname#_as_array = ndarray_from_pyobj('
             '  #atype#,#elsize#,#varname#_Dims,#varname#_Rank,'
             '  capi_#varname#_intent,Py_None,capi_errmess);'},
            {isintent_nothide:
             '    capi_#varname#_as_array = ndarray_from_pyobj('
             '  #atype#,#elsize#,#varname#_Dims,#varname#_Rank,'
             '  capi_#varname#_intent,#varname#_capi,capi_errmess);'},
            """\
    if (capi_#varname#_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = #modulename#_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        #varname# = (#ctype# *)(PyArray_DATA(capi_#varname#_as_array));
""",
            {isstringarray:
             '    slen(#varname#) = f2py_itemsize(#varname#);'},
            {hasinitvalue: [
                {isintent_nothide:
                 '    if (#varname#_capi == Py_None) {'},
                {isintent_hide: '    {'},
                {iscomplexarray: '        #ctype# capi_c;'},
                """\
        int *_i,capi_i=0;
        CFUNCSMESS(\"#name#: Initializing #varname#=#init#\\n\");
        if (initforcomb(PyArray_DIMS(capi_#varname#_as_array),
                        PyArray_NDIM(capi_#varname#_as_array),1)) {
            while ((_i = nextforcomb()))
                #varname#[capi_i++] = #init#; /* fortran way */
        } else {
            PyObject *exc, *val, *tb;
            PyErr_Fetch(&exc, &val, &tb);
            PyErr_SetString(exc ? exc : #modulename#_error,
                \"Initialization of #nth# #varname# failed (initforcomb).\");
            npy_PyErr_ChainExceptionsCause(exc, val, tb);
            f2py_success = 0;
        }
    }
    if (f2py_success) {"""]},
                      ],
        'cleanupfrompyobj': [  # note that this list will be reversed
            '    }  '
            '/* if (capi_#varname#_as_array == NULL) ... else of #varname# */',
            {l_not(l_or(isintent_out, isintent_hide)): """\
    if((PyObject *)capi_#varname#_as_array!=#varname#_capi) {
        Py_XDECREF(capi_#varname#_as_array); }"""},
            {l_and(isintent_hide, l_not(isintent_out))
                   : """        Py_XDECREF(capi_#varname#_as_array);"""},
            {hasinitvalue: '    }  /*if (f2py_success) of #varname# init*/'},
        ],
        '_check': isarray,
        '_depend': ''
    },
    # Scalararray
    {  # Common
        '_check': l_and(isarray, l_not(iscomplexarray))
    }, {  # Not hidden
        '_check': l_and(isarray, l_not(iscomplexarray), isintent_nothide)
    },
    # Integer*1 array
    {'need': '#ctype#',
     '_check': isint1array,
     '_depend': ''
     },
    # Integer*-1 array
    {'need': '#ctype#',
     '_check': isunsigned_chararray,
     '_depend': ''
     },
    # Integer*-2 array
    {'need': '#ctype#',
     '_check': isunsigned_shortarray,
     '_depend': ''
     },
    # Integer*-8 array
    {'need': '#ctype#',
     '_check': isunsigned_long_longarray,
     '_depend': ''
     },
    # Complexarray
    {'need': '#ctype#',
     '_check': iscomplexarray,
     '_depend': ''
     },
    # Character
    {
        'need': 'string',
        '_check': ischaracter,
    },
    # Character array
    {
        'need': 'string',
        '_check': ischaracterarray,
    },
    # Stringarray
    {
        'callfortranappend': {isarrayofstrings: 'flen(#varname#),'},
        'need': 'string',
        '_check': isstringarray
    }
]

################# Rules for checking ###############

check_rules = [
    {
        'frompyobj': {debugcapi: '    fprintf(stderr,\"debug-capi:Checking `#check#\'\\n\");'},
        'need': 'len..'
    }, {
        'frompyobj': '    CHECKSCALAR(#check#,\"#check#\",\"#nth# #varname#\",\"#varshowvalue#\",#varname#) {',
        'cleanupfrompyobj': '    } /*CHECKSCALAR(#check#)*/',
        'need': 'CHECKSCALAR',
        '_check': l_and(isscalar, l_not(iscomplex)),
        '_break': ''
    }, {
        'frompyobj': '    CHECKSTRING(#check#,\"#check#\",\"#nth# #varname#\",\"#varshowvalue#\",#varname#) {',
        'cleanupfrompyobj': '    } /*CHECKSTRING(#check#)*/',
        'need': 'CHECKSTRING',
        '_check': isstring,
        '_break': ''
    }, {
        'need': 'CHECKARRAY',
        'frompyobj': '    CHECKARRAY(#check#,\"#check#\",\"#nth# #varname#\") {',
        'cleanupfrompyobj': '    } /*CHECKARRAY(#check#)*/',
        '_check': isarray,
        '_break': ''
    }, {
        'need': 'CHECKGENERIC',
        'frompyobj': '    CHECKGENERIC(#check#,\"#check#\",\"#nth# #varname#\") {',
        'cleanupfrompyobj': '    } /*CHECKGENERIC(#check#)*/',
    }
]

########## Applying the rules. No need to modify what follows #############

#################### Build C/API module #######################


def buildmodule(m, um):
    """
    Return
    """
    outmess('    Building module "%s"...\n' % (m['name']))
    ret = {}
    mod_rules = defmod_rules[:]
    vrd = capi_maps.modsign2map(m)
    rd = dictappend({'f2py_version': f2py_version}, vrd)
    funcwrappers = []
    funcwrappers2 = []  # F90 codes
    for n in m['interfaced']:
        nb = None
        for bi in m['body']:
            if bi['block'] not in ['interface', 'abstract interface']:
                errmess('buildmodule: Expected interface block. Skipping.\n')
                continue
            for b in bi['body']:
                if b['name'] == n:
                    nb = b
                    break

        if not nb:
            print(
                'buildmodule: Could not find the body of interfaced routine "%s". Skipping.\n' % (n), file=sys.stderr)
            continue
        nb_list = [nb]
        if 'entry' in nb:
            for k, a in nb['entry'].items():
                nb1 = copy.deepcopy(nb)
                del nb1['entry']
                nb1['name'] = k
                nb1['args'] = a
                nb_list.append(nb1)
        for nb in nb_list:
            # requiresf90wrapper must be called before buildapi as it
            # rewrites assumed shape arrays as automatic arrays.
            isf90 = requiresf90wrapper(nb)
            # options is in scope here
            if options['emptygen']:
                b_path = options['buildpath']
                m_name = vrd['modulename']
                outmess('    Generating possibly empty wrappers"\n')
                Path(f"{b_path}/{vrd['coutput']}").touch()
                if isf90:
                    # f77 + f90 wrappers
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers2.f90"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers2.f90').touch()
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers.f"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers.f').touch()
                else:
                    # only f77 wrappers
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers.f"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers.f').touch()
            api, wrap = buildapi(nb)
            if wrap:
                if isf90:
                    funcwrappers2.append(wrap)
                else:
                    funcwrappers.append(wrap)
            ar = applyrules(api, vrd)
            rd = dictappend(rd, ar)

    # Construct COMMON block support
    cr, wrap = common_rules.buildhooks(m)
    if wrap:
        funcwrappers.append(wrap)
    ar = applyrules(cr, vrd)
    rd = dictappend(rd, ar)

    # Construct F90 module support
    mr, wrap = f90mod_rules.buildhooks(m)
    if wrap:
        funcwrappers2.append(wrap)
    ar = applyrules(mr, vrd)
    rd = dictappend(rd, ar)

    for u in um:
        ar = use_rules.buildusevars(u, m['use'][u['name']])
        rd = dictappend(rd, ar)

    needs = cfuncs.get_needs()
    # Add mapped definitions
    needs['typedefs'] += [cvar for cvar in capi_maps.f2cmap_mapped #
                          if cvar in typedef_need_dict.values()]
    code = {}
    for n in needs.keys():
        code[n] = []
        for k in needs[n]:
            c = ''
            if k in cfuncs.includes0:
                c = cfuncs.includes0[k]
            elif k in cfuncs.includes:
                c = cfuncs.includes[k]
            elif k in cfuncs.userincludes:
                c = cfuncs.userincludes[k]
            elif k in cfuncs.typedefs:
                c = cfuncs.typedefs[k]
            elif k in cfuncs.typedefs_generated:
                c = cfuncs.typedefs_generated[k]
            elif k in cfuncs.cppmacros:
                c = cfuncs.cppmacros[k]
            elif k in cfuncs.cfuncs:
                c = cfuncs.cfuncs[k]
            elif k in cfuncs.callbacks:
                c = cfuncs.callbacks[k]
            elif k in cfuncs.f90modhooks:
                c = cfuncs.f90modhooks[k]
            elif k in cfuncs.commonhooks:
                c = cfuncs.commonhooks[k]
            else:
                errmess('buildmodule: unknown need %s.\n' % (repr(k)))
                continue
            code[n].append(c)
    mod_rules.append(code)
    for r in mod_rules:
        if ('_check' in r and r['_check'](m)) or ('_check' not in r):
            ar = applyrules(r, vrd, m)
            rd = dictappend(rd, ar)
    ar = applyrules(module_rules, rd)

    fn = os.path.join(options['buildpath'], vrd['coutput'])
    ret['csrc'] = fn
    with open(fn, 'w') as f:
        f.write(ar['modulebody'].replace('\t', 2 * ' '))
    outmess('    Wrote C/API module "%s" to file "%s"\n' % (m['name'], fn))

    if options['dorestdoc']:
        fn = os.path.join(
            options['buildpath'], vrd['modulename'] + 'module.rest')
        with open(fn, 'w') as f:
            f.write('.. -*- rest -*-\n')
            f.write('\n'.join(ar['restdoc']))
        outmess('    ReST Documentation is saved to file "%s/%smodule.rest"\n' %
                (options['buildpath'], vrd['modulename']))
    if options['dolatexdoc']:
        fn = os.path.join(
            options['buildpath'], vrd['modulename'] + 'module.tex')
        ret['ltx'] = fn
        with open(fn, 'w') as f:
            f.write(
                '%% This file is auto-generated with f2py (version:%s)\n' % (f2py_version))
            if 'shortlatex' not in options:
                f.write(
                    '\\documentclass{article}\n\\usepackage{a4wide}\n\\begin{document}\n\\tableofcontents\n\n')
                f.write('\n'.join(ar['latexdoc']))
            if 'shortlatex' not in options:
                f.write('\\end{document}')
        outmess('    Documentation is saved to file "%s/%smodule.tex"\n' %
                (options['buildpath'], vrd['modulename']))
    if funcwrappers:
        wn = os.path.join(options['buildpath'], vrd['f2py_wrapper_output'])
        ret['fsrc'] = wn
        with open(wn, 'w') as f:
            f.write('C     -*- fortran -*-\n')
            f.write(
                'C     This file is autogenerated with f2py (version:%s)\n' % (f2py_version))
            f.write(
                'C     It contains Fortran 77 wrappers to fortran functions.\n')
            lines = []
            for l in ('\n\n'.join(funcwrappers) + '\n').split('\n'):
                if 0 <= l.find('!') < 66:
                    # don't split comment lines
                    lines.append(l + '\n')
                elif l and l[0] == ' ':
                    while len(l) >= 66:
                        lines.append(l[:66] + '\n     &')
                        l = l[66:]
                    lines.append(l + '\n')
                else:
                    lines.append(l + '\n')
            lines = ''.join(lines).replace('\n     &\n', '\n')
            f.write(lines)
        outmess('    Fortran 77 wrappers are saved to "%s"\n' % (wn))
    if funcwrappers2:
        wn = os.path.join(
            options['buildpath'], '%s-f2pywrappers2.f90' % (vrd['modulename']))
        ret['fsrc'] = wn
        with open(wn, 'w') as f:
            f.write('!     -*- f90 -*-\n')
            f.write(
                '!     This file is autogenerated with f2py (version:%s)\n' % (f2py_version))
            f.write(
                '!     It contains Fortran 90 wrappers to fortran functions.\n')
            lines = []
            for l in ('\n\n'.join(funcwrappers2) + '\n').split('\n'):
                if 0 <= l.find('!') < 72:
                    # don't split comment lines
                    lines.append(l + '\n')
                elif len(l) > 72 and l[0] == ' ':
                    lines.append(l[:72] + '&\n     &')
                    l = l[72:]
                    while len(l) > 66:
                        lines.append(l[:66] + '&\n     &')
                        l = l[66:]
                    lines.append(l + '\n')
                else:
                    lines.append(l + '\n')
            lines = ''.join(lines).replace('\n     &\n', '\n')
            f.write(lines)
        outmess('    Fortran 90 wrappers are saved to "%s"\n' % (wn))
    return ret

################## Build C/API function #############

stnd = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th',
        6: 'th', 7: 'th', 8: 'th', 9: 'th', 0: 'th'}


def buildapi(rout):
    rout, wrap = func2subr.assubr(rout)
    args, depargs = getargs2(rout)
    capi_maps.depargs = depargs
    var = rout['vars']

    if ismoduleroutine(rout):
        outmess('            Constructing wrapper function "%s.%s"...\n' %
                (rout['modulename'], rout['name']))
    else:
        outmess('        Constructing wrapper function "%s"...\n' % (rout['name']))
    # Routine
    vrd = capi_maps.routsign2map(rout)
    rd = dictappend({}, vrd)
    for r in rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)

    # Args
    nth, nthk = 0, 0
    savevrd = {}
    for a in args:
        vrd = capi_maps.sign2map(a, var[a])
        if isintent_aux(var[a]):
            _rules = aux_rules
        else:
            _rules = arg_rules
            if not isintent_hide(var[a]):
                if not isoptional(var[a]):
                    nth = nth + 1
                    vrd['nth'] = repr(nth) + stnd[nth % 10] + ' argument'
                else:
                    nthk = nthk + 1
                    vrd['nth'] = repr(nthk) + stnd[nthk % 10] + ' keyword'
            else:
                vrd['nth'] = 'hidden'
        savevrd[a] = vrd
        for r in _rules:
            if '_depend' in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        if isintent_aux(var[a]):
            _rules = aux_rules
        else:
            _rules = arg_rules
        vrd = savevrd[a]
        for r in _rules:
            if '_depend' not in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
        if 'check' in var[a]:
            for c in var[a]['check']:
                vrd['check'] = c
                ar = applyrules(check_rules, vrd, var[a])
                rd = dictappend(rd, ar)
    if isinstance(rd['cleanupfrompyobj'], list):
        rd['cleanupfrompyobj'].reverse()
    if isinstance(rd['closepyobjfrom'], list):
        rd['closepyobjfrom'].reverse()
    rd['docsignature'] = stripcomma(replace('#docsign##docsignopt##docsignxa#',
                                            {'docsign': rd['docsign'],
                                             'docsignopt': rd['docsignopt'],
                                             'docsignxa': rd['docsignxa']}))
    optargs = stripcomma(replace('#docsignopt##docsignxa#',
                                 {'docsignxa': rd['docsignxashort'],
                                  'docsignopt': rd['docsignoptshort']}
                                 ))
    if optargs == '':
        rd['docsignatureshort'] = stripcomma(
            replace('#docsign#', {'docsign': rd['docsign']}))
    else:
        rd['docsignatureshort'] = replace('#docsign#[#docsignopt#]',
                                          {'docsign': rd['docsign'],
                                           'docsignopt': optargs,
                                           })
    rd['latexdocsignatureshort'] = rd['docsignatureshort'].replace('_', '\\_')
    rd['latexdocsignatureshort'] = rd[
        'latexdocsignatureshort'].replace(',', ', ')
    cfs = stripcomma(replace('#callfortran##callfortranappend#', {
                     'callfortran': rd['callfortran'], 'callfortranappend': rd['callfortranappend']}))
    if len(rd['callfortranappend']) > 1:
        rd['callcompaqfortran'] = stripcomma(replace('#callfortran# 0,#callfortranappend#', {
                                             'callfortran': rd['callfortran'], 'callfortranappend': rd['callfortranappend']}))
    else:
        rd['callcompaqfortran'] = cfs
    rd['callfortran'] = cfs
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
            replace('#docreturn#', {'docreturn': rd['docreturn']})) + ' = '
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
                ['\\begin{description}'] + rd[k][1:] +\
                ['\\end{description}']

    ar = applyrules(routine_rules, rd)
    if ismoduleroutine(rout):
        outmess('              %s\n' % (ar['docshort']))
    else:
        outmess('          %s\n' % (ar['docshort']))
    return ar, wrap


#################### EOF rules.py #######################
<|endoftext|>"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy

from .auxfuncs import (
    getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in,
    isintent_out, islogicalfunction, ismoduleroutine, isscalar,
    issubroutine, issubroutine_wrap, outmess, show
)

from ._isocbind import isoc_kindmap

def var2fixfortran(vars, a, fa=None, f90mode=None):
    if fa is None:
        fa = a
    if a not in vars:
        show(vars)
        outmess('var2fixfortran: No definition for argument "%s".\n' % a)
        return ''
    if 'typespec' not in vars[a]:
        show(vars[a])
        outmess('var2fixfortran: No typespec for argument "%s".\n' % a)
        return ''
    vardef = vars[a]['typespec']
    if vardef == 'type' and 'typename' in vars[a]:
        vardef = '%s(%s)' % (vardef, vars[a]['typename'])
    selector = {}
    lk = ''
    if 'kindselector' in vars[a]:
        selector = vars[a]['kindselector']
        lk = 'kind'
    elif 'charselector' in vars[a]:
        selector = vars[a]['charselector']
        lk = 'len'
    if '*' in selector:
        if f90mode:
            if selector['*'] in ['*', ':', '(*)']:
                vardef = '%s(len=*)' % (vardef)
            else:
                vardef = '%s(%s=%s)' % (vardef, lk, selector['*'])
        else:
            if selector['*'] in ['*', ':']:
                vardef = '%s*(%s)' % (vardef, selector['*'])
            else:
                vardef = '%s*%s' % (vardef, selector['*'])
    else:
        if 'len' in selector:
            vardef = '%s(len=%s' % (vardef, selector['len'])
            if 'kind' in selector:
                vardef = '%s,kind=%s)' % (vardef, selector['kind'])
            else:
                vardef = '%s)' % (vardef)
        elif 'kind' in selector:
            vardef = '%s(kind=%s)' % (vardef, selector['kind'])

    vardef = '%s %s' % (vardef, fa)
    if 'dimension' in vars[a]:
        vardef = '%s(%s)' % (vardef, ','.join(vars[a]['dimension']))
    return vardef

def useiso_c_binding(rout):
    useisoc = False
    for key, value in rout['vars'].items():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isoc_kindmap:
            return True
    return useisoc

def createfuncwrapper(rout, signature=0):
    assert isfunction(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = 'f2py_%s_d%s' % (a, i)
                dv = dict(typespec='integer', intent=['hide'])
                dv['='] = 'shape(%s, %s)' % (a, i)
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = '%s\n      %s' % (ret[0], line)
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)
    newname = '%sf2pywrap' % (name)

    if newname not in vars:
        vars[newname] = vars[name]
        args = [newname] + rout['args'][1:]
    else:
        args = [newname] + rout['args']

    l_tmpl = var2fixfortran(vars, name, '@@@NAME@@@', f90mode)
    if l_tmpl[:13] == 'character*(*)':
        if f90mode:
            l_tmpl = 'character(len=10)' + l_tmpl[13:]
        else:
            l_tmpl = 'character*10' + l_tmpl[13:]
        charselect = vars[name]['charselector']
        if charselect.get('*', '') == '(*)':
            charselect['*'] = '10'

    l1 = l_tmpl.replace('@@@NAME@@@', newname)
    rl = None

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        # gh-23598 fix warning
        # Essentially, this gets called again with modules where the name of the
        # function is added to the arguments, which is not required, and removed
        sargs = sargs.replace(f"{name}, ", '')
        args = [arg for arg in args if arg != name]
        rout['args'] = args
        add('subroutine f2pywrap_%s_%s (%s)' %
            (rout['modulename'], name, sargs))
        if not signature:
            add('use %s, only : %s' % (rout['modulename'], fortranname))
        if useisoc:
            add('use iso_c_binding')
    else:
        add('subroutine f2pywrap%s (%s)' % (name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add('external %s' % (fortranname))
            rl = l_tmpl.replace('@@@NAME@@@', '') + ' ' + fortranname

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    args = args[1:]
    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add('external %s' % (a))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isintent_in(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    add(l1)
    if rl is not None:
        add(rl)

    if need_interface:
        if f90mode:
            # f90 module already defines needed interface
            pass
        else:
            add('interface')
            add(rout['saved_interface'].lstrip())
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        if islogicalfunction(rout):
            add('%s = .not.(.not.%s(%s))' % (newname, fortranname, sargs))
        else:
            add('%s = %s(%s)' % (newname, fortranname, sargs))
    if f90mode:
        add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
    else:
        add('end')
    return ret[0]


def createsubrwrapper(rout, signature=0):
    assert issubroutine(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = 'f2py_%s_d%s' % (a, i)
                dv = dict(typespec='integer', intent=['hide'])
                dv['='] = 'shape(%s, %s)' % (a, i)
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = '%s\n      %s' % (ret[0], line)
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)

    args = rout['args']

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        add('subroutine f2pywrap_%s_%s (%s)' %
            (rout['modulename'], name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not signature:
            add('use %s, only : %s' % (rout['modulename'], fortranname))
    else:
        add('subroutine f2pywrap%s (%s)' % (name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add('external %s' % (fortranname))

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add('external %s' % (a))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    if need_interface:
        if f90mode:
            # f90 module already defines needed interface
            pass
        else:
            add('interface')
            for line in rout['saved_interface'].split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        add('call %s(%s)' % (fortranname, sargs))
    if f90mode:
        add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
    else:
        add('end')
    return ret[0]


def assubr(rout):
    if isfunction_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran function "%s"("%s")...\n' % (
            name, fortranname))
        rout = copy.copy(rout)
        fname = name
        rname = fname
        if 'result' in rout:
            rname = rout['result']
            rout['vars'][fname] = rout['vars'][rname]
        fvar = rout['vars'][fname]
        if not isintent_out(fvar):
            if 'intent' not in fvar:
                fvar['intent'] = []
            fvar['intent'].append('out')
            flag = 1
            for i in fvar['intent']:
                if i.startswith('out='):
                    flag = 0
                    break
            if flag:
                fvar['intent'].append('out=%s' % (rname))
        rout['args'][:] = [fname] + rout['args']
        return rout, createfuncwrapper(rout)
    if issubroutine_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran subroutine "%s"("%s")...\n'
                % (name, fortranname))
        rout = copy.copy(rout)
        return rout, createsubrwrapper(rout)
    return rout, ''
<|endoftext|>import os
import re

# START OF CODE VENDORED FROM `numpy.distutils.from_template`
#############################################################
"""
process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separated words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,\\0,\\1>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,\\0,\\1>
"""

routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I)
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """

    spanlist = []
    ind = 0
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            while True:
                i = astr.rfind('\n', ind, start)
                if i==-1:
                    break
                start = i
                if astr[i:i+7]!='\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end()-1 or len(astr)
        spanlist.append((start, end))
    return spanlist

template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
list_re = re.compile(r"<\s*((.*?))\s*>")

def find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or unique_key(names)
        repl = rep[1].replace(r'\,', '@comma@')
        thelist = conv(repl)
        names[name] = thelist
    return names

def find_and_remove_repl_patterns(astr):
    names = find_repl_patterns(astr)
    astr = re.subn(named_re, '', astr)[0]
    return astr, names

item_re = re.compile(r"\A\\(?P<index>\d+)\Z")
def conv(astr):
    b = astr.split(',')
    l = [x.strip() for x in b]
    for i in range(len(l)):
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    return ','.join(l)

def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey


template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')
def expand_sub(substr, names):
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    lnames = find_repl_patterns(substr)
    substr = named_re.sub(r"<\1>", substr)  # get rid of definition templates

    def listrepl(mobj):
        thelist = conv(mobj.group(1).replace(r'\,', '@comma@'))
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        for key in lnames.keys():    # see if list is already in dictionary
            if lnames[key] == thelist:
                name = key
        if name is None:      # this list is not in the dictionary yet
            name = unique_key(lnames)
            lnames[name] = thelist
        return "<%s>" % name

    substr = list_re.sub(listrepl, substr) # convert all lists to named templates
                                           # newnames are constructed as needed

    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % (r))
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)

            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print("Mismatch in number of replacements (base <{}={}>) "
                      "for <{}={}>. Ignoring.".format(base_rule, ','.join(rules[base_rule]), r, thelist))
    if not rules:
        return substr

    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name, (k+1)*[name])[k]

    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr

def process_str(allstr):
    newstr = allstr
    writestr = ''

    struct = parse_structure(newstr)

    oldend = 0
    names = {}
    names.update(_special_names)
    for sub in struct:
        cleanedstr, defs = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        writestr += cleanedstr
        names.update(defs)
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        oldend =  sub[1]
    writestr += newstr[oldend:]

    return writestr

include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]", re.I)

def resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def process_file(source):
    lines = resolve_includes(source)
    return process_str(''.join(lines))

_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

# END OF CODE VENDORED FROM `numpy.distutils.from_template`
###########################################################
<|endoftext|>"""
Build 'use others module data' mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.3 $"[10:-1]

f2py_version = 'See `f2py -v`'


from .auxfuncs import (
    applyrules, dictappend, gentitle, hasnote, outmess
)


usemodule_rules = {
    'body': """
#begintitle#
static char doc_#apiname#[] = \"\\\nVariable wrapper signature:\\n\\
\t #name# = get_#name#()\\n\\
Arguments:\\n\\
#docstr#\";
extern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);
static PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {
/*#decl#*/
\tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
printf(\"c: %d\\n\",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));
\treturn Py_BuildValue(\"\");
capi_fail:
\treturn NULL;
}
""",
    'method': '\t{\"get_#name#\",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},',
    'need': ['F_MODFUNC']
}

################


def buildusevars(m, r):
    ret = {}
    outmess(
        '\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n' % (m['name']))
    varsmap = {}
    revmap = {}
    if 'map' in r:
        for k in r['map'].keys():
            if r['map'][k] in revmap:
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (
                    r['map'][k], k, revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]] = k
    if r.get('only'):
        for v in r['map'].keys():
            if r['map'][v] in m['vars']:

                if revmap[r['map'][v]] == v:
                    varsmap[v] = r['map'][v]
                else:
                    outmess('\t\t\tIgnoring map "%s=>%s". See above.\n' %
                            (v, r['map'][v]))
            else:
                outmess(
                    '\t\t\tNo definition for variable "%s=>%s". Skipping.\n' % (v, r['map'][v]))
    else:
        for v in m['vars'].keys():
            if v in revmap:
                varsmap[v] = revmap[v]
            else:
                varsmap[v] = v
    for v in varsmap.keys():
        ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))
    return ret


def buildusevar(name, realname, vars, usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (
        name, realname))
    ret = {}
    vrd = {'name': name,
           'realname': realname,
           'REALNAME': realname.upper(),
           'usemodulename': usemodulename,
           'USEMODULENAME': usemodulename.upper(),
           'texname': name.replace('_', '\\_'),
           'begintitle': gentitle('%s=>%s' % (name, realname)),
           'endtitle': gentitle('end of %s=>%s' % (name, realname)),
           'apiname': '#modulename#_use_%s_from_%s' % (realname, usemodulename)
           }
    nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv',
              5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
    vrd['texnamename'] = name
    for i in nummap.keys():
        vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
    if hasnote(vars[realname]):
        vrd['note'] = vars[realname]['note']
    rd = dictappend({}, vrd)

    print(name, realname, vars[realname])
    ret = applyrules(usemodule_rules, rd)
    return ret
<|endoftext|>"""
Build call-back mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__
from .auxfuncs import (
    applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray,
    iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c,
    isintent_hide, isintent_in, isintent_inout, isintent_nothide,
    isintent_out, isoptional, isrequired, isscalar, isstring,
    isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace,
    stripcomma, throw_error
)
from . import cfuncs

f2py_version = __version__.version


################## Rules for callback function ##############

cb_routine_rules = {
    'cbtypedefs': 'typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);',
    'body': """
#begintitle#
typedef struct {
    PyObject *capi;
    PyTupleObject *args_capi;
    int nofargs;
    jmp_buf jmpbuf;
} #name#_t;

#if defined(F2PY_THREAD_LOCAL_DECL) && !defined(F2PY_USE_PYTHON_TLS)

static F2PY_THREAD_LOCAL_DECL #name#_t *_active_#name# = NULL;

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    #name#_t *prev = _active_#name#;
    _active_#name# = ptr;
    return prev;
}

static #name#_t *get_active_#name#(void) {
    return _active_#name#;
}

#else

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PySwapThreadLocalCallbackPtr(key, ptr);
}

static #name#_t *get_active_#name#(void) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PyGetThreadLocalCallbackPtr(key);
}

#endif

/*typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);*/
#static# #rctype# #callbackname# (#optargs##args##strarglens##noargs#) {
    #name#_t cb_local = { NULL, NULL, 0 };
    #name#_t *cb = NULL;
    PyTupleObject *capi_arglist = NULL;
    PyObject *capi_return = NULL;
    PyObject *capi_tmp = NULL;
    PyObject *capi_arglist_list = NULL;
    int capi_j,capi_i = 0;
    int capi_longjmp_ok = 1;
#decl#
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_start_clock();
#endif
    cb = get_active_#name#();
    if (cb == NULL) {
        capi_longjmp_ok = 0;
        cb = &cb_local;
    }
    capi_arglist = cb->args_capi;
    CFUNCSMESS(\"cb:Call-back function #name# (maxnofargs=#maxnofargs#(-#nofoptargs#))\\n\");
    CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    if (cb->capi==NULL) {
        capi_longjmp_ok = 0;
        cb->capi = PyObject_GetAttrString(#modulename#_module,\"#argname#\");
        CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    }
    if (cb->capi==NULL) {
        PyErr_SetString(#modulename#_error,\"cb: Callback #argname# not defined (as an argument or module #modulename# attribute).\\n\");
        goto capi_fail;
    }
    if (F2PyCapsule_Check(cb->capi)) {
    #name#_typedef #name#_cptr;
    #name#_cptr = F2PyCapsule_AsVoidPtr(cb->capi);
    #returncptr#(*#name#_cptr)(#optargs_nm##args_nm##strarglens_nm#);
    #return#
    }
    if (capi_arglist==NULL) {
        capi_longjmp_ok = 0;
        capi_tmp = PyObject_GetAttrString(#modulename#_module,\"#argname#_extra_args\");
        if (capi_tmp) {
            capi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);
            Py_DECREF(capi_tmp);
            if (capi_arglist==NULL) {
                PyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#argname#_extra_args to tuple.\\n\");
                goto capi_fail;
            }
        } else {
            PyErr_Clear();
            capi_arglist = (PyTupleObject *)Py_BuildValue(\"()\");
        }
    }
    if (capi_arglist == NULL) {
        PyErr_SetString(#modulename#_error,\"Callback #argname# argument list is not set.\\n\");
        goto capi_fail;
    }
#setdims#
#ifdef PYPY_VERSION
#define CAPI_ARGLIST_SETITEM(idx, value) PyList_SetItem((PyObject *)capi_arglist_list, idx, value)
    capi_arglist_list = PySequence_List((PyObject *)capi_arglist);
    if (capi_arglist_list == NULL) goto capi_fail;
#else
#define CAPI_ARGLIST_SETITEM(idx, value) PyTuple_SetItem((PyObject *)capi_arglist, idx, value)
#endif
#pyobjfrom#
#undef CAPI_ARGLIST_SETITEM
#ifdef PYPY_VERSION
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist_list);
#else
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist);
#endif
    CFUNCSMESS(\"cb:Call-back calling Python function #argname#.\\n\");
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_start_call_clock();
#endif
#ifdef PYPY_VERSION
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist_list);
    Py_DECREF(capi_arglist_list);
    capi_arglist_list = NULL;
#else
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist);
#endif
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_stop_call_clock();
#endif
    CFUNCSMESSPY(\"cb:capi_return=\",capi_return);
    if (capi_return == NULL) {
        fprintf(stderr,\"capi_return is NULL\\n\");
        goto capi_fail;
    }
    if (capi_return == Py_None) {
        Py_DECREF(capi_return);
        capi_return = Py_BuildValue(\"()\");
    }
    else if (!PyTuple_Check(capi_return)) {
        capi_return = Py_BuildValue(\"(N)\",capi_return);
    }
    capi_j = PyTuple_Size(capi_return);
    capi_i = 0;
#frompyobj#
    CFUNCSMESS(\"cb:#name#:successful\\n\");
    Py_DECREF(capi_return);
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_stop_clock();
#endif
    goto capi_return_pt;
capi_fail:
    fprintf(stderr,\"Call-back #name# failed.\\n\");
    Py_XDECREF(capi_return);
    Py_XDECREF(capi_arglist_list);
    if (capi_longjmp_ok) {
        longjmp(cb->jmpbuf,-1);
    }
capi_return_pt:
    ;
#return#
}
#endtitle#
""",
    'need': ['setjmp.h', 'CFUNCSMESS', 'F2PY_THREAD_LOCAL_DECL'],
    'maxnofargs': '#maxnofargs#',
    'nofoptargs': '#nofoptargs#',
    'docstr': """\
    def #argname#(#docsignature#): return #docreturn#\\n\\
#docstrsigns#""",
    'latexdocstr': """
{{}\\verb@def #argname#(#latexdocsignature#): return #docreturn#@{}}
#routnote#

#latexdocstrsigns#""",
    'docstrshort': 'def #argname#(#docsignature#): return #docreturn#'
}
cb_rout_rules = [
    {  # Init
        'separatorsfor': {'decl': '\n',
                          'args': ',', 'optargs': '', 'pyobjfrom': '\n', 'freemem': '\n',
                          'args_td': ',', 'optargs_td': '',
                          'args_nm': ',', 'optargs_nm': '',
                          'frompyobj': '\n', 'setdims': '\n',
                          'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        'decl': '/*decl*/', 'pyobjfrom': '/*pyobjfrom*/', 'frompyobj': '/*frompyobj*/',
        'args': [], 'optargs': '', 'return': '', 'strarglens': '', 'freemem': '/*freemem*/',
        'args_td': [], 'optargs_td': '', 'strarglens_td': '',
        'args_nm': [], 'optargs_nm': '', 'strarglens_nm': '',
        'noargs': '',
        'setdims': '/*setdims*/',
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '    Required arguments:',
        'docstropt': '    Optional arguments:',
        'docstrout': '    Return objects:',
        'docstrcbs': '    Call-back functions:',
        'docreturn': '', 'docsign': '', 'docsignopt': '',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
        'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
    }, {  # Function
        'decl': '    #ctype# return_value = 0;',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
            '''\
    if (capi_j>capi_i) {
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          "#ctype#_from_pyobj failed in converting return_value of"
          " call-back function #name# to C #ctype#\\n");
    } else {
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }''',
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#.\\n",return_value);'}
        ],
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'}, 'GETSCALARFROMPYTUPLE'],
        'return': '    return return_value;',
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(iscomplexfunction))
    },
    {  # String function
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"debug-capi:cb:#name#:%d:\\n",return_value_len);'},
        'args': '#ctype# return_value,int return_value_len',
        'args_nm': 'return_value,&return_value_len',
        'args_td': '#ctype# ,int',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->\\"");'},
            """\
    if (capi_j>capi_i) {
        GETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len);
    } else {
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }""",
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#\\".\\n",return_value);'}
        ],
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSTRFROMPYTUPLE'],
        'return': 'return;',
        '_check': isstringfunction
    },
    {  # Complex function
        'optargs': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *return_value
#endif
""",
        'optargs_nm': """
#ifndef F2PY_CB_RETURNCOMPLEX
return_value
#endif
""",
        'optargs_td': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *
#endif
""",
        'decl': """
#ifdef F2PY_CB_RETURNCOMPLEX
    #ctype# return_value = {0, 0};
#endif
""",
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
            """\
    if (capi_j>capi_i) {
#ifdef F2PY_CB_RETURNCOMPLEX
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#else
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#endif
    } else {
        fprintf(stderr,
                \"Warning: call-back function #name# did not provide\"
                \" return value (index=%d, type=#ctype#)\\n\",capi_i);
    }""",
            {debugcapi: """\
#ifdef F2PY_CB_RETURNCOMPLEX
    fprintf(stderr,\"#showvalueformat#.\\n\",(return_value).r,(return_value).i);
#else
    fprintf(stderr,\"#showvalueformat#.\\n\",(*return_value).r,(*return_value).i);
#endif
"""}
        ],
        'return': """
#ifdef F2PY_CB_RETURNCOMPLEX
    return return_value;
#else
    return;
#endif
""",
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSCALARFROMPYTUPLE', '#ctype#'],
        '_check': iscomplexfunction
    },
    {'docstrout': '        #pydocsignout#',
     'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                        {hasnote: '--- #note#'}],
     'docreturn': '#rname#,',
     '_check': isfunction},
    {'_check': issubroutine, 'return': 'return;'}
]

cb_arg_rules = [
    {  # Doc
        'docstropt': {l_and(isoptional, isintent_nothide): '        #pydocsign#'},
        'docstrreq': {l_and(isrequired, isintent_nothide): '        #pydocsign#'},
        'docstrout': {isintent_out: '        #pydocsignout#'},
        'latexdocstropt': {l_and(isoptional, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrreq': {l_and(isrequired, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrout': {isintent_out: ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                                          {l_and(hasnote, isintent_hide): '--- #note#',
                                           l_and(hasnote, isintent_nothide): '--- See above.'}]},
        'docsign': {l_and(isrequired, isintent_nothide): '#varname#,'},
        'docsignopt': {l_and(isoptional, isintent_nothide): '#varname#,'},
        'depend': ''
    },
    {
        'args': {
            l_and(isscalar, isintent_c): '#ctype# #varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *#varname_i#_cb_capi',
            isarray: '#ctype# *#varname_i#',
            isstring: '#ctype# #varname_i#'
        },
        'args_nm': {
            l_and(isscalar, isintent_c): '#varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#varname_i#_cb_capi',
            isarray: '#varname_i#',
            isstring: '#varname_i#'
        },
        'args_td': {
            l_and(isscalar, isintent_c): '#ctype#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *',
            isarray: '#ctype# *',
            isstring: '#ctype#'
        },
        'need': {l_or(isscalar, isarray, isstring): '#ctype#'},
        # untested with multiple args
        'strarglens': {isstring: ',int #varname_i#_cb_len'},
        'strarglens_td': {isstring: ',int'},  # untested with multiple args
        # untested with multiple args
        'strarglens_nm': {isstring: ',#varname_i#_cb_len'},
    },
    {  # Scalars
        'decl': {l_not(isintent_c): '    #ctype# #varname_i#=(*#varname_i#_cb_capi);'},
        'error': {l_and(isintent_c, isintent_out,
                        throw_error('intent(c,out) is forbidden for callback scalar arguments')):
                  ''},
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
                      {isintent_out:
                       '    if (capi_j>capi_i)\n        GETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",#varname_i#);'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), l_not( isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);'},
                      {l_and(debugcapi, l_and(iscomplex, isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);'},
                      {l_and(debugcapi, l_and(iscomplex, l_not( isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);'},
                      ],
        'need': [{isintent_out: ['#ctype#_from_pyobj', 'GETSCALARFROMPYTUPLE']},
                 {debugcapi: 'CFUNCSMESS'}],
        '_check': isscalar
    }, {
        'pyobjfrom': [{isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1(#varname_i#)))
            goto capi_fail;"""},
                      {isintent_inout: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#_cb_capi)))
            goto capi_fail;"""}],
        'need': [{isintent_in: 'pyobj_from_#ctype#1'},
                 {isintent_inout: 'pyarr_from_p_#ctype#1'},
                 {iscomplex: '#ctype#'}],
        '_check': l_and(isscalar, isintent_nothide),
        '_optional': ''
    }, {  # String
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->\\"");'},
                      """    if (capi_j>capi_i)
        GETSTRFROMPYTUPLE(capi_return,capi_i++,#varname_i#,#varname_i#_cb_len);""",
                      {debugcapi:
                       '    fprintf(stderr,"#showvalueformat#\\":%d:.\\n",#varname_i#,#varname_i#_cb_len);'},
                      ],
        'need': ['#ctype#', 'GETSTRFROMPYTUPLE',
                 {debugcapi: 'CFUNCSMESS'}, 'string.h'],
        '_check': l_and(isstring, isintent_out)
    }, {
        'pyobjfrom': [
            {debugcapi:
             ('    fprintf(stderr,"debug-capi:cb:#varname#=#showvalueformat#:'
              '%d:\\n",#varname_i#,#varname_i#_cb_len);')},
            {isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len)))
            goto capi_fail;"""},
                      {isintent_inout: """\
    if (cb->nofargs>capi_i) {
        int #varname_i#_cb_dims[] = {#varname_i#_cb_len};
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims)))
            goto capi_fail;
    }"""}],
        'need': [{isintent_in: 'pyobj_from_#ctype#1size'},
                 {isintent_inout: 'pyarr_from_p_#ctype#1'}],
        '_check': l_and(isstring, isintent_nothide),
        '_optional': ''
    },
    # Array ...
    {
        'decl': '    npy_intp #varname_i#_Dims[#rank#] = {#rank*[-1]#};',
        'setdims': '    #cbsetdims#;',
        '_check': isarray,
        '_depend': ''
    },
    {
        'pyobjfrom': [{debugcapi: '    fprintf(stderr,"debug-capi:cb:#varname#\\n");'},
                      {isintent_c: """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr will be inserted to capi_arglist_list that will be
           destroyed when leaving callback function wrapper together
           with tmp_arr. */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_CARRAY,NULL);
""",
                       l_not(isintent_c): """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr will be inserted to capi_arglist_list that will be
           destroyed when leaving callback function wrapper together
           with tmp_arr. */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_FARRAY,NULL);
""",
                       },
                      """
        if (tmp_arr==NULL)
            goto capi_fail;
        if (CAPI_ARGLIST_SETITEM(capi_i++,(PyObject *)tmp_arr))
            goto capi_fail;
}"""],
        '_check': l_and(isarray, isintent_nothide, l_or(isintent_in, isintent_inout)),
        '_optional': '',
    }, {
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
                      """    if (capi_j>capi_i) {
        PyArrayObject *rv_cb_arr = NULL;
        if ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;
        rv_cb_arr =  array_from_pyobj(#atype#,#varname_i#_Dims,#rank#,F2PY_INTENT_IN""",
                      {isintent_c: '|F2PY_INTENT_C'},
                      """,capi_tmp);
        if (rv_cb_arr == NULL) {
            fprintf(stderr,\"rv_cb_arr is NULL\\n\");
            goto capi_fail;
        }
        MEMCOPY(#varname_i#,PyArray_DATA(rv_cb_arr),PyArray_NBYTES(rv_cb_arr));
        if (capi_tmp != (PyObject *)rv_cb_arr) {
            Py_DECREF(rv_cb_arr);
        }
    }""",
                      {debugcapi: '    fprintf(stderr,"<-.\\n");'},
                      ],
        'need': ['MEMCOPY', {iscomplexarray: '#ctype#'}],
        '_check': l_and(isarray, isintent_out)
    }, {
        'docreturn': '#varname#,',
        '_check': isintent_out
    }
]

################## Build call-back module #############
cb_map = {}


def buildcallbacks(m):
    cb_map[m['name']] = []
    for bi in m['body']:
        if bi['block'] == 'interface':
            for b in bi['body']:
                if b:
                    buildcallback(b, m['name'])
                else:
                    errmess('warning: empty body for %s\n' % (m['name']))


def buildcallback(rout, um):
    from . import capi_maps

    outmess('    Constructing call-back function "cb_%s_in_%s"\n' %
            (rout['name'], um))
    args, depargs = getargs(rout)
    capi_maps.depargs = depargs
    var = rout['vars']
    vrd = capi_maps.cb_routsign2map(rout, um)
    rd = dictappend({}, vrd)
    cb_map[um].append([rout['name'], rd['name']])
    for r in cb_rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)
    savevrd = {}
    for i, a in enumerate(args):
        vrd = capi_maps.cb_sign2map(a, var[a], index=i)
        savevrd[a] = vrd
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if '_optional' in r and isoptional(var[a]):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in args:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if ('_optional' not in r) or ('_optional' in r and isrequired(var[a])):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' not in r:
                continue
            if '_optional' in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    if 'args' in rd and 'optargs' in rd:
        if isinstance(rd['optargs'], list):
            rd['optargs'] = rd['optargs'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
            rd['optargs_nm'] = rd['optargs_nm'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
            rd['optargs_td'] = rd['optargs_td'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
            replace('#docreturn#', {'docreturn': rd['docreturn']}))
    optargs = stripcomma(replace('#docsignopt#',
                                 {'docsignopt': rd['docsignopt']}
                                 ))
    if optargs == '':
        rd['docsignature'] = stripcomma(
            replace('#docsign#', {'docsign': rd['docsign']}))
    else:
        rd['docsignature'] = replace('#docsign#[#docsignopt#]',
                                     {'docsign': rd['docsign'],
                                      'docsignopt': optargs,
                                      })
    rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
    rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
                ['\\begin{description}'] + rd[k][1:] +\
                ['\\end{description}']
    if 'args' not in rd:
        rd['args'] = ''
        rd['args_td'] = ''
        rd['args_nm'] = ''
    if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
        rd['noargs'] = 'void'

    ar = applyrules(cb_routine_rules, rd)
    cfuncs.callbacks[rd['name']] = ar['body']
    if isinstance(ar['need'], str):
        ar['need'] = [ar['need']]

    if 'need' in rd:
        for t in cfuncs.typedefs.keys():
            if t in rd['need']:
                ar['need'].append(t)

    cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
    ar['need'].append(rd['name'] + '_typedef')
    cfuncs.needs[rd['name']] = ar['need']

    capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'],
                                      'nofoptargs': ar['nofoptargs'],
                                      'docstr': ar['docstr'],
                                      'latexdocstr': ar['latexdocstr'],
                                      'argname': rd['argname']
                                      }
    outmess('      %s\n' % (ar['docstrshort']))
    return
################## Build call-back function #############
<|endoftext|>"""Fortran to Python Interface Generator.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the terms
of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__all__ = ['run_main', 'get_include']

import sys
import subprocess
import os
import warnings

from numpy.exceptions import VisibleDeprecationWarning
from . import f2py2e
from . import diagnose

run_main = f2py2e.run_main
main = f2py2e.main


def get_include():
    """
    Return the directory that contains the ``fortranobject.c`` and ``.h`` files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.21.1

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

    * Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
      generates ``mymodmodule.c`` and (if needed)
      ``mymod-f2pywrappers.f`` files next to ``mymod.pyf``.
    * Step 2: build your Python extension module. This requires the
      following source files:

      * ``mymodmodule.c``
      * ``mymod-f2pywrappers.f`` (if it was generated in Step 1)
      * ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    """
    return os.path.join(os.path.dirname(__file__), 'src')


def __getattr__(attr):

    # Avoid importing things that aren't needed for building
    # which might import the main numpy module
    if attr == "test":
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)
        return test

    else:
        raise AttributeError("module {!r} has no attribute "
                              "{!r}".format(__name__, attr))


def __dir__():
    return list(globals().keys() | {"test"})
<|endoftext|>"""
Auxiliary functions for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) LICENSE.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import pprint
import sys
import re
import types
from functools import reduce

from . import __version__
from . import cfuncs
from .cfuncs import errmess

__all__ = [
    'applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle',
    'getargs2', 'getcallprotoargument', 'getcallstatement',
    'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode',
    'getusercode1', 'getdimension', 'hasbody', 'hascallstatement', 'hascommon',
    'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote',
    'isallocatable', 'isarray', 'isarrayofstrings',
    'ischaracter', 'ischaracterarray', 'ischaracter_or_characterarray',
    'iscomplex', 'iscstyledirective',
    'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn',
    'isdouble', 'isdummyroutine', 'isexternal', 'isfunction',
    'isfunction_wrap', 'isint1', 'isint1array', 'isinteger', 'isintent_aux',
    'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict',
    'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace',
    'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical',
    'islogicalfunction', 'islong_complex', 'islong_double',
    'islong_doublefunction', 'islong_long', 'islong_longfunction',
    'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isvariable',
    'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray',
    'isstring', 'isstringarray', 'isstring_or_stringarray', 'isstringfunction',
    'issubroutine', 'get_f2py_modulename', 'issubroutine_wrap', 'isthreadsafe',
    'isunsigned', 'isunsigned_char', 'isunsigned_chararray',
    'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short',
    'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace',
    'show', 'stripcomma', 'throw_error', 'isattr_value', 'getuseblocks',
    'process_f2cmap_dict', 'containscommon'
]


f2py_version = __version__.version


show = pprint.pprint

options = {}
debugoptions = []
wrapfuncs = 1


def outmess(t):
    if options.get('verbose', 1):
        sys.stdout.write(t)


def debugcapi(var):
    return 'capi' in debugoptions


def _ischaracter(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


def _isstring(var):
    return 'typespec' in var and var['typespec'] == 'character' and \
           not isexternal(var)


def ischaracter_or_characterarray(var):
    return _ischaracter(var) and 'charselector' not in var


def ischaracter(var):
    return ischaracter_or_characterarray(var) and not isarray(var)


def ischaracterarray(var):
    return ischaracter_or_characterarray(var) and isarray(var)


def isstring_or_stringarray(var):
    return _ischaracter(var) and 'charselector' in var


def isstring(var):
    return isstring_or_stringarray(var) and not isarray(var)


def isstringarray(var):
    return isstring_or_stringarray(var) and isarray(var)


def isarrayofstrings(var):  # obsolete?
    # leaving out '*' for now so that `character*(*) a(m)` and `character
    # a(m,*)` are treated differently. Luckily `character**` is illegal.
    return isstringarray(var) and var['dimension'][-1] == '(*)'


def isarray(var):
    return 'dimension' in var and not isexternal(var)


def isscalar(var):
    return not (isarray(var) or isstring(var) or isexternal(var))


def iscomplex(var):
    return isscalar(var) and \
           var.get('typespec') in ['complex', 'double complex']


def islogical(var):
    return isscalar(var) and var.get('typespec') == 'logical'


def isinteger(var):
    return isscalar(var) and var.get('typespec') == 'integer'


def isreal(var):
    return isscalar(var) and var.get('typespec') == 'real'


def get_kind(var):
    try:
        return var['kindselector']['*']
    except KeyError:
        try:
            return var['kindselector']['kind']
        except KeyError:
            pass


def isint1(var):
    return var.get('typespec') == 'integer' \
        and get_kind(var) == '1' and not isarray(var)


def islong_long(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') not in ['integer', 'logical']:
        return 0
    return get_kind(var) == '8'


def isunsigned_char(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-1'


def isunsigned_short(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-2'


def isunsigned(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-4'


def isunsigned_long_long(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-8'


def isdouble(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '8'


def islong_double(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '16'


def islong_complex(var):
    if not iscomplex(var):
