        onesided: bool | None = None,
        return_complex: bool | None = None,
        align_to_window: bool | None = None,
    ):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.stft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
                align_to_window=align_to_window,
            )
        return torch.stft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex=return_complex,
            align_to_window=align_to_window,
        )

    def istft(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: "Tensor | None" = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool | None = None,
        length: int | None = None,
        return_complex: bool = False,
    ):
        r"""See :func:`torch.istft`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.istft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                onesided=onesided,
                length=length,
                return_complex=return_complex,
            )
        return torch.istft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex=return_complex,
        )

    def resize(self, *sizes):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.resize, (self,), self, *sizes)
        warnings.warn("non-inplace resize is deprecated", stacklevel=2)
        from torch.autograd._functions import Resize

        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        if has_torch_function_variadic(self, tensor):
            return handle_torch_function(Tensor.resize_as, (self, tensor), self, tensor)
        warnings.warn("non-inplace resize_as is deprecated", stacklevel=2)
        from torch.autograd._functions import Resize

        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.split, (self,), self, split_size, dim=dim
            )
        if isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
            except ValueError:
                pass

        if isinstance(split_size, (int, torch.SymInt)):
            return torch._VF.split(self, split_size, dim)  # type: ignore[attr-defined]
        else:
            return torch._VF.split_with_sizes(
                self,
                split_size,
                dim,
            )

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        r"""Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique,
                (self,),
                self,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique(
            self,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        r"""Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique_consecutive,
                (self,),
                self,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique_consecutive(
            self, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return _C._VariableFunctions.rsub(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return self.reciprocal() * other

    __rtruediv__ = __rdiv__
    __itruediv__ = _C.TensorBase.__idiv__

    # pyrefly: ignore [bad-override]
    __pow__ = cast(
        Callable[
            ["torch._C.TensorBase", Union["Tensor", int, float, bool, complex]],
            "Tensor",
        ],
        _handle_torch_function_and_wrap_type_error_to_not_implemented(
            _C.TensorBase.pow
        ),
    )

    __ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C.TensorBase.pow_
    )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return torch.remainder(other, self)

    def __format__(self, format_spec):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and not self.is_meta and type(self) is Tensor:
            # Use detach() here to avoid the warning when converting a scalar Tensor that
            # requires gradients to a python number. It is ok for formatting.
            return self.detach().item().__format__(format_spec)
        return object.__format__(self, format_spec)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
        return torch.pow(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":  # type: ignore[override]
        # TODO(rec): the superclass says it accepts complex here,
        # but torch.floor_divide says it doesn't.
        return torch.floor_divide(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":  # type: ignore[override]
        return torch.floor_divide(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(
        self, other: Union["Tensor", int, float, bool, complex]
    ) -> "Tensor":
        return torch.bitwise_left_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(
        self, other: Union["Tensor", int, float, bool, complex]
    ) -> "Tensor":
        return torch.bitwise_right_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other: "Tensor") -> "Tensor":
        return torch.matmul(other, self)

    __pos__ = _C.TensorBase.positive
    __neg__ = _C.TensorBase.neg
    __abs__ = _C.TensorBase.abs

    def __len__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Using len to get tensor shape might cause the trace to be incorrect. "
                "Recommended usage would be tensor.shape[0]. "
                "Passing a tensor of different shape might lead to errors or silently give "
                "incorrect results.",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return self.shape[0]

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that we get a generator
        # and don't eagerly perform all the indexes.  This could save us
        # work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        # NB: We have intentionally skipped __torch_function__ dispatch here.
        # See gh-54457
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Iterating over a tensor might cause the trace to be incorrect. "
                "Passing a tensor of different shape won't change the number of "
                "iterations executed (and might lead to errors or silently give "
                "incorrect results).",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return iter(self.unbind(0))

    def __hash__(self):
        # Do NOT handle __torch_function__ here as user's default
        # implementation that handle most functions will most likely do it wrong.
        # It can be easily overridden by defining this method on the user
        # subclass if needed.
        return id(self)

    def __dir__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dir__, (self,), self)
        tensor_methods = dir(self.__class__)
        tensor_methods.remove("volatile")  # deprecated
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs

        # property only available dense, cuda tensors
        if (not self.is_cuda) or self.is_sparse:
            keys.remove("__cuda_array_interface__")

        return sorted(keys)

    # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
    __array_priority__ = 1000  # prefer Tensor ops over numpy ones

    def __array__(self, dtype=None):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array again in a suitable tensor when done, to support e.g.
    # `numpy.sin(tensor) -> tensor` or `numpy.greater(tensor, 0) -> ByteTensor`
    def __array_wrap__(self, array):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__array_wrap__, (self,), self, array=array
            )
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            array = array.astype("uint8")
        return torch.from_numpy(array)

    def __contains__(self, element: Any, /) -> bool:
        r"""Check if `element` is present in tensor

        Args:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__contains__, (self,), self, element)
        if isinstance(
            element, (torch.Tensor, Number, torch.SymInt, torch.SymFloat, torch.SymBool)
        ):
            # type hint doesn't understand the __contains__ result array
            return bool((element == self).any().item())  # type: ignore[union-attr]

        raise RuntimeError(
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {type(element)}."
        )

    @property
    def __cuda_array_interface__(self):
        """Array view description for cuda tensors.

        See:
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(  # pyrefly: ignore [bad-argument-count]
                Tensor.__cuda_array_interface__.__get__,  # type: ignore[attr-defined]
                (self,),
                self,  # pyrefly: ignore [bad-argument-type]
            )

        # raise AttributeError for unsupported tensors, so that
        # hasattr(cpu_tensor, "__cuda_array_interface__") is False.
        if not self.is_cuda:
            raise AttributeError(
                f"Can't get __cuda_array_interface__ on non-CUDA tensor type: {self.type()} "
                "If CUDA data is required use tensor.cuda() to copy tensor to device memory."
            )

        if self.is_sparse:
            raise AttributeError(
                f"Can't get __cuda_array_interface__ on sparse type: {self.type()} "
                "Use Tensor.to_dense() to convert to a dense tensor first."
            )

        # RuntimeError, matching tensor.__array__() behavior.
        if self.requires_grad:
            raise RuntimeError(
                "Can't get __cuda_array_interface__ on Variable that requires grad. "
                "If gradients aren't required, use var.detach() to get Variable that doesn't require grad."
            )

        typestr = _dtype_to_typestr(self.dtype)
        itemsize = self.element_size()
        shape = tuple(self.shape)
        if self.is_contiguous():
            # __cuda_array_interface__ v2 requires the strides to be omitted
            # (either not set or set to None) for C-contiguous arrays.
            strides = None
        else:
            strides = tuple(s * itemsize for s in self.stride())
        data_ptr = self.data_ptr() if self.numel() > 0 else 0
        data = (data_ptr, False)  # read-only is false

        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=2)

    def storage_type(self):
        r"""storage_type() -> type

        Returns the type of the underlying storage.

        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage_type, (self,), self)

        torch.storage._warn_typed_storage_removal()

        return self._typed_storage()._get_legacy_storage_class()

    def unflatten(self, dim, sizes):  # type: ignore[override]
        r"""
        unflatten(dim, sizes) -> Tensor

        See :func:`torch.unflatten`.

        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unflatten, (self,), self, dim, sizes)

        if not sizes:
            raise RuntimeError("unflatten: sizes must be non-empty")

        return super().unflatten(dim, sizes)

    def to_sparse_coo(self):
        """Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.

        Examples::

             >>> dense = torch.randn(5, 5)
             >>> sparse = dense.to_sparse_coo()
             >>> sparse._nnz()
             25

        """
        return self.to_sparse()

    def dim_order(self, *, ambiguity_check: bool | list[torch.memory_format] = False):
        """
        dim_order(ambiguity_check=False) -> tuple

        Returns the uniquely determined tuple of int describing the dim order or
        physical layout of :attr:`self`.

        The dim order represents how dimensions are laid out in memory of dense tensors,
        starting from the outermost to the innermost dimension.

        Note that the dim order may not always be uniquely determined.
        If `ambiguity_check` is True, this function raises a RuntimeError when the dim order cannot be uniquely determined;
        If `ambiguity_check` is a list of memory formats, this function raises a RuntimeError when tensor can not be interpreted
        into exactly one of the given memory formats, or it cannot be uniquely determined.
        If `ambiguity_check` is False, it will return one of legal dim order(s) without checking its uniqueness.
        Otherwise, it will raise TypeError.

        Args:
            ambiguity_check (bool or List[torch.memory_format]): The check method for ambiguity of dim order.

        Examples::

            >>> torch.empty((2, 3, 5, 7)).dim_order()
            (0, 1, 2, 3)
            >>> torch.empty((2, 3, 5, 7)).transpose(1, 2).dim_order()
            (0, 2, 1, 3)
            >>> torch.empty((2, 3, 5, 7), memory_format=torch.channels_last).dim_order()
            (0, 2, 3, 1)
            >>> torch.empty((1, 2, 3, 4)).dim_order()
            (0, 1, 2, 3)
            >>> try:
            ...     torch.empty((1, 2, 3, 4)).dim_order(ambiguity_check=True)
            ... except RuntimeError as e:
            ...     print(e)
            The tensor does not have unique dim order, or cannot map to exact one of the given memory formats.
            >>> torch.empty((1, 2, 3, 4)).dim_order(
            ...     ambiguity_check=[torch.contiguous_format, torch.channels_last]
            ... )  # It can be mapped to contiguous format
            (0, 1, 2, 3)
            >>> try:
            ...     torch.empty((1, 2, 3, 4)).dim_order(ambiguity_check="ILLEGAL") # type: ignore[arg-type]
            ... except TypeError as e:
            ...     print(e)
            The ambiguity_check argument must be a bool or a list of memory formats.

        .. warning::
            The dim_order tensor API is experimental and subject to change.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.dim_order, (self,), self)

        if self.is_sparse:
            raise AttributeError(
                f"Can't get dim order on sparse type: {self.type()} "
                "Use Tensor.to_dense() to convert to a dense tensor first."
            )

        # Sanity check ambiguity_check data types
        if not isinstance(ambiguity_check, bool):
            if not isinstance(ambiguity_check, list):
                raise TypeError(
                    "The ambiguity_check argument must be a bool or a list of memory formats."
                )
            for memory_format in ambiguity_check:
                if not isinstance(memory_format, torch.memory_format):
                    raise TypeError(
                        "The ambiguity_check argument must be a bool or a list of memory formats."
                    )

        def invalid_unique_memory_format(tensor, valid_memory_formats):
            """
            Returns True if the tensor cannot be uniquely mapped to any of the given memory formats, False otherwise.
            """

            n_legality = 0

            for memory_format in valid_memory_formats:
                if tensor.is_contiguous(memory_format=memory_format):
                    n_legality += 1

            return n_legality != 1

        def has_multiple_dim_order(tensor):
            """
            Returns True if there're multiple legal dim orders for given tensor, False otherwise.

            The tensor is considered to have multiple legal dim orders if either of the following conditions is met:

            * Singleton Dimensions: There's at least one singleton dimension in the tensor.
              Since their size is 1, they don't affect the memory offset (stride * index
              is zero because index is always zero). Therefore, they can be placed anywhere
              in the dimension order without changing how data is accessed.
            * Same strides: Strides reflect how the tensor is stored in memory.
              If any two dimensions have the same stride, swapping these dimensions won't
              change how data is accessed, leading to multiple correct dimension orders.
            """
            from torch.fx.experimental.symbolic_shapes import guard_or_false

            sizes = tensor.size()
            strides = tensor.stride()

            # Check if there are any duplicate strides
            has_duplicate_strides = any(
                guard_or_false(earlier == later)
                for earlier, later in itertools.pairwise(strides)
            )

            # Check if there are any singleton dimensions
            has_singleton_dims = any(guard_or_false(size == 1) for size in sizes)

            return has_duplicate_strides or has_singleton_dims

        valid_memory_formats = (
            ambiguity_check if isinstance(ambiguity_check, list) else []
        )
        check_multiple_dim_order = (
            ambiguity_check if isinstance(ambiguity_check, bool) else True
        )

        if (
            check_multiple_dim_order and has_multiple_dim_order(self)
        ) and invalid_unique_memory_format(self, valid_memory_formats):
            raise RuntimeError(
                "The tensor does not have unique dim order, or cannot map to exact one of the given memory formats."
            )

        import torch._prims_common as utils

        out_perm, raise_ambiguity = (
            utils.compute_elementwise_output_logical_to_physical_perm(
                self, ambiguity_check=ambiguity_check
            )
        )
        if raise_ambiguity:
            raise RuntimeError("The tensor does not have unique dim order.")
        return tuple(out_perm)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with _C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return _convert(ret, cls)

    __torch_dispatch__ = _C._disabled_torch_dispatch_impl

    def __dlpack__(
        self,
        *,
        stream: Any | None = -1,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[enum.IntEnum, int] | None = None,
        copy: bool | None = None,
        read_only: bool = False,
    ):
        """
        Creates a DLpack `capsule https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange`_
        of the current tensor to be exported to other libraries.

        This function will be called from the `from_dlpack` method
        of the library that will consume the capsule. `from_dlpack` passes the current
        stream to this method as part of the specification.

        Args:
            stream (integer or None): An optional Python integer representing a
                pointer to a CUDA stream. The current stream is synchronized with
                this stream before the capsule is created, and since the capsule
                shares its storage with the tensor this make it safe to access from
                both streams.  If -1 is passed then no synchronization is performed.
                If 1 (on CUDA) or 0 (on ROCM) then the default stream is used for
                synchronization. This API intentionally slightly deviates from the DLPack
                guidance: the default stream is -1 (stream-preserving; no cross-stream sync),
                because many from_dlpack implementations intend stream preservation.
                For non-CUDA devices, -1 is treated the same as None.

            max_version (tuple[int, int] or None): An optional Python tuple with
                2 integers, representing the maximum version the caller supports. If
                None (default), PyTorch will fallback to DLPack 0.8.

            dl_device (tuple[DLDeviceType, int] or None): An optional tuple specifying
                in which device the exported DLPack capsule should be on. If None (default),
                the exported DLPack capsule will be on the same device as ``self``.

            copy (bool or None): An optional boolean indicating whether or not to copy
                ``self``. If None, PyTorch will copy only if necessary.

            read_only (bool): If True, the exported capsule is marked read-only
                (``DLPACK_FLAG_BITMASK_READ_ONLY``) and the data is exported
                through ``const_data_ptr()`` so a copy-on-write tensor is not
                materialized. The consumer must not mutate the data. Requires
                the versioned DLPack protocol (``max_version >= (1, 0)``).
        """
        if has_torch_function_unary(self):
            args = (self,)
            kwargs = {
                "stream": stream,
                "max_version": max_version,
                "dl_device": dl_device,
                "copy": copy,
                "read_only": read_only,
            }
            return handle_torch_function(Tensor.__dlpack__, (self,), *args, **kwargs)

        # DLPack capsules can't capture all of PyTorch's semantics,
        # so we prohibit exporting tensors that would lose their properties like
        # requires_grad and having the conjugate bit set.
        if self.requires_grad:
            raise BufferError(
                "Can't export tensors that require gradient, use tensor.detach()"
            )
        if self.is_conj():
            raise BufferError("Can't export tensors with the conjugate bit set")
        if self.layout != torch.strided:
            raise BufferError(
                "Can't export tensors with layout other than torch.strided"
            )

        if (
            self.device.type == "cuda"
            and self.device.index != torch.cuda.current_device()
        ):
            raise BufferError(
                "Can't export tensors on a different CUDA device index. "
                f"Expected: {self.device.index}. "
                f"Current device: {torch.cuda.current_device()}."
            )

        if stream is not None and type(stream) is not int:
            # Stream pointers in CUDA/ROCm are uniquely numbered and can
            # be retrieved from their integer value.
            raise TypeError("stream must be ``int`` or ``none``")
        elif self.device.type == "cuda" and stream != -1:
            # NB: This logic handles the special case values for default
            # streams and must be kept in sync with from_dlpack in
            # torch/utils/dlpack.py
            is_rocm = torch.version.hip is not None
            is_cuda = not is_rocm

            if stream is None or (is_rocm and stream == 0) or (is_cuda and stream == 1):
                stream = torch.cuda.default_stream()
            else:
                if is_cuda and stream == 2:
                    raise BufferError("per-thread default stream is not supported.")

                device_str = "CUDA" if is_cuda else "ROCm"
                if not (
                    (is_cuda and stream != 0) or (is_rocm and stream not in (1, 2))
                ):
                    raise AssertionError(
                        f"unsupported stream on {device_str}: {stream}."
                    )

                stream = torch.cuda.ExternalStream(stream)

            # Only synchronize on different streams
            current_stream = torch.cuda.current_stream()
            if stream != current_stream:
                event = torch.cuda.Event()
                event.record(current_stream)
                stream.wait_event(event)
        elif self.device.type == "cpu":
            if stream is not None and stream != -1:
                raise AssertionError("stream should be None on cpu.")

        if self.device.type == "xla":
            import torch_xla
            import torch_xla.utils.dlpack as xla_dlpack

            if (
                len(torch_xla.real_devices()) <= 0
                or "cuda" not in torch_xla.real_devices()[0].lower()
            ):
                raise RuntimeError(
                    "Can't export to dlpack an XLA tensor that is not on CUDA."
                )

            # Does not support DLPack 1.0, yet.
            return xla_dlpack.to_dlpack(self)

        if max_version is None or max_version[0] < 1:
            # Fallback to the old, unversioned variant, which has no flags field
            # and therefore cannot represent read-only.
            if read_only:
                raise BufferError(
                    "read_only export requires the versioned DLPack protocol; "
                    "pass max_version=(1, 0) or higher"
                )
            return _C._to_dlpack(self, dl_device=dl_device, copy=copy)

        return _C._to_dlpack_versioned(
            self, dl_device=dl_device, copy=copy, read_only=read_only
        )

    def __dlpack_device__(self) -> tuple[enum.IntEnum, int]:
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack_device__, (self,), self)

        from torch.utils.dlpack import DLDeviceType

        device = self.device
        idx = device.index if device.index is not None else 0
        torch_device_type = device.type
        if torch_device_type == "cuda" and torch.version.hip is not None:
            device_type = DLDeviceType.kDLROCM
        elif torch_device_type == "cpu" and self.is_pinned():
            device_type = DLDeviceType.kDLCUDAHost
        elif torch_device_type == "cuda":
            device_type = DLDeviceType.kDLCUDA
        elif torch_device_type == "cpu":
            device_type = DLDeviceType.kDLCPU
        elif torch_device_type == "xpu":
            device_type = DLDeviceType.kDLOneAPI
        elif self.device.type == "privateuse1":
            device_type = DLDeviceType.kDLExtDev
        elif torch_device_type == "xla":
            import torch_xla

            if (
                len(torch_xla.real_devices()) <= 0
                or "cuda" not in torch_xla.real_devices()[0].lower()
            ):
                raise ValueError(f"Unknown device type {torch_device_type} for Dlpack")

            device_type = DLDeviceType.kDLCUDA
        elif torch_device_type == "mps":
            device_type = DLDeviceType.kDLMetal
        else:
            raise ValueError(f"Unknown device type {torch_device_type} for Dlpack")
        return (device_type, idx)

    __module__ = "torch"


def _convert(ret, cls):
    if cls is Tensor:
        return ret

    if isinstance(ret, Tensor) and not isinstance(ret, cls):
        ret = ret.as_subclass(cls)

    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret
<|endoftext|>import torch


def show() -> str:
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS is exposed, as Timer
# uses them.
def _cxx_flags() -> str:
    """Returns the CXX_FLAGS used when building PyTorch."""
    return torch._C._cxx_flags()


def parallel_info() -> str:
    r"""Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()
<|endoftext|>"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""

import functools
from collections.abc import Callable
from typing import overload, TypeVar
from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = True
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: None = None, recursive: bool = True
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Callable[_P, _T] | None = None, recursive: bool = True
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # cache this on the first invocation to avoid adding too much overhead.
            disable_fn = getattr(fn, "__dynamo_disable", None)
            if disable_fn is None:
                import torch._dynamo

                # We can safely turn off functools.wraps here because the inner
                # already wraps fn in the outer scope.
                disable_fn = torch._dynamo.disable(fn, recursive, wrapping=False)
                fn.__dynamo_disable = disable_fn  # type: ignore[attr-defined]

            return disable_fn(*args, **kwargs)

        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
<|endoftext|># mypy: allow-untyped-defs

import torch


class SobolEngine:
    r"""
    The :class:`torch.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``

    Examples::

        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    """

    MAXBIT = 30
    MAXDIM = 21201

    def __init__(self, dimension, scramble=False, seed=None):
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        cpu = torch.device("cpu")

        self.sobolstate = torch.zeros(
            dimension, self.MAXBIT, device=cpu, dtype=torch.long
        )
        torch._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        if not self.scramble:
            self.shift = torch.zeros(self.dimension, device=cpu, dtype=torch.long)
        else:
            self._scramble()

        self.quasi = self.shift.clone(memory_format=torch.contiguous_format)
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        self.num_generated = 0

    def draw(
        self,
        n: int = 1,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        if self.num_generated == 0:
            if n == 1:
                result = self._first_point.to(dtype)
            else:
                result, self.quasi = torch._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                result = torch.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            result, self.quasi = torch._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        self.num_generated += n

        if out is not None:
            out.resize_as_(result).copy_(result)
            return out

        return result

    def draw_base2(
        self,
        m: int,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        n = 2**m
        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        if n == 0:
            return self
        if self.num_generated == 0:
            torch._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            torch._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        self.num_generated += n
        return self

    def _scramble(self):
        g: torch.Generator | None = None
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)

        cpu = torch.device("cpu")

        # Generate shift vector
        shift_ints = torch.randint(
            2, (self.dimension, self.MAXBIT), device=cpu, generator=g
        )
        self.shift = torch.mv(
            shift_ints, torch.pow(2, torch.arange(0, self.MAXBIT, device=cpu))
        )

        # Generate lower triangular matrices (stacked across dimensions)
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = torch.randint(2, ltm_dims, device=cpu, generator=g).tril()

        torch._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)

    def __repr__(self):
        fmt_string = [f"dimension={self.dimension}"]
        if self.scramble:
            fmt_string += ["scramble=True"]
        if self.seed is not None:
            fmt_string += [f"seed={self.seed}"]
        return self.__class__.__name__ + "(" + ", ".join(fmt_string) + ")"
<|endoftext|># Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None


class CustomClassBaseMeta(type):
    def __instancecheck__(cls, instance):
        # When checking against CustomClassBase itself (not a concrete subclass),
        # delegate to the registration system which correctly covers all
        # custom classes (constant types, metaclass-only symbolic types, and
        # FakeScriptObject wrappers).
        if cls is CustomClassBase:
            from torch._library.opaque_object import is_custom_class_obj

            return is_custom_class_obj(instance)

        if super().__instancecheck__(instance):
            return True

        # Check FakeScriptObject before hasattr to avoid triggering custom
        # __getattr__ on arbitrary user objects (e.g. dict-like objects that
        # raise KeyError on unknown attributes).
        # e.g. test/dynamo/test_dynamic_shapes.py -k test_user_getattr1_dynamic_shapes
        global _FakeScriptObject_cls
        if _FakeScriptObject_cls is None:
            from torch._library.fake_class_registry import FakeScriptObject

            _FakeScriptObject_cls = FakeScriptObject
        if isinstance(instance, _FakeScriptObject_cls) and hasattr(
            instance, "real_obj"
        ):
            return super().__instancecheck__(instance.real_obj)

        return False


class CustomClassBase(metaclass=CustomClassBaseMeta):
    pass


# Backward compatibility aliases
OpaqueBaseMeta = CustomClassBaseMeta
OpaqueBase = CustomClassBase
<|endoftext|>#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2005-2010 ActiveState Software Inc.
# Copyright (c) 2013 Eddy Petrișor

# flake8: noqa

"""
This file is directly from
https://github.com/ActiveState/appdirs/blob/3fe6a83776843a46f20c2e5587afcffe05e03b39/appdirs.py

The license of https://github.com/ActiveState/appdirs copied below:


# This is the MIT license

Copyright (c) 2010 ActiveState Software Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""Utilities for determining application-specific dirs.

See <https://github.com/ActiveState/appdirs> for details and usage.
"""
# Dev Notes:
# - Windows "Known Folders": https://learn.microsoft.com/en-us/windows/win32/shell/csidl
# - macOS File System Programming Guide: https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/Introduction/Introduction.html
# - XDG spec for Un*x: https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html

__version__ = "1.4.4"
__version_info__ = tuple(int(segment) for segment in __version__.split("."))


import os
import sys


unicode = str

if sys.platform.startswith("java"):
    import platform

    os_name = platform.java_ver()[3][0]
    if os_name.startswith("Windows"):  # "Windows XP", "Windows 7", etc.
        system = "win32"
    elif os_name.startswith("Mac"):  # "Mac OS X", etc.
        system = "darwin"
    else:  # "Linux", "SunOS", "FreeBSD", etc.
        # Setting this to "linux2" is not ideal, but only Windows or Mac
        # are actually checked for and the rest of the module expects
        # *sys.platform* style strings.
        system = "linux2"
else:
    system = sys.platform


def user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user data directories are:
        Mac OS X:               ~/Library/Application Support/<AppName>
        Unix:                   ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined
        Win XP (not roaming):   C:\Documents and Settings\<username>\Application Data\<AppAuthor>\<AppName>
        Win XP (roaming):       C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>
        Win 7  (not roaming):   C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>
        Win 7  (roaming):       C:\Users\<username>\AppData\Roaming\<AppAuthor>\<AppName>

    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.
    That means, by default "~/.local/share/<AppName>".
    """
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        path = os.path.normpath(_get_win_folder(const))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
    elif system == "darwin":
        path = os.path.expanduser("~/Library/Application Support/")
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


def site_data_dir(appname=None, appauthor=None, version=None, multipath=False):
    r"""Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of data dirs should be
            returned. By default, the first item from XDG_DATA_DIRS is
            returned, or '/usr/local/share/<AppName>',
            if XDG_DATA_DIRS is not set

    Typical site data directories are:
        Mac OS X:   /Library/Application Support/<AppName>
        Unix:       /usr/local/share/<AppName> or /usr/share/<AppName>
        Win XP:     C:\Documents and Settings\All Users\Application Data\<AppAuthor>\<AppName>
        Vista:      (Fail! "C:\ProgramData" is a hidden *system* directory on Vista.)
        Win 7:      C:\ProgramData\<AppAuthor>\<AppName>   # Hidden, but writeable on Win 7.

    For Unix, this is using the $XDG_DATA_DIRS[0] default.

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    """
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder("CSIDL_COMMON_APPDATA"))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
    elif system == "darwin":
        path = os.path.expanduser("/Library/Application Support")
        if appname:
            path = os.path.join(path, appname)
    else:
        # XDG default for $XDG_DATA_DIRS
        # only first, if multipath is False
        path = os.getenv(
            "XDG_DATA_DIRS", os.pathsep.join(["/usr/local/share", "/usr/share"])
        )
        pathlist = [
            os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)
        ]
        if appname:
            if version:
                appname = os.path.join(appname, version)
            pathlist = [os.sep.join([x, appname]) for x in pathlist]

        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            path = pathlist[0]
        return path

    if appname and version:
        path = os.path.join(path, version)
    return path


def user_config_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific config dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user config directories are:
        Mac OS X:               ~/Library/Preferences/<AppName>
        Unix:                   ~/.config/<AppName>     # or in $XDG_CONFIG_HOME, if defined
        Win *:                  same as user_data_dir

    For Unix, we follow the XDG spec and support $XDG_CONFIG_HOME.
    That means, by default "~/.config/<AppName>".
    """
    if system == "win32":
        path = user_data_dir(appname, appauthor, None, roaming)
    elif system == "darwin":
        path = os.path.expanduser("~/Library/Preferences/")
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


def site_config_dir(appname=None, appauthor=None, version=None, multipath=False):
    r"""Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "multipath" is an optional parameter only applicable to *nix
            which indicates that the entire list of config dirs should be
            returned. By default, the first item from XDG_CONFIG_DIRS is
            returned, or '/etc/xdg/<AppName>', if XDG_CONFIG_DIRS is not set

    Typical site config directories are:
        Mac OS X:   same as site_data_dir
        Unix:       /etc/xdg/<AppName> or $XDG_CONFIG_DIRS[i]/<AppName> for each value in
                    $XDG_CONFIG_DIRS
        Win *:      same as site_data_dir
        Vista:      (Fail! "C:\ProgramData" is a hidden *system* directory on Vista.)

    For Unix, this is using the $XDG_CONFIG_DIRS[0] default, if multipath=False

    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.
    """
    if system == "win32":
        path = site_data_dir(appname, appauthor)
        if appname and version:
            path = os.path.join(path, version)
    elif system == "darwin":
        path = os.path.expanduser("/Library/Preferences")
        if appname:
            path = os.path.join(path, appname)
    else:
        # XDG default for $XDG_CONFIG_DIRS
        # only first, if multipath is False
        path = os.getenv("XDG_CONFIG_DIRS", "/etc/xdg")
        pathlist = [
            os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)
        ]
        if appname:
            if version:
                appname = os.path.join(appname, version)
            pathlist = [os.sep.join([x, appname]) for x in pathlist]

        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            path = pathlist[0]
    return path


def user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    r"""Return full path to the user-specific cache dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Cache" to the base app data dir for Windows. See
            discussion below.

    Typical user cache directories are:
        Mac OS X:   ~/Library/Caches/<AppName>
        Unix:       ~/.cache/<AppName> (XDG default)
        Win XP:     C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>\Cache
        Vista:      C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Cache

    On Windows the only suggestion in the MSDN docs is that local settings go in
    the `CSIDL_LOCAL_APPDATA` directory. This is identical to the non-roaming
    app data dir (the default returned by `user_data_dir` above). Apps typically
    put cache data somewhere *under* the given dir here. Some examples:
        ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
        ...\Acme\SuperApp\Cache\1.0
    OPINION: This function appends "Cache" to the `CSIDL_LOCAL_APPDATA` value.
    This can be disabled with the `opinion=False` option.
    """
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
            if opinion:
                path = os.path.join(path, "Cache")
    elif system == "darwin":
        path = os.path.expanduser("~/Library/Caches")
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


def user_state_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific state dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user state directories are:
        Mac OS X:  same as user_data_dir
        Unix:      ~/.local/state/<AppName>   # or in $XDG_STATE_HOME, if defined
        Win *:     same as user_data_dir

    For Unix, we follow this Debian proposal <https://wiki.debian.org/XDGBaseDirectorySpecification#state>
    to extend the XDG spec and support $XDG_STATE_HOME.

    That means, by default "~/.local/state/<AppName>".
    """
    if system in ["win32", "darwin"]:
        path = user_data_dir(appname, appauthor, None, roaming)
    else:
        path = os.getenv("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


def user_log_dir(appname=None, appauthor=None, version=None, opinion=True):
    r"""Return full path to the user-specific log dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Logs" to the base app data dir for Windows, and "log" to the
            base cache dir for Unix. See discussion below.

    Typical user log directories are:
        Mac OS X:   ~/Library/Logs/<AppName>
        Unix:       ~/.cache/<AppName>/log  # or under $XDG_CACHE_HOME if defined
        Win XP:     C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>\Logs
        Vista:      C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Logs

    On Windows the only suggestion in the MSDN docs is that local settings
    go in the `CSIDL_LOCAL_APPDATA` directory. (Note: I'm interested in
    examples of what some windows apps use for a logs dir.)

    OPINION: This function appends "Logs" to the `CSIDL_LOCAL_APPDATA`
    value for Windows and appends "log" to the user cache dir for Unix.
    This can be disabled with the `opinion=False` option.
    """
    if system == "darwin":
        path = os.path.join(os.path.expanduser("~/Library/Logs"), appname)
    elif system == "win32":
        path = user_data_dir(appname, appauthor, version)
        version = False
        if opinion:
            path = os.path.join(path, "Logs")
    else:
        path = user_cache_dir(appname, appauthor, version)
        version = False
        if opinion:
            path = os.path.join(path, "log")
    if appname and version:
        path = os.path.join(path, version)
    return path


class AppDirs:
    """Convenience wrapper for getting application dirs."""

    def __init__(
        self, appname=None, appauthor=None, version=None, roaming=False, multipath=False
    ):
        self.appname = appname
        self.appauthor = appauthor
        self.version = version
        self.roaming = roaming
        self.multipath = multipath

    @property
    def user_data_dir(self):
        return user_data_dir(
            self.appname, self.appauthor, version=self.version, roaming=self.roaming
        )

    @property
    def site_data_dir(self):
        return site_data_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    @property
    def user_config_dir(self):
        return user_config_dir(
            self.appname, self.appauthor, version=self.version, roaming=self.roaming
        )

    @property
    def site_config_dir(self):
        return site_config_dir(
            self.appname, self.appauthor, version=self.version, multipath=self.multipath
        )

    @property
    def user_cache_dir(self):
        return user_cache_dir(self.appname, self.appauthor, version=self.version)

    @property
    def user_state_dir(self):
        return user_state_dir(self.appname, self.appauthor, version=self.version)

    @property
    def user_log_dir(self):
        return user_log_dir(self.appname, self.appauthor, version=self.version)


# ---- internal support stuff


def _get_win_folder_from_registry(csidl_name):
    """This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
    import winreg as _winreg

    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }[csidl_name]

    key = _winreg.OpenKey(
        _winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
    )
    dir, _type = _winreg.QueryValueEx(key, shell_folder_name)
    return dir


def _get_win_folder_with_pywin32(csidl_name):
    from win32com.shell import shell, shellcon

    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    # Try to make this a unicode path because SHGetFolderPath does
    # not return unicode strings when there is unicode data in the
    # path.
    try:
        dir = unicode(dir)

        # Downgrade to short path name if have highbit chars. See
        # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
        has_high_char = False
        for c in dir:
            if ord(c) > 255:
                has_high_char = True
                break
        if has_high_char:
            try:
                import win32api

                dir = win32api.GetShortPathName(dir)
            except ImportError:
                pass
    except UnicodeError:
        pass
    return dir


def _get_win_folder_with_ctypes(csidl_name):
    import ctypes

    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    return buf.value


def _get_win_folder_with_jna(csidl_name):
    import array

    from com.sun import jna
    from com.sun.jna.platform import win32

    buf_size = win32.WinDef.MAX_PATH * 2
    buf = array.zeros("c", buf_size)
    shell = win32.Shell32.INSTANCE
    shell.SHGetFolderPath(
        None,
        getattr(win32.ShlObj, csidl_name),
        None,
        win32.ShlObj.SHGFP_TYPE_CURRENT,
        buf,
    )
    dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in dir:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf = array.zeros("c", buf_size)
        kernel = win32.Kernel32.INSTANCE
        if kernel.GetShortPathName(dir, buf, buf_size):
            dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    return dir


if system == "win32":
    try:
        import win32com.shell

        _get_win_folder = _get_win_folder_with_pywin32
    except ImportError:
        try:
            from ctypes import windll

            _get_win_folder = _get_win_folder_with_ctypes
        except ImportError:
            try:
                import com.sun.jna

                _get_win_folder = _get_win_folder_with_jna
            except ImportError:
                _get_win_folder = _get_win_folder_from_registry


# ---- self test code

if __name__ == "__main__":
    appname = "MyApp"
    appauthor = "MyCompany"

    props = (
        "user_data_dir",
        "user_config_dir",
        "user_cache_dir",
        "user_state_dir",
        "user_log_dir",
        "site_data_dir",
        "site_config_dir",
    )

    print(f"-- app dirs {__version__} --")

    print("-- app dirs (with optional 'version')")
    dirs = AppDirs(appname, appauthor, version="1.0")
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'version')")
    dirs = AppDirs(appname, appauthor)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'appauthor')")
    dirs = AppDirs(appname)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (with disabled 'appauthor')")
    dirs = AppDirs(appname, appauthor=False)
    for prop in props:
        print(f"{prop}: {getattr(dirs, prop)}")
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
import inspect
import re
import sys
import weakref
from collections.abc import Callable, Sequence
from typing import Any, overload, TYPE_CHECKING, TypeVar, Union
from typing_extensions import deprecated, ParamSpec

import torch
import torch._library as _library
from torch._library.custom_ops import (
    _cast,
    _maybe_get_opdef,
    custom_op,
    CustomOpDef,
    device_types_t,
)
from torch._library.effects import EffectType
from torch._library.infer_schema import infer_schema
from torch._library.triton import triton_op, wrap_triton
from torch._ops import OpOverload
from torch.types import _dtype


__all__ = [
    "Library",
    "impl",
    "define",
    "fallthrough_kernel",
    "impl_abstract",
    "register_autocast",
    "register_fake",
    "register_torch_dispatch",
    "register_vmap",
    "get_ctx",
    "get_kernel",
    "custom_op",
    "triton_op",
    "wrap_triton",
    "infer_schema",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: set[str] = set()
_defs: set[str] = set()

# prim is reserved by TorchScript interpreter
_reserved_namespaces = ["prim"]


def fallthrough_kernel():
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    """
    raise NotImplementedError("fallthrough_kernel() should never be called.")


def _validate_out_schema(schema: "str | torch._C.FunctionSchema") -> None:
    """Validate that a schema has valid out semantics, i.e., it can be tagged with torch.Tag.out.

    Requirements:
    - Must have at least one mutable argument
    - All returns must alias the mutable args in declaration order

    torchgen has equivalent checks (torchgen/model.py), but we reimplement them here
    because (1) it's simple and (2) torchgen uses a different schema object
    (torchgen.model.FunctionSchema vs torch._C.FunctionSchema) so it's difficult to
    share the function.
    """
    if isinstance(schema, str):
        schema = torch._C.parse_schema(schema)
    mutable_args = [
        arg
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    if not mutable_args:
        raise ValueError(
            f"Schema tagged with torch.Tag.out must have at least one mutable argument. "
            f"Got: {schema}"
        )
    positional_mutable = [arg for arg in mutable_args if not arg.kwarg_only]
    if positional_mutable:
        names = [a.name for a in positional_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.out requires all mutable arguments to be "
            f"keyword-only (after the *). Found mutable positional args: {names}. "
            f"Got: {schema}"
        )
    unsupported_mutable = [
        arg for arg in mutable_args if not isinstance(arg.type, torch.TensorType)
    ]
    if unsupported_mutable:
        names = [a.name for a in unsupported_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.out only supports Tensor mutable arguments. "
            f"Found unsupported mutable args: {names}. Got: {schema}"
        )
    returns = schema.returns
    if len(returns) != len(mutable_args):
        raise ValueError(
            f"Schema tagged with torch.Tag.out must return all mutable arguments "
            f"(got {len(mutable_args)} mutable args but {len(returns)} returns). "
            f"Got: {schema}"
        )
    for i, (ret, arg) in enumerate(zip(returns, mutable_args, strict=True)):
        arg_alias = arg.alias_info
        ret_alias = ret.alias_info
        if ret_alias is None:
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must alias mutable arg '{arg.name}'. "
                f"Got: {schema}"
            )
        if not ret_alias.is_write:
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must be a mutable alias "
                f"(e.g., Tensor(a!), not Tensor(a)) of arg '{arg.name}'. "
                f"Got: {schema}"
            )
        # arg_alias is guaranteed non-None by the mutable_args filter above
        if ret_alias.before_set != arg_alias.before_set:  # type: ignore[union-attr]
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must alias mutable arg '{arg.name}' "
                f"(return aliases {ret_alias.before_set} but arg aliases {arg_alias.before_set}). "  # type: ignore[union-attr]
                f"Got: {schema}"
            )


def _validate_inplace_schema(schema: "str | torch._C.FunctionSchema") -> None:
    """Validate that a schema has valid inplace semantics, i.e., it can be tagged with torch.Tag.inplace.

    Requirements:
    - The first positional argument must be a mutable Tensor (Tensor(a!))
    - Must return exactly that Tensor
    - Only the first positional argument may be mutable

    Native inplace ops (defined in native_functions.yaml) have relaxed requirements:
    they may have Tensor[] self and multiple mutable arguments. This validation is
    stricter because it is only used for custom ops defined via torch.library.
    """
    if isinstance(schema, str):
        schema = torch._C.parse_schema(schema)
    args = schema.arguments
    if not args or args[0].kwarg_only:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must have at least one positional argument. "
            f"Got: {schema}"
        )
    first_arg = args[0]
    if first_arg.alias_info is None or not first_arg.alias_info.is_write:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace requires the first positional argument "
            f"to be mutable (e.g., Tensor(a!)). Got: {schema}"
        )
    if not isinstance(first_arg.type, torch.TensorType):
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace requires the first positional argument "
            f"to be a Tensor. Got type '{first_arg.type}' for arg '{first_arg.name}'. "
            f"Got: {schema}"
        )
    # Must return exactly one Tensor aliasing the first arg
    returns = schema.returns
    if len(returns) != 1:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return exactly one value "
            f"(the first argument). Got {len(returns)} returns. Got: {schema}"
        )
    ret = returns[0]
    if not isinstance(ret.type, torch.TensorType):
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must be a Tensor, got type '{ret.type}'). Got: {schema}"
        )
    if ret.alias_info is None:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must alias the first argument). Got: {schema}"
        )
    if not ret.alias_info.is_write:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must be a mutable alias, e.g., Tensor(a!)). Got: {schema}"
        )
    if ret.alias_info.before_set != first_arg.alias_info.before_set:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return aliases {ret.alias_info.before_set} but first arg aliases "
            f"{first_arg.alias_info.before_set}). Got: {schema}"
        )
    other_mutable = [
        arg
        for arg in args[1:]
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    if other_mutable:
        names = [a.name for a in other_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must only mutate the first positional argument. "
            f"Found additional mutable args: {names}. Got: {schema}"
        )


class Library:
    """
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    To create a library to override operators in an existing library (with name ns), set the kind to "IMPL".
    To create a new library (with name ns) to register new operators, set the kind to "DEF".
    To create a fragment of a possibly existing library to register operators (and bypass
    the limitation that there is only one library for a given namespace), set the kind to
    "FRAGMENT".

    Args:
        ns: library name
        kind: "DEF", "IMPL", "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    """

    def __init__(self, ns, kind, dispatch_key=""):
        from torch.fx.operator_schemas import _SCHEMA_TO_SIGNATURE_CACHE

        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError(f"Unsupported kind: {kind}")

        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                f"{ns} is a reserved namespace. Please try creating a library with another name."
            )

        f = sys._getframe(1)
        filename, lineno = f.f_code.co_filename, f.f_lineno
        self.m: Any | None = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        self.ns = ns
        self._op_defs: set[str] = set()
        self._op_impls: set[str] = set()
        self._registration_handles: list[torch._library.utils.RegistrationHandle] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        # Use a finalizer to setup the "destructor" instead of __del__.
        # Python __del__ can lead to weird things (globals and locals may already
        # be gone when __del__ actually gets called!). finalizers help the
        # situation because it lets us capture references and keeps them alive
        weakref.finalize(
            self,
            _del_library,
            _impls,
            self._op_impls,
            _defs,
            self._op_defs,
            self._registration_handles,
            self.m,
            _SCHEMA_TO_SIGNATURE_CACHE,
        )

    def __repr__(self):
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"

    def define(self, schema, alias_analysis="", *, tags=()):
        r"""Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
                                       operator. Tagging an operator changes the operator's behavior
                                       under various PyTorch subsystems; please read the docs for the
                                       torch.Tag carefully before applying it.

        Returns:
            name of the operator as inferred from the schema.

        Example::

            >>> my_lib = Library("mylib", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        """

        # This is added because we also want to disallow PURE_FUNCTION alias analysis which is a valid
        # AliasAnalysis type in C++
        if alias_analysis not in ["", "FROM_SCHEMA", "CONSERVATIVE"]:
            raise RuntimeError(f"Invalid alias_analysis type {alias_analysis}")
        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        if isinstance(tags, torch.Tag):
            tags = (tags,)

        name = schema.split("(")[0]
        packet_name = name.split(".")[0] if "." in name else name
        has_preexisting_packet = hasattr(torch.ops, self.ns) and hasattr(
            getattr(torch.ops, self.ns), packet_name
        )

        if torch.Tag.out in tags:
            _validate_out_schema(schema)
        if torch.Tag.inplace in tags:
            _validate_inplace_schema(schema)

        result = self.m.define(schema, alias_analysis, tuple(tags))
        name = schema.split("(")[0]
        # C++ accepts a name prefixed with the matching namespace ("ns::foo");
        # don't double-prepend the namespace in that case.
        qualname = name if "::" in name else f"{self.ns}::{name}"

        # If the OpOverloadPacket exists already, then this means we're adding a
        # new OpOverload for it. Refresh the packet to include the new OpOverload.
        if has_preexisting_packet:
            ns = getattr(torch.ops, self.ns)
            packet = getattr(ns, packet_name)
            torch._ops._refresh_packet(packet)

        self._op_defs.add(qualname)
        _defs.add(qualname)
        return result

    def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=True):
        r"""Registers the fake impl for an operator defined in the library."""

        source = torch._library.utils.get_source(_stacklevel + 1)
        frame = sys._getframe(_stacklevel)
        caller_module = inspect.getmodule(frame)
        # Can be none if you call register_fake from somewhere there isn't a module
        # (e.g. __main__)
        caller_module_name = None if caller_module is None else caller_module.__name__

        # TODO(rzou): We're gonna need to stage this change with torchvision,
        # since torchvision is github first.
        if caller_module_name is not None and caller_module_name.startswith(
            "torchvision."
        ):
            caller_module_name = None

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(fn, qualname, caller_module_name)
        else:
            func_to_register = fn

        handle = entry.fake_impl.register(
            func_to_register, source, lib=self, allow_override=allow_override
        )
        self._registration_handles.append(handle)

    def _register_torch_dispatch_rule(self, op_name, torch_dispatch_class, fn):
        r"""Registers a torch_dispatch rule for the given operator and torch_dispatch_class.

        This allows for open registration to specify the behavior between the operator
        and the torch_dispatch_class without needing to modify the torch_dispatch_class
        or the operator directly.

        The torch_dispatch_class is either a Tensor subclass with `__torch_dispatch__` or a
        TorchDispatchMode.

        If it is a Tensor subclass, we expect fn to have the following signature:
        (cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any

        If it is a TorchDispatchMode, we expect fn to have the following signature:
        (mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any
        """

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        handle = entry.torch_dispatch_rules.register(torch_dispatch_class, fn)
        self._registration_handles.append(handle)

    def _resolve_op_name(self, op_name, api_name):
        """Resolve op_name (str or OpOverload) to a name string."""
        if isinstance(op_name, str):
            return op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name:
                name = name + "." + overload_name
            return name
        else:
            raise RuntimeError(
                f"{api_name} should be passed either a name or an OpOverload object "
                f"as the first argument, got {type(op_name)}"
            )

    def _impl_with_aoti_compile(self, op_name, dispatch_key=""):
        r"""Register the operator to use the AOTI-compiled implementation.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::

            >>> my_lib = Library("aten", "IMPL")
            >>> warnings.filterwarnings("ignore", message=".*other operators may also be overridden")  # docs: hide
            >>> my_lib._impl_with_aoti_compile("div.Tensor", "CPU")
        """

        if dispatch_key == "":
            dispatch_key = self.dispatch_key
        # pyrefly: ignore [bad-argument-type]
        if not torch.DispatchKeySet(dispatch_key).has(torch._C.DispatchKey.Dense):
            raise AssertionError(
                f"dispatch_key {dispatch_key} does not have Dense in its keyset"
            )

        name = self._resolve_op_name(op_name, "_impl_with_aoti_compile")

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when _impl_with_aoti_compile is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        impl_fn: Callable = self.m.impl_with_aoti_compile
        impl_fn(self.ns, name.split("::")[-1], dispatch_key)

        _impls.add(key)
        self._op_impls.add(key)

    def impl(
        self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=True
    ):
        r"""Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that's the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.
            allow_override: Flag controlling if we want to override an
                         existing registered kernel implementation. This is on
                         by default; pass ``False`` to error if a kernel is
                         already registered for this dispatch key.

        Example::
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> my_lib = Library("aten", "IMPL")
            >>> def div_cpu(self, other):
            >>>     return self * (1 / other)
            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")
        """

        if not callable(fn):
            raise TypeError(
                f"Input function is required to be a callable but found type {type(fn)}"
            )
        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        name = self._resolve_op_name(op_name, "impl")

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if (not allow_override) and key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when impl is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if dispatch_key == "Meta":
            dispatcher_op_name = name
            if "::" not in dispatcher_op_name:
                dispatcher_op_name = f"{self.ns}::{dispatcher_op_name}"

            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                dispatcher_op_name, "CompositeImplicitAutograd"
            ):
                raise RuntimeError(
                    f"We should not register a meta kernel directly to the operator '{name}',"
                    " because it has a CompositeImplicitAutograd kernel in core."
                    " Instead we should let the operator decompose, and ensure that we have meta kernels"
                    " for the base ops that it decomposes into."
                )

        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        self.m.impl(
            name,
            dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd",
            fn,
            with_keyset,
        )

        _impls.add(key)
        self._op_impls.add(key)

    def register_symm_mem_args(self, op_name, arg_names):
        r"""Registers which arguments require symmetric memory allocation for an operator.

        This method allows operators to declaratively specify which arguments need
        symmetric memory treatment. When used with ``torch.compile``, Inductor
        automatically allocates these arguments in P2P-accessible NVLink memory
        (``empty_strided_p2p``), enabling zero-copy inter-GPU communication.

        The operator's schema should include a ``group_name`` (``str``) argument.
        Inductor's ``FallbackKernel._maybe_realize_symm_mem_args`` extracts
        ``group_name`` at compile time to allocate P2P buffers for the correct
        process group. Without ``group_name``, the automatic realization is skipped.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            arg_names: list of argument names that require symmetric memory allocation.

        Example::
            >>> # xdoctest: +SKIP(reason="illustrative example, not runnable")
            >>> my_lib = Library("symm_mem", "FRAGMENT")
            >>> my_lib.define("one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor")
            >>> my_lib.register_symm_mem_args("one_shot_all_reduce", ["input"])
        """
        name = self._resolve_op_name(op_name, "register_symm_mem_args")
        if "::" in name:
            qualname = name
        else:
            qualname = f"{self.ns}::{name}"

        entry = torch._library.simple_registry.singleton.find(qualname)
        handle = entry.symm_mem_args.register(arg_names)
        self._registration_handles.append(handle)

    def fallback(self, fn, dispatch_key="", *, with_keyset=False):
        r"""Registers the function implementation as the fallback for the given key.

        This function only works for a library with global namespace ("_").

        Args:
            fn: function used as fallback for the given dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.

        Example::

            >>> my_lib = Library("_", "IMPL")
            >>> def fallback_kernel(op, *args, **kwargs):
            >>>     # Handle all autocast ops generically
            >>>     # ...
            >>> my_lib.fallback(fallback_kernel, "Autocast")
        """

        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        if self.ns != "_":
            raise RuntimeError(
                f"""Fallback can only be registered using library fragment on the global namespace "_" but it is {self.ns}"""
            )

        if dispatch_key == "":
            raise AssertionError("dispatch_key must not be empty for fallback")
        if self.m is None:
            raise AssertionError("Library object has been destroyed")

        self.m.fallback(dispatch_key, fn, with_keyset)

    def _register_effectful_op(self, op_name: str, effect: EffectType | None):
        """
        Registers an effect to an operator. This is used to register an op that
        has side effects that is not capturable by the schema.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            effect: The effect of the op.
        """
        from torch._higher_order_ops.effects import (
            _register_effectful_op as hoo_register_effect,
        )

        handle = hoo_register_effect(op_name, effect)
        self._registration_handles.append(handle)

    def _destroy(self):
        if self.m is not None:
            self.m.reset()
        self.m = None
        for handle in self._registration_handles:
            handle.destroy()
        self._registration_handles.clear()
        global _impls
        _impls -= self._op_impls
        _clear_torch_ops_cache(self._op_defs)


def _clear_torch_ops_cache(op_defs):
    # Delete the cached torch.ops.ns.foo if it was registered.
    # Otherwise, accessing it leads to a segfault, or to enumerations
    # over the namespace (e.g. _collect_all_valid_cia_ops) tripping on
    # an OpOverloadPacket whose underlying overload was just removed.
    # It's possible that we only registered an overload in this Library
    # and another library owns an alive overload.
    # That's OK - the next time torch.ops.ns.foo gets called, it'll be
    # recomputed to point at the right collection of overloads.
    for qualname in op_defs:
        splits = qualname.split("::")
        if len(splits) != 2:
            # Defense-in-depth: this runs in a shutdown-time finalizer that must
            # never raise, so tolerate any qualname that isn't a clean
            # "namespace::name" instead of unpacking blindly.
            continue
        ns, name_with_overload = splits
        name = name_with_overload.split(".")[0]
        if not hasattr(torch.ops, ns):
            continue
        namespace = getattr(torch.ops, ns)
        # Use vars() to check the instance dict directly, avoiding
        # __getattr__ which calls into C++ via _jit_get_operation.
        # During interpreter shutdown the C++ runtime may already be
        # torn down, causing UnicodeDecodeError or segfaults.
        if name not in vars(namespace):
            continue
        delattr(namespace, name)
        if name in namespace._dir:
            namespace._dir.remove(name)


def _del_library(
    captured_impls,
    op_impls,
    captured_defs,
    op_defs,
    registration_handles,
    m,
    schema_to_signature_cache,
):
    for op_def in op_defs:
        name = op_def
        overload_name = ""
        if "." in op_def:
            name, overload_name = op_def.split(".")
        if (
            name,
            overload_name,
        ) in schema_to_signature_cache:
            del schema_to_signature_cache[(name, overload_name)]

    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()

    if m is not None:
        m.reset()

    _clear_torch_ops_cache(op_defs)


@contextlib.contextmanager
def _scoped_library(*args, **kwargs):
    try:
        lib = Library(*args, **kwargs)
        yield lib
    finally:
        lib._destroy()


_keep_alive: list[Library] = []


NAMELESS_SCHEMA = re.compile(r"\(.*\) -> .*")


@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()):
    r"""Defines a new operator.

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs, like :func:`torch.library.impl` or
    :func:`torch.library.register_fake`.

    Args:
        qualname (str): The qualified name for the operator. Should be
            a string that looks like "namespace::name", e.g. "aten::sin".
            Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"
            for an op that accepts one Tensor and returns one Tensor. It does
            not contain the operator name (that is passed in ``qualname``).
        lib (Optional[Library]): If provided, the lifetime of this operator
            will be tied to the lifetime of the Library object.
        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
            operator. Tagging an operator changes the operator's behavior
            under various PyTorch subsystems; please read the docs for the
            torch.Tag carefully before applying it.

    Example::
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x.sin())

    """
    if not isinstance(qualname, str):
        raise ValueError(
            f"define(qualname, schema): expected qualname "
            f"to be instance of str, got {type(qualname)}"
        )
    namespace, name = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(
            f"define(qualname, schema, ...): expected schema "
            f'to look like e.g. "(Tensor x) -> Tensor" but '
            f'got "{schema}"'
        )
    lib.define(name + schema, alias_analysis="", tags=tags)


@define.register
def _(lib: Library, schema, alias_analysis=""):
    """The old torch.library.define.
    We're keeping this around for BC reasons
    """

    def wrap(f):
        name = lib.define(schema, alias_analysis)
        lib.impl(name, f)
        return f

    return wrap


@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: None = None,
    *,
    lib: Library | None = None,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object],
    *,
    lib: Library | None = None,
) -> None: ...


# Deprecated BC API
@overload
def impl(
    lib: Library,
    name: str,
    dispatch_key: str = "",
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@functools.singledispatch
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[_P, _T] | None = None,
    *,
    lib: Library | None = None,
) -> object:
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    This API may be used as a decorator. You can use nested decorators
    with this API provided they return a function and are placed inside
    this API (see Example 2).

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> # xdoctest: +SKIP("Requires Python <= 3.11")
        >>> import torch
        >>> import numpy as np
        >>> # Example 1: Register function.
        >>> # Define the operator
        >>> torch.library.define("mylib::mysin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylib::mysin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.mysin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example 2: Register function with decorator.
        >>> def custom_decorator(func):
        >>>     def wrapper(*args, **kwargs):
        >>>         return func(*args, **kwargs) + 1
        >>>     return wrapper
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin_plus_one", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin_plus_one", "cpu")
        >>> @custom_decorator
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>>
        >>> y1 = torch.ops.mylib.sin_plus_one(x)
        >>> y2 = torch.sin(x) + 1
        >>> assert torch.allclose(y1, y2)
    """

    return _impl(qualname, types, func, lib=lib, disable_dynamo=False)


if not TYPE_CHECKING:

    @impl.register
    def _(
        lib: Library, name: str, dispatch_key: str = ""
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        """Legacy torch.library.impl API. Kept around for BC"""

        def wrap(f: Callable[_P, _T]) -> Callable[_P, _T]:
            lib.impl(name, f, dispatch_key)
            return f

        return wrap


@overload
def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: None = None,
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object],
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> None: ...


def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object] | None = None,
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> Callable[[Callable[..., object]], None] | None:
    # See impl()
    if isinstance(types, str):
        types = (types,)
    keys = set({})
    for typ in types:
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            # We also support passing a DispatchKey to impl. Please prefer using
            # the higher-level torch.library APIs and only pass DispatchKey to
            # torch.library.impl with caution (or even better, don't use this
            # option and file an issue on GitHub for what you need).
            # We don't advertise this to users because
            # it is very easy to shoot yourself in the foot.
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    def register_(func: Callable[..., object]) -> None:
        namespace, _ = torch._library.utils.parse_namespace(qualname)

        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        if disable_dynamo:

            @torch._disable_dynamo
            def func_no_dynamo(*args, **kwargs):
                return func(*args, **kwargs)

            for key in keys:
                use_lib.impl(qualname, func_no_dynamo, key)
        else:
            for key in keys:
                use_lib.impl(qualname, func, key)

    if func is None:
        return register_
    else:
        register_(func)
        return None


def _device_type_to_key(device_type: str) -> str:
    if device_type == "default":
        # This is technically not correct, because although all device_type
        # DispatchKeys are included in CompositeExplicitAutograd,
        # not everything in CompositeExplicitAutograd is associated with a
        # device_type. I don't really care that much about the difference.
        return "CompositeExplicitAutograd"
    return torch._C._dispatch_key_for_device(device_type)


@deprecated(
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that "
    "instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.",
    category=FutureWarning,
)
def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    r"""This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
    if func is not None:
        _stacklevel = _stacklevel + 1
    return register_fake(qualname, func, lib=lib, _stacklevel=_stacklevel)


_op_identifier = Union[
    str, "torch._ops.OpOverload", "torch._library.custom_ops.CustomOpDef"
]


def _resolve_op_for_registration(
    op: _op_identifier, registration_name: str
) -> str | CustomOpDef:
    if not isinstance(op, (str, OpOverload, CustomOpDef)):
        raise ValueError(
            f"{registration_name}({op}): got unexpected type for op: {type(op)}"
        )
    if isinstance(op, OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef
    if not isinstance(op, str):
        raise AssertionError(f"op must be str at this point, got {type(op).__name__}")
    return op


def _library_for_registration(namespace: str, lib: Library | None) -> Library:
    if lib is not None:
        return lib
    use_lib = Library(namespace, "FRAGMENT")
    _keep_alive.append(use_lib)
    return use_lib


def register_kernel(
    op: _op_identifier,
    device_types: device_types_t,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
):
    """Register an implementation for a device type for this operator.

    Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
    This API may be used as a decorator.

    Args:
        op (str | OpOverload): The operator to register an impl to.
        device_types (str | None | Sequence[str]): The device_types to register an impl to.
            If None, we will register to all device types -- please only use
            this option if your implementation is truly device-type-agnostic.
        func (Callable): The function to register as the implementation for
            the given device types.
        lib (Optional[Library]): If provided, the lifetime of this registration

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> # Create a custom op that works on cpu
        >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> # Add implementations for the cuda device
        >>> @torch.library.register_kernel("mylib::numpy_sin", "cuda")
        >>> def _(x):
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x_cpu = torch.randn(3)
        >>> x_cuda = x_cpu.cuda()
        >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
        >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

    """

    op = _resolve_op_for_registration(op, "register_kernel")
    if isinstance(op, CustomOpDef):
        return op.register_kernel(device_types, func)
    if device_types is None:
        device_types = "CompositeExplicitAutograd"

    return _impl(op, device_types, func, lib=lib, disable_dynamo=True)


def register_autocast(
    op: _op_identifier,
    device_type: str,
    cast_inputs: _dtype,
    /,
    *,
    lib: Library | None = None,
):
    r"""Register an autocast dispatch rule for this custom op.

    Valid ``device_type`` values include any device type that supports autocast.
    See :func:`torch.amp.is_autocast_available` for details.

    Args:
        op (str | OpOverload): The operator to register an autocast dispatch rule to.
        device_type(str):  Device type to use. 'cuda', 'cpu', 'xpu', or any other device type that supports autocast.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`torch.dtype`): When custom op runs in an autocast-enabled region,
            casts incoming floating-point Tensors to the target dtype (non-floating-point Tensors
            are not affected), then executes custom op with autocast disabled.
        lib (Optional[Library]): If provided, the lifetime of this registration

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>>
        >>> # Create a custom op that works on cuda
        >>> @torch.library.custom_op("mylib::my_sin", mutates_args=())
        >>> def my_sin(x: Tensor) -> Tensor:
        >>>     return torch.sin(x)
        >>>
        >>> # Register autocast dispatch rule for the cuda device
        >>> torch.library.register_autocast("mylib::my_sin", "cuda", torch.float16)
        >>>
        >>> x = torch.randn(3, dtype=torch.float32, device="cuda")
        >>> with torch.autocast("cuda", dtype=torch.float16):
        >>>     y = torch.ops.mylib.my_sin(x)
        >>> assert y.dtype == torch.float16

    """
    op = _resolve_op_for_registration(op, "register_autocast")
    if not torch._C._is_autocast_available(device_type):
        raise ValueError(f"Device type '{device_type}' does not support autocast.")

    if isinstance(op, CustomOpDef):
        return op.register_autocast(device_type, cast_inputs)
    qualname = op
    _op = torch._library.utils.lookup_op(qualname)

    namespace, opname = torch._library.utils.parse_namespace(qualname)
    use_lib = _library_for_registration(namespace, lib)

    autocast_key = "Autocast" + torch._C._dispatch_key_for_device(device_type)
    autocast_dispatch_key = getattr(torch._C.DispatchKey, autocast_key)

    def _maybe_override_py_impl(op: torch._ops.OpOverload, dispatch_key):
        def inner(kernel):
            if op.has_kernel_for_dispatch_key(dispatch_key):
                op.py_kernels.pop(dispatch_key)
            return op.py_impl(dispatch_key)(kernel)

        return inner

    @_maybe_override_py_impl(_op, autocast_dispatch_key)
    def _autocast_py_impl(*args, **kwargs):
        if len(kwargs) != 0:
            raise AssertionError("Custom ops do not support kwargs yet.")
        with torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(autocast_dispatch_key)
        ):
            return _op(*_cast(args, device_type, cast_inputs))

    def kernel(_, *args, **kwargs):
        if len(kwargs) != 0:
            raise AssertionError("Custom ops do not support kwargs yet.")
        return _autocast_py_impl(*args, **kwargs)

    return use_lib.impl(opname, kernel, autocast_key, with_keyset=True)


def register_fake(
    op: _op_identifier,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
    _stacklevel: int = 1,
    allow_override: bool = True,
):
    r"""Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    An "FakeTensor implementation" specifies the behavior of this operator on
    Tensors that carry no data ("FakeTensor"). Given some input Tensors with
    certain properties (sizes/strides/storage_offset/device), it specifies
    what the properties of the output Tensors are.

    The FakeTensor implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write a FakeTensor
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The FakeTensor implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        func: Fake tensor implementation.
        lib (Optional[Library]): Library to register the fake tensor to.
        allow_override: Flag controlling if we want to override an
                        existing registered fake impl. This is on by default;
                        pass ``False`` to error if the operator already has a
                        fake impl. This also only applies if the custom operator
                        was not created via torch.library.custom_op, as
                        overriding an existing fake impl is already allowed.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_linear", mutates_args=())
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError("Implementation goes here")
        >>>
        >>> @torch.library.register_fake("mylib::custom_linear")
        >>> def _(x, weight, bias):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> with torch._subclasses.fake_tensor.FakeTensorMode():
        >>>     x = torch.randn(2, 3)
        >>>     w = torch.randn(3, 3)
        >>>     b = torch.randn(3)
        >>>     y = torch.ops.mylib.custom_linear(x, w, b)
        >>>
        >>> assert y.shape == (2, 3)
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_nonzero", mutates_args=())
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     return torch.tensor(res, device=x.device)
        >>>
        >>> @torch.library.register_fake("mylib::custom_nonzero")
        >>> def _(x):
        >>> # Number of nonzero-elements is data-dependent.
        >>> # Since we cannot peek at the data in an fake impl,
        >>> # we use the ctx object to construct a new symint that
        >>> # represents the data-dependent size.
        >>>     ctx = torch.library.get_ctx()
        >>>     nnz = ctx.new_dynamic_size()
        >>>     shape = [nnz, x.dim()]
        >>>     result = x.new_empty(shape, dtype=torch.int64)
        >>>     return result
        >>>
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>>
        >>> x = torch.tensor([0, 1, 2, 3, 4, 0])
        >>> trace = make_fx(torch.ops.mylib.custom_nonzero, tracing_mode="symbolic")(x)
        >>> trace.print_readable()
        >>>
        >>> assert torch.allclose(trace(x), torch.ops.mylib.custom_nonzero(x))

    """
    op = _resolve_op_for_registration(op, "register_fake")
    if isinstance(op, CustomOpDef):
        if func is None:
            return op.register_fake
        else:
            return op.register_fake(func)

    stacklevel = _stacklevel

    def register(func):
        namespace, op_name = torch._library.utils.parse_namespace(op)
        use_lib = _library_for_registration(namespace, lib)
        use_lib._register_fake(
            op_name, func, _stacklevel=stacklevel + 1, allow_override=allow_override
        )
        return func

    if func is None:
        return register
    else:
        stacklevel += 1
        return register(func)


def _register_effectful_op(
    op: _op_identifier,
    effect: EffectType | None,
    *,
    lib: Library | None = None,
) -> None:
    r"""
    To specify that an operator has side-effects, we must register an effect
    type for the operator. This will prevent graph passes in torch.compile from
    reordering operations with the same effect type.

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        effect: Effect type to register. None means the operator is not effectful.
    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_effectful_op({op}): got unexpected type for op: {type(op)}"
        )

    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        opdef.register_effect(effect)
    if not isinstance(op, str):
        raise AssertionError(f"op must be str at this point, got {type(op).__name__}")

    namespace, _ = torch._library.utils.parse_namespace(op)
    if lib is None:
        use_lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(use_lib)
    else:
        use_lib = lib
    use_lib._register_effectful_op(op, effect)


def register_autograd(
    op: _op_identifier,
    backward: Callable,
    /,
    *,
    setup_context: Callable | None = None,
    lib=None,
) -> None:
    r"""Register a backward formula for this custom op.

    In order for an operator to work with autograd, you need to register
    a backward formula:
    1. You must tell us how to compute gradients during the backward pass
    by providing us a "backward" function.
    2. If you need any values from the forward to compute gradients, you can
    use `setup_context` to save values for backward.

    ``backward`` runs during the backward pass. It accepts ``(ctx, *grads)``:
    - ``grads`` is one or more gradients. The number of gradients matches
    the number of outputs of the operator.
    The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
    :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
    same as :meth:`torch.autograd.Function.backward`.

    ``setup_context(ctx, inputs, output)`` runs during the forward pass.
    Please save quantities needed for backward onto the ``ctx`` object via
    either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
    or assigning them as attributes of ``ctx``. If your custom op has
    kwarg-only arguments, we expect the signature of ``setup_context``
    to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

    Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
    they may not directly access :meth:`torch.Tensor.data_ptr` and they must
    not depend on or mutate global state. If you need a non-traceable backward,
    you can make it a separate custom_op that you call inside ``backward_fn``.

    If you need different autograd behavior on different devices, then we
    recommend creating two different custom operators, one for each device
    that needs different behavior, and switching between them at runtime.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> def setup_context(ctx, inputs, output) -> Tensor:
        >>>     x, = inputs
        >>>     ctx.save_for_backward(x)
        >>>
        >>> def backward(ctx, grad):
        >>>     x, = ctx.saved_tensors
        >>>     return grad * x.cos()
        >>>
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_sin", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_sin(x)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, x.cos())
        >>>
        >>> # Example with a keyword-only arg
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, *, val: float) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = x_np * val
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> def setup_context(ctx, inputs, keyword_only_inputs, output) -> Tensor:
        >>>     ctx.val = keyword_only_inputs["val"]
        >>>
        >>> def backward(ctx, grad):
        >>>     return grad * ctx.val
        >>>
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_mul", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_mul(x, val=3.14)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

    """
    op = _resolve_op_for_registration(op, "register_autograd")
    if isinstance(op, CustomOpDef):
        op.register_autograd(backward, setup_context=setup_context)
        return

    qualname = op
    op = torch._library.utils.lookup_op(qualname)
    schema = op._schema
    if _library.utils.is_out(op):
        raise RuntimeError(
            f"Cannot register autograd formula for operator tagged with "
            f"torch.Tag.out: {op}. Out variants do not support autograd."
        )
    if not _library.utils.is_functional_schema(schema):
        raise RuntimeError(
            f"Cannot register autograd formula for non-functional operator "
            f"{op} with schema {schema}. Please create "
            f"a functional operator and register an autograd formula for that."
        )
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_autograd with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    info = _library.autograd.Info(backward, setup_context)
    autograd_kernel = _library.autograd.make_autograd_impl(op, info)
    namespace, opname = torch._library.utils.parse_namespace(qualname)
    use_lib = _library_for_registration(namespace, lib)
    use_lib.impl(opname, autograd_kernel, "Autograd", with_keyset=True)


def register_torch_dispatch(
    op: _op_identifier,
    torch_dispatch_class: Any,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
):
    r"""Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

    This allows for open registration to specify the behavior between the operator
    and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
    or the operator directly.

    The ``torch_dispatch_class`` is either a Tensor subclass with ``__torch_dispatch__`` or a
    TorchDispatchMode.

    If it is a Tensor subclass, we expect ``func`` to have the following signature:
    ``(cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    If it is a TorchDispatchMode, we expect ``func`` to have the following signature:
    ``(mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    ``args`` and ``kwargs`` will have been normalized the same way they are
    in ``__torch_dispatch__`` (see :ref:`torch-dispatch-calling-convention`).

    Examples:

        >>> import torch
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args={})
        >>> def foo(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.clone()
        >>>
        >>> class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
        >>>     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        >>>         return func(*args, **kwargs)
        >>>
        >>> @torch.library.register_torch_dispatch("mylib::foo", MyMode)
        >>> def _(mode, func, types, args, kwargs):
        >>>     x, = args
        >>>     return x + 1
        >>>
        >>> x = torch.randn(3)
        >>> y = foo(x)
        >>> assert torch.allclose(y, x)
        >>>
        >>> with MyMode():
        >>>     y = foo(x)
        >>> assert torch.allclose(y, x + 1)

    """
    op = _resolve_op_for_registration(op, "register_torch_dispatch")
    if isinstance(op, CustomOpDef):
        return op.register_torch_dispatch(torch_dispatch_class, func)

    def register(func):
        namespace, op_name = torch._library.utils.parse_namespace(op)
        use_lib = _library_for_registration(namespace, lib)
        use_lib._register_torch_dispatch_rule(op_name, torch_dispatch_class, func)
        return func

    if func is None:
        return register
    else:
        return register(func)


def register_vmap(
    op: _op_identifier,
    func: Callable | None = None,
    /,
    *,
    lib=None,
):
    r"""Register a vmap implementation to support :func:`torch.vmap` for this custom op.

    This API may be used as a decorator (see examples).

    In order for an operator to work with :func:`torch.vmap`, you may need to register a
    vmap implementation in the following signature:

        ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

    where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.
    We do not support kwarg-only Tensor args.

    It specifies how do we compute the batched version of ``op`` given inputs with an additional
    dimension (specified by ``in_dims``).

    For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``. It is ``None``
    if the arg is not a Tensor or if the arg is not being vmapped over, otherwise, it is an integer
    specifying what dimension of the Tensor is being vmapped over.

    ``info`` is a collection of additional metadata that may be helpful:
    ``info.batch_size`` specifies the size of the dimension being vmapped over, while
    ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.

    The return of the function ``func`` is a tuple of ``(output, out_dims)``. Similar to ``in_dims``,
    ``out_dims`` should be of the same structure as ``output`` and contain one ``out_dim``
    per output that specifies if the output has the vmapped dimension and what index it is in.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>> from typing import Tuple
        >>>
        >>> def to_numpy(tensor):
        >>>     return tensor.cpu().numpy()
        >>>
        >>> lib = torch.library.Library("mylib", "FRAGMENT")
        >>> @torch.library.custom_op("mylib::numpy_cube", mutates_args=())
        >>> def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
        >>>     x_np = to_numpy(x)
        >>>     dx = torch.tensor(3 * x_np ** 2, device=x.device)
        >>>     return torch.tensor(x_np ** 3, device=x.device), dx
        >>>
        >>> def numpy_cube_vmap(info, in_dims, x):
        >>>     result = numpy_cube(x)
        >>>     return result, (in_dims[0], in_dims[0])
        >>>
        >>> torch.library.register_vmap(numpy_cube, numpy_cube_vmap)
        >>>
        >>> x = torch.randn(3)
        >>> torch.vmap(numpy_cube)(x)
        >>>
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
        >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
        >>>
        >>> @torch.library.register_vmap("mylib::numpy_mul")
        >>> def numpy_mul_vmap(info, in_dims, x, y):
        >>>     x_bdim, y_bdim = in_dims
        >>>     x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
        >>>     y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
        >>>     result = x * y
        >>>     result = result.movedim(-1, 0)
        >>>     return result, 0
        >>>
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.randn(3)
        >>> torch.vmap(numpy_mul)(x, y)

    .. note::
        The vmap function should aim to preserve the semantics of the entire custom operator.
        That is, ``grad(vmap(op))`` should be replaceable with a ``grad(map(op))``.

        If your custom operator has any custom behavior in the backward pass, please
        keep this in mind.

    """
    op = _resolve_op_for_registration(op, "register_vmap")
    if isinstance(op, CustomOpDef):
        return op.register_vmap(func)
    qualname = op
    op_overload = torch._library.utils.lookup_op(qualname)
    schema = op_overload._schema
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_vmap with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    def register(func):
        nonlocal lib

        namespace, opname = torch._library.utils.parse_namespace(qualname)
        use_lib = _library_for_registration(namespace, lib)
        if lib is None:
            lib = use_lib

        from torch._functorch.autograd_function import custom_function_call_vmap_helper
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        def wrapped_func(keyset, *args, **kwargs):
            interpreter = retrieve_current_functorch_interpreter()
            return custom_function_call_vmap_helper(
                # pyrefly: ignore[bad-argument-type]
                interpreter,
                func,
                op_overload,
                *args,
                **kwargs,
            )

        use_lib.impl(opname, wrapped_func, "FuncTorchBatched", with_keyset=True)

    if func is None:
        return register
    else:
        return register(func)


# If the op was defined in C++, then we want to make sure there was an
# m.set_python_module(module, ...) call and that the module is the
# same as the module that called torch.library.register_fake.
def _check_pystubs_once(func, qualname, actual_module_name):
    checked = False

    def inner(*args, **kwargs):
        nonlocal checked
        if checked:
            return func(*args, **kwargs)

        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            checked = True
            return func(*args, **kwargs)

        maybe_pystub = torch._C._dispatch_pystub(
            op._schema.name, op._schema.overload_name
        )
        if maybe_pystub is None:
            if torch._library.utils.requires_set_python_module():
                namespace = op.namespace
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' was defined in C++ and has a Python "
                    f"fake impl. In this situation, we require there to also be a "
                    f'companion C++ `m.set_python_module("{actual_module_name}")` '
                    f"call, but we could not find one. Please add that to "
                    f"to the top of the C++ TORCH_LIBRARY({namespace}, ...) block the "
                    f"operator was registered in ({cpp_filename})"
                )
        else:
            pystub_module = maybe_pystub[0]
            if actual_module_name != pystub_module:
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' specified that its python fake impl "
                    f"is in the Python module '{pystub_module}' but it was actually found "
                    f"in '{actual_module_name}'. Please either move the fake impl "
                    f"or correct the m.set_python_module call ({cpp_filename})"
                )
        checked = True
        return func(*args, **kwargs)

    return inner


# NOTE [ctx inside the fake implementation]
# If a user has an operator with data-dependent output shape, then when writing
# a fake implementation they must query the current ctx and use methods on the
# ctx to construct a new unbacked symint.
#
# This is done via us setting the global_ctx_getter function every time a fake
# implementation is invoked.
def get_ctx() -> "torch._library.fake_impl.FakeImplCtx":
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an fake impl
    (see :func:`torch.library.register_fake` for more usage details.
    """
    return torch._library.fake_impl.global_ctx_getter()


def get_kernel(
    op: _op_identifier, dispatch_key: str | torch.DispatchKey
) -> torch._C._SafeKernelFunction:
    """Returns the computed kernel for a given operator and dispatch key.

    This function retrieves the kernel that would be executed for a given
    operator and dispatch key combination. The returned SafeKernelFunction
    can be used to call the kernel in a boxed fashion. The intended use
    case for this function is to retrieve the original kernel for a given
    dispatch key and then register another kernel to the same dispatch key
    that calls into the original kernel for certain cases.

    Args:
        op: Operator name (along with the overload) or OpOverload object
            Can be a string (e.g., "aten::add.Tensor"), an OpOverload, or a CustomOpDef.
        dispatch_key (str | torch.DispatchKey): The dispatch key to get the kernel for.
            Can be a string (e.g., "CPU", "CUDA") or a DispatchKey enum value.

    Returns:
        torch._C._SafeKernelFunction: A safe kernel function that can be used to
            call the kernel.

    Raises:
        RuntimeError: If the operator does not exist.

    Example:
        >>> # Get the CPU kernel for torch.add
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", "CPU")
        >>>
        >>> # You can also use DispatchKey enum
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", torch.DispatchKey.CPU)
        >>>
        >>> # Or use an OpOverload directly
        >>> kernel = torch.library.get_kernel(torch.ops.aten.add.Tensor, "CPU")
        >>>
        >>> # Example: Using get_kernel in a custom op with conditional dispatch
        >>> # Get the original kernel for torch.sin
        >>> original_sin_kernel = torch.library.get_kernel("aten::sin", "CPU")
        >>>
        >>> # If input has negative values, use original sin, otherwise return zeros
        >>> def conditional_sin_impl(dispatch_keys, x):
        >>>     if (x < 0).any():
        >>>         return original_sin_kernel.call_boxed(dispatch_keys, x)
        >>>     else:
        >>>         return torch.zeros_like(x)
        >>>
        >>> lib = torch.library.Library("aten", "IMPL")
        >>> # with_keyset=True so the first argument to the impl is the current DispatchKeySet
        >>> which needs to be the first argument to ``kernel.call_boxed``
        >>> lib.impl("sin", conditional_sin_impl, "CPU", with_keyset=True)
        >>>
        >>> # Test the conditional behavior
        >>> x_positive = torch.tensor([1.0, 2.0])
        >>> x_mixed = torch.tensor([-1.0, 2.0])
        >>> torch.sin(x_positive)
        tensor([0., 0.])
        >>> torch.sin(x_mixed)
        tensor([-0.8415, 0.9093])
    """
    if not isinstance(op, (str, torch._ops.OpOverload)):
        raise ValueError(f"get_kernel({op}): got unexpected type for op: {type(op)}")

    if isinstance(op, torch._ops.OpOverload):
        op = op._name

    if isinstance(dispatch_key, str):
        try:
            dispatch_key = torch._C.DispatchKey.__members__[dispatch_key]
        except KeyError:
            raise ValueError(f"Invalid dispatch key: {dispatch_key}") from None

    return torch._C._dispatch_get_computed_kernel_for_dispatch_key(op, dispatch_key)


_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def opcheck(
    op: torch._ops.OpOverload | torch._ops.OpOverloadPacket | CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,
    atol=None,
    rtol=None,
) -> dict[str, str]:
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    That is, when you use the torch.library/TORCH_LIBRARY APIs to create a
    custom op, you specified metadata (e.g. mutability info) about the custom op
    and these APIs require that the functions you pass them satisfy certain
    properties (e.g. no data pointer access in the fake/meta/abstract kernel)
    ``opcheck`` tests these metadata and properties.

    Concretely, we test the following:

    - test_schema: If the schema matches the implementation of
      the operator. For example: if the schema specifies a Tensor is mutated,
      then we check the implementation mutates the Tensor. If the schema
      specifies that we return a new Tensor, then we check that the
      implementation returns a new Tensor (instead of an existing one or
      a view of an existing one).
    - test_autograd_registration: If the operator supports training
      (autograd): we check that its autograd formula is registered via
      torch.library.register_autograd or a manual registration to one
      or more DispatchKey::Autograd keys. Any other DispatchKey-based
      registrations may lead to undefined behavior.
    - test_faketensor: If the operator has a FakeTensor kernel
      (and if it is correct). The FakeTensor kernel is necessary (
      but not sufficient) for the operator to work with PyTorch compilation
      APIs (torch.compile/export/FX). We check that a FakeTensor kernel
      (also sometimes known as a meta kernel) was registered for the
      operator and that it is correct. This test takes the result of
      running the operator on real tensors and the result of running
      the operator on FakeTensors and checks that they have the same
      Tensor metadata (sizes/strides/dtype/device/etc).
    - test_aot_dispatch_dynamic: If the operator has correct behavior
      with PyTorch compilation APIs (torch.compile/export/FX).
      This checks that the outputs (and gradients, if applicable) are the
      same under eager-mode PyTorch and torch.compile.
      This test is a superset of ``test_faketensor`` and is an e2e test;
      other things it tests are that the operator supports
      functionalization and that the backward pass (if it exists) also
      supports FakeTensor and functionalization.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. If your operator supports
    autograd, please use ``opcheck`` with inputs with ``requires_grad = True``;
    if your operator supports multiple devices (e.g. CPU and CUDA), please
    use ``opcheck`` with inputs on all supported devices.

    Args:
        op: The operator. Must either be a function decorated with
            :func:`torch.library.custom_op` or an OpOverload/OpOverloadPacket
            found in torch.ops.* (e.g. torch.ops.aten.sin, torch.ops.mylib.foo)
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ("test_schema", "test_faketensor")
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.
        rtol (Optional[float]): Relative tolerance for floating point comparisons.
            If specified ``atol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).
        atol (Optional[float]): Absolute tolerance for floating point comparisons.
            If specified ``rtol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).

    .. warning::

        opcheck and :func:`torch.autograd.gradcheck` test different things;
        opcheck tests if your usage of torch.library APIs is correct while
        :func:`torch.autograd.gradcheck` tests if your autograd formula is
        mathematically correct. Use both to test custom ops that support
        gradient computation.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: float) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     z_np = x_np * y
        >>>     return torch.from_numpy(z_np).to(x.device)
        >>>
        >>> @numpy_mul.register_fake
        >>> def _(x, y):
        >>>     return torch.empty_like(x)
        >>>
        >>> def setup_context(ctx, inputs, output):
        >>>     y, = inputs
        >>>     ctx.y = y
        >>>
        >>> def backward(ctx, grad):
        >>>     return grad * ctx.y, None
        >>>
        >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
        >>>
        >>> sample_inputs = [
        >>>     (torch.randn(3), 3.14),
        >>>     (torch.randn(2, 3, device='cuda'), 2.718),
        >>>     (torch.randn(1, 10, requires_grad=True), 1.234),
        >>>     (torch.randn(64, 64, device='cuda', requires_grad=True), 90.18),
        >>> ]
        >>>
        >>> for args in sample_inputs:
        >>>     torch.library.opcheck(numpy_mul, args)

    """
    import torch.testing._internal.optests as optests

    return optests.opcheck(
        op,
        args,
        kwargs,
        test_utils=test_utils,
        raise_exception=raise_exception,
        rtol=rtol,
        atol=atol,
    )
<|endoftext|>_overwrite_module_params_on_conversion: bool = False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will assign new parameters to the module:

    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype
    #. :meth:`nn.Module.to`
    #. :meth:`nn.Module.to_empty`

    Args:
        value (bool): Whether to assign new tensors or not.

    """
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """
    Returns whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an :class:`torch.nn.Module`. Defaults to ``False``.

    See :func:`~torch.__future__.set_overwrite_module_params_on_conversion` for more information.
    """
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module`` and instead
    of ``param.copy_(state_dict[key])`` when loading a state dict into an ``nn.Module``.

    .. note::
        This function takes precedence over :func:`~torch.__future__.get_overwrite_module_params_on_conversion`

    When enabled, the following methods will swap the existing parameters in-place:

    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype
    #. :meth:`nn.Module.to`
    #. :meth:`nn.Module.to_empty`
    #. :meth:`nn.Module.load_state_dict`

    The semantics for :meth:`~nn.Module.load_state_dict` when this is set are as follows:

    #. For each parameter/buffer, its corresponding ``state_dict['key']`` is transformed via
       :meth:`~torch.Tensor.module_load` (i.e. ``res = param.module_load(state_dict['key'])``)
    #. If necessary, ``res`` will be wrapped in an :class:`~nn.Parameter`
    #. The parameter/buffer in the module will be swapped via :func:`~torch.utils.swap_tensors`
       with ``res``

    Args:
        value (bool): Whether to use :func:`~torch.utils.swap_tensors` or not.

    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    Returns whether to use :func:`~torch.utils.swap_tensors` instead of setting .data to
    change the existing parameters in-place when converting an ``nn.Module``. Defaults to ``False``.

    See :func:`~torch.__future__.set_swap_module_params_on_conversion` for more information.
    """
    return _swap_module_params_on_conversion
<|endoftext|># In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Sequence
from typing import Any, IO, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import Self

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from torch import (  # noqa: F401
    device as _device,
    DispatchKey,
    dtype as _dtype,
    layout as _layout,
    qscheme as _qscheme,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge


__all__ = ["Number", "Device", "FileLike", "Storage"]

# Convenience aliases for common composite types that we need
# to talk about in PyTorch
_TensorOrTensors: TypeAlias = Tensor | Sequence[Tensor]  # noqa: PYI047
_TensorOrOptionalTensors: TypeAlias = Tensor | Sequence[Tensor | None]  # noqa: PYI047
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size: TypeAlias = Size | list[int] | tuple[int, ...]  # noqa: PYI042,PYI047
_symsize: TypeAlias = Size | Sequence[int | SymInt]  # noqa: PYI042,PYI047
_dispatchkey: TypeAlias = str | DispatchKey  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = int | SymInt
# float or SymFloat
FloatLikeType: TypeAlias = float | SymFloat
# bool or SymBool
BoolLikeType: TypeAlias = bool | SymBool

py_sym_types = (SymInt, SymFloat, SymBool)  # left un-annotated intentionally
PySymType: TypeAlias = SymInt | SymFloat | SymBool

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = int | float | bool
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

FileLike: TypeAlias = str | os.PathLike[str] | IO[bytes]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = _device | str | int | None


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> Self:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> Self:
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self:
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        raise NotImplementedError
<|endoftext|>"""Implement various linear algebra algorithms for low rank matrices."""

__all__ = ["svd_lowrank", "pca_lowrank"]


import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function


def get_approximate_basis(
    A: Tensor,
    q: int,
    niter: int | None = 2,
    M: Tensor | None = None,
) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`. without instantiating any tensors
    of the size of :math:`A` or :math:`M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              chosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
    matmul = _utils.matmul

    R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable

    X = matmul(A, R)
    if M is not None:
        X = X - matmul(M, R)
    Q = torch.linalg.qr(X).Q
    for _ in range(niter):
        X = matmul(A.mH, Q)
        if M is not None:
            X = X - matmul(M.mH, Q)
        Q = torch.linalg.qr(X).Q
        X = matmul(A, Q)
        if M is not None:
            X = X - matmul(M, Q)
        Q = torch.linalg.qr(X).Q
    return Q


def svd_lowrank(
    A: Tensor,
    q: int | None = 6,
    niter: int | None = 2,
    M: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              chosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: This is a randomized method. To obtain repeatable results,
              set the seed for the pseudorandom number generator

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10x
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`, which will be broadcasted
                              to the size of A in this function.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(
                svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M
            )
    return _svd_lowrank(A, q=q, niter=niter, M=M)


def _svd_lowrank(
    A: Tensor,
    q: int | None = 6,
    niter: int | None = 2,
    M: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    # Algorithm 5.1 in Halko et al., 2009

    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is not None:
        M = M.broadcast_to(A.size())

    # Assume that A is tall
    if m < n:
        A = A.mH
        if M is not None:
            M = M.mH

    Q = get_approximate_basis(A, q, niter=niter, M=M)
    B = matmul(Q.mH, A)
    if M is not None:
        B = B - matmul(Q.mH, M)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V = Vh.mH
    U = Q.matmul(U)

    if m < n:
        U, V = V, U

    return U, S, V


def pca_lowrank(
    A: Tensor,
    q: int | None = None,
    center: bool = True,
    niter: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Performs linear Principal Component Analysis (PCA) on a low-rank
    matrix, batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :math:`A` such that :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`

    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:

                - :math:`A` is a data matrix with ``m`` samples and
                  ``n`` features

                - the :math:`V` columns represent the principal directions

                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of
                  :math:`A^T A / (m - 1)` which is the covariance of
                  ``A`` when ``center=True`` is provided.

                - ``matmul(A, V[:, :k])`` projects data to the first k
                  principal components

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:

                - :math:`U` is m x q matrix

                - :math:`S` is q-vector

                - :math:`V` is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of
                           :math:`A`. By default, ``q = min(6, m,
                           n)``.

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2.

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """

    if not torch.jit.is_scripting():
        if type(A) is not torch.Tensor and has_torch_function((A,)):
            return handle_torch_function(
                pca_lowrank, (A,), A, q=q, center=center, niter=niter
            )

    (m, n) = A.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError(
            f"q(={q}) must be non-negative integer and not greater than min(m, n)={min(m, n)}"
        )
    if not (niter >= 0):
        raise ValueError(f"niter(={niter}) must be non-negative integer")

    dtype = _utils.get_floating_dtype(A)

    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)

    if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
        # reshape c
        column_indices = c.indices()[0]
        indices = torch.zeros(
            2,
            len(column_indices),
            dtype=column_indices.dtype,
            device=column_indices.device,
        )
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(
            indices, c.values(), (n, 1), dtype=dtype, device=A.device
        )

        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = torch.sparse.mm(C_t, ones_m1_t).mT
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        C = A.mean(dim=(-2,), keepdim=True)
        return _svd_lowrank(A - C, q, niter=niter, M=None)
<|endoftext|>"""Ahead-of-time precompilation (``make_fx`` tracer by default; Dynamo planned).

    python_code, cache = torch.compiler.precompile(fn, model, *example_inputs)
    f_c = torch.compiler.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

precompile captures your computation with ``make_fx`` -- a NON-STRICT trace of the ATen
ops that run when ``fn`` executes once on the example inputs. It does not analyze your
Python, so it comes with an explicit contract (the programming model): stay inside it
and the artifact faithfully reproduces ``fn``; step outside it and you get an artifact
that computes the wrong thing.

``precompile`` returns a self-contained, executable ``python_code`` string plus a
companion integrity-tagged ``cache``. With ``backend="inductor"`` (the default) the
captured graph is lowered through the AOT backend contract
(``torch._functorch.aot_autograd.compile_to_python``, AOTAutograd + Inductor);
``python_code`` JIT-compiles kernels on first call and the cache primes them so a warm
reload skips JIT. With ``backend="eager"`` ``python_code`` inlines the captured graph and
runs on its own. Reload with ``torch.compiler.precompile.load(python_code, cache)``.

The full contract, the calling convention, and the cache / code_hash design all live in
Note [precompile programming model] below; every public entry point and guard references
it.
"""

# Note [precompile programming model]
#
# ``fn`` is the WHOLE computation, e.g. ``lambda model, x: model(x)`` for inference
# or ``lambda model, x, t: loss_fn(model(x), t).backward()`` for a training step.
# Among the positional args, the nn.Module arguments have their parameters and
# buffers lifted to explicit graph inputs (via functional reparametrization), so
# nothing live is baked in; the remaining args are the runtime inputs. The artifact
# embeds NO weights -- you pass the model again at runtime.
#
# Because make_fx is a non-strict trace, precompile offers a contract, not a
# guarantee against misuse. The caller MUST uphold the invariants below. The ones
# that are cheaply knowable from the captured graph are ENFORCED (a violation
# raises PrecompileError); the rest are the caller's responsibility and, if broken,
# produce a SILENTLY INCORRECT artifact -- the ordinary consequence of tracing.
#
# 1. Everything live is an input. Every tensor the computation reads must be passed to
#    fn as an explicit tensor argument -- EXCEPT tensors held inside an nn.Module
#    argument, which precompile handles for you. For an nn.Module argument you do NOT
#    enumerate its tensors yourself: precompile lifts every registered parameter and
#    buffer (recursively, including submodules, tied weights collapsed by identity) to
#    explicit graph inputs for you via functional reparametrization, and re-derives the
#    same list from the runtime model you pass to load(). Passing the module is enough --
#    that is the whole point of accepting modules as arguments. What is NOT lifted is
#    anything not reachable
#    through that protocol: tensors closed over by ``fn`` (globals, captured locals)
#    and plain (non-registered) module attributes -- a bare ``self.weight = t`` rather
#    than a registered parameter/buffer. Those are not inputs; a vanilla make_fx trace
#    would bake them in as get_attr constants. Fix by registering them on the module
#    (register_parameter / register_buffer) or passing them as explicit tensor args.
#    ENFORCED: _check_no_constant_tensors rejects any baked tensor constant.
#
# 2. The runtime model must match the traced model structurally. At load time you
#    pass the model again; precompile re-derives the parameter/buffer list from the
#    runtime model in the SAME order (parameters then buffers, interned by tensor
#    identity so tied weights collapse to a single input). The runtime model must
#    have the same named_parameters()/named_buffers() ordering and count and the
#    same weight tying as the example model. Same architecture with different
#    weights is the intended use (swap in a checkpoint); a structurally different
#    model is undefined. requires_grad is ALSO part of the structural contract: which
#    params get a scattered grad is fixed at capture time from the example model's
#    requires_grad (invariant 5), so flipping a param's requires_grad at runtime does
#    not change what the artifact computes. ENFORCED: the driver compares the runtime
#    model's full param/buffer NAME list (order and identity, tied weights collapsed)
#    against the traced list, AND each runtime param/buffer's SHAPE, DTYPE, AND DEVICE
#    against the baked example values, so a reordered or otherwise structurally-different
#    model -- even one with the same count and names but a differently shaped, typed, or
#    placed weight (e.g. a Linear(4,4) swapped for a Linear(4,8), or a CPU weight where a
#    CUDA one was traced) -- is rejected (it cannot silently scatter grads onto the wrong
#    slot, fail deep in a kernel, or compute the wrong thing). Different WEIGHT VALUES with
#    the same shapes/dtypes/devices are the intended use -- WITH ONE INDUCTOR-BACKEND
#    CAVEAT: the inductor backend ALSO specializes each param/buffer's LAYOUT (memory
#    format), since it bakes assert_size_stride on every weight the graph reads. So a
#    same-shape/same-dtype checkpoint whose weight has a DIFFERENT layout (e.g. a
#    non-contiguous view, or a channels_last weight where the example was contiguous) is
#    REJECTED at runtime by the inductor backend (invariant 6). Match the example weight's
#    layout (.contiguous() to match a contiguous example), or use backend='eager' for
#    layout-flexible weights.
#
# 3. Control flow (and, by default, shapes) is specialized to the example. A non-strict
#    trace follows the single path taken for the example inputs: Python ``if``/``for``
#    over tensor values, ``.item()``, and shape-dependent branching are resolved at
#    trace time and baked. Shapes are static BY DEFAULT (capture uses make_fx in its
#    "real" mode, so each size is baked as a constant). You can opt specific user-input
#    dims into being dynamic by marking them with
#    ``torch._dynamo.decorators.mark_unbacked`` before calling: those dims are
#    captured as UNBACKED symints (symbolic capture), which CANNOT be guarded on -- so
#    the artifact is valid for any runtime size of those dims, and a graph that needs to
#    guard on / specialize a marked dim fails LOUDLY at capture (PrecompileError) instead
#    of baking a silently-wrong result.
#
# 4. Boundary effects. Input mutation (including module buffers -- e.g. BatchNorm
#    running stats in training mode), tensor-subclass wrap/unwrap (e.g. DTensor),
#    outputs that alias inputs, and functionalized RNG are SUPPORTED: the inductor
#    backend lowers through torch._functorch.aot_autograd.compile_to_python, which
#    composes AOTAutograd's own codegen'd prelude/epilogue into the artifact (the
#    effect is reflected onto the runtime model / inputs). Effectful ops are not
#    supported yet and raise at capture time (_assert_supported) with a concrete
#    reason; this is an implementation gap, not a fundamental limit. Every other
#    runtime wrapper that can appear in a composable (cacheable) forward graph is
#    codegen'd as source and composed in; the one non-codegen'd wrapper
#    (FakifiedOutWrapper) only activates under fakify_first_call, which makes the graph
#    non-cacheable, so such a graph is rejected before composition ever runs.
#    Distributed capture: a ``compile_on_one_rank`` flag (trace on a single rank and
#    broadcast the artifact to the rest, so every rank need not re-capture) is
#    anticipated and scheduled for a follow-up later in this stack.
#
# 5. Backward is part of the computation. Yes: if you trace ``forward -> loss ->
#    backward``, running the artifact re-runs that whole computation and puts the
#    resulting parameter gradients onto the runtime model. Concretely: the parameter
#    gradients are harvested inside the (functional) graph as extra outputs, and the
#    driver scatters them back onto the runtime model's ``parameters()`` ``.grad``
#    fields -- ACCUMULATING (``p.grad += g``), not overwriting, exactly like eager
#    ``.backward()``, so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged
#    (skip the zero and grads pile up, by design). WHICH params get a grad is fixed at
#    TRACE time, not runtime: only params that actually received a gradient during the
#    traced backward are harvested (recorded by index in GRAD_PARAM_INDICES); a frozen
#    (``requires_grad=False``) or non-contributing param keeps ``.grad = None``, exactly
#    as eager leaves it -- precompile does NOT zero-fill such params, and flipping a
#    param's requires_grad at runtime does not change what gets scattered (invariant 2).
#    Buffers are never harvested (a requires_grad buffer that got a grad is rejected at
#    capture). The artifact therefore returns ``fn``'s own result (``None`` for a bare
#    ``.backward()`` step), not the grads. The grad scatter is the ONLY mutation
#    precompile performs, and it happens in Python outside the graph, so the graph stays
#    functional. precompile does not own optimizer state; bring your own optimizer and
#    zero grads as usual.
#
# 6. Shapes are static by default (dynamic dims are opt-in via mark_unbacked, invariant
#    3), each input's dtype/device is baked, and the inductor backend also specializes
#    on input layout. Each dense user-input leaf's dtype and device are recorded at
#    capture and checked at runtime (both backends): a dtype- or device-mismatched input
#    is rejected with a PrecompileError rather than crashing deep in a kernel or reading
#    a wrong value. The graph is specialized to the example input shapes (invariant 3);
#    tensor-subclass outputs in particular are rebuilt with constant outer sizes/strides,
#    so a different runtime shape is undefined. The inductor backend ADDITIONALLY bakes
#    each read input's stride / memory format (it emits assert_size_stride) -- and this
#    applies to model PARAMETERS/BUFFERS too, not only user inputs, since they are graph
#    inputs the kernels read. So a same-shape runtime input OR a same-shape/same-dtype
#    checkpoint WEIGHT with a DIFFERENT layout (e.g. a contiguous tensor when the example
#    was transposed or channels_last, or a non-contiguous view of a weight) is rejected
#    with a clear PrecompileError; match the example layout or use backend='eager'.
#    This guard is deliberately CONSERVATIVE: a layout-agnostic kernel (e.g. matmul) may
#    well have computed the right answer on the new layout, but precompile cannot
#    recompile to specialize it the way torch.compile does, so it rejects to stay safe
#    rather than risk a silently-wrong result from a layout-sensitive kernel. Pass inputs
#    in the example's layout (``.contiguous()`` to match a contiguous example), or use the
#    layout-flexible eager backend. ENFORCED for read inputs (a layout mismatch raises
#    rather than crashing in assert_size_stride or reading wrong strides).
#
# 7. Both python_code and the cache are trusted, EXECUTABLE input to load(). The cache
#    outer envelope is a plain {"artifact": bytes, ...} dict (read with
#    weights_only=True) carrying a format/version + backend tag AND a code_hash
#    (sha256 of the python_code it accelerates) that load() verifies (raising
#    PrecompileError on mismatch). load() feeds those bytes to
#    torch.compiler.load_cache_artifacts to PRIME the inductor kernel caches, then always
#    EXECs python_code -- with the caches primed the kernels load from the precompiled
#    binaries instead of JIT-compiling. Both the cache priming (it unpickles) and the exec run
#    code you supplied; treat both python_code and the cache like code you are about to
#    run. The code_hash binds the cache to its python_code:
#    load() rejects a (code, cache) pair from different precompile() calls (same
#    backend) rather than silently running the cache's graph under foreign metadata.
#
# self-contained: ``python_code`` runs on its own -- it inlines the composed graph
# module (inductor: kernels JIT-compiled on first call, plus AOTAutograd's codegen'd
# prelude/epilogue) or the captured graph (eager), plus all calling-convention
# metadata. It NEVER reads the cache, and it is the SINGLE SOURCE OF TRUTH for the
# calling convention. The ``cache`` holds ONLY the compiled INDUCTOR artifact and is
# purely an ACCELERATION consumed only by load(): load AST-scrapes the module-level
# calling convention out of python_code, primes the inductor kernel caches from the bundle
# (torch.compiler.load_cache_artifacts), then execs python_code -- so its kernels load
# from the precompiled binaries instead of JIT. With the cache you skip JIT; with only
# python_code you JIT -- same results either way. The
# eager backend has no kernels to accelerate, so the eager cache carries no compiled
# artifact (artifact=None) but is still a full integrity-tagged envelope, and load()
# always runs the graph inlined in python_code. The metadata
# lives in one place (python_code); the envelope carries a code_hash (sha256 of
# python_code) alongside the format/version + backend tag, so load() rejects a
# (python_code, cache) pair that did not come from the same precompile() call.
#
# backend: "inductor" (default) lowers the captured graph through
# torch._functorch.aot_autograd.compile_to_python (AOTAutograd + Inductor, emitting a
# self-contained module). "eager" skips lowering and runs the captured
# ATen graph as-is (analogous to torch.compile(backend="eager")), for inspecting or
# debugging exactly what was traced. The contract above is identical for both
# backends with ONE exception (invariant 6): the inductor backend additionally
# specializes on each input's stride / memory format, while the eager backend is
# layout-flexible. Otherwise the same graph is captured; only its realization differs.
# Two mechanical consequences: the eager backend runs the graph directly on the
# (subclass-level) inputs, so it does not exercise the dense subclass
# flatten/unflatten path that the inductor backend's calling convention requires;
# and because there are no kernels, the eager cache carries no compiled artifact
# (artifact=None) but is still a full integrity-tagged envelope (python_code is the
# whole runnable artifact).
#
# tracer: the capture front-end, orthogonal to backend. "make_fx" (default) is a
# non-strict trace and is the only tracer implemented today -- everything above (the
# invariants, the contract) describes its behavior. "dynamo" is planned (a Dynamo-based
# front-end that analyzes Python rather than specializing to one traced path) and
# currently raises NotImplementedError.

from __future__ import annotations

import hashlib
import io
import logging
from types import MappingProxyType
from typing import Any, cast, NewType, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._subclasses.fake_tensor import FakeTensorMode


# ``precompile`` and ``PrecompileError`` are exposed under the compiler namespace as
# ``torch.compiler.precompile`` / ``torch.compiler.precompile.PrecompileError``
# (re-exported from torch/compiler/__init__.py and registered in
# ``torch.compiler.__all__``); they are deliberately kept out of this private module's
# ``__all__`` so test_public_bindings sees a consistent single public location.
__all__: list[str] = []


# Integrity tag baked into the cache envelope and verified by load() (with the
# code_hash) to reject a foreign / mismatched cache; see Note [precompile programming
# model], invariant 7.
_CACHE_FORMAT = "torch.compiler.precompile"
_CACHE_VERSION = 1


# Index into the caller's positional nn.Module arguments (0-based over the modules,
# not over all args), used to qualify tied-across-modules param/buffer names as m<i>.<n>.
_ModuleIndex = NewType("_ModuleIndex", int)


# Decoded mark_unbacked spec for one dim: (shape_id, min, max, hint_override). shape_id is
# an opaque hashable grouping label (dims sharing it collapse to one unbacked symbol); the
# other three are optional integer sizes and are None wherever the decorator left them unset.
_MarkSpec = tuple[object, int | None, int | None, int | None]
# Per-user-input-leaf runtime bounds harvested from the marks: {dim: (min, max)} (either
# may be None), or None when the leaf has no bounded marked dim.
_LeafBounds = dict[int, tuple[int | None, int | None]] | None


# Reused read-only empty mapping for the mark_unbacked getattr fallbacks (Note [precompile
# reads private dynamo mark attributes]), so the common unmarked leaf reads its (absent)
# _dynamo_* dicts without allocating a throwaway {} per call.
# The value type is Any because these back three DISTINCT private dynamo dicts (dim ->
# shape_id label / (min, max) tuple / hint int); one shared empty default cannot name all
# three, and the private _dynamo_* attrs are untyped (Note above), so Any is the isolated
# boundary here.
_NO_MARKS: Mapping[int, Any] = MappingProxyType({})


class PrecompileError(RuntimeError):
    """The error type raised by ``torch.compiler.precompile`` and its artifacts.

    Raised when capture, lowering, ``load``, or a runtime call violates the precompile
    contract -- e.g. a tensor baked as a constant (invariant 1), an unsupported /
    effectful op, a non-tensor output the inductor backend cannot lower, or a runtime
    input whose shape or memory format differs from the example (invariants 3 and 6).
    See Note [precompile programming model] in this module for the full contract.
    """


def _dense_shape(t: object) -> tuple[int, ...] | None:
    """Return the shape of a plain dense tensor, else ``None`` (non-tensor / subclass).

    Tensor subclasses (e.g. DTensor) go through AOTAutograd's flatten path, so their
    outer shape is not the dense shape the inductor artifact bakes; record ``None`` and
    skip them in the shape check.
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return tuple(t.shape)
    return None


def _dense_dtype(t: object) -> str | None:
    """Return the dtype of a plain dense tensor as a string, else ``None``.

    Recorded as a string (e.g. ``"torch.float32"``) so it serializes into the artifact
    metadata as a literal and compares cleanly against ``str(t.dtype)`` at runtime;
    mirrors the _dense_shape convention (None for non-tensor / subclass leaves). The
    graph is specialized to the example dtype (invariant 6).
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return str(t.dtype)
    return None


def _dense_device(t: object) -> str | None:
    """Return the device (as a string) of a plain dense tensor, else ``None``.

    Recorded as a string so it serializes into the artifact metadata as a literal and
    compares cleanly at runtime; mirrors _dense_shape (None for non-tensor / subclass
    leaves). The graph is specialized to the example device (invariant 6).
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return str(t.device)
    return None


def _resolved_get_attrs(
    gm: torch.fx.GraphModule,
) -> list[tuple[str, object]]:
    """Return ``(target, attr)`` for every ``get_attr`` node, resolving dotted
    qualnames the same way for both capture guards below (missing attr -> None)."""
    resolved = []
    for node in gm.graph.find_nodes(op="get_attr"):
        attr: object = gm
        for part in node.target.split("."):
            attr = getattr(attr, part, None)
        resolved.append((node.target, attr))
    return resolved


# Note [precompile reads private dynamo mark attributes]
#
# The functions below read PRIVATE per-tensor attributes that
# torch._dynamo.decorators.mark_unbacked stamps onto a tensor: it consumes
# _dynamo_unbacked_indices / _dynamo_strict_unbacked_indices / _dynamo_shape_ids /
# _dynamo_unbacked_bounds / _dynamo_hint_overrides, and rejects _dynamo_dynamic_indices
# / _specialize_on (marks it cannot honor). This is a deliberate coupling to a private
# dynamo contract -- mark_unbacked is the documented entry point, and precompile reads
# what it leaves behind rather than exposing its own dynamic-shape kwarg. A stable
# dynamo-owned accessor is the eventual home; until then these names are load-bearing.
def _has_unbacked_marks(args: tuple[object, ...]) -> bool:
    """True if any tensor reachable in ``args`` carries a mark_unbacked dim (backed or
    strict)."""
    return any(
        isinstance(t, torch.Tensor)
        and (
            getattr(t, "_dynamo_unbacked_indices", None)
            or getattr(t, "_dynamo_strict_unbacked_indices", None)
        )
        for t in pytree.tree_leaves(args)
    )


def _reject_unsupported_marks(user_flat: list[object]) -> None:
    """Reject mark options precompile cannot honor, loudly (invariant 3).

    precompile only honors mark_unbacked (backed unbacked dims) and mark_unbacked's
    strict variant. Backed dynamic marks (mark_dynamic -> _dynamo_dynamic_indices) and
    per-dim specialization (_specialize_on) have no analogue in the static/unbacked
    capture path -- silently dropping them would bake a wrong artifact, so reject rather
    than ignore. (mark_unbacked's hint_override is NOT rejected: it is a perf-only
    autotuning size hint, never a guard, so the single artifact is valid regardless; it
    is threaded into the capture ShapeEnv in _fakeify_with_unbacked.) A mark_unbacked dim
    on a tensor SUBCLASS (e.g. DTensor) is rejected: the dynamic capture cannot preserve
    the subclass through the refake, so it too would bake a wrong artifact.
    """
    for t in user_flat:
        if not isinstance(t, torch.Tensor):
            continue
        # mark_unbacked on a tensor subclass (e.g. DTensor) stamps its marks on the OUTER
        # subclass as well as the inner tensor, so precompile's dynamic path picks it up --
        # but _fakeify_with_unbacked refakes a marked leaf via torch.empty, which yields a
        # plain dense tensor and DROPS the subclass, so the trace would run on the wrong
        # type. Reject loudly here rather than silently capturing a subclass-stripped tensor
        # (mirrors the decorator itself, which raises for every non-DTensor subclass).
        if is_traceable_wrapper_subclass(t) and (
            getattr(t, "_dynamo_unbacked_indices", None)
            or getattr(t, "_dynamo_strict_unbacked_indices", None)
        ):
            raise PrecompileError(
                "precompile: an input is a tensor subclass (e.g. DTensor) with a "
                "mark_unbacked dynamic dim, which precompile cannot honor: the dynamic "
                "capture cannot preserve the subclass. Mark a dense input instead, or "
                "capture that dim static (do not mark_unbacked it)."
            )
        if getattr(t, "_dynamo_dynamic_indices", None):
            raise PrecompileError(
                "precompile: an input has a mark_dynamic (backed dynamic) dim, which "
                "precompile cannot honor; it supports only mark_unbacked dynamic dims. "
                "Use torch._dynamo.decorators.mark_unbacked, or leave the dim static."
            )
        specialize_on = getattr(t, "_specialize_on", None)
        if specialize_on and any(v for v in specialize_on.values()):
            raise PrecompileError(
                "precompile: an input has a mark_unbacked specialize_on list, which "
                "precompile cannot honor (it produces a single artifact, not per-value "
                "specializations). Remove specialize_on."
            )


def _read_unbacked_marks(user_flat: list[object]) -> list[dict[int, _MarkSpec]]:
    """Read ``torch._dynamo.decorators.mark_unbacked`` marks off the user-input tensors.

    Dynamic shapes are opt-in via that decorator (the caller marks dims before calling
    precompile), NOT via a precompile kwarg -- so the precompile signature stays simple.
    Returns a per-leaf list aligned to ``user_flat``; each entry maps a marked dim to
    ``(shape_id, min, max, hint_override)`` (None when unset), empty when the leaf has no
    marks. Dims sharing a ``shape_id`` get the SAME unbacked symbol (so they are equal by
    construction); ``min``/``max`` become runtime range asserts; ``hint_override`` is a
    perf-only autotuning size hint applied to the symbol in _fakeify_with_unbacked.
    """
    marks: list[dict[int, _MarkSpec]] = []
    for t in user_flat:
        if not isinstance(t, torch.Tensor):
            marks.append({})
            continue
        # Union the non-strict and strict unbacked index sets. mark_unbacked(strict=True)
        # records ONLY _dynamo_strict_unbacked_indices; precompile already enforces
        # strict's error-on-specialize semantics via the GuardOnDataDependentSymNode ->
        # PrecompileError path, so both are honored identically here. NOTE: the decorator's
        # strict branch returns early, so a strict dim carries no shape_id/min/max/
        # hint_override (those are dropped at mark time) -- combine strict with shape_id/
        # min/max only if that limitation is acceptable; use non-strict to get them.
        idx = set(getattr(t, "_dynamo_unbacked_indices", None) or ())
        idx |= set(getattr(t, "_dynamo_strict_unbacked_indices", None) or ())
        if not idx:
            marks.append({})
            continue
        shape_ids = getattr(t, "_dynamo_shape_ids", _NO_MARKS) or _NO_MARKS
        bounds = getattr(t, "_dynamo_unbacked_bounds", _NO_MARKS) or _NO_MARKS
        hints = getattr(t, "_dynamo_hint_overrides", _NO_MARKS) or _NO_MARKS
        marks.append(
            {
                d: (shape_ids.get(d), *bounds.get(d, (None, None)), hints.get(d))
                for d in idx
            }
        )
    return marks


def _read_input_bounds(marks: list[dict[int, _MarkSpec]]) -> list[_LeafBounds]:
    """Build the per-leaf runtime min/max bounds from the already-read mark_unbacked
    marks, aligned to ``user_flat`` (so ``marks`` is the output of _read_unbacked_marks).

    mark_unbacked promises (in its own docstring) a runtime check that the dim is >= min
    and <= max; those bounds are applied as capture-time torch._check constraints in
    _fakeify_with_unbacked, but unbacked symints cannot be guarded on, so they never
    become a runtime guard on their own. We record them here so the driver enforces them.
    Each entry is None when the leaf has no bounded marked dim, else a dict mapping a
    marked dim index to ``(lo, hi)`` (either may be None); mirrors USER_INPUT_DTYPES.
    """
    bounds: list[_LeafBounds] = []
    for per in marks:
        per_leaf: dict[int, tuple[int | None, int | None]] = {}
        for d, (_shape_id, lo, hi, _hint) in per.items():
            if lo is not None or hi is not None:
                per_leaf[d] = (lo, hi)
        bounds.append(per_leaf or None)
    return bounds


def _detect_memory_format(t: torch.Tensor) -> torch.memory_format:
    """Return the example leaf's memory format so a refaked marked input preserves it.

    A mark_unbacked dim refakes the leaf via torch.empty; defaulting to contiguous would
    bake a contiguous assert_size_stride and reject a channels_last / transposed input
    even at its own layout. Probe the recognized formats and raise on an exotic /
    ambiguous layout we cannot capture rather than silently forcing contiguous.
    """
    if t.is_contiguous(memory_format=torch.contiguous_format):
        return torch.contiguous_format
    if t.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if t.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    raise PrecompileError(
        "precompile: a mark_unbacked input has a memory format that is neither "
        "contiguous, channels_last, nor channels_last_3d (e.g. a transposed or "
        "otherwise non-standard layout); the dynamic-shape capture cannot preserve it. "
        "Pass the input in one of those layouts (.contiguous() to make it contiguous), "
        "or capture the dim static (do not mark_unbacked it)."
    )


def _fakeify_with_unbacked(
    pb_flat: list[Tensor], user_flat: list[object], marks: list[dict[int, _MarkSpec]]
) -> tuple[list[object], FakeTensorMode]:
    """Fakeify the flat capture inputs for an unbacked dynamic-shape capture.

    Params/buffers and unmarked dims become static fakes; each mark_unbacked dim becomes
    an UNBACKED SymInt (unguardable, so the artifact is valid for any runtime size and a
    graph that needs to guard on it fails at capture). Dims sharing a ``shape_id`` reuse
    one symbol; ``min``/``max`` add runtime asserts. Returns ``(flat_fake, fake_mode)``;
    the fake_mode (ShapeEnv) is threaded to the lowering via from_tracing_context.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)
    # shape_id -> unbacked symint (a dynamic SymInt); untyped so grouped dims share one symbol.
    shared: dict[object, Any] = {}
    with fake_mode:
        fake_pb = [fake_mode.from_tensor(t, static_shapes=True) for t in pb_flat]
        fake_user: list[object] = []
        for leaf, per in zip(user_flat, marks):
            if not isinstance(leaf, torch.Tensor):
                fake_user.append(leaf)
            elif not per:
                fake_user.append(fake_mode.from_tensor(leaf, static_shapes=True))
            else:
                sizes: list[Any] = []  # mix of static ints and unbacked SymInts
                for i, s in enumerate(leaf.shape):
                    if i not in per:
                        sizes.append(int(s))
                        continue
                    shape_id, lo, hi, hint = per[i]
                    if shape_id is not None and shape_id in shared:
                        u = shared[shape_id]
                        # Reusing the shared symbol still applies THIS occurrence's
                        # bounds: distinct dims grouped by shape_id may each carry their
                        # own (min, max), and dropping them would lose a runtime assert.
                        if lo is not None:
                            torch._check(u >= lo)
                        if hi is not None:
                            torch._check(u <= hi)
                        sizes.append(u)
                        continue
                    u = shape_env.create_unbacked_symint()
                    torch._check(u >= 0)
                    if lo is not None:
                        torch._check(u >= lo)
                    if hi is not None:
                        torch._check(u <= hi)
                    # hint_override is a perf-only autotuning size hint (not a guard):
                    # thread it onto the fresh symbol so inductor autotuning sees it. For a
                    # shared shape_id the group's one symbol keeps the first hint set here.
                    if hint is not None:
                        shape_env._set_unbacked_var_to_hint_override(u, hint)
                    if shape_id is not None:
                        shared[shape_id] = u
                    sizes.append(u)
                memory_format = _detect_memory_format(leaf)
                f = torch.empty(
                    sizes,
                    dtype=leaf.dtype,
                    device=leaf.device,
                    memory_format=memory_format,
                )
                f.requires_grad_(leaf.requires_grad)
                fake_user.append(f)
    return [*fake_pb, *fake_user], fake_mode


def _check_no_constant_tensors(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 1 of Note [precompile programming model]: everything live
    is an input.

    Every legitimate tensor in a non-strict capture is a placeholder (a lifted
    parameter/buffer or user input) or the result of a ``call_function`` node.
    A ``get_attr`` pointing at a tensor therefore means some tensor was closed
    over (a global, captured local, or non-registered module attribute) and would
    be baked into the graph as a constant, which we forbid.
    """
    offending = [
        (target, tuple(attr.shape), str(attr.dtype))
        for target, attr in _resolved_get_attrs(gm)
        if isinstance(attr, torch.Tensor)
    ]
    if offending:
        raise PrecompileError(
            "precompile traced a tensor that is neither a graph input "
            "(module parameter/buffer or user input) nor an intermediate. Such "
            "tensors would be hard-coded into the graph. This fires for a tensor "
            "closed over by fn (a global or captured local) or a plain "
            "(non-registered) module attribute, and also for a tensor literal "
            "constructed inside fn (e.g. torch.tensor([...])). Offending constants "
            f"(target, shape, dtype): {offending}. Fix by passing the tensor as an "
            "explicit argument; for module state register it as a parameter/buffer, "
            "and for a literal hoist it out of fn and pass it as an argument."
        )


def _assert_no_control_flow_subgraphs(gm: torch.fx.GraphModule) -> None:
    """Reject captured control-flow HOP subgraphs (e.g. from ``torch.cond``).

    They appear as ``get_attr`` nodes pointing at nested ``GraphModule`` submodules.
    The eager backend inlines ``gm.code`` and cannot reach such submodules (they are
    not on the standalone ``_GraphSelf`` holder), and the standalone composition does
    not inline them either, so the artifact would crash at runtime. Fail at capture
    with a concrete reason instead, like ``_assert_supported``.
    """
    offending = [
        target
        for target, attr in _resolved_get_attrs(gm)
        if isinstance(attr, torch.fx.GraphModule)
    ]
    if offending:
        raise PrecompileError(
            "precompile cannot lower a captured control-flow subgraph (e.g. from "
            f"torch.cond / torch.while_loop); not supported yet. Offending get_attr "
            f"targets: {offending}."
        )


def _intern_param_buffers(
    mods: list[torch.nn.Module],
) -> tuple[
    list[Tensor], list[str], list[str], list[tuple[_ModuleIndex, str, int]], int
]:
    """Lift each module's parameters then buffers to a flat list, interning by
    tensor identity so a tied weight becomes a single entry (one optimizer step,
    accumulated gradient -- not one per name).

    Returns ``(pb_flat, param_names, buffer_names, alias_entries, num_params)``,
    where ``alias_entries`` maps each ``(module_index, name)`` to its index in
    ``pb_flat`` (used to reparametrize during capture). This same params-then-
    buffers, intern-by-identity order is reproduced at runtime against the
    user-supplied modules, so the dense list lines up with the compiled graph.

    INVARIANT: the all-modules' params then all-modules' buffers, dedup-by-id ordering
    here is load-bearing and is reproduced VERBATIM by
    ``torch._precompile_driver._extract_param_buffers`` (emitted into the inlined/eager
    load paths). The cached load path uses this function directly, so both must stay
