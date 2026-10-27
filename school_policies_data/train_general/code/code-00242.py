            warnings.warn("Can't get device handle", stacklevel=2)
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID", stacklevel=2)
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


def _transform_uuid_to_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_ordinal(candidate: str, uuids: list[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambiguous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: list[int] = []
    for candidate in candidates:
        if torch.version.hip:
            candidate = candidate.replace(
                "GPU-", "", 1
            )  # Remove GPU-prefix to match amdsmi asic serial
        idx = uuid_to_ordinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return cast(list[int], [])
        rc.append(idx)
    return rc


def _device_count_amdsmi() -> int:
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            uuids = _raw_device_uuid_amdsmi()
            if uuids is None:
                return -1
            # Create string version of visible devices to avoid mypy warnings
            visible_device_str = cast(list[str], visible_devices)
            visible_devices = _transform_uuid_to_ordinals(visible_device_str, uuids)
        else:
            raw_cnt = _raw_device_count_amdsmi()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_nvml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(
                cast(list[str], visible_devices), uuids
            )
        else:
            raw_cnt = _raw_device_count_nvml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _get_nvml_device_index(device: Device) -> int:
    r"""Return the NVML index of the device, taking CUDA_VISIBLE_DEVICES into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices = _transform_uuid_to_ordinals(
            cast(list[str], visible_devices), uuids
        )
    visible_devices = cast(list[int], visible_devices)
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (CUDA_VISIBLE_DEVICES={visible_devices})"
        )
    return visible_devices[idx]


_cached_device_count: int | None = None


def device_count() -> int:
    r"""
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    if _initialized or hasattr(_tls, "is_initializing"):
        r = torch._C._cuda_getDeviceCount()
    else:
        # bypass _device_count_nvml() if rocm (not supported)
        if torch.version.hip:
            nvml_count = _device_count_amdsmi()
        else:
            nvml_count = _device_count_nvml()
        r = torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
    # NB: Do not cache the device count prior to CUDA initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to CUDA initialization.
    if _initialized:
        _cached_device_count = r
    return r


def get_arch_list() -> list[str]:
    r"""Return list CUDA architectures this library was compiled for."""
    if not _is_compiled():
        return []
    arch_flags = torch._C._cuda_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Return NVCC gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize(device: Device = None) -> None:
    r"""Wait for all kernels in all streams on a CUDA device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()


def ipc_collect():
    r"""Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    _lazy_init()
    return torch._C._cuda_ipc_collect()


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Device = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    r"""Return a :class:`Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this returned
       stream is being used.

    Args:
        data_ptr(int): Integer representation of the `cudaStream_t` value that
            is allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """
    _lazy_init()
    streamdata = torch._C._cuda_getStreamFromExternal(
        data_ptr, _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentBlasHandle()


def current_solver_handle():
    r"""Return cusolverDnHandle_t pointer to current cuSOLVER handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentSolverHandle()


def set_sync_debug_mode(debug_mode: int | str) -> None:
    r"""Set the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch._C._cuda_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Return current value of debug mode for cuda synchronizing operations."""
    _lazy_init()
    return torch._C._cuda_get_sync_debug_mode()


def _get_pynvml_handler(device: Device = None):
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "nvidia-ml-py does not seem to be installed or it can't be imported."
            # pyrefly: ignore [invalid-inheritance]
        ) from _PYNVML_ERR
    # pyrefly: ignore [import-error, missing-import, missing-module-attribute]
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded as e:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?") from e

    device = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return handle


def _get_amdsmi_handler(device: Device = None):
    if not _HAS_AMDSMI:
        raise ModuleNotFoundError(
            "amdsmi does not seem to be installed or it can't be imported."
        ) from _AMDSMI_ERR
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        raise RuntimeError(
            "amdsmi driver can't be loaded, requires >=ROCm6.0 installation"
        ) from e
    device = _get_amdsmi_device_index(device)
    handle = amdsmi.amdsmi_get_processor_handles()[device]
    return handle


_cached_hip_to_amdsmi: dict[int, int] | None = None


def _get_amdsmi_device_index_from_hip_index(device: int) -> int:
    r"""Return amdsmi index from HIP device index. They are not always the same.

    Assume amdsmi_init() already completes successfully."""
    global _cached_hip_to_amdsmi
    if _cached_hip_to_amdsmi is None:
        amdsmi_handles = amdsmi.amdsmi_get_processor_handles()

        def gen():
            for amdsmi_idx, handle in enumerate(amdsmi_handles):
                info = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
                if "hip_id" in info:
                    yield info["hip_id"], amdsmi_idx

        _cached_hip_to_amdsmi = dict(gen())
        if not _cached_hip_to_amdsmi and len(amdsmi_handles) > 1:
            warnings.warn(
                "Cannot translate HIP ID to AMD SMI ID due to"
                " lack of translation information prior to ROCm 6.4."
                " Functions that rely on amdsmi"
                " (e.g. temperature()) may operate on wrong devices."
            )
    if device not in _cached_hip_to_amdsmi:
        warnings.warn(
            f"Cannot translate HIP ID {device} to AMD SMI ID due to"
            " undetected HIP ID from amdsmi."
            " amdsmi_get_gpu_enumeration_info() only report these HIP IDs"
            f" {list(_cached_hip_to_amdsmi.keys())}."
            " Functions that rely on amdsmi"
            " (e.g. temperature()) may operate on wrong devices."
        )
    return _cached_hip_to_amdsmi.get(device, device)


def _get_amdsmi_device_index(device: Device) -> int:
    r"""Return the amdsmi index of the device, taking visible_devices into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    visible_device_is_str = type(visible_devices[0]) is str
    if visible_device_is_str:
        uuids = _raw_device_uuid_amdsmi()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices_str = cast(
            list[str], visible_devices
        )  # Create str variable for mypy
        visible_devices = _transform_uuid_to_ordinals(visible_devices_str, uuids)
    idx_map = dict(enumerate(cast(list[int], visible_devices)))
    if idx not in idx_map:
        raise RuntimeError(
            f"device {idx} is not visible (HIP_VISIBLE_DEVICES={visible_devices})"
        )
    if visible_device_is_str:
        return idx_map[idx]
    else:
        return _get_amdsmi_device_index_from_hip_index(idx_map[idx])


def _get_amdsmi_device_memory_used(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    # amdsmi_get_gpu_vram_usage returns mem usage in megabytes
    mem_mega_bytes = amdsmi.amdsmi_get_gpu_vram_usage(handle)["vram_used"]
    mem_bytes = mem_mega_bytes * 1024 * 1024
    return mem_bytes


def _get_amdsmi_memory_usage(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["umc_activity"]


def _get_amdsmi_utilization(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]


def _get_amdsmi_temperature(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_temp_metric(
        handle,
        amdsmi.AmdSmiTemperatureType.JUNCTION,
        amdsmi.AmdSmiTemperatureMetric.CURRENT,
    )


def _get_amdsmi_power_draw(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    socket_power = amdsmi.amdsmi_get_power_info(handle)["average_socket_power"]
    if socket_power != "N/A":
        return socket_power
    else:
        socket_power = amdsmi.amdsmi_get_power_info(handle)["current_socket_power"]
        if socket_power != "N/A":
            return socket_power
        else:
            return 0


def _get_amdsmi_clock_rate(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    clock_info = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
    if "cur_clk" in clock_info:  # ROCm 6.2 deprecation
        clock_rate = clock_info["cur_clk"]
    else:
        clock_rate = clock_info["clk"]
    if clock_rate != "N/A":
        return clock_rate
    else:
        return 0


def device_memory_used(device: Device = None) -> int:
    r"""Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    else:
        return _get_amdsmi_device_memory_used(device)


def memory_usage(device: Device = None) -> int:
    r"""Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    else:
        return _get_amdsmi_memory_usage(device)


def utilization(device: Device = None) -> int:
    r"""Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        return _get_amdsmi_utilization(device)


def temperature(device: Device = None) -> int:
    r"""Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 0 refers to the temperature sensor for the GPU die.
        return pynvml.nvmlDeviceGetTemperature(handle, 0)
    else:
        return _get_amdsmi_temperature(device)


def power_draw(device: Device = None) -> int:
    r"""Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    else:
        return _get_amdsmi_power_draw(device)


def clock_rate(device: Device = None) -> int:
    r"""Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetClockInfo(handle, 1)
    else:
        return _get_amdsmi_clock_rate(device)


def _get_device(device: int | str | torch.device) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the CUDA Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.cuda.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: int | str | torch.device = "cuda"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: int | str | torch.device = "cuda") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


# pyrefly: ignore [deprecated]
from .memory import *  # noqa: F403
from .memory import _use_uvm
from .random import *  # noqa: F403


################################################################################
# Define Storage and Tensor classes
################################################################################


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase:
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for CUDA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for CUDA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for CUDA storage")


class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


del _LegacyStorage
del _CudaLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)


class _WrappedTritonKernel:
    """Just a simple wrapper to store some metadata for testing purposes."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.kernel_invoked = False

    def __call__(self, *args, **kwargs):
        res = self.kernel(*args, **kwargs)
        self.kernel_invoked = True
        return res


def _register_triton_kernels():
    @_WrappedTritonKernel
    def kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_mm

        # pyrefly: ignore [not-callable]
        return bsr_dense_mm(*args, skip_checks=True, **kwargs)

    @_WrappedTritonKernel
    def addmm_kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_addmm

        return bsr_dense_addmm(*args, skip_checks=True, **kwargs)

    has_triton = importlib.util.find_spec("triton") is not None
    if has_triton:
        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_mm_out",
            "_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)",
            kernel_impl,
            "SparseCsrCUDA",
        )

        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_addmm_out",
            (
                "_triton_bsr_dense_addmm_out(Tensor input, Tensor bsr, Tensor dense,"
                " *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)"
            ),
            addmm_kernel_impl,
            "SparseCsrCUDA",
        )


_lazy_call(_register_triton_kernels)


def _compile_kernel(
    kernel_source: str,
    kernel_name: str,
    compute_capability: str | None = None,
    cuda_include_dirs: list | None = None,
    nvcc_options: list | None = None,
):
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

    # Compile the kernel to PTX
    ptx, mangled_name = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [mangled_name])

    if isinstance(result, dict):
        return result[mangled_name]
    else:
        # This branch shouldn't be executed if kernel_names is provided,
        # but MyPy needs this to understand type narrowing
        return getattr(result, mangled_name)


from . import amp, graph_annotations, jiterator, nvtx, profiler, sparse, tunable


_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])


__all__ = [
    # Typed storage and tensors
    "BFloat16Storage",
    # pyrefly: ignore [bad-dunder-all]
    "BFloat16Tensor",
    "BoolStorage",
    # pyrefly: ignore [bad-dunder-all]
    "BoolTensor",
    "ByteStorage",
    # pyrefly: ignore [bad-dunder-all]
    "ByteTensor",
    "CharStorage",
    # pyrefly: ignore [bad-dunder-all]
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "DoubleStorage",
    # pyrefly: ignore [bad-dunder-all]
    "DoubleTensor",
    "FloatStorage",
    # pyrefly: ignore [bad-dunder-all]
    "FloatTensor",
    "HalfStorage",
    # pyrefly: ignore [bad-dunder-all]
    "HalfTensor",
    "IntStorage",
    # pyrefly: ignore [bad-dunder-all]
    "IntTensor",
    "LongStorage",
    # pyrefly: ignore [bad-dunder-all]
    "LongTensor",
    "ShortStorage",
    # pyrefly: ignore [bad-dunder-all]
    "ShortTensor",
    "CUDAGraph",
    "CudaError",
    "DeferredCudaCallError",
    "Event",
    "ExternalStream",
    "Stream",
    "StreamContext",
    "GreenContext",
    "amp",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_disabled",
    "caching_allocator_enable",
    "can_device_access_peer",
    "check_error",
    "cudaStatus",
    "cudart",
    "current_blas_handle",
    "current_solver_handle",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "export_dot",
    "export_graph_data",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_annotations",
    "graph_pool_handle",
    "graphs",
    "has_half",
    "has_magma",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "jiterator",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "memory_usage",
    "MemPool",
    "use_mem_pool",
    "temperature",
    "power_draw",
    "clock_rate",
    "nccl",
    "nvtx",
    "profiler",
    "random",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    "sparse",
    "stream",
    "streams",
    "synchronize",
    "tunable",
    "utilization",
]
<|endoftext|>#!/usr/bin/env python3
"""Post-process a profiler trace to add CUDA graph kernel annotations.

Reads a profiler trace (gzipped or plain JSON) and a kernel annotations
pickle, matches kernel events by their graph node id, and writes an
annotated trace with the annotation fields added to each kernel event's
args (displayed alongside grid/block size in trace viewers).

The annotations pickle is auto-discovered from the trace file's parent
directory (one level up, matching the rank from the trace filename).

Usage:
    python -m torch.cuda._annotate_cuda_graph_trace <trace_file> [-a <annotations_pkl>] [-o <output_file>]

Examples:
    # Auto-discover annotations pickle from trace location
    python -m torch.cuda._annotate_cuda_graph_trace \\
        traces/step_000000000014/000000.*.pt.trace.json.gz

    # Explicit annotations pickle
    python -m torch.cuda._annotate_cuda_graph_trace trace.json.gz -a annotations.pkl
"""

import argparse
import gzip
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_WORK_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _move_overlapping_to_stream(
    trace: dict, default_stream: int = 7, overlap_stream: int = 8
) -> int:
    """Move graphed kernels that overlap with their predecessor to a separate stream.

    Perfetto cannot display overlapping (non-nested) events on the same
    stream -- they get hidden.  This pass detects graphed kernel events on
    *default_stream* whose start timestamp falls before the previous
    kernel's end, and moves them to *overlap_stream* so they're visible.

    Returns the number of events moved.
    """
    graphed_on_default = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel"
        and e.get("tid") == default_stream
        and e.get("args", {}).get("graph node id", 0) != 0
    ]
    graphed_on_default.sort(key=lambda e: e["ts"])

    moved = 0
    prev_end = 0.0
    for event in graphed_on_default:
        ts = event["ts"]
        dur = event.get("dur", 0)
        if ts < prev_end:
            event["tid"] = overlap_stream
            event.get("args", {})["stream"] = overlap_stream
            moved += 1
        else:
            prev_end = ts + dur

    return moved


def _fix_overlapping_timestamps(trace: dict, max_adjust_us: float = 1.0) -> int:
    """Clamp graphed kernel/memcpy timestamps so they don't overlap on the same stream.

    CUPTI can produce slightly overlapping timestamps for consecutive graphed
    events, causing Perfetto to hide events that sit entirely "under" their
    neighbours.  This pass sorts graphed work events per stream and ensures
    each event starts at or after the previous event's end.

    Overlaps larger than *max_adjust_us* are flagged as warnings and left
    unchanged, since they likely indicate a real issue rather than CUPTI
    timestamp jitter.

    Returns the number of events adjusted.
    """
    per_stream: dict[int, list[dict]] = defaultdict(list)
    for event in trace["traceEvents"]:
        if (
            event.get("cat") in _WORK_CATEGORIES
            and event.get("args", {}).get("graph node id", 0) != 0
        ):
            per_stream[event.get("tid")].append(event)

    adjusted = 0
    for tid, events in per_stream.items():
        events.sort(key=lambda e: e["ts"])
        prev_end = 0.0
        for event in events:
            ts = event["ts"]
            dur = event.get("dur", 0)
            if ts < prev_end:
                overlap = prev_end - ts
                if overlap > max_adjust_us:
                    print(
                        f"WARNING: large overlap {overlap:.3f}us on stream {tid} "
                        f"for {event.get('name', '?')[:60]}, skipping adjustment"
                    )
                else:
                    event["ts"] = prev_end
                    adjusted += 1
            prev_end = event["ts"] + dur

    return adjusted


def annotate_trace(
    trace: dict,
    annotations: dict[int, list[Any]],
    default_stream: int = 7,
) -> int:
    """Add annotation fields to kernel events matching the annotations dict.

    Each annotation entry is a list (from nested ``mark_kernels`` scopes).
    Fields from all annotations are merged into the event args; if multiple
    annotations define the same key, later entries in the list win.

    For graphed events (graph_node_id != 0), reassigns ``tid`` and
    ``args["stream"]`` to the stream recorded in annotations, or to
    *default_stream* if there is no annotation.  Also moves the
    corresponding ``ac2g`` flow-finish events to the new tid so that
    CPU-to-GPU correlation arrows are preserved.

    Removes ``gpu_user_annotation`` events and orphaned ``ac2g`` events
    from streams that have no kernel or memcpy events after reassignment,
    since CUPTI replicates these onto every stream during graph replay.

    Returns the number of events annotated.
    """
    # Build an index of ac2g 'f' events keyed by (tid, ts) so we can
    # move them together with the kernel events they correspond to.
    ac2g_f_index: dict[tuple, list] = {}
    for event in trace["traceEvents"]:
        if event.get("cat") == "ac2g" and event.get("ph") == "f":
            key = (event.get("tid"), event.get("ts"))
            ac2g_f_index.setdefault(key, []).append(event)

    annotated = 0
    for event in trace.get("traceEvents", []):
        args = event.get("args", {})
        graph_node_id = args.get("graph node id")
        if graph_node_id is None or graph_node_id == 0:
            continue
        stream_id = None
        if graph_node_id in annotations:
            for ann in annotations[graph_node_id]:
                if isinstance(ann, dict):
                    for key, value in ann.items():
                        # Legacy pickles wrapped bare strings as {"str": ...};
                        # the canonical key is "name".
                        args["name" if key == "str" else key] = str(value)
                    if "stream" in ann:
                        stream_id = int(ann["stream"])
                else:
                    # Legacy pickles stored bare strings unwrapped.
                    args["name"] = str(ann)
            annotated += 1

        # Reassign stream: use annotated stream if available, else default
        if stream_id is None:
            stream_id = default_stream
        old_key = (event.get("tid"), event.get("ts"))
        event["tid"] = stream_id
        args["stream"] = stream_id

        # Move the matching ac2g 'f' event(s) to the same new tid
        for ac2g_event in ac2g_f_index.get(old_key, ()):
            ac2g_event["tid"] = stream_id

    # Remove gpu_user_annotation events and ac2g flow-finish events from
    # streams that have no real kernel/memcpy/memset work -- these are
    # noise replicated by CUPTI onto every stream during graph replay.
    tids_with_work = set()
    for event in trace["traceEvents"]:
        if event.get("cat") in _WORK_CATEGORIES:
            tids_with_work.add(event.get("tid"))

    def _is_noise(event: dict) -> bool:
        cat = event.get("cat")
        if cat == "gpu_user_annotation":
            return event.get("tid") not in tids_with_work
        if cat == "ac2g" and event.get("ph") == "f":
            return event.get("tid") not in tids_with_work
        return False

    original_count = len(trace["traceEvents"])
    trace["traceEvents"] = [
        event for event in trace["traceEvents"] if not _is_noise(event)
    ]
    removed = original_count - len(trace["traceEvents"])
    if removed:
        print(f"Removed {removed} noise events from empty streams")

    # Clean up metadata: remove thread_name / thread_sort_index entries
    # for noise streams that have no real (non-metadata) events, and add
    # thread_name entries for our new annotation streams.
    all_tids_in_trace = {
        e.get("tid") for e in trace["traceEvents"] if e.get("ph") != "M"
    }
    # Find the GPU process pid from existing thread_name metadata
    gpu_pid = 0
    for event in trace["traceEvents"]:
        if (
            event.get("ph") == "M"
            and event.get("name") == "thread_name"
            and str(event.get("args", {}).get("name", "")).startswith("stream ")
        ):
            gpu_pid = event.get("pid", 0)
            break

    # Remove metadata entries for tids with no non-metadata events
    trace["traceEvents"] = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") != "M" or event.get("tid") in all_tids_in_trace
    ]

    # Add thread_name metadata for new annotation tids that lack one
    existing_thread_names = {
        e.get("tid")
        for e in trace["traceEvents"]
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    for tid in sorted(tids_with_work - existing_thread_names):
        trace["traceEvents"].append(
            {
                "ph": "M",
                "pid": gpu_pid,
                "tid": tid,
                "name": "thread_name",
                "args": {"name": f"stream {tid}"},
            }
        )

    return annotated


def load_trace(path: Path) -> dict:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def save_trace(trace: dict, path: Path) -> None:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
    else:
        with open(path, "w") as f:
            json.dump(trace, f)


def _find_annotations_pkl(trace_file: Path) -> Path | None:
    """Auto-discover the annotations pickle from the trace file location.

    Trace files live in e.g. ``traces/step_000000000014/000000.<id>.pt.trace.json.gz``
    where the leading digits are the rank. The pickle lives one level up:
    ``traces/kernel_annotations_rank0_*.pkl``.
    """
    match = re.match(r"^(\d+)", trace_file.name)
    if not match:
        return None
    rank = int(match.group(1))

    traces_dir = trace_file.parent.parent
    candidates = sorted(traces_dir.glob(f"kernel_annotations_rank{rank}_*.pkl"))
    if candidates:
        return candidates[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a profiler trace with CUDA graph kernel annotations."
    )
    parser.add_argument(
        "trace_file", type=Path, help="Input trace file (.json or .json.gz)"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=Path,
        default=None,
        help="Kernel annotations pickle file. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to <trace_file>.annotated.<ext>",
    )
    parser.add_argument(
        "--default-stream",
        type=int,
        default=7,
        help="Stream ID to assign to unannotated graphed events (default: 7).",
    )
    args = parser.parse_args()

    annotations_pkl = args.annotations
    if annotations_pkl is None:
        annotations_pkl = _find_annotations_pkl(args.trace_file)
        if annotations_pkl is None:
            print(
                f"Could not auto-discover annotations pickle for {args.trace_file}. "
                f"Use -a to specify it explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-discovered annotations: {annotations_pkl}")

    with open(annotations_pkl, "rb") as f:
        annotations = pickle.load(f)
    print(f"Loaded {len(annotations)} kernel annotations")

    trace = load_trace(args.trace_file)
    total_events = len(trace.get("traceEvents", []))
    print(f"Loaded trace with {total_events} events")

    count = annotate_trace(trace, annotations, default_stream=args.default_stream)
    print(f"Annotated {count} kernel events")

    overlap_moved = _move_overlapping_to_stream(
        trace, default_stream=args.default_stream
    )
    if overlap_moved:
        print(f"Moved {overlap_moved} overlapping events to stream 8")

    ts_fixed = _fix_overlapping_timestamps(trace)
    if ts_fixed:
        print(f"Fixed {ts_fixed} overlapping graphed event timestamps")

    output = args.output
    if output is None:
        name = args.trace_file.name
        if name.endswith(".json.gz"):
            output = args.trace_file.with_name(
                name.replace(".json.gz", ".annotated.json.gz")
            )
        elif name.endswith(".json"):
            output = args.trace_file.with_suffix(".annotated.json")
        else:
            output = args.trace_file.with_suffix(args.trace_file.suffix + ".annotated")

    save_trace(trace, output)
    print(f"Saved annotated trace to {output}")


if __name__ == "__main__":
    main()
<|endoftext|>import ctypes
import sys
from typing import Any

import torch


try:
    from cuda.bindings import (  # pyrefly: ignore[missing-import]
        driver as _cuda_bindings_driver,
        runtime as _cuda_bindings_runtime,
    )

    _HAS_CUDA_BINDINGS = True
except ImportError:
    _cuda_bindings_driver = None  # type: ignore[assignment]
    _cuda_bindings_runtime = None  # type: ignore[assignment]
    _HAS_CUDA_BINDINGS = False

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


def _get_hip_runtime_library() -> ctypes.CDLL:
    # If ROCm python packages are available, query the OS-independent absolute
    # path to the library provided by those packages, including any version suffix.
    # See https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md#dynamic-library-resolution
    try:
        # pyrefly: ignore [import-error, missing-import]
        import rocm_sdk

        lib = ctypes.CDLL(str(rocm_sdk.find_libraries("amdhip64")[0]))
    except (ImportError, IndexError):
        if sys.platform == "win32":
            lib = ctypes.CDLL(f"amdhip64_{torch.version.hip[0]}.dll")
        else:  # Unix-based systems
            lib = ctypes.CDLL("libamdhip64.so")

    lib.cuGetErrorString = lib.hipGetErrorString  # type: ignore[attr-defined]
    lib.cuModuleLoadData = lib.hipModuleLoadData  # type: ignore[attr-defined]
    lib.cuModuleGetFunction = lib.hipModuleGetFunction  # type: ignore[attr-defined]
    lib.cuLaunchKernel = lib.hipModuleLaunchKernel  # type: ignore[attr-defined]
    lib.cuFuncSetAttribute = lib.hipFuncSetAttribute  # type: ignore[attr-defined]
    return lib


def _get_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:  # Unix-based systems
        return ctypes.CDLL("libcuda.so.1")


# Load GPU driver runtime
def _get_gpu_runtime_library() -> ctypes.CDLL:
    if torch.version.hip:
        return _get_hip_runtime_library()
    else:
        return _get_cuda_library()


# Helper: check CUDA errors
def _check_cuda(result: int) -> None:
    if result == 0:
        return
    err_str = ctypes.c_char_p()
    libcuda = _get_gpu_runtime_library()  # Get reference to CUDA library
    libcuda.cuGetErrorString(result, ctypes.byref(err_str))
    error_message = (
        err_str.value.decode() if err_str.value is not None else "Unknown CUDA error"
    )
    raise RuntimeError(f"CUDA error: {error_message}")


def _check_cuda_bindings(result: Any) -> Any:
    """Check a cuda.bindings (cuda-python) call result for errors.

    All cuda.bindings driver/runtime calls return ``(error, *outputs)``. This
    helper unpacks the tuple, raises on non-success, and returns the
    outputs (``None`` for zero outputs, scalar for one, tuple otherwise).
    """
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("cuda.bindings is not available")
    # We assume that `result`` is a return value from cuda bindings API calls.
    # This means it is always a tuple, of length >= 1 and the first element
    # represents the error with a `value` attribute.
    # See also cuda-python error handling here:
    # https://github.com/NVIDIA/cuda-python/blob/88363f8f17636ecd28772b2b8ad41a33895b41bb/
    # cuda_bindings/cuda/bindings/_example_helpers/helper_cuda.py#L11-L31
    err, *out = result
    if err.value:
        # pyrefly: ignore [missing-attribute]
        if isinstance(err, _cuda_bindings_driver.CUresult):
            # pyrefly: ignore [missing-attribute]
            err2, name = _cuda_bindings_driver.cuGetErrorString(err)
            # pyrefly: ignore [missing-attribute]
            success = _cuda_bindings_driver.CUresult.CUDA_SUCCESS
            # cuGetErrorString can fail and leave name NULL. cudaGetErrorString
            # returns a fallback string for unknown runtime errors.
            err_str = name if err2 == success else "<unknown>"
        # pyrefly: ignore [missing-attribute]
        elif isinstance(err, _cuda_bindings_runtime.cudaError_t):
            # pyrefly: ignore [missing-attribute]
            err_str = _cuda_bindings_runtime.cudaGetErrorString(err)[1]
        else:
            err_str = "unknown error type"
        if isinstance(err_str, bytes):
            err_str = err_str.decode()
        raise RuntimeError(f"CUDA error: {err} ({err_str})")

    if len(out) == 0:
        return None
    if len(out) == 1:
        return out[0]
    return out


def _check_cuda_bindings_driver(result: Any) -> Any:
    """Check a cuda.bindings (cuda-python) driver call result for errors.

    Like ``_check_cuda_bindings`` but for the CUDA driver API, whose calls
    return ``(CUresult, *outputs)`` and report errors via ``cuGetErrorString``.
    """
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("cuda.bindings is not available")
    err, *out = result
    if (
        err
        != _cuda_bindings_driver.CUresult.CUDA_SUCCESS  # pyrefly: ignore[missing-attribute]
    ):
        _, err_str = (
            _cuda_bindings_driver.cuGetErrorString(  # pyrefly: ignore[missing-attribute]
                err
            )
        )
        if isinstance(err_str, bytes):
            err_str = err_str.decode()
        raise RuntimeError(f"CUDA driver error: {err} ({err_str})")
    if len(out) == 0:
        return None
    if len(out) == 1:
        return out[0]
    return out


def _get_hiprtc_library() -> ctypes.CDLL:
    try:
        # pyrefly: ignore [import-error, missing-import]
        import rocm_sdk

        lib = ctypes.CDLL(str(rocm_sdk.find_libraries("hiprtc")[0]))
    except (ImportError, IndexError):
        if sys.platform == "win32":
            version_str = "".join(
                ["0", torch.version.hip[0], "0", torch.version.hip[2]]
            )
            lib = ctypes.CDLL(f"hiprtc{version_str}.dll")
        else:
            lib = ctypes.CDLL("libhiprtc.so")

    # Provide aliases for HIP RTC functions to match NVRTC API
    lib.nvrtcGetErrorString = lib.hiprtcGetErrorString  # type: ignore[attr-defined]
    lib.nvrtcCreateProgram = lib.hiprtcCreateProgram  # type: ignore[attr-defined]
    lib.nvrtcDestroyProgram = lib.hiprtcDestroyProgram  # type: ignore[attr-defined]
    lib.nvrtcCompileProgram = lib.hiprtcCompileProgram  # type: ignore[attr-defined]
    lib.nvrtcGetCUBINSize = lib.hiprtcGetCodeSize  # type: ignore[attr-defined]
    lib.nvrtcGetCUBIN = lib.hiprtcGetCode  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLogSize = lib.hiprtcGetProgramLogSize  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLog = lib.hiprtcGetProgramLog  # type: ignore[attr-defined]
    lib.nvrtcAddNameExpression = lib.hiprtcAddNameExpression  # type: ignore[attr-defined]
    lib.nvrtcGetLoweredName = lib.hiprtcGetLoweredName  # type: ignore[attr-defined]
    return lib


def _get_nvrtc_library() -> ctypes.CDLL:
    major_version = int(torch.version.cuda.split(".")[0])  # type: ignore[union-attr]
    if sys.platform == "win32":
        nvrtc_libs = [
            f"nvrtc64_{major_version}0_0.dll",
        ]
    else:
        nvrtc_libs = [
            f"libnvrtc.so.{major_version}",
            "libnvrtc.so",  # Fallback to unversioned
        ]
    for lib_name in nvrtc_libs:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    raise OSError("Could not find any NVRTC library")


def _get_gpu_rtc_library() -> ctypes.CDLL:
    # Since PyTorch already loads the GPU RTC library, we can use the system library
    # which should be compatible with PyTorch's version
    if torch.version.hip:
        return _get_hiprtc_library()
    else:
        return _get_nvrtc_library()


def _get_gpu_rtc_compatible_flags() -> list[str]:
    """
    Get HIPCC/NVCC flags that are compatible with NVRTC compilation.

    Returns:
        List of HIPCC/NVCC flags that can be safely used with NVRTC.
    """
    from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS, COMMON_NVCC_FLAGS

    nvrtc_unsupported_flags = {
        "--expt-relaxed-constexpr",
    }

    # Filter out unsupported flags
    compatible_flags = [
        flag for flag in COMMON_NVCC_FLAGS if flag not in nvrtc_unsupported_flags
    ]

    if torch.version.hip:
        compatible_flags.extend(COMMON_HIPCC_FLAGS)

    return compatible_flags


def _nvrtc_compile(
    kernel_source: str,
    kernel_name: str,
    compute_capability: str | None = None,
    cuda_include_dirs: list | None = None,
    nvcc_options: list | None = None,
    auto_pch: bool = False,
) -> tuple[bytes, str]:
    """
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, None): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, None): List of directories containing CUDA headers
        nvcc_options (list, None): Additional options to pass to NVRTC
        auto_pch (bool): Enable automatic precompiled headers (CUDA 12.8+)

    Returns:
        Tuple[bytes, str]: The compiled PTX code and mangled kernel name
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load NVRTC library
    libnvrtc = _get_gpu_rtc_library()

    # NVRTC constants
    NVRTC_SUCCESS = 0

    # Helper: check NVRTC errors
    def check_nvrtc(result: int) -> None:
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            error_message = (
                err_str.value.decode()
                if err_str.value is not None
                else "Unknown CUDA error"
            )
            raise RuntimeError(f"CUDA error: {error_message}")

    # Convert source to bytes
    source_bytes = kernel_source.encode("utf-8")

    # Get compute capability if not provided
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if torch.version.hip:
            compute_capability = f"{props.gcnArchName}"
        else:
            compute_capability = f"{props.major}{props.minor}"

    # Prepare compilation options
    options = []
    if torch.version.hip:
        options.append(f"--offload-arch={compute_capability}".encode())
    else:
        options.append(f"--gpu-architecture=sm_{compute_capability}".encode())

    # Auto-detect and add CUDA include paths
    from torch.utils.cpp_extension import include_paths

    cuda_include_paths = include_paths("cuda")
    for cuda_path in cuda_include_paths:
        options.append(f"-I{cuda_path}".encode())

    # Add custom include directories
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}".encode())

    # Enable automatic precompiled headers (CUDA 12.8+)
    if auto_pch:
        if str(torch.version.cuda) < "12.8":
            raise AssertionError(f"PCH requires CUDA 12.8+, got {torch.version.cuda}")
        if nvcc_options is None:
            nvcc_options = []
        nvcc_options.append("--pch")

    # Add custom NVCC options
    if nvcc_options:
        for option in nvcc_options:
            options.append(option.encode("utf-8"))

    nvrtc_compatible_flags = _get_gpu_rtc_compatible_flags()
    options.extend([flag.encode("utf-8") for flag in nvrtc_compatible_flags])

    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)

    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc(
        libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            f"{kernel_name}.cu".encode(),
            0,
            None,
            None,
        )
    )

    # Add kernel name, which can be a template expression
    c_kernel_name = kernel_name.encode("utf-8")
    check_nvrtc(libnvrtc.nvrtcAddNameExpression(prog, c_kernel_name))

    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)

    # Handle compilation errors
    if res != NVRTC_SUCCESS:
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")

    # Get binary
    binary_size = ctypes.c_size_t()
    check_nvrtc(libnvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(binary_size)))
    binary = ctypes.create_string_buffer(binary_size.value)
    check_nvrtc(libnvrtc.nvrtcGetCUBIN(prog, binary))

    # Get mangled name
    c_mangled_name = ctypes.c_char_p()
    check_nvrtc(
        libnvrtc.nvrtcGetLoweredName(prog, c_kernel_name, ctypes.byref(c_mangled_name))
    )
    if c_mangled_name.value is not None:
        mangled_name = c_mangled_name.value.decode()  # make a copy
    else:
        mangled_name = ""

    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    # For some reason, ".value" causes the string to be truncated,
    # likely due to the presence of '\0' in the string. So we use .raw instead.
    return binary.raw, mangled_name


class _CudaModule:
    def __init__(self, module: ctypes.c_void_p) -> None:
        self._module = module
        self._kernels: dict[str, _CudaKernel] = {}

    def __getattr__(self, name: str) -> "_CudaKernel":
        if name in self._kernels:
            return self._kernels[name]

        # Import the CUDA library inside the method
        # pyrefly: ignore [missing-module-attribute]
        from torch.cuda._utils import _get_gpu_runtime_library

        libcuda = _get_gpu_runtime_library()

        func = ctypes.c_void_p()
        try:
            _check_cuda(
                libcuda.cuModuleGetFunction(
                    ctypes.byref(func), self._module, name.encode("utf-8")
                )
            )
            kernel = _CudaKernel(func, self._module)
            self._kernels[name] = kernel
            return kernel

        except RuntimeError as err:
            raise AttributeError(f"No kernel named '{name}' in this module") from err


class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """

    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None:
        self.func = func
        self.module = module
        self._max_shared_mem_bytes = 0

    def __call__(
        self,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        args: list | None = None,
        shared_mem: int = 0,
        stream: Any | None = None,
    ) -> None:
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """
        import torch

        libcuda = torch.cuda._utils._get_gpu_runtime_library()

        if not args:
            args = []

        # Process arguments and convert tensors to pointers
        processed_args: list[ctypes.c_void_p] = []
        c_args = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda and not (arg.is_cpu and arg.is_pinned()):
                    raise ValueError(
                        "All tensor arguments must be CUDA tensors or pinned CPU tensors"
                    )
                # Get pointer to tensor data
                ptr = ctypes.c_void_p(arg.data_ptr())
                processed_args.append(ptr)
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                # Convert integers to C int
                c_int = ctypes.c_int(arg)
                # Store the C int for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_int))
            elif isinstance(arg, float):
                # Python floats are doubles - use double by default
                c_double = ctypes.c_double(arg)
                # Store the C double for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_double))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Convert to array of void pointers
        c_args_array = (ctypes.c_void_p * len(c_args))()
        for i, arg in enumerate(c_args):
            c_args_array[i] = ctypes.cast(arg, ctypes.c_void_p)

        # Get the stream
        if stream is None:
            # Defer import to avoid circular imports
            import torch.cuda

            stream = torch.cuda.current_stream()

        # Check if kernel requires large shared memory but hasn't been configured
        if shared_mem >= 48 * 1024 and (
            self._max_shared_mem_bytes == 0 or shared_mem > self._max_shared_mem_bytes
        ):
            configured_msg = (
                "not configured"
                if self._max_shared_mem_bytes == 0
                else f"only {self._max_shared_mem_bytes} bytes configured"
            )
            raise RuntimeError(
                f"Kernel requires {shared_mem} bytes of shared memory (>= 48KB), "
                f"but {configured_msg}. "
                "Call kernel.set_shared_memory_config(shared_mem) after compilation "
                "and before launching the kernel."
            )

        _check_cuda(
            libcuda.cuLaunchKernel(
                self.func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem,
                stream._as_parameter_,
                c_args_array,
                None,
            )
        )

    def set_shared_memory_config(self, shared_mem_bytes: int) -> None:
        if shared_mem_bytes < 48 * 1024:
            # No configuration needed for <= 48KB, just update the value
            self._max_shared_mem_bytes = shared_mem_bytes
            return

        libcuda = _get_gpu_runtime_library()

        # Get device properties to validate against limits
        device_props = torch.cuda.get_device_properties()
        # HIP doesn't have shared_memory_per_block_optin in device properties, so we hard-code it here
        if torch.version.hip:
            # navi, CDNA1-CDNA3 allows a max of 64KB shared memory,
            # CDNA4 (gfx950) 160KB, and CDNA5 (gfx1250) 320KB.
            gcn_arch = device_props.gcnArchName.split(":", 1)[0]
            if gcn_arch == "gfx950":
                max_shared_mem = 160 * 1024
            elif gcn_arch == "gfx1250":
                max_shared_mem = 320 * 1024
            else:
                max_shared_mem = 65536
        else:
            max_shared_mem = getattr(
                device_props, "shared_memory_per_block_optin", 49152
            )

        if shared_mem_bytes > max_shared_mem:
            raise RuntimeError(
                f"Requested shared memory ({shared_mem_bytes} bytes) exceeds "
                f"device limit ({max_shared_mem} bytes). "
                "Consider reducing block size or shared memory usage."
            )

        # Set the function attribute once
        # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8
        _check_cuda(
            libcuda.cuFuncSetAttribute(
                self.func,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_bytes,
            )
        )

        self._max_shared_mem_bytes = shared_mem_bytes


def _cuda_load_module(
    ptx: str | bytes, kernel_names: list[str] | None = None
) -> _CudaModule | dict[str, "_CudaKernel"]:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load CUDA driver library
    libcuda = _get_gpu_runtime_library()

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    # Get the current stream without directly importing torch.cuda at module level
    stream = torch.cuda.current_stream()
    with stream:
        _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if not kernel_names:
        return _CudaModule(module)

    # Return specific kernels
    kernels = {}
    for name in kernel_names:
        func = ctypes.c_void_p()
        _check_cuda(
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8")
            )
        )
        kernels[name] = _CudaKernel(func, module)
    return kernels


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(f"Expected a cuda or cpu device, but got: {device}")
        elif device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.cuda.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
<|endoftext|># mypy: allow-untyped-defs
import collections
import warnings
from collections.abc import Sequence

import torch.cuda


__all__ = ["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter"]

SUM = 0  # ncclRedOp_t


def is_available(tensors):
    if not hasattr(torch._C, "_nccl_all_reduce"):
        warnings.warn("PyTorch is not compiled with NCCL support", stacklevel=2)
        return False

    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True


def version():
    """
    Returns the version of the NCCL.


    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the NCCL.
    """
    ver = torch._C._nccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = torch._C._nccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)


def unique_id():
    return torch._C._nccl_unique_id()


def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)


def _check_sequence_type(inputs: torch.Tensor | Sequence[torch.Tensor]) -> None:
    if not isinstance(inputs, collections.abc.Container) or isinstance(
        inputs, torch.Tensor
    ):
        raise TypeError("Inputs should be a collection of tensors")


def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    _check_sequence_type(inputs)
    if outputs is None:
        outputs = inputs
    _check_sequence_type(outputs)
    torch._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


# `output` used to be `outputs`, taking in a list of tensors. So we have two
# arguments for BC reasons.
def reduce(
    inputs: Sequence[torch.Tensor],
    output: torch.Tensor | Sequence[torch.Tensor] | None = None,
    root: int = 0,
    op: int = SUM,
    streams: Sequence[torch.cuda.Stream] | None = None,
    comms=None,
    *,
    outputs: Sequence[torch.Tensor] | None = None,
) -> None:
    _check_sequence_type(inputs)
    _output: torch.Tensor
    if outputs is not None:
        if output is not None:
            raise ValueError(
                "'output' and 'outputs' can not be both specified. 'outputs' is deprecated in "
                "favor of 'output', taking in a single output tensor. The signature of reduce is: "
                "reduce(inputs, output=None, root=0, op=SUM, streams=None, comms=None)."
            )
        else:
            warnings.warn(
                "`nccl.reduce` with an output tensor list is deprecated. "
                "Please specify a single output tensor with argument 'output' instead.",
                FutureWarning,
                stacklevel=2,
            )
            _output = outputs[root]
    elif not isinstance(output, torch.Tensor) and isinstance(
        output, collections.abc.Sequence
    ):
        # User called old API with positional arguments of list of output tensors.
        warnings.warn(
            "nccl.reduce with an output tensor list is deprecated. "
            "Please specify a single output tensor.",
            FutureWarning,
            stacklevel=2,
        )
        _output = output[root]
    else:
        _output = inputs[root] if output is None else output
    torch._C._nccl_reduce(inputs, _output, root, op, streams, comms)


def broadcast(
    inputs: Sequence[torch.Tensor], root: int = 0, streams=None, comms=None
) -> None:
    _check_sequence_type(inputs)
    torch._C._nccl_broadcast(inputs, root, streams, comms)


def all_gather(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)


def reduce_scatter(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    op: int = SUM,
    streams=None,
    comms=None,
) -> None:
    _check_sequence_type(inputs)
    _check_sequence_type(outputs)
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)
<|endoftext|>r"""
This module exposes a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library
or more than one technique. For example, a GEMM could be implemented for CUDA or
ROCm using either the blas or blasLt libraries. Further, ROCm's rocblas and
hipblaslt libraries allow the user to query for all possible algorithms and then
choose one. How does one know which implementation is the fastest and should be
chosen? That's what TunableOp provides.

Enabling TunableOp and Tuning Separately
========================================

The TunableOp feature is enabled separately from enabling the tuning phase
itself. Enabling TunableOp means that PyTorch will replace any standard
operators with their Tunable implementations. Any call to a TunableOp first
checks whether it has already been tuned for the given operator inputs. If so,
it will immediately call the tuned operation; no further tuning will take place
even when the tuning setting is enabled. Instead if no tuning result is found,
and tuning is enabled, the TunableOp will benchmark every registered
implementation of that operator for the given set of inputs and select the
fastest.

File Input and Output
=====================

The first time any TunableOp is invoked, the internal database of tuned
operations will be prepared by attempting to read the results from the given
file. The default filename is 'tunableop_results.csv'. To support tuning when
multiple GPUs are used across multiple processes, the GPU device ordinal is
automatically inserted into the filename to avoid multiple processes overwriting
the same file.

If tuning is enabled and new tunings are discovered during the course of your
workload, it will also write out to this same filename with all tunings, both
the ones it read in at startup as well as the new ones found at runtime. This
can be used, for example, to build up a tunings file across many workloads by
reusing the same file. The output file is automatically created when the
application terminates. This behavior can be controlled by the C++ and Python
APIs but not the environment variables.

Assuming you specified a filename, you'll end up with a CSV file with contents
like so::

  Validator,PT_VERSION,2.2.0
  Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
  Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
  Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
  GemmTunableOp_float_NT,nt_25088_4096_64,Gemm_Hipblaslt_1219,1.262
  GemmTunableOp_float_NT,nt_4096_4096_64,Gemm_Rocblas_1216,0.033

Note the "Validator" lines. If you change a library version, or ROCm version, or
PyTorch version, TunableOp will detect this and reject the tunings file because
the prior tunings are likely affected by other software changes.

The remaining lines are the tuned solutions for each TunableOp encountered
during your execution. Each line consists of 4 comma-separated fields: operator
name, operator parameters, solution name, and average execution time. The
execution time is an optional field. The CSV file can be edited, but with
caution. For example, the solution name (field 3) can be changed to "Default"
and it will fall back to the original PyTorch untuned implementation. Or, in the
case of ROCm's hipBLAS or hipBLASLt libraries, if you know the specific solution
index you can override the solution that TunableOp selected by replacing the
value. The operator name and parameters (fields 1 and 2) are internally named
and should not be modified. In the case of GemmTunableOp, field 1 indicates the
datatype and whether the inputs are transposed (T) or not (N) and field 2
indicates the M, N, K input shapes.

There is an option to enable verbose output but it is only recommended for
debugging purposes. This will produce a lot of diagnostic messages but may be
useful to see if TunableOp is being used at all. Otherwise, TunableOp is
completely silent, besides file output, unless there is a warning or error
during its use. The verbose option is only available by setting the environment
variable PYTORCH_TUNABLEOP_VERBOSE=1.

A Note on Tuning Behavior, Warmup, and Cache Effects
====================================================

Tuning an operator consists of iterating through the list or registered
implementations and profiling each one. The profile is established by running a
single implementation in a loop multiple times and taking the average execution
time. There is also an optional warmup phase prior to tuning that can help with
reaching stable power states by the hardware. During tuning of a workload the
various hardware caches will more likely produce hits than when not tuning.
There are options for flushing the instruction cache and rotate the input tensors
which might help produce a more faithful profile of the tuned operator as if the
operator were run within a larger workload instead of in a tight, repetitive loop.

By default, each possible solution for a given operator will be run for either
100 iterations or as many iterations that can be run within 30ms, whichever is
smaller, and its average execution will be calculated. The fastest solution
among all that were successfully profiled will be chosen. A profile might fail
if the given solution doesn't achieve the same accuracy as the default
implementation or if the solution returns an error code.

CUDA cuBLASLt support uses the TunableOp result cache and profiling machinery
to time a configurable number of cuBLASLt heuristic candidates.

Current Tunable Operators
=========================

TunableGemm for ROCm
--------------------

Any call to at::cuda::blas::gemm() or ::bgemm() will be routed through TunableOp
when enabled. Calling gemm() for a given set of input arguments
(transa, transb, m, n, k) on ROCm will attempt to use the fastest available
implementation across both rocblas and hipblaslt. On CUDA, TunableGemm registers
cuBLASLt heuristic candidates for GEMM paths that already use cuBLASLt.

cuBLASLt Heuristic Tuning for CUDA
----------------------------------

The number of cuBLASLt heuristic candidates is controlled by
set_cublaslt_requested_algo_count() or
PYTORCH_TUNABLEOP_CUBLASLT_REQUESTED_ALGO_COUNT, which defaults to 8. If this
count is 1, only the top cuBLASLt heuristic candidate is available.

Offline Tuning
==============

Motivation
----------
There are several use cases for offline tuning.

One use case involves a workload with a high-memory utilization, where regular tuning might lead to running out of memory.

Another use case is for compute-intensive workloads. In such cases, it is more resource-efficient to collect
the GEMMs for the workload once and then tune repeatedly with different tuning parameters or libraries.

Workflow
--------
There are basically two steps:
1) Set the environment variables to collect the untuned GEMM and this will generate ``tunableop_untuned0.csv``:

.. code-block:: bash

   export PYTORCH_TUNABLEOP_ENABLED=1
   export PYTORCH_TUNABLEOP_TUNING=0
   export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
   ...

2) Run a Python script that reads the ``tunableop_untuned0.csv`` and generates the ``tunableop_results0.csv``, like this:

.. code-block:: python

   import torch.cuda.tunable as tunable
   import os

   os.putenv("PYTORCH_TUNABLEOP_ENABLED", "1")
   os.putenv("PYTORCH_TUNABLEOP_TUNING", "1")
   os.putenv("PYTORCH_TUNABLEOP_RECORD_UNTUNED", "0")
   tunable.tune_gemm_in_file("tunableop_untuned0.csv")


It is also possible to take multiple untuned files and distribute the GEMMs for tuning to multiple GPUs
within a single node. In the first step, the GEMMs are first gathered and duplicate GEMMs are eliminated.
Next, the GEMMs are distributed to different GPUs for tuning. After all GEMMs are tuned, the results from
all the GPUs are then gathered into a single file whose base filename has ``_full0`` appended to it
(for example ``tunableop_results_full0.csv``). Finally, this new file, containing the gathered results, will be
duplicated N times, once for each GPU as convenience to the user will run the workload with the tuned
configuration on N GPUs.

.. code-block:: python

   if __name__ == "__main__":
       num_gpus = 8  # number of GPUs that will be used during the tuning process
       tunable.mgpu_tune_gemm_in_file("tunableop_untuned?.csv", num_gpus)

Note that the usage of the ``mgpu_tune_gemm_in_file`` API is different from its single GPU counterpart
(``tune_gemm_in_file``). The body of the Python script that calls the API must be wrapped in ``main()`` as shown
due to the use of concurrent futures module. The argument to ``mgpu_tune_gemm_in_file`` must contain a wild card
expression (``?`` or ``*``) to generate the list of untuned files containing the GEMMs to be processed. The ``num_gpus``
must between 1 and the total number of GPUs available.

Tuning Context
==============

The behavior of TunableOp is currently manipulated through environment
variables, the C++ interface of at::cuda::tunable::getTuningContext(), or the
torch.cuda.tunable python interfaces. The environment variables take precedence
over any setting you manipulate using the C++ or Python APIs.

Environment Variable Interface
------------------------------
Environment variables are cached the first time they are read. You cannot use the
environment variable interface programmatically since the settings become fixed.
Use the C++ or Python APIs instead.

"""

import glob
import multiprocessing as mp
import os
import shutil
import warnings

import torch


__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "record_untuned_enable",
    "record_untuned_is_enabled",
    "set_max_tuning_duration",
    "get_max_tuning_duration",
    "set_max_tuning_iterations",
    "get_max_tuning_iterations",
    "set_cublaslt_requested_algo_count",
    "get_cublaslt_requested_algo_count",
    "set_filename",
    "get_filename",
    "get_results",
    "get_validators",
    "read_file",
    "tune_gemm_in_file",
    "mgpu_tune_gemm_in_file",
    "set_rotating_buffer_size",
    "get_rotating_buffer_size",
    "set_numerical_check_tolerances",
]


def enable(val: bool = True) -> None:
    r"""This is the big on/off switch for all TunableOp implementations."""
    torch._C._cuda_tunableop_enable(val)  # type: ignore[attr-defined]


def is_enabled() -> bool:
    r"""Returns whether the TunableOp feature is enabled."""
    return torch._C._cuda_tunableop_is_enabled()  # type: ignore[attr-defined]


def tuning_enable(val: bool = True) -> None:
    r"""Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
    torch._C._cuda_tunableop_tuning_enable(val)  # type: ignore[attr-defined]


def tuning_is_enabled() -> bool:
    r"""Returns whether TunableOp implementations can be tuned."""
    return torch._C._cuda_tunableop_tuning_is_enabled()  # type: ignore[attr-defined]


def record_untuned_enable(val: bool = True) -> None:
    r"""Enable recording untuned TunableOp operations for offline tuning.

    When enabled, if a tuned entry isn't found, write it to the untuned file.
    """
    torch._C._cuda_record_untuned_enable(val)  # type: ignore[attr-defined]


def record_untuned_is_enabled() -> bool:
    r"""Returns whether TunableOp operations are recorded for offline tuning."""
    return torch._C._cuda_record_untuned_is_enabled()  # type: ignore[attr-defined]


def set_max_tuning_duration(duration: int) -> None:
    r"""Set max time in milliseconds to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_duration(duration)  # type: ignore[attr-defined]


def get_max_tuning_duration() -> int:
    r"""Get max time to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_duration()  # type: ignore[attr-defined]


def set_max_tuning_iterations(iterations: int) -> None:
    r"""Set max number of iterations to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_iterations(iterations)  # type: ignore[attr-defined]


def get_max_tuning_iterations() -> int:
    r"""Get max iterations to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_iterations()  # type: ignore[attr-defined]


def set_cublaslt_requested_algo_count(count: int) -> None:
    r"""Set the number of cuBLASLt heuristic algorithms to request on CUDA.

    Values less than 1 are clamped to 1.
    """
    torch._C._cuda_tunableop_set_cublaslt_requested_algo_count(count)  # type: ignore[attr-defined]


def get_cublaslt_requested_algo_count() -> int:
    r"""Get the number of cuBLASLt heuristic algorithms requested on CUDA."""
    get_count = (
        torch._C._cuda_tunableop_get_cublaslt_requested_algo_count  # type: ignore[attr-defined]
    )
    return get_count()


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    r"""Set the filename to use for input/output of tuning results.

    If :attr:`insert_device_ordinal` is ``True`` then the current device ordinal
    will be added to the given filename automatically. This can be used in a
    1-process-per-gpu scenario to ensure all processes write to a separate file.
    """
    torch._C._cuda_tunableop_set_filename(filename, insert_device_ordinal)  # type: ignore[attr-defined]


def get_filename() -> str:
    r"""Get the results filename."""
    return torch._C._cuda_tunableop_get_filename()  # type: ignore[attr-defined]


def get_results() -> tuple[str, str, str, float]:
    r"""Return all TunableOp results."""
    return torch._C._cuda_tunableop_get_results()  # type: ignore[attr-defined]


def get_validators() -> tuple[str, str]:
    r"""Return the TunableOp validators."""
    return torch._C._cuda_tunableop_get_validators()  # type: ignore[attr-defined]


def read_file(filename: str | None = None) -> bool:
    r"""Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_read_file(filename)  # type: ignore[attr-defined]


def set_rotating_buffer_size(buffer_size: int) -> None:
    r"""Set rotating buffer size to this value in MB, if the buffer size is greater than zero.

    If less than zero, query L2 cache size. If equal to zero, means deactivate rotating buffer.
    """
    return torch._C._cuda_tunableop_set_rotating_buffer_size(buffer_size)  # type: ignore[attr-defined]


def get_rotating_buffer_size() -> int:
    r"""Get the rotating buffer size in kilobytes."""
    return torch._C._cuda_tunableop_get_rotating_buffer_size()  # type: ignore[attr-defined]


def set_numerical_check_tolerances(
    enable: bool, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    r"""Set the atol and rtol values in numeric check"""
    return torch._C._cuda_tunableop_set_numerical_check_tolerances(enable, atol, rtol)  # type: ignore[attr-defined]


def tune_gemm_in_file(filename: str) -> None:
    r"""tune GEMM in file."""

    if not is_enabled():
        raise AssertionError("TunableOp is not enabled")
    if not tuning_is_enabled():
        raise AssertionError("Tuning is not enabled")

    deviceid = torch.cuda.current_device()

    with open(filename) as file:
        for line in file:
            if line.startswith(("Gemm", "ScaledGemm")):
                _process_single_offline_gemm(line, deviceid)


def _gather_unique_untuned_gemm_from_files(filename_pattern: str) -> set[str]:
    r"""Process multiple untuned results file and return a set with duplicates removed."""
    unique_gemm_entries = set()  # set will avoid duplicates

    for file_path in glob.glob(filename_pattern):
        with open(file_path) as file:
            for line in file:
                if line.startswith(("Gemm", "ScaledGemm")):
                    unique_gemm_entries.add(line)

    return unique_gemm_entries


def _gather_tunableop_results() -> None:
    r"""Gather results from multiple tunableop results file and create a single file."""
    gemm_lines = set()
    validator_lines = []

    # Need to allow for the possibility that results filename was
    # set with the Python API instead of with environment variable.
    # Also possible that results filename was not set at all.
    # There are several test cases to check, but ultimately we
    # need a glob-able expression
    results_filename = get_filename()  # Note empty string could be returned here

    if (
        results_filename is not None and results_filename != ""
    ):  # Case were the Python API was used to set the filename
        dot_pos = results_filename.find(".")
        if dot_pos != -1 and dot_pos > 0:
            # Replace the character just to the left of the dot
            filename_pattern = (
                results_filename[: dot_pos - 1] + "?" + results_filename[dot_pos:]
            )
        else:
            filename_pattern = ""  # Needed to make linter happy
    else:  # Case where the environment variable was used to set the filename.
        results_filename_env = os.getenv("PYTORCH_TUNABLEOP_FILENAME")
        if results_filename_env is None or results_filename_env == "":
            filename_pattern = "tunableop_results?.csv"
        elif "%d" in results_filename_env:
            filename_pattern = results_filename_env.replace("%d", "?")
        else:
            filename_pattern = results_filename_env.replace(".", "?.")

    if "?" not in filename_pattern:
        raise AssertionError(
            f"filename_pattern must contain '?', got {filename_pattern!r}"
        )

    FirstFile = False
    matching_files = glob.glob(filename_pattern)
    num_matching_files = len(matching_files)
    for file_path in matching_files:
        with open(file_path) as file:
            for line in file:
                if line.startswith("Validator"):
                    if not (FirstFile):
                        # Only read Validator from first file
                        validator_lines.append(line)
                else:
                    gemm_lines.add(line)

        FirstFile = True

    output_file = filename_pattern.replace("?", "_full0")

    with open(output_file, "w") as out_file:
        for line in validator_lines:
            out_file.write(line)
        for line in gemm_lines:
            out_file.write(line)

    # Create num_matching_copies of the results file
    for i in range(1, num_matching_files):
        duplicate_file = output_file.replace("0", str(i))
        shutil.copy(output_file, duplicate_file)


def _create_matrices(
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    transA: bool,
    transB: bool,
    dtypeA: torch.dtype,
    deviceid: str,
    dtypeB: torch.dtype | None = None,
    randn: bool = True,
    subMatrix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Helper function for _process_single_offline_gemm.
    Creates matrices that are then consumed by one of the Torch GEMM APIs.
    """
    # Fill parameters set for use with ScaledGEMM
    fillA = 0.25
    fillB = 0.75

    if dtypeB is None:
        dtypeB = dtypeA

    if subMatrix:
        # User reference for understanding leading dimension:
        # https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/dgemm.f
        # TO DO: According to lines 108 - 133, there is no lower bound on rowsA,
        # but there is a restriction on rowsB. Using this formula for now as it
        # seems to work for all UTs.
        rowsA = rowsB = max(ldc, k)

        if randn:
            matA = torch.randn(rowsA, lda, dtype=dtypeA, device=deviceid)
            matB = torch.randn(rowsB, ldb, dtype=dtypeA, device=deviceid)
        else:
            matA = torch.full((rowsA, lda), fillA, dtype=dtypeB, device=deviceid)
            matB = torch.full((rowsB, ldb), fillB, dtype=dtypeB, device=deviceid)

        subA = matA[:k, :m].t() if transA else matA[:m, :k]
        subB = matB[:n, :k].t() if transB else matB[:k, :n]
        return subA, subB
    else:
        if randn:
            matA = (
                torch.rand(k, m, dtype=dtypeA, device=deviceid).t()
                if transA
                else torch.rand(m, k, dtype=dtypeA, device=deviceid)
            )
            matB = (
                torch.rand(n, k, dtype=dtypeB, device=deviceid).t()
                if transB
                else torch.rand(k, n, dtype=dtypeB, device=deviceid)
            )
        else:
            matA = (
                torch.full((k, m), fillA, dtype=dtypeA, device=deviceid).t()
                if transA
                else torch.full((m, k), fillA, dtype=dtypeA, device=deviceid)
            )
            matB = (
                torch.full((n, k), fillB, dtype=dtypeB, device=deviceid).t()
                if transB
                else torch.full((k, n), fillB, dtype=dtypeB, device=deviceid)
            )
        return matA, matB


def _create_batch_matrices(
    m: int,
    n: int,
    k: int,
    b: int,
    lda: int,
    ldb: int,
    ldc: int,
    transA: bool,
    transB: bool,
    dtype: torch.dtype,
    deviceid: str,
    subMatrix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Helper function for _process_single_offline_gemm.
    Creates batch matrices that are then consumed by one of the Torch GEMM APIs.
    Similar to _create_matrices but for 3D batch matrices.
    """
    if subMatrix:
        # User reference for understanding leading dimension:
        # https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/dgemm.f
        # TO DO: According to lines 108 - 133, there is no lower bound on rowsA,
        # but there is a restriction on rowsB. Using this formula for now as it
        # seems to work for all UTs.
        rowsA = rowsB = max(ldc, k)

        matA = torch.randn(b, rowsA, lda, dtype=dtype, device=deviceid)
        matB = torch.randn(b, rowsB, ldb, dtype=dtype, device=deviceid)

        subA = matA[:b, :k, :m].transpose(1, 2) if transA else matA[:b, :m, :k]
        subB = matB[:b, :n, :k].transpose(1, 2) if transB else matB[:b, :k, :n]
        return subA, subB
    else:
        matA = (
            torch.rand(b, k, m, dtype=dtype, device=deviceid)
            if transA
            else torch.rand(b, m, k, dtype=dtype, device=deviceid)
        )
        matB = (
            torch.rand(b, n, k, dtype=dtype, device=deviceid)
            if transB
            else torch.rand(b, k, n, dtype=dtype, device=deviceid)
        )
        matA = matA.transpose(1, 2) if transA else matA
        matB = matB.transpose(1, 2) if transB else matB
        return matA, matB


def _process_single_offline_gemm(untuned_gemm_line: str, gpu_id: int) -> None:
    r"""Process a single untuned GEMM."""

    deviceid = "cuda:" + str(gpu_id)
    torch.cuda.set_device(deviceid)

    dtype_dict = {
        "Float": torch.float32,
        "float": torch.float32,
        "tf32": torch.float32,
        "double": torch.float64,
        "BFloat16": torch.bfloat16,
        "Half": torch.half,
        "c10::complex<double>": torch.complex128,
        "c10::complex<float>": torch.complex64,
        "Float8_e4m3fn": torch.float8_e4m3fn,
        "Float8_e5m2": torch.float8_e5m2,
        "Float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "Float8_e5m2fnuz": torch.float8_e5m2fnuz,
    }

    untuned_gemm = untuned_gemm_line.strip().split(",")[:]

    underscore_count = untuned_gemm[0].count("_")

    # Initialize dtype to make linter happy
    dtype = None
    dtypeA = None
    dtypeB = None
    dtypeC = None

    # Extract BLAS parameters
    if underscore_count == 2:
        [op_sig, data_type, layout] = untuned_gemm[0].split("_")
        transB = layout[0] == "T"
        transA = layout[1] == "T"
        dtype = dtype_dict.get(data_type)
        if data_type == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False

    else:  # ScaledGEMM
        count = untuned_gemm[0].count("_")
        if count not in [6, 7]:
            raise AssertionError(f"count must be 6 or 7, got {count}")
        untuned_gemm_temp = untuned_gemm[0].split("_")
        # dtypeC = might not be FP8 type, keep track
        # of the number of underscores
        op_sig = untuned_gemm_temp[0]
        data_typeA = untuned_gemm_temp[1] + "_" + untuned_gemm_temp[2]
        data_typeB = untuned_gemm_temp[3] + "_" + untuned_gemm_temp[4]
        if count == 7:
            data_typeC = untuned_gemm_temp[5] + "_" + untuned_gemm_temp[6]
        else:
            data_typeC = untuned_gemm_temp[5]
        transB = untuned_gemm_temp[count][0] == "T"
        transA = untuned_gemm_temp[count][1] == "T"
        dtypeA = dtype_dict.get(data_typeA)
        dtypeB = dtype_dict.get(data_typeB)
        dtypeC = dtype_dict.get(data_typeC)

    untuned_gemm_temp = untuned_gemm[1].split("_")
    [n, m, k] = [int(g) for g in untuned_gemm_temp[1:4]]
    if op_sig == "GemmStridedBatchedTunableOp":
        if untuned_gemm_temp[6] != "ld":
            raise AssertionError(
                f"expected 'ld' at index 6, got {untuned_gemm_temp[6]!r}"
            )
        [ldb, lda, ldc] = [int(g) for g in untuned_gemm_temp[7:10]]
    else:
        if untuned_gemm_temp[4] != "ld":
            raise AssertionError(
                f"expected 'ld' at index 4, got {untuned_gemm_temp[4]!r}"
            )
        [ldb, lda, ldc] = [int(g) for g in untuned_gemm_temp[5:8]]

    # Detect subMatrix case
    # A GEMM is "tight" (not a sub-matrix) only when each leading dimension
    # equals its expected contiguous value for the given transpose layout.
    # Checking mere membership in {n, m, k} is wrong: a padded leading
    # dimension can coincidentally equal one of n/m/k (e.g. lda == n), which
    # silently rewrites the requested shape and tunes the wrong GEMM.
    # See https://github.com/ROCm/TheRock/issues/5553
    lda_tight = m if transA else k
    ldb_tight = k if transB else n
    ldc_tight = n
    if lda == lda_tight and ldb == ldb_tight and ldc == ldc_tight:
        subMatrix = False
    else:
        subMatrix = True

    if op_sig == "GemmTunableOp":
        # Warnings for unsupported cases:
        if m == 1 or n == 1 or k == 1:
            if (not transA) and (not transB):
                pass  # case is supported
            elif transA and n == 1:
                pass  # case is supported
            else:
                warnings.warn(
                    "Offline tuning is not supported for this GEMM. Use online tuning instead. "
                    + f"Skipped tuning for: {untuned_gemm[1]}",
                    stacklevel=2,
                )
                return

        # Resolve linter issue
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, but got {dtype}")

        matA, matB = _create_matrices(
            m, n, k, lda, ldb, ldc, transA, transB, dtype, deviceid, subMatrix=subMatrix
        )
        torch.mm(matA, matB)

    elif op_sig == "GemmStridedBatchedTunableOp":
        # Warnings for unsupported cases:
        if m == 1 or n == 1 or k == 1:
            warnings.warn(
                "Offline tuning is not supported for this GEMM. Use online tuning instead. "
                + f"Skipped tuning for: {untuned_gemm[1]}",
                stacklevel=2,
            )
            return

        [b] = [int(g) for g in untuned_gemm_temp[5:6]]

        # Resolve linter issue
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, but got {dtype}")

        matA, matB = _create_batch_matrices(
            m,
            n,
            k,
            b,
            lda,
            ldb,
            ldc,
            transA,
            transB,
            dtype,
            deviceid,
            subMatrix=subMatrix,
        )
        torch.bmm(matA, matB)
    elif op_sig == "ScaledGemmTunableOp":
        # Only combination supported by PyTorch
        if transB is not True:
            raise AssertionError(
                f"transB must be True for ScaledGemmTunableOp, got {transB}"
            )
        if transA is not False:
            raise AssertionError(
                f"transA must be False for ScaledGemmTunableOp, got {transA}"
            )

        # Resolve linter issue
        if dtypeA is None or not isinstance(dtypeA, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, but got {dtypeA}")

        matA, matB = _create_matrices(
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transA,
            transB,
            dtypeA,
            deviceid,
            dtypeB=dtypeB,
            randn=False,
            subMatrix=subMatrix,
        )

        if untuned_gemm_temp[8] != "rw":
            raise AssertionError(
                f"expected 'rw' at index 8, got {untuned_gemm_temp[8]!r}"
            )
        if untuned_gemm_temp[9] == "1":
            rowwise = True
        else:
            rowwise = False
        if rowwise:
            scaleA = (
                torch.ones((1, m), device=deviceid)
                if transA
                else torch.ones((m, 1), device=deviceid)
            )
            scaleB = (
                torch.ones((1, n), device=deviceid)
                if transB
                else torch.ones((n, 1), device=deviceid)
            )
        else:
            scaleA = torch.tensor(0.8, device=deviceid)
            scaleB = torch.tensor(0.9, device=deviceid)

        if untuned_gemm_temp[10] != "bias":
            raise AssertionError(
                f"expected 'bias' at index 10, got {untuned_gemm_temp[10]!r}"
            )
        if untuned_gemm_temp[11] == "None":  # no bias vector
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtypeC
            )
        else:  # bias vector present
            fillbias = 0.10
            bias_dtype = dtype_dict.get(untuned_gemm_temp[11])
            bias = (
                torch.full((n,), fillbias, dtype=bias_dtype, device=deviceid)
                if transB
                else torch.full((m,), fillbias, dtype=bias_dtype, device=deviceid)
            )
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtypeC, bias=bias
            )

    elif op_sig == "GemmAndBiasTunableOp":
        # y = x*A^T + b
        if transA == transB:
            raise AssertionError(
                f"transA and transB must differ for GemmAndBiasTunableOp, got transA={transA}, transB={transB}"
            )

        # Resolve linter issue
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, but got {dtype}")

        bias = torch.rand(n, dtype=dtype, device=deviceid)

        X, matA = _create_matrices(
            m, n, k, lda, ldb, ldc, transA, transB, dtype, deviceid, subMatrix=subMatrix
        )
        matA = matA.t()
        torch.nn.functional.linear(X, matA, bias)
    else:
        warnings.warn(f"error: unknown op {op_sig}", stacklevel=2)


def _process_offline_gemms(untuned_gemm_lines: list[str], gpu_id: int) -> None:
    r"""Process multiple untuned GEMMs on a single GPU."""
    _check_tuning_assertions()
    for line in untuned_gemm_lines:
        _process_single_offline_gemm(line, gpu_id)


def _check_tuning_assertions() -> None:
    r"""Helper function for multi-GPU tuning case. Need to check that TunableOp feature
    is enabled and that tuning is enabled.
    """

    if is_enabled() is False:
        warnings.warn("TunableOp was disabled. Trying to enable now.", stacklevel=2)
        enable(True)
    if is_enabled() is not True:
        raise AssertionError("is_enabled() must be True")
    if tuning_is_enabled() is not True:
        raise AssertionError("tuning_is_enabled() must be True")
    if record_untuned_is_enabled() is not False:
        raise AssertionError("record_untuned_is_enabled() must be False")


def mgpu_tune_gemm_in_file(filename_pattern: str, num_gpus: int) -> None:
    r"""Process one or more files and distribute work over one or more GPUs."""
    unique_gemm_entries = _gather_unique_untuned_gemm_from_files(filename_pattern)

    total_gpus = torch.cuda.device_count()

    if not (1 <= num_gpus <= total_gpus):
        raise AssertionError(
            f"num_gpus must be between 1 and {total_gpus}, got {num_gpus}"
        )

    mp_context = mp.get_context("spawn")

    gemm_entries_by_gpu: list[list[str]] = [[] for _ in range(num_gpus)]

    # GEMM are assigned to GPUs in a round robin manner
    for h, line in enumerate(unique_gemm_entries):
        gemm_entries_by_gpu[h % num_gpus].append(line)

    processes = []
    for h, entries in enumerate(gemm_entries_by_gpu):
        if not entries:
            continue
        # TunableOp initializes its output filename once per process, so keep
        # each spawned process bound to a single GPU.
        process = mp_context.Process(target=_process_offline_gemms, args=(entries, h))
        process.start()
        processes.append((h, process))

    failed_processes = []
    for h, process in processes:
        process.join()
        if process.exitcode != 0:
            failed_processes.append((h, process.exitcode))

    if failed_processes:
        raise RuntimeError(f"offline tuning processes failed: {failed_processes}")

    torch.cuda.synchronize()

    _gather_tunableop_results()
<|endoftext|># mypy: allow-untyped-defs
import base64
import io
import json
import operator
import os
import pickle
import subprocess
import sys
import warnings
from functools import lru_cache
from itertools import groupby
from typing import Any


cache = lru_cache(None)

__all__ = ["format_flamegraph", "segments", "memory", "compare"]


def _frame_fmt(f, full_filename=False):
    i = f["line"]
    fname = f["filename"]
    if not full_filename:
        fname = fname.split("/")[-1]
    func = f["name"]
    return f"{fname}:{i}:{func}"


@cache
def _frame_filter(name, filename):
    omit_functions = [
        "unwind::unwind",
        "CapturedTraceback::gather",
        "gather_with_cpp",
        "_start",
        "__libc_start_main",
        "PyEval_",
        "PyObject_",
        "PyFunction_",
    ]
    omit_filenames = [
        "core/boxing",
        "/Register",
        "/Redispatch",
        "pythonrun.c",
        "Modules/main.c",
        "Objects/call.c",
        "Objects/methodobject.c",
        "pycore_ceval.h",
        "ceval.c",
        "cpython/abstract.h",
    ]
    for of in omit_functions:
        if of in name:
            return False
    for of in omit_filenames:
        if of in filename:
            return False
    return True


def _frames_fmt(frames, full_filename=False, reverse=False):
    if reverse:
        frames = reversed(frames)
    return [
        _frame_fmt(f, full_filename)
        for f in frames
        if _frame_filter(f["name"], f["filename"])
    ]


def _block_extra_legacy(b):
    if "history" in b:
        frames = b["history"][0].get("frames", [])
        real_size = b["history"][0]["real_size"]
    else:
        real_size = b.get("requested_size", b["size"])
        frames = []
    return frames, real_size


def _block_extra(b):
    if "frames" not in b:
        # old snapshot format made it more complicated to get frames/allocated size
        return _block_extra_legacy(b)
    return b["frames"], b["requested_size"]


def format_flamegraph(flamegraph_lines, flamegraph_script=None):
    if flamegraph_script is None:
        cache_dir = os.path.expanduser("~/.cache/")
        os.makedirs(cache_dir, exist_ok=True)
        flamegraph_script = f"{cache_dir}/flamegraph.pl"
    if not os.path.exists(flamegraph_script):
        import tempfile
        import urllib.request

        print(f"Downloading flamegraph.pl to: {flamegraph_script}")
        # Download to a temp file next to the target, then atomically move it
        # into place so a concurrent reader never sees a partial file. The temp
        # file lives in the same directory (same filesystem) so os.replace is
        # atomic, and is created with delete=False + manual cleanup: moving a
        # delete=True NamedTemporaryFile out from under its context manager makes
        # __exit__ fail to unlink the now-missing file.
        fd, tmp_name = tempfile.mkstemp(
            suffix=".pl", dir=os.path.dirname(flamegraph_script) or None
        )
        os.close(fd)
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl",
                tmp_name,
            )
            os.chmod(tmp_name, 0o755)
            os.replace(tmp_name, flamegraph_script)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    args = [flamegraph_script, "--countname", "bytes"]
    with subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8"
    ) as p:
        if p.stdin is None:
            raise AssertionError("p.stdin is None")
        if p.stdout is None:
            raise AssertionError("p.stdout is None")
        p.stdin.write(flamegraph_lines)
        p.stdin.close()
        result = p.stdout.read()
        p.stdout.close()
        p.wait()
        if p.wait() != 0:
            raise AssertionError(f"flamegraph process exited with code {p.wait()}")
        return result


def _write_blocks(f, prefix, blocks):
    def frames_fragment(frames):
        if not frames:
            return "<non-python>"
        return ";".join(_frames_fmt(frames, reverse=True))

    for b in blocks:
        if "history" not in b:
            frames, accounted_for_size = _block_extra(b)
            f.write(
                f"{prefix};{b['state']};{frames_fragment(frames)} {accounted_for_size}\n"
            )
        else:
            accounted_for_size = 0
            for h in b["history"]:
                sz = h["real_size"]
                accounted_for_size += sz
                if "frames" in h:
                    frames = h["frames"]
                    f.write(f"{prefix};{b['state']};{frames_fragment(frames)} {sz}\n")
                else:
                    f.write(f"{prefix};{b['state']};<no-context> {sz}\n")
        gaps = b["size"] - accounted_for_size
        if gaps:
            f.write(f"{prefix};{b['state']};<gaps> {gaps}\n")


def segments(snapshot, format_flamegraph=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot["segments"]:
        prefix = f"stream_{seg['stream']};seg_{seg['address']}"
        _write_blocks(f, prefix, seg["blocks"])
    return format_flamegraph(f.getvalue())


def memory(snapshot, format_flamegraph=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot["segments"]:
        prefix = f"stream_{seg['stream']}"
        _write_blocks(f, prefix, seg["blocks"])
    return format_flamegraph(f.getvalue())


def compare(before, after, format_flamegraph=format_flamegraph):
    def _seg_key(seg):
        return (seg["address"], seg["total_size"])

    def _seg_info(seg):
        return f"stream_{seg['stream']};seg_{seg['address']}"

    f = io.StringIO()

    before_segs = {_seg_key(seg) for seg in before}
    after_segs = {_seg_key(seg) for seg in after}

    print(f"only_before = {[a for a, _ in (before_segs - after_segs)]}")
    print(f"only_after = {[a for a, _ in (after_segs - before_segs)]}")

    for seg in before:
        if _seg_key(seg) not in after_segs:
            _write_blocks(f, f"only_before;{_seg_info(seg)}", seg["blocks"])

    for seg in after:
        if _seg_key(seg) not in before_segs:
            _write_blocks(f, f"only_after;{_seg_info(seg)}", seg["blocks"])

    return format_flamegraph(f.getvalue())


def _format_size(num):
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}YiB"


class Bytes:
    def __init__(self, value):
        self.value = value

    def __add__(self, rhs):
        return Bytes(self.value + rhs)

    def __repr__(self):
        return _format_size(self.value)


def calc_active(seg):
    return sum(b["size"] for b in seg["blocks"] if b["state"] == "active_allocated")


def _report_free(free_external, free_internal):
    total = free_external + free_internal
    suffix = ""
    if total != 0:
        pct = (free_internal / total) * 100
        suffix = f" ({pct:.1f}% internal)"
    return f"{Bytes(total)}{suffix}"


PAGE_SIZE = 1024 * 1024 * 20
legend = f"""\

Legend:
    [a     ] - a segment in the allocator
     ^-- a page {Bytes(PAGE_SIZE)} of memory in the segment
    a-z: pages filled with a single block's content
    ' ': page is completely free
    *: page if completely full with multiple blocks
    0-9: page is partially full with tensors of multiple blocks (9 == 90% full)
    (X% internal) - of the free memory, X% is free because we rounded the size of the allocation.
"""


def segsum(data):
    r"""Visually reports how the allocator has filled its segments.

    This printout can help debug fragmentation issues since free fragments
    will appear as gaps in this printout.  The amount of free space is reported
    for each segment.
    We distinguish between internal free memory which occurs because the
    allocator rounds the allocation size, and external free memory, which are
    the gaps between allocations in a segment.
    Args:
        data: snapshot dictionary created from _snapshot()
    """
    out = io.StringIO()
    out.write(f"Summary of segments >= {Bytes(PAGE_SIZE)} in size\n")
    total_reserved = 0
    total_allocated = 0
    free_external = 0
    free_internal = 0
    for seg in sorted(
        data["segments"], key=lambda x: (x["total_size"], calc_active(x))
    ):
        total_reserved += seg["total_size"]

        seg_free_external = 0
        seg_free_internal = 0
        seg_allocated = 0
        all_ranges = []
        boffset = 0
        for b in seg["blocks"]:
            active = b["state"] == "active_allocated"
            if active:
                _, allocated_size = _block_extra(b)
                all_ranges.append((boffset, allocated_size, True))
                seg_allocated += allocated_size
                seg_free_internal += b["size"] - allocated_size
            else:
                seg_free_external += b["size"]

            boffset += b["size"]

        total_allocated += seg_allocated
        free_external += seg_free_external
        free_internal += seg_free_internal

        nseg = (seg["total_size"] - 1) // PAGE_SIZE + 1
        occupied = [" " for _ in range(nseg)]
        frac = [0.0 for _ in range(nseg)]
        active_size = 0
        for i, (start_, size, active) in enumerate(all_ranges):
            active_size += size
            finish_ = start_ + size
            start = start_ // PAGE_SIZE
            finish = (finish_ - 1) // PAGE_SIZE + 1
            m = chr(ord("a" if active else "A") + (i % 26))
            for j in range(start, finish):
                s = max(start_, j * PAGE_SIZE)
                e = min(finish_, (j + 1) * PAGE_SIZE)
                frac[j] += (e - s) / PAGE_SIZE
                if occupied[j] != " ":
                    occupied[j] = "0123456789*"[int(frac[j] * 10)]
                else:
                    occupied[j] = m
        stream = "" if seg["stream"] == 0 else f", stream_{seg['stream']}"
        body = "".join(occupied)
        if seg_free_external + seg_free_internal + seg_allocated != seg["total_size"]:
            raise AssertionError(
                f"Segment size mismatch: {seg_free_external} + {seg_free_internal} + {seg_allocated} != {seg['total_size']}"
            )
        stream = f" stream_{seg['stream']}" if seg["stream"] != 0 else ""
        if seg["total_size"] >= PAGE_SIZE:
            out.write(
                f"[{body}] {Bytes(seg['total_size'])} allocated, "
                f"{_report_free(seg_free_external, seg_free_internal)} free{stream}\n"
            )
    out.write(f"segments: {len(data['segments'])}\n")
    out.write(f"total_reserved: {Bytes(total_reserved)}\n")
    out.write(f"total_allocated: {Bytes(total_allocated)}\n")
    out.write(f"total_free: {_report_free(free_external, free_internal)}\n")
    out.write(legend)
    if free_internal + free_external + total_allocated != total_reserved:
        raise AssertionError(
            f"Memory accounting error: {free_internal} + {free_external} + {total_allocated} != {total_reserved}"
        )
    return out.getvalue()


def trace(data):
    out = io.StringIO()

    def format(entries):
        segment_intervals: list = []
        segment_addr_to_name = {}
        allocation_addr_to_name = {}

        free_names: list = []
        next_name = 0

        def _name():
            nonlocal next_name
            if free_names:
                return free_names.pop()
            r, m = next_name // 26, next_name % 26
            next_name += 1
            return f"{chr(ord('a') + m)}{'' if r == 0 else r}"

        def find_segment(addr):
            for name, saddr, size in segment_intervals:
                if addr >= saddr and addr < saddr + size:
                    return name, saddr
            for i, seg in enumerate(data["segments"]):
                saddr = seg["address"]
                size = seg["allocated_size"]
                if addr >= saddr and addr < saddr + size:
                    return f"seg_{i}", saddr
            return None, None

        count = 0
        out.write(f"{len(entries)} entries\n")

        total_reserved = 0
        for seg in data["segments"]:
            total_reserved += seg["total_size"]

        for count, e in enumerate(entries):
            if e["action"] == "alloc":
                addr, size = e["addr"], e["size"]
                n = _name()
                seg_name, seg_addr = find_segment(addr)
                if seg_name is None:
                    seg_name = "MEM"
                    offset = addr
                else:
                    offset = addr - seg_addr
                out.write(f"{n} = {seg_name}[{offset}:{Bytes(size)}]\n")
                allocation_addr_to_name[addr] = (n, size, count)
                count += size
            elif e["action"] == "free_requested":
                addr, size = e["addr"], e["size"]
                name, _, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f"del {name} # {Bytes(size)}\n")
            elif e["action"] == "free_completed":
                addr, size = e["addr"], e["size"]
                count -= size
                name, _, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f"# free completed for {name} {Bytes(size)}\n")
                if name in allocation_addr_to_name:
                    free_names.append(name)
                    del allocation_addr_to_name[name]
            elif e["action"] == "segment_alloc":
                addr, size = e["addr"], e["size"]
                name = _name()
                out.write(f"{name} = cudaMalloc({addr}, {Bytes(size)})\n")
                segment_intervals.append((name, addr, size))
                segment_addr_to_name[addr] = name
            elif e["action"] == "segment_free":
                addr, size = e["addr"], e["size"]
                name = segment_addr_to_name.get(addr, addr)
                out.write(f"cudaFree({name}) # {Bytes(size)}\n")
                if name in segment_addr_to_name:
                    free_names.append(name)
                    del segment_addr_to_name[name]
            elif e["action"] == "oom":
                size = e["size"]
                free = e["device_free"]
                out.write(
                    f"raise OutOfMemoryError # {Bytes(size)} requested, {Bytes(free)} free in CUDA\n"
                )
            elif e["action"] == "annotate":
                addr = e["addr"]
                name, _, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f"# annotate {name}: {e['user_metadata']}\n")
            else:
                out.write(f"{e}\n")
        out.write(f"TOTAL MEM: {Bytes(count)}")

    for i, d in enumerate(data["device_traces"]):
        if d:
            out.write(f"Device {i} ----------------\n")
            format(d)
    return out.getvalue()


_memory_viz_template = r"""
<!DOCTYPE html>
<html>
<head>
</head>
<body>
<script type="module">
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
const local_files = $SNAPSHOT
add_local_files(local_files, $VIZ_KIND)
</script>
</body>
"""


def _format_viz(data, viz_kind, device):
    if device is not None:
        warnings.warn(
            "device argument is deprecated, plots now contain all device",
            FutureWarning,
            stacklevel=3,
        )
    buffer = pickle.dumps(data)
    buffer += b"\x00" * (3 - len(buffer) % 3)
    # Encode the buffer with base64
    encoded_buffer = base64.b64encode(buffer).decode("utf-8")

    json_format = json.dumps([{"name": "snapshot.pickle", "base64": encoded_buffer}])
    return _memory_viz_template.replace("$VIZ_KIND", repr(viz_kind)).replace(
        "$SNAPSHOT", json_format
    )


def filter_alloc_free_pairs(data):
    for dev_id in range(len(data["device_traces"])):
        # set of indexes of trace events for alloc-free pairs
        filterSet = set()
        # map from addr to index of alloc event
        allocMap = {}
        # set of addrs from free_requested events
        freeRequested = set()
        for idx, event in enumerate(data["device_traces"][dev_id]):
            if event["action"] == "alloc":
                allocMap[event["addr"]] = idx
            elif event["action"] == "free_requested":
                freeRequested.add(event["addr"])
                if allocMap.get(event["addr"]) is not None:
                    filterSet.add(idx)
                    filterSet.add(allocMap[event["addr"]])
                    allocMap.pop(event["addr"])
            elif event["action"] == "free_completed":
                if event["addr"] in freeRequested:
                    freeRequested.remove(event["addr"])
                    filterSet.add(idx)
                else:
                    print(f"free_completed without free_requested: {event}")

        # Remove events whose index is in filterSet
        if filterSet:
            # Create a new list excluding events with indices in filterSet
            data["device_traces"][dev_id] = [
                event
                for idx, event in enumerate(data["device_traces"][dev_id])
                if idx not in filterSet
            ]

    return data


def trace_plot(data, device=None, plot_segments=False, filter_freed=False):
    """Generate a visualization over time of the memory usage recorded by the trace as an html file.

    Args:
        data: Memory snapshot as generated from torch.cuda.memory._snapshot()
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.
        plot_segments (bool, optional): Plots memory returned from cudaMalloc, rather than individual allocations.
                                        Defaults to False.
        filter_freed (bool, optional): Filter out alloc-free paired events to only plot allocations that are not freed yet.
                                        Defaults to False to plot all trace events.

    Returns:
        str: HTML of visualization
    """
    if filter_freed:
        data = filter_alloc_free_pairs(data)

    return _format_viz(
        data,
        "Active Memory Timeline"
        if not plot_segments
        else "Active Cached Memory Timeline",
        device,
    )


def _profile_to_snapshot(profile):
    import torch
    from torch._C._profiler import _EventType
    from torch.profiler._memory_profiler import Action, TensorKey

    memory_profile = profile._memory_profile()

    allocation_stacks = {}
    for event in memory_profile._op_tree.sorted_nodes:
        if event.tag == _EventType.Allocation:
            parent = event.parent
            python_parents = []
            while parent:
                if parent.tag in (_EventType.PyCall, _EventType.PyCCall):
                    python_parents.append(parent)
                parent = parent.parent
            key = TensorKey.from_allocation(event.extra_fields)

            # Corner case: If allocation doesn't have an ID (can't prove it was used as a Tensor)
            #              key will be None. I should add some way to identify these, I just haven't yet.
            if key and event.extra_fields.alloc_size > 0:
                allocation_stacks[key] = python_parents

    device_count = torch.cuda.device_count()
    snapshot: dict[str, list[Any]] = {
        "device_traces": [[] for _ in range(device_count + 1)],
        "segments": [
            {
                "device": device,
                "address": None,
                "total_size": 0,
                "stream": 0,
                "blocks": [],
            }
            for device in range(device_count + 1)
        ],
    }

    def to_device(device):
        if device.type == "cuda":
            return device.index
        else:
            return device_count

    def allocate(size, tensor_key, version, during_trace=True):
        device = to_device(tensor_key.device)
        addr = tensor_key.storage.ptr

        seg = snapshot["segments"][device]  # type: ignore[index]
        if seg["address"] is None or seg["address"] > addr:
            seg["address"] = addr
        seg["total_size"] = max(
            seg["total_size"], addr + size
        )  # record max addr for now, we will make it the size later
        category = memory_profile._categories.get(tensor_key, version)
        category = category.name.lower() if category is not None else "unknown"
        stack = allocation_stacks.get(tensor_key, ())
        stack = [{"filename": "none", "line": 0, "name": p.name} for p in stack]
        r = {
            "action": "alloc",
            "addr": addr,
            "size": size,
            "stream": 0,
            "frames": stack,
            "category": category,
        }
        if during_trace:
            snapshot["device_traces"][device].append(r)
        return r

    def free(alloc, device):
        for e in ("free_requested", "free_completed"):
            snapshot["device_traces"][device].append(
                {
                    "action": e,
                    "addr": alloc["addr"],
                    "size": alloc["size"],
                    "stream": 0,
                    "frames": alloc["frames"],
                }
            )

    kv_to_elem = {}

    # create the device trace
    for _time, action, (tensor_key, version), size in memory_profile.timeline:
        if not isinstance(tensor_key, TensorKey):
            continue
        if action == Action.CREATE:
            kv_to_elem[(tensor_key, version)] = allocate(size, tensor_key, version)
        elif action == Action.DESTROY:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
        elif action == Action.INCREMENT_VERSION:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
            kv_to_elem[(tensor_key, version + 1)] = allocate(
                size, tensor_key, version + 1
            )
        elif action == Action.PREEXISTING:
            kv_to_elem[(tensor_key, version)] = allocate(
                size, tensor_key, version, during_trace=False
            )

    # create the final snapshot state
    blocks_at_end = [
        (to_device(tensor_key.device), event["addr"], event["size"], event["frames"])
        for (tensor_key, version), event in kv_to_elem.items()
    ]
    for device, blocks in groupby(sorted(blocks_at_end), key=operator.itemgetter(0)):
        seg = snapshot["segments"][device]  # type: ignore[index]
        last_addr = seg["address"]
        for _, addr, size, frames in blocks:
            if last_addr < addr:
                seg["blocks"].append({"size": addr - last_addr, "state": "inactive"})
            seg["blocks"].append(
                {
                    "size": size,
                    "state": "active_allocated",
                    "requested_size": size,
                    "frames": frames,
                }
            )
            last_addr = addr + size
        if last_addr < seg["total_size"]:
            seg["blocks"].append(
                {"size": seg["total_size"] - last_addr, "state": "inactive"}
            )

    snapshot["segments"] = [seg for seg in snapshot["segments"] if seg["blocks"]]  # type: ignore[attr-defined]
    for seg in snapshot["segments"]:  # type: ignore[attr-defined, name-defined, no-redef]
        seg["total_size"] -= seg["address"]
        if not seg["blocks"]:
            seg["blocks"].append({"size": seg["total_size"], "state": "inactive"})

    return snapshot


def profile_plot(profile, device=None):
    """Generate a visualization over time of the memory usage recorded by kineto memory profiling as an html file.

    Args:
        profile: profile as generated by `torch.profiler.profile(profile_memory=True)`
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.

    Returns:
        str: HTML of visualization
    """
    snapshot = _profile_to_snapshot(profile)
    return _format_viz(snapshot, "Active Memory Timeline", device)


def segment_plot(data: Any, device=None):
    return _format_viz(data, "Allocator State History", device)


if __name__ == "__main__":
    import os.path

    thedir = os.path.realpath(os.path.dirname(__file__))
    if thedir in sys.path:
        # otherwise we find cuda/random.py as random...
        sys.path.remove(thedir)
    import argparse

    fn_name = "torch.cuda.memory._snapshot()"
    pickled = f"pickled memory statistics from {fn_name}"
    parser = argparse.ArgumentParser(
        description=f"Visualize memory dumps produced by {fn_name}"
    )

    subparsers = parser.add_subparsers(dest="action")

    def _output(p):
        p.add_argument(
            "-o",
            "--output",
            default="output.svg",
            help="flamegraph svg (default: output.svg)",
        )

    description = "Prints overall allocation statistics and a visualization of how the allocators segments are currently filled."
    stats_a = subparsers.add_parser("stats", description=description)
    stats_a.add_argument("input", help=pickled)

    description = "Prints buffer of the most recent allocation events embedded in the snapshot in a Pythonic style."
    trace_a = subparsers.add_parser("trace", description=description)
    trace_a.add_argument("input", help=pickled)

    description = "Generate a flamegraph that visualizes what memory is stored in each allocator segment (aka block)"
    segments_a = subparsers.add_parser("segments", description=description)
    segments_a.add_argument("input", help=pickled)
    _output(segments_a)

    description = (
        "Generate a flamegraph the program locations contributing to CUDA memory usage."
    )
    memory_a = subparsers.add_parser("memory", description=description)
    memory_a.add_argument("input", help=pickled)
    _output(memory_a)

    description = (
        "Generate a flamegraph that shows segments (aka blocks) that have been added "
        "or removed between two different memory snapshots."
    )
    compare_a = subparsers.add_parser("compare", description=description)
    compare_a.add_argument("before", help=pickled)
    compare_a.add_argument("after", help=pickled)
    _output(compare_a)

    plots = (
        (
            "trace_plot",
            "Generate a visualization over time of the memory usage recorded by the trace as an html file.",
        ),
        (
            "segment_plot",
            "Visualize how allocations are packed into allocator segments at each point in a trace as an html file.",
        ),
    )
    for cmd, description in plots:
        trace_plot_a = subparsers.add_parser(cmd, description=description)
        trace_plot_a.add_argument("input", help=pickled)
        help = "visualize trace from this device (default: chooses the only device with trace info or errors)"
        trace_plot_a.add_argument("-d", "--device", type=int, default=None, help=help)
        help = "path to save the visualization(default: output.html)"
        trace_plot_a.add_argument("-o", "--output", default="output.html", help=help)
        if cmd == "trace_plot":
            help = "visualize change to segments rather than individual allocations"
            trace_plot_a.add_argument(
                "-s", "--segments", action="store_true", help=help
            )

            help = (
                "filter out allocation-free pairs to only visualize the allocations that are not freed yet;"
                "useful to reduce the number of events for large traces for debugging OOM"
            )
            trace_plot_a.add_argument(
                "-f", "--filter_freed", action="store_true", help=help
            )

    args = parser.parse_args()

    def _read(name):
        if name == "-":
            data = pickle.load(sys.stdin.buffer)
        else:
            with open(name, "rb") as f:
                data = pickle.load(f)
        if isinstance(data, list):  # segments only...
            data = {"segments": data, "traces": []}
        return data

    def _write(name, data):
        with open(name, "w") as f:
            f.write(data)

    if args.action == "segments":
        data = _read(args.input)
        _write(args.output, segments(data))
    elif args.action == "memory":
        data = _read(args.input)
        _write(args.output, memory(data))
    elif args.action == "stats":
        data = _read(args.input)
        print(segsum(data))
    elif args.action == "trace":
        data = _read(args.input)
        print(trace(data))
    elif args.action == "compare":
        before = _read(args.before)
        after = _read(args.after)
        _write(args.output, compare(before, after))
    elif args.action == "trace_plot":
        data = _read(args.input)
        _write(
            args.output,
            trace_plot(
                data,
                device=args.device,
                plot_segments=args.segments,
                filter_freed=args.filter_freed,
            ),
        )
    elif args.action == "segment_plot":
        data = _read(args.input)
        _write(args.output, segment_plot(data, device=args.device))
<|endoftext|># pylint: disable=useless-parent-delegation
from __future__ import annotations

import ctypes
import gc
import typing
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable
from typing import overload, TYPE_CHECKING, TypeAlias, TypeGuard, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch import Tensor
from torch.cuda._utils import _check_cuda_bindings


try:
    from cuda.bindings import (  # pyrefly: ignore[missing-import]
        driver as _cuda_driver,
        runtime as _cuda_runtime,
    )
except ImportError:
    _cuda_driver = None  # type: ignore[assignment]
    _cuda_runtime = None  # type: ignore[assignment]


if TYPE_CHECKING:
    # importing _POOL_HANDLE at runtime toplevel causes an import cycle
    from torch.cuda import _POOL_HANDLE, MemPool
    from torch.utils._cuda_debug import _CUDAGraphInputLivenessTracker
    from torch.utils.hooks import RemovableHandle

    _GraphPool: TypeAlias = _POOL_HANDLE | MemPool
else:
    _GraphPool: TypeAlias = typing.Any

from .._utils import _dummy_type


__all__ = [
    "is_current_stream_capturing",
    "graph_pool_handle",
    "CUDAGraph",
    "graph",
    "make_graphed_callables",
    "export_dot",
    "export_graph_data",
    "register_graph_capture_start_hook",
    "register_graph_capture_end_hook",
    "register_graph_instantiate_hook",
    "register_graph_replay_start_hook",
    "register_graph_replay_end_hook",
    "register_graph_destroy_hook",
]


_R = TypeVar("_R")
_P = ParamSpec("_P")


def _is_mem_pool(pool: object) -> TypeGuard[MemPool]:
    mempool_cls = getattr(torch.cuda, "MemPool", None)
    return mempool_cls is not None and isinstance(pool, mempool_cls)


def _get_pool_id(pool: _GraphPool | None) -> _POOL_HANDLE | None:
    if pool is None:
        return None
    if _is_mem_pool(pool):
        return typing.cast("_POOL_HANDLE", pool.id)
    return typing.cast("_POOL_HANDLE", pool)


if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_CUDAGraph"] = _dummy_type("_CUDAGraph")
    torch._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch._C.__dict__["_cuda_isCurrentStreamCapturing"] = _dummy_type(
        "_cuda_isCurrentStreamCapturing"
    )

from torch._C import _cuda_isCurrentStreamCapturing, _CUDAGraph, _graph_pool_handle


def is_current_stream_capturing() -> bool:
    r"""Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise.

    If a CUDA context does not exist on the current device, returns False without initializing the context.
    """
    return _cuda_isCurrentStreamCapturing()


# Python shim helps Sphinx process docstrings more reliably.
def graph_pool_handle() -> _POOL_HANDLE:
    r"""Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return torch.cuda._POOL_HANDLE(_graph_pool_handle())


def _require_cuda_bindings() -> None:
    """Raise a uniform error if the cuda-bindings package is unavailable.

    ``_cuda_runtime`` and ``_cuda_driver`` are imported together, so checking one
    suffices.
    """
    if _cuda_runtime is None:
        raise RuntimeError(
            "This CUDAGraph API requires the cuda-bindings package; "
            "install it with `pip install cuda-bindings`."
        )


class _RetainedCallbacks:
    r"""Holds destroy callbacks and retained objects for a single capture cycle
    of a :class:`CUDAGraph`.

    A :class:`CUDAGraph` owns one of these and arms a :func:`weakref.finalize`
    bound to :meth:`fire`. The holder deliberately does NOT reference the graph,
    so the finalizer cannot keep the graph alive; :meth:`fire` can still sync the
    replay streams because they are mirrored onto the holder, not read off the
    graph. Callbacks and objects live in ``OrderedDict``s keyed by
    ``RemovableHandle`` id so registrations can be individually removed.
    """

    def __init__(self) -> None:
        self.callbacks: OrderedDict[int, Callable[[], None]] = OrderedDict()
        self.objects: OrderedDict[int, object] = OrderedDict()
        # Streams the graph was replayed on; drained before firing. A set (not a
        # single stream) because concurrent replays on multiple streams leave
        # work in flight on all of them and we cannot know which finishes last.
        # Stream hashes/compares by value, so same-stream replays dedup to one
        # entry -- the common case. Only populated when a caller requested sync.
        self.replay_streams: set[torch.cuda.Stream] = set()
        self.sync_before_fire: bool = False
        self._fired: bool = False

    def fire(self) -> None:
        # Fired at most once (guards the reset()-then-destroy double trigger).
        if self._fired:
            return
        self._fired = True
        if self.sync_before_fire:
            # Drain in-flight replays before releasing anything the graph may
            # reference. Guarded so a sync failure does not skip the callbacks.
            for stream in self.replay_streams:
                try:
                    stream.synchronize()
                except Exception:
                    pass
        for cb in list(self.callbacks.values()):
            try:
                cb()
            except Exception:
                pass  # match finalizer semantics: swallow, don't abort the rest
        self.callbacks.clear()
        self.objects.clear()


# Global CUDA-graph lifecycle hooks (the module-level counterpart of the per-graph
# _capture_start_hooks / _capture_end_hooks / _post_instantiate_hooks / _replay_*_hooks). A
# consumer (e.g. a profiler observer) registers a hook that fires for every graph, so the graph
# code stays free of consumer knowledge and a consumer sees captures of graphs it did not build
# (an inductor or NCCL capture); registering a hook is the opt-in. Keyed by RemovableHandle id.
# The fan-out (_run_global_hooks) swallows per-hook errors so one consumer cannot break a graph
# lifecycle step for another -- unlike the per-graph hooks, whose errors propagate to the owner.
_global_capture_start_hooks: OrderedDict[int, Callable[[CUDAGraph], None]] = (
    OrderedDict()
)
_global_capture_end_hooks: OrderedDict[int, Callable[[CUDAGraph], None]] = OrderedDict()
_global_instantiate_hooks: OrderedDict[int, Callable[[CUDAGraph], None]] = OrderedDict()
_global_replay_start_hooks: OrderedDict[int, Callable[[CUDAGraph], None]] = (
    OrderedDict()
)
_global_replay_end_hooks: OrderedDict[int, Callable[[CUDAGraph], None]] = OrderedDict()


def _register_global_hook(
    registry: OrderedDict[int, Callable[[CUDAGraph], None]],
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    from torch.utils.hooks import RemovableHandle

    handle = RemovableHandle(registry)
    registry[handle.id] = fn
    return handle


def _run_global_hooks(
    registry: OrderedDict[int, Callable[[CUDAGraph], None]],
    torch_cuda_graph: CUDAGraph,
) -> None:
    for fn in list(registry.values()):
        try:
            fn(torch_cuda_graph)
        except Exception:
            pass


def register_graph_capture_start_hook(
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    """Register a hook run with each CUDA graph as its capture begins. Returns a
    RemovableHandle; call ``.remove()`` to unregister.

    .. warning::
        The hook runs with capture already live on the current stream, so it must not issue
        CUDA work: anything it launches is captured into the graph, and under the default
        ``"global"`` capture error mode an unsafe call (e.g. an allocation) raises. Querying
        capture state is fine. Do preparation that needs CUDA before the capture instead.
    """
    return _register_global_hook(_global_capture_start_hooks, fn)


def register_graph_capture_end_hook(
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    """Register a hook run with each CUDA graph when its capture ends, while the captured
    ``cudaGraph_t`` is still live (see :meth:`CUDAGraph.register_capture_end_hook`). Returns a
    RemovableHandle; call ``.remove()`` to unregister."""
    return _register_global_hook(_global_capture_end_hooks, fn)


def register_graph_instantiate_hook(
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    """Register a hook run with each CUDA graph right after it is instantiated. Returns a
    RemovableHandle; call ``.remove()`` to unregister."""
    return _register_global_hook(_global_instantiate_hooks, fn)


def register_graph_replay_start_hook(
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    """Register a hook run with each CUDA graph at the start of every replay, just before it
    is launched. Returns a RemovableHandle; call ``.remove()`` to unregister.

    .. note::
        Replay is the hot path and this fires for EVERY graph on EVERY replay -- keep it
        cheap. With nothing registered the cost is a single dict emptiness check.
    """
    return _register_global_hook(_global_replay_start_hooks, fn)


def register_graph_replay_end_hook(
    fn: Callable[[CUDAGraph], None],
) -> RemovableHandle:
    """Register a hook run with each CUDA graph at the end of every replay, once the replay is
    *enqueued* (the launch is asynchronous, so the GPU work has not completed). Fires even if
    the launch raised, so a start hook is always balanced by an end. Returns a RemovableHandle;
    call ``.remove()`` to unregister. See the hot-path note on
    :func:`register_graph_replay_start_hook`."""
    return _register_global_hook(_global_replay_end_hooks, fn)


# Graph-destroy hooks, each handed the destroyed graph's exec ids (tools_id >> 32) so a
# consumer can purge its per-graph state. A CUDAGraph arms a single destroy callback (only
# while a hook is registered, see _graph_destroy_hooks_active) that fans out here.
_global_destroy_hooks: OrderedDict[int, Callable[[set[int]], None]] = OrderedDict()


def register_graph_destroy_hook(fn: Callable[[set[int]], None]) -> RemovableHandle:
    """Register ``fn(exec_ids)`` to run when a CUDA graph is destroyed. Returns a handle whose
    ``remove()`` unregisters it."""
    from torch.utils.hooks import RemovableHandle

    handle = RemovableHandle(_global_destroy_hooks)
    _global_destroy_hooks[handle.id] = fn
    return handle


def _graph_destroy_hooks_active() -> bool:
    """True when any graph-destroy hook is registered -- the gate a CUDAGraph checks before
    arming its destroy callback."""
    return bool(_global_destroy_hooks)


def _run_graph_destroy_hooks(exec_graph_ids: set[int]) -> None:
    """Invoke every registered hook with the destroyed exec graph ids, swallowing per-hook
    errors so one failure does not abort the rest (matching the destroy-callback fire
    semantics). The single entry point a graph's destroy callback calls."""
    for fn in list(_global_destroy_hooks.values()):
        try:
            fn(exec_graph_ids)
        except Exception:
            pass


# Python shim helps Sphinx process docstrings more reliably.
class CUDAGraph(_CUDAGraph):
    r"""Wrapper around a CUDA graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            cudaGraphExec_t will be instantiated on GPU at the end of
            ``capture_end`` and the underlying cudaGraph_t will be
            destroyed. Users who want to query or otherwise modify the
            underlying cudaGraph_t before instantiation can set
            ``keep_graph=True`` and access it via ``raw_cuda_graph`` after
            ``capture_end``. Note that the cudaGraphExec_t will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it will be instantiated via an explicit call
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``. It
            is allowed to modify the raw cudaGraph_t after first calling
            ``instantiate``, but the user must call ``instantiate`` again
            manually to make sure the instantiated graph has these
            changes. Pytorch has no means of tracking these changes.

    .. warning::
        This API is in beta and may change in future releases.

    """

    _tracker: _CUDAGraphInputLivenessTracker | None
    # Read-only property exposed from the C++ _CUDAGraph base via pybind;
    # annotated (not assigned) so the type checker sees it without shadowing it.
    _has_graph_exec: bool
    # Stays None unless maybe_stamp_capture_graph_id stamps it during capture_end
    # (requires annotations enabled and cudaGraphNodeGetToolsId available).
    _capture_graph_id: int | None
    # Exec graph id the recorded annotations are currently keyed to, or None
    # before the first remap. Lets a re-instantiate (which produces a fresh exec
    # id) rekey annotations from the previous exec id to the new one.
    _remapped_exec_id: int | None
    # Exec graph ids a consumer has recorded per-graph state under (one per
    # instantiate). Handed to the graph-destroy hooks on destruction so consumers
    # can purge that state and their maps do not grow across the run.
    _recorded_exec_ids: set[int]
    _keep_graph: bool
    # User hooks fired by capture_begin / capture_end / instantiate (see register_*_hook).
    _capture_start_hooks: dict[int, Callable[[CUDAGraph], None]]
    _capture_end_hooks: dict[int, Callable[[CUDAGraph], None]]
    _post_instantiate_hooks: dict[int, Callable[[CUDAGraph], None]]
    # Transient get_graph_data() cache shared across a single instantiate()'s
    # post-instantiate hooks, so multiple consumers (dependency recording, an
    # export_graph_data dump) query the topology once instead of once each.
    # Only populated while _caching_graph_data is set; dropped right after the
    # hooks run so the topology is not retained for the graph's lifetime.
    _caching_graph_data: bool
    _instantiate_graph_data: dict | None
    # Replay lifecycle hooks, fired around each replay (hot path -- only iterated
    # when non-empty). See register_replay_start_hook / register_replay_end_hook.
    _replay_start_hooks: dict[int, Callable[[CUDAGraph], None]]
    _replay_end_hooks: dict[int, Callable[[CUDAGraph], None]]
    # Destroy callbacks / retained objects for the current capture cycle, plus
    # the weakref.finalize armed on this graph that fires the holder on
    # collection. reset() fires the current holder and re-arms a fresh pair.
    _retained: _RetainedCallbacks
    _retained_finalizer: weakref.finalize | None

    def __new__(cls, keep_graph: bool = False) -> Self:
        instance = super().__new__(cls, keep_graph)
        instance._tracker = None
        instance._capture_graph_id = None
        instance._remapped_exec_id = None
        instance._recorded_exec_ids = set()
        instance._keep_graph = keep_graph
        # OrderedDict (not dict): RemovableHandle weak-references the mapping.
        instance._capture_start_hooks = OrderedDict()
        instance._capture_end_hooks = OrderedDict()
        instance._post_instantiate_hooks = OrderedDict()
        instance._caching_graph_data = False
        instance._instantiate_graph_data = None
        instance._replay_start_hooks = OrderedDict()
        instance._replay_end_hooks = OrderedDict()
        instance._retained_finalizer = None
        instance._arm_retained()
        return instance

    def _arm_retained(self) -> None:
        # Install a fresh destroy-callback holder and finalizer. The finalizer is
        # bound to the holder, NOT to `self`: a finalizer arg referencing the
        # graph would keep it alive forever and never fire. The holder does not
        # point back at the graph, so fire() can still sync the last replay
        # stream off it without pinning the graph.
        self._retained = _RetainedCallbacks()
        self._retained_finalizer = weakref.finalize(self, self._retained.fire)
        # When a consumer (e.g. the CUPTI monitor) has registered graph-destroy
        # hooks, arm the per-graph state purge for this capture cycle. The callback
        # captures the exec-id SET OBJECT (empty now, filled as this graph
        # records/instantiates) and the module fan-out function, never self and never
        # a hook: a closure reachable to the graph would pin it past collection so it
        # never fires. reset() re-arms a fresh holder, so this re-registers per cycle;
        # a graph that records nothing just fires on an empty set (a no-op).
        if _graph_destroy_hooks_active():
            exec_ids = self._recorded_exec_ids
            self.register_destroy_callback(lambda: _run_graph_destroy_hooks(exec_ids))

    def register_capture_start_hook(
        self, hook: Callable[[CUDAGraph], None]
    ) -> RemovableHandle:
        r"""Register ``hook(graph)`` to run when capture begins on this graph, right
        after capture is under way on the current stream. Hooks fire in registration
        order. Returns a handle whose ``remove()`` deregisters the hook.

        .. warning::
            The hook runs inside the capture: any CUDA work it issues is captured into
            the graph, and under the default ``"global"`` capture error mode an unsafe
            call raises. See :func:`torch.cuda.graphs.register_graph_capture_start_hook`.
        """
        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._capture_start_hooks)
        self._capture_start_hooks[handle.id] = hook
        return handle

    def register_capture_end_hook(
        self, hook: Callable[[CUDAGraph], None]
    ) -> RemovableHandle:
        r"""Register ``hook(graph)`` to run when capture ends, after capture
        completes but before the graph is finalized. The captured ``cudaGraph_t``
        is live (via :meth:`raw_cuda_graph`) for both ``keep_graph`` modes. Hooks
        fire in registration order. Returns a handle whose ``remove()``
        deregisters the hook.
        """
        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._capture_end_hooks)
        self._capture_end_hooks[handle.id] = hook
        return handle

    def register_post_instantiate_hook(
        self, hook: Callable[[CUDAGraph], None]
    ) -> RemovableHandle:
        r"""Register ``hook(graph)`` to run after each instantiation (including
        re-instantiation, which produces a fresh exec graph). The instantiated
        graph is available via :meth:`raw_cuda_graph_exec`. Hooks fire in
        registration order. Returns a handle whose ``remove()`` deregisters the
        hook.
        """
        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._post_instantiate_hooks)
        self._post_instantiate_hooks[handle.id] = hook
        return handle

    def register_replay_start_hook(
        self, hook: Callable[[CUDAGraph], None]
    ) -> RemovableHandle:
        r"""Register ``hook(graph)`` to run at the start of every :meth:`replay`,
        just before the graph is launched (after any on-demand instantiation, so
        :meth:`raw_cuda_graph_exec` is valid). Hooks fire in registration order.
        Returns a handle whose ``remove()`` deregisters the hook.

        .. note::
            Replay is the hot path and a registered hook runs on every replay --
            keep it cheap. With no hook registered the cost is a single dict
            emptiness check.
        """
        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._replay_start_hooks)
        self._replay_start_hooks[handle.id] = hook
        return handle

    def register_replay_end_hook(
        self, hook: Callable[[CUDAGraph], None]
    ) -> RemovableHandle:
        r"""Register ``hook(graph)`` to run at the end of every :meth:`replay`,
        just after the graph is launched. The launch is asynchronous, so the hook
        runs once the replay is *enqueued*, not once the GPU work completes. Hooks
        fire in registration order. Returns a handle whose ``remove()``
        deregisters the hook. See the hot-path note on
        :meth:`register_replay_start_hook`.

        End hooks fire even if the launch raises -- so a start hook is always
        balanced by an end -- and the launch error then propagates. (Start hooks
        that raise abort the replay before launch, and no end hook fires.)
        """
        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._replay_end_hooks)
        self._replay_end_hooks[handle.id] = hook
        return handle

    def register_destroy_callback(
        self,
        cb: Callable[[], None],
        *,
        synchronize_before_release: bool = False,
    ) -> RemovableHandle:
        r"""Register ``cb()`` to run when this graph is destroyed (finalized) or
        explicitly :meth:`reset`, just before its CUDA resources are freed.
        Callbacks fire once per capture cycle, in registration order; exceptions
        are swallowed so one failure does not abort the rest. ``cb`` must NOT
        reference this graph: the finalizer that fires it is held by a global
        registry, so a callback reachable to the graph keeps the graph alive
        until interpreter exit (it is never collected, hence never fired).
        Returns a handle whose ``remove()`` deregisters the callback.

        Teardown does not synchronize CUDA, and ``cudaGraphExecDestroy`` frees an
        in-flight graph only asynchronously, so a callback that frees device
        memory the graph reads/writes is a use-after-free if a replay is still
        in flight. Pass ``synchronize_before_release=True`` to synchronize every
        stream this graph was replayed on before firing. Otherwise callbacks
        must not free anything the graph references.
        """
        from torch.utils.hooks import RemovableHandle

        if synchronize_before_release:
            self._retained.sync_before_fire = True
        handle = RemovableHandle(self._retained.callbacks)
        self._retained.callbacks[handle.id] = cb
        return handle

    def retain_object(
        self,
        obj: object,
        *,
        synchronize_before_release: bool = False,
    ) -> RemovableHandle:
        r"""Keep ``obj`` alive for this graph's current capture cycle and release
        it when the graph is destroyed (finalized) or explicitly :meth:`reset`.
        No callback runs; normal refcounting drops ``obj`` when the retained
        reference is released. Returns a handle whose ``remove()`` drops the
        retained reference early. As with :meth:`register_destroy_callback`,
        ``obj`` must NOT reference this graph, or the graph is kept alive until
        interpreter exit and ``obj`` is never released.

        ``synchronize_before_release`` has the same meaning and caveats as in
        :meth:`register_destroy_callback`: set it if releasing ``obj`` frees device
        memory the graph reads/writes (e.g. ``obj`` is the last reference to a
        tensor the graph uses) and replays may still be in flight.
        """
        from torch.utils.hooks import RemovableHandle

        if synchronize_before_release:
            self._retained.sync_before_fire = True
        handle = RemovableHandle(self._retained.objects)
        self._retained.objects[handle.id] = obj
        return handle

    def _maybe_remap_annotations(self) -> None:
        # Remap recorded kernel annotations to the current exec graph id. No-op
        # unless a capture id was stamped (annotations enabled). Called from
        # instantiate() -- which replay() routes through for keep_graph=True --
        # so every fresh exec graph (each instantiate() produces a new exec id)
        # rekeys the annotations; remap_to_exec_graph self-skips when the exec id
        # is unchanged.
        if self._capture_graph_id is None:
            return
        from torch.cuda._graph_annotations import remap_to_exec_graph

        remap_to_exec_graph(self)
        # Record the exec id we remapped to so the destroy callback (armed in
        # capture_end_post) hands it to the graph-destroy hooks for per-graph state
        # cleanup. A re-instantiate mints a fresh exec id, so the set accumulates each.
        if self._remapped_exec_id is not None:
            self._recorded_exec_ids.add(self._remapped_exec_id)

    def _release_python_resources(self) -> None:
        # Single source of truth for GC-critical Python resources released by
        # both reset() and __del__. Destroy callbacks are NOT fired here: on the
        # destruction path the armed weakref.finalize fires them (after __del__,
        # before the C++ resources are freed), so user code stays out of
        # __del__; reset() fires + re-arms the holder itself (see reset()).
        tracker, self._tracker = self._tracker, None
        if tracker is not None:
            tracker.stop()

    def __del__(self) -> None:
        try:
            self._release_python_resources()
        except Exception:
            pass  # don't raise under GC

    def capture_begin(
        self,
        pool: _GraphPool | None = None,
        capture_error_mode: str = "global",
        check_input_liveness: bool = False,
    ) -> None:
        r"""Begin capturing CUDA work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) or
                :class:`~torch.cuda.MemPool` that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
            capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
                Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
                may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
                actions in the current thread, and "relaxed" will not error on these actions. Do NOT change this setting
                unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_
            check_input_liveness (bool, optional):
                If ``True``, tracks external tensor inputs during graph capture and
                raises an error if any are deallocated before replay. This helps debug "use after free" errors
                where input tensors are garbage collected between capture and replay. Default: ``False``.

                .. note::
                    Custom CUDA kernels added outside PyTorch (e.g., via cuLaunchKernel or DLPack) are not
                    tracked by this mechanism.
        """
        if self._tracker is not None:
            self._tracker.stop()
            self._tracker = None
        super().capture_begin(
            pool=_get_pool_id(pool), capture_error_mode=capture_error_mode
        )
        if check_input_liveness:
            from torch.utils._cuda_debug import _CUDAGraphInputLivenessTracker

            self._tracker = _CUDAGraphInputLivenessTracker()
            self._tracker.start()
        # Capture is live from here, so a hook must not issue CUDA work (see
        # register_capture_start_hook). Global hooks run before the per-graph ones,
        # matching instantiate().
        if _global_capture_start_hooks:
            _run_global_hooks(_global_capture_start_hooks, self)
        for hook in list(self._capture_start_hooks.values()):
            hook(self)

    def capture_end_pre(self) -> None:
        r"""End capture but do not finalize: leaves the captured ``cudaGraph_t``
        live (for both ``keep_graph`` modes) so it can be inspected before
        :meth:`capture_end_post` instantiates and/or destroys it."""
        super().capture_end_pre()

    def capture_end_post(self) -> None:
        r"""Finalize a capture started by :meth:`capture_end_pre`: destroy the
        template when ``keep_graph=False`` (the graph must already be
        instantiated; :meth:`capture_end` and the context manager do so)."""
        super().capture_end_post()
        if self._tracker is not None:
            self._tracker.stop()

    def capture_end(self) -> None:
        r"""End CUDA graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        self.capture_end_pre()
        # Run the in-window work (stamp, then user capture-end hooks) while the
        # template is live (both keep_graph modes). Errors here are unexpected
        # (stamp) or user bugs (hooks) and propagate -- we deliberately don't
        # wrap them in a finally that calls capture_end_post(), since a failing
        # finalize would mask the real error.
        from torch.cuda._graph_annotations import maybe_stamp_capture_graph_id

        maybe_stamp_capture_graph_id(self)
        if _global_capture_end_hooks:
            _run_global_hooks(_global_capture_end_hooks, self)
        for hook in list(self._capture_end_hooks.values()):
            hook(self)
        if not self._keep_graph:
            # Route keep_graph=False instantiation through Python so the remap
            # and post-instantiate hooks fire from instantiate().
            self.instantiate()
        self.capture_end_post()

    def instantiate(self) -> None:
        r"""Instantiate the CUDA graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the cudaGraph_t returned
        by ``raw_cuda_graph``.
        """
        super().instantiate()
        self._maybe_remap_annotations()
        # Fire registered instantiate hooks (e.g. a profiler observer inspecting the freshly
        # instantiated graph). Registering a hook is the opt-in; no consumer -> a no-op.
        # get_graph_data() is cached for the duration of the hooks so several consumers share
        # one query; the cache is dropped afterwards so nothing is retained for the graph.
        self._caching_graph_data = True
        try:
            if _global_instantiate_hooks:
                _run_global_hooks(_global_instantiate_hooks, self)
            for hook in list(self._post_instantiate_hooks.values()):
                hook(self)
        finally:
            self._caching_graph_data = False
            self._instantiate_graph_data = None

    def replay(self) -> None:
        r"""Replay the CUDA work captured by this graph."""
        if self._tracker is not None:
            self._tracker.check_alive(self.pools())
        if self._retained.sync_before_fire:
            # Record the replay stream on the holder so fire() can drain it
            # before releasing graph-referenced resources, without the holder
            # referencing the graph. Only recorded when a caller opted into sync.
            self._retained.replay_streams.add(torch.cuda.current_stream())
        # With keep_graph=True the exec graph is instantiated on demand here on
        # the first replay; the C++ replay() requires it to already exist. The
        # annotation remap rides on instantiate(), so it is handled by that call.
        if not self._has_graph_exec:
            self.instantiate()
        if _global_replay_start_hooks:
            _run_global_hooks(_global_replay_start_hooks, self)
        if self._replay_start_hooks:
            for hook in list(self._replay_start_hooks.values()):
                hook(self)
        try:
            super().replay()
        finally:
            if _global_replay_end_hooks:
                _run_global_hooks(_global_replay_end_hooks, self)
            if self._replay_end_hooks:
                for hook in list(self._replay_end_hooks.values()):
                    hook(self)

    def reset(self) -> None:
        r"""Delete the graph currently held by this instance."""
        self._release_python_resources()
        # also-fire-on-reset: reset() destroys this capture's CUDA resources and
        # the graph may be re-captured, so fire the current destroy callbacks and
        # re-arm a fresh holder + finalizer for the next capture cycle (the old
        # finalizer is bound to the now-spent holder). detach() cancels it and
        # drops its reference to the spent holder promptly.
        self._retained.fire()
        if self._retained_finalizer is not None:
            self._retained_finalizer.detach()
        # Start a fresh id set BEFORE re-arming: _arm_retained captures it into the
        # next cycle's destroy callback. The just-fired holder dropped the old one.
        self._recorded_exec_ids = set()
        self._arm_retained()
        # Reset-only state: scrubbed here because the object is reused after
        # reset(); on death these ints die with the object.
        self._capture_graph_id = None
        self._remapped_exec_id = None
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return torch.cuda._POOL_HANDLE(super().pool())

    def pools(self) -> list[_POOL_HANDLE]:
        r"""Return opaque tokens for all memory pools retained by this graph."""
        return [torch.cuda._POOL_HANDLE(pool) for pool in super().pools()]

    def _retain_pool(self, pool: _GraphPool) -> None:
        pool_id = _get_pool_id(pool)
        if pool_id is None:
            raise RuntimeError("CUDAGraph._retain_pool expected a memory pool")
        return super()._retain_pool(pool_id)

    def enable_debug_mode(self) -> None:
        r"""Retain the captured graph (equivalent to ``keep_graph=True``) so it
        can be inspected, e.g. via :meth:`debug_dump`. Kept for backward
        compatibility."""
        super().enable_debug_mode()
        # Keep the Python-side flag in sync with the C++ keep_graph_ it now sets.
        self._keep_graph = True

    def debug_dump(self, debug_path: str, *, verbose: bool = True) -> None:
        r"""Dump the captured graph to ``debug_path`` in Graphviz DOT format.

        The graph's template must be live: ``keep_graph=True`` (or
        :meth:`enable_debug_mode`), or called from a capture-end hook. Requires
        the ``cuda.bindings`` package.

        Arguments:
            debug_path (required): Path to dump the graph to.
            verbose: If ``True`` (default), use the most verbose DOT output.
        """
        _dump_graph_dot(self, debug_path, verbose=verbose)

    def raw_cuda_graph(self) -> int:
        r"""Returns the underlying cudaGraph_t. The template must be live: this
        requires ``keep_graph=True`` (it persists after ``capture_end``), or
        access from within a capture-end hook (before the template is destroyed
        for ``keep_graph=False``).

        See the following for APIs for how to manipulate this object: `Graph Management <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>`_ and `cuda-python Graph Management bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#graph-management>`_
        """
        return super().raw_cuda_graph()

    def raw_cuda_graph_exec(self) -> int:
        r"""Returns the underlying cudaGraphExec_t. ``instantiate`` must have been called if ``keep_graph`` is True, or ``capture_end`` must have been called if ``keep_graph`` is False. If you call ``instantiate()`` after ``raw_cuda_graph_exec()``, the previously returned cudaGraphExec_t will be destroyed. It is your responsibility not to use this object after destruction.

        See the following for APIs for how to manipulate this object: `Graph Execution <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH__EXEC.html>`_ and `cuda-python Graph Execution bindings <https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#graph-execution>`_
        """
        return super().raw_cuda_graph_exec()

    def get_graph_data(self) -> dict:
        r"""Return a dictionary describing the graph's topology and node metadata.

        ``keep_graph`` must be True.  The graph must have been instantiated
        (via :meth:`instantiate`) before calling this method.
        Requires the ``cuda.bindings`` package.

        Returns a dictionary with structure::

            {
                "exec_graph_id": int,
                "nodes": [
                    {
                        "index": int,
                        "node_type": str,
                        "tools_id": int,
                        "graph_id": int,
                        "node_id": int,
                        "kernel_name": str or None,
                        "event_ptr": int,
                        "host_fn_addr": int,
                        "host_fn_name": str or None,
                        "dependencies": [int, ...],
                        "dependents": [int, ...],
                    },
                    ...,
                ],
            }

        ``event_ptr`` is the ``cudaEvent_t`` handle (as an int) an event-record
        or event-wait node records / waits on -- these nodes produce no timed
        CUPTI record, so matching a wait to the record that signals it is the
        only way to reason about the cross-stream sync it encodes. It is ``0``
        for other node types.

        ``host_fn_addr`` / ``host_fn_name`` are populated for host nodes (a CPU
        callback run as a graph node): the callback address and a best-effort
        demangled symbol name for it (``None`` when it resolves to no exported
        symbol). They are ``0`` / ``None`` for other node types.

        Each node's ``graph_id`` is remapped to the exec graph id so that
        ``tools_id`` values match those reported by CUPTI-based profilers.
        ``dependencies`` and ``dependents`` are lists of node indices within
        the ``nodes`` list.

        This structure is useful for inspecting a profiler trace and
        establishing whether a particular dependency observed in the profile
        is a true dependency (encoded in the graph) or a fake dependency
        caused by mapping of independent streams to the same hardware
        channel.

        Child-graph and conditional nodes are reported as nodes in their own
        right, but this walk does not descend into their bodies (those are
        separate ``cudaGraph_t`` objects), so the work inside them is absent
        from ``nodes`` and a warning is issued. The ids that *are* reported
        stay valid: the exec graph preserves top-level node ids.
        """
        # Serve the shared cache when instantiate() has it live (see _caching_graph_data).
        if self._instantiate_graph_data is not None:
            return self._instantiate_graph_data
        from torch.cuda._graph_annotations import _is_tools_id_unavailable

        _require_cuda_bindings()
        # Narrow for the type checker (cuda bindings are present past the check).
        if _cuda_runtime is None or _cuda_driver is None:
            raise AssertionError(
                "expected _cuda_runtime and _cuda_driver to be not None"
            )

        if _is_tools_id_unavailable():
            raise RuntimeError(
                "get_graph_data requires cudaGraphNodeGetToolsId which needs "
                "cuda.bindings >= 13.1 and CUDA driver >= 13.1 "
                "(or cuda-compat >= 13.1 in LD_LIBRARY_PATH)"
            )

        node_type_names = {
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "memcpy",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset: "memset",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeHost: "host",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeGraph: "child_graph",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeEmpty: "empty",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent: "wait_event",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord: "event_record",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc: "mem_alloc",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemFree: "mem_free",
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional: "conditional",
        }
        # Node types whose work lives in a separate cudaGraph_t (see the warning below).
        nested_graph_types = {
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeGraph,
            _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional,
        }
        nested_node_types: set[str] = set()

        raw = self.raw_cuda_graph()

        _, num = _check_cuda_bindings(_cuda_runtime.cudaGraphGetNodes(raw, numNodes=0))
        nodes, num = _check_cuda_bindings(
            _cuda_runtime.cudaGraphGetNodes(raw, numNodes=num)
        )

        handle_to_idx: dict[int, int] = {}
        node_infos: list[dict] = []

        for i in range(num):
            node = nodes[i]
            handle_to_idx[int(node)] = i

            ntype = _check_cuda_bindings(_cuda_runtime.cudaGraphNodeGetType(node))
            if ntype in nested_graph_types:
                nested_node_types.add(node_type_names.get(ntype, ntype))
            tools_id = _check_cuda_bindings(_cuda_runtime.cudaGraphNodeGetToolsId(node))
            graph_id = tools_id >> 32
            node_id = tools_id & 0xFFFFFFFF

            kernel_name = None
            if ntype == _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                cu_node = _cuda_driver.CUgraphNode(init_value=int(node))
                err, params = _cuda_driver.cuGraphKernelNodeGetParams(cu_node)
                if err == _cuda_driver.CUresult.CUDA_SUCCESS and int(params.func):
                    cu_func = _cuda_driver.CUfunction(init_value=int(params.func))
                    err, name = _cuda_driver.cuFuncGetName(cu_func)
                    if err == _cuda_driver.CUresult.CUDA_SUCCESS:
                        kernel_name = name.decode() if isinstance(name, bytes) else name

            # Event record/wait nodes carry a cudaEvent_t but emit no timed CUPTI record;
            # capture the handle so a wait node can be matched to the record that signals it.
            # Not best-effort like the kernel-name lookup above (a name can legitimately be
            # unavailable): the node type is already established here, so these calls are
            # expected to succeed. Swallowing a failure would leave event_ptr 0 and make the
            # record/wait match quietly wrong rather than loud.
            event_ptr = 0
            if ntype == _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord:
                event_ptr = int(
                    _check_cuda_bindings(
                        _cuda_runtime.cudaGraphEventRecordNodeGetEvent(node)
                    )
                )
            elif ntype == _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent:
                event_ptr = int(
                    _check_cuda_bindings(
                        _cuda_runtime.cudaGraphEventWaitNodeGetEvent(node)
                    )
                )

            # Host nodes carry no name in the CUPTI record; recover the callback address
            # from the node params and resolve a best-effort symbol name (see
            # _resolve_host_fn_name). Both are None/0 for other node types.
            host_fn_addr = 0
            host_fn_name = None
            if ntype == _cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeHost:
                err, params = _cuda_runtime.cudaGraphHostNodeGetParams(node)
                if err == _cuda_runtime.cudaError_t.cudaSuccess:
                    host_fn_addr = int(params.fn)
                    host_fn_name = _resolve_host_fn_name(host_fn_addr)

            node_infos.append(
                {
                    "index": i,
                    "node_type": node_type_names.get(ntype, str(ntype)),
                    "tools_id": tools_id,
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "kernel_name": kernel_name,
                    "event_ptr": event_ptr,
                    "host_fn_addr": host_fn_addr,
                    "host_fn_name": host_fn_name,
                    "dependencies": [],
                    "dependents": [],
                }
            )

        _, _, _, num_edges = _check_cuda_bindings(
            _cuda_runtime.cudaGraphGetEdges(raw, numEdges=0)
        )
        if num_edges > 0:
            from_nodes, to_nodes, _edge_data, num_edges = _check_cuda_bindings(
                _cuda_runtime.cudaGraphGetEdges(raw, numEdges=num_edges)
            )
            for i in range(num_edges):
                src = handle_to_idx.get(int(from_nodes[i]))
                dst = handle_to_idx.get(int(to_nodes[i]))
                if src is not None and dst is not None:
                    node_infos[src]["dependents"].append(dst)
                    node_infos[dst]["dependencies"].append(src)

        exec_handle = _cuda_runtime.cudaGraphExec_t(
            init_value=self.raw_cuda_graph_exec()
        )
        exec_graph_id = _check_cuda_bindings(
            _cuda_runtime.cudaGraphExecGetId(exec_handle)
        )
        for info in node_infos:
            info["tools_id"] = (exec_graph_id << 32) | info["node_id"]
            info["graph_id"] = exec_graph_id

        if nested_node_types:
            # What is returned stays correct: the exec graph preserves top-level node
            # ids and numbers the nested ones after them. Those nested nodes do execute
            # and do show up in a profiler trace (under this exec graph id, with the
            # appended node ids), they are just missing from the topology below.
            warnings.warn(
                f"get_graph_data: this graph contains {sorted(nested_node_types)} "
                "nodes; the work inside them is in a separate cudaGraph_t that this "
                "walk does not descend into, so it is absent from the returned nodes",
                stacklevel=2,
            )

        data = {
            "exec_graph_id": exec_graph_id,
            "nodes": node_infos,
        }
        if self._caching_graph_data:
            self._instantiate_graph_data = data
        return data


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _resolve_host_fn_name(addr: int) -> str | None:
    """Best-effort symbol name for a host-node callback address via dladdr + demangle.
    Returns None when the address resolves to no exported symbol (e.g. an anonymous
    trampoline). Host nodes carry no name in the CUPTI record, so this is the only source."""
    if not addr:
        return None
    info = _DlInfo()
    if not ctypes.CDLL(None).dladdr(ctypes.c_void_p(addr), ctypes.byref(info)):
        return None
    if not info.dli_sname:
        return None
    return torch._C._demangle(info.dli_sname.decode())


def _dump_graph_dot(cuda_graph: CUDAGraph, path: str, *, verbose: bool = True) -> None:
    """Write ``cuda_graph``'s template to ``path`` in Graphviz DOT format."""
    _require_cuda_bindings()
    flags = (
        _cuda_runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose  # pyrefly: ignore[missing-attribute]
        if verbose
        else 0
    )
    _check_cuda_bindings(
        _cuda_runtime.cudaGraphDebugDotPrint(  # pyrefly: ignore[missing-attribute]
            cuda_graph.raw_cuda_graph(), path.encode(), flags
        )
    )


def export_dot(path: str, *, verbose: bool = True) -> Callable[[CUDAGraph], None]:
    r"""Return a capture-end hook that dumps the captured graph to ``path`` in
    Graphviz DOT format. Register it with
    :meth:`CUDAGraph.register_capture_end_hook`; works for both ``keep_graph``
    modes since it runs while the template is still live."""

    def _hook(cuda_graph: CUDAGraph) -> None:
        _dump_graph_dot(cuda_graph, path, verbose=verbose)

    return _hook


def export_graph_data(path: str) -> Callable[[CUDAGraph], None]:
    r"""Return a post-instantiate hook that pickles :meth:`CUDAGraph.get_graph_data`
    to ``path``. Register it with :meth:`CUDAGraph.register_post_instantiate_hook`:
    ``get_graph_data`` needs the graph instantiated (it remaps node ids to the
    exec graph id), and at post-instantiate time the template is still live, so
    this works for both ``keep_graph`` modes."""

    def _hook(cuda_graph: CUDAGraph) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(cuda_graph.get_graph_data(), f)

    return _hook


class graph:
    r"""Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) or
            :class:`~torch.cuda.MemPool` hinting this graph's capture may share memory
            from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_
        enable_annotations (bool, optional): If ``True``, enables kernel annotation
            recording on entry and automatically calls
            :func:`~torch.cuda._graph_annotations.resolve_pending_annotations` before
            the capture ends.  Annotations are **not** cleared on exit so that multiple
            graphs in the same workload can accumulate annotations.
            Requires ``cuda.bindings`` package and cuda-compat >= 13.1 or CUDA driver >= 13.1.
        check_input_liveness (bool, optional): If ``True``, tracks external tensor inputs during graph capture and
            raises an error if any are deallocated before replay. This helps debug "use after free" errors
            where input tensors are garbage collected between capture and replay. Default: ``False``.

            .. note::
                Custom CUDA kernels added outside PyTorch (e.g., via cuLaunchKernel or DLPack) are not
                tracked by this mechanism.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.

    .. _cudaStreamCaptureMode:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
    """

    default_capture_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: _GraphPool | None = None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
        enable_annotations: bool = False,
        check_input_liveness: bool = False,
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if stream is None and self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        pool_id = _get_pool_id(pool)
        self.pool: tuple[()] | tuple[_POOL_HANDLE] = (
            () if pool_id is None else (pool_id,)
        )
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode
        self._enable_annotations = enable_annotations
        self.check_input_liveness = check_input_liveness

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph
        torch.cuda.synchronize()

        if torch.compiler.config.force_cudagraph_gc:
            # Originally we unconditionally garbage collected here. On one hand
            # that's nice because we have a chance to collect more memory, but
            # on the other hand it is REALLY expensive, especially for doing
            # multiple cudagraph captures in a row. In theory it will only help
            # when a dead python cycle is holding onto CUDA memory.
            gc.collect()

        torch.cuda.empty_cache()
        # pyrefly: ignore [missing-attribute]
        torch._C._host_emptyCache()

        # Scope annotation recording to this capture: stamp/mark_kernels gate on
        # this flag, and __exit__ always clears it.
        from torch.cuda._graph_annotations import _set_annotations_enabled

        _set_annotations_enabled(self._enable_annotations)

        # Stackoverflow seems comfortable with this pattern
        # https://stackoverflow.com/questions/26635684/calling-enter-and-exit-manually#39172487
        self.stream_ctx.__enter__()

        self.cuda_graph.capture_begin(
            # type: ignore[misc]
            *self.pool,
            # pyrefly: ignore [bad-keyword-argument]
            capture_error_mode=self.capture_error_mode,
            # pyrefly: ignore [bad-keyword-argument]
            check_input_liveness=self.check_input_liveness,
        )
        # The capture stream is now capturing into the top-level graph; remember it so
        # mark_kernels can tell a conditional-node body apart from this graph.
        from torch.cuda._graph_annotations import maybe_stamp_capture_root

        maybe_stamp_capture_root(torch.cuda.current_stream())

    def __exit__(self, *args: object) -> None:
        from torch.cuda._graph_annotations import (
            _set_annotations_enabled,
            resolve_pending_annotations,
        )

        try:
            if self._enable_annotations:
                resolve_pending_annotations()

            # capture_end stamps the capture id and, for keep_graph=False,
            # instantiates (which remaps annotations to the exec id). For
            # keep_graph=True the remap is owned by the later instantiate()/replay().
            self.cuda_graph.capture_end()
            self.stream_ctx.__exit__(*args)
        finally:
            # Annotation recording is capture-scoped; clear it unconditionally.
            _set_annotations_enabled(False)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


_ModuleOrCallable: TypeAlias = Union["torch.nn.Module", Callable[..., object]]


@overload
def make_graphed_callables(
    callables: _ModuleOrCallable,
    sample_args: tuple[Tensor, ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _GraphPool | None = None,
    capture_error_mode: str = "global",
    enable_annotations: bool = False,
) -> _ModuleOrCallable: ...


@overload
def make_graphed_callables(
    callables: tuple[_ModuleOrCallable, ...],
    sample_args: tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _GraphPool | None = None,
    capture_error_mode: str = "global",
    enable_annotations: bool = False,
) -> tuple[_ModuleOrCallable, ...]: ...


def make_graphed_callables(
    callables: _ModuleOrCallable | tuple[_ModuleOrCallable, ...],
    sample_args: tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: _GraphPool | None = None,
    capture_error_mode: str = "global",
    enable_annotations: bool = False,
) -> _ModuleOrCallable | tuple[_ModuleOrCallable, ...]:
    r"""Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.
    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables
            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order
            they'll run in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
