        # Subclasses with mutable state should override.
        if transform_fn is not None:
            return transform_fn(self)
        return self


# Subclasses can be found in torch/_dynamo/source.py
@dataclass_with_cached_hash(frozen=True)
class ChainedSource(Source):
    base: Source

    def is_dict_key(self) -> bool:
        # Recurse until you either hit a ConstDictKey or a Source
        return self.base.is_dict_key()

    def is_ephemeral(self) -> bool:
        return self.base.is_ephemeral()

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return self.base.guard_source

    def get_base(self) -> Source:
        current: Source = self
        while isinstance(current, ChainedSource):
            current = current.base
        return current

    @functools.cached_property
    def name(self) -> str:
        return self._name_template.format(self.base.name)

    def get_value(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        cache: dict[Source, Any],
    ) -> Any:
        if self in cache:
            return cache[self]
        tmpvar = "tmp"
        counter = 0
        while tmpvar in locals:
            tmpvar = f"tmp{counter}"
            counter += 1
        locals[tmpvar] = self.base.get_value(globals, locals, cache)
        value = eval(self._name_template.format(tmpvar), globals, locals)
        del locals[tmpvar]
        cache[self] = value
        return value

    def clone(self, transform_fn: Callable[[Source], Source] | None = None) -> Source:
        cloned_fields: dict[str, Any] = {"base": self.base.clone(transform_fn)}
        for f in dataclasses.fields(self):
            if f.name == "base":
                continue
            val = getattr(self, f.name)
            if isinstance(val, Source):
                cloned_fields[f.name] = val.clone(transform_fn)
        result = dataclasses.replace(self, **cloned_fields)
        if transform_fn is not None:
            result = transform_fn(result)
        return result


def detect_fake_mode(inputs: Any = None) -> FakeTensorMode | None:
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import (
        FakeTensor,
        FakeTensorMode,
        get_plain_tensors,
        is_fake_tensor,
        maybe_get_fake_mode,
    )

    # If TracingContext has a fake_mode, use it authoritatively.
    # This is the case when Dynamo is driving compilation - any fake tensors
    # from other modes in the inputs will be refakified by the caller.
    if context := TracingContext.try_get():
        fake_mode = context.fake_mode
        if fake_mode is not None:
            return fake_mode

    fake_modes = []

    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, "active fake mode", i))

    flat_inputs = pytree.tree_leaves(inputs)
    for i, flat_input in enumerate(flat_inputs):
        if is_fake_tensor(flat_input):
            fake_modes.append((maybe_get_fake_mode(flat_input), "fake tensor input", i))
        if is_traceable_wrapper_subclass(flat_input):
            out: list[torch.Tensor | int | torch.SymInt] = []
            get_plain_tensors(flat_input, out=out)  # type: ignore[arg-type]
            fake_tensors: list[FakeTensor] = [
                x
                for x in out
                if isinstance(x, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
            ]
            fake_modes.extend(
                [
                    (maybe_get_fake_mode(tensor), f"subclass input {i}", ix)
                    for ix, tensor in enumerate(fake_tensors)
                ]
            )

    if fake_modes:
        fake_mode, desc1, i1 = fake_modes[0]
        for m, desc2, i2 in fake_modes[1:]:
            if fake_mode is not m:
                raise AssertionError(
                    f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                    f"fake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\n"
                    f"fake mode from {desc2} {i2} allocated at:\n{m.stack}"
                )

        return fake_mode
    else:
        return None


def active_fake_mode() -> FakeTensorMode | None:
    """
    Inspects the dispatch mode stack for an active fake mode and returns it.
    Returns None if no fake mode is active.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for _, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            return m

    return None
<|endoftext|># mypy: allow-untyped-defs
import copyreg
import difflib
import functools
import io
import os
import pickle
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import threading
import warnings
from collections.abc import Callable
from contextlib import closing, contextmanager
from enum import Enum
from typing import Any, cast, Generic, IO, TypeAlias, TypeVar
from typing_extensions import TypeIs

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch._sources import get_source_lines_and_file
from torch._utils import _import_dotted_name
from torch.storage import _get_dtype_from_pickle_storage_type
from torch.types import FileLike, Storage


__all__ = [
    "SourceChangeWarning",
    "mkdtemp",
    "register_package",
    "check_module_version_greater_or_equal",
    "validate_cuda_device",
    "validate_hpu_device",
    "location_tag",
    "default_restore_location",
    "normalize_storage_type",
    "storage_to_tensor_type",
    "save",
    "load",
    "StorageType",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
    "clear_safe_globals",
    "get_safe_globals",
    "add_safe_globals",
    "safe_globals",
    "get_unsafe_globals_in_checkpoint",
    "skip_data",
]

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct("=l").size
INT_SIZE = struct.Struct("=i").size
SHORT_SIZE = struct.Struct("=h").size

MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ","

MAP_LOCATION: TypeAlias = (
    Callable[[Storage, str], Storage] | torch.device | str | dict[str, str] | None
)
STORAGE: TypeAlias = Storage | torch.storage.TypedStorage | torch.UntypedStorage

IS_WINDOWS = sys.platform == "win32"

UNSAFE_MESSAGE = (
    "In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` "
    "from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, "
    "but it can result in arbitrary code execution. Do it only if you got the file from a "
    "trusted source."
)

if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None  # type: ignore[assignment]


def _default_to_weights_only(pickle_module):
    is_fbcode = not hasattr(torch.version, "git_version")
    return pickle_module is None and not is_fbcode


# _serialization_tls is used to store thread local state specific to serialization
# that needs to be propagated to other files, in particular we use this for
# (1) map_location (needed for wrapper subclasses/third party devices to torch._utils)
# (2) skip_data (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
# (3) materialize_fake_tensors (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
class _SerializationLocal(threading.local):
    def __init__(self):
        super().__init__()
        self.map_location: MAP_LOCATION | None = None
        self.skip_data: bool = False
        self.materialize_fake_tensors: bool = False


_serialization_tls = _SerializationLocal()


class SourceChangeWarning(Warning):
    pass


@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)


_package_registry: list[
    tuple[
        int,
        Callable[[STORAGE], str | None],
        Callable[[STORAGE, str], STORAGE | None],
    ]
] = []


class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


def get_default_load_endianness() -> LoadEndianness | None:
    """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """
    from torch.utils.serialization import config

    return config.load.endianness


def set_default_load_endianness(endianness):
    """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    from torch.utils.serialization import config

    config.load.endianness = endianness


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    from torch.utils.serialization import config

    return config.save.compute_crc32


def set_crc32_options(compute_crc32: bool):
    """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 computation flag
    """
    from torch.utils.serialization import config

    config.save.compute_crc32 = compute_crc32


def get_default_mmap_options() -> int | None:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """
    from torch.utils.serialization import config

    return config.load.mmap_flags


def _get_storage_alignment() -> int:
    """
    Gets alignment for storages in torch.save files/

    Defaults to 64.

    Returns:
        storage_alignment: int
    """
    from torch.utils.serialization import config

    return config.save.storage_alignment


class set_default_mmap_options:
    """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """

    def __init__(self, flags: int) -> None:
        if IS_WINDOWS:
            raise RuntimeError(
                "Changing the default mmap options is currently not supported for Windows"
            )
        if flags != MAP_PRIVATE and flags != MAP_SHARED:
            raise ValueError(
                "Invalid argument in function set_default_mmap_options, "
                f"expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got {flags}"
            )
        # global config
        from torch.utils.serialization import config

        self.prev = config.load.mmap_flags
        config.load.mmap_flags = flags

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        from torch.utils.serialization import config

        config.load.mmap_flags = self.prev


def clear_safe_globals() -> None:
    """
    Clears the list of globals that are safe for ``weights_only`` load.
    """
    _weights_only_unpickler._clear_safe_globals()


def get_safe_globals() -> list[Callable | tuple[Callable, str]]:
    """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """
    return _weights_only_unpickler._get_safe_globals()


def add_safe_globals(safe_globals: list[Callable | tuple[Callable, str]]) -> None:
    """
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Each item in the list can either be a function/class or a tuple of the form
    (function/class, string) where string is the full path of the function/class.

    Within the serialized format, each function is identified with its full
    path as ``{__module__}.{__qualname__}``. When calling this API, you can provide this
    full path that should match the one in the checkpoint otherwise the default
    ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Args:
        safe_globals (List[Union[Callable, Tuple[Callable, str]]]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """
    _weights_only_unpickler._add_safe_globals(safe_globals)


class safe_globals(_weights_only_unpickler._safe_globals):
    r"""Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    """


def get_unsafe_globals_in_checkpoint(f: FileLike) -> list[str]:
    """Returns a list of strings of functions/classes in a ``torch.save`` object that are not safe for ``weights_only``.

    For a given function or class ``f``, the corresponding string will be of the form
    ``{f.__module__}.{f.__name__}``.

    This function will return any GLOBALs in the checkpoint that are not in the set marked safe
    for ``weights_only`` (either via :func:`add_safe_globals` or :class:`safe_globals` context or
    allowlisted by ``torch`` by default).

    .. note::
        This function will statically disassemble the pickle file in the checkpoint.
        The implication is any classes dynamically pushed onto the stack during unpickling
        will not be included in the output.

    Args:
        f: File-like object or string containing the checkpoint object saved via ``torch.save``

    Returns:
        A list of strings of pickle GLOBALs in the checkpoint that are not allowlisted for ``weights_only``.
    """
    default_safe_globals_strings = set(
        _weights_only_unpickler._get_allowed_globals().keys()
    )
    user_safe_global_strings = set(
        _weights_only_unpickler._get_user_allowed_globals().keys()
    )
    safe_global_strings = default_safe_globals_strings.union(user_safe_global_strings)

    with _open_file_like(f, "rb") as opened_file:
        if not _is_zipfile(opened_file):
            raise ValueError("Expected input to be a checkpoint returned by torch.save")
        with _open_zipfile_reader(opened_file) as zip_file:
            if _is_torchscript_zip(zip_file):
                raise ValueError(
                    "Expected input to be a checkpoint returned by torch.save but got a torchscript checkpoint"
                )
            data_file = io.BytesIO(zip_file.get_record("data.pkl"))
            all_globals = _weights_only_unpickler.get_globals_in_pkl(data_file)
            return list(all_globals.difference(safe_global_strings))


class skip_data:
    """
    Context-manager that skips writing/reading storage bytes for ``torch.save`` / ``torch.load`` calls.

    For the save path, storages will still be saved, but the space that their bytes would usually be written to
    will be empty space. The storage bytes can then be populated in a separate pass.

    For the load path, tensors will be loaded per the checkpoint but their storages will not be populated with data.

    .. warning::
        The ``skip_data`` context manager is an early prototype and is subject to change.

    Args:
        materialize_fake_tensors: Whether to materialize FakeTensors during save. This is a no-op for the load path.

    Example:
        >>> # xdoctest: +SKIP("NamedTemporaryFile on Windows")
        >>> import tempfile
        >>> t = torch.randn(2, 3)
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     with torch.serialization.skip_data():
        ...         torch.save(t, f.name)
        ...     torch.load(f.name, weights_only=True)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    """

    def __init__(self, materialize_fake_tensors: bool = False):
        self.materialize_fake_tensors = materialize_fake_tensors

    def __enter__(self):
        global _serialization_tls
        self._old_skip_data = _serialization_tls.skip_data
        self._old_materialize_fake_tensors = _serialization_tls.materialize_fake_tensors
        _serialization_tls.skip_data = True
        _serialization_tls.materialize_fake_tensors = self.materialize_fake_tensors

    def __exit__(self, type, value, tb):
        global _serialization_tls
        _serialization_tls.skip_data = self._old_skip_data
        _serialization_tls.materialize_fake_tensors = self._old_materialize_fake_tensors


def _is_zipfile(f) -> bool:
    # This is a stricter implementation than zipfile.is_zipfile().
    # zipfile.is_zipfile() is True if the magic number appears anywhere in the
    # binary. Since we expect the files here to be generated by torch.save or
    # torch.jit.save, it's safe to only check the start bytes and avoid
    # collisions and assume the zip has only 1 file.
    # See bugs.python.org/issue28494.

    start = f.tell()
    # Read the first few bytes and match against the ZIP file signature
    local_header_magic_number = b"PK\x03\x04"
    read_bytes = f.read(len(local_header_magic_number))
    f.seek(start)
    return read_bytes == local_header_magic_number


def register_package(
    priority: int,
    tagger: Callable[[STORAGE], str | None],
    deserializer: Callable[[STORAGE, str], STORAGE | None],
):
    """
    Registers callables for tagging and deserializing storage objects with an associated priority.
    Tagging associates a device with a storage object at save time while deserializing moves a
    storage object to an appropriate device at load time. :attr:`tagger` and :attr:`deserializer`
    are run in the order given by their :attr:`priority` until a tagger/deserializer returns a
    value that is not `None`.

    To override the deserialization behavior for a device in the global registry, one can register a
    tagger with a higher priority than the existing tagger.

    This function can also be used to register a tagger and deserializer for new devices.

    Args:
        priority: Indicates the priority associated with the tagger and deserializer, where a lower
            value indicates higher priority.
        tagger: Callable that takes in a storage object and returns its tagged device as a string
            or None.
        deserializer: Callable that takes in storage object and a device string and returns a storage
            object on the appropriate device or None.

    Returns:
        `None`

    Example:
        >>> def ipu_tag(obj):
        >>>     if obj.device.type == 'ipu':
        >>>         return 'ipu'
        >>> def ipu_deserialize(obj, location):
        >>>     if location.startswith('ipu'):
        >>>         ipu = getattr(torch, "ipu", None)
        >>>         assert ipu is not None, "IPU device module is not loaded"
        >>>         assert torch.ipu.is_available(), "ipu is not available"
        >>>         return obj.ipu(location)
        >>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
    """
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()


def check_module_version_greater_or_equal(
    module,
    req_version_tuple,
    error_if_malformed=True,
):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        version_strs = module.__version__.split(".")
        # Cast module version fields to match the types of the required version
        module_version = tuple(
            type(req_field)(version_strs[idx])
            for idx, req_field in enumerate(req_version_tuple)
        )
        requirement_is_met = module_version >= req_version_tuple

    except Exception as e:
        message = (
            f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared"
            f" with tuple {str(req_version_tuple)}"
        )
        if error_if_malformed:
            raise RuntimeError(message) from e
        else:
            warnings.warn(
                message + ", but continuing assuming that requirement is met",
                stacklevel=2,
            )
            requirement_is_met = True

    return requirement_is_met


def _cpu_tag(obj):
    if obj.device.type == "cpu":
        return "cpu"


def _mps_tag(obj):
    if obj.device.type == "mps":
        return "mps"


def _meta_tag(obj):
    if obj.device.type == "meta":
        return "meta"


def _backend_tag(backend_name, obj):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if obj.device.type == backend_name:
        if obj.device.index is None:
            return backend_name
        else:
            return backend_name + ":" + str(obj.device.index)


def _cpu_deserialize(obj, location):
    if location == "cpu":
        return obj


def _mps_deserialize(obj, location):
    if location.startswith("mps"):
        return obj.mps()


def _meta_deserialize(obj, location):
    if location == "meta":
        return torch.UntypedStorage(obj.nbytes(), device="meta")


def _is_meta_location(map_location):
    """
    Check if map_location specifies the meta device.

    This is used to skip reading storage data from disk when loading
    to the meta device, since meta tensors don't hold actual data.

    Args:
        map_location: The map_location argument passed to torch.load

    Returns:
        True if map_location is definitively the meta device, False otherwise.
        For dict or callable map_location, returns False since we can't
        easily determine the target device without evaluating it.
    """
    if map_location is None:
        return False
    if isinstance(map_location, str):
        return map_location == "meta"
    if isinstance(map_location, torch.device):
        return map_location.type == "meta"
    # dict or callable - can't easily determine
    return False


def _validate_device(location, backend_name):
    """
    Check whether the device index of specified backend is valid

    In case of privateuse1 backend, your must first register a device_module for
    privateuse1 using torch._register_device_module. Implement the following
    methods in device_module like cuda: device_module._utils._get_device_index(location, True),
    device_module.device_count().

    Args:
        location: string of device
        backend_name: the backend name or the name of privateuse1, which can be renamed

    Returns:
        device_index: int
    """
    if not hasattr(torch, backend_name):
        raise RuntimeError(
            f"The {backend_name.upper()} device module is not registered. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_module = getattr(torch, backend_name)
    if hasattr(device_module, "_utils") and hasattr(
        device_module._utils, "_get_device_index"
    ):
        device_index = device_module._utils._get_device_index(location, True)
        device = torch.device(backend_name, device_index)
    else:
        device = torch.device(location)
        device_index = device.index if device.index else 0
    if hasattr(device_module, "is_available") and not device_module.is_available():
        raise RuntimeError(
            f"Attempting to deserialize object on a {backend_name.upper()} "
            f"device but torch.{backend_name}.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    if hasattr(device_module, "device_count"):
        device_count = device_module.device_count()
        if device_index >= device_count:
            raise RuntimeError(
                f"Attempting to deserialize object on {backend_name.upper()} device "
                f"{device_index} but torch.{backend_name}.device_count() is {device_count}. "
                "Please use torch.load with map_location to map your storages "
                "to an existing device."
            )
    return device


def validate_cuda_device(location):
    return _validate_device(location, "cuda").index


def validate_hpu_device(location):
    return _validate_device(location, "hpu").index


def _deserialize(backend_name, obj, location):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if location == backend_name or bool(
        re.match(f"{backend_name}(:|[0-9]+)", location)
    ):
        device = _validate_device(location, backend_name)
        return obj.to(device=device)


register_package(10, _cpu_tag, _cpu_deserialize)
register_package(
    20,
    functools.partial(_backend_tag, "cuda"),
    functools.partial(_deserialize, "cuda"),
)
register_package(21, _mps_tag, _mps_deserialize)
register_package(22, _meta_tag, _meta_deserialize)
register_package(
    23,
    functools.partial(_backend_tag, "privateuse1"),
    functools.partial(_deserialize, "privateuse1"),
)
register_package(
    24,
    functools.partial(_backend_tag, "hpu"),
    functools.partial(_deserialize, "hpu"),
)
register_package(
    25,
    functools.partial(_backend_tag, "xpu"),
    functools.partial(_deserialize, "xpu"),
)
register_package(
    26,
    functools.partial(_backend_tag, "mtia"),
    functools.partial(_deserialize, "mtia"),
)


def location_tag(
    storage: Storage | torch.storage.TypedStorage | torch.UntypedStorage,
):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)
        if location:
            return location
    raise RuntimeError(
        "don't know how to determine data location of " + torch.typename(storage)
    )


def default_restore_location(storage, location):
    """
    Restores `storage` using a deserializer function registered for the `location`.

    This function looks in the registry for deserializer functions that match the `location`.
    If found, it attempts to use them, in priority order, to restore `storage` until one
    returns a not `None` result. If no deserializer can be found in the registry, or all found fail
    to bear a result, it raises a `RuntimeError`.

    Args:
        storage (STORAGE): the storage object to restore
        location (str): the location tag associated with the storage object

    Returns:
        storage: Optional[STORAGE]

    Raises:
        RuntimeError: If no deserializer matching `location` is found in the registry or if
           all matching ones return `None`.
    """
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError(
        "don't know how to restore data location of "
        + torch.typename(storage)
        + " (tagged with "
        + location
        + ")"
    )


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace("Storage", "Tensor"))


def _is_path(name_or_buffer: object) -> TypeIs[str | os.PathLike]:
    return isinstance(name_or_buffer, (str, os.PathLike))


T = TypeVar("T")


class _opener(Generic[T]):
    def __init__(self, file_like: T) -> None:
        self.file_like: T = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_file(_opener[IO[bytes]]):
    def __init__(self, name: str | os.PathLike[str], mode: str) -> None:
        super().__init__(open(name, mode))  # noqa: SIM115

    def __exit__(self, *args):
        self.file_like.close()


class _open_buffer_reader(_opener[IO[bytes]]):
    def __init__(self, buffer: IO[bytes]) -> None:
        super().__init__(buffer)
        _check_seekable(buffer)


class _open_buffer_writer(_opener[IO[bytes]]):
    def __exit__(self, *args):
        self.file_like.flush()


def _open_file_like(name_or_buffer: FileLike, mode: str) -> _opener[IO[bytes]]:
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    else:
        if "w" in mode:
            return _open_buffer_writer(name_or_buffer)
        elif "r" in mode:
            return _open_buffer_reader(name_or_buffer)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


class _open_zipfile_reader(_opener[torch._C.PyTorchFileReader]):
    def __init__(self, name_or_buffer: str | IO[bytes]) -> None:
        super().__init__(torch._C.PyTorchFileReader(name_or_buffer))


class _open_zipfile_writer_file(_opener[torch._C.PyTorchFileWriter]):
    def __init__(self, name: str) -> None:
        self.file_stream = None
        self.name = name
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # PyTorchFileWriter only supports ascii filename.
            # For filenames with non-ascii characters, we rely on Python
            # for writing out the file.
            self.file_stream = open(self.name, mode="wb")  # noqa: SIM115
            super().__init__(
                torch._C.PyTorchFileWriter(  # pyrefly: ignore  # no-matching-overload
                    self.file_stream, get_crc32_options(), _get_storage_alignment()
                )
            )
        else:
            super().__init__(
                torch._C.PyTorchFileWriter(
                    self.name, get_crc32_options(), _get_storage_alignment()
                )
            )

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener[torch._C.PyTorchFileWriter]):
    def __init__(self, buffer: IO[bytes]) -> None:
        if not callable(getattr(buffer, "write", None)):
            msg = f"Buffer of {str(type(buffer)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(
            torch._C.PyTorchFileWriter(
                buffer, get_crc32_options(), _get_storage_alignment()
            )
        )

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        self.buffer.flush()


def _open_zipfile_writer(name_or_buffer: str | IO[bytes]) -> _opener:
    container: type[_opener]
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)  # type: ignore[arg-type]


def _is_compressed_file(f) -> bool:
    compress_modules = ["gzip"]
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False


def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False


def _check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only torch.load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False


def _check_dill_version(pickle_module) -> None:
    """Checks if using dill as the pickle module, and if so, checks if it is the correct version.
    If dill version is lower than 0.3.1, a ValueError is raised.

    Args:
        pickle_module: module used for pickling metadata and objects

    """
    if pickle_module is not None and pickle_module.__name__ == "dill":
        required_dill_version = (0, 3, 1)
        if not check_module_version_greater_or_equal(
            pickle_module, required_dill_version, False
        ):
            raise ValueError(
                (
                    "'torch' supports dill >= {}, but you have dill {}."
                    " Please upgrade dill or switch to 'pickle'"
                ).format(
                    ".".join([str(num) for num in required_dill_version]),
                    pickle_module.__version__,
                )
            )


def _check_save_filelike(f):
    if not _is_path(f) and not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute"
        )


def save(
    obj: object,
    f: FileLike,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    See :ref:`layout-control` for more advanced tools to manipulate a checkpoint.

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # xdoctest: +SKIP("makes cwd dirty")
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, "tensor.pt")
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    torch._C._log_api_usage_once("torch.save")
    _check_dill_version(pickle_module)
    _check_save_filelike(f)

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    if _use_new_zipfile_serialization:
        with _open_zipfile_writer(f) as opened_zipfile:
            _save(
                obj,
                opened_zipfile,
                pickle_module,
                pickle_protocol,
                _disable_byteorder_record,
            )
            return
    else:
        global _serialization_tls
        if _serialization_tls.skip_data:
            raise RuntimeError(
                "Cannot use skip_data=True with _use_new_zipfile_serialization=False"
            )
        with _open_file_like(f, "wb") as opened_file:
            _legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    import torch.nn as nn

    serialized_container_types = {}
    serialized_storages: dict[str, tuple[torch.UntypedStorage, torch.dtype]] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: dict[int, torch.dtype] = {}

    def persistent_id(obj: Any) -> tuple | None:
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = "".join(source_lines)
            except (
                Exception
            ):  # saving the source is optional, so we can ignore any errors
                warnings.warn(
                    "Couldn't retrieve source code for container of "
                    "type " + obj.__name__ + ". It won't be checked "
                    "for correctness upon loading.",
                    stacklevel=2,
                )
            return ("module", obj, source_file, source)

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            storage: torch.UntypedStorage

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj._size()

            elif isinstance(obj, torch.UntypedStorage):
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                dtype = torch.uint8
                storage_numel = storage.nbytes()
            else:
                raise TypeError(f"type not recognized: {type(obj)}")

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            view_metadata: tuple[str, int, int] | None

            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            storage_key = str(storage._cdata)
            location = location_tag(storage)

            # TODO: There's an issue here with FC. It might be impossible to
            # solve, but it's worth noting. Imagine we save a list `[storage,
            # tensor]`, where `tensor.storage()` is the same as `storage`, and
            # `tensor.element_size() > 1`. Let's say that `tensor.dtype ==
            # torch.float`.  The storage will be serialized with element size
            # of 1, since we're choosing to serialize the first occurrence of
            # a duplicate storage. Since this legacy serialization format saves
            # the numel of the storage, rather than nbytes directly, we'll be
            # effectively saving nbytes in this case.  We'll be able to load it
            # and the tensor back up with no problems in _this_ and future
            # versions of pytorch, but in older versions, here's the problem:
            # the storage will be loaded up as a UntypedStorage, and then the
            # FloatTensor will loaded and the UntypedStorage will be assigned to
            # it. Since the storage dtype does not match the tensor dtype, this
            # will cause an error.  If we reverse the list, like `[tensor,
            # storage]`, then we will save the `tensor.storage()` as a faked
            # `FloatStorage`, and the saved size will be the correct
            # dtype-specific numel count that old versions expect. `tensor`
            # will be able to load up properly in old versions, pointing to
            # a FloatStorage. However, `storage` is still being translated to
            # a UntypedStorage, and it will try to resolve to the same
            # FloatStorage that `tensor` contains. This will also cause an
            # error. It doesn't seem like there's any way around this.
            # Probably, we just cannot maintain FC for the legacy format if the
            # saved list contains both a tensor and a storage that point to the
            # same data.  We should still be able to maintain FC for lists of
            # just tensors, as long as all views share the same dtype as the
            # tensor they are viewing.

            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (storage, dtype)
            is_view = storage._cdata != storage._cdata
            if is_view:
                view_metadata = (str(storage._cdata), offset, storage.nbytes())
            else:
                view_metadata = None

            res = (
                "storage",
                storage_type,
                storage_key,
                location,
                storage_numel,
                view_metadata,
            )
            return res
        return None

    sys_info = {
        "protocol_version": PROTOCOL_VERSION,
        "little_endian": sys.byteorder == "little",
        "type_sizes": {
            "short": SHORT_SIZE,
            "int": INT_SIZE,
            "long": LONG_SIZE,
        },
    }

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)

    class PyTorchLegacyPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)  # noqa: F821

    pickler = PyTorchLegacyPickler(f, protocol=pickle_protocol)
    pickler.dump(obj)

    # The class def keeps the persistent_id closure alive, leaking memory.
    del persistent_id

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        storage, dtype = serialized_storages[key]
        storage._write_file(
            f, _should_read_directly(f), True, torch._utils._element_size(dtype)
        )


def _save(
    obj,
    zip_file,
    pickle_module,
    pickle_protocol,
    _disable_byteorder_record,
):
    serialized_storages: dict[str, torch.storage.UntypedStorage] = {}
    id_map: dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: dict[int, torch.dtype] = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(obj, "_fake_device") and obj._fake_device is not None:
                location = str(obj._fake_device)
            else:
                location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()

    class PyTorchPickler(pickle_module.Pickler):  # type: ignore[name-defined]
        def persistent_id(self, obj):
            return persistent_id(obj)  # noqa: F821

    pickler = PyTorchPickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)

    # The class def keeps the persistent_id closure alive, leaking memory.
    del persistent_id

    data_value = data_buf.getvalue()
    zip_file.write_record("data.pkl", data_value, len(data_value))
    # .format_version is used to track
    #     1. version 1 represents the order of storages being changed from
    #        lexicographical based on keys to numerically ordered based on keys
    #     2. version 2 represents including storage_alignment as a record
    #        within the zipfile
    zip_file.write_record(".format_version", "1", len("1"))
    storage_alignment = str(_get_storage_alignment())
    zip_file.write_record(
        ".storage_alignment", storage_alignment, len(storage_alignment)
    )

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in serialized_storages:
        name = f"data/{key}"
        storage = serialized_storages[key]
        num_bytes = storage.nbytes()
        global _serialization_tls
        if _serialization_tls.skip_data:
            zip_file.write_record_metadata(name, num_bytes)
        else:
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != "cpu":
                from torch.utils.serialization import config

                if (
                    config.save.use_pinned_memory_for_d2h
                    and (
                        acc := torch.accelerator.current_accelerator(
                            check_available=True
                        )
                    )
                    is not None
                    and acc.type == storage.device.type
                ):
                    new_storage = torch.empty(
                        num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                    ).untyped_storage()
                    new_storage.copy_(storage)
                    torch.accelerator.current_stream(storage.device.index).synchronize()
                    storage = new_storage
                else:
                    storage = storage.cpu()
            # Now that it is on the CPU we can directly copy it into the zip file
            zip_file.write_record(name, storage, num_bytes)


def load(
    f: FileLike,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool | None = None,
    mmap: bool | None = None,
    **pickle_load_args: Any,
) -> Any:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

    .. warning::
        :func:`torch.load()` uses an unpickler under the hood. **Never load data from an untrusted source.**

        See :ref:`weights-only-security` for more details.

    :func:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``'cpu'``
    for CPU tensors and ``'cuda:device_id'`` (e.g. ``'cuda:2'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn't specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    See :ref:`layout-control` for more advanced tools to manipulate a checkpoint.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        weights_only: Indicates whether unpickler should be restricted to
            loading only tensors, primitive types, dictionaries
            and any types added via :func:`torch.serialization.add_safe_globals`.
            See :ref:`weights-only` for more details. When ``weights_only=True``
            and the checkpoint contains sparse tensors, their invariants (e.g.
            index bounds) are always validated to prevent malformed indices from
            causing out-of-bounds reads later; this is an O(nnz) scan per sparse
            tensor and may be slow for large checkpoints.
        mmap: Indicates whether the file should be mapped rather than loading all the storages into memory.
            Typically, tensor storages in the file will first be moved from disk to CPU memory, after which they
            are moved to the location that they were tagged with when saving, or specified by ``map_location``. This
            second step is a no-op if the final location is CPU. When the ``mmap`` flag is set, instead of copying the
            tensor storages from disk to CPU memory in the first step, ``f`` is mapped, which means tensor storages
            will be lazily loaded when their data is accessed.
        pickle_load_args: optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`,
            only works if :attr:`weights_only=False`, e.g., :attr:`errors=...`.

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding='latin1'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding='bytes'` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.

    Example:
        >>> # xdoctest: +SKIP("undefined filepaths")
        >>> torch.load("tensors.pt", weights_only=True)
        # Load all tensors onto the CPU
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=torch.device("cpu"),
        ...     weights_only=True,
        ... )
        # Load all tensors onto the CPU, using a function
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=lambda storage, loc: storage,
        ...     weights_only=True,
        ... )
        # Load all tensors onto GPU 1
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=lambda storage, loc: storage.cuda(1),  # type: ignore[attr-defined]
        ...     weights_only=True,
        ... )  # type: ignore[attr-defined]
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location={"cuda:1": "cuda:0"},
        ...     weights_only=True,
        ... )
        # Load tensor from io.BytesIO object
        # Loading from a buffer setting weights_only=False, warning this can be unsafe
        >>> with open("tensor.pt", "rb") as f:
        ...     buffer = io.BytesIO(f.read())
        >>> torch.load(buffer, weights_only=False)
        # Load a module with 'ascii' encoding for unpickling
        # Loading from a module setting weights_only=False, warning this can be unsafe
        >>> torch.load("module.pt", encoding="ascii", weights_only=False)
    """
    torch._C._log_api_usage_once("torch.load")
    DOCS_MESSAGE = (
        "\n\nCheck the documentation of torch.load to learn more about types accepted by default with "
        "weights_only https://pytorch.org/docs/stable/generated/torch.load.html."
    )

    def _get_wo_message(message: str) -> str:
        unsafe_global_pattern = r"GLOBAL (\S+) was not an allowed global by default."
        has_unsafe_global = re.search(unsafe_global_pattern, message) is not None
        blocklist_pattern = r"whose module (\S+) is blocked"
        has_blocklist = re.search(blocklist_pattern, message) is not None
        import_pattern = r"(\S+) must be (\S+) to load"
        has_import = re.search(import_pattern, message) is not None
        if has_unsafe_global:
            updated_message = (
                "Weights only load failed. This file can still be loaded, to do so you have two options, "
                "\033[1mdo those steps only if you trust the source of the checkpoint\033[0m. "
                f"\n\t(1) {UNSAFE_MESSAGE}\n\t(2) Alternatively, to load with `weights_only=True` please check "
                "the recommended steps in the following error message.\n\tWeightsUnpickler error: "
                + message
            )
        else:
            if has_import:
                return f"Weights only load failed. {message}\n {UNSAFE_MESSAGE}\n"
            else:
                updated_message = f"Weights only load failed. {UNSAFE_MESSAGE}\n"
                if not has_blocklist:
                    updated_message += (
                        "Please file an issue with the following so that we can make "
                        "`weights_only=True` compatible with your use case: WeightsUnpickler error: "
                    )
            updated_message += "\n\n" + message
        return updated_message + DOCS_MESSAGE

    weights_only_not_set = weights_only is None

    if weights_only_not_set:
        weights_only = _default_to_weights_only(pickle_module)

    true_values = ["1", "y", "yes", "true"]
    # Add ability to force safe only or non-safe weight loads via environment variables
    force_weights_only_load = (
        os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in true_values
    )
    force_no_weights_only_load = (
        os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0").lower() in true_values
    )

    if force_weights_only_load and force_no_weights_only_load:
        raise RuntimeError(
            "Only one of `TORCH_FORCE_WEIGHTS_ONLY_LOAD` or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` "
            "should be set, but both were set."
        )
    elif force_weights_only_load:
        weights_only = True
    elif force_no_weights_only_load:
        # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD can only override if callsite did not explicitly set weights_only
        if weights_only_not_set:
            warnings.warn(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the"
                "`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.",
                UserWarning,
                stacklevel=2,
            )
            weights_only = False

    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
    else:
        if pickle_module is None:
            pickle_module = pickle

    if pickle_load_args != {} and weights_only:
        warnings.warn("pickle_load_args only works if `weights_only=False`.")

    # make flipping default BC-compatible
    if mmap is None:
        from torch.utils.serialization import config

        mmap = config.load.mmap

    _check_dill_version(pickle_module)

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    # Check if the file is a safetensors file
    if _is_path(f):
        fspath = os.fspath(f)
        if fspath.endswith(".safetensors"):
            try:
                import safetensors.torch  # type: ignore[import-not-found]

                # Convert map_location to a device string for safetensors
                device = "cpu"
                if map_location is not None:
                    if isinstance(map_location, (str, bytes)):
                        device = str(map_location)
                    elif isinstance(map_location, torch.device):
                        device = str(map_location)
                    elif isinstance(map_location, dict):
                        raise RuntimeError(
                            "Loading safetensors file with dict map_location is not supported. "
                            "Use a device string or torch.device instead."
                        )
                    elif callable(map_location):
                        raise RuntimeError(
                            "Loading safetensors file with callable map_location is not supported. "
                            "Use a device string or torch.device instead."
                        )

                return safetensors.torch.load_file(fspath, device=device)
            except ImportError as e:
                raise RuntimeError(
                    "Attempting to load a safetensors file but safetensors is not installed. "
                    "Please install safetensors with: pip install safetensors"
                ) from e

    with _open_file_like(f, "rb") as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            overall_storage = None
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "'torch.load' received a zip file that looks like a TorchScript archive"
                        " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                        " silence this warning)",
                        UserWarning,
                        stacklevel=2,
                    )
                    if weights_only:
                        raise RuntimeError(
                            "Cannot use ``weights_only=True`` with TorchScript archives passed to "
                            "``torch.load``. " + UNSAFE_MESSAGE
                        )
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if mmap:
                    if not _is_path(f):
                        raise ValueError(
                            "f must be a file path in order to use the mmap argument"
                        )
                    size = os.path.getsize(f)
                    if not IS_WINDOWS:
                        shared = get_default_mmap_options() == MAP_SHARED
                    else:
                        shared = False
                    overall_storage = torch.UntypedStorage.from_file(
                        os.fspath(f),
                        shared,
                        size,
                    )
                if weights_only:
                    try:
                        return _load(
                            opened_zipfile,
                            map_location,
                            _weights_only_unpickler,
                            overall_storage=overall_storage,
                            weights_only=True,
                            **pickle_load_args,
                        )
                    except pickle.UnpicklingError as e:
                        raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
                return _load(
                    opened_zipfile,
                    map_location,
                    pickle_module,
                    overall_storage=overall_storage,
                    weights_only=False,
                    **pickle_load_args,
                )
        if mmap:
            f_name = "" if not isinstance(f, str) else f"{f}, "
            raise RuntimeError(
                "mmap can only be used with files saved with "
                f"`torch.save({f_name}_use_new_zipfile_serialization=True), "
                "please torch.save your checkpoint with this option in order to use mmap."
            )
        if weights_only:
            try:
                return _legacy_load(
                    opened_file,
                    map_location,
                    _weights_only_unpickler,
                    weights_only=True,
                    **pickle_load_args,
                )
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
        return _legacy_load(
            opened_file,
            map_location,
            pickle_module,
            weights_only=False,
            **pickle_load_args,
        )


# Register pickling support for layout instances such as
# torch.sparse_coo, etc
def _get_layout(name):
    """Get layout extension object from its string representation."""
    cache = _get_layout.cache  # type: ignore[attr-defined]
    if not cache:
        for v in torch.__dict__.values():
            if isinstance(v, torch.layout):
                cache[str(v)] = v
    return cache[name]


# There are yet not good way to type annotate function attributes https://github.com/python/mypy/issues/2087
_get_layout.cache = {}  # type: ignore[attr-defined]
copyreg.pickle(torch.layout, lambda obj: (_get_layout, (str(obj),)))


def _legacy_load(
    f,
    map_location,
    pickle_module,
    *,
    weights_only=False,
    **pickle_load_args,
):
    deserialized_objects: dict[int, Any] = {}

    restore_location = _get_restore_location(map_location)

    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            return super().find_class(mod_name, name)

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading.",
                stacklevel=2,
            )
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + ".patch"
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except OSError:
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            warnings.warn(msg, SourceChangeWarning, stacklevel=2)

    def legacy_load(f):
        deserialized_objects: dict[int, Any] = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with (
            closing(
                tarfile.open(fileobj=f, mode="r:", format=tarfile.PAX_FORMAT)
            ) as tar,
            mkdtemp() as tmpdir,
        ):
            if pickle_module is _weights_only_unpickler:
                raise RuntimeError(
                    "Cannot use ``weights_only=True`` with files saved in the "
                    "legacy .tar format. " + UNSAFE_MESSAGE
                )
            tar.extract("storages", path=tmpdir)
            with open(os.path.join(tmpdir, "storages"), "rb", 0) as f:
                num_storages = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    dtype = storage_type._dtype
                    obj = cast(Storage, torch.UntypedStorage)._new_with_file(
                        f, torch._utils._element_size(dtype)
                    )
                    obj = restore_location(obj, location)
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[key] = torch.storage.TypedStorage(
                        wrap_storage=obj, dtype=dtype, _internal=True
                    )

                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, numel in storage_views:
                    root = deserialized_objects[root_cdata]
                    element_size = torch._utils._element_size(root.dtype)
                    offset_bytes = offset * element_size
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[target_cdata] = torch.storage.TypedStorage(
                        wrap_storage=root._untyped_storage[
                            offset_bytes : offset_bytes + numel * element_size
                        ],
                        dtype=root.dtype,
                        _internal=True,
                    )

            tar.extract("tensors", path=tmpdir)
            with open(os.path.join(tmpdir, "tensors"), "rb", 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, _original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    (ndim,) = struct.unpack("<i", f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    numel = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    stride = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    (storage_offset,) = struct.unpack("<q", f.read(8))
                    tensor = torch.empty((0,), dtype=storage.dtype).set_(
                        storage._untyped_storage, storage_offset, numel, stride
                    )
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile("pickle")
            unpickler = UnpicklerWrapper(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    deserialized_objects = {}

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple):
            raise AssertionError(
                f"saved_id must be a tuple, got {type(saved_id).__name__}"
            )
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == "storage":
            storage_type, root_key, location, numel, view_metadata = data
            location = _maybe_decode_ascii(location)
            dtype = storage_type.dtype

            nbytes = numel * torch._utils._element_size(dtype)

            if root_key not in deserialized_objects:
                if torch._guards.active_fake_mode() is not None:
                    obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
                elif _serialization_tls.skip_data:
                    obj = cast(Storage, torch.UntypedStorage(nbytes))
                    obj = restore_location(obj, location)
                else:
                    obj = cast(Storage, torch.UntypedStorage(nbytes))
                    obj._torch_load_uninitialized = True
                    obj = restore_location(obj, location)
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                typed_storage = torch.storage.TypedStorage(
                    wrap_storage=obj, dtype=dtype, _internal=True
                )
                deserialized_objects[root_key] = typed_storage
            else:
                typed_storage = deserialized_objects[root_key]
                if typed_storage._data_ptr() == 0:
                    typed_storage = torch.storage.TypedStorage(
                        device=typed_storage._untyped_storage.device,
                        dtype=dtype,
                        _internal=True,
                    )

            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                offset_bytes = offset * torch._utils._element_size(dtype)
                view_size_bytes = view_size * torch._utils._element_size(dtype)
                if view_key not in deserialized_objects:
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[view_key] = torch.storage.TypedStorage(
                        wrap_storage=typed_storage._untyped_storage[
                            offset_bytes : offset_bytes + view_size_bytes
                        ],
                        dtype=dtype,
                        _internal=True,
                    )
                res = deserialized_objects[view_key]

            else:
                res = typed_storage
            return res
        else:
            raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error here
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                ) from None
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    if torch._guards.active_fake_mode() is None and not _serialization_tls.skip_data:
        offset = f.tell() if f_should_read_directly else None
        for key in deserialized_storage_keys:
            if key not in deserialized_objects:
                raise AssertionError(
                    f"storage key {key!r} not found in deserialized_objects"
                )
            typed_storage = deserialized_objects[key]
            typed_storage._untyped_storage._set_from_file(
                f,
                offset,
                f_should_read_directly,
                torch._utils._element_size(typed_storage.dtype),
            )
            if offset is not None:
                offset = f.tell()

    torch._utils._validate_loaded_sparse_tensors(weights_only=weights_only)

    return result


def _maybe_decode_ascii(bytes_str: bytes | str) -> str:
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


def _get_restore_location(map_location):
    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):

        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)

    elif isinstance(map_location, (str, bytes)):

        def restore_location(storage, location):
            return default_restore_location(storage, map_location)

    elif isinstance(map_location, torch.device):

        def restore_location(storage, location):
            return default_restore_location(storage, str(map_location))

    else:

        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result

    return restore_location


class StorageType:
    def __init__(self, name):
        self._dtype = _get_dtype_from_pickle_storage_type(name)

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _load(
    zip_file,
    map_location,
    pickle_module,
    pickle_file="data.pkl",
    overall_storage=None,
    *,
    weights_only=False,
    **pickle_load_args,
):
    restore_location = _get_restore_location(map_location)

    loaded_storages = {}

    is_meta_map_location = _is_meta_location(map_location)

    can_calculate_storage_offsets = False
    if zip_file.has_record(".format_version"):
        version = zip_file.get_record(".format_version")
        can_calculate_storage_offsets = version >= b"1"

    # check if byteswapping is needed
    byteordername = "byteorder"
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b"little", b"big"]:
            raise ValueError("Unknown endianness type: " + byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError("Invalid load endianness type")

    storage_alignment = 64
    if zip_file.has_record(".storage_alignment"):
        storage_alignment = int(zip_file.get_record(".storage_alignment"))

    if (
        not zip_file.has_record(byteordername)
        and get_default_load_endianness() is None
        and sys.byteorder == "big"
    ):
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn(
            "The default load endianness for checkpoints without a byteorder mark "
            "on big endian machines was changed from 'native' to 'little' endian, "
            "to avoid this behavior please use "
            "torch.serialization.set_default_load_endianness to set "
            "the desired default load endianness",
            UserWarning,
            stacklevel=2,
        )

    from torch.utils.serialization import config

    calculate_storage_offsets = config.load.calculate_storage_offsets
    run_debug_asserts = os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1"
    current_offset = None
    # constants from miniz.h/miniz.c
    data_descripter_size64 = 24
    data_descripter_size32 = 16
    mz_uint32_max = 0xFFFFFFFF
    offsets: dict[str, int] = dict()

    def _get_offset(key, name, numel):
        """
        Return the offset of the storage associated with key with record name `name` and size numel.
        It is expected that the zipfile header of this storage starts at current_offset.

        WARNING: This function relies on the behavior of the zipwriter in miniz.c. In particular,
        the behavior of `mz_zip_writer_add_mem_ex_v2`. The behavior of this function must be kept
        in sync with that of miniz!

        After reading a storage of size numel that starts at storage_offset
        if it is the first time that storage was read, update nonlocal variable
        current_offset to the start of the next zipfile header by incrementing
        it by numel and the data descriptor size.
        """
        nonlocal current_offset, offsets
        if name in offsets:
            storage_offset = offsets[name]
            return storage_offset

        if current_offset is None:
            if key != "0":
                raise AssertionError(f"expected key '0', got {key!r}")
            current_offset = zip_file.get_record_offset(name)
            local_header_offset = zip_file.get_record_header_offset(name)
            storage_offset = current_offset
        else:
            storage_offset = zip_file.get_record_offset_no_read(
                current_offset, name, numel, storage_alignment
            )
            local_header_offset = current_offset

        # This is only actually needed for storages that have typed_storage._data_ptr() == 0
        # after being read. Otherwise persistent_load would never "re-call" load_tensor
        # for a given key.
        offsets[name] = storage_offset

        # Increment current_offset to offset where next zipfile header starts
        current_offset = storage_offset + numel
        # add size of data descriptor after payload
        if numel > 0:
            if local_header_offset >= mz_uint32_max or numel >= mz_uint32_max:
                current_offset += data_descripter_size64
            else:
                current_offset += data_descripter_size32

        return storage_offset

    def load_tensor(dtype, nbytes, key, location):
        name = f"data/{key}"
        if torch._guards.detect_fake_mode(None) is not None or is_meta_map_location:
            storage = torch.UntypedStorage(nbytes, device="meta")
            if can_calculate_storage_offsets:
                storage._checkpoint_offset = _get_offset(key, name, nbytes)
            else:
                storage._checkpoint_offset = zip_file.get_record_offset(name)
        elif _serialization_tls.skip_data:
            storage = torch.UntypedStorage(nbytes)
        elif overall_storage is not None:
            if can_calculate_storage_offsets and calculate_storage_offsets:
                storage_offset = _get_offset(key, name, nbytes)
                if run_debug_asserts:
                    if storage_offset != zip_file.get_record_offset(name):
                        raise RuntimeError(
                            "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                            f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                            f"{zip_file.get_record_offset(name)}"
                        )
            else:
                storage_offset = zip_file.get_record_offset(name)
            storage = overall_storage[storage_offset : storage_offset + nbytes]
        else:
            if can_calculate_storage_offsets and run_debug_asserts:
                # This is debug code that we use to test the validity of
                # torch.utils.serialization.config.load.calculate_storage_offsets throughout CI
                storage_offset = _get_offset(key, name, nbytes)
                if storage_offset != zip_file.get_record_offset(name):
                    raise RuntimeError(
                        "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                        f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                        f"{zip_file.get_record_offset(name)}"
                    )
            storage = (
                zip_file.get_storage_from_record(name, nbytes, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # swap here if byteswapping is needed
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage

        if is_meta_map_location:
            # Skip restore_location for meta map_location. Since we already created
            # a meta storage above, calling restore_location would just redundantly
            # call _meta_deserialize which creates another meta storage with the same
            # size.
            wrap_storage = storage
        elif torch._guards.detect_fake_mode(None) is None:
            wrap_storage = restore_location(storage, location)
        else:
            storage._fake_device = location
            wrap_storage = storage

        typed_storage = torch.storage.TypedStorage(
            wrap_storage=wrap_storage,
            dtype=dtype,
            _internal=True,
        )

        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage

        return typed_storage

    def persistent_load(saved_id):
        if not isinstance(saved_id, tuple):
            raise AssertionError(
                f"saved_id must be a tuple, got {type(saved_id).__name__}"
            )
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename != "storage":
            raise AssertionError(
                f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            )
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = load_tensor(
                dtype, nbytes, key, _maybe_decode_ascii(location)
            )

        return typed_storage

    load_module_mapping: dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record(pickle_file))

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    # Needed for tensors where storage device and rebuild tensor device are
    # not connected (wrapper subclasses and tensors rebuilt using numpy)
    global _serialization_tls
    _serialization_tls.map_location = map_location
    result = unpickler.load()
    _serialization_tls.map_location = None

    torch._utils._validate_loaded_sparse_tensors(weights_only=weights_only)
    torch._C._log_api_usage_metadata(
        "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
    )
    return result


def _is_torchscript_zip(zip_file):
    return "constants.pkl" in zip_file.get_all_records()
<|endoftext|># mypy: allow-untyped-defs
"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import ast
import builtins
import collections
import contextlib
import enum
import functools
import inspect
import io
import pickle
import sys
import textwrap
import threading
import types
import typing
import warnings
import weakref
from typing import (  # noqa: UP035, F401  # (Dict, List, Tuple) imported by torch.jit.annotations
    Any,
    Callable,
    Dict,
    Final,
    ForwardRef,
    get_args,
    get_origin,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec

import torch

# This is needed. `torch._jit_internal` is imported before `torch.distributed.__init__`.
# Explicitly ask to import `torch.distributed.__init__` first.
# Otherwise, "AttributeError: module 'torch' has no attribute 'distributed'" is raised.
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future


class HasGetattr(Protocol):
    def __getattr__(self, key: str) -> Any: ...


_P = ParamSpec("_P")
_R = TypeVar("_R")

BuiltinUnionType: type | tuple[type, ...] = types.UnionType

LockType: type
try:
    import _thread

    LockType = _thread.LockType
except ImportError:
    import _dummy_thread  # type: ignore[import-not-found]

    LockType = _dummy_thread.LockType

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


FAKE_FILENAME_PREFIX = "__torch_jit_dataclass"


def is_final(ann) -> bool:
    return (
        hasattr(ann, "__module__")
        and ann.__module__ in {"typing", "typing_extensions"}
        and (get_origin(ann) is Final or isinstance(ann, type(Final)))
    )


# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls:
    def __getitem__(self, types):
        return


# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1


def is_scripting() -> bool:
    r"""
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
            if torch.jit.is_scripting():
                return torch.linear(x)
            else:
                return unsupported_linear_op(x)
    """
    return False


# Retrieves a fully-qualified name (module hierarchy + classname) for a given obj.
def _qualified_name(obj, mangle_name=True) -> str:
    # This special case allows us to override the qualified name on a type.
    # It's currently used in conjunction with tracing, where we create a
    # fake module to filter only supported attributes. However, since this
    # new type is defined as a local class, we need a mechanism to override
    # its qualname so it appears correctly in the TorchScript system. This,
    # we set '_jit_override_qualname' with the original traced module's
    # qualified name, which is picked up here
    if hasattr(obj, "_jit_override_qualname"):
        return obj._jit_override_qualname
    # short-circuit in cases where the object already has a known qualified name
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name

    if getattr(obj, "__name__", None):
        name = obj.__name__
    # Enum classes do not have `__name__` attr, instead they have `name`.
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError("Could not get name of python class object")

    if name == "<lambda>":
        name = "_lambda"  # make name a valid identifier

    module_name = obj.__module__

    # If the module is actually a torchbind module, then we should short circuit
    if module_name == "torch._classes":
        return obj.qualified_name

    # The Python docs are very clear that `__module__` can be None, but I can't
    # figure out when it actually would be.
    if module_name is None:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            "__module__ can't be None."
        )

    # if getattr(sys.modules[module_name], name) is not obj:
    #     raise RuntimeError(f"Could not get qualified name for class '{name}': "
    #                        f"the attr {name} on module {module_name} is not the class")

    # torch.package and TorchScript have separate mangling schemes to avoid
    # name collisions from multiple packages. To avoid them interfering with
    # each other, normalize the package managing here.
    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace("<", "_")
        module_name = module_name.replace(">", "_")

    # The PythonExceptionValue C++ class in torch/csrc/jit/python/python_sugared_value.h
    # does not need mangle the python class name.
    if mangle_name:
        # __main__ is a builtin module, so rewrite it to "__torch__".
        if module_name == "__main__":
            module_name = "__torch__"
        else:
            # Everything else gets a "__torch__" prefix to avoid name collisions
            # with the names of user values.
            module_name = "__torch__." + module_name

    if "." in name:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            f"'{name}' is not a valid identifier"
        )

    return module_name + "." + name


class SourceLoader:
    def __init__(self):
        self.content = {}

    def cache(self, fn, source):
        self.content[fn] = source

    def get_source(self, fn):
        return self.content.get(fn)


loader = SourceLoader()


def createResolutionCallbackFromEnv(lookup_base: HasGetattr) -> Callable[[str], Any]:
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """

    def lookupInModule(qualified_name: str, module: Any) -> Any:
        if "." in qualified_name:
            base, remaining_pieces = qualified_name.split(".", maxsplit=1)
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

    def parseNestedExpr(expr: str, module: Any) -> tuple[Any, int]:
        i = 0
        while i < len(expr) and expr[i] not in (",", "[", "]"):
            i += 1

        # Special case logic for the empty Tuple as a subscript (used
        # in the type annotation `Tuple[()]`)
        if expr[:i] == "()":
            return (), i

        base = lookupInModule(expr[:i].strip(), module)
        if base is None:
            raise AssertionError(f"Unresolvable type {expr[:i]}")
        if i == len(expr) or expr[i] != "[":
            return base, i

        if expr[i] != "[":
            raise AssertionError(f"expected '[' at position {i}, got {expr[i]!r}")
        parts = []
        while expr[i] != "]":
            part_len = 0
            i += 1
            part, part_len = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        if len(parts) > 1:
            return base[tuple(parts)], i + 1
        else:
            return base[parts[0]], i + 1

    def parseExpr(expr: str, module: Any) -> Any:
        try:
            value, len_parsed = parseNestedExpr(expr, module)
            if len_parsed != len(expr):
                raise AssertionError(
                    "whole expression was not parsed, falling back to c++ parser"
                )
            return value
        except Exception:
            """
            The python resolver fails in several cases in known unit tests, and is intended
            to fall back gracefully to the c++ resolver in general.  For example, python 2 style
            annotations which are frequent in our unit tests often fail with types e.g. int not
            resolvable from the calling frame.
            """
            return None

    return lambda expr: parseExpr(expr, lookup_base)


def createResolutionCallbackFromFrame(frames_up: int = 0) -> Callable[[str], Any]:
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))


        def baz():
            foo = 2
            bar()


        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        if frame is None:
            raise AssertionError(f"frame is None at iteration {i}")
        frame = frame.f_back
        i += 1

    if frame is None:
        raise AssertionError("frame is None after traversing frames_up")
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    class env:
        def __getattr__(self, key: str) -> Any:
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)

    return createResolutionCallbackFromEnv(env())


def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures


# [local resolution in python]
# Depending on where a variable is defined, and where it is used, we may
# or may not be able to recover its value when recursively compiling a
# script function. Remember in the general case, a module or function is
# first defined and then later scripted. This means we do not have a
# chance to capture the active frames when the function is defined. Hence any
# name resolution has to happen later on the created closure. The way
# python captures type annotations restricts what we can recover. The
# follow example illustrates the different cases:
#
#         class MyGlobalClass:
#         ...
#         def my_local_scope():
#             @torch.jit.script
#             class MyClass:
#                 ...
#             @torch.jit.script
#             class MyClassUsedAsVar:
#                 ...
#             def eg(x: MyClass, y: MyGlobalClass):
#                 a_local_capture : Foo
#                 return MyClassUsedAsVar(x)
#
# MyGlobalClass is defined in the __globals__ dictionary of function
# 'eg', so it is always recoverable. my_local_scope introduces a new local
# variable scope in the function. Classes defined here are only visible as
# local variables. For the case of MyClassUsedAsVar, it is captured
# because it is used as a variable inside the body of the function, and we
# can resolve it using the captures returned from `get_closure`. However,
# the type annotations are not captured by the closure. In Python
# 3.0--3.9, the _value_ of MyClass and MyGlobalClass will be available as
# annotations on `eg``, but starting in Python 4.0, they will represented as
# strings and no longer present. Furthermore, since the body of `eg` does
# not reference those names, they do not appear in the list of closed over
# variables. We anticipate
# that most users will not run into this issue because their modules and
# functions will be defined at a global scope like MyGlobalClass. In cases
# where they are not, it is possible to work around issues by declaring the
# values global in the function.
# In Python 3.9 declaring class as global will make it invisible to
# `inspect.getsource`, see https://bugs.python.org/issue42666 .
# This could be worked around by manually adding it to `global()` dictionary.


def createResolutionCallbackFromClosure(fn) -> Callable[[str], Any]:
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    closure = get_closure(fn)

    class closure_lookup:
        # This is a class since `closure` is a dict and it's easier in
        # `env_helper` if everything just works with `getattr` calls
        def __getattr__(self, key: str) -> Any:
            if key in closure:
                return closure[key]
            elif hasattr(typing, key):
                return getattr(typing, key)
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            return None

    return createResolutionCallbackFromEnv(closure_lookup())


@functools.cache
def can_compile_class(cls) -> bool:
    # If any of the functions on a type don't have a code object, this type can't
    # be compiled and is probably a builtin / bound from C
    if is_ignored_fn(cls):
        return False

    # Ignore the following list of built-in classes.
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False

    names = cls.__dict__
    fns = [
        getattr(cls, name)
        for name in names
        if inspect.isroutine(getattr(cls, name, None))
    ]
    has_code = [hasattr(fn, "__code__") for fn in fns]
    return all(has_code)


def get_callable_argument_names(fn) -> list[str]:
    """
    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.
    Returns an empty list when other types of arguments are present.

    This is used by `torch.jit.trace` to assign meaningful argument names to
    traced functions and modules.

    Args:
        fn: A callable.
    Returns:
        Argument names: List[str]
    """
    # inspect.signature may fail, give up in that case.
    try:
        callable_signature = inspect.signature(fn)
    except Exception:
        return []

    argument_names = []
    for name, param in callable_signature.parameters.items():
        # All four other types of arguments do not map to individual values
        # with a keyword as name.
        if param.kind != param.POSITIONAL_OR_KEYWORD:
            continue

        argument_names.append(name)

    return argument_names


def get_annotation_str(annotation):
    """
    Convert an AST node containing a type annotation to the string present in the source
    that represents the same annotation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Attribute):
        return ".".join([get_annotation_str(annotation.value), annotation.attr])
    elif isinstance(annotation, ast.Subscript):
        # In Python3.9+ subscript indices are not wrapped in ast.Index
        subscript_slice = annotation.slice
        return f"{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]"
    elif isinstance(annotation, ast.Tuple):
        return ",".join([get_annotation_str(elt) for elt in annotation.elts])
    elif isinstance(annotation, ast.Constant):
        return f"{annotation.value}"

    # If an AST node is not handled here, it's probably handled in ScriptTypeParser.
    return None


def get_type_hint_captures(fn):
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
    # First, try to get the source of the function. We'll need to parse it to find the actual string names
    # that were used to annotate the types, since inspect.signature() will only return the class object that
    # the annotation refers to, not the string name. If we can't get the source, simply return an empty dict.
    # This may happen in cases where the function is synthesized dynamically at runtime.
    src = loader.get_source(fn)
    if src is None:
        try:
            src = inspect.getsource(fn)
        except OSError as e:
            raise OSError(
                f"Failed to get source for {fn} using inspect.getsource"
            ) from e

    # Gather a dictionary of parameter name -> type, skipping any parameters whose annotated
    # types are strings. These are only understood by TorchScript in the context of a type annotation
    # that refers to a class in its own definition, but trying to include a mapping for this in the result
    # function would cause infinite recursion because the class is currently being compiled.
    # In addition, there is logic in ScriptTypeParser to handle this.
    signature = inspect.signature(fn)
    name_to_type = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
        and not isinstance(parameter.annotation, str)
    }

    # Then, get the literal type annotations from the function declaration
    # by source inspection. This accounts for the case in which aliases are used
    # to annotate the arguments (e.g device_t = torch.device, and then d: device_t).
    # frontend.py cannot be used here because it includes _jit_internal, so use ast instead.
    a = ast.parse(textwrap.dedent(src))
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f"Expected {fn} to be a function")
    f = a.body[0]

    # Prepare a dictionary of source annotation -> type, which will be the final result of this function,
    # by using the parsed AST (f) to reconstruct source annotations as strings for each parameter and mapping
    # them to the type object corresponding to the annotation via name_to_type using the parameter name.
    annotation_to_type = {}

    for arg in f.args.args:
        # Get the source type annotation string for this argument if possible.
        arg_annotation_str = (
            get_annotation_str(arg.annotation) if arg.annotation else None
        )

        # If the argument has no annotation or get_annotation_str cannot convert it to a string,
        # arg_annotation_str will be None. Skip this arg; ScriptTypeParser will probably handle
        # this in the latter case.
        if arg_annotation_str is None:
            continue

        # Insert {arg_annotation_str: type} into annotation_to_type if possible. One reason arg_name may not
        # be present in name_to_type is that the annotation itself is a string and not a type object
        # (common for self-referential annotations in classes). Once again, let ScriptTypeParser handle this.
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]

    # If there is a valid return annotation, include it in annotation_to_type. As with argument annotations,
    # the literal annotation has to be convertible to a string by get_annotation_str, and the actual type
    # of the annotation cannot be a string.
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = (
        return_annotation is not inspect.Parameter.empty
        and not isinstance(return_annotation, str)
    )
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation

    return annotation_to_type


def createResolutionCallbackForClassMethods(cls: type) -> Callable[[str], Any]:
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    # cls is a type here, so `ismethod` is false since the methods on the type
    # aren't bound to anything, so Python treats them as regular functions
    fns = [
        getattr(cls, name)
        for name in cls.__dict__
        if inspect.isroutine(getattr(cls, name))
    ]
    # Skip built-ins, as they do not have global scope nor type hints
    # Needed to support `enum.Enum` derived classes in Python-3.11
    # That adds `_new_member_` property which is an alias to `__new__`
    # Skip __annotate__ added by PEP 649 for deferred annotation evaluation
    fns = [
        fn
        for fn in fns
        if not inspect.isbuiltin(fn)
        and hasattr(fn, "__globals__")
        and fn.__name__ != "__annotate__"
    ]
    captures = {}

    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    def lookup_in_class(key: str) -> Any:
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)

    return lookup_in_class


def boolean_dispatch(
    arg_name,
    arg_index,
    default,
    if_true,
    if_false,
    module_name,
    func_name,
):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    def fn(*args, **kwargs):
        dispatch_flag = default
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # neither function has a docstring
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from an exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"


def export(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT  # type:ignore[attr-defined]
    return fn


def unused(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(  # noqa: B010
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        if prop.fset:
            setattr(  # noqa: B010
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED  # type: ignore[attr-defined]
    return fn


# No op context manager from python side
class _IgnoreContextManager(contextlib.AbstractContextManager):
    def __init__(self, **kwargs):
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb

                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x


        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

    if callable(drop):
        # used without any args, so drop is actually a function
        #   @torch.jit.ignore
        #   def fn(...):
        fn = drop
        # pyrefly: ignore [missing-attribute]
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    if not isinstance(drop, bool):
        raise RuntimeError(
            f"Argument to @torch.jit.ignore must be a bool or a function but got {drop}"
        )

    # for backwards compat
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        warnings.warn(
            "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

        drop = drop_on_export
    elif drop:
        warnings.warn(
            "ignore(True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

    def decorator(fn):
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    return decorator


def _drop(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    fn._torchscript_modifier = FunctionModifiers._DROP  # type: ignore[attr-defined]
    return fn


def _copy_to_script_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER  # type: ignore[attr-defined]
    return fn


def module_has_exports(mod):
    for name in dir(mod):
        if hasattr(mod, name):
            item = getattr(mod, name)
            if callable(item):
                if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                    return True
    return False


# WARNING: should_drop is currently being used by our JIT code coverage plug-in to mark JIT'd code as covered. If you
# rename this function, please update references in tools/coverage_plugins_package/src/coverage_plugins/jit_plugin.py to
# allow JIT'd code to still be covered.
def should_drop(fn) -> bool:
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED or attr is FunctionModifiers._DROP


def is_ignored_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return (
        mod is FunctionModifiers.UNUSED
        or mod is FunctionModifiers.IGNORE
        or mod is FunctionModifiers._DROP
    )


def _is_drop_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers._DROP


def is_static_fn(cls, fn) -> bool:
    return isinstance(inspect.getattr_static(cls, fn, default=None), staticmethod)


def get_static_fn(cls, fn):
    return inspect.getattr_static(cls, fn).__func__


def get_torchscript_modifier(fn):
    if not callable(fn):
        return None
    if hasattr(fn, "__func__"):
        fn = fn.__func__
    return getattr(fn, "_torchscript_modifier", FunctionModifiers.DEFAULT)


def copy_torchscript_modifier(orig, new) -> None:
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr


# overloading registration
# overloads get registered in this file, and compiled in torch/jit/__init__.py
# so that they can be imported in nn/functional.py without an import cycle

# qualified_name => list[overload_functions]
_overloaded_fns: dict[str, list[Callable]] = {}  # noqa: T484


_OVERLOAD_EXAMPLE = """
Example usage of overload function:
@torch.jit._overload
def my_function(x: type0) -> type0: # decl 1
    pass

@torch.jit._overload
def my_function(x: type1) -> type1: # decl 2
    pass

def my_function(x):                 # implementation
    if isinstance(x, type0):
        return x
    elif isinstance(x, type1):
        return x
"""


def get_overload_no_implementation_error_message(kind, obj):
    sourcelines, file_lineno, filename = get_source_lines_and_file(obj)
    return (
        f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make '
        f"sure a definition is provided and defined after all overload declarations.\n"
        f'File "{filename}", line {file_lineno}:\n'
        + "".join(sourcelines)
        + "\n"
        + _OVERLOAD_EXAMPLE
    )


def _check_overload_body(func):
    try:
        parsed_def = parse_def(func)
    except OSError:
        # Parsing the function definition can raise an OSError if source is unavailable.
        # Since this is just an initial check, just raise a warning if this is the case.
        warnings.warn(
            f"Unable to retrieve source for @torch.jit._overload function: {func}.",
            stacklevel=2,
        )
        return

    body = parsed_def.ast.body[0].body

    def is_pass(x):
        return isinstance(x, ast.Pass)

    def is_ellipsis(x):
        return (
            isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Constant)
            and x.value.value is Ellipsis
        )

    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = (
            "Only `pass` statement or `...` can be the body of overload declaration:\n"
        )
        msg += "\n".join(parsed_def.source.split("\n")[:3])
        msg += " <- Expecting `pass` or `...` here!\n" + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)


def _overload(func: Callable[_P, _R]) -> Callable[_P, _R]:
    _check_overload_body(func)
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func


def _get_fn_overloads(qual_name):
    return _overloaded_fns.get(qual_name)


def _clear_fn_overloads(qual_name) -> None:
    del _overloaded_fns[qual_name]


def get_class_name_lineno(method) -> tuple[str, int]:
    current_frame = inspect.currentframe()

    # one for the get_class_name call, one for _overload_method call
    for i in range(2):
        if current_frame is None:
            raise AssertionError(f"current_frame is None at iteration {i}")
        current_frame = current_frame.f_back

    if current_frame is None:
        raise AssertionError("current_frame is None after traversing frames")
    class_name = current_frame.f_code.co_name
    line_no = current_frame.f_code.co_firstlineno
    return class_name, line_no


# At the point the decorator is applied to class methods the method
# has no reference to its owning class. _qualified_name would not include
# the class it is defined in, so any methods with the same name in the same file
# would have the same _qualified_name, even if they were defined in different
# classes. This problem only exists in python 2.
# We get around this problem by looking at the stack frame and identifying
# the class name, and throwing an error whenever overloads are used
# when modules of the same name are in the same file

# qualified_name => class name => list[overload_functions]
_overloaded_methods: dict[str, dict[str, list[Callable]]] = {}  # noqa: T484


# (qualified_name, class name) => class_fileno
_overloaded_method_class_fileno: dict[tuple[str, str], int] = {}


def _overload_method(func: Callable[_P, _R]) -> Callable[_P, _R]:
    try:
        _check_overload_body(func)
    except IndentationError:
        # CPython 3.13.8 has a bug (https://github.com/python/cpython/issues/139783)
        # where inspect.getsourcelines() returns truncated source when a decorator
        # is followed by a comment, causing ast.parse() to fail with IndentationError.
        # Fixed in 3.13.9. Swallow the error on affected versions; re-raise otherwise.
        if sys.version_info[:3] == (3, 13, 8):
            import warnings

            warnings.warn(
                "Skipping overload body check due to a known CPython 3.13.8 bug "
                "(https://github.com/python/cpython/issues/139783). "
                "Consider upgrading to Python 3.13.9+.",
                stacklevel=2,
            )
        else:
            raise
    qual_name = _qualified_name(func)
    global _overloaded_methods
    class_name_map = _overloaded_methods.get(qual_name)
    if class_name_map is None:
        class_name_map = {}
        _overloaded_methods[qual_name] = class_name_map

    class_name, line_no = get_class_name_lineno(func)
    method_overloads = class_name_map.get(class_name)
    if method_overloads is None:
        method_overloads = []
        class_name_map[class_name] = method_overloads
        _overloaded_method_class_fileno[(qual_name, class_name)] = line_no
    else:
        existing_lineno = _overloaded_method_class_fileno[(qual_name, class_name)]
        if existing_lineno != line_no:
            raise RuntimeError(
                "Cannot currently overload the same method name in two different"
                " classes with the same name in the same module"
            )

    method_overloads.append(func)
    return func


def _get_overloaded_methods(method, mod_class):
    # TODO: __name__ not set for submodules in recursive script
    if not hasattr(method, "__name__"):
        return None
    qual_name = _qualified_name(method)
    class_name_map = _overloaded_methods.get(qual_name)
    if class_name_map is None:
        return None
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None

    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise AssertionError(
            "Overloads are not usable when a module is redeclared within the same file: "
            + str(method)
        )
    return overloads


def is_tuple(ann) -> bool:
    # Check for typing.Tuple missing args (but `tuple` is fine)
    if ann is typing.Tuple:  # noqa: UP006
        raise_error_container_parameter_missing("Tuple")

    # For some reason Python 3.7 violates the Type[A, B].__origin__ == Type rule
    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is tuple


def is_list(ann) -> bool:
    # Check for typing.List missing args (but `list` is fine)
    if ann is typing.List:  # noqa: UP006
        raise_error_container_parameter_missing("List")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is list


def is_dict(ann) -> bool:
    # Check for typing.Dict missing args (but `dict` is fine)
    if ann is typing.Dict:  # noqa: UP006
        raise_error_container_parameter_missing("Dict")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    return ann.__module__ in ("builtins", "typing") and ann_origin is dict


def is_union(ann):
    if ann is Union:
        raise_error_container_parameter_missing("Union")

    return isinstance(ann, BuiltinUnionType) or (
        hasattr(ann, "__module__")
        and ann.__module__ == "typing"
        and (get_origin(ann) is Union)
    )


def is_optional(ann):
    if ann is Optional:
        raise_error_container_parameter_missing("Optional")

    def is_optional_as_optional(ann):
        return (
            hasattr(ann, "__module__")
            and ann.__module__ == "typing"
            and (get_origin(ann) is Optional)
        )

    def is_union_as_optional(ann):
        ann_args = get_args(ann)
        return len(ann_args) == 2 and (None in ann_args or type(None) in ann_args)

    return is_optional_as_optional(ann) or (is_union(ann) and is_union_as_optional(ann))


def is_future(ann) -> bool:
    if ann is Future:
        raise RuntimeError(
            "Attempted to use Future without a "
            "contained type. Please add a contained type, e.g. "
            "Future[int]"
        )
    return get_origin(ann) is Future


def is_await(ann) -> bool:
    if ann is _Await:
        return True
    return get_origin(ann) is _Await


if torch.distributed.rpc.is_available():
    from torch._C._distributed_rpc import PyRRef
    from torch.distributed.rpc import RRef

    def is_rref(ann) -> bool:
        if ann is RRef:
            raise RuntimeError(
                "Attempted to use RRef without a "
                "contained type. Please add a contained type, e.g. "
                "RRef[int]"
            )
        return get_origin(ann) is RRef

    def is_rref_instance(obj) -> bool:
        return isinstance(obj, PyRRef)

else:

    def is_rref_instance(obj) -> bool:
        # If the RPC module doesn't exist then RRefs don't exist either.
        return False


def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return boolean_dispatched.get(fn)


def _get_named_tuple_properties(
    obj,
    loc: torch._C._jit_tree_views.SourceRange | None = None,
    rcb=None,
):
    if loc is None:
        loc = fake_range()

    if not issubclass(obj, tuple) or not hasattr(obj, "_fields"):
        raise AssertionError(
            f"expected namedtuple (tuple subclass with _fields), got {obj}"
        )
    if hasattr(obj, "_field_defaults"):
        defaults = [
            obj._field_defaults[field]
            for field in obj._fields
            if field in obj._field_defaults
        ]
    else:
        defaults = []

    obj_annotations = inspect.get_annotations(obj)
    if len(obj_annotations) == 0 and hasattr(obj, "__base__"):
        obj_annotations = inspect.get_annotations(
            # pyrefly: ignore [bad-argument-type]
            obj.__base__
        )

    annotations = []
    for field in obj._fields:
        if field in obj_annotations:
            field_type = obj_annotations[field]
            # [Note: ForwardRef annotations in NamedTuple attributes]
            # NamedTuple types are slightly different from normal types.
            #
            # Normally, annotations are evaluated like this (during jit.script):
            # 1. Load strings of python code into c++ and parse.
            # 2. Get annotations as strings
            # 3. Use the PythonResolver's resolution callback (rcb) to convert
            #    the string into a python object
            # 4. We call into annotations.py:ann_to_type to convert python obj
            #    from step 3 into a type that torchscript understands.
            #
            # NamedTuples are more complicated, because it has sub-types.
            # Normally, once we have the NamedTuple type object from #3,
            # we can just look at the annotation literal values and use
            # ann_to_type directly on them.
            #
            # But sometimes, users will annotate with string literals, e.g.
            #    x: 'int'
            # This also happens with PEP563 (from __forward__ import annotations)
            #
            # These annotations appear in the annotation dict as ForwardRef('int').
            #
            # Then, we need to convert the string into a python object. This
            # requires having local context for custom objects or imported types.
            # rcb() is what gives us this. So, we plumb rcb through the stack so
            # it can be used in this context for the if block below.
            #
            # FAQ:
            # - Why do we need this special handling for NamedTuple but string
            #   annotations work fine for normal types? Normally, we parse the
            #   string directly and then call rcb() directly from C++.
            # - Why not use ForwardRef._evaluate? For that, we need globals()
            #   and locals() for the local context where the NamedTuple was defined.
            #   rcb is what lets us look up into these. So, basically rcb does the
            #   hard work for us.
            if isinstance(field_type, ForwardRef) and rcb is not None:
                rcb_type = rcb(field_type.__forward_arg__)
                # rcb returns None if it can't find anything.
                if rcb_type is None:
                    raise ValueError(
                        f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}."
                        f" Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858."
                        f" Issue occurred at {loc.highlight()}"
                    )
                field_type = rcb_type
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            annotations.append(the_type)
        else:
            annotations.append(torch._C.TensorType.getInferred())
    return type(obj).__name__, obj._fields, annotations, defaults


def _create_named_tuple(
    t,
    unqual_name: str,
    field_names: list[str],
    defaults: tuple[Any, ...],
):
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)  # type: ignore[call-arg, no-redef, misc]
    return TupleType(*t)


@contextlib.contextmanager
def _disable_emit_hooks():
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    try:
        yield
    finally:
        torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None:
    def __enter__(self) -> None:
        self.hooks = torch._C._jit_get_emit_hooks()
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args) -> None:
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])


def _is_exception(obj) -> bool:
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, Exception)


def raise_error_container_parameter_missing(target_type) -> None:
    if target_type.endswith("ict"):
        raise RuntimeError(
            f"Attempted to use {target_type} without "
            "contained types. Please add contained type, e.g. "
            f"{target_type}[int, int]"
        )
    raise RuntimeError(
        f"Attempted to use {target_type} without a "
        "contained type. Please add a contained type, e.g. "
        f"{target_type}[int]"
    )


_RAW_TYPE_NAME_MAPPING = {
    dict: "dict",
    list: "list",
    tuple: "tuple",
    typing.Dict: "Dict",  # noqa: UP006
    typing.List: "List",  # noqa: UP006
    typing.Optional: "Optional",
    typing.Tuple: "Tuple",  # noqa: UP006
}


def check_args_exist(target_type) -> None:
    if name := _RAW_TYPE_NAME_MAPPING.get(target_type):
        raise_error_container_parameter_missing(name)


def check_empty_containers(obj) -> None:
    if obj == [] or obj == {} or obj == ():
        warnings.warn(
            "The inner type of a container is lost when "
            "calling torch.jit.isinstance in eager mode. For "
            "example, List[int] would become list and "
            "therefore falsely return True for List[float] or"
            " List[str].",
            stacklevel=2,
        )


# supports List/Dict/Tuple and Optional types
# TODO support future
def container_checker(obj, target_type) -> bool:
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        return False
    elif origin_type is list or origin_type is typing.List:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, list):
            return False
        arg_type = get_args(target_type)[0]
        arg_origin = get_origin(arg_type)
        for el in obj:
            # check if nested container, ex: List[List[str]]
            if arg_origin:  # processes nested container, ex: List[List[str]]
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
        return True
    elif origin_type is typing.Dict or origin_type is dict:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for key, val in obj.items():
            # check if keys are of right type
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
        return True
    elif origin_type is typing.Tuple or origin_type is tuple:  # noqa: UP006
        check_empty_containers(obj)
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for el, el_type in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
        return True
    elif origin_type is Union or issubclass(
        origin_type,
        BuiltinUnionType,
    ):  # also handles Optional
        if obj is None:  # check before recursion because None is always fine
            return True
        inner_types = get_args(target_type)
        for t in inner_types:
            t_origin = get_origin(t)
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                return True
    return False


def _isinstance(obj, target_type) -> bool:
    if isinstance(target_type, collections.abc.Container):
        if not isinstance(target_type, tuple):
            raise RuntimeError(
                "The second argument to "
                "`torch.jit.isinstance` must be a type "
                "or a tuple of types"
            )
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False

    origin_type = get_origin(target_type)
    if origin_type:
        return container_checker(obj, target_type)

    # Check to handle non-typed optional origin returns as none instead
    #    of as optional in 3.7-3.8
    check_args_exist(target_type)

    # handle non-containers
    return isinstance(obj, target_type)


class _TensorExtractor(pickle.Pickler):
    def __init__(self, *args, tensors: list[torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ""
        # Since we just want to extract tensors, we don't mind if an object is
        # unpicklable if it doesn't contain tensors, as we can just ignore/skip
        # it. To play it safe, we only do so for common objects that we're sure
        # don't contain tensors. Feel free to add new types here. Note also that
        # even if a type isn't listed here this won't block users, since they
        # can just add a __getstate__ or __reduce__ method to their class.
        if isinstance(obj, LockType):
            return ""
        # Futures and RRefs don't technically contain a value, they just offer
        # the means to access a value.
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ""
        if isinstance(obj, CAwait):
            return ""
        if isinstance(obj, torch.cuda.Event):
            return ""
        if isinstance(obj, threading.Thread):
            return ""
        return None


def _extract_tensors(obj):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
    tensors: list[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors


def _get_model_id(obj) -> str | None:
    if isinstance(obj, torch.jit.ScriptModule):
        return str(obj._c._type())
    elif isinstance(obj, torch.jit.ScriptFunction):
        return obj.qualified_name
    else:
        return None


# In Python-3.11+ typed enums (i.e. IntEnum for example) retain number of base class methods in subclass
# that were previously dropped. To preserve the behavior, explicitly drop them there

if sys.version_info >= (3, 11):
    _drop(enum.Enum.__new__)
    _drop(enum.Enum.__format__)
    _drop(enum.Enum.__repr__)
    _drop(enum.Enum.__str__)
<|endoftext|>"""
This makes the functions in torch._C._VariableFunctions available as
    torch._VF.<funcname>
without mypy being able to find them.

A subset of those functions are mapped to ATen functions in
torch/jit/_builtins.py

See https://github.com/pytorch/pytorch/issues/21478 for the reason for
introducing torch._VF

"""

import sys
import types

import torch


class VFModule(types.ModuleType):
    vf: types.ModuleType

    def __init__(self, name: str):
        super().__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, name: str) -> object:
        return getattr(self.vf, name)


sys.modules[__name__] = VFModule(__name__)
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any
from typing_extensions import deprecated
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from torch.serialization import MAP_LOCATION


class _Faketqdm:  # type: ignore[no-redef]
    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # Don't bother implementing; use real tqdm if you want
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write("\n")


try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

__all__ = [
    "download_url_to_file",
    "get_dir",
    "help",
    "list",
    "load",
    "load_state_dict_from_url",
    "set_dir",
]

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
_PATH_SEP_PATTERN = re.compile(r"[/\\]")

_TRUSTED_REPO_OWNERS = (
    "facebookresearch",
    "facebookincubator",
    "pytorch",
    "fairinternal",
)
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
VAR_DEPENDENCY = "dependencies"
MODULE_HUBCONF = "hubconf.py"
READ_DATA_CHUNK = 128 * 1024
_hub_dir: str | None = None


@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


# Copied from tools/shared/module_loader to be included in torch package
def _import_module(name, path):
    import importlib.util
    from importlib.abc import Loader

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise AssertionError(f"failed to load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    if not isinstance(spec.loader, Loader):
        raise AssertionError(f"expected Loader, got {type(spec.loader)}")
    spec.loader.exec_module(module)
    return module


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _safe_extract_zip(zip_file, extract_to):
    """
    Safely extract a zip file, preventing zipslip attacks.

    Args:
        zip_file: ZipFile object to extract
        extract_to: Directory to extract to

    Raises:
        ValueError: If any archive entry contains unsafe paths
    """
    # Normalize the extraction directory path
    extract_to = Path(extract_to).resolve(strict=False)

    for member in zip_file.infolist():
        # Get the normalized path
        filename = os.path.normpath(member.filename)

        # Check for directory traversal attempts
        if filename.startswith(("/", "\\")):
            raise ValueError(f"Archive entry has absolute path: {member.filename}")

        if len(filename) >= 2 and filename[1] == ":" and filename[0].isalpha():
            raise ValueError(f"Archive entry has absolute path: {member.filename}")

        if ".." in re.split(_PATH_SEP_PATTERN, filename):
            raise ValueError(
                f"Archive entry contains directory traversal: {member.filename}"
            )

        # Construct the full extraction path and verify it's within extract_to
        out = (extract_to / filename).resolve(strict=False)

        if not out.is_relative_to(extract_to):
            raise ValueError(
                f"Archive entry escapes target directory: {member.filename}"
            )

        # Extract the member safely
        zip_file.extract(member, extract_to)


def _git_archive_link(repo_owner, repo_name, ref):
    # See https://docs.github.com/en/rest/reference/repos#download-a-repository-archive-zip
    return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )
    return torch_home


def _parse_repo_info(github):
    if ":" in github:
        repo_info, ref = github.split(":")
    else:
        repo_info, ref = github, None
    repo_owner, repo_name = repo_info.split("/")

    if ref is None:
        # The ref wasn't specified by the user, so we need to figure out the
        # default branch: main or master. Our assumption is that if main exists
        # then it's the default branch, otherwise it's master.
        try:
            with urlopen(f"https://github.com/{repo_owner}/{repo_name}/tree/main/"):
                ref = "main"
        except HTTPError as e:
            if e.code == 404:
                ref = "master"
            else:
                raise
        except URLError as e:
            # No internet connection, need to check for cache as last resort
            for possible_ref in ("main", "master"):
                if os.path.exists(
                    f"{get_dir()}/{repo_owner}_{repo_name}_{possible_ref}"
                ):
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(
                    "It looks like there is no internet connection and the "
                    f"repo could not be found in the cache ({get_dir()})"
                ) from e
    return repo_owner, repo_name, ref


def _read_url(url):
    with urlopen(url) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


def _validate_not_a_forked_repo(repo_owner, repo_name, ref):
    # Use urlopen to avoid depending on local git.
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags",
    ):
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}"
            try:
                response = json.loads(_read_url(Request(url, headers=headers)))
            except HTTPError:
                # Retry without token in case it had insufficient permissions.
                del headers["Authorization"]
                response = json.loads(_read_url(Request(url, headers=headers)))
            # Empty response means no more data to process
            if not response:
                break
            for br in response:
                if br["name"] == ref or br["commit"]["sha"].startswith(ref):
                    return

    raise ValueError(
        f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
        "If it's a commit from a forked repo, please call hub.load() with forked repo directly."
    )


def _get_cache_or_reload(
    github,
    force_reload,
    trust_repo,
    verbose=True,
    skip_validation=False,
):
    # Setup hub_dir to save downloaded files
    hub_dir = get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    # Parse github repo information
    repo_owner, repo_name, ref = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need
    # to worry about it.
    normalized_br = ref.replace("/", "_")
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, owner_name_branch)
    # Check that the repo is in the trusted list
    _check_repo_is_trusted(
        repo_owner,
        repo_name,
        owner_name_branch,
        trust_repo=trust_repo,
    )

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write(f"Using cache found in {repo_dir}\n")
    else:
        # Validate the tag/branch is from the original repo instead of a forked repo
        if not skip_validation:
            _validate_not_a_forked_repo(repo_owner, repo_name, ref)

        cached_file = os.path.join(hub_dir, normalized_br + ".zip")
        _remove_if_exists(cached_file)

        try:
            url = _git_archive_link(repo_owner, repo_name, ref)
            sys.stdout.write(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, progress=False)
        except HTTPError as err:
            if err.code == 300:
                # Getting a 300 Multiple Choices error likely means that the ref is both a tag and a branch
                # in the repo. This can be disambiguated by explicitly using refs/heads/ or refs/tags
                # See https://git-scm.com/book/en/v2/Git-Internals-Git-References
                # Here, we do the same as git: we throw a warning, and assume the user wanted the branch
                warnings.warn(
                    f"The ref {ref} is ambiguous. Perhaps it is both a tag and a branch in the repo? "
                    "Torchhub will now assume that it's a branch. "
                    "You can disambiguate tags and branches by explicitly passing refs/heads/branch_name or "
                    "refs/tags/tag_name as the ref. That might require using skip_validation=True.",
                    stacklevel=2,
                )
                disambiguated_branch_ref = f"refs/heads/{ref}"
                url = _git_archive_link(
                    repo_owner, repo_name, ref=disambiguated_branch_ref
                )
                download_url_to_file(url, cached_file, progress=False)
            else:
                raise

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder using safe extraction
            _safe_extract_zip(cached_zipfile, hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def _check_repo_is_trusted(
    repo_owner,
    repo_name,
    owner_name_branch,
    trust_repo,
):
    hub_dir = get_dir()
    filepath = os.path.join(hub_dir, "trusted_list")

    if not os.path.exists(filepath):
        Path(filepath).touch()
    with open(filepath) as file:
        trusted_repos = tuple(line.strip() for line in file)

    # To minimize friction of introducing the new trust_repo mechanism, we consider that
    # if a repo was already downloaded by torchhub, then it is already trusted (even if it's not in the allowlist)
    trusted_repos_legacy = next(os.walk(hub_dir))[1]

    owner_name = "_".join([repo_owner, repo_name])
    is_trusted = (
        owner_name in trusted_repos
        or owner_name_branch in trusted_repos_legacy
        or repo_owner in _TRUSTED_REPO_OWNERS
    )

    if (trust_repo is False) or (trust_repo == "check" and not is_trusted):
        response = input(
            f"The repository {owner_name} does not belong to the list of trusted repositories and as such cannot be downloaded. "
            "Do you trust this repository and wish to add it to the trusted list of repositories (y/N)?"
        )
        if response.lower() in ("y", "yes"):
            if is_trusted:
                print("The repository is already trusted.")
        elif response.lower() in ("n", "no", ""):
            raise Exception("Untrusted repository.")  # noqa: TRY002
        else:
            raise ValueError(f"Unrecognized response {response}.")

    # At this point we're sure that the user trusts the repo (or wants to trust it)
    if not is_trusted:
        with open(filepath, "a") as file:
            file.write(owner_name + "\n")


def _check_module_exists(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if missing_deps:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing_deps)}")


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")

    # Note that if a missing dependency is imported at top level of hubconf, it will
    # throw before this function. It's a chicken and egg situation where we have to
    # load hubconf to know what're the dependencies, but to import hubconf it requires
    # a missing package. This is fine, Python will throw proper error message for users.
    _check_dependencies(m)

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError(f"Cannot find callable {model} in hubconf")

    return func


def get_dir() -> str:
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn(
            "TORCH_HUB is deprecated, please use env TORCH_HOME instead", stacklevel=2
        )

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")


def set_dir(d: str | os.PathLike) -> None:
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = os.path.expanduser(d)


def list(
    github,
    force_reload=False,
    skip_validation=False,
    trust_repo="check",
    verbose=True,
):
    r"""
    List all callable entrypoints available in the repo specified by ``github``.

    Args:
        github (str): a string with format "repo_owner/repo_name[:ref]" with an optional
            ref (tag or branch). If ``ref`` is not specified, the default branch is assumed to be ``main`` if
            it exists, and otherwise ``master``.
            Example: 'pytorch/vision:0.10'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.

            Default is ``"check"``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Default is ``True``.

    Returns:
        list: The available callables entrypoint

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> entrypoints = torch.hub.list("pytorch/vision", force_reload=True)
    """
    repo_dir = _get_cache_or_reload(
        github,
        force_reload,
        trust_repo,
        verbose=verbose,
        skip_validation=skip_validation,
    )

    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    # We take functions starting with '_' as internal helper functions
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]

    return entrypoints


def help(github, model, force_reload=False, skip_validation=False, trust_repo="check"):
    r"""
    Show the docstring of entrypoint ``model``.

    Args:
        github (str): a string with format <repo_owner/repo_name[:ref]> with an optional
            ref (a tag or a branch). If ``ref`` is not specified, the default branch is assumed
            to be ``main`` if it exists, and otherwise ``master``.
            Example: 'pytorch/vision:0.10'
        model (str): a string of entrypoint name defined in repo's ``hubconf.py``
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the ref
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.

            Default is ``"check"``.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> print(torch.hub.help("pytorch/vision", "resnet18", force_reload=True))
    """
    repo_dir = _get_cache_or_reload(
        github,
        force_reload,
        trust_repo,
        verbose=True,
        skip_validation=skip_validation,
    )

    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


def load(
    repo_or_dir,
    model,
    *args,
    source="github",
    trust_repo="check",
    force_reload=False,
    verbose=True,
    skip_validation=False,
    **kwargs,
):
    r"""
    Load a model from a github repo or a local directory.

    Note: Loading a model is the typical use case, but this can also be used to
    for loading other objects such as tokenizers, loss functions, etc.

    If ``source`` is 'github', ``repo_or_dir`` is expected to be
    of the form ``repo_owner/repo_name[:ref]`` with an optional
    ref (a tag or a branch).

    If ``source`` is 'local', ``repo_or_dir`` is expected to be a
    path to a local directory.

    Args:
        repo_or_dir (str): If ``source`` is 'github',
            this should correspond to a github repo with format ``repo_owner/repo_name[:ref]`` with
            an optional ref (tag or branch), for example 'pytorch/vision:0.10'. If ``ref`` is not specified,
            the default branch is assumed to be ``main`` if it exists, and otherwise ``master``.
            If ``source`` is 'local'  then it should be a path to a local directory.
        model (str): the name of a callable (entrypoint) defined in the
            repo/dir's ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        source (str, optional): 'github' or 'local'. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is 'github'.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.

            Default is ``"check"``.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = 'local'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = 'local'``.
            Default is ``True``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        The output of the ``model`` callable when called with the given
        ``*args`` and ``**kwargs``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # from a github repo
        >>> repo = "pytorch/vision"
        >>> model = torch.hub.load(
        ...     repo, "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1"
        ... )
        >>> # from a local directory
        >>> path = "/some/local/path/pytorch/vision"
        >>> # xdoctest: +SKIP
        >>> model = torch.hub.load(path, "resnet50", weights="ResNet50_Weights.DEFAULT")
    """
    source = source.lower()

    if source not in ("github", "local"):
        raise ValueError(
            f'Unknown source: "{source}". Allowed values: "github" | "local".'
        )

    if source == "github":
        repo_or_dir = _get_cache_or_reload(
            repo_or_dir,
            force_reload,
            trust_repo,
            verbose=verbose,
            skip_validation=skip_validation,
        )

    model = _load_local(repo_or_dir, model, *args, **kwargs)
    return model


def _load_local(hubconf_dir, model, *args, **kwargs):
    r"""
    Load a model from a local directory with a ``hubconf.py``.

    Args:
        hubconf_dir (str): path to a local directory that contains a
            ``hubconf.py``.
        model (str): name of an entrypoint defined in the directory's
            ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> # xdoctest: +SKIP("stub local path")
        >>> path = "/some/local/path/pytorch/vision"
        >>> model = _load_local(
        ...     path,
        ...     "resnet50",
        ...     weights="ResNet50_Weights.IMAGENET1K_V1",
        ... )
    """
    with _add_to_sys_path(hubconf_dir):
        hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

        entry = _load_entry_from_hubconf(hub_module, model)
        model = entry(*args, **kwargs)

    return model


def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        ...     "/tmp/temporary_file",
        ... )

    """
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for _ in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")  # noqa: SIM115
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")
    req = Request(url, headers={"User-Agent": "torch.hub"})
    try:
        with urlopen(req) as u:
            meta = u.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            file_size = None
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])

            sha256 = hashlib.sha256() if hash_prefix is not None else None
            with tqdm(
                total=file_size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = u.read(READ_DATA_CHUNK)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if sha256 is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if sha256 is not None and hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(
                        f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                    )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename: str) -> bool:
    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as zf:
            infolist = zf.infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False


@deprecated(
    "Falling back to the old format < 1.6. This support will be "
    "deprecated in favor of default zipfile format introduced in 1.6. "
    "Please redo torch.save() to save it in the new zipfile format.",
    category=FutureWarning,
)
def _legacy_zip_load(
    filename: str,
    model_dir: str,
    map_location: MAP_LOCATION,
    weights_only: bool,
) -> dict[str, Any]:
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        # Use safe extraction to prevent zipslip attacks
        _safe_extract_zip(f, model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(
        extracted_file, map_location=map_location, weights_only=weights_only
    )


def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> state_dict = torch.hub.load_state_dict_from_url(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        ... )

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead",
            stacklevel=2,
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stdout.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)
<|endoftext|>"""Runtime driver for torch.compiler.precompile artifacts, authored as real code.

Nothing here is imported or run by torch at runtime, and the generated artifacts never
import this module. Instead torch._precompile emits these function bodies VERBATIM (via
inspect.getsource, on the to_python_code / emit path only) into the self-contained
python_code string, after the calling-convention metadata and the compiled/captured
graph. Authoring the driver as real code -- instead of a hand-written string literal --
lets pyrefly / ruff / IDEs type-check and navigate the load-bearing driver logic that
would otherwise be invisible inside a string (and drops the wall of ``# noqa: F821``).

Keeping python_code self-contained and version-frozen (its behavior is hashed via
code_hash, invariant 7) still holds: the artifact carries the driver TEXT, it does not
import it, so there is no torch-version skew. The emit path runs getsource in-process
where torch source is present; load() never touches this module.

The names the emitted bodies read from the artifact's own namespace -- the metadata
constants, the ``_torch`` / ``_pytree`` import aliases, and the graph's ``call`` -- are
declared under TYPE_CHECKING below so the bodies type-check here; at emit time they
resolve against the metadata + graph sections that precede the driver in python_code.

INVARIANT: ``_extract_param_buffers`` reproduces
``torch._precompile._intern_param_buffers``'s params-then-buffers, intern-by-identity
ordering VERBATIM; the two must stay in sync (see Note [precompile programming model]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch as _torch
import torch.utils._pytree as _pytree


if TYPE_CHECKING:
    # Calling-convention metadata the emitted driver reads from the artifact namespace,
    # where torch._precompile._build_metadata_section emits each as a literal assignment
    # ahead of the driver. Bound here with placeholder values (not bare annotations) so
    # static tools treat them as real names in the bodies below; this block is not emitted.
    MODULE_POSITIONS: list[int] = []
    NUM_POSITIONAL_ARGS: int = 0
    PARAM_NAMES: list[str] = []
    BUFFER_NAMES: list[str] = []
    PARAM_SHAPES: list[tuple[int, ...]] = []
    BUFFER_SHAPES: list[tuple[int, ...]] = []
    PARAM_DTYPES: list[str] = []
    BUFFER_DTYPES: list[str] = []
    PARAM_DEVICES: list[str] = []
    BUFFER_DEVICES: list[str] = []
    GRAD_PARAM_INDICES: list[int] = []
    IN_SPEC: str | None = None
    OUT_SPEC: str = ""
    USER_INPUT_SHAPES: list[tuple[int | None, ...] | None] = []
    USER_INPUT_DTYPES: list[str | None] = []
    USER_INPUT_DEVICES: list[str | None] = []
    USER_INPUT_BOUNDS: list[dict[int, tuple[int | None, int | None]] | None] = []

    # The compiled/captured graph's entry point, emitted before the driver.
    def call(flat_inputs: list[object]) -> list[object]: ...


def _extract_param_buffers(mods):
    """Lift the runtime modules' params then buffers, interning by identity, in the
    same order as capture, so the list lines up with the compiled/captured graph. Returns
    (pb, names) where names mirrors PARAM_NAMES + BUFFER_NAMES. This ordering AND the
    naming must match torch._precompile._intern_param_buffers verbatim (its INVARIANT)."""
    multi = len(mods) > 1
    seen = set()
    pb = []
    names = []

    def intern(mi, n, t):
        if id(t) not in seen:
            seen.add(id(t))
            pb.append(t)
            names.append(f"m{mi}.{n}" if multi else n)

    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            intern(mi, n, p)
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            intern(mi, n, b)
    return pb, names


def _fail(msg):
    # Imported lazily (only when a guard fails) so a normal run does not couple the
    # standalone artifact to torch._precompile's import surface.
    from torch._precompile import PrecompileError as _PrecompileError

    raise _PrecompileError(msg)


def _check_structure(pb, names):
    # Verify the runtime model's extracted param/buffer NAMES match the baked
    # PARAM_NAMES + BUFFER_NAMES (count AND order/identity), so a reordered or
    # structurally-drifted same-count model is caught precisely (invariant 2) rather
    # than scattering grads onto the wrong slot. Then check each tensor's SHAPE, DTYPE and
    # DEVICE against the baked example: the graph is specialized to the example shapes and
    # can bake a device literal, so a same-named but differently shaped/typed/placed
    # runtime tensor would silently miscompute or fail deep in a kernel.
    expected = list(PARAM_NAMES) + list(BUFFER_NAMES)
    if names != expected:
        _fail(
            f"precompile: the runtime model's param/buffer names {names!r} do not match "
            f"the traced model's {expected!r}; the runtime model must be structurally "
            f"identical to the traced model (invariant 2)."
        )
    expected_shapes = list(PARAM_SHAPES) + list(BUFFER_SHAPES)
    expected_dtypes = list(PARAM_DTYPES) + list(BUFFER_DTYPES)
    expected_devices = list(PARAM_DEVICES) + list(BUFFER_DEVICES)
    for _nm, _t, _shp, _dt, _dev in zip(
        names, pb, expected_shapes, expected_dtypes, expected_devices
    ):
        if tuple(_t.shape) != tuple(_shp):
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} has shape "
                f"{tuple(_t.shape)} but the traced model's was {tuple(_shp)}; the runtime "
                f"model must be structurally identical to the traced model (invariant 2)."
            )
        if str(_t.dtype) != _dt:
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} has dtype {_t.dtype} but "
                f"the traced model's was {_dt}; the runtime model must be structurally "
                f"identical to the traced model (invariant 2)."
            )
        if str(_t.device) != _dev:
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} is on device {_t.device} "
                f"but the traced model's was {_dev}; the runtime model must be "
                f"structurally identical to the traced model (invariant 2)."
            )


def _eager_forward(*args):
    """Run the captured ATen graph eagerly. Pass the same args the traced fn took --
    the module(s) in the same positions plus the runtime inputs. The module(s) must
    be structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    The eager backend runs the graph as captured: inputs (including tensor
    subclasses) are passed through unchanged (no dense flatten/unflatten), and the
    graph's flat outputs are reassembled into fn's output structure. If fn ran a
    backward, the trailing grad outputs (one per GRAD_PARAM_INDICES entry) are
    parameter grads, scattered (accumulated) onto the params that received one like
    eager .backward() -- frozen / non-contributing params keep .grad = None."""
    if len(args) != NUM_POSITIONAL_ARGS:
        _fail(
            f"precompile: expected {NUM_POSITIONAL_ARGS} positional args (the same as "
            f"the traced fn), got {len(args)} (invariant 2)."
        )
    mods = []
    for _i in MODULE_POSITIONS:
        if not isinstance(args[_i], _torch.nn.Module):
            _fail(
                f"precompile: argument at position {_i} must be the nn.Module the traced "
                f"fn took (invariant 2), got {type(args[_i]).__name__}."
            )
        mods.append(args[_i])
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _runtime_in_spec = _pytree.tree_flatten(tuple(user_inputs))
    if IN_SPEC is not None and _runtime_in_spec != _pytree.treespec_loads(IN_SPEC):
        _fail(
            "precompile: runtime inputs have a different structure than the traced "
            "example inputs (invariant 3); they must match in nesting and count."
        )
    # Reject a SHAPE / DTYPE / DEVICE mismatch (invariants 3 and 6) up front. Mirrors the
    # inductor driver checks (keep the two drivers in sync). The eager backend has no
    # assert_size_stride, so only these are checked (layout-flexible). The eager backend
    # rejects mark_unbacked up front, so every dim here is static and USER_INPUT_BOUNDS is
    # always all-None; there is no bounds branch (it would be dead code). USER_INPUT_BOUNDS
    # is still emitted in the metadata for the inductor driver, so it is not consumed here.
    if len(user_flat) != len(USER_INPUT_SHAPES):
        _fail(
            "precompile: runtime inputs flattened to a different number of leaves than "
            "the traced example (invariant 3); they must match the traced structure."
        )
    for _t, _shp, _dt, _dev in zip(
        user_flat, USER_INPUT_SHAPES, USER_INPUT_DTYPES, USER_INPUT_DEVICES
    ):
        if _shp is None or not isinstance(_t, _torch.Tensor):
            continue
        _act = tuple(_t.shape)
        if len(_act) != len(_shp) or any(a != e for a, e in zip(_act, _shp)):
            _fail(
                f"precompile: a runtime input has shape {_act} but the artifact was "
                f"traced with shape {tuple(_shp)}; the graph is specialized to the static "
                f"dims (invariant 3). Retrace for this shape, or use backend='eager'."
            )
        if _dt is not None and str(_t.dtype) != _dt:
            _fail(
                f"precompile: a runtime input has dtype {_t.dtype} but the artifact was "
                f"traced with dtype {_dt}; the graph is specialized to the example dtype "
                f"(invariant 6). Cast the input to the traced dtype, or retrace."
            )
        if _dev is not None and str(_t.device) != _dev:
            _fail(
                f"precompile: a runtime input is on device {_t.device} but the artifact "
                f"was traced on device {_dev}; the graph is specialized to the example "
                f"device (invariant 6). Move the input to the traced device, or retrace."
            )
    pb, _names = _extract_param_buffers(mods)
    _check_structure(pb, _names)
    with _torch.no_grad():
        out = list(call([*pb, *user_flat]))
    if GRAD_PARAM_INDICES:
        n = len(GRAD_PARAM_INDICES)
        grads = out[len(out) - n :]
        out = out[: len(out) - n]
        for idx, g in zip(GRAD_PARAM_INDICES, grads):
            p = pb[idx]
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))


def _inductor_forward(*args):
    """Run the compiled computation. Pass the same args the traced fn took -- the
    module(s) in the same positions plus the runtime inputs. The module(s) must be
    structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    Module params/buffers are extracted (no weights are baked into the artifact) and,
    together with the runtime inputs, passed to the composed ``call`` -- which is the
    AOTAutograd+Inductor graph with its own prelude/epilogue, so it handles tensor-
    subclass wrap/unwrap and input mutation (e.g. BatchNorm running stats) internally
    and disables grad itself. If fn ran a backward, the trailing grad outputs (one per
    GRAD_PARAM_INDICES entry) are parameter grads: they are scattered (accumulated)
    onto the params that received one, mirroring eager .backward() (frozen /
    non-contributing params keep .grad = None), and the artifact returns fn's own
    result. Nothing here reads an external cache: the kernels JIT-compile from the
    inlined source on first call. A runtime input whose shape, dtype, or device differs
    from the traced example is rejected up front (invariants 3 and 6), and a differing
    stride / memory format is rejected via the inlined assert_size_stride (invariant 6);
    use backend="eager" for layout-flexible execution."""
    if len(args) != NUM_POSITIONAL_ARGS:
        _fail(
            f"precompile: expected {NUM_POSITIONAL_ARGS} positional args (the same as "
            f"the traced fn), got {len(args)} (invariant 2)."
        )
    mods = []
    for _i in MODULE_POSITIONS:
        if not isinstance(args[_i], _torch.nn.Module):
            _fail(
                f"precompile: argument at position {_i} must be the nn.Module the traced "
                f"fn took (invariant 2), got {type(args[_i]).__name__}."
            )
        mods.append(args[_i])
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _runtime_in_spec = _pytree.tree_flatten(tuple(user_inputs))
    if IN_SPEC is not None and _runtime_in_spec != _pytree.treespec_loads(IN_SPEC):
        _fail(
            "precompile: runtime inputs have a different structure than the traced "
            "example inputs (invariant 3); they must match in nesting and count."
        )
    # Reject a SHAPE / DTYPE / DEVICE / BOUNDS mismatch (invariants 3 and 6) up front.
    # Mirrors the eager driver checks (keep the two drivers in sync). Stride/memory-format
    # is enforced by the inlined assert_size_stride (pinned at capture).
    if len(user_flat) != len(USER_INPUT_SHAPES):
        _fail(
            "precompile: runtime inputs flattened to a different number of leaves than "
            "the traced example (invariant 3); they must match the traced structure."
        )
    for _t, _shp, _dt, _dev, _bnd in zip(
        user_flat,
        USER_INPUT_SHAPES,
        USER_INPUT_DTYPES,
        USER_INPUT_DEVICES,
        USER_INPUT_BOUNDS,
    ):
        if _shp is None or not isinstance(_t, _torch.Tensor):
            continue
        # A dim recorded as None was captured dynamic (unbacked); any size is valid.
        _act = tuple(_t.shape)
        if len(_act) != len(_shp) or any(
            e is not None and a != e for a, e in zip(_act, _shp)
        ):
            _fail(
                f"precompile: a runtime input has shape {_act} but the artifact was "
                f"traced with shape {tuple(_shp)} (None = a dynamic dim, any size); the "
                f"graph is specialized to the static dims (invariant 3). Retrace, mark "
                f"the dim dynamic via mark_unbacked, or use backend='eager'."
            )
        if _dt is not None and str(_t.dtype) != _dt:
            _fail(
                f"precompile: a runtime input has dtype {_t.dtype} but the artifact was "
                f"traced with dtype {_dt}; the graph is specialized to the example dtype "
                f"(invariant 6). Cast the input to the traced dtype, or retrace."
            )
        if _dev is not None and str(_t.device) != _dev:
            _fail(
                f"precompile: a runtime input is on device {str(_t.device)!r} but the "
                f"artifact was traced on device {_dev!r}; the graph is specialized to the "
                f"example device (invariant 6). Move the input to the traced device, or "
                f"retrace."
            )
        if _bnd is not None:
            for _d, (_lo, _hi) in _bnd.items():
                _sz = _t.shape[_d]
                if _lo is not None and _sz < _lo:
                    _fail(
                        f"precompile: runtime input dim {_d} has size {_sz} but "
                        f"mark_unbacked declared min={_lo} (invariant 3)."
                    )
                if _hi is not None and _sz > _hi:
                    _fail(
                        f"precompile: runtime input dim {_d} has size {_sz} but "
