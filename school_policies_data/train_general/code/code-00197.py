        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_tensor_v3,
        torch._utils._rebuild_sparse_tensor,
        torch._utils._rebuild_meta_tensor_no_storage,
        torch._utils._rebuild_nested_tensor,
        torch._utils._rebuild_wrapper_subclass,
        # Allowlisting this, but not allowlisting the numpy functions by default
        # Reasoning is that we don't have control over the numpy functions, but
        # this utility is provided by pytorch
        torch._utils._rebuild_device_tensor_from_numpy,
        # In 2.6, we should no longer have a dependency on numpy and the above
        # _rebuild_device_tensor_from_numpy function.
        torch._utils._rebuild_device_tensor_from_cpu_tensor,
    }


# Unpickling machinery
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    rc: dict[str, Any] = {
        "collections.OrderedDict": OrderedDict,
        "collections.Counter": Counter,
        "torch.nn.parameter.Parameter": torch.nn.Parameter,
        "torch.serialization._get_layout": torch.serialization._get_layout,
        "torch.Size": torch.Size,
        "torch.Tensor": torch.Tensor,
        "torch.device": torch.device,
        "_codecs.encode": encode,  # for bytes
        "builtins.bytearray": bytearray,  # for bytearray
        "builtins.set": set,  # for set
        "builtins.complex": complex,  # for complex
    }

    # dtype
    for t in torch.storage._dtype_to_storage_type_map():
        rc[str(t)] = t
    for t in torch.storage._new_dtypes():
        rc[str(t)] = t
    for t in [getattr(torch, f"uint{x}") for x in range(1, 8)]:
        rc[str(t)] = t
    for t in [getattr(torch, f"int{x}") for x in range(1, 8)]:
        rc[str(t)] = t

    # Tensor classes
    for tt in torch._tensor_classes:
        rc[f"{tt.__module__}.{tt.__name__}"] = tt
    # Storage classes
    for ts in torch._storage_classes:
        if ts not in (torch.storage.TypedStorage, torch.storage.UntypedStorage):
            # Wrap legacy storage types in a dummy class
            rc[f"{ts.__module__}.{ts.__name__}"] = torch.serialization.StorageType(
                ts.__name__
            )
        else:
            rc[f"{ts.__module__}.{ts.__name__}"] = ts
    # Quantization specific
    for qt in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
        torch.per_channel_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine_float_qparams,
    ]:
        rc[str(qt)] = qt
    # Rebuild functions
    for f in _tensor_rebuild_functions():
        rc[f"torch._utils.{f.__name__}"] = f

    # Handles Tensor Subclasses, Tensor's with attributes.
    # NOTE: It calls into above rebuild functions for regular Tensor types.
    rc["torch._tensor._rebuild_from_type_v2"] = torch._tensor._rebuild_from_type_v2
    return rc


def _read_global_instruction(readline: Callable) -> tuple[str, str]:
    module = readline()[:-1].decode("utf-8")
    name = readline()[:-1].decode("utf-8")
    # Patch since torch.save default protocol is 2
    # users will be running this code in python > 3
    if (module, name) in NAME_MAPPING:
        module, name = NAME_MAPPING[(module, name)]
    elif module in IMPORT_MAPPING:
        module = IMPORT_MAPPING[module]
    return module, name


def get_globals_in_pkl(file) -> set[str]:
    globals_in_checkpoint = set()
    read = file.read
    readline = file.readline
    op_to_bytes_to_read = {
        NEWOBJ[0]: 0,
        REDUCE[0]: 0,
        BUILD[0]: 0,
        APPEND[0]: 0,
        APPENDS[0]: 0,
        SETITEM[0]: 0,
        SETITEMS[0]: 0,
        MARK[0]: 0,
        TUPLE[0]: 0,
        TUPLE1[0]: 0,
        TUPLE2[0]: 0,
        TUPLE3[0]: 0,
        NONE[0]: 0,
        NEWFALSE[0]: 0,
        NEWTRUE[0]: 0,
        EMPTY_TUPLE[0]: 0,
        EMPTY_LIST[0]: 0,
        EMPTY_DICT[0]: 0,
        EMPTY_SET[0]: 0,
        BINPERSID[0]: 0,
        BININT[0]: 4,
        BININT1[0]: 1,
        BININT2[0]: 2,
        BINFLOAT[0]: 8,
        BINGET[0]: 1,
        LONG_BINGET[0]: 4,
        BINPUT[0]: 1,
        LONG_BINPUT[0]: 4,
    }
    while True:
        key = read(1)
        if not key:
            raise EOFError
        if not isinstance(key, bytes_types):
            raise AssertionError(f"Expected bytes, got {type(key).__name__}")
        if key[0] == GLOBAL[0]:
            module, name = _read_global_instruction(readline)
            globals_in_checkpoint.add(f"{module}.{name}")
        elif key[0] in op_to_bytes_to_read:
            bytes_to_read = op_to_bytes_to_read[key[0]]
            if bytes_to_read:
                read(bytes_to_read)
        # ops where bytes to read depends on the data
        elif key[0] == BINUNICODE[0]:
            strlen = unpack("<I", read(4))[0]
            if strlen > maxsize:
                raise UnpicklingError("String is too long")
            read(strlen)
        elif key[0] in {SHORT_BINSTRING[0], LONG1[0]}:
            strlen = read(1)[0]
            read(strlen)
        # first and last op
        elif key[0] == PROTO[0]:
            read(1)[0]
        elif key[0] == STOP[0]:
            return globals_in_checkpoint
        else:
            raise UnpicklingError(f"Unsupported operand {key[0]}")


class Unpickler:
    def __init__(self, file, *, encoding: str = "bytes"):
        self.encoding = encoding
        self.readline = file.readline
        self.read = file.read
        self.memo: dict[int, Any] = {}
        self.proto: int = -1

    def load(self):
        """Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        """
        self.metastack = []
        self.stack: list[Any] = []
        self.append = self.stack.append
        read = self.read
        while True:
            key = read(1)
            if not key:
                raise EOFError
            if not isinstance(key, bytes_types):
                raise AssertionError(f"Expected bytes, got {type(key).__name__}")
            # Risky operators
            if key[0] == GLOBAL[0]:
                module, name = _read_global_instruction(self.readline)
                full_path = f"{module}.{name}"
                if module in _blocklisted_modules:
                    raise UnpicklingError(
                        f"Trying to load unsupported GLOBAL {full_path} whose module {module} is blocked."
                    )
                if full_path in _get_allowed_globals():
                    self.append(_get_allowed_globals()[full_path])
                elif full_path in _get_user_allowed_globals():
                    self.append(_get_user_allowed_globals()[full_path])
                elif full_path in (
                    [
                        "torch.nested._internal.nested_tensor.NestedTensor",
                        "torch.nested._internal.nested_tensor._rebuild_njt",
                        "torch._dynamo.decorators._DimRange",
                    ]
                ):
                    raise UnpicklingError(
                        "``torch.nested`` and ``torch._dynamo`` must be imported to load nested jagged tensors (NJTs)"
                    )
                elif full_path in (
                    [
                        "torch.distributed.device_mesh.DeviceMesh",
                        "torch.distributed.tensor._dtensor_spec.DTensorSpec",
                        "torch.distributed.tensor._dtensor_spec.TensorMeta",
                        "torch.distributed.tensor.DTensor",
                        "torch.distributed.tensor.placement_types.Partial",
                        "torch.distributed.tensor.placement_types.Replicate",
                        "torch.distributed.tensor.placement_types.Shard",
                    ]
                ):
                    raise UnpicklingError(
                        "``torch.distributed.tensor`` must be imported to load DTensors"
                    )
                else:
                    builtins_name = "builtins"
                    if (
                        builtins_name in full_path
                        and builtins_name == full_path[: len(builtins_name)]
                    ):
                        full_path = full_path[len(builtins_name) :]
                        full_path = (
                            full_path[1:]
                            if len(full_path) > 0 and full_path[0] == "."
                            else builtins_name + full_path
                        )
                    raise UnpicklingError(
                        f"Unsupported global: GLOBAL {full_path} was not an allowed global by default. "
                        f"Please use `torch.serialization.add_safe_globals([{full_path}])` or the "
                        f"`torch.serialization.safe_globals([{full_path}])` context manager to allowlist this global "
                        "if you trust this class/function."
                    )
            elif key[0] == NEWOBJ[0]:
                args = self.stack.pop()
                cls = self.stack.pop()
                if cls is torch.nn.Parameter:
                    self.append(torch.nn.Parameter(*args))
                elif (
                    cls in _get_user_allowed_globals().values()
                    or cls in _get_allowed_globals().values()
                ):
                    result = cls.__new__(cls, *args)
                    if cls in torch._tensor_classes and "sparse" in cls.__module__:
                        _sparse_tensors_to_validate.append(result)
                    self.append(result)
                else:
                    raise UnpicklingError(
                        "Can only create new object for nn.Parameter or classes allowlisted "
                        f"via `add_safe_globals` but got {cls}"
                    )
            elif key[0] == REDUCE[0]:
                args = self.stack.pop()
                func = self.stack[-1]
                if (
                    func not in _get_allowed_globals().values()
                    and func not in _get_user_allowed_globals().values()
                ):
                    error_msg = (
                        f"Trying to call reduce for unrecognized function {func}"
                    )
                    if hasattr(func, "__self__"):
                        error_msg += f" which belongs to {func.__self__}"
                    raise UnpicklingError(error_msg)
                result = func(*args)
                if func in torch._tensor_classes and "sparse" in func.__module__:
                    _sparse_tensors_to_validate.append(result)
                self.stack[-1] = result
            elif key[0] == BUILD[0]:
                state = self.stack.pop()
                inst = self.stack[-1]
                if type(inst) is torch.Tensor:
                    # Legacy unpickling

                    inst.set_(*state)
                elif type(inst) is torch.nn.Parameter:
                    inst.__setstate__(state)
                elif type(inst) is OrderedDict:
                    inst.__dict__.update(state)
                elif (
                    type(inst) in _get_user_allowed_globals().values()
                    or type(inst) in _get_allowed_globals().values()
                ):
                    if hasattr(inst, "__setstate__"):
                        inst.__setstate__(state)
                    else:
                        # mimics load_build in pickle
                        # https://github.com/python/cpython/blob/f0c6fccd08904787a39269367f09f263d496114c/Lib/pickle.py#L1854-L1867
                        slotstate = None
                        if isinstance(state, tuple) and len(state) == 2:
                            state, slotstate = state
                        if state:
                            inst.__dict__.update(state)
                        if slotstate:
                            for k, v in slotstate.items():
                                setattr(inst, k, v)
                else:
                    raise UnpicklingError(
                        "Can only build Tensor, Parameter, OrderedDict or types allowlisted "
                        f"via `add_safe_globals`, but got {type(inst)}"
                    )
            # Stack manipulation
            elif key[0] == APPEND[0]:
                item = self.stack.pop()
                list_obj = self.stack[-1]
                self._check_append_target(list_obj)
                list_obj.append(item)
            elif key[0] == APPENDS[0]:
                items = self.pop_mark()
                list_obj = self.stack[-1]
                self._check_append_target(list_obj)
                list_obj.extend(items)
            elif key[0] == SETITEM[0]:
                (v, k) = (self.stack.pop(), self.stack.pop())
                self._check_set_item_target("SETITEM")
                self.stack[-1][k] = v
            elif key[0] == SETITEMS[0]:
                items = self.pop_mark()
                self._check_set_item_target("SETITEMS")
                for i in range(0, len(items), 2):
                    self.stack[-1][items[i]] = items[i + 1]
            elif key[0] == MARK[0]:
                self.metastack.append(self.stack)
                self.stack = []
                self.append = self.stack.append
            elif key[0] == TUPLE[0]:
                items = self.pop_mark()
                self.append(tuple(items))
            elif key[0] == TUPLE1[0]:
                self.stack[-1] = (self.stack[-1],)
            elif key[0] == TUPLE2[0]:
                self.stack[-2:] = [(self.stack[-2], self.stack[-1])]
            elif key[0] == TUPLE3[0]:
                self.stack[-3:] = [(self.stack[-3], self.stack[-2], self.stack[-1])]
            # Basic types construction
            elif key[0] == NONE[0]:
                self.append(None)
            elif key[0] == NEWFALSE[0]:
                self.append(False)
            elif key[0] == NEWTRUE[0]:
                self.append(True)
            elif key[0] == EMPTY_TUPLE[0]:
                self.append(())
            elif key[0] == EMPTY_LIST[0]:
                self.append([])
            elif key[0] == EMPTY_DICT[0]:
                self.append({})
            elif key[0] == EMPTY_SET[0]:
                self.append(set())
            elif key[0] == BININT[0]:
                self.append(unpack("<i", read(4))[0])
            elif key[0] == BININT1[0]:
                self.append(self.read(1)[0])
            elif key[0] == BININT2[0]:
                self.append(unpack("<H", read(2))[0])
            elif key[0] == BINFLOAT[0]:
                self.append(unpack(">d", self.read(8))[0])
            elif key[0] == BINUNICODE[0]:
                strlen = unpack("<I", read(4))[0]
                if strlen > maxsize:
                    raise UnpicklingError("String is too long")
                strval = str(read(strlen), "utf-8", "surrogatepass")
                self.append(strval)
            elif key[0] == SHORT_BINSTRING[0]:
                strlen = read(1)[0]
                strdata = read(strlen)
                if self.encoding != "bytes":
                    strdata = strdata.decode(self.encoding, "strict")
                self.append(strdata)
            elif key[0] == BINPERSID[0]:
                pid = self.stack.pop()
                # Only allow persistent load of storage
                if type(pid) is not tuple and type(pid) is not int:
                    raise UnpicklingError(
                        f"persistent_load id must be tuple or int, but got {type(pid)}"
                    )
                if (
                    type(pid) is tuple
                    and len(pid) > 0
                    and torch.serialization._maybe_decode_ascii(pid[0]) != "storage"
                ):
                    raise UnpicklingError(
                        f"Only persistent_load of storage is allowed, but got {type(pid[0])}"
                    )
                self.append(self.persistent_load(pid))
            elif key[0] in [BINGET[0], LONG_BINGET[0]]:
                idx = (read(1) if key[0] == BINGET[0] else unpack("<I", read(4)))[0]
                self.append(self.memo[idx])
            elif key[0] in [BINPUT[0], LONG_BINPUT[0]]:
                i = (read(1) if key[0] == BINPUT[0] else unpack("<I", read(4)))[0]
                if i < 0:
                    raise ValueError("negative argument")
                self.memo[i] = self.stack[-1]
            elif key[0] == LONG1[0]:
                n = read(1)[0]
                data = read(n)
                self.append(decode_long(data))
            # First and last deserializer ops
            elif key[0] == PROTO[0]:
                self.proto = read(1)[0]
                if self.proto != 2:
                    warnings.warn(
                        f"Detected pickle protocol {self.proto} in the checkpoint, which was "
                        "not the default pickle protocol used by `torch.load` (2). The weights_only "
                        "Unpickler might not support all instructions implemented by this protocol, "
                        "please file an issue for adding support if you encounter this.",
                        stacklevel=2,
                    )
            elif key[0] == STOP[0]:
                rc = self.stack.pop()
                return rc
            else:
                raise UnpicklingError(f"Unsupported operand {key[0]}")

    # Return a list of items pushed in the stack after last MARK instruction.
    def pop_mark(self):
        items = self.stack
        self.stack = self.metastack.pop()
        self.append = self.stack.append
        return items

    def _check_set_item_target(self, opcode: str):
        if type(self.stack[-1]) not in [dict, OrderedDict, Counter]:
            raise UnpicklingError(
                f"Can only {opcode} for dict, collections.OrderedDict, "
                f"collections.Counter, but got {type(self.stack[-1])}"
            )

    def _check_append_target(self, list_obj):
        # list subclasses allowlisted via add_safe_globals can be appended to
        # (e.g. traceback.StackSummary); plain list is always allowed
        if type(list_obj) is not list and not (
            isinstance(list_obj, list)
            and type(list_obj) in _get_user_allowed_globals().values()
        ):
            raise UnpicklingError(f"Can only append to lists, but got {type(list_obj)}")

    def persistent_load(self, pid):
        raise UnpicklingError("unsupported persistent id encountered")


def load(file, *, encoding: str = "ASCII"):
    return Unpickler(file, encoding=encoding).load()
<|endoftext|># mypy: allow-untyped-defs
import abc
import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Concatenate, final, Generic, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._C import (
    _dispatch_is_included_in_alias as is_included_in_alias,
    DispatchKey,
    DispatchKeySet,
)
from torch._functorch.pyfunctorch import dispatch_functorch, TransformType
from torch.utils._python_dispatch import TorchDispatchMode


try:
    from types import NotImplementedType  # Python 3.10+
except ImportError:  # pragma: no cover
    NotImplementedType = type(NotImplemented)  # type: ignore[misc]


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


_T = TypeVar("_T", default=Any)
_P = ParamSpec("_P", default=...)


# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags")


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if not _SET_GLOBAL_FLAGS:
        yield
        return
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


class OperatorBase:
    """
    Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator
    (which represents Python-only operators that are unrepresentable in TorchScript).
    """

    def __init__(self):
        # The dispatch cache precomputes a mapping of dispatch key that the
        # dispatcher wants to dispatch to, to an actual implementation of the
        # dispatch key.  Confusingly, the actual implementation could *also* be a
        # dispatch key, but in this case, this refers to the C++ kernel that
        # was registered to some dispatch key.  Aliases are permitted in the
        # latter but not the former; for example, you might lookup the
        # entry for AutogradCPU, and this maps you to the Autograd key for
        # the generic autograd kernel that works for all devices.  Since this
        # is the Python dispatcher, you can also put an arbitrary Python
        # callable to call instead.  This handler gets precisely the
        # args/kwargs that the operator was __call__'ed with.
        # NB: This name is hard-coded in torch/csrc/autograd/python_variable.cpp
        # for use with OpOverload; cache lookup is done entirely from C++
        # for speed.
        # TODO: The cache is NOT currently used by HigherOrderOperator, but it should!
        self._dispatch_cache: dict[DispatchKey, DispatchKey | Callable[..., Any]] = {}

        # This table allows you to override the behavior of a particular
        # dispatch key to call a custom Python function, rather than the
        # ordinary C++ configured behavior.  This is the raison d'etre of  # codespell:ignore
        # Python dispatcher: to let you program the dispatcher from Python
        # in case you need something unusual, and don't want to clobber
        # the existing registrations using the Python operator registration
        # API.
        self.py_kernels: dict[DispatchKey, Callable[..., Any]] = {}

        # This table allows you to override the behavior of a particular
        # operator for a particular TorchDispatchMode.  In practice,
        # we are using this mostly for ProxyTensorMode.  Modes can be
        # thought of as an open world extension of dispatch keys, so it
        # makes sense that you should be able to register them, the same
        # way you can register dispatch keys.
        self.python_key_table: dict[
            type[TorchDispatchMode | torch.Tensor], Callable[..., Any]
        ] = {}

        # This table allows you to override the behavior of functorch
        # transformations.  NB: this currently only does something for
        # HigherOrderOperator
        self.functorch_table = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False

    def py_impl(
        self,
        k: type[TorchDispatchMode] | type[torch.Tensor] | TransformType | DispatchKey,
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        def inner(fn: Callable[_P, _T]) -> Callable[_P, _T]:
            if inspect.isclass(k) and (
                issubclass(k, TorchDispatchMode) or issubclass(k, torch.Tensor)
            ):
                if k in self.python_key_table:
                    raise AssertionError(f"{k} already registered in python_key_table")
                # TODO(voz): Should we replace setting DispatchKey.Python entirely with setting mode keys?
                self.python_key_table[k] = fn
                self._dispatch_cache.clear()
                return fn

            if isinstance(k, TransformType):
                if k in self.functorch_table:
                    raise AssertionError(f"{k} already registered in functorch_table")
                self.functorch_table[k] = fn
                return fn

            if not isinstance(k, DispatchKey):
                raise AssertionError(f"expected DispatchKey, got {type(k)}")
            if k == DispatchKey.Python:
                raise AssertionError(
                    "Please register a mode for the DispatchKey.Python key instead."
                )

            if k in self.py_kernels:
                raise RuntimeError(
                    f"Trying to override a python impl for {k} on operator {self.name()}"
                )
            self.py_kernels[k] = fn
            self._dispatch_cache.clear()
            return fn

        return inner

    # Registers an implementation to all **3** variants of functionalization that we have:
    # - DispatchKey.Functionalize
    # - functorch.TransformType.Functionalize
    # - FunctionalTensorMode
    # Example:
    #   @py_functionalize_impl
    #   def functionalize_rule(ctx, inner_f, *args):
    #       args_unwrapped = ctx.unwrap_tensors(args)
    #       with ctx.redispatch_to_next():
    #           out = ctx.functionalize(inner_f)(*args_unwrapped)
    #           return ctx.wrap_tensors(out)
    def py_functionalize_impl(
        self, fn: Callable[Concatenate["BaseFunctionalizeAPI", _P], _T]
    ) -> Callable[Concatenate["BaseFunctionalizeAPI", _P], _T]:
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI,
        )

        # Construct our three flavors of functionalization,
        # each of which have slightly different wrap/unwrap/redispatch policies
        def functionalize_dk_fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return fn(CppFunctionalizeAPI(), *args, **kwargs)

        def functionalize_dispatch_mode_fn(
            mode: FunctionalTensorMode | None, *args: _P.args, **kwargs: _P.kwargs
        ) -> _T | NotImplementedType:
            from torch._higher_order_ops.utils import has_user_subclass
            from torch._subclasses import FakeTensor
            from torch._subclasses.functional_tensor import FunctionalTensor

            if has_user_subclass(
                (args, kwargs),
                allowed_subclasses=(FakeTensor, FunctionalTensor),
            ):
                return NotImplemented
            return fn(PythonFunctionalizeAPI(mode), *args, **kwargs)

        def functionalize_functorch_fn(
            interpreter, *args: _P.args, **kwargs: _P.kwargs
        ) -> _T:
            return fn(FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)

        self.py_impl(DispatchKey.Functionalize)(functionalize_dk_fn)
        self.py_impl(FunctionalTensorMode)(functionalize_dispatch_mode_fn)
        self.py_impl(TransformType.Functionalize)(functionalize_functorch_fn)

        return fn

    def name(self):
        raise NotImplementedError


# Equivalent to computeDispatchTableEntryWithDebug
def resolve_key(op: OperatorBase, k: DispatchKey):  # type: ignore[valid-type]
    # 1. (Direct) operator registration
    if op.has_kernel_for_dispatch_key(k):
        return k
    # 2.1 Use CompositeExplicitAutogradNonFunctional kernel if available
    cand = DispatchKey.CompositeExplicitAutogradNonFunctional
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.2 Use CompositeExplicitAutograd kernel if available
    cand = DispatchKey.CompositeExplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    has_backend_kernel = op.has_kernel_for_any_dispatch_key(
        torch._C._dispatch_get_backend_keyset_from_autograd(k)
    ) or op.has_kernel_for_dispatch_key(DispatchKey.CompositeExplicitAutograd)
    # 2.3. Use CompositeImplicitAutograd kernel if available
    cand = DispatchKey.CompositeImplicitAutogradNestedTensor
    if (
        (k != DispatchKey.Undefined and is_included_in_alias(k, cand))
        and op.has_kernel_for_dispatch_key(cand)
        and not has_backend_kernel
    ):
        return cand
    cand = DispatchKey.CompositeImplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        if k == DispatchKey.AutogradOther and op.has_kernel_for_any_dispatch_key(
            torch._C._dispatch_autogradother_backends
        ):
            raise RuntimeError("ambiguous autogradother kernel")
        elif not has_backend_kernel:
            return cand
    # 2.4. For autograd backend keys, use kernel from DispatchKey::Autograd if available
    cand = DispatchKey.Autograd
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.5 Use kernel from DispatchKey::FuncTorchBatchedDecomposition if available
    cand = DispatchKey.FuncTorchBatchedDecomposition
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # Backend fallback
    if torch._C._dispatch_has_backend_fallback(k):
        # The dispatch key itself will implicitly route to backend fallback.
        # This is probably not great for the pure Python implementation.
        return k
    raise NotImplementedError(f"could not find kernel for {op} at dispatch key {k}")


_higher_order_ops: dict[str, "HigherOrderOperator"] = {}

_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = [
    DispatchKey.PythonDispatcher,  # type: ignore[attr-defined]
    DispatchKey.PythonTLSSnapshot,  # type: ignore[attr-defined]
    DispatchKey.ADInplaceOrView,
    DispatchKey.BackendSelect,
    DispatchKey.AutocastCPU,  # type: ignore[attr-defined]
    DispatchKey.AutocastCUDA,  # type: ignore[attr-defined]
    DispatchKey.AutocastXPU,  # type: ignore[attr-defined]
]


class HigherOrderOperator(OperatorBase, abc.ABC):
    # The HigherOrderOperator will appear as torch.ops.higher_order.{name}
    #
    # If you're creating a new HigherOrderOperator, please do not change the
    # default. Adding operators to the global torch.ops namespace is a bad
    # practice due to name collisions.
    def __init__(
        self, name, *, cacheable=False, supports_training_input_mutation=False
    ):
        super().__init__()
        if type(self) is HigherOrderOperator:
            raise RuntimeError(
                "Direct instantiation of HigherOrderOperator is not allowed. Please subclass it."
            )
        self._name = name

        # Make _OPNamespace not scream, this whole name based association needs a good hard look
        self.__name__ = name
        _higher_order_ops[name] = self
        self._ns = "higher_order"
        self.__module__ = "torch.ops.higher_order"
        self._cacheable = cacheable
        # When True, allows mutating inputs that don't require grad during
        # training. Mutations must be handled by auto_functionalize before
        # reaching autograd.
        self._supports_training_input_mutation = supports_training_input_mutation

        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        # [NOTE] We have to register pre-dispatch key implementation
        # because sometimes HOP use aot-dispatch tracing to detect certain
        # mutations. This is problematic when we are functionalizing HOP
        # during pre-dispatch because when the inner tracer starts, it will see
        # that PreDispatch key is still active. In that case, we just redispatch
        # it to next key. This is only safe to do when PreDispatch key stack has no
        # active modes.

    def py_impl(
        self,
        k: type[TorchDispatchMode] | type[torch.Tensor] | TransformType | DispatchKey,
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        if isinstance(k, DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        return super().py_impl(k)

    def py_autograd_impl(
        self,
        fn: Callable[_P, _T],
    ) -> Callable[_P, _T]:
        def maybe_run_autograd(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if not torch.is_grad_enabled() or pytree.tree_all_only(
                torch.Tensor,
                lambda t: not t.requires_grad,  # type: ignore[union-attr]
                (*args, kwargs),
            ):
                with torch._C._AutoDispatchBelowAutograd():
                    return self(*args, **kwargs)

            from torch._higher_order_ops.utils import _has_gen_schema

            if not self._supports_training_input_mutation and _has_gen_schema(self):
                schema = self.gen_schema(*args, **kwargs)
                if any(arg.is_write for arg in schema.arguments):
                    raise RuntimeError(
                        f"The {self.name()} HigherOrderOperator does not currently support training "
                        "with in-place input or buffer mutations "
                        "If you require this feature, please submit an issue to PyTorch. "
                        "Alternatively, consider creating your own custom autograd.Function. "
                    )

            return fn(*args, **kwargs)

        self.py_impl(DispatchKey.Autograd)(maybe_run_autograd)

        return fn

    @property
    def namespace(self):
        return self._ns

    @final
    def cacheable(self) -> bool:
        from torch._functorch.autograd_function import AutogradFunctionApply

        return (
            self._cacheable
            or f"{self.__module__}.{self.__name__}"
            in torch._inductor.config.unsafe_marked_cacheable_functions
            or (
                isinstance(self, AutogradFunctionApply)
                and torch._functorch.config.autograd_cache_allow_custom_autograd_functions
            )
        )

    def fallthrough(self, dispatch_key):
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def _get_overloaded_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        # Default HOP behavior matches handle_torch_function_no_python_arg_parser
        # in torch/csrc/utils/python_arg_parser.cpp.
        overloaded_args: list[torch.Tensor] = []

        def has_python_key(tensor):
            return torch._C._dispatch_keys(tensor).has("Python")

        def check_overloaded(arg):
            if isinstance(arg, torch.Tensor) and has_python_key(arg):
                overloaded_args.append(arg)

        for arg in (*args, *kwargs.values()):
            check_overloaded(arg)
            if isinstance(arg, (list, tuple)):
                for a in arg:
                    check_overloaded(a)

        return tuple(overloaded_args)

    # Use positional-only argument to avoid naming collide with custom ops arguments
    # that are named "self".
    def dispatch(self, /, dispatch_key, *args, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            if isinstance(kernel, DispatchKey):
                raise AssertionError(f"unexpected DispatchKey in cache: {kernel}")
            return kernel(*args, **kwargs)

        if dispatch_key == DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)

        if dispatch_key == DispatchKey.Python:
            overloaded_args = self._get_overloaded_args(args, kwargs)

            # Step 1: dispatch on any user TorchDispatchModes
            from torch.utils._python_dispatch import _pop_mode_temporarily

            curr_mode = _get_current_dispatch_mode()
            if curr_mode is not None:
                if type(curr_mode) in self.python_key_table:
                    handler = self.python_key_table[type(curr_mode)]
                    with _pop_mode_temporarily() as mode:
                        # "natural" calling convention: (mode, *args, **kwargs)
                        # TODO(rzou): we should support torch_dispatch calling convention too.
                        result = handler(mode, *args, **kwargs)
                else:
                    if curr_mode.supports_higher_order_operators:
                        with _pop_mode_temporarily() as mode:
                            return curr_mode.__torch_dispatch__(self, [], args, kwargs)
                    else:
                        raise NotImplementedError(
                            f"There was no rule registered for HigherOrderOperator {self._name} and mode {curr_mode}."
                            f"Hint: set {curr_mode}'s supports_higher_order_operators to True."
                            f" This causes all higher order operators to pass through {curr_mode}'s __torch_dispatch__,"
                            f" so handle them accordingly by"
                            f" adding support for HigerOrderOperators (in this case, {self._name}) in"
                            f" {curr_mode}.__torch_dispatch__ or"
                            f" returning NotImplemented when not supported."
                        )
                if result is not NotImplemented:
                    return result

            # Step 2: dispatch on any subclasses
            for arg in overloaded_args:
                subclass_type = type(arg)
                if (
                    subclass_type.__torch_dispatch__
                    is torch._C._disabled_torch_dispatch_impl
                ):
                    continue

                # In some case, people are using FakeTensor without a FakeTensorMode.
                # For example, some sparse arch model has a mix of FakeTensor and real
                # tensor for weights during lowering, and ppl tends to run eager evaluation
                # on the model without setting up the FakeTensorMode.
                # In this case, we pull FakeTensorMode impl.
                if subclass_type is torch._subclasses.fake_tensor.FakeTensor:
                    subclass_type = torch._subclasses.fake_tensor.FakeTensorMode  # type: ignore[assignment]
                    handler = self.python_key_table[subclass_type]
                    result = handler(arg.fake_mode, *args, **kwargs)  # type: ignore[attr-defined]
                    return result

                if subclass_type in self.python_key_table:
                    handler = self.python_key_table[subclass_type]
                    # "natural" calling convention: (*args, **kwargs)
                    # TODO(rzou): we should support torch_dispatch calling convention too.
                    result = handler(*args, **kwargs)
                else:
                    raise NotImplementedError(
                        f"There was no rule registered for HOP {self._name} and subclass {subclass_type}. "
                        f"We recommend filing an issue."
                    )
                if result is not NotImplemented:
                    return result

            # All handlers returned NotImplemented
            raise TypeError(
                f"HigherOrderOperator '{self._name}' is not supported for the given input types. "
                f"This typically happens when using custom tensor types or dispatch modes that don't "
                f"have implementations for this operation.\n\n"
                f"Current mode: {curr_mode}\n"
                f"Input types: {[type(a).__name__ for a in overloaded_args]}\n\n"
                f"To fix this, can add support for '{self._name}' in {curr_mode}'s __torch_dispatch__\n"
            )

        functionality_key = torch._C._to_functionality_key(dispatch_key)  # type: ignore[attr-defined]
        if functionality_key == DispatchKey.PreDispatch:
            from torch.utils._python_dispatch import _pop_mode_temporarily

            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                _len_torch_dispatch_stack_pre_dispatch() > 0
            ) and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                DispatchKey.Python
            ):
                curr_mode = _get_current_dispatch_mode_pre_dispatch()
                if curr_mode is None:
                    raise AssertionError(
                        "Illegal invocation of dispatch on DispatchKey.PreDispatch without a mode."
                    )
                if type(curr_mode) not in self.python_key_table:
                    raise AssertionError(
                        f"Current active mode {curr_mode} not registered"
                    )
                handler = self.python_key_table[type(curr_mode)]
                with _pop_mode_temporarily(functionality_key) as mode:
                    return handler(mode, *args, **kwargs)

        final_key = resolve_key(self, dispatch_key)

        # This can current fail due to backend fallbacks.  You just have to
        # register them by hand for HigherOrderOperator.
        if final_key not in self.py_kernels:
            raise NotImplementedError(
                f"could not find kernel for HigherOrderOperator {self._name} "
                f"at dispatch key {final_key} (resolved from {dispatch_key})"
            )

        # [NOTE] We shouldn't cache PreDispatch kernel here because depending
        # on what modes are active, predispatch behaviour is different.
        # Also we do same thing for normal ops:
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        if dispatch_key != DispatchKey.PreDispatch:
            self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]
        kernel = self.py_kernels[final_key]
        # It's illegal to register DispatchKey to py_kernels, since there's no
        # C++ kernel to call into
        if isinstance(kernel, DispatchKey):
            raise AssertionError(f"unexpected DispatchKey in py_kernels: {kernel}")
        return kernel(*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, /, *args, **kwargs):
        flat_args = _to_flat_tuple(args, kwargs)
        if torch.overrides.has_torch_function(flat_args):
            return torch.overrides.handle_torch_function(
                self, flat_args, *args, **kwargs
            )
        del flat_args

        dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
        return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)

    # NOTE [HigherOrderOperator Schema]
    # Each invocation of a HigherOrderOperator (hop) should have its own schema because
    # the subgraphs and the arguments can be different even for the same hop.
    #
    # Each hop should implement its own gen_schema method, which should
    # take the same input as the __call__ method and returns a FunctionSchema.
    # The schema provides a unified way to check if the hop mutates its inputs,
    # which can be useful in implementing optimizations.
    #
    # If the hop doesn't implement the gen_schema method,
    # we expect it to be functional. It should not mutate its inputs and there
    # are no input, output aliasing via views or direct referencing.
    def gen_schema(self, *args, **kwargs):
        raise NotImplementedError(
            f"HigherOrderOperator {self._name} does not implement a gen_schema. "
            f"This is OK as long as the hop is functional. "
            f"e.g. it should not mutate its inputs and there are no input, output aliasing "
            f"via views or direct referencing."
        )

    def __str__(self):
        return f"{self.name()}"

    def name(self):
        return self._name

    # it's a no-op since HigherOrderOperator is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self


def _to_flat_tuple(args, kwargs):
    return pytree.arg_tree_leaves(*args, **kwargs)


def _compute_keyset(args, kwargs, non_fallthrough_keys):
    tensors = _get_tensors(args, kwargs)
    return key_extractor(tensors, non_fallthrough_keys)


def _get_tensors(args, kwargs):
    flat_all = _to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)


# Note - this should maintain identical impl to the C++ dispatcher key extraction logic
# at ATen/core/dispatch/DispatchKeyExtractor.h
def key_extractor(tensors, key_mask):
    key_set = torch._C._dispatch_tls_local_include_set()
    for tensor in tensors:
        key_set = key_set | torch._C._dispatch_keys(tensor)
    key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
    key_set = key_set & key_mask
    return key_set


# Mode stack for PreDispatchKey
# it should always have three keys with
# priority given to FunctionalTensorMode and
# then ProxyTorchDispatchMode. It means that
# slot 0 belongs to ProxyTorchDispatchMode and
# slot 1 belongs to FunctionalTensorMode.
#
# SchemaCheckMode is separate from the other 2,
# and is only valid when the stack is empty.
# SchemaCheckMode is for testing purposes, and
# is meant to run in eager mode on concrete inputs,
# checking for incorrect schemas in regards to
# aliasing or mutating ops.
class _ModeStackStateForPreDispatch:
    def __init__(self):
        self.__infra_modes = [None, None]
        self._schema_check_mode = None

    def set(self, index, mode):
        if index >= len(self.__infra_modes):
            raise AssertionError(
                f"index {index} out of bounds for infra_modes length {len(self.__infra_modes)}"
            )
        self.__infra_modes[index] = mode

    def get(self, index):
        if index >= len(self.__infra_modes):
            raise AssertionError(
                f"index {index} out of bounds for infra_modes length {len(self.__infra_modes)}"
            )
        return self.__infra_modes[index]

    def count(self):
        return len([i for i in self.__infra_modes if i is not None]) + int(
            self._schema_check_mode is not None
        )


_mode_stack_state_for_pre_dispatch = _ModeStackStateForPreDispatch()


def unset_mode_pre_dispatch(mode_key, schema_check=False):
    current_mode_stack_pre_dispatch = mode_stack_state_for_pre_dispatch()
    valid_keys = (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )
    if mode_key is not None and mode_key not in valid_keys:
        raise AssertionError(
            f"mode_key must be None or one of {valid_keys}, got {mode_key}"
        )
    if schema_check:
        if mode_key is not None:
            raise AssertionError("mode_key must be None when schema_check is True")

    def _unset_mode():
        # NOTE: Using `is` rather than `==` to work around slow enum comparison in
        # pybind11.
        if mode_key is torch._C._TorchDispatchModeKey.PROXY:
            current_mode = current_mode_stack_pre_dispatch.get(0)
            mode_stack_state_for_pre_dispatch().set(0, None)
            return current_mode
        elif mode_key is torch._C._TorchDispatchModeKey.FUNCTIONAL:
            current_mode = current_mode_stack_pre_dispatch.get(1)
            mode_stack_state_for_pre_dispatch().set(1, None)
            return current_mode
        else:
            current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
            mode_stack_state_for_pre_dispatch()._schema_check_mode = None
            return current_mode

    current_mode = _unset_mode()

    new_pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()
    # When we are unsetting a mode, we need to check if there is
    # active mode left on the PreDispatch key. If there is nothing
    # active, we need to remove PreDispatch key from local dispatch include
    # set.
    if new_pre_dispatch_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(DispatchKey.PreDispatch, False)

    return current_mode


def _set_mode_pre_dispatch(mode):
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    if not isinstance(
        mode,
        (
            FunctionalTensorMode,
            ProxyTorchDispatchMode,
            SchemaCheckMode,
        ),
    ):
        raise AssertionError(
            f"mode must be FunctionalTensorMode, ProxyTorchDispatchMode, or SchemaCheckMode, got {type(mode)}"
        )

    previous_mode_stack_len = _len_torch_dispatch_stack_pre_dispatch()
    if isinstance(mode, SchemaCheckMode):
        current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
        if previous_mode_stack_len > 0:
            raise AssertionError(
                "SchemaCheckMode for pre-dispatch must be used exclusively, found other modes on the stack"
            )
        mode_stack_state_for_pre_dispatch()._schema_check_mode = mode
    elif isinstance(mode, FunctionalTensorMode):
        current_mode = mode_stack_state_for_pre_dispatch().get(1)
        if current_mode is not None:
            raise AssertionError(
                f"FunctionalTensorMode slot already occupied by {current_mode}"
            )
        mode_stack_state_for_pre_dispatch().set(1, mode)
    else:
        current_mode = mode_stack_state_for_pre_dispatch().get(0)
        if current_mode is not None:
            raise AssertionError(
                f"ProxyTorchDispatchMode slot already occupied by {current_mode}"
            )
        mode_stack_state_for_pre_dispatch().set(0, mode)

    # When we are setting a mode, we need to check if there is
    # active mode left on the PreDispatch key. If there was nothing
    # active before setting this mode, it means that PreDispatch key
    # was turned off. So we need to turn it on again.
    if previous_mode_stack_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(DispatchKey.PreDispatch, True)


def _pop_mode_from_pre_dispatch():
    mode_stack = mode_stack_state_for_pre_dispatch()
    pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()

    if pre_dispatch_len == 0:
        raise AssertionError("Trying to pop empty mode stack")

    if mode_stack._schema_check_mode is not None:
        return unset_mode_pre_dispatch(None, schema_check=True)
    if mode_stack.get(1) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    if mode_stack.get(0) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)


def _len_torch_dispatch_stack_pre_dispatch():
    return mode_stack_state_for_pre_dispatch().count()


def _get_dispatch_mode_pre_dispatch(mode_key):
    # NOTE: Using `is` rather than `==` to work around slow enum comparison in pybind11.
    if mode_key is torch._C._TorchDispatchModeKey.PROXY:
        return mode_stack_state_for_pre_dispatch().get(0)
    else:
        if mode_key is not torch._C._TorchDispatchModeKey.FUNCTIONAL:
            raise AssertionError(
                f"mode_key must be PROXY or FUNCTIONAL, got {mode_key}"
            )
        return mode_stack_state_for_pre_dispatch().get(1)


def _get_current_dispatch_mode_pre_dispatch():
    if mode_stack_state_for_pre_dispatch()._schema_check_mode is not None:
        return mode_stack_state_for_pre_dispatch()._schema_check_mode
    else:
        stack_len = mode_stack_state_for_pre_dispatch().count()
        if stack_len == 2:
            return mode_stack_state_for_pre_dispatch().get(1)
        if stack_len == 1:
            return (
                mode_stack_state_for_pre_dispatch().get(1)
                if mode_stack_state_for_pre_dispatch().get(1) is not None
                else mode_stack_state_for_pre_dispatch().get(0)
            )
    return None


def mode_stack_state_for_pre_dispatch():
    global _mode_stack_state_for_pre_dispatch
    return _mode_stack_state_for_pre_dispatch


cached_ops: set["OpOverload"] = set()


def add_cached_op(op_overload):
    global cached_ops
    cached_ops.add(op_overload)


def reset_cached_ops():
    global cached_ops
    cached_ops.clear()


def get_cached_ops():
    global cached_ops
    return cached_ops


@dataclass
class _PyObjectDispatcher(Generic[_P, _T]):
    # [NOTE: PyObject Dispatcher aka pyobj_dispatcher]
    #
    # Custom operators whose kernels are implemented in Python currently need
    # to make 1+ roundtrips into the C++ PyTorch dispatcher. These roundtrips
    # are expensive; the main expensive thing is converting a PyObject
    # to an IValue requires copying the at::Tensor, incurring at::Tensor
    # and PyObject (at::Tensor owns a PyObject) refcount bumps.
    #
    # Instead, we introduce a new type of dispatching, "PyObject Dispatching".
    # When dispatching an operator, we avoid converting PyObject into IValues,
    # instead doing the dispatch key computation in a reimplementation in the
    # Python-C API. This dispatch is implemented faithfully compared to the
    # C++ dispatcher and shares helper functions.
    #
    # After we have computed a DispatchKey to dispatch on, we query the C++
    # Dispatcher for the kernel to be dispatched on. If the kernel is a Python
    # kernel, then we directly pass the PyObject args to the Python kernel.
    # if the kernel is a C++ kernel, then we perform a C++ Dispatcher redispatch
    # (which ends up doing the expensive at::Tensor copies).
    dispatch: Callable[_P, _T]
    redispatch: Callable[Concatenate[DispatchKeySet, _P], _T]


# Each OpOverload object contains pointer to a specific operator overload, a pointer to the parent `OpOverloadPacket` object.
# You can obtain an OpOverload object through attribute query on OpOverloadPacket.
class OpOverload(OperatorBase, Generic[_P, _T]):
    def __init__(
        self,
        overloadpacket: "OpOverloadPacket",
        op: Callable[_P, _T],
        op_dk: Callable[Concatenate[DispatchKey, _P], _T],
        schema: torch._C.FunctionSchema,
        tags: list[Any],
    ) -> None:
        super().__init__()
        self._op = op
        # _op may be swapped to the PyObject dispatch callable. Keep the C++
        # dispatcher handle separately for code that must bypass that fast path.
        self._cpp_dispatch_handle = op
        self._op_dk = op_dk
        self._schema = schema
        self._pyobj_dispatcher: _PyObjectDispatcher[_P, _T] | None = None
        self._overloadpacket = overloadpacket
        self._tags = tags
        self._overloadname = (
            "default" if schema.overload_name == "" else schema.overload_name
        )
        if tags:
            self._nondeterministic_seeded = torch.Tag.nondeterministic_seeded in tags
        self._name = self._schema.name
        if schema.overload_name:
            self._name += "." + schema.overload_name
        self.__name__ = f"{self._schema.name.split('::')[1]}.{self._overloadname}"
        self.__module__ = overloadpacket.__module__
        op.__module__ = overloadpacket.__module__
        self.__qualname__ = self._name
        self.__annotations__ = {}

        # If the OpOverload was constructed from a Library.def in Python.
        self._defined_in_python = self.__qualname__ in torch.library._defs

        # Logic replicated from aten/src/ATen/native/MathBitsFallback.h
        is_write = None
        for a in self._schema.arguments:  # pyrefly: ignore  # bad-assignment
            if a.alias_info is None:
                continue
            if is_write is None:
                is_write = a.alias_info.is_write
            else:
                # We will conservatively call mixed mutable/non-mutable
                # aliased inputs as NOT a view
                is_write = a.alias_info.is_write or is_write
        self.is_view = is_write is not None and not is_write

    @cached_property
    def _namespace(self) -> str:
        return self._schema.name.split("::", maxsplit=1)[0]

    @cached_property
    def _opname(self) -> str:
        return self._schema.name.split("::", maxsplit=1)[1]

    @cached_property
    def _handle(self) -> torch._C._DispatchOperatorHandle:
        # Handle to the C++ dispatcher operator entry, used for boxed dispatch.
        return torch._C._dispatch_find_schema_or_throw(
            self._schema.name, self._schema.overload_name
        )

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f"<OpOverload(op='{self._namespace}.{self._opname}', overload='{self._overloadname}')>"

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if (
            self._pyobj_dispatcher is not None
            and torch._C._peek_should_skip_torch_function()
        ):
            return self._cpp_dispatch_handle(*args, **kwargs)
        return self._op(*args, **kwargs)

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def redispatch(
        self, /, keyset: torch._C.DispatchKeySet, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        pyobj_dispatcher = self._pyobj_dispatcher
        if pyobj_dispatcher is not None:
            return pyobj_dispatcher.redispatch(keyset, *args, **kwargs)
        return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]

    def __hash__(self):
        return hash(self._cpp_dispatch_handle)

    # `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    def has_kernel_for_dispatch_key(self, k: DispatchKey) -> bool:
        return super().has_kernel_for_dispatch_key(
            k
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    def has_kernel_for_any_dispatch_key(self, ks: torch._C.DispatchKeySet) -> bool:
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(
            self.name(), ks
        ) or super().has_kernel_for_any_dispatch_key(ks)

    @property
    def namespace(self) -> str:
        return self._namespace

    def _can_decompose(self) -> bool:
        dk = DispatchKey.CompositeImplicitAutograd
        return dk in self.py_kernels or torch._C._dispatch_has_kernel_for_dispatch_key(
            self.name(), dk
        )

    def decompose(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        dk = DispatchKey.CompositeImplicitAutograd
        if dk in self.py_kernels:
            # NB: This branch is not too necessary anymore, because we can
            # apply Python CompositeImplicitAutograd *before* tracing
            # using Python dispatcher (also taking advantage of the autograd
            # formula).  But it's included for completeness
            return self.py_kernels[dk](*args, **kwargs)
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        else:
            return NotImplemented  # pyrefly: ignore [bad-return]

    # Remove a dispatch key from the dispatch cache.  This will force it to get
    # recomputed the next time.  Does nothing
    # WARNING: if you register a dispatch key to py_kernels of an OpOverload,
    # calling _del_dispatch on that key is NOT sufficient to apply your change,
    # because a single registration may affect MULTIPLE dispatch keys (e.g.,
    # registering Autograd affects AutogradCPU).  del_dispatch is to be used
    # only if you are specifically modifying how get_dispatch handles a
    # particular input 'key'.
    def _uncache_dispatch(self, key: DispatchKey) -> None:
        self._dispatch_cache.pop(key, None)

    def _is_pyobj_dispatcher_enabled(self) -> bool:
        return self._pyobj_dispatcher is not None

    def _can_enable_pyobj_dispatch(self) -> bool:
        # TODO(#187974): Support non-Tensor returns by normalizing Python kernel
        # returns against the operator schema in the PyObject fast path.
        return all(
            isinstance(ret.type, torch.TensorType) for ret in self._schema.returns
        )

    def _enable_pyobj_dispatch(self, enabled: bool = True) -> None:
        if self._is_pyobj_dispatcher_enabled() == enabled:
            return
        if not enabled:
            self._pyobj_dispatcher = None
            self._op = self._cpp_dispatch_handle
            return
        if not self._can_enable_pyobj_dispatch():
            return
        dispatch, redispatch = torch._C._dispatch_make_pyobj_dispatch_fns(
            self._handle,
            self._cpp_dispatch_handle,
            self._handle.redispatch_boxed,
        )
        self._pyobj_dispatcher = _PyObjectDispatcher(dispatch, redispatch)
        self._op = dispatch

    # This implements the pre-computation logic for the Python dispatcher.
    def _get_dispatch(self, key: DispatchKey) -> DispatchKey | Callable[_P, _T]:
        # This is only called upon a cache miss
        if key in self._dispatch_cache:
            raise AssertionError(f"{self} {key} already in dispatch cache")

        if key == DispatchKey.Python:
            if not isinstance(self, TorchBindOpOverload) and not self.python_key_table:
                self._dispatch_cache[key] = key
                add_cached_op(self)
                return key

            def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                from torch.utils._python_dispatch import _get_current_dispatch_mode

                # TODO: We also need to handle tensor subclasses here
                # TODO(voz): We should walk all the nodes here / turn it into a list, topmode is ok for now.
                curr_mode = type(_get_current_dispatch_mode())
                if curr_mode is None:
                    raise AssertionError(
                        "Illegal invocation of dispatch on DispatchKey.Python without a mode."
                    )

                if curr_mode not in self.python_key_table:
                    if isinstance(self, TorchBindOpOverload):
                        with (
                            torch.utils._python_dispatch._pop_mode_temporarily() as mode
                        ):
                            return torch._library.utils.handle_dispatch_mode(
                                mode, self, *args, **kwargs
                            )
                    else:
                        return self._op_dk(key, *args, **kwargs)

                with torch.utils._python_dispatch._pop_mode_temporarily() as mode:
                    return self.python_key_table[curr_mode](mode, *args, **kwargs)  # type: ignore[index]

            self._dispatch_cache[key] = handler
            add_cached_op(self)
            return handler

        functionality_key = torch._C._to_functionality_key(key)  # type: ignore[attr-defined]
        if functionality_key == DispatchKey.PreDispatch:
            curr_stack_len = _len_torch_dispatch_stack_pre_dispatch()
            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                curr_stack_len > 0
                and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                    DispatchKey.Python
                )
            ):

                def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                    @contextlib.contextmanager
                    def _temporarily_pop_modes_from_pre_dispatch():
                        top_mode = _pop_mode_from_pre_dispatch()
                        try:
                            yield top_mode
                        finally:
                            _set_mode_pre_dispatch(top_mode)

                    with _temporarily_pop_modes_from_pre_dispatch() as curr_mode:
                        return torch._library.utils.handle_dispatch_mode(
                            curr_mode, self, *args, **kwargs
                        )

                # Note [Not Caching Per-Dispatch-Key Mode Handlers]
                # Note that we're not caching this handler.  There isn't really a point, since the slow bit
                # is the handler itself (in python).
                # Also, not caching means that we don't have to reset the cache when any existing
                # modes go out of scope (which in of itself takes time to loop through all operators).
                return handler

        final_key = resolve_key(self, key)

        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        cache_result = key != DispatchKey.PreDispatch

        # TODO: We could potentially have lots of debugging wrappers against
        # dispatch keys; design some general registration mechanism instead of
        # having if statement for each of them
        if key == DispatchKey.Functionalize:
            import torch._dispatch.python as pydispatch

            if pydispatch.CROSSREF_FUNCTIONALIZE:
                handler = pydispatch.make_crossref_functionalize(self, final_key)  # type: ignore[assignment]
                if cache_result:
                    self._dispatch_cache[key] = handler
                    add_cached_op(self)
                return handler

        r = self.py_kernels.get(final_key, final_key)
        if cache_result:
            self._dispatch_cache[key] = r
            add_cached_op(self)
        return r

    def name(self):
        return self._name

    @property
    def overloadpacket(self):
        return self._overloadpacket

    @property
    def op(self):
        return self._op

    @property
    def tags(self):
        return self._tags

    # TODO: add more methods to expose information about input and output arguments


# TorchBindOpOverload are those custom ops which have at least one overload's
# schema consists of torch.ScriptObject (i.e. custom class) input.
# TorchBindOpOverload will skip C++ dispatcher and purely dispatched in python
# when its inputs contain FakeScriptObject in a similar way as higher order ops.
class TorchBindOpOverload(OpOverload[_P, _T]):
    def _fallthrough_keys(self) -> list[DispatchKey]:
        # TODO: we should be calling the fallback for these, but a fallthrough is almost close
        # enough to the fallback in most cases that we care about.
        _DEFAULT_FALLTHROUGH_KEYS = [
            DispatchKey.Autograd,
            DispatchKey.AutogradCPU,
            DispatchKey.AutogradCUDA,
            DispatchKey.ADInplaceOrView,
            DispatchKey.BackendSelect,
            DispatchKey.PythonTLSSnapshot,
            DispatchKey.PythonDispatcher,
            DispatchKey.Functionalize,
        ]

        def _may_use_fallthrough_instead_of_fallback(key: DispatchKey):
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), key):
                return torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                    self.name(), key
                )

            return (
                key not in self.py_kernels
                or self.py_kernels[key] is torch.library.fallthrough_kernel
            )

        return [
            key
            for key in _DEFAULT_FALLTHROUGH_KEYS
            if _may_use_fallthrough_instead_of_fallback(key)
        ]

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if _must_dispatch_in_python(args, kwargs):
            # When any inputs are FakeScriptObject, we need to
            # skip c++ dispatcher and dispatch in python through _get_dispatch of python_dispatcher
            # because C++ dispatcher will check the schema and cannot recognize FakeScriptObject.
            return self._dispatch_in_python(self._fallthrough_keys(), *args, **kwargs)
        return self._op(*args, **kwargs)

    def _dispatch_in_python(
        self, fallthrough_keys: list[DispatchKey], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        non_fallthrough_keys = torch._C._dispatch_keyset_full()
        for key in fallthrough_keys:
            non_fallthrough_keys = non_fallthrough_keys.remove(key)

        dispatch_key_set = _compute_keyset(args, kwargs, non_fallthrough_keys)
        dispatch_key = dispatch_key_set.highestPriorityTypeId()

        handler = (
            self._get_dispatch(dispatch_key)
            if dispatch_key not in self._dispatch_cache
            else self._dispatch_cache[dispatch_key]
        )

        if isinstance(handler, DispatchKey):
            # fallthrough keys can be registered at runtime via torch.library.impl
            # so need to add it to fallthrough_keys and re-dispatch.
            if torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                self.name(), dispatch_key
            ):
                return self._dispatch_in_python(
                    fallthrough_keys + [dispatch_key],
                    *args,
                    **kwargs,
                )

            raise RuntimeError(
                f"Torchbind op {self} received a FakeScriptObject input when dispatching {handler}."
                f" but no python implementation is found."
                f" Please file an issue on this when you encounter this error."
                f" This error can happen when you export or compile the model."
                f" It can still happen even if a C++ implementation for {dispatch_key}. "
                f" has been registered. That's because FakeScriptObject purely lives in python and cannot work "
                f" with a C++ implementation."
            )

        if not isinstance(handler, Callable):  # type: ignore[arg-type]
            raise AssertionError(f"handler must be callable, got {type(handler)}")
        return handler(*args, **kwargs)


def _contains_fake_script_object(obj) -> bool:
    """Check if obj is or contains a FakeScriptObject.
    This is load-bearing for TorchBindOpOverloads so we avoid pytree
    since it's much slower.
    """
    if isinstance(obj, torch._library.fake_class_registry.FakeScriptObject):
        return True
    elif isinstance(obj, (list, tuple)):
        return any(_contains_fake_script_object(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_fake_script_object(v) for v in obj.values())
    return False


def _must_dispatch_in_python(args, kwargs) -> bool:
    return any(_contains_fake_script_object(arg) for arg in args) or (
        bool(kwargs) and any(_contains_fake_script_object(v) for v in kwargs.values())
    )


def _has_script_object_arg(schema: torch.FunctionSchema) -> bool:
    return any(isinstance(arg.type, torch.ClassType) for arg in schema.arguments)


# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket(Generic[_P, _T]):
    __file__: ClassVar[str] = "torch.ops"

    def __init__(
        self,
        qualified_op_name: str,
        op_name: str,
        op: Callable[_P, _T],
        overload_names: list[str],
    ) -> None:
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names
        self._dir: list[str] = []
        self._has_torchbind_op_overload = any(
            _has_script_object_arg(schema) for schema in self._schemas.values()
        )

    # it's a no-op since OpOverloadPacket object is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    @property
    def op(self):
        return self._op

    @property
    def _schemas(self):
        return {
            overload_name: torch._C._get_schema(self._qualified_op_name, overload_name)
            for overload_name in self._overload_names
        }

    def __getattr__(self, key: str) -> OpOverload[_P, _T]:
        # ensure that query for dunder attributes that does not exist on
        # opoverloadpacket but instead exists on the self._op object does not unnecessarily call
        # `_get_operation_overload` (which is an expensive operation).
        # This is done to prevent any potential slowdown. This list can be extended
        # if there exists other attributes like `__name__` that only exist on self._op and not on the
        # opoverloadpacket.
        # This is ok since we are guaranteed that an overload name for an aten op can't start with '__'
        try:
            if key.startswith("__"):
                return getattr(self._op, key)
        except AttributeError:
            # for consistency because it seems weird to
            # throw an attribute error with a message containing
            # an object name different from the one the attribute
            # query was performed on.
            raise AttributeError(
                f"'{str(self)}' can't have an overload name beginning with '__' and the "
                f"underlying op {str(self._op)} has no attribute {key} either."
            ) from None

        try:
            # This is ok since we are guaranteed that an overload name for an aten op can't be 'default'
            use_key = "" if key == "default" else key
            # TODO: disallow access to overloads registered by JIT
            op_dk_tags = torch._C._get_operation_overload(
                self._qualified_op_name, use_key
            )
            if op_dk_tags is None:
                raise AttributeError(
                    f"The underlying op of '{str(self)}' has no overload name '{key}'"
                )

            op_, op_dk_, tags = op_dk_tags
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload: OpOverload[_P, _T] = (
                OpOverload(self, op_, op_dk_, schema, tags)
                if not _has_script_object_arg(schema)
                else TorchBindOpOverload(self, op_, op_dk_, schema, tags)
            )
            # cache the overload object
            setattr(self, key, overload)
            self._dir.append(key)
            return overload
        except RuntimeError:
            raise AttributeError(
                f"The underlying op of '{str(self)}' has no overload name '{key}'"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # overloading __call__ to ensure torch.ops.foo.bar()
        # is still callable from JIT
        # We save the function ptr as the `op` attribute on
        # OpOverloadPacket to access it here.

        # Directly calling OverloadPacket goes into C++, which will check
        # the schema and cause an error for torchbind op when inputs consist of FakeScriptObject so we
        # intercept it here and call TorchBindOpverload instead.
        if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
            # pyrefly: ignore [bad-argument-type]
            return _call_overload_packet_from_python(self, *args, **kwargs)
        return self._op(*args, **kwargs)

    # TODO: use this to make a __dir__
    def overloads(self):
        return [n if n else "default" for n in self._overload_names]

    def op_overloads(self):
        return [getattr(self, n) for n in self.overloads()]


# Note - this mirrors the logic of the cpp_function defined in jit/python/init.cpp
# _jit_get_operations, which calls _get_operation_for_overload_or_packet.
def _call_overload_packet_from_python(
    op: OpOverloadPacket[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    # Reuse the torch function handling logic in cpp
    torch_function_called, ret = torch._C._maybe_call_torch_function_for_op_packet(
        op, *args, **kwargs
    )

    if torch_function_called:
        return ret

    # The following mirrors getOpWithStack.
    # In cpp, we do a schema matching for the arguments, and call ToIValue
    # to check whether the arguments are valid. But need to do similar things here
    # and check the schema whether the FakeScriptObject is the corresponding fake class
    # of the actual class used in schema.
    exceptions = {}
    found_op = None
    for overload_name in op.overloads():
        op_overload = getattr(op, overload_name)
        try:
            _ = torch._C._check_schema_allow_fake_script_object(
                op_overload._schema, *args, **kwargs
            )
            found_op = op_overload
            break
        except RuntimeError as e:
            exceptions[overload_name] = e

    if found_op:
        return found_op(*args, **kwargs)

    err_msg = (
        f"Fail to match any TorchBindOverload of {op} with following exceptions:\n"
    )
    for key, msg in exceptions.items():
        err_msg += f"Overload name {key}:\n {msg}\n"
    raise RuntimeError(err_msg)


# Resolution of torch.fn is different from torch.ops.aten.fn
# torch.fn uses the Python argparser, matches with the
# appropriate schema, and calls into the unboxed version of the method
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT.
# JIT creates a stack of all the overloads and then tries to match the
# correct one at runtime and always calls into the boxed version of the method
# Autograd codegen creates VariableType, TracerType,
# inplace or view type and python bindings.
# Aten codegen generates tensor methods for the tensor class.

# _OpNamespace is a subclass of ModuleType because the torch script
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# to work from script, we need to ensure ops and foo are modules


class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """

    __file__ = "torch.ops"

    def __init__(self, name: str) -> None:
        super().__init__("torch.ops." + name)
        self.name = name
        self._dir: list[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    def __getattr__(self, op_name: str) -> OpOverloadPacket:
        if op_name in ("__origin__", "__self__"):
            raise AttributeError(
                f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'"
            )

        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        namespace_name = self.name
        qualified_op_name = f"{namespace_name}::{op_name}"
        module_name = self.__module__ + "." + namespace_name

        try:
            op, overload_names = _get_packet(qualified_op_name, module_name)
            if op is None:
                raise AttributeError(
                    f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
                )
        except RuntimeError as e:
            # Turn this into AttributeError so getattr(obj, key, default)
            # works (this is called by TorchScript with __origin__)
            raise AttributeError(
                f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
            ) from e

        op.__module__ = module_name
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        setattr(self, op_name, opoverloadpacket)
        self._dir.append(op_name)
        return opoverloadpacket


def _get_packet(qualname, op_module):
    op, overload_names = torch._C._jit_get_operation(qualname)
    if op is not None:
        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        torch.jit._builtins._register_builtin(op, qualname)
        op.__module__ = op_module
    return op, overload_names


def _refresh_packet(packet):
    op, overload_names = _get_packet(packet._qualified_op_name, packet._op.__module__)
    if op is None:
        raise AssertionError(f"failed to get packet for {packet._qualified_op_name}")
    packet._op = op
    packet._overload_names = overload_names


class _HigherOrderNamespace(types.ModuleType):
    __file__ = "torch.ops"

    def __init__(self) -> None:
        super().__init__("torch.ops.higher_order")
        self._dir: list[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    def __getattr__(self, name: str) -> HigherOrderOperator:
        # Following _OpNamespace.__getattr__, we cache the op on this object.
        op = _higher_order_ops.get(name)
        if op is None:
            raise AttributeError(
                f"'_HigherOrderNamespace' 'torch.ops.higher_order' object has no attribute '{name}'"
            )
        setattr(self, name, op)
        self._dir.append(name)
        return op


class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super().__init__("torch.ops")
        self.loaded_libraries = set()
        self.higher_order = _HigherOrderNamespace()
        self._dir = []

    def __getattr__(self, name: str) -> _OpNamespace:
        # Here we are creating `torch.ops.my_namespace`
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    def import_module(self, module):
        """
        Imports a Python module that has torch.library registrations.

        Generally, to extend PyTorch with custom operators, a user will
        create a Python module whose import triggers registration of
        the custom operators via a torch.ops.load_library call or a call
        to one or more torch.library.* APIs.

        It is unexpected for Python modules to have side effects, so some
        linters and formatters will complain. Use this API to import Python
        modules that contain these torch.library side effects.

        Args:
            module (str): The name of the Python module to import

        """
        importlib.import_module(module)

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        path = _utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            try:
                ctypes.CDLL(path)
            except Exception as e:
                raise OSError(f"Could not load this library: {path}") from e
        self.loaded_libraries.add(path)


# The ops "namespace"
ops: _Ops = _Ops()
<|endoftext|># mypy: allow-untyped-defs

from __future__ import annotations

import collections
import copy
import functools
import io
import threading
import warnings
from typing import Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self

import torch
from torch._utils import _to, _type
from torch.types import _bool, _int, Storage


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


__all__ = ["TypedStorage", "UntypedStorage"]


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


_share_memory_lock = threading.Lock()
_share_memory_map: dict[int, threading.RLock] = {}

T = TypeVar("T", bound="_StorageBase | TypedStorage")


def _throws_on_data_ptr_access(storage: _StorageBase | TypedStorage) -> _bool:
    # Whether storage.data_ptr() would raise (e.g. FakeTensor,
    # FunctionalTensor, or a cudagraph-invalidated storage). Non-throwing
    # replacement for wrapping data_ptr() in a try/except RuntimeError.
    return torch._C._storage_throw_on_immutable_data_ptr(
        storage._cdata
    ) or torch._C._storage_throw_on_mutable_data_ptr(storage._cdata)


class _StorageBase:
    _cdata: Any
    is_sparse: _bool = False
    is_sparse_csr: _bool = False
    device: torch.device
    # Used when
    # (1) stashing FakeTensor device onto storage in torch.serialization.skip_data
    # (2) stashing device onto storage to propagate to FakeTensor when torch.load under FakeTensorMode
    _fake_device: torch.device | None = None
    # Used when loading with FakeTensorMode to give information about offset of storage in torch.saved-file
    _checkpoint_offset: int | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self) -> _int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    def copy_(self, source: T, non_blocking: _bool | None = None) -> T:
        raise NotImplementedError

    def new(self) -> _StorageBase | TypedStorage:
        raise NotImplementedError

    def nbytes(self) -> _int:
        raise NotImplementedError

    def size(self) -> _int:
        return self.nbytes()

    def type(
        self, dtype: str | None = None, non_blocking: _bool = False
    ) -> _StorageBase | TypedStorage:
        return _type(self, dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False) -> _StorageBase | TypedStorage:
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination GPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("cuda", device) if device else torch.device("cuda")
        return self.to(device=device2, non_blocking=non_blocking)

    def hpu(self, device=None, non_blocking=False) -> _StorageBase | TypedStorage:
        """Returns a copy of this object in HPU memory.

        If this object is already in HPU memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination HPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("hpu", device) if device else torch.device("hpu")
        return self.to(device=device2, non_blocking=non_blocking)

    def element_size(self) -> _int:
        raise NotImplementedError

    def get_device(self) -> _int:
        return self.device.index

    def data_ptr(self) -> _int:
        raise NotImplementedError

    def resizable(self) -> _bool:
        raise NotImplementedError

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    def _share_fd_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _new_using_filename_cpu(cls, size: _int) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_using_fd_cpu(cls, size: _int) -> Self:
        raise NotImplementedError

    @classmethod
    def from_buffer(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_shared_filename_cpu(
        cls,
        manager,
        obj,
        size,
        *,
        device=None,
        dtype=None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        return cls._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _release_ipc_counter_cuda(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def _shared_decref(self) -> _StorageBase | TypedStorage:
        raise NotImplementedError

    def _write_file(self, *args, **kwargs):
        raise NotImplementedError

    def resize_(self, size: _int):
        raise NotImplementedError

    def _weak_ref(self, *args, **kwargs) -> _StorageBase | TypedStorage:
        raise NotImplementedError

    def _set_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def _set_cdata(self, *args, **kwargs):
        raise NotImplementedError

    def _share_cuda_(self, *args, **kwargs):
        raise NotImplementedError

    def is_shared(self) -> _bool:
        raise NotImplementedError

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def _shared_incref(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_cuda(self):
        raise NotImplementedError

    @property
    def is_hpu(self):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, shared, nbytes) -> _StorageBase | TypedStorage:
        raise NotImplementedError

    @classmethod
    def _expired(cls, *args, **kwargs) -> _StorageBase | TypedStorage:
        raise NotImplementedError

    def _byteswap(self, *args, **kwargs):
        raise NotImplementedError

    def _get_filename(self, *args, **kwargs) -> str | None:
        raise NotImplementedError

    def __repr__(self):
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str

    def __iter__(self):
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault("torch", {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def __sizeof__(self):
        return super().__sizeof__() + self.size()

    def clone(self):
        """Return a copy of this storage."""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        """Return a list containing the elements of this storage."""
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        if self.device.type != "cpu":
            return torch.UntypedStorage(self.size()).copy_(self, False)
        return self

    def mps(self):
        """Return a MPS copy of this storage if it's not already on the MPS."""
        if self.device.type != "mps":
            return torch.UntypedStorage(self.size(), device="mps").copy_(self, False)
        return self

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def to(self, *, device: DeviceLikeType, non_blocking: _bool = False):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        return _to(self, device, non_blocking)

    def double(self):
        """Casts this storage to double type."""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
        return self._to(torch.float8_e4m3fnuz)

    def is_pinned(self, device: DeviceLikeType | None = None):
        r"""Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        return (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .is_pinned(device)
        )

    def pin_memory(self, device: DeviceLikeType | None = None):
        r"""Copy the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != "cpu":
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")

        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        from torch.multiprocessing import get_sharing_strategy

        if self.device.type not in ("cpu", "meta"):
            pass  # only CPU uses POSIX shared memory
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        """Create a new storage in shared memory with the same data type."""
        from torch.multiprocessing import get_sharing_strategy

        device = torch.device(device)
        if device.type != "cpu":
            return cls(size, device=device)
        elif get_sharing_strategy() == "file_system":
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        return self

    def byteswap(self, dtype):
        """Swap bytes in underlying data."""
        elem_size = torch._utils._element_size(dtype)
        # for complex types, don't swap first and second numbers
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)


def _share_memory_lock_protected(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        to_free = None
        to_wait = None
        with _share_memory_lock:
            key = self._cdata
            if key in _share_memory_map:
                to_wait = _share_memory_map[key]
            else:
                _share_memory_map[key] = threading.RLock()
                _share_memory_map[key].acquire()
                to_free = key

        # If we're already in the process of sharing the storage, wait
        # for it to be done.
        if to_wait is not None:
            with to_wait:
                pass

        try:
            return fn(self, *args, **kwargs)
        finally:
            # If we acquired the storage lock here and we're done working on it
            # we can now release it and free the entry.
            if to_free is not None:
                # Ensure that the cdata from the storage didn't change and only
                # the data_ptr did.
                if self._cdata != to_free:
                    raise AssertionError(
                        f"storage cdata changed unexpectedly: expected {to_free}, got {self._cdata}"
                    )
                with _share_memory_lock:
                    _share_memory_map[to_free].release()
                    del _share_memory_map[to_free]

    return wrapper


class UntypedStorage(torch._C.StorageBase, _StorageBase):
    def __getitem__(self, *args, **kwargs):
        if self.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def is_hpu(self):
        return self.device.type == "hpu"

    @property
    def filename(self) -> str | None:
        """Returns the file name associated with this storage.

        The file name will be a string if the storage is on CPU and was created via
        :meth:`~torch.from_file()` with ``shared`` as ``True``. This attribute is ``None`` otherwise.
        """
        return self._get_filename()

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), weights_only=False)


@functools.cache
def _new_dtypes():
    # These are dtypes serialized as UntypedStorage unlike those in
    # _dtype_to_storage_type_map
    return {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
        torch.bits8,
        torch.bits16,
        torch.bits1x8,
        torch.bits2x4,
        torch.bits4x2,
        torch.complex32,
        torch.bcomplex32,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }


@functools.cache
def _dtype_to_storage_type_map():
    # NOTE: We should no longer add dtypes to this map. This map
    # is only used for BC/FC with older PyTorch versions. Going forward,
    # new dtypes of TypedStorage should not translate to a legacy
    # <type>Storage class. Instead, new dtypes of TypedStorage should
    # be serialized as an UntypedStorage paired with a torch.dtype
    return {
        torch.double: "DoubleStorage",
        torch.float: "FloatStorage",
        torch.half: "HalfStorage",
        torch.long: "LongStorage",
        torch.int: "IntStorage",
        torch.int16: "ShortStorage",
        torch.int8: "CharStorage",
        torch.uint8: "ByteStorage",
        torch.bool: "BoolStorage",
        torch.bfloat16: "BFloat16Storage",
        torch.cdouble: "ComplexDoubleStorage",
        torch.cfloat: "ComplexFloatStorage",
        torch.qint8: "QInt8Storage",
        torch.qint32: "QInt32Storage",
        torch.quint8: "QUInt8Storage",
        torch.quint4x2: "QUInt4x2Storage",
        torch.quint2x4: "QUInt2x4Storage",
    }


@functools.cache
def _storage_type_to_dtype_map():
    dtype_map = {val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map


def _get_storage_from_sequence(sequence, dtype, device):
    if dtype in [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]:
        interpret_dtypes = {
            torch.quint8: torch.uint8,
            torch.quint4x2: torch.uint8,
            torch.quint2x4: torch.uint8,
            torch.qint32: torch.int32,
            torch.qint8: torch.int8,
        }
        tmp_tensor = torch.tensor(
            sequence, dtype=interpret_dtypes[dtype], device=device
        )

    else:
        tmp_tensor = torch.tensor(sequence, dtype=dtype, device=device)

    return tmp_tensor._typed_storage()._untyped_storage


def _isint(x):
    if HAS_NUMPY:
        return isinstance(x, (int, np.integer))  # pyrefly: ignore [missing-attribute]
    else:
        return isinstance(x, int)


_always_warn_typed_storage_removal = False


def _get_always_warn_typed_storage_removal():
    return _always_warn_typed_storage_removal


def _set_always_warn_typed_storage_removal(always_warn):
    global _always_warn_typed_storage_removal
    if not isinstance(always_warn, bool):
        raise AssertionError(
            f"always_warn must be bool, got {type(always_warn).__name__}"
        )
    _always_warn_typed_storage_removal = always_warn


def _warn_typed_storage_removal(stacklevel=2):
    global _always_warn_typed_storage_removal

    def is_first_time():
        if not hasattr(_warn_typed_storage_removal, "has_warned"):
            return True
        else:
            return not _warn_typed_storage_removal.__dict__["has_warned"]

    if _get_always_warn_typed_storage_removal() or is_first_time():
        message = (
            "TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class. This should only matter "
            "to you if you are using storages directly.  To access UntypedStorage "
            "directly, use tensor.untyped_storage() instead of tensor.storage()"
        )
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_typed_storage_removal.__dict__["has_warned"] = True


def _reset_warn_typed_storage_removal():
    _warn_typed_storage_removal.__dict__["has_warned"] = False


def _get_device_from_module(module: str):
    last_part = module.rsplit(".", 1)[-1]
    if last_part in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
        return last_part
    else:
        return "cpu"


class TypedStorage:
    is_sparse: _bool = False
    # Used when stashing FakeTensor device onto storage in torch.save(metadata_only=True)
    _fake_device: torch.device | None = None

    dtype: torch.dtype

    @property
    def _dtype(self):
        return self.dtype

    @property
    def filename(self) -> str | None:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
        or ``None`` if the storage was not created by memory mapping a file."""
        return self._untyped_storage.filename

    def fill_(self, value):
        _warn_typed_storage_removal()
        self._setitem(slice(0, self._size()), value)
        return self

    def __new__(
        cls,
        *args,
        wrap_storage=None,
        dtype=None,
        device=None,
        _internal=False,
    ):
        if not _internal:
            _warn_typed_storage_removal()

        if cls == torch.storage._LegacyStorage:
            raise RuntimeError(
                "Only child classes of _LegacyStorage can be instantiated"
            )

        if cls == TypedStorage:
            return super().__new__(cls)

        else:
            arg_error_msg = (
                f"{cls}.__new__ received an invalid combination "
                f"of arguments. Expected one of:\n"
                " * no arguments\n"
                " * (int size)\n"
                " * (Sequence data)\n"
                " * (*, UntypedStorage wrap_storage)"
            )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'device' cannot be specified"
                )

            if dtype is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'dtype' cannot be specified"
                )

            if wrap_storage is None:
                if len(args) > 1:
                    raise RuntimeError(
                        arg_error_msg + "\nToo many positional arguments"
                    )

                if (
                    len(args) == 1
                    and not _isint(args[0])
                    and not isinstance(args[0], collections.abc.Sequence)
                ):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

                return TypedStorage(
                    *args,
                    dtype=cls._dtype,
                    device=_get_device_from_module(cls.__module__),
                    _internal=True,
                )

            else:
                if len(args) != 0:
                    raise RuntimeError(
                        arg_error_msg
                        + "\nNo positional arguments should be given when using "
                        "'wrap_storage'"
                    )

                if not isinstance(wrap_storage, torch.UntypedStorage):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                    )

                cls_device = _get_device_from_module(cls.__module__)

                if wrap_storage.device.type != cls_device:
                    raise RuntimeError(
                        arg_error_msg
                        + f"\nDevice of 'wrap_storage' must be {cls_device}"
                        f", but got {wrap_storage.device.type}"
                    )

                return TypedStorage(
                    *args,
                    wrap_storage=wrap_storage,
                    dtype=cls.dtype,
                    _internal=True,
                )

    def __init__(
        self,
        *args,
        device=None,
        dtype=None,
        wrap_storage=None,
        _internal=False,
    ):
        if not _internal:
            _warn_typed_storage_removal()
        arg_error_msg = (
            "TypedStorage.__init__ received an invalid combination "
            "of arguments. Expected one of:\n"
            " * (*, torch.device device, torch.dtype dtype)\n"
            " * (int size, *, torch.device device, torch.dtype dtype)\n"
            " * (Sequence data, *, torch.device device, torch.dtype dtype)\n"
            " * (*, UntypedStorage wrap_storage, torch.dtype dtype)"
        )

        if wrap_storage is not None:
            if len(args) != 0:
                raise RuntimeError(
                    arg_error_msg
                    + "\nNo positional arguments should be given when using "
                    "'wrap_storage'"
                )

            if dtype is None:
                raise RuntimeError(
                    arg_error_msg + "\nArgument 'dtype' must be specified"
                )

            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}"
                )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg
                    + "\nArgument 'device' should not be specified when 'wrap_storage' is given"
                )

            self.dtype = dtype

            if not isinstance(wrap_storage, torch.UntypedStorage):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                )

            self._untyped_storage = wrap_storage

        else:
            self.dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)

            if self.dtype in [
                torch.quint8,
                torch.quint4x2,
                torch.quint2x4,
                torch.qint32,
                torch.qint8,
            ]:
                if device.type == "cuda":
                    raise RuntimeError(
                        "Cannot create CUDA storage with quantized dtype"
                    )

            if len(args) == 0:
                self._untyped_storage = torch.UntypedStorage(device=device)

            elif len(args) == 1:
                if _isint(args[0]):
                    self._untyped_storage = torch.UntypedStorage(
                        int(args[0]) * self._element_size(), device=device
                    )
                elif isinstance(args[0], collections.abc.Sequence):
                    self._untyped_storage = _get_storage_from_sequence(
                        args[0], self.dtype, device
                    )
                else:
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

            else:
                raise RuntimeError(arg_error_msg + "\nToo many positional arguments")

    @property
    def is_cuda(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "cuda"

    @property
    def is_hpu(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "hpu"

    def untyped(self):
        """Return the internal :class:`torch.UntypedStorage`."""
        _warn_typed_storage_removal()
        return self._untyped_storage

    def _new_wrapped_storage(self, untyped_storage) -> Self:
        if type(untyped_storage) is not torch.UntypedStorage:
            raise AssertionError(
                f"expected UntypedStorage, got {type(untyped_storage).__name__}"
            )

        if type(self) is TypedStorage:
            return cast(
                Self,
                TypedStorage(
                    wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
                ),
            )
        else:
            return type(self)(wrap_storage=untyped_storage)

    def __len__(self):
        _warn_typed_storage_removal()
        return self._size()

    def _maybe_wrap_index(self, idx, is_stop=False):
        if idx is None:
            if is_stop:
                return self._size()
            else:
                return 0

        else:
            if type(idx) is not int:
                raise TypeError(f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if (idx > self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                if idx > 0:
                    return idx
                else:
                    return idx % self._size()
            else:
                if (idx >= self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                return idx % self._size()

    def __setitem__(self, idx, value):
        _warn_typed_storage_removal()
        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if torch.is_storage(value):
            raise RuntimeError(f"cannot set item with value type {type(value)}")
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor(
                [], dtype=tmp_dtype, device=self._untyped_storage.device
            )
            tmp_tensor.set_(
                TypedStorage(
                    wrap_storage=self._untyped_storage, dtype=tmp_dtype, _internal=True
                )
            )
        else:
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)

        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        _warn_typed_storage_removal()
        return self._getitem(idx)

    def _getitem(self, idx):
        if self._untyped_storage.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")

        # NOTE: Before TypedStorage existed, indexing with a slice used to be
        # possible for <type>Storage objects. However, it would return
        # a storage view, which would be a hassle to implement in TypedStorage,
        # so it was disabled
        if isinstance(idx, slice):
            raise RuntimeError(
                "slices are only supported in UntypedStorage.__getitem__"
            )
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")

        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            return TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=interpret_dtypes[self.dtype],
                _internal=True,
            )._getitem(idx)

        idx_wrapped = self._maybe_wrap_index(idx)
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)
            return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking: bool | None = None):
        _warn_typed_storage_removal()
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)
        else:
            self._untyped_storage.copy_(source, non_blocking)
        return self

    def nbytes(self):
        _warn_typed_storage_removal()
        return self._nbytes()

    # For internal use only, to avoid deprecation warning
    def _nbytes(self):
        return self._untyped_storage.nbytes()

    def type(
        self,
        dtype: str | None = None,
        non_blocking: bool = False,
    ) -> _StorageBase | TypedStorage | str:
        _warn_typed_storage_removal()
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()

            if legacy_class is not None:
                return legacy_class.__module__ + "." + legacy_class.__name__

            return ".".join([self.__module__, type(self).__name__])

        else:
            return self._untyped_storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False) -> Self:
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create CUDA storage with quantized dtype")
        cuda_storage = self._untyped_storage.cuda(device, non_blocking)
        return self._new_wrapped_storage(cuda_storage)

    def hpu(self, device=None, non_blocking=False) -> Self:
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create HPU storage with quantized dtype")
        hpu_storage = self._untyped_storage.hpu(device, non_blocking)
        return self._new_wrapped_storage(hpu_storage)

    def to(self, *, device: DeviceLikeType, non_blocking: bool = False) -> Self:
        _warn_typed_storage_removal()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError(
                f"Cannot create {device.type.upper()} storage with quantized dtype"
            )
        to_storage = self._untyped_storage.to(device=device, non_blocking=non_blocking)
        return self._new_wrapped_storage(to_storage)

    def element_size(self):
        _warn_typed_storage_removal()
        return self._element_size()

    # For internal use only, to avoid deprecation warning
    def _element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> _int:
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    def __str__(self):
        _warn_typed_storage_removal()
        info_str = (
            f"[{torch.typename(self)}(dtype={self.dtype}, "
            f"device={self.device}) of size {len(self)}]"
        )
        if self.device.type == "meta":
            return "...\n" + info_str
        else:
            data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
            return data_str + "\n" + info_str

    def __repr__(self):
        _warn_typed_storage_removal()
        return str(self)

    def __iter__(self):
        _warn_typed_storage_removal()
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(copy.copy(self._untyped_storage))

    def __deepcopy__(self, memo):
        _warn_typed_storage_removal()
        return self._deepcopy(memo)

    # For internal use only, to avoid deprecation warning
    def _deepcopy(self, memo):
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))

    def __sizeof__(self):
        _warn_typed_storage_removal()
        return super().__sizeof__() + self.nbytes()

    def clone(self):
        """Return a copy of this storage."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        """Return a list containing the elements of this storage."""
        _warn_typed_storage_removal()
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: DeviceLikeType | None = None):
        r"""Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        _warn_typed_storage_removal()
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: DeviceLikeType | None = None):
        r"""Copy the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(
            self._untyped_storage.pin_memory(device=device)
        )

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        _warn_typed_storage_removal()
        return self._share_memory_()

    # For internal use only, to avoid deprecation warning
    def _share_memory_(self):
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        """Create a new storage in shared memory with the same data type."""
        if device is None:
            device = "cpu"
        device = torch.device(device)
        untyped_storage = torch.UntypedStorage._new_shared(
            size * self._element_size(), device=device
        )
        return TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    @property
    def _cdata(self):
        return self._untyped_storage._cdata

    @property
    def device(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device

    def size(self):
        _warn_typed_storage_removal()
        return self._size()

    # For internal use only, to avoid deprecation warning
    def _size(self):
        # NB: don't indirect through __len__, as that requires
        # an int to be returned
        return self._untyped_storage.nbytes() // self._element_size()

    def pickle_storage_type(self):
        _warn_typed_storage_removal()
        return self._pickle_storage_type()

    # For internal use only, to avoid deprecation warning
    def _pickle_storage_type(self):
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError as e:
            raise KeyError(f"dtype {self.dtype} is not recognized") from e

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        _warn_typed_storage_removal()
        return self._data_ptr()

    # For internal use only, to avoid deprecation warning
    def _data_ptr(self):
        return self._untyped_storage.data_ptr()

    def resizable(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.resizable()

    def resize_(self, size):
        _warn_typed_storage_removal()
        self._resize_(size)

    # For internal use only, to avoid deprecation warning
    def _resize_(self, size):
        self._untyped_storage.resize_(size * self._element_size())

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        return UntypedStorage._free_weak_ref(*args, **kwargs)

    def _weak_ref(self, *args, **kwargs):
        return self._untyped_storage._weak_ref(*args, **kwargs)

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        return cls._from_buffer(*args, **kwargs)

    @classmethod
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
        if cls == TypedStorage:
            dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)
            if device.type != "cpu":
                raise RuntimeError(
                    f"TypedStorage.from_buffer: Not available for device {device.type}"
                )
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        else:
            if dtype is not None or len(args) == 5:
                raise RuntimeError(
                    "from_buffer: 'dtype' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )
            if device is not None:
                raise RuntimeError(
                    "from_buffer: 'device' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )

            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=self.dtype, device=self.device)
            .set_(self)
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type."""
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        _warn_typed_storage_removal()
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        _warn_typed_storage_removal()
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        _warn_typed_storage_removal()
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        _warn_typed_storage_removal()
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        _warn_typed_storage_removal()
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fnuz)

    @classmethod
    def from_file(cls, filename, shared, size):
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
        _warn_typed_storage_removal()
        if cls == TypedStorage:
            raise RuntimeError("from_file can only be called on derived classes")
        untyped_storage = UntypedStorage.from_file(
            filename, shared, size * torch._utils._element_size(cls.dtype)
        )
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        return UntypedStorage._expired(*args, **kwargs)

    def _write_file(self, *args, **kwargs):
        return self._untyped_storage._write_file(*args, **kwargs)

    def _set_from_file(self, *args, **kwargs):
        return self._untyped_storage._set_from_file(*args, **kwargs)

    def _set_cdata(self, *args, **kwargs):
        return self._untyped_storage._set_cdata(*args, **kwargs)

    def _share_cuda_(self, *args, **kwargs):
        return self._untyped_storage._share_cuda_(*args, **kwargs)

    def is_shared(self):
        _warn_typed_storage_removal()
        return self._is_shared()

    # For internal use only, to avoid deprecation warning
    def _is_shared(self):
        return self._untyped_storage.is_shared()

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)

    def _share_filename_cpu_(self, *args, **kwargs):
        (
            manager_handle,
            storage_handle,
            size,
        ) = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
        return manager_handle, storage_handle, size // self._element_size()

    def _shared_decref(self):
        self._untyped_storage._shared_decref()
        return self

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    def _shared_incref(self, *args, **kwargs):
        return self._untyped_storage._shared_incref(*args, **kwargs)

    def _share_fd_cpu_(self, *args, **kwargs):
        fd, size = self._untyped_storage._share_fd_cpu_(*args, **kwargs)
        return fd, size // self._element_size()

    def _get_legacy_storage_class(self):
        if self.dtype not in _dtype_to_storage_type_map():
            return None

        storage_name = _dtype_to_storage_type_map()[self.dtype]

        if self.device.type not in [
            "cpu",
            "cuda",
            "hpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            return None

        module = (
            torch if self.device.type == "cpu" else getattr(torch, self.device.type)
        )

        try:
            return getattr(module, storage_name)
        except AttributeError:
            return None


TypedStorage.type.__doc__ = _type.__doc__
TypedStorage.cuda.__doc__ = _StorageBase.cuda.__doc__
TypedStorage.hpu.__doc__ = _StorageBase.hpu.__doc__
TypedStorage.to.__doc__ = _to.__doc__


class _LegacyStorageMeta(type):
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        if type(instance) is TypedStorage:
            cls_device = _get_device_from_module(cls.__module__)
            return (cls_device == instance.device.type) and (
                cls.dtype == instance.dtype
            )
        return False


class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    @classmethod
    def _new_shared(cls, size):  # type: ignore[override]
        """Create a new storage in shared memory with the same data type."""
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(
            wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(
                manager, obj, bytes_size
            )
        )


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError as e:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized'
        ) from e
<|endoftext|>"""Various linear algebra utility methods for internal use."""

import torch
from torch import Tensor


def is_sparse(A: Tensor) -> bool:
    """Check if tensor A is a sparse COO tensor.

    All other sparse storage formats (CSR, CSC, etc...) will return False.
    """
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def get_floating_dtype(A: Tensor) -> torch.dtype:
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32


def matmul(A: Tensor | None, B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def bform(X: Tensor, A: Tensor | None, Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(X.mT, matmul(A, Y))


def qform(A: Tensor | None, S: Tensor) -> Tensor:
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A: Tensor) -> Tensor:
    """Return orthogonal basis of A columns."""
    return torch.linalg.qr(A).Q


def symeig(A: Tensor, largest: bool | None = False) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = torch.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z


# These functions were deprecated and removed
# This nice error message can be removed in version 1.13+
def cholesky(input: Tensor, upper: bool = False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.cholesky` is deprecated in favor of `torch.linalg.cholesky`.\n"
        "L = torch.cholesky(A) should be replaced with:\n"
        "L = torch.linalg.cholesky(A)\n"
        "U = torch.cholesky(A, upper=True) should be replaced with:\n"
        "U = torch.linalg.cholesky(A).mH"
    )


def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `torch.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in `torch.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.solve` is deprecated in favor of `torch.linalg.solve`. "
        "`torch.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\n"
        "To get the LU factorization see `torch.lu`, which can be used with `torch.lu_solve` or `torch.lu_unpack`.\n"
        "X = torch.solve(B, A).solution "
        "should be replaced with:\n"
        "X = torch.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.lstsq` is deprecated in favor of `torch.linalg.lstsq`.\n"
        "`torch.linalg.lstsq` has reversed arguments and does not return the QR decomposition in "
        "the returned tuple (although it returns other information about the problem).\n\n"
        "To get the QR decomposition consider using `torch.linalg.qr`.\n\n"
        "The returned solution in `torch.lstsq` stored the residuals of the solution in the "
        "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = torch.linalg.lstsq(A, B).solution"
    )


def _symeig(
    input,
    eigenvectors=False,
    upper=True,
    *,
    out=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "The default behavior has changed from using the upper triangular portion of the matrix by default "
        "to using the lower triangular portion.\n\n"
        "L, _ = torch.symeig(A, upper=upper) "
        "should be replaced with:\n"
        "L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\n"
        "and\n\n"
        "L, V = torch.symeig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')"
    )


def eig(
    self: Tensor,
    eigenvectors: bool = False,
    *,
    e=None,
    v=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors "
        "mimicking complex tensors.\n\n"
        "L, _ = torch.eig(A) "
        "should be replaced with:\n"
        "L_complex = torch.linalg.eigvals(A)\n\n"
        "and\n\n"
        "L, V = torch.eig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L_complex, V_complex = torch.linalg.eig(A)"
    )


def qr(
    input: Tensor,
    some: bool = True,
    *,
    out=None,
) -> tuple[Tensor, Tensor]:
    reduced = "reduced" if some else "complete"
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.qr` is deprecated in favor of `torch.linalg.qr` and the boolean parameter "
        "`some` has been replaced with a string parameter `mode`.\n\n"
        "Q, R = torch.qr(A) "
        "should be replaced with:\n"
        "Q, R = torch.linalg.qr(A, mode='reduced')\n\n"
        "and\n\n"
        "Q, R = torch.qr(A, some=False) "
        "should be replaced with:\n"
        "Q, R = torch.linalg.qr(A, mode='complete')\n\n"
        f"Since `some` was set to {some}, use mode='{reduced}'."
    )
<|endoftext|>from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import logging
import re
import sys
import threading
import traceback
import unittest.mock
import weakref
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, overload, TYPE_CHECKING, TypeVar
from typing_extensions import dataclass_transform

import torch
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._traceback import CapturedTraceback, format_frame
from torch.utils.weak import WeakTensorKeyDictionary


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    import dis
    from collections.abc import Callable, Generator, Iterator
    from types import CodeType

    import sympy

    from torch._dynamo.backends.distributed import DDPOptimizerContext
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.guards import GuardCheckSpec
    from torch._dynamo.output_graph import CodeOptions
    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta
    from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
    from torch._subclasses.fake_tensor import FakeTensorMode


"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""

COMPILE_ID_PATTERN = re.compile(r"^(?P<frame_id>\d+)/(?P<frame_compile_id>\d+)$")
CA_COMPILE_ID_PATTERN = re.compile(
    r"^!(?P<compiled_autograd_id>\d+)(?:/(?P<frame_id>\d+)/(?P<frame_compile_id>\d+))?$"
)

# [Note: Updating CompiledId]
#
# CompiledId represents a unique program-level identifier, and we want to keep that
# property as the codebase evolves. This property is relied on even outside of the pytorch
# repo, e.g. tlparse or other internal tooling. The in-memory format can be freely changed,
# as those dependencies only consume the string serialization.
#
# The string form should be:
# 1. Program-level uid: CompileId can uniquely identify a compiled graph.
# 2. Storage efficient: This object is logged in nearly every entry. We should elide symbols when possible.
# 3. Compact: The string form is directly displayed by some tools. Special symbols are okay.


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileId:
    frame_id: int | None
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int | None

    # torch.compiling a compiled autograd graph
    compiled_autograd_id: int | None = None

    # TODO: consider also tracking the recompilation count
    # See Note: Updating CompileId

    def __str__(self) -> str:
        # NOTE: Keep this in sync with both from_string and the tlparse repo
        if self.compiled_autograd_id is not None:
            if (self.frame_id is None) != (self.frame_compile_id is None):
                raise AssertionError(
                    f"frame_id and frame_compile_id must both be None or both be set, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            frame_str = ""
            if self.frame_id is not None:
                frame_str = f"/{self.frame_id}/{self.frame_compile_id}"

            return f"!{self.compiled_autograd_id}{frame_str}"
        else:
            if self.frame_id is None or self.frame_compile_id is None:
                raise AssertionError(
                    f"frame_id and frame_compile_id must not be None, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            return f"{self.frame_id}/{self.frame_compile_id}"

    @classmethod
    def from_string(cls, compile_id: str | None) -> CompileId | None:
        """
        Factory method that creates a CompileId from its string representation.
        Keep this in sync with the __str__ method.
        """
        if compile_id is None:
            return None
        try:
            for pattern in (COMPILE_ID_PATTERN, CA_COMPILE_ID_PATTERN):
                if match := pattern.match(compile_id):
                    groups = match.groupdict()
                    for k, v in groups.items():
                        if v is not None:
                            groups[k] = int(v)
                    return cls(**groups)  # type: ignore[arg-type]
            else:
                raise ValueError

        except Exception as e:
            raise ValueError(f"Invalid compile_id '{compile_id}'") from e


class TraceId(NamedTuple):
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self) -> str:
        # Keep this in sync with tlparse repo
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_SPECIALIZED_NN_MODULE = 2
    GLOBAL_SPECIALIZED_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8
    BACKWARD_STATE = 9
    EPHEMERAL = 10
    SYNTHETIC_LOCAL = 11
    LOCAL_UNSPECIALIZED_NN_MODULE = 12
    GLOBAL_UNSPECIALIZED_NN_MODULE = 13
    LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 14
    GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 15
    TEMP_LOCAL = 16

    def is_fsdp_module(self) -> bool:
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    def is_specialized_nn_module(self) -> bool:
        import torch._dynamo.config as config

        if config._unsafe_skip_fsdp_module_guards:
            return (
                self
                in (
                    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
                    GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
                )
                or self.is_fsdp_module()
            )
        return self in (
            GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
        )

    def is_unspecialized_nn_module(self) -> bool:
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_unspecialized_builtin_nn_module(self) -> bool:
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_local(self) -> bool:
        return self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )


"""
Base class for a "GuardBuilder" role.

The GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little
confusing, as it's not a builder, but for the sake of avoiding a lot of renames and keeping the original reference
to torchdynamo's GuardBuilder.

Note: create_fn is invoked with a GuardBuilderBase and a Guard. A GuardBuilder is chosen based
on GuardSource's select function.

There is value in keeping this GuardBuilderBase empty to keep layering clean.
"""


class GuardBuilderBase:
    pass


@dataclasses.dataclass(frozen=True)
class SLoc:
    framework_loc: traceback.FrameSummary | str | None
    maybe_user_loc: str | None

    def __str__(self) -> str:
        floc = (
            self.framework_loc
            if isinstance(self.framework_loc, str)
            else ""
            if self.framework_loc is None
            else format_frame(self.framework_loc)
        )
        if self.maybe_user_loc is not None:
            return f"{self.maybe_user_loc} ({floc})"
        else:
            return f"({floc})"


class ShapeGuard(NamedTuple):
    expr: sympy.logic.boolalg.Boolean
    sloc: SLoc
    size_oblivious: bool


@dataclasses.dataclass(slots=True)
class Guard:
    # originating_source is the source that called the make_guard method to
    # construct this guard object. The property name specifies what exactly it
    # is the guard is guarding on.  The meaning of the name is dependent on the
    # create_fn; you must look at the use-site inside create_fn to know what
    # name means.
    #
    # That being said, although you might think this is just a "name", name is
    # usually an arbitrary Python expression that will be evaluated with all
    # globals (and locals, if you create a LOCAL guard) to extract the Python
    # object that we want to perform guard tests on.  This evaluation
    # typically happens in GuardBuilder.eval.  In these cases, name is
    # typically produced by originating_source.name (not to be confused with
    # GuardSource - the property source).
    #
    # Occasionally, name is not a valid Python expression; sometimes
    # it is meaningless.  Example create_fns that are like this include
    # GRAD_MODE and SHAPE_ENV.
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: list[str] | None = None
    code_list: list[str] | None = None
    obj_weakref: object | None = None
    guarded_class_weakref: weakref.ReferenceType[Any] | None = None

    stack: CapturedTraceback | None = None
    user_stack: traceback.StackSummary | None = None
    _hash: int | None = None
    _unserializable: bool = False
    _force_dict_keys_match: bool = False

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self) -> tuple[bool, int, int, str, int]:
        # Put the duplicate input guards at the end. The duplicate guards have
        # two sources while guard.name only considers one source.

        is_duplicate_input = (
            isinstance(self.create_fn, functools.partial)
            and self.create_fn.func is torch._dynamo.guards.GuardBuilder.DUPLICATE_INPUT
        )
        return (
            is_duplicate_input,
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.inner_create_fn().__code__.co_firstlineno,
        )

    def __lt__(self, other: Guard) -> bool:
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self) -> Callable[[GuardBuilderBase, Guard], Any]:
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    def name(self) -> str:
        return self.originating_source.name

    @property
    def source(self) -> GuardSource:
        return self.originating_source.guard_source

    @staticmethod
    def weakref_to_str(obj_weakref: object) -> str:
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f"<weakref at {hex(id(obj_weakref))}; dead>"
        else:
            return str(obj_weakref)

    def __repr__(self) -> str:
        s = f"""
        {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.inner_create_fn().__name__}
        {{
            'guard_types': {self.guard_types},
            'code': {self.code_list},
            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}
            'guarded_class': {self.guarded_class_weakref}
        }}
        """
        return s

    def __str__(self) -> str:
        output = f"Name: {repr(self.name)}\n"
        source = self.source.name.lower() if self.source else ""
        output += f"    Source: {source}\n"
        output += f"    Create Function: {self.inner_create_fn().__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, builder: GuardBuilderBase) -> Any:
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.exception("Error while creating guard:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise

    def is_specialized_nn_module(self) -> bool:
        return self.source.is_specialized_nn_module()

    def is_fsdp_module(self) -> bool:
        return self.source.is_fsdp_module()

    def is_local(self) -> bool:
        return self.source.is_local()

    def create_fn_name(self) -> str:
        if isinstance(self.create_fn, functools.partial):
            create_fn = self.create_fn.func  # type: ignore[attr-defined]
        else:
            create_fn = self.create_fn
        return create_fn.__name__

    def set_export_info(
        self,
        guard_type: str,
        guarded_class: weakref.ReferenceType[Any] | None,
        code_list: list[str],
        obj_weakref: object,
    ) -> None:
        if not self.guard_types:
            self.guard_types = []

        self.guard_types.append(guard_type)

        if self.guarded_class_weakref not in (guarded_class, None):
            raise AssertionError(
                f"Guarded class id must be identical, or None, "
                f"got {self.guarded_class_weakref} vs {guarded_class}"
            )
        self.guarded_class_weakref = guarded_class

        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        # Some objects are ephemeral, e.g., list[slice(1, 2)]. If we have
        # multiple guards on the same object, the weakref can die between the
        # invocation of set_export_info calls. So a dead weakref is also
        # acceptable.
        is_valid = (
            self.obj_weakref in (obj_weakref, None)
            or callable(self.obj_weakref)
            and self.obj_weakref() is None
        )
        if not is_valid:
            raise AssertionError(
                f"Guarded object must be identical, None or ephemeral (dead weakref), "
                f"got {self.obj_weakref} vs {obj_weakref}"
            )
        self.obj_weakref = obj_weakref


T = TypeVar("T")

"""
Parent structure for guard env expressions.
A GuardEnvExpr can have any subtype.
Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""


@dataclasses.dataclass(frozen=True)
class GuardEnvExpr:
    pass


"""
A class representing a pair of duplicate inputs.
input_pos_a and input_pos_b are input positions we have deduped.
"""


@dataclasses.dataclass(frozen=True)
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self) -> None:
        if self.input_source_a == self.input_source_b:
            raise AssertionError(
                f"input_source_a and input_source_b must be different, "
                f"got {self.input_source_a}"
            )


"""
A class representing storage overlap relations among inputs that aliases the same storage.

Given that a set of tensors alias the same storage, this guard checks whether they actually
have overlapping storages.

While non_overlapping_sources represent input tensors that definitely don't have any storage
overlapping with any other input, overlapping_sources represent tensors that either:

1. Do overlap some other input tensor
2. Might not overlap some other input tensor, but we are not sure
"""


@dataclasses.dataclass(frozen=True)
class StorageOverlap(GuardEnvExpr):
    overlapping_sources: list[Source]
    non_overlapping_sources: list[Source]


"""
Checkpointable is an interface for driving state snapshotting, left purposely vague for now.

copy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that
can also be taken in at restore_graphstate(T) calls.

When to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable
does not provide any guarantees around consistency, idempotency, or safety of calling its APIs, yet.

In the future, it will have a closer coupling to a generic Checkpoint management system.
"""


class Checkpointable(Generic[T]):
    @abstractmethod
    def copy_graphstate(self) -> T: ...

    @abstractmethod
    def restore_graphstate(self, state: T) -> None: ...


class GuardsCheckpointState:
    """
    The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
    """

    dynamo_guards: OrderedSet[Guard]

    def __init__(self, dynamo_guards: OrderedSet[Guard]) -> None:
        self.dynamo_guards = dynamo_guards

    def diff(self, other: GuardsCheckpointState) -> OrderedSet[Guard] | None:
        """
        Produces a delta against another GuardsCheckpointState.

        Returns None if no delta is found, otherwise, return an OrderedSet() of mismatched
        Guard type objects.
        """
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GuardsCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContextCheckpointState:
    nn_modules: dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules: dict[str, torch.nn.Module]) -> None:
        self.nn_modules = nn_modules

    def diff(self, other: ModuleContextCheckpointState) -> set[str] | None:
        """
        Produces a delta against another ModuleContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        module key names.
        """
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuleContextCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    def __init__(self) -> None:
        self.nn_modules: dict[str, Any] = {}

    def copy_graphstate(self) -> ModuleContextCheckpointState:
        return ModuleContextCheckpointState(dict(self.nn_modules))

    def restore_graphstate(self, state: ModuleContextCheckpointState) -> None:
        if not isinstance(state, ModuleContextCheckpointState):
            raise AssertionError(
                f"expected ModuleContextCheckpointState, got {type(state)}"
            )
        self.nn_modules = state.nn_modules


class GlobalContextCheckpointState:
    global_state: dict[str, tuple[Callable, Any]] = {}

    def __init__(self, global_states: dict[str, tuple[Callable, Any]]) -> None:
        self.global_state = global_states

    def diff(self, other: GlobalContextCheckpointState) -> set[str] | None:
        """
        Produces a delta against another GlobalContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        global key names.
        """
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalContextCheckpointState):
            return False
        return self.diff(other) is None


class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    This keeps track of the global torch state during tracing of a function.
    For example, torch.is_grad_enabled.
    """

    _supported_global_states = {
        "grad_enabled",
        "autocast_enabled",
        "autocast_cpu_enabled",
        "autocast_gpu_dtype",
        "autocast_cpu_dtype",
        "autocast_cache_enabled",
    }

    def __init__(self) -> None:
        self.global_state: dict[str, tuple[Callable, Any]] = {}

    def copy_graphstate(self) -> GlobalContextCheckpointState:
        return GlobalContextCheckpointState(self.global_state)

    def restore_graphstate(self, state: GlobalContextCheckpointState) -> None:
        if not isinstance(state, GlobalContextCheckpointState):
            raise AssertionError(
                f"expected GlobalContextCheckpointState, got {type(state)}"
            )
        self.global_state = state.global_state
        if not (
            len(self.global_state) == len(self._supported_global_states)
            and set(self.global_state.keys()) == self._supported_global_states
        ):
            raise AssertionError(
                f"Global state mismatch: got keys {set(self.global_state.keys())}, "
                f"expected {self._supported_global_states}"
            )
        for func, args in self.global_state.values():
            func(args)


# Like a Set[Guard] but will record the user stack on all guards at the
# time they were installed at their destination
class GuardsSet:
    def __init__(self, inner: OrderedSet[Guard] | None = None) -> None:
        if inner is None:
            self.inner: OrderedSet[Guard] = OrderedSet()
        else:
            self.inner = inner
        # Map from source to list of guards with that source. Incrementally
        # built in `add` method.
        self.source_to_guards: defaultdict[Source, list[Guard]] = defaultdict(list)
        for guard in self.inner:
            self.track_guard_by_source(guard)

    def track_guard_by_source(self, guard: Guard) -> None:
        source = guard.originating_source
        self.source_to_guards[source].append(guard)

    def __iter__(self) -> Iterator[Guard]:
        return iter(self.inner)

    def __len__(self) -> int:
        return len(self.inner)

    # Subtraction along with bool is typically used to determine the delta of
    # added guards between checkpoints for higher order ops
    def __sub__(self, other: GuardsSet) -> GuardsSet:
        return GuardsSet(self.inner - other.inner)

    def __bool__(self) -> bool:
        return bool(self.inner)

    def clear(self) -> None:
        self.inner = OrderedSet()
        self.source_to_guards = defaultdict(list)

    def add(
        self, guard: Guard, *, collect_debug_stack: bool = True, skip: int = 0
    ) -> None:
        if guard in self.inner:
            return
        if collect_debug_stack:
            if guard.stack is None:
                guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = TracingContext.extract_stack()
        self.inner.add(guard)
        self.track_guard_by_source(guard)

    def update(self, *others: set[Guard]) -> None:
        for o in others:
            for g in o:
                self.add(g, skip=1)

    def get_guards_for_source(self, source: Source) -> list[Guard]:
        """Return all guards with the given originating_source."""
        return list(self.source_to_guards[source])

    def remove_guards_with_source(self, source: Source) -> None:
        """Delete all guards that contains a given source"""
        from ._dynamo.source import is_from_source

        self.inner = OrderedSet(
            g for g in self.inner if not is_from_source(g.originating_source, source)
        )
        # Rebuild the index since is_from_source walks the chain, so
        # multiple source keys may need removal.
        self.source_to_guards = defaultdict(list)
        for guard in self.inner:
            self.track_guard_by_source(guard)


"""
A GuardsContext is a checkpointable representation of all the guards in the current tracing
context. Its lifecycle is bound 1:1 to the tracing context, and it should never be instantiated
directly outside of it. For passing around internal state representations of this object,
prefer to extract them with copy_graphstate to produce a GuardsCheckpointState.
"""


class GuardsContext(Checkpointable[GuardsCheckpointState]):
    def __init__(self) -> None:
        self.dynamo_guards: GuardsSet = GuardsSet()
        self.aotautograd_guards: list[GuardEnvExpr] = []
        self.skip_install: bool = False

    @contextlib.contextmanager
    def skip_guard_install(self) -> Generator[None, None, None]:
        old = self.skip_install
        self.skip_install = True
        try:
            yield
        finally:
            self.skip_install = old

    def copy_graphstate(self) -> GuardsCheckpointState:
        return GuardsCheckpointState(OrderedSet(self.dynamo_guards.inner))

    def restore_graphstate(self, state: GuardsCheckpointState) -> None:
        # NB: "steals" the passed in state
        if not isinstance(state, GuardsCheckpointState):
            raise AssertionError(f"expected GuardsCheckpointState, got {type(state)}")
        self.dynamo_guards = GuardsSet(state.dynamo_guards)


class HopSubgraphCache:
    @abstractmethod
    def add_dynamo_installed_submodule(
        self, fn_code: CodeType, identifier: str
    ) -> None: ...

    @abstractmethod
    def get_dynamo_installed_submodules(self, fn_code: CodeType) -> list[str]: ...

    @abstractmethod
    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_autograd_key_entry(self, identifier: str) -> Callable | None: ...

    @abstractmethod
    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_proxy_dispatch_entry(self, identifier: str) -> Callable | None: ...

    @abstractmethod
    def add_functionalize_schema_entry(
        self, key: object, schema: torch._C.FunctionSchema
    ) -> None: ...

    @abstractmethod
    def get_functionalize_schema_entry(
        self, key: object
    ) -> torch._C.FunctionSchema | None: ...

    @abstractmethod
    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int: ...

    @abstractmethod
    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[torch.fx.GraphModule | None, int | None]: ...


@dataclass
class InvokeSubgraphReuseEntry:
    body_name: str
    body_gmod: torch.fx.GraphModule
    config: NestedCompileRegionOptions | None
    subgraph_input_mapping: list[
        Any
    ]  # list[LiftedArgOrigin] - defined in invoke_subgraph.py
    single_tensor_output: bool
    # Per-output tensor metadata (shape, stride, dtype, device, requires_grad)
    # cached from the first trace so we can construct fresh FakeTensors on
    # cache hit without re-running body_gmod.
    output_metadata: list[
        tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device, bool]
    ]
    # 1-1 mapping to flat_vts: source for each flattened arg/kwarg, or None if
    # the VT has no source. On cache hit, we build a source replacement mapping
    # (old arg sources → new arg sources) to rewrite captured variable sources
    # for the current invocation.
    arg_sources: list[Source | None]
    # Number of user-visible outputs (from the function return value).
    # The graph may have additional outputs from side-effect intermediates;
    # stamp_out_subgraph uses this to return only the user-visible slice.
    num_user_outputs: int = 0


@dataclass
class InvokeSubgraphReuseCondition:
    # Per flattened input VT: (InputTag, metadata).
    #   (InputTag.TENSOR, TensorMetadata)
    #   (InputTag.SYMNODE, sym_num — same object implies same symbol)
    #   (InputTag.CONSTANT, value)
    #   (InputTag.OBJECT, None)
    # Tensor metadata is checked here because TENSOR_MATCH guards for
    # subgraph inputs may already exist before tracing and thus won't
    # appear in the guard delta.
    input_checks: list[tuple[Any, object]]  # list[tuple[InputTag, object]]

    # Guards captured during the trace (delta + source-mapped).
    # Each entry: (source, handler, expected_value, guard)
    # handler is a pre-resolved GuardCheckSpec from GUARD_VALUE_DISPATCH.
    guards: list[tuple[Source, GuardCheckSpec, object, Guard]]

    # TreeSpec from pytree.tree_flatten of the (args, kwargs) structure.
    # On cache hit, we verify the new call has the same treespec.
    treespec: pytree.TreeSpec | None = None

    # All sources accessed via VariableBuilder during the subgraph trace.
    # On cache hit, we check if any modified VT's source is a base of one
    # of these to detect mutations on captured variables.
    traced_sources: OrderedSet[Source] = dataclasses.field(default_factory=OrderedSet)


class InvokeSubgraphCache(HopSubgraphCache):
    def __init__(self) -> None:
        self.autograd_cache: dict[str, Callable] = {}
        self.proxy_dispatch_cache: dict[str, Callable] = {}
        self.functionalize_schema_cache: dict[object, torch._C.FunctionSchema] = {}
        self.dynamo_installed_submodules: dict[CodeType, list[str]] = defaultdict(list)
        self.lazy_bwd_cache: dict[
            str, dict[tuple[object], tuple[torch.fx.GraphModule, int]]
        ] = defaultdict(dict)
        self.effects_cache: dict[
            str, set
        ] = {}  # Maps identifier -> set of effect types
        # fn.__code__ → list of (condition, cache_entry) pairs. Walked linearly
        # on lookup; first matching condition wins.
        self.subgraph_reuse_cache: dict[
            CodeType,
            list[tuple[InvokeSubgraphReuseCondition, InvokeSubgraphReuseEntry]],
        ] = defaultdict(list)
        # fn_code → {hash_key → cache_entry}. Used by user-provided
        # reuse_hash_fn for O(1) subgraph reuse lookup.
        self.subgraph_reuse_key_cache: dict[
            CodeType, dict[int, InvokeSubgraphReuseEntry]
        ] = defaultdict(dict)

    def add_dynamo_installed_submodule(
        self, fn_code: CodeType, identifier: str
    ) -> None:
        self.dynamo_installed_submodules[fn_code].append(identifier)

    def get_dynamo_installed_submodules(self, fn_code: CodeType) -> list[str]:
        return self.dynamo_installed_submodules.get(fn_code, [])

    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None:
        self.autograd_cache[identifier] = key

    def get_autograd_key_entry(self, identifier: str) -> Callable | None:
        return self.autograd_cache.get(identifier, None)

    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None:
        self.proxy_dispatch_cache[identifier] = key

    def get_proxy_dispatch_entry(self, identifier: str) -> Callable | None:
        return self.proxy_dispatch_cache.get(identifier, None)

    def add_functionalize_schema_entry(
        self, key: object, schema: torch._C.FunctionSchema
    ) -> None:
        self.functionalize_schema_cache[key] = schema

    def get_functionalize_schema_entry(
        self, key: object
    ) -> torch._C.FunctionSchema | None:
        return self.functionalize_schema_cache.get(key, None)

    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int:
        # Save the number of existing graph modules in the dictionary to get the suffix
        num_gmods = len(self.lazy_bwd_cache[identifier])
        self.lazy_bwd_cache[identifier][tangent_metadata] = (gmod, num_gmods)
        return num_gmods

    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[torch.fx.GraphModule | None, int | None]:
        if identifier not in self.lazy_bwd_cache:
            return (None, None)

        return self.lazy_bwd_cache[identifier].get(tangent_metadata, (None, None))

    def add_effects(self, identifier: str, effects: set) -> None:
        """Store the effect types for a given invoke_subgraph identifier."""
        if prev_effects := self.effects_cache.get(identifier, None):
            if effects != prev_effects:
                raise AssertionError(
                    "Different number of effects were found for invoke_subgraph "
                    f"call with identifier {identifier}. \n"
                    f"Previously we had the following effects: {prev_effects}.\n"
                    f"But now we have: {effects}."
                )
        self.effects_cache[identifier] = effects

    def get_effects(self, identifier: str) -> set | None:
        """Retrieve the effect types for a given invoke_subgraph identifier."""
        return self.effects_cache.get(identifier, None)

    def add_reuse_entry(
        self,
        fn_code: CodeType,
        condition: InvokeSubgraphReuseCondition,
        entry: InvokeSubgraphReuseEntry,
        max_reuse_entries: int = 8,
    ) -> None:
        entries = self.subgraph_reuse_cache[fn_code]
        if len(entries) >= max_reuse_entries:
            raise RuntimeError(
                f"invoke_subgraph: exceeded maximum reuse entries "
                f"({max_reuse_entries}) for function code {fn_code}. "
                f"This most likely means a guard keeps failing on every "
                f"invocation, preventing subgraph reuse. "
                f"Set TORCH_LOGS='+hierarchical_compile' to identify which "
                f"guard is failing. If reuse is genuinely not possible and "
                f"you need more cache entries, increase the limit via the "
                f"max_reuse_entries argument to nested_compile_region()."
            )
        entries.append((condition, entry))

    def find_reuse_entry(
        self,
        fn_code: CodeType,
        evaluator: Callable[
            [InvokeSubgraphReuseCondition, InvokeSubgraphReuseEntry], bool
        ],
    ) -> InvokeSubgraphReuseEntry | None:
        entries = self.subgraph_reuse_cache.get(fn_code, [])
        for i, (condition, entry) in enumerate(entries):
            if evaluator(condition, entry):
                # MRU: move the hit entry to the front for faster future lookups
                if i > 0:
                    entries.insert(0, entries.pop(i))
                return entry
        return None

    def find_reuse_entry_by_key(
        self, fn_code: CodeType, hash_key: int
    ) -> InvokeSubgraphReuseEntry | None:
        return self.subgraph_reuse_key_cache.get(fn_code, {}).get(hash_key)

    def add_reuse_entry_by_key(
        self,
        fn_code: CodeType,
        hash_key: int,
        entry: InvokeSubgraphReuseEntry,
        max_reuse_entries: int = 8,
    ) -> None:
        key_cache = self.subgraph_reuse_key_cache[fn_code]
        if len(key_cache) >= max_reuse_entries and hash_key not in key_cache:
            raise RuntimeError(
                f"invoke_subgraph: exceeded maximum reuse entries "
                f"({max_reuse_entries}) for function code {fn_code} (hash-key path). "
                f"Increase the limit via the max_reuse_entries argument to "
                f"nested_compile_region()."
            )
        key_cache[hash_key] = entry


class HopDispatchSetCache:
    def __init__(self) -> None:
        # Delayed import to avoid circular dependency
        from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

        self.hop_cache_map = {invoke_subgraph: InvokeSubgraphCache()}

    def get_cache(self, op: torch._ops.HigherOrderOperator) -> HopSubgraphCache | None:
        if op not in self.hop_cache_map:
            return None
        return self.hop_cache_map[op]  # type: ignore[index]


class _TLSStorage(threading.local):
    # Default the hot-path attributes to None per thread so that
    # TracingContext.try_get() / CompileContext.try_get() -- called on every
    # torch.compile'd call -- hit a present attribute instead of paying for an
    # AttributeError raise+catch inside getattr(). Without this, a thread that
    # never ran compilation itself (e.g. a worker thread executing compiled code
    # compiled on another thread) takes the slow getattr-miss path on every call.
    def __init__(self) -> None:
        self.tracing_context: TracingContext | None = None
        self.compile_context: CompileContext | None = None


_TLS = _TLSStorage()

"""
TracingContext is the source of truth for all currently accumulated information
needed to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems
are open to managing their own TracingContext with that in mind.

The purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid
having to plumb complex subsystems across multiple verticals.

Ex: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.
Accessing the current tracing context via
TracingContext.get() allows users to accumulate their own guards for processing, without needing to know how
to plumb objects back up to where frame interpretation happened.

Note that you can end up with multiple TracingContext for a single compilation
of a frame, as we reset the TracingContext whenever we restart analysis.
CompileContext is a more overarching context that encompasses multiple restarts.
"""


class CompileContext:
    @staticmethod
    def get() -> CompileContext:
        if _TLS.compile_context is None:
            raise AssertionError("compile_context is not set")
        return _TLS.compile_context

    @staticmethod
    def try_get() -> CompileContext | None:
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id: CompileId | None) -> None:
        if compile_id is not None and not isinstance(compile_id, CompileId):
            raise AssertionError(
                f"compile_id must be None or CompileId, got {type(compile_id)}"
            )
        self.compile_id: CompileId | None = compile_id
        self.attempt = 0
        # Verbose ShapeEnv guards produced.
        self.shape_env_guards: list[str] = []

    @staticmethod
    def current_compile_id() -> CompileId | None:
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id() -> TraceId | None:
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)


@dataclass
class InlinedCodeCache:
    """Cache for code-object-derived data used during inlining."""

    instructions: list[Any]
    indexof: dict[Any, int]
    code_options: CodeOptions


class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """

    @staticmethod
    def try_get() -> TracingContext | None:
        return getattr(_TLS, "tracing_context", None)

    @staticmethod
    def get() -> TracingContext:
        if ctx := TracingContext.try_get():
            return ctx
        raise RuntimeError(
            "TracingContext.get() must be called within an ongoing trace."
        )

    def __init__(self, fake_mode: FakeTensorMode | None) -> None:
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.previously_inlined_functions: dict[Any, Any] = dict()
        self.previously_cleaned_instructions: dict[Any, Any] = dict()
        # Combined cache for inlined code data (instructions, indexof, code_options)
        self.inlined_code_cache: dict[Any, InlinedCodeCache] = dict()
        self.fake_mode: FakeTensorMode | None = fake_mode
        self.frame_summary_stack: list[traceback.FrameSummary] = []
        # This is morally part of frame_summary_stack, but it is kept separate
        # for clarity.  As we process a frame, this variable gets updated
        # to keep track of what line we are in the function.  We make a
        # function call, this gets cleared and the frame location is pushed
        # to frame_summary_stack (prepping this variable for the inner frame's
        # progress)
        self.loc_in_frame: tuple[str, int, str] | None = None
        self.loc_in_frame_positions: dis.Positions | None = None
        # this is only set after aot_autograd
        self.fw_metadata: ViewAndMutationMeta | None = None
        # this is only set when the DDPOptimizer is used
        self.ddp_optimizer_ctx: DDPOptimizerContext | None = None
        # this is only set after aot_autograd
        self.aot_graph_name: list[str] | None = None
        self.params_flat: list[Any] | None = None
        self.params_flat_unwrap_subclasses: list[Any] | None = None
        self.params_unwrapped_to_flat_index: list[Any] | None = None
        # this is for extended return calling convention from backend
        # compiler to aot_autograd
        # Per output, what the compiler specified stride of the output is,
        # or None if no stride is known.  This is always the HINT, it
        # is never a SymInt (it would be better if it was a SymInt, but
        # I can't conveniently get this from Inductor atm.  Also, be
        # careful not to accidentally induce guards on the SymInt if
        # you ever do change this in aot_autograd.py; you should check
        # on permutations preferentially.)
        self.output_strides: list[tuple[int, ...] | None] | None = None
        # When this is True, whenever we encounter an int in Dynamo tracing,
        # we will (1) force unspec it and (2) force it as a size-like unbacked
        # integer.  This is currently used when processing certain lists of
        # ints that are known to be size-like and may have 0/1 entries that we
        # must not specialize on.
        self.force_unspec_int_unbacked_size_like = False
        # See note [Tensor Fakification and Symbol Caching]
        self.tensor_to_context = WeakTensorKeyDictionary()

        # If this true, Aot Autograd will return output Fake Tensors with appropriate
        # meta on the first invocation
        # see note: [Returning Fake Tensors on First AOT Autograd Call]
        self.fakify_first_call = False
        self.hop_dispatch_set_cache = HopDispatchSetCache()
        # list of code objects for inlined functions
        self.traced_code: list[CodeType] = []
        self.cudagraph_annotation: Any = None

    def clear(self) -> None:
        # Look at the note in output_graph.py in function `save_global_state`
        # for the context on clearing global context.
        self.global_context.global_state = {}
        self.previously_inlined_functions.clear()
        self.previously_cleaned_instructions.clear()
        self.inlined_code_cache.clear()

    @staticmethod
    @contextmanager
    def patch(**kwargs: Any) -> Generator[None, None, None]:
        prior = {}
        ctx = TracingContext.get()

        for key in kwargs:
            # KeyError on invalid entry
            prior[key] = getattr(ctx, key)
        for key, val in kwargs.items():
            setattr(ctx, key, val)
        try:
            yield
        finally:
            for key, val in prior.items():
                setattr(ctx, key, val)

    @staticmethod
    def extract_stack() -> traceback.StackSummary:
        self = TracingContext.try_get()
        if self is None:
            return traceback.StackSummary()
        stack = self.frame_summary_stack
        if self.loc_in_frame is not None:
            stack = stack + [self._populate_loc_in_frame_summary()]
        return traceback.StackSummary.from_list(stack)

    def _populate_loc_in_frame_summary(self) -> traceback.FrameSummary:
        if self.loc_in_frame is None:
            raise AssertionError("loc_in_frame must not be None")
        filename, lineno, frame_name = self.loc_in_frame
        # colno/end_colno kwargs were added to FrameSummary in 3.11
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 11) and self.loc_in_frame_positions is not None:
            kwargs["colno"] = self.loc_in_frame_positions.col_offset
            kwargs["end_colno"] = self.loc_in_frame_positions.end_col_offset
        return traceback.FrameSummary(
            filename,
            lineno,
            frame_name,
            lookup_line=False,
            **kwargs,
        )

    # Call this when you want to call into some code that isn't necessarily
    # associated with the current frame state
    @staticmethod
    @contextlib.contextmanager
    def clear_frame() -> Generator[None, None, None]:
        tc = TracingContext.get()
        with (
            unittest.mock.patch.object(tc, "frame_summary_stack", []),
            unittest.mock.patch.object(tc, "loc_in_frame", None),
            unittest.mock.patch.object(tc, "loc_in_frame_positions", None),
        ):
            try:
                yield
            except Exception as e:
                # Prevent real_stack from getting attached
                #
                # The invariant is that if an Exception as real_stack, we've
                # appropriately attached a user stack and we no longer need to
                # attach anything. Because we cannot conveniently interpose
                # when an exception is thrown, we instead interpose everywhere
                # we set what the user stack is set (using the context
                # manager). However, our compiler stack does "tail calls"
                # (when it calls into user compiler), at which point the
                # parent exception frames would incorrectly attach an
                # incorrect frame.
                #
                # However, if, somehow, someone raised an exception with this
                # scope that had a stack (for example, because they are
                # restoring the user stack state appropriately as they process
                # node by node), we should respect it. Thus, we cannot
                # unconditionally set None.
                if not hasattr(e, "real_stack"):
                    e.real_stack = None  # type: ignore[attr-defined]
                raise

    @staticmethod
    @contextlib.contextmanager
    def current_frame(
        frame_summary: traceback.FrameSummary | None,
    ) -> Generator[None, None, None]:
        # frame_summary can be None to solely take advantage of real_stack
        # attachment to thrown exceptions
        tc = TracingContext.get()
        if frame_summary is not None:
            tc.frame_summary_stack.append(frame_summary)
        old = tc.loc_in_frame
        old_positions = tc.loc_in_frame_positions
        tc.loc_in_frame = None
        tc.loc_in_frame_positions = None
        try:
            yield
        except Exception as e:
            if not hasattr(e, "real_stack"):
                e.real_stack = tc.extract_stack()  # type: ignore[attr-defined]
            raise
        finally:
            if frame_summary is not None:
                tc.frame_summary_stack.pop()
            tc.loc_in_frame = old
            tc.loc_in_frame_positions = old_positions

    @staticmethod
    @contextlib.contextmanager
    def report_output_strides() -> Generator[
        list[tuple[int, ...] | None] | None, None, None
    ]:
        tc = TracingContext.try_get()
        if tc is None:
            yield None
            return
        old_output_strides = tc.output_strides
        tc.output_strides = []
        try:
            yield tc.output_strides
        finally:
            tc.output_strides = old_output_strides

    @staticmethod
    def set_current_loc(
        filename: str,
        lineno: int,
        frame_name: str,
        positions: dis.Positions | None = None,
    ) -> None:
        # Save the current location in the frame. Lazily generate the
        # framesummary.
        tc = TracingContext.get()
        tc.loc_in_frame = (filename, lineno, frame_name)
        tc.loc_in_frame_positions = positions

    @staticmethod
    def get_traced_code() -> list[CodeType] | None:
        tc = TracingContext.try_get()
        if tc is None:
            return None
        return tc.traced_code


@contextmanager
def compile_context(
    context: CompileContext | None,
) -> Generator[CompileContext | None, None, None]:
    old_context = getattr(_TLS, "compile_context", None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context


@contextmanager
def tracing(
    context: TracingContext | None,
) -> Generator[TracingContext | None, None, None]:
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """
    old_context = getattr(_TLS, "tracing_context", None)
    _TLS.tracing_context = context
    try:
        yield context
    except Exception as e:
        if not hasattr(e, "real_stack") and context is not None:
            e.real_stack = context.extract_stack()  # type: ignore[attr-defined]
        raise
    finally:
        if (
            context is not None
            and context.fake_mode is not None
            and context.fake_mode.shape_env is not None
        ):
            context.fake_mode.shape_env.cleanup()
        _TLS.tracing_context = old_context


@overload
def dataclass_with_cached_hash(cls: type[T], **kwargs: Any) -> type[T]: ...


@overload
def dataclass_with_cached_hash(
    cls: None = None, **kwargs: Any
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform()
def dataclass_with_cached_hash(
    cls: type[T] | None = None, **kwargs: Any
) -> type[T] | Callable[[type[T]], type[T]]:
    def wrap(cls_inner: type[T]) -> type[T]:
        new_cls = dataclasses.dataclass(cls_inner, **kwargs)
        old_hash = cls_inner.__hash__

        def __hash__(self) -> int:
            if not hasattr(self, "_hash"):
                object.__setattr__(self, "_hash", old_hash(self))
            return self._hash

        def __reduce__(self):
            # Exclude _hash from pickling to ensure deterministic cache keys.
            # The _hash is a cached value that can be nondeterministically computed
            # (e.g., based on id() of objects), so it should not affect pickling.
            fields = dataclasses.fields(self)
            field_values = tuple(getattr(self, f.name) for f in fields if f.init)
            return (self.__class__, field_values)

        new_cls.__hash__ = __hash__
        new_cls.__reduce__ = __reduce__
        return new_cls  # type: ignore[return-value]

    if cls is None:
        return wrap

    return wrap(cls)


# Subclasses can be found in torch/_dynamo/source.py
# TODO(voz): Consider a toplevel torch/_source.py
@dataclass_with_cached_hash(frozen=True)
class Source:
    def is_dict_key(self) -> bool:
        return False

    def is_ephemeral(self) -> bool:
        return False

    def reconstruct(self, codegen: PyCodegen) -> None:
        raise NotImplementedError

    def reconstruct_pycode(self, codegen: PyCodegen) -> str:
        """
        Reconstructs the source into a string of Python code. This method should
        be eventually implemented for all subclasses of Source to achieve full
        coverage of python wrapper code generation.
        """
        raise NotImplementedError

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        raise NotImplementedError

    @property
    def _name_template(self) -> str:
        """
        A template for the name of the source. Used to prevent code duplication between
        `name` and `get_value`.

        For non-ChainedSources, `name` and `get_value` use the returned string directly.

        For ChainedSources, `name` and `get_value` expect the return to be a format string
        with `{0}` present - `name` and `get_value` will apply different values to this function's
        returned format string.
        """
        raise NotImplementedError

    @functools.cached_property
    def name(self) -> str:
        return self._name_template

    def get_value(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        cache: dict[Source, Any],
    ) -> Any:
        if self in cache:
            return cache[self]
        value = eval(self._name_template, globals, locals)
        cache[self] = value
        return value

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        if self.guard_source is GuardSource.CONSTANT:
            raise NotImplementedError
        return Guard(self, fn)

    def is_specialized_nn_module(self) -> bool:
        return self.guard_source.is_specialized_nn_module()

    def subguards_allowed(self) -> bool:
        """True if you can guard on attributes of this"""
        return self.guard_source != GuardSource.SYNTHETIC_LOCAL

    def clone(self, transform_fn: Callable[[Source], Source] | None = None) -> Source:
        # Frozen dataclass, so returning self is effectively a clone.
