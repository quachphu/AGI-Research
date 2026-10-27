                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        old_is_fx_tracing_flag = _get_is_fx_tracing()
        _set_is_fx_tracing(True)
        try:
            if isinstance(root, torch.nn.Module):
                # do real recompilation for _LazyGraphModule before retracing since the trace
                # method can not trace the _lazy_forward method. Got error:
                #   https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
                # without this.
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(
                    root  # pyrefly: ignore[bad-argument-type]
                )

                self.root = root

                if not hasattr(type(root), self.traced_func_name):
                    raise AssertionError(
                        f"traced_func_name={self.traced_func_name} doesn't exist in "
                        f"{type(root).__name__}"
                    )

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: type[Tracer] | None = getattr(self, "__class__", None)
            self.graph = Graph(tracer_cls=tracer_cls)
            if hasattr(fn, "__code__"):
                code = fn.__code__
                self.graph._co_fields = {
                    "co_name": code.co_name,
                    "co_filename": code.co_filename,
                    "co_firstlineno": code.co_firstlineno,
                }

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: dict[
                _ConstantAttributeType,
                str,
            ] = {}

            def collect_tensor_attrs(
                m: torch.nn.Module, prefix_atoms: list[str]
            ) -> None:
                for k, v in m.__dict__.items():
                    if isinstance(v, _constant_attribute_types):
                        self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            if not isinstance(fn, FunctionType):
                raise AssertionError(f"Expected FunctionType, got {type(fn)}")

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args
            )

            parameter_proxy_cache: dict[
                str, Proxy
            ] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod: torch.nn.Module, attr: str) -> Any:
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(
                mod: torch.nn.Module, *args: Any, **kwargs: Any
            ) -> Any:
                def forward(*args: Any, **kwargs: Any) -> Any:
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,  # type: ignore[has-type]
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                return self.call_module(mod, forward, args, kwargs)

            with _new_patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module,
                    "__call__",
                    module_call_wrapper,
                    deduplicate=False,
                )
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(
                        patcher, module.__dict__, self._autowrap_function_ids
                    )
                ann = inspect.get_annotations(inspect.unwrap(fn))
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(fn(*args)),),
                    {},
                    type_expr=ann.get("return", None),
                )

            self.submodule_paths = None
        except RuntimeError as e:
            if e.args and isinstance(e.args[0], str) and "data-dependent" in e.args[0]:
                partial_fx_graph = self.graph.python_code(
                    root_module="self",
                    verbose=True,
                ).src
                e.partial_fx_graph = partial_fx_graph  # type: ignore[attr-defined]
                raise

            raise
        finally:
            _set_is_fx_tracing(old_is_fx_tracing_flag)
        return self.graph

    def __deepcopy__(self, memo: dict[int, Any]) -> "Tracer":
        # _autowrap_search contains modules, which cannot be deepcopied.
        new_tracer = Tracer.__new__(Tracer)

        for k, v in self.__dict__.items():
            if k == "_autowrap_search":
                new_obj = copy.copy(v)
            else:
                new_obj = copy.deepcopy(v, memo)

            new_tracer.__dict__[k] = new_obj

        return new_tracer

    def _proxy_placeholder(
        self,
        name: str,
        concrete_args: dict[str, Any] | None,
        sig: inspect.Signature,
        fn_for_analysis: Callable[..., Any],
    ) -> Any:
        if concrete_args is not None and name in concrete_args:
            cnt = 0

            def replace_ph(x: object) -> object:
                nonlocal cnt
                cnt += 1
                param = sig.parameters[name]
                default: tuple[Any, ...] = (
                    () if param.default is inspect.Parameter.empty else (param.default,)
                )
                out = self.create_proxy(
                    "placeholder", f"{name}_{str(cnt)}", default, {}
                )
                if isinstance(x, PHBase):
                    if x != PH:
                        # Transfer attrs in the case where you're using a placeholder other
                        # than the singleton PH (PH has no attributes to transfer).
                        # Proxies were created out of the placeholders.
                        # Transfer any metadata (put on the placeholders in the form of
                        # attributes set by the user) from the placeholder to the
                        # underlying nodes (the proxy is unwrapped by the user, but
                        # the metadata should hold).
                        _transfer_attrs(fr=x, to=out.node)

                    return out
                # Union[int, bool] == bool in Python <= 3.6
                if (
                    type(x) is bool
                    or type(x) in base_types
                    and type(x) is not torch.Tensor
                ):
                    torch._assert(
                        out == x,
                        f"{name} has been specialized to have value {x} but got another value",
                    )
                elif x is None:
                    args = (
                        out,
                        f"{name} has been specialized to have value None but got another value",
                    )
                    self.create_proxy("call_function", _assert_is_none, args, {})
                else:
                    warnings.warn(
                        f"Was not able to add assertion to guarantee correct input {name} to "
                        f"specialized function. It is up to the user to make sure that your inputs match the "
                        f"inputs you specialized the function with."
                    )

                return x

            return pytree.tree_map(replace_ph, concrete_args[name])
        if name[0] == "*":
            default: tuple[Any, ...] = ()
        else:
            param = sig.parameters[name]
            default = (  # type: ignore[assignment]
                () if param.default is inspect.Parameter.empty else (param.default,)
            )
        return self.create_proxy(
            "placeholder",
            name,
            default,
            {},
            type_expr=fn_for_analysis.__annotations__.get(name, None),
        )


# Dictionary of (id(globals dict), function name) => globals_dict to patch for
# the purposes of the wrap() API.
# We key by the globals dict id and function name to ensure we're wrapping a given
# function only once.
_wrapped_fns_to_patch: dict[tuple[int, str], dict[str, Any]] = {}

# List of methods on classes to wrap (class type, function name)
# this currently only works for Tensor.* methods that aren't traced properly
_wrapped_methods_to_patch: list[tuple[type, str]] = []

if os.environ.get("FX_PATCH_GETITEM") == "1":
    # This change is needed to trace models like PositionalEmbedding from BERT:
    # https://github.com/pytorch/benchmark/blob/master/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py
    # but causes issues in quantization documented here:
    # https://github.com/pytorch/pytorch/issues/50710
    # once that is fixed we can make this the default behavior.
    _wrapped_methods_to_patch.append((torch.Tensor, "__getitem__"))


def _find_proxy(*objects_to_search: object) -> Proxy | None:
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
    proxy = None

    def find_proxy(x: object) -> None:
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x

    # pyrefly: ignore[bad-specialization]
    map_aggregate(objects_to_search, find_proxy)
    return proxy


def _create_wrapped_func(orig_fn: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(orig_fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        """
        Given a closed-over ``orig_function`` to invoke, search the args and kwargs for
        a Proxy object. If there is one, emit a ``call_function`` node to preserve the
        call to this leaf function directly. Otherwise, just return the results of
        this function call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy(
                "call_function", orig_fn, args, kwargs
            )
            return_proxy.node.meta["is_wrapped"] = True
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped


def _create_wrapped_method(cls: type, name: str) -> Callable[..., Any]:
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """
        Search the args and kwargs for a Proxy object. If there is one,
        emit a ``call_method`` node to preserve the call to this method
        directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy("call_method", name, args, kwargs)
        return orig_fn(*args, **kwargs)

    return wrapped


class _PatchedFn(NamedTuple):
    frame_dict: Any  # dict[str, Any] for SetItem/Del, type for SetAttr
    fn_name: str
    orig_fn: Callable[..., Any] | None
    new_fn: Callable[..., Any]

    def revert(self) -> None:
        raise NotImplementedError

    def patch(self) -> None:
        raise NotImplementedError


class _PatchedFnSetItem(_PatchedFn):
    def revert(self) -> None:
        self.frame_dict[self.fn_name] = self.orig_fn

    def patch(self) -> None:
        self.frame_dict[self.fn_name] = self.new_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self) -> None:
        del self.frame_dict[self.fn_name]

    def patch(self) -> None:
        self.frame_dict[self.fn_name] = self.new_fn


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self) -> None:
        setattr(self.frame_dict, self.fn_name, self.orig_fn)

    def patch(self) -> None:
        setattr(self.frame_dict, self.fn_name, self.new_fn)


class _Patcher:
    def __init__(self) -> None:
        super().__init__()
        self.patches_made: list[_PatchedFn] = []
        self.visited: set[int] = set()

    def patch(
        self,
        frame_dict: dict[str, Any],
        name: str,
        new_fn: Callable[..., Any],
        deduplicate: bool = True,
    ) -> None:
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None, new_fn))
            self.patches_made[-1].patch()
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItem(frame_dict, name, frame_dict[name], new_fn)
            )
            self.patches_made[-1].patch()

    def patch_method(
        self, cls: type, name: str, new_fn: Callable[..., Any], deduplicate: bool = True
    ) -> None:
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn, new_fn))
        self.patches_made[-1].patch()

    def visit_once(self, thing: object) -> bool:
        """Return True on the first call to with thing, otherwise false"""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def revert_all_patches(self) -> list[_PatchedFn]:
        """
        Remove all the stored patcheds. It doesn't modify patches_made.
        """
        for patch in self.patches_made:
            patch.revert()
        return self.patches_made

    def reapply_all_patches(self) -> list[_PatchedFn]:
        """
        Patch all the stored patcheds. It doesn't modify patches_made.
        """
        for patch in self.patches_made:
            patch.patch()
        return self.patches_made

    def __enter__(self) -> "_Patcher":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()


CURRENT_PATCHER: _Patcher | None = None


@contextlib.contextmanager
def _new_patcher() -> Iterator[_Patcher]:
    global CURRENT_PATCHER
    prior_patcher = CURRENT_PATCHER
    try:
        CURRENT_PATCHER = _Patcher()
        yield CURRENT_PATCHER
    finally:
        # Clear all the patches made by when using current patcher.
        if CURRENT_PATCHER is None:
            raise AssertionError("CURRENT_PATCHER is None in finally block")
        CURRENT_PATCHER.revert_all_patches()
        CURRENT_PATCHER = prior_patcher


@contextlib.contextmanager
def _maybe_revert_all_patches() -> Iterator[None]:
    current_patcher = CURRENT_PATCHER
    patches_made = None
    patches_removed = None
    try:
        if current_patcher is not None:
            patches_removed = current_patcher.revert_all_patches()
        yield
    finally:
        if current_patcher is not None:
            patches_made = current_patcher.reapply_all_patches()
        if patches_made != patches_removed:
            raise AssertionError(
                "CURRENT_PATCHER was changed during a revert_all_patches"
            )


def _patch_wrapped_functions(patcher: _Patcher) -> None:
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for (_, name), frame_dict in _wrapped_fns_to_patch.copy().items():
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(
    patcher: _Patcher, frame_dict: dict[str, Any], function_ids: set[int]
) -> None:
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (
                not name.startswith("_")
                and callable(value)
                and id(value) in function_ids
            ):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


@overload
def wrap(fn_or_name: _F) -> _F: ...
@overload
def wrap(fn_or_name: str) -> str: ...
@compatibility(is_backward_compatible=True)
def wrap(fn_or_name: str | Callable[..., Any]) -> str | Callable[..., Any]:
    """
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y


        torch.fx.wrap("my_custom_function")


        def fn_to_be_traced(x, y):
            # When symbolic tracing, the below call to my_custom_function will be inserted into
            # the graph rather than tracing it.
            return my_custom_function(x, y)

    This function can also equivalently be used as a decorator::

        # foo/bar/baz.py
        @torch.fx.wrap
        def my_custom_function(x, y):
            return x * x + y * y

    A wrapped function can be thought of a "leaf function", analogous to the concept of
    "leaf modules", that is, they are functions that are left as calls in the FX trace
    rather than traced through.

    Args:

        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the
            graph when it's called
    """
    if not callable(fn_or_name) and not isinstance(fn_or_name, str):
        raise RuntimeError(
            "Unsupported type for global function! Must be either a callable or "
            "string name"
        )

    if callable(fn_or_name):
        if isinstance(fn_or_name, str):  # to make mypy happy
            raise AssertionError("Unexpected: fn_or_name is both callable and str")
        fn_name = fn_or_name.__name__
    else:
        if not isinstance(fn_or_name, str):
            raise AssertionError(
                f"fn_or_name must be a global function or string name, got "
                f"{type(fn_or_name)}"
            )
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    if currentframe is None:
        raise AssertionError("inspect.currentframe() returned None")
    f = currentframe.f_back
    if f is None:
        raise AssertionError("currentframe.f_back is None")
    if f.f_code.co_name != "<module>":
        raise NotImplementedError("wrap must be called at the top level of a module")

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    _wrapped_fns_to_patch[(id(f.f_globals), fn_name)] = f.f_globals
    return fn_or_name


@compatibility(is_backward_compatible=True)
def symbolic_trace(
    root: torch.nn.Module | Callable[..., Any],
    concrete_args: dict[str, Any] | None = None,
) -> GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a * 2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `concrete_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, concrete_args={"b": False})
        assert f(3, False) == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `concrete_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn't be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out


        f = fx.symbolic_trace(
            f, concrete_args={"x": {"a": fx.PH, "b": fx.PH, "c": fx.PH}}
        )
        assert f({"a": 1, "b": 2, "c": 4}) == 7


    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return _make_graph_module(tracer.root, graph, name)


@wrap
def _assert_is_none(value: object, msg: str) -> None:
    if value is not None:
        raise AssertionError(msg)
<|endoftext|>from typing import Any

from torch.fx.proxy import Proxy

from ._compatibility import compatibility


__all__ = ["annotate"]


@compatibility(is_backward_compatible=False)
def annotate(val: Any, type: type) -> Any:
    """
    Annotates a Proxy object with a given type.

    This function annotates a val with a given type if a type of the val is a torch.fx.Proxy object
    Args:
        val (object): An object to be annotated if its type is torch.fx.Proxy.
        type (object): A type to be assigned to a given proxy object as val.
    Returns:
        The given val.
    Raises:
        RuntimeError: If a val already has a type in its node.
    """
    if isinstance(val, Proxy):
        if val.node.type:
            raise RuntimeError(
                f"Tried to annotate a value that already had a type on it!"
                f" Existing type is {val.node.type} "
                f"and new type is {type}. "
                f"This could happen if you tried to annotate a function parameter "
                f"value (in which case you should use the type slot "
                f"on the function signature) or you called "
                f"annotate on the same value twice"
            )
        else:
            val.node.type = type
        return val
    else:
        return val
<|endoftext|>from __future__ import annotations

import base64
import contextlib
import contextvars
import copy
import hashlib
import importlib
import itertools
import linecache
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Generator, Iterator
    from typing import Self

    from ._symbolic_trace import Tracer
    from .experimental.symbolic_shapes import ShapeEnv
    from .node import Node

import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer

from ._compatibility import compatibility
from .experimental import _config as fx_experimental_config
from .graph import (
    _BoxedCodeGen,
    _custom_builtins,
    _is_from_torch,
    _override_sym_repr,
    _PyTreeCodeGen,
    Graph,
    PythonCode,
)


__all__ = [
    "reduce_graph_module",
    "reduce_package_graph_module",
    "GraphModule",
]

_USER_PRESERVED_ATTRIBUTES_KEY = "_user_preserved_attributes"
FX_GRAPH_MODULE_FILE_PREFIX = "fx_generated_"


# Normal exec loses the source code, however we can work with
# the linecache module to recover it.
# Using _exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
class _EvalCacheLoader:
    def __init__(self) -> None:
        self.eval_cache = {}
        self.next_id = 0

    def cache(
        self, src: str, globals: dict[str, Any], co_fields: dict[str, Any] | None = None
    ) -> str:
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """

        key = self._get_key()
        if co_fields:
            if "co_filename" in co_fields:
                # If only co_filename is provided, use it directly as the key
                if "co_firstlineno" not in co_fields or "co_name" not in co_fields:
                    key = co_fields["co_filename"]
                else:
                    # Full co_fields with all three components
                    key += f" from {co_fields['co_filename']}:{co_fields['co_firstlineno']} in {co_fields['co_name']}"
        self.eval_cache[key] = src

        # Don't mutate globals so that this loader is only used
        # to populate linecache, and doesn't interact with other modules
        # that might check `__loader__`
        globals_copy = globals.copy()
        globals_copy["__file__"] = key
        globals_copy["__name__"] = key
        globals_copy["__loader__"] = self
        globals_copy["__spec__"] = importlib.machinery.ModuleSpec(key, self)  # type: ignore[bad-argument-type]
        linecache.lazycache(key, globals_copy)

        return key

    # Part of the loader protocol (PEP 302)
    # linecache will use this method when trying to find source code
    def get_source(self, module_name: str) -> str | None:
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]
        return None

    def _get_key(self) -> str:
        key = f"<eval_with_key>.{self.next_id}"
        self.next_id += 1
        return key


_loader = _EvalCacheLoader()


_share_torchbind_and_process_group = contextvars.ContextVar(
    "_share_torchbind_and_process_group", default=False
)


@contextlib.contextmanager
def _share_torchbind_and_process_group_on_deepcopy() -> Iterator[None]:
    """Inside this context, ``GraphModule.__deepcopy__`` smuggles torchbind
    objects without ``__getstate__``/``__setstate__`` (e.g. ProcessGroup)
    through deepcopy as shared references instead of crashing.
    """
    token = _share_torchbind_and_process_group.set(True)
    try:
        yield
    finally:
        _share_torchbind_and_process_group.reset(token)


def _exec_with_source(
    src: str, globals: dict[str, Any], co_fields: dict[str, Any] | None = None
) -> None:
    key = _loader.cache(src, globals, co_fields)
    # dont_inherit=True prevents this module's `from __future__ import
    # annotations` from leaking into the generated code, which would turn
    # type annotations into strings and break downstream consumers like
    # TorchScript that expect real type objects.
    # TODO: Fix TorchScript BC to avoid breakages like these
    exec(compile(src, key, "exec", dont_inherit=True), globals)


def _forward_from_src(
    src: str, globals: dict[str, Any], co_fields: dict[str, Any] | None = None
) -> Callable[..., Any]:
    return _method_from_src(
        method_name="forward", src=src, globals=globals, co_fields=co_fields
    )


def _method_from_src(
    method_name: str,
    src: str,
    globals: dict[str, Any],
    co_fields: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    # avoid mutating the passed in dict
    globals_copy = globals.copy()
    _exec_with_source(src, globals_copy, co_fields)
    fn = globals_copy[method_name]
    del globals_copy[method_name]
    return fn


def _format_import_statement(name: str, obj: object, importer: Importer) -> str:
    if name in _custom_builtins:
        return _custom_builtins[name].import_str
    if _is_from_torch(name):
        return "import torch"
    module_name, attr_name = importer.get_name(obj)
    return f"from {module_name} import {attr_name} as {name}"


def _format_import_block(globals: dict[str, Any], importer: Importer) -> str:
    import_strs: set[str] = set()
    for name, obj in globals.items():
        if isinstance(obj, str):
            # Forward-reference string annotations are globals in generated code.
            import_strs.add(f"{name} = {obj!r}")
        else:
            import_strs.add(_format_import_statement(name, obj, importer))
    # Sort the imports so we have a stable import block that allows us to
    # hash the graph module and get a consistent key for use in a cache.
    return "\n".join(sorted(import_strs))


@compatibility(is_backward_compatible=True)
def reduce_graph_module(body: dict[str, Any], import_block: str) -> torch.nn.Module:
    # BC: attribute name was changed from `code` to `_code` to facilitate
    # making `code` into a property and adding a docstring to it
    fn_src = body.get("_code") or body["code"]
    forward = _forward_from_src(import_block + fn_src, {})
    return _deserialize_graph_module(forward, body)


@compatibility(is_backward_compatible=True)
def reduce_package_graph_module(
    importer: PackageImporter, body: dict[str, Any], generated_module_name: str
) -> torch.nn.Module:
    forward = importer.import_module(generated_module_name).forward
    return _deserialize_graph_module(forward, body)


# We create a dummy class here because symbolic_trace pulls the forward()
# function off of the class, rather than the instance. This class is used
# in _deserialize_graph_module() below.
class _CodeOnlyModule(torch.nn.Module):
    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__()
        self.__dict__ = body


def _deserialize_graph_module(
    forward: Callable[..., Any],
    body: dict[str, Any],
    graph_module_cls: type | None = None,
) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """

    # Try to retrieve the forward source in a backward-compatible way
    _CodeOnlyModule.forward = forward

    tracer_cls = body.get("_tracer_cls")
    if tracer_cls is None:
        from ._symbolic_trace import Tracer

        tracer_cls = Tracer

    graphmodule_cls_name = body.get("_graphmodule_cls_name", "GraphModule")

    # This is a workaround for a mypy linter issue related to
    # passing base class as an argument - https://github.com/python/mypy/issues/5865.
    cls_tracer: Any = tracer_cls

    class KeepModules(cls_tracer):
        # we shouldn't trace into any of the submodules,
        # because they were not traced in the original GraphModule
        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True

    com = _CodeOnlyModule(body)

    tracer_extras = body.get("_tracer_extras", {})
    graph = KeepModules().trace(com, **tracer_extras)

    # Recover node.meta["stack_trace"] after re-tracing
    node_meta_stack_trace = body.get("_graphmodule_graph_node_meta_stack_trace")
    if node_meta_stack_trace is not None:
        del body["_graphmodule_graph_node_meta_stack_trace"]
        for node in graph.nodes:
            if node_meta_stack_trace.get(node.name, None) is not None:
                node.meta["stack_trace"] = node_meta_stack_trace[node.name]

    # Manually set Tracer class on the reconstructed Graph, to avoid
    # referencing the private local subclass KeepModules.
    graph._tracer_cls = tracer_cls
    from ._lazy_graph_module import _make_graph_module

    gm = _make_graph_module(
        com, graph, class_name=graphmodule_cls_name, graph_module_cls=graph_module_cls
    )

    # The GraphModule constructor only retains attributes referenced by the graph.
    # In this case, our goal is return a GraphModule as close to identical as the one
    # put into the package. If any additional attributes were present in body,
    # we should keep them.
    for k, v in body.items():
        if not hasattr(gm, k):
            setattr(gm, k, v)
    return gm


# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(
    from_module: torch.nn.Module, to_module: torch.nn.Module, target: str
) -> None:
    *prefix, field = target.split(".")
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            # we have already installed one of its parents
            # (e.g. target = root.linear.weight, but we have already installed root.linear)
            # once we install a parent, we no longer need to copy the children
            # since all the needed properties will already be present
            return

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = f, t

    orig = getattr(from_module, field)
    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
        persistent = field not in from_module._non_persistent_buffers_set
        to_module.register_buffer(field, orig, persistent=persistent)
    else:
        setattr(to_module, field, orig)


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(from_obj: object, to_module: torch.nn.Module, target: str) -> None:
    *prefix, field = target.split(".")
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(from_obj, torch.Tensor) and not isinstance(
        from_obj, torch.nn.Parameter
    ):
        to_module.register_buffer(field, from_obj)
    else:
        setattr(to_module, field, from_obj)


# Recursively look up target from a graph module.
def _get_attr(model: torch.nn.Module, attr_name: str) -> Any:
    return _get_attr_via_attr_list(model, attr_name.split("."))


def _del_attr(model: torch.nn.Module, attr_name: str) -> None:
    attr_names = attr_name.split(".")
    t = _get_attr_via_attr_list(model, attr_names[:-1])
    return delattr(t, attr_names[-1])


def _get_attr_via_attr_list(model: torch.nn.Module, attr_list: list[str]) -> Any:
    if len(attr_list) == 0:
        return model
    *prefix, field = attr_list
    t = model
    for item in prefix:
        t = getattr(t, item, None)  # type: ignore[assignment]
        if t is None:
            raise AssertionError(f"Attribute '{item}' not found in model")

    return getattr(t, field)


def _has_attr(model: torch.nn.Module, attr_name: str) -> bool:
    *prefix, field = attr_name.split(".")
    t = model
    for item in prefix:
        t = hasattr(t, item)  # type: ignore[assignment]
        if t is False:
            return False

    return hasattr(t, field)


def _print_readable(
    module: torch.nn.Module,
    module_name: str,
    print_output: bool = True,
    include_stride: bool = False,
    include_device: bool = False,
    colored: bool = False,
    expanded_def: bool = False,
    additional_meta: list[str] | None = None,
) -> str:
    graph = module.graph
    if graph is None or not isinstance(graph, torch.fx.Graph):
        raise AssertionError("print_readable must be used on a module with a graph")

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=True,
        include_stride=include_stride,
        include_device=include_device,
        colored=colored,
        expanded_def=expanded_def,
        additional_meta=additional_meta,
    )
    module_code = verbose_python_code.src
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 4)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if isinstance(submodule, GraphModule):
            submodule_code_list.append(
                _print_readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                    additional_meta=additional_meta,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 4)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output


def _metadata_hash(code: str, node_metadata: dict[int, dict[str, Any]]) -> str:
    """
    Create a content-addressed hash from code and metadata.

    Args:
        code: The source code string
        lineno_map: Mapping from line numbers to node indices
        node_metadata: Metadata for each node

    Returns:
        A 51-character base32-encoded hash
    """
    import json

    # Create a deterministic string representation of all components
    # We use JSON to ensure consistent serialization
    hash_data = {
        "code": code,
        "node_metadata": node_metadata,
    }
    hashing_str = json.dumps(hash_data).encode("utf-8")

    # [:51] to strip off the "Q====" suffix common to every hash value.
    return (
        base64.b32encode(hashlib.sha256(hashing_str).digest())[:51]
        .decode("utf-8")
        .lower()
    )


class _WrappedCall:
    def __init__(self, cls: type, cls_call: Callable[..., Any] | None) -> None:
        self.cls = cls
        self.cls_call = cls_call

    # Previously, if an error occurred when valid
    # symbolically-traced code was run with an invalid input, the
    # user would see the source of the error as coming from
    # `File "<eval_with_key_N">`, where N is some number. We use
    # this function to generate a more informative error message. We
    # return the traceback itself, a message explaining that the
    # error occurred in a traced Module's generated forward
    # function, and five lines of context surrounding the faulty
    # line
    @staticmethod
    def _generate_error_message(frame_summary: traceback.FrameSummary) -> str:
        # auxiliary variables (for readability)
        err_lineno = frame_summary.lineno
        if err_lineno is None:
            raise AssertionError("frame_summary.lineno is None")
        line = frame_summary.line
        if line is None:
            raise AssertionError("frame_summary.line is None")
        err_line_len = len(line)
        all_src_lines = linecache.getlines(frame_summary.filename)

        # constituent substrings of the error message
        tb_repr = torch._dynamo.disable(
            traceback.format_exc,
            reason="do not trace into traceback.format_exc when generating error message",
        )()
        custom_msg = (
            "Call using an FX-traced Module, "
            f"line {err_lineno} of the traced Module's "
            "generated forward function:"
        )
        before_err = "".join(all_src_lines[err_lineno - 2 : err_lineno])
        marker = "~" * err_line_len + "~~~ <--- HERE"
        err_and_after_err = "\n".join(all_src_lines[err_lineno : err_lineno + 2])

        # joined message
        return "\n".join([tb_repr, custom_msg, before_err, marker, err_and_after_err])

    def __call__(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            if self.cls_call is not None:
                return self.cls_call(obj, *args, **kwargs)
            else:
                return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
        except Exception as e:
            if not e.__traceback__:
                raise AssertionError("Exception has no traceback") from e
            topmost_framesummary: traceback.FrameSummary = (
                traceback.StackSummary.extract(traceback.walk_tb(e.__traceback__))[-1]
            )
            if "eval_with_key" in topmost_framesummary.filename:
                print(
                    _WrappedCall._generate_error_message(topmost_framesummary),
                    file=sys.stderr,
                )
                raise e.with_traceback(None)  # noqa: B904
            else:
                raise e


@compatibility(is_backward_compatible=True)
class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        # address issue described at https://github.com/pytorch/pytorch/issues/63883
        # in other words, traverse class hierarchy to fix the redundant class definition problem
        for t in cls.__mro__:
            c = t.__qualname__.split(".")[-1]
            if c != "GraphModuleImpl":
                cls = t
                break

        class GraphModuleImpl(cls):  # type: ignore[misc, valid-type]
            pass

        return super().__new__(GraphModuleImpl)

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        class_name: str = "GraphModule",
    ) -> None:
        """
        Construct a GraphModule.

        Args:

            root (Union[torch.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

            class_name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.
        """
        super().__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, torch.nn.Module):
            if hasattr(root, "training"):
                self.training = root.training

            # When we pickle/unpickle graph module, we don't want to drop any module or attributes.
            if isinstance(root, _CodeOnlyModule):
                for k, _ in root.named_children():
                    _copy_attr(root, self, k)

                for k, _ in root.named_buffers():
                    _copy_attr(root, self, k)

                for k, _ in root.named_parameters():
                    _copy_attr(root, self, k)

            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    if not isinstance(node.target, str):
                        raise AssertionError(
                            f"Expected node.target to be str, got {type(node.target)}"
                        )
                    _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    if not isinstance(node.target, str):
                        raise AssertionError(
                            f"Expected node.target to be str, got {type(node.target)}"
                        )
                    if node.target not in root:
                        raise RuntimeError(
                            "Node "
                            + str(node)
                            + " referenced target "
                            + node.target
                            + " but that target was not provided in ``root``!"
                        )
                    targets_to_copy.append(node.target)
            # Sort targets in ascending order of the # of atoms.
            # This will ensure that less deeply nested attributes are assigned
            # before more deeply nested attributes. For example, foo.bar
            # will be assigned before foo.bar.baz. Otherwise, we might assign
            # the user-provided ``foo.bar`` and wipe out the previously-assigned
            # ``foo.bar.baz``
            targets_to_copy.sort(key=lambda t: t.count("."))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError("Unsupported type " + str(root) + " passed for root!")

        self.graph = graph

        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because torch.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls: type[Tracer] | None = None
        if (
            self.graph._tracer_cls
            and "<locals>" not in self.graph._tracer_cls.__qualname__
        ):
            # pyrefly: ignore [bad-assignment]
            self._tracer_cls = self.graph._tracer_cls

        self._tracer_extras = {}
        if self.graph._tracer_extras:
            self._tracer_extras = self.graph._tracer_extras

        # Dictionary to store metadata
        self.meta: dict[str, Any] = {}
        self._replace_hooks: list[Callable[[Node, str, Node], object]] = []
        self._create_node_hooks: list[Callable[[Node], object]] = []
        self._erase_node_hooks: list[Callable[[Node], object]] = []
        # Used to remove hooks from deepcopied graph modules within a context manager.
        self._deepcopy_hooks: list[Callable[[GraphModule], object]] = []
        self.shape_env: ShapeEnv | None = None

    # TorchScript breaks trying to compile the graph setter because of the
    # continued string literal. Issue here: https://github.com/pytorch/pytorch/issues/44842
    #
    # Shouldn't be an issue since these methods shouldn't be used in TorchScript anyway
    __jit_unused_properties__ = ["graph", "_boxed_call"]

    @property
    def _boxed_call(self) -> bool:
        return isinstance(self._graph._codegen, _BoxedCodeGen)

    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    @graph.setter
    def graph(self, g: Graph) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        if not isinstance(g, Graph):
            raise AssertionError(f"Expected a Graph instance, but got {type(g)}")
        self._graph = g
        g.owning_module = self
        self.recompile()

    @compatibility(is_backward_compatible=False)
    def to_folder(
        self, folder: str | os.PathLike[str], module_name: str = "FxModule"
    ) -> None:
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
        folder = Path(folder)
        Path(folder).mkdir(exist_ok=True)
        torch.save(self.state_dict(), folder / "state_dict.pt")
        tab = " " * 4
        custom_builtins = "\n".join([v.import_str for v in _custom_builtins.values()])
        model_str = f"""
import torch
{custom_builtins}

from torch.nn import *
class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

        def _gen_model_repr(module_name: str, module: torch.nn.Module) -> str | None:
            safe_reprs = [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules: list[str] = []
        for module_name, module in self.named_children():
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f"{module_name}.pt"
                torch.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
                # weights_only=False as this is legacy code that saves the model
                module_load_str = f"torch.load(r'{module_file}', weights_only=False)"
                model_str += f"{tab * 2}setattr(self, '{module_name}', {module_load_str}) # {module_repr}\n"
            else:
                model_str += f"{tab * 2}setattr(self, '{module_name}', {module_str})\n"

        for buffer_name, buffer in self._buffers.items():
            if buffer is None:
                continue
            model_str += f"{tab * 2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"

        for param_name, param in self._parameters.items():
            if param is None:
                continue
            model_str += f"{tab * 2}setattr(self, '{param_name}', torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype})))\n"

        model_str += (
            f"{tab * 2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
        )
        model_str += f"{_addindent(self.code, 4)}\n"

        module_file = folder / "module.py"
        module_file.write_text(model_str)

        init_file = folder / "__init__.py"
        init_file.write_text("from .module import *")

        if len(blobified_modules) > 0:
            warnings.warn(
                "Was not able to save the following children modules as reprs -"
                f"saved as pickled files instead: {blobified_modules}"
            )

    @compatibility(is_backward_compatible=True)
    def add_submodule(self, target: str, m: torch.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)
        """
        *prefix, field = target.split(".")
        mod: torch.nn.Module = self

        for item in prefix:
            submod = getattr(mod, item, None)

            if submod is None:
                submod = torch.nn.Module()
                setattr(mod, item, submod)

            if not isinstance(submod, torch.nn.Module):
                return False

            mod = submod

        mod.add_module(field, m)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_submodule(self, target: str) -> bool:
        """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
        atoms = target.split(".")
        path, target_submod = atoms[:-1], atoms[-1]
        mod: torch.nn.Module = self

        # Get the parent module
        for item in path:
            if not hasattr(mod, item):
                return False

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                return False

        if not hasattr(mod, target_submod):
            return False

        if not isinstance(getattr(mod, target_submod), torch.nn.Module):
            return False

        delattr(mod, target_submod)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_all_unused_submodules(self) -> None:
        """
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a
        ``get_attr`` node

        This method can be called to clean up an ``nn.Module`` without
        manually calling ``delete_submodule`` on each unused submodule.
        """
        used: set[str] = set()

        for node in self.graph.nodes:
            if node.op in ("call_module", "get_attr") and isinstance(node.target, str):
                # A list of strings representing the different parts
                # of the path. For example, `foo.bar.baz` gives us
                # ["foo", "bar", "baz"]
                fullpath = node.target.split(".")

                # If we're looking at multiple parts of a path,
                # join them with a dot. Otherwise, return that single
                # element without doing anything to it.
                def join_fn(x: str, y: str) -> str:
                    return ".".join([x, y] if y else [x])

                # Progressively collect all the names of intermediate
                # modules. For example, if we have the target
                # `foo.bar.baz`, we'll add `foo`, `foo.bar`, and
                # `foo.bar.baz` to the set.
                used.update(itertools.accumulate(fullpath, join_fn))

                # For a `call_module` node, also register all recursive submodules
                # as used
                if node.op == "call_module":
                    try:
                        str_target = cast(str, node.target)
                        submod = self.get_submodule(str_target)

                        for submod_name, _ in submod.named_modules():
                            if submod_name != "":
                                used.add(".".join([str_target, submod_name]))
                    except AttributeError:
                        # Node referenced nonexistent submodule, don't need to
                        # worry about GCing anything
                        pass

        to_delete = [name for name, _ in self.named_modules() if name not in used]

        for name in to_delete:
            self.delete_submodule(name)

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, "_code"):
            raise RuntimeError(
                "Code has not been generated! Please report a bug to PyTorch"
            )
        return self._code

    @compatibility(is_backward_compatible=True)
    def recompile(self) -> PythonCode:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        # Do not import anything inside recompile, it might slow down the
        # function and cause perf regression. Import outside of the method instead.
        if isinstance(self._graph._codegen, _PyTreeCodeGen):
            self._in_spec = self._graph._codegen.pytree_info.in_spec
            self._out_spec = self._graph._codegen.pytree_info.out_spec

        python_code = self._graph.python_code(
            root_module="self",
            record_func=fx_experimental_config.enrich_profiler_metadata,
        )
        self._code = python_code.src
        self._lineno_map = python_code._lineno_map
        self._prologue_start = python_code._prologue_start

        cls = type(self)
        co_fields: dict[str, Any] = (
            self._graph._co_fields if hasattr(self._graph, "_co_fields") else {}
        )

        if fx_experimental_config.enrich_profiler_metadata:
            # Generate metadata and register for profiler augmentation
            node_metadata: dict[int, dict[str, Any]] = {}
            for i, node in enumerate(self._graph.nodes):
                node_metadata[i] = {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "stack_trace": node.meta.get("stack_trace", None),
                }

            # Generate a content-addressed filename based on hash of code and metadata
            # This ensures the same code+metadata always generates the same filename
            hash_value = _metadata_hash(self._code, node_metadata)
            file_stem = f"{FX_GRAPH_MODULE_FILE_PREFIX}_{hash_value}"
            filename = f"{file_stem}.py"

            # Only include co_filename to use it directly as the cache key
            co_fields = {
                "co_filename": filename,
            }

            # Store metadata in global in-memory registry
            metadata = {
                "lineno_map": python_code._lineno_map,
                "prologue_start": python_code._prologue_start,
                "node_metadata": node_metadata,
            }

            # Register metadata in the global registry
            from torch.fx.traceback import _register_fx_metadata

            _register_fx_metadata(filename, metadata)

            # Replace the placeholder in generated code with actual filename
            # The double hash ## convention is used by post-processing to find the fx markers
            self._code = self._code.replace(
                "torch._C._profiler._RecordFunctionFast('## ENTER_GRAPH_PLACEHOLDER_KEY ##')",
                f"torch._C._profiler._RecordFunctionFast('## {filename} ##')",
            )

        cls.forward = _forward_from_src(self._code, python_code.globals, co_fields)

        # Determine whether this class explicitly defines a __call__ implementation
        # to wrap. If it does, save it in order to have wrapped_call invoke it.
        # If it does not, wrapped_call can use a dynamic call to super() instead.
        # In most cases, super().__call__ should be torch.nn.Module.__call__.
        # We do not want to hold a reference to Module.__call__ here; doing so will
        # bypass patching of torch.nn.Module.__call__ done while symbolic tracing.
        cls_call = cls.__call__ if "__call__" in vars(cls) else None

        if "_wrapped_call" not in vars(cls):
            cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

        self._recompile_submodules()

        def call_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            return self._wrapped_call(self, *args, **kwargs)

        cls.__call__ = call_wrapped  # type: ignore[method-assign]

        return python_code

    def _recompile_submodules(self) -> list[tuple[str, PythonCode]]:
        """
        Recompile all submodules of this graph module, returning their respective PythonCodes
        in a similar format to named_children()
        """
        results: list[tuple[str, PythonCode]] = []
        for name, mod in self.named_children():
            if isinstance(mod, GraphModule):
                results.append((name, mod.recompile()))
        return results

    # Passing Tracer as argument allows subclasses extending fx.GraphModule
    # define their own Tracer (extending fx.Tracer).

    def __reduce_package__(self, exporter: PackageExporter) -> tuple[Any, ...]:
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        # Store node.meta["stack_trace"] so we can recover them after re-tracing during deserialization
        node_meta_stack_trace = {
            node.name: node.meta["stack_trace"]
            for node in self.graph.nodes
            if "stack_trace" in node.meta
        }
        dict_without_graph["_graphmodule_graph_node_meta_stack_trace"] = (
            node_meta_stack_trace
        )

        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        python_code = self.recompile()
        import_block = _format_import_block(python_code.globals, exporter.importer)
        module_code = import_block + self.code
        exporter.save_source_string(generated_module_name, module_code)
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """
        Serialization of GraphModule. We serialize only the generated code, not
        the underlying ``Graph``. This is because ``Graph`` does not have on-disk
        backward-compatibility guarantees, whereas Python source code does.
        On the deserialization side, we symbolically trace through the generated
        code to regenerate the underlying ``Graph``
        """
        dict_without_graph = self.__dict__.copy()

        python_code = self.recompile()
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        return (reduce_graph_module, (dict_without_graph, import_block))

    def _deepcopy_init(self) -> Callable[..., None]:
        return GraphModule.__init__

    # because __reduce__ is defined for serialization,
    # we need to define deepcopy otherwise it will call __reduce__
    # and cause symbolic tracing to occur every time we try to copy the object
    def __deepcopy__(self, memo: dict[int, Any]) -> GraphModule:
        res = type(self).__new__(type(self))
        memo[id(self)] = res
        # Opt-in: smuggle non-pickleable torchbind / ProcessGroup objects
        # through deepcopy as shared references.
        if _share_torchbind_and_process_group.get():
            import torch.distributed as _dist

            _PG = _dist.ProcessGroup if _dist.is_available() else ()
            for v in self.__dict__.values():
                if isinstance(v, torch.ScriptObject) and not (
                    v._has_method("__getstate__")  # type: ignore[attr-defined]
                    and v._has_method("__setstate__")  # type: ignore[attr-defined]
                ):
                    memo.setdefault(id(v), v)
                elif isinstance(v, _PG):
                    memo.setdefault(id(v), v)
        fake_mod = _CodeOnlyModule(copy.deepcopy(self.__dict__, memo))
        self._deepcopy_init()(res, fake_mod, fake_mod.__dict__["_graph"])
        # hooks are lost during `GraphModule.__init__`, so we need to copy over
        # them explicitly, note right now we are only copying state_dict related
        # hooks, to reduce bc-related issues, we can copy forward/backward related
        # hooks in the future as well if needed
        extra_preserved_attrs = [
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
            "_replace_hooks",
            "_create_node_hooks",
            "_erase_node_hooks",
            "_deepcopy_hooks",
        ]
        for attr in extra_preserved_attrs:
            if attr in self.__dict__:
                setattr(res, attr, copy.deepcopy(self.__dict__[attr], memo))
        res.meta = copy.deepcopy(getattr(self, "meta", {}), memo)
        if _USER_PRESERVED_ATTRIBUTES_KEY in res.meta:
            for attr_name, attr in res.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
                setattr(res, attr_name, attr)
        if hasattr(self, "_deepcopy_hooks"):
            for hook in self._deepcopy_hooks:
                hook(res)
        return res

    def __copy__(self) -> GraphModule:
        from ._lazy_graph_module import _make_graph_module

        res = _make_graph_module(self, self.graph)
        res.meta = getattr(self, "meta", {})
        return res

    @compatibility(is_backward_compatible=False)
    def print_readable(
        self,
        print_output: bool = True,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        *,
        # If `fast_sympy_print` is True then we use a sympy printer which is faster
        # but may result in less-readable output.
        fast_sympy_print: bool = False,
        expanded_def: bool = False,
        additional_meta: list[str] | None = None,
    ) -> str:
        """
        Return the Python code generated for current GraphModule and its children GraphModules.

        Args:
            additional_meta: Optional list of meta keys to include in the output.
                For each key in the list, if it exists in node.meta, its value
                will be shown in the format "key: value".
                Example: `print_readable(additional_meta=["seq_nr"])`.
        """
        ctx_mgr = contextlib.ExitStack()
        with ctx_mgr:
            if fast_sympy_print:
                from torch._inductor.utils import sympy_str

                def fast_repr(expr: torch.types.PySymType) -> str:
                    return sympy_str(expr.node.expr)

                ctx_mgr.enter_context(_override_sym_repr(fast_repr))

            r = _print_readable(
                self,
                self._get_name(),
                print_output,
                include_stride,
                include_device,
                colored,
                expanded_def,
                additional_meta,
            )
            return r

    def __str__(self) -> str:
        orig_str = super().__str__()
        print_readable_reminder = (
            "# To see more debug info, please use `graph_module.print_readable()`"
        )
        return "\n".join([orig_str, self._code, print_readable_reminder])

    def _replicate_for_data_parallel(self) -> GraphModule:
        new_gm = self.__copy__()
        object.__setattr__(new_gm, "_is_replica", True)
        return new_gm

    @contextlib.contextmanager
    def _set_replace_hook(
        self, f: Callable[[Node, str, Node], object]
    ) -> Generator[None, None, None]:
        """
        Takes a callable which will be called every time when we replace a node
        to a new node, or change the node's name. Callable takes three arguments:
        the old node we're changing, and NAME of the new node, followed by the
        user node which consumes the old node to be replaced.
        """
        if not callable(f):
            raise AssertionError("Replace hook must be a callable.")
        self._register_replace_node_hook(f)
        try:
            yield
        finally:
            self._unregister_replace_node_hook(f)

    def _register_replace_node_hook(
        self, f: Callable[[Node, str, Node], object]
    ) -> None:
        """
        Takes a callable which will be called every time when we replace a node
        to a new node, or change the node's name. Callable takes three arguments:
        the old node we're changing, and NAME of the new node, followed by the
        user node which consumes the old node to be replaced.
        """
        if not callable(f):
            raise AssertionError("create_node hook must be a callable.")
        self._replace_hooks.append(f)

    def _unregister_replace_node_hook(
        self, f: Callable[[Node, str, Node], object]
    ) -> None:
        """
        Takes a callable which was previously registered to be called every time when we replace a node.
        This function will unregister that callable so it is no longer invoked on node replacement.
        """
        if not callable(f):
            raise AssertionError("create_node hook must be a callable.")
        self._replace_hooks.remove(f)

    def _register_create_node_hook(self, f: Callable[[Node], object]) -> None:
        """
        Takes a callable which will be called after we create a new node. The
        callable takes the newly created node as input and returns None.
        """
        if not callable(f):
            raise AssertionError("create_node hook must be a callable.")
        self._create_node_hooks.append(f)

    def _unregister_create_node_hook(self, f: Callable[[Node], object]) -> None:
        """
        Takes a callable which was previously registered to be called after we create a node.
        This function will unregister that callable so it is no longer invoked on node creation.
        """
        if not callable(f):
            raise AssertionError("create_node hook must be a callable.")
        self._create_node_hooks.remove(f)

    def _register_erase_node_hook(self, f: Callable[[Node], object]) -> None:
        """
        Takes a callable which will be called after we erase a node. The
        callable takes the node that is being erased as input and returns None.
        """
        if not callable(f):
            raise AssertionError("erase_node hook must be a callable.")
        self._erase_node_hooks.append(f)

    def _unregister_erase_node_hook(self, f: Callable[[Node], object]) -> None:
        """
        Takes a callable which was previously registered to be called after we erase a node.
        This function will unregister that callable so it is no longer invoked on node erasure.
        """
        if not callable(f):
            raise AssertionError("erase_node hook must be a callable.")
        self._erase_node_hooks.remove(f)

    def _register_deepcopy_hook(self, f: Callable[[GraphModule], object]) -> None:
        """
        Takes a callable which will be called when we deepcopy this graph module. The
        callable takes the resulting deepcopied graph module.
        """
        if not callable(f):
            raise AssertionError("deepcopy hook must be a callable.")
        self._deepcopy_hooks.append(f)

    def _unregister_deepcopy_hook(self, f: Callable[[GraphModule], object]) -> None:
        """
        Takes a callable which was previously registered to be called after deepcopy.
        This function will unregister that callable so it is no longer invoked on deepcopy.
        """
        if not callable(f):
            raise AssertionError("deepcopy hook must be a callable.")
        self._deepcopy_hooks.remove(f)
<|endoftext|>import collections
import copy
import dis
import enum
import inspect
import logging
import operator
import sys
import traceback
import types
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from typing import Any, cast, TypeVar
from typing_extensions import Never

import torch
import torch.fx.traceback as fx_traceback
from torch._C import _fx_map_arg as map_arg
from torch._library.opaque_object import is_opaque_constant_type
from torch._logging import getArtifactLogger
from torch.utils._pytree import tree_map_
from torch.utils._traceback import CapturedTraceback

from ._compatibility import compatibility
from .graph import Graph, magic_methods, reflectable_magic_methods
from .immutable_collections import immutable_dict, immutable_list
from .node import Argument, base_types, Node, Target
from .operator_schemas import check_for_mutable_operation


__all__ = [
    "TracerBase",
    "GraphAppendingTracer",
    "TraceError",
    "Proxy",
    "MetaProxy",
    "Attribute",
    "ParameterProxy",
    "Scope",
    "ScopeContextManager",
]


log = logging.getLogger(__name__)
annotation_log = getArtifactLogger(__name__, "annotation")


def _is_arbitrary_callable(obj: object) -> bool:
    """
    Returns True if obj is an arbitrary callable (function, lambda, method, etc.)
    that requires special tracing to handle. These cannot be symbolically traced
    using the standard Proxy mechanism.
    """
    import functools
    import types

    return isinstance(
        obj,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            functools.partial,
        ),
    )


def _find_arbitrary_callable(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """
    Recursively searches args and kwargs for any arbitrary callable.
    Returns the first arbitrary callable found, or None if none exist.
    """
    found = None

    _T = TypeVar("_T")

    def check(obj: _T) -> _T:
        nonlocal found
        if found is not None:
            return obj
        if _is_arbitrary_callable(obj):
            found = obj
        return obj

    tree_map_(check, args)
    tree_map_(check, kwargs)
    return found


@compatibility(is_backward_compatible=False)
class Scope:
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)


        class M(torch.nn.Module):
            def __init__(self) -> None:
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    """

    def __init__(self, module_path: str, module_type: Any) -> None:
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


@compatibility(is_backward_compatible=False)
class ScopeContextManager:
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_scope: Scope,
    ) -> None:
        super().__init__()
        # Keep a copy of prev scope to restore on exit
        self._prev_scope = copy.copy(scope)
        # Update scope to current scope
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # Save a reference so we can restore it
        self._scope = scope

    def __enter__(self) -> Scope:
        return self._scope

    def __exit__(self, *args: object) -> None:
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
        return


_COPY_META_FIELDS = [
    "nn_module_stack",
    "torch_fn",
    "source_fn_stack",
    "original_aten",
    "recompute",
    "ac_graph_id",
    "has_backward_hook",
    "from_node",
    "quantization_tag",  # TODO deprecated
    "_numeric_debug_handle",  # TODO deprecated
    "custom",
    "partitioner_tag",
]


# Set of (filename, co_name) pairs registered as user-code anchors. Frames
# matching any entry are preserved by ``_filter_traceback_frames`` when
# ``_record_forward_stack_traces_only`` is True. Populated via
# ``_register_stack_trace_anchor`` (e.g. by ``annotate_fn``).
_STACK_TRACE_ANCHORS: set[tuple[str, str]] = set()


def _register_stack_trace_anchor(fn: Callable[..., Any]) -> None:
    code = getattr(fn, "__code__", None)
    if code is not None:
        _STACK_TRACE_ANCHORS.add((code.co_filename, code.co_name))


@compatibility(is_backward_compatible=True)
class TracerBase:
    graph: Graph
    record_stack_traces: bool = False
    # When record_stack_traces is True, only reocrd stack traces
    # with forward function names.
    # This helps when we want stack trace back to model code
    _record_forward_stack_traces_only: bool = False
    # Feature flag for mutable schema checking
    # Enableby default in 1.12
    check_mutable_operations: bool = False
    # Feature flag for assert tracing
    trace_asserts: bool = False
    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = False

    # Name of the function to be traced. It will only be used when
    # ``root`` is an instance of ``nn.Module``
    traced_func_name: str = "forward"

    # Maps the containing module's name to the operator name
    scope: Scope

    # Records the module call stack
    module_stack: OrderedDict[str, tuple[str, Any]]

    # Mapping of node name to module scope
    node_name_to_scope: dict[str, tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """

        if kind == "call_function" and self.check_mutable_operations:
            target_fn = cast(Callable[..., Any], target)
            check_for_mutable_operation(target_fn, args, kwargs)

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        # TODO node_name_to_scope will be deprecated in favor of
        # node.meta['nn_module_stack']
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )

        # Optionally set stack trace on the created Node for debugging purposes
        if fx_traceback.has_preserved_node_meta():
            current_meta: dict[str, Any] = fx_traceback.get_current_meta()

            stack_trace = current_meta.get("stack_trace")
            if stack_trace:
                node.stack_trace = stack_trace

                if fx_traceback.GRADIENT_ACC_SPECIAL_STACK in stack_trace:
                    node.meta["is_gradient_acc"] = True
                    node.meta["autograd_backward"] = True

            # Explicitly set the stack_trace, nn_module_stack and source_fn on the node.meta
            # If other meta fields are needed, they can be added here
            for field in _COPY_META_FIELDS:
                if field in current_meta:
                    node.meta[field] = copy.copy(current_meta[field])

            new_seq_nr = _get_seq_nr(node.name)
            if new_seq_nr is not None:
                annotation_log.debug(
                    "Assigning new_seq_nr %s to %s", new_seq_nr, node.name
                )
                node.meta["seq_nr"] = new_seq_nr

            # See Note [Functionalization View Replay Annotation]
            # Overriding some node meta with the original node meta of the
            # regenerated node.
            replay_node: Node | None = fx_traceback.get_current_replay_node()
            if replay_node is not None:
                node.meta["is_functional_regenerated"] = True
                if "custom" in replay_node.meta:
                    node.meta["custom"] = replay_node.meta.get("custom")
                if "stack_trace" in replay_node.meta:
                    node.stack_trace = replay_node.meta.get("stack_trace")

            if current_meta.get("autograd_backward", False):
                node.meta["autograd_backward"] = True

        elif self.module_stack:
            node.meta["nn_module_stack"] = copy.copy(self.module_stack)

        if self.record_stack_traces and not node.stack_trace:
            user_stack_summary = CapturedTraceback.extract().summary()
            if user_stack_summary:
                user_stack_summary = self._filter_traceback_frames(user_stack_summary)
                if user_stack_summary:
                    node.stack_trace = "".join(user_stack_summary.format()).strip()

        log.debug("create_node %s", node)
        return node

    def _filter_traceback_frames(
        self, user_stack_summary: traceback.StackSummary
    ) -> traceback.StackSummary:
        # This method can be overridden to customize the frame filtering logic
        # for the recorded stack trace
        user_frames: list[traceback.FrameSummary] = []
        if self._record_forward_stack_traces_only:
            user_frames = [
                frame
                for frame in user_stack_summary
                if (
                    frame.name == "forward"
                    or frame.filename.endswith("torch/__init__.py")
                    or (frame.filename, frame.name) in _STACK_TRACE_ANCHORS
                )
            ]
        else:
            first_forward = -1
            for i, frame in enumerate(user_stack_summary):
                if frame.name == "forward":
                    user_frames = user_stack_summary[i:]
                    first_forward = i
                    break

            # Not having a "forward" call in the stacktrace implies the
            # stacktrace will probably be irrelevant
            if first_forward == -1:
                user_frames: list[traceback.FrameSummary] = []

        from torch.fx.experimental.symbolic_shapes import uninteresting_files

        user_frames = [
            frame
            for frame in user_frames
            if frame.filename not in uninteresting_files()
        ]

        return traceback.StackSummary.from_list(user_frames)

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> "Proxy":
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] | None = None,
    ) -> "Proxy":
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        """

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        if not isinstance(args_, tuple):
            raise AssertionError(f"Expected args_ to be tuple, got {type(args_)}")
        if not isinstance(kwargs_, dict):
            raise AssertionError(f"Expected kwargs_ to be dict, got {type(kwargs_)}")

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)

        return proxy

    def _find_user_frame(self) -> types.FrameType | None:
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        # We have to do a little dance here. Basically, walk up the callstack and
        # record the first frame not in the pytorch source. This is the frame executing
        # the user code during tracing.
        frame = inspect.currentframe()

        pt_files = [
            "torch/fx/proxy.py",
            "torch/fx/_symbolic_trace.py",
            "torch/fx/experimental/proxy_tensor.py",
            "torch/_ops.py",
            "torch/_tensor.py",
            "torch/utils/_python_dispatch.py",
            "torch/_prims_common/wrappers.py",
            "torch/_refs/__init__.py",
            "torch/_refs/nn/functional/__init__.py",
            "torch/utils/_stats.py",
        ]
        while frame:
            frame = frame.f_back
            if frame and all(
                not frame.f_code.co_filename.endswith(file) for file in pt_files
            ):
                break

        if not frame:
            return None

        return frame

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        # IMPORTANT: Are you here because you are trying to proxy a new type into
        # the graph? Please Please Please contact someone on the PyTorch Compiler team;
        # the considerations are subtle.
        #
        # 1) When you add a new type, all of the downstream consumers and pass writers
        # need to handle the new type. torch.fx is intended to be easy to write
        # passes for, so we will push back against new types.
        # 2) In torch.compile's IR, there are only specific operations that go
        # into the graph. In particular, Tensor operations should go into the graph,
        # but non-Tensor operations shouldn't. What that means is that constructors
        # for new types *SHOULD NOT* become nodes in the FX graph.
        handler = _create_arg_bypass.get(type(a))
        if handler is not None:
            # this is just a performance optimization and can be removed if needed
            # for common types, we have a fast path to avoid isinstance() overhead
            # this doesn't remove the checks below since we need to handle subclasses
            return handler(self, a)

        if isinstance(a, Proxy):
            return a.node  # most common arg type goes first
        elif hasattr(a, "__fx_create_arg__"):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple):
            if hasattr(a, "_fields"):
                # NamedTuple constructors don't seem to like getting a generator
                # expression as an argument to their constructor, so build this
                # intermediate tuple and unpack it into the NamedTuple constructor
                args = [self.create_arg(elem) for elem in a]
                return type(a)(*args)  # type: ignore[arg-type]
            return type(a)([self.create_arg(elem) for elem in a])
        elif isinstance(a, list):
            return [self.create_arg(elem) for elem in a]
        elif isinstance(a, dict):
            return _create_arg_dict(self, a)
        elif isinstance(a, slice):
            return slice(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        elif isinstance(a, range):
            return range(  # pyrefly: ignore[no-matching-overload]
                self.create_arg(a.start),  # pyrefly: ignore[bad-argument-type]
                self.create_arg(a.stop),  # pyrefly: ignore[bad-argument-type]
                self.create_arg(a.step),  # pyrefly: ignore[bad-argument-type]
            )

        elif isinstance(a, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return a  # pyrefly: ignore[bad-return]

        elif is_opaque_constant_type(type(a)):
            return a

        elif is_dataclass(a):
            kwargs = {
                field.name: self.create_arg(getattr(a, field.name))
                for field in fields(a)
            }
            return self.create_node("call_function", a.__class__, (), kwargs)

        elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
            return a  # pyrefly: ignore[bad-return]

        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: "Proxy") -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to control flow"
        )

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: "Proxy") -> Iterator:  # pyrefly: ignore[implicit-any]
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError(
            "Proxy object cannot be iterated. This can be "
            "attempted when the Proxy is used in a loop or"
            " as a *args or **kwargs function argument. "
            "See the torch.fx docs on pytorch.org for a "
            "more detailed explanation of what types of "
            "control flow can be traced, and check out the"
            " Proxy docstring for help troubleshooting "
            "Proxy iteration errors"
        )

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> "Proxy":
        """Called when a proxy object has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator if ** is supposed to work in your custom tracer.
        """
        return Attribute(obj, "keys")()


def _get_seq_nr(node_name: str = "") -> int | None:
    """
    Returns the seq_nr node meta for the current proxy node that we're creating.
    The seq_nr number in node meta is related to but not the same as the "sequence number"
    in autograd.
    We use the seq_nr to correlate forward and backward nodes in the traced FX graphs
    (e.g. in copy_fwd_metadata_to_bw_nodes).
    The corresponding forward and backward FX graph nodes should have the same seq_nr.

    The ordering of the seq_nr in the graph does not indicate when the node is executed.
    For example, the nodes in the invoke_subgraph HOP's subgraph may be called multiple
    times by different HOP nodes, but these nodes only have a single seq_nr, which may be
    smaller, the same, or larger than the calling HOP.

    `node_name` is the name of the node that we're creating. It is used for logging only.
    """
    current_meta: dict[str, Any] = fx_traceback.get_current_meta()
    new_seq_nr = None
    # The sequence_nr increments every time a new autograd Node
    # is created. During the FWD pass we store the sequence_nr
    # corresponding to the last autograd Node created on this fx
    # node's meta.  A single aten op can create multiple autograd
    # nodes as is the case with in-place foreach ops. During the
    # BWD pass we retrieve the sequence_nr stored on the current
    # executing autograd Node. See NOTE [ Sequence Number ].
    if current_meta.get("in_grad_fn", 0) > 0:
        # This branch is used to get seq_nr for backward nodes
        annotation_log.debug("%s: seq_nr from current_meta grad_fn_seq_nr", node_name)
        new_seq_nr = current_meta["grad_fn_seq_nr"][-1]

    elif torch.fx.traceback._is_preserving_node_seq_nr():
        # Special case where we preserve seq_nr from currently tracing node
        # Used to preserve seq_nr when re-tracing subgraphs in HOP
        annotation_log.debug("%s: seq_nr from current_meta seq_nr", node_name)
        new_seq_nr = current_meta.get("seq_nr")
    else:
        # Here we decrement to account for the sequence_nr having
        # just been incremented while tracing this lowered aten op.
        # This branch is used to get seq_nr for forward nodes
        new_seq_nr = torch.autograd._get_sequence_nr() - 1

    if not torch.fx.traceback._is_preserving_node_seq_nr():
        # See Note [Functionalization View Replay Annotation]
        # Overriding some node meta with the original node meta of the
        # regenerated node.
        replay_node: Node | None = fx_traceback.get_current_replay_node()
        if replay_node is not None:
            if "seq_nr" in replay_node.meta:
                annotation_log.debug("%s: seq_nr from replay_node", node_name)
                new_seq_nr = replay_node.meta["seq_nr"]

    return new_seq_nr


# used in Proxy object when just appending to the graph while not tracing.
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack: OrderedDict[str, tuple[str, Any]] = collections.OrderedDict()
        self.node_name_to_scope: dict[str, tuple[str, type]] = {}


@compatibility(is_backward_compatible=False)
def assert_fn(x: object) -> None:
    if not x:
        raise AssertionError("Assertion failed")


@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    pass


@compatibility(is_backward_compatible=True)
class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/README.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: "TracerBase | None" = None) -> None:
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    def __getattr__(self, k: str) -> "Attribute":
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __deepcopy__(self, memo: dict[int, Any]) -> "Proxy":
        # We have to explicitly override this method, because otherwise deepcopy
        # will go to __getattr__(self, "__deepcopy__") and return a
        # Attribute(__deepcopy__), and may go into an infinite loop in some cases.
        import copy

        new_dict: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            try:
                new_obj = copy.deepcopy(v, memo)
            except Exception:
                log.warning(
                    "Shallow copy %s of Proxy because it cannot be deepcopied. "
                    "Proxy is created for node %s",
                    k,
                    self.node.name,
                )
                new_obj = copy.copy(v)
            new_dict[k] = new_obj
        if "node" not in new_dict:
            raise AssertionError("'node' not in new_dict during proxy unpickling")
        if "tracer" not in new_dict:
            raise AssertionError("'tracer' not in new_dict during proxy unpickling")
        new_proxy = Proxy(new_dict["node"], new_dict["tracer"])
        for k, v in new_dict.items():
            new_proxy.__dict__[k] = v
        return new_proxy

    def __setstate__(self, d: dict[str, Any]) -> None:
        # This is called when being unpickled/loaded.
        self.__dict__ = d

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs
        )

    def __iter__(self) -> Iterator["Proxy"]:
        frame = inspect.currentframe()
        if frame is None:
            raise AssertionError("inspect.currentframe() returned None")
        calling_frame = frame.f_back
        if calling_frame is None:
            raise AssertionError("frame.f_back is None")
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __abs__(self) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.abs, (self,), {})

    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            if frame is None:
                raise AssertionError("inspect.currentframe() returned None")
            calling_frame = frame.f_back
            if calling_frame is None:
                raise AssertionError("frame.f_back is None")
            insts = list(dis.get_instructions(calling_frame.f_code))
            if sys.version_info >= (3, 11):
                from bisect import bisect_left

                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == "POP_JUMP_IF_TRUE":
                first = insts[cur + 1]
                if inst.arg is None:
                    raise AssertionError("inst.arg is None for POP_JUMP_IF_TRUE")
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (
                    first.opname == "LOAD_GLOBAL"
                    and first.argval == "AssertionError"
                    or first.opname == "LOAD_ASSERTION_ERROR"
                )
                if starts_with_assert and last.opname == "RAISE_VARARGS":
                    self.tracer.create_proxy("call_function", assert_fn, (self,), {})
                    return True

        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self) -> "Proxy":
        return self.tracer.keys(self)

    def __len__(self) -> int:
        raise RuntimeError(
            "'len' is not supported in symbolic tracing by default. If you want "
            "this call to be recorded, please call torch.fx.wrap('len') at "
            "module scope"
        )

    @classmethod
    def __torch_function__(
        cls,
        orig_method: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "Proxy":
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: dict[TracerBase, None] = {}

        def find_tracer(a: Any) -> None:
            if isinstance(a, cls):
                tracers[a.tracer] = None

        tree_map_(find_tracer, args)
        tree_map_(find_tracer, kwargs)

        if len(tracers) > 1:
            raise RuntimeError(
                f"Found multiple different tracers {list(tracers.keys())} while "
                f"trying to trace operations {orig_method}"
            )
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy("call_method", orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy(
                "call_method", orig_method.__name__, args, kwargs
            )
        else:
            if isinstance(orig_method, torch._ops.HigherOrderOperator):
                bad_callable = _find_arbitrary_callable(args, kwargs)
                if bad_callable is not None:
                    raise RuntimeError(
                        f"Unable to symbolically trace the HigherOrderOperator "
                        f"{orig_method._name} because it received an arbitrary "
                        f"callable argument {bad_callable}. Use make_fx or dynamo "
                        f"tracing instead."
                    )
            return tracer.create_proxy(
                "call_function",
                orig_method,
                args,
                kwargs,
                name=tracer.graph._target_to_str(orig_method.__name__),
            )


@compatibility(is_backward_compatible=False)
class MetaProxy(Proxy):
    """
    A Proxy subclass that propagates metadata (meta['val']) during graph tracing.
    """

    def __init__(
        self, node: Node, tracer: "TracerBase | None" = None, fake_mode: Any = None
    ) -> None:
        super().__init__(node, tracer)
        self.fake_mode = fake_mode

    def __repr__(self) -> str:
        return f"MetaProxy({self.node.name})"

    @classmethod
    def __torch_function__(
        cls,
        orig_method: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "MetaProxy":
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        meta_proxy = None
        for arg in args:
            if isinstance(arg, MetaProxy):
                meta_proxy = arg
                break

        if meta_proxy is None:
            raise AssertionError(
                "No MetaProxy found in arguments, but one is expected."
            )

        proxy = super().__torch_function__(orig_method, types, args, kwargs)
        with meta_proxy.fake_mode:
            proxy.node.meta["val"] = orig_method(
                *[a.node.meta["val"] if isinstance(a, Proxy) else a for a in args],
                **kwargs,
            )
        return MetaProxy(proxy.node, proxy.tracer, meta_proxy.fake_mode)


@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: Proxy, attr: str) -> None:
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self) -> Node:  # pyrefly: ignore[bad-override]
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(
        self, tracer: TracerBase, node: Node, name: str, param: torch.nn.Parameter
    ) -> None:
        super().__init__(node, tracer)
        if not isinstance(param, torch.nn.Parameter):
            raise AssertionError(f"Expected Parameter, got {type(param)}")
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f"ParameterProxy({self.name})"

    @property
    def shape(self) -> torch.Size:
        return self.param.shape

    def size(self) -> torch.Size:
        return self.param.size()

    def dim(self) -> int:
        return self.param.dim()

    @property
    def ndim(self) -> int:
        return self.param.ndim

    def numel(self) -> int:
        return self.param.numel()

    def nelement(self) -> int:
        return self.param.nelement()


for method in magic_methods:

    def _scope(method: str) -> None:
        def impl(*args: Any, **kwargs: Any) -> "Proxy":
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy("call_function", target, args, kwargs)

        impl.__name__ = method
        as_magic = f"__{method.strip('_')}__"
        setattr(Proxy, as_magic, impl)

    _scope(method)


def _define_reflectable(orig_method_name: str) -> None:
    method_name = f"__r{orig_method_name.strip('_')}__"

    def impl(self: "Proxy", rhs: Any) -> "Proxy":
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy("call_function", target, (rhs, self), {})

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)


def _no_nodes_error(arg: Argument) -> Never:
    raise RuntimeError(
        "Keys for dictionaries used as an argument cannot contain a "
        f"Node. Got key: {arg}"
    )


def _create_arg_dict(self: TracerBase, a: dict[Any, Any]) -> dict[Any, Argument]:
    r: dict[Any, Argument] = {}
    for k, v in a.items():
        if not isinstance(k, str):
            # Check for invalid dict keys. We do not want a Proxy to appear
            # anywhere within the key. Since keys can be collection types,
            # we iterate through the key with map_arg
            k = self.create_arg(k)
            map_arg(k, _no_nodes_error)
        r[k] = self.create_arg(v)
    return r


_create_arg_bypass = {
    t: lambda self, a: a
    for t in [
        *base_types,
        type(None),
        type(...),
        torch._ops.OpOverload,
        torch._ops.HigherOrderOperator,
    ]
}
_create_arg_bypass[Proxy] = lambda self, a: a.node
_create_arg_bypass[tuple] = lambda self, a: tuple(self.create_arg(elem) for elem in a)
_create_arg_bypass[list] = lambda self, a: [self.create_arg(elem) for elem in a]
_create_arg_bypass[dict] = _create_arg_dict
_create_arg_bypass[immutable_list] = _create_arg_bypass[list]
_create_arg_bypass[immutable_dict] = _create_arg_bypass[dict]
<|endoftext|>from typing import NamedTuple

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node, Target
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata


__all__ = [
    "replace_target_nodes_with",
    "size_bytes",
    "get_size_of_all_nodes",
    "get_tensor_meta",
    "get_size_of_node",
]


@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
) -> None:
    """
    Modifies all nodes in fx_module.graph.nodes which match the specified op code
    and target, and updates them to match the new op code and target.
    """
    new_graph = Graph()
    val_map: dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            if not isinstance(args, tuple):
                raise AssertionError(f"Expected tuple, got {type(args)}")
            if not isinstance(kwargs, dict):
                raise AssertionError(f"Expected dict, got {type(kwargs)}")
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph


@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int


@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: list[torch.Tensor] | None = None
) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    if args is not None:
        # Mark shape and dtype for each node (node.shape and node.dtype)
        ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return


@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> TensorMetadata:
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        raise RuntimeError(
            f"Node {node} has no tensor metadata associated with it! "
            f"Check that shape propagation has run."
        )

    return tensor_meta


@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, consider all parameters
    if node.op == "call_module":
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for _name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    # Assume for now if it's quantized then it's qint8 or quint8
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized(
            [], dtype=tensor_meta.dtype
        ).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)
<|endoftext|>from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


__all__ = [
    "default_matching",
    "extract_attrs_for_lowering",
    "lift_lowering_attrs_to_nodes",
]


# Matching method matches the attribute name of current version to the attribute name of `target_version`
@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """Default matching method"""
    return name


# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.
# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.
# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.
module_fetch_book: dict[type, tuple[int, list[str], Callable[[str, int], str]]] = {
    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),
    torch.nn.modules.conv.Conv2d: (
        1,
        [
            "weight",
            "bias",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
        ],
        default_matching,
    ),
    torch.nn.modules.batchnorm.BatchNorm2d: (
        2,
        ["weight", "bias", "running_mean", "running_var", "eps"],
        default_matching,
    ),
    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),
    torch.nn.modules.pooling.MaxPool2d: (
        1,
        ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
        default_matching,
    ),
    torch.nn.modules.activation.ReLU: (1, ["inplace"], default_matching),
}


@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    attrs_for_lowering: dict[str, Any] = {}
    attrs_for_lowering["name"] = torch.typename(mod)

    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(
                f"Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, "
                "please upgrade the module_fetch_book, open an issue and @842974287 "
                "or report a bug to AIACC team directly."
            )
        for attr in param_to_fetch:
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        raise RuntimeError(
            f"{torch.typename(mod)} is not in the module_fetch_book yet, "
            "please add it to the module_fetch_book, open an issue and @842974287 "
            "or report a bug to AIACC team directly."
        )
    return attrs_for_lowering


@compatibility(is_backward_compatible=False)
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module."""
    submodules = dict(fx_module.named_modules())

    for node in fx_module.graph.nodes:
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                lift_lowering_attrs_to_nodes(submodules[node.target])
            else:
                node.attrs_for_lowering = extract_attrs_for_lowering(
                    submodules[node.target]
                )
<|endoftext|>import dis
import functools
import logging
import sys
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING


# Import sympy and ShapeEnv during TYPE_CHECKING since importing sympy is slow
if TYPE_CHECKING:
    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx._compatibility import compatibility
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule


__all__ = ["insert_deferred_runtime_asserts"]

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code_verbose")


def _get_example_value(node: fx.Node) -> str | None:
    """
    Get the example value key for a node, since dynamo uses "example_value"
    while non-strict export uses "val.
    """
    if "example_value" in node.meta:
        return node.meta["example_value"]
    elif "val" in node.meta:
        return node.meta["val"]
    else:
        return None


def _get_sym_val(node: fx.Node) -> Optional["sympy.Expr"]:
    val = _get_example_value(node)
    if isinstance(val, py_sym_types):
        return val.node.expr
    return None


@compatibility(is_backward_compatible=True)
def insert_deferred_runtime_asserts(
    gm: GraphModule,
    shape_env: ShapeEnv,
    name: str,
    export: bool = False,
) -> None:
    """
    During tracing, we may have discovered that some data-dependent values
    had runtime assert on them; e.g., torch.empty(x.item()) induces a runtime
    that x.item() >= 0.  These asserts can happen unpredictably during fake
    tensor propagation, so we cannot conveniently insert them into the FX graph
    when they occur.  Instead, we accumulate them in the ShapeEnv, and in this
    pass insert them into the graph as proper tests.

    This pass also deduplicates size-related computation, CSE-ing ops that produce
    symbolic values and/or are involved in runtime asserts. Additionally, shape calls
    (size/stride/storage_offset) are turned into compute on input sizes if possible,
    allowing intermediate tensors to be freed earlier. For example, here dynamo will
    DCE the cat and repeat calls:

        z = torch.cat([x, x], dim=0)  # 2*s0
        w = z.repeat(y.shape[0])  # 2*s0*s1
        _w = w.shape[0]
        # something with _w, but not w ...

        # turns into ->
        _w0 = 2 * s0
        _w = _w0 * s1

        # where s0, s1 are either SymInt graph inputs, or the result of added size calls

    Redundant compiler-generated asserts that assert the same expression, and
    redundant constrain_range calls are also deduplicated. Additionally, because
    single-symbol bound checks (e.g. u0 >= 0, u0 <= 5) accumulate information in
    the ShapeEnv, the ShapeEnv contains min/max bounds for each symbol, and we
    add bound checks at the end of this pass when an equivalent explicit assert
    is not already present in the graph.
    """

    # Import sympy locally
    import sympy

    from torch._export.passes._node_metadata_hook import _set_node_metadata_hook
    from torch.fx.experimental.symbolic_shapes import (
        _get_placeholder_expr,
        _has_uninterpretable_sympy_function,
        free_symbols,
        resolve_unbacked_bindings,
        RuntimeAssert,
    )
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.reference import (
        OptimizedPythonReferenceAnalysis,
        PythonReferenceAnalysis,
    )
    from torch.utils._sympy.value_ranges import ValueRanges

    # TODO: Request simplification on runtime asserts before emitting them
    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()
    graph = gm.graph
    tracer = fx.proxy.GraphAppendingTracer(graph)
    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # We are going to mutate the dict
    expr_to_proxy: dict[sympy.Expr, fx.Proxy] = {}
    placeholders = set()
    first_non_placeholder = None
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break
        else:
            placeholders.add(node)

    def _is_intermediate_tensor_sym_call(node: fx.Node) -> bool:
        """
        If a size/stride/storage offset call on an intermediate tensor,
        we can try to compute the value from input shapes instead.
        """
        return (
            (val := _get_sym_val(node)) is not None
            and not isinstance(val, sympy.Number)
            # this holds back from reifying anything in torch.utils._sympy.functions.py that's unsupported
            and not _has_uninterpretable_sympy_function(val)
            and any(
                isinstance(arg, fx.Node)
                and isinstance(_get_example_value(arg), (torch.Tensor, torch.Size))
                and arg.op != "placeholder"
                for arg in node.args
            )
        )

    # Figure out what key to use, val or example_value
    val_key = "val"
    for node in graph.nodes:
        if "example_value" in node.meta:
            val_key = "example_value"
            break
        elif "val" in node.meta:
            break

    # Note: DO NOT register one _set_node_metadata_hook(_node_metadata_hook)
    # for each nodes in the graph.
    # _set_node_metadata_hook is expensive and this can cause compile
    # time to regress significantly.
    def _node_metadata_hook(
        node: torch.fx.Node,
        stack_trace: str | None = None,
        nn_module_stack: dict[str, Any] | None = None,
        custom: dict[str, Any] | None = None,
        skip_val: bool = False,
    ) -> None:
        if not skip_val:
            fake_args = pytree.tree_map(
                lambda arg: (
                    _get_example_value(arg) if isinstance(arg, torch.fx.Node) else arg
                ),
                node.args,
            )
            try:
                target = node.target
                if node.op == "call_method":
                    if not isinstance(node.target, str):
                        raise AssertionError(
                            f"Expected str target, got {type(node.target)}"
                        )
                    target = getattr(fake_args[0], node.target)
                    fake_args = fake_args[1:]
                node.meta[val_key] = target(*fake_args)  # type: ignore[operator]
            except NotImplementedError:
                # This can happen when attempting to reify a symbol with an unsupported call_function node,
                # e.g. with NestedTensors + sym_size.int via match_symbol().
                # This seems to be fine, as the node gets CSE'd and deleted later in favor of a SymInt graph input.
                pass
        if stack_trace is not None:
            node.meta["stack_trace"] = stack_trace
        if nn_module_stack is not None:
            node.meta["nn_module_stack"] = nn_module_stack
        if custom is not None:
            node.meta["custom"] = custom

    # Track asserts/checks we've added
    added_asserts: set[sympy.Expr] = set()
    constrained_unbacked_symbols: set[sympy.Symbol] = set()

    Analysis = PythonReferenceAnalysis if export else OptimizedPythonReferenceAnalysis

    def _sympy_interp(
        expr_to_proxy: dict[sympy.Expr, fx.Proxy], expr: sympy.Expr
    ) -> fx.Proxy:
        # Lower a sympy expression into the graph and return the resulting
        # Proxy (use .node to get the FX node representing ``expr``).
        # sympy_interp() with hash consing
        from sympy import Integer, Number, Symbol
        from sympy.logic.boolalg import BooleanAtom

        from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp

        # hash cons
        if expr in expr_to_proxy:
            return expr_to_proxy[expr]
        # base cases, don't cache
        if isinstance(expr, (Integer, Number, Symbol, BooleanAtom)):
            return sympy_interp(Analysis, expr_to_proxy, expr)

        # hash cons on arguments, run expr handler
        expr_to_proxy[expr] = _run_sympy_handler(
            Analysis,
            [_sympy_interp(expr_to_proxy, arg) for arg in expr.args],
            expr,
        )
        return expr_to_proxy[expr]

    def _is_bound_expr_for_symbol(expr: "sympy.Expr") -> bool:
        # This is probably unnecessary, but since torch._check() calls for single-symbol bounds
        # like u0 >= 0, 10 >= u0 accumulate range info in the ShapeEnv, we designate these calls as redundant
        # and instead add 2 runtime asserts at the end of this pass, if the min/max bounds are non-trivial.
        if len(expr.args) != 2 or expr.func not in (sympy.LessThan, sympy.GreaterThan):
            return False
        lhs, rhs = expr.args
        return (isinstance(lhs, sympy.Symbol) and isinstance(rhs, sympy.Number)) or (
            isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number)
        )

    assert_targets = (
        torch._check,
        torch._assert_scalar,
        torch.ops.aten._assert_scalar.default,
    )
    preserve_runtime_assert_key = "_fx_runtime_assert_preserve"

    def _assertion_condition(node: fx.Node) -> Any:
        if node.args:
            return node.args[0]
        return node.kwargs.get("cond", node.kwargs.get("self"))

    def _assertion_message(
        node: fx.Node, cond: Any, assert_expr: Optional["sympy.Expr"]
    ) -> str:
        def static_string_from_callable(fn: Callable[[], object]) -> str | None:
            code = getattr(fn, "__code__", None)
            if (
                code is None
                or code.co_argcount != 0
                or code.co_posonlyargcount != 0
                or code.co_kwonlyargcount != 0
            ):
                return None

            # torch._check message callables are lazy. Preserve common constant
            # string lambdas without executing arbitrary user code at compile time.
            ignored_opnames = {
                "CACHE",
                "COPY_FREE_VARS",
                "EXTENDED_ARG",
                "NOP",
                "RESUME",
            }
            instructions = [
                inst
                for inst in dis.get_instructions(fn)
                if inst.opname not in ignored_opnames
            ]
            if (
                len(instructions) == 1
                and instructions[0].opname == "RETURN_CONST"
                and isinstance(instructions[0].argval, str)
            ):
                return instructions[0].argval
            if (
                len(instructions) == 2
                and instructions[0].opname == "LOAD_CONST"
                and isinstance(instructions[0].argval, str)
                and instructions[1].opname == "RETURN_VALUE"
            ):
                return instructions[0].argval
            return None

        msg = None
        if len(node.args) > 1:
            msg = node.args[1]
        elif node.target is torch._check:
            msg = node.kwargs.get("message")
        else:
            msg = node.kwargs.get("assert_msg")

        if isinstance(msg, str):
            return msg
        if isinstance(msg, fx.Node) and msg.op == "get_attr":
            attr = gm
            if not isinstance(msg.target, str):
                raise AssertionError(f"Expected str target, got {type(msg.target)}")
            for atom in msg.target.split("."):
                attr = getattr(attr, atom)
            if callable(attr):
                static_msg = static_string_from_callable(attr)
                if static_msg is not None:
                    return static_msg
            if isinstance(attr, str):
                return attr
        return f"Runtime assertion failed for expression {assert_expr} on node '{cond}'"

    def _is_user_runtime_assert(node: fx.Node) -> bool:
        if node.target in (torch._check, torch._assert_scalar):
            return True
        return (
            node.target is torch.ops.aten._assert_scalar.default
            and node.meta.get(preserve_runtime_assert_key) is True
        )

    def _replace_with_aten_assert(
        node: fx.Node, cond: Any, assert_expr: Optional["sympy.Expr"]
    ) -> None:
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten._assert_scalar.default,
                (cond, _assertion_message(node, cond, assert_expr)),
            )
            new_node.meta.update(node.meta)
            new_node.meta[preserve_runtime_assert_key] = True
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    user_assert_exprs: set[sympy.Expr] = set()
    for node in graph.nodes:
        if node.target in assert_targets and _is_user_runtime_assert(node):
            cond = _assertion_condition(node)
            if isinstance(cond, fx.Node):
                assert_expr = _get_sym_val(cond)
                if assert_expr is not None:
                    user_assert_exprs.add(assert_expr)

    def add_runtime_asserts(ras: list[RuntimeAssert]) -> None:
        for ra in ras:
            if ra.expr in user_assert_exprs:
                added_asserts.add(ra.expr)
                continue
            if (
                # redundant
                ra.expr in added_asserts
                # if we've already added a constrain_range call for this symbol,
                # then single-symbol bound asserts like u0 >= 0, u0 <= 5 are redundant.
                or (
                    len(ra.expr.free_symbols) == 1
                    and next(iter(ra.expr.free_symbols)) in constrained_unbacked_symbols
                    and _is_bound_expr_for_symbol(ra.expr)
                )
                # don't try to reify sympy functions we can't turn into FX nodes
                or _has_uninterpretable_sympy_function(ra.expr)
            ):
                continue

            log.debug("inserting runtime assert %s", ra.expr)
            # Need to process ALL free symbols, not just unbacked ones
            fvs = free_symbols(ra.expr)
            missing = fvs - expr_to_proxy.keys()
            if missing:
                i1 = min(missing, key=str)
                # TODO: Remove relaxing assert on unbacked_symint https://github.com/pytorch/pytorch/issues/119689
                # assert shape_env.is_unbacked_symint(i1), i1
                ras_by_symbol.setdefault(i1, []).append(ra)
            else:
                # Convert the sympy expression into a sequence of FX
                # nodes
                with _set_node_metadata_hook(
                    gm,
                    functools.partial(
                        _node_metadata_hook,
                        stack_trace=node.meta.get("stack_trace"),
                        nn_module_stack=node.meta.get("nn_module_stack"),
                        # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                        # as the input node to the assertion
                        custom=node.meta.get("custom"),
                    ),
                ):
                    res = _sympy_interp(expr_to_proxy, ra.expr).node

                    graph.call_function(
                        torch.ops.aten._assert_scalar.default,
                        # TODO: use ra.msg here, but it's pretty
                        # useless right now
                        (
                            res,
                            f"Runtime assertion failed for expression {ra.expr} on node '{res}'",
                        ),
                    )
                added_asserts.add(ra.expr)

    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        # Placeholders can match symbols, but when we destructure them
        # with size we have to make sure we insert the nodes after all
        # the placeholders
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else first_non_placeholder
        ):
            # Unfortunately, this logic still must remain because manual
            # make_fx calls may not explicitly bind all symbolic ints as
            # arguments to the function, so we must infer it from the other
            # arguments
            if (
                node in placeholders
                and (example_value := _get_example_value(node)) is not None
            ):

                def match_symbol(symint: object, cb: Callable[[], fx.Node]) -> None:
                    if (
                        isinstance(symint, torch.SymInt)
                        and isinstance(symint.node, SymNode)
                        and isinstance(
                            s := _get_placeholder_expr(symint.node), sympy.Symbol
                        )
                        and s not in expr_to_proxy
                    ):
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[s] = fx.Proxy(cb(), tracer=tracer)

                        log.debug("expr_to_proxy[%s] = %s", s, expr_to_proxy[s])

                match_symbol(example_value, lambda: node)

                if isinstance(t := example_value, torch.Tensor):
                    for i, s in enumerate(t.size()):
                        match_symbol(
                            s,
                            lambda: graph.create_size_node(node, i),
                        )
                    if not is_sparse_any(t):
                        for i, s in enumerate(t.stride()):
                            match_symbol(
                                s,
                                lambda: graph.create_stride_node(node, i),
                            )
                        match_symbol(
                            t.storage_offset(),
                            lambda: graph.create_storage_offset_node(node),
                        )

            # Handle asserts that aren't associated with any symbol.  This
            # doesn't really have to be in the loop as it will only run once,
            # it just needs to happen right after the placeholders.
            # insert this after placeholders & added sym nodes, and before non-placeholders.
            if node == first_non_placeholder:
                add_runtime_asserts(ras_by_symbol.pop(None, []))  # type: ignore[call-overload]

            # deduplicate asserts already present in graph, and remove trivial asserts
            if node.target in assert_targets:
                cond = _assertion_condition(node)
                assert_expr = _get_sym_val(cond) if isinstance(cond, fx.Node) else None
                if cond == True:  # noqa: E712
                    arg = cond
                    gm.graph.erase_node(node)
                    if isinstance(arg, fx.Node) and not arg.users:
                        gm.graph.erase_node(arg)
                else:
                    if _is_user_runtime_assert(node):
                        if node.target is not torch.ops.aten._assert_scalar.default:
                            _replace_with_aten_assert(node, cond, assert_expr)
                        if assert_expr is not None:
                            added_asserts.add(assert_expr)
                    elif (
                        assert_expr is not None
                        and assert_expr in expr_to_proxy
                        and assert_expr in added_asserts
                    ):
                        arg = cond
                        gm.graph.erase_node(node)
                        if isinstance(arg, fx.Node) and not arg.users:
                            gm.graph.erase_node(arg)
                    else:
                        if assert_expr is not None:
                            added_asserts.add(assert_expr)

            # hash cons, replace function calls that return torch.SymInts with direct references to
            # FX nodes built up to reify the sympy expression.
            if (
                node.op != "placeholder"
                and (sym_expr := _get_sym_val(node)) is not None
            ):
                # this guards against deleting calls like item() that produce new untracked symbols
                def has_new_untracked_symbols() -> bool:
                    # pyrefly: ignore [missing-attribute]
                    for symbol in sym_expr.free_symbols:
                        if symbol not in expr_to_proxy:
                            return True
                    return False

                # this guards against deleting calls that produce unbacked bindings we haven't yet seen.
                # in this case looking at sym_expr.free_symbols might not be enough, if the example value has a hint
                # (is backed), but produces an unbacked symbol. In this case keep the node alive.
                resolved_unbacked_bindings = resolve_unbacked_bindings(
                    shape_env, node.meta.get("unbacked_bindings", {})
                )

                def has_new_unbacked_bindings() -> bool:
                    if resolved_unbacked_bindings is None:
                        raise AssertionError("resolved_unbacked_bindings is None")
                    for key in resolved_unbacked_bindings:
                        if key not in expr_to_proxy:
                            return True
                    return False

                # maybe re-reify expression, replace current node
                if (
                    sym_expr in expr_to_proxy
                    or (  # example value is redundant
                        _is_intermediate_tensor_sym_call(node)
                        # shape call on intermediate tensor, turn into computation on input shapes
                        and not has_new_untracked_symbols()
                    )
                ) and not has_new_unbacked_bindings():
                    if _is_intermediate_tensor_sym_call(
                        node
                    ):  # reify from input shapes
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[sym_expr] = _sympy_interp(
                                expr_to_proxy,
                                sym_expr,
                            )  # type: ignore[arg-type]
                        # won't try DCE-ing tensor compute here
                    hash_node = expr_to_proxy[sym_expr].node  # type: ignore[arg-type]
                    node.replace_all_uses_with(hash_node)
                    gm.graph.erase_node(node)
                    log.debug(
                        "CSE node %s -> %s for expr %s",
                        node,
                        hash_node,
                        sym_expr,
                    )

                # store node in hash cons, don't delete/replace

                elif sym_expr not in expr_to_proxy and not isinstance(
                    sym_expr,
                    (sympy.Number, sympy.logic.boolalg.BooleanAtom),
                ):  # don't hash cons primitives
                    expr_to_proxy[sym_expr] = fx.Proxy(node, tracer=tracer)  # type: ignore[arg-type]

            # We add sym_constrain_range calls for symbols later in any case if they're size-like or range-constrained,
            # so calls before that are redundant.
            if node.target in (
                torch.ops.aten.sym_constrain_range.default,
                torch.ops.aten.sym_constrain_range_for_size.default,
            ):
                gm.graph.erase_node(node)

            defs = []

            # AOTAutograd will create new symbols as the unbacked_bindings keys, which PropagateSymInts will set as
            # equivalent, but the refinement calls we perform in this pass may struggle with associating the two.
            # More concretely, when re-exporting/tracing, constraining only the new symbol may not communicate enough
            # information about the old symbol when we re-export, raising errors on data-dependent guards.
            # Call resolve_unbacked_bindings() to get the original symbol if present, otherwise we take it as is.
            if unbacked_bindings := resolve_unbacked_bindings(
                shape_env, node.meta.get("unbacked_bindings")
            ):
                for s, keypath in unbacked_bindings.items():
                    defs.append(s)

                    # TODO: some CSE when generating these nodes can probably
                    # help reduce graph size and improve compile time
                    def _lower_symint_divisor(d: torch.SymInt) -> fx.Node | int:
                        p = _sympy_interp(expr_to_proxy, d.node.expr)
                        return p.node if isinstance(p, fx.Proxy) else p

                    if s not in expr_to_proxy:
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[s] = fx.Proxy(
                                graph._resolve_unbacked_binding(
                                    node,
                                    tuple(keypath),
                                    lower_symint=_lower_symint_divisor,
                                ),
                                tracer=tracer,
                            )
                        log.debug("expr_to_proxy[%s] = %s", s, expr_to_proxy[s])

            for i0 in defs:
                ras = ras_by_symbol.pop(i0, [])
                # Before we perform any asserts, first apply range
                # refinement.  This is important, because if we are going
                # to retrace the graph (and we typically are if we send
                # the graph to AOTAutograd), we need to make sure we apply
                # range refinement (ala _check_is_size) first, BEFORE we
                # run any of the asserts.  Otherwise, we may decide to
                # perform substitutions based on the asserts which we then
                # can't back out, because value ranges can only be applied
                # to asserts.)
                #
                # A perhaps better long term plan is to avoid this order
                # dependence by making it possible to refine ranges on
                # arbitrary expressions, not just symbols.  But it is not
                # so easy to make use of this information, see
                # https://twitter.com/ezyang/status/1745801370299482492
                # We actually made an attempt at this in
                # https://github.com/pytorch/pytorch/pull/119043
                # which didn't work.
                #
                # Another idea for how to do this:
                # - Have bound_sympy be the source of truth of the ranges of any expression
                # - Cache intermediate results for every subexpression of bound_sympy
                # - This cache should be possible to edit to refine ranges
                #
                # One issue with this proposal is that if
                # we have a bound on 2x, we are not going to be able to
                # apply it for 4x.  Similarly, we may have bounds for an
                # equivalent expression that we are not applying because
                # it's not a perfect match (e.g. x < y vs y > x)".
                #
                # The first issue we already have it and it's impossible
                # to solve in general, so any implementation on a best
                # effort basis should do.
                #
                # The second issue is a preexisting one. It can be mitigated
                # with a normalization algorithm. In general, it may also
                # be on a best effort basis, but since our grammar is not
                # terribly difficult, chances are we could even fully
                # normalize SymPy expressions... who knows.
                if i0 in constrained_unbacked_symbols:
                    continue  # constrain symbol just once

                vr = shape_env.var_to_range[i0]
                if vr.is_int and vr.upper == sys.maxsize - 1:
                    # treat upper bound == sys.maxsize - 1 for int symbols as +oo
                    # to avoid redundant runtime assert
                    vr = ValueRanges(vr.lower, int_oo)
                if not shape_env._default_unspecified_value_range().issubset(vr):
                    # The runtime range is constrained, so add a runtime
                    # assert and also explicitly refine the range
                    # (refinement should not be necessary once runtime
                    # asserts cause refinement, but that's NYI)
                    def convert(s: Any) -> int | float | None:
                        if s in (sympy.oo, -sympy.oo, int_oo, -int_oo):
                            return None
                        try:
                            if vr.is_int:
                                return int(s)
                            if vr.is_float:
                                return float(s)
                        except (TypeError, OverflowError):
                            return None
                        return None

                    with _set_node_metadata_hook(
                        gm,
                        functools.partial(
                            _node_metadata_hook,
                            stack_trace=node.meta.get("stack_trace"),
                            nn_module_stack=node.meta.get("nn_module_stack"),
                            # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                            # as the input node to the assertion
                            custom=node.meta.get("custom"),
                        ),
                    ):
                        if (min_val := convert(vr.lower)) is not None:
                            ge_expr = i0 >= min_val
                            if ge_expr in user_assert_exprs:
                                added_asserts.add(ge_expr)
                            else:
                                ge = _sympy_interp(expr_to_proxy, ge_expr).node
                                graph.call_function(
                                    torch.ops.aten._assert_scalar.default,
                                    (
                                        ge,
                                        f"Runtime assertion failed for expression {ge_expr} on node '{ge}'",
                                    ),
                                )
                                added_asserts.add(ge_expr)
                        if (max_val := convert(vr.upper)) is not None:
                            le_expr = i0 <= max_val
                            if le_expr in user_assert_exprs:
                                added_asserts.add(le_expr)
                            else:
                                le = _sympy_interp(expr_to_proxy, le_expr).node
                                graph.call_function(
                                    torch.ops.aten._assert_scalar.default,
                                    (
                                        le,
                                        f"Runtime assertion failed for expression {le_expr} on node '{le}'",
                                    ),
                                )
                                added_asserts.add(le_expr)

                constrained_unbacked_symbols.add(i0)
                add_runtime_asserts(ras)

    # delete unused reified symbols
    for expr, proxy in expr_to_proxy.items():
        if (
            isinstance(expr, sympy.Symbol)
            and proxy.node.op != "placeholder"  # keep placeholders intact
            and not proxy.node.users
        ):
            log.debug("deleting unused reified symbol for %s", expr)
            gm.graph.erase_node(proxy.node)
<|endoftext|>import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg

from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    Names,
    NodeList,
    NodeSet,
    TensorOrTensors,
    Tensors,
)


__all__ = [
    "FxNetMinimizerBadModuleError",
    "FxNetMinimizerRunFuncError",
    "FxNetMinimizerResultMismatchError",
]

_LOGGER = logging.getLogger(__name__)


@compatibility(is_backward_compatible=False)
class FxNetMinimizerBadModuleError(Exception):
    """
    Raised if failed to split out a minimize module
    """


@compatibility(is_backward_compatible=False)
class FxNetMinimizerRunFuncError(Exception):
    """
    Raised if error occurs during run_a or run_b functions
    """


@compatibility(is_backward_compatible=False)
class FxNetMinimizerResultMismatchError(Exception):
    """
    Raised if comparing function thinks the results are mismatching.
    """


@dataclass
class _MinimizerSettingBase:
    """
    Args:
    `accumulate_error`: Instead of using a's input for both converted module to verify
    , use the previous outputs of each converted module as input to accumulate the
    errors.

    `traverse_method`: "sequential" or "binary" or "accumulate"
    Determine the way of traverse the nodes in FX module.

    `find_all`: Minimizer will go through the entire model and return all problematic nodes.

    `return_intermediate`: If true, when using `run_nodes()` function to run the
    model, intermediate results of all the ops will be returned as output.

    `all_outputs`: If true, when using `_run_and_compare()` function,
    all the output nodes in the subgraph will be used for comparison.
    """

    accumulate_error: bool = False
    traverse_method: str = "sequential"
    find_all: bool = False
    return_intermediate: bool = False
    all_outputs: bool = False

    def __str__(self) -> str:
        settings_str = "FX Minimizer Settings:\n"

        for k, v in vars(self).items():
            settings_str += f"\t{k}: {v}\n"

        return settings_str


class _MinimizerBase:
    """
    This class is used to automatically find problematic nodes in a model. It takes a FX
    graphmodule and generate some submodules while traverse the graph. Then two functions
    `run_a` and `run_b` will be used to run the same submodule and a function `compare_fn`
    will be used to compare the results.

    Currently we provide two ways to traverse the graph and generate submodules.
        1. Sequential traversal: this will traverse the graph node by node and generate
           one submodule with one single node.
        2. Binary searching: this will do a binary search style traversal on the graph.

    For internal Users, a guide can be found here https://fb.quip.com/HDtuAgiKGfkP.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[
            [TensorOrTensors, TensorOrTensors, Names], tuple[float, bool]
        ],
        settings: _MinimizerSettingBase,
        module_exporter: Callable[[Tensors, torch.fx.GraphModule, str], None]
        | None = None,
        exclusion_fn: Callable[[NodeList, int, int], None] | None = None,
    ) -> None:
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")

        self.module = module
        self.sample_input = sample_input
        self.compare_fn = compare_fn
        self.module_exporter = module_exporter
        self.settings = settings
        self.exclusion_fn = exclusion_fn

        # Stores outputs of run_a function
        self.a_outputs: dict[str, Any] = {}

        # Stores outputs of run_b function
        self.b_outputs: dict[str, Any] = {}

        # Stores the results of compare_fn
        self.results: dict[Any, Any] = {}

        # Stores the report for the runs
        self.reports: list[list[str]] = []

        # Current iteration
        self.iteration: int = 0

        callable_nodes = {
            node for node in self.module.graph.nodes if node.op in CALLABLE_NODE_OPS
        }
        self.run_shape_prop()
        self.fusions = FxNetAccFusionsFinder(self.module, callable_nodes)()

        # Check if number of input in sample_input matches the number of placeholders
        placeholders = [
            node.name for node in self.module.graph.nodes if node.op == "placeholder"
        ]
        if len(placeholders) != len(self.sample_input):
            raise AssertionError(
                f"Placeholder count ({len(placeholders)}) does not match "
                f"sample_input count ({len(self.sample_input)})"
            )

        # Store sample_input
        for i, name in enumerate(placeholders):
            self.a_outputs[name] = sample_input[i]
            self.b_outputs[name] = sample_input[i]

    def run_shape_prop(self) -> None:
        """
        Helper function to run shape propagation on module. Can be overridden by
        subclasses for custom shape propagation logic.
        """
        ShapeProp(self.module).propagate(*self.sample_input)

    def run_a(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1
    ) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_b().
        """
        raise RuntimeError("run_a() is not implemented.")

    def run_b(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1
    ) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_a().
        """
        raise RuntimeError("run_b() is not implemented.")

    def _store_outputs(
        self,
        a_result: TensorOrTensors,
        b_result: TensorOrTensors,
        submodule: torch.fx.GraphModule,
    ) -> None:
        """
        Store the outputs of self.run_a() and self.run_b() into self.a_outputs and
        self.b_outputs, so that we can use them when executing preceding nodes that
        use those outputs as inputs.

        Args:
            a_result: Output of self.run_a(). Could be a tensor or tensors.
            b_result: Output of self.run_b(). Could be a tensor or tensors.
            submodule: The module that generates a_result and b_result.
        """
        output_node = next(
            node for node in submodule.graph.nodes if node.op == "output"
        )

        # Only one output
        if isinstance(output_node.args[0], torch.fx.Node):
            self.a_outputs[output_node.args[0].name] = a_result
            self.b_outputs[output_node.args[0].name] = b_result
        # Multiple outputs
        else:
            for i, arg in enumerate(output_node.args[0]):
                self.a_outputs[arg.name] = a_result[i]
                self.b_outputs[arg.name] = b_result[i]

    def _get_submod_inputs(
        self, main_module: torch.fx.GraphModule, submod_path: str
    ) -> tuple[Tensors, Tensors]:
        """
        Try get submodule inputs from stored outputs. If not found then use
        torch_glow.get_submod_inputs to get the inputs.

        If accumulate_error is False, use a_input for run_a() and run_b()
        otherwise use a_input for run_a and b_input for run_b.

        Args:
            main_module: Top-level fx module.
            submod_path: Path to the submodule we want to run and compare results.

        Returns:
            a_input: List of tensor(s) that will be used by run_a() as submodule inputs.
            b_input: List of tensor(s) that will be used by run_b() as submodule inputs.
        """
        a_input = []
        b_input = []
        submodule = getattr(main_module, submod_path)
        placeholders = [
            node.name for node in submodule.graph.nodes if node.op == "placeholder"
        ]

        # If all placeholder can be found in stored outputs, use stored
        # outputs as inputs. Otherwise, use `torch_glow.get_submod_inputs`
        # to get the inputs.
        if set(placeholders) <= self.a_outputs.keys():
            for name in placeholders:
                a_input.append(self.a_outputs[name])
                b_input.append(self.b_outputs[name])
        else:
            if self.settings.accumulate_error:
                print(f"Can't find previous stored outputs named {placeholders}!")

            def get_inputs(self: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                nonlocal a_input
                a_input = inputs

            # Use forward hook to get the inputs to the submodule
            handle = submodule.register_forward_pre_hook(get_inputs)
            main_module(*self.sample_input)
            handle.remove()

            b_input = a_input

        if not self.settings.accumulate_error:
            return a_input, a_input

        return a_input, b_input

    def _tag_nodes(self, selected_nodes: NodeSet) -> None:
        """
        Tag selected nodes with tag "minimize". Nodes with the same tags will
        be split to the same submodule afterwards.

        Args:
            selected_nodes: Nodes that we want to minimize. We will tag those nodes
                with "minimize", all preceding nodes with "main_0" and all following
                nodes with "main_1".
        """
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            if node in selected_nodes:
                node.tag = "minimize"
            elif any(
                n.tag in {"minimize", "main_1"}
                for n in node.all_input_nodes
                if n.op in CALLABLE_NODE_OPS
            ):
                node.tag = "main_1"
            else:
                node.tag = "main_0"

    def _build_submodule(self, nodes: NodeSet) -> tuple[torch.fx.GraphModule, str]:
        """
        Split self.module so that one submodule consists of `nodes` and only `nodes`.

        Args:
            nodes: Nodes that we want to include in the minimize submodule.

        Returns:
            split_module (torch.fx.GraphModule): the module after split.
            submodule_name (str): the name of the submodule that consists of `nodes`.
        """
        # Color provided nodes
        self._tag_nodes(nodes)

        # Split module based on coloring
        split_module = split_by_tags(self.module, ["main_0", "minimize", "main_1"])

        # Find submodule containing colored nodes
        submodule_name: str = ""
        for child_name, _ in split_module.named_children():  # type: ignore[union-attr]
            # Skip submodules we're not interested in at the moment
            if "minimize" not in child_name:
                continue

            if submodule_name == "":
                submodule_name = child_name
            else:
                raise FxNetMinimizerBadModuleError(
                    f"Expected only one minimize submodule with nodes {nodes}"
                )

        if submodule_name == "":
            raise FxNetMinimizerBadModuleError(
                f"Minimize submodule was not found with nodes {nodes}"
            )

        return split_module, submodule_name  # type: ignore[return-value]

    def _run_and_compare(
        self,
        split_module: torch.fx.GraphModule,
        submod_name: str,
        output_names: Names,
        report_idx: int = -1,
    ) -> None:
        """
        Run the submodule in `split_module` that has name `submod_name`
        using `self.run_a` and `self.run_b` and compare their results.

        Args:
            split_module: Main module that contains the minimize submodule.
            submod_name: Name of the minimize submodule.
            output_names: Names of the node we want to output. If None, we
                will use the original output.
        """
        submodule = getattr(split_module, submod_name)
        a_input, b_input = self._get_submod_inputs(split_module, submod_name)

        if len(self.reports) == 0:
            self.reports.append([])
            self.iteration = 1

        report = self.reports[report_idx if report_idx >= 0 else self.iteration - 1]
        report.append("Run and compare ...")

        if output_names and not self.settings.all_outputs:
            output_nodes: NodeList = []
            for node in submodule.graph.nodes:
                if node.op == "output":
                    submodule.graph.erase_node(node)

                if node.name in output_names:
                    output_nodes.append(node)

            submodule.graph.output(
                output_nodes[0] if len(output_nodes) == 1 else tuple(output_nodes)
            )
            submodule.graph.lint()
            submodule.recompile()

        # Use name of args in output node as key to store comparison result
        for node in submodule.graph.nodes:
            if node.op == "output":
                result_key = map_arg(node.args, lambda x: x.name)

        try:
            a_result = self.run_a(submodule, a_input, report_idx)
            b_result = self.run_b(submodule, b_input, report_idx)
            self._store_outputs(a_result, b_result, submodule)
        except Exception as e:
            report.append(f"Exception raised when running {submod_name}: {e}")
            raise FxNetMinimizerRunFuncError(  # noqa: B904
                f"Exception raised when running {submod_name}: {e}"
            )

        # Compare results
        names: Names = output_names
        if output_names is None:
            names = [str(v) for v in result_key]  # type: ignore[possibly-undefined]

        numeric_result, bool_result = self.compare_fn(a_result, b_result, names)

        self.results[result_key] = numeric_result  # type: ignore[possibly-undefined]
        report.append(f"Numerical accuracy = {numeric_result}")
        if not bool_result:
            report.append(f"Result mismatch for {result_key}")  # type: ignore[possibly-undefined]
            if self.module_exporter:
                if isinstance(result_key, tuple):  # type: ignore[possibly-undefined]
                    # pyrefly: ignore [unbound-name]
                    result_key = result_key[-1]
                # If the result is still a tuple (happens in non-sequential mode),
                # we only use the first element as name.
                if isinstance(result_key, tuple):  # type: ignore[possibly-undefined]
                    # pyrefly: ignore [unbound-name]
                    result_key = str(result_key[0])
                # pyre-ignore[29]: not a function
                self.module_exporter(
                    a_input,
                    submodule,
                    # pyrefly: ignore [unbound-name]
                    result_key + "_cpu",
                )
                # pyre-ignore[29]: not a function
                self.module_exporter(
                    b_input,
                    submodule,
                    # pyrefly: ignore [unbound-name]
                    result_key + "_acc",
                )
            raise FxNetMinimizerResultMismatchError(f"Result mismatch for {result_key}")  # type: ignore[possibly-undefined]

    def _binary_search_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        Recursive binary search implementation.
        """
        culprits: NodeSet = set()
        nodes: NodeList = all_nodes[start_idx:end_idx]

        report: list[str] = []
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, start_idx, end_idx)
            if len(nodes) == 0:
                report = ["All nodes are excluded by user"]
                self.reports.append(report)
                return culprits

        first_node_name = nodes[0].name
        output_node_name = nodes[-1].name
        self.iteration += 1
        self.reports.append(report)
        report.append(f"Binary search iteration {self.iteration}")
        report.append(
            f"From node index {start_idx}:{first_node_name} to {end_idx - 1}:{output_node_name}. "
            f"Size of the interested node list is {len(nodes)}"
        )
        cur_nodes: NodeSet = set(nodes)

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [output_node_name])

        except (FxNetMinimizerRunFuncError, FxNetMinimizerResultMismatchError):
            if len(nodes) == 1:
                report.append(
                    f"This is the last node in the sub-module. "
                    f"Search in the current branch is successful with culprit = {cur_nodes}."
                )
                self.print_report(report)
                return cur_nodes

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            mid = len(nodes) // 2
            culprits = self._binary_search_impl(all_nodes, start_idx, start_idx + mid)

            if len(culprits) != 0 and not self.settings.find_all:
                return culprits

            culprits = self._binary_search_impl(all_nodes, start_idx + mid, end_idx)

            if len(culprits) == 0:
                report.append(
                    f"Further split and lowering found no errors. "
                    f"Unable to minimize the submodule with list of nodes: {nodes}"
                )
                self.print_report(report)

            return culprits
        else:
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()

    def _binary_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Binary search on `nodes` for culprit.
        """
        return self._binary_search_impl(nodes, 0, len(nodes))

    def _sequential_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Traverse `nodes` one by one and determine if any of them is a culprit.
        """
        culprits: NodeSet = set()

        for node in nodes:
            report: list[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Sequential traverse iteration {self.iteration}.")
            report.append(f"Visit node: {node.name}")

            _LOGGER.info("Visit node: %s", node.name)
            node_list: NodeList = [node]
            if self.exclusion_fn is not None:
                self.exclusion_fn(node_list, -1, -1)
                if len(node_list) == 0:
                    report.append(f"User exclusion : {node.name}")
                    self.print_report(report)
                    if not self.settings.find_all:
                        return culprits
                    else:
                        continue

            cur_nodes: NodeSet = {node}

            if node in self.fusions:
                cur_nodes = self.fusions[node]

            try:
                split_module, submod_name = self._build_submodule(cur_nodes)
                self._run_and_compare(split_module, submod_name, [node.name])
                self.print_report(report)
            except FxNetMinimizerResultMismatchError:
                culprits.add(node)
                report.append(f"Found culprit from numeric error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits
            except FxNetMinimizerRunFuncError:
                culprits.update(cur_nodes)
                report.append(f"Found culprit from run error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits

        return culprits

    def _block_traverse_impl(
        self, nodes: NodeList, start_idx: int, end_idx: int, find_last_node: bool
    ) -> int | None:
        """
        Recursive block search implementation.
        find_last_node: If True, search for the last node which result in numerics difference
        if False: find first node in sorted node list
        """
        report: list[str] = []

        mid = (start_idx + end_idx) // 2
        cur_nodes_list: NodeList = nodes[: mid + 1] if find_last_node else nodes[mid:]

        if self.exclusion_fn:
            self.exclusion_fn(cur_nodes_list, -1, -1)

        cur_nodes = set(cur_nodes_list)

        first_node_name = cur_nodes_list[0].name
        last_node_name = cur_nodes_list[-1].name
        target_node_name = last_node_name if find_last_node else first_node_name

        self.iteration += 1
        self.reports.append(report)
        report.extend(
            [
                "=" * 30,
                f"Block search iteration {self.iteration}",
            ]
        )
        report.extend(
            [
                f"Search for {'last' if find_last_node else 'first'} node in culprits",
                f"From node index {start_idx}:{nodes[start_idx].name} to {end_idx}:{nodes[end_idx].name}. ",
                f"Subgraph constructed by {first_node_name} to {last_node_name}",
                f"Targeting node: {target_node_name}",
                f"Size of the interested node list is {end_idx - start_idx + 1}",
            ]
        )
        report_idx = len(self.reports) - 1

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(
                split_module, submod_name, [last_node_name], report_idx
            )
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            report.append(
                f"Culprits found from node {first_node_name} to {last_node_name}."
            )

            if start_idx == mid == end_idx:
                report.extend(
                    [
                        "This is the last node in the sub-module. ",
                        "Search in the current branch is successful with node :",
                        f"{start_idx}, node name: {nodes[start_idx].name}.",
                    ]
                )
                self.print_report(report)
                return start_idx

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            if find_last_node:
                return self._block_traverse_impl(nodes, start_idx, mid, find_last_node)
            else:
                return self._block_traverse_impl(
                    nodes, mid + 1, end_idx, find_last_node
                )
        else:
            report.append(
                f"Culprits not found from node start to {mid}:{nodes[mid].name}."
            )

            if start_idx == mid == end_idx:
                # We did not find anything if the pointers have not moved
                if (start_idx == 0 and not find_last_node) or (
                    start_idx == len(nodes) - 1 and find_last_node
                ):
                    report.append(
                        f"At {'last' if find_last_node else 'first'} node, no culprits found."
                    )
                    self.print_report(report)
                    return None

                # Otherwise, we have converged on the border between discrepancy and valid
                return start_idx + (1 if find_last_node else -1)

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            if find_last_node:
                return self._block_traverse_impl(
                    nodes, mid + 1, end_idx, find_last_node
                )
            else:
                return self._block_traverse_impl(nodes, start_idx, mid, find_last_node)

    def _block_traverse(self, nodes: NodeList, find_last_node: bool | None) -> NodeSet:
        """
        Traverse topologically sorted node list
        Find minimum block (start_idx, end_idx) which contains the culprit
        1st pass: search for end_idx by finding the last node in culprit block
        where Numerical accuracy (0, end_idx) > threshold
        2nd pass: search for start_idx by finding the first node in culprit block
        where Numerical accuracy (start_idx, end_idx) < threshold
        Form minimum block by (start_idx - 1, end_idx)
        """
        culprits: NodeSet = set()
        first_node_name = nodes[0].name
        last_node_name = nodes[-1].name
        last_node_report = [f"Block search from {first_node_name} to {last_node_name}"]
        last_node_report.append("*" * 50)
        self.reports.append(last_node_report)

        start_idx = 0
        end_idx = len(nodes) - 1

        final_start_idx: int | None = start_idx
        final_end_idx: int | None = end_idx

        run_both = find_last_node is None

        # step 1: find (0, end_idx) of culprit block
        if run_both or find_last_node:
            last_node_report.append("Start searching for last node in culprit")
            self.print_report(last_node_report)
            final_end_idx = self._block_traverse_impl(nodes, start_idx, end_idx, True)

            if final_end_idx is None:
                last_node_report.append("No culprits found")
                self.print_report(last_node_report)
                return culprits

            last_node_report.extend(
                [
                    "Finish Pass 1",
                    f"Find end_idx = {final_end_idx}:{nodes[final_end_idx].name}",
                ]
            )
            self.print_report(last_node_report)

        # step 2: reduce culprit block to (start_idx, end_idx)
        if run_both or not find_last_node:
            first_node_report = ["Start searching for first node in culprit"]
            self.print_report(first_node_report)
            final_start_idx = self._block_traverse_impl(
                nodes[0 : end_idx + 1], start_idx, final_end_idx or end_idx, False
            )

            if final_start_idx is None:
                last_node_report.append("No culprits found")
                self.print_report(last_node_report)
                return culprits

            first_node_report.append("*" * 50)
            self.reports.append(first_node_report)
            first_node_report.extend(
                [
                    "Finish Pass 2",
                    f"Find start_idx = {final_start_idx}:{nodes[final_start_idx].name}",
                ]
            )
            self.print_report(first_node_report)

        # step 3: form module with minimum culprits. These indexes are guaranteed to exist
        range_start, range_end = cast(int, final_start_idx), cast(int, final_end_idx)
        culprits.update(nodes[range_start : range_end + 1])
        result_report = [
            f"Finish searching, found minimum block ({nodes[range_start]},{nodes[range_end]})"
        ]
        self.reports.append(result_report)
        self.print_report(result_report)
        return culprits

    def _defined_traverse(self, nodes: NodeList) -> NodeSet:
        """
        run user defined `nodes` and determine if it is a culprit.
        """
        culprits: NodeSet = set()
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, -1, -1)
        if len(nodes) == 0:
            report = ["All nodes are excluded by user"]
            self.reports.append(report)
            return culprits

        first_node_name = nodes[0].name
        output_node_name = nodes[-1].name
        report = [f"Defined graph from {first_node_name} to {output_node_name}"]
        cur_nodes: NodeSet = set(nodes)
        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [output_node_name])
            self.print_report(report)
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            report.append(f"Found culprit {cur_nodes}")
            self.print_report(report)
            return culprits

        return culprits

    def _accumulate_traverse(self, nodes: NodeList) -> NodeSet:
        culprits: NodeSet = set()
        nodes_to_run: NodeSet = set()

        # find_all is not supported for accumulate traversal because all the
        # ops run on NNPI. So we return after the first op that raises error.
        if self.settings.find_all:
            print("'Find All' mode is not supported in accumulate traversal.")
            return culprits

        for node in nodes:
            report: list[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Accumulate traverse iteration {self.iteration}.")

            nodes_to_run.add(node)

            node_name = node.name
            if node_name is not None and isinstance(node_name, tuple):
                node_name = node_name[0]
            if node_name is None or not isinstance(node_name, str):
                raise AssertionError(f"minimize: node_name: {node_name}")

            report.append(f"Add node: {node_name}")

            try:
                split_module, submod_name = self._build_submodule(nodes_to_run)
                self._run_and_compare(split_module, submod_name, [node_name])
                self.print_report(report)
            except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
                culprits.add(node)
                report.append(f"Found culprit {node}")
                self.print_report(report)
                return culprits

        return culprits

    def _skip_traverse_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        culprits: NodeSet = set()
        nodes: NodeList = all_nodes[start_idx:end_idx]
        cur_nodes: NodeSet = set(nodes)
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, start_idx, end_idx)
            cur_nodes = set(nodes)
        else:
            for node in nodes:
                if node in self.fusions:
                    cur_nodes.update(self.fusions[node])
        report: list[str] = []
        self.reports.append(report)
        self.iteration += 1
        report.append(f" Nodes block {self.iteration}.")
        report.append(
            f"From node index {start_idx} to {end_idx - 1}. "
            f"Size of the interested node list is {len(nodes)}"
        )

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [])
        except FxNetMinimizerResultMismatchError:
            culprits.update(cur_nodes)
            report.append(f"Found culprit from numeric error: {cur_nodes}")
            self.print_report(report)
            return culprits
        except FxNetMinimizerRunFuncError:
            culprits.update(cur_nodes)
            report.append(f"Found culprit from run error: {cur_nodes}")
            self.print_report(report)
            return culprits
        else:
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()

    def _skip_traverse(self, all_nodes: NodeList, skip_nodes: list[str]) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        start_idx = 0
        num_nodes = len(all_nodes)
        idx = 0
        culprits = set()
        while idx < num_nodes:
            node = all_nodes[idx]
            if node.name in skip_nodes:  # skip the node
                if idx > start_idx:
                    culprits = self._skip_traverse_impl(all_nodes, start_idx, idx)
                start_idx = idx + 1
            elif idx == num_nodes - 1 and start_idx <= idx:  # last node
                culprits = self._skip_traverse_impl(all_nodes, start_idx, idx + 1)
            idx += 1

        return culprits

    def _collect_nodes(self, start: str | None, end: str | None) -> NodeList:
        """
        Collect nodes in the model that between nodes with name of `start` and `end`.
        These two nodes are also included.
        """
        nodes: NodeList = []
        add_node = start is None

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            if node.name == start:
                add_node = True

            if add_node:
                nodes.append(node)

            if node.name == end:
                break

        return nodes

    def run_nodes(self, start: str | None = None, end: str | None = None) -> None:
        """
        Run part of the model from `start` node to `end` node. If `start` is None
        then we start from the beginning of the model. If `end` is None then we
        stop at the end of the model.

        Args:
            start: The name of the node which is the first node of the submodule
                we want to run. If set to None, then we'll start with the first
                node of the model.
            end: The name of the node which is the last node of the submodule we
                want to run. If set to None, we'll end with the last node of the
                model.
        """
        nodes = self._collect_nodes(start, end)
        cur_nodes = set(nodes)

        for node in nodes:
            if node in self.fusions:
                cur_nodes.update(self.fusions[node])

        output_names: list[str] = []
        if self.settings.return_intermediate:
            output_names = [node.name for node in nodes]

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, output_names)
        except (
            FxNetMinimizerRunFuncError,
            FxNetMinimizerResultMismatchError,
        ) as e:
            print(e)

    def print_report(self, report: list[str]) -> None:
        for i in range(len(report)):
            if i > 0:
                print(" . " + report[i])
            else:
                print(report[i])

    def print_reports(self) -> None:
        for report in self.reports:
            self.print_report(report)

    def minimize(
        self,
        start: str | None = None,
        end: str | None = None,
        skip_nodes: list[str] | None = None,
        find_last_node: bool | None = None,
    ) -> NodeSet:
        """
        Minimizing the model from node with name `start` to node with name `end` base
        on self.settings. Find culprits that causes FxNetMinimizerRunFuncError or
        FxNetMinimizerResultMismatchError errors.

        Args:
            start: The name of the node where we want to start minimizing. If set
                to None, then we'll start with the first node of the model.
            end: The name of the node where we want to terminate minimizing. If
                set to None, we'll end with the last node of the model.
            skip_nodes: The names of nodes where we want to skip during minimizing.
                It'll create subgraphs without these skip nodes under the hood.
                Only applicable in mode "skip".
            find_last_node: True if only last_node of a culprits is needed in mode "block".
                False if only the first_node of a culprits is needed.
                Only applicable in mode "block".

        Returns:
            nodes: A list of nodes that causes FxNetMinimizerRunFuncError or
                FxNetMinimizerResultMismatchError errors during minimizing.
        """

        print(self.settings)
        print(self.module.graph)

        nodes = self._collect_nodes(start, end)

        if self.settings.traverse_method == "sequential":
            return self._sequential_traverse(nodes)

        if self.settings.traverse_method == "binary":
            return self._binary_traverse(nodes)

        if self.settings.traverse_method == "accumulate":
            return self._accumulate_traverse(nodes)

        if self.settings.traverse_method == "skip":
            if skip_nodes is None:
                raise RuntimeError(
                    "'skip_nodes' can't be None when 'traverse_method' is 'skip'."
                )
            return self._skip_traverse(nodes, skip_nodes)

        if self.settings.traverse_method == "defined":
            return self._defined_traverse(nodes)

        if self.settings.traverse_method == "block":
            return self._block_traverse(nodes, find_last_node)

        raise RuntimeError(f"Unknown traverse method {self.settings.traverse_method}!")
<|endoftext|>import operator
import typing

import torch


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (:class:`torch.fx.Graph`): The graph to be annotated
    """
    for node in graph.nodes:
        if node.target is operator.getitem:
            sequence_node, index_node = node.args
            if not sequence_node.type:
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):
                parameterized_types = typing.get_args(sequence_node.type)
                if sequence_node.type._name == "Tuple":
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type._name == "List":
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # Generic Alias Type
            elif hasattr(sequence_node.type, "__origin__"):
                parameterized_types = typing.get_args(sequence_node.type)
                if sequence_node.type.__origin__ is tuple:
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type.__origin__ is list:
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):
                if sequence_node.type == torch.Tensor:
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__
                field_name = sequence_node.type._fields[index_node]
                node.type = sequence_node_field_types[field_name]
<|endoftext|>import copy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module

from .tools_common import CALLABLE_NODE_OPS, is_node_output_tensor, NodeList


if TYPE_CHECKING:
    from .splitter_base import Subgraph


__all__ = [
    "getattr_recursive",
    "setattr_recursive",
    "Component",
    "split_by_tags",
    "move_non_tensor_nodes_on_boundary",
]


@compatibility(is_backward_compatible=False)
def getattr_recursive(obj: object, name: str) -> Any:
    for layer in name.split("."):
        if isinstance(obj, torch.nn.ModuleList):
            if hasattr(obj, "_modules") and layer in obj._modules:
                obj = obj._modules[layer]
            else:
                return None
        elif hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


@compatibility(is_backward_compatible=False)
def setattr_recursive(obj: object, attr: str, value: object) -> None:
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)


@compatibility(is_backward_compatible=False)
@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """

    graph: torch.fx.Graph
    order: int
    name: str

    # Stores the placeholder nodes in `graph`.
    input_placeholders: list[torch.fx.Node] = field(default_factory=list)

    # Store the nodes in original graph that are placeholder in `graph`.
    orig_inputs: list[torch.fx.Node] = field(default_factory=list)

    # Store the nodes in original graph that are outputs in `graph`.
    orig_outputs: list[torch.fx.Node] = field(default_factory=list)

    # Mapping from get_attr node in original graph to get_attr node in `graph`.
    getattr_maps: dict[torch.fx.Node, torch.fx.Node] = field(default_factory=dict)
    constructor_args: list[str] = field(default_factory=list)
    gm: torch.fx.GraphModule | None = None


@compatibility(is_backward_compatible=False)
def split_by_tags(
    gm: torch.fx.GraphModule,
    tags: list[str],
    return_fqn_mapping: bool = False,
    return_tuple: bool = False,
    GraphModuleCls: type[torch.fx.GraphModule] = torch.fx.GraphModule,
) -> torch.fx.GraphModule | tuple[torch.fx.GraphModule, dict[str, str]]:
    """
    Splits a GraphModule using tags on its graph nodes. We honor the order of
    tags. For example, we have tags = ["a", "b", "c"], the function will create
    the initial submodules in the order of "a", "b", "c".

    To set a tag:

    .. code-block:: python

        gm.graph.nodes[idx].tag = "mytag"

    This will result in all nodes with the same tag being extracted and placed in their
    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder
    and output nodes are created when needed while get_attr nodes get copied to submodules
    where they are used.

    Given the following module def:

    .. code-block:: python

        class SimpleModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(...)
                self.linear2 = torch.nn.Linear(...)
                self.linear3 = torch.nn.Linear(...)

            def forward(self, in1, in2):
                r1 = self.linear1(in1)
                r2 = self.linear2(in2)
                r3 = torch.cat([r1, r2])
                return self.linear3(r3)

    Marking the node corresponding to in1 with the tag sc.REQUEST_ONLY.lower() results in the following split:

    ro:

    .. code-block:: python

        def forward(self, in1):
            self = self.root
            linear1 = self.linear1(in1)
            return linear1

    main:

    .. code-block:: python

        def forward(self, in2, linear1):
            self = self.root
            linear2 = self.linear2(in2)
            cat_1 = torch.cat([linear1, linear2])
            linear3 = self.linear3(cat_1)
            return linear3

    main:

    .. code-block:: python

        def forward(self, in1, in2):
            self = self.root
            ro_0 = self.ro_0(in1)
            main_1 = self.main_1(in2, ro_0)
            return main_1

    Returns:
        split_gm: torch fx graph after split

        orig_to_split_fqn_mapping: a map between the original fqn and the fqn
            after split for call_module and get_attr.
    """

    def flatten(x: torch.fx.node.Argument) -> NodeList:
        """
        Stores nodes in x to a list and returns the list.
        """
        r: NodeList = []
        map_arg(x, r.append)
        return r

    # Mapping from node in original module to node in created submodule.
    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}

    # Mapping from node in original module or created submodules to
    # corresponding component.
    node_to_component: dict[torch.fx.Node, Component] = {}

    # Mapping from tag to the corresponding component.
    tag_to_component: dict[str, Component] = {}

