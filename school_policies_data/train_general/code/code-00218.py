        # 1. foo of type float
        # 2. names_ages of type Dict[str, int]

    .. testcleanup::

        del AttributeModule
        del m

    Note: it's now preferred to instead use type annotations instead of `torch.jit.Attribute`:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.nn.Module):
            names: Dict[str, int]

            def __init__(self) -> None:
                super().__init__()
                self.names = {}

        m = AttributeModule()

    .. testcleanup::

        del AttributeModule
        del m

    Args:
        value: An initial value to be assigned to attribute.
        type: A Python type

    Returns:
        Returns `value`
"""


def _get_type_trace_db():
    # This is a private API. Use of this for external purposes is discouraged.
    return type_trace_db


# Gets a function from the name of a method on a type
def _get_function_from_type(cls, name):
    return getattr(cls, name, None)


# ScriptClasses must be new-style classes because we construct them using their
# __new__ method.
def _is_new_style_class(cls):
    if hasattr(cls, "__class__"):
        return "__dict__" in dir(cls) or hasattr(cls, "__slots__")


# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of Module.
# This allows us to reuse most of nn.Module while still storing the
# data in C++.
# Each OrderedDict needs to support:
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)


class OrderedDictWrapper:
    def __init__(self, _c):
        self._c = _c

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __len__(self):
        return len(self.values())

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        return self._c.items()

    def __setitem__(self, k, v):
        if k not in self:
            raise RuntimeError(
                f"Can't add a new parameter after ScriptModule construction. Tried to add '{k}"
            )
        self._c.setattr(k, v)

    def __contains__(self, k):
        return self._c.contains(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module, python_dict):
        super().__init__(torch._C.ModuleDict(module))
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = python_dict

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        # Cases where sub-module can be re-assigned after ScriptModule construction
        # 1. If the attr is a module interface type, it's guaranteed that the module is
        #    not inlined in the graph, so it's safe to swap a new ScriptModule in.
        # 2. if the new value if a ScriptModule with the same JIT type, IR won't change
        #    and it's legit to swap a new module in.
        # In these two cases we allow swapping a new scripted module and update the
        # corresponding python module dict to keep sync.
        # Note: the value to be swapped in has to be ScriptModule instead of nn.Module,
        # otherwise it's illegal and we throw error.
        if isinstance(v, ScriptModule):
            self._c.setattr(k, v)
            self._python_modules[k] = v
        else:
            raise RuntimeError(
                "Cannot re-assign modules in a ScriptModule with non-scripted "
                f"module, tried to replace existing module '{k}': {v}"
            )

    def __getitem__(self, k):
        return self._python_modules[k]


# For each user-defined class that subclasses ScriptModule, this meta-class:
# (1) finds all the methods annotated with @script_method in a ScriptModule and
#     removes them from the class attributes
# (2) puts a wrapper around the class's __init__ method to recursively compile
#     all of the script_methods with the module after the original __init__ has
#     run. This has to occur after the user-defined __init__ so that submodules and
#     parameters are initialized _before_ the script compiler resolve references to
#     `self.param` or `self.module`.
class ScriptMeta(type):
    def __init__(cls, name, bases, attrs):
        # Aggregate all the ScriptMethods and constants from superclasses
        cls._methods: dict[str, Any] = {}
        cls._constants_set = set(getattr(cls, "__constants__", ()))
        for base in reversed(bases):
            for k, v in getattr(base, "_methods", {}).items():
                cls._methods[k] = v
            base_constants: set = getattr(base, "_constants_set", set())
            cls._constants_set = cls._constants_set.union(base_constants)

        # find all the script methods of the current class
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v

        if getattr(cls, "_disable_script_meta", False):
            # We leave built-in ScriptModule types alone, since this metaclass
            # is only for compiling user classes that inherit from
            # ScriptModule.
            super().__init__(name, bases, attrs)
            return

        original_init = getattr(cls, "__init__", lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            num_methods = len(cls._methods)
            original_init(self, *args, **kwargs)
            added_methods_in_init = len(cls._methods) > num_methods

            if type(self) is cls:

                def make_stubs(module):
                    cls = type(module)
                    if hasattr(cls, "_methods"):
                        return [v for k, v in sorted(cls._methods.items())]
                    else:
                        return infer_methods_to_compile(module)

                self.__dict__["_actual_script_module"] = (
                    torch.jit._recursive.create_script_module(
                        self, make_stubs, share_types=not added_methods_in_init
                    )
                )

                # Delete the Python attributes that now shadow the ScriptModule
                # ones, so that __getattr__ and __setattr__ will properly find
                # the scripted versions.
                concrete_type = self._actual_script_module._concrete_type
                for name in concrete_type.get_attributes():
                    delattr(self, name)
                for name, _ in concrete_type.get_modules():
                    delattr(self, name)
                for name in ("_parameters", "_buffers", "_modules"):
                    delattr(self, name)

        cls.__init__ = init_then_script  # type: ignore[misc]
        super().__init__(name, bases, attrs)


class _CachedForward:
    def __get__(self, obj, cls):
        return self.__getattr__("forward")  # type: ignore[attr-defined]


class ScriptWarning(Warning):
    pass


def script_method(fn):
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.script_method` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)


class ConstMap:
    def __init__(self, const_mapping: Mapping[str, Any]) -> None:
        self.const_mapping = const_mapping

    def __getattr__(self, attr: str) -> Any:
        return self.const_mapping[attr]


def unpackage_script_module(
    importer: PackageImporter, script_module_id: str
) -> torch.nn.Module:
    """
    Call by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.

    Performs work of loading and returning a ScriptModule from a ``torch.package`` archive.
    """
    if not isinstance(importer.zip_reader, torch._C.PyTorchFileReader):
        raise RuntimeError(
            "Loading ScriptObjects from a PackageImporter created from a "
            "directory is not supported. Use a package archive file instead."
        )
    cu = torch._C.CompilationUnit()
    cpp_module = torch._C._import_ir_module_from_package(
        cu,
        importer.zip_reader,
        importer.storage_context,
        validate_map_location(importer.last_map_location),
        script_module_id,
    )
    return wrap_cpp_module(cpp_module)


if _enabled:
    _magic_methods = [
        "__iter__",
        "__len__",
        "__neg__",
        "__mul__",
        "__contains__",
        "__add__",
        "__sub__",
        "__pow__",
        "__truediv__",
        "__mod__",
        "__ne__",
        "__eq__",
        "__lt__",
        "__gt__",
        "__le__",
        "__ge__",
        "__and__",
        "__or__",
        "__xor__",
        "__getitem__",
        "__setitem__",
        "__call__",
        "__int__",
        "__float__",
        "__bool__",
        "__str__",
        "__enter__",
        "__exit__",
    ]

    class RecursiveScriptClass:
        """Wrapper for a TorchScript class instance for use in Python.

        An analogue of RecursiveScriptModule for regular objects that are not modules.
        This class is a wrapper around a torch._C.ScriptObject that represents an instance
        of a TorchScript class and allows it to be used in Python.

        Attributes:
            _c [torch._C.ScriptObject]: The C++ object to which attribute lookups and method
                calls are forwarded.
            _props [Dict[str, property]]: A dictionary of properties fetched from self._c and
                exposed on this wrapper.
        """

        def __init__(self, cpp_class):
            super().__init__()
            self.__dict__["_initializing"] = True
            self._c = cpp_class

            # Add wrapped object's properties to this class instance.
            self._props = {
                prop.name: property(prop.getter, prop.setter)
                for prop in self._c._properties()
            }

            self.__dict__["_initializing"] = False

        def __getattr__(self, attr: str) -> Any:
            if self.__dict__.get("_initializing"):
                return super().__getattr__(attr)  # type: ignore[misc]

            if attr in self._props:
                return self._props[attr].fget()  # type: ignore[call-arg, misc]

            return getattr(self._c, attr)

        def __setattr__(self, attr: str, value: Any) -> None:
            if self.__dict__.get("_initializing"):
                return super().__setattr__(attr, value)

            if attr in self._props:
                return self._props[attr].fset(value)  # type: ignore[call-arg, misc]

            setattr(self._c, attr, value)

        # Delegate calls to magic methods like __len__ to the C++ module backing the
        # RecursiveScriptClass.
        def forward_magic_method(
            self, method_name: str, *args: Any, **kwargs: Any
        ) -> Any:
            if not self._c._has_method(method_name):
                raise TypeError

            self_method = self.__getattr__(method_name)
            return self_method(*args, **kwargs)

        def __getstate__(self):
            raise pickle.PickleError("ScriptClasses cannot be pickled")

        def __iadd__(self, other: Self) -> Self:
            if self._c._has_method("__iadd__"):
                return self.forward_magic_method("__iadd__", other)
            else:
                return self.forward_magic_method("__add__", other)

    for method_name in _magic_methods:

        def method_template(self, *args, **kwargs):
            return self.forward_magic_method(method_name, *args, **kwargs)

        setattr(RecursiveScriptClass, method_name, method_template)

    # this is a Python 'non-data descriptor' that causes the first access
    # to ScriptModule's forward to look up the forward method and stash
    # it in the objects dict. Due to the standard rules for attribute lookup,
    # subsequent lookups will just directly return the previously looked up method.
    # This is necessary because nn.Module defines forward as a method. If we
    # did nothing, __getattr__ would not be called. Instead we'd get nn.Module.forward
    # which always throws an exception.

    class ScriptModule(Module, metaclass=ScriptMeta):
        r"""Wrapper for C++ torch::jit::Module with methods, attributes, and parameters.

        A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same way as on a normal ``nn.Module``.
        """

        __jit_unused_properties__ = [
            "code",
            "code_with_constants",
            "graph",
            "inlined_graph",
            "original_name",
        ]

        def __init__(self) -> None:
            super().__init__()

        forward: Callable[..., Any] = _CachedForward()  # type: ignore[assignment]

        def __getattr__(self, attr: str) -> Any:
            if "_actual_script_module" not in self.__dict__:
                return super().__getattr__(attr)
            return getattr(self._actual_script_module, attr)

        def __setattr__(self, attr: str, value: Any) -> None:
            if "_actual_script_module" not in self.__dict__:
                # Unwrap torch.jit.Attribute into a regular setattr + record
                # the provided type in __annotations__.
                #
                # This ensures that if we use the attr again in `__init__`, it
                # will look like the actual value, not an instance of Attribute.
                # pyrefly: ignore [invalid-argument]
                if isinstance(value, Attribute):
                    # NB: Ensure that we set __annotations__ on the specific
                    # class in question, and not on a superclass (which would
                    # be wrong wrong wrong!).
                    # See also https://github.com/pytorch/pytorch/issues/39463
                    if "__annotations__" not in self.__class__.__dict__:
                        self.__class__.__annotations__ = {}
                    self.__annotations__[attr] = value.type
                    value = value.value
                return super().__setattr__(attr, value)

            setattr(self._actual_script_module, attr, value)

        def define(self, src):
            if "_actual_script_module" in self.__dict__:
                # If we have completed initialization, just defer to the
                # backing RecursiveScriptModule to eagerly compile the provided
                # source.
                return self._actual_script_module.define(src)

            # Otherwise, we are still in the object's __init__.
            # In that case, add `src` as a stub to be compiled.
            #
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            ast = torch._C._parse_source_def(src)
            self._methods[ast.name().name] = ScriptMethodStub(rcb, ast, None)

        def _replicate_for_data_parallel(self):
            return self._actual_script_module._replicate_for_data_parallel()

        def __reduce_package__(self, exporter: PackageExporter):
            """Save a ScriptModule inside of a ``torch.package`` archive.

            Called by ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` when
            saving TorchScript objects. Performs act of saving a ScriptModule inside of
            a ``torch.package`` archive.

            Returns method to load the ScriptModule from a ``torch.package.PackageImporter``'s
            Pickler's ``persistent_load`` function.
            """
            script_module_id = exporter.get_unique_id()
            exporter.script_module_serializer.serialize(self._c, int(script_module_id))
            return (unpackage_script_module, (script_module_id,))

    class RecursiveScriptModule(ScriptModule):
        # XXX: RecursiveScriptModule inherits from ScriptModule for the sole
        # reason that it retains the existing isinstance(ScriptModule)
        # behavior.
        r"""Retain the existing isinstance(ScriptModule) behavior.

        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions, a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\s code to run without the need for a Python interpreter.

        ``ScriptModule``\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """

        _disable_script_meta = True

        def __init__(self, cpp_module):
            self.__dict__["_initializing"] = True
            self._c = cpp_module
            super().__init__()
            # Delete the 'training' attribute set up by `Module.__init__`. It
            # will get set on the underlying cpp module, so we delete it here
            # to avoid this version shadowing the cpp module version.
            delattr(self, "training")

        @staticmethod
        def _construct(cpp_module, init_fn):
            """
            Construct a RecursiveScriptModule that's ready for use.

            PyTorch code should use this to construct a RecursiveScriptModule instead
            of instead of calling `__init__` directly, as it makes sure the
            object is properly finalized (and in the future, we may take
            control of how the RecursiveScriptModule instance is created).

            Args:
                cpp_module:  The C++ Module that will hold the actual state of
                             this RecursiveScriptModule instance.
                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.
            """
            script_module = RecursiveScriptModule(cpp_module)
            init_fn(script_module)

            # Finalize the ScriptModule: replace the nn.Module state with our
            # custom implementations and flip the _initializing bit.
            # pyrefly: ignore [missing-attribute]
            RecursiveScriptModule._finalize_scriptmodule(script_module)
            return script_module

        @staticmethod
        def _finalize_scriptmodule(script_module):
            script_module._parameters = OrderedDictWrapper(
                torch._C.ParameterDict(script_module._c)
            )
            script_module._buffers = OrderedDictWrapper(
                torch._C.BufferDict(script_module._c)
            )
            script_module._modules = OrderedModuleDict(
                script_module._c, script_module._modules
            )
            script_module._initializing = False

        def _reconstruct(self, cpp_module):
            """
            Re-construct an instance of RecursiveScriptModule using an instance of a C++ module.

            Args:
                cpp_module: The C++ module that this RecursiveScriptModule will be rebuilt around.
            """
            self.__init__(cpp_module)  # type: ignore[misc]

            # Copy the concrete type from the C++ module to this ScriptModule.
            self._concrete_type = torch._C.ConcreteModuleType.from_jit_type(
                self._c._type()
            )

            # Copy submodules from the C++ module to this ScriptModule.
            modules = {}
            for name, cpp_module in torch._C.ModuleDict(self._c).items():
                modules[name] = wrap_cpp_module(cpp_module)
            self._modules = OrderedModuleDict(self._c, modules)  # type: ignore[assignment]

            # Copy parameters and buffers.
            self._parameters = OrderedDictWrapper(torch._C.ParameterDict(self._c))  # type: ignore[assignment]
            self._buffers = OrderedDictWrapper(torch._C.BufferDict(self._c))  # type: ignore[assignment]

            # Get rid of the functions from the old C++ module.
            self.__dict__ = {
                k: v
                for k, v in self.__dict__.items()
                if not isinstance(v, torch._C.ScriptMethod)
            }
            self.__dict__["_initializing"] = False

        @property
        def graph(self):
            r"""Return a string representation of the internal graph for the ``forward`` method."""
            return self._c._get_method("forward").graph

        @property
        def inlined_graph(self):
            r"""
            Return a string representation of the internal graph for the ``forward`` method.

            This graph will be preprocessed to inline all function and method calls.
            """
            return self.forward.inlined_graph  # type: ignore[attr-defined]

        @property
        def code(self):
            r"""
            Return a pretty-printed representation (as valid Python syntax) of the internal graph for the ``forward`` method.

            """
            return self.forward.code  # type: ignore[attr-defined]

        @property
        def code_with_constants(self):
            r"""Return a tuple.

            Returns a tuple of:

            [0] a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `code`.
            [1] a ConstMap following the CONSTANT.cN format of the output in [0].
            The indices in the [0] output are keys to the underlying constant's values.

            """
            r = self.forward.code_with_constants  # type: ignore[attr-defined]
            return (r[0], ConstMap(r[1]))

        def save(self, f, **kwargs):
            r"""Save with a file-like object.

            save(f, _extra_files={})

            See :func:`torch.jit.save <torch.jit.save>` which accepts a file-like object.
            This function, torch.save(), converts the object to a string, treating it as a path.
            DO NOT confuse these two functions when it comes to the 'f' parameter functionality.
            """
            return self._c.save(str(f), **kwargs)

        @deprecated(
            "Lite Interpreter is deprecated. Please consider switching to ExecuTorch. \
            https://docs.pytorch.org/executorch/stable/getting-started.html"
        )
        def _save_for_lite_interpreter(self, *args, **kwargs):
            r"""Add (or update) the bytecode session to the script model.

            _save_for_lite_interpreter(f)

            The updated model is used
            in lite interpreter for mobile applications.

            Args:
                f: a string containing a file name.
                _extra_files: Map from filename to contents which will be stored as part of 'f'.

            """
            warnings.warn(
                "Lite Interpreter is deprecated. Please consider switching to ExecuTorch. \
                https://docs.pytorch.org/executorch/stable/getting-started.html",
                FutureWarning,
                stacklevel=2,
            )
            return self._c._save_for_mobile(*args, **kwargs)

        @deprecated(
            "Lite Interpreter is deprecated. Please consider switching to ExecuTorch. \
            https://docs.pytorch.org/executorch/stable/getting-started.html"
        )
        def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs):
            warnings.warn(
                "Lite Interpreter is deprecated. Please consider switching to ExecuTorch. \
                https://docs.pytorch.org/executorch/stable/getting-started.html",
                FutureWarning,
                stacklevel=2,
            )
            return self._c._save_to_buffer_for_mobile(*args, **kwargs)

        def save_to_buffer(self, *args: Any, **kwargs: Any) -> Any:
            return self._c.save_to_buffer(*args, **kwargs)

        def get_debug_state(self, *args: Any, **kwargs: Any) -> Any:
            return self._c.get_debug_state()

        def extra_repr(self) -> str:
            return f"original_name={self.original_name}"

        def graph_for(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward.graph_for(self, *args, **kwargs)  # type: ignore[attr-defined]

        @property
        def original_name(self):
            # pyrefly: ignore [unnecessary-comparison]
            if type(self) is str(self._c._type().name()):
                return ""
            return str(self._c._type().name())

        def define(self, src):
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            self._c._define(self._concrete_type, src, rcb)

        def __getattr__(self, attr: str) -> Any:
            if "_initializing" not in self.__dict__:
                raise RuntimeError(
                    "ScriptModule has not been initialized, did you forget to call super's init?"
                )

            if self._initializing:
                return super().__getattr__(attr)

            # _modules check is before hasattr since modules are included as attributes in _c,
            # but we want to get the python wrapper from _modules instead of the raw _c object.
            if attr in self._modules:
                return self._modules[attr]
            elif self._c.hasattr(attr):
                return self._c.getattr(attr)
            elif self._c._has_method(attr):
                script_method = self._c._get_method(attr)
                # cache method so future calls do not go through __getattr__
                # to improve invocation performance
                self.__dict__[attr] = script_method
                return script_method

            return super().__getattr__(attr)

        def __setattr__(self, attr: str, value: Any) -> None:
            if self._initializing:
                return super().__setattr__(attr, value)

            if attr in self._modules:
                self._modules[attr] = value
            elif self._c.hasattr(attr):
                self._c.setattr(attr, value)
            elif (
                hasattr(self, "_concrete_type")
                and attr in self._concrete_type.get_constants()
            ):
                # TODO: we don't have _concrete_type set after load(), and in general we lose constant information.
                # We should encode constants as class type attributes (or something) so it persists across save/load.
                raise AttributeError(
                    f"Cannot mutate TorchScript constant value: '{attr}'. Value: '{value}'"
                )
            else:
                # We allow setting Python attributes on the ScriptModule, for
                # when people want to stash some convenience info on it.
                # TODO: it's possible that the following is confusing:
                #   s = torch.jit.script(...)
                #   s.python_attr = ...
                #   s.save()   <--- this doesn't have `python_attr`
                # It's fairly trivial to save enough info to warn in this case.
                return super().__setattr__(attr, value)

        def __copy__(self) -> Self:
            return torch.jit._recursive.wrap_cpp_module(copy.copy(self._c))

        def __deepcopy__(self, memo: dict[int, Any] | None) -> Self:
            return torch.jit._recursive.wrap_cpp_module(copy.deepcopy(self._c, memo))

        # Python magic methods do method lookups on an object's class type, instead of looking up
        # the method defines on the class instance. In order to continue to expose the magic methods
        # of builtin-containers (ModuleList, Sequential, ModuleDict) to Python, we
        # define magic methods here as a shim to the correct attribute.
        def forward_magic_method(
            self, method_name: str, *args: Any, **kwargs: Any
        ) -> Any:
            self_method = getattr(self, method_name)
            if getattr(self_method, "__func__", None) == getattr(
                RecursiveScriptModule, method_name
            ):
                raise NotImplementedError
            return self_method(*args, **kwargs)

        def __iter__(self) -> Iterator[Any]:
            return self.forward_magic_method("__iter__")

        def __getitem__(self, idx: int) -> Any:
            return self.forward_magic_method("__getitem__", idx)

        def __len__(self):
            return self.forward_magic_method("__len__")

        def __contains__(self, key):
            return self.forward_magic_method("__contains__", key)

        # dir is defined by the base nn.Module, so instead of throwing if
        # it is not overridden, we call into the nn.Module __dir__ method
        def __dir__(self) -> Sequence[str]:
            self_method = self.__dir__
            if (
                self_method.__func__  # type: ignore[attr-defined]
                is _get_function_from_type(RecursiveScriptModule, "__dir__")
            ):
                return super().__dir__()
            return self_method()

        # to resolve bool(value), Python looks if __bool__ is defined then __iter__
        # is defined then returns true for classes. Since __iter__() on this
        # class throws if it isn't overridden, we define __bool__ to preserve default behavior
        def __bool__(self):
            self_method = self.__bool__
            if (
                self_method.__func__  # type: ignore[attr-defined]
                is _get_function_from_type(RecursiveScriptModule, "__bool__")
            ):
                return True
            return self_method()

        def _replicate_for_data_parallel(self):
            # we have to initialize ScriptModule properly so that
            # it works with pybind11
            def init_fn(script_module):
                # Don't do anything here, we'll initialize the ScriptModule below
                return

            # pyrefly: ignore [missing-attribute]
            return RecursiveScriptModule._construct(
                self._c._replicate_for_data_parallel(), init_fn
            )

    # Need to copy all RecursiveScriptModule methods to ScriptModule.
    #
    # This is because `super().foo()` does not use
    # `__getattr__` to look up `foo`. So we need to make each method available on
    # the ScriptModule manually.
    # pyrefly: ignore [missing-attribute]
    for name, item in RecursiveScriptModule.__dict__.items():
        if not callable(item) and not isinstance(item, property):
            continue
        if name.startswith("__") or hasattr(ScriptModule, name):
            continue
        # We can copy over the implementation wholesale because besides the
        # `super()` thing above, ScriptModule behaves exactly like
        # RecursiveScriptModule
        setattr(ScriptModule, name, item)

    def _get_methods(cls):
        import inspect

        # In Python 3 unbound methods are functions, but in Python 2 they are methods
        return inspect.getmembers(
            cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)
        )

    _compiled_methods_allowlist = {
        "forward",
        "register_buffer",
        "register_parameter",
        "register_module",
        "add_module",
        "_apply",
        "apply",
        "cuda",
        "cpu",
        "to",
        "type",
        "float",
        "double",
        "half",
        "state_dict",
        "_save_to_state_dict",
        "load_state_dict",
        "_load_from_state_dict",
        "_named_members",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "zero_grad",
        "share_memory",
        "_get_name",
        "extra_repr",
        "_slow_forward",
        "_tracing_name",
        "eval",
        "train",
        "get_extra_state",
        "set_extra_state",
    }

    def _make_fail(name):
        def fail(self, *args, **kwargs):
            raise RuntimeError(name + " is not supported on ScriptModules")

        return fail

    for name, method in _get_methods(torch.nn.Module):
        if name.startswith("__") or name.endswith("_call_impl"):
            continue
        if (
            # pyrefly: ignore [missing-attribute]
            name not in RecursiveScriptModule.__dict__
            and name not in _compiled_methods_allowlist
        ):
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))


else:
    # TODO MAKE SURE THAT DISABLING WORKS
    class RecursiveScriptClass:  # type: ignore[no-redef]
        pass

    class ScriptModule(torch.nn.Module):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()

    class RecursiveScriptModule(ScriptModule):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()


def call_prepare_scriptable_func_impl(obj, memo):
    if not isinstance(obj, torch.nn.Module):
        return obj

    obj_id = id(obj)

    # If obj_id is in memo, obj has already been prepared or is being
    # prepared in another call up the stack.
    if obj_id in memo:
        return memo[id(obj)]

    obj = (
        # pyrefly: ignore [not-callable]
        obj.__prepare_scriptable__() if hasattr(obj, "__prepare_scriptable__") else obj
    )  # type: ignore[operator]
    # Record obj in memo to avoid infinite recursion in the case of cycles in the module
    # hierarchy when recursing below.
    memo[obj_id] = obj

    new_obj_dict = {}

    for name, sub_module in obj.__dict__.items():
        if name == "_modules":
            for k, v in sub_module.items():
                # Mirror the elif below: skip already-scripted children (re-assigning
                # a jit-ignored grandchild back into a ScriptModule would raise).
                if isinstance(v, torch.nn.Module) and not isinstance(v, ScriptModule):
                    sub_module[k] = call_prepare_scriptable_func_impl(v, memo)
            new_obj_dict[name] = sub_module
        elif isinstance(sub_module, torch.nn.Module) and not isinstance(
            sub_module, ScriptModule
        ):
            new_obj_dict[name] = call_prepare_scriptable_func_impl(sub_module, memo)
        else:
            new_obj_dict[name] = sub_module

    for v in new_obj_dict.values():
        obj.__dict__[name] = v

    return obj


def call_prepare_scriptable_func(obj):
    memo: dict[int, torch.nn.Module] = {}
    return call_prepare_scriptable_func_impl(obj, memo)


def create_script_dict(obj):
    """
    Create a ``torch._C.ScriptDict`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python dictionary that is used to initialize the ``ScriptDict``
                    returned by this function.

    Returns:
        An instance of ``torch._C.ScriptDict`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    return torch._C.ScriptDict(obj)  # type: ignore[attr-defined]


def create_script_list(obj, type_hint=None):
    """
    Create a ``torch._C.ScriptList`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python list that is used to initialize the ``ScriptList``
                    returned by this function.
    Returns:
        An instance of ``torch._C.ScriptList`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    return torch._C.ScriptList(obj)  # type: ignore[attr-defined]


_TOPLEVEL: bool = True


def _script_impl(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
):
    global type_trace_db

    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=3,
        )

    # No-op for modules, functions, class instances that are already scripted
    if isinstance(obj, RecursiveScriptClass):
        return obj
    if isinstance(obj, ScriptModule):
        return obj
    if isinstance(obj, ScriptFunction):
        return obj

    if example_inputs:
        # If MonkeyType is installed, enable profile directed type annotation
        # Check if example_inputs are defined and generate call traces
        # for the method by running eager mode version of the method with
        # the provide example inputs. This logs all the traces in type_trace_db
        type_trace_db = JitTypeTraceStore()
        if monkeytype_trace:
            # pyrefly: ignore [bad-argument-count]
            monkeytype_config = JitTypeTraceConfig(type_trace_db)
            with monkeytype_trace(monkeytype_config):
                if isinstance(example_inputs, dict):
                    # If the obj is an nn.Module or a class, then each method is
                    # executed with the arguments provided in the example inputs.
                    # example inputs here will be of type Dict(class.method, (arguments))
                    # This is used to infer type annotations for those methods
                    # which are not called directly under the hood of monkeytype.
                    for module, example_input in example_inputs.items():
                        for example in example_input:
                            module(*example)
                elif isinstance(example_inputs, list):
                    for examples in example_inputs:
                        obj(*examples)
                else:
                    raise ValueError(
                        "Error: Unable to infer types. Please format the inputs to type `List[Tuple]`"
                        " or `Dict[Callable, List[Tuple]]` to be run with MonkeyType."
                    )
        else:
            warnings.warn(
                "Warning: monkeytype is not installed. Please install https://github.com/Instagram/MonkeyType "
                "to enable Profile-Directed Typing in TorchScript. Refer to "
                "https://github.com/Instagram/MonkeyType/blob/master/README.rst to install MonkeyType. ",
                stacklevel=2,
            )

    if isinstance(obj, torch.nn.Module):
        obj = call_prepare_scriptable_func(obj)
        return torch.jit._recursive.create_script_module(
            obj, torch.jit._recursive.infer_methods_to_compile
        )
    else:
        obj = (
            obj.__prepare_scriptable__()
            if hasattr(obj, "__prepare_scriptable__")
            else obj
        )  # type: ignore[operator]

    if isinstance(obj, dict):
        return create_script_dict(obj)
    if isinstance(obj, list):
        return create_script_list(obj)

    if inspect.isclass(obj):
        qualified_name = _qualified_name(obj)
        # If this type is a `nn.Module` subclass, they probably meant to pass
        # an instance instead of a Module
        if issubclass(obj, torch.nn.Module):
            raise RuntimeError(
                f"Type '{obj}' cannot be compiled since it inherits from nn.Module, pass an instance instead"
            )

        # Enums are automatically usable in TorchScript, explicitly scripting
        # is not necessary, but not harmful either.
        if issubclass(obj, enum.Enum):
            return obj

        if not _is_new_style_class(obj):
            raise RuntimeError(
                "TorchScript classes must be new-style classes. "
                "Please inherit from 'object'."
            )
        if len(obj.mro()) > 2:
            raise RuntimeError(
                "TorchScript classes does not support inheritance yet. "
                "Please directly inherit from 'object'."
            )
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
        qualified_name = _qualified_name(obj)
        # this is a decorated fn, and we need to the underlying fn and its rcb
        if hasattr(obj, "__script_if_tracing_wrapper"):
            obj = obj.__original_fn  # type: ignore[union-attr]
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)

        # some functions are explicitly marked as not supported in script mode
        if hasattr(obj, "__script_unsupported"):
            raise RuntimeError("TorchScript error: " + obj.__script_unsupported)

        _check_directly_compile_overloaded(obj)
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            maybe_already_compiled_fn._torchdynamo_inline = obj  # type: ignore[attr-defined]
            return maybe_already_compiled_fn
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        fn = torch._C._jit_script_compile(
            qualified_name, ast, _rcb, get_default_args(obj)
        )
        # Forward docstrings
        fn.__doc__ = obj.__doc__
        fn.__name__ = "ScriptFunction"
        fn.__qualname__ = "torch.jit.ScriptFunction"
        # Allow torch.compile() to inline
        fn._torchdynamo_inline = obj  # type: ignore[attr-defined]
        _set_jit_function_cache(obj, fn)
        return fn
    else:
        return torch.jit._recursive.create_script_class(obj)


def script(
    obj: Any,
    optimize: None = None,
    _frames_up: int = 0,
    _rcb: Callable[[str], Any] | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> Any:
    r"""Script the function.

    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be
    subsequently passed by reference between Python and TorchScript with zero copy overhead.

    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists
     and as a decorator ``@torch.jit.script`` for torchscript-classes and functions.

    Args:
        obj (Callable, class, or nn.Module):  The ``nn.Module``, function, class type,
                                                  dictionary, or list to compile.
        example_inputs (Union[List[Tuple], Dict[Callable, List[Tuple]], None]): Provide example inputs
            to annotate the arguments for a function or ``nn.Module``.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned. If ``obj`` is a ``dict``, then
        ``script`` returns an instance of `torch._C.ScriptDict`. If ``obj`` is a ``list``,
        then ``script`` returns an instance of `torch._C.ScriptList`.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    ****Scripting a function using example_inputs**
        Example inputs can be used to annotate a function arguments.

        Example (annotating a function before scripting):

        .. testcode::

            import torch

            def test_sum(a, b):
                return a + b

            # Annotate the arguments to be int
            scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])

            print(type(scripted_fn))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(scripted_fn.code)

            # Call the function using the TorchScript interpreter
            scripted_fn(20, 100)

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super().__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                    input = F.relu(self.conv1(input))
                    input = F.relu(self.conv2(input))
                    return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb

                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99


            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))

        Example ( Annotating forward of nn.Module using example_inputs)::

            import torch
            import torch.nn as nn
            from typing import NamedTuple

            class MyModule(NamedTuple):
            result: List[int]

            class TestNNModule(torch.nn.Module):
                def forward(self, a) -> MyModule:
                    result = MyModule(result=a)
                    return result

            pdt_model = TestNNModule()

            # Runs the pdt_model in eager model with the inputs provided and annotates the arguments of forward
            scripted_model = torch.jit.script(pdt_model, example_inputs={pdt_model: [([10, 20, ], ), ], })

            # Run the scripted_model with actual inputs
            print(scripted_model([20]))
    """
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.script` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    if not _enabled:
        return obj
    try:
        global _TOPLEVEL
        prev = _TOPLEVEL
        _TOPLEVEL = False
        ret = _script_impl(
            obj=obj,
            optimize=optimize,
            _frames_up=_frames_up + 1,
            _rcb=_rcb,
            example_inputs=example_inputs,
        )

        if prev:
            log_torchscript_usage("script", model_id=_get_model_id(ret))

        return ret
    finally:
        _TOPLEVEL = prev


# overloads are registered in _jit_internal and compiled here so that _overload
# can be used in nn/functional.py without an import cycle


def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    for name, overload_value in overload_defaults.items():
        if name not in impl_defaults or impl_defaults[name] != overload_value:
            raise torch.jit.frontend.FrontendError(
                loc,
                "Default parameters on overloads do not affect the runtime so they "
                "must equal to the default parameter on the implementation function. Found on "
                f"parameter {name}",
            )


def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    overload_decl = get_jit_def(overload_fn, overload_fn.__name__).decl()
    overload_signature = torch.jit.annotations.get_signature(
        overload_fn, None, None, inspect.ismethod(overload_fn)
    )
    impl_ast = get_jit_def(impl_fn, impl_fn.__name__)
    overload_defaults = get_default_args(overload_fn)
    implementation_defaults = get_default_args(impl_fn)
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    _check_overload_defaults(
        implementation_defaults, overload_defaults, overload_decl.range()
    )
    fn = torch._C._jit_script_compile_overload(
        qual_name,
        overload_decl,
        impl_ast,
        _rcb,
        implementation_defaults,
        overload_signature,
    )
    return fn


def _get_overloads(obj):
    # check for cached compiled fns
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    qual_name = _qualified_name(obj)
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns

    if obj in uncompiled_overloads:
        raise RuntimeError(
            _jit_internal.get_overload_no_implementation_error_message("function", obj)
        )

    compiled_fns = [
        _compile_function_with_overload(overload_fn, qual_name, obj)
        for overload_fn in uncompiled_overloads
    ]

    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns

    # cache compilation, remove information stored to do compilation
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns


def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(
            f"Function {qual_name} cannot be directly compiled because it"
            " is overloaded. It must be used in a context of a function"
            " where its inputs can determine which overload to call."
        )


def interface(obj: _T) -> _T:
    r"""Decorate to annotate classes or modules of different types.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    This decorator can be used to define an interface that can be used to annotate
    classes or modules of different types. This can be used for to annotate a submodule
    or attribute class that could have different types that implement the same
    interface, or which could be swapped at runtime; or to store a list of modules or
    classes of varying types.

    It is sometimes used to implement "Callables" - functions or modules that implement
    an interface but whose implementations differ and which can be swapped out.

    Example:
    .. testcode::

        import torch
        from typing import List

        @torch.jit.interface
        class InterfaceType:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                pass

        # implements InterfaceType
        @torch.jit.script
        class Impl1:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x.relu()

        class Impl2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.val = torch.rand(())

            @torch.jit.export
            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.val

        def user_fn(impls: List[InterfaceType], idx: int, val: torch.Tensor) -> torch.Tensor:
            return impls[idx].run(val)

        user_fn_jit = torch.jit.script(user_fn)

        impls = [Impl1(), torch.jit.script(Impl2())]
        val = torch.rand(4, 4)
        user_fn_jit(impls, 0, val)
        user_fn_jit(impls, 1, val)
    """
    warnings.warn(
        "`torch.jit.interface` is deprecated. Please use `torch.compile` instead.",
        FutureWarning,
    )
    if not inspect.isclass(obj):
        raise RuntimeError("interface must be applied to a class")
    if not _is_new_style_class(obj):
        raise RuntimeError("TorchScript interfaces must inherit from 'object'")

    # Expected MRO is:
    #   User module
    #   torch.nn.modules.module.Module
    #   object
    is_module_interface = issubclass(obj, torch.nn.Module) and len(obj.mro()) == 3

    if not is_module_interface and len(obj.mro()) > 2:
        raise RuntimeError(
            "TorchScript interface does not support inheritance yet. "
            "Please directly inherit from 'object' or 'nn.Module'."
        )

    qualified_name = _qualified_name(obj)
    rcb = _jit_internal.createResolutionCallbackFromFrame(1)
    # if this type is a `nn.Module` subclass, generate a module interface type
    # instead of a class interface type; a module interface type only compiles
    # the user provided methods as part of the interface
    ast = get_jit_class_def(obj, obj.__name__)
    mangled_classname = torch._C._jit_script_interface_compile(
        qualified_name, ast, rcb, is_module_interface
    )
    obj.__torch_script_interface__ = mangled_classname
    return obj


def _recursive_compile_class(obj, loc):
    _qual_name = _qualified_name(obj)
    # We're starting a new compilation, so update the error call stack in
    # case it fails
    error_stack = torch._C.CallStack(_qual_name, loc)  # noqa: F841
    rcb = _jit_internal.createResolutionCallbackForClassMethods(obj)
    return _compile_and_register_class(obj, rcb, _qual_name)


CompilationUnit = torch._C.CompilationUnit
set_module(CompilationUnit, "torch.jit")


def pad(s: str, padding: int, offset: int = 0, char: str = " "):
    if padding >= len(s):
        padding -= len(s)
    return "".join([char for _ in range(padding + offset)]) + s


class _ScriptProfileColumn:
    def __init__(self, header: str, alignment: int = 4, offset: int = 0):
        self.header = header
        self.alignment = alignment
        self.offset = offset
        self.rows: dict[int, Any] = {}

    def add_row(self, lineno: int, value: Any):
        self.rows[lineno] = value

    def materialize(self):
        max_length = len(self.header)
        rows: list[tuple[int, str]] = []
        for key, value in self.rows.items():
            cell = str(value)
            rows.append((key, cell))
            max_length = max(len(cell), max_length)

        if self.alignment > 0:
            padding = max_length + self.alignment
            padding -= padding % self.alignment
        else:
            padding = 0

        rows = [(key, pad(cell, padding, self.offset)) for key, cell in rows]
        return pad(self.header, padding, self.offset), rows


class _ScriptProfileTable:
    def __init__(self, cols: list[_ScriptProfileColumn], source_range: list[int]):
        self.cols = cols
        self.source_range = source_range

    def dump_string(self):
        outputs: list[str] = []
        cells: list[tuple[str, dict[int, str]]] = []
        header_buffer = ""
        for col in self.cols:
            header, rows = col.materialize()
            header_buffer += header
            cells.append((header, dict(rows)))

        outputs.append(header_buffer)
        outputs.append(pad("", len(header_buffer), 0, "="))
        for line in self.source_range:
            row_buffer = ""
            for header, rows in cells:
                cell = rows.get(line)
                if cell is None:
                    row_buffer += pad("", len(header))
                else:
                    row_buffer += cell
            outputs.append(row_buffer)
        return "\n".join(outputs)


class _ScriptProfile:
    def __init__(self) -> None:
        self.profile = classes.profiling._ScriptProfile()

    def enable(self):
        self.profile.enable()

    def disable(self):
        self.profile.disable()

    def dump_string(self) -> str:
        outputs: list[str] = []
        for source_stats in self.profile._dump_stats():
            source_ref = source_stats.source()
            source_lines = source_ref.text().splitlines()
            dedent = min(len(line) - len(line.lstrip(" ")) for line in source_lines)
            source_lines = [line[dedent:] for line in source_lines]

            start_line = source_ref.starting_lineno()
            end_line = start_line + len(source_lines)
            source_range = range(start_line, end_line)
            lineno = _ScriptProfileColumn("Line #")
            hits = _ScriptProfileColumn("Hits")
            time_ns = _ScriptProfileColumn("Time (ns)")
            line_contents = _ScriptProfileColumn("Line Contents", 0, 1)
            stats = source_stats.line_map()
            for line in source_range:
                lineno.add_row(line, line)
                line_contents.add_row(line, source_lines[line - start_line])
                stat = stats.get(line)
                if stat is not None:
                    hits.add_row(line, stat.count())
                    time_ns.add_row(line, stat.duration_ns())

            table = _ScriptProfileTable(
                [lineno, hits, time_ns, line_contents], list(source_range)
            )
            outputs.append(table.dump_string())
        return "\n\n".join(outputs)

    def dump(self):
        print(self.dump_string())


def _unwrap_optional(x):
    if x is None:
        raise AssertionError("Unwrapping null optional")
    return x


_register_builtin(_unwrap_optional, "aten::_unwrap_optional")
_register_builtin(_jit_internal.is_scripting, "aten::is_scripting")
_register_builtin(has_torch_function, "aten::has_torch_function")
_register_builtin(has_torch_function_unary, "aten::has_torch_function")
_register_builtin(has_torch_function_variadic, "aten::has_torch_function")
<|endoftext|># mypy: allow-untyped-defs
"""Async API.

This module contains the API for parallelism in TorchScript, notably:
    * torch.jit.fork
    * torch.jit.wait

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import warnings

import torch
from torch._jit_internal import Future
from torch.jit._builtins import _register_builtin
from torch.utils import set_module


set_module(Future, "torch.jit")


def fork(func, *args, **kwargs):
    r"""
    Create an asynchronous task executing `func` and a reference to the value of the result of this execution.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    `fork` will return immediately, so the return value of `func` may not have been computed yet. To force completion
    of the task and access the return value invoke `torch.jit.wait` on the Future. `fork` invoked
    with a `func` which returns `T` is typed as `torch.jit.Future[T]`. `fork` calls can be arbitrarily
    nested, and may be invoked with positional and keyword arguments.
    Asynchronous execution will only occur when run in TorchScript. If run in pure python,
    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked
    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.

    .. warning::
        `fork` tasks will execute non-deterministically. We recommend only spawning
        parallel fork tasks for pure functions that do not modify their inputs,
        module attributes, or global state.

    Args:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be invoked. If executed in TorchScript, it will execute asynchronously,
            otherwise it will not. Traced invocations of fork will be captured in the IR.
        ``*args``, ``**kwargs``: arguments to invoke `func` with.
    Returns:
        `torch.jit.Future[T]`: a reference to the execution of `func`. The value `T`
        can only be accessed by forcing completion of `func` through `torch.jit.wait`.

    Example (fork a free function):

    .. code-block:: python

        import torch
        from torch import Tensor


        def foo(a: Tensor, b: int) -> Tensor:
            return a + b


        def bar(a):
            fut: torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
            return torch.jit.wait(fut)


        script_bar = torch.jit.script(bar)
        input = torch.tensor(2)
        # only the scripted version executes asynchronously
        assert script_bar(input) == bar(input)
        # trace is not run asynchronously, but fork is captured in IR
        graph = torch.jit.trace(bar, (input,)).graph
        assert "fork" in str(graph)

    Example (fork a module method):

    .. code-block:: python

        import torch
        from torch import Tensor


        class AddMod(torch.nn.Module):
            def forward(self, a: Tensor, b: int):
                return a + b


        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(self).__init__()
                self.mod = AddMod()

            def forward(self, input):
                fut = torch.jit.fork(self.mod, a, b=2)
                return torch.jit.wait(fut)


        input = torch.tensor(2)
        mod = Mod()
        assert mod(input) == torch.jit.script(mod).forward(input)
    """
    warnings.warn(
        "`torch.jit.fork` is deprecated. Please use `torch.compile` instead.",
        FutureWarning,
    )
    return torch._C.fork(func, *args, **kwargs)


def wait(future):
    r"""
    Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    See :func:`~fork` for docs and examples.
    Args:
        future (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`
    Returns:
        `T`: the return value of the completed task
    """
    warnings.warn(
        "`torch.jit.wait` is deprecated. Please use `torch.compile` instead.",
        FutureWarning,
    )
    return torch._C.wait(future)


_register_builtin(wait, "aten::wait")
<|endoftext|># mypy: allow-untyped-defs
import inspect
import sys
import typing
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import CodeType

import torch


_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore[import]

    # pyrefly: ignore [import-error, missing-import]
    from monkeytype import trace as monkeytype_trace
    from monkeytype.config import _startswith, LIB_PATHS  # type: ignore[import]
    from monkeytype.db.base import (  # type: ignore[import]
        CallTraceStore,
        CallTraceStoreLogger,
        CallTraceThunk,
    )
    from monkeytype.tracing import CallTrace, CodeFilter  # type: ignore[import]
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False


# Checks whether a class is defined in `torch.*` modules
def is_torch_native_class(cls):
    if not hasattr(cls, "__module__"):
        return False

    parent_modules = cls.__module__.split(".")
    if not parent_modules:
        return False

    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch


def get_type(type):
    """Convert the given type to a torchScript acceptable format."""
    if isinstance(type, str):
        return type
    elif inspect.getmodule(type) == typing:
        # If the type is a type imported from typing
        # like Tuple, List, Dict then replace `typing.`
        # with a null string. This needs to be done since
        # typing.List is not accepted by TorchScript.
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + ".", "")
    elif is_torch_native_class(type):
        # If the type is a subtype of torch module, then TorchScript expects a fully qualified name
        # for the type which is obtained by combining the module name and type name.
        return type.__module__ + "." + type.__name__
    else:
        # For all other types use the name for the type.
        return type.__name__


def get_optional_of_element_type(types):
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extract the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
    elem_type = types[1] if type(None) is types[0] else types[0]
    elem_type = get_type(elem_type)

    # Optional type is internally converted to Union[type, NoneType], which
    # is not supported yet in TorchScript. Hence, representing the optional type as string.
    return "Optional[" + elem_type + "]"


def get_qualified_name(func):
    return func.__qualname__


if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def log(self, trace: CallTrace) -> None:
            # pyrefly: ignore [missing-attribute]
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):
        def __init__(self) -> None:
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]) -> None:
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: str | None = None,
            limit: int = 2000,
        ) -> list[CallTraceThunk]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    # TODO: To remove this check once Union support in TorchScript lands.
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = "Any"
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> dict:
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):
        def __init__(self, s: JitTypeTraceStore) -> None:
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
            # pyrefly: ignore [bad-argument-count]
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            return self.s

        def code_filter(self) -> CodeFilter | None:
            return jit_code_filter

else:
    # When MonkeyType is not installed, we provide dummy class definitions
    # for the below classes.
    class JitTypeTraceStoreLogger:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    class JitTypeTraceStore:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            self.trace_records = None

    class JitTypeTraceConfig:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    monkeytype_trace = None  # type: ignore[assignment]


def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
    # Filter code without a source file and exclude this check for 'forward' calls.
    if code.co_name != "forward" and (
        not code.co_filename or code.co_filename[0] == "<"
    ):
        return False

    filename = Path(code.co_filename).resolve()
    return not any(_startswith(filename, lib_path) for lib_path in LIB_PATHS)
<|endoftext|>from types import TracebackType

import torch


class _InsertPoint:
    def __init__(
        self,
        insert_point_graph: torch._C.Graph,
        insert_point: torch._C.Node | torch._C.Block,
    ) -> None:
        self.insert_point = insert_point
        self.g = insert_point_graph
        self.guard = None

    def __enter__(self) -> None:
        self.prev_insert_point = self.g.insertPoint()
        self.g.setInsertPoint(self.insert_point)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.g.setInsertPoint(self.prev_insert_point)


def insert_point_guard(
    self: torch._C.Graph, insert_point: torch._C.Node | torch._C.Block
) -> _InsertPoint:
    return _InsertPoint(self, insert_point)
<|endoftext|># mypy: allow-untyped-defs
import os

import torch
from torch.jit._serialization import validate_map_location


def _load_for_lite_interpreter(f, map_location=None):
    r"""
    Load a :class:`LiteScriptModule` saved with :func:`torch.jit._save_for_lite_interpreter`.

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location: a string or torch.device used to dynamically remap
            storages to an alternative set of devices.

    Returns:
        A :class:`LiteScriptModule` object.

    Example:

    .. testcode::

        import torch
        import io

        # Load LiteScriptModule from saved file path
        torch.jit._load_for_lite_interpreter('lite_script_module.pt')

        # Load LiteScriptModule from io.BytesIO object
        with open('lite_script_module.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.mobile._load_for_lite_interpreter(buffer)
    """
    if isinstance(f, (str, os.PathLike)):
        if not os.path.exists(f):
            raise ValueError(f"The provided filename {f} does not exist")
        if os.path.isdir(f):
            raise ValueError(f"The provided filename {f} is a directory")

    map_location = validate_map_location(map_location)

    if isinstance(f, (str, os.PathLike)):
        cpp_module = torch._C._load_for_lite_interpreter(os.fspath(f), map_location)
    else:
        cpp_module = torch._C._load_for_lite_interpreter_from_buffer(
            f.read(),
            map_location,
        )

    return LiteScriptModule(cpp_module)


class LiteScriptModule:
    def __init__(self, cpp_module) -> None:
        self._c = cpp_module
        super().__init__()

    def __call__(self, *input):
        return self._c.forward(input)

    def find_method(self, method_name):
        return self._c.find_method(method_name)

    def forward(self, *input):
        return self._c.forward(input)

    def run_method(self, method_name, *input):
        return self._c.run_method(method_name, input)


def _export_operator_list(module: LiteScriptModule):
    r"""Return a set of root operator names (with overload name) that are used by any method in this mobile module."""
    return torch._C._export_operator_list(module._c)


def _get_model_bytecode_version(f_input) -> int:
    r"""Take a file-like object to return an integer.

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        version: An integer. If the integer is -1, the version is invalid. A warning
            will show in the log.

    Example:
    .. testcode::

        from torch.jit.mobile import _get_model_bytecode_version

        # Get bytecode version from a saved file path
        version = _get_model_bytecode_version("path/to/model.ptl")

    """
    if isinstance(f_input, (str, os.PathLike)):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if isinstance(f_input, (str, os.PathLike)):
        return torch._C._get_model_bytecode_version(os.fspath(f_input))
    else:
        return torch._C._get_model_bytecode_version_from_buffer(f_input.read())


def _get_mobile_model_contained_types(f_input) -> int:
    r"""Take a file-like object and return a set of string, like ("int", "Optional").

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        type_list: A set of string, like ("int", "Optional"). These are types used in bytecode.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_mobile_model_contained_types

        # Get type list from a saved file path
        type_list = _get_mobile_model_contained_types("path/to/model.ptl")

    """
    if isinstance(f_input, (str, os.PathLike)):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if isinstance(f_input, (str, os.PathLike)):
        return torch._C._get_mobile_model_contained_types(os.fspath(f_input))
    else:
        return torch._C._get_mobile_model_contained_types_from_buffer(f_input.read())


def _backport_for_mobile(f_input, f_output, to_version):
    r"""Take an input string containing a file name (file-like object) and a new destination to return a boolean.

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        f_output: path to new model destination
        to_version: the expected output model bytecode version
    Returns:
        success: A boolean. If backport success, return true, otherwise false
    """
    if isinstance(f_input, (str, os.PathLike)):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if (isinstance(f_input, (str, os.PathLike))) and (
        isinstance(f_output, (str, os.PathLike))
    ):
        return torch._C._backport_for_mobile(
            os.fspath(f_input),
            os.fspath(f_output),
            to_version,
        )
    else:
        return torch._C._backport_for_mobile_from_buffer(
            f_input.read(),
            str(f_output),
            to_version,
        )


def _backport_for_mobile_to_buffer(f_input, to_version):
    r"""Take a string containing a file name (file-like object).

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    """
    if isinstance(f_input, (str, os.PathLike)):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if isinstance(f_input, (str, os.PathLike)):
        return torch._C._backport_for_mobile_to_buffer(os.fspath(f_input), to_version)
    else:
        return torch._C._backport_for_mobile_from_buffer_to_buffer(
            f_input.read(),
            to_version,
        )


def _get_model_ops_and_info(f_input):
    r"""Retrieve the root (top level) operators of a model and their corresponding compatibility info.

    These root operators can call other operators within them (traced ops), and
    a root op can call many different traced ops depending on internal code paths in the root op.
    These traced ops are not returned by this function. Those operators are abstracted into the
    runtime as an implementation detail (and the traced ops themselves can also call other operators)
    making retrieving them difficult and their value from this api negligible since they will differ
    between which runtime version the model is run on. Because of this, there is a false positive this
    api can't prevent in a compatibility usecase. All the root ops of a model are present in a
    target runtime, but not all the traced ops are which prevents a model from being able to run.
    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        Operators and info: A Dictionary mapping strings (the qualified names of the root operators)
        of the model to their OperatorInfo structs.

    Example:

    .. testcode::

        from torch.jit.mobile import _get_model_ops_and_info

        # Get bytecode version from a saved file path
        ops_and_info = _get_model_ops_and_info("path/to/model.ptl")

    """
    if isinstance(f_input, (str, os.PathLike)):
        if not os.path.exists(f_input):
            raise ValueError(f"The provided filename {f_input} does not exist")
        if os.path.isdir(f_input):
            raise ValueError(f"The provided filename {f_input} is a directory")

    if isinstance(f_input, (str, os.PathLike)):
        return torch._C._get_model_ops_and_info(os.fspath(f_input))
    else:
        return torch._C._get_model_ops_and_info(f_input.read())
<|endoftext|>"""
Tools to help with tensor property propagation.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

from typing import Any

import torch
from torch import TensorType
from torch._C import Graph


def apply_input_props_using_example(graph: Graph, example_input: list[Any]) -> None:
    """
    Applies properties for each tensor in the graph inputs
    using the example supplied.
    """
    graph_inputs = list(graph.inputs())
    if len(graph_inputs) == 0:
        return

    # Strip self args off for methods
    in_0 = graph_inputs[0]
    if isinstance(in_0.type(), torch._C.ClassType) and in_0.debugName() == "self":
        graph_inputs = graph_inputs[1:]

    if not len(graph_inputs) == len(example_input):
        raise RuntimeError(
            "Number of inputs in graph does not match number of inputs in the example"
        )

    for i, (graph_i, example_i) in enumerate(zip(graph_inputs, example_input)):
        if example_i is None:
            continue  # Skip the type check

        if isinstance(example_i, torch.Tensor) != isinstance(
            graph_i.type(), TensorType
        ):
            raise RuntimeError(
                f"Input {i} does not match type of example", graph_i, example_i
            )

        if isinstance(example_i, torch.Tensor):
            graph_i.setType(TensorType.create_from_tensor(example_i))  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, overload, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["OptState", "GradScaler"]


class _MultiDeviceReplicator:
    """Lazily serves copies of a tensor to requested devices.

    Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        self.master = master_tensor
        self._per_device_tensors: dict[torch.device, torch.Tensor] = {}

    def get(self, device: torch.device) -> torch.Tensor:
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


# Defines default_factory for GradScaler's _per_optimizer_states defaultdict,
# as well as associated "enum" values.  Prefers defining these at top level because
# - Lambdas can't be pickled, so we don't want to supply a lambda as the factory.
# - Defining READY, UNSCALED, STEPPED and _refresh_per_optimizer_state within GradScaler
#   causes a circular reference, which we'd rather avoid.
class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_per_optimizer_state() -> dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


class GradScaler:
    """An instance ``scaler`` of :class:`GradScaler`.

    Helps perform the steps of gradient scaling
    conveniently.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
    * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
    * ``scaler.update()`` updates ``scaler``'s scale factor.

    Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage
    (along with autocasting) in more complex cases like gradient clipping, gradient accumulation, gradient penalty,
    and multiple losses/optimizers.

    ``scaler`` dynamically estimates the scale factor each iteration.  To minimize gradient underflow,
    a large scale factor should be used.  However, ``float16`` values can "overflow" (become inf or NaN) if
    the scale factor is too large.  Therefore, the optimal scale factor is the largest factor that can be used
    without incurring inf or NaN gradient values.
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).

    Args:
        device (str, optional, default="cuda"): Device type to use. Possible values are: 'cuda' and 'cpu'.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
    """

    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._device = device
        self._enabled = enabled
        if self._device == "cuda":
            if enabled and torch.cuda.amp.common.amp_definitely_not_available():
                warnings.warn(
                    "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
                    stacklevel=2,
                )
                self._enabled = False

        if self._enabled:
            if growth_factor <= 1.0:
                raise AssertionError("The growth factor must be > 1.0.")
            if backoff_factor >= 1.0:
                raise AssertionError("The backoff factor must be < 1.0.")

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale: torch.Tensor | None = None
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker: torch.Tensor | None = None
            self._per_optimizer_states: dict[int, dict[str, Any]] = defaultdict(
                _refresh_per_optimizer_state
            )

    def _check_scale_growth_tracker(
        self, funcname: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        if self._scale is None:
            raise AssertionError(f"Attempted {funcname} but _scale is None.  " + fix)
        if self._growth_tracker is None:
            raise AssertionError(
                f"Attempted {funcname} but _growth_tracker is None.  " + fix
            )
        return (self._scale, self._growth_tracker)

    def _lazy_init_scale_growth_tracker(self, dev: torch.device) -> None:
        if self._growth_tracker is not None:
            raise AssertionError("_growth_tracker initialized before _scale")
        self._scale = torch.full((), self._init_scale, dtype=torch.float32, device=dev)
        self._growth_tracker = torch.full(
            (), self._init_growth_tracker, dtype=torch.int32, device=dev
        )

    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    @overload
    def scale(self, outputs: list[torch.Tensor]) -> list[torch.Tensor]: ...

    @overload
    def scale(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]: ...

    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...

    def scale(
        self,
        outputs: torch.Tensor | Iterable[torch.Tensor],
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            # _scale is now guaranteed to be set
            scale = self._scale
            if scale is None:
                raise AssertionError("_scale should not be None after lazy init")
            return outputs * scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: list[
            _MultiDeviceReplicator
        ] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val: torch.Tensor | Iterable[torch.Tensor]):
            if isinstance(val, torch.Tensor):
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    # _scale is now guaranteed to be set
                    scale = self._scale
                    if scale is None:
                        raise AssertionError(
                            "_scale should not be None after lazy init"
                        )
                    stash.append(_MultiDeviceReplicator(scale))
                return val * stash[0].get(val.device)
            if isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                return iterable
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ) -> dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads: dict[
            torch.device, dict[torch.dtype, list[torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if not isinstance(param, torch.Tensor):
                        raise AssertionError(
                            f"expected param to be torch.Tensor, got {type(param).__name__}"
                        )
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )

        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Divides ("unscales") the optimizer's gradient tensors by the scale factor.

        :meth:`unscale_` is optional, serving cases where you need to
        :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
        between the backward pass(es) and :meth:`step`.
        If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.

        Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale_` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
            and only after all gradients for that optimizer's assigned parameters have been accumulated.
            Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

        .. warning::
            :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
        """
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        if self._scale is None:
            raise AssertionError("_scale is None in unscale_")
        inv_scale = (
            self._scale.double().reciprocal().float()
            if self._scale.device != torch.device("mps:0")
            else self._scale.reciprocal()
        )
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, False
        )
        optimizer_state["stage"] = OptState.UNSCALED

    def _maybe_opt_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_state: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> float | None:
        retval: float | None = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        return retval

    def step(
        self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any
    ) -> float | None:
        """Invoke ``unscale_(optimizer)`` followed by parameter update, if gradients are not infs/NaN.

        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError(
                "Closure use is not currently supported if GradScaler is enabled."
            )

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        retval: float | None = None

        if getattr(optimizer, "_step_supports_amp_scaling", False):
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            # The contract above is being deprecated to avoid introducing `grad_scaler: GradScaler` argument
            # to `Optimizer.step`. The new behavior is going to add two Tensor attributes of `grad_scale`
            # and `found_inf` to the passed optimizer so that the optimizer can utilize those
            # to skip the parameter updates or unscale gradients before updating parameters in
            # the fused kernel, e.g. `FusedAdamMathFunctor`.
            # In this behavior, `GradScaler._check_inf_per_device` is called if `OptState.READY`,
            # while the method is expected to be called by users side, i.e. their optimizers.
            kwargs_ = kwargs
            has_grad_scaler_kwarg = (
                "grad_scaler" in inspect.signature(optimizer.step).parameters
            )
            if has_grad_scaler_kwarg:
                warnings.warn(
                    "GradScaler is going to stop passing itself as a keyword argument to the passed "
                    "optimizer. In the near future GradScaler registers `grad_scale: Tensor` and "
                    "`found_inf: Tensor` to the passed optimizer and let the optimizer use them directly.",
                    FutureWarning,
                    stacklevel=2,
                )
                kwargs_.update({"grad_scaler": self})
            else:
                if optimizer_state["stage"] is OptState.READY:
                    self._check_inf_per_device(optimizer)
                scaler = self._get_scale_async()
                if scaler is None:
                    raise AssertionError("_get_scale_async returned None")
                found_inf = cast(
                    torch.Tensor,
                    sum(
                        [
                            t.to(scaler.device, non_blocking=True)
                            for t in optimizer_state["found_inf_per_device"].values()
                        ]
                    ),
                )
                # Take the product of the scales, if the user has already set `optimizer.grad_scale`.
                optimizer.grad_scale = (  # type: ignore[attr-defined]
                    getattr(optimizer, "grad_scale", None)
                    if optimizer_state["stage"] == OptState.UNSCALED
                    else scaler * getattr(optimizer, "grad_scale", 1)
                )
                optimizer.found_inf = found_inf  # type: ignore[attr-defined]
            retval = optimizer.step(*args, **kwargs_)
            optimizer_state["stage"] = OptState.STEPPED
            if not has_grad_scaler_kwarg:
                del optimizer.grad_scale  # type: ignore[attr-defined]
                del optimizer.found_inf  # type: ignore[attr-defined]
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        if len(optimizer_state["found_inf_per_device"]) == 0:
            raise AssertionError("No inf checks were recorded for this optimizer.")

        retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """Update the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.Tensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.

        .. warning::
            For performance reasons, we do not check the scale factor value to avoid synchronizations,
            so the scale factor is not guaranteed to be above 1. If the scale falls below 1 and/or
            you are seeing NaNs in your gradients or loss, something is likely wrong. For example,
            bf16-pretrained models are often incompatible with AMP/fp16 due to differing dynamic ranges.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            if self._scale is None:
                raise AssertionError("_scale is None in update")
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)
            else:
                reason = (
                    "new_scale should be a float or a 1-element torch.cuda.FloatTensor or "
                    "torch.FloatTensor with requires_grad=False."
                )
                if new_scale.device.type != self._device:
                    raise AssertionError(reason)
                if new_scale.numel() != 1:
                    raise AssertionError(reason)
                if new_scale.requires_grad is True:
                    raise AssertionError(reason)
                self._scale.copy_(new_scale)
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            if len(found_infs) == 0:
                raise AssertionError("No inf checks were recorded prior to update.")

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def _get_scale_async(self) -> torch.Tensor | None:
        return self._scale

    def get_scale(self) -> float:
        """Return a Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
        if self._enabled:
            return (
                self._init_scale
                if (scale := self._get_scale_async()) is None
                else cast(float, scale.item())
            )
        return 1.0

    def get_growth_factor(self) -> float:
        r"""Return a Python float containing the scale growth factor."""
        return self._growth_factor

    def set_growth_factor(self, new_factor: float) -> None:
        r"""Set a new scale growth factor.

        Args:
            new_scale (float):  Value to use as the new scale growth factor.
        """
        self._growth_factor = new_factor

    def get_backoff_factor(self) -> float:
        r"""Return a Python float containing the scale backoff factor."""
        return self._backoff_factor

    def set_backoff_factor(self, new_factor: float) -> None:
        r"""Set a new scale backoff factor.

        Args:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
        self._backoff_factor = new_factor

    def get_growth_interval(self) -> int:
        r"""Return a Python int containing the growth interval."""
        return self._growth_interval

    def set_growth_interval(self, new_interval: int) -> None:
        r"""Set a new growth interval.

        Args:
            new_interval (int):  Value to use as the new growth interval.
        """
        self._growth_interval = new_interval

    def _get_growth_tracker(self) -> int:
        if self._enabled:
            return (
                self._init_growth_tracker
                if self._growth_tracker is None
                else cast(int, self._growth_tracker.item())
            )
        return 0

    def is_enabled(self) -> bool:
        r"""Return a bool indicating whether this instance is enabled."""
        return self._enabled

    def state_dict(self) -> dict[str, Any]:
        r"""Return the state of the scaler as a :class:`dict`.

        It contains five entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor
        * ``"growth_interval"`` - a Python int containing the current growth interval
        * ``"_growth_tracker"`` - a Python int containing the number of recent consecutive unskipped steps.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler's state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        """
        if self._enabled:
            return {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
            }
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load the scaler state.

        If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = cast(float, state_dict["scale"])
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = cast(float, state_dict["growth_factor"])
        self._backoff_factor = cast(float, state_dict["backoff_factor"])
        self._growth_interval = cast(int, state_dict["growth_interval"])
        self._init_growth_tracker = cast(int, state_dict["_growth_tracker"])
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self._enabled:
            if len(self._per_optimizer_states) != 0:
                raise AssertionError(
                    "A GradScaler instance may only be pickled at the beginning "
                    "of an iteration, or at the end after scaler.update()."
                )
            # Pickling _scale and _growth_tracker Tensors directly triggers
            # "warnings.warn("pickle support for Storage will be removed in 1.5..."
            # so instead, we set the unpickled instance up to reinitialize them lazily.
            state["_init_scale"] = self.get_scale()
            state["_init_growth_tracker"] = self._get_growth_tracker()
            state["_scale"] = None
            state["_growth_tracker"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _check_inf_per_device(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        _scale, _ = self._check_scale_growth_tracker("_check_inf_per_device")

        dummy_inv_scale = torch.full((), 1.0, dtype=torch.float32, device=_scale.device)
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=_scale.device)

        self._per_optimizer_states[id(optimizer)]["found_inf_per_device"] = (
            self._unscale_grads_(optimizer, dummy_inv_scale, found_inf, True)
        )

        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]

    def _found_inf_per_device(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
<|endoftext|>import torch
from torch._inductor.analysis.device_info import datasheet_dram_bw_gbs, datasheet_tops
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch.fx.experimental.symbolic_shapes import (
    optimization_hint,
    statically_known_true,
)
from torch.utils._ordered_set import OrderedSet

from .flop_counter import flop_registry


aten = torch.ops.aten

_FLOAT_TYPES = OrderedSet(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
)

# No fall-back kernel needed/exists for view ops
_VIEW_OPS = OrderedSet(
    [
        aten.lift_fresh,
        aten.t,
        aten.transpose,
        aten.view,
        aten.detach,
        aten._unsafe_view,
        aten.split,
        aten.adjoint,
        aten.as_strided,
        aten.diagonal,
        aten.expand,
        aten.expand_as,
        aten.movedim,
        aten.permute,
        aten.select,
        aten.squeeze,
        aten.mT,
        aten.mH,
        aten.real,
        aten.imag,
        aten.view_as,
        aten.unflatten,
        aten.unfold,
        aten.unbind,
        aten.unsqueeze,
        aten.vsplit,
        aten.hsplit,
        aten.split_with_sizes,
        aten.swapaxes,
        aten.swapdims,
        aten.chunk,
    ]
)
# We can ignore benchmarking tensor create ops
_CREATE_OPS = OrderedSet(
    [
        aten.randint,
        aten.randn,
        aten.rand,
        aten.randn_like,
        aten.rand_like,
        aten.randint_like,
        aten.arange,
        aten.ones_like,
        aten.zeros_like,
    ]
)

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS


def flops_to_ns(
    flops: float | int, dtype: "torch.dtype", gpu_type: str | None = None
) -> float:
    """Convert a FLOPs count to estimated nanoseconds on the GPU.

    Uses 75% of theoretical peak and converts FLOPs to MACs (divide by 2).

    If ``gpu_type`` names a device in the datasheet, its pinned datasheet
    TFLOPS are used instead of querying the current device, making the
    estimate deterministic and hardware-independent.
    """
    if gpu_type is not None:
        is_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
        tflops = datasheet_tops(dtype, is_tf32=is_tf32, device_name=gpu_type)
        if tflops is None:
            raise ValueError(
                f"gpu_type {gpu_type!r} has no datasheet entry for {dtype}"
            )
    else:
        tflops = get_device_tflops(dtype)
    peak_gpu_flops = tflops * 1e12
    if peak_gpu_flops == 0:
        return 0.0
    macs = flops / 2
    return (macs / (0.75 * peak_gpu_flops)) * 1e9


def get_compute_time(
    func_packet, args, kwargs, out, out_dtypes, node_meta=None, gpu_type=None
) -> float:  # type: ignore[no-untyped-def]
    """
    Estimates the compute time of an aten operator.

    Args:
        func_packet: The operator overload packet.
        args: The arguments to the operator.
        kwargs: The keyword arguments to the operator.
        out: The output of the operator.
        out_dtypes: The output data types.
        node_meta: Optional FX node meta dict. Passed through to the flop
            formula as ``_node_meta`` kwarg so formulas can read annotations
            like ``sparsity_hint``.
        gpu_type: Optional datasheet device name to pin the peak FLOPS to
            instead of querying the current device.

    Returns:
        float: The estimated compute time in nanoseconds.
    """
    if func_packet in flop_registry:
        if len(out_dtypes) != 1:
            raise AssertionError(
                f"Only support single out dtype got {out_dtypes} for {func_packet}"
            )
        dtype = out_dtypes.pop()
        flop_count_func = flop_registry[func_packet]
        extra_kwargs = {}
        if node_meta is not None:
            extra_kwargs["_node_meta"] = node_meta
        flop_count = flop_count_func(*args, **kwargs, out_val=out, **extra_kwargs)
        return flops_to_ns(flop_count, dtype, gpu_type=gpu_type)
    return 0.0


def get_num_bytes(t: torch.Tensor) -> int:
    """
    Calculates the memory consumption of a tensor.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        int: The memory consumption of the tensor in bytes.
    """
    real_numel = 1
    for size, stride in zip(t.shape, t.stride()):
        # For dims with stride=0 (expanded/broadcast), only 1 element accessed
        if not statically_known_true(stride == 0):
            real_numel *= optimization_hint(size, fallback=0)

    return real_numel * t.element_size()


def get_transfer_time(flat_args_kwargs, flat_outs, gpu_type=None) -> float:  # type: ignore[no-untyped-def]
    """
    Estimates the memory transfer time of input and output tensors.

    Args:
        flat_args_kwargs (List[torch.Tensor]): The flat list of arguments and keyword arguments.
        flat_outs (List[torch.Tensor]): The flat list of outputs.
        gpu_type: Optional datasheet device name to pin the DRAM bandwidth to
            instead of querying the current device.

    Returns:
        float: The estimated memory transfer time in nanoseconds.
    """
    if gpu_type is not None:
        gpu_memory_bandwidth = datasheet_dram_bw_gbs(gpu_type)
        if gpu_memory_bandwidth is None:
            raise ValueError(f"gpu_type {gpu_type!r} not found in datasheet")
    else:
        gpu_memory_bandwidth = get_gpu_dram_gbps()
    read_bytes = sum(
        get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor)
    )
    write_bytes = sum(
        get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
    )
    counted_bytes = read_bytes + write_bytes
    # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
    transfer_time = counted_bytes / gpu_memory_bandwidth
    return transfer_time
<|endoftext|># mypy: allow-untyped-defs
import collections


Entry = collections.namedtuple("Entry", "version, hash")


def update_hash(seed, value):
    # Good old boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    return seed ^ (hash(value) + 0x9E3779B9 + (seed << 6) + (seed >> 2))


def hash_source_files(hash_value, source_files):
    for filename in source_files:
        with open(filename, "rb") as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_arguments(hash_value, build_arguments):
    for group in build_arguments:
        if group:
            for argument in group:
                hash_value = update_hash(hash_value, argument)
    return hash_value


class ExtensionVersioner:
    def __init__(self) -> None:
        self.entries = {}

    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    def bump_version_if_changed(
        self,
        name,
        source_files,
        build_arguments,
        build_directory,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
    ):
        hash_value = 0
        hash_value = hash_source_files(hash_value, source_files)
        hash_value = hash_build_arguments(hash_value, build_arguments)
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, with_sycl)
        hash_value = update_hash(hash_value, is_python_module)
        hash_value = update_hash(hash_value, is_standalone)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version
<|endoftext|>from collections.abc import Callable
from typing import TypeVar


F = TypeVar("F")


# Allows one to expose an API in a private submodule publicly as per the definition
# in PyTorch's public api policy.
#
# It is a temporary solution while we figure out if it should be the long-term solution
# or if we should amend PyTorch's public api policy. The concern is that this approach
# may not be very robust because it's not clear what __module__ is used for.
# However, both numpy and jax overwrite the __module__ attribute of their APIs
# without problem, so it seems fine.
def exposed_in(module: str) -> Callable[[F], F]:
    def wrapper(fn: F) -> F:
        fn.__module__ = module
        return fn

    return wrapper
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs
import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO

__all__ = ["FakeObject", "FakeClass", "DumpUnpickler", "main"]

class FakeObject:
    def __init__(self, module, name, args) -> None:
        self.module = module
        self.name = name
        self.args = args
        # NOTE: We don't distinguish between state never set and state set to None.
        self.state = None

    def __repr__(self) -> str:
        state_str = "" if self.state is None else f"(state={self.state!r})"
        return f"{self.module}.{self.name}{self.args!r}{state_str}"

    def __setstate__(self, state):
        self.state = state

    @staticmethod
    def pp_format(printer, obj, stream, indent, allowance, context, level) -> None:
        if not obj.args and obj.state is None:
            stream.write(repr(obj))
            return
        if obj.state is None:
            stream.write(f"{obj.module}.{obj.name}")
            printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
            return
        if not obj.args:
            stream.write(f"{obj.module}.{obj.name}()(state=\n")
            indent += printer._indent_per_level
            stream.write(" " * indent)
            printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
            stream.write(")")
            return
        raise Exception("Need to implement")  # noqa: TRY002


class FakeClass:
    def __init__(self, module, name) -> None:
        self.module = module
        self.name = name
        self.__new__ = self.fake_new  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"{self.module}.{self.name}"

    def __call__(self, *args):
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        return FakeObject(self.module, self.name, args[1:])


class DumpUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    def __init__(
            self,
            file,
            *,
            catch_invalid_utf8=False,
            **kwargs) -> None:
        super().__init__(file, **kwargs)
        self.catch_invalid_utf8 = catch_invalid_utf8

    def find_class(self, module, name):
        return FakeClass(module, name)

    def persistent_load(self, pid):
        return FakeObject("pers", "obj", (pid,))

    dispatch = dict(pickle._Unpickler.dispatch)  # type: ignore[attr-defined]

    # Custom objects in TorchScript are able to return invalid UTF-8 strings
    # from their pickle (__getstate__) functions.  Install a custom loader
    # for strings that catches the decode exception and replaces it with
    # a sentinel object.
    def load_binunicode(self) -> None:
        strlen, = struct.unpack("<I", self.read(4))  # type: ignore[attr-defined]
        if strlen > sys.maxsize:
            raise Exception("String too long.")  # noqa: TRY002
        str_bytes = self.read(strlen)  # type: ignore[attr-defined]
        obj: Any
        try:
            obj = str(str_bytes, "utf-8", "surrogatepass")
        except UnicodeDecodeError as exn:
            if not self.catch_invalid_utf8:
                raise
            obj = FakeObject("builtin", "UnicodeDecodeError", (str(exn),))
        self.append(obj)  # type: ignore[attr-defined]
    dispatch[pickle.BINUNICODE[0]] = load_binunicode  # type: ignore[assignment]

    @classmethod
    def dump(cls, in_stream, out_stream):
        value = cls(in_stream).load()
        pprint.pprint(value, stream=out_stream)
        return value


def main(argv, output_stream=None) -> int | None:
    if len(argv) != 2:
        # Don't spam stderr if not using stdout.
        if output_stream is not None:
            raise Exception("Pass argv of length 2.")  # noqa: TRY002
        sys.stderr.write("usage: show_pickle PICKLE_FILE\n")
        sys.stderr.write("  PICKLE_FILE can be any of:\n")
        sys.stderr.write("    path to a pickle file\n")
        sys.stderr.write("    file.zip@member.pkl\n")
        sys.stderr.write("    file.zip@*/pattern.*\n")
        sys.stderr.write("      (shell glob pattern for members)\n")
        sys.stderr.write("      (only first match will be shown)\n")
        return 2

    fname = argv[1]
    handle: IO[bytes]
    if "@" not in fname:
        with open(fname, "rb") as handle:
            DumpUnpickler.dump(handle, output_stream)
    else:
        zfname, mname = fname.split("@", 1)
        with zipfile.ZipFile(zfname) as zf:
            if "*" not in mname:
                with zf.open(mname) as handle:
                    DumpUnpickler.dump(handle, output_stream)
            else:
                found = False
                for info in zf.infolist():
                    if fnmatch.fnmatch(info.filename, mname):
                        with zf.open(info) as handle:
                            DumpUnpickler.dump(handle, output_stream)
                        found = True
                        break
                if not found:
                    raise Exception(f"Could not find member matching {mname} in {zfname}")  # noqa: TRY002


if __name__ == "__main__":
    # This hack works on every version of Python I've tested.
    # I've tested on the following versions:
    #   3.7.4
    if True:
        pprint.PrettyPrinter._dispatch[FakeObject.__repr__] = FakeObject.pp_format  # type: ignore[attr-defined]

    sys.exit(main(sys.argv))
<|endoftext|># NOTE! PLEASE KEEP THIS FILE *FREE* OF TORCH DEPS! IT SHOULD BE IMPORTABLE ANYWHERE.
# IF YOU FEEL AN OVERWHELMING URGE TO ADD A TORCH DEP, MAKE A TRAMPOLINE FILE A LA torch._dynamo.utils
# AND SCRUB AWAY TORCH NOTIONS THERE.
import collections
import functools
from collections import OrderedDict
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec


simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def count_label(label: str) -> None:
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1


def count(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        return fn(*args, **kwargs)

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
import os
import time
import warnings


class FileBaton:
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1, warn_after_seconds=None) -> None:
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            warn_after_seconds: The seconds to wait before showing
                lock file path to warn existing lock file.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None
        self.warn_after_seconds = warn_after_seconds

    def try_acquire(self) -> bool | None:
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self) -> None:
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        has_warned = False

        start_time = time.time()
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

            if self.warn_after_seconds is not None:
                if time.time() - start_time > self.warn_after_seconds and not has_warned:
                    warnings.warn(f'Waited on lock file "{self.lock_file_path}" for '
                                  f'{self.warn_after_seconds} seconds.', stacklevel=2)
                    has_warned = True

    def release(self) -> None:
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
<|endoftext|>"""
IndentedBuffer: a line-oriented buffer that tracks indentation as source is
appended and renders it (optionally with a line map for traceback remapping).

This is the shared primitive underneath inductor's codegen buffers
(torch._inductor.utils re-exports these names and subclasses IndentedBuffer for
its JIT/AOTI variants), dynamo's guard codegen (IndentedBufferWithPrefix), and
AOTAutograd's runtime-wrapper codegen (PySourceBuilder). It lives in torch.utils
so those layers can share it without importing across the _functorch -> _inductor
edge or pulling in the heavy torch._inductor.utils module. The cluster depends
only on the standard library.
"""

from __future__ import annotations

import contextlib
import math
import textwrap
from dataclasses import dataclass
from io import StringIO
from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class LineContext(NamedTuple):
    context: Any


@dataclass
class ValueWithLineMap:
    value: str
    line_map: list[tuple[int, LineContext]]


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        self._lines: list[DeferredLineBase | LineContext | str] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> ValueWithLineMap:
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def get_lines_ref(self):
        return self._lines

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    def prefix(self) -> str:
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        self.writeline("\n")

    def writeline(self, line: LineContext | DeferredLineBase | str) -> None:
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writeline_jit(self, line: LineContext | DeferredLineBase | str) -> None:
        """Write to JIT buffer only. On a plain IndentedBuffer, same as writeline."""
        self.writeline(line)

    def writeline_aot(self, line: LineContext | DeferredLineBase | str) -> None:
        """Write to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def splice_jit(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        """Splice to JIT buffer only. On a plain IndentedBuffer, same as splice."""
        self.splice(other_code, strip=strip)

    def splice_aot(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        """Splice to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def writelines(self, lines: Sequence[LineContext | DeferredLineBase | str]) -> None:
        for line in lines:
            self.writeline(line)

    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        self._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        self._indent -= offset

    def splice(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")

            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for s in other_code.split("\n"):
                IndentedBuffer.writeline(self, s)

    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self) -> str:
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other: Self) -> IndentedBuffer:
        if self._indent != other._indent:
            raise AssertionError(f"Indent mismatch: {self._indent} != {other._indent}")
        res = IndentedBuffer(initial_indent=self._indent)
        # TODO(rec): or should this be self.__class__(initial_indent=self._indent)?
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res

    def contains(self, new_line: DeferredLineBase | LineContext | str) -> bool:
        return new_line in self._lines


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line: str):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> str | None:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> Self:
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix: str) -> Self:
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self) -> Self:
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index: int | slice) -> Self:
        return self._new_line(self.line[index])

    def __bool__(self) -> bool:
        return bool(self.line)

    def __len__(self) -> int:
        return len(self.line)
<|endoftext|>import functools
import importlib.util
from types import ModuleType


def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False


@functools.lru_cache
def dill_available() -> bool:
    return _check_module_exists("dill")


@functools.lru_cache
def import_dill() -> ModuleType | None:
    if not dill_available():
        return None

    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    return dill
<|endoftext|>from collections.abc import Callable
from typing import Generic, TypeVar


R = TypeVar("R")


class Thunk(Generic[R]):
    """
    A simple lazy evaluation implementation that lets you delay
    execution of a function.  It properly handles releasing the
    function once it is forced.
    """

    f: Callable[[], R] | None
    r: R | None

    __slots__ = ["f", "r"]

    def __init__(self, f: Callable[[], R]) -> None:
        self.f = f
        self.r = None

    def force(self) -> R:
        if self.f is None:
            return self.r  # type: ignore[return-value]
        self.r = self.f()
        self.f = None
        return self.r
<|endoftext|># mypy: allow-untyped-defs
import copy
import glob
import importlib
import importlib.abc
import importlib.util
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import threading
import types
import collections
from pathlib import Path
import errno
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

import torch
import torch._appdirs
from ._filelock import FileLock
from ._cpp_extension_versioner import ExtensionVersioner
from typing_extensions import deprecated
from torch.torch_version import TorchVersion, Version
from torch._utils_internal import get_file_path


IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

# Torch root used below to locate compiled artifacts (lib/, include/, bin/).
# Editable installs using scikit-build-core redirect mode place these in the
# installed package directory rather than the source tree this file loads
# from, so resolve it the same way torch._utils_internal does instead of from
# this file's location.
_TORCH_PATH = get_file_path("torch")
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 20, 0)

VersionRange = tuple[tuple[int, ...], tuple[int, ...]]
VersionMap = dict[str, VersionRange]
# The following values were taken from the following GitHub gist that
# summarizes the minimum valid major versions of g++/clang++ for each supported
# CUDA version: https://gist.github.com/ax3l/9489132
# Or from include/crt/host_config.h in the CUDA SDK
# The second value is the exclusive(!) upper bound, i.e. min <= version < max
CUDA_GCC_VERSIONS: VersionMap = {
    '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
    '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.4': ((6, 0, 0), (12, 0)),
    '11.5': ((6, 0, 0), (12, 0)),
    '11.6': ((6, 0, 0), (12, 0)),
    '11.7': ((6, 0, 0), (12, 0)),
    '12.0': ((6, 0, 0), (13, 0)),
    '12.1': ((6, 0, 0), (13, 0)),
    '12.2': ((6, 0, 0), (13, 0)),
    '12.3': ((6, 0, 0), (14, 0)),
    '12.4': ((6, 0, 0), (14, 0)),
    '12.5': ((6, 0, 0), (14, 0)),
    '12.6': ((6, 0, 0), (14, 0)),
    '12.7': ((6, 0, 0), (14, 0)),
    '12.8': ((6, 0, 0), (15, 0)),
    '12.9': ((6, 0, 0), (15, 0)),
    '13.0': ((6, 0, 0), (16, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {
    '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
    '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
    '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
    '12.0': ((7, 0), (15, 0)),
    '12.1': ((7, 0), (15, 0)),
    '12.2': ((7, 0), (16, 0)),
    '12.3': ((7, 0), (16, 0)),
    '12.4': ((7, 0), (17, 0)),
    '12.5': ((7, 0), (18, 0)),
    '12.6': ((7, 0), (18, 0)),
    '12.7': ((7, 0), (19, 0)),
    '12.8': ((7, 0), (19, 0)),
    '12.9': ((7, 0), (19, 0)),
    '13.0': ((7, 0), (21, 0)),
}

__all__ = ["get_default_build_root", "check_compiler_ok_for_platform", "get_compiler_abi_compatibility_and_version", "BuildExtension",  # noqa: F822, RUF100
           "CppExtension", "CUDAExtension", "SyclExtension", "include_paths", "library_paths", "load", "load_inline", "is_ninja_available",
           "verify_ninja_availability", "remove_extension_h_precompiler_headers", "get_cxx_compiler", "check_compiler_is_gcc"]

# setuptools.command.build_ext imports Cython when available, so keep it off the
# cpp_extension import path until BuildExtension is explicitly requested.
_BUILD_EXTENSION_CLASS = None
_BUILD_EXTENSION_BASE_INITIALIZED = False
_BUILD_EXTENSION_LOCK = threading.Lock()


if TYPE_CHECKING:
    from setuptools.command.build_ext import build_ext as _LazyBuildExt
else:
    class _LazyBuildExt(setuptools.Command):
        pass


def _get_build_extension():
    global _BUILD_EXTENSION_BASE_INITIALIZED
    build_extension = _BUILD_EXTENSION_CLASS
    if build_extension is None:
        raise RuntimeError("BuildExtension class is not initialized")
    if not _BUILD_EXTENSION_BASE_INITIALIZED:
        with _BUILD_EXTENSION_LOCK:
            if not _BUILD_EXTENSION_BASE_INITIALIZED:
                from setuptools.command.build_ext import build_ext

                build_extension.__bases__ = (build_ext,)
                globals()["BuildExtension"] = build_extension
                _BUILD_EXTENSION_BASE_INITIALIZED = True
    return build_extension


def __getattr__(name):
    if name == "BuildExtension":
        return _get_build_extension()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Taken directly from python stdlib < 3.9
# See https://github.com/pytorch/pytorch/issues/48617
def _nt_quote_args(args: list[str] | None) -> list[str]:
    """Quote command-line arguments for DOS/Windows conventions.

    Just wraps every argument which contains blanks in double quotes, and
    returns a new argument list.
    """
    # Cover None-type
    if not args:
        return []
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]

def _find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        logger.warning("No CUDA runtime is found, using CUDA_HOME='%s'", cuda_home)
    return cuda_home

def _find_rocm_home() -> str | None:
    """Find the ROCm install path."""
    # Guess #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Guess #2: Support for ROCm distribution from TheRock
        # rocm-sdk-core installs everything under <site-packages>/_rocm_sdk_core
        # (include/, lib/, bin/, ...), so the module's own location is the
        # ROCM_HOME we want. Use find_spec to locate it without importing.
        spec = importlib.util.find_spec('_rocm_sdk_core')
        if spec is not None and spec.origin is not None:
            rocm_home = str(Path(spec.origin).parent.resolve())
    if rocm_home is None:
        # Guess #3
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(
                os.path.realpath(hipcc_path)))
            # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            # Guess #4
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        logger.warning("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
    return rocm_home

def _find_sycl_home() -> str | None:
    sycl_home = None
    icpx_path = shutil.which('icpx')
    # Guess 1: for source code build developer/user, we'll have icpx in PATH,
    # which will tell us the SYCL_HOME location.
    if icpx_path is not None:
        sycl_home = os.path.dirname(os.path.dirname(
            os.path.realpath(icpx_path)))

    # Guess 2: for users install Pytorch with XPU support, the sycl runtime is
    # inside intel-sycl-rt, which is automatically installed via pip dependency.
    else:
        try:
            files = importlib.metadata.files('intel-sycl-rt') or []
            for f in files:
                if f.name == "libsycl.so":
                    sycl_home = os.path.dirname(Path(f.locate()).parent.resolve())
                    break
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Trying to find SYCL_HOME from intel-sycl-rt package, but it is not installed.")
    return sycl_home

def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. '
                      'Please set it to your ROCm install root.')
    return os.path.join(ROCM_HOME, *paths)

def _join_sycl_home(*paths) -> str:
    """
    Join paths with SYCL_HOME, or raises an error if it SYCL_HOME is not found.

    This is basically a lazy way of raising an error for missing SYCL_HOME
    only once we need to get any SYCL-specific path.
    """
    if SYCL_HOME is None:
        raise OSError('SYCL runtime is not detected. Please setup the pytorch '
                      'prerequisites for Intel GPU following the instruction in '
                      'https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support '
                      'or install intel-sycl-rt via pip.')

    return os.path.join(SYCL_HOME, *paths)



def _wrap_compiler(compiler: str | list[str]) -> list[str]:
    """Prepend a compiler wrapper (ccache/sccache) if available.

    Accepts a compiler as a string or list. Always returns a list with the
    wrapper prepended, or the original value as a list if no wrapper is found.
    Disabled when TORCH_NO_COMPILER_WRAPPER is set.
    """
    if isinstance(compiler, str):
        compiler = [compiler]
    # hipcc with ccache/sccache is currently broken
    # I.e. compilation fails with
    #  sccache: caused by: Compiler not supported: "sh: 1: /usr/local/cuda/bin/nvcc: not found\n
    # sh: 1: nvcc: not found\nDevice not supported - Defaulting to AMD\n
    # failed to execute:/opt/rocm/lib/llvm/bin/clang++  -O3  -E -x c /tmp/sccachei1cosZ/testfile.c\n"
    if os.environ.get('TORCH_NO_COMPILER_WRAPPER') or IS_WINDOWS or torch.version.hip is not None:
        return compiler
    for wrapper in ('ccache', 'sccache'):
        if shutil.which(wrapper):
            return [wrapper] + compiler
    return compiler


def _windows_quote(arg: str) -> str:
    """Quote a single argument per the MSVC ``CommandLineToArgvW`` rules."""
    if arg and not any(c in arg for c in ' \t\n\v"'):
        return arg
    out = ['"']
    backslashes = 0
    for c in arg:
        if c == '\\':
            backslashes += 1
            continue
        if c == '"':
            out.append('\\' * (2 * backslashes + 1))
            out.append('"')
        else:
            out.append('\\' * backslashes)
            out.append(c)
        backslashes = 0
    out.append('\\' * (2 * backslashes))
    out.append('"')
    return ''.join(out)


def _shell_join(cmd: list[str]) -> str:
    """Quote and join ``cmd`` for the current platform's shell.

    ``shlex.join`` produces POSIX-style quoting which ninja does not parse
    correctly on Windows for paths like ``C:\\Program Files\\...``.
    """
    if IS_WINDOWS:
        return ' '.join(_windows_quote(a) for a in cmd)
    return shlex.join(cmd)


ABI_INCOMPATIBILITY_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) may be ABI-incompatible with PyTorch!"
    "Please use a compiler that is ABI-compatible with GCC 5.0 and above."
    "See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html."
    "See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6"
    "for instructions on how to install GCC 5 or higher."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
WRONG_COMPILER_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) is not compatible with the compiler Pytorch was"
    "built with for this platform, which is %s on %s. Please"
    "use %s to compile your extension. Alternatively, you may"
    "compile PyTorch from source using %s, and then you can also use"
    "%s to compile your extension."
    "See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help"
    "with compiling PyTorch from source."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
CUDA_MISMATCH_MESSAGE = (
    "The detected CUDA version (%s) mismatches the version that was used to compile"
    "PyTorch (%s). Please make sure to use the same CUDA versions."
)
CUDA_MISMATCH_WARN = (
    "The detected CUDA version (%s) has a minor version mismatch with the version that was used to compile PyTorch (%s). Most likely this shouldn't be a problem."
)
CUDA_NOT_FOUND_MESSAGE = (
    "CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH"
    "environment variable or add NVCC to your system PATH. The extension compilation will fail."
)
ROCM_HOME = _find_rocm_home() if (torch.cuda._is_compiled() and torch.version.hip) else None
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = bool(ROCM_HOME is not None and torch.version.hip is not None)
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

CUDA_HOME = _find_cuda_home() if (torch.cuda._is_compiled() and torch.version.cuda) else None
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
SYCL_HOME = _find_sycl_home() if torch.xpu._is_compiled() else None
WINDOWS_CUDA_HOME = os.environ.get('WINDOWS_CUDA_HOME')  # used for AOTI cross-compilation

# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']

MSVC_IGNORE_CUDAFE_WARNINGS = [
    'base_class_has_different_dll_interface',
    'field_without_dll_interface',
    'dll_interface_conflict_none_assumed',
    'dll_interface_conflict_dllexport_assumed'
]

COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

COMMON_HIP_FLAGS = [
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
    '-DHIPBLAS_V2',
]

if not IS_WINDOWS:
    COMMON_HIP_FLAGS.append('-fPIC')

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
    '-DHIP_ENABLE_WARP_SYNC_BUILTINS=1'
]

if IS_WINDOWS:
    # Compatibility flags, similar to those set in cmake/Dependencies.cmake.
    COMMON_HIPCC_FLAGS.append('-fms-extensions')
    # Suppress warnings about dllexport.
    COMMON_HIPCC_FLAGS.append('-Wno-ignored-attributes')


def _get_icpx_version() -> str:
    icpx = 'icx' if IS_WINDOWS else 'icpx'
    compiler_info = subprocess.check_output([icpx, '--version'])
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())
    version = ['0', '0', '0'] if match is None else list(match.groups())
    version = list(map(int, version))
    if len(version) != 3:
        raise AssertionError("Failed to parse DPC++ compiler version")
    # Aligning version format with what torch.version.xpu() returns
    return f"{version[0]}{version[1]:02}{version[2]:02}"


def _get_sycl_arch_list():
    if 'TORCH_XPU_ARCH_LIST' in os.environ:
        return os.environ.get('TORCH_XPU_ARCH_LIST')
    arch_list = torch.xpu.get_arch_list()
    # Dropping dg2* archs since they lack hardware support for fp64 and require
    # special consideration from the user. If needed these platforms can
    # be requested thru TORCH_XPU_ARCH_LIST environment variable.
    arch_list = [x for x in arch_list if not x.startswith('dg2')]
    return ','.join(arch_list)


# If arch list returned by _get_sycl_arch_list() is empty, then sycl kernels will be compiled
# for default spir64 target and avoid device specific compilations entirely. Further, kernels
# will be JIT compiled at runtime.
def _append_sycl_targets_if_missing(cflags) -> None:
    if any(flag.startswith('-fsycl-targets=') for flag in cflags):
        # do nothing: user has manually specified sycl targets
        return
    if _get_sycl_arch_list() != '':
        # AOT (spir64_gen) + JIT (spir64)
        cflags.append('-fsycl-targets=spir64_gen,spir64')
    else:
        # JIT (spir64)
        cflags.append('-fsycl-targets=spir64')

def _get_sycl_device_flags(cflags):
    # We need last occurrence of -fsycl-targets as it will be the one taking effect.
    # So searching in reversed list.
    flags = [f for f in reversed(cflags) if f.startswith('-fsycl-targets=')]
    if not flags:
        raise AssertionError("bug: -fsycl-targets should have been amended to cflags")

    arch_list = _get_sycl_arch_list()
    if arch_list != '':
        flags += [f'-Xs "-device {arch_list}"']
    return flags

_COMMON_SYCL_FLAGS = [
    '-fsycl',
]

_SYCL_DLINK_FLAGS = [
    *_COMMON_SYCL_FLAGS,
    '-fsycl-link',
    '--offload-compress',
]

JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}

min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode

def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> list[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']

def _maybe_write(filename, new_content) -> None:
    r'''
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    '''
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, 'w') as source_file:
        source_file.write(new_content)

def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


def check_compiler_ok_for_platform(compiler: str) -> bool:
    """
    Verify that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    if IS_WINDOWS:
        return True
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return False
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(compiler_path)
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except subprocess.CalledProcessError:
        # If '-v' fails, try '--version'
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            # Though on Ubuntu it's sometimes called "Ubuntu clang version"
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def get_compiler_abi_compatibility_and_version(compiler) -> tuple[bool, TorchVersion]:
    """
    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    """
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        logger.warning(WRONG_COMPILER_WARNING, compiler, _accepted_compilers_for_platform()[0], sys.platform, _accepted_compilers_for_platform()[0], compiler, compiler)
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            compiler_info = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
        version = ['0', '0', '0'] if match is None else list(match.groups())
    except (subprocess.CalledProcessError, OSError):
        logger.warning('Error checking compiler version for %s', compiler, exc_info=True)
        return (False, TorchVersion('0.0.0'))

    # convert alphanumeric string to numeric string
    # amdclang++ returns str like 0.0.0git, others return 0.0.0
    numeric_version = [re.sub(r'\D', '', v) for v in version]

    if tuple(map(int, numeric_version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(numeric_version)))

    compiler = f'{compiler} {".".join(numeric_version)}'
    logger.warning(ABI_INCOMPATIBILITY_WARNING, compiler)

    return (False, TorchVersion('.'.join(numeric_version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
    if not os.path.exists(nvcc):
        raise FileNotFoundError(f"nvcc not found at '{nvcc}'. Ensure CUDA path '{CUDA_HOME}' is correct.")

    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
    if cuda_version is None:
        return

    cuda_str_version = cuda_version.group(1)
    cuda_ver = Version(cuda_str_version)
    if torch.version.cuda is None:
        return

    torch_cuda_version = Version(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
        logger.warning(CUDA_MISMATCH_WARN, cuda_str_version, torch.version.cuda)

    if not (sys.platform.startswith('linux') and
            os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and
            _is_binary_build()):
        return

    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS

    if cuda_str_version not in cuda_compiler_bounds:
        logger.warning('There are no %s version bounds defined for CUDA version %s', compiler_name, cuda_str_version)
    else:
        min_compiler_version, max_excl_compiler_version = cuda_compiler_bounds[cuda_str_version]
        # Special case for 11.4.0, which has lower compiler bounds than 11.4.1
        if "V11.4.48" in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        min_compiler_version_str = '.'.join(map(str, min_compiler_version))
        max_excl_compiler_version_str = '.'.join(map(str, max_excl_compiler_version))

        version_bound_str = f'>={min_compiler_version_str}, <{max_excl_compiler_version_str}'

        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is less '
                f'than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is greater '
                f'than the maximum required version by CUDA {cuda_str_version}. '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )


# Specify Visual Studio C runtime library for hipcc
def _set_hipcc_runtime_lib(is_standalone, debug) -> None:
    if is_standalone:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static')
    else:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll')

def _append_sycl_std_if_no_std_present(cflags) -> None:
    if not any(flag.startswith('-sycl-std=') for flag in cflags):
        cflags.append('-sycl-std=2020')


def _wrap_sycl_host_flags(cflags):
    host_cflags = []
    host_cxx = get_cxx_compiler()
    if IS_WINDOWS:
        for flag in cflags:
            if flag.startswith("-I"):
                flag = flag.replace("\\", "\\\\").replace("-I", "/I")
            else:
                flag = flag.replace("-D", "/D")
            flag = flag.replace('"', '\\"')
            host_cflags.append(flag)
        joined_host_cflags = ' '.join(host_cflags)

        external_include = _join_sycl_home("include").replace("\\", "\\\\")

        # Some versions of DPC++ compiler pass paths to SYCL headers as user include paths (`-I`) rather
        # than system paths (`-isystem`). This makes host compiler to report warnings encountered in the
        # SYCL headers, such as deprecated warnings, even if warned API is not actually used in the program.
        # We expect that this issue will be addressed in the later version of DPC++ compiler. To workaround the
        # issue now we wrap paths to SYCL headers in `/external:I`. Warning free compilation is especially important
        # for Windows build as `/sdl` compilation flag assumes that and we will fail compilation otherwise.
        wrapped_host_cflags = [
            f"-fsycl-host-compiler={host_cxx}",
            f'-fsycl-host-compiler-options="\\"/external:I{external_include}\\" /external:W0 {joined_host_cflags}"',
        ]
    else:
        joined_host_cflags = ' '.join(cflags)
        wrapped_host_cflags = [
            f"-fsycl-host-compiler={host_cxx}",
            shlex.quote(f"-fsycl-host-compiler-options={joined_host_cflags}"),
        ]
    return wrapped_host_cflags


class BuildExtension(_LazyBuildExt):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++20``) as well as mixed
    C++/CUDA/SYCL compilation (and support for CUDA/SYCL files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages/compilers (the only expected values are ``cxx``, ``nvcc`` or
    ``sycl``) to a list of additional compiler flags to supply to the compiler.
    This makes it possible to supply different flags to the C++, CUDA and SYCL
    compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options."""
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '%s. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                logger.warning(msg, 'we could not find ninja.')
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        sycl_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not (cuda_ext and sycl_ext) and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                elif ext == '.sycl':
                    sycl_ext = True

                # This check accounts on a case when cuda and sycl sources
                # are mixed in the same extension. We can stop checking
                # sources if both are found or there is no more sources.
                if cuda_ext and sycl_ext:
                    break

            extension = next(extension_iter, None)

        if sycl_ext:
            if not self.use_ninja:
                raise AssertionError("ninja is required to build sycl extensions.")

        if cuda_ext and not IS_HIP_EXTENSION:
            _check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx', 'nvcc' and 'sycl' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx', 'nvcc' or 'sycl' is
            # passed to extra_compile_args in CUDAExtension or SyclExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc', 'sycl']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')

            if IS_HIP_EXTENSION:
                self._hipify_compile_flags(extension)

            if extension.py_limited_api:
                # compile any extension that has passed in py_limited_api to the
                # Extension constructor with the Py_LIMITED_API flag set to our
                # min supported CPython version.
                # See https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
                self._add_compile_flag(extension, f'-DPy_LIMITED_API={min_supported_cpython}')
            self._define_torch_extension_name(extension)

            if 'nvcc_dlink' in extension.extra_compile_args:
                if not self.use_ninja:
                    raise AssertionError(
                        f"With dlink=True, ninja is required to build cuda extension {extension.name}."
                    )

        # Register .cu, .cuh, .hip, .mm and .sycl as valid source extensions.
        # NOTE: At the moment .sycl is not a standard extension for SYCL supported
        # by compiler. Here we introduce a torch level convention that SYCL sources
        # should have .sycl file extension.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip', '.sycl']
        if torch.backends.mps.is_built():
            self.compiler.src_extensions += ['.mm']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh', '.hip']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++20'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags + _get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                _ccbin is not None
                and not any(flag.startswith(('-ccbin', '--compiler-bindir')) for flag in cflags)
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths) -> None:
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', _wrap_compiler(nvcc))
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if IS_HIP_EXTENSION:
                        cflags = COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                else:
                    self.compiler.set_executable('compiler_so', _wrap_compiler(list(original_compiler)))
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']
                if IS_HIP_EXTENSION:
                    cflags = COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567  # codespell:ignore
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc/sycl to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
                cuda_dlink_post_cflags = [shlex.quote(f) for f in cuda_dlink_post_cflags]
            else:
                cuda_dlink_post_cflags = None

            sycl_post_cflags = None
            sycl_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = extra_cc_cflags + common_cflags + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs['sycl']
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = extra_cc_cflags + common_cflags + post_cflags
                append_std17_if_no_std_present(host_cflags)
                # escaping quoted arguments to pass them thru SYCL compiler
                icpx_version = _get_icpx_version()
                if int(icpx_version) >= 20250200:
                    host_cflags = [item.replace('"', '\\"') for item in host_cflags]
                else:
                    host_cflags = [item.replace('"', '\\\\"') for item in host_cflags]
                # Note the order: shlex.quote sycl_flags first, _wrap_sycl_host_flags
                # second. Reason is that sycl host flags are quoted, space containing
                # strings passed to SYCL compiler.
                sycl_cflags = [shlex.quote(f) for f in sycl_cflags]
                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = [shlex.quote(f) for f in sycl_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (COMMON_NVCC_FLAGS +
                    cflags + _get_cuda_arch_flags(cflags))

        def win_hip_flags(cflags):
            return (COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + _get_rocm_arch_flags(cflags))

        def win_filter_msvc_include_dirs(pp_opts) -> list[str]:
            """Filter out MSVC include dirs from pp_opts for oneAPI 2025.3+."""
            # oneAPI 2025.3+ changed include path ordering to match MSVC behavior.
            # Filter out MSVC headers to avoid conflicting declarations with oneAPI's std headers.
            icpx_version = int(_get_icpx_version())
            if icpx_version >= 20250300:
                vc_tools_dir = os.path.normcase(os.environ.get('VCToolsInstallDir', ''))
                if vc_tools_dir:
                    pp_opts = [
                        path for path in pp_opts
                        if vc_tools_dir not in os.path.normcase(path)
                    ]
            return pp_opts

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')  # codespell:ignore
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        if IS_HIP_EXTENSION:
                            nvcc = _get_hipcc_path()
                        else:
                            nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        if IS_HIP_EXTENSION:
                            cflags = win_hip_flags(cflags)
                        else:
                            cflags = win_cuda_flags(cflags) + ['-std=c++20', '--use-local-env']
                            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                                cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        cmd = _wrap_compiler([nvcc, '-c', src, '-o', obj] + include_list + cflags)
                    else:
                        if isinstance(self.cflags, dict):
                            cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                            append_std17_if_no_std_present(cflags)
                            cmd += cflags
                        elif isinstance(self.cflags, list):
                            cflags = COMMON_MSVC_FLAGS + self.cflags
                            append_std17_if_no_std_present(cflags)
                            cmd += cflags
                        cmd = _wrap_compiler(cmd)

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources,
                                   output_dir=None,
                                   macros=None,
                                   include_dirs=None,
                                   debug=0,
                                   extra_preargs=None,
                                   extra_postargs=None,
                                   depends=None,
                                   is_standalone=False):
            if not self.compiler.initialized:
                self.compiler.initialize()
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any.
            # For ninja build, the build location is not local, but instead, the build happens
            # in a script-created build folder. Thus, relative paths lose their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here.
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            # Replace space with \ when using hipcc (hipcc passes includes to clang without ""s so clang sees space in include paths as new argument)
            if IS_HIP_EXTENSION:
                pp_opts = ["-I{}".format(s[2:].replace(" ", "\\")) if s.startswith('-I') else s for s in pp_opts]
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            cflags = cflags + common_cflags + pp_opts + COMMON_MSVC_FLAGS
            if IS_HIP_EXTENSION:
                _set_hipcc_runtime_lib(is_standalone, debug)
                common_cflags.extend(COMMON_HIP_FLAGS)
            else:
                common_cflags.extend(COMMON_MSVC_FLAGS)
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['-std=c++20']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                if not IS_HIP_EXTENSION:
                    cuda_cflags.append('--use-local-env')
                    for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                        cuda_cflags.append('-Xcudafe')
                        cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = win_hip_flags(cuda_post_cflags)
                else:
                    cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None

            sycl_cflags = None
