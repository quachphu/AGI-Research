        if self.__origin__ is Generic:
            if Protocol in bases:
                return ()
            i = bases.index(self)
            for b in bases[i+1:]:
                if isinstance(b, _BaseGenericAlias) and b is not self:
                    return ()
        return (self.__origin__,)


# _nparams is the number of accepted parameters, e.g. 0 for Hashable,
# 1 for List and 2 for Dict.  It may be -1 if variable number of
# parameters are accepted (needs custom __getitem__).

class _SpecialGenericAlias(_BaseGenericAlias, _root=True):
    def __init__(self, origin, nparams, *, inst=True, name=None):
        if name is None:
            name = origin.__name__
        super().__init__(origin, inst=inst, name=name)
        self._nparams = nparams
        if origin.__module__ == 'builtins':
            self.__doc__ = f'A generic version of {origin.__qualname__}.'
        else:
            self.__doc__ = f'A generic version of {origin.__module__}.{origin.__qualname__}.'

    @_tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        msg = "Parameters to generic types must be types."
        params = tuple(_type_check(p, msg) for p in params)
        _check_generic(self, params, self._nparams)
        return self.copy_with(params)

    def copy_with(self, params):
        return _GenericAlias(self.__origin__, params,
                             name=self._name, inst=self._inst)

    def __repr__(self):
        return 'typing.' + self._name

    def __subclasscheck__(self, cls):
        if isinstance(cls, _SpecialGenericAlias):
            return issubclass(cls.__origin__, self.__origin__)
        if not isinstance(cls, _GenericAlias):
            return issubclass(cls, self.__origin__)
        return super().__subclasscheck__(cls)

    def __reduce__(self):
        return self._name

    def __or__(self, right):
        return Union[self, right]

    def __ror__(self, left):
        return Union[left, self]

class _CallableGenericAlias(_GenericAlias, _root=True):
    def __repr__(self):
        assert self._name == 'Callable'
        args = self.__args__
        if len(args) == 2 and _is_param_expr(args[0]):
            return super().__repr__()
        return (f'typing.Callable'
                f'[[{", ".join([_type_repr(a) for a in args[:-1]])}], '
                f'{_type_repr(args[-1])}]')

    def __reduce__(self):
        args = self.__args__
        if not (len(args) == 2 and _is_param_expr(args[0])):
            args = list(args[:-1]), args[-1]
        return operator.getitem, (Callable, args)


class _CallableType(_SpecialGenericAlias, _root=True):
    def copy_with(self, params):
        return _CallableGenericAlias(self.__origin__, params,
                                     name=self._name, inst=self._inst,
                                     _typevar_types=(TypeVar, ParamSpec),
                                     _paramspec_tvars=True)

    def __getitem__(self, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Callable must be used as "
                            "Callable[[arg, ...], result].")
        args, result = params
        # This relaxes what args can be on purpose to allow things like
        # PEP 612 ParamSpec.  Responsibility for whether a user is using
        # Callable[...] properly is deferred to static type checkers.
        if isinstance(args, list):
            params = (tuple(args), result)
        else:
            params = (args, result)
        return self.__getitem_inner__(params)

    @_tp_cache
    def __getitem_inner__(self, params):
        args, result = params
        msg = "Callable[args, result]: result must be a type."
        result = _type_check(result, msg)
        if args is Ellipsis:
            return self.copy_with((_TypingEllipsis, result))
        if not isinstance(args, tuple):
            args = (args,)
        args = tuple(_type_convert(arg) for arg in args)
        params = args + (result,)
        return self.copy_with(params)


class _TupleType(_SpecialGenericAlias, _root=True):
    @_tp_cache
    def __getitem__(self, params):
        if params == ():
            return self.copy_with((_TypingEmpty,))
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 2 and params[1] is ...:
            msg = "Tuple[t, ...]: t must be a type."
            p = _type_check(params[0], msg)
            return self.copy_with((p, _TypingEllipsis))
        msg = "Tuple[t0, t1, ...]: each t must be a type."
        params = tuple(_type_check(p, msg) for p in params)
        return self.copy_with(params)


class _UnionGenericAlias(_GenericAlias, _root=True):
    def copy_with(self, params):
        return Union[params]

    def __eq__(self, other):
        if not isinstance(other, (_UnionGenericAlias, types.UnionType)):
            return NotImplemented
        return set(self.__args__) == set(other.__args__)

    def __hash__(self):
        return hash(frozenset(self.__args__))

    def __repr__(self):
        args = self.__args__
        if len(args) == 2:
            if args[0] is type(None):
                return f'typing.Optional[{_type_repr(args[1])}]'
            elif args[1] is type(None):
                return f'typing.Optional[{_type_repr(args[0])}]'
        return super().__repr__()

    def __instancecheck__(self, obj):
        return self.__subclasscheck__(type(obj))

    def __subclasscheck__(self, cls):
        for arg in self.__args__:
            if issubclass(cls, arg):
                return True

    def __reduce__(self):
        func, (origin, args) = super().__reduce__()
        return func, (Union, args)


def _value_and_type_iter(parameters):
    return ((p, type(p)) for p in parameters)


class _LiteralGenericAlias(_GenericAlias, _root=True):

    def __eq__(self, other):
        if not isinstance(other, _LiteralGenericAlias):
            return NotImplemented

        return set(_value_and_type_iter(self.__args__)) == set(_value_and_type_iter(other.__args__))

    def __hash__(self):
        return hash(frozenset(_value_and_type_iter(self.__args__)))


class _ConcatenateGenericAlias(_GenericAlias, _root=True):
    def copy_with(self, params):
        if isinstance(params[-1], (list, tuple)):
            return (*params[:-1], *params[-1])
        if isinstance(params[-1], _ConcatenateGenericAlias):
            params = (*params[:-1], *params[-1].__args__)
        elif not isinstance(params[-1], ParamSpec):
            raise TypeError("The last parameter to Concatenate should be a "
                            "ParamSpec variable.")
        return super().copy_with(params)


class Generic:
    """Abstract base class for generic types.

    A generic type is typically declared by inheriting from
    this class parameterized with one or more type variables.
    For example, a generic mapping type might be defined as::

      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.

    This class can then be used as follows::

      def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default
    """
    __slots__ = ()
    _is_protocol = False

    @_tp_cache
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        if not params and cls is not Tuple:
            raise TypeError(
                f"Parameter list to {cls.__qualname__}[...] cannot be empty")
        params = tuple(_type_convert(p) for p in params)
        if cls in (Generic, Protocol):
            # Generic and Protocol can only be subscripted with unique type variables.
            if not all(isinstance(p, (TypeVar, ParamSpec)) for p in params):
                raise TypeError(
                    f"Parameters to {cls.__name__}[...] must all be type variables "
                    f"or parameter specification variables.")
            if len(set(params)) != len(params):
                raise TypeError(
                    f"Parameters to {cls.__name__}[...] must all be unique")
        else:
            # Subscripting a regular Generic subclass.
            if any(isinstance(t, ParamSpec) for t in cls.__parameters__):
                params = _prepare_paramspec_params(cls, params)
            else:
                _check_generic(cls, params, len(cls.__parameters__))
        return _GenericAlias(cls, params,
                             _typevar_types=(TypeVar, ParamSpec),
                             _paramspec_tvars=True)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        tvars = []
        if '__orig_bases__' in cls.__dict__:
            error = Generic in cls.__orig_bases__
        else:
            error = Generic in cls.__bases__ and cls.__name__ != 'Protocol'
        if error:
            raise TypeError("Cannot inherit from plain Generic")
        if '__orig_bases__' in cls.__dict__:
            tvars = _collect_type_vars(cls.__orig_bases__, (TypeVar, ParamSpec))
            # Look for Generic[T1, ..., Tn].
            # If found, tvars must be a subset of it.
            # If not found, tvars is it.
            # Also check for and reject plain Generic,
            # and reject multiple Generic[...].
            gvars = None
            for base in cls.__orig_bases__:
                if (isinstance(base, _GenericAlias) and
                        base.__origin__ is Generic):
                    if gvars is not None:
                        raise TypeError(
                            "Cannot inherit from Generic[...] multiple types.")
                    gvars = base.__parameters__
            if gvars is not None:
                tvarset = set(tvars)
                gvarset = set(gvars)
                if not tvarset <= gvarset:
                    s_vars = ', '.join(str(t) for t in tvars if t not in gvarset)
                    s_args = ', '.join(str(g) for g in gvars)
                    raise TypeError(f"Some type variables ({s_vars}) are"
                                    f" not listed in Generic[{s_args}]")
                tvars = gvars
        cls.__parameters__ = tuple(tvars)


class _TypingEmpty:
    """Internal placeholder for () or []. Used by TupleMeta and CallableMeta
    to allow empty list/tuple in specific places, without allowing them
    to sneak in where prohibited.
    """


class _TypingEllipsis:
    """Internal placeholder for ... (ellipsis)."""


_TYPING_INTERNALS = ['__parameters__', '__orig_bases__',  '__orig_class__',
                     '_is_protocol', '_is_runtime_protocol']

_SPECIAL_NAMES = ['__abstractmethods__', '__annotations__', '__dict__', '__doc__',
                  '__init__', '__module__', '__new__', '__slots__',
                  '__subclasshook__', '__weakref__', '__class_getitem__']

# These special attributes will be not collected as protocol members.
EXCLUDED_ATTRIBUTES = _TYPING_INTERNALS + _SPECIAL_NAMES + ['_MutableMapping__marker']


def _get_protocol_attrs(cls):
    """Collect protocol members from a protocol class objects.

    This includes names actually defined in the class dictionary, as well
    as names that appear in annotations. Special names (above) are skipped.
    """
    attrs = set()
    for base in cls.__mro__[:-1]:  # without object
        if base.__name__ in ('Protocol', 'Generic'):
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in list(base.__dict__.keys()) + list(annotations.keys()):
            if not attr.startswith('_abc_') and attr not in EXCLUDED_ATTRIBUTES:
                attrs.add(attr)
    return attrs


def _is_callable_members_only(cls):
    # PEP 544 prohibits using issubclass() with protocols that have non-method members.
    return all(callable(getattr(cls, attr, None)) for attr in _get_protocol_attrs(cls))


def _no_init_or_replace_init(self, *args, **kwargs):
    cls = type(self)

    if cls._is_protocol:
        raise TypeError('Protocols cannot be instantiated')

    # Already using a custom `__init__`. No need to calculate correct
    # `__init__` to call. This can lead to RecursionError. See bpo-45121.
    if cls.__init__ is not _no_init_or_replace_init:
        return

    # Initially, `__init__` of a protocol subclass is set to `_no_init_or_replace_init`.
    # The first instantiation of the subclass will call `_no_init_or_replace_init` which
    # searches for a proper new `__init__` in the MRO. The new `__init__`
    # replaces the subclass' old `__init__` (ie `_no_init_or_replace_init`). Subsequent
    # instantiation of the protocol subclass will thus use the new
    # `__init__` and no longer call `_no_init_or_replace_init`.
    for base in cls.__mro__:
        init = base.__dict__.get('__init__', _no_init_or_replace_init)
        if init is not _no_init_or_replace_init:
            cls.__init__ = init
            break
    else:
        # should not happen
        cls.__init__ = object.__init__

    cls.__init__(self, *args, **kwargs)


def _caller(depth=1, default='__main__'):
    try:
        return sys._getframe(depth + 1).f_globals.get('__name__', default)
    except (AttributeError, ValueError):  # For platforms without _getframe()
        return None


def _allow_reckless_class_checks(depth=3):
    """Allow instance and class checks for special stdlib modules.

    The abc and functools modules indiscriminately call isinstance() and
    issubclass() on the whole MRO of a user class, which may contain protocols.
    """
    try:
        return sys._getframe(depth).f_globals['__name__'] in ['abc', 'functools']
    except (AttributeError, ValueError):  # For platforms without _getframe().
        return True


_PROTO_ALLOWLIST = {
    'collections.abc': [
        'Callable', 'Awaitable', 'Iterable', 'Iterator', 'AsyncIterable',
        'Hashable', 'Sized', 'Container', 'Collection', 'Reversible',
    ],
    'contextlib': ['AbstractContextManager', 'AbstractAsyncContextManager'],
}


class _ProtocolMeta(ABCMeta):
    # This metaclass is really unfortunate and exists only because of
    # the lack of __instancehook__.
    def __instancecheck__(cls, instance):
        # We need this method for situations where attributes are
        # assigned in __init__.
        if (
            getattr(cls, '_is_protocol', False) and
            not getattr(cls, '_is_runtime_protocol', False) and
            not _allow_reckless_class_checks(depth=2)
        ):
            raise TypeError("Instance and class checks can only be used with"
                            " @runtime_checkable protocols")

        if ((not getattr(cls, '_is_protocol', False) or
                _is_callable_members_only(cls)) and
                issubclass(instance.__class__, cls)):
            return True
        if cls._is_protocol:
            if all(hasattr(instance, attr) and
                    # All *methods* can be blocked by setting them to None.
                    (not callable(getattr(cls, attr, None)) or
                     getattr(instance, attr) is not None)
                    for attr in _get_protocol_attrs(cls)):
                return True
        return super().__instancecheck__(instance)


class Protocol(Generic, metaclass=_ProtocolMeta):
    """Base class for protocol classes.

    Protocol classes are defined as::

        class Proto(Protocol):
            def meth(self) -> int:
                ...

    Such classes are primarily used with static type checkers that recognize
    structural subtyping (static duck-typing), for example::

        class C:
            def meth(self) -> int:
                return 0

        def func(x: Proto) -> int:
            return x.meth()

        func(C())  # Passes static type check

    See PEP 544 for details. Protocol classes decorated with
    @typing.runtime_checkable act as simple-minded runtime protocols that check
    only the presence of given attributes, ignoring their type signatures.
    Protocol classes can be generic, they are defined as::

        class GenProto(Protocol[T]):
            def meth(self) -> T:
                ...
    """
    __slots__ = ()
    _is_protocol = True
    _is_runtime_protocol = False

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        # Determine if this is a protocol or a concrete subclass.
        if not cls.__dict__.get('_is_protocol', False):
            cls._is_protocol = any(b is Protocol for b in cls.__bases__)

        # Set (or override) the protocol subclass hook.
        def _proto_hook(other):
            if not cls.__dict__.get('_is_protocol', False):
                return NotImplemented

            # First, perform various sanity checks.
            if not getattr(cls, '_is_runtime_protocol', False):
                if _allow_reckless_class_checks():
                    return NotImplemented
                raise TypeError("Instance and class checks can only be used with"
                                " @runtime_checkable protocols")
            if not _is_callable_members_only(cls):
                if _allow_reckless_class_checks():
                    return NotImplemented
                raise TypeError("Protocols with non-method members"
                                " don't support issubclass()")
            if not isinstance(other, type):
                # Same error message as for issubclass(1, int).
                raise TypeError('issubclass() arg 1 must be a class')

            # Second, perform the actual structural compatibility check.
            for attr in _get_protocol_attrs(cls):
                for base in other.__mro__:
                    # Check if the members appears in the class dictionary...
                    if attr in base.__dict__:
                        if base.__dict__[attr] is None:
                            return NotImplemented
                        break

                    # ...or in annotations, if it is a sub-protocol.
                    annotations = getattr(base, '__annotations__', {})
                    if (isinstance(annotations, collections.abc.Mapping) and
                            attr in annotations and
                            issubclass(other, Generic) and other._is_protocol):
                        break
                else:
                    return NotImplemented
            return True

        if '__subclasshook__' not in cls.__dict__:
            cls.__subclasshook__ = _proto_hook

        # We have nothing more to do for non-protocols...
        if not cls._is_protocol:
            return

        # ... otherwise check consistency of bases, and prohibit instantiation.
        for base in cls.__bases__:
            if not (base in (object, Generic) or
                    base.__module__ in _PROTO_ALLOWLIST and
                    base.__name__ in _PROTO_ALLOWLIST[base.__module__] or
                    issubclass(base, Generic) and base._is_protocol):
                raise TypeError('Protocols can only inherit from other'
                                ' protocols, got %r' % base)
        cls.__init__ = _no_init_or_replace_init


class _AnnotatedAlias(_GenericAlias, _root=True):
    """Runtime representation of an annotated type.

    At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
    with extra annotations. The alias behaves like a normal typing alias,
    instantiating is the same as instantiating the underlying type, binding
    it to types is also the same.
    """
    def __init__(self, origin, metadata):
        if isinstance(origin, _AnnotatedAlias):
            metadata = origin.__metadata__ + metadata
            origin = origin.__origin__
        super().__init__(origin, origin)
        self.__metadata__ = metadata

    def copy_with(self, params):
        assert len(params) == 1
        new_type = params[0]
        return _AnnotatedAlias(new_type, self.__metadata__)

    def __repr__(self):
        return "typing.Annotated[{}, {}]".format(
            _type_repr(self.__origin__),
            ", ".join(repr(a) for a in self.__metadata__)
        )

    def __reduce__(self):
        return operator.getitem, (
            Annotated, (self.__origin__,) + self.__metadata__
        )

    def __eq__(self, other):
        if not isinstance(other, _AnnotatedAlias):
            return NotImplemented
        return (self.__origin__ == other.__origin__
                and self.__metadata__ == other.__metadata__)

    def __hash__(self):
        return hash((self.__origin__, self.__metadata__))

    def __getattr__(self, attr):
        if attr in {'__name__', '__qualname__'}:
            return 'Annotated'
        return super().__getattr__(attr)


class Annotated:
    """Add context specific metadata to a type.

    Example: Annotated[int, runtime_check.Unsigned] indicates to the
    hypothetical runtime_check module that this type is an unsigned int.
    Every other consumer of this type can ignore this metadata and treat
    this type as int.

    The first argument to Annotated must be a valid type.

    Details:

    - It's an error to call `Annotated` with less than two arguments.
    - Nested Annotated are flattened::

        Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

    - Instantiating an annotated type is equivalent to instantiating the
    underlying type::

        Annotated[C, Ann1](5) == C(5)

    - Annotated can be used as a generic type alias::

        Optimized = Annotated[T, runtime.Optimize()]
        Optimized[int] == Annotated[int, runtime.Optimize()]

        OptimizedList = Annotated[List[T], runtime.Optimize()]
        OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type Annotated cannot be instantiated.")

    @_tp_cache
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) < 2:
            raise TypeError("Annotated[...] should be used "
                            "with at least two arguments (a type and an "
                            "annotation).")
        msg = "Annotated[t, ...]: t must be a type."
        origin = _type_check(params[0], msg, allow_special_forms=True)
        metadata = tuple(params[1:])
        return _AnnotatedAlias(origin, metadata)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "Cannot subclass {}.Annotated".format(cls.__module__)
        )


def runtime_checkable(cls):
    """Mark a protocol class as a runtime protocol.

    Such protocol can be used with isinstance() and issubclass().
    Raise TypeError if applied to a non-protocol class.
    This allows a simple-minded structural check very similar to
    one trick ponies in collections.abc such as Iterable.
    For example::

        @runtime_checkable
        class Closable(Protocol):
            def close(self): ...

        assert isinstance(open('/some/file'), Closable)

    Warning: this will check only the presence of the required methods,
    not their type signatures!
    """
    if not issubclass(cls, Generic) or not cls._is_protocol:
        raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                        ' got %r' % cls)
    cls._is_runtime_protocol = True
    return cls


def cast(typ, val):
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val


def _get_defaults(func):
    """Internal helper to extract the default arguments, by name."""
    try:
        code = func.__code__
    except AttributeError:
        # Some built-in functions don't have __code__, __defaults__, etc.
        return {}
    pos_count = code.co_argcount
    arg_names = code.co_varnames
    arg_names = arg_names[:pos_count]
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__
    res = dict(kwdefaults) if kwdefaults else {}
    pos_offset = pos_count - len(defaults)
    for name, value in zip(arg_names[pos_offset:], defaults):
        assert name not in res
        res[name] = value
    return res


_allowed_types = (types.FunctionType, types.BuiltinFunctionType,
                  types.MethodType, types.ModuleType,
                  WrapperDescriptorType, MethodWrapperType, MethodDescriptorType)


def get_type_hints(obj, globalns=None, localns=None, include_extras=False):
    """Return type hints for an object.

    This is often the same as obj.__annotations__, but it handles
    forward references encoded as string literals, adds Optional[t] if a
    default value equal to None is set and recursively replaces all
    'Annotated[T, ...]' with 'T' (unless 'include_extras=True').

    The argument may be a module, class, method, or function. The annotations
    are returned as a dictionary. For classes, annotations include also
    inherited members.

    TypeError is raised if the argument is not of a type that can contain
    annotations, and an empty dictionary is returned if no annotations are
    present.

    BEWARE -- the behavior of globalns and localns is counterintuitive
    (unless you are familiar with how eval() and exec() work).  The
    search order is locals first, then globals.

    - If no dict arguments are passed, an attempt is made to use the
      globals from obj (or the respective module's globals for classes),
      and these are also used as the locals.  If the object does not appear
      to have globals, an empty dictionary is used.  For classes, the search
      order is globals first then locals.

    - If one dict argument is passed, it is used for both globals and
      locals.

    - If two dict arguments are passed, they specify globals and
      locals, respectively.
    """

    if getattr(obj, '__no_type_check__', None):
        return {}
    # Classes require a special treatment.
    if isinstance(obj, type):
        hints = {}
        for base in reversed(obj.__mro__):
            if globalns is None:
                base_globals = getattr(sys.modules.get(base.__module__, None), '__dict__', {})
            else:
                base_globals = globalns
            ann = base.__dict__.get('__annotations__', {})
            if isinstance(ann, types.GetSetDescriptorType):
                ann = {}
            base_locals = dict(vars(base)) if localns is None else localns
            if localns is None and globalns is None:
                # This is surprising, but required.  Before Python 3.10,
                # get_type_hints only evaluated the globalns of
                # a class.  To maintain backwards compatibility, we reverse
                # the globalns and localns order so that eval() looks into
                # *base_globals* first rather than *base_locals*.
                # This only affects ForwardRefs.
                base_globals, base_locals = base_locals, base_globals
            for name, value in ann.items():
                if value is None:
                    value = type(None)
                if isinstance(value, str):
                    value = ForwardRef(value, is_argument=False, is_class=True)
                value = _eval_type(value, base_globals, base_locals)
                hints[name] = value
        return hints if include_extras else {k: _strip_annotations(t) for k, t in hints.items()}

    if globalns is None:
        if isinstance(obj, types.ModuleType):
            globalns = obj.__dict__
        else:
            nsobj = obj
            # Find globalns for the unwrapped object.
            while hasattr(nsobj, '__wrapped__'):
                nsobj = nsobj.__wrapped__
            globalns = getattr(nsobj, '__globals__', {})
        if localns is None:
            localns = globalns
    elif localns is None:
        localns = globalns
    hints = getattr(obj, '__annotations__', None)
    if hints is None:
        # Return empty annotations for something that _could_ have them.
        if isinstance(obj, _allowed_types):
            return {}
        else:
            raise TypeError('{!r} is not a module, class, method, '
                            'or function.'.format(obj))
    defaults = _get_defaults(obj)
    hints = dict(hints)
    for name, value in hints.items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            # class-level forward refs were handled above, this must be either
            # a module-level annotation or a function argument annotation
            value = ForwardRef(
                value,
                is_argument=not isinstance(obj, types.ModuleType),
                is_class=False,
            )
        value = _eval_type(value, globalns, localns)
        if name in defaults and defaults[name] is None:
            value = Optional[value]
        hints[name] = value
    return hints if include_extras else {k: _strip_annotations(t) for k, t in hints.items()}


def _strip_annotations(t):
    """Strips the annotations from a given type.
    """
    if isinstance(t, _AnnotatedAlias):
        return _strip_annotations(t.__origin__)
    if isinstance(t, _GenericAlias):
        stripped_args = tuple(_strip_annotations(a) for a in t.__args__)
        if stripped_args == t.__args__:
            return t
        return t.copy_with(stripped_args)
    if isinstance(t, GenericAlias):
        stripped_args = tuple(_strip_annotations(a) for a in t.__args__)
        if stripped_args == t.__args__:
            return t
        return GenericAlias(t.__origin__, stripped_args)
    if isinstance(t, types.UnionType):
        stripped_args = tuple(_strip_annotations(a) for a in t.__args__)
        if stripped_args == t.__args__:
            return t
        return functools.reduce(operator.or_, stripped_args)

    return t


def get_origin(tp):
    """Get the unsubscripted version of a type.

    This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
    and Annotated. Return None for unsupported types. Examples::

        get_origin(Literal[42]) is Literal
        get_origin(int) is None
        get_origin(ClassVar[int]) is ClassVar
        get_origin(Generic) is Generic
        get_origin(Generic[T]) is Generic
        get_origin(Union[T, int]) is Union
        get_origin(List[Tuple[T, T]][int]) == list
        get_origin(P.args) is P
    """
    if isinstance(tp, _AnnotatedAlias):
        return Annotated
    if isinstance(tp, (_BaseGenericAlias, GenericAlias,
                       ParamSpecArgs, ParamSpecKwargs)):
        return tp.__origin__
    if tp is Generic:
        return Generic
    if isinstance(tp, types.UnionType):
        return types.UnionType
    return None


def get_args(tp):
    """Get type arguments with all substitutions performed.

    For unions, basic simplifications used by Union constructor are performed.
    Examples::
        get_args(Dict[str, int]) == (str, int)
        get_args(int) == ()
        get_args(Union[int, Union[T, int], str][int]) == (int, str)
        get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        get_args(Callable[[], T][int]) == ([], int)
    """
    if isinstance(tp, _AnnotatedAlias):
        return (tp.__origin__,) + tp.__metadata__
    if isinstance(tp, (_GenericAlias, GenericAlias)):
        res = tp.__args__
        if (tp.__origin__ is collections.abc.Callable
                and not (len(res) == 2 and _is_param_expr(res[0]))):
            res = (list(res[:-1]), res[-1])
        return res
    if isinstance(tp, types.UnionType):
        return tp.__args__
    return ()


def is_typeddict(tp):
    """Check if an annotation is a TypedDict class

    For example::
        class Film(TypedDict):
            title: str
            year: int

        is_typeddict(Film)  # => True
        is_typeddict(Union[list, str])  # => False
    """
    return isinstance(tp, _TypedDictMeta)


def no_type_check(arg):
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods and classes defined in that class
    (but not to methods defined in its superclasses or subclasses).

    This mutates the function(s) or class(es) in place.
    """
    if isinstance(arg, type):
        arg_attrs = arg.__dict__.copy()
        for attr, val in arg.__dict__.items():
            if val in arg.__bases__ + (arg,):
                arg_attrs.pop(attr)
        for obj in arg_attrs.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
            if isinstance(obj, type):
                no_type_check(obj)
    try:
        arg.__no_type_check__ = True
    except TypeError:  # built-in classes
        pass
    return arg


def no_type_check_decorator(decorator):
    """Decorator to give another decorator the @no_type_check effect.

    This wraps the decorator with something that wraps the decorated
    function in @no_type_check.
    """

    @functools.wraps(decorator)
    def wrapped_decorator(*args, **kwds):
        func = decorator(*args, **kwds)
        func = no_type_check(func)
        return func

    return wrapped_decorator


def _overload_dummy(*args, **kwds):
    """Helper for @overload to raise when called."""
    raise NotImplementedError(
        "You should not call an overloaded function. "
        "A series of @overload-decorated functions "
        "outside a stub module should always be followed "
        "by an implementation that is not @overload-ed.")


def overload(func):
    """Decorator for overloaded functions/methods.

    In a stub file, place two or more stub definitions for the same
    function in a row, each decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...

    In a non-stub file (i.e. a regular .py file), do the same but
    follow it with an implementation.  The implementation should *not*
    be decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...
      def utf8(value):
          # implementation goes here
    """
    return _overload_dummy


def final(f):
    """A decorator to indicate final methods and final classes.

    Use this decorator to indicate to type checkers that the decorated
    method cannot be overridden, and decorated class cannot be subclassed.
    For example:

      class Base:
          @final
          def done(self) -> None:
              ...
      class Sub(Base):
          def done(self) -> None:  # Error reported by type checker
                ...

      @final
      class Leaf:
          ...
      class Other(Leaf):  # Error reported by type checker
          ...

    There is no runtime checking of these properties.
    """
    return f


# Some unconstrained type variables.  These are used by the container types.
# (These are not for export.)
T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
T_co = TypeVar('T_co', covariant=True)  # Any type covariant containers.
V_co = TypeVar('V_co', covariant=True)  # Any type covariant containers.
VT_co = TypeVar('VT_co', covariant=True)  # Value type covariant containers.
T_contra = TypeVar('T_contra', contravariant=True)  # Ditto contravariant.
# Internal type variable used for Type[].
CT_co = TypeVar('CT_co', covariant=True, bound=type)

# A useful type variable with constraints.  This represents string types.
# (This one *is* for export!)
AnyStr = TypeVar('AnyStr', bytes, str)


# Various ABCs mimicking those in collections.abc.
_alias = _SpecialGenericAlias

Hashable = _alias(collections.abc.Hashable, 0)  # Not generic.
Awaitable = _alias(collections.abc.Awaitable, 1)
Coroutine = _alias(collections.abc.Coroutine, 3)
AsyncIterable = _alias(collections.abc.AsyncIterable, 1)
AsyncIterator = _alias(collections.abc.AsyncIterator, 1)
Iterable = _alias(collections.abc.Iterable, 1)
Iterator = _alias(collections.abc.Iterator, 1)
Reversible = _alias(collections.abc.Reversible, 1)
Sized = _alias(collections.abc.Sized, 0)  # Not generic.
Container = _alias(collections.abc.Container, 1)
Collection = _alias(collections.abc.Collection, 1)
Callable = _CallableType(collections.abc.Callable, 2)
Callable.__doc__ = \
    """Callable type; Callable[[int], str] is a function of (int) -> str.

    The subscription syntax must always be used with exactly two
    values: the argument list and the return type.  The argument list
    must be a list of types or ellipsis; the return type must be a single type.

    There is no syntax to indicate optional or keyword arguments,
    such function types are rarely used as callback types.
    """
AbstractSet = _alias(collections.abc.Set, 1, name='AbstractSet')
MutableSet = _alias(collections.abc.MutableSet, 1)
# NOTE: Mapping is only covariant in the value type.
Mapping = _alias(collections.abc.Mapping, 2)
MutableMapping = _alias(collections.abc.MutableMapping, 2)
Sequence = _alias(collections.abc.Sequence, 1)
MutableSequence = _alias(collections.abc.MutableSequence, 1)
ByteString = _alias(collections.abc.ByteString, 0)  # Not generic
# Tuple accepts variable number of parameters.
Tuple = _TupleType(tuple, -1, inst=False, name='Tuple')
Tuple.__doc__ = \
    """Tuple type; Tuple[X, Y] is the cross-product type of X and Y.

    Example: Tuple[T1, T2] is a tuple of two elements corresponding
    to type variables T1 and T2.  Tuple[int, float, str] is a tuple
    of an int, a float and a string.

    To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].
    """
List = _alias(list, 1, inst=False, name='List')
Deque = _alias(collections.deque, 1, name='Deque')
Set = _alias(set, 1, inst=False, name='Set')
FrozenSet = _alias(frozenset, 1, inst=False, name='FrozenSet')
MappingView = _alias(collections.abc.MappingView, 1)
KeysView = _alias(collections.abc.KeysView, 1)
ItemsView = _alias(collections.abc.ItemsView, 2)
ValuesView = _alias(collections.abc.ValuesView, 1)
ContextManager = _alias(contextlib.AbstractContextManager, 1, name='ContextManager')
AsyncContextManager = _alias(contextlib.AbstractAsyncContextManager, 1, name='AsyncContextManager')
Dict = _alias(dict, 2, inst=False, name='Dict')
DefaultDict = _alias(collections.defaultdict, 2, name='DefaultDict')
OrderedDict = _alias(collections.OrderedDict, 2)
Counter = _alias(collections.Counter, 1)
ChainMap = _alias(collections.ChainMap, 2)
Generator = _alias(collections.abc.Generator, 3)
AsyncGenerator = _alias(collections.abc.AsyncGenerator, 2)
Type = _alias(type, 1, inst=False, name='Type')
Type.__doc__ = \
    """A special construct usable to annotate class objects.

    For example, suppose we have the following classes::

      class User: ...  # Abstract base for User classes
      class BasicUser(User): ...
      class ProUser(User): ...
      class TeamUser(User): ...

    And a function that takes a class argument that's a subclass of
    User and returns an instance of the corresponding class::

      U = TypeVar('U', bound=User)
      def new_user(user_class: Type[U]) -> U:
          user = user_class()
          # (Here we could write the user object to a database)
          return user

      joe = new_user(BasicUser)

    At this point the type checker knows that joe has type BasicUser.
    """


@runtime_checkable
class SupportsInt(Protocol):
    """An ABC with one abstract method __int__."""
    __slots__ = ()

    @abstractmethod
    def __int__(self) -> int:
        pass


@runtime_checkable
class SupportsFloat(Protocol):
    """An ABC with one abstract method __float__."""
    __slots__ = ()

    @abstractmethod
    def __float__(self) -> float:
        pass


@runtime_checkable
class SupportsComplex(Protocol):
    """An ABC with one abstract method __complex__."""
    __slots__ = ()

    @abstractmethod
    def __complex__(self) -> complex:
        pass


@runtime_checkable
class SupportsBytes(Protocol):
    """An ABC with one abstract method __bytes__."""
    __slots__ = ()

    @abstractmethod
    def __bytes__(self) -> bytes:
        pass


@runtime_checkable
class SupportsIndex(Protocol):
    """An ABC with one abstract method __index__."""
    __slots__ = ()

    @abstractmethod
    def __index__(self) -> int:
        pass


@runtime_checkable
class SupportsAbs(Protocol[T_co]):
    """An ABC with one abstract method __abs__ that is covariant in its return type."""
    __slots__ = ()

    @abstractmethod
    def __abs__(self) -> T_co:
        pass


@runtime_checkable
class SupportsRound(Protocol[T_co]):
    """An ABC with one abstract method __round__ that is covariant in its return type."""
    __slots__ = ()

    @abstractmethod
    def __round__(self, ndigits: int = 0) -> T_co:
        pass


def _make_nmtuple(name, types, module, defaults = ()):
    fields = [n for n, t in types]
    types = {n: _type_check(t, f"field {n} annotation must be a type")
             for n, t in types}
    nm_tpl = collections.namedtuple(name, fields,
                                    defaults=defaults, module=module)
    nm_tpl.__annotations__ = nm_tpl.__new__.__annotations__ = types
    return nm_tpl


# attributes prohibited to set in NamedTuple class syntax
_prohibited = frozenset({'__new__', '__init__', '__slots__', '__getnewargs__',
                         '_fields', '_field_defaults',
                         '_make', '_replace', '_asdict', '_source'})

_special = frozenset({'__module__', '__name__', '__annotations__'})


class NamedTupleMeta(type):

    def __new__(cls, typename, bases, ns):
        assert bases[0] is _NamedTuple
        types = ns.get('__annotations__', {})
        default_names = []
        for field_name in types:
            if field_name in ns:
                default_names.append(field_name)
            elif default_names:
                raise TypeError(f"Non-default namedtuple field {field_name} "
                                f"cannot follow default field"
                                f"{'s' if len(default_names) > 1 else ''} "
                                f"{', '.join(default_names)}")
        nm_tpl = _make_nmtuple(typename, types.items(),
                               defaults=[ns[n] for n in default_names],
                               module=ns['__module__'])
        # update from user namespace without overriding special namedtuple attributes
        for key in ns:
            if key in _prohibited:
                raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
            elif key not in _special and key not in nm_tpl._fields:
                setattr(nm_tpl, key, ns[key])
        return nm_tpl


def NamedTuple(typename, fields=None, /, **kwargs):
    """Typed version of namedtuple.

    Usage in Python versions >= 3.6::

        class Employee(NamedTuple):
            name: str
            id: int

    This is equivalent to::

        Employee = collections.namedtuple('Employee', ['name', 'id'])

    The resulting class has an extra __annotations__ attribute, giving a
    dict that maps field names to types.  (The field names are also in
    the _fields attribute, which is part of the namedtuple API.)
    Alternative equivalent keyword syntax is also accepted::

        Employee = NamedTuple('Employee', name=str, id=int)

    In Python versions <= 3.5 use::

        Employee = NamedTuple('Employee', [('name', str), ('id', int)])
    """
    if fields is None:
        fields = kwargs.items()
    elif kwargs:
        raise TypeError("Either list of fields or keywords"
                        " can be provided to NamedTuple, not both")
    try:
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        module = None
    return _make_nmtuple(typename, fields, module=module)

_NamedTuple = type.__new__(NamedTupleMeta, 'NamedTuple', (), {})

def _namedtuple_mro_entries(bases):
    if len(bases) > 1:
        raise TypeError("Multiple inheritance with NamedTuple is not supported")
    assert bases[0] is NamedTuple
    return (_NamedTuple,)

NamedTuple.__mro_entries__ = _namedtuple_mro_entries


class _TypedDictMeta(type):
    def __new__(cls, name, bases, ns, total=True):
        """Create new typed dict class object.

        This method is called when TypedDict is subclassed,
        or when TypedDict is instantiated. This way
        TypedDict supports all three syntax forms described in its docstring.
        Subclasses and instances of TypedDict return actual dictionaries.
        """
        for base in bases:
            if type(base) is not _TypedDictMeta:
                raise TypeError('cannot inherit from both a TypedDict type '
                                'and a non-TypedDict base class')
        tp_dict = type.__new__(_TypedDictMeta, name, (dict,), ns)

        annotations = {}
        own_annotations = ns.get('__annotations__', {})
        own_annotation_keys = set(own_annotations.keys())
        msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
        own_annotations = {
            n: _type_check(tp, msg, module=tp_dict.__module__)
            for n, tp in own_annotations.items()
        }
        required_keys = set()
        optional_keys = set()

        for base in bases:
            annotations.update(base.__dict__.get('__annotations__', {}))
            required_keys.update(base.__dict__.get('__required_keys__', ()))
            optional_keys.update(base.__dict__.get('__optional_keys__', ()))

        annotations.update(own_annotations)
        if total:
            required_keys.update(own_annotation_keys)
        else:
            optional_keys.update(own_annotation_keys)

        tp_dict.__annotations__ = annotations
        tp_dict.__required_keys__ = frozenset(required_keys)
        tp_dict.__optional_keys__ = frozenset(optional_keys)
        if not hasattr(tp_dict, '__total__'):
            tp_dict.__total__ = total
        return tp_dict

    __call__ = dict  # static method

    def __subclasscheck__(cls, other):
        # Typed dicts are only for static structural subtyping.
        raise TypeError('TypedDict does not support instance and class checks')

    __instancecheck__ = __subclasscheck__


def TypedDict(typename, fields=None, /, *, total=True, **kwargs):
    """A simple typed namespace. At runtime it is equivalent to a plain dict.

    TypedDict creates a dictionary type that expects all of its
    instances to have a certain set of keys, where each key is
    associated with a value of a consistent type. This expectation
    is not checked at runtime but is only enforced by type checkers.
    Usage::

        class Point2D(TypedDict):
            x: int
            y: int
            label: str

        a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
        b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

        assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

    The type info can be accessed via the Point2D.__annotations__ dict, and
    the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
    TypedDict supports two additional equivalent forms::

        Point2D = TypedDict('Point2D', x=int, y=int, label=str)
        Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

    By default, all keys must be present in a TypedDict. It is possible
    to override this by specifying totality.
    Usage::

        class point2D(TypedDict, total=False):
            x: int
            y: int

    This means that a point2D TypedDict can have any of the keys omitted.A type
    checker is only expected to support a literal False or True as the value of
    the total argument. True is the default, and makes all items defined in the
    class body be required.

    The class syntax is only supported in Python 3.6+, while two other
    syntax forms work for Python 2.7 and 3.2+
    """
    if fields is None:
        fields = kwargs
    elif kwargs:
        raise TypeError("TypedDict takes either a dict or keyword arguments,"
                        " but not both")

    ns = {'__annotations__': dict(fields)}
    try:
        # Setting correct module is necessary to make typed dict classes pickleable.
        ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return _TypedDictMeta(typename, (), ns, total=total)

_TypedDict = type.__new__(_TypedDictMeta, 'TypedDict', (), {})
TypedDict.__mro_entries__ = lambda bases: (_TypedDict,)


class NewType:
    """NewType creates simple unique types with almost zero
    runtime overhead. NewType(name, tp) is considered a subtype of tp
    by static type checkers. At runtime, NewType(name, tp) returns
    a dummy callable that simply returns its argument. Usage::

        UserId = NewType('UserId', int)

        def name_by_id(user_id: UserId) -> str:
            ...

        UserId('user')          # Fails type check

        name_by_id(42)          # Fails type check
        name_by_id(UserId(42))  # OK

        num = UserId(5) + 1     # type: int
    """

    def __init__(self, name, tp):
        self.__qualname__ = name
        if '.' in name:
            name = name.rpartition('.')[-1]
        self.__name__ = name
        self.__supertype__ = tp
        def_mod = _caller()
        if def_mod != 'typing':
            self.__module__ = def_mod

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'

    def __call__(self, x):
        return x

    def __reduce__(self):
        return self.__qualname__

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]


# Python-version-specific alias (Python 2: unicode; Python 3: str)
Text = str


# Constant that's True when type checking, but False here.
TYPE_CHECKING = False


class IO(Generic[AnyStr]):
    """Generic base class for TextIO and BinaryIO.

    This is an abstract, generic version of the return of open().

    NOTE: This does not distinguish between the different possible
    classes (text vs. binary, read vs. write vs. read/write,
    append-only, unbuffered).  The TextIO and BinaryIO subclasses
    below capture the distinctions between text vs. binary, which is
    pervasive in the interface; however we currently do not offer a
    way to track the other distinctions in the type system.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def closed(self) -> bool:
        pass

    @abstractmethod
    def fileno(self) -> int:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def isatty(self) -> bool:
        pass

    @abstractmethod
    def read(self, n: int = -1) -> AnyStr:
        pass

    @abstractmethod
    def readable(self) -> bool:
        pass

    @abstractmethod
    def readline(self, limit: int = -1) -> AnyStr:
        pass

    @abstractmethod
    def readlines(self, hint: int = -1) -> List[AnyStr]:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int:
        pass

    @abstractmethod
    def seekable(self) -> bool:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def truncate(self, size: int = None) -> int:
        pass

    @abstractmethod
    def writable(self) -> bool:
        pass

    @abstractmethod
    def write(self, s: AnyStr) -> int:
        pass

    @abstractmethod
    def writelines(self, lines: List[AnyStr]) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> 'IO[AnyStr]':
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback) -> None:
        pass


class BinaryIO(IO[bytes]):
    """Typed version of the return of open() in binary mode."""

    __slots__ = ()

    @abstractmethod
    def write(self, s: Union[bytes, bytearray]) -> int:
        pass

    @abstractmethod
    def __enter__(self) -> 'BinaryIO':
        pass


class TextIO(IO[str]):
    """Typed version of the return of open() in text mode."""

    __slots__ = ()

    @property
    @abstractmethod
    def buffer(self) -> BinaryIO:
        pass

    @property
    @abstractmethod
    def encoding(self) -> str:
        pass

    @property
    @abstractmethod
    def errors(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def line_buffering(self) -> bool:
        pass

    @property
    @abstractmethod
    def newlines(self) -> Any:
        pass

    @abstractmethod
    def __enter__(self) -> 'TextIO':
        pass


class io:
    """Wrapper namespace for IO generic classes."""

    __all__ = ['IO', 'TextIO', 'BinaryIO']
    IO = IO
    TextIO = TextIO
    BinaryIO = BinaryIO


io.__name__ = __name__ + '.io'
sys.modules[io.__name__] = io

Pattern = _alias(stdlib_re.Pattern, 1)
Match = _alias(stdlib_re.Match, 1)

class re:
    """Wrapper namespace for re type aliases."""

    __all__ = ['Pattern', 'Match']
    Pattern = Pattern
    Match = Match


re.__name__ = __name__ + '.re'
sys.modules[re.__name__] = re
<|endoftext|>#!/usr/bin/env python3
"""Generate Python documentation in HTML or text for interactive use.

At the Python interactive prompt, calling help(thing) on a Python object
documents the object, and calling help() starts up an interactive
help session.

Or, at the shell command line outside of Python:

Run "pydoc <name>" to show documentation on something.  <name> may be
the name of a function, module, package, or a dotted reference to a
class or function within a module or module in a package.  If the
argument contains a path segment delimiter (e.g. slash on Unix,
backslash on Windows) it is treated as the path to a Python source file.

Run "pydoc -k <keyword>" to search for a keyword in the synopsis lines
of all available modules.

Run "pydoc -n <hostname>" to start an HTTP server with the given
hostname (default: localhost) on the local machine.

Run "pydoc -p <port>" to start an HTTP server on the given port on the
local machine.  Port number 0 can be used to get an arbitrary unused port.

Run "pydoc -b" to start an HTTP server on an arbitrary unused port and
open a web browser to interactively browse documentation.  Combine with
the -n and -p options to control the hostname and port used.

Run "pydoc -w <name>" to write out the HTML documentation for a module
to a file named "<name>.html".

Module docs for core modules are assumed to be in

    /usr/share/doc/pythonX.Y/html/library

if the pythonX.Y-doc package is installed or in

    https://docs.python.org/X.Y/library/

This can be overridden by setting the PYTHONDOCS environment variable
to a different URL or to a local directory containing the Library
Reference Manual pages.
"""
__all__ = ['help']
__author__ = "Ka-Ping Yee <ping@lfw.org>"
__date__ = "26 February 2001"

__credits__ = """Guido van Rossum, for an excellent programming language.
Tommy Burnette, the original creator of manpy.
Paul Prescod, for all his work on onlinehelp.
Richard Chamberlain, for the first implementation of textdoc.
"""

# Known bugs that can't be fixed here:
#   - synopsis() cannot be prevented from clobbering existing
#     loaded modules.
#   - If the __file__ attribute on a module is a relative path and
#     the current directory is changed with os.chdir(), an incorrect
#     path will be displayed.

import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import types
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only


# --------------------------------------------------------- common routines

def pathdirs():
    """Convert sys.path into a list of absolute, existing, unique paths."""
    dirs = []
    normdirs = []
    for dir in sys.path:
        dir = os.path.abspath(dir or '.')
        normdir = os.path.normcase(dir)
        if normdir not in normdirs and os.path.isdir(dir):
            dirs.append(dir)
            normdirs.append(normdir)
    return dirs

def _isclass(object):
    return inspect.isclass(object) and not isinstance(object, types.GenericAlias)

def _findclass(func):
    cls = sys.modules.get(func.__module__)
    if cls is None:
        return None
    for name in func.__qualname__.split('.')[:-1]:
        cls = getattr(cls, name)
    if not _isclass(cls):
        return None
    return cls

def _finddoc(obj):
    if inspect.ismethod(obj):
        name = obj.__func__.__name__
        self = obj.__self__
        if (_isclass(self) and
            getattr(getattr(self, name, None), '__func__') is obj.__func__):
            # classmethod
            cls = self
        else:
            cls = self.__class__
    elif inspect.isfunction(obj):
        name = obj.__name__
        cls = _findclass(obj)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif inspect.isbuiltin(obj):
        name = obj.__name__
        self = obj.__self__
        if (_isclass(self) and
            self.__qualname__ + '.' + name == obj.__qualname__):
            # classmethod
            cls = self
        else:
            cls = self.__class__
    # Should be tested before isdatadescriptor().
    elif isinstance(obj, property):
        func = obj.fget
        name = func.__name__
        cls = _findclass(func)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif inspect.ismethoddescriptor(obj) or inspect.isdatadescriptor(obj):
        name = obj.__name__
        cls = obj.__objclass__
        if getattr(cls, name) is not obj:
            return None
        if inspect.ismemberdescriptor(obj):
            slots = getattr(cls, '__slots__', None)
            if isinstance(slots, dict) and name in slots:
                return slots[name]
    else:
        return None
    for base in cls.__mro__:
        try:
            doc = _getowndoc(getattr(base, name))
        except AttributeError:
            continue
        if doc is not None:
            return doc
    return None

def _getowndoc(obj):
    """Get the documentation string for an object if it is not
    inherited from its class."""
    try:
        doc = object.__getattribute__(obj, '__doc__')
        if doc is None:
            return None
        if obj is not type:
            typedoc = type(obj).__doc__
            if isinstance(typedoc, str) and typedoc == doc:
                return None
        return doc
    except AttributeError:
        return None

def _getdoc(object):
    """Get the documentation string for an object.

    All tabs are expanded to spaces.  To clean up docstrings that are
    indented to line up with blocks of code, any whitespace than can be
    uniformly removed from the second line onwards is removed."""
    doc = _getowndoc(object)
    if doc is None:
        try:
            doc = _finddoc(object)
        except (AttributeError, TypeError):
            return None
    if not isinstance(doc, str):
        return None
    return inspect.cleandoc(doc)

def getdoc(object):
    """Get the doc string or comments for an object."""
    result = _getdoc(object) or inspect.getcomments(object)
    return result and re.sub('^ *\n', '', result.rstrip()) or ''

def splitdoc(doc):
    """Split a doc string into a synopsis line (if any) and the rest."""
    lines = doc.strip().split('\n')
    if len(lines) == 1:
        return lines[0], ''
    elif len(lines) >= 2 and not lines[1].rstrip():
        return lines[0], '\n'.join(lines[2:])
    return '', '\n'.join(lines)

def classname(object, modname):
    """Get a class name and qualify it with a module name if necessary."""
    name = object.__name__
    if object.__module__ != modname:
        name = object.__module__ + '.' + name
    return name

def isdata(object):
    """Check if an object is of a type that probably means it's data."""
    return not (inspect.ismodule(object) or _isclass(object) or
                inspect.isroutine(object) or inspect.isframe(object) or
                inspect.istraceback(object) or inspect.iscode(object))

def replace(text, *pairs):
    """Do a series of global replacements on a string."""
    while pairs:
        text = pairs[1].join(text.split(pairs[0]))
        pairs = pairs[2:]
    return text

def cram(text, maxlen):
    """Omit part of a string if needed to make it fit in a maximum length."""
    if len(text) > maxlen:
        pre = max(0, (maxlen-3)//2)
        post = max(0, maxlen-3-pre)
        return text[:pre] + '...' + text[len(text)-post:]
    return text

_re_stripid = re.compile(r' at 0x[0-9a-f]{6,16}(>+)$', re.IGNORECASE)
def stripid(text):
    """Remove the hexadecimal id from a Python object representation."""
    # The behaviour of %p is implementation-dependent in terms of case.
    return _re_stripid.sub(r'\1', text)

def _is_bound_method(fn):
    """
    Returns True if fn is a bound method, regardless of whether
    fn was implemented in Python or in C.
    """
    if inspect.ismethod(fn):
        return True
    if inspect.isbuiltin(fn):
        self = getattr(fn, '__self__', None)
        return not (inspect.ismodule(self) or (self is None))
    return False


def allmethods(cl):
    methods = {}
    for key, value in inspect.getmembers(cl, inspect.isroutine):
        methods[key] = 1
    for base in cl.__bases__:
        methods.update(allmethods(base)) # all your base are belong to us
    for key in methods.keys():
        methods[key] = getattr(cl, key)
    return methods

def _split_list(s, predicate):
    """Split sequence s via predicate, and return pair ([true], [false]).

    The return value is a 2-tuple of lists,
        ([x for x in s if predicate(x)],
         [x for x in s if not predicate(x)])
    """

    yes = []
    no = []
    for x in s:
        if predicate(x):
            yes.append(x)
        else:
            no.append(x)
    return yes, no

def visiblename(name, all=None, obj=None):
    """Decide whether to show documentation on a variable."""
    # Certain special names are redundant or internal.
    # XXX Remove __initializing__?
    if name in {'__author__', '__builtins__', '__cached__', '__credits__',
                '__date__', '__doc__', '__file__', '__spec__',
                '__loader__', '__module__', '__name__', '__package__',
                '__path__', '__qualname__', '__slots__', '__version__'}:
        return 0
    # Private names are hidden, but special names are displayed.
    if name.startswith('__') and name.endswith('__'): return 1
    # Namedtuples have public fields and methods with a single leading underscore
    if name.startswith('_') and hasattr(obj, '_fields'):
        return True
    if all is not None:
        # only document that which the programmer exported in __all__
        return name in all
    else:
        return not name.startswith('_')

def classify_class_attrs(object):
    """Wrap inspect.classify_class_attrs, with fixup for data descriptors."""
    results = []
    for (name, kind, cls, value) in inspect.classify_class_attrs(object):
        if inspect.isdatadescriptor(value):
            kind = 'data descriptor'
            if isinstance(value, property) and value.fset is None:
                kind = 'readonly property'
        results.append((name, kind, cls, value))
    return results

def sort_attributes(attrs, object):
    'Sort the attrs list in-place by _fields and then alphabetically by name'
    # This allows data descriptors to be ordered according
    # to a _fields attribute if present.
    fields = getattr(object, '_fields', [])
    try:
        field_order = {name : i-len(fields) for (i, name) in enumerate(fields)}
    except TypeError:
        field_order = {}
    keyfunc = lambda attr: (field_order.get(attr[0], 0), attr[0])
    attrs.sort(key=keyfunc)

# ----------------------------------------------------- module manipulation

def ispackage(path):
    """Guess whether a path refers to a package directory."""
    if os.path.isdir(path):
        for ext in ('.py', '.pyc'):
            if os.path.isfile(os.path.join(path, '__init__' + ext)):
                return True
    return False

def source_synopsis(file):
    line = file.readline()
    while line[:1] == '#' or not line.strip():
        line = file.readline()
        if not line: break
    line = line.strip()
    if line[:4] == 'r"""': line = line[1:]
    if line[:3] == '"""':
        line = line[3:]
        if line[-1:] == '\\': line = line[:-1]
        while not line.strip():
            line = file.readline()
            if not line: break
        result = line.split('"""')[0].strip()
    else: result = None
    return result

def synopsis(filename, cache={}):
    """Get the one-line summary out of a module file."""
    mtime = os.stat(filename).st_mtime
    lastupdate, result = cache.get(filename, (None, None))
    if lastupdate is None or lastupdate < mtime:
        # Look for binary suffixes first, falling back to source.
        if filename.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            loader_cls = importlib.machinery.SourcelessFileLoader
        elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            loader_cls = importlib.machinery.ExtensionFileLoader
        else:
            loader_cls = None
        # Now handle the choice.
        if loader_cls is None:
            # Must be a source file.
            try:
                file = tokenize.open(filename)
            except OSError:
                # module can't be opened, so skip it
                return None
            # text modules can be directly examined
            with file:
                result = source_synopsis(file)
        else:
            # Must be a binary module, which has to be imported.
            loader = loader_cls('__temp__', filename)
            # XXX We probably don't need to pass in the loader here.
            spec = importlib.util.spec_from_file_location('__temp__', filename,
                                                          loader=loader)
            try:
                module = importlib._bootstrap._load(spec)
            except:
                return None
            del sys.modules['__temp__']
            result = module.__doc__.splitlines()[0] if module.__doc__ else None
        # Cache the result.
        cache[filename] = (mtime, result)
    return result

class ErrorDuringImport(Exception):
    """Errors that occurred while trying to import something to document it."""
    def __init__(self, filename, exc_info):
        self.filename = filename
        self.exc, self.value, self.tb = exc_info

    def __str__(self):
        exc = self.exc.__name__
        return 'problem in %s - %s: %s' % (self.filename, exc, self.value)

def importfile(path):
    """Import a Python source file or compiled file given its path."""
    magic = importlib.util.MAGIC_NUMBER
    with open(path, 'rb') as file:
        is_bytecode = magic == file.read(len(magic))
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    if is_bytecode:
        loader = importlib._bootstrap_external.SourcelessFileLoader(name, path)
    else:
        loader = importlib._bootstrap_external.SourceFileLoader(name, path)
    # XXX We probably don't need to pass in the loader here.
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    try:
        return importlib._bootstrap._load(spec)
    except:
        raise ErrorDuringImport(path, sys.exc_info())

def safeimport(path, forceload=0, cache={}):
    """Import a module; handle errors; return None if the module isn't found.

    If the module *is* found but an exception occurs, it's wrapped in an
    ErrorDuringImport exception and reraised.  Unlike __import__, if a
    package path is specified, the module at the end of the path is returned,
    not the package at the beginning.  If the optional 'forceload' argument
    is 1, we reload the module from disk (unless it's a dynamic extension)."""
    try:
        # If forceload is 1 and the module has been previously loaded from
        # disk, we always have to reload the module.  Checking the file's
        # mtime isn't good enough (e.g. the module could contain a class
        # that inherits from another module that has changed).
        if forceload and path in sys.modules:
            if path not in sys.builtin_module_names:
                # Remove the module from sys.modules and re-import to try
                # and avoid problems with partially loaded modules.
                # Also remove any submodules because they won't appear
                # in the newly loaded module's namespace if they're already
                # in sys.modules.
                subs = [m for m in sys.modules if m.startswith(path + '.')]
                for key in [path] + subs:
                    # Prevent garbage collection.
                    cache[key] = sys.modules[key]
                    del sys.modules[key]
        module = __import__(path)
    except:
        # Did the error occur before or after the module was found?
        (exc, value, tb) = info = sys.exc_info()
        if path in sys.modules:
            # An error occurred while executing the imported module.
            raise ErrorDuringImport(sys.modules[path].__file__, info)
        elif exc is SyntaxError:
            # A SyntaxError occurred before we could execute the module.
            raise ErrorDuringImport(value.filename, info)
        elif issubclass(exc, ImportError) and value.name == path:
            # No such module in the path.
            return None
        else:
            # Some other error occurred during the importing process.
            raise ErrorDuringImport(path, sys.exc_info())
    for part in path.split('.')[1:]:
        try: module = getattr(module, part)
        except AttributeError: return None
    return module

# ---------------------------------------------------- formatter base class

class Doc:

    PYTHONDOCS = os.environ.get("PYTHONDOCS",
                                "https://docs.python.org/%d.%d/library"
                                % sys.version_info[:2])

    def document(self, object, name=None, *args):
        """Generate documentation for an object."""
        args = (object, name) + args
        # 'try' clause is to attempt to handle the possibility that inspect
        # identifies something in a way that pydoc itself has issues handling;
        # think 'super' and how it is a descriptor (which raises the exception
        # by lacking a __name__ attribute) and an instance.
        try:
            if inspect.ismodule(object): return self.docmodule(*args)
            if _isclass(object): return self.docclass(*args)
            if inspect.isroutine(object): return self.docroutine(*args)
        except AttributeError:
            pass
        if inspect.isdatadescriptor(object): return self.docdata(*args)
        return self.docother(*args)

    def fail(self, object, name=None, *args):
        """Raise an exception for unimplemented types."""
        message = "don't know how to document object%s of type %s" % (
            name and ' ' + repr(name), type(object).__name__)
        raise TypeError(message)

    docmodule = docclass = docroutine = docother = docproperty = docdata = fail

    def getdocloc(self, object, basedir=sysconfig.get_path('stdlib')):
        """Return the location of module docs or None"""

        try:
            file = inspect.getabsfile(object)
        except TypeError:
            file = '(built-in)'

        docloc = os.environ.get("PYTHONDOCS", self.PYTHONDOCS)

        basedir = os.path.normcase(basedir)
        if (isinstance(object, type(os)) and
            (object.__name__ in ('errno', 'exceptions', 'gc', 'imp',
                                 'marshal', 'posix', 'signal', 'sys',
                                 '_thread', 'zipimport') or
             (file.startswith(basedir) and
              not file.startswith(os.path.join(basedir, 'dist-packages')) and
              not file.startswith(os.path.join(basedir, 'site-packages')))) and
            object.__name__ not in ('xml.etree', 'test.pydoc_mod')):
            if docloc.startswith(("http://", "https://")):
                docloc = "{}/{}.html".format(docloc.rstrip("/"), object.__name__.lower())
            else:
                docloc = os.path.join(docloc, object.__name__.lower() + ".html")
        else:
            docloc = None
        return docloc

# -------------------------------------------- HTML documentation generator

class HTMLRepr(Repr):
    """Class for safely making an HTML representation of a Python object."""
    def __init__(self):
        Repr.__init__(self)
        self.maxlist = self.maxtuple = 20
        self.maxdict = 10
        self.maxstring = self.maxother = 100

    def escape(self, text):
        return replace(text, '&', '&amp;', '<', '&lt;', '>', '&gt;')

    def repr(self, object):
        return Repr.repr(self, object)

    def repr1(self, x, level):
        if hasattr(type(x), '__name__'):
            methodname = 'repr_' + '_'.join(type(x).__name__.split())
            if hasattr(self, methodname):
                return getattr(self, methodname)(x, level)
        return self.escape(cram(stripid(repr(x)), self.maxother))

    def repr_string(self, x, level):
        test = cram(x, self.maxstring)
        testrepr = repr(test)
        if '\\' in test and '\\' not in replace(testrepr, r'\\', ''):
            # Backslashes are only literal in the string and are never
            # needed to make any special characters, so show a raw string.
            return 'r' + testrepr[0] + self.escape(test) + testrepr[0]
        return re.sub(r'((\\[\\abfnrtv\'"]|\\[0-9]..|\\x..|\\u....)+)',
                      r'<font color="#c040c0">\1</font>',
                      self.escape(testrepr))

    repr_str = repr_string

    def repr_instance(self, x, level):
        try:
            return self.escape(cram(stripid(repr(x)), self.maxstring))
        except:
            return self.escape('<%s instance>' % x.__class__.__name__)

    repr_unicode = repr_string

class HTMLDoc(Doc):
    """Formatter class for HTML documentation."""

    # ------------------------------------------- HTML formatting utilities

    _repr_instance = HTMLRepr()
    repr = _repr_instance.repr
    escape = _repr_instance.escape

    def page(self, title, contents):
        """Format an HTML page."""
        return '''\
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html><head><title>Python: %s</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head><body bgcolor="#f0f0f8">
%s
</body></html>''' % (title, contents)

    def heading(self, title, fgcol, bgcol, extras=''):
        """Format a page heading."""
        return '''
<table width="100%%" cellspacing=0 cellpadding=2 border=0 summary="heading">
<tr bgcolor="%s">
<td valign=bottom>&nbsp;<br>
<font color="%s" face="helvetica, arial">&nbsp;<br>%s</font></td
><td align=right valign=bottom
><font color="%s" face="helvetica, arial">%s</font></td></tr></table>
    ''' % (bgcol, fgcol, title, fgcol, extras or '&nbsp;')

    def section(self, title, fgcol, bgcol, contents, width=6,
                prelude='', marginalia=None, gap='&nbsp;'):
        """Format a section with a heading."""
        if marginalia is None:
            marginalia = '<tt>' + '&nbsp;' * width + '</tt>'
        result = '''<p>
<table width="100%%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="%s">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="%s" face="helvetica, arial">%s</font></td></tr>
    ''' % (bgcol, fgcol, title)
        if prelude:
            result = result + '''
<tr bgcolor="%s"><td rowspan=2>%s</td>
<td colspan=2>%s</td></tr>
<tr><td>%s</td>''' % (bgcol, marginalia, prelude, gap)
        else:
            result = result + '''
<tr><td bgcolor="%s">%s</td><td>%s</td>''' % (bgcol, marginalia, gap)

        return result + '\n<td width="100%%">%s</td></tr></table>' % contents

    def bigsection(self, title, *args):
        """Format a section with a big heading."""
        title = '<big><strong>%s</strong></big>' % title
        return self.section(title, *args)

    def preformat(self, text):
        """Format literal preformatted text."""
        text = self.escape(text.expandtabs())
        return replace(text, '\n\n', '\n \n', '\n\n', '\n \n',
                             ' ', '&nbsp;', '\n', '<br>\n')

    def multicolumn(self, list, format, cols=4):
        """Format a list of items into a multi-column list."""
        result = ''
        rows = (len(list)+cols-1)//cols
        for col in range(cols):
            result = result + '<td width="%d%%" valign=top>' % (100//cols)
            for i in range(rows*col, rows*col+rows):
                if i < len(list):
                    result = result + format(list[i]) + '<br>\n'
            result = result + '</td>'
        return '<table width="100%%" summary="list"><tr>%s</tr></table>' % result

    def grey(self, text): return '<font color="#909090">%s</font>' % text

    def namelink(self, name, *dicts):
        """Make a link for an identifier, given name-to-URL mappings."""
        for dict in dicts:
            if name in dict:
                return '<a href="%s">%s</a>' % (dict[name], name)
        return name

    def classlink(self, object, modname):
        """Make a link for a class."""
        name, module = object.__name__, sys.modules.get(object.__module__)
        if hasattr(module, name) and getattr(module, name) is object:
            return '<a href="%s.html#%s">%s</a>' % (
                module.__name__, name, classname(object, modname))
        return classname(object, modname)

    def modulelink(self, object):
        """Make a link for a module."""
        return '<a href="%s.html">%s</a>' % (object.__name__, object.__name__)

    def modpkglink(self, modpkginfo):
        """Make a link for a module or package to display in an index."""
        name, path, ispackage, shadowed = modpkginfo
        if shadowed:
            return self.grey(name)
        if path:
            url = '%s.%s.html' % (path, name)
        else:
            url = '%s.html' % name
        if ispackage:
            text = '<strong>%s</strong>&nbsp;(package)' % name
        else:
            text = name
        return '<a href="%s">%s</a>' % (url, text)

    def filelink(self, url, path):
        """Make a link to source file."""
        return '<a href="file:%s">%s</a>' % (url, path)

    def markup(self, text, escape=None, funcs={}, classes={}, methods={}):
        """Mark up some plain text, given a context of symbols to look for.
        Each context dictionary maps object names to anchor names."""
        escape = escape or self.escape
        results = []
        here = 0
        pattern = re.compile(r'\b((http|https|ftp)://\S+[\w/]|'
                                r'RFC[- ]?(\d+)|'
                                r'PEP[- ]?(\d+)|'
                                r'(self\.)?(\w+))')
        while True:
            match = pattern.search(text, here)
            if not match: break
            start, end = match.span()
            results.append(escape(text[here:start]))

            all, scheme, rfc, pep, selfdot, name = match.groups()
            if scheme:
                url = escape(all).replace('"', '&quot;')
                results.append('<a href="%s">%s</a>' % (url, url))
            elif rfc:
                url = 'http://www.rfc-editor.org/rfc/rfc%d.txt' % int(rfc)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif pep:
                url = 'https://www.python.org/dev/peps/pep-%04d/' % int(pep)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif selfdot:
                # Create a link for methods like 'self.method(...)'
                # and use <strong> for attributes like 'self.attr'
                if text[end:end+1] == '(':
                    results.append('self.' + self.namelink(name, methods))
                else:
                    results.append('self.<strong>%s</strong>' % name)
            elif text[end:end+1] == '(':
                results.append(self.namelink(name, methods, funcs, classes))
            else:
                results.append(self.namelink(name, classes))
            here = end
        results.append(escape(text[here:]))
        return ''.join(results)

    # ---------------------------------------------- type-specific routines

    def formattree(self, tree, modname, parent=None):
        """Produce HTML for a class tree as given by inspect.getclasstree()."""
        result = ''
        for entry in tree:
            if type(entry) is type(()):
                c, bases = entry
                result = result + '<dt><font face="helvetica, arial">'
                result = result + self.classlink(c, modname)
                if bases and bases != (parent,):
                    parents = []
                    for base in bases:
                        parents.append(self.classlink(base, modname))
                    result = result + '(' + ', '.join(parents) + ')'
                result = result + '\n</font></dt>'
            elif type(entry) is type([]):
                result = result + '<dd>\n%s</dd>\n' % self.formattree(
                    entry, modname, c)
        return '<dl>\n%s</dl>\n' % result

    def docmodule(self, object, name=None, mod=None, *ignored):
        """Produce HTML documentation for a module object."""
        name = object.__name__ # ignore the passed-in name
        try:
            all = object.__all__
        except AttributeError:
            all = None
        parts = name.split('.')
        links = []
        for i in range(len(parts)-1):
            links.append(
                '<a href="%s.html"><font color="#ffffff">%s</font></a>' %
                ('.'.join(parts[:i+1]), parts[i]))
        linkedname = '.'.join(links + parts[-1:])
        head = '<big><big><strong>%s</strong></big></big>' % linkedname
        try:
            path = inspect.getabsfile(object)
            url = urllib.parse.quote(path)
            filelink = self.filelink(url, path)
        except TypeError:
            filelink = '(built-in)'
        info = []
        if hasattr(object, '__version__'):
            version = str(object.__version__)
            if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
                version = version[11:-1].strip()
            info.append('version %s' % self.escape(version))
        if hasattr(object, '__date__'):
            info.append(self.escape(str(object.__date__)))
        if info:
            head = head + ' (%s)' % ', '.join(info)
        docloc = self.getdocloc(object)
        if docloc is not None:
            docloc = '<br><a href="%(docloc)s">Module Reference</a>' % locals()
        else:
            docloc = ''
        result = self.heading(
            head, '#ffffff', '#7799ee',
            '<a href=".">index</a><br>' + filelink + docloc)

        modules = inspect.getmembers(object, inspect.ismodule)

        classes, cdict = [], {}
        for key, value in inspect.getmembers(object, _isclass):
            # if __all__ exists, believe it.  Otherwise use old heuristic.
            if (all is not None or
                (inspect.getmodule(value) or object) is object):
                if visiblename(key, all, object):
                    classes.append((key, value))
                    cdict[key] = cdict[value] = '#' + key
        for key, value in classes:
            for base in value.__bases__:
                key, modname = base.__name__, base.__module__
                module = sys.modules.get(modname)
                if modname != name and module and hasattr(module, key):
                    if getattr(module, key) is base:
                        if not key in cdict:
                            cdict[key] = cdict[base] = modname + '.html#' + key
        funcs, fdict = [], {}
        for key, value in inspect.getmembers(object, inspect.isroutine):
            # if __all__ exists, believe it.  Otherwise use old heuristic.
            if (all is not None or
                inspect.isbuiltin(value) or inspect.getmodule(value) is object):
                if visiblename(key, all, object):
                    funcs.append((key, value))
                    fdict[key] = '#-' + key
                    if inspect.isfunction(value): fdict[value] = fdict[key]
        data = []
        for key, value in inspect.getmembers(object, isdata):
            if visiblename(key, all, object):
                data.append((key, value))

        doc = self.markup(getdoc(object), self.preformat, fdict, cdict)
        doc = doc and '<tt>%s</tt>' % doc
        result = result + '<p>%s</p>\n' % doc

        if hasattr(object, '__path__'):
            modpkgs = []
            for importer, modname, ispkg in pkgutil.iter_modules(object.__path__):
                modpkgs.append((modname, name, ispkg, 0))
            modpkgs.sort()
            contents = self.multicolumn(modpkgs, self.modpkglink)
            result = result + self.bigsection(
                'Package Contents', '#ffffff', '#aa55cc', contents)
        elif modules:
            contents = self.multicolumn(
                modules, lambda t: self.modulelink(t[1]))
            result = result + self.bigsection(
                'Modules', '#ffffff', '#aa55cc', contents)

        if classes:
            classlist = [value for (key, value) in classes]
            contents = [
                self.formattree(inspect.getclasstree(classlist, 1), name)]
            for key, value in classes:
                contents.append(self.document(value, key, name, fdict, cdict))
            result = result + self.bigsection(
                'Classes', '#ffffff', '#ee77aa', ' '.join(contents))
        if funcs:
            contents = []
            for key, value in funcs:
                contents.append(self.document(value, key, name, fdict, cdict))
            result = result + self.bigsection(
                'Functions', '#ffffff', '#eeaa77', ' '.join(contents))
        if data:
            contents = []
            for key, value in data:
                contents.append(self.document(value, key))
            result = result + self.bigsection(
                'Data', '#ffffff', '#55aa55', '<br>\n'.join(contents))
        if hasattr(object, '__author__'):
            contents = self.markup(str(object.__author__), self.preformat)
            result = result + self.bigsection(
                'Author', '#ffffff', '#7799ee', contents)
        if hasattr(object, '__credits__'):
            contents = self.markup(str(object.__credits__), self.preformat)
            result = result + self.bigsection(
                'Credits', '#ffffff', '#7799ee', contents)

        return result

    def docclass(self, object, name=None, mod=None, funcs={}, classes={},
                 *ignored):
        """Produce HTML documentation for a class object."""
        realname = object.__name__
        name = name or realname
        bases = object.__bases__

        contents = []
        push = contents.append

        # Cute little class to pump out a horizontal rule between sections.
        class HorizontalRule:
            def __init__(self):
                self.needone = 0
            def maybe(self):
                if self.needone:
                    push('<hr>\n')
                self.needone = 1
        hr = HorizontalRule()

        # List the mro, if non-trivial.
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            hr.maybe()
            push('<dl><dt>Method resolution order:</dt>\n')
            for base in mro:
                push('<dd>%s</dd>\n' % self.classlink(base,
                                                      object.__module__))
            push('</dl>\n')

        def spill(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        # Some descriptors may meet a failure in their __get__.
                        # (bug #1785)
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value, name, mod,
                                        funcs, classes, mdict, object))
                    push('\n')
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    base = self.docother(getattr(object, name), name, mod)
                    doc = getdoc(value)
                    if not doc:
                        push('<dl><dt>%s</dl>\n' % base)
                    else:
                        doc = self.markup(getdoc(value), self.preformat,
                                          funcs, classes, mdict)
                        doc = '<dd><tt>%s</tt>' % doc
                        push('<dl><dt>%s%s</dl>\n' % (base, doc))
                    push('\n')
            return attrs

        attrs = [(name, kind, cls, value)
                 for name, kind, cls, value in classify_class_attrs(object)
                 if visiblename(name, obj=object)]

        mdict = {}
        for key, kind, homecls, value in attrs:
            mdict[key] = anchor = '#' + name + '-' + key
            try:
                value = getattr(object, name)
            except Exception:
                # Some descriptors may meet a failure in their __get__.
                # (bug #1785)
                pass
            try:
                # The value may not be hashable (e.g., a data attr with
                # a dict or list value).
                mdict[value] = anchor
            except TypeError:
                pass

        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            attrs, inherited = _split_list(attrs, lambda t: t[2] is thisclass)

            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = 'defined here'
            else:
                tag = 'inherited from %s' % self.classlink(thisclass,
                                                           object.__module__)
            tag += ':<br>\n'

            sort_attributes(attrs, object)

            # Pump out the attrs, segregated by kind.
            attrs = spill('Methods %s' % tag, attrs,
                          lambda t: t[1] == 'method')
            attrs = spill('Class methods %s' % tag, attrs,
                          lambda t: t[1] == 'class method')
            attrs = spill('Static methods %s' % tag, attrs,
                          lambda t: t[1] == 'static method')
            attrs = spilldescriptors("Readonly properties %s" % tag, attrs,
                                     lambda t: t[1] == 'readonly property')
            attrs = spilldescriptors('Data descriptors %s' % tag, attrs,
                                     lambda t: t[1] == 'data descriptor')
            attrs = spilldata('Data and other attributes %s' % tag, attrs,
                              lambda t: t[1] == 'data')
            assert attrs == []
            attrs = inherited

        contents = ''.join(contents)

        if name == realname:
            title = '<a name="%s">class <strong>%s</strong></a>' % (
                name, realname)
        else:
            title = '<strong>%s</strong> = <a name="%s">class %s</a>' % (
                name, name, realname)
        if bases:
            parents = []
            for base in bases:
                parents.append(self.classlink(base, object.__module__))
            title = title + '(%s)' % ', '.join(parents)

        decl = ''
        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != '()':
                decl = name + self.escape(argspec) + '\n\n'

        doc = getdoc(object)
        if decl:
            doc = decl + (doc or '')
        doc = self.markup(doc, self.preformat, funcs, classes, mdict)
        doc = doc and '<tt>%s<br>&nbsp;</tt>' % doc

        return self.section(title, '#000000', '#ffc8d8', contents, 3, doc)

    def formatvalue(self, object):
        """Format an argument default value as text."""
        return self.grey('=' + self.repr(object))

    def docroutine(self, object, name=None, mod=None,
                   funcs={}, classes={}, methods={}, cl=None):
        """Produce HTML documentation for a function or method object."""
        realname = object.__name__
        name = name or realname
        anchor = (cl and cl.__name__ or '') + '-' + name
        note = ''
        skipdocs = 0
        if _is_bound_method(object):
            imclass = object.__self__.__class__
            if cl:
                if imclass is not cl:
                    note = ' from ' + self.classlink(imclass, mod)
            else:
                if object.__self__ is not None:
                    note = ' method of %s instance' % self.classlink(
                        object.__self__.__class__, mod)
                else:
                    note = ' unbound %s method' % self.classlink(imclass,mod)

        if (inspect.iscoroutinefunction(object) or
                inspect.isasyncgenfunction(object)):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''

        if name == realname:
            title = '<a name="%s"><strong>%s</strong></a>' % (anchor, realname)
        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                reallink = '<a href="#%s">%s</a>' % (
                    cl.__name__ + '-' + realname, realname)
                skipdocs = 1
            else:
                reallink = realname
            title = '<a name="%s"><strong>%s</strong></a> = %s' % (
                anchor, name, reallink)
        argspec = None
        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                argspec = str(signature)
                if realname == '<lambda>':
                    title = '<strong>%s</strong> <em>lambda</em> ' % name
                    # XXX lambda's won't usually have func_annotations['return']
                    # since the syntax doesn't support but it is possible.
                    # So removing parentheses isn't truly safe.
                    argspec = argspec[1:-1] # remove parentheses
        if not argspec:
            argspec = '(...)'

        decl = asyncqualifier + title + self.escape(argspec) + (note and
               self.grey('<font face="helvetica, arial">%s</font>' % note))

        if skipdocs:
            return '<dl><dt>%s</dt></dl>\n' % decl
        else:
            doc = self.markup(
                getdoc(object), self.preformat, funcs, classes, methods)
            doc = doc and '<dd><tt>%s</tt></dd>' % doc
            return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)

    def docdata(self, object, name=None, mod=None, cl=None):
        """Produce html documentation for a data descriptor."""
        results = []
        push = results.append

        if name:
            push('<dl><dt><strong>%s</strong></dt>\n' % name)
        doc = self.markup(getdoc(object), self.preformat)
        if doc:
            push('<dd><tt>%s</tt></dd>\n' % doc)
        push('</dl>\n')

        return ''.join(results)

    docproperty = docdata

    def docother(self, object, name=None, mod=None, *ignored):
        """Produce HTML documentation for a data object."""
        lhs = name and '<strong>%s</strong> = ' % name or ''
        return lhs + self.repr(object)

    def index(self, dir, shadowed=None):
        """Generate an HTML index for a directory of modules."""
        modpkgs = []
        if shadowed is None: shadowed = {}
        for importer, name, ispkg in pkgutil.iter_modules([dir]):
            if any((0xD800 <= ord(ch) <= 0xDFFF) for ch in name):
                # ignore a module if its name contains a surrogate character
                continue
            modpkgs.append((name, '', ispkg, name in shadowed))
            shadowed[name] = 1

        modpkgs.sort()
        contents = self.multicolumn(modpkgs, self.modpkglink)
        return self.bigsection(dir, '#ffffff', '#ee77aa', contents)

# -------------------------------------------- text documentation generator

class TextRepr(Repr):
    """Class for safely making a text representation of a Python object."""
    def __init__(self):
        Repr.__init__(self)
        self.maxlist = self.maxtuple = 20
        self.maxdict = 10
        self.maxstring = self.maxother = 100

    def repr1(self, x, level):
        if hasattr(type(x), '__name__'):
            methodname = 'repr_' + '_'.join(type(x).__name__.split())
            if hasattr(self, methodname):
                return getattr(self, methodname)(x, level)
        return cram(stripid(repr(x)), self.maxother)

    def repr_string(self, x, level):
        test = cram(x, self.maxstring)
        testrepr = repr(test)
        if '\\' in test and '\\' not in replace(testrepr, r'\\', ''):
            # Backslashes are only literal in the string and are never
            # needed to make any special characters, so show a raw string.
            return 'r' + testrepr[0] + test + testrepr[0]
        return testrepr

    repr_str = repr_string

    def repr_instance(self, x, level):
        try:
            return cram(stripid(repr(x)), self.maxstring)
        except:
            return '<%s instance>' % x.__class__.__name__

class TextDoc(Doc):
    """Formatter class for text documentation."""

    # ------------------------------------------- text formatting utilities

    _repr_instance = TextRepr()
    repr = _repr_instance.repr

    def bold(self, text):
        """Format a string in bold by overstriking."""
        return ''.join(ch + '\b' + ch for ch in text)

    def indent(self, text, prefix='    '):
        """Indent text by prepending a given prefix to each line."""
        if not text: return ''
        lines = [prefix + line for line in text.split('\n')]
        if lines: lines[-1] = lines[-1].rstrip()
        return '\n'.join(lines)

    def section(self, title, contents):
        """Format a section with a given heading."""
        clean_contents = self.indent(contents).rstrip()
        return self.bold(title) + '\n' + clean_contents + '\n\n'

    # ---------------------------------------------- type-specific routines

    def formattree(self, tree, modname, parent=None, prefix=''):
        """Render in text a class tree as returned by inspect.getclasstree()."""
        result = ''
        for entry in tree:
            if type(entry) is type(()):
                c, bases = entry
                result = result + prefix + classname(c, modname)
                if bases and bases != (parent,):
                    parents = (classname(c, modname) for c in bases)
                    result = result + '(%s)' % ', '.join(parents)
                result = result + '\n'
            elif type(entry) is type([]):
                result = result + self.formattree(
                    entry, modname, c, prefix + '    ')
        return result

    def docmodule(self, object, name=None, mod=None):
        """Produce text documentation for a given module object."""
        name = object.__name__ # ignore the passed-in name
        synop, desc = splitdoc(getdoc(object))
        result = self.section('NAME', name + (synop and ' - ' + synop))
        all = getattr(object, '__all__', None)
        docloc = self.getdocloc(object)
        if docloc is not None:
            result = result + self.section('MODULE REFERENCE', docloc + """

The following documentation is automatically generated from the Python
source files.  It may be incomplete, incorrect or include features that
are considered implementation detail and may vary between Python
implementations.  When in doubt, consult the module reference at the
location listed above.
""")

        if desc:
            result = result + self.section('DESCRIPTION', desc)

        classes = []
        for key, value in inspect.getmembers(object, _isclass):
            # if __all__ exists, believe it.  Otherwise use old heuristic.
            if (all is not None
                or (inspect.getmodule(value) or object) is object):
                if visiblename(key, all, object):
                    classes.append((key, value))
        funcs = []
        for key, value in inspect.getmembers(object, inspect.isroutine):
            # if __all__ exists, believe it.  Otherwise use old heuristic.
            if (all is not None or
                inspect.isbuiltin(value) or inspect.getmodule(value) is object):
                if visiblename(key, all, object):
                    funcs.append((key, value))
        data = []
        for key, value in inspect.getmembers(object, isdata):
            if visiblename(key, all, object):
                data.append((key, value))

        modpkgs = []
        modpkgs_names = set()
        if hasattr(object, '__path__'):
            for importer, modname, ispkg in pkgutil.iter_modules(object.__path__):
                modpkgs_names.add(modname)
                if ispkg:
                    modpkgs.append(modname + ' (package)')
                else:
                    modpkgs.append(modname)

            modpkgs.sort()
            result = result + self.section(
                'PACKAGE CONTENTS', '\n'.join(modpkgs))

        # Detect submodules as sometimes created by C extensions
        submodules = []
        for key, value in inspect.getmembers(object, inspect.ismodule):
            if value.__name__.startswith(name + '.') and key not in modpkgs_names:
                submodules.append(key)
        if submodules:
            submodules.sort()
            result = result + self.section(
                'SUBMODULES', '\n'.join(submodules))

        if classes:
            classlist = [value for key, value in classes]
            contents = [self.formattree(
                inspect.getclasstree(classlist, 1), name)]
            for key, value in classes:
                contents.append(self.document(value, key, name))
            result = result + self.section('CLASSES', '\n'.join(contents))

        if funcs:
            contents = []
            for key, value in funcs:
                contents.append(self.document(value, key, name))
            result = result + self.section('FUNCTIONS', '\n'.join(contents))

        if data:
            contents = []
            for key, value in data:
                contents.append(self.docother(value, key, name, maxlen=70))
            result = result + self.section('DATA', '\n'.join(contents))

        if hasattr(object, '__version__'):
            version = str(object.__version__)
            if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
                version = version[11:-1].strip()
            result = result + self.section('VERSION', version)
        if hasattr(object, '__date__'):
            result = result + self.section('DATE', str(object.__date__))
        if hasattr(object, '__author__'):
            result = result + self.section('AUTHOR', str(object.__author__))
        if hasattr(object, '__credits__'):
            result = result + self.section('CREDITS', str(object.__credits__))
        try:
            file = inspect.getabsfile(object)
        except TypeError:
            file = '(built-in)'
        result = result + self.section('FILE', file)
        return result

    def docclass(self, object, name=None, mod=None, *ignored):
        """Produce text documentation for a given class object."""
        realname = object.__name__
        name = name or realname
        bases = object.__bases__

        def makename(c, m=object.__module__):
            return classname(c, m)

        if name == realname:
            title = 'class ' + self.bold(realname)
        else:
            title = self.bold(name) + ' = class ' + realname
        if bases:
            parents = map(makename, bases)
            title = title + '(%s)' % ', '.join(parents)

        contents = []
        push = contents.append

        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != '()':
                push(name + argspec + '\n')

        doc = getdoc(object)
        if doc:
            push(doc + '\n')

        # List the mro, if non-trivial.
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            push("Method resolution order:")
            for base in mro:
                push('    ' + makename(base))
            push('')

        # List the built-in subclasses, if any:
        subclasses = sorted(
            (str(cls.__name__) for cls in type.__subclasses__(object)
             if not cls.__name__.startswith("_") and cls.__module__ == "builtins"),
            key=str.lower
        )
        no_of_subclasses = len(subclasses)
        MAX_SUBCLASSES_TO_DISPLAY = 4
        if subclasses:
            push("Built-in subclasses:")
            for subclassname in subclasses[:MAX_SUBCLASSES_TO_DISPLAY]:
                push('    ' + subclassname)
            if no_of_subclasses > MAX_SUBCLASSES_TO_DISPLAY:
                push('    ... and ' +
                     str(no_of_subclasses - MAX_SUBCLASSES_TO_DISPLAY) +
                     ' other subclasses')
            push('')

        # Cute little class to pump out a horizontal rule between sections.
        class HorizontalRule:
            def __init__(self):
                self.needone = 0
            def maybe(self):
                if self.needone:
                    push('-' * 70)
                self.needone = 1
        hr = HorizontalRule()

        def spill(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        # Some descriptors may meet a failure in their __get__.
                        # (bug #1785)
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value,
                                        name, mod, object))
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    doc = getdoc(value)
                    try:
                        obj = getattr(object, name)
                    except AttributeError:
                        obj = homecls.__dict__[name]
                    push(self.docother(obj, name, mod, maxlen=70, doc=doc) +
                         '\n')
            return attrs

        attrs = [(name, kind, cls, value)
                 for name, kind, cls, value in classify_class_attrs(object)
                 if visiblename(name, obj=object)]

        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            attrs, inherited = _split_list(attrs, lambda t: t[2] is thisclass)

            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = "defined here"
            else:
                tag = "inherited from %s" % classname(thisclass,
                                                      object.__module__)

            sort_attributes(attrs, object)

            # Pump out the attrs, segregated by kind.
            attrs = spill("Methods %s:\n" % tag, attrs,
                          lambda t: t[1] == 'method')
            attrs = spill("Class methods %s:\n" % tag, attrs,
                          lambda t: t[1] == 'class method')
            attrs = spill("Static methods %s:\n" % tag, attrs,
                          lambda t: t[1] == 'static method')
            attrs = spilldescriptors("Readonly properties %s:\n" % tag, attrs,
                                     lambda t: t[1] == 'readonly property')
            attrs = spilldescriptors("Data descriptors %s:\n" % tag, attrs,
                                     lambda t: t[1] == 'data descriptor')
            attrs = spilldata("Data and other attributes %s:\n" % tag, attrs,
                              lambda t: t[1] == 'data')

            assert attrs == []
            attrs = inherited

        contents = '\n'.join(contents)
        if not contents:
            return title + '\n'
        return title + '\n' + self.indent(contents.rstrip(), ' |  ') + '\n'

    def formatvalue(self, object):
        """Format an argument default value as text."""
        return '=' + self.repr(object)

    def docroutine(self, object, name=None, mod=None, cl=None):
        """Produce text documentation for a function or method object."""
        realname = object.__name__
        name = name or realname
        note = ''
        skipdocs = 0
        if _is_bound_method(object):
            imclass = object.__self__.__class__
            if cl:
                if imclass is not cl:
                    note = ' from ' + classname(imclass, mod)
            else:
                if object.__self__ is not None:
                    note = ' method of %s instance' % classname(
                        object.__self__.__class__, mod)
                else:
                    note = ' unbound %s method' % classname(imclass,mod)

        if (inspect.iscoroutinefunction(object) or
                inspect.isasyncgenfunction(object)):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''

        if name == realname:
            title = self.bold(realname)
        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                skipdocs = 1
            title = self.bold(name) + ' = ' + realname
        argspec = None

        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                argspec = str(signature)
                if realname == '<lambda>':
                    title = self.bold(name) + ' lambda '
                    # XXX lambda's won't usually have func_annotations['return']
                    # since the syntax doesn't support but it is possible.
                    # So removing parentheses isn't truly safe.
                    argspec = argspec[1:-1] # remove parentheses
        if not argspec:
            argspec = '(...)'
        decl = asyncqualifier + title + argspec + note

        if skipdocs:
            return decl + '\n'
        else:
            doc = getdoc(object) or ''
            return decl + '\n' + (doc and self.indent(doc).rstrip() + '\n')

    def docdata(self, object, name=None, mod=None, cl=None):
        """Produce text documentation for a data descriptor."""
        results = []
        push = results.append

        if name:
            push(self.bold(name))
            push('\n')
        doc = getdoc(object) or ''
        if doc:
            push(self.indent(doc))
            push('\n')
        return ''.join(results)

    docproperty = docdata

    def docother(self, object, name=None, mod=None, parent=None, maxlen=None, doc=None):
        """Produce text documentation for a data object."""
        repr = self.repr(object)
        if maxlen:
            line = (name and name + ' = ' or '') + repr
            chop = maxlen - len(line)
            if chop < 0: repr = repr[:chop] + '...'
        line = (name and self.bold(name) + ' = ' or '') + repr
        if not doc:
            doc = getdoc(object)
        if doc:
            line += '\n' + self.indent(str(doc)) + '\n'
        return line

class _PlainTextDoc(TextDoc):
    """Subclass of TextDoc which overrides string styling"""
    def bold(self, text):
        return text

# --------------------------------------------------------- user interfaces

def pager(text):
    """The first time this is called, determine what kind of pager to use."""
    global pager
    pager = getpager()
    pager(text)

def getpager():
    """Decide what method to use for paging through text."""
    if not hasattr(sys.stdin, "isatty"):
        return plainpager
    if not hasattr(sys.stdout, "isatty"):
        return plainpager
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return plainpager
    use_pager = os.environ.get('MANPAGER') or os.environ.get('PAGER')
    if use_pager:
        if sys.platform == 'win32': # pipes completely broken in Windows
            return lambda text: tempfilepager(plain(text), use_pager)
        elif os.environ.get('TERM') in ('dumb', 'emacs'):
            return lambda text: pipepager(plain(text), use_pager)
        else:
            return lambda text: pipepager(text, use_pager)
    if os.environ.get('TERM') in ('dumb', 'emacs'):
        return plainpager
    if sys.platform == 'win32':
        return lambda text: tempfilepager(plain(text), 'more <')
    if hasattr(os, 'system') and os.system('(pager) 2>/dev/null') == 0:
        return lambda text: pipepager(text, 'pager')
    if hasattr(os, 'system') and os.system('(less) 2>/dev/null') == 0:
        return lambda text: pipepager(text, 'less')

    import tempfile
    (fd, filename) = tempfile.mkstemp()
    os.close(fd)
    try:
        if hasattr(os, 'system') and os.system('more "%s"' % filename) == 0:
            return lambda text: pipepager(text, 'more')
        else:
            return ttypager
    finally:
        os.unlink(filename)

def plain(text):
    """Remove boldface formatting from text."""
    return re.sub('.\b', '', text)

def pipepager(text, cmd):
    """Page through text by feeding it to another program."""
    import subprocess
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                            errors='backslashreplace')
    try:
        with proc.stdin as pipe:
            try:
                pipe.write(text)
            except KeyboardInterrupt:
                # We've hereby abandoned whatever text hasn't been written,
                # but the pager is still in control of the terminal.
                pass
    except OSError:
        pass # Ignore broken pipes caused by quitting the pager program.
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            # Ignore ctl-c like the pager itself does.  Otherwise the pager is
            # left running and the terminal is in raw mode and unusable.
            pass

def tempfilepager(text, cmd):
    """Page through text by invoking a program on a temporary file."""
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'pydoc.out')
        with open(filename, 'w', errors='backslashreplace',
                  encoding=os.device_encoding(0) if
                  sys.platform == 'win32' else None
                  ) as file:
            file.write(text)
        os.system(cmd + ' "' + filename + '"')

def _escape_stdout(text):
    # Escape non-encodable characters to avoid encoding errors later
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    return text.encode(encoding, 'backslashreplace').decode(encoding)

def ttypager(text):
    """Page through text on a text terminal."""
    lines = plain(_escape_stdout(text)).split('\n')
    try:
        import tty
        fd = sys.stdin.fileno()
        old = tty.tcgetattr(fd)
        tty.setcbreak(fd)
        getchar = lambda: sys.stdin.read(1)
    except (ImportError, AttributeError, io.UnsupportedOperation):
        tty = None
        getchar = lambda: sys.stdin.readline()[:-1][:1]

    try:
        try:
            h = int(os.environ.get('LINES', 0))
        except ValueError:
            h = 0
        if h <= 1:
            h = 25
        r = inc = h - 1
        sys.stdout.write('\n'.join(lines[:inc]) + '\n')
        while lines[r:]:
            sys.stdout.write('-- more --')
            sys.stdout.flush()
            c = getchar()

            if c in ('q', 'Q'):
                sys.stdout.write('\r          \r')
                break
            elif c in ('\r', '\n'):
                sys.stdout.write('\r          \r' + lines[r] + '\n')
                r = r + 1
                continue
            if c in ('b', 'B', '\x1b'):
                r = r - inc - inc
                if r < 0: r = 0
            sys.stdout.write('\n' + '\n'.join(lines[r:r+inc]) + '\n')
            r = r + inc

    finally:
        if tty:
            tty.tcsetattr(fd, tty.TCSAFLUSH, old)

def plainpager(text):
    """Simply print unformatted text.  This is the ultimate fallback."""
    sys.stdout.write(plain(_escape_stdout(text)))

def describe(thing):
    """Produce a short description of the given thing."""
    if inspect.ismodule(thing):
        if thing.__name__ in sys.builtin_module_names:
            return 'built-in module ' + thing.__name__
        if hasattr(thing, '__path__'):
            return 'package ' + thing.__name__
        else:
            return 'module ' + thing.__name__
    if inspect.isbuiltin(thing):
        return 'built-in function ' + thing.__name__
    if inspect.isgetsetdescriptor(thing):
        return 'getset descriptor %s.%s.%s' % (
            thing.__objclass__.__module__, thing.__objclass__.__name__,
            thing.__name__)
    if inspect.ismemberdescriptor(thing):
        return 'member descriptor %s.%s.%s' % (
            thing.__objclass__.__module__, thing.__objclass__.__name__,
            thing.__name__)
    if _isclass(thing):
        return 'class ' + thing.__name__
    if inspect.isfunction(thing):
        return 'function ' + thing.__name__
    if inspect.ismethod(thing):
        return 'method ' + thing.__name__
    return type(thing).__name__

def locate(path, forceload=0):
    """Locate an object by name or dotted path, importing as necessary."""
    parts = [part for part in path.split('.') if part]
    module, n = None, 0
    while n < len(parts):
        nextmodule = safeimport('.'.join(parts[:n+1]), forceload)
        if nextmodule: module, n = nextmodule, n + 1
        else: break
    if module:
        object = module
    else:
        object = builtins
    for part in parts[n:]:
        try:
            object = getattr(object, part)
        except AttributeError:
            return None
    return object

# --------------------------------------- interactive interpreter interface

text = TextDoc()
plaintext = _PlainTextDoc()
html = HTMLDoc()

def resolve(thing, forceload=0):
    """Given an object or a path to an object, get the object and its name."""
    if isinstance(thing, str):
        object = locate(thing, forceload)
        if object is None:
            raise ImportError('''\
No Python documentation found for %r.
Use help() to get the interactive help utility.
Use help(str) for help on the str class.''' % thing)
        return object, thing
    else:
        name = getattr(thing, '__name__', None)
        return thing, name if isinstance(name, str) else None

def render_doc(thing, title='Python Library Documentation: %s', forceload=0,
        renderer=None):
    """Render text documentation, given an object or a path to an object."""
    if renderer is None:
        renderer = text
    object, name = resolve(thing, forceload)
    desc = describe(object)
    module = inspect.getmodule(object)
    if name and '.' in name:
        desc += ' in ' + name[:name.rfind('.')]
    elif module and module is not object:
        desc += ' in module ' + module.__name__

    if not (inspect.ismodule(object) or
              _isclass(object) or
              inspect.isroutine(object) or
              inspect.isdatadescriptor(object) or
              _getdoc(object)):
        # If the passed object is a piece of data or an instance,
        # document its available methods instead of its value.
        if hasattr(object, '__origin__'):
            object = object.__origin__
        else:
            object = type(object)
            desc += ' object'
    return title % desc + '\n\n' + renderer.document(object, name)

def doc(thing, title='Python Library Documentation: %s', forceload=0,
        output=None):
    """Display text documentation, given an object or a path to an object."""
    try:
        if output is None:
            pager(render_doc(thing, title, forceload))
        else:
            output.write(render_doc(thing, title, forceload, plaintext))
    except (ImportError, ErrorDuringImport) as value:
        print(value)

def writedoc(thing, forceload=0):
    """Write HTML documentation to a file in the current directory."""
    try:
        object, name = resolve(thing, forceload)
        page = html.page(describe(object), html.document(object, name))
        with open(name + '.html', 'w', encoding='utf-8') as file:
            file.write(page)
        print('wrote', name + '.html')
    except (ImportError, ErrorDuringImport) as value:
        print(value)

def writedocs(dir, pkgpath='', done=None):
    """Write out HTML documentation for all modules in a directory tree."""
    if done is None: done = {}
    for importer, modname, ispkg in pkgutil.walk_packages([dir], pkgpath):
        writedoc(modname)
    return

class Helper:

    # These dictionaries map a topic name to either an alias, or a tuple
    # (label, seealso-items).  The "label" is the label of the corresponding
    # section in the .rst file under Doc/ and an index into the dictionary
    # in pydoc_data/topics.py.
    #
    # CAUTION: if you change one of these dictionaries, be sure to adapt the
    #          list of needed labels in Doc/tools/extensions/pyspecific.py and
    #          regenerate the pydoc_data/topics.py file by running
    #              make pydoc-topics
    #          in Doc/ and copying the output file into the Lib/ directory.

    keywords = {
        'False': '',
        'None': '',
        'True': '',
        'and': 'BOOLEAN',
        'as': 'with',
        'assert': ('assert', ''),
        'async': ('async', ''),
        'await': ('await', ''),
        'break': ('break', 'while for'),
        'class': ('class', 'CLASSES SPECIALMETHODS'),
        'continue': ('continue', 'while for'),
        'def': ('function', ''),
        'del': ('del', 'BASICMETHODS'),
        'elif': 'if',
        'else': ('else', 'while for'),
        'except': 'try',
        'finally': 'try',
        'for': ('for', 'break continue while'),
        'from': 'import',
        'global': ('global', 'nonlocal NAMESPACES'),
        'if': ('if', 'TRUTHVALUE'),
        'import': ('import', 'MODULES'),
        'in': ('in', 'SEQUENCEMETHODS'),
        'is': 'COMPARISON',
        'lambda': ('lambda', 'FUNCTIONS'),
        'nonlocal': ('nonlocal', 'global NAMESPACES'),
        'not': 'BOOLEAN',
        'or': 'BOOLEAN',
        'pass': ('pass', ''),
        'raise': ('raise', 'EXCEPTIONS'),
        'return': ('return', 'FUNCTIONS'),
        'try': ('try', 'EXCEPTIONS'),
        'while': ('while', 'break continue if TRUTHVALUE'),
        'with': ('with', 'CONTEXTMANAGERS EXCEPTIONS yield'),
        'yield': ('yield', ''),
    }
    # Either add symbols to this dictionary or to the symbols dictionary
    # directly: Whichever is easier. They are merged later.
    _strprefixes = [p + q for p in ('b', 'f', 'r', 'u') for q in ("'", '"')]
    _symbols_inverse = {
        'STRINGS' : ("'", "'''", '"', '"""', *_strprefixes),
        'OPERATORS' : ('+', '-', '*', '**', '/', '//', '%', '<<', '>>', '&',
                       '|', '^', '~', '<', '>', '<=', '>=', '==', '!=', '<>'),
        'COMPARISON' : ('<', '>', '<=', '>=', '==', '!=', '<>'),
        'UNARY' : ('-', '~'),
        'AUGMENTEDASSIGNMENT' : ('+=', '-=', '*=', '/=', '%=', '&=', '|=',
                                '^=', '<<=', '>>=', '**=', '//='),
        'BITWISE' : ('<<', '>>', '&', '|', '^', '~'),
        'COMPLEX' : ('j', 'J')
    }
    symbols = {
        '%': 'OPERATORS FORMATTING',
        '**': 'POWER',
        ',': 'TUPLES LISTS FUNCTIONS',
        '.': 'ATTRIBUTES FLOAT MODULES OBJECTS',
        '...': 'ELLIPSIS',
        ':': 'SLICINGS DICTIONARYLITERALS',
        '@': 'def class',
        '\\': 'STRINGS',
        '_': 'PRIVATENAMES',
        '__': 'PRIVATENAMES SPECIALMETHODS',
        '`': 'BACKQUOTES',
        '(': 'TUPLES FUNCTIONS CALLS',
        ')': 'TUPLES FUNCTIONS CALLS',
        '[': 'LISTS SUBSCRIPTS SLICINGS',
        ']': 'LISTS SUBSCRIPTS SLICINGS'
    }
    for topic, symbols_ in _symbols_inverse.items():
        for symbol in symbols_:
            topics = symbols.get(symbol, topic)
            if topic not in topics:
                topics = topics + ' ' + topic
            symbols[symbol] = topics

    topics = {
        'TYPES': ('types', 'STRINGS UNICODE NUMBERS SEQUENCES MAPPINGS '
                  'FUNCTIONS CLASSES MODULES FILES inspect'),
        'STRINGS': ('strings', 'str UNICODE SEQUENCES STRINGMETHODS '
                    'FORMATTING TYPES'),
        'STRINGMETHODS': ('string-methods', 'STRINGS FORMATTING'),
        'FORMATTING': ('formatstrings', 'OPERATORS'),
        'UNICODE': ('strings', 'encodings unicode SEQUENCES STRINGMETHODS '
                    'FORMATTING TYPES'),
        'NUMBERS': ('numbers', 'INTEGER FLOAT COMPLEX TYPES'),
        'INTEGER': ('integers', 'int range'),
        'FLOAT': ('floating', 'float math'),
        'COMPLEX': ('imaginary', 'complex cmath'),
        'SEQUENCES': ('typesseq', 'STRINGMETHODS FORMATTING range LISTS'),
        'MAPPINGS': 'DICTIONARIES',
        'FUNCTIONS': ('typesfunctions', 'def TYPES'),
        'METHODS': ('typesmethods', 'class def CLASSES TYPES'),
        'CODEOBJECTS': ('bltin-code-objects', 'compile FUNCTIONS TYPES'),
        'TYPEOBJECTS': ('bltin-type-objects', 'types TYPES'),
        'FRAMEOBJECTS': 'TYPES',
        'TRACEBACKS': 'TYPES',
        'NONE': ('bltin-null-object', ''),
        'ELLIPSIS': ('bltin-ellipsis-object', 'SLICINGS'),
        'SPECIALATTRIBUTES': ('specialattrs', ''),
        'CLASSES': ('types', 'class SPECIALMETHODS PRIVATENAMES'),
        'MODULES': ('typesmodules', 'import'),
        'PACKAGES': 'import',
        'EXPRESSIONS': ('operator-summary', 'lambda or and not in is BOOLEAN '
                        'COMPARISON BITWISE SHIFTING BINARY FORMATTING POWER '
                        'UNARY ATTRIBUTES SUBSCRIPTS SLICINGS CALLS TUPLES '
                        'LISTS DICTIONARIES'),
        'OPERATORS': 'EXPRESSIONS',
        'PRECEDENCE': 'EXPRESSIONS',
        'OBJECTS': ('objects', 'TYPES'),
        'SPECIALMETHODS': ('specialnames', 'BASICMETHODS ATTRIBUTEMETHODS '
                           'CALLABLEMETHODS SEQUENCEMETHODS MAPPINGMETHODS '
                           'NUMBERMETHODS CLASSES'),
        'BASICMETHODS': ('customization', 'hash repr str SPECIALMETHODS'),
        'ATTRIBUTEMETHODS': ('attribute-access', 'ATTRIBUTES SPECIALMETHODS'),
        'CALLABLEMETHODS': ('callable-types', 'CALLS SPECIALMETHODS'),
        'SEQUENCEMETHODS': ('sequence-types', 'SEQUENCES SEQUENCEMETHODS '
                             'SPECIALMETHODS'),
        'MAPPINGMETHODS': ('sequence-types', 'MAPPINGS SPECIALMETHODS'),
        'NUMBERMETHODS': ('numeric-types', 'NUMBERS AUGMENTEDASSIGNMENT '
                          'SPECIALMETHODS'),
        'EXECUTION': ('execmodel', 'NAMESPACES DYNAMICFEATURES EXCEPTIONS'),
        'NAMESPACES': ('naming', 'global nonlocal ASSIGNMENT DELETION DYNAMICFEATURES'),
        'DYNAMICFEATURES': ('dynamic-features', ''),
        'SCOPING': 'NAMESPACES',
        'FRAMES': 'NAMESPACES',
        'EXCEPTIONS': ('exceptions', 'try except finally raise'),
        'CONVERSIONS': ('conversions', ''),
        'IDENTIFIERS': ('identifiers', 'keywords SPECIALIDENTIFIERS'),
        'SPECIALIDENTIFIERS': ('id-classes', ''),
        'PRIVATENAMES': ('atom-identifiers', ''),
        'LITERALS': ('atom-literals', 'STRINGS NUMBERS TUPLELITERALS '
                     'LISTLITERALS DICTIONARYLITERALS'),
        'TUPLES': 'SEQUENCES',
        'TUPLELITERALS': ('exprlists', 'TUPLES LITERALS'),
        'LISTS': ('typesseq-mutable', 'LISTLITERALS'),
        'LISTLITERALS': ('lists', 'LISTS LITERALS'),
        'DICTIONARIES': ('typesmapping', 'DICTIONARYLITERALS'),
        'DICTIONARYLITERALS': ('dict', 'DICTIONARIES LITERALS'),
        'ATTRIBUTES': ('attribute-references', 'getattr hasattr setattr ATTRIBUTEMETHODS'),
        'SUBSCRIPTS': ('subscriptions', 'SEQUENCEMETHODS'),
        'SLICINGS': ('slicings', 'SEQUENCEMETHODS'),
        'CALLS': ('calls', 'EXPRESSIONS'),
        'POWER': ('power', 'EXPRESSIONS'),
        'UNARY': ('unary', 'EXPRESSIONS'),
        'BINARY': ('binary', 'EXPRESSIONS'),
        'SHIFTING': ('shifting', 'EXPRESSIONS'),
        'BITWISE': ('bitwise', 'EXPRESSIONS'),
        'COMPARISON': ('comparisons', 'EXPRESSIONS BASICMETHODS'),
        'BOOLEAN': ('booleans', 'EXPRESSIONS TRUTHVALUE'),
        'ASSERTION': 'assert',
        'ASSIGNMENT': ('assignment', 'AUGMENTEDASSIGNMENT'),
        'AUGMENTEDASSIGNMENT': ('augassign', 'NUMBERMETHODS'),
        'DELETION': 'del',
        'RETURNING': 'return',
        'IMPORTING': 'import',
        'CONDITIONAL': 'if',
        'LOOPING': ('compound', 'for while break continue'),
        'TRUTHVALUE': ('truth', 'if while and or not BASICMETHODS'),
        'DEBUGGING': ('debugger', 'pdb'),
        'CONTEXTMANAGERS': ('context-managers', 'with'),
    }

    def __init__(self, input=None, output=None):
        self._input = input
        self._output = output

    @property
    def input(self):
        return self._input or sys.stdin

    @property
    def output(self):
        return self._output or sys.stdout

    def __repr__(self):
        if inspect.stack()[1][3] == '?':
            self()
            return ''
        return '<%s.%s instance>' % (self.__class__.__module__,
                                     self.__class__.__qualname__)

    _GoInteractive = object()
    def __call__(self, request=_GoInteractive):
        if request is not self._GoInteractive:
            self.help(request)
        else:
            self.intro()
            self.interact()
            self.output.write('''
You are now leaving help and returning to the Python interpreter.
If you want to ask for help on a particular object directly from the
interpreter, you can type "help(object)".  Executing "help('string')"
has the same effect as typing a particular string at the help> prompt.
''')

    def interact(self):
        self.output.write('\n')
        while True:
            try:
                request = self.getline('help> ')
                if not request: break
            except (KeyboardInterrupt, EOFError):
                break
            request = request.strip()

            # Make sure significant trailing quoting marks of literals don't
            # get deleted while cleaning input
            if (len(request) > 2 and request[0] == request[-1] in ("'", '"')
                    and request[0] not in request[1:-1]):
                request = request[1:-1]
            if request.lower() in ('q', 'quit'): break
            if request == 'help':
                self.intro()
            else:
                self.help(request)

    def getline(self, prompt):
        """Read one line, using input() when appropriate."""
        if self.input is sys.stdin:
            return input(prompt)
        else:
            self.output.write(prompt)
            self.output.flush()
            return self.input.readline()

    def help(self, request):
        if type(request) is type(''):
            request = request.strip()
            if request == 'keywords': self.listkeywords()
            elif request == 'symbols': self.listsymbols()
            elif request == 'topics': self.listtopics()
            elif request == 'modules': self.listmodules()
            elif request[:8] == 'modules ':
                self.listmodules(request.split()[1])
            elif request in self.symbols: self.showsymbol(request)
            elif request in ['True', 'False', 'None']:
                # special case these keywords since they are objects too
                doc(eval(request), 'Help on %s:')
            elif request in self.keywords: self.showtopic(request)
            elif request in self.topics: self.showtopic(request)
            elif request: doc(request, 'Help on %s:', output=self._output)
            else: doc(str, 'Help on %s:', output=self._output)
        elif isinstance(request, Helper): self()
        else: doc(request, 'Help on %s:', output=self._output)
        self.output.write('\n')

    def intro(self):
        self.output.write('''
Welcome to Python {0}'s help utility!

If this is your first time using Python, you should definitely check out
the tutorial on the internet at https://docs.python.org/{0}/tutorial/.

Enter the name of any module, keyword, or topic to get help on writing
Python programs and using Python modules.  To quit this help utility and
return to the interpreter, just type "quit".

To get a list of available modules, keywords, symbols, or topics, type
"modules", "keywords", "symbols", or "topics".  Each module also comes
with a one-line summary of what it does; to list the modules whose name
or summary contain a given string such as "spam", type "modules spam".
'''.format('%d.%d' % sys.version_info[:2]))

    def list(self, items, columns=4, width=80):
        items = list(sorted(items))
        colw = width // columns
        rows = (len(items) + columns - 1) // columns
        for row in range(rows):
            for col in range(columns):
                i = col * rows + row
                if i < len(items):
                    self.output.write(items[i])
                    if col < columns - 1:
                        self.output.write(' ' + ' ' * (colw - 1 - len(items[i])))
            self.output.write('\n')

    def listkeywords(self):
        self.output.write('''
Here is a list of the Python keywords.  Enter any keyword to get more help.

''')
        self.list(self.keywords.keys())

    def listsymbols(self):
        self.output.write('''
Here is a list of the punctuation symbols which Python assigns special meaning
to. Enter any symbol to get more help.

''')
        self.list(self.symbols.keys())

    def listtopics(self):
        self.output.write('''
Here is a list of available topics.  Enter any topic name to get more help.

''')
        self.list(self.topics.keys())

    def showtopic(self, topic, more_xrefs=''):
        try:
            import pydoc_data.topics
        except ImportError:
            self.output.write('''
Sorry, topic and keyword documentation is not available because the
module "pydoc_data.topics" could not be found.
''')
            return
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        if type(target) is type(''):
            return self.showtopic(target, more_xrefs)

        label, xrefs = target
        try:
            doc = pydoc_data.topics.topics[label]
        except KeyError:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        doc = doc.strip() + '\n'
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        if xrefs:
            import textwrap
            text = 'Related help topics: ' + ', '.join(xrefs.split()) + '\n'
            wrapped_text = textwrap.wrap(text, 72)
            doc += '\n%s\n' % '\n'.join(wrapped_text)
        pager(doc)

    def _gettopic(self, topic, more_xrefs=''):
        """Return unbuffered tuple of (topic, xrefs).

        If an error occurs here, the exception is caught and displayed by
        the url handler.

        This function duplicates the showtopic method but returns its
        result directly so it can be formatted for display in an html page.
        """
        try:
            import pydoc_data.topics
        except ImportError:
            return('''
Sorry, topic and keyword documentation is not available because the
module "pydoc_data.topics" could not be found.
''' , '')
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            raise ValueError('could not find topic')
        if isinstance(target, str):
            return self._gettopic(target, more_xrefs)
        label, xrefs = target
        doc = pydoc_data.topics.topics[label]
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        return doc, xrefs

    def showsymbol(self, symbol):
        target = self.symbols[symbol]
        topic, _, xrefs = target.partition(' ')
        self.showtopic(topic, xrefs)

    def listmodules(self, key=''):
        if key:
            self.output.write('''
Here is a list of modules whose name or summary contains '{}'.
If there are any, enter a module name to get more help.

'''.format(key))
            apropos(key)
        else:
            self.output.write('''
Please wait a moment while I gather a list of all available modules...

''')
            modules = {}
            def callback(path, modname, desc, modules=modules):
                if modname and modname[-9:] == '.__init__':
                    modname = modname[:-9] + ' (package)'
                if modname.find('.') < 0:
                    modules[modname] = 1
            def onerror(modname):
                callback(None, modname, None)
            ModuleScanner().run(callback, onerror=onerror)
            self.list(modules.keys())
            self.output.write('''
Enter any module name to get more help.  Or, type "modules spam" to search
for modules whose name or summary contain the string "spam".
''')

help = Helper()

class ModuleScanner:
    """An interruptible scanner that searches module synopses."""

    def run(self, callback, key=None, completer=None, onerror=None):
        if key: key = key.lower()
        self.quit = False
        seen = {}

        for modname in sys.builtin_module_names:
            if modname != '__main__':
                seen[modname] = 1
                if key is None:
                    callback(None, modname, '')
                else:
                    name = __import__(modname).__doc__ or ''
                    desc = name.split('\n')[0]
                    name = modname + ' - ' + desc
                    if name.lower().find(key) >= 0:
                        callback(None, modname, desc)

        for importer, modname, ispkg in pkgutil.walk_packages(onerror=onerror):
            if self.quit:
                break

            if key is None:
                callback(None, modname, '')
            else:
                try:
                    spec = pkgutil._get_spec(importer, modname)
                except SyntaxError:
                    # raised by tests for bad coding cookies or BOM
                    continue
                loader = spec.loader
                if hasattr(loader, 'get_source'):
                    try:
                        source = loader.get_source(modname)
                    except Exception:
                        if onerror:
                            onerror(modname)
                        continue
                    desc = source_synopsis(io.StringIO(source)) or ''
                    if hasattr(loader, 'get_filename'):
                        path = loader.get_filename(modname)
                    else:
                        path = None
                else:
                    try:
                        module = importlib._bootstrap._load(spec)
                    except ImportError:
                        if onerror:
                            onerror(modname)
                        continue
                    desc = module.__doc__.splitlines()[0] if module.__doc__ else ''
                    path = getattr(module,'__file__',None)
                name = modname + ' - ' + desc
                if name.lower().find(key) >= 0:
                    callback(path, modname, desc)

        if completer:
            completer()

def apropos(key):
    """Print all the one-line module summaries that contain a substring."""
    def callback(path, modname, desc):
        if modname[-9:] == '.__init__':
            modname = modname[:-9] + ' (package)'
        print(modname, desc and '- ' + desc)
    def onerror(modname):
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore') # ignore problems during import
        ModuleScanner().run(callback, key, onerror=onerror)

# --------------------------------------- enhanced web browser interface

def _start_server(urlhandler, hostname, port):
    """Start an HTTP server thread on a specific port.

    Start an HTML/text server thread, so HTML or text documents can be
    browsed dynamically and interactively with a web browser.  Example use:

        >>> import time
        >>> import pydoc

        Define a URL handler.  To determine what the client is asking
        for, check the URL and content_type.

        Then get or generate some text or HTML code and return it.

        >>> def my_url_handler(url, content_type):
        ...     text = 'the URL sent was: (%s, %s)' % (url, content_type)
        ...     return text

        Start server thread on port 0.
        If you use port 0, the server will pick a random port number.
        You can then use serverthread.port to get the port number.

        >>> port = 0
        >>> serverthread = pydoc._start_server(my_url_handler, port)

        Check that the server is really started.  If it is, open browser
        and get first page.  Use serverthread.url as the starting page.

        >>> if serverthread.serving:
        ...    import webbrowser

        The next two lines are commented out so a browser doesn't open if
        doctest is run on this module.

        #...    webbrowser.open(serverthread.url)
        #True

        Let the server do its thing. We just need to monitor its status.
        Use time.sleep so the loop doesn't hog the CPU.

        >>> starttime = time.monotonic()
        >>> timeout = 1                    #seconds

        This is a short timeout for testing purposes.

        >>> while serverthread.serving:
        ...     time.sleep(.01)
        ...     if serverthread.serving and time.monotonic() - starttime > timeout:
        ...          serverthread.stop()
        ...          break

        Print any errors that may have occurred.

        >>> print(serverthread.error)
        None
   """
    import http.server
    import email.message
    import select
    import threading

    class DocHandler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            """Process a request from an HTML browser.

            The URL received is in self.path.
            Get an HTML page from self.urlhandler and send it.
            """
            if self.path.endswith('.css'):
                content_type = 'text/css'
            else:
                content_type = 'text/html'
            self.send_response(200)
            self.send_header('Content-Type', '%s; charset=UTF-8' % content_type)
            self.end_headers()
            self.wfile.write(self.urlhandler(
                self.path, content_type).encode('utf-8'))

        def log_message(self, *args):
            # Don't log messages.
            pass

    class DocServer(http.server.HTTPServer):

        def __init__(self, host, port, callback):
            self.host = host
            self.address = (self.host, port)
            self.callback = callback
            self.base.__init__(self, self.address, self.handler)
            self.quit = False

        def serve_until_quit(self):
            while not self.quit:
                rd, wr, ex = select.select([self.socket.fileno()], [], [], 1)
                if rd:
                    self.handle_request()
            self.server_close()

        def server_activate(self):
            self.base.server_activate(self)
            if self.callback:
                self.callback(self)

    class ServerThread(threading.Thread):

        def __init__(self, urlhandler, host, port):
            self.urlhandler = urlhandler
            self.host = host
            self.port = int(port)
            threading.Thread.__init__(self)
            self.serving = False
            self.error = None

        def run(self):
            """Start the server."""
            try:
                DocServer.base = http.server.HTTPServer
                DocServer.handler = DocHandler
                DocHandler.MessageClass = email.message.Message
                DocHandler.urlhandler = staticmethod(self.urlhandler)
                docsvr = DocServer(self.host, self.port, self.ready)
                self.docserver = docsvr
                docsvr.serve_until_quit()
            except Exception as e:
                self.error = e

        def ready(self, server):
            self.serving = True
            self.host = server.host
            self.port = server.server_port
            self.url = 'http://%s:%d/' % (self.host, self.port)

        def stop(self):
            """Stop the server and this thread nicely"""
            self.docserver.quit = True
            self.join()
            # explicitly break a reference cycle: DocServer.callback
            # has indirectly a reference to ServerThread.
            self.docserver = None
            self.serving = False
            self.url = None

    thread = ServerThread(urlhandler, hostname, port)
    thread.start()
    # Wait until thread.serving is True to make sure we are
    # really up before returning.
    while not thread.error and not thread.serving:
        time.sleep(.01)
    return thread


def _url_handler(url, content_type="text/html"):
    """The pydoc url handler for use with the pydoc server.

    If the content_type is 'text/css', the _pydoc.css style
    sheet is read and returned if it exits.

    If the content_type is 'text/html', then the result of
    get_html_page(url) is returned.
    """
    class _HTMLDoc(HTMLDoc):

        def page(self, title, contents):
            """Format an HTML page."""
            css_path = "pydoc_data/_pydoc.css"
            css_link = (
                '<link rel="stylesheet" type="text/css" href="%s">' %
                css_path)
            return '''\
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html><head><title>Pydoc: %s</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
%s</head><body bgcolor="#f0f0f8">%s<div style="clear:both;padding-top:.5em;">%s</div>
</body></html>''' % (title, css_link, html_navbar(), contents)


    html = _HTMLDoc()

    def html_navbar():
        version = html.escape("%s [%s, %s]" % (platform.python_version(),
                                               platform.python_build()[0],
                                               platform.python_compiler()))
        return """
            <div style='float:left'>
                Python %s<br>%s
            </div>
            <div style='float:right'>
                <div style='text-align:center'>
                  <a href="index.html">Module Index</a>
                  : <a href="topics.html">Topics</a>
                  : <a href="keywords.html">Keywords</a>
                </div>
                <div>
                    <form action="get" style='display:inline;'>
                      <input type=text name=key size=15>
                      <input type=submit value="Get">
                    </form>&nbsp;
                    <form action="search" style='display:inline;'>
                      <input type=text name=key size=15>
                      <input type=submit value="Search">
                    </form>
                </div>
            </div>
            """ % (version, html.escape(platform.platform(terse=True)))

    def html_index():
        """Module Index page."""

        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)

        heading = html.heading(
            '<big><big><strong>Index of Modules</strong></big></big>',
            '#ffffff', '#7799ee')
        names = [name for name in sys.builtin_module_names
                 if name != '__main__']
        contents = html.multicolumn(names, bltinlink)
        contents = [heading, '<p>' + html.bigsection(
            'Built-in Modules', '#ffffff', '#ee77aa', contents)]

        seen = {}
        for dir in sys.path:
            contents.append(html.index(dir, seen))

        contents.append(
            '<p align=right><font color="#909090" face="helvetica,'
            'arial"><strong>pydoc</strong> by Ka-Ping Yee'
            '&lt;ping@lfw.org&gt;</font>')
        return 'Index of Modules', ''.join(contents)

    def html_search(key):
        """Search results page."""
        # scan for modules
        search_result = []

        def callback(path, modname, desc):
            if modname[-9:] == '.__init__':
                modname = modname[:-9] + ' (package)'
            search_result.append((modname, desc and '- ' + desc))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # ignore problems during import
            def onerror(modname):
                pass
            ModuleScanner().run(callback, key, onerror=onerror)

        # format page
        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)

        results = []
        heading = html.heading(
            '<big><big><strong>Search Results</strong></big></big>',
            '#ffffff', '#7799ee')
        for name, desc in search_result:
            results.append(bltinlink(name) + desc)
        contents = heading + html.bigsection(
            'key = %s' % key, '#ffffff', '#ee77aa', '<br>'.join(results))
        return 'Search Results', contents

    def html_topics():
        """Index of topic texts available."""

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)

        heading = html.heading(
            '<big><big><strong>INDEX</strong></big></big>',
            '#ffffff', '#7799ee')
        names = sorted(Helper.topics.keys())

        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection(
            'Topics', '#ffffff', '#ee77aa', contents)
        return 'Topics', contents

    def html_keywords():
        """Index of keywords."""
        heading = html.heading(
            '<big><big><strong>INDEX</strong></big></big>',
            '#ffffff', '#7799ee')
        names = sorted(Helper.keywords.keys())

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)

        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection(
            'Keywords', '#ffffff', '#ee77aa', contents)
        return 'Keywords', contents

    def html_topicpage(topic):
        """Topic or keyword help page."""
        buf = io.StringIO()
        htmlhelp = Helper(buf, buf)
        contents, xrefs = htmlhelp._gettopic(topic)
        if topic in htmlhelp.keywords:
            title = 'KEYWORD'
        else:
            title = 'TOPIC'
        heading = html.heading(
            '<big><big><strong>%s</strong></big></big>' % title,
            '#ffffff', '#7799ee')
        contents = '<pre>%s</pre>' % html.markup(contents)
        contents = html.bigsection(topic , '#ffffff','#ee77aa', contents)
        if xrefs:
            xrefs = sorted(xrefs.split())

            def bltinlink(name):
                return '<a href="topic?key=%s">%s</a>' % (name, name)

            xrefs = html.multicolumn(xrefs, bltinlink)
            xrefs = html.section('Related help topics: ',
                                 '#ffffff', '#ee77aa', xrefs)
        return ('%s %s' % (title, topic),
                ''.join((heading, contents, xrefs)))

    def html_getobj(url):
        obj = locate(url, forceload=1)
        if obj is None and url != 'None':
            raise ValueError('could not find object')
        title = describe(obj)
        content = html.document(obj, url)
        return title, content

    def html_error(url, exc):
        heading = html.heading(
            '<big><big><strong>Error</strong></big></big>',
            '#ffffff', '#7799ee')
        contents = '<br>'.join(html.escape(line) for line in
                               format_exception_only(type(exc), exc))
        contents = heading + html.bigsection(url, '#ffffff', '#bb0000',
                                             contents)
        return "Error - %s" % url, contents

    def get_html_page(url):
        """Generate an HTML page for url."""
        complete_url = url
        if url.endswith('.html'):
            url = url[:-5]
        try:
            if url in ("", "index"):
                title, content = html_index()
            elif url == "topics":
                title, content = html_topics()
            elif url == "keywords":
                title, content = html_keywords()
            elif '=' in url:
                op, _, url = url.partition('=')
                if op == "search?key":
                    title, content = html_search(url)
                elif op == "topic?key":
                    # try topics first, then objects.
                    try:
                        title, content = html_topicpage(url)
                    except ValueError:
                        title, content = html_getobj(url)
                elif op == "get?key":
                    # try objects first, then topics.
                    if url in ("", "index"):
                        title, content = html_index()
                    else:
                        try:
                            title, content = html_getobj(url)
                        except ValueError:
                            title, content = html_topicpage(url)
                else:
                    raise ValueError('bad pydoc url')
            else:
                title, content = html_getobj(url)
        except Exception as exc:
            # Catch any errors and display them in an error page.
            title, content = html_error(complete_url, exc)
        return html.page(title, content)

    if url.startswith('/'):
        url = url[1:]
    if content_type == 'text/css':
        path_here = os.path.dirname(os.path.realpath(__file__))
        css_path = os.path.join(path_here, url)
        with open(css_path) as fp:
            return ''.join(fp.readlines())
    elif content_type == 'text/html':
        return get_html_page(url)
    # Errors outside the url handler are caught by the server.
    raise TypeError('unknown content type %r for url %s' % (content_type, url))


def browse(port=0, *, open_browser=True, hostname='localhost'):
    """Start the enhanced pydoc web server and open a web browser.

    Use port '0' to start the server on an arbitrary port.
    Set open_browser to False to suppress opening a browser.
    """
    import webbrowser
    serverthread = _start_server(_url_handler, hostname, port)
    if serverthread.error:
        print(serverthread.error)
        return
    if serverthread.serving:
        server_help_msg = 'Server commands: [b]rowser, [q]uit'
        if open_browser:
            webbrowser.open(serverthread.url)
        try:
            print('Server ready at', serverthread.url)
            print(server_help_msg)
            while serverthread.serving:
                cmd = input('server> ')
                cmd = cmd.lower()
                if cmd == 'q':
                    break
                elif cmd == 'b':
                    webbrowser.open(serverthread.url)
                else:
                    print(server_help_msg)
        except (KeyboardInterrupt, EOFError):
            print()
        finally:
            if serverthread.serving:
                serverthread.stop()
                print('Server stopped')


# -------------------------------------------------- command-line interface

def ispath(x):
    return isinstance(x, str) and x.find(os.sep) >= 0

def _get_revised_path(given_path, argv0):
    """Ensures current directory is on returned path, and argv0 directory is not

    Exception: argv0 dir is left alone if it's also pydoc's directory.

    Returns a new path entry list, or None if no adjustment is needed.
    """
    # Scripts may get the current directory in their path by default if they're
    # run with the -m switch, or directly from the current directory.
    # The interactive prompt also allows imports from the current directory.

    # Accordingly, if the current directory is already present, don't make
    # any changes to the given_path
    if '' in given_path or os.curdir in given_path or os.getcwd() in given_path:
        return None

    # Otherwise, add the current directory to the given path, and remove the
    # script directory (as long as the latter isn't also pydoc's directory.
    stdlib_dir = os.path.dirname(__file__)
    script_dir = os.path.dirname(argv0)
    revised_path = given_path.copy()
    if script_dir in given_path and not os.path.samefile(script_dir, stdlib_dir):
        revised_path.remove(script_dir)
    revised_path.insert(0, os.getcwd())
    return revised_path


# Note: the tests only cover _get_revised_path, not _adjust_cli_path itself
def _adjust_cli_sys_path():
    """Ensures current directory is on sys.path, and __main__ directory is not.

    Exception: __main__ dir is left alone if it's also pydoc's directory.
    """
    revised_path = _get_revised_path(sys.path, sys.argv[0])
    if revised_path is not None:
        sys.path[:] = revised_path


def cli():
    """Command-line interface (looks at sys.argv to decide what to do)."""
    import getopt
    class BadUsage(Exception): pass

    _adjust_cli_sys_path()

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'bk:n:p:w')
        writing = False
        start_server = False
        open_browser = False
        port = 0
        hostname = 'localhost'
        for opt, val in opts:
            if opt == '-b':
                start_server = True
                open_browser = True
            if opt == '-k':
                apropos(val)
                return
            if opt == '-p':
                start_server = True
                port = val
            if opt == '-w':
                writing = True
            if opt == '-n':
                start_server = True
                hostname = val

        if start_server:
            browse(port, hostname=hostname, open_browser=open_browser)
            return

        if not args: raise BadUsage
        for arg in args:
            if ispath(arg) and not os.path.exists(arg):
                print('file %r does not exist' % arg)
                break
            try:
                if ispath(arg) and os.path.isfile(arg):
                    arg = importfile(arg)
                if writing:
                    if ispath(arg) and os.path.isdir(arg):
                        writedocs(arg)
                    else:
                        writedoc(arg)
                else:
                    help.help(arg)
            except ErrorDuringImport as value:
                print(value)

    except (getopt.error, BadUsage):
        cmd = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        print("""pydoc - the Python documentation tool

{cmd} <name> ...
    Show text documentation on something.  <name> may be the name of a
    Python keyword, topic, function, module, or package, or a dotted
    reference to a class or function within a module or module in a
    package.  If <name> contains a '{sep}', it is used as the path to a
    Python source file to document. If name is 'keywords', 'topics',
    or 'modules', a listing of these things is displayed.

{cmd} -k <keyword>
    Search for a keyword in the synopsis lines of all available modules.

{cmd} -n <hostname>
    Start an HTTP server with the given hostname (default: localhost).

{cmd} -p <port>
    Start an HTTP server on the given port on the local machine.  Port
    number 0 can be used to get an arbitrary unused port.

{cmd} -b
    Start an HTTP server on an arbitrary unused port and open a web browser
    to interactively browse documentation.  This option can be used in
    combination with -n and/or -p.

{cmd} -w <name> ...
    Write out the HTML documentation for a module to a file in the current
    directory.  If <name> contains a '{sep}', it is treated as a filename; if
    it names a directory, documentation is written for all the contents.
""".format(cmd=cmd, sep=os.sep))

if __name__ == '__main__':
    cli()
<|endoftext|>"""Redo the builtin repr() (representation) but with limits on most sizes."""

__all__ = ["Repr", "repr", "recursive_repr"]

import builtins
from itertools import islice
from _thread import get_ident

def recursive_repr(fillvalue='...'):
    'Decorator to make a repr function return fillvalue for a recursive call'

    def decorating_function(user_function):
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        wrapper.__qualname__ = getattr(user_function, '__qualname__')
        wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
        return wrapper

    return decorating_function

class Repr:

    def __init__(self):
        self.maxlevel = 6
        self.maxtuple = 6
        self.maxlist = 6
        self.maxarray = 5
        self.maxdict = 4
        self.maxset = 6
        self.maxfrozenset = 6
        self.maxdeque = 6
        self.maxstring = 30
        self.maxlong = 40
        self.maxother = 30

    def repr(self, x):
        return self.repr1(x, self.maxlevel)

    def repr1(self, x, level):
        typename = type(x).__name__
        if ' ' in typename:
            parts = typename.split()
            typename = '_'.join(parts)
        if hasattr(self, 'repr_' + typename):
            return getattr(self, 'repr_' + typename)(x, level)
        else:
            return self.repr_instance(x, level)

    def _repr_iterable(self, x, level, left, right, maxiter, trail=''):
        n = len(x)
        if level <= 0 and n:
            s = '...'
        else:
            newlevel = level - 1
            repr1 = self.repr1
            pieces = [repr1(elem, newlevel) for elem in islice(x, maxiter)]
            if n > maxiter:  pieces.append('...')
            s = ', '.join(pieces)
            if n == 1 and trail:  right = trail + right
        return '%s%s%s' % (left, s, right)

    def repr_tuple(self, x, level):
        return self._repr_iterable(x, level, '(', ')', self.maxtuple, ',')

    def repr_list(self, x, level):
        return self._repr_iterable(x, level, '[', ']', self.maxlist)

    def repr_array(self, x, level):
        if not x:
            return "array('%s')" % x.typecode
        header = "array('%s', [" % x.typecode
        return self._repr_iterable(x, level, header, '])', self.maxarray)

    def repr_set(self, x, level):
        if not x:
            return 'set()'
        x = _possibly_sorted(x)
        return self._repr_iterable(x, level, '{', '}', self.maxset)

    def repr_frozenset(self, x, level):
        if not x:
            return 'frozenset()'
        x = _possibly_sorted(x)
        return self._repr_iterable(x, level, 'frozenset({', '})',
                                   self.maxfrozenset)

    def repr_deque(self, x, level):
        return self._repr_iterable(x, level, 'deque([', '])', self.maxdeque)

    def repr_dict(self, x, level):
        n = len(x)
        if n == 0: return '{}'
        if level <= 0: return '{...}'
        newlevel = level - 1
        repr1 = self.repr1
        pieces = []
        for key in islice(_possibly_sorted(x), self.maxdict):
            keyrepr = repr1(key, newlevel)
            valrepr = repr1(x[key], newlevel)
            pieces.append('%s: %s' % (keyrepr, valrepr))
        if n > self.maxdict: pieces.append('...')
        s = ', '.join(pieces)
        return '{%s}' % (s,)

    def repr_str(self, x, level):
        s = builtins.repr(x[:self.maxstring])
        if len(s) > self.maxstring:
            i = max(0, (self.maxstring-3)//2)
            j = max(0, self.maxstring-3-i)
            s = builtins.repr(x[:i] + x[len(x)-j:])
            s = s[:i] + '...' + s[len(s)-j:]
        return s

    def repr_int(self, x, level):
        s = builtins.repr(x) # XXX Hope this isn't too slow...
        if len(s) > self.maxlong:
            i = max(0, (self.maxlong-3)//2)
            j = max(0, self.maxlong-3-i)
            s = s[:i] + '...' + s[len(s)-j:]
        return s

    def repr_instance(self, x, level):
        try:
            s = builtins.repr(x)
            # Bugs in x.__repr__() can cause arbitrary
            # exceptions -- then make up something
        except Exception:
            return '<%s instance at %#x>' % (x.__class__.__name__, id(x))
        if len(s) > self.maxother:
            i = max(0, (self.maxother-3)//2)
            j = max(0, self.maxother-3-i)
            s = s[:i] + '...' + s[len(s)-j:]
        return s


def _possibly_sorted(x):
    # Since not all sequences of items can be sorted and comparison
    # functions may raise arbitrary exceptions, return an unsorted
    # sequence in that case.
    try:
        return sorted(x)
    except Exception:
        return list(x)

aRepr = Repr()
repr = aRepr.repr
<|endoftext|>"""Common operations on Posix pathnames.

Instead of importing this module directly, import os and refer to
this module as os.path.  The "os.path" name is an alias for this
module on Posix systems; on other systems (e.g. Windows),
os.path provides the same operations in a manner specific to that
platform, and is an alias to another module (e.g. ntpath).

Some of this can actually be useful on non-Posix systems too, e.g.
for manipulation of the pathname component of URLs.
"""

# Strings representing various path-related bits and pieces.
# These are primarily for export; internally, they are hardcoded.
# Should be set before imports for resolving cyclic dependency.
curdir = '.'
pardir = '..'
extsep = '.'
sep = '/'
pathsep = ':'
defpath = '/bin:/usr/bin'
altsep = None
devnull = '/dev/null'

import os
import sys
import stat
import genericpath
from genericpath import *

__all__ = ["normcase","isabs","join","splitdrive","split","splitext",
           "basename","dirname","commonprefix","getsize","getmtime",
           "getatime","getctime","islink","exists","lexists","isdir","isfile",
           "ismount", "expanduser","expandvars","normpath","abspath",
           "samefile","sameopenfile","samestat",
           "curdir","pardir","sep","pathsep","defpath","altsep","extsep",
           "devnull","realpath","supports_unicode_filenames","relpath",
           "commonpath"]


def _get_sep(path):
    if isinstance(path, bytes):
        return b'/'
    else:
        return '/'

# Normalize the case of a pathname.  Trivial in Posix, string.lower on Mac.
# On MS-DOS this may also turn slashes into backslashes; however, other
# normalizations (such as optimizing '../' away) are not allowed
# (another function should be defined to do that).

def normcase(s):
    """Normalize case of pathname.  Has no effect under Posix"""
    return os.fspath(s)


# Return whether a path is absolute.
# Trivial in Posix, harder on the Mac or MS-DOS.

def isabs(s):
    """Test whether a path is absolute"""
    s = os.fspath(s)
    sep = _get_sep(s)
    return s.startswith(sep)


# Join pathnames.
# Ignore the previous parts if a part is absolute.
# Insert a '/' unless the first part is empty or already ends in '/'.

def join(a, *p):
    """Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator."""
    a = os.fspath(a)
    sep = _get_sep(a)
    path = a
    try:
        if not p:
            path[:0] + sep  #23780: Ensure compatible data type even if p is null.
        for b in map(os.fspath, p):
            if b.startswith(sep):
                path = b
            elif not path or path.endswith(sep):
                path += b
            else:
                path += sep + b
    except (TypeError, AttributeError, BytesWarning):
        genericpath._check_arg_types('join', a, *p)
        raise
    return path


# Split a path in head (everything up to the last '/') and tail (the
# rest).  If the path ends in '/', tail will be empty.  If there is no
# '/' in the path, head  will be empty.
# Trailing '/'es are stripped from head unless it is the root.

def split(p):
    """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
    everything after the final slash.  Either part may be empty."""
    p = os.fspath(p)
    sep = _get_sep(p)
    i = p.rfind(sep) + 1
    head, tail = p[:i], p[i:]
    if head and head != sep*len(head):
        head = head.rstrip(sep)
    return head, tail


# Split a path in root and extension.
# The extension is everything starting at the last dot in the last
# pathname component; the root is everything before that.
# It is always true that root + ext == p.

def splitext(p):
    p = os.fspath(p)
    if isinstance(p, bytes):
        sep = b'/'
        extsep = b'.'
    else:
        sep = '/'
        extsep = '.'
    return genericpath._splitext(p, sep, None, extsep)
splitext.__doc__ = genericpath._splitext.__doc__

# Split a pathname into a drive specification and the rest of the
# path.  Useful on DOS/Windows/NT; on Unix, the drive is always empty.

def splitdrive(p):
    """Split a pathname into drive and path. On Posix, drive is always
    empty."""
    p = os.fspath(p)
    return p[:0], p


# Return the tail (basename) part of a path, same as split(path)[1].

def basename(p):
    """Returns the final component of a pathname"""
    p = os.fspath(p)
    sep = _get_sep(p)
    i = p.rfind(sep) + 1
    return p[i:]


# Return the head (dirname) part of a path, same as split(path)[0].

def dirname(p):
    """Returns the directory component of a pathname"""
    p = os.fspath(p)
    sep = _get_sep(p)
    i = p.rfind(sep) + 1
    head = p[:i]
    if head and head != sep*len(head):
        head = head.rstrip(sep)
    return head


# Is a path a symbolic link?
# This will always return false on systems where os.lstat doesn't exist.

def islink(path):
    """Test whether a path is a symbolic link"""
    try:
        st = os.lstat(path)
    except (OSError, ValueError, AttributeError):
        return False
    return stat.S_ISLNK(st.st_mode)

# Being true for dangling symbolic links is also useful.

def lexists(path):
    """Test whether a path exists.  Returns True for broken symbolic links"""
    try:
        os.lstat(path)
    except (OSError, ValueError):
        return False
    return True


# Is a path a mount point?
# (Does this work for all UNIXes?  Is it even guaranteed to work by Posix?)

def ismount(path):
    """Test whether a path is a mount point"""
    try:
        s1 = os.lstat(path)
    except (OSError, ValueError):
        # It doesn't exist -- so not a mount point. :-)
        return False
    else:
        # A symlink can never be a mount point
        if stat.S_ISLNK(s1.st_mode):
            return False

    path = os.fspath(path)
    if isinstance(path, bytes):
        parent = join(path, b'..')
    else:
        parent = join(path, '..')
    parent = realpath(parent)
    try:
        s2 = os.lstat(parent)
    except (OSError, ValueError):
        return False

    dev1 = s1.st_dev
    dev2 = s2.st_dev
    if dev1 != dev2:
        return True     # path/.. on a different device as path
    ino1 = s1.st_ino
    ino2 = s2.st_ino
    if ino1 == ino2:
        return True     # path/.. is the same i-node as path
    return False


# Expand paths beginning with '~' or '~user'.
# '~' means $HOME; '~user' means that user's home directory.
# If the path doesn't begin with '~', or if the user or $HOME is unknown,
# the path is returned unchanged (leaving error reporting to whatever
# function is called with the expanded path as argument).
# See also module 'glob' for expansion of *, ? and [...] in pathnames.
# (A function should also be defined to do full *sh-style environment
# variable expansion.)

def expanduser(path):
    """Expand ~ and ~user constructions.  If user or $HOME is unknown,
    do nothing."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        tilde = b'~'
    else:
        tilde = '~'
    if not path.startswith(tilde):
        return path
    sep = _get_sep(path)
    i = path.find(sep, 1)
    if i < 0:
        i = len(path)
    if i == 1:
        if 'HOME' not in os.environ:
            import pwd
            try:
                userhome = pwd.getpwuid(os.getuid()).pw_dir
            except KeyError:
                # bpo-10496: if the current user identifier doesn't exist in the
                # password database, return the path unchanged
                return path
        else:
            userhome = os.environ['HOME']
    else:
        import pwd
        name = path[1:i]
        if isinstance(name, bytes):
            name = str(name, 'ASCII')
        try:
            pwent = pwd.getpwnam(name)
        except KeyError:
            # bpo-10496: if the user name from the path doesn't exist in the
            # password database, return the path unchanged
            return path
        userhome = pwent.pw_dir
    # if no user home, return the path unchanged on VxWorks
    if userhome is None and sys.platform == "vxworks":
        return path
    if isinstance(path, bytes):
        userhome = os.fsencode(userhome)
        root = b'/'
    else:
        root = '/'
    userhome = userhome.rstrip(root)
    return (userhome + path[i:]) or root


# Expand paths containing shell variable substitutions.
# This expands the forms $variable and ${variable} only.
# Non-existent variables are left unchanged.

_varpattern = r'\$(\w+|\{[^}]*\}?)'
_varsub = None
_varsubb = None

def expandvars(path):
    """Expand shell variables of form $var and ${var}.  Unknown variables
    are left unchanged."""
    path = os.fspath(path)
    global _varsub, _varsubb
    if isinstance(path, bytes):
        if b'$' not in path:
            return path
        if not _varsubb:
            import re
            _varsubb = re.compile(_varpattern.encode(), re.ASCII).sub
        sub = _varsubb
        start = b'{'
        end = b'}'
        environ = getattr(os, 'environb', None)
    else:
        if '$' not in path:
            return path
        if not _varsub:
            import re
            _varsub = re.compile(_varpattern, re.ASCII).sub
        sub = _varsub
        start = '{'
        end = '}'
        environ = os.environ

    def repl(m):
        name = m[1]
        if name.startswith(start):
            if not name.endswith(end):
                return m[0]
            name = name[1:-1]
        try:
            if environ is None:
                value = os.fsencode(os.environ[os.fsdecode(name)])
            else:
                value = environ[name]
        except KeyError:
            return m[0]
        else:
            return value

    return sub(repl, path)


# Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A/B.
# It should be understood that this may change the meaning of the path
# if it contains symbolic links!

def normpath(path):
    """Normalize path, eliminating double slashes, etc."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        sep = b'/'
        empty = b''
        dot = b'.'
        dotdot = b'..'
    else:
        sep = '/'
        empty = ''
        dot = '.'
        dotdot = '..'
    if path == empty:
        return dot
    initial_slashes = path.startswith(sep)
    # POSIX allows one or two initial slashes, but treats three or more
    # as single slash.
    # (see https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13)
    if (initial_slashes and
        path.startswith(sep*2) and not path.startswith(sep*3)):
        initial_slashes = 2
    comps = path.split(sep)
    new_comps = []
    for comp in comps:
        if comp in (empty, dot):
            continue
        if (comp != dotdot or (not initial_slashes and not new_comps) or
             (new_comps and new_comps[-1] == dotdot)):
            new_comps.append(comp)
        elif new_comps:
            new_comps.pop()
    comps = new_comps
    path = sep.join(comps)
    if initial_slashes:
        path = sep*initial_slashes + path
    return path or dot


def abspath(path):
    """Return an absolute path."""
    path = os.fspath(path)
    if not isabs(path):
        if isinstance(path, bytes):
            cwd = os.getcwdb()
        else:
            cwd = os.getcwd()
        path = join(cwd, path)
    return normpath(path)


# Return a canonical path (i.e. the absolute location of a file on the
# filesystem).

def realpath(filename, *, strict=False):
    """Return the canonical path of the specified filename, eliminating any
symbolic links encountered in the path."""
    filename = os.fspath(filename)
    path, ok = _joinrealpath(filename[:0], filename, strict, {})
    return abspath(path)

# Join two paths, normalizing and eliminating any symbolic links
# encountered in the second path.
def _joinrealpath(path, rest, strict, seen):
    if isinstance(path, bytes):
        sep = b'/'
        curdir = b'.'
        pardir = b'..'
    else:
        sep = '/'
        curdir = '.'
        pardir = '..'

    if isabs(rest):
        rest = rest[1:]
        path = sep

    while rest:
        name, _, rest = rest.partition(sep)
        if not name or name == curdir:
            # current dir
            continue
        if name == pardir:
            # parent dir
            if path:
                path, name = split(path)
                if name == pardir:
                    path = join(path, pardir, pardir)
            else:
                path = pardir
            continue
        newpath = join(path, name)
        try:
            st = os.lstat(newpath)
        except OSError:
            if strict:
                raise
            is_link = False
        else:
            is_link = stat.S_ISLNK(st.st_mode)
        if not is_link:
            path = newpath
            continue
        # Resolve the symbolic link
        if newpath in seen:
            # Already seen this path
            path = seen[newpath]
            if path is not None:
                # use cached value
                continue
            # The symlink is not resolved, so we must have a symlink loop.
            if strict:
                # Raise OSError(errno.ELOOP)
                os.stat(newpath)
            else:
                # Return already resolved part + rest of the path unchanged.
                return join(newpath, rest), False
        seen[newpath] = None # not resolved symlink
        path, ok = _joinrealpath(path, os.readlink(newpath), strict, seen)
        if not ok:
            return join(path, rest), False
        seen[newpath] = path # resolved symlink

    return path, True


supports_unicode_filenames = (sys.platform == 'darwin')

def relpath(path, start=None):
    """Return a relative version of a path"""

    if not path:
        raise ValueError("no path specified")

    path = os.fspath(path)
    if isinstance(path, bytes):
        curdir = b'.'
        sep = b'/'
        pardir = b'..'
    else:
        curdir = '.'
        sep = '/'
        pardir = '..'

    if start is None:
        start = curdir
    else:
        start = os.fspath(start)

    try:
        start_list = [x for x in abspath(start).split(sep) if x]
        path_list = [x for x in abspath(path).split(sep) if x]
        # Work out how much of the filepath is shared by start and path.
        i = len(commonprefix([start_list, path_list]))

        rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return curdir
        return join(*rel_list)
    except (TypeError, AttributeError, BytesWarning, DeprecationWarning):
        genericpath._check_arg_types('relpath', path, start)
        raise


# Return the longest common sub-path of the sequence of paths given as input.
# The paths are not normalized before comparing them (this is the
# responsibility of the caller). Any trailing separator is stripped from the
# returned path.

def commonpath(paths):
    """Given a sequence of path names, returns the longest common sub-path."""

    if not paths:
        raise ValueError('commonpath() arg is an empty sequence')

    paths = tuple(map(os.fspath, paths))
    if isinstance(paths[0], bytes):
        sep = b'/'
        curdir = b'.'
    else:
        sep = '/'
        curdir = '.'

    try:
        split_paths = [path.split(sep) for path in paths]

        try:
            isabs, = set(p[:1] == sep for p in paths)
        except ValueError:
            raise ValueError("Can't mix absolute and relative paths") from None

        split_paths = [[c for c in s if c and c != curdir] for s in split_paths]
        s1 = min(split_paths)
        s2 = max(split_paths)
        common = s1
        for i, c in enumerate(s1):
            if c != s2[i]:
                common = s1[:i]
                break

        prefix = sep if isabs else sep[:0]
        return prefix + sep.join(common)
    except (TypeError, AttributeError):
        genericpath._check_arg_types('commonpath', *paths)
        raise
<|endoftext|>"""Selectors module.

This module allows high-level and efficient I/O multiplexing, built upon the
`select` module primitives.
"""


from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys


# generic events, that must be mapped to implementation-specific ones
EVENT_READ = (1 << 0)
EVENT_WRITE = (1 << 1)


def _fileobj_to_fd(fileobj):
    """Return a file descriptor from a file object.

    Parameters:
    fileobj -- file object or file descriptor

    Returns:
    corresponding file descriptor

    Raises:
    ValueError if the object is invalid
    """
    if isinstance(fileobj, int):
        fd = fileobj
    else:
        try:
            fd = int(fileobj.fileno())
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Invalid file object: "
                             "{!r}".format(fileobj)) from None
    if fd < 0:
        raise ValueError("Invalid file descriptor: {}".format(fd))
    return fd


SelectorKey = namedtuple('SelectorKey', ['fileobj', 'fd', 'events', 'data'])

SelectorKey.__doc__ = """SelectorKey(fileobj, fd, events, data)

    Object used to associate a file object to its backing
    file descriptor, selected event mask, and attached data.
"""
if sys.version_info >= (3, 5):
    SelectorKey.fileobj.__doc__ = 'File object registered.'
    SelectorKey.fd.__doc__ = 'Underlying file descriptor.'
    SelectorKey.events.__doc__ = 'Events that must be waited for on this file object.'
    SelectorKey.data.__doc__ = ('''Optional opaque data associated to this file object.
    For example, this could be used to store a per-client session ID.''')


class _SelectorMapping(Mapping):
    """Mapping of file objects to selector keys."""

    def __init__(self, selector):
        self._selector = selector

    def __len__(self):
        return len(self._selector._fd_to_key)

    def __getitem__(self, fileobj):
        try:
            fd = self._selector._fileobj_lookup(fileobj)
            return self._selector._fd_to_key[fd]
        except KeyError:
            raise KeyError("{!r} is not registered".format(fileobj)) from None

    def __iter__(self):
        return iter(self._selector._fd_to_key)


class BaseSelector(metaclass=ABCMeta):
    """Selector abstract base class.

    A selector supports registering file objects to be monitored for specific
    I/O events.

    A file object is a file descriptor or any object with a `fileno()` method.
    An arbitrary object can be attached to the file object, which can be used
    for example to store context information, a callback, etc.

    A selector can use various implementations (select(), poll(), epoll()...)
    depending on the platform. The default `Selector` class uses the most
    efficient implementation on the current platform.
    """

    @abstractmethod
    def register(self, fileobj, events, data=None):
        """Register a file object.

        Parameters:
        fileobj -- file object or file descriptor
        events  -- events to monitor (bitwise mask of EVENT_READ|EVENT_WRITE)
        data    -- attached data

        Returns:
        SelectorKey instance

        Raises:
        ValueError if events is invalid
        KeyError if fileobj is already registered
        OSError if fileobj is closed or otherwise is unacceptable to
                the underlying system call (if a system call is made)

        Note:
        OSError may or may not be raised
        """
        raise NotImplementedError

    @abstractmethod
    def unregister(self, fileobj):
        """Unregister a file object.

        Parameters:
        fileobj -- file object or file descriptor

        Returns:
        SelectorKey instance

        Raises:
        KeyError if fileobj is not registered

        Note:
        If fileobj is registered but has since been closed this does
        *not* raise OSError (even if the wrapped syscall does)
        """
        raise NotImplementedError

    def modify(self, fileobj, events, data=None):
        """Change a registered file object monitored events or attached data.

        Parameters:
        fileobj -- file object or file descriptor
        events  -- events to monitor (bitwise mask of EVENT_READ|EVENT_WRITE)
        data    -- attached data

        Returns:
        SelectorKey instance

        Raises:
        Anything that unregister() or register() raises
        """
        self.unregister(fileobj)
        return self.register(fileobj, events, data)

    @abstractmethod
    def select(self, timeout=None):
        """Perform the actual selection, until some monitored file objects are
        ready or a timeout expires.

        Parameters:
        timeout -- if timeout > 0, this specifies the maximum wait time, in
                   seconds
                   if timeout <= 0, the select() call won't block, and will
                   report the currently ready file objects
                   if timeout is None, select() will block until a monitored
                   file object becomes ready

        Returns:
        list of (key, events) for ready file objects
        `events` is a bitwise mask of EVENT_READ|EVENT_WRITE
        """
        raise NotImplementedError

    def close(self):
        """Close the selector.

        This must be called to make sure that any underlying resource is freed.
        """
        pass

    def get_key(self, fileobj):
        """Return the key associated to a registered file object.

        Returns:
        SelectorKey for this file object
        """
        mapping = self.get_map()
        if mapping is None:
            raise RuntimeError('Selector is closed')
        try:
            return mapping[fileobj]
        except KeyError:
            raise KeyError("{!r} is not registered".format(fileobj)) from None

    @abstractmethod
    def get_map(self):
        """Return a mapping of file objects to selector keys."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _BaseSelectorImpl(BaseSelector):
    """Base selector implementation."""

    def __init__(self):
        # this maps file descriptors to keys
        self._fd_to_key = {}
        # read-only mapping returned by get_map()
        self._map = _SelectorMapping(self)

    def _fileobj_lookup(self, fileobj):
        """Return a file descriptor from a file object.

        This wraps _fileobj_to_fd() to do an exhaustive search in case
        the object is invalid but we still have it in our map.  This
        is used by unregister() so we can unregister an object that
        was previously registered even if it is closed.  It is also
        used by _SelectorMapping.
        """
        try:
            return _fileobj_to_fd(fileobj)
        except ValueError:
            # Do an exhaustive search.
            for key in self._fd_to_key.values():
                if key.fileobj is fileobj:
                    return key.fd
            # Raise ValueError after all.
            raise

    def register(self, fileobj, events, data=None):
        if (not events) or (events & ~(EVENT_READ | EVENT_WRITE)):
            raise ValueError("Invalid events: {!r}".format(events))

        key = SelectorKey(fileobj, self._fileobj_lookup(fileobj), events, data)

        if key.fd in self._fd_to_key:
            raise KeyError("{!r} (FD {}) is already registered"
                           .format(fileobj, key.fd))

        self._fd_to_key[key.fd] = key
        return key

    def unregister(self, fileobj):
        try:
            key = self._fd_to_key.pop(self._fileobj_lookup(fileobj))
        except KeyError:
            raise KeyError("{!r} is not registered".format(fileobj)) from None
        return key

    def modify(self, fileobj, events, data=None):
        try:
            key = self._fd_to_key[self._fileobj_lookup(fileobj)]
        except KeyError:
            raise KeyError("{!r} is not registered".format(fileobj)) from None
        if events != key.events:
            self.unregister(fileobj)
            key = self.register(fileobj, events, data)
        elif data != key.data:
            # Use a shortcut to update the data.
            key = key._replace(data=data)
            self._fd_to_key[key.fd] = key
        return key

    def close(self):
        self._fd_to_key.clear()
        self._map = None

    def get_map(self):
        return self._map

    def _key_from_fd(self, fd):
        """Return the key associated to a given file descriptor.

        Parameters:
        fd -- file descriptor

        Returns:
        corresponding key, or None if not found
        """
        try:
            return self._fd_to_key[fd]
        except KeyError:
            return None


class SelectSelector(_BaseSelectorImpl):
    """Select-based selector."""

    def __init__(self):
        super().__init__()
        self._readers = set()
        self._writers = set()

    def register(self, fileobj, events, data=None):
        key = super().register(fileobj, events, data)
        if events & EVENT_READ:
            self._readers.add(key.fd)
        if events & EVENT_WRITE:
            self._writers.add(key.fd)
        return key

    def unregister(self, fileobj):
        key = super().unregister(fileobj)
        self._readers.discard(key.fd)
        self._writers.discard(key.fd)
        return key

    if sys.platform == 'win32':
        def _select(self, r, w, _, timeout=None):
            r, w, x = select.select(r, w, w, timeout)
            return r, w + x, []
    else:
        _select = select.select

    def select(self, timeout=None):
        timeout = None if timeout is None else max(timeout, 0)
        ready = []
        try:
            r, w, _ = self._select(self._readers, self._writers, [], timeout)
        except InterruptedError:
            return ready
        r = set(r)
        w = set(w)
        for fd in r | w:
            events = 0
            if fd in r:
                events |= EVENT_READ
            if fd in w:
                events |= EVENT_WRITE

            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready


class _PollLikeSelector(_BaseSelectorImpl):
    """Base class shared between poll, epoll and devpoll selectors."""
    _selector_cls = None
    _EVENT_READ = None
    _EVENT_WRITE = None

    def __init__(self):
        super().__init__()
        self._selector = self._selector_cls()

    def register(self, fileobj, events, data=None):
        key = super().register(fileobj, events, data)
        poller_events = 0
        if events & EVENT_READ:
            poller_events |= self._EVENT_READ
        if events & EVENT_WRITE:
            poller_events |= self._EVENT_WRITE
        try:
            self._selector.register(key.fd, poller_events)
        except:
            super().unregister(fileobj)
            raise
        return key

    def unregister(self, fileobj):
        key = super().unregister(fileobj)
        try:
            self._selector.unregister(key.fd)
        except OSError:
            # This can happen if the FD was closed since it
            # was registered.
            pass
        return key

    def modify(self, fileobj, events, data=None):
        try:
            key = self._fd_to_key[self._fileobj_lookup(fileobj)]
        except KeyError:
            raise KeyError(f"{fileobj!r} is not registered") from None

        changed = False
        if events != key.events:
            selector_events = 0
            if events & EVENT_READ:
                selector_events |= self._EVENT_READ
            if events & EVENT_WRITE:
                selector_events |= self._EVENT_WRITE
            try:
                self._selector.modify(key.fd, selector_events)
            except:
                super().unregister(fileobj)
                raise
            changed = True
        if data != key.data:
            changed = True

        if changed:
            key = key._replace(events=events, data=data)
            self._fd_to_key[key.fd] = key
        return key

    def select(self, timeout=None):
        # This is shared between poll() and epoll().
        # epoll() has a different signature and handling of timeout parameter.
        if timeout is None:
            timeout = None
        elif timeout <= 0:
            timeout = 0
        else:
            # poll() has a resolution of 1 millisecond, round away from
            # zero to wait *at least* timeout seconds.
            timeout = math.ceil(timeout * 1e3)
        ready = []
        try:
            fd_event_list = self._selector.poll(timeout)
        except InterruptedError:
            return ready
        for fd, event in fd_event_list:
            events = 0
            if event & ~self._EVENT_READ:
                events |= EVENT_WRITE
            if event & ~self._EVENT_WRITE:
                events |= EVENT_READ

            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready


if hasattr(select, 'poll'):

    class PollSelector(_PollLikeSelector):
        """Poll-based selector."""
        _selector_cls = select.poll
        _EVENT_READ = select.POLLIN
        _EVENT_WRITE = select.POLLOUT


if hasattr(select, 'epoll'):

    class EpollSelector(_PollLikeSelector):
        """Epoll-based selector."""
        _selector_cls = select.epoll
        _EVENT_READ = select.EPOLLIN
        _EVENT_WRITE = select.EPOLLOUT

        def fileno(self):
            return self._selector.fileno()

        def select(self, timeout=None):
            if timeout is None:
                timeout = -1
            elif timeout <= 0:
                timeout = 0
            else:
                # epoll_wait() has a resolution of 1 millisecond, round away
                # from zero to wait *at least* timeout seconds.
                timeout = math.ceil(timeout * 1e3) * 1e-3

            # epoll_wait() expects `maxevents` to be greater than zero;
            # we want to make sure that `select()` can be called when no
            # FD is registered.
            max_ev = max(len(self._fd_to_key), 1)

            ready = []
            try:
                fd_event_list = self._selector.poll(timeout, max_ev)
            except InterruptedError:
                return ready
            for fd, event in fd_event_list:
                events = 0
                if event & ~select.EPOLLIN:
                    events |= EVENT_WRITE
                if event & ~select.EPOLLOUT:
