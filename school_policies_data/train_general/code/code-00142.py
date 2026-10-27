            msg = "Passed object must be a class or attrs instance."
            raise TypeError(msg)

        return fields(type_)

    attrs = getattr(cls, "__attrs_attrs__", None)

    if attrs is None:
        if generic_base is not None:
            attrs = getattr(generic_base, "__attrs_attrs__", None)
            if attrs is not None:
                # Even though this is global state, stick it on here to speed
                # it up. We rely on `cls` being cached for this to be
                # efficient.
                cls.__attrs_attrs__ = attrs
                return attrs
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)

    return attrs


def fields_dict(cls):
    """
    Return an ordered dictionary of *attrs* attributes for a class, whose keys
    are the attribute names.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    Returns:
        dict[str, attrs.Attribute]: Dict of attribute name to definition

    .. versionadded:: 18.1.0
    """
    if not isinstance(cls, type):
        msg = "Passed object must be a class."
        raise TypeError(msg)
    attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs is None:
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)
    return {a.name: a for a in attrs}


def validate(inst):
    """
    Validate all attributes on *inst* that have a validator.

    Leaves all exceptions through.

    Args:
        inst: Instance of a class with *attrs* attributes.
    """
    if _config._run_validators is False:
        return

    for a in fields(inst.__class__):
        v = a.validator
        if v is not None:
            v(inst, a, getattr(inst, a.name))


def _is_slot_attr(a_name, base_attr_map):
    """
    Check if the attribute name comes from a slot class.
    """
    cls = base_attr_map.get(a_name)
    return cls and "__slots__" in cls.__dict__


def _make_init_script(
    cls,
    attrs,
    pre_init,
    pre_init_has_args,
    post_init,
    frozen,
    slots,
    cache_hash,
    base_attr_map,
    is_exc,
    cls_on_setattr,
    attrs_init,
) -> tuple[str, dict, dict]:
    has_cls_on_setattr = (
        cls_on_setattr is not None and cls_on_setattr is not setters.NO_OP
    )

    if frozen and has_cls_on_setattr:
        msg = "Frozen classes can't use on_setattr."
        raise ValueError(msg)

    needs_cached_setattr = cache_hash or frozen
    filtered_attrs = []
    attr_dict = {}
    for a in attrs:
        if not a.init and a.default is NOTHING:
            continue

        filtered_attrs.append(a)
        attr_dict[a.name] = a

        if a.on_setattr is not None:
            if frozen is True and a.on_setattr is not setters.NO_OP:
                msg = "Frozen classes can't use on_setattr."
                raise ValueError(msg)

            needs_cached_setattr = True
        elif has_cls_on_setattr and a.on_setattr is not setters.NO_OP:
            needs_cached_setattr = True

    script, globs, annotations = _attrs_to_init_script(
        filtered_attrs,
        frozen,
        slots,
        pre_init,
        pre_init_has_args,
        post_init,
        cache_hash,
        base_attr_map,
        is_exc,
        needs_cached_setattr,
        has_cls_on_setattr,
        "__attrs_init__" if attrs_init else "__init__",
    )
    if cls.__module__ in sys.modules:
        # This makes typing.get_type_hints(CLS.__init__) resolve string types.
        globs.update(sys.modules[cls.__module__].__dict__)

    globs.update({"NOTHING": NOTHING, "attr_dict": attr_dict})

    if needs_cached_setattr:
        # Save the lookup overhead in __init__ if we need to circumvent
        # setattr hooks.
        globs["_cached_setattr_get"] = _OBJ_SETATTR.__get__

    return script, globs, annotations


def _setattr(attr_name: str, value_var: str, has_on_setattr: bool) -> str:
    """
    Use the cached object.setattr to set *attr_name* to *value_var*.
    """
    return f"_setattr('{attr_name}', {value_var})"


def _setattr_with_converter(
    attr_name: str, value_var: str, has_on_setattr: bool, converter: Converter
) -> str:
    """
    Use the cached object.setattr to set *attr_name* to *value_var*, but run
    its converter first.
    """
    return f"_setattr('{attr_name}', {converter._fmt_converter_call(attr_name, value_var)})"


def _assign(attr_name: str, value: str, has_on_setattr: bool) -> str:
    """
    Unless *attr_name* has an on_setattr hook, use normal assignment. Otherwise
    relegate to _setattr.
    """
    if has_on_setattr:
        return _setattr(attr_name, value, True)

    return f"self.{attr_name} = {value}"


def _assign_with_converter(
    attr_name: str, value_var: str, has_on_setattr: bool, converter: Converter
) -> str:
    """
    Unless *attr_name* has an on_setattr hook, use normal assignment after
    conversion. Otherwise relegate to _setattr_with_converter.
    """
    if has_on_setattr:
        return _setattr_with_converter(attr_name, value_var, True, converter)

    return f"self.{attr_name} = {converter._fmt_converter_call(attr_name, value_var)}"


def _determine_setters(
    frozen: bool, slots: bool, base_attr_map: dict[str, type]
):
    """
    Determine the correct setter functions based on whether a class is frozen
    and/or slotted.
    """
    if frozen is True:
        if slots is True:
            return (), _setattr, _setattr_with_converter

        # Dict frozen classes assign directly to __dict__.
        # But only if the attribute doesn't come from an ancestor slot
        # class.
        # Note _inst_dict will be used again below if cache_hash is True

        def fmt_setter(
            attr_name: str, value_var: str, has_on_setattr: bool
        ) -> str:
            if _is_slot_attr(attr_name, base_attr_map):
                return _setattr(attr_name, value_var, has_on_setattr)

            return f"_inst_dict['{attr_name}'] = {value_var}"

        def fmt_setter_with_converter(
            attr_name: str,
            value_var: str,
            has_on_setattr: bool,
            converter: Converter,
        ) -> str:
            if has_on_setattr or _is_slot_attr(attr_name, base_attr_map):
                return _setattr_with_converter(
                    attr_name, value_var, has_on_setattr, converter
                )

            return f"_inst_dict['{attr_name}'] = {converter._fmt_converter_call(attr_name, value_var)}"

        return (
            ("_inst_dict = self.__dict__",),
            fmt_setter,
            fmt_setter_with_converter,
        )

    # Not frozen -- we can just assign directly.
    return (), _assign, _assign_with_converter


def _attrs_to_init_script(
    attrs: list[Attribute],
    is_frozen: bool,
    is_slotted: bool,
    call_pre_init: bool,
    pre_init_has_args: bool,
    call_post_init: bool,
    does_cache_hash: bool,
    base_attr_map: dict[str, type],
    is_exc: bool,
    needs_cached_setattr: bool,
    has_cls_on_setattr: bool,
    method_name: str,
) -> tuple[str, dict, dict]:
    """
    Return a script of an initializer for *attrs*, a dict of globals, and
    annotations for the initializer.

    The globals are required by the generated script.
    """
    lines = ["self.__attrs_pre_init__()"] if call_pre_init else []

    if needs_cached_setattr:
        lines.append(
            # Circumvent the __setattr__ descriptor to save one lookup per
            # assignment. Note _setattr will be used again below if
            # does_cache_hash is True.
            "_setattr = _cached_setattr_get(self)"
        )

    extra_lines, fmt_setter, fmt_setter_with_converter = _determine_setters(
        is_frozen, is_slotted, base_attr_map
    )
    lines.extend(extra_lines)

    args = []  # Parameters in the definition of __init__
    pre_init_args = []  # Parameters in the call to __attrs_pre_init__
    kw_only_args = []  # Used for both 'args' and 'pre_init_args' above
    attrs_to_validate = []

    # This is a dictionary of names to validator and converter callables.
    # Injecting this into __init__ globals lets us avoid lookups.
    names_for_globals = {}
    annotations = {"return": None}

    for a in attrs:
        if a.validator:
            attrs_to_validate.append(a)

        attr_name = a.name
        has_on_setattr = a.on_setattr is not None or (
            a.on_setattr is not setters.NO_OP and has_cls_on_setattr
        )
        # a.alias is set to maybe-mangled attr_name in _ClassBuilder if not
        # explicitly provided
        arg_name = a.alias

        has_factory = isinstance(a.default, Factory)
        maybe_self = "self" if has_factory and a.default.takes_self else ""

        if a.converter is not None and not isinstance(a.converter, Converter):
            converter = Converter(a.converter)
        else:
            converter = a.converter

        if a.init is False:
            if has_factory:
                init_factory_name = _INIT_FACTORY_PAT % (a.name,)
                if converter is not None:
                    lines.append(
                        fmt_setter_with_converter(
                            attr_name,
                            init_factory_name + f"({maybe_self})",
                            has_on_setattr,
                            converter,
                        )
                    )
                    names_for_globals[converter._get_global_name(a.name)] = (
                        converter.converter
                    )
                else:
                    lines.append(
                        fmt_setter(
                            attr_name,
                            init_factory_name + f"({maybe_self})",
                            has_on_setattr,
                        )
                    )
                names_for_globals[init_factory_name] = a.default.factory
            elif converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name,
                        f"attr_dict['{attr_name}'].default",
                        has_on_setattr,
                        converter,
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(
                    fmt_setter(
                        attr_name,
                        f"attr_dict['{attr_name}'].default",
                        has_on_setattr,
                    )
                )
        elif a.default is not NOTHING and not has_factory:
            arg = f"{arg_name}=attr_dict['{attr_name}'].default"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
                pre_init_args.append(arg_name)

            if converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))

        elif has_factory:
            arg = f"{arg_name}=NOTHING"
            if a.kw_only:
                kw_only_args.append(arg)
            else:
                args.append(arg)
                pre_init_args.append(arg_name)
            lines.append(f"if {arg_name} is not NOTHING:")

            init_factory_name = _INIT_FACTORY_PAT % (a.name,)
            if converter is not None:
                lines.append(
                    "    "
                    + fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                lines.append("else:")
                lines.append(
                    "    "
                    + fmt_setter_with_converter(
                        attr_name,
                        init_factory_name + "(" + maybe_self + ")",
                        has_on_setattr,
                        converter,
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(
                    "    " + fmt_setter(attr_name, arg_name, has_on_setattr)
                )
                lines.append("else:")
                lines.append(
                    "    "
                    + fmt_setter(
                        attr_name,
                        init_factory_name + "(" + maybe_self + ")",
                        has_on_setattr,
                    )
                )
            names_for_globals[init_factory_name] = a.default.factory
        else:
            if a.kw_only:
                kw_only_args.append(arg_name)
            else:
                args.append(arg_name)
                pre_init_args.append(arg_name)

            if converter is not None:
                lines.append(
                    fmt_setter_with_converter(
                        attr_name, arg_name, has_on_setattr, converter
                    )
                )
                names_for_globals[converter._get_global_name(a.name)] = (
                    converter.converter
                )
            else:
                lines.append(fmt_setter(attr_name, arg_name, has_on_setattr))

        if a.init is True:
            if a.type is not None and converter is None:
                annotations[arg_name] = a.type
            elif converter is not None and converter._first_param_type:
                # Use the type from the converter if present.
                annotations[arg_name] = converter._first_param_type

    if attrs_to_validate:  # we can skip this if there are no validators.
        names_for_globals["_config"] = _config
        lines.append("if _config._run_validators is True:")
        for a in attrs_to_validate:
            val_name = "__attr_validator_" + a.name
            attr_name = "__attr_" + a.name
            lines.append(f"    {val_name}(self, {attr_name}, self.{a.name})")
            names_for_globals[val_name] = a.validator
            names_for_globals[attr_name] = a

    if call_post_init:
        lines.append("self.__attrs_post_init__()")

    # Because this is set only after __attrs_post_init__ is called, a crash
    # will result if post-init tries to access the hash code.  This seemed
    # preferable to setting this beforehand, in which case alteration to field
    # values during post-init combined with post-init accessing the hash code
    # would result in silent bugs.
    if does_cache_hash:
        if is_frozen:
            if is_slotted:
                init_hash_cache = f"_setattr('{_HASH_CACHE_FIELD}', None)"
            else:
                init_hash_cache = f"_inst_dict['{_HASH_CACHE_FIELD}'] = None"
        else:
            init_hash_cache = f"self.{_HASH_CACHE_FIELD} = None"
        lines.append(init_hash_cache)

    # For exceptions we rely on BaseException.__init__ for proper
    # initialization.
    if is_exc:
        vals = ",".join(f"self.{a.name}" for a in attrs if a.init)

        lines.append(f"BaseException.__init__(self, {vals})")

    args = ", ".join(args)
    pre_init_args = ", ".join(pre_init_args)
    if kw_only_args:
        # leading comma & kw_only args
        args += f"{', ' if args else ''}*, {', '.join(kw_only_args)}"
        pre_init_kw_only_args = ", ".join(
            [
                f"{kw_arg_name}={kw_arg_name}"
                # We need to remove the defaults from the kw_only_args.
                for kw_arg_name in (kwa.split("=")[0] for kwa in kw_only_args)
            ]
        )
        pre_init_args += ", " if pre_init_args else ""
        pre_init_args += pre_init_kw_only_args

    if call_pre_init and pre_init_has_args:
        # If pre init method has arguments, pass the values given to __init__.
        lines[0] = f"self.__attrs_pre_init__({pre_init_args})"

    # Python <3.12 doesn't allow backslashes in f-strings.
    NL = "\n    "
    return (
        f"""def {method_name}(self, {args}):
    {NL.join(lines) if lines else "pass"}
""",
        names_for_globals,
        annotations,
    )


def _default_init_alias_for(name: str) -> str:
    """
    The default __init__ parameter name for a field.

    This performs private-name adjustment via leading-unscore stripping,
    and is the default value of Attribute.alias if not provided.
    """

    return name.lstrip("_")


class Attribute:
    """
    *Read-only* representation of an attribute.

    .. warning::

       You should never instantiate this class yourself.

    The class has *all* arguments of `attr.ib` (except for ``factory`` which is
    only syntactic sugar for ``default=Factory(...)`` plus the following:

    - ``name`` (`str`): The name of the attribute.
    - ``alias`` (`str`): The __init__ parameter name of the attribute, after
      any explicit overrides and default private-attribute-name handling.
    - ``alias_is_default`` (`bool`): Whether the ``alias`` was automatically
      generated (``True``) or explicitly provided by the user (``False``).
    - ``inherited`` (`bool`): Whether or not that attribute has been inherited
      from a base class.
    - ``eq_key`` and ``order_key`` (`typing.Callable` or `None`): The
      callables that are used for comparing and ordering objects by this
      attribute, respectively. These are set by passing a callable to
      `attr.ib`'s ``eq``, ``order``, or ``cmp`` arguments. See also
      :ref:`comparison customization <custom-comparison>`.

    Instances of this class are frequently used for introspection purposes
    like:

    - `fields` returns a tuple of them.
    - Validators get them passed as the first argument.
    - The :ref:`field transformer <transform-fields>` hook receives a list of
      them.
    - The ``alias`` property exposes the __init__ parameter name of the field,
      with any overrides and default private-attribute handling applied.


    .. versionadded:: 20.1.0 *inherited*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.2.0 *inherited* is not taken into account for
        equality checks and hashing anymore.
    .. versionadded:: 21.1.0 *eq_key* and *order_key*
    .. versionadded:: 22.2.0 *alias*
    .. versionadded:: 26.1.0 *alias_is_default*

    For the full version history of the fields, see `attr.ib`.
    """

    # These slots must NOT be reordered because we use them later for
    # instantiation.
    __slots__ = (  # noqa: RUF023
        "name",
        "default",
        "validator",
        "repr",
        "eq",
        "eq_key",
        "order",
        "order_key",
        "hash",
        "init",
        "metadata",
        "type",
        "converter",
        "kw_only",
        "inherited",
        "on_setattr",
        "alias",
        "alias_is_default",
    )

    def __init__(
        self,
        name,
        default,
        validator,
        repr,
        cmp,  # XXX: unused, remove along with other cmp code.
        hash,
        init,
        inherited,
        metadata=None,
        type=None,
        converter=None,
        kw_only=False,
        eq=None,
        eq_key=None,
        order=None,
        order_key=None,
        on_setattr=None,
        alias=None,
        alias_is_default=None,
    ):
        eq, eq_key, order, order_key = _determine_attrib_eq_order(
            cmp, eq_key or eq, order_key or order, True
        )

        # Cache this descriptor here to speed things up later.
        bound_setattr = _OBJ_SETATTR.__get__(self)

        # Despite the big red warning, people *do* instantiate `Attribute`
        # themselves.
        bound_setattr("name", name)
        bound_setattr("default", default)
        bound_setattr("validator", validator)
        bound_setattr("repr", repr)
        bound_setattr("eq", eq)
        bound_setattr("eq_key", eq_key)
        bound_setattr("order", order)
        bound_setattr("order_key", order_key)
        bound_setattr("hash", hash)
        bound_setattr("init", init)
        bound_setattr("converter", converter)
        bound_setattr(
            "metadata",
            (
                types.MappingProxyType(dict(metadata))  # Shallow copy
                if metadata
                else _EMPTY_METADATA_SINGLETON
            ),
        )
        bound_setattr("type", type)
        bound_setattr("kw_only", kw_only)
        bound_setattr("inherited", inherited)
        bound_setattr("on_setattr", on_setattr)
        bound_setattr("alias", alias)
        bound_setattr(
            "alias_is_default",
            alias is None if alias_is_default is None else alias_is_default,
        )

    def __setattr__(self, name, value):
        raise FrozenInstanceError

    @classmethod
    def from_counting_attr(
        cls, name: str, ca: _CountingAttr, kw_only: bool, type=None
    ):
        # The 'kw_only' argument is the class-level setting, and is used if the
        # attribute itself does not explicitly set 'kw_only'.
        # type holds the annotated value. deal with conflicts:
        if type is None:
            type = ca.type
        elif ca.type is not None:
            msg = f"Type annotation and type argument cannot both be present for '{name}'."
            raise ValueError(msg)
        return cls(
            name,
            ca._default,
            ca._validator,
            ca.repr,
            None,
            ca.hash,
            ca.init,
            False,
            ca.metadata,
            type,
            ca.converter,
            kw_only if ca.kw_only is None else ca.kw_only,
            ca.eq,
            ca.eq_key,
            ca.order,
            ca.order_key,
            ca.on_setattr,
            ca.alias,
            ca.alias is None,
        )

    # Don't use attrs.evolve since fields(Attribute) doesn't work
    def evolve(self, **changes):
        """
        Copy *self* and apply *changes*.

        This works similarly to `attrs.evolve` but that function does not work
        with :class:`attrs.Attribute`.

        It is mainly meant to be used for `transform-fields`.

        .. versionadded:: 20.3.0
        """
        new = copy.copy(self)

        new._setattrs(changes.items())

        if "alias" in changes and "alias_is_default" not in changes:
            # Explicit alias provided -- no longer the default.
            _OBJ_SETATTR.__get__(new)("alias_is_default", False)
        elif (
            "name" in changes
            and "alias" not in changes
            # Don't auto-generate alias if the user picked picked the old one.
            and self.alias_is_default
        ):
            # Name changed, alias was auto-generated -- update it.
            _OBJ_SETATTR.__get__(new)(
                "alias", _default_init_alias_for(new.name)
            )

        return new

    # Don't use _add_pickle since fields(Attribute) doesn't work
    def __getstate__(self):
        """
        Play nice with pickle.
        """
        return tuple(
            getattr(self, name) if name != "metadata" else dict(self.metadata)
            for name in self.__slots__
        )

    def __setstate__(self, state):
        """
        Play nice with pickle.
        """
        if len(state) < len(self.__slots__):
            # Pre-26.1.0 pickle without alias_is_default -- infer it
            # heuristically.
            state_dict = dict(zip(self.__slots__, state))
            alias_is_default = state_dict.get(
                "alias"
            ) is None or state_dict.get("alias") == _default_init_alias_for(
                state_dict["name"]
            )
            state = (*state, alias_is_default)

        self._setattrs(zip(self.__slots__, state))

    def _setattrs(self, name_values_pairs):
        bound_setattr = _OBJ_SETATTR.__get__(self)
        for name, value in name_values_pairs:
            if name != "metadata":
                bound_setattr(name, value)
            else:
                bound_setattr(
                    name,
                    (
                        types.MappingProxyType(dict(value))
                        if value
                        else _EMPTY_METADATA_SINGLETON
                    ),
                )


_a = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=(name != "alias_is_default"),
        cmp=None,
        eq=True,
        order=False,
        hash=(name != "metadata"),
        init=True,
        inherited=False,
        alias=_default_init_alias_for(name),
    )
    for name in Attribute.__slots__
]

Attribute = _add_hash(
    _add_eq(
        _add_repr(Attribute, attrs=_a),
        attrs=[a for a in _a if a.name != "inherited"],
    ),
    attrs=[a for a in _a if a.hash and a.name != "inherited"],
)


class _CountingAttr:
    """
    Intermediate representation of attributes that uses a counter to preserve
    the order in which the attributes have been defined.

    *Internal* data structure of the attrs library.  Running into is most
    likely the result of a bug like a forgotten `@attr.s` decorator.
    """

    __slots__ = (
        "_default",
        "_validator",
        "alias",
        "converter",
        "counter",
        "eq",
        "eq_key",
        "hash",
        "init",
        "kw_only",
        "metadata",
        "on_setattr",
        "order",
        "order_key",
        "repr",
        "type",
    )
    __attrs_attrs__ = (
        *tuple(
            Attribute(
                name=name,
                alias=_default_init_alias_for(name),
                default=NOTHING,
                validator=None,
                repr=True,
                cmp=None,
                hash=True,
                init=True,
                kw_only=False,
                eq=True,
                eq_key=None,
                order=False,
                order_key=None,
                inherited=False,
                on_setattr=None,
            )
            for name in (
                "counter",
                "_default",
                "repr",
                "eq",
                "order",
                "hash",
                "init",
                "on_setattr",
                "alias",
            )
        ),
        Attribute(
            name="metadata",
            alias="metadata",
            default=None,
            validator=None,
            repr=True,
            cmp=None,
            hash=False,
            init=True,
            kw_only=False,
            eq=True,
            eq_key=None,
            order=False,
            order_key=None,
            inherited=False,
            on_setattr=None,
        ),
    )
    cls_counter = 0

    def __init__(
        self,
        default,
        validator,
        repr,
        cmp,
        hash,
        init,
        converter,
        metadata,
        type,
        kw_only,
        eq,
        eq_key,
        order,
        order_key,
        on_setattr,
        alias,
    ):
        _CountingAttr.cls_counter += 1
        self.counter = _CountingAttr.cls_counter
        self._default = default
        self._validator = validator
        self.converter = converter
        self.repr = repr
        self.eq = eq
        self.eq_key = eq_key
        self.order = order
        self.order_key = order_key
        self.hash = hash
        self.init = init
        self.metadata = metadata
        self.type = type
        self.kw_only = kw_only
        self.on_setattr = on_setattr
        self.alias = alias

    def validator(self, meth):
        """
        Decorator that adds *meth* to the list of validators.

        Returns *meth* unchanged.

        .. versionadded:: 17.1.0
        """
        if self._validator is None:
            self._validator = meth
        else:
            self._validator = and_(self._validator, meth)
        return meth

    def default(self, meth):
        """
        Decorator that allows to set the default for an attribute.

        Returns *meth* unchanged.

        Raises:
            DefaultAlreadySetError: If default has been set before.

        .. versionadded:: 17.1.0
        """
        if self._default is not NOTHING:
            raise DefaultAlreadySetError

        self._default = Factory(meth, takes_self=True)

        return meth


_CountingAttr = _add_eq(_add_repr(_CountingAttr))


class ClassProps:
    """
    Effective class properties as derived from parameters to `attr.s()` or
    `define()` decorators.

    This is the same data structure that *attrs* uses internally to decide how
    to construct the final class.

    Warning:

        This feature is currently **experimental** and is not covered by our
        strict backwards-compatibility guarantees.


    Attributes:
        is_exception (bool):
            Whether the class is treated as an exception class.

        is_slotted (bool):
            Whether the class is `slotted <slotted classes>`.

        has_weakref_slot (bool):
            Whether the class has a slot for weak references.

        is_frozen (bool):
            Whether the class is frozen.

        kw_only (KeywordOnly):
            Whether / how the class enforces keyword-only arguments on the
            ``__init__`` method.

        collected_fields_by_mro (bool):
            Whether the class fields were collected by method resolution order.
            That is, correctly but unlike `dataclasses`.

        added_init (bool):
            Whether the class has an *attrs*-generated ``__init__`` method.

        added_repr (bool):
            Whether the class has an *attrs*-generated ``__repr__`` method.

        added_eq (bool):
            Whether the class has *attrs*-generated equality methods.

        added_ordering (bool):
            Whether the class has *attrs*-generated ordering methods.

        hashability (Hashability): How `hashable <hashing>` the class is.

        added_match_args (bool):
            Whether the class supports positional `match <match>` over its
            fields.

        added_str (bool):
            Whether the class has an *attrs*-generated ``__str__`` method.

        added_pickling (bool):
            Whether the class has *attrs*-generated ``__getstate__`` and
            ``__setstate__`` methods for `pickle`.

        on_setattr_hook (Callable[[Any, Attribute[Any], Any], Any] | None):
            The class's ``__setattr__`` hook.

        field_transformer (Callable[[Attribute[Any]], Attribute[Any]] | None):
            The class's `field transformers <transform-fields>`.

    .. versionadded:: 25.4.0
    """

    class Hashability(enum.Enum):
        """
        The hashability of a class.

        .. versionadded:: 25.4.0
        """

        HASHABLE = "hashable"
        """Write a ``__hash__``."""
        HASHABLE_CACHED = "hashable_cache"
        """Write a ``__hash__`` and cache the hash."""
        UNHASHABLE = "unhashable"
        """Set ``__hash__`` to ``None``."""
        LEAVE_ALONE = "leave_alone"
        """Don't touch ``__hash__``."""

    class KeywordOnly(enum.Enum):
        """
        How attributes should be treated regarding keyword-only parameters.

        .. versionadded:: 25.4.0
        """

        NO = "no"
        """Attributes are not keyword-only."""
        YES = "yes"
        """Attributes in current class without kw_only=False are keyword-only."""
        FORCE = "force"
        """All attributes are keyword-only."""

    __slots__ = (  # noqa: RUF023 -- order matters for __init__
        "is_exception",
        "is_slotted",
        "has_weakref_slot",
        "is_frozen",
        "kw_only",
        "collected_fields_by_mro",
        "added_init",
        "added_repr",
        "added_eq",
        "added_ordering",
        "hashability",
        "added_match_args",
        "added_str",
        "added_pickling",
        "on_setattr_hook",
        "field_transformer",
    )

    def __init__(
        self,
        is_exception,
        is_slotted,
        has_weakref_slot,
        is_frozen,
        kw_only,
        collected_fields_by_mro,
        added_init,
        added_repr,
        added_eq,
        added_ordering,
        hashability,
        added_match_args,
        added_str,
        added_pickling,
        on_setattr_hook,
        field_transformer,
    ):
        self.is_exception = is_exception
        self.is_slotted = is_slotted
        self.has_weakref_slot = has_weakref_slot
        self.is_frozen = is_frozen
        self.kw_only = kw_only
        self.collected_fields_by_mro = collected_fields_by_mro
        self.added_init = added_init
        self.added_repr = added_repr
        self.added_eq = added_eq
        self.added_ordering = added_ordering
        self.hashability = hashability
        self.added_match_args = added_match_args
        self.added_str = added_str
        self.added_pickling = added_pickling
        self.on_setattr_hook = on_setattr_hook
        self.field_transformer = field_transformer

    @property
    def is_hashable(self):
        return (
            self.hashability is ClassProps.Hashability.HASHABLE
            or self.hashability is ClassProps.Hashability.HASHABLE_CACHED
        )


_cas = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=True,
        init=True,
        inherited=False,
        alias=_default_init_alias_for(name),
    )
    for name in ClassProps.__slots__
]

ClassProps = _add_eq(_add_repr(ClassProps, attrs=_cas), attrs=_cas)


class Factory:
    """
    Stores a factory callable.

    If passed as the default value to `attrs.field`, the factory is used to
    generate a new value.

    Args:
        factory (typing.Callable):
            A callable that takes either none or exactly one mandatory
            positional argument depending on *takes_self*.

        takes_self (bool):
            Pass the partially initialized instance that is being initialized
            as a positional argument.

    .. versionadded:: 17.1.0  *takes_self*
    """

    __slots__ = ("factory", "takes_self")

    def __init__(self, factory, takes_self=False):
        self.factory = factory
        self.takes_self = takes_self

    def __getstate__(self):
        """
        Play nice with pickle.
        """
        return tuple(getattr(self, name) for name in self.__slots__)

    def __setstate__(self, state):
        """
        Play nice with pickle.
        """
        for name, value in zip(self.__slots__, state):
            setattr(self, name, value)


_f = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=True,
        init=True,
        inherited=False,
    )
    for name in Factory.__slots__
]

Factory = _add_hash(_add_eq(_add_repr(Factory, attrs=_f), attrs=_f), attrs=_f)


class Converter:
    """
    Stores a converter callable.

    Allows for the wrapped converter to take additional arguments. The
    arguments are passed in the order they are documented.

    Args:
        converter (Callable): A callable that converts the passed value.

        takes_self (bool):
            Pass the partially initialized instance that is being initialized
            as a positional argument. (default: `False`)

        takes_field (bool):
            Pass the field definition (an :class:`Attribute`) into the
            converter as a positional argument. (default: `False`)

    .. versionadded:: 24.1.0
    """

    __slots__ = (
        "__call__",
        "_first_param_type",
        "_global_name",
        "converter",
        "takes_field",
        "takes_self",
    )

    def __init__(self, converter, *, takes_self=False, takes_field=False):
        self.converter = converter
        self.takes_self = takes_self
        self.takes_field = takes_field

        ex = _AnnotationExtractor(converter)
        self._first_param_type = ex.get_first_param_type()

        if not (self.takes_self or self.takes_field):
            self.__call__ = lambda value, _, __: self.converter(value)
        elif self.takes_self and not self.takes_field:
            self.__call__ = lambda value, instance, __: self.converter(
                value, instance
            )
        elif not self.takes_self and self.takes_field:
            self.__call__ = lambda value, __, field: self.converter(
                value, field
            )
        else:
            self.__call__ = self.converter

        rt = ex.get_return_type()
        if rt is not None:
            self.__call__.__annotations__["return"] = rt

    @staticmethod
    def _get_global_name(attr_name: str) -> str:
        """
        Return the name that a converter for an attribute name *attr_name*
        would have.
        """
        return f"__attr_converter_{attr_name}"

    def _fmt_converter_call(self, attr_name: str, value_var: str) -> str:
        """
        Return a string that calls the converter for an attribute name
        *attr_name* and the value in variable named *value_var* according to
        `self.takes_self` and `self.takes_field`.
        """
        if not (self.takes_self or self.takes_field):
            return f"{self._get_global_name(attr_name)}({value_var})"

        if self.takes_self and self.takes_field:
            return f"{self._get_global_name(attr_name)}({value_var}, self, attr_dict['{attr_name}'])"

        if self.takes_self:
            return f"{self._get_global_name(attr_name)}({value_var}, self)"

        return f"{self._get_global_name(attr_name)}({value_var}, attr_dict['{attr_name}'])"

    def __getstate__(self):
        """
        Return a dict containing only converter and takes_self -- the rest gets
        computed when loading.
        """
        return {
            "converter": self.converter,
            "takes_self": self.takes_self,
            "takes_field": self.takes_field,
        }

    def __setstate__(self, state):
        """
        Load instance from state.
        """
        self.__init__(**state)


_f = [
    Attribute(
        name=name,
        default=NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        order=False,
        hash=True,
        init=True,
        inherited=False,
    )
    for name in ("converter", "takes_self", "takes_field")
]

Converter = _add_hash(
    _add_eq(_add_repr(Converter, attrs=_f), attrs=_f), attrs=_f
)


def make_class(
    name, attrs, bases=(object,), class_body=None, **attributes_arguments
):
    r"""
    A quick way to create a new class called *name* with *attrs*.

    .. note::

        ``make_class()`` is a thin wrapper around `attr.s`, not `attrs.define`
        which means that it doesn't come with some of the improved defaults.

        For example, if you want the same ``on_setattr`` behavior as in
        `attrs.define`, you have to pass the hooks yourself: ``make_class(...,
        on_setattr=setters.pipe(setters.convert, setters.validate)``

    .. warning::

        It is *your* duty to ensure that the class name and the attribute names
        are valid identifiers. ``make_class()`` will *not* validate them for
        you.

    Args:
        name (str): The name for the new class.

        attrs (list | dict):
            A list of names or a dictionary of mappings of names to `attr.ib`\
            s / `attrs.field`\ s.

            The order is deduced from the order of the names or attributes
            inside *attrs*.  Otherwise the order of the definition of the
            attributes is used.

        bases (tuple[type, ...]): Classes that the new class will subclass.

        class_body (dict):
            An optional dictionary of class attributes for the new class.

        attributes_arguments: Passed unmodified to `attr.s`.

    Returns:
        type: A new class with *attrs*.

    .. versionadded:: 17.1.0 *bases*
    .. versionchanged:: 18.1.0 If *attrs* is ordered, the order is retained.
    .. versionchanged:: 23.2.0 *class_body*
    .. versionchanged:: 25.2.0 Class names can now be unicode.
    """
    # Class identifiers are converted into the normal form NFKC while parsing
    name = unicodedata.normalize("NFKC", name)

    if isinstance(attrs, dict):
        cls_dict = attrs
    elif isinstance(attrs, (list, tuple)):
        cls_dict = {a: attrib() for a in attrs}
    else:
        msg = "attrs argument must be a dict or a list."
        raise TypeError(msg)

    pre_init = cls_dict.pop("__attrs_pre_init__", None)
    post_init = cls_dict.pop("__attrs_post_init__", None)
    user_init = cls_dict.pop("__init__", None)

    body = {}
    if class_body is not None:
        body.update(class_body)
    if pre_init is not None:
        body["__attrs_pre_init__"] = pre_init
    if post_init is not None:
        body["__attrs_post_init__"] = post_init
    if user_init is not None:
        body["__init__"] = user_init

    type_ = types.new_class(name, bases, {}, lambda ns: ns.update(body))

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the class is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    with contextlib.suppress(AttributeError, ValueError):
        type_.__module__ = sys._getframe(1).f_globals.get(
            "__name__", "__main__"
        )

    # We do it here for proper warnings with meaningful stacklevel.
    cmp = attributes_arguments.pop("cmp", None)
    (
        attributes_arguments["eq"],
        attributes_arguments["order"],
    ) = _determine_attrs_eq_order(
        cmp,
        attributes_arguments.get("eq"),
        attributes_arguments.get("order"),
        True,
    )

    cls = _attrs(these=cls_dict, **attributes_arguments)(type_)
    # Only add type annotations now or "_attrs()" will complain:
    cls.__annotations__ = {
        k: v.type for k, v in cls_dict.items() if v.type is not None
    }
    return cls


# These are required by within this module so we define them here and merely
# import into .validators / .converters.


@attrs(slots=True, unsafe_hash=True)
class _AndValidator:
    """
    Compose many validators to a single one.
    """

    _validators = attrib()

    def __call__(self, inst, attr, value):
        for v in self._validators:
            v(inst, attr, value)


def and_(*validators):
    """
    A validator that composes multiple validators into one.

    When called on a value, it runs all wrapped validators.

    Args:
        validators (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of validators.

    .. versionadded:: 17.1.0
    """
    vals = []
    for validator in validators:
        vals.extend(
            validator._validators
            if isinstance(validator, _AndValidator)
            else [validator]
        )

    return _AndValidator(tuple(vals))


def pipe(*converters):
    """
    A converter that composes multiple converters into one.

    When called on a value, it runs all wrapped converters, returning the
    *last* value.

    Type annotations will be inferred from the wrapped converters', if they
    have any.

        converters (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of converters.

    .. versionadded:: 20.1.0
    """

    return_instance = any(isinstance(c, Converter) for c in converters)

    if return_instance:

        def pipe_converter(val, inst, field):
            for c in converters:
                val = (
                    c(val, inst, field) if isinstance(c, Converter) else c(val)
                )

            return val

    else:

        def pipe_converter(val):
            for c in converters:
                val = c(val)

            return val

    if not converters:
        # If the converter list is empty, pipe_converter is the identity.
        A = TypeVar("A")
        pipe_converter.__annotations__.update({"val": A, "return": A})
    else:
        # Get parameter type from first converter.
        t = _AnnotationExtractor(converters[0]).get_first_param_type()
        if t:
            pipe_converter.__annotations__["val"] = t

        last = converters[-1]
        if not PY_3_11_PLUS and isinstance(last, Converter):
            last = last.__call__

        # Get return type from last converter.
        rt = _AnnotationExtractor(last).get_return_type()
        if rt:
            pipe_converter.__annotations__["return"] = rt

    if return_instance:
        return Converter(pipe_converter, takes_self=True, takes_field=True)
    return pipe_converter
<|endoftext|># SPDX-License-Identifier: MIT

import inspect
import platform
import sys
import threading

from collections.abc import Mapping, Sequence  # noqa: F401
from typing import _GenericAlias


PYPY = platform.python_implementation() == "PyPy"
PY_3_10_PLUS = sys.version_info[:2] >= (3, 10)
PY_3_11_PLUS = sys.version_info[:2] >= (3, 11)
PY_3_12_PLUS = sys.version_info[:2] >= (3, 12)
PY_3_13_PLUS = sys.version_info[:2] >= (3, 13)
PY_3_14_PLUS = sys.version_info[:2] >= (3, 14)


if PY_3_14_PLUS:
    import annotationlib

    # We request forward-ref annotations to not break in the presence of
    # forward references.

    def _get_annotations(cls):
        return annotationlib.get_annotations(
            cls, format=annotationlib.Format.FORWARDREF
        )

else:

    def _get_annotations(cls):
        """
        Get annotations for *cls*.
        """
        return cls.__dict__.get("__annotations__", {})


class _AnnotationExtractor:
    """
    Extract type annotations from a callable, returning None whenever there
    is none.
    """

    __slots__ = ["sig"]

    def __init__(self, callable):
        try:
            self.sig = inspect.signature(callable)
        except (ValueError, TypeError):  # inspect failed
            self.sig = None

    def get_first_param_type(self):
        """
        Return the type annotation of the first argument if it's not empty.
        """
        if not self.sig:
            return None

        params = list(self.sig.parameters.values())
        if params and params[0].annotation is not inspect.Parameter.empty:
            return params[0].annotation

        return None

    def get_return_type(self):
        """
        Return the return type if it's not empty.
        """
        if (
            self.sig
            and self.sig.return_annotation is not inspect.Signature.empty
        ):
            return self.sig.return_annotation

        return None


# Thread-local global to track attrs instances which are already being repr'd.
# This is needed because there is no other (thread-safe) way to pass info
# about the instances that are already being repr'd through the call stack
# in order to ensure we don't perform infinite recursion.
#
# For instance, if an instance contains a dict which contains that instance,
# we need to know that we're already repr'ing the outside instance from within
# the dict's repr() call.
#
# This lives here rather than in _make.py so that the functions in _make.py
# don't have a direct reference to the thread-local in their globals dict.
# If they have such a reference, it breaks cloudpickle.
repr_context = threading.local()


def get_generic_base(cl):
    """If this is a generic class (A[str]), return the generic base for it."""
    if cl.__class__ is _GenericAlias:
        return cl.__origin__
    return None
<|endoftext|># SPDX-License-Identifier: MIT


import functools
import types

from ._make import __ne__


_operation_names = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def cmp_using(
    eq=None,
    lt=None,
    le=None,
    gt=None,
    ge=None,
    require_same_type=True,
    class_name="Comparable",
):
    """
    Create a class that can be passed into `attrs.field`'s ``eq``, ``order``,
    and ``cmp`` arguments to customize field comparison.

    The resulting class will have a full set of ordering methods if at least
    one of ``{lt, le, gt, ge}`` and ``eq``  are provided.

    Args:
        eq (typing.Callable | None):
            Callable used to evaluate equality of two objects.

        lt (typing.Callable | None):
            Callable used to evaluate whether one object is less than another
            object.

        le (typing.Callable | None):
            Callable used to evaluate whether one object is less than or equal
            to another object.

        gt (typing.Callable | None):
            Callable used to evaluate whether one object is greater than
            another object.

        ge (typing.Callable | None):
            Callable used to evaluate whether one object is greater than or
            equal to another object.

        require_same_type (bool):
            When `True`, equality and ordering methods will return
            `NotImplemented` if objects are not of the same type.

        class_name (str | None): Name of class. Defaults to "Comparable".

    See `comparison` for more details.

    .. versionadded:: 21.1.0
    """

    body = {
        "__slots__": ["value"],
        "__init__": _make_init(),
        "_requirements": [],
        "_is_comparable_to": _is_comparable_to,
    }

    # Add operations.
    num_order_functions = 0
    has_eq_function = False

    if eq is not None:
        has_eq_function = True
        body["__eq__"] = _make_operator("eq", eq)
        body["__ne__"] = __ne__

    if lt is not None:
        num_order_functions += 1
        body["__lt__"] = _make_operator("lt", lt)

    if le is not None:
        num_order_functions += 1
        body["__le__"] = _make_operator("le", le)

    if gt is not None:
        num_order_functions += 1
        body["__gt__"] = _make_operator("gt", gt)

    if ge is not None:
        num_order_functions += 1
        body["__ge__"] = _make_operator("ge", ge)

    type_ = types.new_class(
        class_name, (object,), {}, lambda ns: ns.update(body)
    )

    # Add same type requirement.
    if require_same_type:
        type_._requirements.append(_check_same_type)

    # Add total ordering if at least one operation was defined.
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            # functools.total_ordering requires __eq__ to be defined,
            # so raise early error here to keep a nice stack.
            msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
            raise ValueError(msg)
        type_ = functools.total_ordering(type_)

    return type_


def _make_init():
    """
    Create __init__ method.
    """

    def __init__(self, value):
        """
        Initialize object with *value*.
        """
        self.value = value

    return __init__


def _make_operator(name, func):
    """
    Create operator method.
    """

    def method(self, other):
        if not self._is_comparable_to(other):
            return NotImplemented

        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented

        return result

    method.__name__ = f"__{name}__"
    method.__doc__ = (
        f"Return a {_operation_names[name]} b.  Computed by attrs."
    )

    return method


def _is_comparable_to(self, other):
    """
    Check whether `other` is comparable to `self`.
    """
    return all(func(self, other) for func in self._requirements)


def _check_same_type(self, other):
    """
    Return True if *self* and *other* are of the same type, False otherwise.
    """
    return other.value.__class__ is self.value.__class__
<|endoftext|># SPDX-License-Identifier: MIT

"""
Commonly useful validators.
"""

import operator
import re

from contextlib import contextmanager
from re import Pattern

from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError


__all__ = [
    "and_",
    "deep_iterable",
    "deep_mapping",
    "disabled",
    "ge",
    "get_disabled",
    "gt",
    "in_",
    "instance_of",
    "is_callable",
    "le",
    "lt",
    "matches_re",
    "max_len",
    "min_len",
    "not_",
    "optional",
    "or_",
    "set_disabled",
]


def set_disabled(disabled):
    """
    Globally disable or enable running validators.

    By default, they are run.

    Args:
        disabled (bool): If `True`, disable running all validators.

    .. warning::

        This function is not thread-safe!

    .. versionadded:: 21.3.0
    """
    set_run_validators(not disabled)


def get_disabled():
    """
    Return a bool indicating whether validators are currently disabled or not.

    Returns:
        bool:`True` if validators are currently disabled.

    .. versionadded:: 21.3.0
    """
    return not get_run_validators()


@contextmanager
def disabled():
    """
    Context manager that disables running validators within its context.

    .. warning::

        This context manager is not thread-safe!

    .. versionadded:: 21.3.0
    .. versionchanged:: 26.1.0 The contextmanager is nestable.
    """
    prev = get_run_validators()
    set_run_validators(False)
    try:
        yield
    finally:
        set_run_validators(prev)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InstanceOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not isinstance(value, self.type):
            msg = f"'{attr.name}' must be {self.type!r} (got {value!r} that is a {value.__class__!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<instance_of validator for type {self.type!r}>"


def instance_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `isinstance` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type]): The type to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _InstanceOfValidator(type)


@attrs(repr=False, frozen=True, slots=True)
class _MatchesReValidator:
    pattern = attrib()
    match_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.match_func(value):
            msg = f"'{attr.name}' must match regex {self.pattern.pattern!r} ({value!r} doesn't)"
            raise ValueError(
                msg,
                attr,
                self.pattern,
                value,
            )

    def __repr__(self):
        return f"<matches_re validator for pattern {self.pattern!r}>"


def matches_re(regex, flags=0, func=None):
    r"""
    A validator that raises `ValueError` if the initializer is called with a
    string that doesn't match *regex*.

    Args:
        regex (str, re.Pattern):
            A regex string or precompiled pattern to match against

        flags (int):
            Flags that will be passed to the underlying re function (default 0)

        func (typing.Callable):
            Which underlying `re` function to call. Valid options are
            `re.fullmatch`, `re.search`, and `re.match`; the default `None`
            means `re.fullmatch`. For performance reasons, the pattern is
            always precompiled using `re.compile`.

    .. versionadded:: 19.2.0
    .. versionchanged:: 21.3.0 *regex* can be a pre-compiled pattern.
    """
    valid_funcs = (re.fullmatch, None, re.search, re.match)
    if func not in valid_funcs:
        msg = "'func' must be one of {}.".format(
            ", ".join(
                sorted((e and e.__name__) or "None" for e in set(valid_funcs))
            )
        )
        raise ValueError(msg)

    if isinstance(regex, Pattern):
        if flags:
            msg = "'flags' can only be used with a string pattern; pass flags to re.compile() instead"
            raise TypeError(msg)
        pattern = regex
    else:
        pattern = re.compile(regex, flags)

    if func is re.match:
        match_func = pattern.match
    elif func is re.search:
        match_func = pattern.search
    else:
        match_func = pattern.fullmatch

    return _MatchesReValidator(pattern, match_func)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OptionalValidator:
    validator = attrib()

    def __call__(self, inst, attr, value):
        if value is None:
            return

        self.validator(inst, attr, value)

    def __repr__(self):
        return f"<optional validator for {self.validator!r} or None>"


def optional(validator):
    """
    A validator that makes an attribute optional.  An optional attribute is one
    which can be set to `None` in addition to satisfying the requirements of
    the sub-validator.

    Args:
        validator
            (typing.Callable | tuple[typing.Callable] | list[typing.Callable]):
            A validator (or validators) that is used for non-`None` values.

    .. versionadded:: 15.1.0
    .. versionchanged:: 17.1.0 *validator* can be a list of validators.
    .. versionchanged:: 23.1.0 *validator* can also be a tuple of validators.
    """
    if isinstance(validator, (list, tuple)):
        return _OptionalValidator(_AndValidator(validator))

    return _OptionalValidator(validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InValidator:
    options = attrib()
    _original_options = attrib(hash=False)

    def __call__(self, inst, attr, value):
        try:
            in_options = value in self.options
        except TypeError:  # e.g. `1 in "abc"`
            in_options = False

        if not in_options:
            msg = f"'{attr.name}' must be in {self._original_options!r} (got {value!r})"
            raise ValueError(
                msg,
                attr,
                self._original_options,
                value,
            )

    def __repr__(self):
        return f"<in_ validator with options {self._original_options!r}>"


def in_(options):
    """
    A validator that raises a `ValueError` if the initializer is called with a
    value that does not belong in the *options* provided.

    The check is performed using ``value in options``, so *options* has to
    support that operation.

    To keep the validator hashable, dicts, lists, and sets are transparently
    transformed into a `tuple`.

    Args:
        options: Allowed options.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected options, and the value it got.

    .. versionadded:: 17.1.0
    .. versionchanged:: 22.1.0
       The ValueError was incomplete until now and only contained the human
       readable error message. Now it contains all the information that has
       been promised since 17.1.0.
    .. versionchanged:: 24.1.0
       *options* that are a list, dict, or a set are now transformed into a
       tuple to keep the validator hashable.
    """
    repr_options = options
    if isinstance(options, (list, dict, set)):
        options = tuple(options)

    return _InValidator(options, repr_options)


@attrs(repr=False, slots=False, unsafe_hash=True)
class _IsCallableValidator:
    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not callable(value):
            message = (
                "'{name}' must be callable "
                "(got {value!r} that is a {actual!r})."
            )
            raise NotCallableError(
                msg=message.format(
                    name=attr.name, value=value, actual=value.__class__
                ),
                value=value,
            )

    def __repr__(self):
        return "<is_callable validator>"


def is_callable():
    """
    A validator that raises a `attrs.exceptions.NotCallableError` if the
    initializer is called with a value for this particular attribute that is
    not callable.

    .. versionadded:: 19.1.0

    Raises:
        attrs.exceptions.NotCallableError:
            With a human readable error message containing the attribute
            (`attrs.Attribute`) name, and the value it got.
    """
    return _IsCallableValidator()


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepIterable:
    member_validator = attrib(validator=is_callable())
    iterable_validator = attrib(
        default=None, validator=optional(is_callable())
    )

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.iterable_validator is not None:
            self.iterable_validator(inst, attr, value)

        for member in value:
            self.member_validator(inst, attr, member)

    def __repr__(self):
        iterable_identifier = (
            ""
            if self.iterable_validator is None
            else f" {self.iterable_validator!r}"
        )
        return (
            f"<deep_iterable validator for{iterable_identifier}"
            f" iterables of {self.member_validator!r}>"
        )


def deep_iterable(member_validator, iterable_validator=None):
    """
    A validator that performs deep validation of an iterable.

    Args:
        member_validator: Validator(s) to apply to iterable members.

        iterable_validator:
            Validator(s) to apply to iterable itself (optional).

    Raises
        TypeError: if any sub-validators fail

    .. versionadded:: 19.1.0

    .. versionchanged:: 25.4.0
       *member_validator* and *iterable_validator* can now be a list or tuple
       of validators.
    """
    if isinstance(member_validator, (list, tuple)):
        member_validator = and_(*member_validator)
    if isinstance(iterable_validator, (list, tuple)):
        iterable_validator = and_(*iterable_validator)
    return _DeepIterable(member_validator, iterable_validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepMapping:
    key_validator = attrib(validator=optional(is_callable()))
    value_validator = attrib(validator=optional(is_callable()))
    mapping_validator = attrib(validator=optional(is_callable()))

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.mapping_validator is not None:
            self.mapping_validator(inst, attr, value)

        for key in value:
            if self.key_validator is not None:
                self.key_validator(inst, attr, key)
            if self.value_validator is not None:
                self.value_validator(inst, attr, value[key])

    def __repr__(self):
        return f"<deep_mapping validator for objects mapping {self.key_validator!r} to {self.value_validator!r}>"


def deep_mapping(
    key_validator=None, value_validator=None, mapping_validator=None
):
    """
    A validator that performs deep validation of a dictionary.

    All validators are optional, but at least one of *key_validator* or
    *value_validator* must be provided.

    Args:
        key_validator: Validator(s) to apply to dictionary keys.

        value_validator: Validator(s) to apply to dictionary values.

        mapping_validator:
            Validator(s) to apply to top-level mapping attribute.

    .. versionadded:: 19.1.0

    .. versionchanged:: 25.4.0
       *key_validator* and *value_validator* are now optional, but at least one
       of them must be provided.

    .. versionchanged:: 25.4.0
       *key_validator*, *value_validator*, and *mapping_validator* can now be a
       list or tuple of validators.

    Raises:
        TypeError: If any sub-validator fails on validation.

        ValueError:
            If neither *key_validator* nor *value_validator* is provided on
            instantiation.
    """
    if key_validator is None and value_validator is None:
        msg = (
            "At least one of key_validator or value_validator must be provided"
        )
        raise ValueError(msg)

    if isinstance(key_validator, (list, tuple)):
        key_validator = and_(*key_validator)
    if isinstance(value_validator, (list, tuple)):
        value_validator = and_(*value_validator)
    if isinstance(mapping_validator, (list, tuple)):
        mapping_validator = and_(*mapping_validator)

    return _DeepMapping(key_validator, value_validator, mapping_validator)


@attrs(repr=False, frozen=True, slots=True)
class _NumberValidator:
    bound = attrib()
    compare_op = attrib()
    compare_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.compare_func(value, self.bound):
            msg = f"'{attr.name}' must be {self.compare_op} {self.bound}: {value}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<Validator for x {self.compare_op} {self.bound}>"


def lt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number larger or equal to *val*.

    The validator uses `operator.lt` to compare the values.

    Args:
        val: Exclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<", operator.lt)


def le(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number greater than *val*.

    The validator uses `operator.le` to compare the values.

    Args:
        val: Inclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<=", operator.le)


def ge(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller than *val*.

    The validator uses `operator.ge` to compare the values.

    Args:
        val: Inclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">=", operator.ge)


def gt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller or equal to *val*.

    The validator uses `operator.gt` to compare the values.

    Args:
       val: Exclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">", operator.gt)


@attrs(repr=False, frozen=True, slots=True)
class _MaxLengthValidator:
    max_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) > self.max_length:
            msg = f"Length of '{attr.name}' must be <= {self.max_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<max_len validator for {self.max_length}>"


def max_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is longer than *length*.

    Args:
        length (int): Maximum length of the string or iterable

    .. versionadded:: 21.3.0
    """
    return _MaxLengthValidator(length)


@attrs(repr=False, frozen=True, slots=True)
class _MinLengthValidator:
    min_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) < self.min_length:
            msg = f"Length of '{attr.name}' must be >= {self.min_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<min_len validator for {self.min_length}>"


def min_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is shorter than *length*.

    Args:
        length (int): Minimum length of the string or iterable

    .. versionadded:: 22.1.0
    """
    return _MinLengthValidator(length)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _SubclassOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not issubclass(value, self.type):
            msg = f"'{attr.name}' must be a subclass of {self.type!r} (got {value!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<subclass_of validator for type {self.type!r}>"


def _subclass_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `issubclass` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type, ...]): The type(s) to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _SubclassOfValidator(type)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _NotValidator:
    validator = attrib()
    msg = attrib(
        converter=default_if_none(
            "not_ validator child '{validator!r}' "
            "did not raise a captured error"
        )
    )
    exc_types = attrib(
        validator=deep_iterable(
            member_validator=_subclass_of(Exception),
            iterable_validator=instance_of(tuple),
        ),
    )

    def __call__(self, inst, attr, value):
        try:
            self.validator(inst, attr, value)
        except self.exc_types:
            pass  # suppress error to invert validity
        else:
            raise ValueError(
                self.msg.format(
                    validator=self.validator,
                    exc_types=self.exc_types,
                ),
                attr,
                self.validator,
                value,
                self.exc_types,
            )

    def __repr__(self):
        return f"<not_ validator wrapping {self.validator!r}, capturing {self.exc_types!r}>"


def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
    """
    A validator that wraps and logically 'inverts' the validator passed to it.
    It will raise a `ValueError` if the provided validator *doesn't* raise a
    `ValueError` or `TypeError` (by default), and will suppress the exception
    if the provided validator *does*.

    Intended to be used with existing validators to compose logic without
    needing to create inverted variants, for example, ``not_(in_(...))``.

    Args:
        validator: A validator to be logically inverted.

        msg (str):
            Message to raise if validator fails. Formatted with keys
            ``exc_types`` and ``validator``.

        exc_types (tuple[type, ...]):
            Exception type(s) to capture. Other types raised by child
            validators will not be intercepted and pass through.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the validator that failed to raise an
            exception, the value it got, and the expected exception types.

    .. versionadded:: 22.2.0
    """
    try:
        exc_types = tuple(exc_types)
    except TypeError:
        exc_types = (exc_types,)
    return _NotValidator(validator, msg, exc_types)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OrValidator:
    validators = attrib()

    def __call__(self, inst, attr, value):
        for v in self.validators:
            try:
                v(inst, attr, value)
            except Exception:  # noqa: BLE001, PERF203, S112
                continue
            else:
                return

        msg = f"None of {self.validators!r} satisfied for value {value!r}"
        raise ValueError(msg)

    def __repr__(self):
        return f"<or validator wrapping {self.validators!r}>"


def or_(*validators):
    """
    A validator that composes multiple validators into one.

    When called on a value, it runs all wrapped validators until one of them is
    satisfied.

    Args:
        validators (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of validators.

    Raises:
        ValueError:
            If no validator is satisfied. Raised with a human-readable error
            message listing all the wrapped validators and the value that
            failed all of them.

    .. versionadded:: 24.1.0
    """
    vals = []
    for v in validators:
        vals.extend(v.validators if isinstance(v, _OrValidator) else [v])

    return _OrValidator(tuple(vals))
<|endoftext|># SPDX-License-Identifier: MIT

"""
Classes Without Boilerplate
"""

from functools import partial
from typing import Callable, Literal, Protocol

from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, has, resolve_types
from ._make import (
    NOTHING,
    Attribute,
    Converter,
    Factory,
    _Nothing,
    attrib,
    attrs,
    evolve,
    fields,
    fields_dict,
    make_class,
    validate,
)
from ._next_gen import define, field, frozen, mutable
from ._version_info import VersionInfo


s = attributes = attrs
ib = attr = attrib
dataclass = partial(attrs, auto_attribs=True)  # happy Easter ;)


class AttrsInstance(Protocol):
    pass


NothingType = Literal[_Nothing.NOTHING]

__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "Converter",
    "Factory",
    "NothingType",
    "asdict",
    "assoc",
    "astuple",
    "attr",
    "attrib",
    "attributes",
    "attrs",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "get_run_validators",
    "has",
    "ib",
    "make_class",
    "mutable",
    "resolve_types",
    "s",
    "set_run_validators",
    "setters",
    "validate",
    "validators",
]


def _make_getattr(mod_name: str) -> Callable:
    """
    Create a metadata proxy for packaging information that uses *mod_name* in
    its warnings and errors.
    """

    def __getattr__(name: str) -> str:
        if name not in ("__version__", "__version_info__"):
            msg = f"module {mod_name} has no attribute {name}"
            raise AttributeError(msg)

        from importlib.metadata import metadata

        meta = metadata("attrs")

        if name == "__version_info__":
            return VersionInfo._from_version_string(meta["version"])

        return meta["version"]

    return __getattr__


__getattr__ = _make_getattr(__name__)
<|endoftext|># SPDX-License-Identifier: MIT


import copy

from ._compat import get_generic_base
from ._make import _OBJ_SETATTR, NOTHING, fields
from .exceptions import AttrsAttributeNotFoundError


_ATOMIC_TYPES = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        complex,
        bytes,
        type(...),
        type,
        range,
        property,
    }
)


def asdict(
    inst,
    recurse=True,
    filter=None,
    dict_factory=dict,
    retain_collection_types=False,
    value_serializer=None,
):
    """
    Return the *attrs* attribute values of *inst* as a dict.

    Optionally recurse into other *attrs*-decorated classes.

    Args:
        inst: Instance of an *attrs*-decorated class.

        recurse (bool): Recurse into classes that are also *attrs*-decorated.

        filter (~typing.Callable):
            A callable whose return code determines whether an attribute or
            element is included (`True`) or dropped (`False`).  Is called with
            the `attrs.Attribute` as the first argument and the value as the
            second argument.

        dict_factory (~typing.Callable):
            A callable to produce dictionaries from.  For example, to produce
            ordered dictionaries instead of normal Python dictionaries, pass in
            ``collections.OrderedDict``.

        retain_collection_types (bool):
            Do not convert to `list` when encountering an attribute whose type
            is `tuple` or `set`.  Only meaningful if *recurse* is `True`.

        value_serializer (typing.Callable | None):
            A hook that is called for every attribute or dict key/value.  It
            receives the current instance, field and value and must return the
            (updated) value.  The hook is run *after* the optional *filter* has
            been applied.

    Returns:
        Return type of *dict_factory*.

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  versionadded:: 16.0.0 *dict_factory*
    ..  versionadded:: 16.1.0 *retain_collection_types*
    ..  versionadded:: 20.3.0 *value_serializer*
    ..  versionadded:: 21.3.0
        If a dict has a collection for a key, it is serialized as a tuple.
    """
    attrs = fields(inst.__class__)
    rv = dict_factory()
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue

        if value_serializer is not None:
            v = value_serializer(inst, a, v)

        if recurse is True:
            value_type = type(v)
            if value_type in _ATOMIC_TYPES:
                rv[a.name] = v
            elif has(value_type):
                rv[a.name] = asdict(
                    v,
                    recurse=True,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
            elif issubclass(value_type, (tuple, list, set, frozenset)):
                cf = value_type if retain_collection_types is True else list
                items = [
                    _asdict_anything(
                        i,
                        is_key=False,
                        filter=filter,
                        dict_factory=dict_factory,
                        retain_collection_types=retain_collection_types,
                        value_serializer=value_serializer,
                    )
                    for i in v
                ]
                try:
                    rv[a.name] = cf(items)
                except TypeError:
                    if not issubclass(cf, tuple):
                        raise
                    # Workaround for TypeError: cf.__new__() missing 1 required
                    # positional argument (which appears, for a namedturle)
                    rv[a.name] = cf(*items)
            elif issubclass(value_type, dict):
                df = dict_factory
                rv[a.name] = df(
                    (
                        _asdict_anything(
                            kk,
                            is_key=True,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                        _asdict_anything(
                            vv,
                            is_key=False,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                    )
                    for kk, vv in v.items()
                )
            else:
                rv[a.name] = v
        else:
            rv[a.name] = v
    return rv


def _asdict_anything(
    val,
    is_key,
    filter,
    dict_factory,
    retain_collection_types,
    value_serializer,
):
    """
    ``asdict`` only works on attrs instances, this works on anything.
    """
    val_type = type(val)
    if val_type in _ATOMIC_TYPES:
        rv = val
        if value_serializer is not None:
            rv = value_serializer(None, None, rv)
    elif getattr(val_type, "__attrs_attrs__", None) is not None:
        # Attrs class.
        rv = asdict(
            val,
            recurse=True,
            filter=filter,
            dict_factory=dict_factory,
            retain_collection_types=retain_collection_types,
            value_serializer=value_serializer,
        )
    elif issubclass(val_type, (tuple, list, set, frozenset)):
        if retain_collection_types is True:
            cf = val.__class__
        elif is_key:
            cf = tuple
        else:
            cf = list

        rv = cf(
            [
                _asdict_anything(
                    i,
                    is_key=False,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
                for i in val
            ]
        )
    elif issubclass(val_type, dict):
        df = dict_factory
        rv = df(
            (
                _asdict_anything(
                    kk,
                    is_key=True,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
                _asdict_anything(
                    vv,
                    is_key=False,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
            )
            for kk, vv in val.items()
        )
    else:
        rv = val
        if value_serializer is not None:
            rv = value_serializer(None, None, rv)

    return rv


def astuple(
    inst,
    recurse=True,
    filter=None,
    tuple_factory=tuple,
    retain_collection_types=False,
):
    """
    Return the *attrs* attribute values of *inst* as a tuple.

    Optionally recurse into other *attrs*-decorated classes.

    Args:
        inst: Instance of an *attrs*-decorated class.

        recurse (bool):
            Recurse into classes that are also *attrs*-decorated.

        filter (~typing.Callable):
            A callable whose return code determines whether an attribute or
            element is included (`True`) or dropped (`False`).  Is called with
            the `attrs.Attribute` as the first argument and the value as the
            second argument.

        tuple_factory (~typing.Callable):
            A callable to produce tuples from. For example, to produce lists
            instead of tuples.

        retain_collection_types (bool):
            Do not convert to `list` or `dict` when encountering an attribute
            which type is `tuple`, `dict` or `set`. Only meaningful if
            *recurse* is `True`.

    Returns:
        Return type of *tuple_factory*

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  versionadded:: 16.2.0
    """
    attrs = fields(inst.__class__)
    rv = []
    retain = retain_collection_types  # Very long. :/
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue
        value_type = type(v)
        if recurse is True:
            if value_type in _ATOMIC_TYPES:
                rv.append(v)
            elif has(value_type):
                rv.append(
                    astuple(
                        v,
                        recurse=True,
                        filter=filter,
                        tuple_factory=tuple_factory,
                        retain_collection_types=retain,
                    )
                )
            elif issubclass(value_type, (tuple, list, set, frozenset)):
                cf = v.__class__ if retain is True else list
                items = [
                    (
                        astuple(
                            j,
                            recurse=True,
                            filter=filter,
                            tuple_factory=tuple_factory,
                            retain_collection_types=retain,
                        )
                        if has(j.__class__)
                        else j
                    )
                    for j in v
                ]
                try:
                    rv.append(cf(items))
                except TypeError:
                    if not issubclass(cf, tuple):
                        raise
                    # Workaround for TypeError: cf.__new__() missing 1 required
                    # positional argument (which appears, for a namedturle)
                    rv.append(cf(*items))
            elif issubclass(value_type, dict):
                df = value_type if retain is True else dict
                rv.append(
                    df(
                        (
                            (
                                astuple(
                                    kk,
                                    tuple_factory=tuple_factory,
                                    retain_collection_types=retain,
                                )
                                if has(kk.__class__)
                                else kk
                            ),
                            (
                                astuple(
                                    vv,
                                    tuple_factory=tuple_factory,
                                    retain_collection_types=retain,
                                )
                                if has(vv.__class__)
                                else vv
                            ),
                        )
                        for kk, vv in v.items()
                    )
                )
            else:
                rv.append(v)
        else:
            rv.append(v)

    return rv if tuple_factory is list else tuple_factory(rv)


def has(cls):
    """
    Check whether *cls* is a class with *attrs* attributes.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

    Returns:
        bool:
    """
    attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs is not None:
        return True

    # No attrs, maybe it's a specialized generic (A[str])?
    generic_base = get_generic_base(cls)
    if generic_base is not None:
        generic_attrs = getattr(generic_base, "__attrs_attrs__", None)
        if generic_attrs is not None:
            # Stick it on here for speed next time.
            cls.__attrs_attrs__ = generic_attrs
        return generic_attrs is not None
    return False


def assoc(inst, **changes):
    """
    Copy *inst* and apply *changes*.

    This is different from `evolve` that applies the changes to the arguments
    that create the new instance.

    `evolve`'s behavior is preferable, but there are `edge cases`_ where it
    doesn't work. Therefore `assoc` is deprecated, but will not be removed.

    .. _`edge cases`: https://github.com/python-attrs/attrs/issues/251

    Args:
        inst: Instance of a class with *attrs* attributes.

        changes: Keyword changes in the new copy.

    Returns:
        A copy of inst with *changes* incorporated.

    Raises:
        attrs.exceptions.AttrsAttributeNotFoundError:
            If *attr_name* couldn't be found on *cls*.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    ..  deprecated:: 17.1.0
        Use `attrs.evolve` instead if you can. This function will not be
        removed du to the slightly different approach compared to
        `attrs.evolve`, though.
    """
    new = copy.copy(inst)
    attrs = fields(inst.__class__)
    for k, v in changes.items():
        a = getattr(attrs, k, NOTHING)
        if a is NOTHING:
            msg = f"{k} is not an attrs attribute on {new.__class__}."
            raise AttrsAttributeNotFoundError(msg)
        _OBJ_SETATTR(new, k, v)
    return new


def resolve_types(
    cls, globalns=None, localns=None, attribs=None, include_extras=True
):
    """
    Resolve any strings and forward annotations in type annotations.

    This is only required if you need concrete types in :class:`Attribute`'s
    *type* field. In other words, you don't need to resolve your types if you
    only use them for static type checking.

    With no arguments, names will be looked up in the module in which the class
    was created. If this is not what you want, for example, if the name only
    exists inside a method, you may pass *globalns* or *localns* to specify
    other dictionaries in which to look up these names. See the docs of
    `typing.get_type_hints` for more details.

    Args:
        cls (type): Class to resolve.

        globalns (dict | None): Dictionary containing global variables.

        localns (dict | None): Dictionary containing local variables.

        attribs (list | None):
            List of attribs for the given class. This is necessary when calling
            from inside a ``field_transformer`` since *cls* is not an *attrs*
            class yet.

        include_extras (bool):
            Resolve more accurately, if possible. Pass ``include_extras`` to
            ``typing.get_hints``, if supported by the typing module. On
            supported Python versions (3.9+), this resolves the types more
            accurately.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class and you didn't pass any attribs.

        NameError: If types cannot be resolved because of missing variables.

    Returns:
        *cls* so you can use this function also as a class decorator. Please
        note that you have to apply it **after** `attrs.define`. That means the
        decorator has to come in the line **before** `attrs.define`.

    ..  versionadded:: 20.1.0
    ..  versionadded:: 21.1.0 *attribs*
    ..  versionadded:: 23.1.0 *include_extras*
    """
    # Since calling get_type_hints is expensive we cache whether we've
    # done it already.
    if getattr(cls, "__attrs_types_resolved__", None) != cls:
        import typing

        kwargs = {
            "globalns": globalns,
            "localns": localns,
            "include_extras": include_extras,
        }

        hints = typing.get_type_hints(cls, **kwargs)
        for field in fields(cls) if attribs is None else attribs:
            if field.name in hints:
                # Since fields have been frozen we must work around it.
                _OBJ_SETATTR(field, "type", hints[field.name])
        # We store the class we resolved so that subclasses know they haven't
        # been resolved.
        cls.__attrs_types_resolved__ = cls

    # Return the class so you can use it as a decorator too.
    return cls
<|endoftext|># SPDX-License-Identifier: MIT

"""
Commonly useful filters for `attrs.asdict` and `attrs.astuple`.
"""

from ._make import Attribute


def _split_what(what):
    """
    Returns a tuple of `frozenset`s of classes and attributes.
    """
    return (
        frozenset(cls for cls in what if isinstance(cls, type)),
        frozenset(cls for cls in what if isinstance(cls, str)),
        frozenset(cls for cls in what if isinstance(cls, Attribute)),
    )


def include(*what):
    """
    Create a filter that only allows *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to include. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.1.0 Accept strings with field names.
    """
    cls, names, attrs = _split_what(what)

    def include_(attribute, value):
        return (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return include_


def exclude(*what):
    """
    Create a filter that does **not** allow *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to exclude. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.3.0 Accept field name string as input argument
    """
    cls, names, attrs = _split_what(what)

    def exclude_(attribute, value):
        return not (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return exclude_
<|endoftext|># SPDX-License-Identifier: MIT

"""
Commonly used hooks for on_setattr.
"""

from . import _config
from .exceptions import FrozenAttributeError


def pipe(*setters):
    """
    Run all *setters* and return the return value of the last one.

    .. versionadded:: 20.1.0
    """

    def wrapped_pipe(instance, attrib, new_value):
        rv = new_value

        for setter in setters:
            rv = setter(instance, attrib, rv)

        return rv

    return wrapped_pipe


def frozen(_, __, ___):
    """
    Prevent an attribute to be modified.

    .. versionadded:: 20.1.0
    """
    raise FrozenAttributeError


def validate(instance, attrib, new_value):
    """
    Run *attrib*'s validator on *new_value* if it has one.

    .. versionadded:: 20.1.0
    """
    if _config._run_validators is False:
        return new_value

    v = attrib.validator
    if not v:
        return new_value

    v(instance, attrib, new_value)

    return new_value


def convert(instance, attrib, new_value):
    """
    Run *attrib*'s converter -- if it has one -- on *new_value* and return the
    result.

    .. versionadded:: 20.1.0
    """
    c = attrib.converter
    if c:
        # This can be removed once we drop 3.8 and use attrs.Converter instead.
        from ._make import Converter

        if not isinstance(c, Converter):
            return c(new_value)

        return c(new_value, instance, attrib)

    return new_value


# Sentinel for disabling class-wide *on_setattr* hooks for certain attributes.
# Sphinx's autodata stopped working, so the docstring is inlined in the API
# docs.
NO_OP = object()
<|endoftext|># SPDX-License-Identifier: MIT

"""
Commonly useful converters.
"""

import typing

from ._compat import _AnnotationExtractor
from ._make import NOTHING, Converter, Factory, pipe


__all__ = [
    "default_if_none",
    "optional",
    "pipe",
    "to_bool",
]


def optional(converter):
    """
    A converter that allows an attribute to be optional. An optional attribute
    is one which can be set to `None`.

    Type annotations will be inferred from the wrapped converter's, if it has
    any.

    Args:
        converter (typing.Callable):
            the converter that is used for non-`None` values.

    .. versionadded:: 17.1.0
    """

    if isinstance(converter, Converter):

        def optional_converter(val, inst, field):
            if val is None:
                return None
            return converter(val, inst, field)

    else:

        def optional_converter(val):
            if val is None:
                return None
            return converter(val)

    xtr = _AnnotationExtractor(converter)

    t = xtr.get_first_param_type()
    if t:
        optional_converter.__annotations__["val"] = typing.Optional[t]

    rt = xtr.get_return_type()
    if rt:
        optional_converter.__annotations__["return"] = typing.Optional[rt]

    if isinstance(converter, Converter):
        return Converter(optional_converter, takes_self=True, takes_field=True)

    return optional_converter


def default_if_none(default=NOTHING, factory=None):
    """
    A converter that allows to replace `None` values by *default* or the result
    of *factory*.

    Args:
        default:
            Value to be used if `None` is passed. Passing an instance of
            `attrs.Factory` is supported, however the ``takes_self`` option is
            *not*.

        factory (typing.Callable):
            A callable that takes no parameters whose result is used if `None`
            is passed.

    Raises:
        TypeError: If **neither** *default* or *factory* is passed.

        TypeError: If **both** *default* and *factory* are passed.

        ValueError:
            If an instance of `attrs.Factory` is passed with
            ``takes_self=True``.

    .. versionadded:: 18.2.0
    """
    if default is NOTHING and factory is None:
        msg = "Must pass either `default` or `factory`."
        raise TypeError(msg)

    if default is not NOTHING and factory is not None:
        msg = "Must pass either `default` or `factory` but not both."
        raise TypeError(msg)

    if factory is not None:
        default = Factory(factory)

    if isinstance(default, Factory):
        if default.takes_self:
            msg = "`takes_self` is not supported by default_if_none."
            raise ValueError(msg)

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default.factory()

    else:

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default

    return default_if_none_converter


def to_bool(val):
    """
    Convert "boolean" strings (for example, from environment variables) to real
    booleans.

    Values mapping to `True`:

    - ``True``
    - ``"true"`` / ``"t"``
    - ``"yes"`` / ``"y"``
    - ``"on"``
    - ``"1"``
    - ``1``

    Values mapping to `False`:

    - ``False``
    - ``"false"`` / ``"f"``
    - ``"no"`` / ``"n"``
    - ``"off"``
    - ``"0"``
    - ``0``

    Raises:
        ValueError: For any other value.

    .. versionadded:: 21.3.0
    """
    if isinstance(val, str):
        val = val.lower()

    if val in (True, "true", "t", "yes", "y", "on", "1", 1):
        return True
    if val in (False, "false", "f", "no", "n", "off", "0", 0):
        return False

    msg = f"Cannot convert value to bool: {val!r}"
    raise ValueError(msg)
<|endoftext|># SPDX-License-Identifier: MIT

"""
These are keyword-only APIs that call `attr.s` and `attr.ib` with different
default values.
"""

from functools import partial

from . import setters
from ._funcs import asdict as _asdict
from ._funcs import astuple as _astuple
from ._make import (
    _DEFAULT_ON_SETATTR,
    NOTHING,
    _frozen_setattrs,
    attrib,
    attrs,
)
from .exceptions import NotAnAttrsClassError, UnannotatedAttributeError


def define(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    unsafe_hash=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
    force_kw_only=False,
):
    r"""
    A class decorator that adds :term:`dunder methods` according to
    :term:`fields <field>` specified using :doc:`type annotations <types>`,
    `field()` calls, or the *these* argument.

    Since *attrs* patches or replaces an existing class, you cannot use
    `object.__init_subclass__` with *attrs* classes, because it runs too early.
    As a replacement, you can define ``__attrs_init_subclass__`` on your class.
    It will be called by *attrs* classes that subclass it after they're
    created. See also :ref:`init-subclass`.

    Args:
        slots (bool):
            Create a :term:`slotted class <slotted classes>` that's more
            memory-efficient. Slotted classes are generally superior to the
            default dict classes, but have some gotchas you should know about,
            so we encourage you to read the :term:`glossary entry <slotted
            classes>`.

        auto_detect (bool):
            Instead of setting the *init*, *repr*, *eq*, and *hash* arguments
            explicitly, assume they are set to True **unless any** of the
            involved methods for one of the arguments is implemented in the
            *current* class (meaning, it is *not* inherited from some base
            class).

            So, for example by implementing ``__eq__`` on a class yourself,
            *attrs* will deduce ``eq=False`` and will create *neither*
            ``__eq__`` *nor* ``__ne__`` (but Python classes come with a
            sensible ``__ne__`` by default, so it *should* be enough to only
            implement ``__eq__`` in most cases).

            Passing :data:`True` or :data:`False` to *init*, *repr*, *eq*, or *hash*
            overrides whatever *auto_detect* would determine.

        auto_exc (bool):
            If the class subclasses `BaseException` (which implicitly includes
            any subclass of any exception), the following happens to behave
            like a well-behaved Python exception class:

            - the values for *eq*, *order*, and *hash* are ignored and the
              instances compare and hash by the instance's ids [#]_ ,
            - all attributes that are either passed into ``__init__`` or have a
              default value are additionally available as a tuple in the
              ``args`` attribute,
            - the value of *str* is ignored leaving ``__str__`` to base
              classes.

            .. [#]
               Note that *attrs* will *not* remove existing implementations of
               ``__hash__`` or the equality methods. It just won't add own
               ones.

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            A callable that is run whenever the user attempts to set an
            attribute (either by assignment like ``i.x = 42`` or by using
            `setattr` like ``setattr(i, "x", 42)``). It receives the same
            arguments as validators: the instance, the attribute that is being
            modified, and the new value.

            If no exception is raised, the attribute is set to the return value
            of the callable.

            If a list of callables is passed, they're automatically wrapped in
            an `attrs.setters.pipe`.

            If left None, the default behavior is to run converters and
            validators whenever an attribute is set.

        init (bool):
            Create a ``__init__`` method that initializes the *attrs*
            attributes. Leading underscores are stripped for the argument name,
            unless an alias is set on the attribute.

            .. seealso::
                `init` shows advanced ways to customize the generated
                ``__init__`` method, including executing code before and after.

        repr(bool):
            Create a ``__repr__`` method with a human readable representation
            of *attrs* attributes.

        str (bool):
            Create a ``__str__`` method that is identical to ``__repr__``. This
            is usually not necessary except for `Exception`\ s.

        eq (bool | None):
            If True or None (default), add ``__eq__`` and ``__ne__`` methods
            that check two instances for equality.

            .. seealso::
                `comparison` describes how to customize the comparison behavior
                going as far comparing NumPy arrays.

        order (bool | None):
            If True, add ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__``
            methods that behave like *eq* above and allow instances to be
            ordered.

            They compare the instances as if they were tuples of their *attrs*
            attributes if and only if the types of both classes are
            *identical*.

            If `None` mirror value of *eq*.

            .. seealso:: `comparison`

        unsafe_hash (bool | None):
            If None (default), the ``__hash__`` method is generated according
            how *eq* and *frozen* are set.

            1. If *both* are True, *attrs* will generate a ``__hash__`` for
               you.
            2. If *eq* is True and *frozen* is False, ``__hash__`` will be set
               to None, marking it unhashable (which it is).
            3. If *eq* is False, ``__hash__`` will be left untouched meaning
               the ``__hash__`` method of the base class will be used. If the
               base class is `object`, this means it will fall back to id-based
               hashing.

            Although not recommended, you can decide for yourself and force
            *attrs* to create one (for example, if the class is immutable even
            though you didn't freeze it programmatically) by passing True or
            not.  Both of these cases are rather special and should be used
            carefully.

            .. seealso::

                - Our documentation on `hashing`,
                - Python's documentation on `object.__hash__`,
                - and the `GitHub issue that led to the default \ behavior
                  <https://github.com/python-attrs/attrs/issues/136>`_ for more
                  details.

        hash (bool | None):
            Deprecated alias for *unsafe_hash*. *unsafe_hash* takes precedence.

        cache_hash (bool):
            Ensure that the object's hash code is computed only once and stored
            on the object.  If this is set to True, hashing must be either
            explicitly or implicitly enabled for this class.  If the hash code
            is cached, avoid any reassignments of fields involved in hash code
            computation or mutations of the objects those fields point to after
            object creation.  If such changes occur, the behavior of the
            object's hash code is undefined.

        frozen (bool):
            Make instances immutable after initialization.  If someone attempts
            to modify a frozen instance, `attrs.exceptions.FrozenInstanceError`
            is raised.

            .. note::

                1. This is achieved by installing a custom ``__setattr__``
                   method on your class, so you can't implement your own.

                2. True immutability is impossible in Python.

                3. This *does* have a minor a runtime performance `impact
                   <how-frozen>` when initializing new instances.  In other
                   words: ``__init__`` is slightly slower with ``frozen=True``.

                4. If a class is frozen, you cannot modify ``self`` in
                   ``__attrs_post_init__`` or a self-written ``__init__``. You
                   can circumvent that limitation by using
                   ``object.__setattr__(self, "attribute_name", value)``.

                5. Subclasses of a frozen class are frozen too.

        kw_only (bool):
            Make attributes keyword-only in the generated ``__init__`` (if
            *init* is False, this parameter is ignored).  Attributes that
            explicitly set ``kw_only=False`` are not affected; base class
            attributes are also not affected.

            Also see *force_kw_only*.

        weakref_slot (bool):
            Make instances weak-referenceable.  This has no effect unless
            *slots* is True.

        field_transformer (~typing.Callable | None):
            A function that is called with the original class object and all
            fields right before *attrs* finalizes the class.  You can use this,
            for example, to automatically add converters or validators to
            fields based on their types.

            .. seealso:: `transform-fields`

        match_args (bool):
            If True (default), set ``__match_args__`` on the class to support
            :pep:`634` (*Structural Pattern Matching*). It is a tuple of all
            non-keyword-only ``__init__`` parameter names on Python 3.10 and
            later. Ignored on older Python versions.

        collect_by_mro (bool):
            If True, *attrs* collects attributes from base classes correctly
            according to the `method resolution order
            <https://docs.python.org/3/howto/mro.html>`_. If False, *attrs*
            will mimic the (wrong) behavior of `dataclasses` and :pep:`681`.

            See also `issue #428
            <https://github.com/python-attrs/attrs/issues/428>`_.

        force_kw_only (bool):
            A back-compat flag for restoring pre-25.4.0 behavior.  If True and
            ``kw_only=True``, all attributes are made keyword-only, including
            base class attributes, and those set to ``kw_only=False`` at the
            attribute level.  Defaults to False.

            See also `issue #980
            <https://github.com/python-attrs/attrs/issues/980>`_.

        getstate_setstate (bool | None):
            .. note::

                This is usually only interesting for slotted classes and you
                should probably just set *auto_detect* to True.

            If True, ``__getstate__`` and ``__setstate__`` are generated and
            attached to the class. This is necessary for slotted classes to be
            pickleable. If left None, it's True by default for slotted classes
            and False for dict classes.

            If *auto_detect* is True, and *getstate_setstate* is left None, and
            **either** ``__getstate__`` or ``__setstate__`` is detected
            directly on the class (meaning: not inherited), it is set to False
            (this is usually what you want).

        auto_attribs (bool | None):
            If True, look at type annotations to determine which attributes to
            use, like `dataclasses`. If False, it will only look for explicit
            :func:`field` class attributes, like classic *attrs*.

            If left None, it will guess:

            1. If any attributes are annotated and no unannotated
               `attrs.field`\ s are found, it assumes *auto_attribs=True*.
            2. Otherwise it assumes *auto_attribs=False* and tries to collect
               `attrs.field`\ s.

            If *attrs* decides to look at type annotations, **all** fields
            **must** be annotated. If *attrs* encounters a field that is set to
            a :func:`field` / `attr.ib` but lacks a type annotation, an
            `attrs.exceptions.UnannotatedAttributeError` is raised.  Use
            ``field_name: typing.Any = field(...)`` if you don't want to set a
            type.

            .. warning::

                For features that use the attribute name to create decorators
                (for example, :ref:`validators <validators>`), you still *must*
                assign :func:`field` / `attr.ib` to them. Otherwise Python will
                either not find the name or try to use the default value to
                call, for example, ``validator`` on it.

            Attributes annotated as `typing.ClassVar`, and attributes that are
            neither annotated nor set to an `field()` are **ignored**.

        these (dict[str, object]):
            A dictionary of name to the (private) return value of `field()`
            mappings. This is useful to avoid the definition of your attributes
            within the class body because you can't (for example, if you want
            to add ``__repr__`` methods to Django models) or don't want to.

            If *these* is not `None`, *attrs* will *not* search the class body
            for attributes and will *not* remove any attributes from it.

            The order is deduced from the order of the attributes inside
            *these*.

            Arguably, this is a rather obscure feature.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.3.0 Converters are also run ``on_setattr``.
    .. versionadded:: 22.2.0
       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
    .. versionchanged:: 24.1.0
       Instances are not compared as tuples of attributes anymore, but using a
       big ``and`` condition. This is faster and has more correct behavior for
       uncomparable values like `math.nan`.
    .. versionadded:: 24.1.0
       If a class has an *inherited* classmethod called
       ``__attrs_init_subclass__``, it is executed after the class is created.
    .. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
    .. versionadded:: 24.3.0
       Unless already present, a ``__replace__`` method is automatically
       created for `copy.replace` (Python 3.13+ only).
    .. versionchanged:: 25.4.0
       *kw_only* now only applies to attributes defined in the current class,
       and respects attribute-level ``kw_only=False`` settings.
    .. versionadded:: 25.4.0
       Added *force_kw_only* to go back to the previous *kw_only* behavior.

    .. note::

        The main differences to the classic `attr.s` are:

        - Automatically detect whether or not *auto_attribs* should be `True`
          (c.f. *auto_attribs* parameter).
        - Converters and validators run when attributes are set by default --
          if *frozen* is `False`.
        - *slots=True*

          Usually, this has only upsides and few visible effects in everyday
          programming. But it *can* lead to some surprising behaviors, so
          please make sure to read :term:`slotted classes`.

        - *auto_exc=True*
        - *auto_detect=True*
        - *order=False*
        - *force_kw_only=False*
        - Some options that were only relevant on Python 2 or were kept around
          for backwards-compatibility have been removed.

    """

    def do_it(cls, auto_attribs):
        return attrs(
            maybe_cls=cls,
            these=these,
            repr=repr,
            hash=hash,
            unsafe_hash=unsafe_hash,
            init=init,
            slots=slots,
            frozen=frozen,
            weakref_slot=weakref_slot,
            str=str,
            auto_attribs=auto_attribs,
            kw_only=kw_only,
            cache_hash=cache_hash,
            auto_exc=auto_exc,
            eq=eq,
            order=order,
            auto_detect=auto_detect,
            collect_by_mro=True,
            getstate_setstate=getstate_setstate,
            on_setattr=on_setattr,
            field_transformer=field_transformer,
            match_args=match_args,
            force_kw_only=force_kw_only,
        )

    def wrap(cls):
        """
        Making this a wrapper ensures this code runs during class creation.

        We also ensure that frozen-ness of classes is inherited.
        """
        nonlocal frozen, on_setattr

        had_on_setattr = on_setattr not in (None, setters.NO_OP)

        # By default, mutable classes convert & validate on setattr.
        if frozen is False and on_setattr is None:
            on_setattr = _DEFAULT_ON_SETATTR

        # However, if we subclass a frozen class, we inherit the immutability
        # and disable on_setattr.
        for base_cls in cls.__bases__:
            if base_cls.__setattr__ is _frozen_setattrs:
                if had_on_setattr:
                    msg = "Frozen classes can't use on_setattr (frozen-ness was inherited)."
                    raise ValueError(msg)

                on_setattr = setters.NO_OP
                break

        if auto_attribs is not None:
            return do_it(cls, auto_attribs)

        try:
            return do_it(cls, True)
        except UnannotatedAttributeError:
            return do_it(cls, False)

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


mutable = define
frozen = partial(define, frozen=True, on_setattr=None)


def field(
    *,
    default=NOTHING,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=None,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
):
    """
    Create a new :term:`field` / :term:`attribute` on a class.

    ..  warning::

        Does **nothing** unless the class is also decorated with
        `attrs.define` (or similar)!

    Args:
        default:
            A value that is used if an *attrs*-generated ``__init__`` is used
            and no value is passed while instantiating or the attribute is
            excluded using ``init=False``.

            If the value is an instance of `attrs.Factory`, its callable will
            be used to construct a new value (useful for mutable data types
            like lists or dicts).

            If a default is not set (or set manually to `attrs.NOTHING`), a
            value *must* be supplied when instantiating; otherwise a
            `TypeError` will be raised.

            .. seealso:: `defaults`

        factory (~typing.Callable):
            Syntactic sugar for ``default=attr.Factory(factory)``.

        validator (~typing.Callable | list[~typing.Callable]):
            Callable that is called by *attrs*-generated ``__init__`` methods
            after the instance has been initialized.  They receive the
            initialized instance, the :func:`~attrs.Attribute`, and the passed
            value.

            The return value is *not* inspected so the validator has to throw
            an exception itself.

            If a `list` is passed, its items are treated as validators and must
            all pass.

            Validators can be globally disabled and re-enabled using
            `attrs.validators.get_disabled` / `attrs.validators.set_disabled`.

            The validator can also be set using decorator notation as shown
            below.

            .. seealso:: :ref:`validators`

        repr (bool | ~typing.Callable):
            Include this attribute in the generated ``__repr__`` method. If
            True, include the attribute; if False, omit it. By default, the
            built-in ``repr()`` function is used. To override how the attribute
            value is formatted, pass a ``callable`` that takes a single value
            and returns a string. Note that the resulting string is used as-is,
            which means it will be used directly *instead* of calling
            ``repr()`` (the default).

        eq (bool | ~typing.Callable):
            If True (default), include this attribute in the generated
            ``__eq__`` and ``__ne__`` methods that check two instances for
            equality. To override how the attribute value is compared, pass a
            callable that takes a single value and returns the value to be
            compared.

            .. seealso:: `comparison`

        order (bool | ~typing.Callable):
            If True (default), include this attributes in the generated
            ``__lt__``, ``__le__``, ``__gt__`` and ``__ge__`` methods. To
            override how the attribute value is ordered, pass a callable that
            takes a single value and returns the value to be ordered.

            .. seealso:: `comparison`

        hash (bool | None):
            Include this attribute in the generated ``__hash__`` method.  If
            None (default), mirror *eq*'s value.  This is the correct behavior
            according the Python spec.  Setting this value to anything else
            than None is *discouraged*.

            .. seealso:: `hashing`

        init (bool):
            Include this attribute in the generated ``__init__`` method.

            It is possible to set this to False and set a default value. In
            that case this attributed is unconditionally initialized with the
            specified default value or factory.

            .. seealso:: `init`

        converter (typing.Callable | Converter):
            A callable that is called by *attrs*-generated ``__init__`` methods
            to convert attribute's value to the desired format.

            If a vanilla callable is passed, it is given the passed-in value as
            the only positional argument. It is possible to receive additional
            arguments by wrapping the callable in a `Converter`.

            Either way, the returned value will be used as the new value of the
            attribute.  The value is converted before being passed to the
            validator, if any.

            .. seealso:: :ref:`converters`

        metadata (dict | None):
            An arbitrary mapping, to be used by third-party code.

            .. seealso:: `extending-metadata`.

        type (type):
            The type of the attribute. Nowadays, the preferred method to
            specify the type is using a variable annotation (see :pep:`526`).
            This argument is provided for backwards-compatibility and for usage
            with `make_class`. Regardless of the approach used, the type will
            be stored on ``Attribute.type``.

            Please note that *attrs* doesn't do anything with this metadata by
            itself. You can use it as part of your own code or for `static type
            checking <types>`.

        kw_only (bool | None):
            Make this attribute keyword-only in the generated ``__init__`` (if
            *init* is False, this parameter is ignored).  If None (default),
            mirror the setting from `attrs.define`.

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            Allows to overwrite the *on_setattr* setting from `attr.s`. If left
            None, the *on_setattr* value from `attr.s` is used. Set to
            `attrs.setters.NO_OP` to run **no** `setattr` hooks for this
            attribute -- regardless of the setting in `define()`.

        alias (str | None):
            Override this attribute's parameter name in the generated
            ``__init__`` method. If left None, default to ``name`` stripped
            of leading underscores. See `private-attributes`.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0
       *eq*, *order*, and *cmp* also accept a custom callable
    .. versionadded:: 22.2.0 *alias*
    .. versionadded:: 23.1.0
       The *type* parameter has been re-added; mostly for `attrs.make_class`.
       Please note that type checkers ignore this metadata.
    .. versionchanged:: 25.4.0
       *kw_only* can now be None, and its default is also changed from False to
       None.

    .. seealso::

       `attr.ib`
    """
    return attrib(
        default=default,
        validator=validator,
        repr=repr,
        hash=hash,
        init=init,
        metadata=metadata,
        type=type,
        converter=converter,
        factory=factory,
        kw_only=kw_only,
        eq=eq,
        order=order,
        on_setattr=on_setattr,
        alias=alias,
    )


def asdict(inst, *, recurse=True, filter=None, value_serializer=None):
    """
    Same as `attr.asdict`, except that collections types are always retained
    and dict is always used as *dict_factory*.

    .. versionadded:: 21.3.0
    """
    return _asdict(
        inst=inst,
        recurse=recurse,
        filter=filter,
        value_serializer=value_serializer,
        retain_collection_types=True,
    )


def astuple(inst, *, recurse=True, filter=None):
    """
    Same as `attr.astuple`, except that collections types are always retained
    and `tuple` is always used as the *tuple_factory*.

    .. versionadded:: 21.3.0
    """
    return _astuple(
        inst=inst, recurse=recurse, filter=filter, retain_collection_types=True
    )


def inspect(cls):
    """
    Inspect the class and return its effective build parameters.

    Warning:
        This feature is currently **experimental** and is not covered by our
        strict backwards-compatibility guarantees.

    Args:
        cls: The *attrs*-decorated class to inspect.

    Returns:
        The effective build parameters of the class.

    Raises:
        NotAnAttrsClassError: If the class is not an *attrs*-decorated class.

    .. versionadded:: 25.4.0
    """
    try:
        return cls.__dict__["__attrs_props__"]
    except KeyError:
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg) from None
<|endoftext|># SPDX-License-Identifier: MIT

__all__ = ["get_run_validators", "set_run_validators"]

_run_validators = True


def set_run_validators(run):
    """
    Set whether or not validators are run.  By default, they are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.set_disabled()`
        instead.
    """
    if not isinstance(run, bool):
        msg = "'run' must be bool."
        raise TypeError(msg)
    global _run_validators
    _run_validators = run


def get_run_validators():
    """
    Return whether or not validators are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.get_disabled()`
        instead.
    """
    return _run_validators
<|endoftext|># SPDX-License-Identifier: MIT

from __future__ import annotations


class FrozenError(AttributeError):
    """
    A frozen/immutable instance or attribute have been attempted to be
    modified.

    It mirrors the behavior of ``namedtuples`` by using the same error message
    and subclassing `AttributeError`.

    .. versionadded:: 20.1.0
    """

    def __init__(self):
        msg = "can't set attribute"
        super().__init__(msg)
        self.msg = msg


class FrozenInstanceError(FrozenError):
    """
    A frozen instance has been attempted to be modified.

    .. versionadded:: 16.1.0
    """


class FrozenAttributeError(FrozenError):
    """
    A frozen attribute has been attempted to be modified.

    .. versionadded:: 20.1.0
    """


class AttrsAttributeNotFoundError(ValueError):
    """
    An *attrs* function couldn't find an attribute that the user asked for.

    .. versionadded:: 16.2.0
    """


class NotAnAttrsClassError(ValueError):
    """
    A non-*attrs* class has been passed into an *attrs* function.

    .. versionadded:: 16.2.0
    """


class DefaultAlreadySetError(RuntimeError):
    """
    A default has been set when defining the field and is attempted to be reset
    using the decorator.

    .. versionadded:: 17.1.0
    """


class UnannotatedAttributeError(RuntimeError):
    """
    A class with ``auto_attribs=True`` has a field without a type annotation.

    .. versionadded:: 17.3.0
    """


class PythonTooOldError(RuntimeError):
    """
    It was attempted to use an *attrs* feature that requires a newer Python
    version.

    .. versionadded:: 18.2.0
    """


class NotCallableError(TypeError):
    """
    A field requiring a callable has been set with a value that is not
    callable.

    .. versionadded:: 19.2.0
    """

    def __init__(self, msg, value):
        super(TypeError, self).__init__(msg, value)
        self.msg = msg
        self.value = value

    def __str__(self):
        return str(self.msg)
<|endoftext|>from __future__ import annotations

import codecs
import contextlib
import datetime
import email.message
import json as jsonlib
import re
import typing
import urllib.request
from collections.abc import AsyncGenerator, Mapping
from http.cookiejar import Cookie, CookieJar

from ._content import ByteStream, UnattachedStream, encode_request, encode_response
from ._decoders import (
    ByteChunker,
    ContentDecoder,
    IdentityDecoder,
    LineDecoder,
    MultiDecoder,
    TextChunker,
    TextDecoder,
)
from ._exceptions import (
    CookieConflict,
    HTTPStatusError,
    RequestNotRead,
    ResponseNotRead,
    StreamClosed,
    StreamConsumed,
    request_context,
)
from ._multipart import get_multipart_boundary_from_content_type
from ._status_codes import codes
from ._types import (
    AsyncByteStream,
    CookieTypes,
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    ResponseContent,
    ResponseExtensions,
    SyncByteStream,
)
from ._urls import URL
from ._utils import to_bytes_or_str, to_str

__all__ = ["Cookies", "Headers", "Request", "Response"]

SENSITIVE_HEADERS = {"authorization", "proxy-authorization"}

_T = typing.TypeVar("_T")


def _is_known_encoding(encoding: str) -> bool:
    """
    Return `True` if `encoding` is a known codec.
    """
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True


def _normalize_header_key(key: str | bytes, encoding: str | None = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header key.
    """
    return key if isinstance(key, bytes) else key.encode(encoding or "ascii")


def _normalize_header_value(value: str | bytes, encoding: str | None = None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header value.
    """
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Header value must be str or bytes, not {type(value)}")
    return value.encode(encoding or "ascii")


def _parse_content_type_charset(content_type: str) -> str | None:
    # We used to use `cgi.parse_header()` here, but `cgi` became a dead battery.
    # See: https://peps.python.org/pep-0594/#cgi
    msg = email.message.Message()
    msg["content-type"] = content_type
    return msg.get_content_charset(failobj=None)


def _parse_header_links(value: str) -> list[dict[str, str]]:
    """
    Returns a list of parsed link headers, for more info see:
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Link
    The generic syntax of those is:
    Link: < uri-reference >; param1=value1; param2="value2"
    So for instance:
    Link; '<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;'
    would return
        [
            {"url": "http:/.../front.jpeg", "type": "image/jpeg"},
            {"url": "http://.../back.jpeg"},
        ]
    :param value: HTTP Link entity-header field
    :return: list of parsed link headers
    """
    links: list[dict[str, str]] = []
    replace_chars = " '\""
    value = value.strip(replace_chars)
    if not value:
        return links
    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""
        link = {"url": url.strip("<> '\"")}
        for param in params.split(";"):
            try:
                key, value = param.split("=")
            except ValueError:
                break
            link[key.strip(replace_chars)] = value.strip(replace_chars)
        links.append(link)
    return links


def _obfuscate_sensitive_headers(
    items: typing.Iterable[tuple[typing.AnyStr, typing.AnyStr]],
) -> typing.Iterator[tuple[typing.AnyStr, typing.AnyStr]]:
    for k, v in items:
        if to_str(k.lower()) in SENSITIVE_HEADERS:
            v = to_bytes_or_str("[secure]", match_type_of=v)
        yield k, v


class Headers(typing.MutableMapping[str, str]):
    """
    HTTP headers, as a case-insensitive multi-dict.
    """

    def __init__(
        self,
        headers: HeaderTypes | None = None,
        encoding: str | None = None,
    ) -> None:
        self._list: list[tuple[bytes, bytes, bytes]] = []

        if isinstance(headers, Headers):
            self._list = list(headers._list)
        elif isinstance(headers, Mapping):
            for k, v in headers.items():
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))
        elif headers is not None:
            for k, v in headers:
                bytes_key = _normalize_header_key(k, encoding)
                bytes_value = _normalize_header_value(v, encoding)
                self._list.append((bytes_key, bytes_key.lower(), bytes_value))

        self._encoding = encoding

    @property
    def encoding(self) -> str:
        """
        Header encoding is mandated as ascii, but we allow fallbacks to utf-8
        or iso-8859-1.
        """
        if self._encoding is None:
            for encoding in ["ascii", "utf-8"]:
                for key, value in self.raw:
                    try:
                        key.decode(encoding)
                        value.decode(encoding)
                    except UnicodeDecodeError:
                        break
                else:
                    # The else block runs if 'break' did not occur, meaning
                    # all values fitted the encoding.
                    self._encoding = encoding
                    break
            else:
                # The ISO-8859-1 encoding covers all 256 code points in a byte,
                # so will never raise decode errors.
                self._encoding = "iso-8859-1"
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        self._encoding = value

    @property
    def raw(self) -> list[tuple[bytes, bytes]]:
        """
        Returns a list of the raw header items, as byte pairs.
        """
        return [(raw_key, value) for raw_key, _, value in self._list]

    def keys(self) -> typing.KeysView[str]:
        return {key.decode(self.encoding): None for _, key, _value in self._list}.keys()

    def values(self) -> typing.ValuesView[str]:
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f", {str_value}"
            else:
                values_dict[str_key] = str_value
        return values_dict.values()

    def items(self) -> typing.ItemsView[str, str]:
        """
        Return `(key, value)` items of headers. Concatenate headers
        into a single comma separated value when a key occurs multiple times.
        """
        values_dict: dict[str, str] = {}
        for _, key, value in self._list:
            str_key = key.decode(self.encoding)
            str_value = value.decode(self.encoding)
            if str_key in values_dict:
                values_dict[str_key] += f", {str_value}"
            else:
                values_dict[str_key] = str_value
        return values_dict.items()

    def multi_items(self) -> list[tuple[str, str]]:
        """
        Return a list of `(key, value)` pairs of headers. Allow multiple
        occurrences of the same key without concatenating into a single
        comma separated value.
        """
        return [(key.decode(self.encoding), value.decode(self.encoding)) for _, key, value in self._list]

    @typing.overload
    def get(self, key: str, /) -> str | None: ...

    @typing.overload
    def get(self, key: str, default: _T) -> str | _T: ...

    def get(self, key: str, default: _T | None = None) -> str | _T | None:
        """
        Return a header value. If multiple occurrences of the header occur
        then concatenate them together with commas.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def get_list(self, key: str, split_commas: bool = False) -> list[str]:
        """
        Return a list of all header values for a given key.
        If `split_commas=True` is passed, then any comma separated header
        values are split into multiple return strings.
        """
        get_header_key = key.lower().encode(self.encoding)

        values = [
            item_value.decode(self.encoding)
            for _, item_key, item_value in self._list
            if item_key.lower() == get_header_key
        ]

        if not split_commas:
            return values

        split_values: list[str] = []
        for value in values:
            split_values.extend([item.strip() for item in value.split(",")])
        return split_values

    def update(self, headers: HeaderTypes | None = None) -> None:  # type: ignore
        headers = Headers(headers)
        for key in headers.keys():
            if key in self:
                self.pop(key)
        self._list.extend(headers._list)

    def copy(self) -> Headers:
        return Headers(self, encoding=self.encoding)

    def __or__(self, other: Mapping[str, str]) -> Headers:
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = self.copy()
        merged.update(other)
        return merged

    def __ror__(self, other: Mapping[str, str]) -> Headers:
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = Headers(other)
        merged.update(self)
        return merged

    def __ior__(self, other: HeaderTypes) -> Headers:
        self.update(other)
        return self

    def __getitem__(self, key: str) -> str:
        """
        Return a single header value.

        If there are multiple headers with the same key, then we concatenate
        them with commas. See: https://tools.ietf.org/html/rfc7230#section-3.2.2
        """
        normalized_key = key.lower().encode(self.encoding)

        items = [
            header_value.decode(self.encoding)
            for _, header_key, header_value in self._list
            if header_key == normalized_key
        ]

        if items:
            return ", ".join(items)

        raise KeyError(key)

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header `key` to `value`, removing any duplicate entries.
        Retains insertion order.
        """
        set_key = key.encode(self._encoding or "utf-8")
        set_value = value.encode(self._encoding or "utf-8")
        lookup_key = set_key.lower()

        found_indexes = [idx for idx, (_, item_key, _) in enumerate(self._list) if item_key == lookup_key]

        for idx in reversed(found_indexes[1:]):
            del self._list[idx]

        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, lookup_key, set_value)
        else:
            self._list.append((set_key, lookup_key, set_value))

    def __delitem__(self, key: str) -> None:
        """
        Remove the header `key`.
        """
        del_key = key.lower().encode(self.encoding)

        pop_indexes = [idx for idx, (_, item_key, _) in enumerate(self._list) if item_key.lower() == del_key]

        if not pop_indexes:
            raise KeyError(key)

        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __contains__(self, key: typing.Any) -> bool:
        header_key = key.lower().encode(self.encoding)
        return header_key in [key for _, key, _ in self._list]

    def __iter__(self) -> typing.Iterator[typing.Any]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: typing.Any) -> bool:
        try:
            other_headers = Headers(other)
        except ValueError:
            return False

        self_list = [(key, value) for _, key, value in self._list]
        other_list = [(key, value) for _, key, value in other_headers._list]
        return sorted(self_list) == sorted(other_list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        encoding_str = ""
        if self.encoding != "ascii":
            encoding_str = f", encoding={self.encoding!r}"

        as_list = list(_obfuscate_sensitive_headers(self.multi_items()))
        as_dict = dict(as_list)

        no_duplicate_keys = len(as_dict) == len(as_list)
        if no_duplicate_keys:
            return f"{class_name}({as_dict!r}{encoding_str})"
        return f"{class_name}({as_list!r}{encoding_str})"


class Request:
    def __init__(
        self,
        method: str,
        url: URL | str,
        *,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: typing.Any | None = None,
        stream: SyncByteStream | AsyncByteStream | None = None,
        extensions: RequestExtensions | None = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url) if params is None else URL(url, params=params)
        self.headers = Headers(headers)
        self.extensions = {} if extensions is None else dict(extensions)

        if cookies:
            Cookies(cookies).set_cookie_header(self)

        if stream is None:
            content_type: str | None = self.headers.get("content-type")
            headers, stream = encode_request(
                content=content,
                data=data,
                files=files,
                json=json,
                boundary=get_multipart_boundary_from_content_type(
                    content_type=content_type.encode(self.headers.encoding) if content_type else None
                ),
            )
            self._prepare(headers)
            self.stream = stream
            # Load the request body, except for streaming content.
            if isinstance(stream, ByteStream):
                self.read()
        else:
            # There's an important distinction between `Request(content=...)`,
            # and `Request(stream=...)`.
            #
            # Using `content=...` implies automatically populated `Host` and content
            # headers, of either `Content-Length: ...` or `Transfer-Encoding: chunked`.
            #
            # Using `stream=...` will not automatically include *any*
            # auto-populated headers.
            #
            # As an end-user you don't really need `stream=...`. It's only
            # useful when:
            #
            # * Preserving the request stream when copying requests, eg for redirects.
            # * Creating request instances on the *server-side* of the transport API.
            self.stream = stream

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            # Ignore Content-Length if Transfer-Encoding has been set explicitly.
            if key.lower() == "content-length" and "Transfer-Encoding" in self.headers:
                continue
            # Ignore Transfer-Encoding if Content-Length has been set explicitly.
            if key.lower() == "transfer-encoding" and "Content-Length" in self.headers:
                continue
            self.headers.setdefault(key, value)

        auto_headers: list[tuple[bytes, bytes]] = []

        has_host = "Host" in self.headers
        has_content_length = "Content-Length" in self.headers or "Transfer-Encoding" in self.headers

        if not has_host and self.url.host:
            auto_headers.append((b"Host", self.url.netloc))
        if not has_content_length and self.method in ("POST", "PUT", "PATCH", "QUERY"):
            auto_headers.append((b"Content-Length", b"0"))

        self.headers = Headers(auto_headers + self.headers.raw)

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise RequestNotRead()
        return self._content

    def read(self) -> bytes:
        """
        Read and return the request content.
        """
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.Iterable)
            self._content = b"".join(self.stream)
            if not isinstance(self.stream, ByteStream):
                # If a streaming request has been read entirely into memory, then
                # we can replace the stream with a raw bytes implementation,
                # to ensure that any non-replayable streams can still be used.
                self.stream = ByteStream(self._content)
        return self._content

    async def aread(self) -> bytes:
        """
        Read and return the request content.
        """
        if not hasattr(self, "_content"):
            assert isinstance(self.stream, typing.AsyncIterable)
            self._content = b"".join([part async for part in self.stream])
            if not isinstance(self.stream, ByteStream):
                # If a streaming request has been read entirely into memory, then
                # we can replace the stream with a raw bytes implementation,
                # to ensure that any non-replayable streams can still be used.
                self.stream = ByteStream(self._content)
        return self._content

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        url = str(self.url)
        return f"<{class_name}({self.method!r}, {url!r})>"

    def __getstate__(self) -> dict[str, typing.Any]:
        return {name: value for name, value in self.__dict__.items() if name not in ["extensions", "stream"]}

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.extensions = {}
        self.stream = UnattachedStream()


class Response:
    def __init__(
        self,
        status_code: int,
        *,
        headers: HeaderTypes | None = None,
        content: ResponseContent | None = None,
        text: str | None = None,
        html: str | None = None,
        json: typing.Any = None,
        stream: SyncByteStream | AsyncByteStream | None = None,
        request: Request | None = None,
        extensions: ResponseExtensions | None = None,
        history: list[Response] | None = None,
        default_encoding: str | typing.Callable[[bytes], str | None] = "utf-8",
    ) -> None:
        self.status_code = status_code
        self.headers = Headers(headers)

        self._request: Request | None = request

        # When follow_redirects=False and a redirect is received,
        # the client will set `response.next_request`.
        self.next_request: Request | None = None

        self.extensions = {} if extensions is None else dict(extensions)
        self.history = [] if history is None else list(history)

        self.is_closed = False
        self.is_stream_consumed = False

        self.default_encoding = default_encoding

        if stream is None:
            headers, stream = encode_response(content, text, html, json)
            self._prepare(headers)
            self.stream = stream
            if isinstance(stream, ByteStream):
                # Load the response body, except for streaming content.
                self.read()
        else:
            # There's an important distinction between `Response(content=...)`,
            # and `Response(stream=...)`.
            #
            # Using `content=...` implies automatically populated content headers,
            # of either `Content-Length: ...` or `Transfer-Encoding: chunked`.
            #
            # Using `stream=...` will not automatically include any content headers.
            #
            # As an end-user you don't really need `stream=...`. It's only
            # useful when creating response instances having received a stream
            # from the transport API.
            self.stream = stream

        self._num_bytes_downloaded = 0

    def _prepare(self, default_headers: dict[str, str]) -> None:
        for key, value in default_headers.items():
            # Ignore Transfer-Encoding if the Content-Length has been set explicitly.
            if key.lower() == "transfer-encoding" and "content-length" in self.headers:
                continue
            self.headers.setdefault(key, value)

    @property
    def elapsed(self) -> datetime.timedelta:
        """
        Returns the time taken for the complete request/response
        cycle to complete.
        """
        if not hasattr(self, "_elapsed"):
            stream_elapsed: datetime.timedelta | None = getattr(self.stream, "elapsed", None)
            if stream_elapsed is not None:
                return stream_elapsed
            raise RuntimeError("'.elapsed' may only be accessed after the response has been read or closed.")
        return self._elapsed

    @elapsed.setter
    def elapsed(self, elapsed: datetime.timedelta) -> None:
        self._elapsed = elapsed

    @property
    def request(self) -> Request:
        """
        Returns the request instance associated to the current response.
        """
        if self._request is None:
            raise RuntimeError("The request instance has not been set on this response.")
        return self._request

    @request.setter
    def request(self, value: Request) -> None:
        self._request = value

    @property
    def http_version(self) -> str:
        try:
            http_version: bytes = self.extensions["http_version"]
        except KeyError:
            return "HTTP/1.1"
        else:
            return http_version.decode("ascii", errors="ignore")

    @property
    def reason_phrase(self) -> str:
        try:
            reason_phrase: bytes = self.extensions["reason_phrase"]
        except KeyError:
            return codes.get_reason_phrase(self.status_code)
        else:
            return reason_phrase.decode("ascii", errors="ignore")

    @property
    def url(self) -> URL:
        """
        Returns the URL for which the request was made.
        """
        return self.request.url

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            raise ResponseNotRead()
        return self._content

    @property
    def text(self) -> str:
        if not hasattr(self, "_text"):
            content = self.content
            if not content:
                self._text = ""
            else:
                decoder = TextDecoder(encoding=self.encoding or "utf-8")
                self._text = "".join([decoder.decode(self.content), decoder.flush()])
        return self._text

    @property
    def encoding(self) -> str | None:
        """
        Return an encoding to use for decoding the byte content into text.
        The priority for determining this is given by...

        * `.encoding = <>` has been set explicitly.
        * The encoding as specified by the charset parameter in the Content-Type header.
        * The encoding as determined by `default_encoding`, which may either be
          a string like "utf-8" indicating the encoding to use, or may be a callable
          which enables charset autodetection.
        """
        if not hasattr(self, "_encoding"):
            encoding = self.charset_encoding
            if encoding is None or not _is_known_encoding(encoding):
                if isinstance(self.default_encoding, str):
                    encoding = self.default_encoding
                elif hasattr(self, "_content"):
                    encoding = self.default_encoding(self._content)
            self._encoding = encoding or "utf-8"
        return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        """
        Set the encoding to use for decoding the byte content into text.

        If the `text` attribute has been accessed, attempting to set the
        encoding will throw a ValueError.
        """
        if hasattr(self, "_text"):
            raise ValueError("Setting encoding after `text` has been accessed is not allowed.")
        self._encoding = value

    @property
    def charset_encoding(self) -> str | None:
        """
        Return the encoding, as specified by the Content-Type header.
        """
        content_type = self.headers.get("Content-Type")
        if content_type is None:
            return None

        return _parse_content_type_charset(content_type)

    def _get_content_decoder(self) -> ContentDecoder:
        """
        Returns a decoder instance which can be used to decode the raw byte
        content, depending on the Content-Encoding used in the response.
        """
        if not hasattr(self, "_decoder"):
            values = self.headers.get_list("content-encoding", split_commas=True)
            encodings = [value.strip().lower() for value in values]
            decoder = MultiDecoder([encoding for encoding in encodings if encoding != "identity"])
            if len(decoder.children) == 1:
                self._decoder = decoder.children[0]
            elif decoder.children:
                self._decoder = decoder
            else:
                self._decoder = IdentityDecoder()

        return self._decoder

    @property
    def is_informational(self) -> bool:
        """
        A property which is `True` for 1xx status codes, `False` otherwise.
        """
        return codes.is_informational(self.status_code)

    @property
    def is_success(self) -> bool:
        """
        A property which is `True` for 2xx status codes, `False` otherwise.
        """
        return codes.is_success(self.status_code)

    @property
    def is_redirect(self) -> bool:
        """
        A property which is `True` for 3xx status codes, `False` otherwise.

        Note that not all responses with a 3xx status code indicate a URL redirect.

        Use `response.has_redirect_location` to determine responses with a properly
        formed URL redirection.
        """
        return codes.is_redirect(self.status_code)

    @property
    def is_client_error(self) -> bool:
        """
        A property which is `True` for 4xx status codes, `False` otherwise.
        """
        return codes.is_client_error(self.status_code)

    @property
    def is_server_error(self) -> bool:
        """
        A property which is `True` for 5xx status codes, `False` otherwise.
        """
        return codes.is_server_error(self.status_code)

    @property
    def is_error(self) -> bool:
        """
        A property which is `True` for 4xx and 5xx status codes, `False` otherwise.
        """
        return codes.is_error(self.status_code)

    @property
    def has_redirect_location(self) -> bool:
        """
        Returns True for 3xx responses with a properly formed URL redirection,
        `False` otherwise.
        """
        return (
            self.status_code
            in (
                # 301 (Cacheable redirect. Method may change to GET.)
                codes.MOVED_PERMANENTLY,
                # 302 (Uncacheable redirect. Method may change to GET.)
                codes.FOUND,
                # 303 (Client should make a GET or HEAD request.)
                codes.SEE_OTHER,
                # 307 (Equiv. 302, but retain method)
                codes.TEMPORARY_REDIRECT,
                # 308 (Equiv. 301, but retain method)
                codes.PERMANENT_REDIRECT,
            )
            and "Location" in self.headers
        )

    def raise_for_status(self) -> Response:
        """
        Raise the `HTTPStatusError` if one occurred.
        """
        request = self._request
        if request is None:
            raise RuntimeError(
                "Cannot call `raise_for_status` as the request instance has not been set on this response."
            )

        if self.is_success:
            return self

        if self.has_redirect_location:
            message = (
                "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
                "Redirect location: '{0.headers[location]}'\n"
                "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
            )
        else:
            message = (
                "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
                "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{0.status_code}"
            )

        status_class = self.status_code // 100
        error_types = {
            1: "Informational response",
            3: "Redirect response",
            4: "Client error",
            5: "Server error",
        }
        error_type = error_types.get(status_class, "Invalid status code")
        message = message.format(self, error_type=error_type)
        raise HTTPStatusError(message, request=request, response=self)

    def json(self, **kwargs: typing.Any) -> typing.Any:
        return jsonlib.loads(self.content, **kwargs)

    @property
    def cookies(self) -> Cookies:
        if not hasattr(self, "_cookies"):
            self._cookies = Cookies()
            self._cookies.extract_cookies(self)
        return self._cookies

    @property
    def links(self) -> dict[str | None, dict[str, str]]:
        """
        Returns the parsed header links of the response, if any
        """
        header = self.headers.get("link")
        if header is None:
            return {}

        return {(link.get("rel") or link.get("url")): link for link in _parse_header_links(header)}

    @property
    def num_bytes_downloaded(self) -> int:
        return self._num_bytes_downloaded

    def __repr__(self) -> str:
        return f"<Response [{self.status_code} {self.reason_phrase}]>"

    def __getstate__(self) -> dict[str, typing.Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ["extensions", "stream", "is_closed", "_decoder"]
        }

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.is_closed = True
        self.extensions = {}
        self.stream = UnattachedStream()

    def read(self) -> bytes:
        """
        Read and return the response content.
        """
        if not hasattr(self, "_content"):
            self._content = b"".join(self.iter_bytes())
        return self._content

    def iter_bytes(self, chunk_size: int | None = None) -> typing.Iterator[bytes]:
        """
        A byte-iterator over the decoded response content.
        This allows us to handle gzip, deflate, brotli, and zstd encoded responses.
        """
        if hasattr(self, "_content"):
            chunk_size = len(self._content) if chunk_size is None else chunk_size
            for i in range(0, len(self._content), max(chunk_size, 1)):
                yield self._content[i : i + chunk_size]
        else:
            decoder = self._get_content_decoder()
            chunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                for raw_bytes in self.iter_raw():
                    for decoded in decoder.decode(raw_bytes):
                        for chunk in chunker.decode(decoded):
                            yield chunk
                for decoded in decoder.flush():
                    for chunk in chunker.decode(decoded):
                        yield chunk  # pragma: no cover
                for chunk in chunker.flush():
                    yield chunk

    def iter_text(self, chunk_size: int | None = None) -> typing.Iterator[str]:
        """
        A str-iterator over the decoded response content
        that handles both gzip, deflate, etc but also detects the content's
        string encoding.
        """
        decoder = TextDecoder(encoding=self.encoding or "utf-8")
        chunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            for byte_content in self.iter_bytes():
                text_content = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk  # pragma: no cover
            for chunk in chunker.flush():
                yield chunk

    def iter_lines(self) -> typing.Iterator[str]:
        decoder = LineDecoder()
        with request_context(request=self._request):
            for text in self.iter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    def iter_raw(self, chunk_size: int | None = None) -> typing.Iterator[bytes]:
        """
        A byte-iterator over the raw response content.
        """
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, SyncByteStream):
            raise RuntimeError("Attempted to call a sync iterator on an async stream.")

        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker = ByteChunker(chunk_size=chunk_size)

        try:
            with request_context(request=self._request):
                for raw_stream_bytes in self.stream:
                    self._num_bytes_downloaded += len(raw_stream_bytes)
                    for chunk in chunker.decode(raw_stream_bytes):
                        yield chunk

            for chunk in chunker.flush():
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        """
        Close the response and release the connection.
        Automatically called if the response body is read to completion.
        """
        if not isinstance(self.stream, SyncByteStream):
            raise RuntimeError("Attempted to call a sync close on an async stream.")

        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                self.stream.close()

    async def aread(self) -> bytes:
        """
        Read and return the response content.
        """
        if not hasattr(self, "_content"):
            async with contextlib.aclosing(self.aiter_bytes()) as parts:
                self._content = b"".join([part async for part in parts])
        return self._content

    async def aiter_bytes(self, chunk_size: int | None = None) -> typing.AsyncGenerator[bytes, None]:
        """
        A byte-iterator over the decoded response content.
        This allows us to handle gzip, deflate, brotli, and zstd encoded responses.
        """
        if hasattr(self, "_content"):
            chunk_size = len(self._content) if chunk_size is None else chunk_size
            for i in range(0, len(self._content), max(chunk_size, 1)):
                yield self._content[i : i + chunk_size]
        else:
            decoder = self._get_content_decoder()
            chunker = ByteChunker(chunk_size=chunk_size)
            with request_context(request=self._request):
                async with contextlib.aclosing(self.aiter_raw()) as raw_stream:
                    async for raw_bytes in raw_stream:
                        for decoded in decoder.decode(raw_bytes):
                            for chunk in chunker.decode(decoded):
                                yield chunk
                for decoded in decoder.flush():
                    for chunk in chunker.decode(decoded):
                        yield chunk  # pragma: no cover
                for chunk in chunker.flush():
                    yield chunk

    async def aiter_text(self, chunk_size: int | None = None) -> typing.AsyncIterator[str]:
        """
        A str-iterator over the decoded response content
        that handles both gzip, deflate, etc but also detects the content's
        string encoding.
        """
        decoder = TextDecoder(encoding=self.encoding or "utf-8")
        chunker = TextChunker(chunk_size=chunk_size)
        with request_context(request=self._request):
            async for byte_content in self.aiter_bytes():
                text_content = decoder.decode(byte_content)
                for chunk in chunker.decode(text_content):
                    yield chunk
            text_content = decoder.flush()
            for chunk in chunker.decode(text_content):
                yield chunk  # pragma: no cover
            for chunk in chunker.flush():
                yield chunk

    async def aiter_lines(self) -> typing.AsyncIterator[str]:
        decoder = LineDecoder()
        with request_context(request=self._request):
            async for text in self.aiter_text():
                for line in decoder.decode(text):
                    yield line
            for line in decoder.flush():
                yield line

    async def aiter_raw(self, chunk_size: int | None = None) -> typing.AsyncGenerator[bytes, None]:
        """
        A byte-iterator over the raw response content.
        """
        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async iterator on a sync stream.")

        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        chunker = ByteChunker(chunk_size=chunk_size)

        stream = self.stream.__aiter__()
        try:
            with request_context(request=self._request):
                async for raw_stream_bytes in stream:
                    self._num_bytes_downloaded += len(raw_stream_bytes)
                    for chunk in chunker.decode(raw_stream_bytes):
                        yield chunk

            for chunk in chunker.flush():
                yield chunk
        finally:
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()
            await self.aclose()

    async def aclose(self) -> None:
        """
        Close the response and release the connection.
        Automatically called if the response body is read to completion.
        """
        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async close on a sync stream.")

        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                await self.stream.aclose()


class Cookies(typing.MutableMapping[str, str]):
    """
    HTTP Cookies, as a mutable mapping.
    """

    def __init__(self, cookies: CookieTypes | None = None) -> None:
        if cookies is None or isinstance(cookies, dict):
            self.jar = CookieJar()
            if isinstance(cookies, dict):
                for key, value in cookies.items():
                    self.set(key, value)
        elif isinstance(cookies, list):
            self.jar = CookieJar()
            for key, value in cookies:
                self.set(key, value)
        elif isinstance(cookies, Cookies):
            self.jar = CookieJar()
            for cookie in cookies.jar:
                self.jar.set_cookie(cookie)
        else:
            self.jar = cookies

    def extract_cookies(self, response: Response) -> None:
        """
        Loads any cookies based on the response `Set-Cookie` headers.
        """
        if "set-cookie" not in response.headers and "set-cookie2" not in response.headers:
            return

        urllib_response = self._CookieCompatResponse(response)
        urllib_request = self._CookieCompatRequest(response.request)

        self.jar.extract_cookies(urllib_response, urllib_request)  # type: ignore

    def set_cookie_header(self, request: Request) -> None:
        """
        Sets an appropriate 'Cookie:' HTTP header on the `Request`.
        """
        urllib_request = self._CookieCompatRequest(request)
        self.jar.add_cookie_header(urllib_request)

    def set(self, name: str, value: str, domain: str = "", path: str = "/") -> None:
        """
        Set a cookie value by name. May optionally include domain and path.
        """
        kwargs = {
            "version": 0,
            "name": name,
            "value": value,
            "port": None,
            "port_specified": False,
            "domain": domain,
            "domain_specified": bool(domain),
            "domain_initial_dot": domain.startswith("."),
            "path": path,
            "path_specified": bool(path),
            "secure": False,
            "expires": None,
            "discard": True,
            "comment": None,
            "comment_url": None,
            "rest": {"HttpOnly": None},
            "rfc2109": False,
        }
        cookie = Cookie(**kwargs)  # type: ignore
        self.jar.set_cookie(cookie)

    def get(  # type: ignore
        self,
        name: str,
        default: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> str | None:
        """
        Get a cookie by name. May optionally include domain and path
        in order to specify exactly which cookie to retrieve.
        """
        value = None
        for cookie in self.jar:
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if value is not None:
                            message = f"Multiple cookies exist with name={name}"
                            raise CookieConflict(message)
                        value = cookie.value

        if value is None:
            return default
        return value

    def delete(
        self,
        name: str,
        domain: str | None = None,
        path: str | None = None,
    ) -> None:
        """
        Delete a cookie by name. May optionally include domain and path
        in order to specify exactly which cookie to delete.
        """
        if domain is not None and path is not None:
            return self.jar.clear(domain, path, name)

        remove = [
            cookie
            for cookie in self.jar
            if cookie.name == name
            and (domain is None or cookie.domain == domain)
            and (path is None or cookie.path == path)
        ]

        for cookie in remove:
            self.jar.clear(cookie.domain, cookie.path, cookie.name)

    def clear(self, domain: str | None = None, path: str | None = None) -> None:
        """
        Delete all cookies. Optionally include a domain and path in
        order to only delete a subset of all the cookies.
        """
        args: list[str] = []
        if domain is not None:
            args.append(domain)
        if path is not None:
            assert domain is not None
            args.append(path)
        self.jar.clear(*args)

    def update(self, cookies: CookieTypes | None = None) -> None:  # type: ignore
        cookies = Cookies(cookies)
        for cookie in cookies.jar:
            self.jar.set_cookie(cookie)

    def __setitem__(self, name: str, value: str) -> None:
        return self.set(name, value)

    def __getitem__(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def __delitem__(self, name: str) -> None:
        return self.delete(name)

    def __len__(self) -> int:
        return len(self.jar)

    def __iter__(self) -> typing.Iterator[str]:
        return (cookie.name for cookie in self.jar)

    def __bool__(self) -> bool:
        for _ in self.jar:
            return True
        return False

    def __repr__(self) -> str:
        cookies_repr = ", ".join(
            [f"<Cookie {cookie.name}={cookie.value} for {cookie.domain} />" for cookie in self.jar]
        )

        return f"<Cookies[{cookies_repr}]>"

    class _CookieCompatRequest(urllib.request.Request):
        """
        Wraps a `Request` instance up in a compatibility interface suitable
        for use with `CookieJar` operations.
        """

        def __init__(self, request: Request) -> None:
            super().__init__(
                url=str(request.url),
                headers=dict(request.headers),
                method=request.method,
            )
            self.request = request

        def add_unredirected_header(self, key: str, val: str) -> None:
            super().add_unredirected_header(key, val)
            self.request.headers[key] = val

    class _CookieCompatResponse:
        """
        Wraps a `Request` instance up in a compatibility interface suitable
        for use with `CookieJar` operations.
        """

        def __init__(self, response: Response) -> None:
            self.response = response

        def info(self) -> email.message.Message:
            info = email.message.Message()
            for key, value in self.response.headers.multi_items():
                # Note that setting `info[key]` here is an "append" operation,
                # not a "replace" operation.
                # https://docs.python.org/3/library/email.compat32-message.html#email.message.Message.__setitem__
                info[key] = value
            return info
<|endoftext|>from __future__ import annotations

import functools
import json
import sys
import typing

import click
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table

from ._client import Client
from ._exceptions import RequestError
from ._models import Response
from ._status_codes import codes

if typing.TYPE_CHECKING:
    import httpcore2  # pragma: no cover


def print_help() -> None:
    console = rich.console.Console()

    console.print("[bold]HTTPX :butterfly:", justify="center")
    console.print()
    console.print("A next generation HTTP client.", justify="center")
    console.print()
    console.print("Usage: [bold]httpx2[/bold] [cyan]<URL> [OPTIONS][/cyan] ", justify="left")
    console.print()

    table = rich.table.Table.grid(padding=1, pad_edge=True)
    table.add_column("Parameter", no_wrap=True, justify="left", style="bold")
    table.add_column("Description")
    table.add_row(
        "-m, --method [cyan]METHOD",
        "Request method, such as GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD, QUERY.\n"
        "[Default: GET, or POST if a request body is included]",
    )
    table.add_row(
        "-p, --params [cyan]<NAME VALUE> ...",
        "Query parameters to include in the request URL.",
    )
    table.add_row("-c, --content [cyan]TEXT", "Byte content to include in the request body.")
    table.add_row("-d, --data [cyan]<NAME VALUE> ...", "Form data to include in the request body.")
    table.add_row(
        "-f, --files [cyan]<NAME FILENAME> ...",
        "Form files to include in the request body.",
    )
    table.add_row("-j, --json [cyan]TEXT", "JSON data to include in the request body.")
    table.add_row(
        "-h, --headers [cyan]<NAME VALUE> ...",
        "Include additional HTTP headers in the request.",
    )
    table.add_row("--cookies [cyan]<NAME VALUE> ...", "Cookies to include in the request.")
    table.add_row(
        "--auth [cyan]<USER PASS>",
        "Username and password to include in the request. Specify '-' for the password"
        " to use a password prompt. Note that using --verbose/-v will expose"
        " the Authorization header, including the password encoding"
        " in a trivially reversible format.",
    )

    table.add_row(
        "--proxy [cyan]URL",
        "Send the request via a proxy. Should be the URL giving the proxy address.",
    )

    table.add_row(
        "--timeout [cyan]FLOAT",
        "Timeout value to use for network operations, such as establishing the"
        " connection, reading some data, etc... [Default: 5.0]",
    )

    table.add_row("--follow-redirects", "Automatically follow redirects.")
    table.add_row("--verify / --no-verify", "Enable SSL verification.")
    table.add_row("--http2", "Send the request using HTTP/2, if the remote server supports it.")

    table.add_row(
        "--download [cyan]FILE",
        "Save the response content as a file, rather than displaying it.",
    )

    table.add_row("-v, --verbose", "Verbose output. Show request as well as response.")
    table.add_row("--help", "Show this message and exit.")
    console.print(table)


def get_lexer_for_response(response: Response) -> str:
    content_type = response.headers.get("Content-Type")
    if content_type is not None:
        mime_type, _, _ = content_type.partition(";")
        try:
            return typing.cast(str, pygments.lexers.get_lexer_for_mimetype(mime_type.strip()).name)
        except pygments.util.ClassNotFound:  # pragma: no cover
            pass
    return ""  # pragma: no cover


def format_request_headers(request: httpcore2.Request, http2: bool = False) -> str:
    version = "HTTP/2" if http2 else "HTTP/1.1"
    headers = [(name.lower() if http2 else name, value) for name, value in request.headers]
    method = request.method.decode("ascii")
    target = request.url.target.decode("ascii")
    lines = [f"{method} {target} {version}"] + [
        f"{name.decode('ascii')}: {value.decode('ascii')}" for name, value in headers
    ]
    return "\n".join(lines)


def format_response_headers(
    http_version: bytes,
    status: int,
    reason_phrase: bytes | None,
    headers: list[tuple[bytes, bytes]],
) -> str:
    version = http_version.decode("ascii")
    reason = codes.get_reason_phrase(status) if reason_phrase is None else reason_phrase.decode("ascii")
    lines = [f"{version} {status} {reason}"] + [
        f"{name.decode('ascii')}: {value.decode('ascii')}" for name, value in headers
    ]
    return "\n".join(lines)


def print_request_headers(request: httpcore2.Request, http2: bool = False) -> None:
    console = rich.console.Console()
    http_text = format_request_headers(request, http2=http2)
    syntax = rich.syntax.Syntax(http_text, "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax("", "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)


def print_response_headers(
    http_version: bytes,
    status: int,
    reason_phrase: bytes | None,
    headers: list[tuple[bytes, bytes]],
) -> None:
    console = rich.console.Console()
    http_text = format_response_headers(http_version, status, reason_phrase, headers)
    syntax = rich.syntax.Syntax(http_text, "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)
    syntax = rich.syntax.Syntax("", "http", theme="ansi_dark", word_wrap=True)
    console.print(syntax)


def print_response(response: Response) -> None:
    console = rich.console.Console()
    lexer_name = get_lexer_for_response(response)
    if lexer_name:
        if lexer_name.lower() == "json":
            try:
                data = response.json()
                text = json.dumps(data, indent=4)
            except ValueError:  # pragma: no cover
                text = response.text
        else:
            text = response.text

        syntax = rich.syntax.Syntax(text, lexer_name, theme="ansi_dark", word_wrap=True)
        console.print(syntax)
    else:
        console.print(f"<{len(response.content)} bytes of binary data>")


_PCTRTT = tuple[tuple[str, str], ...]
_PCTRTTT = tuple[_PCTRTT, ...]
_PeerCertRetDictType = dict[str, str | _PCTRTTT | _PCTRTT]


def format_certificate(cert: _PeerCertRetDictType) -> str:  # pragma: no cover
    lines: list[str] = []
    for key, value in cert.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"*   {key}:")
            for item in value:
                if key in ("subject", "issuer"):
                    lines.extend(f"*     {sub_item[0]}: {sub_item[1]!r}" for sub_item in item)
                elif isinstance(item, tuple) and len(item) == 2:
                    lines.append(f"*     {item[0]}: {item[1]!r}")
                else:
                    lines.append(f"*     {item!r}")
        else:
            lines.append(f"*   {key}: {value!r}")
    return "\n".join(lines)


def trace(name: str, info: typing.Mapping[str, typing.Any], verbose: bool = False) -> None:
    console = rich.console.Console()
    if name == "connection.connect_tcp.started" and verbose:
        host = info["host"]
        console.print(f"* Connecting to {host!r}")
    elif name == "connection.connect_tcp.complete" and verbose:
        stream = info["return_value"]
        server_addr = stream.get_extra_info("server_addr")
        console.print(f"* Connected to {server_addr[0]!r} on port {server_addr[1]}")
    elif name == "connection.start_tls.complete" and verbose:  # pragma: no cover
        stream = info["return_value"]
        ssl_object = stream.get_extra_info("ssl_object")
        version = ssl_object.version()
        cipher = ssl_object.cipher()
        server_cert = ssl_object.getpeercert()
        alpn = ssl_object.selected_alpn_protocol()
        console.print(f"* SSL established using {version!r} / {cipher[0]!r}")
        console.print(f"* Selected ALPN protocol: {alpn!r}")
        if server_cert:
            console.print("* Server certificate:")
            console.print(format_certificate(server_cert))
    elif name == "http11.send_request_headers.started" and verbose:
        request = info["request"]
        print_request_headers(request, http2=False)
    elif name == "http2.send_request_headers.started" and verbose:  # pragma: no cover
        request = info["request"]
        print_request_headers(request, http2=True)
    elif name == "http11.receive_response_headers.complete":
        http_version, status, reason_phrase, headers = info["return_value"]
        print_response_headers(http_version, status, reason_phrase, headers)
    elif name == "http2.receive_response_headers.complete":  # pragma: no cover
        status, headers = info["return_value"]
        http_version = b"HTTP/2"
        reason_phrase = None
        print_response_headers(http_version, status, reason_phrase, headers)


def download_response(response: Response, download: typing.BinaryIO) -> None:
    console = rich.console.Console()
    console.print()
    content_length = response.headers.get("Content-Length")
    with rich.progress.Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.BarColumn(bar_width=None),
        rich.progress.DownloadColumn(),
        rich.progress.TransferSpeedColumn(),
    ) as progress:
        description = f"Downloading [bold]{rich.markup.escape(download.name)}"
        download_task = progress.add_task(
            description,
            total=int(content_length or 0),
            start=content_length is not None,
        )
        for chunk in response.iter_bytes():
            download.write(chunk)
            progress.update(download_task, completed=response.num_bytes_downloaded)


def validate_json(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> typing.Any:
    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover
        raise click.BadParameter("Not valid JSON")


def validate_auth(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> typing.Any:
    if value == (None, None):
        return None

    username, password = value
    if password == "-":  # pragma: no cover
        password = click.prompt("Password", hide_input=True)
    return (username, password)


def handle_help(
    ctx: click.Context,
    param: click.Option | click.Parameter,
    value: typing.Any,
) -> None:
    if not value or ctx.resilient_parsing:
        return

    print_help()
    ctx.exit()


@click.command(add_help_option=False)
@click.argument("url", type=str)
@click.option(
    "--method",
    "-m",
    "method",
    type=str,
    help=(
        "Request method, such as GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD, QUERY. "
        "[Default: GET, or POST if a request body is included]"
    ),
)
@click.option(
    "--params",
    "-p",
    "params",
    type=(str, str),
    multiple=True,
    help="Query parameters to include in the request URL.",
)
@click.option(
    "--content",
    "-c",
    "content",
    type=str,
    help="Byte content to include in the request body.",
)
@click.option(
    "--data",
    "-d",
    "data",
    type=(str, str),
    multiple=True,
    help="Form data to include in the request body.",
)
@click.option(
    "--files",
    "-f",
    "files",
    type=(str, click.File(mode="rb")),
    multiple=True,
    help="Form files to include in the request body.",
)
@click.option(
    "--json",
    "-j",
    "json",
    type=str,
    callback=validate_json,
    help="JSON data to include in the request body.",
)
@click.option(
    "--headers",
    "-h",
    "headers",
    type=(str, str),
    multiple=True,
    help="Include additional HTTP headers in the request.",
)
@click.option(
    "--cookies",
    "cookies",
    type=(str, str),
    multiple=True,
    help="Cookies to include in the request.",
)
@click.option(
    "--auth",
    "auth",
    type=(str, str),
    default=(None, None),
    callback=validate_auth,
    help=(
        "Username and password to include in the request. "
        "Specify '-' for the password to use a password prompt. "
        "Note that using --verbose/-v will expose the Authorization header, "
        "including the password encoding in a trivially reversible format."
    ),
)
@click.option(
    "--proxy",
    "proxy",
    type=str,
    default=None,
    help="Send the request via a proxy. Should be the URL giving the proxy address.",
)
@click.option(
    "--timeout",
    "timeout",
    type=float,
    default=5.0,
    help=(
        "Timeout value to use for network operations, such as establishing the "
        "connection, reading some data, etc... [Default: 5.0]"
    ),
)
@click.option(
    "--follow-redirects",
    "follow_redirects",
    is_flag=True,
    default=False,
    help="Automatically follow redirects.",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Enable SSL verification.",
)
@click.option(
    "--http2",
    "http2",
    type=bool,
    is_flag=True,
    default=False,
    help="Send the request using HTTP/2, if the remote server supports it.",
)
@click.option(
    "--download",
    type=click.File("wb"),
    help="Save the response content as a file, rather than displaying it.",
)
@click.option(
    "--verbose",
    "-v",
    type=bool,
    is_flag=True,
    default=False,
    help="Verbose. Show request as well as response.",
)
@click.option(
    "--help",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=handle_help,
    help="Show this message and exit.",
)
def main(
    url: str,
    method: str,
    params: list[tuple[str, str]],
    content: str,
    data: list[tuple[str, str]],
    files: list[tuple[str, click.File]],
    json: str,
    headers: list[tuple[str, str]],
    cookies: list[tuple[str, str]],
    auth: tuple[str, str] | None,
    proxy: str,
    timeout: float,
    follow_redirects: bool,
    verify: bool,
    http2: bool,
    download: typing.BinaryIO | None,
    verbose: bool,
) -> None:
    """
    An HTTP command line client.
    Sends a request and displays the response.
    """
    if not method:
        method = "POST" if content or data or files or json else "GET"

    try:
        with Client(proxy=proxy, timeout=timeout, http2=http2, verify=verify) as client:
            with client.stream(
                method,
                url,
                params=list(params),
                content=content,
                data=dict(data),
                files=files,  # type: ignore[arg-type]
                json=json,
                headers=headers,
                cookies=dict(cookies),
                auth=auth,
                follow_redirects=follow_redirects,
                extensions={"trace": functools.partial(trace, verbose=verbose)},
            ) as response:
                if download is not None:
                    download_response(response, download)
                else:
                    response.read()
                    if response.content:
                        print_response(response)

    except RequestError as exc:
        console = rich.console.Console()
        console.print(f"[red]{type(exc).__name__}[/red]: {exc}")
        sys.exit(1)

    sys.exit(0 if response.is_success else 1)
<|endoftext|>from __future__ import annotations

import datetime
import enum
import logging
import time
import typing
import warnings
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType

from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import (
    DEFAULT_KEEPALIVE_PING_INTERVAL_SECONDS,
    DEFAULT_KEEPALIVE_PING_TIMEOUT_SECONDS,
    DEFAULT_LIMITS,
    DEFAULT_MAX_EVENT_SIZE_BYTES,
    DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_TIMEOUT_CONFIG,
    Limits,
    Proxy,
    Timeout,
)
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import (
    InvalidURL,
    RemoteProtocolError,
    TooManyRedirects,
    request_context,
)
from ._models import Cookies, Headers, Request, Response
from ._sse import EventSource
from ._status_codes import codes
from ._transports import AsyncHTTPTransport, HTTPTransport
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._types import (
    AsyncByteStream,
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    SyncByteStream,
    TimeoutTypes,
)
from ._urls import URL, QueryParams
from ._utils import URLPattern, get_environment_proxies

if typing.TYPE_CHECKING:
    import ssl  # pragma: no cover

    from .websockets._api import AsyncWebSocketSession, WebSocketSession

__all__ = ["USE_CLIENT_DEFAULT", "AsyncClient", "Client"]

# The type annotation for @classmethod and context managers here follows PEP 484
# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
T = typing.TypeVar("T", bound="Client")
U = typing.TypeVar("U", bound="AsyncClient")


def _is_https_redirect(url: URL, location: URL) -> bool:
    """
    Return 'True' if 'location' is a HTTPS upgrade of 'url'
    """
    origin = url.origin
    location_origin = location.origin

    return (
        origin.host == location_origin.host
        and origin.scheme == "http"
        and origin.port == 80
        and location_origin.scheme == "https"
        and location_origin.port == 443
    )


class UseClientDefault:
    """
    For some parameters such as `auth=...` and `timeout=...` we need to be able
    to indicate the default "unset" state, in a way that is distinctly different
    to using `None`.

    The default "unset" state indicates that whatever default is set on the
    client should be used. This is different to setting `None`, which
    explicitly disables the parameter, possibly overriding a client default.

    For example we use `timeout=USE_CLIENT_DEFAULT` in the `request()` signature.
    Omitting the `timeout` parameter will send a request using whatever default
    timeout has been configured on the client. Including `timeout=None` will
    ensure no timeout is used.

    Note that user code shouldn't need to use the `USE_CLIENT_DEFAULT` constant,
    but it is used internally when a parameter is not included.
    """


USE_CLIENT_DEFAULT = UseClientDefault()


logger = logging.getLogger("httpx2")

USER_AGENT = f"python-httpx2/{__version__}"
ACCEPT_ENCODING = ", ".join([key for key in SUPPORTED_DECODERS.keys() if key != "identity"])


class ClientState(enum.Enum):
    # UNOPENED:
    #   The client has been instantiated, but has not been used to send a request,
    #   or been opened by entering the context of a `with` block.
    UNOPENED = 1
    # OPENED:
    #   The client has either sent a request, or is within a `with` block.
    OPENED = 2
    # CLOSED:
    #   The client has either exited the `with` block, or `close()` has
    #   been called explicitly.
    CLOSED = 3


class BoundSyncStream(SyncByteStream):
    """
    A byte stream that tracks elapsed time for a response. Once closed, the
    elapsed time is available via the `elapsed` attribute, and the response
    can read it back from `response.stream.elapsed`.
    """

    def __init__(self, stream: SyncByteStream, start: float) -> None:
        self._stream = stream
        self._start = start
        self.elapsed: datetime.timedelta | None = None

    def __iter__(self) -> typing.Iterator[bytes]:
        yield from self._stream

    def close(self) -> None:
        self.elapsed = datetime.timedelta(seconds=time.perf_counter() - self._start)
        self._stream.close()


class BoundAsyncStream(AsyncByteStream):
    """
    An async byte stream that tracks elapsed time for a response. Once closed,
    the elapsed time is available via the `elapsed` attribute, and the response
    can read it back from `response.stream.elapsed`.
    """

    def __init__(self, stream: AsyncByteStream, start: float) -> None:
        self._stream = stream
        self._start = start
        self.elapsed: datetime.timedelta | None = None

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        stream = self._stream.__aiter__()
        try:
            async for chunk in stream:
                yield chunk
        finally:
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()

    async def aclose(self) -> None:
        self.elapsed = datetime.timedelta(seconds=time.perf_counter() - self._start)
        await self._stream.aclose()


EventHook = typing.Callable[..., typing.Any]


class BaseClient:
    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: None | (typing.Mapping[str, list[EventHook]]) = None,
        base_url: URL | str = "",
        trust_env: bool = True,
        default_encoding: str | typing.Callable[[bytes], str | None] = "utf-8",
    ) -> None:
        event_hooks = {} if event_hooks is None else event_hooks

        self._base_url = self._enforce_trailing_slash(URL(base_url))

        self._auth = self._build_auth(auth)
        self._params = QueryParams(params)
        self.headers = Headers(headers)
        self._cookies = Cookies(cookies)
        self._timeout = Timeout(timeout)
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self._event_hooks = {
            "request": list(event_hooks.get("request", [])),
            "response": list(event_hooks.get("response", [])),
        }
        self._trust_env = trust_env
        self._default_encoding = default_encoding
        self._state = ClientState.UNOPENED

    @property
    def is_closed(self) -> bool:
        """
        Check if the client being closed
        """
        return self._state == ClientState.CLOSED

    @property
    def trust_env(self) -> bool:
        return self._trust_env

    def _enforce_trailing_slash(self, url: URL) -> URL:
        if url.raw_path.endswith(b"/"):
            return url
        return url.copy_with(raw_path=url.raw_path + b"/")

    def _get_proxy_map(self, proxy: ProxyTypes | None, allow_env_proxies: bool) -> dict[str, Proxy | None]:
        if proxy is None:
            if allow_env_proxies:
                return {key: None if url is None else Proxy(url=url) for key, url in get_environment_proxies().items()}
            return {}
        else:
            proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
            return {"all://": proxy}

    @property
    def timeout(self) -> Timeout:
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None:
        self._timeout = Timeout(timeout)

    @property
    def event_hooks(self) -> dict[str, list[EventHook]]:
        return self._event_hooks

    @event_hooks.setter
    def event_hooks(self, event_hooks: dict[str, list[EventHook]]) -> None:
        self._event_hooks = {
            "request": list(event_hooks.get("request", [])),
            "response": list(event_hooks.get("response", [])),
        }

    @property
    def auth(self) -> Auth | None:
        """
        Authentication class used when none is passed at the request-level.

        See also [Authentication][0].

        [0]: quickstart.md#authentication
        """
        return self._auth

    @auth.setter
    def auth(self, auth: AuthTypes) -> None:
        self._auth = self._build_auth(auth)

    @property
    def base_url(self) -> URL:
        """
        Base URL to use when sending requests with relative URLs.
        """
        return self._base_url

    @base_url.setter
    def base_url(self, url: URL | str) -> None:
        self._base_url = self._enforce_trailing_slash(URL(url))

    @property
    def headers(self) -> Headers:
        """
        HTTP headers to include when sending requests.
        """
        return self._headers

    @headers.setter
    def headers(self, headers: HeaderTypes) -> None:
        client_headers = Headers(
            {
                b"Accept": b"*/*",
                b"Accept-Encoding": ACCEPT_ENCODING.encode("ascii"),
                b"Connection": b"keep-alive",
                b"User-Agent": USER_AGENT.encode("ascii"),
            }
        )
        client_headers.update(headers)
        self._headers = client_headers

    @property
    def cookies(self) -> Cookies:
        """
        Cookie values to include when sending requests.
        """
        return self._cookies

    @cookies.setter
    def cookies(self, cookies: CookieTypes) -> None:
        self._cookies = Cookies(cookies)

    @property
    def params(self) -> QueryParams:
        """
        Query parameters to include in the URL when sending requests.
        """
        return self._params

    @params.setter
