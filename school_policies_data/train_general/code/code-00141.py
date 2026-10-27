        environment: "Environment",
        func: t.Callable[..., str],
        name: str,
        arguments: t.List[str],
        catch_kwargs: bool,
        catch_varargs: bool,
        caller: bool,
        default_autoescape: t.Optional[bool] = None,
    ):
        self._environment = environment
        self._func = func
        self._argument_count = len(arguments)
        self.name = name
        self.arguments = arguments
        self.catch_kwargs = catch_kwargs
        self.catch_varargs = catch_varargs
        self.caller = caller
        self.explicit_caller = "caller" in arguments

        if default_autoescape is None:
            if callable(environment.autoescape):
                default_autoescape = environment.autoescape(None)
            else:
                default_autoescape = environment.autoescape

        self._default_autoescape = default_autoescape

    @internalcode
    @pass_eval_context
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> str:
        # This requires a bit of explanation,  In the past we used to
        # decide largely based on compile-time information if a macro is
        # safe or unsafe.  While there was a volatile mode it was largely
        # unused for deciding on escaping.  This turns out to be
        # problematic for macros because whether a macro is safe depends not
        # on the escape mode when it was defined, but rather when it was used.
        #
        # Because however we export macros from the module system and
        # there are historic callers that do not pass an eval context (and
        # will continue to not pass one), we need to perform an instance
        # check here.
        #
        # This is considered safe because an eval context is not a valid
        # argument to callables otherwise anyway.  Worst case here is
        # that if no eval context is passed we fall back to the compile
        # time autoescape flag.
        if args and isinstance(args[0], EvalContext):
            autoescape = args[0].autoescape
            args = args[1:]
        else:
            autoescape = self._default_autoescape

        # try to consume the positional arguments
        arguments = list(args[: self._argument_count])
        off = len(arguments)

        # For information why this is necessary refer to the handling
        # of caller in the `macro_body` handler in the compiler.
        found_caller = False

        # if the number of arguments consumed is not the number of
        # arguments expected we start filling in keyword arguments
        # and defaults.
        if off != self._argument_count:
            for name in self.arguments[len(arguments) :]:
                try:
                    value = kwargs.pop(name)
                except KeyError:
                    value = missing
                if name == "caller":
                    found_caller = True
                arguments.append(value)
        else:
            found_caller = self.explicit_caller

        # it's important that the order of these arguments does not change
        # if not also changed in the compiler's `function_scoping` method.
        # the order is caller, keyword arguments, positional arguments!
        if self.caller and not found_caller:
            caller = kwargs.pop("caller", None)
            if caller is None:
                caller = self._environment.undefined("No caller defined", name="caller")
            arguments.append(caller)

        if self.catch_kwargs:
            arguments.append(kwargs)
        elif kwargs:
            if "caller" in kwargs:
                raise TypeError(
                    f"macro {self.name!r} was invoked with two values for the special"
                    " caller argument. This is most likely a bug."
                )
            raise TypeError(
                f"macro {self.name!r} takes no keyword argument {next(iter(kwargs))!r}"
            )
        if self.catch_varargs:
            arguments.append(args[self._argument_count :])
        elif len(args) > self._argument_count:
            raise TypeError(
                f"macro {self.name!r} takes not more than"
                f" {len(self.arguments)} argument(s)"
            )

        return self._invoke(arguments, autoescape)

    async def _async_invoke(self, arguments: t.List[t.Any], autoescape: bool) -> str:
        rv = await self._func(*arguments)  # type: ignore

        if autoescape:
            return Markup(rv)

        return rv  # type: ignore

    def _invoke(self, arguments: t.List[t.Any], autoescape: bool) -> str:
        if self._environment.is_async:
            return self._async_invoke(arguments, autoescape)  # type: ignore

        rv = self._func(*arguments)

        if autoescape:
            rv = Markup(rv)

        return rv

    def __repr__(self) -> str:
        name = "anonymous" if self.name is None else repr(self.name)
        return f"<{type(self).__name__} {name}>"


class Undefined:
    """The default undefined type. This can be printed, iterated, and treated as
    a boolean. Any other operation will raise an :exc:`UndefinedError`.

    >>> foo = Undefined(name='foo')
    >>> str(foo)
    ''
    >>> not foo
    True
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = (
        "_undefined_hint",
        "_undefined_obj",
        "_undefined_name",
        "_undefined_exception",
    )

    def __init__(
        self,
        hint: t.Optional[str] = None,
        obj: t.Any = missing,
        name: t.Optional[str] = None,
        exc: t.Type[TemplateRuntimeError] = UndefinedError,
    ) -> None:
        self._undefined_hint = hint
        self._undefined_obj = obj
        self._undefined_name = name
        self._undefined_exception = exc

    @property
    def _undefined_message(self) -> str:
        """Build a message about the undefined value based on how it was
        accessed.
        """
        if self._undefined_hint:
            return self._undefined_hint

        if self._undefined_obj is missing:
            return f"{self._undefined_name!r} is undefined"

        if not isinstance(self._undefined_name, str):
            return (
                f"{object_type_repr(self._undefined_obj)} has no"
                f" element {self._undefined_name!r}"
            )

        return (
            f"{object_type_repr(self._undefined_obj)!r} has no"
            f" attribute {self._undefined_name!r}"
        )

    @internalcode
    def _fail_with_undefined_error(
        self, *args: t.Any, **kwargs: t.Any
    ) -> "te.NoReturn":
        """Raise an :exc:`UndefinedError` when operations are performed
        on the undefined value.
        """
        raise self._undefined_exception(self._undefined_message)

    @internalcode
    def __getattr__(self, name: str) -> t.Any:
        # Raise AttributeError on requests for names that appear to be unimplemented
        # dunder methods to keep Python's internal protocol probing behaviors working
        # properly in cases where another exception type could cause unexpected or
        # difficult-to-diagnose failures.
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)

        return self._fail_with_undefined_error()

    __add__ = __radd__ = __sub__ = __rsub__ = _fail_with_undefined_error
    __mul__ = __rmul__ = __div__ = __rdiv__ = _fail_with_undefined_error
    __truediv__ = __rtruediv__ = _fail_with_undefined_error
    __floordiv__ = __rfloordiv__ = _fail_with_undefined_error
    __mod__ = __rmod__ = _fail_with_undefined_error
    __pos__ = __neg__ = _fail_with_undefined_error
    __call__ = __getitem__ = _fail_with_undefined_error
    __lt__ = __le__ = __gt__ = __ge__ = _fail_with_undefined_error
    __int__ = __float__ = __complex__ = _fail_with_undefined_error
    __pow__ = __rpow__ = _fail_with_undefined_error

    def __eq__(self, other: t.Any) -> bool:
        return type(self) is type(other)

    def __ne__(self, other: t.Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return id(type(self))

    def __str__(self) -> str:
        return ""

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> t.Iterator[t.Any]:
        yield from ()

    async def __aiter__(self) -> t.AsyncIterator[t.Any]:
        for _ in ():
            yield

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Undefined"


def make_logging_undefined(
    logger: t.Optional["logging.Logger"] = None, base: t.Type[Undefined] = Undefined
) -> t.Type[Undefined]:
    """Given a logger object this returns a new undefined class that will
    log certain failures.  It will log iterations and printing.  If no
    logger is given a default logger is created.

    Example::

        logger = logging.getLogger(__name__)
        LoggingUndefined = make_logging_undefined(
            logger=logger,
            base=Undefined
        )

    .. versionadded:: 2.8

    :param logger: the logger to use.  If not provided, a default logger
                   is created.
    :param base: the base class to add logging functionality to.  This
                 defaults to :class:`Undefined`.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stderr))

    def _log_message(undef: Undefined) -> None:
        logger.warning("Template variable warning: %s", undef._undefined_message)

    class LoggingUndefined(base):  # type: ignore
        __slots__ = ()

        def _fail_with_undefined_error(  # type: ignore
            self, *args: t.Any, **kwargs: t.Any
        ) -> "te.NoReturn":
            try:
                super()._fail_with_undefined_error(*args, **kwargs)
            except self._undefined_exception as e:
                logger.error("Template variable error: %s", e)  # type: ignore
                raise e

        def __str__(self) -> str:
            _log_message(self)
            return super().__str__()  # type: ignore

        def __iter__(self) -> t.Iterator[t.Any]:
            _log_message(self)
            return super().__iter__()  # type: ignore

        def __bool__(self) -> bool:
            _log_message(self)
            return super().__bool__()  # type: ignore

    return LoggingUndefined


class ChainableUndefined(Undefined):
    """An undefined that is chainable, where both ``__getattr__`` and
    ``__getitem__`` return itself rather than raising an
    :exc:`UndefinedError`.

    >>> foo = ChainableUndefined(name='foo')
    >>> str(foo.bar['baz'])
    ''
    >>> foo.bar['baz'] + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined

    .. versionadded:: 2.11.0
    """

    __slots__ = ()

    def __html__(self) -> str:
        return str(self)

    def __getattr__(self, name: str) -> "ChainableUndefined":
        # Raise AttributeError on requests for names that appear to be unimplemented
        # dunder methods to avoid confusing Python with truthy non-method objects that
        # do not implement the protocol being probed for. e.g., copy.copy(Undefined())
        # fails spectacularly if getattr(Undefined(), '__setstate__') returns an
        # Undefined object instead of raising AttributeError to signal that it does not
        # support that style of object initialization.
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)

        return self

    def __getitem__(self, _name: str) -> "ChainableUndefined":  # type: ignore[override]
        return self


class DebugUndefined(Undefined):
    """An undefined that returns the debug info when printed.

    >>> foo = DebugUndefined(name='foo')
    >>> str(foo)
    '{{ foo }}'
    >>> not foo
    True
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = ()

    def __str__(self) -> str:
        if self._undefined_hint:
            message = f"undefined value printed: {self._undefined_hint}"

        elif self._undefined_obj is missing:
            message = self._undefined_name  # type: ignore

        else:
            message = (
                f"no such element: {object_type_repr(self._undefined_obj)}"
                f"[{self._undefined_name!r}]"
            )

        return f"{{{{ {message} }}}}"


class StrictUndefined(Undefined):
    """An undefined that barks on print and iteration as well as boolean
    tests and all kinds of comparisons.  In other words: you can do nothing
    with it except checking if it's defined using the `defined` test.

    >>> foo = StrictUndefined(name='foo')
    >>> str(foo)
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    >>> not foo
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = ()
    __iter__ = __str__ = __len__ = Undefined._fail_with_undefined_error
    __eq__ = __ne__ = __bool__ = __hash__ = Undefined._fail_with_undefined_error
    __contains__ = Undefined._fail_with_undefined_error
<|endoftext|>"""Extension API for adding custom tags and behavior."""

import pprint
import re
import typing as t

from markupsafe import Markup

from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .lexer import Token
    from .lexer import TokenStream
    from .parser import Parser

    class _TranslationsBasic(te.Protocol):
        def gettext(self, message: str) -> str: ...

        def ngettext(self, singular: str, plural: str, n: int) -> str:
            pass

    class _TranslationsContext(_TranslationsBasic):
        def pgettext(self, context: str, message: str) -> str: ...

        def npgettext(
            self, context: str, singular: str, plural: str, n: int
        ) -> str: ...

    _SupportedTranslations = t.Union[_TranslationsBasic, _TranslationsContext]


# I18N functions available in Jinja templates. If the I18N library
# provides ugettext, it will be assigned to gettext.
GETTEXT_FUNCTIONS: t.Tuple[str, ...] = (
    "_",
    "gettext",
    "ngettext",
    "pgettext",
    "npgettext",
)
_ws_re = re.compile(r"\s*\n\s*")


class Extension:
    """Extensions can be used to add extra functionality to the Jinja template
    system at the parser level.  Custom extensions are bound to an environment
    but may not store environment specific data on `self`.  The reason for
    this is that an extension can be bound to another environment (for
    overlays) by creating a copy and reassigning the `environment` attribute.

    As extensions are created by the environment they cannot accept any
    arguments for configuration.  One may want to work around that by using
    a factory function, but that is not possible as extensions are identified
    by their import name.  The correct way to configure the extension is
    storing the configuration values on the environment.  Because this way the
    environment ends up acting as central configuration storage the
    attributes may clash which is why extensions have to ensure that the names
    they choose for configuration are not too generic.  ``prefix`` for example
    is a terrible name, ``fragment_cache_prefix`` on the other hand is a good
    name as includes the name of the extension (fragment cache).
    """

    identifier: t.ClassVar[str]

    def __init_subclass__(cls) -> None:
        cls.identifier = f"{cls.__module__}.{cls.__name__}"

    #: if this extension parses this is the list of tags it's listening to.
    tags: t.Set[str] = set()

    #: the priority of that extension.  This is especially useful for
    #: extensions that preprocess values.  A lower value means higher
    #: priority.
    #:
    #: .. versionadded:: 2.4
    priority = 100

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def bind(self, environment: Environment) -> "te.Self":
        """Create a copy of this extension bound to another environment."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.environment = environment
        return rv

    def preprocess(
        self, source: str, name: t.Optional[str], filename: t.Optional[str] = None
    ) -> str:
        """This method is called before the actual lexing and can be used to
        preprocess the source.  The `filename` is optional.  The return value
        must be the preprocessed source.
        """
        return source

    def filter_stream(
        self, stream: "TokenStream"
    ) -> t.Union["TokenStream", t.Iterable["Token"]]:
        """It's passed a :class:`~jinja2.lexer.TokenStream` that can be used
        to filter tokens returned.  This method has to return an iterable of
        :class:`~jinja2.lexer.Token`\\s, but it doesn't have to return a
        :class:`~jinja2.lexer.TokenStream`.
        """
        return stream

    def parse(self, parser: "Parser") -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """If any of the :attr:`tags` matched this method is called with the
        parser as first argument.  The token the parser stream is pointing at
        is the name token that matched.  This method has to return one or a
        list of multiple nodes.
        """
        raise NotImplementedError()

    def attr(
        self, name: str, lineno: t.Optional[int] = None
    ) -> nodes.ExtensionAttribute:
        """Return an attribute node for the current extension.  This is useful
        to pass constants on extensions to generated template code.

        ::

            self.attr('_my_attribute', lineno=lineno)
        """
        return nodes.ExtensionAttribute(self.identifier, name, lineno=lineno)

    def call_method(
        self,
        name: str,
        args: t.Optional[t.List[nodes.Expr]] = None,
        kwargs: t.Optional[t.List[nodes.Keyword]] = None,
        dyn_args: t.Optional[nodes.Expr] = None,
        dyn_kwargs: t.Optional[nodes.Expr] = None,
        lineno: t.Optional[int] = None,
    ) -> nodes.Call:
        """Call a method of the extension.  This is a shortcut for
        :meth:`attr` + :class:`jinja2.nodes.Call`.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        return nodes.Call(
            self.attr(name, lineno=lineno),
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=lineno,
        )


@pass_context
def _gettext_alias(
    __context: Context, *args: t.Any, **kwargs: t.Any
) -> t.Union[t.Any, Undefined]:
    return __context.call(__context.resolve("gettext"), *args, **kwargs)


def _make_new_gettext(func: t.Callable[[str], str]) -> t.Callable[..., str]:
    @pass_context
    def gettext(__context: Context, __string: str, **variables: t.Any) -> str:
        rv = __context.call(func, __string)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, even if there are no
        # variables. This makes translation strings more consistent
        # and predictable. This requires escaping
        return rv % variables  # type: ignore

    return gettext


def _make_new_ngettext(func: t.Callable[[str, str, int], str]) -> t.Callable[..., str]:
    @pass_context
    def ngettext(
        __context: Context,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("num", __num)
        rv = __context.call(func, __singular, __plural, __num)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return ngettext


def _make_new_pgettext(func: t.Callable[[str, str], str]) -> t.Callable[..., str]:
    @pass_context
    def pgettext(
        __context: Context, __string_ctx: str, __string: str, **variables: t.Any
    ) -> str:
        variables.setdefault("context", __string_ctx)
        rv = __context.call(func, __string_ctx, __string)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return pgettext


def _make_new_npgettext(
    func: t.Callable[[str, str, str, int], str],
) -> t.Callable[..., str]:
    @pass_context
    def npgettext(
        __context: Context,
        __string_ctx: str,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("context", __string_ctx)
        variables.setdefault("num", __num)
        rv = __context.call(func, __string_ctx, __singular, __plural, __num)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return npgettext


class InternationalizationExtension(Extension):
    """This extension adds gettext support to Jinja."""

    tags = {"trans"}

    # TODO: the i18n extension is currently reevaluating values in a few
    # situations.  Take this example:
    #   {% trans count=something() %}{{ count }} foo{% pluralize
    #     %}{{ count }} fooss{% endtrans %}
    # something is called twice here.  One time for the gettext value and
    # the other time for the n-parameter of the ngettext function.

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals["_"] = _gettext_alias
        environment.extend(
            install_gettext_translations=self._install,
            install_null_translations=self._install_null,
            install_gettext_callables=self._install_callables,
            uninstall_gettext_translations=self._uninstall,
            extract_translations=self._extract,
            newstyle_gettext=False,
        )

    def _install(
        self, translations: "_SupportedTranslations", newstyle: t.Optional[bool] = None
    ) -> None:
        # ugettext and ungettext are preferred in case the I18N library
        # is providing compatibility with older Python versions.
        gettext = getattr(translations, "ugettext", None)
        if gettext is None:
            gettext = translations.gettext
        ngettext = getattr(translations, "ungettext", None)
        if ngettext is None:
            ngettext = translations.ngettext

        pgettext = getattr(translations, "pgettext", None)
        npgettext = getattr(translations, "npgettext", None)
        self._install_callables(
            gettext, ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext
        )

    def _install_null(self, newstyle: t.Optional[bool] = None) -> None:
        import gettext

        translations = gettext.NullTranslations()

        if hasattr(translations, "pgettext"):
            # Python < 3.8
            pgettext = translations.pgettext
        else:

            def pgettext(c: str, s: str) -> str:  # type: ignore[misc]
                return s

        if hasattr(translations, "npgettext"):
            npgettext = translations.npgettext
        else:

            def npgettext(c: str, s: str, p: str, n: int) -> str:  # type: ignore[misc]
                return s if n == 1 else p

        self._install_callables(
            gettext=translations.gettext,
            ngettext=translations.ngettext,
            newstyle=newstyle,
            pgettext=pgettext,
            npgettext=npgettext,
        )

    def _install_callables(
        self,
        gettext: t.Callable[[str], str],
        ngettext: t.Callable[[str, str, int], str],
        newstyle: t.Optional[bool] = None,
        pgettext: t.Optional[t.Callable[[str, str], str]] = None,
        npgettext: t.Optional[t.Callable[[str, str, str, int], str]] = None,
    ) -> None:
        if newstyle is not None:
            self.environment.newstyle_gettext = newstyle  # type: ignore
        if self.environment.newstyle_gettext:  # type: ignore
            gettext = _make_new_gettext(gettext)
            ngettext = _make_new_ngettext(ngettext)

            if pgettext is not None:
                pgettext = _make_new_pgettext(pgettext)

            if npgettext is not None:
                npgettext = _make_new_npgettext(npgettext)

        self.environment.globals.update(
            gettext=gettext, ngettext=ngettext, pgettext=pgettext, npgettext=npgettext
        )

    def _uninstall(self, translations: "_SupportedTranslations") -> None:
        for key in ("gettext", "ngettext", "pgettext", "npgettext"):
            self.environment.globals.pop(key, None)

    def _extract(
        self,
        source: t.Union[str, nodes.Template],
        gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    ) -> t.Iterator[
        t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]
    ]:
        if isinstance(source, str):
            source = self.environment.parse(source)
        return extract_from_ast(source, gettext_functions)

    def parse(self, parser: "Parser") -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a translatable tag."""
        lineno = next(parser.stream).lineno

        context = None
        context_token = parser.stream.next_if("string")

        if context_token is not None:
            context = context_token.value

        # find all the variables referenced.  Additionally a variable can be
        # defined in the body of the trans block too, but this is checked at
        # a later state.
        plural_expr: t.Optional[nodes.Expr] = None
        plural_expr_assignment: t.Optional[nodes.Assign] = None
        num_called_num = False
        variables: t.Dict[str, nodes.Expr] = {}
        trimmed = None
        while parser.stream.current.type != "block_end":
            if variables:
                parser.stream.expect("comma")

            # skip colon for python compatibility
            if parser.stream.skip_if("colon"):
                break

            token = parser.stream.expect("name")
            if token.value in variables:
                parser.fail(
                    f"translatable variable {token.value!r} defined twice.",
                    token.lineno,
                    exc=TemplateAssertionError,
                )

            # expressions
            if parser.stream.current.type == "assign":
                next(parser.stream)
                variables[token.value] = var = parser.parse_expression()
            elif trimmed is None and token.value in ("trimmed", "notrimmed"):
                trimmed = token.value == "trimmed"
                continue
            else:
                variables[token.value] = var = nodes.Name(token.value, "load")

            if plural_expr is None:
                if isinstance(var, nodes.Call):
                    plural_expr = nodes.Name("_trans", "load")
                    variables[token.value] = plural_expr
                    plural_expr_assignment = nodes.Assign(
                        nodes.Name("_trans", "store"), var
                    )
                else:
                    plural_expr = var
                num_called_num = token.value == "num"

        parser.stream.expect("block_end")

        plural = None
        have_plural = False
        referenced = set()

        # now parse until endtrans or pluralize
        singular_names, singular = self._parse_block(parser, True)
        if singular_names:
            referenced.update(singular_names)
            if plural_expr is None:
                plural_expr = nodes.Name(singular_names[0], "load")
                num_called_num = singular_names[0] == "num"

        # if we have a pluralize block, we parse that too
        if parser.stream.current.test("name:pluralize"):
            have_plural = True
            next(parser.stream)
            if parser.stream.current.type != "block_end":
                token = parser.stream.expect("name")
                if token.value not in variables:
                    parser.fail(
                        f"unknown variable {token.value!r} for pluralization",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )
                plural_expr = variables[token.value]
                num_called_num = token.value == "num"
            parser.stream.expect("block_end")
            plural_names, plural = self._parse_block(parser, False)
            next(parser.stream)
            referenced.update(plural_names)
        else:
            next(parser.stream)

        # register free names as simple name expressions
        for name in referenced:
            if name not in variables:
                variables[name] = nodes.Name(name, "load")

        if not have_plural:
            plural_expr = None
        elif plural_expr is None:
            parser.fail("pluralize without variables", lineno)

        if trimmed is None:
            trimmed = self.environment.policies["ext.i18n.trimmed"]
        if trimmed:
            singular = self._trim_whitespace(singular)
            if plural:
                plural = self._trim_whitespace(plural)

        node = self._make_node(
            singular,
            plural,
            context,
            variables,
            plural_expr,
            bool(referenced),
            num_called_num and have_plural,
        )
        node.set_lineno(lineno)
        if plural_expr_assignment is not None:
            return [plural_expr_assignment, node]
        else:
            return node

    def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str] = _ws_re) -> str:
        return _ws_re.sub(" ", string.strip())

    def _parse_block(
        self, parser: "Parser", allow_pluralize: bool
    ) -> t.Tuple[t.List[str], str]:
        """Parse until the next block tag with a given name."""
        referenced = []
        buf = []

        while True:
            if parser.stream.current.type == "data":
                buf.append(parser.stream.current.value.replace("%", "%%"))
                next(parser.stream)
            elif parser.stream.current.type == "variable_begin":
                next(parser.stream)
                name = parser.stream.expect("name").value
                referenced.append(name)
                buf.append(f"%({name})s")
                parser.stream.expect("variable_end")
            elif parser.stream.current.type == "block_begin":
                next(parser.stream)
                block_name = (
                    parser.stream.current.value
                    if parser.stream.current.type == "name"
                    else None
                )
                if block_name == "endtrans":
                    break
                elif block_name == "pluralize":
                    if allow_pluralize:
                        break
                    parser.fail(
                        "a translatable section can have only one pluralize section"
                    )
                elif block_name == "trans":
                    parser.fail(
                        "trans blocks can't be nested; did you mean `endtrans`?"
                    )
                parser.fail(
                    f"control structures in translatable sections are not allowed; "
                    f"saw `{block_name}`"
                )
            elif parser.stream.eos:
                parser.fail("unclosed translation block")
            else:
                raise RuntimeError("internal parser error")

        return referenced, concat(buf)

    def _make_node(
        self,
        singular: str,
        plural: t.Optional[str],
        context: t.Optional[str],
        variables: t.Dict[str, nodes.Expr],
        plural_expr: t.Optional[nodes.Expr],
        vars_referenced: bool,
        num_called_num: bool,
    ) -> nodes.Output:
        """Generates a useful node from the data provided."""
        newstyle = self.environment.newstyle_gettext  # type: ignore
        node: nodes.Expr

        # no variables referenced?  no need to escape for old style
        # gettext invocations only if there are vars.
        if not vars_referenced and not newstyle:
            singular = singular.replace("%%", "%")
            if plural:
                plural = plural.replace("%%", "%")

        func_name = "gettext"
        func_args: t.List[nodes.Expr] = [nodes.Const(singular)]

        if context is not None:
            func_args.insert(0, nodes.Const(context))
            func_name = f"p{func_name}"

        if plural_expr is not None:
            func_name = f"n{func_name}"
            func_args.extend((nodes.Const(plural), plural_expr))

        node = nodes.Call(nodes.Name(func_name, "load"), func_args, [], None, None)

        # in case newstyle gettext is used, the method is powerful
        # enough to handle the variable expansion and autoescape
        # handling itself
        if newstyle:
            for key, value in variables.items():
                # the function adds that later anyways in case num was
                # called num, so just skip it.
                if num_called_num and key == "num":
                    continue
                node.kwargs.append(nodes.Keyword(key, value))

        # otherwise do that here
        else:
            # mark the return value as safe if we are in an
            # environment with autoescaping turned on
            node = nodes.MarkSafeIfAutoescape(node)
            if variables:
                node = nodes.Mod(
                    node,
                    nodes.Dict(
                        [
                            nodes.Pair(nodes.Const(key), value)
                            for key, value in variables.items()
                        ]
                    ),
                )
        return nodes.Output([node])


class ExprStmtExtension(Extension):
    """Adds a `do` tag to Jinja that works like the print statement just
    that it doesn't print the return value.
    """

    tags = {"do"}

    def parse(self, parser: "Parser") -> nodes.ExprStmt:
        node = nodes.ExprStmt(lineno=next(parser.stream).lineno)
        node.node = parser.parse_tuple()
        return node


class LoopControlExtension(Extension):
    """Adds break and continue to the template engine."""

    tags = {"break", "continue"}

    def parse(self, parser: "Parser") -> t.Union[nodes.Break, nodes.Continue]:
        token = next(parser.stream)
        if token.value == "break":
            return nodes.Break(lineno=token.lineno)
        return nodes.Continue(lineno=token.lineno)


class DebugExtension(Extension):
    """A ``{% debug %}`` tag that dumps the available variables,
    filters, and tests.

    .. code-block:: html+jinja

        <pre>{% debug %}</pre>

    .. code-block:: text

        {'context': {'cycler': <class 'jinja2.utils.Cycler'>,
                     ...,
                     'namespace': <class 'jinja2.utils.Namespace'>},
         'filters': ['abs', 'attr', 'batch', 'capitalize', 'center', 'count', 'd',
                     ..., 'urlencode', 'urlize', 'wordcount', 'wordwrap', 'xmlattr'],
         'tests': ['!=', '<', '<=', '==', '>', '>=', 'callable', 'defined',
                   ..., 'odd', 'sameas', 'sequence', 'string', 'undefined', 'upper']}

    .. versionadded:: 2.11.0
    """

    tags = {"debug"}

    def parse(self, parser: "Parser") -> nodes.Output:
        lineno = parser.stream.expect("name:debug").lineno
        context = nodes.ContextReference()
        result = self.call_method("_render", [context], lineno=lineno)
        return nodes.Output([result], lineno=lineno)

    def _render(self, context: Context) -> str:
        result = {
            "context": context.get_all(),
            "filters": sorted(self.environment.filters.keys()),
            "tests": sorted(self.environment.tests.keys()),
        }

        # Set the depth since the intent is to show the top few names.
        return pprint.pformat(result, depth=3, compact=True)


def extract_from_ast(
    ast: nodes.Template,
    gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    babel_style: bool = True,
) -> t.Iterator[
    t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]
]:
    """Extract localizable strings from the given template node.  Per
    default this function returns matches in babel style that means non string
    parameters as well as keyword arguments are returned as `None`.  This
    allows Babel to figure out what you really meant if you are using
    gettext functions that allow keyword arguments for placeholder expansion.
    If you don't want that behavior set the `babel_style` parameter to `False`
    which causes only strings to be returned and parameters are always stored
    in tuples.  As a consequence invalid gettext calls (calls without a single
    string parameter or string parameters after non-string parameters) are
    skipped.

    This example explains the behavior:

    >>> from jinja2 import Environment
    >>> env = Environment()
    >>> node = env.parse('{{ (_("foo"), _(), ngettext("foo", "bar", 42)) }}')
    >>> list(extract_from_ast(node))
    [(1, '_', 'foo'), (1, '_', ()), (1, 'ngettext', ('foo', 'bar', None))]
    >>> list(extract_from_ast(node, babel_style=False))
    [(1, '_', ('foo',)), (1, 'ngettext', ('foo', 'bar'))]

    For every string found this function yields a ``(lineno, function,
    message)`` tuple, where:

    * ``lineno`` is the number of the line on which the string was found,
    * ``function`` is the name of the ``gettext`` function used (if the
      string was extracted from embedded Python code), and
    *   ``message`` is the string, or a tuple of strings for functions
         with multiple string arguments.

    This extraction function operates on the AST and is because of that unable
    to extract any comments.  For comment support you have to use the babel
    extraction interface or extract comments yourself.
    """
    out: t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]

    for node in ast.find_all(nodes.Call):
        if (
            not isinstance(node.node, nodes.Name)
            or node.node.name not in gettext_functions
        ):
            continue

        strings: t.List[t.Optional[str]] = []

        for arg in node.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                strings.append(arg.value)
            else:
                strings.append(None)

        for _ in node.kwargs:
            strings.append(None)
        if node.dyn_args is not None:
            strings.append(None)
        if node.dyn_kwargs is not None:
            strings.append(None)

        if not babel_style:
            out = tuple(x for x in strings if x is not None)

            if not out:
                continue
        else:
            if len(strings) == 1:
                out = strings[0]
            else:
                out = tuple(strings)

        yield node.lineno, node.node.name, out


class _CommentFinder:
    """Helper class to find comments in a token stream.  Can only
    find comments for gettext calls forwards.  Once the comment
    from line 4 is found, a comment for line 1 will not return a
    usable value.
    """

    def __init__(
        self, tokens: t.Sequence[t.Tuple[int, str, str]], comment_tags: t.Sequence[str]
    ) -> None:
        self.tokens = tokens
        self.comment_tags = comment_tags
        self.offset = 0
        self.last_lineno = 0

    def find_backwards(self, offset: int) -> t.List[str]:
        try:
            for _, token_type, token_value in reversed(
                self.tokens[self.offset : offset]
            ):
                if token_type in ("comment", "linecomment"):
                    try:
                        prefix, comment = token_value.split(None, 1)
                    except ValueError:
                        continue
                    if prefix in self.comment_tags:
                        return [comment.rstrip()]
            return []
        finally:
            self.offset = offset

    def find_comments(self, lineno: int) -> t.List[str]:
        if not self.comment_tags or self.last_lineno > lineno:
            return []
        for idx, (token_lineno, _, _) in enumerate(self.tokens[self.offset :]):
            if token_lineno > lineno:
                return self.find_backwards(self.offset + idx)
        return self.find_backwards(len(self.tokens))


def babel_extract(
    fileobj: t.BinaryIO,
    keywords: t.Sequence[str],
    comment_tags: t.Sequence[str],
    options: t.Dict[str, t.Any],
) -> t.Iterator[
    t.Tuple[
        int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]], t.List[str]
    ]
]:
    """Babel extraction method for Jinja templates.

    .. versionchanged:: 2.3
       Basic support for translation comments was added.  If `comment_tags`
       is now set to a list of keywords for extraction, the extractor will
       try to find the best preceding comment that begins with one of the
       keywords.  For best results, make sure to not have more than one
       gettext call in one line of code and the matching comment in the
       same line or the line before.

    .. versionchanged:: 2.5.1
       The `newstyle_gettext` flag can be set to `True` to enable newstyle
       gettext calls.

    .. versionchanged:: 2.7
       A `silent` option can now be provided.  If set to `False` template
       syntax errors are propagated instead of being ignored.

    :param fileobj: the file-like object the messages should be extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results.
    :param options: a dictionary of additional options (optional)
    :return: an iterator over ``(lineno, funcname, message, comments)`` tuples.
             (comments will be empty currently)
    """
    extensions: t.Dict[t.Type[Extension], None] = {}

    for extension_name in options.get("extensions", "").split(","):
        extension_name = extension_name.strip()

        if not extension_name:
            continue

        extensions[import_string(extension_name)] = None

    if InternationalizationExtension not in extensions:
        extensions[InternationalizationExtension] = None

    def getbool(options: t.Mapping[str, str], key: str, default: bool = False) -> bool:
        return options.get(key, str(default)).lower() in {"1", "on", "yes", "true"}

    silent = getbool(options, "silent", True)
    environment = Environment(
        options.get("block_start_string", defaults.BLOCK_START_STRING),
        options.get("block_end_string", defaults.BLOCK_END_STRING),
        options.get("variable_start_string", defaults.VARIABLE_START_STRING),
        options.get("variable_end_string", defaults.VARIABLE_END_STRING),
        options.get("comment_start_string", defaults.COMMENT_START_STRING),
        options.get("comment_end_string", defaults.COMMENT_END_STRING),
        options.get("line_statement_prefix") or defaults.LINE_STATEMENT_PREFIX,
        options.get("line_comment_prefix") or defaults.LINE_COMMENT_PREFIX,
        getbool(options, "trim_blocks", defaults.TRIM_BLOCKS),
        getbool(options, "lstrip_blocks", defaults.LSTRIP_BLOCKS),
        defaults.NEWLINE_SEQUENCE,
        getbool(options, "keep_trailing_newline", defaults.KEEP_TRAILING_NEWLINE),
        tuple(extensions),
        cache_size=0,
        auto_reload=False,
    )

    if getbool(options, "trimmed"):
        environment.policies["ext.i18n.trimmed"] = True
    if getbool(options, "newstyle_gettext"):
        environment.newstyle_gettext = True  # type: ignore

    source = fileobj.read().decode(options.get("encoding", "utf-8"))
    try:
        node = environment.parse(source)
        tokens = list(environment.lex(environment.preprocess(source)))
    except TemplateSyntaxError:
        if not silent:
            raise
        # skip templates with syntax errors
        return

    finder = _CommentFinder(tokens, comment_tags)
    for lineno, func, message in extract_from_ast(node, keywords):
        yield lineno, func, message, finder.find_comments(lineno)


#: nicer import names
i18n = InternationalizationExtension
do = ExprStmtExtension
loopcontrols = LoopControlExtension
debug = DebugExtension
<|endoftext|>"""Implements a Jinja / Python combination lexer. The ``Lexer`` class
is used to do some preprocessing. It filters out invalid operators like
the bitshift operators we don't allow in templates. It separates
template code and python code in expressions.
"""

import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern

from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

# cache for the lexers. Exists in order to be able to have multiple
# environments with the same lexer
_lexer_cache: t.MutableMapping[t.Tuple, "Lexer"] = LRUCache(50)  # type: ignore

# static regular expressions
whitespace_re = re.compile(r"\s+")
newline_re = re.compile(r"(\r\n|\r|\n)")
string_re = re.compile(
    r"('([^'\\]*(?:\\.[^'\\]*)*)'" r'|"([^"\\]*(?:\\.[^"\\]*)*)")', re.S
)
integer_re = re.compile(
    r"""
    (
        0b(_?[0-1])+ # binary
    |
        0o(_?[0-7])+ # octal
    |
        0x(_?[\da-f])+ # hex
    |
        [1-9](_?\d)* # decimal
    |
        0(_?0)* # decimal zero
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
float_re = re.compile(
    r"""
    (?<!\.)  # doesn't start with a .
    (\d+_)*\d+  # digits, possibly _ separated
    (
        (\.(\d+_)*\d+)?  # optional fractional part
        e[+\-]?(\d+_)*\d+  # exponent part
    |
        \.(\d+_)*\d+  # required fractional part
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# internal the tokens and keep references to them
TOKEN_ADD = intern("add")
TOKEN_ASSIGN = intern("assign")
TOKEN_COLON = intern("colon")
TOKEN_COMMA = intern("comma")
TOKEN_DIV = intern("div")
TOKEN_DOT = intern("dot")
TOKEN_EQ = intern("eq")
TOKEN_FLOORDIV = intern("floordiv")
TOKEN_GT = intern("gt")
TOKEN_GTEQ = intern("gteq")
TOKEN_LBRACE = intern("lbrace")
TOKEN_LBRACKET = intern("lbracket")
TOKEN_LPAREN = intern("lparen")
TOKEN_LT = intern("lt")
TOKEN_LTEQ = intern("lteq")
TOKEN_MOD = intern("mod")
TOKEN_MUL = intern("mul")
TOKEN_NE = intern("ne")
TOKEN_PIPE = intern("pipe")
TOKEN_POW = intern("pow")
TOKEN_RBRACE = intern("rbrace")
TOKEN_RBRACKET = intern("rbracket")
TOKEN_RPAREN = intern("rparen")
TOKEN_SEMICOLON = intern("semicolon")
TOKEN_SUB = intern("sub")
TOKEN_TILDE = intern("tilde")
TOKEN_WHITESPACE = intern("whitespace")
TOKEN_FLOAT = intern("float")
TOKEN_INTEGER = intern("integer")
TOKEN_NAME = intern("name")
TOKEN_STRING = intern("string")
TOKEN_OPERATOR = intern("operator")
TOKEN_BLOCK_BEGIN = intern("block_begin")
TOKEN_BLOCK_END = intern("block_end")
TOKEN_VARIABLE_BEGIN = intern("variable_begin")
TOKEN_VARIABLE_END = intern("variable_end")
TOKEN_RAW_BEGIN = intern("raw_begin")
TOKEN_RAW_END = intern("raw_end")
TOKEN_COMMENT_BEGIN = intern("comment_begin")
TOKEN_COMMENT_END = intern("comment_end")
TOKEN_COMMENT = intern("comment")
TOKEN_LINESTATEMENT_BEGIN = intern("linestatement_begin")
TOKEN_LINESTATEMENT_END = intern("linestatement_end")
TOKEN_LINECOMMENT_BEGIN = intern("linecomment_begin")
TOKEN_LINECOMMENT_END = intern("linecomment_end")
TOKEN_LINECOMMENT = intern("linecomment")
TOKEN_DATA = intern("data")
TOKEN_INITIAL = intern("initial")
TOKEN_EOF = intern("eof")

# bind operators to token types
operators = {
    "+": TOKEN_ADD,
    "-": TOKEN_SUB,
    "/": TOKEN_DIV,
    "//": TOKEN_FLOORDIV,
    "*": TOKEN_MUL,
    "%": TOKEN_MOD,
    "**": TOKEN_POW,
    "~": TOKEN_TILDE,
    "[": TOKEN_LBRACKET,
    "]": TOKEN_RBRACKET,
    "(": TOKEN_LPAREN,
    ")": TOKEN_RPAREN,
    "{": TOKEN_LBRACE,
    "}": TOKEN_RBRACE,
    "==": TOKEN_EQ,
    "!=": TOKEN_NE,
    ">": TOKEN_GT,
    ">=": TOKEN_GTEQ,
    "<": TOKEN_LT,
    "<=": TOKEN_LTEQ,
    "=": TOKEN_ASSIGN,
    ".": TOKEN_DOT,
    ":": TOKEN_COLON,
    "|": TOKEN_PIPE,
    ",": TOKEN_COMMA,
    ";": TOKEN_SEMICOLON,
}

reverse_operators = {v: k for k, v in operators.items()}
assert len(operators) == len(reverse_operators), "operators dropped"
operator_re = re.compile(
    f"({'|'.join(re.escape(x) for x in sorted(operators, key=lambda x: -len(x)))})"
)

ignored_tokens = frozenset(
    [
        TOKEN_COMMENT_BEGIN,
        TOKEN_COMMENT,
        TOKEN_COMMENT_END,
        TOKEN_WHITESPACE,
        TOKEN_LINECOMMENT_BEGIN,
        TOKEN_LINECOMMENT_END,
        TOKEN_LINECOMMENT,
    ]
)
ignore_if_empty = frozenset(
    [TOKEN_WHITESPACE, TOKEN_DATA, TOKEN_COMMENT, TOKEN_LINECOMMENT]
)


def _describe_token_type(token_type: str) -> str:
    if token_type in reverse_operators:
        return reverse_operators[token_type]

    return {
        TOKEN_COMMENT_BEGIN: "begin of comment",
        TOKEN_COMMENT_END: "end of comment",
        TOKEN_COMMENT: "comment",
        TOKEN_LINECOMMENT: "comment",
        TOKEN_BLOCK_BEGIN: "begin of statement block",
        TOKEN_BLOCK_END: "end of statement block",
        TOKEN_VARIABLE_BEGIN: "begin of print statement",
        TOKEN_VARIABLE_END: "end of print statement",
        TOKEN_LINESTATEMENT_BEGIN: "begin of line statement",
        TOKEN_LINESTATEMENT_END: "end of line statement",
        TOKEN_DATA: "template data / text",
        TOKEN_EOF: "end of template",
    }.get(token_type, token_type)


def describe_token(token: "Token") -> str:
    """Returns a description of the token."""
    if token.type == TOKEN_NAME:
        return token.value

    return _describe_token_type(token.type)


def describe_token_expr(expr: str) -> str:
    """Like `describe_token` but for token expressions."""
    if ":" in expr:
        type, value = expr.split(":", 1)

        if type == TOKEN_NAME:
            return value
    else:
        type = expr

    return _describe_token_type(type)


def count_newlines(value: str) -> int:
    """Count the number of newline characters in the string.  This is
    useful for extensions that filter a stream.
    """
    return len(newline_re.findall(value))


def compile_rules(environment: "Environment") -> t.List[t.Tuple[str, str]]:
    """Compiles all the rules from the environment into a list of rules."""
    e = re.escape
    rules = [
        (
            len(environment.comment_start_string),
            TOKEN_COMMENT_BEGIN,
            e(environment.comment_start_string),
        ),
        (
            len(environment.block_start_string),
            TOKEN_BLOCK_BEGIN,
            e(environment.block_start_string),
        ),
        (
            len(environment.variable_start_string),
            TOKEN_VARIABLE_BEGIN,
            e(environment.variable_start_string),
        ),
    ]

    if environment.line_statement_prefix is not None:
        rules.append(
            (
                len(environment.line_statement_prefix),
                TOKEN_LINESTATEMENT_BEGIN,
                r"^[ \t\v]*" + e(environment.line_statement_prefix),
            )
        )
    if environment.line_comment_prefix is not None:
        rules.append(
            (
                len(environment.line_comment_prefix),
                TOKEN_LINECOMMENT_BEGIN,
                r"(?:^|(?<=\S))[^\S\r\n]*" + e(environment.line_comment_prefix),
            )
        )

    return [x[1:] for x in sorted(rules, reverse=True)]


class Failure:
    """Class that raises a `TemplateSyntaxError` if called.
    Used by the `Lexer` to specify known errors.
    """

    def __init__(
        self, message: str, cls: t.Type[TemplateSyntaxError] = TemplateSyntaxError
    ) -> None:
        self.message = message
        self.error_class = cls

    def __call__(self, lineno: int, filename: t.Optional[str]) -> "te.NoReturn":
        raise self.error_class(self.message, lineno, filename)


class Token(t.NamedTuple):
    lineno: int
    type: str
    value: str

    def __str__(self) -> str:
        return describe_token(self)

    def test(self, expr: str) -> bool:
        """Test a token against a token expression.  This can either be a
        token type or ``'token_type:token_value'``.  This can only test
        against string values and types.
        """
        # here we do a regular string equality check as test_any is usually
        # passed an iterable of not interned strings.
        if self.type == expr:
            return True

        if ":" in expr:
            return expr.split(":", 1) == [self.type, self.value]

        return False

    def test_any(self, *iterable: str) -> bool:
        """Test against multiple token expressions."""
        return any(self.test(expr) for expr in iterable)


class TokenStreamIterator:
    """The iterator for tokenstreams.  Iterate over the stream
    until the eof token is reached.
    """

    def __init__(self, stream: "TokenStream") -> None:
        self.stream = stream

    def __iter__(self) -> "TokenStreamIterator":
        return self

    def __next__(self) -> Token:
        token = self.stream.current

        if token.type is TOKEN_EOF:
            self.stream.close()
            raise StopIteration

        next(self.stream)
        return token


class TokenStream:
    """A token stream is an iterable that yields :class:`Token`\\s.  The
    parser however does not iterate over it but calls :meth:`next` to go
    one token ahead.  The current active token is stored as :attr:`current`.
    """

    def __init__(
        self,
        generator: t.Iterable[Token],
        name: t.Optional[str],
        filename: t.Optional[str],
    ):
        self._iter = iter(generator)
        self._pushed: te.Deque[Token] = deque()
        self.name = name
        self.filename = filename
        self.closed = False
        self.current = Token(1, TOKEN_INITIAL, "")
        next(self)

    def __iter__(self) -> TokenStreamIterator:
        return TokenStreamIterator(self)

    def __bool__(self) -> bool:
        return bool(self._pushed) or self.current.type is not TOKEN_EOF

    @property
    def eos(self) -> bool:
        """Are we at the end of the stream?"""
        return not self

    def push(self, token: Token) -> None:
        """Push a token back to the stream."""
        self._pushed.append(token)

    def look(self) -> Token:
        """Look at the next token."""
        old_token = next(self)
        result = self.current
        self.push(result)
        self.current = old_token
        return result

    def skip(self, n: int = 1) -> None:
        """Got n tokens ahead."""
        for _ in range(n):
            next(self)

    def next_if(self, expr: str) -> t.Optional[Token]:
        """Perform the token test and return the token if it matched.
        Otherwise the return value is `None`.
        """
        if self.current.test(expr):
            return next(self)

        return None

    def skip_if(self, expr: str) -> bool:
        """Like :meth:`next_if` but only returns `True` or `False`."""
        return self.next_if(expr) is not None

    def __next__(self) -> Token:
        """Go one token ahead and return the old one.

        Use the built-in :func:`next` instead of calling this directly.
        """
        rv = self.current

        if self._pushed:
            self.current = self._pushed.popleft()
        elif self.current.type is not TOKEN_EOF:
            try:
                self.current = next(self._iter)
            except StopIteration:
                self.close()

        return rv

    def close(self) -> None:
        """Close the stream."""
        self.current = Token(self.current.lineno, TOKEN_EOF, "")
        self._iter = iter(())
        self.closed = True

    def expect(self, expr: str) -> Token:
        """Expect a given token type and return it.  This accepts the same
        argument as :meth:`jinja2.lexer.Token.test`.
        """
        if not self.current.test(expr):
            expr = describe_token_expr(expr)

            if self.current.type is TOKEN_EOF:
                raise TemplateSyntaxError(
                    f"unexpected end of template, expected {expr!r}.",
                    self.current.lineno,
                    self.name,
                    self.filename,
                )

            raise TemplateSyntaxError(
                f"expected token {expr!r}, got {describe_token(self.current)!r}",
                self.current.lineno,
                self.name,
                self.filename,
            )

        return next(self)


def get_lexer(environment: "Environment") -> "Lexer":
    """Return a lexer which is probably cached."""
    key = (
        environment.block_start_string,
        environment.block_end_string,
        environment.variable_start_string,
        environment.variable_end_string,
        environment.comment_start_string,
        environment.comment_end_string,
        environment.line_statement_prefix,
        environment.line_comment_prefix,
        environment.trim_blocks,
        environment.lstrip_blocks,
        environment.newline_sequence,
        environment.keep_trailing_newline,
    )
    lexer = _lexer_cache.get(key)

    if lexer is None:
        _lexer_cache[key] = lexer = Lexer(environment)

    return lexer


class OptionalLStrip(tuple):  # type: ignore[type-arg]
    """A special tuple for marking a point in the state that can have
    lstrip applied.
    """

    __slots__ = ()

    # Even though it looks like a no-op, creating instances fails
    # without this.
    def __new__(cls, *members, **kwargs):  # type: ignore
        return super().__new__(cls, members)


class _Rule(t.NamedTuple):
    pattern: t.Pattern[str]
    tokens: t.Union[str, t.Tuple[str, ...], t.Tuple[Failure]]
    command: t.Optional[str]


class Lexer:
    """Class that implements a lexer for a given environment. Automatically
    created by the environment class, usually you don't have to do that.

    Note that the lexer is not automatically bound to an environment.
    Multiple environments can share the same lexer.
    """

    def __init__(self, environment: "Environment") -> None:
        # shortcuts
        e = re.escape

        def c(x: str) -> t.Pattern[str]:
            return re.compile(x, re.M | re.S)

        # lexing rules for tags
        tag_rules: t.List[_Rule] = [
            _Rule(whitespace_re, TOKEN_WHITESPACE, None),
            _Rule(float_re, TOKEN_FLOAT, None),
            _Rule(integer_re, TOKEN_INTEGER, None),
            _Rule(name_re, TOKEN_NAME, None),
            _Rule(string_re, TOKEN_STRING, None),
            _Rule(operator_re, TOKEN_OPERATOR, None),
        ]

        # assemble the root lexing rule. because "|" is ungreedy
        # we have to sort by length so that the lexer continues working
        # as expected when we have parsing rules like <% for block and
        # <%= for variables. (if someone wants asp like syntax)
        # variables are just part of the rules if variable processing
        # is required.
        root_tag_rules = compile_rules(environment)

        block_start_re = e(environment.block_start_string)
        block_end_re = e(environment.block_end_string)
        comment_end_re = e(environment.comment_end_string)
        variable_end_re = e(environment.variable_end_string)

        # block suffix if trimming is enabled
        block_suffix_re = "\\n?" if environment.trim_blocks else ""

        self.lstrip_blocks = environment.lstrip_blocks

        self.newline_sequence = environment.newline_sequence
        self.keep_trailing_newline = environment.keep_trailing_newline

        root_raw_re = (
            rf"(?P<raw_begin>{block_start_re}(\-|\+|)\s*raw\s*"
            rf"(?:\-{block_end_re}\s*|{block_end_re}))"
        )
        root_parts_re = "|".join(
            [root_raw_re] + [rf"(?P<{n}>{r}(\-|\+|))" for n, r in root_tag_rules]
        )

        # global lexing rules
        self.rules: t.Dict[str, t.List[_Rule]] = {
            "root": [
                # directives
                _Rule(
                    c(rf"(.*?)(?:{root_parts_re})"),
                    OptionalLStrip(TOKEN_DATA, "#bygroup"),  # type: ignore
                    "#bygroup",
                ),
                # data
                _Rule(c(".+"), TOKEN_DATA, None),
            ],
            # comments
            TOKEN_COMMENT_BEGIN: [
                _Rule(
                    c(
                        rf"(.*?)((?:\+{comment_end_re}|\-{comment_end_re}\s*"
                        rf"|{comment_end_re}{block_suffix_re}))"
                    ),
                    (TOKEN_COMMENT, TOKEN_COMMENT_END),
                    "#pop",
                ),
                _Rule(c(r"(.)"), (Failure("Missing end of comment tag"),), None),
            ],
            # blocks
            TOKEN_BLOCK_BEGIN: [
                _Rule(
                    c(
                        rf"(?:\+{block_end_re}|\-{block_end_re}\s*"
                        rf"|{block_end_re}{block_suffix_re})"
                    ),
                    TOKEN_BLOCK_END,
                    "#pop",
                ),
            ]
            + tag_rules,
            # variables
            TOKEN_VARIABLE_BEGIN: [
                _Rule(
                    c(rf"\-{variable_end_re}\s*|{variable_end_re}"),
                    TOKEN_VARIABLE_END,
                    "#pop",
                )
            ]
            + tag_rules,
            # raw block
            TOKEN_RAW_BEGIN: [
                _Rule(
                    c(
                        rf"(.*?)((?:{block_start_re}(\-|\+|))\s*endraw\s*"
                        rf"(?:\+{block_end_re}|\-{block_end_re}\s*"
                        rf"|{block_end_re}{block_suffix_re}))"
                    ),
                    OptionalLStrip(TOKEN_DATA, TOKEN_RAW_END),  # type: ignore
                    "#pop",
                ),
                _Rule(c(r"(.)"), (Failure("Missing end of raw directive"),), None),
            ],
            # line statements
            TOKEN_LINESTATEMENT_BEGIN: [
                _Rule(c(r"\s*(\n|$)"), TOKEN_LINESTATEMENT_END, "#pop")
            ]
            + tag_rules,
            # line comments
            TOKEN_LINECOMMENT_BEGIN: [
                _Rule(
                    c(r"(.*?)()(?=\n|$)"),
                    (TOKEN_LINECOMMENT, TOKEN_LINECOMMENT_END),
                    "#pop",
                )
            ],
        }

    def _normalize_newlines(self, value: str) -> str:
        """Replace all newlines with the configured sequence in strings
        and template data.
        """
        return newline_re.sub(self.newline_sequence, value)

    def tokenize(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> TokenStream:
        """Calls tokeniter + tokenize and wraps it in a token stream."""
        stream = self.tokeniter(source, name, filename, state)
        return TokenStream(self.wrap(stream, name, filename), name, filename)

    def wrap(
        self,
        stream: t.Iterable[t.Tuple[int, str, str]],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> t.Iterator[Token]:
        """This is called with the stream as returned by `tokenize` and wraps
        every token in a :class:`Token` and converts the value.
        """
        for lineno, token, value_str in stream:
            if token in ignored_tokens:
                continue

            value: t.Any = value_str

            if token == TOKEN_LINESTATEMENT_BEGIN:
                token = TOKEN_BLOCK_BEGIN
            elif token == TOKEN_LINESTATEMENT_END:
                token = TOKEN_BLOCK_END
            # we are not interested in those tokens in the parser
            elif token in (TOKEN_RAW_BEGIN, TOKEN_RAW_END):
                continue
            elif token == TOKEN_DATA:
                value = self._normalize_newlines(value_str)
            elif token == "keyword":
                token = value_str
            elif token == TOKEN_NAME:
                value = value_str

                if not value.isidentifier():
                    raise TemplateSyntaxError(
                        "Invalid character in identifier", lineno, name, filename
                    )
            elif token == TOKEN_STRING:
                # try to unescape string
                try:
                    value = (
                        self._normalize_newlines(value_str[1:-1])
                        .encode("ascii", "backslashreplace")
                        .decode("unicode-escape")
                    )
                except Exception as e:
                    msg = str(e).split(":")[-1].strip()
                    raise TemplateSyntaxError(msg, lineno, name, filename) from e
            elif token == TOKEN_INTEGER:
                value = int(value_str.replace("_", ""), 0)
            elif token == TOKEN_FLOAT:
                # remove all "_" first to support more Python versions
                value = literal_eval(value_str.replace("_", ""))
            elif token == TOKEN_OPERATOR:
                token = operators[value_str]

            yield Token(lineno, token, value)

    def tokeniter(
        self,
        source: str,
        name: t.Optional[str],
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> t.Iterator[t.Tuple[int, str, str]]:
        """This method tokenizes the text and returns the tokens in a
        generator. Use this method if you just want to tokenize a template.

        .. versionchanged:: 3.0
            Only ``\\n``, ``\\r\\n`` and ``\\r`` are treated as line
            breaks.
        """
        lines = newline_re.split(source)[::2]

        if not self.keep_trailing_newline and lines[-1] == "":
            del lines[-1]

        source = "\n".join(lines)
        pos = 0
        lineno = 1
        stack = ["root"]

        if state is not None and state != "root":
            assert state in ("variable", "block"), "invalid state"
            stack.append(state + "_begin")

        statetokens = self.rules[stack[-1]]
        source_length = len(source)
        balancing_stack: t.List[str] = []
        newlines_stripped = 0
        line_starting = True

        while True:
            # tokenizer loop
            for regex, tokens, new_state in statetokens:
                m = regex.match(source, pos)

                # if no match we try again with the next rule
                if m is None:
                    continue

                # we only match blocks and variables if braces / parentheses
                # are balanced. continue parsing with the lower rule which
                # is the operator rule. do this only if the end tags look
                # like operators
                if balancing_stack and tokens in (
                    TOKEN_VARIABLE_END,
                    TOKEN_BLOCK_END,
                    TOKEN_LINESTATEMENT_END,
                ):
                    continue

                # tuples support more options
                if isinstance(tokens, tuple):
                    groups: t.Sequence[str] = m.groups()

                    if isinstance(tokens, OptionalLStrip):
                        # Rule supports lstrip. Match will look like
                        # text, block type, whitespace control, type, control, ...
                        text = groups[0]
                        # Skipping the text and first type, every other group is the
                        # whitespace control for each type. One of the groups will be
                        # -, +, or empty string instead of None.
                        strip_sign = next(g for g in groups[2::2] if g is not None)

                        if strip_sign == "-":
                            # Strip all whitespace between the text and the tag.
                            stripped = text.rstrip()
                            newlines_stripped = text[len(stripped) :].count("\n")
                            groups = [stripped, *groups[1:]]
                        elif (
                            # Not marked for preserving whitespace.
                            strip_sign != "+"
                            # lstrip is enabled.
                            and self.lstrip_blocks
                            # Not a variable expression.
                            and not m.groupdict().get(TOKEN_VARIABLE_BEGIN)
                        ):
                            # The start of text between the last newline and the tag.
                            l_pos = text.rfind("\n") + 1

                            if l_pos > 0 or line_starting:
                                # If there's only whitespace between the newline and the
                                # tag, strip it.
                                if whitespace_re.fullmatch(text, l_pos):
                                    groups = [text[:l_pos], *groups[1:]]

                    for idx, token in enumerate(tokens):
                        # failure group
                        if isinstance(token, Failure):
                            raise token(lineno, filename)
                        # bygroup is a bit more complex, in that case we
                        # yield for the current token the first named
                        # group that matched
                        elif token == "#bygroup":
                            for key, value in m.groupdict().items():
                                if value is not None:
                                    yield lineno, key, value
                                    lineno += value.count("\n")
                                    break
                            else:
                                raise RuntimeError(
                                    f"{regex!r} wanted to resolve the token dynamically"
                                    " but no group matched"
                                )
                        # normal group
                        else:
                            data = groups[idx]

                            if data or token not in ignore_if_empty:
                                yield lineno, token, data  # type: ignore[misc]

                            lineno += data.count("\n") + newlines_stripped
                            newlines_stripped = 0

                # strings as token just are yielded as it.
                else:
                    data = m.group()

                    # update brace/parentheses balance
                    if tokens == TOKEN_OPERATOR:
                        if data == "{":
                            balancing_stack.append("}")
                        elif data == "(":
                            balancing_stack.append(")")
                        elif data == "[":
                            balancing_stack.append("]")
                        elif data in ("}", ")", "]"):
                            if not balancing_stack:
                                raise TemplateSyntaxError(
                                    f"unexpected '{data}'", lineno, name, filename
                                )

                            expected_op = balancing_stack.pop()

                            if expected_op != data:
                                raise TemplateSyntaxError(
                                    f"unexpected '{data}', expected '{expected_op}'",
                                    lineno,
                                    name,
                                    filename,
                                )

                    # yield items
                    if data or tokens not in ignore_if_empty:
                        yield lineno, tokens, data

                    lineno += data.count("\n")

                line_starting = m.group()[-1:] == "\n"
                # fetch new position into new variable so that we can check
                # if there is a internal parsing error which would result
                # in an infinite loop
                pos2 = m.end()

                # handle state changes
                if new_state is not None:
                    # remove the uppermost state
                    if new_state == "#pop":
                        stack.pop()
                    # resolve the new state by group checking
                    elif new_state == "#bygroup":
                        for key, value in m.groupdict().items():
                            if value is not None:
                                stack.append(key)
                                break
                        else:
                            raise RuntimeError(
                                f"{regex!r} wanted to resolve the new state dynamically"
                                f" but no group matched"
                            )
                    # direct state name given
                    else:
                        stack.append(new_state)

                    statetokens = self.rules[stack[-1]]
                # we are still at the same position and no stack change.
                # this means a loop without break condition, avoid that and
                # raise error
                elif pos2 == pos:
                    raise RuntimeError(
                        f"{regex!r} yielded empty string without stack change"
                    )

                # publish new function and start again
                pos = pos2
                break
            # if loop terminated without break we haven't found a single match
            # either we are at the end of the file or we have a problem
            else:
                # end of text
                if pos >= source_length:
                    return

                # something went wrong
                raise TemplateSyntaxError(
                    f"unexpected char {source[pos]!r} at {pos}", lineno, name, filename
                )
<|endoftext|>"""Parse tokens from the lexer into nodes for the compiler."""

import typing
import typing as t

from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

_ImportInclude = t.TypeVar("_ImportInclude", nodes.Import, nodes.Include)
_MacroCall = t.TypeVar("_MacroCall", nodes.Macro, nodes.CallBlock)

_statement_keywords = frozenset(
    [
        "for",
        "if",
        "block",
        "extends",
        "print",
        "macro",
        "include",
        "from",
        "import",
        "set",
        "with",
        "autoescape",
    ]
)
_compare_operators = frozenset(["eq", "ne", "lt", "lteq", "gt", "gteq"])

_math_nodes: t.Dict[str, t.Type[nodes.Expr]] = {
    "add": nodes.Add,
    "sub": nodes.Sub,
    "mul": nodes.Mul,
    "div": nodes.Div,
    "floordiv": nodes.FloorDiv,
    "mod": nodes.Mod,
}


class Parser:
    """This is the central parsing class Jinja uses.  It's passed to
    extensions and can be used to parse expressions or statements.
    """

    def __init__(
        self,
        environment: "Environment",
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> None:
        self.environment = environment
        self.stream = environment._tokenize(source, name, filename, state)
        self.name = name
        self.filename = filename
        self.closed = False
        self.extensions: t.Dict[
            str, t.Callable[[Parser], t.Union[nodes.Node, t.List[nodes.Node]]]
        ] = {}
        for extension in environment.iter_extensions():
            for tag in extension.tags:
                self.extensions[tag] = extension.parse
        self._last_identifier = 0
        self._tag_stack: t.List[str] = []
        self._end_token_stack: t.List[t.Tuple[str, ...]] = []

    def fail(
        self,
        msg: str,
        lineno: t.Optional[int] = None,
        exc: t.Type[TemplateSyntaxError] = TemplateSyntaxError,
    ) -> "te.NoReturn":
        """Convenience method that raises `exc` with the message, passed
        line number or last line number as well as the current name and
        filename.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        raise exc(msg, lineno, self.name, self.filename)

    def _fail_ut_eof(
        self,
        name: t.Optional[str],
        end_token_stack: t.List[t.Tuple[str, ...]],
        lineno: t.Optional[int],
    ) -> "te.NoReturn":
        expected: t.Set[str] = set()
        for exprs in end_token_stack:
            expected.update(map(describe_token_expr, exprs))
        if end_token_stack:
            currently_looking: t.Optional[str] = " or ".join(
                map(repr, map(describe_token_expr, end_token_stack[-1]))
            )
        else:
            currently_looking = None

        if name is None:
            message = ["Unexpected end of template."]
        else:
            message = [f"Encountered unknown tag {name!r}."]

        if currently_looking:
            if name is not None and name in expected:
                message.append(
                    "You probably made a nesting mistake. Jinja is expecting this tag,"
                    f" but currently looking for {currently_looking}."
                )
            else:
                message.append(
                    f"Jinja was looking for the following tags: {currently_looking}."
                )

        if self._tag_stack:
            message.append(
                "The innermost block that needs to be closed is"
                f" {self._tag_stack[-1]!r}."
            )

        self.fail(" ".join(message), lineno)

    def fail_unknown_tag(
        self, name: str, lineno: t.Optional[int] = None
    ) -> "te.NoReturn":
        """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
        self._fail_ut_eof(name, self._end_token_stack, lineno)

    def fail_eof(
        self,
        end_tokens: t.Optional[t.Tuple[str, ...]] = None,
        lineno: t.Optional[int] = None,
    ) -> "te.NoReturn":
        """Like fail_unknown_tag but for end of template situations."""
        stack = list(self._end_token_stack)
        if end_tokens is not None:
            stack.append(end_tokens)
        self._fail_ut_eof(None, stack, lineno)

    def is_tuple_end(
        self, extra_end_rules: t.Optional[t.Tuple[str, ...]] = None
    ) -> bool:
        """Are we at the end of a tuple?"""
        if self.stream.current.type in ("variable_end", "block_end", "rparen"):
            return True
        elif extra_end_rules is not None:
            return self.stream.current.test_any(extra_end_rules)  # type: ignore
        return False

    def free_identifier(self, lineno: t.Optional[int] = None) -> nodes.InternalName:
        """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
        self._last_identifier += 1
        rv = object.__new__(nodes.InternalName)
        nodes.Node.__init__(rv, f"fi{self._last_identifier}", lineno=lineno)
        return rv

    def parse_statement(self) -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a single statement."""
        token = self.stream.current
        if token.type != "name":
            self.fail("tag name expected", token.lineno)
        self._tag_stack.append(token.value)
        pop_tag = True
        try:
            if token.value in _statement_keywords:
                f = getattr(self, f"parse_{self.stream.current.value}")
                return f()  # type: ignore
            if token.value == "call":
                return self.parse_call_block()
            if token.value == "filter":
                return self.parse_filter_block()
            ext = self.extensions.get(token.value)
            if ext is not None:
                return ext(self)

            # did not work out, remove the token we pushed by accident
            # from the stack so that the unknown tag fail function can
            # produce a proper error message.
            self._tag_stack.pop()
            pop_tag = False
            self.fail_unknown_tag(token.value, token.lineno)
        finally:
            if pop_tag:
                self._tag_stack.pop()

    def parse_statements(
        self, end_tokens: t.Tuple[str, ...], drop_needle: bool = False
    ) -> t.List[nodes.Node]:
        """Parse multiple statements into a list until one of the end tokens
        is reached.  This is used to parse the body of statements as it also
        parses template data if appropriate.  The parser checks first if the
        current token is a colon and skips it if there is one.  Then it checks
        for the block end and parses until if one of the `end_tokens` is
        reached.  Per default the active token in the stream at the end of
        the call is the matched end token.  If this is not wanted `drop_needle`
        can be set to `True` and the end token is removed.
        """
        # the first token may be a colon for python compatibility
        self.stream.skip_if("colon")

        # in the future it would be possible to add whole code sections
        # by adding some sort of end of statement token and parsing those here.
        self.stream.expect("block_end")
        result = self.subparse(end_tokens)

        # we reached the end of the template too early, the subparser
        # does not check for this, so we do that now
        if self.stream.current.type == "eof":
            self.fail_eof(end_tokens)

        if drop_needle:
            next(self.stream)
        return result

    def parse_set(self) -> t.Union[nodes.Assign, nodes.AssignBlock]:
        """Parse an assign statement."""
        lineno = next(self.stream).lineno
        target = self.parse_assign_target(with_namespace=True)
        if self.stream.skip_if("assign"):
            expr = self.parse_tuple()
            return nodes.Assign(target, expr, lineno=lineno)
        filter_node = self.parse_filter(None)
        body = self.parse_statements(("name:endset",), drop_needle=True)
        return nodes.AssignBlock(target, filter_node, body, lineno=lineno)

    def parse_for(self) -> nodes.For:
        """Parse a for loop."""
        lineno = self.stream.expect("name:for").lineno
        target = self.parse_assign_target(extra_end_rules=("name:in",))
        self.stream.expect("name:in")
        iter = self.parse_tuple(
            with_condexpr=False, extra_end_rules=("name:recursive",)
        )
        test = None
        if self.stream.skip_if("name:if"):
            test = self.parse_expression()
        recursive = self.stream.skip_if("name:recursive")
        body = self.parse_statements(("name:endfor", "name:else"))
        if next(self.stream).value == "endfor":
            else_ = []
        else:
            else_ = self.parse_statements(("name:endfor",), drop_needle=True)
        return nodes.For(target, iter, body, else_, test, recursive, lineno=lineno)

    def parse_if(self) -> nodes.If:
        """Parse an if construct."""
        node = result = nodes.If(lineno=self.stream.expect("name:if").lineno)
        while True:
            node.test = self.parse_tuple(with_condexpr=False)
            node.body = self.parse_statements(("name:elif", "name:else", "name:endif"))
            node.elif_ = []
            node.else_ = []
            token = next(self.stream)
            if token.test("name:elif"):
                node = nodes.If(lineno=self.stream.current.lineno)
                result.elif_.append(node)
                continue
            elif token.test("name:else"):
                result.else_ = self.parse_statements(("name:endif",), drop_needle=True)
            break
        return result

    def parse_with(self) -> nodes.With:
        node = nodes.With(lineno=next(self.stream).lineno)
        targets: t.List[nodes.Expr] = []
        values: t.List[nodes.Expr] = []
        while self.stream.current.type != "block_end":
            if targets:
                self.stream.expect("comma")
            target = self.parse_assign_target()
            target.set_ctx("param")
            targets.append(target)
            self.stream.expect("assign")
            values.append(self.parse_expression())
        node.targets = targets
        node.values = values
        node.body = self.parse_statements(("name:endwith",), drop_needle=True)
        return node

    def parse_autoescape(self) -> nodes.Scope:
        node = nodes.ScopedEvalContextModifier(lineno=next(self.stream).lineno)
        node.options = [nodes.Keyword("autoescape", self.parse_expression())]
        node.body = self.parse_statements(("name:endautoescape",), drop_needle=True)
        return nodes.Scope([node])

    def parse_block(self) -> nodes.Block:
        node = nodes.Block(lineno=next(self.stream).lineno)
        node.name = self.stream.expect("name").value
        node.scoped = self.stream.skip_if("name:scoped")
        node.required = self.stream.skip_if("name:required")

        # common problem people encounter when switching from django
        # to jinja.  we do not support hyphens in block names, so let's
        # raise a nicer error message in that case.
        if self.stream.current.type == "sub":
            self.fail(
                "Block names in Jinja have to be valid Python identifiers and may not"
                " contain hyphens, use an underscore instead."
            )

        node.body = self.parse_statements(("name:endblock",), drop_needle=True)

        # enforce that required blocks only contain whitespace or comments
        # by asserting that the body, if not empty, is just TemplateData nodes
        # with whitespace data
        if node.required:
            for body_node in node.body:
                if not isinstance(body_node, nodes.Output) or any(
                    not isinstance(output_node, nodes.TemplateData)
                    or not output_node.data.isspace()
                    for output_node in body_node.nodes
                ):
                    self.fail("Required blocks can only contain comments or whitespace")

        self.stream.skip_if("name:" + node.name)
        return node

    def parse_extends(self) -> nodes.Extends:
        node = nodes.Extends(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        return node

    def parse_import_context(
        self, node: _ImportInclude, default: bool
    ) -> _ImportInclude:
        if self.stream.current.test_any(
            "name:with", "name:without"
        ) and self.stream.look().test("name:context"):
            node.with_context = next(self.stream).value == "with"
            self.stream.skip()
        else:
            node.with_context = default
        return node

    def parse_include(self) -> nodes.Include:
        node = nodes.Include(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        if self.stream.current.test("name:ignore") and self.stream.look().test(
            "name:missing"
        ):
            node.ignore_missing = True
            self.stream.skip(2)
        else:
            node.ignore_missing = False
        return self.parse_import_context(node, True)

    def parse_import(self) -> nodes.Import:
        node = nodes.Import(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect("name:as")
        node.target = self.parse_assign_target(name_only=True).name
        return self.parse_import_context(node, False)

    def parse_from(self) -> nodes.FromImport:
        node = nodes.FromImport(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect("name:import")
        node.names = []

        def parse_context() -> bool:
            if self.stream.current.value in {
                "with",
                "without",
            } and self.stream.look().test("name:context"):
                node.with_context = next(self.stream).value == "with"
                self.stream.skip()
                return True
            return False

        while True:
            if node.names:
                self.stream.expect("comma")
            if self.stream.current.type == "name":
                if parse_context():
                    break
                target = self.parse_assign_target(name_only=True)
                if target.name.startswith("_"):
                    self.fail(
                        "names starting with an underline can not be imported",
                        target.lineno,
                        exc=TemplateAssertionError,
                    )
                if self.stream.skip_if("name:as"):
                    alias = self.parse_assign_target(name_only=True)
                    node.names.append((target.name, alias.name))
                else:
                    node.names.append(target.name)
                if parse_context() or self.stream.current.type != "comma":
                    break
            else:
                self.stream.expect("name")
        if not hasattr(node, "with_context"):
            node.with_context = False
        return node

    def parse_signature(self, node: _MacroCall) -> None:
        args = node.args = []
        defaults = node.defaults = []
        self.stream.expect("lparen")
        while self.stream.current.type != "rparen":
            if args:
                self.stream.expect("comma")
            arg = self.parse_assign_target(name_only=True)
            arg.set_ctx("param")
            if self.stream.skip_if("assign"):
                defaults.append(self.parse_expression())
            elif defaults:
                self.fail("non-default argument follows default argument")
            args.append(arg)
        self.stream.expect("rparen")

    def parse_call_block(self) -> nodes.CallBlock:
        node = nodes.CallBlock(lineno=next(self.stream).lineno)
        if self.stream.current.type == "lparen":
            self.parse_signature(node)
        else:
            node.args = []
            node.defaults = []

        call_node = self.parse_expression()
        if not isinstance(call_node, nodes.Call):
            self.fail("expected call", node.lineno)
        node.call = call_node
        node.body = self.parse_statements(("name:endcall",), drop_needle=True)
        return node

    def parse_filter_block(self) -> nodes.FilterBlock:
        node = nodes.FilterBlock(lineno=next(self.stream).lineno)
        node.filter = self.parse_filter(None, start_inline=True)  # type: ignore
        node.body = self.parse_statements(("name:endfilter",), drop_needle=True)
        return node

    def parse_macro(self) -> nodes.Macro:
        node = nodes.Macro(lineno=next(self.stream).lineno)
        node.name = self.parse_assign_target(name_only=True).name
        self.parse_signature(node)
        node.body = self.parse_statements(("name:endmacro",), drop_needle=True)
        return node

    def parse_print(self) -> nodes.Output:
        node = nodes.Output(lineno=next(self.stream).lineno)
        node.nodes = []
        while self.stream.current.type != "block_end":
            if node.nodes:
                self.stream.expect("comma")
            node.nodes.append(self.parse_expression())
        return node

    @typing.overload
    def parse_assign_target(
        self, with_tuple: bool = ..., name_only: "te.Literal[True]" = ...
    ) -> nodes.Name: ...

    @typing.overload
    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        with_namespace: bool = False,
    ) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]: ...

    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        with_namespace: bool = False,
    ) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]:
        """Parse an assignment target.  As Jinja allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.  If
        `with_namespace` is enabled, a namespace assignment may be parsed.
        """
        target: nodes.Expr

        if name_only:
            token = self.stream.expect("name")
            target = nodes.Name(token.value, "store", lineno=token.lineno)
        else:
            if with_tuple:
                target = self.parse_tuple(
                    simplified=True,
                    extra_end_rules=extra_end_rules,
                    with_namespace=with_namespace,
                )
            else:
                target = self.parse_primary(with_namespace=with_namespace)

            target.set_ctx("store")

        if not target.can_assign():
            self.fail(
                f"can't assign to {type(target).__name__.lower()!r}", target.lineno
            )

        return target  # type: ignore

    def parse_expression(self, with_condexpr: bool = True) -> nodes.Expr:
        """Parse an expression.  Per default all expressions are parsed, if
        the optional `with_condexpr` parameter is set to `False` conditional
        expressions are not parsed.
        """
        if with_condexpr:
            return self.parse_condexpr()
        return self.parse_or()

    def parse_condexpr(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        expr1 = self.parse_or()
        expr3: t.Optional[nodes.Expr]

        while self.stream.skip_if("name:if"):
            expr2 = self.parse_or()
            if self.stream.skip_if("name:else"):
                expr3 = self.parse_condexpr()
            else:
                expr3 = None
            expr1 = nodes.CondExpr(expr2, expr1, expr3, lineno=lineno)
            lineno = self.stream.current.lineno
        return expr1

    def parse_or(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_and()
        while self.stream.skip_if("name:or"):
            right = self.parse_and()
            left = nodes.Or(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_and(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_not()
        while self.stream.skip_if("name:and"):
            right = self.parse_not()
            left = nodes.And(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_not(self) -> nodes.Expr:
        if self.stream.current.test("name:not"):
            lineno = next(self.stream).lineno
            return nodes.Not(self.parse_not(), lineno=lineno)
        return self.parse_compare()

    def parse_compare(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        expr = self.parse_math1()
        ops = []
        while True:
            token_type = self.stream.current.type
            if token_type in _compare_operators:
                next(self.stream)
                ops.append(nodes.Operand(token_type, self.parse_math1()))
            elif self.stream.skip_if("name:in"):
                ops.append(nodes.Operand("in", self.parse_math1()))
            elif self.stream.current.test("name:not") and self.stream.look().test(
                "name:in"
            ):
                self.stream.skip(2)
                ops.append(nodes.Operand("notin", self.parse_math1()))
            else:
                break
            lineno = self.stream.current.lineno
        if not ops:
            return expr
        return nodes.Compare(expr, ops, lineno=lineno)

    def parse_math1(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_concat()
        while self.stream.current.type in ("add", "sub"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_concat()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_concat(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        args = [self.parse_math2()]
        while self.stream.current.type == "tilde":
            next(self.stream)
            args.append(self.parse_math2())
        if len(args) == 1:
            return args[0]
        return nodes.Concat(args, lineno=lineno)

    def parse_math2(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_pow()
        while self.stream.current.type in ("mul", "div", "floordiv", "mod"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_pow()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_pow(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_unary()
        while self.stream.current.type == "pow":
            next(self.stream)
            right = self.parse_unary()
            left = nodes.Pow(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_unary(self, with_filter: bool = True) -> nodes.Expr:
        token_type = self.stream.current.type
        lineno = self.stream.current.lineno
        node: nodes.Expr

        if token_type == "sub":
            next(self.stream)
            node = nodes.Neg(self.parse_unary(False), lineno=lineno)
        elif token_type == "add":
            next(self.stream)
            node = nodes.Pos(self.parse_unary(False), lineno=lineno)
        else:
            node = self.parse_primary()
        node = self.parse_postfix(node)
        if with_filter:
            node = self.parse_filter_expr(node)
        return node

    def parse_primary(self, with_namespace: bool = False) -> nodes.Expr:
        """Parse a name or literal value. If ``with_namespace`` is enabled, also
        parse namespace attr refs, for use in assignments."""
        token = self.stream.current
        node: nodes.Expr
        if token.type == "name":
            next(self.stream)
            if token.value in ("true", "false", "True", "False"):
                node = nodes.Const(token.value in ("true", "True"), lineno=token.lineno)
            elif token.value in ("none", "None"):
                node = nodes.Const(None, lineno=token.lineno)
            elif with_namespace and self.stream.current.type == "dot":
                # If namespace attributes are allowed at this point, and the next
                # token is a dot, produce a namespace reference.
                next(self.stream)
                attr = self.stream.expect("name")
                node = nodes.NSRef(token.value, attr.value, lineno=token.lineno)
            else:
                node = nodes.Name(token.value, "load", lineno=token.lineno)
        elif token.type == "string":
            next(self.stream)
            buf = [token.value]
            lineno = token.lineno
            while self.stream.current.type == "string":
                buf.append(self.stream.current.value)
                next(self.stream)
            node = nodes.Const("".join(buf), lineno=lineno)
        elif token.type in ("integer", "float"):
            next(self.stream)
            node = nodes.Const(token.value, lineno=token.lineno)
        elif token.type == "lparen":
            next(self.stream)
            node = self.parse_tuple(explicit_parentheses=True)
            self.stream.expect("rparen")
        elif token.type == "lbracket":
            node = self.parse_list()
        elif token.type == "lbrace":
            node = self.parse_dict()
        else:
            self.fail(f"unexpected {describe_token(token)!r}", token.lineno)
        return node

    def parse_tuple(
        self,
        simplified: bool = False,
        with_condexpr: bool = True,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        explicit_parentheses: bool = False,
        with_namespace: bool = False,
    ) -> t.Union[nodes.Tuple, nodes.Expr]:
        """Works like `parse_expression` but if multiple expressions are
        delimited by a comma a :class:`~jinja2.nodes.Tuple` node is created.
        This method could also return a regular expression instead of a tuple
        if no commas where found.

        The default parsing mode is a full tuple.  If `simplified` is `True`
        only names and literals are parsed; ``with_namespace`` allows namespace
        attr refs as well. The `no_condexpr` parameter is forwarded to
        :meth:`parse_expression`.

        Because tuples do not require delimiters and may end in a bogus comma
        an extra hint is needed that marks the end of a tuple.  For example
        for loops support tuples between `for` and `in`.  In that case the
        `extra_end_rules` is set to ``['name:in']``.

        `explicit_parentheses` is true if the parsing was triggered by an
        expression in parentheses.  This is used to figure out if an empty
        tuple is a valid expression or not.
        """
        lineno = self.stream.current.lineno
        if simplified:

            def parse() -> nodes.Expr:
                return self.parse_primary(with_namespace=with_namespace)

        else:

            def parse() -> nodes.Expr:
                return self.parse_expression(with_condexpr=with_condexpr)

        args: t.List[nodes.Expr] = []
        is_tuple = False

        while True:
            if args:
                self.stream.expect("comma")
            if self.is_tuple_end(extra_end_rules):
                break
            args.append(parse())
            if self.stream.current.type == "comma":
                is_tuple = True
            else:
                break
            lineno = self.stream.current.lineno

        if not is_tuple:
            if args:
                return args[0]

            # if we don't have explicit parentheses, an empty tuple is
            # not a valid expression.  This would mean nothing (literally
            # nothing) in the spot of an expression would be an empty
            # tuple.
            if not explicit_parentheses:
                self.fail(
                    "Expected an expression,"
                    f" got {describe_token(self.stream.current)!r}"
                )

        return nodes.Tuple(args, "load", lineno=lineno)

    def parse_list(self) -> nodes.List:
        token = self.stream.expect("lbracket")
        items: t.List[nodes.Expr] = []
        while self.stream.current.type != "rbracket":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbracket":
                break
            items.append(self.parse_expression())
        self.stream.expect("rbracket")
        return nodes.List(items, lineno=token.lineno)

    def parse_dict(self) -> nodes.Dict:
        token = self.stream.expect("lbrace")
        items: t.List[nodes.Pair] = []
        while self.stream.current.type != "rbrace":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbrace":
                break
            key = self.parse_expression()
            self.stream.expect("colon")
            value = self.parse_expression()
            items.append(nodes.Pair(key, value, lineno=key.lineno))
        self.stream.expect("rbrace")
        return nodes.Dict(items, lineno=token.lineno)

    def parse_postfix(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "dot" or token_type == "lbracket":
                node = self.parse_subscript(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_filter_expr(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "pipe":
                node = self.parse_filter(node)  # type: ignore
            elif token_type == "name" and self.stream.current.value == "is":
                node = self.parse_test(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_subscript(
        self, node: nodes.Expr
    ) -> t.Union[nodes.Getattr, nodes.Getitem]:
        token = next(self.stream)
        arg: nodes.Expr

        if token.type == "dot":
            attr_token = self.stream.current
            next(self.stream)
            if attr_token.type == "name":
                return nodes.Getattr(
                    node, attr_token.value, "load", lineno=token.lineno
                )
            elif attr_token.type != "integer":
                self.fail("expected name or number", attr_token.lineno)
            arg = nodes.Const(attr_token.value, lineno=attr_token.lineno)
            return nodes.Getitem(node, arg, "load", lineno=token.lineno)
        if token.type == "lbracket":
            args: t.List[nodes.Expr] = []
            while self.stream.current.type != "rbracket":
                if args:
                    self.stream.expect("comma")
                args.append(self.parse_subscribed())
            self.stream.expect("rbracket")
            if len(args) == 1:
                arg = args[0]
            else:
                arg = nodes.Tuple(args, "load", lineno=token.lineno)
            return nodes.Getitem(node, arg, "load", lineno=token.lineno)
        self.fail("expected subscript expression", token.lineno)

    def parse_subscribed(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        args: t.List[t.Optional[nodes.Expr]]

        if self.stream.current.type == "colon":
            next(self.stream)
            args = [None]
        else:
            node = self.parse_expression()
            if self.stream.current.type != "colon":
                return node
            next(self.stream)
            args = [node]

        if self.stream.current.type == "colon":
            args.append(None)
        elif self.stream.current.type not in ("rbracket", "comma"):
            args.append(self.parse_expression())
        else:
            args.append(None)

        if self.stream.current.type == "colon":
            next(self.stream)
            if self.stream.current.type not in ("rbracket", "comma"):
                args.append(self.parse_expression())
            else:
                args.append(None)
        else:
            args.append(None)

        return nodes.Slice(lineno=lineno, *args)  # noqa: B026

    def parse_call_args(
        self,
    ) -> t.Tuple[
        t.List[nodes.Expr],
        t.List[nodes.Keyword],
        t.Optional[nodes.Expr],
        t.Optional[nodes.Expr],
    ]:
        token = self.stream.expect("lparen")
        args = []
        kwargs = []
        dyn_args = None
        dyn_kwargs = None
        require_comma = False

        def ensure(expr: bool) -> None:
            if not expr:
                self.fail("invalid syntax for function call expression", token.lineno)

        while self.stream.current.type != "rparen":
            if require_comma:
                self.stream.expect("comma")

                # support for trailing comma
                if self.stream.current.type == "rparen":
                    break

            if self.stream.current.type == "mul":
                ensure(dyn_args is None and dyn_kwargs is None)
                next(self.stream)
                dyn_args = self.parse_expression()
            elif self.stream.current.type == "pow":
                ensure(dyn_kwargs is None)
                next(self.stream)
                dyn_kwargs = self.parse_expression()
            else:
                if (
                    self.stream.current.type == "name"
                    and self.stream.look().type == "assign"
                ):
                    # Parsing a kwarg
                    ensure(dyn_kwargs is None)
                    key = self.stream.current.value
                    self.stream.skip(2)
                    value = self.parse_expression()
                    kwargs.append(nodes.Keyword(key, value, lineno=value.lineno))
                else:
                    # Parsing an arg
                    ensure(dyn_args is None and dyn_kwargs is None and not kwargs)
                    args.append(self.parse_expression())

            require_comma = True

        self.stream.expect("rparen")
        return args, kwargs, dyn_args, dyn_kwargs

    def parse_call(self, node: nodes.Expr) -> nodes.Call:
        # The lparen will be expected in parse_call_args, but the lineno
        # needs to be recorded before the stream is advanced.
        token = self.stream.current
        args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        return nodes.Call(node, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno)

    def parse_filter(
        self, node: t.Optional[nodes.Expr], start_inline: bool = False
    ) -> t.Optional[nodes.Expr]:
        while self.stream.current.type == "pipe" or start_inline:
            if not start_inline:
                next(self.stream)
            token = self.stream.expect("name")
            name = token.value
            while self.stream.current.type == "dot":
                next(self.stream)
                name += "." + self.stream.expect("name").value
            if self.stream.current.type == "lparen":
                args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
            else:
                args = []
                kwargs = []
                dyn_args = dyn_kwargs = None
            node = nodes.Filter(
                node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno
            )
            start_inline = False
        return node

    def parse_test(self, node: nodes.Expr) -> nodes.Expr:
        token = next(self.stream)
        if self.stream.current.test("name:not"):
            next(self.stream)
            negated = True
        else:
            negated = False
        name = self.stream.expect("name").value
        while self.stream.current.type == "dot":
            next(self.stream)
            name += "." + self.stream.expect("name").value
        dyn_args = dyn_kwargs = None
        kwargs: t.List[nodes.Keyword] = []
        if self.stream.current.type == "lparen":
            args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        elif self.stream.current.type in {
            "name",
            "string",
            "integer",
            "float",
            "lparen",
            "lbracket",
            "lbrace",
        } and not self.stream.current.test_any("name:else", "name:or", "name:and"):
            if self.stream.current.test("name:is"):
                self.fail("You cannot chain multiple tests with is")
            arg_node = self.parse_primary()
            arg_node = self.parse_postfix(arg_node)
            args = [arg_node]
        else:
            args = []
        node = nodes.Test(
            node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno
        )
        if negated:
            node = nodes.Not(node, lineno=token.lineno)
        return node

    def subparse(
        self, end_tokens: t.Optional[t.Tuple[str, ...]] = None
    ) -> t.List[nodes.Node]:
        body: t.List[nodes.Node] = []
        data_buffer: t.List[nodes.Node] = []
        add_data = data_buffer.append

        if end_tokens is not None:
            self._end_token_stack.append(end_tokens)

        def flush_data() -> None:
            if data_buffer:
                lineno = data_buffer[0].lineno
                body.append(nodes.Output(data_buffer[:], lineno=lineno))
                del data_buffer[:]

        try:
            while self.stream:
                token = self.stream.current
                if token.type == "data":
                    if token.value:
                        add_data(nodes.TemplateData(token.value, lineno=token.lineno))
                    next(self.stream)
                elif token.type == "variable_begin":
                    next(self.stream)
                    add_data(self.parse_tuple(with_condexpr=True))
                    self.stream.expect("variable_end")
                elif token.type == "block_begin":
                    flush_data()
                    next(self.stream)
                    if end_tokens is not None and self.stream.current.test_any(
                        *end_tokens
                    ):
                        return body
                    rv = self.parse_statement()
                    if isinstance(rv, list):
                        body.extend(rv)
                    else:
                        body.append(rv)
                    self.stream.expect("block_end")
                else:
                    raise AssertionError("internal parsing error")

            flush_data()
        finally:
            if end_tokens is not None:
                self._end_token_stack.pop()
        return body

    def parse(self) -> nodes.Template:
        """Parse the whole template into a `Template` node."""
        result = nodes.Template(self.subparse(), lineno=1)
        result.set_environment(self.environment)
        return result
<|endoftext|>import typing as t

from . import nodes
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    import typing_extensions as te

VAR_LOAD_PARAMETER = "param"
VAR_LOAD_RESOLVE = "resolve"
VAR_LOAD_ALIAS = "alias"
VAR_LOAD_UNDEFINED = "undefined"


def find_symbols(
    nodes: t.Iterable[nodes.Node], parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    visitor = FrameSymbolVisitor(sym)
    for node in nodes:
        visitor.visit(node)
    return sym


def symbols_for_node(
    node: nodes.Node, parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    sym.analyze_node(node)
    return sym


class Symbols:
    def __init__(
        self, parent: t.Optional["Symbols"] = None, level: t.Optional[int] = None
    ) -> None:
        if level is None:
            if parent is None:
                level = 0
            else:
                level = parent.level + 1

        self.level: int = level
        self.parent = parent
        self.refs: t.Dict[str, str] = {}
        self.loads: t.Dict[str, t.Any] = {}
        self.stores: t.Set[str] = set()

    def analyze_node(self, node: nodes.Node, **kwargs: t.Any) -> None:
        visitor = RootVisitor(self)
        visitor.visit(node, **kwargs)

    def _define_ref(
        self, name: str, load: t.Optional[t.Tuple[str, t.Optional[str]]] = None
    ) -> str:
        ident = f"l_{self.level}_{name}"
        self.refs[name] = ident
        if load is not None:
            self.loads[ident] = load
        return ident

    def find_load(self, target: str) -> t.Optional[t.Any]:
        if target in self.loads:
            return self.loads[target]

        if self.parent is not None:
            return self.parent.find_load(target)

        return None

    def find_ref(self, name: str) -> t.Optional[str]:
        if name in self.refs:
            return self.refs[name]

        if self.parent is not None:
            return self.parent.find_ref(name)

        return None

    def ref(self, name: str) -> str:
        rv = self.find_ref(name)
        if rv is None:
            raise AssertionError(
                "Tried to resolve a name to a reference that was"
                f" unknown to the frame ({name!r})"
            )
        return rv

    def copy(self) -> "te.Self":
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.refs = self.refs.copy()
        rv.loads = self.loads.copy()
        rv.stores = self.stores.copy()
        return rv

    def store(self, name: str) -> None:
        self.stores.add(name)

        # If we have not see the name referenced yet, we need to figure
        # out what to set it to.
        if name not in self.refs:
            # If there is a parent scope we check if the name has a
            # reference there.  If it does it means we might have to alias
            # to a variable there.
            if self.parent is not None:
                outer_ref = self.parent.find_ref(name)
                if outer_ref is not None:
                    self._define_ref(name, load=(VAR_LOAD_ALIAS, outer_ref))
                    return

            # Otherwise we can just set it to undefined.
            self._define_ref(name, load=(VAR_LOAD_UNDEFINED, None))

    def declare_parameter(self, name: str) -> str:
        self.stores.add(name)
        return self._define_ref(name, load=(VAR_LOAD_PARAMETER, None))

    def load(self, name: str) -> None:
        if self.find_ref(name) is None:
            self._define_ref(name, load=(VAR_LOAD_RESOLVE, name))

    def branch_update(self, branch_symbols: t.Sequence["Symbols"]) -> None:
        stores: t.Set[str] = set()

        for branch in branch_symbols:
            stores.update(branch.stores)

        stores.difference_update(self.stores)

        for sym in branch_symbols:
            self.refs.update(sym.refs)
            self.loads.update(sym.loads)
            self.stores.update(sym.stores)

        for name in stores:
            target = self.find_ref(name)
            assert target is not None, "should not happen"

            if self.parent is not None:
                outer_target = self.parent.find_ref(name)
                if outer_target is not None:
                    self.loads[target] = (VAR_LOAD_ALIAS, outer_target)
                    continue
            self.loads[target] = (VAR_LOAD_RESOLVE, name)

    def dump_stores(self) -> t.Dict[str, str]:
        rv: t.Dict[str, str] = {}
        node: t.Optional[Symbols] = self

        while node is not None:
            for name in sorted(node.stores):
                if name not in rv:
                    rv[name] = self.find_ref(name)  # type: ignore

            node = node.parent

        return rv

    def dump_param_targets(self) -> t.Set[str]:
        rv = set()
        node: t.Optional[Symbols] = self

        while node is not None:
            for target, (instr, _) in self.loads.items():
                if instr == VAR_LOAD_PARAMETER:
                    rv.add(target)

            node = node.parent

        return rv


class RootVisitor(NodeVisitor):
    def __init__(self, symbols: "Symbols") -> None:
        self.sym_visitor = FrameSymbolVisitor(symbols)

    def _simple_visit(self, node: nodes.Node, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes():
            self.sym_visitor.visit(child)

    visit_Template = _simple_visit
    visit_Block = _simple_visit
    visit_Macro = _simple_visit
    visit_FilterBlock = _simple_visit
    visit_Scope = _simple_visit
    visit_If = _simple_visit
    visit_ScopedEvalContextModifier = _simple_visit

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes(exclude=("call",)):
            self.sym_visitor.visit(child)

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_For(
        self, node: nodes.For, for_branch: str = "body", **kwargs: t.Any
    ) -> None:
        if for_branch == "body":
            self.sym_visitor.visit(node.target, store_as_param=True)
            branch = node.body
        elif for_branch == "else":
            branch = node.else_
        elif for_branch == "test":
            self.sym_visitor.visit(node.target, store_as_param=True)
            if node.test is not None:
                self.sym_visitor.visit(node.test)
            return
        else:
            raise RuntimeError("Unknown for branch")

        if branch:
            for item in branch:
                self.sym_visitor.visit(item)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.targets:
            self.sym_visitor.visit(target)
        for child in node.body:
            self.sym_visitor.visit(child)

    def generic_visit(self, node: nodes.Node, *args: t.Any, **kwargs: t.Any) -> None:
        raise NotImplementedError(f"Cannot find symbols for {type(node).__name__!r}")


class FrameSymbolVisitor(NodeVisitor):
    """A visitor for `Frame.inspect`."""

    def __init__(self, symbols: "Symbols") -> None:
        self.symbols = symbols

    def visit_Name(
        self, node: nodes.Name, store_as_param: bool = False, **kwargs: t.Any
    ) -> None:
        """All assignments to names go through this function."""
        if store_as_param or node.ctx == "param":
            self.symbols.declare_parameter(node.name)
        elif node.ctx == "store":
            self.symbols.store(node.name)
        elif node.ctx == "load":
            self.symbols.load(node.name)

    def visit_NSRef(self, node: nodes.NSRef, **kwargs: t.Any) -> None:
        self.symbols.load(node.name)

    def visit_If(self, node: nodes.If, **kwargs: t.Any) -> None:
        self.visit(node.test, **kwargs)
        original_symbols = self.symbols

        def inner_visit(nodes: t.Iterable[nodes.Node]) -> "Symbols":
            self.symbols = rv = original_symbols.copy()

            for subnode in nodes:
                self.visit(subnode, **kwargs)

            self.symbols = original_symbols
            return rv

        body_symbols = inner_visit(node.body)
        elif_symbols = inner_visit(node.elif_)
        else_symbols = inner_visit(node.else_ or ())
        self.symbols.branch_update([body_symbols, elif_symbols, else_symbols])

    def visit_Macro(self, node: nodes.Macro, **kwargs: t.Any) -> None:
        self.symbols.store(node.name)

    def visit_Import(self, node: nodes.Import, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)
        self.symbols.store(node.target)

    def visit_FromImport(self, node: nodes.FromImport, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)

        for name in node.names:
            if isinstance(name, tuple):
                self.symbols.store(name[1])
            else:
                self.symbols.store(name)

    def visit_Assign(self, node: nodes.Assign, **kwargs: t.Any) -> None:
        """Visit assignments in the correct order."""
        self.visit(node.node, **kwargs)
        self.visit(node.target, **kwargs)

    def visit_For(self, node: nodes.For, **kwargs: t.Any) -> None:
        """Visiting stops at for blocks.  However the block sequence
        is visited as part of the outer scope.
        """
        self.visit(node.iter, **kwargs)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        self.visit(node.call, **kwargs)

    def visit_FilterBlock(self, node: nodes.FilterBlock, **kwargs: t.Any) -> None:
        self.visit(node.filter, **kwargs)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.values:
            self.visit(target)

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        """Stop visiting at block assigns."""
        self.visit(node.target, **kwargs)

    def visit_Scope(self, node: nodes.Scope, **kwargs: t.Any) -> None:
        """Stop visiting at scopes."""

    def visit_Block(self, node: nodes.Block, **kwargs: t.Any) -> None:
        """Stop visiting at blocks."""

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        """Do not visit into overlay scopes."""
<|endoftext|>"""Functions that expose information about templates that might be
interesting for introspection.
"""

import typing as t

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame

if t.TYPE_CHECKING:
    from .environment import Environment


class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment: "Environment") -> None:
        super().__init__(environment, "<introspection>", "<introspection>")
        self.undeclared_identifiers: t.Set[str] = set()

    def write(self, x: str) -> None:
        """Don't write."""

    def enter_frame(self, frame: Frame) -> None:
        """Remember all undeclared identifiers."""
        super().enter_frame(frame)

        for _, (action, param) in frame.symbols.loads.items():
            if action == "resolve" and param not in self.environment.globals:
                self.undeclared_identifiers.add(param)


def find_undeclared_variables(ast: nodes.Template) -> t.Set[str]:
    """Returns a set of all variables in the AST that will be looked up from
    the context at runtime.  Because at compile time it's not known which
    variables will be used depending on the path the execution takes at
    runtime, all variables are returned.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
    >>> meta.find_undeclared_variables(ast) == {'bar'}
    True

    .. admonition:: Implementation

       Internally the code generator is used for finding undeclared variables.
       This is good to know because the code generator might raise a
       :exc:`TemplateAssertionError` during compilation and as a matter of
       fact this function can currently raise that exception as well.
    """
    codegen = TrackingCodeGenerator(ast.environment)  # type: ignore
    codegen.visit(ast)
    return codegen.undeclared_identifiers


_ref_types = (nodes.Extends, nodes.FromImport, nodes.Import, nodes.Include)
_RefType = t.Union[nodes.Extends, nodes.FromImport, nodes.Import, nodes.Include]


def find_referenced_templates(ast: nodes.Template) -> t.Iterator[t.Optional[str]]:
    """Finds all the referenced templates from the AST.  This will return an
    iterator over all the hardcoded template extensions, inclusions and
    imports.  If dynamic inheritance or inclusion is used, `None` will be
    yielded.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% extends "layout.html" %}{% include helper %}')
    >>> list(meta.find_referenced_templates(ast))
    ['layout.html', None]

    This function is useful for dependency tracking.  For example if you want
    to rebuild parts of the website after a layout template has changed.
    """
    template_name: t.Any

    for node in ast.find_all(_ref_types):
        template: nodes.Expr = node.template  # type: ignore

        if not isinstance(template, nodes.Const):
            # a tuple with some non consts in there
            if isinstance(template, (nodes.Tuple, nodes.List)):
                for template_name in template.items:
                    # something const, only yield the strings and ignore
                    # non-string consts that really just make no sense
                    if isinstance(template_name, nodes.Const):
                        if isinstance(template_name.value, str):
                            yield template_name.value
                    # something dynamic in there
                    else:
                        yield None
            # something dynamic we don't know about here
            else:
                yield None
            continue
        # constant is a basestring, direct template name
        if isinstance(template.value, str):
            yield template.value
        # a tuple or list (latter *should* not happen) made of consts,
        # yield the consts that are strings.  We could warn here for
        # non string values
        elif isinstance(node, nodes.Include) and isinstance(
            template.value, (tuple, list)
        ):
            for template_name in template.value:
                if isinstance(template_name, str):
                    yield template_name
        # something else we don't care about, we could warn here
        else:
            yield None
<|endoftext|>import typing as t

from .filters import FILTERS as DEFAULT_FILTERS  # noqa: F401
from .tests import TESTS as DEFAULT_TESTS  # noqa: F401
from .utils import Cycler
from .utils import generate_lorem_ipsum
from .utils import Joiner
from .utils import Namespace

if t.TYPE_CHECKING:
    import typing_extensions as te

# defaults for the parser / lexer
BLOCK_START_STRING = "{%"
BLOCK_END_STRING = "%}"
VARIABLE_START_STRING = "{{"
VARIABLE_END_STRING = "}}"
COMMENT_START_STRING = "{#"
COMMENT_END_STRING = "#}"
LINE_STATEMENT_PREFIX: t.Optional[str] = None
LINE_COMMENT_PREFIX: t.Optional[str] = None
TRIM_BLOCKS = False
LSTRIP_BLOCKS = False
NEWLINE_SEQUENCE: "te.Literal['\\n', '\\r\\n', '\\r']" = "\n"
KEEP_TRAILING_NEWLINE = False

# default filters, tests and namespace

DEFAULT_NAMESPACE = {
    "range": range,
    "dict": dict,
    "lipsum": generate_lorem_ipsum,
    "cycler": Cycler,
    "joiner": Joiner,
    "namespace": Namespace,
}

# default policies
DEFAULT_POLICIES: t.Dict[str, t.Any] = {
    "compiler.ascii_str": True,
    "urlize.rel": "noopener",
    "urlize.target": None,
    "urlize.extra_schemes": None,
    "truncate.leeway": 5,
    "json.dumps_function": None,
    "json.dumps_kwargs": {"sort_keys": True},
    "ext.i18n.trimmed": False,
}
<|endoftext|>import typing as t

if t.TYPE_CHECKING:
    from .runtime import Undefined


class TemplateError(Exception):
    """Baseclass for all template errors."""

    def __init__(self, message: t.Optional[str] = None) -> None:
        super().__init__(message)

    @property
    def message(self) -> t.Optional[str]:
        return self.args[0] if self.args else None


class TemplateNotFound(IOError, LookupError, TemplateError):
    """Raised if a template does not exist.

    .. versionchanged:: 2.11
        If the given name is :class:`Undefined` and no message was
        provided, an :exc:`UndefinedError` is raised.
    """

    # Silence the Python warning about message being deprecated since
    # it's not valid here.
    message: t.Optional[str] = None

    def __init__(
        self,
        name: t.Optional[t.Union[str, "Undefined"]],
        message: t.Optional[str] = None,
    ) -> None:
        IOError.__init__(self, name)

        if message is None:
            from .runtime import Undefined

            if isinstance(name, Undefined):
                name._fail_with_undefined_error()

            message = name

        self.message = message
        self.name = name
        self.templates = [name]

    def __str__(self) -> str:
        return str(self.message)


class TemplatesNotFound(TemplateNotFound):
    """Like :class:`TemplateNotFound` but raised if multiple templates
    are selected.  This is a subclass of :class:`TemplateNotFound`
    exception, so just catching the base exception will catch both.

    .. versionchanged:: 2.11
        If a name in the list of names is :class:`Undefined`, a message
        about it being undefined is shown rather than the empty string.

    .. versionadded:: 2.2
    """

    def __init__(
        self,
        names: t.Sequence[t.Union[str, "Undefined"]] = (),
        message: t.Optional[str] = None,
    ) -> None:
        if message is None:
            from .runtime import Undefined

            parts = []

            for name in names:
                if isinstance(name, Undefined):
                    parts.append(name._undefined_message)
                else:
                    parts.append(name)

            parts_str = ", ".join(map(str, parts))
            message = f"none of the templates given were found: {parts_str}"

        super().__init__(names[-1] if names else None, message)
        self.templates = list(names)


class TemplateSyntaxError(TemplateError):
    """Raised to tell the user that there is a problem with the template."""

    def __init__(
        self,
        message: str,
        lineno: int,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.lineno = lineno
        self.name = name
        self.filename = filename
        self.source: t.Optional[str] = None

        # this is set to True if the debug.translate_syntax_error
        # function translated the syntax error into a new traceback
        self.translated = False

    def __str__(self) -> str:
        # for translated errors we only return the message
        if self.translated:
            return t.cast(str, self.message)

        # otherwise attach some stuff
        location = f"line {self.lineno}"
        name = self.filename or self.name
        if name:
            location = f'File "{name}", {location}'
        lines = [t.cast(str, self.message), "  " + location]

        # if the source is set, add the line to the output
        if self.source is not None:
            try:
                line = self.source.splitlines()[self.lineno - 1]
            except IndexError:
                pass
            else:
                lines.append("    " + line.strip())

        return "\n".join(lines)

    def __reduce__(self):  # type: ignore
        # https://bugs.python.org/issue1692335 Exceptions that take
        # multiple required arguments have problems with pickling.
        # Without this, raises TypeError: __init__() missing 1 required
        # positional argument: 'lineno'
        return self.__class__, (self.message, self.lineno, self.name, self.filename)


class TemplateAssertionError(TemplateSyntaxError):
    """Like a template syntax error, but covers cases where something in the
    template caused an error at compile time that wasn't necessarily caused
    by a syntax error.  However it's a direct subclass of
    :exc:`TemplateSyntaxError` and has the same attributes.
    """


class TemplateRuntimeError(TemplateError):
    """A generic runtime error in the template engine.  Under some situations
    Jinja may raise this exception.
    """


class UndefinedError(TemplateRuntimeError):
    """Raised if a template tries to operate on :class:`Undefined`."""


class SecurityError(TemplateRuntimeError):
    """Raised if a template tries to do something insecure if the
    sandbox is enabled.
    """


class FilterArgumentError(TemplateRuntimeError):
    """This error is raised if a filter was called with inappropriate
    arguments
    """
<|endoftext|># SPDX-License-Identifier: MIT


from functools import total_ordering

from ._funcs import astuple
from ._make import attrib, attrs


@total_ordering
@attrs(eq=False, order=False, slots=True, frozen=True)
class VersionInfo:
    """
    A version object that can be compared to tuple of length 1--4:

    >>> attr.VersionInfo(19, 1, 0, "final")  <= (19, 2)
    True
    >>> attr.VersionInfo(19, 1, 0, "final") < (19, 1, 1)
    True
    >>> vi = attr.VersionInfo(19, 2, 0, "final")
    >>> vi < (19, 1, 1)
    False
    >>> vi < (19,)
    False
    >>> vi == (19, 2,)
    True
    >>> vi == (19, 2, 1)
    False

    .. versionadded:: 19.2
    """

    year = attrib(type=int)
    minor = attrib(type=int)
    micro = attrib(type=int)
    releaselevel = attrib(type=str)

    @classmethod
    def _from_version_string(cls, s):
        """
        Parse *s* and return a _VersionInfo.
        """
        v = s.split(".")
        if len(v) == 3:
            v.append("final")

        return cls(
            year=int(v[0]), minor=int(v[1]), micro=int(v[2]), releaselevel=v[3]
        )

    def _ensure_tuple(self, other):
        """
        Ensure *other* is a tuple of a valid length.

        Returns a possibly transformed *other* and ourselves as a tuple of
        the same length as *other*.
        """

        if self.__class__ is other.__class__:
            other = astuple(other)

        if not isinstance(other, tuple):
            raise NotImplementedError

        if not (1 <= len(other) <= 4):
            raise NotImplementedError

        return astuple(self)[: len(other)], other

    def __eq__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        return us == them

    def __lt__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        # Since alphabetically "dev0" < "final" < "post1" < "post2", we don't
        # have to do anything special with releaselevel for now.
        return us < them

    def __hash__(self):
        return hash((self.year, self.minor, self.micro, self.releaselevel))
<|endoftext|># SPDX-License-Identifier: MIT

from __future__ import annotations

import abc
import contextlib
import copy
import enum
import inspect
import itertools
import linecache
import sys
import types
import unicodedata
import weakref

from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, NamedTuple, TypeVar

# We need to import _compat itself in addition to the _compat members to avoid
# having the thread-local in the globals here.
from . import _compat, _config, setters
from ._compat import (
    PY_3_10_PLUS,
    PY_3_11_PLUS,
    PY_3_13_PLUS,
    _AnnotationExtractor,
    _get_annotations,
    get_generic_base,
)
from .exceptions import (
    DefaultAlreadySetError,
    FrozenInstanceError,
    NotAnAttrsClassError,
    UnannotatedAttributeError,
)


# This is used at least twice, so cache it here.
_OBJ_SETATTR = object.__setattr__
_INIT_FACTORY_PAT = "__attr_factory_%s"
_CLASSVAR_PREFIXES = (
    "typing.ClassVar",
    "t.ClassVar",
    "ClassVar",
    "typing_extensions.ClassVar",
)
# we don't use a double-underscore prefix because that triggers
# name mangling when trying to create a slot for the field
# (when slots=True)
_HASH_CACHE_FIELD = "_attrs_cached_hash"

_EMPTY_METADATA_SINGLETON = types.MappingProxyType({})

# Unique object for unequivocal getattr() defaults.
_SENTINEL = object()

_DEFAULT_ON_SETATTR = setters.pipe(setters.convert, setters.validate)


class _Nothing(enum.Enum):
    """
    Sentinel to indicate the lack of a value when `None` is ambiguous.

    If extending attrs, you can use ``typing.Literal[NOTHING]`` to show
    that a value may be ``NOTHING``.

    .. versionchanged:: 21.1.0 ``bool(NOTHING)`` is now False.
    .. versionchanged:: 22.2.0 ``NOTHING`` is now an ``enum.Enum`` variant.
    """

    NOTHING = enum.auto()

    def __repr__(self):
        return "NOTHING"

    def __bool__(self):
        return False


NOTHING = _Nothing.NOTHING
"""
Sentinel to indicate the lack of a value when `None` is ambiguous.

When using in 3rd party code, use `attrs.NothingType` for type annotations.
"""


class _CacheHashWrapper(int):
    """
    An integer subclass that pickles / copies as None

    This is used for non-slots classes with ``cache_hash=True``, to avoid
    serializing a potentially (even likely) invalid hash value. Since `None`
    is the default value for uncalculated hashes, whenever this is copied,
    the copy's value for the hash should automatically reset.

    See GH #613 for more details.
    """

    def __reduce__(self, _none_constructor=type(None), _args=()):  # noqa: B008
        return _none_constructor, _args


def attrib(
    default=NOTHING,
    validator=None,
    repr=True,
    cmp=None,
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
    Create a new field / attribute on a class.

    Identical to `attrs.field`, except it's not keyword-only.

    Consider using `attrs.field` in new code (``attr.ib`` will *never* go away,
    though).

    ..  warning::

        Does **nothing** unless the class is also decorated with
        `attr.s` (or similar)!


    .. versionadded:: 15.2.0 *convert*
    .. versionadded:: 16.3.0 *metadata*
    .. versionchanged:: 17.1.0 *validator* can be a ``list`` now.
    .. versionchanged:: 17.1.0
       *hash* is `None` and therefore mirrors *eq* by default.
    .. versionadded:: 17.3.0 *type*
    .. deprecated:: 17.4.0 *convert*
    .. versionadded:: 17.4.0
       *converter* as a replacement for the deprecated *convert* to achieve
       consistency with other noun-based arguments.
    .. versionadded:: 18.1.0
       ``factory=f`` is syntactic sugar for ``default=attr.Factory(f)``.
    .. versionadded:: 18.2.0 *kw_only*
    .. versionchanged:: 19.2.0 *convert* keyword argument removed.
    .. versionchanged:: 19.2.0 *repr* also accepts a custom callable.
    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
    .. versionadded:: 19.2.0 *eq* and *order*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.3.0 *kw_only* backported to Python 2
    .. versionchanged:: 21.1.0
       *eq*, *order*, and *cmp* also accept a custom callable
    .. versionchanged:: 21.1.0 *cmp* undeprecated
    .. versionadded:: 22.2.0 *alias*
    .. versionchanged:: 25.4.0
       *kw_only* can now be None, and its default is also changed from False to
       None.
    """
    eq, eq_key, order, order_key = _determine_attrib_eq_order(
        cmp, eq, order, True
    )

    if hash is not None and hash is not True and hash is not False:
        msg = "Invalid value for hash.  Must be True, False, or None."
        raise TypeError(msg)

    if factory is not None:
        if default is not NOTHING:
            msg = (
                "The `default` and `factory` arguments are mutually exclusive."
            )
            raise ValueError(msg)
        if not callable(factory):
            msg = "The `factory` argument must be a callable."
            raise ValueError(msg)
        default = Factory(factory)

    if metadata is None:
        metadata = {}

    # Apply syntactic sugar by auto-wrapping.
    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)

    if validator and isinstance(validator, (list, tuple)):
        validator = and_(*validator)

    if converter and isinstance(converter, (list, tuple)):
        converter = pipe(*converter)

    return _CountingAttr(
        default=default,
        validator=validator,
        repr=repr,
        cmp=None,
        hash=hash,
        init=init,
        converter=converter,
        metadata=metadata,
        type=type,
        kw_only=kw_only,
        eq=eq,
        eq_key=eq_key,
        order=order,
        order_key=order_key,
        on_setattr=on_setattr,
        alias=alias,
    )


def _compile_and_eval(
    script: str,
    globs: dict[str, Any] | None,
    locs: Mapping[str, object] | None = None,
    filename: str = "",
) -> None:
    """
    Evaluate the script with the given global (globs) and local (locs)
    variables.
    """
    bytecode = compile(script, filename, "exec")
    eval(bytecode, globs, locs)


def _linecache_and_compile(
    script: str,
    filename: str,
    globs: dict[str, Any] | None,
    locals: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """
    Cache the script with _linecache_, compile it and return the _locals_.
    """

    locs = {} if locals is None else locals

    # In order of debuggers like PDB being able to step through the code,
    # we add a fake linecache entry.
    count = 1
    base_filename = filename
    while True:
        linecache_tuple = (
            len(script),
            None,
            script.splitlines(True),
            filename,
        )
        old_val = linecache.cache.setdefault(filename, linecache_tuple)
        if old_val == linecache_tuple:
            break

        filename = f"{base_filename[:-1]}-{count}>"
        count += 1

    _compile_and_eval(script, globs, locs, filename)

    return locs


def _make_attr_tuple_class(cls_name: str, attr_names: list[str]) -> type:
    """
    Create a tuple subclass to hold `Attribute`s for an `attrs` class.

    The subclass is a bare tuple with properties for names.

    class MyClassAttributes(tuple):
        __slots__ = ()
        x = property(itemgetter(0))
    """
    attr_class_name = f"{cls_name}Attributes"
    body = {}
    for i, attr_name in enumerate(attr_names):

        def getter(self, i=i):
            return self[i]

        body[attr_name] = property(getter)
    return type(attr_class_name, (tuple,), body)


# Tuple class for extracted attributes from a class definition.
# `base_attrs` is a subset of `attrs`.
class _Attributes(NamedTuple):
    attrs: type
    base_attrs: list[Attribute]
    base_attrs_map: dict[str, type]


def _is_class_var(annot):
    """
    Check whether *annot* is a typing.ClassVar.

    The string comparison hack is used to avoid evaluating all string
    annotations which would put attrs-based classes at a performance
    disadvantage compared to plain old classes.
    """
    annot = str(annot)

    # Annotation can be quoted.
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]

    return annot.startswith(_CLASSVAR_PREFIXES)


def _has_own_attribute(cls, attrib_name):
    """
    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).
    """
    return attrib_name in cls.__dict__


def _collect_base_attrs(
    cls, taken_attr_names
) -> tuple[list[Attribute], dict[str, type]]:
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.
    """
    base_attrs = []
    base_attr_map = {}  # A dictionary of base attrs to their classes.

    # Traverse the MRO and collect attributes.
    for base_cls in reversed(cls.__mro__[1:-1]):
        for a in getattr(base_cls, "__attrs_attrs__", []):
            if a.inherited or a.name in taken_attr_names:
                continue

            a = a.evolve(inherited=True)  # noqa: PLW2901
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls

    # For each name, only keep the freshest definition i.e. the furthest at the
    # back.  base_attr_map is fine because it gets overwritten with every new
    # instance.
    filtered = []
    seen = set()
    for a in reversed(base_attrs):
        if a.name in seen:
            continue
        filtered.insert(0, a)
        seen.add(a.name)

    return filtered, base_attr_map


def _collect_base_attrs_broken(cls, taken_attr_names):
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.

    N.B. *taken_attr_names* will be mutated.

    Adhere to the old incorrect behavior.

    Notably it collects from the front and considers inherited attributes which
    leads to the buggy behavior reported in #428.
    """
    base_attrs = []
    base_attr_map = {}  # A dictionary of base attrs to their classes.

    # Traverse the MRO and collect attributes.
    for base_cls in cls.__mro__[1:-1]:
        for a in getattr(base_cls, "__attrs_attrs__", []):
            if a.name in taken_attr_names:
                continue

            a = a.evolve(inherited=True)  # noqa: PLW2901
            taken_attr_names.add(a.name)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls

    return base_attrs, base_attr_map


def _transform_attrs(
    cls,
    these,
    auto_attribs,
    kw_only,
    collect_by_mro,
    field_transformer,
) -> _Attributes:
    """
    Transform all `_CountingAttr`s on a class into `Attribute`s.

    If *these* is passed, use that and don't look for them on the class.

    If *collect_by_mro* is True, collect them in the correct MRO order,
    otherwise use the old -- incorrect -- order.  See #428.

    Return an `_Attributes`.
    """
    cd = cls.__dict__
    anns = _get_annotations(cls)

    if these is not None:
        ca_list = list(these.items())
    elif auto_attribs is True:
        ca_names = {
            name
            for name, attr in cd.items()
            if attr.__class__ is _CountingAttr
        }
        ca_list = []
        annot_names = set()
        for attr_name, type in anns.items():
            if _is_class_var(type):
                continue
            annot_names.add(attr_name)
            a = cd.get(attr_name, NOTHING)

            if a.__class__ is not _CountingAttr:
                a = attrib(a)
            ca_list.append((attr_name, a))

        unannotated = ca_names - annot_names
        if unannotated:
            raise UnannotatedAttributeError(
                "The following `attr.ib`s lack a type annotation: "
                + ", ".join(
                    sorted(unannotated, key=lambda n: cd.get(n).counter)
                )
                + "."
            )
    else:
        ca_list = sorted(
            (
                (name, attr)
                for name, attr in cd.items()
                if attr.__class__ is _CountingAttr
            ),
            key=lambda e: e[1].counter,
        )

    fca = Attribute.from_counting_attr
    no = ClassProps.KeywordOnly.NO
    own_attrs = [
        fca(
            attr_name,
            ca,
            kw_only is not no,
            anns.get(attr_name),
        )
        for attr_name, ca in ca_list
    ]

    if collect_by_mro:
        base_attrs, base_attr_map = _collect_base_attrs(
            cls, {a.name for a in own_attrs}
        )
    else:
        base_attrs, base_attr_map = _collect_base_attrs_broken(
            cls, {a.name for a in own_attrs}
        )

    if kw_only is ClassProps.KeywordOnly.FORCE:
        own_attrs = [a.evolve(kw_only=True) for a in own_attrs]
        base_attrs = [a.evolve(kw_only=True) for a in base_attrs]

    attrs = base_attrs + own_attrs

    # Resolve default field alias before executing field_transformer, so that
    # the transformer receives fully populated Attribute objects with usable
    # alias values.
    for a in attrs:
        if not a.alias:
            # Evolve is very slow, so we hold our nose and do it dirty.
            _OBJ_SETATTR.__get__(a)("alias", _default_init_alias_for(a.name))
            _OBJ_SETATTR.__get__(a)("alias_is_default", True)

    if field_transformer is not None:
        attrs = tuple(field_transformer(cls, attrs))

    # Check attr order after executing the field_transformer.
    # Mandatory vs non-mandatory attr order only matters when they are part of
    # the __init__ signature and when they aren't kw_only (which are moved to
    # the end and can be mandatory or non-mandatory in any order, as they will
    # be specified as keyword args anyway). Check the order of those attrs:
    had_default = False
    for a in (a for a in attrs if a.init is not False and a.kw_only is False):
        if had_default is True and a.default is NOTHING:
            msg = f"No mandatory attributes allowed after an attribute with a default value or factory.  Attribute in question: {a!r}"
            raise ValueError(msg)

        if had_default is False and a.default is not NOTHING:
            had_default = True

    # Resolve default field alias for any new attributes that the
    # field_transformer may have added without setting an alias.
    for a in attrs:
        if not a.alias:
            _OBJ_SETATTR.__get__(a)("alias", _default_init_alias_for(a.name))
            _OBJ_SETATTR.__get__(a)("alias_is_default", True)

    # Create AttrsClass *after* applying the field_transformer since it may
    # add or remove attributes!
    attr_names = [a.name for a in attrs]
    AttrsClass = _make_attr_tuple_class(cls.__name__, attr_names)

    return _Attributes(AttrsClass(attrs), base_attrs, base_attr_map)


def _make_cached_property_getattr(cached_properties, original_getattr, cls):
    lines = [
        # Wrapped to get `__class__` into closure cell for super()
        # (It will be replaced with the newly constructed class after construction).
        "def wrapper(_cls):",
        "    __class__ = _cls",
        "    def __getattr__(self, item, cached_properties=cached_properties, original_getattr=original_getattr, _cached_setattr_get=_cached_setattr_get):",
        "         func = cached_properties.get(item)",
        "         if func is not None:",
        "              result = func(self)",
        "              _setter = _cached_setattr_get(self)",
        "              _setter(item, result)",
        "              return result",
    ]
    if original_getattr is not None:
        lines.append(
            "         return original_getattr(self, item)",
        )
    else:
        lines.extend(
            [
                "         try:",
                "             return super().__getattribute__(item)",
                "         except AttributeError:",
                "             if not hasattr(super(), '__getattr__'):",
                "                 raise",
                "             return super().__getattr__(item)",
                "         original_error = f\"'{self.__class__.__name__}' object has no attribute '{item}'\"",
                "         raise AttributeError(original_error)",
            ]
        )

    lines.extend(
        [
            "    return __getattr__",
            "__getattr__ = wrapper(_cls)",
        ]
    )

    unique_filename = _generate_unique_filename(cls, "getattr")

    glob = {
        "cached_properties": cached_properties,
        "_cached_setattr_get": _OBJ_SETATTR.__get__,
        "original_getattr": original_getattr,
    }

    return _linecache_and_compile(
        "\n".join(lines), unique_filename, glob, locals={"_cls": cls}
    )["__getattr__"]


def _frozen_setattrs(self, name, value):
    """
    Attached to frozen classes as __setattr__.
    """
    if isinstance(self, BaseException) and name in (
        "__cause__",
        "__context__",
        "__traceback__",
        "__suppress_context__",
        "__notes__",
    ):
        BaseException.__setattr__(self, name, value)
        return

    raise FrozenInstanceError


def _frozen_delattrs(self, name):
    """
    Attached to frozen classes as __delattr__.
    """
    if isinstance(self, BaseException) and name == "__notes__":
        BaseException.__delattr__(self, name)
        return

    raise FrozenInstanceError


def evolve(*args, **changes):
    """
    Create a new instance, based on the first positional argument with
    *changes* applied.

    .. tip::

       On Python 3.13 and later, you can also use `copy.replace` instead.

    Args:

        inst:
            Instance of a class with *attrs* attributes. *inst* must be passed
            as a positional argument.

        changes:
            Keyword changes in the new copy.

    Returns:
        A copy of inst with *changes* incorporated.

    Raises:
        TypeError:
            If *attr_name* couldn't be found in the class ``__init__``.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    .. versionadded:: 17.1.0
    .. deprecated:: 23.1.0
       It is now deprecated to pass the instance using the keyword argument
       *inst*. It will raise a warning until at least April 2024, after which
       it will become an error. Always pass the instance as a positional
       argument.
    .. versionchanged:: 24.1.0
       *inst* can't be passed as a keyword argument anymore.
    """
    try:
        (inst,) = args
    except ValueError:
        msg = (
            f"evolve() takes 1 positional argument, but {len(args)} were given"
        )
        raise TypeError(msg) from None

    cls = inst.__class__
    attrs = fields(cls)
    for a in attrs:
        if not a.init:
            continue
        attr_name = a.name  # To deal with private attributes.
        init_name = a.alias
        if init_name not in changes:
            changes[init_name] = getattr(inst, attr_name)

    return cls(**changes)


class _ClassBuilder:
    """
    Iteratively build *one* class.
    """

    __slots__ = (
        "_add_method_dunders",
        "_attr_names",
        "_attrs",
        "_base_attr_map",
        "_base_names",
        "_cache_hash",
        "_cls",
        "_cls_dict",
        "_delete_attribs",
        "_frozen",
        "_has_custom_setattr",
        "_has_post_init",
        "_has_pre_init",
        "_is_exc",
        "_on_setattr",
        "_pre_init_has_args",
        "_repr_added",
        "_script_snippets",
        "_slots",
        "_weakref_slot",
        "_wrote_own_setattr",
    )

    def __init__(
        self,
        cls: type,
        these,
        auto_attribs: bool,
        props: ClassProps,
        has_custom_setattr: bool,
    ):
        attrs, base_attrs, base_map = _transform_attrs(
            cls,
            these,
            auto_attribs,
            props.kw_only,
            props.collected_fields_by_mro,
            props.field_transformer,
        )

        self._cls = cls
        self._cls_dict = dict(cls.__dict__) if props.is_slotted else {}
        self._attrs = attrs
        self._base_names = {a.name for a in base_attrs}
        self._base_attr_map = base_map
        self._attr_names = tuple(a.name for a in attrs)
        self._slots = props.is_slotted
        self._frozen = props.is_frozen
        self._weakref_slot = props.has_weakref_slot
        self._cache_hash = (
            props.hashability is ClassProps.Hashability.HASHABLE_CACHED
        )
        self._has_pre_init = bool(getattr(cls, "__attrs_pre_init__", False))
        self._pre_init_has_args = False
        if self._has_pre_init:
            # Check if the pre init method has more arguments than just `self`
            # We want to pass arguments if pre init expects arguments
            pre_init_func = cls.__attrs_pre_init__
            pre_init_signature = inspect.signature(pre_init_func)
            self._pre_init_has_args = len(pre_init_signature.parameters) > 1
        self._has_post_init = bool(getattr(cls, "__attrs_post_init__", False))
        self._delete_attribs = not bool(these)
        self._is_exc = props.is_exception
        self._on_setattr = props.on_setattr_hook

        self._has_custom_setattr = has_custom_setattr
        self._wrote_own_setattr = False

        self._cls_dict["__attrs_attrs__"] = self._attrs
        self._cls_dict["__attrs_props__"] = props

        if props.is_frozen:
            self._cls_dict["__setattr__"] = _frozen_setattrs
            self._cls_dict["__delattr__"] = _frozen_delattrs

            self._wrote_own_setattr = True
        elif self._on_setattr in (
            _DEFAULT_ON_SETATTR,
            setters.validate,
            setters.convert,
        ):
            has_validator = has_converter = False
            for a in attrs:
                if a.validator is not None:
                    has_validator = True
                if a.converter is not None:
                    has_converter = True

                if has_validator and has_converter:
                    break
            if (
                (
                    self._on_setattr == _DEFAULT_ON_SETATTR
                    and not (has_validator or has_converter)
                )
                or (self._on_setattr == setters.validate and not has_validator)
                or (self._on_setattr == setters.convert and not has_converter)
            ):
                # If class-level on_setattr is set to convert + validate, but
                # there's no field to convert or validate, pretend like there's
                # no on_setattr.
                self._on_setattr = None

        if props.added_pickling:
            (
                self._cls_dict["__getstate__"],
                self._cls_dict["__setstate__"],
            ) = self._make_getstate_setstate()

        # tuples of script, globs, hook
        self._script_snippets: list[
            tuple[str, dict, Callable[[dict, dict], Any]]
        ] = []
        self._repr_added = False

        # We want to only do this check once; in 99.9% of cases these
        # exist.
        if not hasattr(self._cls, "__module__") or not hasattr(
            self._cls, "__qualname__"
        ):
            self._add_method_dunders = self._add_method_dunders_safe
        else:
            self._add_method_dunders = self._add_method_dunders_unsafe

    def __repr__(self):
        return f"<_ClassBuilder(cls={self._cls.__name__})>"

    def _eval_snippets(self) -> None:
        """
        Evaluate any registered snippets in one go.
        """
        script = "\n".join([snippet[0] for snippet in self._script_snippets])
        globs = {}
        for _, snippet_globs, _ in self._script_snippets:
            globs.update(snippet_globs)

        locs = _linecache_and_compile(
            script,
            _generate_unique_filename(self._cls, "methods"),
            globs,
        )

        for _, _, hook in self._script_snippets:
            hook(self._cls_dict, locs)

    def build_class(self):
        """
        Finalize class based on the accumulated configuration.

        Builder cannot be used after calling this method.
        """
        self._eval_snippets()
        if self._slots is True:
            cls = self._create_slots_class()
            self._cls.__attrs_base_of_slotted__ = weakref.ref(cls)
        else:
            cls = self._patch_original_class()
            if PY_3_10_PLUS:
                cls = abc.update_abstractmethods(cls)

        # The method gets only called if it's not inherited from a base class.
        # _has_own_attribute does NOT work properly for classmethods.
        if (
            getattr(cls, "__attrs_init_subclass__", None)
            and "__attrs_init_subclass__" not in cls.__dict__
        ):
            cls.__attrs_init_subclass__()

        return cls

    def _patch_original_class(self):
        """
        Apply accumulated methods and return the class.
        """
        cls = self._cls
        base_names = self._base_names

        # Clean class of attribute definitions (`attr.ib()`s).
        if self._delete_attribs:
            for name in self._attr_names:
                if (
                    name not in base_names
                    and getattr(cls, name, _SENTINEL) is not _SENTINEL
                ):
                    # An AttributeError can happen if a base class defines a
                    # class variable and we want to set an attribute with the
                    # same name by using only a type annotation.
                    with contextlib.suppress(AttributeError):
                        delattr(cls, name)

        # Attach our dunder methods.
        for name, value in self._cls_dict.items():
            setattr(cls, name, value)

        # If we've inherited an attrs __setattr__ and don't write our own,
        # reset it to object's.
        if not self._wrote_own_setattr and getattr(
            cls, "__attrs_own_setattr__", False
        ):
            cls.__attrs_own_setattr__ = False

            if not self._has_custom_setattr:
                cls.__setattr__ = _OBJ_SETATTR

        return cls

    def _create_slots_class(self):
        """
        Build and return a new class with a `__slots__` attribute.
        """
        cd = {
            k: v
            for k, v in self._cls_dict.items()
            if k not in (*tuple(self._attr_names), "__dict__", "__weakref__")
        }

        # 3.14.0rc2+
        if hasattr(sys, "_clear_type_descriptors"):
            sys._clear_type_descriptors(self._cls)

        # If our class doesn't have its own implementation of __setattr__
        # (either from the user or by us), check the bases, if one of them has
        # an attrs-made __setattr__, that needs to be reset. We don't walk the
        # MRO because we only care about our immediate base classes.
        # XXX: This can be confused by subclassing a slotted attrs class with
        # XXX: a non-attrs class and subclass the resulting class with an attrs
        # XXX: class.  See `test_slotted_confused` for details.  For now that's
        # XXX: OK with us.
        if not self._wrote_own_setattr:
            cd["__attrs_own_setattr__"] = False

            if not self._has_custom_setattr:
                for base_cls in self._cls.__bases__:
                    if base_cls.__dict__.get("__attrs_own_setattr__", False):
                        cd["__setattr__"] = _OBJ_SETATTR
                        break

        # Traverse the MRO to collect existing slots
        # and check for an existing __weakref__.
        existing_slots = {}
        weakref_inherited = False
        for base_cls in self._cls.__mro__[1:-1]:
            if base_cls.__dict__.get("__weakref__", None) is not None:
                weakref_inherited = True
            existing_slots.update(
                {
                    name: getattr(base_cls, name)
                    for name in getattr(base_cls, "__slots__", [])
                }
            )

        base_names = set(self._base_names)

        names = self._attr_names
        if (
            self._weakref_slot
            and "__weakref__" not in getattr(self._cls, "__slots__", ())
            and "__weakref__" not in names
            and not weakref_inherited
        ):
            names += ("__weakref__",)

        cached_properties = {
            name: cached_prop.func
            for name, cached_prop in cd.items()
            if isinstance(cached_prop, cached_property)
        }

        # Collect methods with a `__class__` reference that are shadowed in the new class.
        # To know to update them.
        additional_closure_functions_to_update = []
        if cached_properties:
            class_annotations = _get_annotations(self._cls)
            for name, func in cached_properties.items():
                # Add cached properties to names for slotting.
                names += (name,)
                # Clear out function from class to avoid clashing.
                del cd[name]
                additional_closure_functions_to_update.append(func)
                annotation = inspect.signature(func).return_annotation
                if annotation is not inspect.Parameter.empty:
                    class_annotations[name] = annotation

            original_getattr = cd.get("__getattr__")
            if original_getattr is not None:
                additional_closure_functions_to_update.append(original_getattr)

            cd["__getattr__"] = _make_cached_property_getattr(
                cached_properties, original_getattr, self._cls
            )

        # We only add the names of attributes that aren't inherited.
        # Setting __slots__ to inherited attributes wastes memory.
        slot_names = [name for name in names if name not in base_names]

        # There are slots for attributes from current class
        # that are defined in parent classes.
        # As their descriptors may be overridden by a child class,
        # we collect them here and update the class dict
        reused_slots = {
            slot: slot_descriptor
            for slot, slot_descriptor in existing_slots.items()
            if slot in slot_names
        }
        slot_names = [name for name in slot_names if name not in reused_slots]
        cd.update(reused_slots)
        if self._cache_hash:
            slot_names.append(_HASH_CACHE_FIELD)

        cd["__slots__"] = tuple(slot_names)

        cd["__qualname__"] = self._cls.__qualname__

        # Create new class based on old class and our methods.
        cls = type(self._cls)(self._cls.__name__, self._cls.__bases__, cd)

        # The following is a fix for
        # <https://github.com/python-attrs/attrs/issues/102>.
        # If a method mentions `__class__` or uses the no-arg super(), the
        # compiler will bake a reference to the class in the method itself
        # as `method.__closure__`.  Since we replace the class with a
        # clone, we rewrite these references so it keeps working.
        for item in itertools.chain(
            cls.__dict__.values(), additional_closure_functions_to_update
        ):
            if isinstance(item, (classmethod, staticmethod)):
                # Class- and staticmethods hide their functions inside.
                # These might need to be rewritten as well.
                closure_cells = getattr(item.__func__, "__closure__", None)
            elif isinstance(item, property):
                # Workaround for property `super()` shortcut (PY3-only).
                # There is no universal way for other descriptors.
                closure_cells = getattr(item.fget, "__closure__", None)
            else:
                closure_cells = getattr(item, "__closure__", None)

            if not closure_cells:  # Catch None or the empty list.
                continue
            for cell in closure_cells:
                try:
                    match = cell.cell_contents is self._cls
                except ValueError:  # noqa: PERF203
                    # ValueError: Cell is empty
                    pass
                else:
                    if match:
                        cell.cell_contents = cls
        return cls

    def add_repr(self, ns):
        script, globs = _make_repr_script(self._attrs, ns)

        def _attach_repr(cls_dict, globs):
            cls_dict["__repr__"] = self._add_method_dunders(globs["__repr__"])

        self._script_snippets.append((script, globs, _attach_repr))
        self._repr_added = True
        return self

    def add_str(self):
        if not self._repr_added:
            msg = "__str__ can only be generated if a __repr__ exists."
            raise ValueError(msg)

        def __str__(self):
            return self.__repr__()

        self._cls_dict["__str__"] = self._add_method_dunders(__str__)
        return self

    def _make_getstate_setstate(self):
        """
        Create custom __setstate__ and __getstate__ methods.
        """
        # __weakref__ is not writable.
        state_attr_names = tuple(
            an for an in self._attr_names if an != "__weakref__"
        )

        def slots_getstate(self):
            """
            Automatically created by attrs.
            """
            return {name: getattr(self, name) for name in state_attr_names}

        hash_caching_enabled = self._cache_hash

        def slots_setstate(self, state):
            """
            Automatically created by attrs.
            """
            __bound_setattr = _OBJ_SETATTR.__get__(self)
            if isinstance(state, tuple):
                # Backward compatibility with attrs instances pickled with
                # attrs versions before v22.2.0 which stored tuples.
                for name, value in zip(state_attr_names, state):
                    __bound_setattr(name, value)
            else:
                for name in state_attr_names:
                    if name in state:
                        __bound_setattr(name, state[name])

            # The hash code cache is not included when the object is
            # serialized, but it still needs to be initialized to None to
            # indicate that the first call to __hash__ should be a cache
            # miss.
            if hash_caching_enabled:
                __bound_setattr(_HASH_CACHE_FIELD, None)

        return slots_getstate, slots_setstate

    def make_unhashable(self):
        self._cls_dict["__hash__"] = None
        return self

    def add_hash(self):
        script, globs = _make_hash_script(
            self._cls,
            self._attrs,
            frozen=self._frozen,
            cache_hash=self._cache_hash,
        )

        def attach_hash(cls_dict: dict, locs: dict) -> None:
            cls_dict["__hash__"] = self._add_method_dunders(locs["__hash__"])

        self._script_snippets.append((script, globs, attach_hash))

        return self

    def add_init(self):
        script, globs, annotations = _make_init_script(
            self._cls,
            self._attrs,
            self._has_pre_init,
            self._pre_init_has_args,
            self._has_post_init,
            self._frozen,
            self._slots,
            self._cache_hash,
            self._base_attr_map,
            self._is_exc,
            self._on_setattr,
            attrs_init=False,
        )

        def _attach_init(cls_dict, globs):
            init = globs["__init__"]
            init.__annotations__ = annotations
            cls_dict["__init__"] = self._add_method_dunders(init)

        self._script_snippets.append((script, globs, _attach_init))

        return self

    def add_replace(self):
        self._cls_dict["__replace__"] = self._add_method_dunders(evolve)
        return self

    def add_match_args(self):
        self._cls_dict["__match_args__"] = tuple(
            field.name
            for field in self._attrs
            if field.init and not field.kw_only
        )

    def add_attrs_init(self):
        script, globs, annotations = _make_init_script(
            self._cls,
            self._attrs,
            self._has_pre_init,
            self._pre_init_has_args,
            self._has_post_init,
            self._frozen,
            self._slots,
            self._cache_hash,
            self._base_attr_map,
            self._is_exc,
            self._on_setattr,
            attrs_init=True,
        )

        def _attach_attrs_init(cls_dict, globs):
            init = globs["__attrs_init__"]
            init.__annotations__ = annotations
            cls_dict["__attrs_init__"] = self._add_method_dunders(init)

        self._script_snippets.append((script, globs, _attach_attrs_init))

        return self

    def add_eq(self):
        cd = self._cls_dict

        script, globs = _make_eq_script(self._attrs)

        def _attach_eq(cls_dict, globs):
            cls_dict["__eq__"] = self._add_method_dunders(globs["__eq__"])

        self._script_snippets.append((script, globs, _attach_eq))

        cd["__ne__"] = __ne__

        return self

    def add_order(self):
        cd = self._cls_dict

        cd["__lt__"], cd["__le__"], cd["__gt__"], cd["__ge__"] = (
            self._add_method_dunders(meth)
            for meth in _make_order(self._cls, self._attrs)
        )

        return self

    def add_setattr(self):
        sa_attrs = {}
        for a in self._attrs:
            on_setattr = a.on_setattr or self._on_setattr
            if on_setattr and on_setattr is not setters.NO_OP:
                sa_attrs[a.name] = a, on_setattr

        if not sa_attrs:
            return self

        if self._has_custom_setattr:
            # We need to write a __setattr__ but there already is one!
            msg = "Can't combine custom __setattr__ with on_setattr hooks."
            raise ValueError(msg)

        # docstring comes from _add_method_dunders
        def __setattr__(self, name, val):
            try:
                a, hook = sa_attrs[name]
            except KeyError:
                nval = val
            else:
                nval = hook(self, a, val)

            _OBJ_SETATTR(self, name, nval)

        self._cls_dict["__attrs_own_setattr__"] = True
        self._cls_dict["__setattr__"] = self._add_method_dunders(__setattr__)
        self._wrote_own_setattr = True

        return self

    def _add_method_dunders_unsafe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method*.
        """
        method.__module__ = self._cls.__module__

        method.__qualname__ = f"{self._cls.__qualname__}.{method.__name__}"

        method.__doc__ = (
            f"Method generated by attrs for class {self._cls.__qualname__}."
        )

        return method

    def _add_method_dunders_safe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method* if possible.
        """
        with contextlib.suppress(AttributeError):
            method.__module__ = self._cls.__module__

        with contextlib.suppress(AttributeError):
            method.__qualname__ = f"{self._cls.__qualname__}.{method.__name__}"

        with contextlib.suppress(AttributeError):
            method.__doc__ = f"Method generated by attrs for class {self._cls.__qualname__}."

        return method


def _determine_attrs_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    # cmp takes precedence due to bw-compatibility.
    if cmp is not None:
        return cmp, cmp

    # If left None, equality is set to the specified default and ordering
    # mirrors equality.
    if eq is None:
        eq = default_eq

    if order is None:
        order = eq

    if eq is False and order is True:
        msg = "`order` can only be True if `eq` is True too."
        raise ValueError(msg)

    return eq, order


def _determine_attrib_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    def decide_callable_or_boolean(value):
        """
        Decide whether a key function is used.
        """
        if callable(value):
            value, key = True, value
        else:
            key = None
        return value, key

    # cmp takes precedence due to bw-compatibility.
    if cmp is not None:
        cmp, cmp_key = decide_callable_or_boolean(cmp)
        return cmp, cmp_key, cmp, cmp_key

    # If left None, equality is set to the specified default and ordering
    # mirrors equality.
    if eq is None:
        eq, eq_key = default_eq, None
    else:
        eq, eq_key = decide_callable_or_boolean(eq)

    if order is None:
        order, order_key = eq, eq_key
    else:
        order, order_key = decide_callable_or_boolean(order)

    if eq is False and order is True:
        msg = "`order` can only be True if `eq` is True too."
        raise ValueError(msg)

    return eq, eq_key, order, order_key


def _determine_whether_to_implement(
    cls, flag, auto_detect, dunders, default=True
):
    """
    Check whether we should implement a set of methods for *cls*.

    *flag* is the argument passed into @attr.s like 'init', *auto_detect* the
    same as passed into @attr.s and *dunders* is a tuple of attribute names
    whose presence signal that the user has implemented it themselves.

    Return *default* if no reason for either for or against is found.
    """
    if flag is True or flag is False:
        return flag

    if flag is None and auto_detect is False:
        return default

    # Logically, flag is None and auto_detect is True here.
    for dunder in dunders:
        if _has_own_attribute(cls, dunder):
            return False

    return default


def attrs(
    maybe_cls=None,
    these=None,
    repr_ns=None,
    repr=None,
    cmp=None,
    hash=None,
    init=None,
    slots=False,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=False,
    kw_only=False,
    cache_hash=False,
    auto_exc=False,
    eq=None,
    order=None,
    auto_detect=False,
    collect_by_mro=False,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
    unsafe_hash=None,
    force_kw_only=True,
):
    r"""
    A class decorator that adds :term:`dunder methods` according to the
    specified attributes using `attr.ib` or the *these* argument.

    Consider using `attrs.define` / `attrs.frozen` in new code (``attr.s`` will
    *never* go away, though).

    Args:
        repr_ns (str):
            When using nested classes, there was no way in Python 2 to
            automatically detect that.  This argument allows to set a custom
            name for a more meaningful ``repr`` output.  This argument is
            pointless in Python 3 and is therefore deprecated.

    .. caution::
        Refer to `attrs.define` for the rest of the parameters, but note that they
        can have different defaults.

        Notably, leaving *on_setattr* as `None` will **not** add any hooks.

    .. versionadded:: 16.0.0 *slots*
    .. versionadded:: 16.1.0 *frozen*
    .. versionadded:: 16.3.0 *str*
    .. versionadded:: 16.3.0 Support for ``__attrs_post_init__``.
    .. versionchanged:: 17.1.0
       *hash* supports `None` as value which is also the default now.
    .. versionadded:: 17.3.0 *auto_attribs*
    .. versionchanged:: 18.1.0
       If *these* is passed, no attributes are deleted from the class body.
    .. versionchanged:: 18.1.0 If *these* is ordered, the order is retained.
    .. versionadded:: 18.2.0 *weakref_slot*
    .. deprecated:: 18.2.0
       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now raise a
       `DeprecationWarning` if the classes compared are subclasses of
       each other. ``__eq`` and ``__ne__`` never tried to compared subclasses
       to each other.
    .. versionchanged:: 19.2.0
       ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__`` now do not consider
       subclasses comparable anymore.
    .. versionadded:: 18.2.0 *kw_only*
    .. versionadded:: 18.2.0 *cache_hash*
    .. versionadded:: 19.1.0 *auto_exc*
    .. deprecated:: 19.2.0 *cmp* Removal on or after 2021-06-01.
    .. versionadded:: 19.2.0 *eq* and *order*
    .. versionadded:: 20.1.0 *auto_detect*
    .. versionadded:: 20.1.0 *collect_by_mro*
    .. versionadded:: 20.1.0 *getstate_setstate*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionadded:: 20.3.0 *field_transformer*
    .. versionchanged:: 21.1.0
       ``init=False`` injects ``__attrs_init__``
    .. versionchanged:: 21.1.0 Support for ``__attrs_pre_init__``
    .. versionchanged:: 21.1.0 *cmp* undeprecated
    .. versionadded:: 21.3.0 *match_args*
    .. versionadded:: 22.2.0
       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
    .. deprecated:: 24.1.0 *repr_ns*
    .. versionchanged:: 24.1.0
       Instances are not compared as tuples of attributes anymore, but using a
       big ``and`` condition. This is faster and has more correct behavior for
       uncomparable values like `math.nan`.
    .. versionadded:: 24.1.0
       If a class has an *inherited* classmethod called
       ``__attrs_init_subclass__``, it is executed after the class is created.
    .. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
    .. versionchanged:: 25.4.0
       *kw_only* now only applies to attributes defined in the current class,
       and respects attribute-level ``kw_only=False`` settings.
    .. versionadded:: 25.4.0 *force_kw_only*
    """
    if repr_ns is not None:
        import warnings

        warnings.warn(
            DeprecationWarning(
                "The `repr_ns` argument is deprecated and will be removed in or after August 2025."
            ),
            stacklevel=2,
        )

    eq_, order_ = _determine_attrs_eq_order(cmp, eq, order, None)

    #  unsafe_hash takes precedence due to PEP 681.
    if unsafe_hash is not None:
        hash = unsafe_hash

    if isinstance(on_setattr, (list, tuple)):
        on_setattr = setters.pipe(*on_setattr)

    def wrap(cls):
        nonlocal hash
        is_frozen = frozen or _has_frozen_base_class(cls)
        is_exc = auto_exc is True and issubclass(cls, BaseException)
        has_own_setattr = auto_detect and _has_own_attribute(
            cls, "__setattr__"
        )

        if has_own_setattr and is_frozen:
            msg = "Can't freeze a class with a custom __setattr__."
            raise ValueError(msg)

        eq = not is_exc and _determine_whether_to_implement(
            cls, eq_, auto_detect, ("__eq__", "__ne__")
        )

        Hashability = ClassProps.Hashability

        if is_exc:
            hashability = Hashability.LEAVE_ALONE
        elif hash is True:
            hashability = (
                Hashability.HASHABLE_CACHED
                if cache_hash
                else Hashability.HASHABLE
            )
        elif hash is False:
            hashability = Hashability.LEAVE_ALONE
        elif hash is None:
            if auto_detect is True and _has_own_attribute(cls, "__hash__"):
                hashability = Hashability.LEAVE_ALONE
            elif eq is True and is_frozen is True:
                hashability = (
                    Hashability.HASHABLE_CACHED
                    if cache_hash
                    else Hashability.HASHABLE
                )
            elif eq is False:
                hashability = Hashability.LEAVE_ALONE
            else:
                hashability = Hashability.UNHASHABLE
        else:
            msg = "Invalid value for hash.  Must be True, False, or None."
            raise TypeError(msg)

        KeywordOnly = ClassProps.KeywordOnly
        if kw_only:
            kwo = KeywordOnly.FORCE if force_kw_only else KeywordOnly.YES
        else:
            kwo = KeywordOnly.NO

        props = ClassProps(
            is_exception=is_exc,
            is_frozen=is_frozen,
            is_slotted=slots,
            collected_fields_by_mro=collect_by_mro,
            added_init=_determine_whether_to_implement(
                cls, init, auto_detect, ("__init__",)
            ),
            added_repr=_determine_whether_to_implement(
                cls, repr, auto_detect, ("__repr__",)
            ),
            added_eq=eq,
            added_ordering=not is_exc
            and _determine_whether_to_implement(
                cls,
                order_,
                auto_detect,
                ("__lt__", "__le__", "__gt__", "__ge__"),
            ),
            hashability=hashability,
            added_match_args=match_args,
            kw_only=kwo,
            has_weakref_slot=weakref_slot,
            added_str=str,
            added_pickling=_determine_whether_to_implement(
                cls,
                getstate_setstate,
                auto_detect,
                ("__getstate__", "__setstate__"),
                default=slots,
            ),
            on_setattr_hook=on_setattr,
            field_transformer=field_transformer,
        )

        if not props.is_hashable and cache_hash:
            msg = "Invalid value for cache_hash.  To use hash caching, hashing must be either explicitly or implicitly enabled."
            raise TypeError(msg)

        builder = _ClassBuilder(
            cls,
            these,
            auto_attribs=auto_attribs,
            props=props,
            has_custom_setattr=has_own_setattr,
        )

        if props.added_repr:
            builder.add_repr(repr_ns)

        if props.added_str:
            builder.add_str()

        if props.added_eq:
            builder.add_eq()
        if props.added_ordering:
            builder.add_order()

        if not frozen:
            builder.add_setattr()

        if props.is_hashable:
            builder.add_hash()
        elif props.hashability is Hashability.UNHASHABLE:
            builder.make_unhashable()

        if props.added_init:
            builder.add_init()
        else:
            builder.add_attrs_init()
            if cache_hash:
                msg = "Invalid value for cache_hash.  To use hash caching, init must be True."
                raise TypeError(msg)

        if PY_3_13_PLUS and not _has_own_attribute(cls, "__replace__"):
            builder.add_replace()

        if (
            PY_3_10_PLUS
            and match_args
            and not _has_own_attribute(cls, "__match_args__")
        ):
            builder.add_match_args()

        return builder.build_class()

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


_attrs = attrs
"""
Internal alias so we can use it in functions that take an argument called
*attrs*.
"""


def _has_frozen_base_class(cls):
    """
    Check whether *cls* has a frozen ancestor by looking at its
    __setattr__.
    """
    return cls.__setattr__ is _frozen_setattrs


def _generate_unique_filename(cls: type, func_name: str) -> str:
    """
    Create a "filename" suitable for a function being generated.
    """
    return (
        f"<attrs generated {func_name} {cls.__module__}."
        f"{getattr(cls, '__qualname__', cls.__name__)}>"
    )


def _make_hash_script(
    cls: type, attrs: list[Attribute], frozen: bool, cache_hash: bool
) -> tuple[str, dict]:
    attrs = tuple(
        a for a in attrs if a.hash is True or (a.hash is None and a.eq is True)
    )

    tab = "        "

    type_hash = hash(_generate_unique_filename(cls, "hash"))
    # If eq is custom generated, we need to include the functions in globs
    globs = {}

    hash_def = "def __hash__(self"
    hash_func = "hash(("
    closing_braces = "))"
    if not cache_hash:
        hash_def += "):"
    else:
        hash_def += ", *"

        hash_def += ", _cache_wrapper=__import__('attr._make')._make._CacheHashWrapper):"
        hash_func = "_cache_wrapper(" + hash_func
        closing_braces += ")"

    method_lines = [hash_def]

    def append_hash_computation_lines(prefix, indent):
        """
        Generate the code for actually computing the hash code.
        Below this will either be returned directly or used to compute
        a value which is then cached, depending on the value of cache_hash
        """

        method_lines.extend(
            [
                indent + prefix + hash_func,
                indent + f"        {type_hash},",
            ]
        )

        for a in attrs:
            if a.eq_key:
                cmp_name = f"_{a.name}_key"
                globs[cmp_name] = a.eq_key
                method_lines.append(
                    indent + f"        {cmp_name}(self.{a.name}),"
                )
            else:
                method_lines.append(indent + f"        self.{a.name},")

        method_lines.append(indent + "    " + closing_braces)

    if cache_hash:
        method_lines.append(tab + f"if self.{_HASH_CACHE_FIELD} is None:")
        if frozen:
            append_hash_computation_lines(
                f"object.__setattr__(self, '{_HASH_CACHE_FIELD}', ", tab * 2
            )
            method_lines.append(tab * 2 + ")")  # close __setattr__
        else:
            append_hash_computation_lines(
                f"self.{_HASH_CACHE_FIELD} = ", tab * 2
            )
        method_lines.append(tab + f"return self.{_HASH_CACHE_FIELD}")
    else:
        append_hash_computation_lines("return ", tab)

    script = "\n".join(method_lines)
    return script, globs


def _add_hash(cls: type, attrs: list[Attribute]):
    """
    Add a hash method to *cls*.
    """
    script, globs = _make_hash_script(
        cls, attrs, frozen=False, cache_hash=False
    )
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__hash__")
    )
    cls.__hash__ = globs["__hash__"]
    return cls


def __ne__(self, other):
    """
    Check equality and either forward a NotImplemented or
    return the result negated.
    """
    result = self.__eq__(other)
    if result is NotImplemented:
        return NotImplemented

    return not result


def _make_eq_script(attrs: list) -> tuple[str, dict]:
    """
    Create __eq__ method for *cls* with *attrs*.
    """
    attrs = [a for a in attrs if a.eq]

    lines = [
        "def __eq__(self, other):",
        "    if other.__class__ is not self.__class__:",
        "        return NotImplemented",
    ]

    globs = {}
    if attrs:
        lines.append("    return  (")
        for a in attrs:
            if a.eq_key:
                cmp_name = f"_{a.name}_key"
                # Add the key function to the global namespace
                # of the evaluated function.
                globs[cmp_name] = a.eq_key
                lines.append(
                    f"        {cmp_name}(self.{a.name}) == {cmp_name}(other.{a.name})"
                )
            else:
                lines.append(f"        self.{a.name} == other.{a.name}")
            if a is not attrs[-1]:
                lines[-1] = f"{lines[-1]} and"
        lines.append("    )")
    else:
        lines.append("    return True")

    script = "\n".join(lines)

    return script, globs


def _make_order(cls, attrs):
    """
    Create ordering methods for *cls* with *attrs*.
    """
    attrs = [a for a in attrs if a.order]

    def attrs_to_tuple(obj):
        """
        Save us some typing.
        """
        return tuple(
            key(value) if key else value
            for value, key in (
                (getattr(obj, a.name), a.order_key) for a in attrs
            )
        )

    def __lt__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) < attrs_to_tuple(other)

        return NotImplemented

    def __le__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) <= attrs_to_tuple(other)

        return NotImplemented

    def __gt__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) > attrs_to_tuple(other)

        return NotImplemented

    def __ge__(self, other):
        """
        Automatically created by attrs.
        """
        if other.__class__ is self.__class__:
            return attrs_to_tuple(self) >= attrs_to_tuple(other)

        return NotImplemented

    return __lt__, __le__, __gt__, __ge__


def _add_eq(cls, attrs=None):
    """
    Add equality methods to *cls* with *attrs*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__

    script, globs = _make_eq_script(attrs)
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__eq__")
    )
    cls.__eq__ = globs["__eq__"]
    cls.__ne__ = __ne__

    return cls


def _make_repr_script(attrs, ns) -> tuple[str, dict]:
    """
    Create the source and globs for a __repr__ and return it.
    """
    # Figure out which attributes to include, and which function to use to
    # format them. The a.repr value can be either bool or a custom
    # callable.
    attr_names_with_reprs = tuple(
        (a.name, (repr if a.repr is True else a.repr), a.init)
        for a in attrs
        if a.repr is not False
    )
    globs = {
        name + "_repr": r for name, r, _ in attr_names_with_reprs if r != repr
    }
    globs["_compat"] = _compat
    globs["AttributeError"] = AttributeError
    globs["NOTHING"] = NOTHING
    attribute_fragments = []
    for name, r, i in attr_names_with_reprs:
        accessor = (
            "self." + name if i else 'getattr(self, "' + name + '", NOTHING)'
        )
        fragment = (
            "%s={%s!r}" % (name, accessor)
            if r == repr
            else "%s={%s_repr(%s)}" % (name, name, accessor)
        )
        attribute_fragments.append(fragment)
    repr_fragment = ", ".join(attribute_fragments)

    if ns is None:
        cls_name_fragment = '{self.__class__.__qualname__.rsplit(">.", 1)[-1]}'
    else:
        cls_name_fragment = ns + ".{self.__class__.__name__}"

    lines = [
        "def __repr__(self):",
        "  try:",
        "    already_repring = _compat.repr_context.already_repring",
        "  except AttributeError:",
        "    already_repring = {id(self),}",
        "    _compat.repr_context.already_repring = already_repring",
        "  else:",
        "    if id(self) in already_repring:",
        "      return '...'",
        "    else:",
        "      already_repring.add(id(self))",
        "  try:",
        f"    return f'{cls_name_fragment}({repr_fragment})'",
        "  finally:",
        "    already_repring.remove(id(self))",
    ]

    return "\n".join(lines), globs


def _add_repr(cls, ns=None, attrs=None):
    """
    Add a repr method to *cls*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__

    script, globs = _make_repr_script(attrs, ns)
    _compile_and_eval(
        script, globs, filename=_generate_unique_filename(cls, "__repr__")
    )
    cls.__repr__ = globs["__repr__"]
    return cls


def fields(cls):
    """
    Return the tuple of *attrs* attributes for a class or instance.

    The tuple also allows accessing the fields by their names (see below for
    examples).

    Args:
        cls (type): Class or instance to introspect.

    Raises:
        TypeError: If *cls* is neither a class nor an *attrs* instance.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    Returns:
        tuple (with name accessors) of `attrs.Attribute`

    .. versionchanged:: 16.2.0 Returned tuple allows accessing the fields
       by name.
    .. versionchanged:: 23.1.0 Add support for generic classes.
    .. versionchanged:: 26.1.0 Add support for instances.
    """
    generic_base = get_generic_base(cls)

    if generic_base is None and not isinstance(cls, type):
        type_ = type(cls)
        if getattr(type_, "__attrs_attrs__", None) is None:
