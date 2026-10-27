    .. deprecated:: 8.2
        Will be removed in Click 9.0.
    """

    def __init__(self, ctx: Context | None = None) -> None:
        #: The :class:`~click.Context` for this parser.  This might be
        #: `None` for some advanced use cases.
        self.ctx = ctx
        #: This controls how the parser deals with interspersed arguments.
        #: If this is set to `False`, the parser will stop on the first
        #: non-option.  Click uses this to implement nested subcommands
        #: safely.
        self.allow_interspersed_args: bool = True
        #: This tells the parser how to deal with unknown options.  By
        #: default it will error out (which is sensible), but there is a
        #: second mode where it will ignore it and continue processing
        #: after shifting all the unknown options into the resulting args.
        self.ignore_unknown_options: bool = False

        if ctx is not None:
            self.allow_interspersed_args = ctx.allow_interspersed_args
            self.ignore_unknown_options = ctx.ignore_unknown_options

        self._short_opt: dict[str, _Option] = {}
        self._long_opt: dict[str, _Option] = {}
        self._opt_prefixes = {"-", "--"}
        self._args: list[_Argument] = []

    def add_option(
        self,
        obj: CoreOption,
        opts: cabc.Sequence[str],
        dest: str | None,
        action: str | None = None,
        nargs: int = 1,
        const: t.Any | None = None,
    ) -> None:
        """Adds a new option named `dest` to the parser.  The destination
        is not inferred (unlike with optparse) and needs to be explicitly
        provided.  Action can be any of ``store``, ``store_const``,
        ``append``, ``append_const`` or ``count``.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        opts = [_normalize_opt(opt, self.ctx) for opt in opts]
        option = _Option(obj, opts, dest, action=action, nargs=nargs, const=const)
        self._opt_prefixes.update(option.prefixes)
        for opt in option._short_opts:
            self._short_opt[opt] = option
        for opt in option._long_opts:
            self._long_opt[opt] = option

    def add_argument(self, obj: CoreArgument, dest: str | None, nargs: int = 1) -> None:
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        self._args.append(_Argument(obj, dest=dest, nargs=nargs))

    def parse_args(
        self, args: list[str]
    ) -> tuple[dict[str, t.Any], list[str], list[CoreParameter]]:
        """Parses positional arguments and returns ``(values, args, order)``
        for the parsed options and arguments as well as the leftover
        arguments if there are any.  The order is a list of objects as they
        appear on the command line.  If arguments appear multiple times they
        will be memorized multiple times as well.
        """
        state = _ParsingState(args)
        try:
            self._process_args_for_options(state)
            self._process_args_for_args(state)
        except UsageError:
            if self.ctx is None or not self.ctx.resilient_parsing:
                raise
        return state.opts, state.largs, state.order

    def _process_args_for_args(self, state: _ParsingState) -> None:
        pargs, args = _unpack_args(
            state.largs + state.rargs, [x.nargs for x in self._args]
        )

        for idx, arg in enumerate(self._args):
            arg.process(pargs[idx], state)

        state.largs = args
        state.rargs = []

    def _process_args_for_options(self, state: _ParsingState) -> None:
        while state.rargs:
            arg = state.rargs.pop(0)
            arglen = len(arg)
            # Double dashes always handled explicitly regardless of what
            # prefixes are valid.
            if arg == "--":
                return
            elif arg[:1] in self._opt_prefixes and arglen > 1:
                self._process_opts(arg, state)
            elif self.allow_interspersed_args:
                state.largs.append(arg)
            else:
                state.rargs.insert(0, arg)
                return

        # Say this is the original argument list:
        # [arg0, arg1, ..., arg(i-1), arg(i), arg(i+1), ..., arg(N-1)]
        #                            ^
        # (we are about to process arg(i)).
        #
        # Then rargs is [arg(i), ..., arg(N-1)] and largs is a *subset* of
        # [arg0, ..., arg(i-1)] (any options and their arguments will have
        # been removed from largs).
        #
        # The while loop will usually consume 1 or more arguments per pass.
        # If it consumes 1 (eg. arg is an option that takes no arguments),
        # then after _process_arg() is done the situation is:
        #
        #   largs = subset of [arg0, ..., arg(i)]
        #   rargs = [arg(i+1), ..., arg(N-1)]
        #
        # If allow_interspersed_args is false, largs will always be
        # *empty* -- still a subset of [arg0, ..., arg(i-1)], but
        # not a very interesting subset!

    def _match_long_opt(
        self, opt: str, explicit_value: str | None, state: _ParsingState
    ) -> None:
        if opt not in self._long_opt:
            raise NoSuchOption(opt, possibilities=self._long_opt, ctx=self.ctx)

        option = self._long_opt[opt]
        if option.takes_value:
            # At this point it's safe to modify rargs by injecting the
            # explicit value, because no exception is raised in this
            # branch.  This means that the inserted value will be fully
            # consumed.
            if explicit_value is not None:
                state.rargs.insert(0, explicit_value)

            value = self._get_value_from_state(opt, option, state)

        elif explicit_value is not None:
            raise BadOptionUsage(
                opt, _("Option {name!r} does not take a value.").format(name=opt)
            )

        else:
            value = UNSET

        option.process(value, state)

    def _match_short_opt(self, arg: str, state: _ParsingState) -> None:
        stop = False
        i = 1
        prefix = arg[0]
        unknown_options = []

        for ch in arg[1:]:
            opt = _normalize_opt(f"{prefix}{ch}", self.ctx)
            option = self._short_opt.get(opt)
            i += 1

            if not option:
                if self.ignore_unknown_options:
                    unknown_options.append(ch)
                    continue
                raise NoSuchOption(opt, ctx=self.ctx)
            if option.takes_value:
                # Any characters left in arg?  Pretend they're the
                # next arg, and stop consuming characters of arg.
                if i < len(arg):
                    state.rargs.insert(0, arg[i:])
                    stop = True

                value = self._get_value_from_state(opt, option, state)

            else:
                value = UNSET

            option.process(value, state)

            if stop:
                break

        # If we got any unknown options we recombine the string of the
        # remaining options and re-attach the prefix, then report that
        # to the state as new large.  This way there is basic combinatorics
        # that can be achieved while still ignoring unknown arguments.
        if self.ignore_unknown_options and unknown_options:
            state.largs.append(f"{prefix}{''.join(unknown_options)}")

    def _get_value_from_state(
        self, option_name: str, option: _Option, state: _ParsingState
    ) -> str | cabc.Sequence[str] | T_UNSET | T_FLAG_NEEDS_VALUE:
        nargs = option.nargs

        value: str | cabc.Sequence[str] | T_UNSET | T_FLAG_NEEDS_VALUE

        if len(state.rargs) < nargs:
            if option.obj._flag_needs_value:
                # Option allows omitting the value.
                value = FLAG_NEEDS_VALUE
            else:
                raise BadOptionUsage(
                    option_name,
                    ngettext(
                        "Option {name!r} requires an argument.",
                        "Option {name!r} requires {nargs} arguments.",
                        nargs,
                    ).format(name=option_name, nargs=nargs),
                )
        elif nargs == 1:
            next_rarg = state.rargs[0]

            if (
                option.obj._flag_needs_value
                and isinstance(next_rarg, str)
                and next_rarg[:1] in self._opt_prefixes
                and len(next_rarg) > 1
            ):
                # The next arg looks like the start of an option, don't
                # use it as the value if omitting the value is allowed.
                value = FLAG_NEEDS_VALUE
            else:
                value = state.rargs.pop(0)
        else:
            value = tuple(state.rargs[:nargs])
            del state.rargs[:nargs]

        return value

    def _process_opts(self, arg: str, state: _ParsingState) -> None:
        explicit_value = None
        # Long option handling happens in two parts.  The first part is
        # supporting explicitly attached values.  In any case, we will try
        # to long match the option first.
        if "=" in arg:
            long_opt, explicit_value = arg.split("=", 1)
        else:
            long_opt = arg
        norm_long_opt = _normalize_opt(long_opt, self.ctx)

        # At this point we will match the (assumed) long option through
        # the long option matching code.  Note that this allows options
        # like "-foo" to be matched as long options.
        try:
            self._match_long_opt(norm_long_opt, explicit_value, state)
        except NoSuchOption:
            # At this point the long option matching failed, and we need
            # to try with short options.  However there is a special rule
            # which says, that if we have a two character options prefix
            # (applies to "--foo" for instance), we do not dispatch to the
            # short option code and will instead raise the no option
            # error.
            if arg[:2] not in self._opt_prefixes:
                self._match_short_opt(arg, state)
                return

            if not self.ignore_unknown_options:
                raise

            state.largs.append(arg)


def __getattr__(name: str) -> object:
    import warnings

    if name in {
        "OptionParser",
        "Argument",
        "Option",
        "split_opt",
        "normalize_opt",
        "ParsingState",
    }:
        warnings.warn(
            f"'parser.{name}' is deprecated and will be removed in Click 9.0."
            " The old parser is available in 'optparse'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[f"_{name}"]

    if name == "split_arg_string":
        from .shell_completion import split_arg_string

        warnings.warn(
            "Importing 'parser.split_arg_string' is deprecated, it will only be"
            " available in 'shell_completion' in Click 9.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return split_arg_string

    raise AttributeError(name)
<|endoftext|>from __future__ import annotations

import typing as t
from threading import local

if t.TYPE_CHECKING:
    from .core import Context

_local = local()


@t.overload
def get_current_context(silent: t.Literal[False] = False) -> Context: ...


@t.overload
def get_current_context(silent: bool = ...) -> Context | None: ...


def get_current_context(silent: bool = False) -> Context | None:
    """Returns the current click context.  This can be used as a way to
    access the current context object from anywhere.  This is a more implicit
    alternative to the :func:`pass_context` decorator.  This function is
    primarily useful for helpers such as :func:`echo` which might be
    interested in changing its behavior based on the current context.

    To push the current context, :meth:`Context.scope` can be used.

    .. versionadded:: 5.0

    :param silent: if set to `True` the return value is `None` if no context
                   is available.  The default behavior is to raise a
                   :exc:`RuntimeError`.
    """
    try:
        return t.cast("Context", _local.stack[-1])
    except (AttributeError, IndexError) as e:
        if not silent:
            raise RuntimeError("There is no active click context.") from e

    return None


def push_context(ctx: Context) -> None:
    """Pushes a new context to the current stack."""
    _local.__dict__.setdefault("stack", []).append(ctx)


def pop_context() -> None:
    """Removes the top level from the stack."""
    _local.stack.pop()


def resolve_color_default(color: bool | None = None) -> bool | None:
    """Internal helper to get the default value of the color flag.  If a
    value is passed it's returned unchanged, otherwise it's looked up from
    the current context.
    """
    if color is not None:
        return color

    ctx = get_current_context(silent=True)

    if ctx is not None:
        return ctx.color

    return None
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import inspect
import io
import itertools
import os
import re
import sys
import typing as t
from contextlib import AbstractContextManager
from contextlib import redirect_stdout
from gettext import gettext as _

from . import _compat
from ._compat import isatty
from ._compat import strip_ansi
from .exceptions import Abort
from .exceptions import UsageError
from .globals import resolve_color_default
from .types import Choice
from .types import convert_type
from .types import ParamType
from .utils import _LazyFile
from .utils import echo

if t.TYPE_CHECKING:
    from ._termui_impl import ProgressBar

V = t.TypeVar("V")

# The prompt functions to use.  The doc tools currently override these
# functions to customize how they work.
visible_prompt_func: t.Callable[[str], str] = input

_ansi_colors = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "reset": 39,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
}
_ansi_reset_all = "\033[0m"


_HIDDEN_INPUT_MASK = "'***'"


def _mask_hidden_input(message: str, value: str) -> str:
    """Replace occurrences of ``value`` in ``message`` with a fixed mask.

    Both ``repr(value)`` (the form built-in :class:`ParamType` errors use
    via ``{value!r}``) and the raw value are masked. The raw-value pass
    uses word-boundary lookarounds so a substring like ``"1"`` does not
    match inside ``"10"``, and ``"ent"`` does not match inside
    ``"Authentication"``. The empty string is skipped to avoid matching
    at every boundary.
    """
    message = message.replace(repr(value), _HIDDEN_INPUT_MASK)
    if value:
        message = re.sub(
            rf"(?<!\w){re.escape(value)}(?!\w)", _HIDDEN_INPUT_MASK, message
        )
    return message


def hidden_prompt_func(prompt: str) -> str:
    import getpass

    return getpass.getpass(prompt)


def _readline_prompt(func: t.Callable[[str], str], text: str, err: bool) -> str:
    """Call a prompt function, passing the full prompt so readline can
    handle line editing and cursor positioning correctly.

    The prompt is handed to *func* (such as :func:`input`) rather than
    written through :func:`echo`, so it has to strip ANSI color and style
    codes itself when the destination stream does not support them. Without
    this the prompt would keep codes that :func:`echo` removes from the
    rest of the output.
    """
    stream = sys.stderr if err else sys.stdout

    # Look up ``should_strip_ansi`` on the module so that ``CliRunner``,
    # which patches it there during test isolation, is honored.
    if _compat.should_strip_ansi(stream, resolve_color_default()):
        text = strip_ansi(text)

    if err:
        with redirect_stdout(sys.stderr):
            return func(text)
    return func(text)


def _build_prompt(
    text: str,
    suffix: str,
    show_default: bool | str = False,
    default: object | None = None,
    show_choices: bool = True,
    type: object | None = None,
) -> str:
    prompt = text
    if type is not None and show_choices and isinstance(type, Choice):
        prompt += f" ({', '.join(map(str, type.choices))})"
    default_preview = ""
    if show_default:
        if isinstance(show_default, str):
            default_preview = f" [({show_default})]"
        elif default is not None:
            default_preview = f" [{_format_default(default)}]"
    return f"{prompt}{default_preview}{suffix}"


def _format_default(default: V) -> V | str:
    if isinstance(default, (io.IOBase, _LazyFile)):
        name = getattr(default, "name", None)

        if name is not None:
            return str(name)

    return default


@t.overload
def prompt(
    text: str,
    default: str | None = None,
    hide_input: bool = False,
    confirmation_prompt: bool | str = False,
    type: None = None,
    value_proc: None = None,
    prompt_suffix: str = ": ",
    show_default: bool | str = True,
    err: bool = False,
    show_choices: bool = True,
) -> str: ...


@t.overload
def prompt(
    text: str,
    default: V | str | None = None,
    hide_input: bool = False,
    confirmation_prompt: bool | str = False,
    type: ParamType[V, str] | type[V] | None = None,
    value_proc: t.Callable[[str], V] | None = None,
    prompt_suffix: str = ": ",
    show_default: bool | str = True,
    err: bool = False,
    show_choices: bool = True,
) -> V: ...


def prompt(
    text: str,
    default: V | str | None = None,
    hide_input: bool = False,
    confirmation_prompt: bool | str = False,
    type: ParamType[V, str] | type[V] | None = None,
    value_proc: t.Callable[[str], V] | None = None,
    prompt_suffix: str = ": ",
    show_default: bool | str = True,
    err: bool = False,
    show_choices: bool = True,
) -> V:
    """Prompts a user for input.  This is a convenience function that can
    be used to prompt a user for input later.

    If the user aborts the input by sending an interrupt signal, this
    function will catch it and raise a :exc:`Abort` exception.

    :param text: the text to show for the prompt.
    :param default: the default value to use if no input happens.  If this
                    is not given it will prompt until it's aborted.
    :param hide_input: if this is set to true then the input value will
                       be hidden.
    :param confirmation_prompt: Prompt a second time to confirm the
        value. Can be set to a string instead of ``True`` to customize
        the message.
    :param type: the type to use to check the value against.
    :param value_proc: if this parameter is provided it's a function that
                       is invoked instead of the type conversion to
                       convert a value.
    :param prompt_suffix: a suffix that should be added to the prompt.
    :param show_default: shows or hides the default value in the prompt.
                         If this value is a string, it shows that string
                         in parentheses instead of the actual value.
    :param err: if set to true the file defaults to ``stderr`` instead of
                ``stdout``, the same as with echo.
    :param show_choices: Show or hide choices if the passed type is a Choice.
                         For example if type is a Choice of either day or week,
                         show_choices is true and text is "Group by" then the
                         prompt will be "Group by (day, week): ".

    .. versionchanged:: 8.5.0
        Generically typed: the return type is narrowed by ``type``,
        ``value_proc``, or ``default`` instead of being ``Any``. Runtime
        behavior is unchanged.

    .. versionchanged:: 8.3.3
        ``show_default`` can be a string to show a custom value instead
        of the actual default, matching the help text behavior.

    .. versionchanged:: 8.3.1
        A space is no longer appended to the prompt.

    .. versionadded:: 8.0
        ``confirmation_prompt`` can be a custom string.

    .. versionadded:: 7.0
        Added the ``show_choices`` parameter.

    .. versionadded:: 6.0
        Added unicode support for cmd.exe on Windows.

    .. versionadded:: 4.0
        Added the `err` parameter.

    """

    def prompt_func(text: str) -> str:
        f = hidden_prompt_func if hide_input else visible_prompt_func
        try:
            return _readline_prompt(f, text, err)
        except (KeyboardInterrupt, EOFError):
            # getpass doesn't print a newline if the user aborts input with ^C.
            # Allegedly this behavior is inherited from getpass(3).
            # A doc bug has been filed at https://bugs.python.org/issue24711
            if hide_input:
                echo(None, err=err)
            raise Abort() from None

    if value_proc is None:
        value_proc = convert_type(type, default)

    prompt = _build_prompt(
        text, prompt_suffix, show_default, default, show_choices, type
    )

    if confirmation_prompt:
        if confirmation_prompt is True:
            confirmation_prompt = _("Repeat for confirmation")

        confirmation_prompt = _build_prompt(confirmation_prompt, prompt_suffix)

    while True:
        while True:
            value = prompt_func(prompt)
            if value:
                break
            elif default is not None:
                # Defaults of any type are accepted and round trip through
                # value_proc like typed input, so the annotation is only
                # accurate for typed input.
                value = t.cast("str", default)
                break
        try:
            result = value_proc(value)
        except UsageError as e:
            message = _mask_hidden_input(e.message, value) if hide_input else e.message
            echo(_("Error: {message}").format(message=message), err=err)
            continue
        if not confirmation_prompt:
            return result
        while True:
            value2 = prompt_func(confirmation_prompt)
            is_empty = not value and not value2
            if value2 or is_empty:
                break
        if value == value2:
            return result
        echo(_("Error: The two entered values do not match."), err=err)


def confirm(
    text: str,
    default: bool | None = False,
    abort: bool = False,
    prompt_suffix: str = ": ",
    show_default: bool = True,
    err: bool = False,
) -> bool:
    """Prompts for confirmation (yes/no question).

    If the user aborts the input by sending a interrupt signal this
    function will catch it and raise a :exc:`Abort` exception.

    :param text: the question to ask.
    :param default: The default value to use when no input is given. If
        ``None``, repeat until input is given.
    :param abort: if this is set to `True` a negative answer aborts the
                  exception by raising :exc:`Abort`.
    :param prompt_suffix: a suffix that should be added to the prompt.
    :param show_default: shows or hides the default value in the prompt.
    :param err: if set to true the file defaults to ``stderr`` instead of
                ``stdout``, the same as with echo.

    .. versionchanged:: 8.3.1
        A space is no longer appended to the prompt.

    .. versionchanged:: 8.0
        Repeat until input is given if ``default`` is ``None``.

    .. versionadded:: 4.0
        Added the ``err`` parameter.
    """
    prompt = _build_prompt(
        text,
        prompt_suffix,
        show_default,
        "y/n" if default is None else ("Y/n" if default else "y/N"),
    )

    while True:
        try:
            value = _readline_prompt(visible_prompt_func, prompt, err).lower().strip()
        except (KeyboardInterrupt, EOFError):
            raise Abort() from None
        if value in ("y", "yes"):
            rv = True
        elif value in ("n", "no"):
            rv = False
        elif default is not None and value == "":
            rv = default
        else:
            echo(_("Error: invalid input"), err=err)
            continue
        break
    if abort and not rv:
        raise Abort()
    return rv


def get_pager_file(
    color: bool | None = None,
) -> t.ContextManager[t.TextIO]:
    """Context manager.

    Yields a writable file-like object which can be used as an output pager.

    .. versionadded:: 8.4.0

    :param color: controls if the pager supports ANSI colors or not.  The
                  default is autodetection.
    """
    from ._termui_impl import get_pager_file

    color = resolve_color_default(color)

    return get_pager_file(color=color)


def echo_via_pager(
    text_or_generator: cabc.Iterable[str] | t.Callable[[], cabc.Iterable[str]] | str,
    color: bool | None = None,
) -> None:
    """This function takes a text and shows it via an environment specific
    pager on stdout.

    .. versionchanged:: 3.0
       Added the `color` flag.

    :param text_or_generator: the text to page, or alternatively, a
                              generator emitting the text to page.
    :param color: controls if the pager supports ANSI colors or not.  The
                  default is autodetection.
    """

    if inspect.isgeneratorfunction(text_or_generator):
        i = t.cast("t.Callable[[], cabc.Iterable[str]]", text_or_generator)()
    elif isinstance(text_or_generator, str):
        i = [text_or_generator]
    else:
        i = iter(t.cast("cabc.Iterable[str]", text_or_generator))

    # convert every element of i to a text type if necessary
    text_generator = (el if isinstance(el, str) else str(el) for el in i)

    with get_pager_file(color=color) as pager:
        for text in itertools.chain(text_generator, "\n"):
            pager.write(text)
            # Flush after each write so a slow generator streams to the pager
            # incrementally rather than staying invisible until the pipe buffer
            # fills (~8 KB).
            pager.flush()


@t.overload
def progressbar(
    *,
    length: int,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[int]: ...


@t.overload
def progressbar(
    iterable: cabc.Iterable[V] | None = None,
    length: int | None = None,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    item_show_func: t.Callable[[V | None], str | None] | None = None,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[V]: ...


def progressbar(
    iterable: cabc.Iterable[V] | None = None,
    length: int | None = None,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    item_show_func: t.Callable[[V | None], str | None] | None = None,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[V]:
    """This function creates an iterable context manager that can be used
    to iterate over something while showing a progress bar.  It will
    either iterate over the `iterable` or `length` items (that are counted
    up).  While iteration happens, this function will print a rendered
    progress bar to the given `file` (defaults to stdout) and will attempt
    to calculate remaining time and more.  By default, this progress bar
    will not be rendered if the file is not a terminal.

    The context manager creates the progress bar.  When the context
    manager is entered the progress bar is already created.  With every
    iteration over the progress bar, the iterable passed to the bar is
    advanced and the bar is updated.  When the context manager exits,
    a newline is printed and the progress bar is finalized on screen.

    Note: The progress bar is currently designed for use cases where the
    total progress can be expected to take at least several seconds.
    Because of this, the ProgressBar class object won't display
    progress that is considered too fast, and progress where the time
    between steps is less than a second.

    No printing must happen or the progress bar will be unintentionally
    destroyed.

    Example usage::

        with progressbar(items) as bar:
            for item in bar:
                do_something_with(item)

    Alternatively, if no iterable is specified, one can manually update the
    progress bar through the `update()` method instead of directly
    iterating over the progress bar.  The update method accepts the number
    of steps to increment the bar with::

        with progressbar(length=chunks.total_bytes) as bar:
            for chunk in chunks:
                process_chunk(chunk)
                bar.update(chunks.bytes)

    The ``update()`` method also takes an optional value specifying the
    ``current_item`` at the new position. This is useful when used
    together with ``item_show_func`` to customize the output for each
    manual step::

        with click.progressbar(
            length=total_size,
            label='Unzipping archive',
            item_show_func=lambda a: a.filename
        ) as bar:
            for archive in zip_file:
                archive.extract()
                bar.update(archive.size, archive)

    :param iterable: an iterable to iterate over.  If not provided the length
                     is required.
    :param length: the number of items to iterate over.  By default the
                   progressbar will attempt to ask the iterator about its
                   length, which might or might not work.  If an iterable is
                   also provided this parameter can be used to override the
                   length.  If an iterable is not provided the progress bar
                   will iterate over a range of that length.
    :param label: the label to show next to the progress bar.
    :param hidden: hide the progressbar. Defaults to ``False``. When no tty is
        detected, it will only print the progressbar label. Setting this to
        ``False`` also disables that.
    :param show_eta: enables or disables the estimated time display.  This is
                     automatically disabled if the length cannot be
                     determined.
    :param show_percent: enables or disables the percentage display.  The
                         default is `True` if the iterable has a length or
                         `False` if not.
    :param show_pos: enables or disables the absolute position display.  The
                     default is `False`.
    :param item_show_func: A function called with the current item which
        can return a string to show next to the progress bar. If the
        function returns ``None`` nothing is shown. The current item can
        be ``None``, such as when entering and exiting the bar.
    :param fill_char: the character to use to show the filled part of the
                      progress bar.
    :param empty_char: the character to use to show the non-filled part of
                       the progress bar.
    :param bar_template: the format string to use as template for the bar.
                         The parameters in it are ``label`` for the label,
                         ``bar`` for the progress bar and ``info`` for the
                         info section.
    :param info_sep: the separator between multiple info items (eta etc.)
    :param width: the width of the progress bar in characters, 0 means full
                  terminal width
    :param file: The file to write to. If this is not a terminal then
        only the label is printed.
    :param color: controls if the terminal supports ANSI colors or not.  The
                  default is autodetection.  This is only needed if ANSI
                  codes are included anywhere in the progress bar output
                  which is not the case by default.
    :param update_min_steps: Render only when this many updates have
        completed. This allows tuning for very fast iterators.

    .. versionadded:: 8.2
        The ``hidden`` argument.

    .. versionchanged:: 8.0
        Output is shown even if execution time is less than 0.5 seconds.

    .. versionchanged:: 8.0
        ``item_show_func`` shows the current item, not the previous one.

    .. versionchanged:: 8.0
        Labels are echoed if the output is not a TTY. Reverts a change
        in 7.0 that removed all output.

    .. versionadded:: 8.0
       The ``update_min_steps`` parameter.

    .. versionadded:: 4.0
        The ``color`` parameter and ``update`` method.

    .. versionadded:: 2.0
    """
    from ._termui_impl import ProgressBar

    color = resolve_color_default(color)
    return ProgressBar(
        iterable=iterable,
        length=length,
        hidden=hidden,
        show_eta=show_eta,
        show_percent=show_percent,
        show_pos=show_pos,
        item_show_func=item_show_func,
        fill_char=fill_char,
        empty_char=empty_char,
        bar_template=bar_template,
        info_sep=info_sep,
        file=file,
        label=label,
        width=width,
        color=color,
        update_min_steps=update_min_steps,
    )


def clear() -> None:
    """Clears the terminal screen.  This will have the effect of clearing
    the whole visible space of the terminal and moving the cursor to the
    top left.  This does not do anything if not connected to a terminal.

    .. versionadded:: 2.0
    """
    if not isatty(sys.stdout):
        return

    # ANSI escape \033[2J clears the screen, \033[1;1H moves the cursor
    echo("\033[2J\033[1;1H", nl=False)


def _interpret_color(color: int | tuple[int, int, int] | str, offset: int = 0) -> str:
    """Interprets a color value and returns the corresponding ANSI code."""
    if isinstance(color, str) and color in _ansi_colors:
        return str(_ansi_colors[color] + offset)

    # bool is an int subclass: without the exclusion, True and False would
    # silently render as the palette indices 1 and 0.
    elif isinstance(color, int) and not isinstance(color, bool):
        if 0 <= color <= 255:
            return f"{38 + offset};5;{color:d}"

    elif (
        isinstance(color, (tuple, list))
        and len(color) == 3
        and all(
            isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255
            for c in color
        )
    ):
        r, g, b = color
        return f"{38 + offset};2;{r:d};{g:d};{b:d}"

    raise ValueError(_("Unknown color {colour!r}").format(colour=color))


def style(
    text: t.Any,
    fg: int | tuple[int, int, int] | str | None = None,
    bg: int | tuple[int, int, int] | str | None = None,
    bold: bool | None = None,
    dim: bool | None = None,
    underline: bool | None = None,
    overline: bool | None = None,
    italic: bool | None = None,
    blink: bool | None = None,
    reverse: bool | None = None,
    strikethrough: bool | None = None,
    reset: bool = True,
) -> str:
    """Styles a text with ANSI styles and returns the new string.  By
    default the styling is self contained which means that at the end
    of the string a reset code is issued.  This can be prevented by
    passing ``reset=False``.

    Examples::

        click.echo(click.style('Hello World!', fg='green'))
        click.echo(click.style('ATTENTION!', blink=True))
        click.echo(click.style('Some things', reverse=True, fg='cyan'))
        click.echo(click.style('More colors', fg=(255, 12, 128), bg=117))

    Supported color names:

    * ``black`` (might be a gray)
    * ``red``
    * ``green``
    * ``yellow`` (might be an orange)
    * ``blue``
    * ``magenta``
    * ``cyan``
    * ``white`` (might be light gray)
    * ``bright_black``
    * ``bright_red``
    * ``bright_green``
    * ``bright_yellow``
    * ``bright_blue``
    * ``bright_magenta``
    * ``bright_cyan``
    * ``bright_white``
    * ``reset`` (reset the color code only)

    If the terminal supports it, color may also be specified as:

    -   An integer in the interval [0, 255]. The terminal must support
        8-bit/256-color mode.
    -   An RGB tuple of three integers in [0, 255]. The terminal must
        support 24-bit/true-color mode.

    See https://en.wikipedia.org/wiki/ANSI_color and
    https://gist.github.com/XVilka/8346728 for more information.

    :param text: the string to style with ansi codes.
    :param fg: if provided this will become the foreground color.
    :param bg: if provided this will become the background color.
    :param bold: if provided this will enable or disable bold mode.
    :param dim: if provided this will enable or disable dim mode.  This is
                badly supported.
    :param underline: if provided this will enable or disable underline.
    :param overline: if provided this will enable or disable overline.
    :param italic: if provided this will enable or disable italic.
    :param blink: if provided this will enable or disable blinking.
    :param reverse: if provided this will enable or disable inverse
                    rendering (foreground becomes background and the
                    other way round).
    :param strikethrough: if provided this will enable or disable
        striking through text.
    :param reset: by default a reset-all code is added at the end of the
                  string which means that styles do not carry over.  This
                  can be disabled to compose styles.

    .. versionchanged:: 8.5.0
        All invalid color values raise :exc:`ValueError`. 256-color index
        ``0`` is no longer ignored.

    .. versionchanged:: 8.0
        A non-string ``message`` is converted to a string.

    .. versionchanged:: 8.0
       Added support for 256 and RGB color codes.

    .. versionchanged:: 8.0
        Added the ``strikethrough``, ``italic``, and ``overline``
        parameters.

    .. versionchanged:: 7.0
        Added support for bright colors.

    .. versionadded:: 2.0
    """
    if not isinstance(text, str):
        text = str(text)

    bits = []

    if fg is not None:
        bits.append(f"\033[{_interpret_color(fg)}m")

    if bg is not None:
        bits.append(f"\033[{_interpret_color(bg, 10)}m")

    if bold is not None:
        bits.append(f"\033[{1 if bold else 22}m")
    if dim is not None:
        bits.append(f"\033[{2 if dim else 22}m")
    if underline is not None:
        bits.append(f"\033[{4 if underline else 24}m")
    if overline is not None:
        bits.append(f"\033[{53 if overline else 55}m")
    if italic is not None:
        bits.append(f"\033[{3 if italic else 23}m")
    if blink is not None:
        bits.append(f"\033[{5 if blink else 25}m")
    if reverse is not None:
        bits.append(f"\033[{7 if reverse else 27}m")
    if strikethrough is not None:
        bits.append(f"\033[{9 if strikethrough else 29}m")
    bits.append(text)
    if reset:
        bits.append(_ansi_reset_all)
    return "".join(bits)


def unstyle(text: str) -> str:
    """Removes ANSI styling information from a string.  Usually it's not
    necessary to use this function as Click's echo function will
    automatically remove styling if necessary.

    .. versionadded:: 2.0

    :param text: the text to remove style information from.
    """
    return strip_ansi(text)


def secho(
    message: t.Any | None = None,
    file: t.IO[t.AnyStr] | None = None,
    nl: bool = True,
    err: bool = False,
    color: bool | None = None,
    **styles: t.Any,
) -> None:
    """This function combines :func:`echo` and :func:`style` into one
    call.  As such the following two calls are the same::

        click.secho('Hello World!', fg='green')
        click.echo(click.style('Hello World!', fg='green'))

    All keyword arguments are forwarded to the underlying functions
    depending on which one they go with.

    Non-string types will be converted to :class:`str`. However,
    :class:`bytes` are passed directly to :meth:`echo` without applying
    style. If you want to style bytes that represent text, call
    :meth:`bytes.decode` first.

    .. versionchanged:: 8.0
        A non-string ``message`` is converted to a string. Bytes are
        passed through without style applied.

    .. versionadded:: 2.0
    """
    if message is not None and not isinstance(message, (bytes, bytearray)):
        message = style(message, **styles)

    return echo(message, file=file, nl=nl, err=err, color=color)


@t.overload
def edit(
    text: bytes | bytearray,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = False,
    extension: str = ".txt",
) -> bytes | None: ...


@t.overload
def edit(
    text: str,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
) -> str | None: ...


@t.overload
def edit(
    text: None = None,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
    filename: str
    | os.PathLike[str]
    | cabc.Iterable[str | os.PathLike[str]]
    | None = None,
) -> None: ...


def edit(
    text: str | bytes | bytearray | None = None,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
    filename: str
    | os.PathLike[str]
    | cabc.Iterable[str | os.PathLike[str]]
    | None = None,
) -> str | bytes | bytearray | None:
    r"""Edits the given text in the defined editor.  If an editor is given
    (should be the full path to the executable but the regular operating
    system search path is used for finding the executable) it overrides
    the detected editor.  Optionally, some environment variables can be
    used.  If the editor is closed without changes, `None` is returned.  In
    case a file is edited directly the return value is always `None` and
    `require_save` and `extension` are ignored.

    If the editor cannot be opened a :exc:`UsageError` is raised.

    Note for Windows: to simplify cross-platform usage, the newlines are
    automatically converted from POSIX to Windows and vice versa.  As such,
    the message here will have ``\n`` as newline markers.

    :param text: the text to edit.
    :param editor: optionally the editor to use.  Defaults to automatic
                   detection.
    :param env: environment variables to forward to the editor.
    :param require_save: if this is true, then not saving in the editor
                         will make the return value become `None`.
    :param extension: the extension to tell the editor about.  This defaults
                      to `.txt` but changing this might change syntax
                      highlighting.
    :param filename: if provided it will edit this file instead of the
                     provided text contents.  It will not use a temporary
                     file as an indirection in that case. It accepts a path
                     or any iterable of paths. If the editor supports
                     editing multiple files at once, a sequence of files may
                     be passed as well. Invoke `click.file` once per file
                     instead if multiple files cannot be managed at once or
                     editing the files serially is desired.

    .. versionchanged:: 8.2.0
        ``filename`` now accepts any ``Iterable[str]`` in addition to a ``str``
        if the ``editor`` supports editing multiple files at once.

    .. versionchanged:: 8.5.0
        ``filename`` accepts ``os.PathLike`` values in addition to strings.

    """
    from ._termui_impl import Editor

    ed = Editor(editor=editor, env=env, require_save=require_save, extension=extension)

    if filename is None:
        return ed.edit(text)

    if isinstance(filename, (str, os.PathLike)):
        filename = (filename,)

    ed.edit_files(filenames=filename)
    return None


def launch(url: str, wait: bool = False, locate: bool = False) -> int:
    """This function launches the given URL (or filename) in the default
    viewer application for this file type.  If this is an executable, it
    might launch the executable in a new session.  The return value is
    the exit code of the launched application.  Usually, ``0`` indicates
    success.

    Examples::

        click.launch('https://click.palletsprojects.com/')
        click.launch('/my/downloaded/file', locate=True)

    .. versionadded:: 2.0

    :param url: URL or filename of the thing to launch.
    :param wait: Wait for the program to exit before returning. This
        only works if the launched program blocks. In particular,
        ``xdg-open`` on Linux does not block.
    :param locate: if this is set to `True` then instead of launching the
                   application associated with the URL it will attempt to
                   launch a file manager with the file located.  This
                   might have weird effects if the URL does not point to
                   the filesystem.
    """
    from ._termui_impl import open_url

    return open_url(url, wait=wait, locate=locate)


# If this is provided, getchar() calls into this instead.  This is used
# for unittesting purposes.
_getchar: t.Callable[[bool], str] | None = None


def getchar(echo: bool = False) -> str:
    """Fetches a single character from the terminal and returns it.  This
    will always return a unicode character and under certain rare
    circumstances this might return more than one character.  The
    situations which more than one character is returned is when for
    whatever reason multiple characters end up in the terminal buffer or
    standard input was not actually a terminal.

    Note that this will always read from the terminal, even if something
    is piped into the standard input.

    Note for Windows: in rare cases when typing non-ASCII characters, this
    function might wait for a second character and then return both at once.
    This is because certain Unicode characters look like special-key markers.

    .. versionadded:: 2.0

    :param echo: if set to `True`, the character read will also show up on
                 the terminal.  The default is to not show it.
    """
    global _getchar

    if _getchar is None:
        from ._termui_impl import getchar as f

        _getchar = f

    return _getchar(echo)


def raw_terminal() -> AbstractContextManager[int]:
    from ._termui_impl import raw_terminal as f

    return f()


def pause(info: str | None = None, err: bool = False) -> None:
    """This command stops execution and waits for the user to press any
    key to continue.  This is similar to the Windows batch "pause"
    command.  If the program is not run through a terminal, this command
    will instead do nothing.

    .. versionadded:: 2.0

    .. versionadded:: 4.0
       Added the `err` parameter.

    :param info: The message to print before pausing. Defaults to
        ``"Press any key to continue..."``.
    :param err: if set to message goes to ``stderr`` instead of
                ``stdout``, the same as with echo.
    """
    if not isatty(sys.stdin) or not isatty(sys.stdout):
        return

    if info is None:
        info = _("Press any key to continue...")

    try:
        if info:
            echo(info, nl=False, err=err)
        try:
            getchar()
        except (KeyboardInterrupt, EOFError):
            pass
    finally:
        if info:
            echo(err=err)
<|endoftext|>from __future__ import annotations

import collections.abc as cabc
import typing as t
from gettext import gettext as _
from gettext import ngettext

from ._compat import get_text_stderr
from .globals import resolve_color_default
from .utils import echo
from .utils import format_filename

if t.TYPE_CHECKING:
    from .core import Command
    from .core import Context
    from .core import Parameter


def _join_param_hints(param_hint: cabc.Sequence[str] | str | None) -> str | None:
    if param_hint is not None and not isinstance(param_hint, str):
        return " / ".join(repr(x) for x in param_hint)

    return param_hint


def _format_possibilities(possibilities: list[str]) -> str:
    possibility_str = ", ".join(repr(p) for p in sorted(possibilities))
    return ngettext(
        "Did you mean {possibility}?",
        "(Did you mean one of: {possibilities}?)",
        len(possibilities),
    ).format(possibility=possibility_str, possibilities=possibility_str)


class ClickException(Exception):
    """An exception that Click can handle and show to the user."""

    #: The exit code for this exception.
    exit_code: t.ClassVar[int] = 1

    show_color: t.Final[bool | None]
    message: t.Final[str]

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # The context will be removed by the time we print the message, so cache
        # the color settings here to be used later on (in `show`)
        self.show_color = resolve_color_default()
        self.message = message

    def format_message(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        if file is None:
            file = get_text_stderr()

        echo(
            _("Error: {message}").format(message=self.format_message()),
            file=file,
            color=self.show_color,
        )


class UsageError(ClickException):
    """An internal exception that signals a usage error.  This typically
    aborts any further handling.

    :param message: the error message to display.
    :param ctx: optionally the context that caused this error.  Click will
                fill in the context automatically in some situations.
    """

    exit_code: t.ClassVar[int] = 2

    ctx: Context | None
    cmd: t.Final[Command | None]

    def __init__(self, message: str, ctx: Context | None = None) -> None:
        super().__init__(message)
        self.ctx = ctx
        self.cmd = self.ctx.command if self.ctx else None

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        if file is None:
            file = get_text_stderr()
        color = None
        hint = ""
        if (
            self.ctx is not None
            and self.ctx.command.get_help_option(self.ctx) is not None
        ):
            help_names = self.ctx.command.get_help_option_names(self.ctx)
            # Pick the longest name (like ``--help`` over ``-h``) for
            # readability in error messages.
            hint = _("Try '{command} {option}' for help.").format(
                command=self.ctx.command_path,
                option=max(help_names, key=len),
            )
            hint = f"{hint}\n"
        if self.ctx is not None:
            color = self.ctx.color
            echo(f"{self.ctx.get_usage()}\n{hint}", file=file, color=color)
        echo(
            _("Error: {message}").format(message=self.format_message()),
            file=file,
            color=color,
        )


class BadParameter(UsageError):
    """An exception that formats out a standardized error message for a
    bad parameter.  This is useful when thrown from a callback or type as
    Click will attach contextual information to it (for instance, which
    parameter it is).

    .. versionadded:: 2.0

    :param param: the parameter object that caused this error.  This can
                  be left out, and Click will attach this info itself
                  if possible.
    :param param_hint: a string that shows up as parameter name.  This
                       can be used as alternative to `param` in cases
                       where custom validation should happen.  If it is
                       a string it's used as such, if it's a list then
                       each item is quoted and separated.
    """

    param: Parameter | None
    param_hint: cabc.Sequence[str] | str | None

    def __init__(
        self,
        message: str,
        ctx: Context | None = None,
        param: Parameter | None = None,
        param_hint: cabc.Sequence[str] | str | None = None,
    ) -> None:
        super().__init__(message, ctx)
        self.param = param
        self.param_hint = param_hint

    def format_message(self) -> str:
        if self.param_hint is not None:
            param_hint = self.param_hint
        elif self.param is not None:
            param_hint = self.param.get_error_hint(self.ctx)
        else:
            return _("Invalid value: {message}").format(message=self.message)

        return _("Invalid value for {param_hint}: {message}").format(
            param_hint=_join_param_hints(param_hint), message=self.message
        )


class MissingParameter(BadParameter):
    """Raised if click required an option or argument but it was not
    provided when invoking the script.

    .. versionadded:: 4.0

    :param param_type: a string that indicates the type of the parameter.
                       The default is to inherit the parameter type from
                       the given `param`.  Valid values are ``'parameter'``,
                       ``'option'`` or ``'argument'``.
    """

    param_type: t.Final[str | None]

    def __init__(
        self,
        message: str | None = None,
        ctx: Context | None = None,
        param: Parameter | None = None,
        param_hint: cabc.Sequence[str] | str | None = None,
        param_type: str | None = None,
    ) -> None:
        super().__init__(message or "", ctx, param, param_hint)
        self.param_type = param_type

    def format_message(self) -> str:
        if self.param_hint is not None:
            param_hint: cabc.Sequence[str] | str | None = self.param_hint
        elif self.param is not None:
            param_hint = self.param.get_error_hint(self.ctx)
        else:
            param_hint = None

        param_hint = _join_param_hints(param_hint)
        param_hint = f" {param_hint}" if param_hint else ""

        param_type = self.param_type
        if param_type is None and self.param is not None:
            param_type = self.param.param_type_name

        msg = self.message
        if self.param is not None:
            msg_extra = self.param.type.get_missing_message(
                param=self.param, ctx=self.ctx
            )
            if msg_extra:
                if msg:
                    msg += f". {msg_extra}"
                else:
                    msg = msg_extra

        msg = f" {msg}" if msg else ""

        # Translate param_type for known types.
        if param_type == "argument":
            missing = _("Missing argument")
        elif param_type == "option":
            missing = _("Missing option")
        elif param_type == "parameter":
            missing = _("Missing parameter")
        else:
            missing = _("Missing {param_type}").format(param_type=param_type)

        return f"{missing}{param_hint}.{msg}"

    def __str__(self) -> str:
        if not self.message:
            param_name = self.param.name if self.param else None
            return _("Missing parameter: {param_name}").format(param_name=param_name)
        else:
            return self.message


class NoSuchOption(UsageError):
    """Raised if Click attempted to handle an option that does not exist.

    .. versionadded:: 4.0
    """

    option_name: t.Final[str]
    possibilities: t.Final[list[str] | None]

    def __init__(
        self,
        option_name: str,
        message: str | None = None,
        possibilities: cabc.Iterable[str] | None = None,
        ctx: Context | None = None,
    ) -> None:
        if message is None:
            message = _("No such option {name!r}.").format(name=option_name)

        super().__init__(message, ctx)
        self.option_name = option_name

        if possibilities:
            from difflib import get_close_matches

            possibilities_ = get_close_matches(option_name, possibilities)
        else:
            possibilities_ = None
        self.possibilities = possibilities_

    def format_message(self) -> str:
        if not self.possibilities:
            return self.message
        return f"{self.message} {_format_possibilities(self.possibilities)}"


class NoSuchCommand(UsageError):
    """Raised if Click attempted to handle a command that does not exist.

    .. versionadded:: 8.4.0
    """

    command_name: t.Final[str]
    possibilities: t.Final[list[str] | None]

    def __init__(
        self,
        command_name: str,
        message: str | None = None,
        possibilities: cabc.Iterable[str] | None = None,
        ctx: Context | None = None,
    ) -> None:
        if message is None:
            message = _("No such command {name!r}.").format(name=command_name)

        super().__init__(message, ctx)
        self.command_name = command_name

        if possibilities:
            from difflib import get_close_matches

            possibilities_ = get_close_matches(command_name, possibilities)
        else:
            possibilities_ = None
        self.possibilities = possibilities_

    def format_message(self) -> str:
        if not self.possibilities:
            return self.message
        return f"{self.message} {_format_possibilities(self.possibilities)}"


class BadOptionUsage(UsageError):
    """Raised if an option is generally supplied but the use of the option
    was incorrect.  This is for instance raised if the number of arguments
    for an option is not correct.

    .. versionadded:: 4.0

    :param option_name: the name of the option being used incorrectly.
    """

    option_name: t.Final[str]

    def __init__(
        self, option_name: str, message: str, ctx: Context | None = None
    ) -> None:
        super().__init__(message, ctx)
        self.option_name = option_name


class BadArgumentUsage(UsageError):
    """Raised if an argument is generally supplied but the use of the argument
    was incorrect.  This is for instance raised if the number of values
    for an argument is not correct.

    .. versionadded:: 6.0
    """


class NoArgsIsHelpError(UsageError):
    ctx: Context

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx.get_help(), ctx=ctx)

    def show(self, file: t.IO[t.Any] | None = None) -> None:
        echo(self.format_message(), file=file, err=True, color=self.ctx.color)


class FileError(ClickException):
    """Raised if a file cannot be opened."""

    ui_filename: t.Final[str]
    filename: t.Final[str]

    def __init__(self, filename: str, hint: str | None = None) -> None:
        if hint is None:
            hint = _("unknown error")

        super().__init__(hint)
        self.ui_filename = format_filename(filename)
        self.filename = filename

    def format_message(self) -> str:
        return _("Could not open file {filename!r}: {message}").format(
            filename=self.ui_filename, message=self.message
        )


class Abort(RuntimeError):
    """An internal signalling exception that signals Click to abort."""


class Exit(RuntimeError):
    """An exception that indicates that the application should exit with some
    status code.

    :param code: the status code to exit with.
    """

    __slots__ = ("exit_code",)

    exit_code: t.Final[int]

    def __init__(self, code: int = 0) -> None:
        self.exit_code = code
<|endoftext|>"""Query string handling."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, SupportsInt, Union, cast

from multidict import istr

from ._quoters import QUERY_PART_QUOTER, QUERY_QUOTER

SimpleQuery = Union[str, SupportsInt, float]
QueryVariable = Union[SimpleQuery, Sequence[SimpleQuery]]
Query = Union[
    None, str, Mapping[str, QueryVariable], Sequence[tuple[str, QueryVariable]]
]


def query_var(v: SimpleQuery) -> str:
    """Convert a query variable to a string.

    Note: Objects implementing the ``__int__`` data model method (typed as
    ``SupportsInt``; e.g. ``uuid.UUID``) are converted via ``int()`` first.
    Callers should convert such values to ``str`` explicitly if the string
    representation is desired.
    """
    cls = type(v)
    if cls is int:  # Fast path for non-subclassed int
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if math.isinf(v):
            raise ValueError("float('inf') is not supported")
        if math.isnan(v):
            raise ValueError("float('nan') is not supported")
        return str(float(v))
    if cls is not bool and isinstance(v, SupportsInt):
        return str(int(v))
    raise TypeError(
        "Invalid variable type: value "
        "should be str, int or float, got {!r} "
        "of type {}".format(v, cls)
    )


def get_str_query_from_sequence_iterable(
    items: Iterable[tuple[str | istr, QueryVariable]],
) -> str:
    """Return a query string from a sequence of (key, value) pairs.

    value is a single value or a sequence of values for the key

    The sequence of values must be a list or tuple.
    """
    quoter = QUERY_PART_QUOTER
    pairs = [
        f"{quoter(k)}={quoter(v if type(v) is str else query_var(v))}"
        for k, val in items
        for v in (
            val if type(val) is not str and isinstance(val, (list, tuple)) else (val,)
        )
    ]
    return "&".join(pairs)


def get_str_query_from_iterable(
    items: Iterable[tuple[str | istr, SimpleQuery]],
) -> str:
    """Return a query string from an iterable.

    The iterable must contain (key, value) pairs.

    The values are not allowed to be sequences, only single values are
    allowed. For sequences, use `_get_str_query_from_sequence_iterable`.
    """
    quoter = QUERY_PART_QUOTER
    # A listcomp is used since listcomps are inlined on CPython 3.12+ and
    # they are a bit faster than a generator expression.
    pairs = [
        f"{quoter(k)}={quoter(v if type(v) is str else query_var(v))}" for k, v in items
    ]
    return "&".join(pairs)


def get_str_query(*args: Any, **kwargs: Any) -> str | None:
    """Return a query string from supported args."""
    query: (
        str
        | Mapping[str, QueryVariable]
        | Sequence[tuple[str | istr, SimpleQuery]]
        | None
    )
    if kwargs:
        if args:
            msg = "Either kwargs or single query parameter must be present"
            raise ValueError(msg)
        query = kwargs
    elif len(args) == 1:
        query = args[0]
    else:
        raise ValueError("Either kwargs or single query parameter must be present")

    if query is None:
        return None
    if not query:
        return ""
    if type(query) is dict:
        return get_str_query_from_sequence_iterable(query.items())
    if type(query) is str or isinstance(query, str):
        return QUERY_QUOTER(query)
    if isinstance(query, Mapping):
        return get_str_query_from_sequence_iterable(query.items())
    if isinstance(query, (bytes, bytearray, memoryview)):
        msg = "Invalid query type: bytes, bytearray and memoryview are forbidden"
        raise TypeError(msg)
    if isinstance(query, Sequence):
        # We don't expect sequence values if we're given a list of pairs
        # already; only mappings like builtin `dict` which can't have the
        # same key pointing to multiple values are allowed to use
        # `_query_seq_pairs`.
        if TYPE_CHECKING:
            query = cast(Sequence[tuple[Union[str, istr], SimpleQuery]], query)
        return get_str_query_from_iterable(query)
    raise TypeError(
        "Invalid query type: only str, mapping or "
        "sequence of (key, value) pairs is allowed"
    )
<|endoftext|>"""Utilities for working with paths."""

from collections.abc import Sequence
from contextlib import suppress


def normalize_path_segments(segments: Sequence[str]) -> list[str]:
    """Drop '.' and '..' from a sequence of str segments"""

    resolved_path: list[str] = []

    for seg in segments:
        if seg == "..":
            # ignore any .. segments that would otherwise cause an
            # IndexError when popped from resolved_path if
            # resolving for rfc3986
            with suppress(IndexError):
                resolved_path.pop()
        elif seg != ".":
            resolved_path.append(seg)

    if segments and segments[-1] in (".", ".."):
        # do some post-processing here.
        # if the last segment was a relative dir,
        # then we need to append the trailing '/'
        resolved_path.append("")

    return resolved_path


def normalize_path(path: str) -> str:
    # Drop '.' and '..' from str path
    prefix = ""
    if path and path[0] == "/":
        # preserve the "/" root element of absolute paths, copying it to the
        # normalised output as per sections 5.2.4 and 6.2.2.3 of rfc3986.
        prefix = "/"
        path = path[1:]

    segments = path.split("/")
    return prefix + "/".join(normalize_path_segments(segments))
<|endoftext|>"""URL parsing utilities."""

import codecs
import re
import unicodedata
from functools import lru_cache
from urllib.parse import parse_qsl, scheme_chars, uses_netloc

from ._quoters import QUOTER, UNQUOTER_PLUS

# Leading and trailing C0 control and space to be stripped per WHATWG spec.
# == "".join([chr(i) for i in range(0, 0x20 + 1)])
WHATWG_C0_CONTROL_OR_SPACE = (
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10"
    "\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f "
)

# Unsafe bytes to be removed per WHATWG spec
UNSAFE_URL_BYTES_TO_REMOVE = ["\t", "\r", "\n"]
USES_AUTHORITY = frozenset(uses_netloc)

SplitURLType = tuple[str, str, str, str, str]


def split_url(url: str) -> SplitURLType:
    """Split URL into parts."""
    # Adapted from urllib.parse.urlsplit
    # Only lstrip url as some applications rely on preserving trailing space.
    # (https://url.spec.whatwg.org/#concept-basic-url-parser would strip both)
    url = url.lstrip(WHATWG_C0_CONTROL_OR_SPACE)
    for b in UNSAFE_URL_BYTES_TO_REMOVE:
        if b in url:
            url = url.replace(b, "")

    scheme = netloc = query = fragment = ""
    i = url.find(":")
    if i > 0 and url[0] in scheme_chars:
        for c in url[1:i]:
            if c not in scheme_chars:
                break
        else:
            scheme, url = url[:i].lower(), url[i + 1 :]
    has_hash = "#" in url
    has_question_mark = "?" in url
    if url[:2] == "//":
        delim = len(url)  # position of end of domain part of url, default is end
        if has_hash and has_question_mark:
            delim_chars = "/?#"
        elif has_question_mark:
            delim_chars = "/?"
        elif has_hash:
            delim_chars = "/#"
        else:
            delim_chars = "/"
        for c in delim_chars:  # look for delimiters; the order is NOT important
            wdelim = url.find(c, 2)  # find first of this delim
            if wdelim >= 0 and wdelim < delim:  # if found
                delim = wdelim  # use earliest delim position
        netloc = url[2:delim]
        url = url[delim:]
        # Backslash is not valid in the authority component per RFC 3986.
        # WHATWG parsers treat \ as a path separator for special schemes, so
        # accepting it in the authority can cause host parsing ambiguity.
        if "\\" in netloc:
            raise ValueError(
                "Invalid URL: backslash ('\\') is not allowed in the authority "
                "component per RFC 3986."
            )
        has_left_bracket = "[" in netloc
        has_right_bracket = "]" in netloc
        if (has_left_bracket and not has_right_bracket) or (
            has_right_bracket and not has_left_bracket
        ):
            raise ValueError("Invalid IPv6 URL")
        if has_left_bracket:
            # Per RFC 3986, brackets are only valid at the START of the host
            # for IP-literal addresses. Text before '[' (e.g. '127.0.0.1[::1]')
            # is invalid and must be rejected to prevent SSRF bypasses. The
            # count checks reject URLs with more than one bracket pair in the
            # host subcomponent (e.g. 'http://[:localhost[]].google:80'),
            # which would otherwise resolve to an unintended host.
            hostinfo = netloc.rpartition("@")[2]
            if hostinfo[0] != "[" or hostinfo.count("[") > 1 or hostinfo.count("]") > 1:
                raise ValueError("Invalid IPv6 URL")
            bracketed_host, _, after_bracket = hostinfo[1:].partition("]")
            # Per RFC 3986 §3.2.2, after the closing ']' of an IP-literal
            # only ":" <port> or end-of-authority is valid. Any other text
            # (e.g. '[::1]allowed.example:1') must be rejected to prevent
            # host-confusion where the suffix is silently dropped.
            if after_bracket and after_bracket[0] != ":":
                raise ValueError("Invalid IPv6 URL")
            # Valid bracketed hosts are defined in
            # https://www.rfc-editor.org/rfc/rfc3986#page-49
            # https://url.spec.whatwg.org/
            if bracketed_host and bracketed_host[0] == "v":
                if not re.match(r"\Av[a-fA-F0-9]+\..+\Z", bracketed_host):
                    raise ValueError("IPvFuture address is invalid")
            elif ":" not in bracketed_host:
                raise ValueError("The IPv6 content between brackets is not valid")
    if has_hash:
        url, _, fragment = url.partition("#")
    if has_question_mark:
        url, _, query = url.partition("?")
    if netloc and not netloc.isascii():
        _check_netloc(netloc)
    return scheme, netloc, url, query, fragment


def _check_netloc(netloc: str) -> None:
    # Adapted from urllib.parse._checknetloc
    # looking for characters like \u2100 that expand to 'a/c'
    # IDNA uses NFKC equivalence, so normalize for this check

    # ignore characters already included
    # but not the surrounding text
    n = netloc.replace("@", "").replace(":", "").replace("#", "").replace("?", "")
    normalized_netloc = unicodedata.normalize("NFKC", n)
    if n == normalized_netloc:
        return
    # Note that there are no unicode decompositions for the character '@' so
    # its currently impossible to have test coverage for this branch, however if the
    # one should be added in the future we want to make sure its still checked.
    for c in "/?#@:%":  # pragma: no branch
        if c in normalized_netloc:
            raise ValueError(
                f"netloc '{netloc}' contains invalid "
                "characters under NFKC normalization"
            )


@lru_cache  # match the same size as urlsplit
def split_netloc(
    netloc: str,
) -> tuple[str | None, str | None, str | None, int | None]:
    """Split netloc into username, password, host and port."""
    if "@" not in netloc:
        username: str | None = None
        password: str | None = None
        hostinfo = netloc
    else:
        userinfo, _, hostinfo = netloc.rpartition("@")
        username, have_password, password = userinfo.partition(":")
        if not have_password:
            password = None

    if "[" in hostinfo:
        if hostinfo[0] != "[" or hostinfo.count("[") > 1 or hostinfo.count("]") > 1:
            raise ValueError("Invalid IPv6 URL")
        _, _, bracketed = hostinfo.partition("[")
        hostname, _, port_str = bracketed.partition("]")
        # Defense-in-depth: after ']' only ':port' or empty is valid.
        # split_url() should have already rejected invalid suffixes,
        # but guard here too for callers that use split_netloc() directly.
        if port_str and port_str[0] != ":":
            raise ValueError("Invalid IPv6 URL")
        _, _, port_str = port_str.partition(":")
    else:
        hostname, _, port_str = hostinfo.partition(":")

    if not port_str:
        return username or None, password, hostname or None, None

    try:
        port = int(port_str)
    except ValueError:
        raise ValueError("Invalid URL: port can't be converted to integer")
    if not (0 <= port <= 65535):
        raise ValueError("Port out of range 0-65535")
    return username or None, password, hostname or None, port


def unsplit_result(
    scheme: str, netloc: str, url: str, query: str, fragment: str
) -> str:
    """Unsplit a URL without any normalization."""
    if netloc or (scheme and scheme in USES_AUTHORITY) or url[:2] == "//":
        if url and url[:1] != "/":
            url = f"{scheme}://{netloc}/{url}" if scheme else f"{scheme}:{url}"
        else:
            url = f"{scheme}://{netloc}{url}" if scheme else f"//{netloc}{url}"
    elif scheme:
        url = f"{scheme}:{url}"
    if query:
        url = f"{url}?{query}"
    return f"{url}#{fragment}" if fragment else url


@lru_cache  # match the same size as urlsplit
def make_netloc(
    user: str | None,
    password: str | None,
    host: str | None,
    port: int | None,
    encode: bool = False,
) -> str:
    """Make netloc from parts.

    The user and password are encoded if encode is True.

    The host must already be encoded with _encode_host.
    """
    if host is None:
        return ""
    ret = host
    if port is not None:
        ret = f"{ret}:{port}"
    if user is None and password is None:
        return ret
    if password is not None:
        if not user:
            user = ""
        elif encode:
            user = QUOTER(user)
        if encode:
            password = QUOTER(password)
        user = f"{user}:{password}"
    elif user and encode:
        user = QUOTER(user)
    return f"{user}@{ret}" if user else ret


def query_to_pairs(
    query_string: str, *, max_fields: int | None = None, encoding: str = "utf-8"
) -> list[tuple[str, str]]:
    """Parse a query string into a list of decoded name, value pairs.

    The result is the same as
    ``urllib.parse.parse_qsl(query_string, keep_blank_values=True,
    encoding=encoding, max_num_fields=max_fields)``.

    Raises :exc:`ValueError` if *max_fields* is not ``None`` and the
    query string has more than *max_fields* fields. An empty query string
    returns an empty list on every Python version, even when *max_fields*
    is ``0``, where ``parse_qsl`` on Python 3.10 raises instead.
    """
    if not query_string:
        return []
    if max_fields is not None and query_string.count("&") >= max_fields:
        raise ValueError("Max number of fields exceeded")
    pairs: list[tuple[str, str]] = []
    if "%" not in query_string:
        # Nothing to decode except '+', which is the same in every encoding
        for name_value in query_string.replace("+", " ").split("&"):
            if name_value:
                name, _, value = name_value.partition("=")
                pairs.append((name, value))
        return pairs
    if encoding != "utf-8" and codecs.lookup(encoding).name != "utf-8":
        return parse_qsl(query_string, keep_blank_values=True, encoding=encoding)
    for name_value in query_string.split("&"):
        if name_value:
            name, _, value = name_value.partition("=")
            pairs.append((UNQUOTER_PLUS(name), UNQUOTER_PLUS(value)))
    return pairs
<|endoftext|>from ._parse import query_to_pairs
from ._query import Query, QueryVariable, SimpleQuery
from ._url import URL, cache_clear, cache_configure, cache_info

__version__ = "1.25.1"

__all__ = (
    "URL",
    "SimpleQuery",
    "QueryVariable",
    "Query",
    "cache_clear",
    "cache_configure",
    "cache_info",
    "query_to_pairs",
)
<|endoftext|>import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import _CacheInfo, lru_cache
from importlib.util import find_spec
from ipaddress import ip_address
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)
from urllib.parse import SplitResult, scheme_chars, uses_relative

import idna
from multidict import MultiDict, MultiDictProxy, istr
from propcache.api import under_cached_property as cached_property

from ._parse import (
    USES_AUTHORITY,
    SplitURLType,
    make_netloc,
    query_to_pairs,
    split_netloc,
    split_url,
    unsplit_result,
)
from ._path import normalize_path, normalize_path_segments
from ._query import (
    Query,
    QueryVariable,
    SimpleQuery,
    get_str_query,
    get_str_query_from_iterable,
    get_str_query_from_sequence_iterable,
)
from ._quoters import (
    FRAGMENT_QUOTER,
    FRAGMENT_REQUOTER,
    PATH_QUOTER,
    PATH_REQUOTER,
    PATH_SAFE_UNQUOTER,
    QS_UNQUOTER,
    QUERY_QUOTER,
    QUERY_REQUOTER,
    QUOTER,
    REQUOTER,
    UNQUOTER,
    human_quote,
)

# Avoid Pydantic import if not used (increases yarl's import time by 3-7x).
HAS_PYDANTIC = find_spec("pydantic_core") is not None
if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema


DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}
USES_RELATIVE = frozenset(uses_relative)
_SCHEME_CHARS = frozenset(scheme_chars)

# Special schemes https://url.spec.whatwg.org/#special-scheme
# are not allowed to have an empty host https://url.spec.whatwg.org/#url-representation
SCHEME_REQUIRES_HOST = frozenset(("http", "https", "ws", "wss", "ftp"))


# reg-name: unreserved / pct-encoded / sub-delims
# this pattern matches anything that is *not* in those classes. and is only used
# on lower-cased ASCII values.
NOT_REG_NAME = re.compile(
    r"""
        # any character not in the unreserved or sub-delims sets, plus %
        # (validated with the additional check for pct-encoded sequences below)
        [^a-z0-9\-._~!$&'()*+,;=%]
    |
        # % only allowed if it is part of a pct-encoded
        # sequence of 2 hex digits.
        %(?![0-9a-f]{2})
    """,
    re.VERBOSE,
)

# Invisible default-ignorable / format code points that must not appear in a
# host (soft hyphen, zero-width space, word joiner, bidi controls, variation
# selectors, ...). Depending on the code point IDNA either silently deletes it
# (so ``e<ZWSP>vil.com`` encodes to ``evil.com``) or folds it into a different
# punycode host; either way the parsed host differs from the string an
# application validated. The set is the union of two authoritative sources,
# matching the two encoders _idna_encode dispatches to:
#
# 1. Unicode Default_Ignorable_Code_Point (uts46=True path via the ``idna``
#    package). Ranges taken from the DerivedCoreProperties data file:
#    https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
#    (the E0000..E0FFF block is contiguous under this property).
# 2. RFC 3454 (Stringprep) Table B.1 "commonly mapped to nothing", used by the
#    stdlib ``str.encode("idna")`` / IDNA2003 nameprep fallback:
#    https://www.rfc-editor.org/rfc/rfc3454#appendix-B.1
#    This is the source of U+1806, which is not Default_Ignorable.
#
# Coverage is pinned to the installed ``idna``/Unicode data by a sweep test
# (test_default_ignorable_covers_idna_stripped in tests/test_url.py) that
# brute-forces every code point through _idna_encode and fails if any point it
# silently deletes is not matched here.
_DEFAULT_IGNORABLE_RE = re.compile(
    "["
    "\u00ad"  # SOFT HYPHEN
    "\u034f"  # COMBINING GRAPHEME JOINER
    "\u061c"  # ARABIC LETTER MARK
    "\u115f-\u1160"  # HANGUL CHOSEONG/JUNGSEONG FILLER
    "\u17b4-\u17b5"  # KHMER VOWEL INHERENT AQ/AA
    "\u1806"  # MONGOLIAN TODO SOFT HYPHEN (nameprep maps to nothing)
    "\u180b-\u180f"  # MONGOLIAN FVS ONE..FOUR and VOWEL SEPARATOR
    "\u200b-\u200f"  # ZERO WIDTH SPACE..RIGHT-TO-LEFT MARK
    "\u202a-\u202e"  # bidi embedding/override controls
    "\u2060-\u206f"  # WORD JOINER..NOMINAL DIGIT SHAPES
    "\u3164"  # HANGUL FILLER
    "\ufe00-\ufe0f"  # VARIATION SELECTOR-1..16
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\uffa0"  # HALFWIDTH HANGUL FILLER
    "\ufff0-\ufff8"  # reserved default-ignorables
    "\U0001bca0-\U0001bca3"  # SHORTHAND FORMAT controls
    "\U0001d173-\U0001d17a"  # MUSICAL SYMBOL begin/end controls
    "\U000e0000-\U000e0fff"  # tags and VARIATION SELECTOR SUPPLEMENT
    "]"
)

# Zone IDs are OS-specific text strings with no format defined by the RFCs:
# https://datatracker.ietf.org/doc/html/rfc4007#section-11.2
# RFC 9844 §6.3 recommends rejecting characters inappropriate for the
# environment; for yarl we reject ASCII control characters (CTL):
# https://datatracker.ietf.org/doc/html/rfc9844#section-6-3
_ZONE_ID_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f]")

_T = TypeVar("_T")

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


class UndefinedType(Enum):
    """Singleton type for use with not set sentinel values."""

    _singleton = 0


UNDEFINED = UndefinedType._singleton


class CacheInfo(TypedDict):
    """Host encoding cache."""

    idna_encode: _CacheInfo
    idna_decode: _CacheInfo
    ip_address: _CacheInfo
    host_validate: _CacheInfo
    encode_host: _CacheInfo


class _InternalURLCache(TypedDict, total=False):
    _val: SplitURLType
    _origin: "URL"
    absolute: bool
    hash: int
    scheme: str
    raw_authority: str
    authority: str
    raw_user: str | None
    user: str | None
    raw_password: str | None
    password: str | None
    raw_host: str | None
    host: str | None
    host_subcomponent: str | None
    host_port_subcomponent: str | None
    port: int | None
    explicit_port: int | None
    raw_path: str
    path: str
    _parsed_query: list[tuple[str, str]]
    query: "MultiDictProxy[str]"
    raw_query_string: str
    query_string: str
    path_qs: str
    raw_path_qs: str
    raw_fragment: str
    fragment: str
    raw_parts: tuple[str, ...]
    parts: tuple[str, ...]
    parent: "URL"
    raw_name: str
    name: str
    raw_suffix: str
    suffix: str
    raw_suffixes: tuple[str, ...]
    suffixes: tuple[str, ...]


def rewrite_module(obj: _T) -> _T:
    obj.__module__ = "yarl"
    return obj


def _encode_relative_scheme_colon(path: str) -> str:
    """Re-encode a scheme-shaped leading ``:`` in a relative path to ``%3A``."""
    colon_pos = path.find(":")
    if colon_pos <= 0:
        return path
    for c in path[:colon_pos]:
        if c not in _SCHEME_CHARS:
            return path
    return path[:colon_pos] + "%3A" + path[colon_pos + 1 :]


@lru_cache
def encode_url(url_str: str) -> "URL":
    """Parse unencoded URL."""
    cache: _InternalURLCache = {}
    host: str | None
    scheme, netloc, path, query, fragment = split_url(url_str)
    if not netloc:  # netloc
        host = ""
    else:
        if ":" in netloc or "@" in netloc or "[" in netloc:
            # Complex netloc
            username, password, host, port = split_netloc(netloc)
        else:
            username = password = port = None
            host = netloc
        if host is None:
            if scheme in SCHEME_REQUIRES_HOST:
                msg = (
                    "Invalid URL: host is required for "
                    f"absolute urls with the {scheme} scheme"
                )
                raise ValueError(msg)
            else:
                host = ""
        host = _encode_host(host, validate_host=False)
        # Remove brackets as host encoder adds back brackets for IPv6 addresses
        cache["raw_host"] = host[1:-1] if "[" in host else host
        cache["explicit_port"] = port
        if password is None and username is None:
            # Fast path for URLs without user, password
            netloc = host if port is None else f"{host}:{port}"
            cache["raw_user"] = None
            cache["raw_password"] = None
        else:
            raw_user = REQUOTER(username) if username else username
            raw_password = REQUOTER(password) if password else password
            netloc = make_netloc(raw_user, raw_password, host, port)
            cache["raw_user"] = raw_user
            cache["raw_password"] = raw_password

    if path:
        path = PATH_REQUOTER(path)
        if netloc and "." in path:
            path = normalize_path(path)
        elif not scheme and not netloc:
            path = _encode_relative_scheme_colon(path)
    if query:
        query = QUERY_REQUOTER(query)
    if fragment:
        fragment = FRAGMENT_REQUOTER(fragment)

    cache["scheme"] = scheme
    cache["raw_path"] = "/" if not path and netloc else path
    cache["raw_query_string"] = query
    cache["raw_fragment"] = fragment

    self = object.__new__(URL)
    self._scheme = scheme
    self._netloc = netloc
    self._path = path
    self._query = query
    self._fragment = fragment
    self._cache = cache
    return self


@lru_cache
def pre_encoded_url(url_str: str) -> "URL":
    """Parse pre-encoded URL."""
    self = object.__new__(URL)
    val = split_url(url_str)
    self._scheme, self._netloc, self._path, self._query, self._fragment = val
    self._cache = {}
    return self


@lru_cache
def build_pre_encoded_url(
    scheme: str,
    authority: str,
    user: str | None,
    password: str | None,
    host: str,
    port: int | None,
    path: str,
    query_string: str,
    fragment: str,
) -> "URL":
    """Build a pre-encoded URL from parts."""
    self = object.__new__(URL)
    self._scheme = scheme
    if authority:
        self._netloc = authority
    elif host:
        if port is not None:
            port = None if port == DEFAULT_PORTS.get(scheme) else port
        if user is None and password is None:
            self._netloc = host if port is None else f"{host}:{port}"
        else:
            self._netloc = make_netloc(user, password, host, port)
    else:
        self._netloc = ""
    if path and not scheme and not self._netloc and ":" in path:
        path = _encode_relative_scheme_colon(path)
    self._path = path
    self._query = query_string
    self._fragment = fragment
    self._cache = {}
    return self


def from_parts_uncached(
    scheme: str, netloc: str, path: str, query: str, fragment: str
) -> "URL":
    """Create a new URL from parts."""
    self = object.__new__(URL)
    self._scheme = scheme
    self._netloc = netloc
    if path and not scheme and not netloc and ":" in path:
        path = _encode_relative_scheme_colon(path)
    self._path = path
    self._query = query
    self._fragment = fragment
    self._cache = {}
    return self


from_parts = lru_cache(from_parts_uncached)


@rewrite_module
class URL:
    # Don't derive from str
    # follow pathlib.Path design
    # probably URL will not suffer from pathlib problems:
    # it's intended for libraries like aiohttp,
    # not to be passed into standard library functions like os.open etc.

    # URL grammar (RFC 3986)
    # pct-encoded = "%" HEXDIG HEXDIG
    # reserved    = gen-delims / sub-delims
    # gen-delims  = ":" / "/" / "?" / "#" / "[" / "]" / "@"
    # sub-delims  = "!" / "$" / "&" / "'" / "(" / ")"
    #             / "*" / "+" / "," / ";" / "="
    # unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
    # URI         = scheme ":" hier-part [ "?" query ] [ "#" fragment ]
    # hier-part   = "//" authority path-abempty
    #             / path-absolute
    #             / path-rootless
    #             / path-empty
    # scheme      = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
    # authority   = [ userinfo "@" ] host [ ":" port ]
    # userinfo    = *( unreserved / pct-encoded / sub-delims / ":" )
    # host        = IP-literal / IPv4address / reg-name
    # IP-literal = "[" ( IPv6address / IPvFuture  ) "]"
    # IPvFuture  = "v" 1*HEXDIG "." 1*( unreserved / sub-delims / ":" )
    # IPv6address =                            6( h16 ":" ) ls32
    #             /                       "::" 5( h16 ":" ) ls32
    #             / [               h16 ] "::" 4( h16 ":" ) ls32
    #             / [ *1( h16 ":" ) h16 ] "::" 3( h16 ":" ) ls32
    #             / [ *2( h16 ":" ) h16 ] "::" 2( h16 ":" ) ls32
    #             / [ *3( h16 ":" ) h16 ] "::"    h16 ":"   ls32
    #             / [ *4( h16 ":" ) h16 ] "::"              ls32
    #             / [ *5( h16 ":" ) h16 ] "::"              h16
    #             / [ *6( h16 ":" ) h16 ] "::"
    # ls32        = ( h16 ":" h16 ) / IPv4address
    #             ; least-significant 32 bits of address
    # h16         = 1*4HEXDIG
    #             ; 16 bits of address represented in hexadecimal
    # IPv4address = dec-octet "." dec-octet "." dec-octet "." dec-octet
    # dec-octet   = DIGIT                 ; 0-9
    #             / %x31-39 DIGIT         ; 10-99
    #             / "1" 2DIGIT            ; 100-199
    #             / "2" %x30-34 DIGIT     ; 200-249
    #             / "25" %x30-35          ; 250-255
    # reg-name    = *( unreserved / pct-encoded / sub-delims )
    # port        = *DIGIT
    # path          = path-abempty    ; begins with "/" or is empty
    #               / path-absolute   ; begins with "/" but not "//"
    #               / path-noscheme   ; begins with a non-colon segment
    #               / path-rootless   ; begins with a segment
    #               / path-empty      ; zero characters
    # path-abempty  = *( "/" segment )
    # path-absolute = "/" [ segment-nz *( "/" segment ) ]
    # path-noscheme = segment-nz-nc *( "/" segment )
    # path-rootless = segment-nz *( "/" segment )
    # path-empty    = 0<pchar>
    # segment       = *pchar
    # segment-nz    = 1*pchar
    # segment-nz-nc = 1*( unreserved / pct-encoded / sub-delims / "@" )
    #               ; non-zero-length segment without any colon ":"
    # pchar         = unreserved / pct-encoded / sub-delims / ":" / "@"
    # query       = *( pchar / "/" / "?" )
    # fragment    = *( pchar / "/" / "?" )
    # URI-reference = URI / relative-ref
    # relative-ref  = relative-part [ "?" query ] [ "#" fragment ]
    # relative-part = "//" authority path-abempty
    #               / path-absolute
    #               / path-noscheme
    #               / path-empty
    # absolute-URI  = scheme ":" hier-part [ "?" query ]
    __slots__ = ("_cache", "_scheme", "_netloc", "_path", "_query", "_fragment")

    _cache: _InternalURLCache
    _scheme: str
    _netloc: str
    _path: str
    _query: str
    _fragment: str

    def __new__(
        cls,
        val: Union[str, SplitResult, "URL", UndefinedType] = UNDEFINED,
        *,
        encoded: bool = False,
        strict: bool | None = None,
    ) -> "URL":
        if strict is not None:  # pragma: no cover
            warnings.warn("strict parameter is ignored")
        if type(val) is str:
            return pre_encoded_url(val) if encoded else encode_url(val)
        if type(val) is cls:
            return val
        if type(val) is SplitResult:
            if not encoded:
                raise ValueError("Cannot apply decoding to SplitResult")
            return from_parts(*val)
        if isinstance(val, str):
            return pre_encoded_url(str(val)) if encoded else encode_url(str(val))
        if val is UNDEFINED:
            # Special case for UNDEFINED since it might be unpickling and we do
            # not want to cache as the `__set_state__` call would mutate the URL
            # object in the `pre_encoded_url` or `encoded_url` caches.
            self = object.__new__(URL)
            self._scheme = self._netloc = self._path = self._query = self._fragment = ""
            self._cache = {}
            return self
        raise TypeError("Constructor parameter should be str")

    @classmethod
    def build(
        cls,
        *,
        scheme: str = "",
        authority: str = "",
        user: str | None = None,
        password: str | None = None,
        host: str = "",
        port: int | None = None,
        path: str = "",
        query: Query | None = None,
        query_string: str = "",
        fragment: str = "",
        encoded: bool = False,
    ) -> "URL":
        """Creates and returns a new URL"""

        if authority and (user or password or host or port):
            raise ValueError(
                'Can\'t mix "authority" with "user", "password", "host" or "port".'
            )
        if port is not None and not isinstance(port, int):
            raise TypeError(f"The port is required to be int, got {type(port)!r}.")
        if port and not host:
            raise ValueError('Can\'t build URL with "port" but without "host".')
        if query and query_string:
            raise ValueError('Only one of "query" or "query_string" should be passed')
        if (
            scheme is None  # type: ignore[redundant-expr]
            or authority is None  # type: ignore[redundant-expr]
            or host is None  # type: ignore[redundant-expr]
            or path is None  # type: ignore[redundant-expr]
            or query_string is None  # type: ignore[redundant-expr]
            or fragment is None
        ):
            raise TypeError(
                'NoneType is illegal for "scheme", "authority", "host", "path", '
                '"query_string", and "fragment" args, use empty string instead.'
            )

        if query:
            query_string = get_str_query(query) or ""

        if encoded:
            return build_pre_encoded_url(
                scheme,
                authority,
                user,
                password,
                host,
                port,
                path,
                query_string,
                fragment,
            )

        self = object.__new__(URL)
        self._scheme = scheme
        _host: str | None = None
        if authority:
            user, password, _host, port = split_netloc(authority)
            _host = _encode_host(_host, validate_host=False) if _host else ""
        elif host:
            _host = _encode_host(host, validate_host=True)
        else:
            self._netloc = ""

        if _host is not None:
            if port is not None:
                port = None if port == DEFAULT_PORTS.get(scheme) else port
            if user is None and password is None:
                self._netloc = _host if port is None else f"{_host}:{port}"
            else:
                self._netloc = make_netloc(user, password, _host, port, True)

        path = PATH_QUOTER(path) if path else path
        if path and self._netloc:
            if "." in path:
                path = normalize_path(path)
            if path[0] != "/":
                msg = (
                    "Path in a URL with authority should "
                    "start with a slash ('/') if set"
                )
                raise ValueError(msg)

        if path and not self._scheme and not self._netloc and ":" in path:
            path = _encode_relative_scheme_colon(path)
        self._path = path
        if not query and query_string:
            query_string = QUERY_QUOTER(query_string)
        self._query = query_string
        self._fragment = FRAGMENT_QUOTER(fragment) if fragment else fragment
        self._cache = {}
        return self

    def __init_subclass__(cls) -> NoReturn:
        raise TypeError(f"Inheriting a class {cls!r} from URL is forbidden")

    def __str__(self) -> str:
        if not self._path and self._netloc and (self._query or self._fragment):
            path = "/"
        else:
            path = self._path
        if (port := self.explicit_port) is not None and port == DEFAULT_PORTS.get(
            self._scheme
        ):
            # port normalization - using None for default ports to remove from rendering
            # https://datatracker.ietf.org/doc/html/rfc3986.html#section-6.2.3
            host = self.host_subcomponent
            netloc = make_netloc(self.raw_user, self.raw_password, host, None)
        else:
            netloc = self._netloc
        return unsplit_result(self._scheme, netloc, path, self._query, self._fragment)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{str(self)}')"

    def __bytes__(self) -> bytes:
        return str(self).encode("ascii")

    def __eq__(self, other: object) -> bool:
        if type(other) is not URL:
            return NotImplemented

        path1 = "/" if not self._path and self._netloc else self._path
        path2 = "/" if not other._path and other._netloc else other._path
        return (
            self._scheme == other._scheme
            and self._netloc == other._netloc
            and path1 == path2
            and self._query == other._query
            and self._fragment == other._fragment
        )

    def __hash__(self) -> int:
        if (ret := self._cache.get("hash")) is None:
            path = "/" if not self._path and self._netloc else self._path
            ret = self._cache["hash"] = hash(
                (self._scheme, self._netloc, path, self._query, self._fragment)
            )
        return ret

    def __le__(self, other: object) -> bool:
        if type(other) is not URL:
            return NotImplemented
        return self._val <= other._val

    def __lt__(self, other: object) -> bool:
        if type(other) is not URL:
            return NotImplemented
        return self._val < other._val

    def __ge__(self, other: object) -> bool:
        if type(other) is not URL:
            return NotImplemented
        return self._val >= other._val

    def __gt__(self, other: object) -> bool:
        if type(other) is not URL:
            return NotImplemented
        return self._val > other._val

    def __truediv__(self, name: str) -> "URL":
        if not isinstance(name, str):
            return NotImplemented
        return self._make_child((str(name),))

    def __mod__(self, query: Query) -> "URL":
        return self.update_query(query)

    def __bool__(self) -> bool:
        return bool(self._netloc or self._path or self._query or self._fragment)

    def __getstate__(self) -> tuple[SplitURLType]:
        # Return a plain tuple rather than a ``SplitResult``. Constructing a
        # ``SplitResult`` via ``tuple.__new__`` skips its ``__init__`` and on
        # Python 3.15+ leaves ``_keep_empty`` unset, which breaks pickling: the
        # new ``SplitResult.__getstate__`` indexes a state that ends up as
        # ``None`` (gh-1632). ``__setstate__`` already unpacks both shapes, so
        # pickles produced by older yarl releases (which embed a real
        # ``SplitResult``) still load correctly.
        return (self._val,)

    def __setstate__(
        self, state: tuple[SplitURLType] | tuple[None, _InternalURLCache]
    ) -> None:
        if state[0] is None and isinstance(state[1], dict):
            # default style pickle
            val = state[1]["_val"]
        else:
            unused: list[object]
            val, *unused = state
        self._scheme, self._netloc, self._path, self._query, self._fragment = val
        self._cache = {}

    def _cache_netloc(self) -> None:
        """Cache the netloc parts of the URL."""
        c = self._cache
        split_loc = split_netloc(self._netloc)
        c["raw_user"], c["raw_password"], c["raw_host"], c["explicit_port"] = split_loc

    def is_absolute(self) -> bool:
        """A check for absolute URLs.

        Return True for absolute ones (having scheme or starting
        with //), False otherwise.

        Is is preferred to call the .absolute property instead
        as it is cached.
        """
        return self.absolute

    def is_default_port(self) -> bool:
        """A check for default port.

        Return True if port is default for specified scheme,
        e.g. 'http://python.org' or 'http://python.org:80', False
        otherwise.

        Return False for relative URLs.

        """
        if (explicit := self.explicit_port) is None:
            # If the explicit port is None, then the URL must be
            # using the default port unless its a relative URL
            # which does not have an implicit port / default port
            return self._netloc != ""
        return explicit == DEFAULT_PORTS.get(self._scheme)

    def origin(self) -> "URL":
        """Return an URL with scheme, host and port parts only.

        user, password, path, query and fragment are removed.

        """
        # TODO: add a keyword-only option for keeping user/pass maybe?
        return self._origin

    @cached_property
    def _val(self) -> SplitURLType:
        return (self._scheme, self._netloc, self._path, self._query, self._fragment)

    @cached_property
    def _origin(self) -> "URL":
        """Return an URL with scheme, host and port parts only.

        user, password, path, query and fragment are removed.
        """
        if not (netloc := self._netloc):
            raise ValueError("URL should be absolute")
        if not (scheme := self._scheme):
            raise ValueError("URL should have scheme")
        if "@" in netloc:
            encoded_host = self.host_subcomponent
            netloc = make_netloc(None, None, encoded_host, self.explicit_port)
        elif not self._path and not self._query and not self._fragment:
            return self
        return from_parts(scheme, netloc, "", "", "")

    def relative(self) -> "URL":
        """Return a relative part of the URL.

        scheme, user, password, host and port are removed.

        """
        if not self._netloc:
            raise ValueError("URL should be absolute")
        return from_parts("", "", self._path, self._query, self._fragment)

    @cached_property
    def absolute(self) -> bool:
        """A check for absolute URLs.

        Return True for absolute ones (having scheme or starting
        with //), False otherwise.

        """
        # `netloc`` is an empty string for relative URLs
        # Checking `netloc` is faster than checking `hostname`
        # because `hostname` is a property that does some extra work
        # to parse the host from the `netloc`
        return self._netloc != ""

    @cached_property
    def scheme(self) -> str:
        """Scheme for absolute URLs.

        Empty string for relative URLs or URLs starting with //

        """
        return self._scheme

    @cached_property
    def raw_authority(self) -> str:
        """Encoded authority part of URL.

        Empty string for relative URLs.

        """
        return self._netloc

    @cached_property
    def authority(self) -> str:
        """Decoded authority part of URL.

        Empty string for relative URLs.

        """
        return make_netloc(self.user, self.password, self.host, self.port)

    @cached_property
    def raw_user(self) -> str | None:
        """Encoded user part of URL.

        None if user is missing.

        """
        # not .username
        self._cache_netloc()
        return self._cache["raw_user"]

    @cached_property
    def user(self) -> str | None:
        """Decoded user part of URL.

        None if user is missing.

        """
        if (raw_user := self.raw_user) is None:
            return None
        return UNQUOTER(raw_user)

    @cached_property
    def raw_password(self) -> str | None:
        """Encoded password part of URL.

        None if password is missing.

        """
        self._cache_netloc()
        return self._cache["raw_password"]

    @cached_property
    def password(self) -> str | None:
        """Decoded password part of URL.

        None if password is missing.

        """
        if (raw_password := self.raw_password) is None:
            return None
        return UNQUOTER(raw_password)

    @cached_property
    def raw_host(self) -> str | None:
        """Encoded host part of URL.

        None for relative URLs.

        When working with IPv6 addresses, use the `host_subcomponent` property instead
        as it will return the host subcomponent with brackets.
        """
        # Use host instead of hostname for sake of shortness
        # May add .hostname prop later
        self._cache_netloc()
        return self._cache["raw_host"]

    @cached_property
    def host(self) -> str | None:
        """Decoded host part of URL.

        None for relative URLs.

        For IPv6 hosts that carry an RFC 6874 zone identifier, the
        ``%25`` zone separator is decoded back to ``%``; the encoded
        form is still available via :attr:`raw_host` and
        :attr:`host_subcomponent`.

        """
        if (raw := self.raw_host) is None:
            return None
        if raw and raw[-1].isdigit() or ":" in raw:
            # IP addresses are never IDNA encoded. The replace decodes
            # every %25 in the raw host, i.e. the RFC 6874 zone
            # separator and any %25 that percent-encodes a literal %
            # inside the zone identifier.
            if "%25" in raw:
                return raw.replace("%25", "%")
            return raw
        return _idna_decode(raw)

    @cached_property
    def host_subcomponent(self) -> str | None:
        """Return the host subcomponent part of URL.

        None for relative URLs.

        https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.2

        `IP-literal = "[" ( IPv6address / IPvFuture  ) "]"`

        Examples:
        - `http://example.com:8080` -> `example.com`
        - `http://example.com:80` -> `example.com`
        - `https://127.0.0.1:8443` -> `127.0.0.1`
        - `https://[::1]:8443` -> `[::1]`
        - `http://[::1]` -> `[::1]`

        """
        if (raw := self.raw_host) is None:
            return None
        return f"[{raw}]" if ":" in raw else raw

    @cached_property
    def host_port_subcomponent(self) -> str | None:
        """Return the host and port subcomponent part of URL.

        Trailing dots are removed from the host part.

        This value is suitable for use in the Host header of an HTTP request.

        None for relative URLs.

        https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.2
        `IP-literal = "[" ( IPv6address / IPvFuture  ) "]"`
        https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.3
        port        = *DIGIT

        Examples:
        - `http://example.com:8080` -> `example.com:8080`
        - `http://example.com:80` -> `example.com`
        - `http://example.com.:80` -> `example.com`
        - `https://127.0.0.1:8443` -> `127.0.0.1:8443`
        - `https://[::1]:8443` -> `[::1]:8443`
        - `http://[::1]` -> `[::1]`

        """
        if (raw := self.raw_host) is None:
            return None
        if raw[-1] == ".":
            # Remove all trailing dots from the netloc as while
            # they are valid FQDNs in DNS, TLS validation fails.
            # See https://github.com/aio-libs/aiohttp/issues/3636.
            # To avoid string manipulation we only call rstrip if
            # the last character is a dot.
            raw = raw.rstrip(".")
        port = self.explicit_port
        if port is None or port == DEFAULT_PORTS.get(self._scheme):
            return f"[{raw}]" if ":" in raw else raw
        return f"[{raw}]:{port}" if ":" in raw else f"{raw}:{port}"

    @cached_property
    def port(self) -> int | None:
        """Port part of URL, with scheme-based fallback.

        None for relative URLs or URLs without explicit port and
        scheme without default port substitution.

        """
        if (explicit_port := self.explicit_port) is not None:
            return explicit_port
        return DEFAULT_PORTS.get(self._scheme)

    @cached_property
    def explicit_port(self) -> int | None:
        """Port part of URL, without scheme-based fallback.

        None for relative URLs or URLs without explicit port.

        """
        self._cache_netloc()
        return self._cache["explicit_port"]

    @cached_property
    def raw_path(self) -> str:
        """Encoded path of URL.

        / for absolute URLs without path part.

        """
        return self._path if self._path or not self._netloc else "/"

    @cached_property
    def path(self) -> str:
        """Decoded path of URL.

        / for absolute URLs without path part.

        """
        return UNQUOTER(self._path) if self._path else "/" if self._netloc else ""

    @cached_property
    def path_safe(self) -> str:
        """Decoded path of URL.

        / for absolute URLs without path part.

        / (%2F) and % (%25) are not decoded

        """
        if self._path:
            return PATH_SAFE_UNQUOTER(self._path)
        return "/" if self._netloc else ""

    @cached_property
    def _parsed_query(self) -> list[tuple[str, str]]:
        """Parse query part of URL."""
        return query_to_pairs(self._query)

    @cached_property
    def query(self) -> "MultiDictProxy[str]":
        """A MultiDictProxy representing parsed query parameters in decoded
        representation.

        Empty value if URL has no query part.

        """
        return MultiDictProxy(MultiDict(self._parsed_query))

    @cached_property
    def raw_query_string(self) -> str:
        """Encoded query part of URL.

        Empty string if query is missing.

        """
        return self._query

    @cached_property
    def query_string(self) -> str:
        """Decoded query part of URL.

        Empty string if query is missing.

        """
        return QS_UNQUOTER(self._query) if self._query else ""

    @cached_property
    def path_qs(self) -> str:
        """Decoded path of URL with query."""
        return self.path if not (q := self.query_string) else f"{self.path}?{q}"

    @cached_property
    def raw_path_qs(self) -> str:
        """Encoded path of URL with query."""
        if q := self._query:
            return f"{self._path}?{q}" if self._path or not self._netloc else f"/?{q}"
        return self._path if self._path or not self._netloc else "/"

    @cached_property
    def raw_fragment(self) -> str:
        """Encoded fragment part of URL.

        Empty string if fragment is missing.

        """
        return self._fragment

    @cached_property
    def fragment(self) -> str:
        """Decoded fragment part of URL.

        Empty string if fragment is missing.

        """
        return UNQUOTER(self._fragment) if self._fragment else ""

    @cached_property
    def raw_parts(self) -> tuple[str, ...]:
        """A tuple containing encoded *path* parts.

        ('/',) for absolute URLs if *path* is missing.

        """
        path = self._path
        if self._netloc:
            return ("/", *path[1:].split("/")) if path else ("/",)
        if path and path[0] == "/":
            return ("/", *path[1:].split("/"))
        return tuple(path.split("/"))

    @cached_property
    def parts(self) -> tuple[str, ...]:
        """A tuple containing decoded *path* parts.

        ('/',) for absolute URLs if *path* is missing.

        """
        return tuple(UNQUOTER(part) for part in self.raw_parts)

    @cached_property
    def parent(self) -> "URL":
        """A new URL with last part of path removed and cleaned up query and
        fragment.

        """
        path = self._path
        if not path or path == "/":
            if self._fragment or self._query:
                return from_parts(self._scheme, self._netloc, path, "", "")
            return self
        parts = path.split("/")
        return from_parts(self._scheme, self._netloc, "/".join(parts[:-1]), "", "")

    @cached_property
    def raw_name(self) -> str:
        """The last part of raw_parts."""
        parts = self.raw_parts
        if not self._netloc:
            return parts[-1]
        parts = parts[1:]
        return parts[-1] if parts else ""

    @cached_property
    def name(self) -> str:
        """The last part of parts."""
        return UNQUOTER(self.raw_name)

    @cached_property
    def raw_suffix(self) -> str:
        name = self.raw_name
        i = name.rfind(".")
        return name[i:] if 0 < i < len(name) - 1 else ""

    @cached_property
    def suffix(self) -> str:
        return UNQUOTER(self.raw_suffix)

    @cached_property
    def raw_suffixes(self) -> tuple[str, ...]:
        name = self.raw_name
        if name.endswith("."):
            return ()
        name = name.lstrip(".")
        return tuple("." + suffix for suffix in name.split(".")[1:])

    @cached_property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(UNQUOTER(suffix) for suffix in self.raw_suffixes)

    def _make_child(self, paths: "Sequence[str]", encoded: bool = False) -> "URL":
        """
        add paths to self._path, accounting for absolute vs relative paths,
        keep existing, but do not create new, empty segments
        """
        parsed: list[str] = []
        needs_normalize: bool = False
        for idx, path in enumerate(reversed(paths)):
            # empty segment of last is not removed
            last = idx == 0
            if path and path[0] == "/":
                raise ValueError(
                    f"Appending path {path!r} starting from slash is forbidden"
                )
            # We need to quote the path if it is not already encoded
            # This cannot be done at the end because the existing
            # path is already quoted and we do not want to double quote
            # the existing path.
            path = path if encoded else PATH_QUOTER(path)
            needs_normalize |= "." in path
            segments = path.split("/")
            segments.reverse()
            # remove trailing empty segment for all but the last path
            parsed += segments[1:] if not last and segments[0] == "" else segments

        if (path := self._path) and (old_segments := path.split("/")):
            # If the old path ends with a slash, the last segment is an empty string
            # and should be removed before adding the new path segments.
            old = old_segments[:-1] if old_segments[-1] == "" else old_segments
            old.reverse()
            parsed += old

        # If the netloc is present, inject a leading slash when adding a
        # path to an absolute URL where there was none before.
        if (netloc := self._netloc) and parsed and parsed[-1] != "":
            parsed.append("")

        parsed.reverse()
        if not netloc or not needs_normalize:
            return from_parts(self._scheme, netloc, "/".join(parsed), "", "")

        path = "/".join(normalize_path_segments(parsed))
        # If normalizing the path segments removed the leading slash, add it back.
        if path and path[0] != "/":
            path = f"/{path}"
        return from_parts(self._scheme, netloc, path, "", "")

    def with_scheme(self, scheme: str) -> "URL":
        """Return a new URL with scheme replaced."""
        # N.B. doesn't cleanup query/fragment
        if not isinstance(scheme, str):
            raise TypeError("Invalid scheme type")
        lower_scheme = scheme.lower()
        netloc = self._netloc
        if not netloc and lower_scheme in SCHEME_REQUIRES_HOST:
            msg = (
                "scheme replacement is not allowed for "
                f"relative URLs for the {lower_scheme} scheme"
            )
            raise ValueError(msg)
        return from_parts(lower_scheme, netloc, self._path, self._query, self._fragment)

    def with_user(self, user: str | None) -> "URL":
        """Return a new URL with user replaced.

        Autoencode user if needed.

        Clear user/password if user is None.

        """
        # N.B. doesn't cleanup query/fragment
        if user is None:
            password = None
        elif isinstance(user, str):
            user = QUOTER(user)
            password = self.raw_password
        else:
            raise TypeError("Invalid user type")
        if not (netloc := self._netloc):
            raise ValueError("user replacement is not allowed for relative URLs")
        encoded_host = self.host_subcomponent or ""
        netloc = make_netloc(user, password, encoded_host, self.explicit_port)
        return from_parts(self._scheme, netloc, self._path, self._query, self._fragment)

    def with_password(self, password: str | None) -> "URL":
        """Return a new URL with password replaced.

        Autoencode password if needed.

        Clear password if argument is None.

        """
        # N.B. doesn't cleanup query/fragment
        if password is None:
            pass
        elif isinstance(password, str):
            password = QUOTER(password)
        else:
            raise TypeError("Invalid password type")
        if not (netloc := self._netloc):
            raise ValueError("password replacement is not allowed for relative URLs")
        encoded_host = self.host_subcomponent or ""
        port = self.explicit_port
        netloc = make_netloc(self.raw_user, password, encoded_host, port)
        return from_parts(self._scheme, netloc, self._path, self._query, self._fragment)

    def with_host(self, host: str) -> "URL":
        """Return a new URL with host replaced.

        Autoencode host if needed.

        Changing host for relative URLs is not allowed, use .join()
        instead.

        """
        # N.B. doesn't cleanup query/fragment
        if not isinstance(host, str):
            raise TypeError("Invalid host type")
        if not (netloc := self._netloc):
            raise ValueError("host replacement is not allowed for relative URLs")
        if not host:
            raise ValueError("host removing is not allowed")
        encoded_host = _encode_host(host, validate_host=True) if host else ""
        port = self.explicit_port
        netloc = make_netloc(self.raw_user, self.raw_password, encoded_host, port)
        return from_parts(self._scheme, netloc, self._path, self._query, self._fragment)

    def with_port(self, port: int | None) -> "URL":
        """Return a new URL with port replaced.

        Clear port to default if None is passed.

        """
        # N.B. doesn't cleanup query/fragment
        if port is not None:
            if isinstance(port, bool) or not isinstance(port, int):
                raise TypeError(f"port should be int or None, got {type(port)}")
            if not (0 <= port <= 65535):
                raise ValueError(f"port must be between 0 and 65535, got {port}")
        if not (netloc := self._netloc):
            raise ValueError("port replacement is not allowed for relative URLs")
        encoded_host = self.host_subcomponent or ""
        netloc = make_netloc(self.raw_user, self.raw_password, encoded_host, port)
        return from_parts(self._scheme, netloc, self._path, self._query, self._fragment)

    def with_path(
        self,
        path: str,
        *,
        encoded: bool = False,
        keep_query: bool = False,
        keep_fragment: bool = False,
    ) -> "URL":
        """Return a new URL with path replaced."""
        netloc = self._netloc
        if not encoded:
            path = PATH_QUOTER(path)
            if netloc:
                path = normalize_path(path) if "." in path else path
        if path and path[0] != "/":
            path = f"/{path}"
        query = self._query if keep_query else ""
        fragment = self._fragment if keep_fragment else ""
        return from_parts(self._scheme, netloc, path, query, fragment)

    @overload
    def with_query(self, query: Query) -> "URL": ...

    @overload
    def with_query(self, **kwargs: QueryVariable) -> "URL": ...

    def with_query(self, *args: Any, **kwargs: Any) -> "URL":
        """Return a new URL with query part replaced.

        Accepts any Mapping (e.g. dict, multidict.MultiDict instances)
        or str, autoencode the argument if needed.

        A sequence of (key, value) pairs is supported as well.

        It also can take an arbitrary number of keyword arguments.

        Clear query if None is passed.

        """
        # N.B. doesn't cleanup query/fragment
        query = get_str_query(*args, **kwargs) or ""
        return from_parts_uncached(
            self._scheme, self._netloc, self._path, query, self._fragment
        )

    @overload
    def extend_query(self, query: Query) -> "URL": ...

    @overload
    def extend_query(self, **kwargs: QueryVariable) -> "URL": ...

    def extend_query(self, *args: Any, **kwargs: Any) -> "URL":
        """Return a new URL with query part combined with the existing.

        This method will not remove existing query parameters.

        Example:
        >>> url = URL('http://example.com/?a=1&b=2')
        >>> url.extend_query(a=3, c=4)
        URL('http://example.com/?a=1&b=2&a=3&c=4')
        """
        if not (new_query := get_str_query(*args, **kwargs)):
            return self
        if query := self._query:
            # both strings are already encoded so we can use a simple
            # string join
            query += new_query if query[-1] == "&" else f"&{new_query}"
        else:
            query = new_query
        return from_parts_uncached(
            self._scheme, self._netloc, self._path, query, self._fragment
        )

    @overload
    def update_query(self, query: Query) -> "URL": ...

    @overload
    def update_query(self, **kwargs: QueryVariable) -> "URL": ...

    def update_query(self, *args: Any, **kwargs: Any) -> "URL":
        """Return a new URL with query part updated.

        This method will overwrite existing query parameters.

        Example:
        >>> url = URL('http://example.com/?a=1&b=2')
        >>> url.update_query(a=3, c=4)
        URL('http://example.com/?a=3&b=2&c=4')
        """
        in_query: (
            str
            | Mapping[str, QueryVariable]
            | Sequence[tuple[str | istr, SimpleQuery]]
            | None
        )
        if kwargs:
            if args:
                msg = "Either kwargs or single query parameter must be present"
                raise ValueError(msg)
            in_query = kwargs
        elif len(args) == 1:
            in_query = args[0]
        else:
            raise ValueError("Either kwargs or single query parameter must be present")

        if in_query is None:
            query = ""
        elif not in_query:
            query = self._query
        elif isinstance(in_query, Mapping):
            qm: MultiDict[QueryVariable] = MultiDict(self._parsed_query)
            qm.update(in_query)
            query = get_str_query_from_sequence_iterable(qm.items())
        elif isinstance(in_query, str):
            qstr: MultiDict[str] = MultiDict(self._parsed_query)
            qstr.update(query_to_pairs(in_query))
            query = get_str_query_from_iterable(qstr.items())
        elif isinstance(in_query, (bytes, bytearray, memoryview)):
            msg = "Invalid query type: bytes, bytearray and memoryview are forbidden"
            raise TypeError(msg)
        elif isinstance(in_query, Sequence):
            # We don't expect sequence values if we're given a list of pairs
            # already; only mappings like builtin `dict` which can't have the
            # same key pointing to multiple values are allowed to use
            # `_query_seq_pairs`.
            if TYPE_CHECKING:
                in_query = cast(
                    Sequence[tuple[Union[str, istr], SimpleQuery]], in_query
                )
            qs: MultiDict[SimpleQuery] = MultiDict(self._parsed_query)
            qs.update(in_query)
            query = get_str_query_from_iterable(qs.items())
        else:
            raise TypeError(
                "Invalid query type: only str, mapping or "
                "sequence of (key, value) pairs is allowed"
            )
        return from_parts_uncached(
            self._scheme, self._netloc, self._path, query, self._fragment
        )

    def without_query_params(self, *query_params: str) -> "URL":
        """Remove some keys from query part and return new URL."""
        params_to_remove = set(query_params) & self.query.keys()
        if not params_to_remove:
            return self
        return self.with_query(
            tuple(
                (name, value)
                for name, value in self.query.items()
                if name not in params_to_remove
            )
        )

    def with_fragment(self, fragment: str | None) -> "URL":
        """Return a new URL with fragment replaced.

        Autoencode fragment if needed.

        Clear fragment to default if None is passed.

        """
        # N.B. doesn't cleanup query/fragment
        if fragment is None:
            raw_fragment = ""
        elif not isinstance(fragment, str):
            raise TypeError("Invalid fragment type")
        else:
            raw_fragment = FRAGMENT_QUOTER(fragment)
        if self._fragment == raw_fragment:
            return self
        return from_parts(
            self._scheme, self._netloc, self._path, self._query, raw_fragment
        )

    def with_name(
        self,
        name: str,
        *,
        keep_query: bool = False,
        keep_fragment: bool = False,
    ) -> "URL":
        """Return a new URL with name (last part of path) replaced.

        Query and fragment parts are cleaned up.

        Name is encoded if needed.

        """
        # N.B. DOES cleanup query/fragment
        if not isinstance(name, str):
            raise TypeError("Invalid name type")
        if "/" in name:
            raise ValueError("Slash in name is not allowed")
        name = PATH_QUOTER(name)
        if name in (".", ".."):
            raise ValueError(". and .. values are forbidden")
        parts = list(self.raw_parts)
        if netloc := self._netloc:
            if len(parts) == 1:
                parts.append(name)
            else:
                parts[-1] = name
            parts[0] = ""  # replace leading '/'
        else:
            parts[-1] = name
            if parts[0] == "/":
                parts[0] = ""  # replace leading '/'

        query = self._query if keep_query else ""
        fragment = self._fragment if keep_fragment else ""
        return from_parts(self._scheme, netloc, "/".join(parts), query, fragment)

    def with_suffix(
        self,
        suffix: str,
        *,
        keep_query: bool = False,
        keep_fragment: bool = False,
    ) -> "URL":
        """Return a new URL with suffix (file extension of name) replaced.

        Query and fragment parts are cleaned up.

        suffix is encoded if needed.
        """
        if not isinstance(suffix, str):
            raise TypeError("Invalid suffix type")
        if suffix and not suffix[0] == "." or suffix == "." or "/" in suffix:
            raise ValueError(f"Invalid suffix {suffix!r}")
        name = self.raw_name
        if not name:
            raise ValueError(f"{self!r} has an empty name")
        old_suffix = self.raw_suffix
        suffix = PATH_QUOTER(suffix)
        name = name + suffix if not old_suffix else name[: -len(old_suffix)] + suffix
        if name in (".", ".."):
            raise ValueError(". and .. values are forbidden")
        parts = list(self.raw_parts)
        if netloc := self._netloc:
            if len(parts) == 1:
                parts.append(name)
            else:
                parts[-1] = name
            parts[0] = ""  # replace leading '/'
        else:
            parts[-1] = name
            if parts[0] == "/":
                parts[0] = ""  # replace leading '/'

        query = self._query if keep_query else ""
        fragment = self._fragment if keep_fragment else ""
        return from_parts(self._scheme, netloc, "/".join(parts), query, fragment)

    def join(self, url: "URL") -> "URL":
        """Join URLs

        Construct a full (“absolute”) URL by combining a “base URL”
        (self) with another URL (url).

        Informally, this uses components of the base URL, in
        particular the addressing scheme, the network location and
        (part of) the path, to provide missing components in the
        relative URL.

        """
        if type(url) is not URL:
            raise TypeError("url should be URL")

        scheme = url._scheme or self._scheme
        if scheme != self._scheme or scheme not in USES_RELATIVE:
            return url

        # scheme is in uses_authority as uses_authority is a superset of uses_relative
        if (join_netloc := url._netloc) and scheme in USES_AUTHORITY:
            return from_parts(scheme, join_netloc, url._path, url._query, url._fragment)

        orig_path = self._path
        if join_path := url._path:
            if join_path[0] == "/":
                path = join_path
            elif not orig_path:
                path = f"/{join_path}"
            elif orig_path[-1] == "/":
                path = f"{orig_path}{join_path}"
            else:
                # …
                # and relativizing ".."
                # parts[0] is / for absolute urls,
                # this join will add a double slash there
                path = "/".join([*self.parts[:-1], ""]) + join_path
                # which has to be removed
                if orig_path[0] == "/":
                    path = path[1:]
            path = normalize_path(path) if "." in path else path
        else:
            path = orig_path

        return from_parts(
            scheme,
            self._netloc,
            path,
            url._query if join_path or url._query else self._query,
            url._fragment if join_path or url._fragment else self._fragment,
        )

    def joinpath(self, *other: str, encoded: bool = False) -> "URL":
        """Return a new URL with the elements in other appended to the path."""
        return self._make_child(other, encoded=encoded)

    def human_repr(self) -> str:
        """Return decoded human readable string for URL representation."""
        user = human_quote(self.user, "#/:?@[]\\")
        password = human_quote(self.password, "#/:?@[]\\")
        if (host := self.host) and ":" in host:
            host = f"[{host}]"
        path = human_quote(self.path, "#?")
        if TYPE_CHECKING:
            assert path is not None
        if not self._scheme and not self._netloc:
            path = _encode_relative_scheme_colon(path)
        query_string = "&".join(
            "{}={}".format(human_quote(k, "#&+;="), human_quote(v, "#&+;="))
            for k, v in self.query.items()
        )
        fragment = human_quote(self.fragment, "")
        if TYPE_CHECKING:
            assert fragment is not None
        netloc = make_netloc(user, password, host, self.explicit_port)
        return unsplit_result(self._scheme, netloc, path, query_string, fragment)

    if HAS_PYDANTIC:
        # Borrowed from https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        @classmethod
        def __get_pydantic_json_schema__(
            cls,
            core_schema: "CoreSchema",
            handler: "GetJsonSchemaHandler",
        ) -> "JsonSchemaValue":
            field_schema: dict[str, Any] = {}
            field_schema.update(type="string", format="uri")
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: type[Self] | type[str],
            handler: "GetCoreSchemaHandler",
        ) -> "CoreSchema":
            # Lazy import: pulling in pydantic_core at module load time
            # increases yarl's import cost 3-7x for users who don't use
            # pydantic. Keep this import function-scoped.
            from pydantic_core import core_schema  # noqa: PLC0415

            from_str_schema = core_schema.chain_schema(
                [
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(URL),
                ]
            )

            return core_schema.json_or_python_schema(
                json_schema=from_str_schema,
                python_schema=core_schema.union_schema(
                    [
                        # check if it's an instance first before doing any further work
                        core_schema.is_instance_schema(URL),
                        from_str_schema,
                    ]
                ),
                serialization=core_schema.plain_serializer_function_ser_schema(str),
            )


_DEFAULT_IDNA_SIZE = 256
_DEFAULT_ENCODE_SIZE = 512


@lru_cache(_DEFAULT_IDNA_SIZE)
def _idna_decode(raw: str) -> str:
    try:
        return idna.decode(raw.encode("ascii"))
    except UnicodeError:  # e.g. '::1'
        return raw.encode("ascii").decode("idna")


@lru_cache(_DEFAULT_IDNA_SIZE)
def _idna_encode(host: str) -> str:
    try:
        return idna.encode(host, uts46=True).decode("ascii")
    except UnicodeError:
        return host.encode("idna").decode("ascii")


@lru_cache(_DEFAULT_ENCODE_SIZE)
def _encode_host(host: str, validate_host: bool) -> str:
    """Encode host part of URL."""
    # If the host ends with a digit or contains a colon, its likely
    # an IP address.
    if host and (host[-1].isdigit() or ":" in host):
        # RFC 6874 spells the IPv6 zone separator as the percent-encoded
        # ``%25``; bare ``%`` is still accepted so that hosts constructed
        # programmatically (e.g. ``with_host("fe80::1%1")``) keep working.
        part = "%25" if "%25" in host else "%"
        raw_ip, sep, zone = host.partition(part)
        # If it looks like an IP, we check with _ip_compressed_version
        # and fall-through if its not an IP address. This is a performance
        # optimization to avoid parsing IP addresses as much as possible
        # because it is orders of magnitude slower than almost any other
        # operation this library does.
        # Might be an IP address, check it
        #
        # IP Addresses can look like:
        # https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.2
        # - 127.0.0.1 (last character is a digit)
        # - 2001:db8::ff00:42:8329 (contains a colon)
        # - 2001:db8::ff00:42:8329%eth0 (contains a colon)
        # - [2001:db8::ff00:42:8329] (contains a colon -- brackets should
        #                             have been removed before it gets here)
        # Rare IP Address formats are not supported per:
        # https://datatracker.ietf.org/doc/html/rfc3986#section-7.4
        #
        # IP parsing is slow, so its wrapped in an LRU
        try:
            ip = ip_address(raw_ip)
        except ValueError:
            pass
        else:
            if sep and validate_host and (not zone or _ZONE_ID_UNSAFE_RE.search(zone)):
                raise ValueError("Invalid characters in zone identifier")
            # These checks should not happen in the
            # LRU to keep the cache size small
            host = ip.compressed
            if ip.version == 6:
                return f"[{host}{sep}{zone}]" if sep else f"[{host}]"
            return f"{host}{sep}{zone}" if sep else host

    # IDNA encoding is slow, skip it for ASCII-only strings
    if host.isascii():
        # Check for invalid characters explicitly; _idna_encode() does this
        # for non-ascii host names.
        host = host.lower()
        if validate_host and (invalid := NOT_REG_NAME.search(host)):
            value, pos, extra = invalid.group(), invalid.start(), ""
            if value == "@" or (value == ":" and "@" in host[pos:]):
                # this looks like an authority string
                extra = (
                    ", if the value includes a username or password, "
                    "use 'authority' instead of 'host'"
                )
            raise ValueError(
                f"Host {host!r} cannot contain {value!r} (at position {pos}){extra}"
            ) from None
        return host

    # IDNA/UTS-46 mapping silently deletes default-ignorable code points, which
    # would turn e.g. ``e<ZWSP>vil.com`` into ``evil.com``, a different host
    # than the string the caller supplied. Reject them on every path (this runs
    # regardless of ``validate_host`` since the plain ``URL(str)`` constructor
    # encodes with ``validate_host=False``) so the parsed host cannot diverge
    # from the input, matching idna/httpx/urllib3.
    if invalid := _DEFAULT_IGNORABLE_RE.search(host):
        raise ValueError(
            f"Host {host!r} cannot contain {invalid.group()!r} "
            f"(at position {invalid.start()})"
        ) from None
    encoded = _idna_encode(host)
    # IDNA uses NFKC equivalence, so normalization can expand a non-ascii
    # character into an ASCII delimiter (e.g. the fullwidth solidus U+FF0F
    # becomes '/'). The ascii branch above rejects such delimiters directly;
    # apply the same check to the IDNA output so the builder APIs agree with
    # the parser's _check_netloc.
    if validate_host and (invalid := NOT_REG_NAME.search(encoded)):
        raise ValueError(
            f"Host {host!r} cannot contain {invalid.group()!r} "
            f"after IDNA normalization to {encoded!r}"
        ) from None
    return encoded


@rewrite_module
def cache_clear() -> None:
    """Clear all LRU caches."""
    _idna_encode.cache_clear()
    _idna_decode.cache_clear()
    _encode_host.cache_clear()


@rewrite_module
def cache_info() -> CacheInfo:
    """Report cache statistics."""
    return {
        "idna_encode": _idna_encode.cache_info(),
        "idna_decode": _idna_decode.cache_info(),
        "ip_address": _encode_host.cache_info(),
        "host_validate": _encode_host.cache_info(),
        "encode_host": _encode_host.cache_info(),
    }


@rewrite_module
def cache_configure(
    *,
    idna_encode_size: int | None = _DEFAULT_IDNA_SIZE,
    idna_decode_size: int | None = _DEFAULT_IDNA_SIZE,
    ip_address_size: int | None | UndefinedType = UNDEFINED,
    host_validate_size: int | None | UndefinedType = UNDEFINED,
    encode_host_size: int | None | UndefinedType = UNDEFINED,
) -> None:
    """Configure LRU cache sizes."""
    global _idna_decode, _idna_encode, _encode_host
    # ip_address_size, host_validate_size are no longer
    # used, but are kept for backwards compatibility.
    if ip_address_size is not UNDEFINED or host_validate_size is not UNDEFINED:
        warnings.warn(
            "cache_configure() no longer accepts the "
            "ip_address_size or host_validate_size arguments, "
            "they are used to set the encode_host_size instead "
            "and will be removed in the future",
            DeprecationWarning,
            stacklevel=2,
        )

    if encode_host_size is not None:
        for size in (ip_address_size, host_validate_size):
            if size is None:
                encode_host_size = None
            elif encode_host_size is UNDEFINED:
                if size is not UNDEFINED:
                    encode_host_size = size
            elif size is not UNDEFINED:
                if TYPE_CHECKING:
                    assert isinstance(size, int)
                    assert isinstance(encode_host_size, int)
                encode_host_size = max(size, encode_host_size)
        if encode_host_size is UNDEFINED:
            encode_host_size = _DEFAULT_ENCODE_SIZE

    _encode_host = lru_cache(encode_host_size)(_encode_host.__wrapped__)
    _idna_decode = lru_cache(idna_decode_size)(_idna_decode.__wrapped__)
    _idna_encode = lru_cache(idna_encode_size)(_idna_encode.__wrapped__)
<|endoftext|>"""Quoting and unquoting utilities for URL parts."""

from urllib.parse import quote

from ._quoting import _Quoter, _Unquoter

QUOTER = _Quoter(requote=False)
REQUOTER = _Quoter()
PATH_QUOTER = _Quoter(safe="@:", protected="/+", requote=False)
PATH_REQUOTER = _Quoter(safe="@:", protected="/+")
QUERY_QUOTER = _Quoter(safe="?/:@", protected="=+&;", qs=True, requote=False)
QUERY_REQUOTER = _Quoter(safe="?/:@", protected="=+&;", qs=True)
QUERY_PART_QUOTER = _Quoter(safe="?/:@", qs=True, requote=False)
FRAGMENT_QUOTER = _Quoter(safe="?/:@", requote=False)
FRAGMENT_REQUOTER = _Quoter(safe="?/:@")

UNQUOTER = _Unquoter()
PATH_SAFE_UNQUOTER = _Unquoter(ignore="/%")
QS_UNQUOTER = _Unquoter(qs=True)
# to match urllib.parse.unquote_plus
UNQUOTER_PLUS = _Unquoter(plus=True, replace_invalid=True)


def human_quote(s: str | None, unsafe: str) -> str | None:
    if not s:
        return s
    for c in "%" + unsafe:
        if c in s:
            s = s.replace(c, f"%{ord(c):02X}")
    if s.isprintable():
        return s
    return "".join(c if c.isprintable() else quote(c) for c in s)
<|endoftext|>import os
import sys
from typing import TYPE_CHECKING

__all__ = ("_Quoter", "_Unquoter")


NO_EXTENSIONS = bool(os.environ.get("YARL_NO_EXTENSIONS"))  # type: bool
if sys.implementation.name != "cpython":
    NO_EXTENSIONS = True


if TYPE_CHECKING or NO_EXTENSIONS:
    from ._quoting_py import _Quoter, _Unquoter
else:
    try:
        from ._quoting_c import _Quoter, _Unquoter
    except ImportError:  # pragma: no cover
        from ._quoting_py import _Quoter, _Unquoter  # type: ignore[assignment]
<|endoftext|>import re
from string import ascii_letters, ascii_lowercase, digits, hexdigits
from typing import overload

BASCII_LOWERCASE = ascii_lowercase.encode("ascii")
SUB_DELIMS_WITHOUT_QS = "!$'()*,"
QS = "+&=;"
UNRESERVED = ascii_letters + digits + "-._~"
ALLOWED = UNRESERVED + SUB_DELIMS_WITHOUT_QS


_IS_HEX = re.compile(b"[A-Z0-9][A-Z0-9]")
# Every two hex digit escape body, in any case, to the byte it encodes
_PCT_BYTES = {a + b: int(a + b, 16) for a in hexdigits for b in hexdigits}
_ASCII_LIMIT = 0x80
_ASCII_CHARS = tuple(map(chr, range(_ASCII_LIMIT)))
_UTF8_CONT_MIN = 0x80  # continuation bytes are 10xxxxxx
_UTF8_CONT_MAX = 0xBF
_UTF8_CONT_PAYLOAD = 0x3F
_UTF8_CONT_BITS = 6
_UTF8_LEAD2_PAYLOAD = 0x1F
_UTF8_LEAD3_PAYLOAD = 0x0F
_UTF8_LEAD4_PAYLOAD = 0x07
# Strict UTF-8 as accepted by CPython's decoder, see table 3-7 of the Unicode
# standard: lead byte to the sequence length, the range of the second byte and
# the payload bits of the lead byte.
_UTF8_LEADS = {
    **{
        lead: (2, _UTF8_CONT_MIN, _UTF8_CONT_MAX, _UTF8_LEAD2_PAYLOAD)
        for lead in range(0xC2, 0xE0)  # 0xC0 and 0xC1 only start overlong forms
    },
    0xE0: (3, 0xA0, _UTF8_CONT_MAX, _UTF8_LEAD3_PAYLOAD),  # E0 80..9F is overlong
    **{
        lead: (3, _UTF8_CONT_MIN, _UTF8_CONT_MAX, _UTF8_LEAD3_PAYLOAD)
        for lead in (*range(0xE1, 0xED), 0xEE, 0xEF)
    },
    0xED: (3, _UTF8_CONT_MIN, 0x9F, _UTF8_LEAD3_PAYLOAD),  # ED A0..BF is a surrogate
    0xF0: (4, 0x90, _UTF8_CONT_MAX, _UTF8_LEAD4_PAYLOAD),  # F0 80..8F is overlong
    **{
        lead: (4, _UTF8_CONT_MIN, _UTF8_CONT_MAX, _UTF8_LEAD4_PAYLOAD)
        for lead in range(0xF1, 0xF4)
    },
    0xF4: (4, _UTF8_CONT_MIN, 0x8F, _UTF8_LEAD4_PAYLOAD),  # F4 90..BF is past U+10FFFF
}


class _Quoter:
    def __init__(
        self,
        *,
        safe: str = "",
        protected: str = "",
        qs: bool = False,
        requote: bool = True,
    ) -> None:
        if not safe.isascii() or not protected.isascii():
            raise ValueError("Only safe symbols with ORD < 128 are allowed")
        # A safe '%' would be left alone while requoting decodes '%XX', and a
        # safe ' ' would be left alone while qs turns it into '+'
        if requote and ("%" in safe or "%" in protected):
            raise ValueError(
                "safe and protected cannot contain '%' when requote is enabled"
            )
        if qs and (" " in safe or " " in protected):
            raise ValueError("safe and protected cannot contain ' ' when qs is enabled")
        self._safe = safe
        self._protected = protected
        self._qs = qs
        self._requote = requote

    @overload
    def __call__(self, val: str) -> str: ...
    @overload
    def __call__(self, val: None) -> None: ...
    def __call__(self, val: str | None) -> str | None:
        if val is None:
            return None
        if not isinstance(val, str):
            raise TypeError("Argument should be str")
        if not val:
            return ""
        bval = val.encode("utf8", errors="ignore")
        ret = bytearray()
        pct = bytearray()
        safe = self._safe
        safe += ALLOWED
        if not self._qs:
            safe += QS
        safe += self._protected
        bsafe = safe.encode("ascii")
        idx = 0
        while idx < len(bval):
            ch = bval[idx]
            idx += 1

            if pct:
                if ch in BASCII_LOWERCASE:
                    ch = ch - 32  # convert to uppercase
                pct.append(ch)
                if len(pct) == 3:  # pragma: no branch   # peephole optimizer
                    buf = pct[1:]
                    if not _IS_HEX.match(buf):
                        ret.extend(b"%25")
                        pct.clear()
                        idx -= 2
                        continue
                    try:
                        unquoted = chr(int(pct[1:].decode("ascii"), base=16))
                    except ValueError:
                        ret.extend(b"%25")
                        pct.clear()
                        idx -= 2
                        continue

                    if unquoted in self._protected:
                        ret.extend(pct)
                    elif unquoted in safe:
                        ret.append(ord(unquoted))
                    else:
                        ret.extend(pct)
                    pct.clear()

                # special case, if we have only one char after "%"
                elif len(pct) == 2 and idx == len(bval):
                    ret.extend(b"%25")
                    pct.clear()
                    idx -= 1

                continue

            elif ch == ord("%") and self._requote:
                pct.clear()
                pct.append(ch)

                # special case if "%" is last char
                if idx == len(bval):
                    ret.extend(b"%25")

                continue

            if self._qs and ch == ord(" "):
                ret.append(ord("+"))
                continue
            if ch in bsafe:
                ret.append(ch)
                continue

            ret.extend((f"%{ch:02X}").encode("ascii"))

        ret2 = ret.decode("ascii")
        if ret2 == val:
            return val
        return ret2


class _Unquoter:
    def __init__(
        self,
        *,
        ignore: str = "",
        qs: bool = False,
        plus: bool = False,
        replace_invalid: bool = False,
    ) -> None:
        # Requoting leaves these characters as is, so they would be decoded
        # even when they are in ignore
        decoded_anyway = ALLOWED if qs else ALLOWED + QS
        for ch in ignore:
            if not ch.isascii():
                raise ValueError(f"ignore cannot contain {ch!r}, it is not ASCII")
            if ch in decoded_anyway:
                raise ValueError(f"ignore cannot contain {ch!r}, it is decoded anyway")
        # Write U+FFFD for escapes that are not valid UTF-8, like urllib does,
        # instead of keeping them as is
        self._invalid = "\ufffd" if replace_invalid else ""
        # '+' means a space in query strings and in urllib.parse.unquote_plus
        self._plus_is_space = qs or plus
        # What to write for each decoded ASCII character, its escape when it
        # is ignored or a query string delimiter
        self._ascii_output = list(_ASCII_CHARS)
        for ch in ignore + (QS if qs else ""):
            self._ascii_output[ord(ch)] = f"%{ord(ch):02X}"

    @overload
    def __call__(self, val: str) -> str: ...
    @overload
    def __call__(self, val: None) -> None: ...
    def __call__(self, val: str | None) -> str | None:
        if val is None:
            return None
        if not isinstance(val, str):
            raise TypeError("Argument should be str")
        if not val:
            return ""
        if self._plus_is_space and "+" in val:
            val = val.replace("+", " ")
        if (pos := val.find("%")) == -1:
            return val
        ascii_output = self._ascii_output
        invalid = self._invalid
        ret = []
        # An incomplete UTF-8 sequence: the number of bytes seen, where its
        # escapes start in val and the code point decoded so far
        pending = pending_start = code_point = need = low = high = 0
        # idx is the end of the part of val already handled; plain runs
        # between '%' characters are appended as a single slice.
        idx = 0
        while pos != -1:
            byte = _PCT_BYTES.get(val[pos + 1 : pos + 3])
            if pending and (pos > idx or byte is None or not low <= byte <= high):
                # Not a valid sequence, write the pending escapes as invalid
                ret.append(invalid or val[pending_start:idx])
                pending = 0
            if pos > idx:
                ret.append(val[idx:pos])
            if byte is None:
                # A '%' that does not start an escape is kept as is, as part
                # of the next plain run
                idx = pos
                pos = val.find("%", pos + 1)
                continue
            idx = pos + 3
            if pending:
                code_point = code_point << _UTF8_CONT_BITS | byte & _UTF8_CONT_PAYLOAD
                pending += 1
                low, high = _UTF8_CONT_MIN, _UTF8_CONT_MAX
                if pending == need:
                    ret.append(chr(code_point))
                    pending = 0
            elif byte < _ASCII_LIMIT:
                ret.append(ascii_output[byte])
            elif (lead := _UTF8_LEADS.get(byte)) is not None:
                need, low, high, payload = lead
                code_point = byte & payload
                pending = 1
                pending_start = pos
            else:
                ret.append(invalid or val[pos:idx])
            pos = val.find("%", idx)

        if pending:
            ret.append(invalid or val[pending_start:idx])
        ret.append(val[idx:])

        ret2 = "".join(ret)
        if ret2 == val:
            return val
        return ret2
<|endoftext|>"""
Digest authentication middleware for aiohttp client.

This middleware implements HTTP Digest Authentication according to RFC 7616,
providing a more secure alternative to Basic Authentication. It supports all
standard hash algorithms including MD5, SHA, SHA-256, SHA-512 and their session
variants, as well as both 'auth' and 'auth-int' quality of protection (qop) options.
"""

import hashlib
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Final, Literal, TypedDict

from yarl import URL

from . import hdrs
from .client_exceptions import ClientError
from .client_middlewares import ClientHandlerType
from .client_reqrep import ClientRequest, ClientResponse
from .payload import Payload


class DigestAuthChallenge(TypedDict, total=False):
    realm: str
    nonce: str
    qop: str
    algorithm: str
    opaque: str
    domain: str
    stale: str


DigestFunctions: dict[str, Callable[[bytes], "hashlib._Hash"]] = {
    "MD5": hashlib.md5,
    "MD5-SESS": hashlib.md5,
    "SHA": hashlib.sha1,
    "SHA-SESS": hashlib.sha1,
    "SHA256": hashlib.sha256,
    "SHA256-SESS": hashlib.sha256,
    "SHA-256": hashlib.sha256,
    "SHA-256-SESS": hashlib.sha256,
    "SHA512": hashlib.sha512,
    "SHA512-SESS": hashlib.sha512,
    "SHA-512": hashlib.sha512,
    "SHA-512-SESS": hashlib.sha512,
}


# Compile the regex pattern once at module level for performance
_HEADER_PAIRS_PATTERN = re.compile(
    r'(?:^|\s|,\s*)(\w+)(?:\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]+)))?'
    if sys.version_info < (3, 11)
    else r'(?:^|\s|,\s*)((?>\w+))(?:\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]+)))?'
    # +------------|--------|--|--||--|--|----|------|---|---||-----|-> Match valid start/sep
    #              +--------|--|--||--|--|----|------|---|---||-----|-> alphanumeric key (atomic group reduces backtracking)
    #                       +--|--||--|--|----|------|---|---||-----|-> optional value; absent => bare auth-scheme token
    #                          +--||--|--|----|------|---|---||-----|-> maybe whitespace
    #                             +|--|--|----|------|---|---||-----|-> = (delimiter)
    #                              +--|--|----|------|---|---||-----|-> maybe whitespace
    #                                 +--|----|------|---|---||-----|-> group quoted or unquoted
    #                                    +----|------|---|---||-----|-> if quoted...
    #                                         +------|---|---||-----|-> anything but " or \
    #                                                +---|---||-----|-> escaped characters allowed
    #                                                    +---||-----|-> or can be empty string
    #                                                        +|-----|-> if unquoted...
    #                                                         +-----|-> anything but , or <space>
    #                                                               +-> at least one char req'd
)


# RFC 7616: Challenge parameters to extract
CHALLENGE_FIELDS: Final[
    tuple[
        Literal["realm", "nonce", "qop", "algorithm", "opaque", "domain", "stale"], ...
    ]
] = (
    "realm",
    "nonce",
    "qop",
    "algorithm",
    "opaque",
    "domain",
    "stale",
)

# Supported digest authentication algorithms
# Use a tuple of sorted keys for predictable documentation and error messages
SUPPORTED_ALGORITHMS: Final[tuple[str, ...]] = tuple(sorted(DigestFunctions.keys()))

# RFC 7616: Fields that require quoting in the Digest auth header
# These fields must be enclosed in double quotes in the Authorization header.
# Algorithm, qop, and nc are never quoted per RFC specifications.
# This frozen set is used by the template-based header construction to
# automatically determine which fields need quotes.
QUOTED_AUTH_FIELDS: Final[frozenset[str]] = frozenset(
    {"username", "realm", "nonce", "uri", "response", "opaque", "cnonce"}
)


def escape_quotes(value: str) -> str:
    """Escape backslashes and double quotes for HTTP quoted-strings."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def unescape_quotes(value: str) -> str:
    """Unescape backslashes and double quotes in HTTP quoted-strings."""
    return value.replace('\\"', '"').replace("\\\\", "\\")


def parse_header_pairs(header: str) -> dict[str, str]:
    """
    Parse key-value pairs from the first challenge of a WWW-Authenticate header.

    This function handles the complex format of WWW-Authenticate header values,
    supporting both quoted and unquoted values, proper handling of commas in
    quoted values, and whitespace variations per RFC 7616.

    A single header may carry several challenges
    (https://www.rfc-editor.org/rfc/rfc7235#section-4.1). Parsing
    stops at the next auth-scheme token so a later challenge's parameters cannot
    overwrite the first challenge's values; a leading scheme token is skipped.

    Examples of supported formats:
      - key1="value1", key2=value2
      - key1 = "value1" , key2="value, with, commas"
      - key1=value1,key2="value2"
      - realm="example.com", nonce="12345", qop="auth"

    Args:
        header: The header value string to parse

    Returns:
        Dictionary mapping parameter names to their values
    """
    pairs: dict[str, str] = {}
    for match in _HEADER_PAIRS_PATTERN.finditer(header):
        key = match.group(1)
        quoted_val, unquoted_val = match.group(2), match.group(3)
        if quoted_val is None and unquoted_val is None:
            # Bare token with no "=value": an auth-scheme name, not a parameter.
            # Skip a leading scheme; once parameters exist, a new scheme marks
            # the start of the next challenge, so stop here.
            if pairs:
                break
            continue
        pairs[key] = (
            unescape_quotes(quoted_val) if quoted_val is not None else unquoted_val
        )
    return pairs


class DigestAuthMiddleware:
    """
    HTTP digest authentication middleware for aiohttp client.

    This middleware intercepts 401 Unauthorized responses containing a Digest
    authentication challenge, calculates the appropriate digest credentials,
    and automatically retries the request with the proper Authorization header.

    Features:
    - Handles all aspects of Digest authentication handshake automatically
    - Supports all standard hash algorithms:
      - MD5, MD5-SESS
      - SHA, SHA-SESS
      - SHA256, SHA256-SESS, SHA-256, SHA-256-SESS
      - SHA512, SHA512-SESS, SHA-512, SHA-512-SESS
    - Supports 'auth' and 'auth-int' quality of protection modes
    - Properly handles quoted strings and parameter parsing
    - Includes replay attack protection with client nonce count tracking
    - Supports preemptive authentication per RFC 7616 Section 3.6

    Origin scoping:
    The credentials are scoped to the origin of the first request the
    middleware handles. A request to a different origin is passed through
    untouched, so it never receives a digest response computed from those
    credentials, unless that origin falls within a protection space the
    anchor origin advertised through the RFC 7616 ``domain`` directive. Make
    the first request through the middleware against the intended origin, as
    the anchor is pinned to it and not reset for the life of the instance.

    Standards compliance:
    - RFC 7616: HTTP Digest Access Authentication (primary reference)
    - RFC 2617: HTTP Authentication (deprecated by RFC 7616)
    - RFC 1945: Section 11.1 (username restrictions)

    Implementation notes:
    The core digest calculation is inspired by the implementation in
    https://github.com/requests/requests/blob/v2.18.4/requests/auth.py
    with added support for modern digest auth features and error handling.
    """

    def __init__(
        self,
        login: str,
        password: str,
        preemptive: bool = True,
    ) -> None:
        if login is None:
            raise ValueError("None is not allowed as login value")

        if password is None:
            raise ValueError("None is not allowed as password value")

        if ":" in login:
            raise ValueError('A ":" is not allowed in username (RFC 1945#section-11.1)')

        self._login_str: Final[str] = login
        self._login_bytes: Final[bytes] = login.encode("utf-8")
        self._password_bytes: Final[bytes] = password.encode("utf-8")

        self._last_nonce_bytes = b""
        self._nonce_count = 0
        self._challenge: DigestAuthChallenge = {}
        self._preemptive: bool = preemptive
        # Set of URLs defining the protection space
        self._protection_space: list[str] = []
        # Origin the credentials are scoped to; set on the first request.
        self._origin: URL | None = None

    async def _encode(self, method: str, url: URL, body: Payload | Literal[b""]) -> str:
        """
        Build digest authorization header for the current challenge.

        Args:
            method: The HTTP method (GET, POST, etc.)
            url: The request URL
            body: The request body (used for qop=auth-int)

        Returns:
            A fully formatted Digest authorization header string

        Raises:
            ClientError: If the challenge is missing required parameters or
                         contains unsupported values

        """
        challenge = self._challenge
        if "realm" not in challenge:
            raise ClientError(
                "Malformed Digest auth challenge: Missing 'realm' parameter"
            )

        if "nonce" not in challenge:
            raise ClientError(
                "Malformed Digest auth challenge: Missing 'nonce' parameter"
            )

        # Empty realm values are allowed per RFC 7616 (SHOULD, not MUST, contain host name)
        realm = challenge["realm"]
        nonce = challenge["nonce"]

        # Empty nonce values are not allowed as they are security-critical for replay protection
        if not nonce:
            raise ClientError(
                "Security issue: Digest auth challenge contains empty 'nonce' value"
            )

        qop_raw = challenge.get("qop", "")
        # Preserve original algorithm case for response while using uppercase for processing
        algorithm_original = challenge.get("algorithm", "MD5")
        algorithm = algorithm_original.upper()
        opaque = challenge.get("opaque", "")

        # Convert string values to bytes once
        nonce_bytes = nonce.encode("utf-8")
        realm_bytes = realm.encode("utf-8")
        # Use the encoded request-target (raw_path_qs) since that is what is
        # transmitted on the wire and what the server signs against. Using the
        # decoded form would cause digest verification to fail when the path
        # or query string contains percent-encoded reserved characters.
        path = URL(url).raw_path_qs

        # Process QoP
        qop = ""
        qop_bytes = b""
        if qop_raw:
            valid_qops = {"auth", "auth-int"}.intersection(
                {q.strip() for q in qop_raw.split(",") if q.strip()}
            )
            if not valid_qops:
                raise ClientError(
                    f"Digest auth error: Unsupported Quality of Protection (qop) value(s): {qop_raw}"
                )

            qop = "auth-int" if "auth-int" in valid_qops else "auth"
            qop_bytes = qop.encode("utf-8")

        if algorithm not in DigestFunctions:
            raise ClientError(
                f"Digest auth error: Unsupported hash algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS)}"
            )
        hash_fn: Final = DigestFunctions[algorithm]

        def H(x: bytes) -> bytes:
            """RFC 7616 Section 3: Hash function H(data) = hex(hash(data))."""
            return hash_fn(x).hexdigest().encode()

        def KD(s: bytes, d: bytes) -> bytes:
            """RFC 7616 Section 3: KD(secret, data) = H(concat(secret, ":", data))."""
            return H(b":".join((s, d)))

        # Calculate A1 and A2
        A1 = b":".join((self._login_bytes, realm_bytes, self._password_bytes))
        A2 = f"{method.upper()}:{path}".encode()
        if qop == "auth-int":
            if isinstance(body, Payload):  # will always be empty bytes unless Payload
                entity_bytes = await body.as_bytes()  # Get bytes from Payload
            else:
                entity_bytes = body
            entity_hash = H(entity_bytes)
            A2 = b":".join((A2, entity_hash))

        HA1 = H(A1)
        HA2 = H(A2)

        # Nonce count handling
        if nonce_bytes == self._last_nonce_bytes:
            self._nonce_count += 1
        else:
            self._nonce_count = 1

        self._last_nonce_bytes = nonce_bytes
        ncvalue = f"{self._nonce_count:08x}"
        ncvalue_bytes = ncvalue.encode("utf-8")

        # Generate client nonce
        cnonce = hashlib.sha1(
            b"".join(
                [
                    str(self._nonce_count).encode("utf-8"),
                    nonce_bytes,
                    time.ctime().encode("utf-8"),
                    os.urandom(8),
                ]
            )
        ).hexdigest()[:16]
        cnonce_bytes = cnonce.encode("utf-8")

        # Special handling for session-based algorithms
        if algorithm.upper().endswith("-SESS"):
            HA1 = H(b":".join((HA1, nonce_bytes, cnonce_bytes)))

        # Calculate the response digest
        if qop:
            noncebit = b":".join(
                (nonce_bytes, ncvalue_bytes, cnonce_bytes, qop_bytes, HA2)
            )
            response_digest = KD(HA1, noncebit)
        else:
            response_digest = KD(HA1, b":".join((nonce_bytes, HA2)))

        # Define a dict mapping of header fields to their values
        # Group fields into always-present, optional, and qop-dependent
        header_fields = {
            # Always present fields
            "username": escape_quotes(self._login_str),
            "realm": escape_quotes(realm),
            "nonce": escape_quotes(nonce),
            "uri": path,
            "response": response_digest.decode(),
            "algorithm": algorithm_original,
        }

        # Optional fields
        if opaque:
            header_fields["opaque"] = escape_quotes(opaque)

        # QoP-dependent fields
        if qop:
            header_fields["qop"] = qop
            header_fields["nc"] = ncvalue
            header_fields["cnonce"] = cnonce

        # Build header using templates for each field type
        pairs: list[str] = []
        for field, value in header_fields.items():
            if field in QUOTED_AUTH_FIELDS:
                pairs.append(f'{field}="{value}"')
            else:
                pairs.append(f"{field}={value}")

        return f"Digest {', '.join(pairs)}"

    def _in_protection_space(self, url: URL) -> bool:
        """
        Check if the given URL is within the current protection space.

        According to RFC 7616, a URI is in the protection space if any URI
        in the protection space is a prefix of it (after both have been made absolute).
        """
        request_str = str(url)
        for space_str in self._protection_space:
            # Check if request starts with space URL
            if not request_str.startswith(space_str):
                continue
            # Exact match or space ends with / (proper directory prefix)
            if len(request_str) == len(space_str) or space_str[-1] == "/":
                return True
            # Check next char is / to ensure proper path boundary
            if request_str[len(space_str)] == "/":
                return True
        return False

    def _authenticate(self, response: ClientResponse) -> bool:
        """
        Takes the given response and tries digest-auth, if needed.

        Returns true if the original request must be resent.
        """
        if response.status != 401:
            return False

        auth_header = response.headers.get("www-authenticate", "")
        if not auth_header:
            return False  # No authentication header present

        method, sep, headers = auth_header.partition(" ")
        if not sep:
            # No space found in www-authenticate header
            return False  # Malformed auth header, missing scheme separator

        if method.lower() != "digest":
            # Not a digest auth challenge (could be Basic, Bearer, etc.)
            return False

        if not headers:
            # We have a digest scheme but no parameters
            return False  # Malformed digest header, missing parameters

        # We have a digest auth header with content
        if not (header_pairs := parse_header_pairs(headers)):
            # Failed to parse any key-value pairs
            return False  # Malformed digest header, no valid parameters

        # Extract challenge parameters
        self._challenge = {}
        for field in CHALLENGE_FIELDS:
            if (value := header_pairs.get(field)) is not None:
                self._challenge[field] = value

        # Update protection space based on domain parameter or default to origin
        origin = response.url.origin()
        self._protection_space = []

        if domain := self._challenge.get("domain"):
            # Parse space-separated list of URIs
            for uri in domain.split():
                # Remove quotes if present
                uri = uri.strip('"')
                if not uri:
                    continue
                if uri.startswith("/"):
                    # Path-absolute, relative to origin
                    self._protection_space.append(str(origin.join(URL(uri))))
                else:
                    # Absolute URI
                    self._protection_space.append(str(URL(uri)))

        if not self._protection_space:
            self._protection_space = [str(origin)]

        # Return True only if we found at least one challenge parameter
        return bool(self._challenge)

    async def __call__(
        self, request: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        """Run the digest auth middleware."""
        # Credentials are scoped to the first request's origin. Other origins
        # pass through untouched unless a challenge from the anchor origin
        # advertised them via RFC 7616 domain; mirrors aiohttp stripping
        # Authorization on cross-origin redirects.
        origin = request.url.origin()
        if self._origin is None:
            self._origin = origin
        elif origin != self._origin and not self._in_protection_space(request.url):
            return await handler(request)

        response = None
        for retry_count in range(2):
            # Apply authorization header if:
            # 1. This is a retry after 401 (retry_count > 0), OR
            # 2. Preemptive auth is enabled AND we have a challenge AND the URL is in protection space
            if retry_count > 0 or (
                self._preemptive
                and self._challenge
                and self._in_protection_space(request.url)
            ):
                request.headers[hdrs.AUTHORIZATION] = await self._encode(
                    request.method, request.url, request.body
                )

            # Send the request
            response = await handler(request)

            # Check if we need to authenticate
            if not self._authenticate(response):
                break

        # At this point, response is guaranteed to be defined
        assert response is not None
        return response
<|endoftext|>import datetime
import functools
import logging
import os
import re
import time as time_mod
from collections.abc import Iterable
from typing import Callable, ClassVar, NamedTuple

from .abc import AbstractAccessLogger
from .web_request import BaseRequest
from .web_response import StreamResponse


class KeyMethod(NamedTuple):
    key: str | tuple[str, str]
    method: Callable[[BaseRequest, StreamResponse, float], str]


class AccessLogger(AbstractAccessLogger):
    """Helper object to log access.

    Usage:
        log = logging.getLogger("spam")
        log_format = "%a %{User-Agent}i"
        access_logger = AccessLogger(log, log_format)
        access_logger.log(request, response, time)

    Format:
        %%  The percent sign
        %a  Remote IP-address (IP-address of proxy if using reverse proxy)
        %t  Time when the request was started to process
        %P  The process ID of the child that serviced the request
        %r  First line of request
        %s  Response status code
        %b  Size of response in bytes, including HTTP headers
        %T  Time taken to serve the request, in seconds
        %Tf Time taken to serve the request, in seconds with floating fraction
            in .06f format
        %D  Time taken to serve the request, in microseconds
        %{FOO}i  request.headers['FOO']
        %{FOO}o  response.headers['FOO']
        %{FOO}e  os.environ['FOO']

    """

    LOG_FORMAT_MAP = {
        "a": "remote_address",
        "t": "request_start_time",
        "P": "process_id",
        "r": "first_request_line",
        "s": "response_status",
        "b": "response_size",
        "T": "request_time",
        "Tf": "request_time_frac",
        "D": "request_time_micro",
        "i": "request_header",
        "o": "response_header",
    }

    LOG_FORMAT = '%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    FORMAT_RE = re.compile(r"%(\{([A-Za-z0-9\-_]+)\}([ioe])|[atPrsbOD]|Tf?)")
    CLEANUP_RE = re.compile(r"(%[^s])")
    _FORMAT_CACHE: dict[str, tuple[str, list[KeyMethod]]] = {}

    _cached_tz: ClassVar[datetime.timezone | None] = None
    _cached_tz_expires: ClassVar[float] = 0.0

    def __init__(self, logger: logging.Logger, log_format: str = LOG_FORMAT) -> None:
        """Initialise the logger.

        logger is a logger object to be used for logging.
        log_format is a string with apache compatible log format description.

        """
        super().__init__(logger, log_format=log_format)

        _compiled_format = AccessLogger._FORMAT_CACHE.get(log_format)
        if not _compiled_format:
            _compiled_format = self.compile_format(log_format)
            AccessLogger._FORMAT_CACHE[log_format] = _compiled_format

        self._log_format, self._methods = _compiled_format

    def compile_format(self, log_format: str) -> tuple[str, list[KeyMethod]]:
        """Translate log_format into form usable by modulo formatting

        All known atoms will be replaced with %s
        Also methods for formatting of those atoms will be added to
        _methods in appropriate order

        For example we have log_format = "%a %t"
        This format will be translated to "%s %s"
        Also contents of _methods will be
        [self._format_a, self._format_t]
        These method will be called and results will be passed
        to translated string format.

        Each _format_* method receive 'args' which is list of arguments
        given to self.log

        Exceptions are _format_e, _format_i and _format_o methods which
        also receive key name (by functools.partial)

        """
        # list of (key, method) tuples, we don't use an OrderedDict as users
        # can repeat the same key more than once
        methods = list()

        for atom in self.FORMAT_RE.findall(log_format):
            if atom[1] == "":
                format_key1 = self.LOG_FORMAT_MAP[atom[0]]
                m = getattr(AccessLogger, "_format_%s" % atom[0])
                key_method = KeyMethod(format_key1, m)
            else:
                format_key2 = (self.LOG_FORMAT_MAP[atom[2]], atom[1])
                m = getattr(AccessLogger, "_format_%s" % atom[2])
                key_method = KeyMethod(format_key2, functools.partial(m, atom[1]))

            methods.append(key_method)

        log_format = self.FORMAT_RE.sub(r"%s", log_format)
        log_format = self.CLEANUP_RE.sub(r"%\1", log_format)
        return log_format, methods

    @staticmethod
    def _format_i(
        key: str, request: BaseRequest, response: StreamResponse, time: float
    ) -> str:
        if request is None:
            return "(no headers)"

        # suboptimal, make istr(key) once
        return request.headers.get(key, "-")

    @staticmethod
    def _format_o(
        key: str, request: BaseRequest, response: StreamResponse, time: float
    ) -> str:
        # suboptimal, make istr(key) once
        return response.headers.get(key, "-")

    @staticmethod
    def _format_a(request: BaseRequest, response: StreamResponse, time: float) -> str:
        if request is None:
            return "-"
        ip = request.remote
        return ip if ip is not None else "-"

    @classmethod
    def _get_local_time(cls) -> datetime.datetime:
        if cls._cached_tz is None or time_mod.time() >= cls._cached_tz_expires:
            gmtoff = time_mod.localtime().tm_gmtoff
            cls._cached_tz = tz = datetime.timezone(datetime.timedelta(seconds=gmtoff))

            now = datetime.datetime.now(tz)
            # Expire at every 30 mins, as any DST change should occur at 0/30 mins past.
            d = now + datetime.timedelta(minutes=30)
            d = d.replace(minute=30 if d.minute >= 30 else 0, second=0, microsecond=0)
            cls._cached_tz_expires = d.timestamp()
            return now

        return datetime.datetime.now(cls._cached_tz)

    @staticmethod
    def _format_t(request: BaseRequest, response: StreamResponse, time: float) -> str:
        now = AccessLogger._get_local_time()
        start_time = now - datetime.timedelta(seconds=time)
        return start_time.strftime("[%d/%b/%Y:%H:%M:%S %z]")

    @staticmethod
    def _format_P(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return "<%s>" % os.getpid()

    @staticmethod
    def _format_r(request: BaseRequest, response: StreamResponse, time: float) -> str:
        if request is None:
            return "-"
        return f"{request.method} {request.path_qs} HTTP/{request.version.major}.{request.version.minor}"

    @staticmethod
    def _format_s(request: BaseRequest, response: StreamResponse, time: float) -> int:
        return response.status

    @staticmethod
    def _format_b(request: BaseRequest, response: StreamResponse, time: float) -> int:
        return response.body_length

    @staticmethod
    def _format_T(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return str(round(time))

    @staticmethod
    def _format_Tf(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return "%06f" % time

    @staticmethod
    def _format_D(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return str(round(time * 1000000))

    def _format_line(
        self, request: BaseRequest, response: StreamResponse, time: float
    ) -> Iterable[tuple[str | tuple[str, str], str]]:
        return [(key, method(request, response, time)) for key, method in self._methods]

    @property
    def enabled(self) -> bool:
        """Check if logger is enabled."""
        # Avoid formatting the log line if it will not be emitted.
        return self.logger.isEnabledFor(logging.INFO)

    def log(self, request: BaseRequest, response: StreamResponse, time: float) -> None:
        try:
            fmt_info = self._format_line(request, response, time)

            values = list()
            extra: dict[str, str | dict[str, str]] = dict()
            for key, value in fmt_info:
                values.append(value)

                if isinstance(key, str):
                    extra[key] = value
                else:
                    k1, k2 = key
                    dct: dict[str, str] = extra.get(k1, {})  # type: ignore[assignment]
                    dct[k2] = value
                    extra[k1] = dct

            self.logger.info(self._log_format % tuple(values), extra=extra)
        except Exception:
            self.logger.exception("Error in logging")
<|endoftext|>import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from asyncio.base_events import BaseEventLoop
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from hashlib import md5, sha1, sha256
from http.cookies import Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Union

import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL

from . import hdrs, helpers, http, multipart, payload
from ._cookie_helpers import (
    parse_cookie_header,
    parse_set_cookie_headers,
    preserve_morsel_with_coded_value,
)
from .abc import AbstractStreamWriter
from .client_exceptions import (
    ClientConnectionError,
    ClientOSError,
    ClientResponseError,
    ContentTypeError,
    InvalidURL,
    ServerFingerprintMismatch,
    UploadAbortedError,
)
from .compression_utils import HAS_BROTLI, HAS_ZSTD
from .formdata import FormData
from .helpers import (
    _SENTINEL,
    BaseTimerContext,
    BasicAuth,
    HeadersMixin,
    TimerNoop,
    _basic_auth_no_warn,
    noop,
    reify,
    sentinel,
    set_exception,
    set_result,
)
from .http import (
    SERVER_SOFTWARE,
    HttpVersion,
    HttpVersion10,
    HttpVersion11,
    StreamWriter,
)
from .streams import StreamReader
from .typedefs import (
    DEFAULT_JSON_DECODER,
    JSONDecoder,
    LooseCookies,
    LooseHeaders,
    Query,
    RawHeaders,
)

if TYPE_CHECKING:
    import ssl
    from ssl import SSLContext
else:
    try:
        import ssl
        from ssl import SSLContext
    except ImportError:  # pragma: no cover
        ssl = None  # type: ignore[assignment]
        SSLContext = object  # type: ignore[misc,assignment]


__all__ = ("ClientRequest", "ClientResponse", "RequestInfo", "Fingerprint")


if TYPE_CHECKING:
    from .client import ClientSession
    from .connector import Connection
    from .tracing import Trace


_CONNECTION_CLOSED_EXCEPTION = ClientConnectionError("Connection closed")
_CONTAINS_CONTROL_CHAR_RE = re.compile(r"[^-!#$%&'*+.^_`|~0-9a-zA-Z]")
json_re = re.compile(r"^application/(?:[\w.+-]+?\+)?json")
_DIGITS_RE = re.compile(r"\d+", re.ASCII)
_LINK_PARAM_RE = re.compile(r"^([^\s=]+)\s*=\s*(?:(['\"])(.*?)\2|(\S*))$", re.M)


def _gen_default_accept_encoding() -> str:
    encodings = [
        "gzip",
        "deflate",
    ]
    if HAS_BROTLI:
        encodings.append("br")
    if HAS_ZSTD:
        encodings.append("zstd")
    return ", ".join(encodings)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ContentDisposition:
    type: str | None
    parameters: "MappingProxyType[str, str]"
    filename: str | None


class _RequestInfo(NamedTuple):
    url: URL
    method: str
    headers: "CIMultiDictProxy[str]"
    real_url: URL


class RequestInfo(_RequestInfo):

    def __new__(
        cls,
        url: URL,
        method: str,
        headers: "CIMultiDictProxy[str]",
        real_url: URL | _SENTINEL = sentinel,
    ) -> "RequestInfo":
        """Create a new RequestInfo instance.

        For backwards compatibility, the real_url parameter is optional.
        """
        return tuple.__new__(
            cls, (url, method, headers, url if real_url is sentinel else real_url)
        )


class Fingerprint:
    HASHFUNC_BY_DIGESTLEN = {
        16: md5,
        20: sha1,
        32: sha256,
    }

    def __init__(self, fingerprint: bytes) -> None:
        digestlen = len(fingerprint)
        hashfunc = self.HASHFUNC_BY_DIGESTLEN.get(digestlen)
        if not hashfunc:
            raise ValueError("fingerprint has invalid length")
        elif hashfunc is md5 or hashfunc is sha1:
            raise ValueError("md5 and sha1 are insecure and not supported. Use sha256.")
        self._hashfunc = hashfunc
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> bytes:
        return self._fingerprint

    def check(self, transport: asyncio.Transport) -> None:
        if not transport.get_extra_info("sslcontext"):
            return
        sslobj = transport.get_extra_info("ssl_object")
        cert = sslobj.getpeercert(binary_form=True)
        got = self._hashfunc(cert).digest()
        if got != self._fingerprint:
            host, port, *_ = transport.get_extra_info("peername")
            raise ServerFingerprintMismatch(self._fingerprint, got, host, port)


if ssl is not None:
    SSL_ALLOWED_TYPES = (ssl.SSLContext, bool, Fingerprint, type(None))
else:  # pragma: no cover
    SSL_ALLOWED_TYPES = (bool, type(None))


def _merge_ssl_params(
    ssl: Union["SSLContext", bool, Fingerprint],
    verify_ssl: bool | None,
    ssl_context: Optional["SSLContext"],
    fingerprint: bytes | None,
) -> Union["SSLContext", bool, Fingerprint]:
    if ssl is None:
        ssl = True  # Double check for backwards compatibility
    if verify_ssl is not None and not verify_ssl:
        warnings.warn(
            "verify_ssl is deprecated, use ssl=False instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if ssl is not True:
            raise ValueError(
                "verify_ssl, ssl_context, fingerprint and ssl "
                "parameters are mutually exclusive"
            )
        else:
            ssl = False
    if ssl_context is not None:
        warnings.warn(
            "ssl_context is deprecated, use ssl=context instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if ssl is not True:
            raise ValueError(
                "verify_ssl, ssl_context, fingerprint and ssl "
                "parameters are mutually exclusive"
            )
        else:
            ssl = ssl_context
    if fingerprint is not None:
        warnings.warn(
            "fingerprint is deprecated, use ssl=Fingerprint(fingerprint) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if ssl is not True:
            raise ValueError(
                "verify_ssl, ssl_context, fingerprint and ssl "
                "parameters are mutually exclusive"
            )
        else:
            ssl = Fingerprint(fingerprint)
    if not isinstance(ssl, SSL_ALLOWED_TYPES):
        raise TypeError(
            "ssl should be SSLContext, bool, Fingerprint or None, "
            f"got {ssl!r} instead."
        )
    return ssl


_SSL_SCHEMES = frozenset(("https", "wss"))


# ConnectionKey is a NamedTuple because it is used as a key in a dict
# and a set in the connector. Since a NamedTuple is a tuple it uses
# the fast native tuple __hash__ and __eq__ implementation in CPython.
class ConnectionKey(NamedTuple):
    # the key should contain an information about used proxy / TLS
    # to prevent reusing wrong connections from a pool
    host: str
    port: int | None
    is_ssl: bool
    ssl: SSLContext | bool | Fingerprint
    proxy: URL | None
    proxy_auth: BasicAuth | None
    proxy_headers_hash: int | None  # hash(CIMultiDict)
    server_hostname: str | None = None


def _is_expected_content_type(
    response_content_type: str, expected_content_type: str
) -> bool:
    if expected_content_type == "application/json":
        return json_re.match(response_content_type) is not None
    return expected_content_type in response_content_type


def _warn_if_unclosed_payload(payload: payload.Payload, stacklevel: int = 2) -> None:
    """Warn if the payload is not closed.

    Callers must check that the body is a Payload before calling this method.

    Args:
        payload: The payload to check
        stacklevel: Stack level for the warning (default 2 for direct callers)
    """
    if not payload.autoclose and not payload.consumed:
        warnings.warn(
            "The previous request body contains unclosed resources. "
            "Use await request.update_body() instead of setting request.body "
            "directly to properly close resources and avoid leaks.",
            ResourceWarning,
            stacklevel=stacklevel,
        )


# Unflushed checkpoints kept before forcing a refresh in _add_bytes.
_MAX_UPLOAD_CHECKPOINTS = 256


class UploadTracker:
    """Tracks upload progress of a single client request.

    Pass a fresh instance via the request's ``upload_tracker`` argument and
    read ``bytes_written``, ``attempts`` and ``upload_complete`` while the
    request runs.

    Must be created inside a running event loop. A tracker observes exactly
    one request: passing it to a second request raises :exc:`RuntimeError`.
    """

    def __init__(self) -> None:
        self.attempts = 0
        self.upload_complete: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._bound = False
        # Outcome of the latest attempt: finished, failed with _exc, or
        # neither (never sent, or cut short).
        self._finished = False
        self._exc: BaseException | None = None
        # Set once the request reaches a terminal point (returned or raised);
        # no further attempts can start after that.
        self._final = False
        # Set while the current attempt's body is being written.
        self._writer: AbstractStreamWriter | None = None
        # Payload bytes handed to the writer for the current attempt.
        self._accepted = 0
        # Payload bytes known to have left the transport for the kernel.
        self._flushed = 0
        # (payload total, wire total) pairs: the last fully flushed point
        # first, then accepted chunks not yet out of the transport's buffer.
        self._checkpoints: deque[tuple[int, int]] = deque()

    @property
    def bytes_written(self) -> int:
        """Payload bytes of the current attempt sent to the kernel.

        Computed from the transport's unsent buffer on access, so it
        excludes bytes still queued in the event loop; it cannot see past
        the kernel's own socket buffering.
        """
        self._refresh_flushed()
        return self._flushed

    def _refresh_flushed(self) -> None:
        writer = self._writer
        if writer is None:
            return
        transport = writer.transport
        if transport is None or transport.is_closing():
            # Torn down: bytes dropped from the buffer were never sent,
            # so keep the last confirmed value.
            return
        wire_sent = writer.output_size - transport.get_write_buffer_size()
        checkpoints = self._checkpoints
        while len(checkpoints) > 1 and checkpoints[1][1] <= wire_sent:
            checkpoints.popleft()
        # A live _writer implies the sentinel pair from _attempt_writing
        # is present, and the loop above never pops the head.
        flushed, wire = checkpoints[0]
        if len(checkpoints) > 1 and wire_sent > wire:
            # Inside a partially flushed chunk: scale the payload bytes
            # linearly over the chunk's wire bytes (exact for identity
            # writes, an estimate under compression).
            payload_cum, wire_cum = checkpoints[1]
            flushed += (payload_cum - flushed) * (wire_sent - wire) // (wire_cum - wire)
        self._flushed = flushed

    def _bind(self) -> None:
        if self._bound:
            raise RuntimeError("UploadTracker is already bound to a request")
        self._bound = True

    def _attempt_started(self) -> int:
        """A new upload attempt is dispatched; returns its generation token.

        Called before connecting, so a resend failing early settles as
        aborted instead of reporting the superseded attempt's outcome.
        """
        self.attempts += 1
        self._writer = None
        self._accepted = 0
        self._flushed = 0
        self._checkpoints.clear()
        self._finished = False
        self._exc = None
        return self.attempts

    def _attempt_writing(self, gen: int, writer: AbstractStreamWriter) -> None:
        """The attempt's body write has begun; settling now defers to it."""
        if gen == self.attempts:
            self._writer = writer
            # Baseline past any bytes already on the wire (e.g. headers
            # sent for the 100-continue preamble).
            self._checkpoints.append((0, writer.output_size))

    def _add_bytes(self, gen: int, size: int, wire_position: int) -> None:
        # A stale writer of a superseded attempt (e.g. still being torn
        # down while a redirect resends the body) must not corrupt the counter.
        if gen == self.attempts and self._writer is not None:
            self._accepted += size
            checkpoints = self._checkpoints
            # A compressor may swallow a chunk without producing output;
            # the wire position is then unchanged and no checkpoint is due.
            if wire_position > checkpoints[-1][1]:
                checkpoints.append((self._accepted, wire_position))
                # Bound the queue when nobody polls bytes_written: flushed
                # entries are only dropped on refresh.
                if len(checkpoints) > _MAX_UPLOAD_CHECKPOINTS:
                    self._refresh_flushed()

    def _attempt_finished(self, gen: int) -> None:
        if gen == self.attempts:
            self._finished = True
            # The body was written in full; the tail still in transit is
            # ordered ahead of the response that completes the request.
            self._flushed = self._accepted
            self._checkpoints.clear()
            self._writer = None
            if self._final:
                self._settle()

    def _attempt_failed(self, gen: int, exc: BaseException | None) -> None:
        """The attempt did not send the body in full.

        ``exc`` is the upload error, or ``None`` when the attempt was cut
        short (cancelled) rather than failed.
        """
        if gen == self.attempts:
            self._exc = exc
            # Keep the last kernel-confirmed value and drop the writer.
            self._refresh_flushed()
            self._checkpoints.clear()
            self._writer = None
            if self._final:
                self._settle()

    def _finalize(self) -> None:
        """Mark the request terminal: no further attempts will start.

        Settles upload_complete unless the final attempt is still writing,
        in which case the attempt's own terminal event settles it.
        """
        self._final = True
        if self._writer is None:
            self._settle()

    def _settle(self) -> None:
        fut = self.upload_complete
        if fut.done():
            return
        if self._finished:
            fut.set_result(None)
            return
        if self._exc is not None:
            fut.set_exception(self._exc)
        else:
            fut.set_exception(UploadAbortedError("The request body was not fully sent"))
        # Avoid 'exception was never retrieved' when doing a .done() poll.
        fut.exception()


class ClientResponse(HeadersMixin):

    # Some of these attributes are None when created,
    # but will be set by the start() method.
    # As the end user will likely never see the None values, we cheat the types below.
    # from the Status-Line of the response
    version: HttpVersion | None = None  # HTTP-Version
    status: int = None  # type: ignore[assignment] # Status-Code
    reason: str | None = None  # Reason-Phrase

    content: StreamReader = None  # type: ignore[assignment] # Payload stream
    _body: bytes | None = None
    _headers: CIMultiDictProxy[str] = None  # type: ignore[assignment]
    _history: tuple["ClientResponse", ...] = ()
    _raw_headers: RawHeaders = None  # type: ignore[assignment]

    _connection: Optional["Connection"] = None  # current connection
    _cookies: SimpleCookie | None = None
    _raw_cookie_headers: tuple[str, ...] | None = None
    _continue: Optional["asyncio.Future[bool]"] = None
    _source_traceback: traceback.StackSummary | None = None
    _session: Optional["ClientSession"] = None
    # set up by ClientRequest after ClientResponse object creation
    # post-init stage allows to not change ctor signature
    _closed = True  # to allow __del__ for non-initialized properly response
    _released = False
    _in_context = False

    _resolve_charset: Callable[["ClientResponse", bytes], str] = lambda *_: "utf-8"

    __writer: Optional["asyncio.Task[None]"] = None
    _stream_writer: Optional[AbstractStreamWriter] = None
    _output_size: int = 0
    _upload_complete: Optional[asyncio.Future[None]] = None

    def __init__(
        self,
        method: str,
        url: URL,
        *,
        writer: "asyncio.Task[None] | None",
        continue100: Optional["asyncio.Future[bool]"],
        timer: BaseTimerContext,
        request_info: RequestInfo,
        traces: list["Trace"],
        loop: asyncio.AbstractEventLoop,
        session: "ClientSession",
        stream_writer: AbstractStreamWriter,
    ) -> None:
        # URL forbids subclasses, so a simple type check is enough.
        assert type(url) is URL

        self.method = method

        self._real_url = url
        self._url = url.with_fragment(None) if url.raw_fragment else url
        if writer is None:  # Request already sent
            self._output_size = stream_writer.output_size
        else:
            self._stream_writer = stream_writer
            self._writer = writer
        if continue100 is not None:
            self._continue = continue100
        self._request_info = request_info
        self._timer = timer if timer is not None else TimerNoop()
        self._cache: dict[str, Any] = {}
        self._traces = traces
        self._loop = loop
        # Save reference to _resolve_charset, so that get_encoding() will still
        # work after the response has finished reading the body.
        # TODO: Fix session=None in tests (see ClientRequest.__init__).
        if session is not None:
            # store a reference to session #1985
            self._session = session
            self._resolve_charset = session._resolve_charset
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

    def __reset_writer(self, _: object = None) -> None:
        self.__writer = None
        if self._stream_writer is not None:
            self._output_size = self._stream_writer.output_size
            self._stream_writer = None
        if self._upload_complete is not None and not self._upload_complete.done():
            self._upload_complete.set_result(None)

    @property
    def _writer(self) -> Optional["asyncio.Task[None]"]:
        """The writer task for streaming data.

        _writer is only provided for backwards compatibility
        for subclasses that may need to access it.
        """
        return self.__writer

    @_writer.setter
    def _writer(self, writer: Optional["asyncio.Task[None]"]) -> None:
        """Set the writer task for streaming data."""
        if self.__writer is not None:
            self.__writer.remove_done_callback(self.__reset_writer)
        self.__writer = writer
        if writer is None:
            return
        if writer.done():
            # The writer is already done, so we can clear it immediately.
            self.__reset_writer()
        else:
            writer.add_done_callback(self.__reset_writer)

    @property
    def output_size(self) -> int:
        """Number of bytes sent for this request.

        .. deprecated:: 3.14.4
           Use :class:`UploadTracker` instead.
        """
        warnings.warn(
            "ClientResponse.output_size is deprecated, "
            "use aiohttp.UploadTracker instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._stream_writer is not None:
            return self._stream_writer.output_size
        return self._output_size

    @property
    def upload_complete(self) -> "asyncio.Future[None]":
        """Future set when the request body has been fully sent.

        .. deprecated:: 3.14.4
           Use :class:`UploadTracker` instead.
        """
        warnings.warn(
            "ClientResponse.upload_complete is deprecated, "
            "use aiohttp.UploadTracker instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._upload_complete is None:
            self._upload_complete = self._loop.create_future()
            if self._stream_writer is None:  # upload already finished
                self._upload_complete.set_result(None)
        return self._upload_complete

    @property
    def cookies(self) -> SimpleCookie:
        if self._cookies is None:
