) -> Callable[[Sequence[Any] | None], list[Any] | None]:
    """Turn Click's tuple (from ``multiple=True``) into a list, applying ``inner`` per item."""

    def convertor(value: Sequence[Any] | None) -> list[Any] | None:
        if value is None or (default_value is None and len(value) == 0):
            return None
        return [inner(item) if inner else item for item in value]

    return convertor


def _make_convertor(
    annotation: Any, is_list: bool, default_value: Any | None, info: ParameterInfo
) -> Callable[[Any], Any] | None:
    """Build the value convertor for a param (``None`` when no coercion is needed).

    An explicit ``click_type`` (e.g. ``SoftChoice``) owns its own conversion, so no
    enum convertor is added on top of it.
    """
    inner = None
    if info.click_type is None and isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        inner = _enum_convertor(annotation)
    if is_list:
        return _list_convertor(inner, default_value)
    return inner


def _unwrap_optional_and_list(annotation: Any) -> tuple[Any, bool]:
    """Strip ``T | None`` / ``Optional[T]`` and ``list[T]``; return ``(base_type, is_list)``."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"Unsupported union annotation {annotation!r}; only ``T | None`` is supported.")
        annotation = non_none[0]
        origin = get_origin(annotation)
    if origin in (list, Sequence):
        return get_args(annotation)[0], True
    return annotation, False


def _split_annotated(annotation: Any) -> tuple[Any, ParameterInfo | None]:
    """Split ``Annotated[T, Option(...)]`` into ``(T, marker)``; ``(annotation, None)`` otherwise."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        marker = next((meta for meta in args[1:] if isinstance(meta, ParameterInfo)), None)
        if marker is None and (typer_marker := next((meta for meta in args[1:] if _is_typer_marker(meta)), None)):
            marker = _from_typer_marker(typer_marker, from_annotated=True)
        return args[0], marker
    return annotation, None


# ---------------------------------------------------------------------------
# Typer-marker compatibility shim (transition helper).
#
# ``typer_factory`` is exposed publicly and downstream CLIs (e.g. `transformers`)
# still register commands whose parameters carry ``typer.Option`` /
# ``typer.Argument`` markers. Recognize those markers structurally (no typer
# import required) and translate the fields this framework supports, so such
# commands keep working while downstream migrates.
# TODO: remove once transformers pins huggingface_hub>=1.22.0.
# ---------------------------------------------------------------------------


def _is_typer_marker(obj: Any) -> bool:
    return type(obj).__module__.partition(".")[0] == "typer" and type(obj).__name__ in ("OptionInfo", "ArgumentInfo")


def _from_typer_marker(meta: Any, *, from_annotated: bool) -> ParameterInfo:
    """Translate a ``typer.models.OptionInfo`` / ``ArgumentInfo`` into our marker."""
    param_decls = list(getattr(meta, "param_decls", None) or ())
    if type(meta).__name__ == "OptionInfo":
        # In ``Annotated[...]`` usage typer stores the first flag name in ``default``
        # (``Option("--flag", "-f")`` -> ``default="--flag"``, ``param_decls=("-f",)``).
        default = getattr(meta, "default", ...)
        if from_annotated and isinstance(default, str):
            param_decls = [default, *param_decls]
        cls: type[ParameterInfo] = Option
    else:
        cls = Argument
    return cls(
        *param_decls,
        help=getattr(meta, "help", None),
        show_default=getattr(meta, "show_default", True),
        hidden=getattr(meta, "hidden", False),
        is_eager=getattr(meta, "is_eager", False),
        callback=getattr(meta, "callback", None),
        min=getattr(meta, "min", None),
        click_type=getattr(meta, "click_type", None),
    )


# ---------------------------------------------------------------------------
# 3. Signature -> Click parameters (mirrors ``typer.main.get_click_param``).
# ---------------------------------------------------------------------------


def _wrap_param_callback(
    user_callback: Callable[[Any], Any], convertor: Callable[[Any], Any] | None
) -> Callable[[click.Context, click.Parameter, Any], Any]:
    """Adapt a Typer-style value parser ``fn(value) -> parsed`` to a Click param callback."""

    def click_callback(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        return user_callback(convertor(value) if convertor else value)

    return click_callback


def _build_click_param(
    name: str, annotation: Any, info: ParameterInfo | None, signature_default: Any
) -> tuple[click.Parameter, Callable[[Any], Any] | None]:
    """Build a Click ``Parameter`` (and its post-parse value convertor) for one function param."""
    required = signature_default is inspect.Parameter.empty
    default_value = None if required else signature_default
    if isinstance(default_value, enum.Enum):
        default_value = default_value.value
    if info is None:
        # A bare annotation with no marker: required positional -> Argument, else Option.
        info = Argument() if required else Option()

    base_type, is_list = _unwrap_optional_and_list(annotation)
    convertor = _make_convertor(base_type, is_list, default_value, info)

    # A user ``callback`` consumes the value itself (and the convertor with it); otherwise the
    # convertor runs later in the command handler wrapper. This keeps conversion single-pass.
    if info.callback is not None:
        param_callback: Callable[..., Any] | None = _wrap_param_callback(info.callback, convertor)
        handler_convertor: Callable[[Any], Any] | None = None
    else:
        param_callback = None
        handler_convertor = convertor

    if isinstance(info, Argument):
        argument = HfArgument(
            [name],
            type=_get_click_type(base_type, info),
            required=required,
            default=default_value,
            nargs=-1 if is_list else 1,
            is_eager=info.is_eager,
            callback=param_callback,
            help=info.help,
            show_default=info.show_default,
            hidden=info.hidden,
        )
        return argument, handler_convertor

    # Option: prepend the Python name so ``click.Parameter.name`` matches the function kwarg,
    # regardless of the user-facing flag names ("-v"/"--version", "--type"/"--repo-type", ...).
    is_flag = base_type is bool
    decls = [name]
    if info.param_decls:
        decls.extend(info.param_decls)
    else:
        kebab = get_command_name(name)
        decls.append(f"--{kebab}/--no-{kebab}" if is_flag else f"--{kebab}")
    option = HfOption(
        decls,
        # Click infers the flag type itself; passing a type alongside ``is_flag`` is rejected.
        type=None if is_flag else _get_click_type(base_type, info),
        required=required,
        default=default_value,
        is_flag=is_flag,
        multiple=is_list,
        is_eager=info.is_eager,
        callback=param_callback,
        help=info.help,
        show_default=info.show_default,
        hidden=info.hidden,
    )
    return option, handler_convertor


def _build_params(
    func: Callable[..., Any],
) -> tuple[list[click.Parameter], dict[str, Callable[[Any], Any]], str | None]:
    """Introspect ``func`` and return ``(params, convertors, context_param_name)``."""
    params: list[click.Parameter] = []
    convertors: dict[str, Callable[[Any], Any]] = {}
    context_param_name: str | None = None
    for name, sig_param in inspect.signature(func).parameters.items():
        # Read raw signature annotations rather than ``typing.get_type_hints``: the CLI never uses
        # ``from __future__ import annotations``, so annotations are already live objects, and
        # get_type_hints mangles ``Annotated[X | None, ...]`` on Python 3.10 (drops the origin).
        annotation = sig_param.annotation if sig_param.annotation is not inspect.Parameter.empty else str
        base_type, info = _split_annotated(annotation)
        if isinstance(base_type, type) and (
            issubclass(base_type, click.Context)
            # typer.Context subclasses typer's *vendored* click since typer 0.26, so an
            # issubclass check against real click misses it. Match it structurally.
            or (base_type.__module__.partition(".")[0] == "typer" and base_type.__name__ == "Context")
        ):
            context_param_name = name
            continue
        signature_default = sig_param.default
        if _is_typer_marker(signature_default):
            # Old typer style: the marker *is* the signature default and carries the value.
            if info is None:
                info = _from_typer_marker(signature_default, from_annotated=False)
            marker_default = getattr(signature_default, "default", ...)
            signature_default = inspect.Parameter.empty if marker_default is ... else marker_default
        click_param, convertor = _build_click_param(name, base_type, info, signature_default)
        if convertor is not None and click_param.name is not None:
            convertors[click_param.name] = convertor
        params.append(click_param)
    return params, convertors, context_param_name


def _make_handler(
    func: Callable[..., Any],
    convertors: dict[str, Callable[[Any], Any]],
    context_param_name: str | None,
) -> Callable[..., Any]:
    """Wrap ``func`` so Click's kwargs map back to it, applying convertors and injecting the ctx."""

    def handler(**kwargs: Any) -> Any:
        call_kwargs = {key: (convertors[key](value) if key in convertors else value) for key, value in kwargs.items()}
        if context_param_name is not None:
            call_kwargs[context_param_name] = click.get_current_context()
        return func(**call_kwargs)

    functools.update_wrapper(handler, func)
    return handler


# ---------------------------------------------------------------------------
# 4. Click parameter subclasses — port Typer's help records so arguments show up
#    in ``--help`` (plain ``click.Argument`` renders none) and defaults/metavars
#    read identically. Rich/env-var branches are dropped (never enabled here).
# ---------------------------------------------------------------------------


def _split_opt(opt: str) -> tuple[str, str]:
    """Split an option string into ``(prefix, name)`` (Click's ``split_opt``, inlined)."""
    first = opt[:1]
    if first.isalnum():
        return "", opt
    if opt[1:2] == first:
        return opt[:2], opt[2:]
    return first, opt[1:]


def _extract_default_help_str(param: click.Parameter, ctx: click.Context) -> Any:
    # Resilient parsing avoids type casting failing while rendering the default.
    resilient = ctx.resilient_parsing
    ctx.resilient_parsing = True
    try:
        return param.get_default(ctx, call=False)
    finally:
        ctx.resilient_parsing = resilient


def _default_string(
    param: "HfArgument | HfOption", ctx: click.Context, show_default_is_str: bool, default_value: Any
) -> str:
    if show_default_is_str:
        return f"({param.show_default})"
    if isinstance(default_value, (list, tuple)):
        return ", ".join(_default_string(param, ctx, show_default_is_str, item) for item in default_value)
    if isinstance(default_value, enum.Enum):
        return str(default_value.value)
    if inspect.isfunction(default_value):
        return "(dynamic)"
    if isinstance(param, HfOption) and param.is_bool_flag and param.secondary_opts:
        # Boolean toggle: show the opt name (without prefix) matching the current default.
        if default_value:
            return _split_opt(param.opts[0])[1] if param.opts else str(default_value)
        return _split_opt(param.secondary_opts[0])[1]
    if isinstance(param, HfOption) and param.is_bool_flag and not param.secondary_opts and not default_value:
        return ""
    return str(default_value)


def _describe_number_range(param_type: "click.IntRange | click.FloatRange") -> str:
    """Human-readable range hint like ``x>=1`` or ``1<=x<10``.

    Reproduces click's private ``_NumberRangeBase._describe_range()`` using the public
    ``IntRange``/``FloatRange`` attributes, so the framework stays on Click's public API.
    """
    if param_type.min is None:
        return f"x{'<' if param_type.max_open else '<='}{param_type.max}"
    if param_type.max is None:
        return f"x{'>' if param_type.min_open else '>='}{param_type.min}"
    left = "<" if param_type.min_open else "<="
    right = "<" if param_type.max_open else "<="
    return f"{param_type.min}{left}x{right}{param_type.max}"


def _build_help_extra(param: "HfArgument | HfOption", ctx: click.Context, base_help: str) -> str:
    """Append the ``[default: ...; required]`` suffix to a help string (shared by arg/option)."""
    extra: list[str] = []
    default_value = _extract_default_help_str(param, ctx)
    show_default_is_str = isinstance(param.show_default, str)
    if show_default_is_str or (default_value is not None and (param.show_default or ctx.show_default)):
        default_string = _default_string(param, ctx, show_default_is_str, default_value)
        if default_string:
            extra.append(f"default: {default_string}")
    # Numeric range hints are shown for options only (matches Typer; arguments omit them).
    if isinstance(param, HfOption) and isinstance(param.type, (click.IntRange, click.FloatRange)):
        range_str = _describe_number_range(param.type)
        if range_str:
            extra.append(range_str)
    if param.required:
        extra.append("required")
    if extra:
        suffix = f"[{'; '.join(extra)}]"
        return f"{base_help}  {suffix}" if base_help else suffix
    return base_help


class HfArgument(click.Argument):
    """Positional argument that renders help text and a metavar (Click's does neither)."""

    def __init__(
        self,
        param_decls: Sequence[str],
        *,
        help: str | None = None,
        show_default: bool | str = True,
        hidden: bool = False,
        **attrs: Any,
    ) -> None:
        self.show_default = show_default
        self.hidden = hidden
        super().__init__(param_decls, **attrs)
        # Set *after* ``super().__init__``: Click >= 8.5 gives ``Argument`` its own ``help``
        # parameter and resets ``self.help`` to ``None`` in its constructor.
        self.help = help

    def make_metavar(self, ctx: click.Context) -> str:
        if self.metavar is not None:
            var = self.metavar
            if not self.required and not var.startswith("["):
                var = f"[{var}]"
            return var
        var = (self.name or "").upper()
        if not self.required:
            var = f"[{var}]"
        type_var = self.type.get_metavar(self, ctx=ctx)
        if type_var:
            var += f":{type_var}"
        if self.nargs != 1:
            var += "..."
        return var

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        if self.hidden:
            return None
        return self.make_metavar(ctx=ctx), _build_help_extra(self, ctx, self.help or "")


class HfOption(click.Option):
    """Option whose help record ports Typer's default/metavar rendering verbatim."""

    show_default: bool | str

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        if self.hidden:
            return None

        any_prefix_is_slash = False

        def _write_opts(opts: Sequence[str]) -> str:
            nonlocal any_prefix_is_slash
            rv, any_slashes = click.formatting.join_options(opts)
            if any_slashes:
                any_prefix_is_slash = True
            if not self.is_flag and not self.count:
                rv += f" {self.make_metavar(ctx=ctx)}"
            return rv

        rv = [_write_opts(self.opts)]
        if self.secondary_opts:
            rv.append(_write_opts(self.secondary_opts))

        help_text = _build_help_extra(self, ctx, self.help or "")
        return ("; " if any_prefix_is_slash else " / ").join(rv), help_text


# ---------------------------------------------------------------------------
# 5. Command / group base classes and the decorator API (replace ``typer.Typer``).
# ---------------------------------------------------------------------------


def _format_params(command: click.Command, ctx: click.Context, formatter: click.HelpFormatter) -> None:
    """Render params split into "Arguments" and "Options" sections (mirrors Typer)."""
    args: list[tuple[str, str]] = []
    opts: list[tuple[str, str]] = []
    for param in command.get_params(ctx):
        record = param.get_help_record(ctx)
        if record is None:
            continue
        if param.param_type_name == "argument":
            args.append(record)
        elif param.param_type_name == "option":
            opts.append(record)
    if args:
        with formatter.section("Arguments"):
            formatter.write_dl(args)
    if opts:
        with formatter.section("Options"):
            formatter.write_dl(opts)


class HfCommand(click.Command):
    """Leaf command that renders arguments and options in separate help sections."""

    def format_arguments(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Click >= 8.5 added a `format_arguments` step to `format_help` that prints its own
        # "Positional arguments" section. `_format_params` already lists arguments (Typer-style),
        # so this override avoids printing them twice. Never called on Click < 8.5.
        return None

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _format_params(self, ctx, formatter)


def build_command(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    cls: type[click.Command] | None = None,
    help: str | None = None,
    epilog: str | None = None,
    short_help: str | None = None,
    options_metavar: str = "[OPTIONS]",
    add_help_option: bool = True,
    no_args_is_help: bool = False,
    hidden: bool = False,
    deprecated: bool = False,
    context_settings: dict[str, Any] | None = None,
) -> click.Command:
    """Build a Click ``Command`` from a function with ``Annotated`` params.

    Replaces ``typer.main.get_command_from_info``.
    """
    params, convertors, context_param_name = _build_params(func)
    handler = _make_handler(func, convertors, context_param_name)
    command_help = inspect.cleandoc(help) if help else inspect.getdoc(func)
    return (cls or HfCommand)(
        name=name if name is not None else get_command_name(func.__name__),
        callback=handler,
        params=params,
        help=command_help,
        epilog=epilog,
        short_help=short_help,
        options_metavar=options_metavar,
        add_help_option=add_help_option,
        no_args_is_help=no_args_is_help,
        hidden=hidden,
        deprecated=deprecated,
        context_settings=context_settings or {},
    )


class HfGroup(click.Group):
    """Command group with the decorator API the CLI relies on (``command``/``callback``/``add_group``).

    Subclasses (see ``HFCliTyperGroup``) layer on styling, aliases, topics and error
    enrichment; this base only wires functions to Click via :func:`build_command`.
    """

    #: Command class used by ``@group.command()`` unless overridden per call.
    command_class: type[click.Command] = HfCommand

    def format_arguments(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # See `HfCommand.format_arguments`.
        return None

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _format_params(self, ctx, formatter)
        self.format_commands(ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:
        # Preserve declaration order rather than Click's alphabetical default.
        return list(self.commands)

    def command(  # type: ignore  # deliberately narrows click.Group.command (builds from annotated signatures)
        self, name: str | None = None, *, cls: type[click.Command] | None = None, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            command = build_command(func, name=name, cls=cls or self.command_class, **kwargs)
            self.add_command(command, command.name)
            return func

        return decorator

    def add_group(self, group: click.Group, *, name: str, hidden: bool = False) -> None:
        """Register a subgroup under ``name`` (which may carry pipe aliases, e.g. ``"repos | repo"``)."""
        group.name = name
        group.hidden = hidden
        self.add_command(group, name)

    def group_callback(
        self, *, invoke_without_command: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the function invoked for this group (named ``group_callback`` because Click
        already uses the ``callback`` attribute for the group's own handler)."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            params, convertors, context_param_name = _build_params(func)
            self.callback = _make_handler(func, convertors, context_param_name)
            self.params = [*self.params, *params]
            self.invoke_without_command = invoke_without_command
            if invoke_without_command:
                # Match Typer: the subcommand is optional when the group runs without one.
                self.subcommand_metavar = "[COMMAND] [ARGS]..."
            if self.help is None and func.__doc__:
                self.help = inspect.cleandoc(func.__doc__)
            return func

        return decorator
<|endoftext|>"""Interactive isometric city explorer — easter egg for `hf repos ls --explore`."""

import dataclasses
import math
import os
import random
import re
import select
import shutil
import sys
import time

from huggingface_hub.hf_api import RepoStorageInfo

from ._file_listing import format_size


Color = tuple[int, int, int]

# (top_face, left_face, right_face) — lighter to darker for 3D effect
_TYPE_COLORS: dict[str, tuple[Color, Color, Color]] = {
    "model": ((175, 148, 240), (138, 112, 208), (105, 80, 180)),
    "dataset": ((245, 128, 128), (222, 92, 92), (190, 60, 60)),
    "space": ((245, 175, 85), (218, 140, 55), (185, 110, 30)),
    "bucket": ((112, 185, 242), (70, 150, 220), (40, 118, 192)),
}
_EXTRA_COLORS: tuple[Color, Color, Color] = ((168, 176, 188), (128, 136, 148), (90, 98, 110))
_GRID_COLOR: Color = (178, 182, 190)

_DX = 4  # isometric half-width (pixels)
_DY = 2  # isometric half-height (pixels)
_MAX_H = 16  # tallest tile (pixels)
_MIN_H = 1
_COLS = 6
_EXT = 1  # grid extension beyond tiles
_MAX_TILES = 30

# Cursor sprite — pixel art arrow pointer
_OUTLINE: Color = (30, 30, 30)
_FILL: Color = (255, 255, 255)

_CURSOR_GRID = [
    "  X  ",
    " XWX ",
    "XWWWX",
    " XWX ",
    "  X  ",
]
_CURSOR_PALETTE: dict[str, Color] = {
    "X": _OUTLINE,
    "W": _FILL,
}
_CURSOR_H = len(_CURSOR_GRID)

_MOVE_FRAMES = 8
_MOVE_DELAY = 0.03
_CURSOR_PAD = _CURSOR_H + 16
_GAP = 3
_MIN_TERM_W = 100
_MIN_TERM_H = 24
_SUMMARY_W = 24


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TileInfo:
    grid_row: int
    grid_col: int
    height: int
    top: Color
    left: Color
    right: Color
    repo: RepoStorageInfo | None


@dataclasses.dataclass
class CityData:
    tiles: list[TileInfo]
    rows: int
    cols: int
    x_off: int
    y_off: int
    buf_w: int
    buf_h: int
    total_storage: int
    extra_count: int
    extra_storage: int
    all_repos: list[RepoStorageInfo]


# ---------------------------------------------------------------------------
# City layout
# ---------------------------------------------------------------------------


def _prepare_city_data(repos: list[RepoStorageInfo]) -> CityData:
    sorted_repos = sorted(repos, key=lambda r: r.storage, reverse=True)
    display = sorted_repos[:_MAX_TILES]
    extra_count = max(0, len(sorted_repos) - _MAX_TILES)
    extra_storage = sum(r.storage for r in sorted_repos[_MAX_TILES:])
    total_storage = sum(r.storage for r in repos)
    max_storage = max(1, display[0].storage)

    n = len(display) + (1 if extra_count > 0 else 0)
    cols = min(n, _COLS)
    rows = math.ceil(n / cols) if cols > 0 else 1

    tiles: list[TileInfo] = []
    for i, repo in enumerate(display):
        r, c = divmod(i, cols)
        h = max(_MIN_H, round(math.sqrt(repo.storage / max_storage) * _MAX_H))
        top, left, right = _TYPE_COLORS.get(repo.type, _EXTRA_COLORS)
        tiles.append(TileInfo(r, c, h, top, left, right, repo))
    if extra_count > 0:
        r, c = divmod(len(display), cols)
        h = max(_MIN_H, round(math.sqrt(extra_storage / max_storage) * _MAX_H))
        tiles.append(TileInfo(r, c, h, *_EXTRA_COLORS, None))

    r_lo, r_hi = -_EXT, rows - 1 + _EXT
    c_lo, c_hi = -_EXT, cols - 1 + _EXT

    xs: list[int] = []
    ys: list[int] = []
    for rr in range(r_lo, r_hi + 1):
        for cc in range(c_lo, c_hi + 1):
            cx, cy = (cc - rr) * _DX, (cc + rr) * _DY
            xs.extend([cx - _DX, cx + _DX])
            ys.extend([cy, cy + 2 * _DY])
    for tile in tiles:
        ys.append((tile.grid_col + tile.grid_row) * _DY - tile.height)

    x_off = -min(xs)
    y_off = -min(ys)
    buf_w = max(xs) - min(xs) + 1
    buf_h = max(ys) - min(ys) + 1
    if buf_h % 2:
        buf_h += 1

    return CityData(
        tiles=tiles,
        rows=rows,
        cols=cols,
        x_off=x_off,
        y_off=y_off,
        buf_w=buf_w,
        buf_h=buf_h,
        total_storage=total_storage,
        extra_count=extra_count,
        extra_storage=extra_storage,
        all_repos=repos,
    )


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_diamond_outline(buf: list[list[Color | None]], cx: int, cy: int) -> None:
    t = (cx, cy)
    r = (cx + _DX, cy + _DY)
    b = (cx, cy + 2 * _DY)
    ll = (cx - _DX, cy + _DY)
    _draw_line(buf, *t, *r, _GRID_COLOR)
    _draw_line(buf, *r, *b, _GRID_COLOR)
    _draw_line(buf, *b, *ll, _GRID_COLOR)
    _draw_line(buf, *ll, *t, _GRID_COLOR)


def _draw_block(
    buf: list[list[Color | None]],
    cx: int,
    cy: int,
    h: int,
    top: Color,
    left: Color,
    right: Color,
) -> None:
    _fill_poly(
        buf,
        [(cx - _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx, cy + 2 * _DY), (cx - _DX, cy + _DY)],
        left,
    )
    _fill_poly(
        buf,
        [(cx, cy + 2 * _DY - h), (cx + _DX, cy + _DY - h), (cx + _DX, cy + _DY), (cx, cy + 2 * _DY)],
        right,
    )
    _fill_poly(
        buf,
        [(cx, cy - h), (cx + _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx - _DX, cy + _DY - h)],
        top,
    )


def _fill_poly(buf: list[list[Color | None]], verts: list[tuple[int, int]], color: Color) -> None:
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    all_y = [v[1] for v in verts]
    y0 = max(0, min(all_y))
    y1 = min(bh - 1, max(all_y))
    n = len(verts)
    for y in range(y0, y1 + 1):
        xl: float = float("inf")
        xr: float = float("-inf")
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            if ay == by:
                if y == ay:
                    xl = min(xl, float(min(ax, bx)))
                    xr = max(xr, float(max(ax, bx)))
                continue
            if not (min(ay, by) <= y <= max(ay, by)):
                continue
            t = (y - ay) / (by - ay)
            ix = ax + t * (bx - ax)
            xl = min(xl, ix)
            xr = max(xr, ix)
        if xl <= xr:
            for x in range(max(0, round(xl)), min(bw, round(xr) + 1)):
                buf[y][x] = color


def _draw_line(buf: list[list[Color | None]], x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy)
    if steps == 0:
        if 0 <= y0 < bh and 0 <= x0 < bw:
            buf[y0][x0] = color
        return
    xi = (x1 - x0) / steps
    yi = (y1 - y0) / steps
    fx, fy = float(x0), float(y0)
    for _ in range(steps + 1):
        px, py = round(fx), round(fy)
        if 0 <= py < bh and 0 <= px < bw:
            buf[py][px] = color
        fx += xi
        fy += yi


# ---------------------------------------------------------------------------
# Pixel buffer → terminal
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _visible_len(s: str) -> int:
    return len(_strip_ansi(s))


def _pixels_to_lines(buf: list[list[Color | None]]) -> list[str]:
    height = len(buf)
    width = len(buf[0]) if buf else 0
    lines: list[str] = []
    for row in range(0, height, 2):
        last = -1
        for col in range(width - 1, -1, -1):
            top = buf[row][col]
            bot = buf[row + 1][col] if row + 1 < height else None
            if top or bot:
                last = col
                break
        if last < 0:
            lines.append("")
            continue

        parts: list[str] = []
        cfg: Color | None = None
        cbg: Color | None = None

        for col in range(last + 1):
            top = buf[row][col]
            bot = buf[row + 1][col] if row + 1 < height else None

            if not top and not bot:
                if cfg is not None or cbg is not None:
                    parts.append("\033[0m")
                    cfg = cbg = None
                parts.append(" ")
                continue

            if top and bot and top == bot:
                nfg, nbg, ch = top, None, "█"
            elif top and bot:
                nfg, nbg, ch = bot, top, "▄"
            elif top:
                nfg, nbg, ch = top, None, "▀"
            else:
                nfg, nbg, ch = bot, None, "▄"  # type: ignore[assignment]

            esc = ""
            if nfg != cfg:
                esc += f"\033[38;2;{nfg[0]};{nfg[1]};{nfg[2]}m"
                cfg = nfg
            if nbg != cbg:
                esc += "\033[49m" if nbg is None else f"\033[48;2;{nbg[0]};{nbg[1]};{nbg[2]}m"
                cbg = nbg
            parts.append(esc + ch)

        if cfg is not None or cbg is not None:
            parts.append("\033[0m")
        lines.append("".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_base_buffer(city: CityData) -> list[list[Color | None]]:
    buf: list[list[Color | None]] = [[None] * city.buf_w for _ in range(city.buf_h)]

    for tile in city.tiles:
        cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
        cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
        _draw_diamond_outline(buf, cx, cy)

    sorted_tiles = sorted(city.tiles, key=lambda t: (t.grid_row + t.grid_col, t.grid_col))
    for tile in sorted_tiles:
        cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
        cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
        _draw_block(buf, cx, cy, tile.height, tile.top, tile.left, tile.right)

    return buf


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------


def _colored_square(color: Color) -> str:
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m■\033[0m"


def _build_summary(
    repos: list[RepoStorageInfo],
    total_storage: int,
    extra_count: int,
) -> list[str]:
    lines: list[str] = [""]
    lines.append("  Storage Overview")
    lines.append("  " + "─" * 16)
    lines.append(f"  {format_size(total_storage, human_readable=True)} total")
    lines.append("")

    order = ["model", "dataset", "space", "bucket"]
    labels = {"model": "Models", "dataset": "Datasets", "space": "Spaces", "bucket": "Buckets"}
    for rtype in order:
        group = [r for r in repos if r.type == rtype]
        if not group:
            continue
        storage = sum(r.storage for r in group)
        sq = _colored_square(_TYPE_COLORS[rtype][0])
        lines.append(f"  {sq} {labels[rtype]}")
        lines.append(f"    {len(group)} repos · {format_size(storage, human_readable=True)}")
        lines.append("")

    if extra_count > 0:
        sq = _colored_square(_EXTRA_COLORS[0])
        lines.append(f"  {sq} +{extra_count} more repos")

    return lines


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


def _build_cursor() -> list[tuple[int, int, Color]]:
    pixels: list[tuple[int, int, Color]] = []
    for ri, row in enumerate(_CURSOR_GRID):
        for ci, ch in enumerate(row):
            if ch in _CURSOR_PALETTE:
                pixels.append((ci - len(row) // 2, ri - _CURSOR_H + 1, _CURSOR_PALETTE[ch]))
    return pixels


_CURSOR_PIXELS = _build_cursor()


# ---------------------------------------------------------------------------
# Interactive game
# ---------------------------------------------------------------------------


def run_city_game(repos: list[RepoStorageInfo]) -> None:
    """Launch the interactive city explorer."""
    if not repos:
        print("No repositories found.")
        return

    try:
        import termios
        import tty
    except ImportError:
        print("Interactive mode requires a Unix-like terminal (Linux/macOS).")
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Interactive mode requires a terminal.")
        return

    term = shutil.get_terminal_size()
    if term.columns < _MIN_TERM_W or term.lines < _MIN_TERM_H:
        print(f"Your terminal is {term.columns}×{term.lines} characters.")
        print(f"Please resize to at least {_MIN_TERM_W}×{_MIN_TERM_H} to explore the city!")
        return

    city = _prepare_city_data(repos)

    tiles_with_repos = [t for t in city.tiles if t.repo is not None]
    start_tile = random.choice(tiles_with_repos) if tiles_with_repos else city.tiles[0]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?1049h\033[?25l\033[2J")
        sys.stdout.flush()
        _game_loop(city, start_tile.grid_row, start_tile.grid_col)
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _game_loop(city: CityData, cur_row: int, cur_col: int) -> None:
    tile_map: dict[tuple[int, int], TileInfo] = {(t.grid_row, t.grid_col): t for t in city.tiles}
    city = dataclasses.replace(city, buf_h=city.buf_h + _CURSOR_PAD, y_off=city.y_off + _CURSOR_PAD)
    base_buf = _render_base_buffer(city)

    summary = _build_summary(city.all_repos, city.total_storage, city.extra_count)

    # Intro: cursor drops onto starting tile
    tx, ty = _tile_top_center(city, cur_row, cur_col, tile_map)
    for i in range(1, _MOVE_FRAMES + 1):
        t = i / _MOVE_FRAMES
        t = t * t * (3 - 2 * t)
        drop_y = ty - 16 * (1 - t)
        frame = _copy_buf(base_buf)
        _highlight_tile(frame, city, tile_map[(cur_row, cur_col)])
        _draw_cursor(frame, tx, round(drop_y))
        _present(city, frame, tile_map.get((cur_row, cur_col)), summary)
        time.sleep(_MOVE_DELAY)

    while True:
        cx, cy = _tile_top_center(city, cur_row, cur_col, tile_map)
        frame = _copy_buf(base_buf)
        _highlight_tile(frame, city, tile_map[(cur_row, cur_col)])
        _draw_cursor(frame, cx, cy)
        _present(city, frame, tile_map.get((cur_row, cur_col)), summary)

        key = _read_key()
        if key in ("q", "Q", "esc", "\x03"):
            return

        dr, dc = _key_to_direction(key)
        if dr == 0 and dc == 0:
            continue

        nr, nc = cur_row + dr, cur_col + dc
        if (nr, nc) not in tile_map:
            continue

        ex, ey = _tile_top_center(city, nr, nc, tile_map)
        for i in range(1, _MOVE_FRAMES + 1):
            t = i / _MOVE_FRAMES
            t = t * t * (3 - 2 * t)
            bx = cx + (ex - cx) * t
            by = cy + (ey - cy) * t
            frame = _copy_buf(base_buf)
            _highlight_tile(frame, city, tile_map[(nr, nc)])
            _draw_cursor(frame, round(bx), round(by))
            _present(city, frame, tile_map.get((nr, nc)), summary)
            time.sleep(_MOVE_DELAY)

        cur_row, cur_col = nr, nc


def _tile_top_center(city: CityData, row: int, col: int, tile_map: dict[tuple[int, int], TileInfo]) -> tuple[int, int]:
    tile = tile_map.get((row, col))
    h = tile.height if tile else 1
    cx = city.x_off + (col - row) * _DX
    cy = city.y_off + (col + row) * _DY
    return cx, cy + _DY - h


def _key_to_direction(key: str) -> tuple[int, int]:
    match key:
        case "w" | "W" | "\x1b[A":
            return -1, 0
        case "s" | "S" | "\x1b[B":
            return 1, 0
        case "a" | "A" | "\x1b[D":
            return 0, -1
        case "d" | "D" | "\x1b[C":
            return 0, 1
        case _:
            return 0, 0


def _draw_cursor(buf: list[list[Color | None]], cx: int, cy: int) -> None:
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    for dx, dy, color in _CURSOR_PIXELS:
        px, py = cx + dx, cy + dy
        if 0 <= py < bh and 0 <= px < bw:
            buf[py][px] = color


def _highlight_tile(buf: list[list[Color | None]], city: CityData, tile: TileInfo) -> None:
    cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
    cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
    h = tile.height
    _fill_poly(
        buf,
        [(cx, cy - h), (cx + _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx - _DX, cy + _DY - h)],
        _brighten(tile.top, 35),
    )


def _brighten(color: Color, amount: int) -> Color:
    return (min(255, color[0] + amount), min(255, color[1] + amount), min(255, color[2] + amount))


def _present(
    city: CityData,
    buf: list[list[Color | None]],
    tile: TileInfo | None,
    summary: list[str],
) -> None:
    city_lines = _pixels_to_lines(buf)
    while city_lines and not _strip_ansi(city_lines[0]).strip():
        city_lines.pop(0)
    while city_lines and not _strip_ansi(city_lines[-1]).strip():
        city_lines.pop()

    city_w = max((_visible_len(line) for line in city_lines), default=0)
    term = shutil.get_terminal_size()
    panel_max_w = max(20, term.columns - city_w - _SUMMARY_W - 2 * _GAP)

    info = _build_info_panel(tile, city, panel_max_w)

    n = max(len(summary), len(city_lines), len(info))
    summary_lo = max(0, (n - len(summary)) // 2)
    info_lo = max(0, (n - len(info)) // 2)

    lines: list[str] = []
    for i in range(n):
        si = i - summary_lo
        lt = summary[si] if 0 <= si < len(summary) else ""
        lpad = max(0, _SUMMARY_W - _visible_len(lt))

        ct = city_lines[i] if i < len(city_lines) else ""
        cpad = max(0, city_w - _visible_len(ct))

        ri = i - info_lo
        rt = info[ri] if 0 <= ri < len(info) else ""

        lines.append(lt + " " * lpad + " " * _GAP + ct + " " * cpad + " " * _GAP + rt)

    lines.append("")
    lines.append("  \033[90mWASD/Arrows: move · Q/ESC: quit\033[0m")

    while len(lines) < term.lines - 1:
        lines.append("")

    output = "\033[H"
    for line in lines[: term.lines - 1]:
        output += line + "\033[K\r\n"
    sys.stdout.write(output)
    sys.stdout.flush()


def _build_info_panel(tile: TileInfo | None, city: CityData, max_w: int) -> list[str]:
    reset = "\033[0m"
    gray = "\033[90m"
    bold = "\033[1m"
    indent = "  "
    content_w = max_w - len(indent)

    lines: list[str] = [""]
    lines.append(f"{indent}{bold}City Explorer{reset}")
    lines.append(indent + "─" * min(22, content_w))
    lines.append("")

    if tile is None:
        lines.append(f"{indent}{gray}Move to a tile")
        lines.append(f"{indent}to see details.{reset}")
        return lines

    if tile.repo is None:
        lines.append(f"{indent}{gray}+{city.extra_count} more repos{reset}")
        lines.append(f"{indent}{gray}{format_size(city.extra_storage, human_readable=True)} combined{reset}")
        return lines

    repo = tile.repo
    name = repo.id
    if len(name) > content_w:
        name = name[: content_w - 3] + "..."
    lines.append(f"{indent}{bold}{name}{reset}")
    lines.append("")

    type_ansi = {
        "model": "\033[38;2;175;148;240m",
        "dataset": "\033[38;2;245;128;128m",
        "space": "\033[38;2;245;175;85m",
        "bucket": "\033[38;2;112;185;242m",
    }
    tc = type_ansi.get(repo.type, "")

    lines.append(f"{indent}Type       {tc}{repo.type}{reset}")
    lines.append(f"{indent}Visibility {repo.visibility}")
    lines.append(f"{indent}Storage    {format_size(repo.storage, human_readable=True)}")
    lines.append(f"{indent}Usage      {repo.storage_percent:.1f}%")
    lines.append("")

    bar_w = min(18, content_w)
    filled = max(0, min(bar_w, round(repo.storage_percent / 100 * bar_w)))
    lines.append(f"{indent}{tc}{'█' * filled}{gray}{'░' * (bar_w - filled)}{reset}")

    return lines


def _copy_buf(buf: list[list[Color | None]]) -> list[list[Color | None]]:
    return [row[:] for row in buf]


def _read_key() -> str:
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        if _has_input(fd, 0.05):
            ch2 = os.read(fd, 1)
            if ch2 == b"[" and _has_input(fd, 0.05):
                ch3 = os.read(fd, 1)
                return f"\x1b[{ch3.decode()}"
        return "esc"
    return ch.decode("utf-8", errors="replace")


def _has_input(fd: int, timeout: float) -> bool:
    r, _, _ = select.select([fd], [], [], timeout)
    return bool(r)
<|endoftext|># Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to authenticate to the Hugging Face Hub and interact with your repositories."""

from typing import Annotated

import click

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import whoami

from .._login import _save_oauth_token, auth_list, auth_switch, login, logout
from ..errors import CLIError
from ..utils import get_stored_tokens, get_token, logging, select_choice
from ..utils._oauth_device import poll_device_token, request_device_code
from ._cli_utils import TokenOpt, typer_factory
from ._framework import Option
from ._output import OutputFormat, out


logger = logging.get_logger(__name__)


auth_cli = typer_factory(help="Manage authentication (login, logout, etc.).")


@auth_cli.command(
    "login",
    examples=[
        "hf auth login",
        "hf auth login --token $HF_TOKEN",
        "hf auth login --token $HF_TOKEN --add-to-git-credential",
        "hf auth login --force",
    ],
)
def auth_login(
    token: TokenOpt = None,
    add_to_git_credential: Annotated[
        bool,
        Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
    force: Annotated[
        bool,
        Option(
            help="Force re-login even if already logged in.",
        ),
    ] = False,
) -> None:
    """Login from your browser, or using a token from huggingface.co/settings/tokens."""
    if token is not None or out.mode == OutputFormat.human:
        # `--token` bypasses any prompt; in human mode the gh-style menu lives in `login()`.
        login(token=token, add_to_git_credential=add_to_git_credential, skip_if_logged_in=not force)
        return

    # Logging in is an interactive flow: besides human mode, only agent mode is supported.
    if out.mode != OutputFormat.agent:
        raise CLIError(
            "`hf auth login` is interactive and does not support --format json/quiet. "
            "Pass --token for a non-interactive login."
        )

    # agent mode: never prompt; print instructions the agent can relay to its user.
    if not force and get_token() is not None:
        out.text(agent="Already logged in. Use `hf auth login --force` to re-login.")
        return
    device_info = request_device_code()
    out.text(
        agent=(
            f"Ask the user to open {device_info['verification_uri_complete']} in a browser and enter the code "
            f"{device_info['user_code']}. The code expires in {device_info['expires_in']} seconds. "
            "Waiting for authorization..."
        )
    )
    response = poll_device_token(device_info)
    token_name, username = _save_oauth_token(response)
    out.text(agent=f"Login successful: logged in as {username} (token saved as '{token_name}').")


@auth_cli.command(
    "logout",
    examples=["hf auth logout", "hf auth logout --token-name my-token"],
)
def auth_logout(
    token_name: Annotated[
        str | None,
        Option(help="Name of token to logout"),
    ] = None,
) -> None:
    """Logout from a specific token."""
    logout(token_name=token_name)


def _select_token_name() -> str | None:
    token_names = list(get_stored_tokens().keys())

    if not token_names:
        logger.error("No stored tokens found. Please login first.")
        return None

    if out.mode != OutputFormat.human:
        raise CLIError("Use --token-name to select a token in non-interactive mode.")
    return token_names[select_choice("Select a token to switch to:", token_names)]


@auth_cli.command(
    "switch",
    examples=["hf auth switch", "hf auth switch --token-name my-token"],
)
def auth_switch_cmd(
    token_name: Annotated[
        str | None,
        Option(
            help="Name of the token to switch to",
        ),
    ] = None,
    add_to_git_credential: Annotated[
        bool,
        Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
) -> None:
    """Switch between access tokens."""
    if token_name is None:
        token_name = _select_token_name()
    if token_name is None:
        print("No token name provided. Aborting.")
        raise click.exceptions.Exit()
    auth_switch(token_name, add_to_git_credential=add_to_git_credential)


@auth_cli.command("list | ls", examples=["hf auth list"])
def auth_list_cmd() -> None:
    """List all stored access tokens."""
    auth_list()


@auth_cli.command("token", examples=["hf auth token", "hf auth token | xargs curl -H 'Authorization: Bearer {}'"])
def auth_token() -> None:
    """Print the current access token to stdout."""
    token = get_token()
    if token is None:
        out.error("Not logged in. Run `hf auth login` first.")
        raise click.exceptions.Exit(code=1)
    print(token)
    out.hint("Run `hf auth whoami` to see which account this token belongs to.")


@auth_cli.command("whoami", examples=["hf auth whoami", "hf auth whoami --format json"])
def auth_whoami() -> None:
    """Find out which huggingface.co account you are logged in as."""

    token = get_token()
    if token is None:
        out.error("Not logged in. Run `hf auth login` first.")
        raise click.exceptions.Exit(code=1)

    info = whoami(token)
    orgs = ",".join(org["name"] for org in info["orgs"]) or None
    endpoint = ENDPOINT if ENDPOINT != "https://huggingface.co" else None
    out.result("Logged in", user=info["name"], orgs=orgs, endpoint=endpoint)
<|endoftext|># Copyright 2025-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the 'hf cache' command group with cache management subcommands."""

import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

import click

from huggingface_hub.errors import CLIError

from ..utils import (
    ANSI,
    CachedFileInfo,
    CachedRepoInfo,
    CachedRevisionInfo,
    CacheNotFound,
    HFCacheInfo,
    HfUri,
    _format_size,
    _shared_blobs,
    parse_hf_uri,
    scan_cache_dir,
)
from ..utils._parsing import parse_duration, parse_size
from ._cli_utils import RepoIdArg, RepoTypeOpt, RevisionOpt, TokenOpt, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import out


cache_cli = typer_factory(help="Manage local cache directory.")


#### Cache helper utilities


@dataclass(frozen=True)
class _DeletionResolution:
    revisions: frozenset[str]
    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]]
    missing: tuple[str, ...]
    files: dict[CachedFileInfo, str] = field(default_factory=dict)  # selected files, mapped to their display label


_FILTER_PATTERN = re.compile(r"(?P<key>[a-zA-Z_]+)\s*(?P<op>==|!=|>=|<=|>|<|=)\s*(?P<value>.+)")
_ALLOWED_OPERATORS = {"=", "!=", ">", "<", ">=", "<="}
_FILTER_KEYS = {"accessed", "modified", "refs", "size", "type"}
_SORT_KEYS = {"accessed", "modified", "name", "size"}
_SORT_PATTERN = re.compile(r"(?P<key>[a-zA-Z_]+)(?::(?P<order>asc|desc))?")
_SORT_DEFAULT_ORDER = {
    # Default ordering: accessed/modified/size are descending (newest/biggest first), name is ascending
    "accessed": "desc",
    "modified": "desc",
    "size": "desc",
    "name": "asc",
}


# Dynamically generate SortOptions enum from _SORT_KEYS
_sort_options_dict = {}
for key in sorted(_SORT_KEYS):
    _sort_options_dict[key] = key
    _sort_options_dict[f"{key}_asc"] = f"{key}:asc"
    _sort_options_dict[f"{key}_desc"] = f"{key}:desc"

SortOptions = Enum("SortOptions", _sort_options_dict, type=str, module=__name__)  # type: ignore


@dataclass(frozen=True)
class CacheDeletionCounts:
    """Simple counters summarizing cache deletions for CLI messaging."""

    repo_count: int
    partial_revision_count: int
    total_revision_count: int


CacheEntry = tuple[CachedRepoInfo, CachedRevisionInfo | None]
RepoRefsMap = dict[CachedRepoInfo, frozenset[str]]


def summarize_deletions(
    selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]],
) -> CacheDeletionCounts:
    """Summarize deletions across repositories."""
    repo_count = 0
    total_revisions = 0
    revisions_in_full_repos = 0

    for repo, revisions in selected_by_repo.items():
        total_revisions += len(revisions)
        if len(revisions) == len(repo.revisions):
            repo_count += 1
            revisions_in_full_repos += len(revisions)

    partial_revision_count = total_revisions - revisions_in_full_repos
    return CacheDeletionCounts(repo_count, partial_revision_count, total_revisions)


def _prune_summary(revision_count: int, incomplete_count: int, shared_blob_count: int) -> str:
    """Build the human-readable summary of what `hf cache prune` is about to delete."""
    parts: list[str] = []
    if revision_count:
        parts.append(f"{revision_count} unreferenced revision(s)")
    if incomplete_count:
        parts.append(f"{incomplete_count} incomplete download(s)")
    if shared_blob_count:
        parts.append(f"{shared_blob_count} unreferenced shared blob(s)")
    if len(parts) > 1:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return "".join(parts)


def print_cache_selected_revisions(selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]]) -> None:
    """Pretty-print selected cache revisions during confirmation prompts."""
    for repo in sorted(selected_by_repo.keys(), key=lambda repo: (repo.repo_type, repo.repo_id.lower())):
        repo_key = f"{repo.repo_type}/{repo.repo_id}"
        revisions = sorted(selected_by_repo[repo], key=lambda rev: rev.commit_hash)
        if len(revisions) == len(repo.revisions):
            out.text(f"  - {repo_key} (entire repo)")
            continue

        out.text(f"  - {repo_key}:")
        for revision in revisions:
            refs = " ".join(sorted(revision.refs)) or "(detached)"
            out.text(f"      {revision.commit_hash} [{refs}] {revision.size_on_disk_str}")


def build_cache_index(
    hf_cache_info: HFCacheInfo,
) -> tuple[
    dict[str, CachedRepoInfo],
    dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]],
]:
    """Create lookup tables so CLI commands can resolve repo ids and revisions quickly."""
    repo_lookup: dict[str, CachedRepoInfo] = {}
    revision_lookup: dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]] = {}
    for repo in hf_cache_info.repos:
        repo_key = repo.cache_id.lower()
        repo_lookup[repo_key] = repo
        for revision in repo.revisions:
            revision_lookup[revision.commit_hash.lower()] = (repo, revision)
    return repo_lookup, revision_lookup


def _parse_hf_uri_target(target: str) -> HfUri:
    """Parse and validate an hf:// target passed to `hf cache rm`."""
    uri = parse_hf_uri(target)
    if not uri.is_repo:
        raise CLIError("Only repository hf:// URIs are supported by `hf cache rm`.")
    if uri.revision is not None:
        hint = (
            "Drop '@<revision>' to delete the file from all cached revisions."
            if uri.path_in_repo
            else "Pass the revision hash instead, see 'hf cache ls --revisions'."
        )
        raise CLIError(f"Revisions in hf:// URIs are not supported by `hf cache rm`. {hint}")
    return uri


def _cache_id(uri: HfUri) -> str:
    """Return the `type/id` cache id of a repo URI."""
    return f"{uri.type}/{uri.id}"


def collect_cache_entries(
    hf_cache_info: HFCacheInfo, *, include_revisions: bool
) -> tuple[list[CacheEntry], RepoRefsMap]:
    """Flatten cache metadata into rows consumed by `hf cache ls`."""
    entries: list[CacheEntry] = []
    repo_refs_map: RepoRefsMap = {}
    sorted_repos = sorted(hf_cache_info.repos, key=lambda repo: (repo.repo_type, repo.repo_id.lower()))
    for repo in sorted_repos:
        repo_refs_map[repo] = frozenset({ref for revision in repo.revisions for ref in revision.refs})
        if include_revisions:
            for revision in sorted(repo.revisions, key=lambda rev: rev.commit_hash):
                entries.append((repo, revision))
        else:
            entries.append((repo, None))
    if include_revisions:
        entries.sort(
            key=lambda entry: (
                entry[0].cache_id,
                entry[1].commit_hash if entry[1] is not None else "",
            )
        )
    else:
        entries.sort(key=lambda entry: entry[0].cache_id)
    return entries, repo_refs_map


def compile_cache_filter(
    expr: str, repo_refs_map: RepoRefsMap
) -> Callable[[CachedRepoInfo, CachedRevisionInfo | None, float], bool]:
    """Convert a `hf cache ls` filter expression into the yes/no test we apply to each cache entry before displaying it."""
    match = _FILTER_PATTERN.fullmatch(expr.strip())
    if not match:
        raise ValueError(f"Invalid filter expression: '{expr}'.")

    key = match.group("key").lower()
    op = match.group("op")
    value_raw = match.group("value").strip()

    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Unsupported operator '{op}' in filter '{expr}'. Must be one of {list(_ALLOWED_OPERATORS)}.")

    if key not in _FILTER_KEYS:
        raise ValueError(f"Unsupported filter key '{key}' in '{expr}'. Must be one of {list(_FILTER_KEYS)}.")
    # at this point we know that key is in `_FILTER_KEYS`
    if key == "size":
        size_threshold = parse_size(value_raw)
        return lambda repo, revision, _: _compare_numeric(
            revision.size_on_disk if revision is not None else repo.size_on_disk,
            op,
            size_threshold,
        )

    if key in {"modified", "accessed"}:
        seconds = parse_duration(value_raw.strip())

        def _time_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, now: float) -> bool:
            timestamp = (
                repo.last_accessed
                if key == "accessed"
                else revision.last_modified
                if revision is not None
                else repo.last_modified
            )
            if timestamp is None:
                return False
            return _compare_numeric(now - timestamp, op, seconds)

        return _time_filter

    if key == "type":
        expected = value_raw.lower()

        if op != "=":
            raise ValueError(f"Only '=' is supported for 'type' filters. Got '{op}'.")

        def _type_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            return repo.repo_type.lower() == expected

        return _type_filter

    else:  # key == "refs"
        if op != "=":
            raise ValueError(f"Only '=' is supported for 'refs' filters. Got {op}.")

        def _refs_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            refs = revision.refs if revision is not None else repo_refs_map.get(repo, frozenset())
            return value_raw.lower() in [ref.lower() for ref in refs]

        return _refs_filter


def _compare_numeric(left: float | None, op: str, right: float) -> bool:
    """Evaluate numeric comparisons for filters."""
    if left is None:
        return False

    comparisons = {
        "=": left == right,
        "!=": left != right,
        ">": left > right,
        "<": left < right,
        ">=": left >= right,
        "<=": left <= right,
    }

    if op not in comparisons:
        raise ValueError(f"Unsupported numeric comparison operator: {op}")

    return comparisons[op]


def compile_cache_sort(sort_expr: str) -> tuple[Callable[[CacheEntry], tuple[Any, ...]], bool]:
    """Convert a `hf cache ls` sort expression into a key function for sorting entries.

    Returns:
        A tuple of (key_function, reverse_flag) where reverse_flag indicates whether
        to sort in descending order (True) or ascending order (False).
    """
    match = _SORT_PATTERN.fullmatch(sort_expr.strip().lower())
    if not match:
        raise ValueError(f"Invalid sort expression: '{sort_expr}'. Expected format: 'key' or 'key:asc' or 'key:desc'.")

    key = match.group("key").lower()
    explicit_order = match.group("order")

    if key not in _SORT_KEYS:
        raise ValueError(f"Unsupported sort key '{key}' in '{sort_expr}'. Must be one of {list(_SORT_KEYS)}.")

    # Use explicit order if provided, otherwise use default for the key
    order = explicit_order if explicit_order else _SORT_DEFAULT_ORDER[key]
    reverse = order == "desc"

    def _sort_key(entry: CacheEntry) -> tuple[Any, ...]:
        repo, revision = entry

        if key == "name":
            # Sort by cache_id (repo type/id)
            value: Any = repo.cache_id.lower()
            return (value,)

        if key == "size":
            # Use revision size if available, otherwise repo size
            value = revision.size_on_disk if revision is not None else repo.size_on_disk
            return (value,)

        if key == "accessed":
            # For revisions, accessed is not available per-revision, use repo's last_accessed
            # For repos, use repo's last_accessed
            value = repo.last_accessed if repo.last_accessed is not None else 0.0
            return (value,)

        if key == "modified":
            # Use revision's last_modified if available, otherwise repo's last_modified
            if revision is not None:
                value = revision.last_modified if revision.last_modified is not None else 0.0
            else:
                value = repo.last_modified if repo.last_modified is not None else 0.0
            return (value,)

        # Should never reach here due to validation above
        raise ValueError(f"Unsupported sort key: {key}")

    return _sort_key, reverse


def _resolve_deletion_targets(hf_cache_info: HFCacheInfo, targets: list[str]) -> _DeletionResolution:
    """Resolve the deletion targets into a deletion resolution."""
    targets = [stripped for target in targets if (stripped := target.strip())]
    uris = {target: _parse_hf_uri_target(target) for target in targets if target.startswith("hf://")}
    file_uris = {target: uri for target, uri in uris.items() if uri.path_in_repo}
    if file_uris:
        if len(file_uris) != len(set(targets)):
            raise CLIError("File targets cannot be mixed with repository or revision targets.")
        return _resolve_file_targets(hf_cache_info, file_uris)

    repo_lookup, revision_lookup = build_cache_index(hf_cache_info)

    selected: dict[CachedRepoInfo, set[CachedRevisionInfo]] = defaultdict(set)
    revisions: set[str] = set()
    missing: list[str] = []

    for target in targets:
        lowered = target.lower()

        if re.fullmatch(r"[0-9a-fA-F]{40}", lowered):
            match = revision_lookup.get(lowered)
            if match is None:
                missing.append(target)
                continue
            repo, revision = match
            selected[repo].add(revision)
            revisions.add(revision.commit_hash)
            continue

        cache_id = _cache_id(uris[target]) if target in uris else target
        matched_repo = repo_lookup.get(cache_id.lower())
        if matched_repo is None:
            missing.append(target)
            continue

        for revision in matched_repo.revisions:
            selected[matched_repo].add(revision)
            revisions.add(revision.commit_hash)

    frozen_selected = {repo: frozenset(revs) for repo, revs in selected.items()}
    return _DeletionResolution(
        revisions=frozenset(revisions),
        selected=frozen_selected,
        missing=tuple(missing),
    )


def _resolve_file_targets(hf_cache_info: HFCacheInfo, file_uris: dict[str, HfUri]) -> _DeletionResolution:
    """Select the cached files matching hf:// file URIs, in every cached revision of their repo."""
    repo_lookup, _ = build_cache_index(hf_cache_info)
    files: dict[CachedFileInfo, str] = {}
    missing: list[str] = []
    for target, uri in file_uris.items():
        repo = repo_lookup.get(_cache_id(uri).lower())
        if repo is None:
            missing.append(target)
            continue
        matches = {
            file: f"{repo.cache_id}@{revision.commit_hash}/{uri.path_in_repo}"
            for revision in repo.revisions
            for file in revision.files
            if file.file_path.relative_to(revision.snapshot_path).as_posix() == uri.path_in_repo
        }
        if not matches:
            missing.append(target)
        files.update(matches)
    return _DeletionResolution(revisions=frozenset(), selected={}, missing=tuple(missing), files=files)


#### Cache CLI commands


@cache_cli.command(
    "list | ls",
    examples=[
        "hf cache ls",
        "hf cache ls --revisions",
        'hf cache ls --filter "size>1GB" --limit 20',
        "hf cache ls --format json",
    ],
)
def ls(
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    revisions: Annotated[
        bool,
        Option(
            help="Include revisions in the output instead of aggregated repositories.",
        ),
    ] = False,
    filter: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter",
            help="Filter entries (e.g. 'size>1GB', 'type=model', 'accessed>7d'). Can be used multiple times.",
        ),
    ] = None,
    sort: Annotated[
        SortOptions | None,
        Option(
            help="Sort entries by key. Supported keys: 'accessed', 'modified', 'name', 'size'. "
            "Append ':asc' or ':desc' to explicitly set the order (e.g., 'modified:asc'). "
            "Defaults: 'accessed', 'modified', 'size' default to 'desc' (newest/biggest first); "
            "'name' defaults to 'asc' (alphabetical).",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Option(
            help="Limit the number of results returned. Returns only the top N entries after sorting.",
        ),
    ] = None,
    show_warnings: Annotated[
        bool,
        Option(
            help="Show warnings about cache inconsistencies.",
        ),
    ] = False,
) -> None:
    """List cached repositories or revisions."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    filters = filter or []

    entries, repo_refs_map = collect_cache_entries(hf_cache_info, include_revisions=revisions)
    try:
        filter_fns = [compile_cache_filter(expr, repo_refs_map) for expr in filters]
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    now = time.time()
    for fn in filter_fns:
        entries = [entry for entry in entries if fn(entry[0], entry[1], now)]

    # Apply sorting if requested
    if sort:
        try:
            sort_key_fn, reverse = compile_cache_sort(sort.value)
            entries.sort(key=sort_key_fn, reverse=reverse)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc

    # Apply limit if requested
    if limit is not None:
        if limit < 0:
            raise click.BadParameter(f"Limit must be a positive integer, got {limit}.")
        entries = entries[:limit]

    if revisions:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "revision": revision.commit_hash,
                "snapshot_path": str(revision.snapshot_path),
                "size": revision.size_on_disk_str,
                "last_modified": revision.last_modified_str,
                "refs": sorted(revision.refs),
            }
            for repo, revision in entries
            if revision is not None
        ]
        out.table(
            items,
            headers=["id", "revision", "size", "last_modified", "refs"],
            id_key="revision",
            alignments={"size": "right"},
        )
    else:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size": repo.size_on_disk_str,
                "last_accessed": repo.last_accessed_str or "",
                "last_modified": repo.last_modified_str,
                "refs": sorted(repo_refs_map.get(repo, frozenset())),
            }
            for repo, _ in entries
        ]
        out.table(
            items,
            headers=["id", "size", "last_accessed", "last_modified", "refs"],
            id_key="id",
            alignments={"size": "right"},
        )

    if entries:
        unique_repos = {repo for repo, _ in entries}
        repo_count = len(unique_repos)
        if revisions:
            revision_count = sum(1 for _, rev in entries if rev is not None)
            total_size = sum(rev.size_on_disk for _, rev in entries if rev is not None)
        else:
            revision_count = sum(len(repo.revisions) for repo in unique_repos)
            if filters or limit is not None:
                total_size = sum(repo.size_on_disk for repo in unique_repos)
            else:
                # Per-repo sizes double-count blobs shared across repos: report the physical total.
                total_size = hf_cache_info.size_on_disk
        out.text(
            ANSI.bold(
                f"\nFound {repo_count} repo(s) for a total of {revision_count} revision(s)"
                f" and {_format_size(total_size)} on disk."
            )
        )

    if len(hf_cache_info.warnings):
        if show_warnings:
            for warning in hf_cache_info.warnings:
                out.warning(str(warning).rstrip(".") + ". Repo ignored.")
        else:
            out.warning(
                f"Found {len(hf_cache_info.warnings)} cache inconsistencies. Re-run with `--show-warnings` to display them."
            )

    incomplete_files = hf_cache_info.incomplete_files
    if incomplete_files:
        out.hint(
            f"Found {len(incomplete_files)} incomplete download(s) totalling "
            f"{_format_size(hf_cache_info.incomplete_size_on_disk)}. "
            "Remove them with 'hf cache prune'."
        )


@cache_cli.command(
    examples=[
        "hf cache rm model/gpt2",
        "hf cache rm hf://models/openai-community/gpt2",
        "hf cache rm hf://models/openai-community/gpt2/model.safetensors",
        "hf cache rm <revision_hash>",
        "hf cache rm model/gpt2 --dry-run",
        "hf cache rm model/gpt2 --yes",
    ],
)
def rm(
    targets: Annotated[
        list[str],
        Argument(
            help="One or more repo IDs (e.g. model/bert-base-uncased), hf:// URIs (repo or file), or revision hashes to delete.",
        ),
    ],
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove cached repositories, revisions or files."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    resolution = _resolve_deletion_targets(hf_cache_info, targets)

    if resolution.missing:
        details = "\n".join(f"  - {entry}" for entry in resolution.missing)
        out.warning(f"Could not find in cache:\n{details}")

    if resolution.files:
        strategy = hf_cache_info.delete_files(*resolution.files)
        out.text(f"About to delete {len(resolution.files)} file(s) totalling {strategy.expected_freed_size_str}.")
        for label in sorted(resolution.files.values()):
            out.text(f"  - {label}")
        if dry_run:
            out.result(
                "Dry run: no files were deleted.",
                dry_run=True,
                files=len(resolution.files),
                size=strategy.expected_freed_size_str,
            )
            return
        out.confirm("Proceed with deletion?", yes=yes)
        strategy.execute()
        out.result(
            f"Deleted {len(resolution.files)} file(s); freed {strategy.expected_freed_size_str}.",
            files_deleted=len(resolution.files),
            freed=strategy.expected_freed_size_str,
        )
        return

    if len(resolution.revisions) == 0:
        out.text("Nothing to delete.")
        raise click.exceptions.Exit(code=0)

    strategy = hf_cache_info.delete_revisions(*sorted(resolution.revisions))
    counts = summarize_deletions(resolution.selected)

    summary_parts: list[str] = []
    if counts.repo_count:
        summary_parts.append(f"{counts.repo_count} repo(s)")
    if counts.partial_revision_count:
        summary_parts.append(f"{counts.partial_revision_count} revision(s)")
    if not summary_parts:
        summary_parts.append(f"{counts.total_revision_count} revision(s)")

    summary_text = " and ".join(summary_parts)
    out.text(f"About to delete {summary_text} totalling {strategy.expected_freed_size_str}.")
    print_cache_selected_revisions(resolution.selected)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            repos=counts.repo_count,
            revisions=counts.total_revision_count,
            size=strategy.expected_freed_size_str,
        )
        return

    out.confirm("Proceed with deletion?", yes=yes)

    strategy.execute()
    counts = summarize_deletions(resolution.selected)
    out.result(
        f"Deleted {counts.repo_count} repo(s) and {counts.total_revision_count} revision(s);"
        f" freed {strategy.expected_freed_size_str}.",
        repos_deleted=counts.repo_count,
        revisions_deleted=counts.total_revision_count,
        freed=strategy.expected_freed_size_str,
    )


@cache_cli.command(examples=["hf cache prune", "hf cache prune --dry-run"])
def prune(
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to Hugging Face cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove detached revisions and incomplete downloads from the cache."""
    try:
        hf_cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]] = {}
    revisions: set[str] = set()
    for repo in hf_cache_info.repos:
        detached = frozenset(revision for revision in repo.revisions if len(revision.refs) == 0)
        if not detached:
            continue
        selected[repo] = detached
        revisions.update(revision.commit_hash for revision in detached)

    incomplete_files = hf_cache_info.incomplete_files
    # Shared blobs left behind when a repo folder is deleted manually or by an older client.
    unreferenced_blobs = (
        _shared_blobs.unreferenced_shared_blobs(hf_cache_info.cache_dir) if hf_cache_info.cache_dir is not None else {}
    )

    if len(revisions) == 0 and not incomplete_files and not unreferenced_blobs:
        out.text("No unreferenced revisions, incomplete downloads or shared blobs found. Nothing to prune.")
        return

    strategy = hf_cache_info.delete_revisions(*sorted(revisions))
    counts = summarize_deletions(selected)
    total_freed = (
        strategy.expected_freed_size + hf_cache_info.incomplete_size_on_disk + sum(unreferenced_blobs.values())
    )

    summary = _prune_summary(counts.total_revision_count, len(incomplete_files), len(unreferenced_blobs))
    out.text(f"About to delete {summary} ({_format_size(total_freed)} total).")
    print_cache_selected_revisions(selected)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            revisions=counts.total_revision_count,
            incomplete=len(incomplete_files),  # might be overstated but it's fine
            shared_blobs=len(unreferenced_blobs),
            size=_format_size(total_freed),
        )
        return

    out.confirm("Proceed?", yes=yes)

    strategy.execute()
    for incomplete_file in incomplete_files:
        try:
            incomplete_file.file_path.unlink()
        except FileNotFoundError:
            pass  # already removed (e.g. by a full-repo deletion above)
        except OSError as exc:
            out.warning(f"Could not delete incomplete file {incomplete_file.file_path}: {exc}")
    if hf_cache_info.cache_dir is not None:
        for store_path in unreferenced_blobs:
            _shared_blobs.sweep_shared_blob(store_path, cache_dir=hf_cache_info.cache_dir)
    out.result(
        f"Deleted {summary}; freed {_format_size(total_freed)}.",
        revisions_deleted=counts.total_revision_count,
        incomplete_deleted=len(incomplete_files),
        shared_blobs_deleted=len(unreferenced_blobs),
        freed=_format_size(total_freed),
    )


@cache_cli.command(
    examples=[
        "hf cache verify gpt2",
        "hf cache verify gpt2 --revision refs/pr/1",
        "hf cache verify my-dataset --repo-type dataset",
    ],
)
def verify(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoTypeOpt.model,
    revision: RevisionOpt = None,
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to use when verifying files from cache (defaults to Hugging Face cache).",
        ),
    ] = None,
    local_dir: Annotated[
        str | None,
        Option(
            help="If set, verify files under this directory instead of the cache.",
        ),
    ] = None,
    fail_on_missing_files: Annotated[
        bool,
        Option(
            "--fail-on-missing-files",
            help="Fail if some files exist on the remote but are missing locally.",
        ),
    ] = False,
    fail_on_extra_files: Annotated[
        bool,
        Option(
            "--fail-on-extra-files",
            help="Fail if some files exist locally but are not present on the remote revision.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Verify checksums for a single repo revision from cache or a local directory.

    Examples:
      - Verify main revision in cache: `hf cache verify gpt2`
      - Verify specific revision: `hf cache verify gpt2 --revision refs/pr/1`
      - Verify dataset: `hf cache verify karpathy/fineweb-edu-100b-shuffle --repo-type dataset`
      - Verify local dir: `hf cache verify deepseek-ai/DeepSeek-OCR --local-dir /path/to/repo`
    """

    if local_dir is not None and cache_dir is not None:
        out.error("Cannot pass both --local-dir and --cache-dir. Use one or the other.")
        raise click.exceptions.Exit(code=2)

    api = get_hf_api(token=token)

    result = api.verify_repo_checksums(
        repo_id=repo_id,
        repo_type=repo_type.value if hasattr(repo_type, "value") else str(repo_type),
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=token,
    )

    exit_code = 0

    if result.mismatches:
        details = "\n".join(
            f"  - {m['path']}: expected {m['expected']} ({m['algorithm']}), got {m['actual']}"
            for m in result.mismatches
        )
        out.text(f"❌ Checksum verification failed for the following file(s):\n{details}")
        exit_code = 1

    if result.missing_paths:
        if fail_on_missing_files:
            details = "\n".join(f"  - {p}" for p in result.missing_paths)
            out.text(f"❌ Missing files (present remotely, absent locally):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(result.missing_paths)} remote file(s) are missing locally. "
                "Use --fail-on-missing-files for details."
            )

    if result.extra_paths:
        if fail_on_extra_files:
            details = "\n".join(f"  - {p}" for p in result.extra_paths)
            out.text(f"❌ Extra files (present locally, absent remotely):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(result.extra_paths)} local file(s) do not exist on the remote repo. "
                "Use --fail-on-extra-files for details."
            )

    verified_location = result.verified_path

    if exit_code != 0:
        out.error(
            f"Verification failed for '{repo_id}' ({repo_type.value}) in {verified_location}.\n  Revision: {result.revision}"
        )
        raise click.exceptions.Exit(code=exit_code)

    out.result(
        f"Verified {result.checked_count} file(s) for {repo_type.value} '{repo_id}'. All checksums match.",
        repo_id=repo_id,
        repo_type=repo_type.value,
        checked=result.checked_count,
        path=str(verified_location),
    )
<|endoftext|># Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Output framework for the `hf` CLI."""

import dataclasses
import datetime
import json
import re
import shutil
import sys
from collections.abc import Sequence
from enum import Enum
from functools import cache
from typing import Any, cast

import click

from huggingface_hub import constants
from huggingface_hub.errors import ConfirmationError
from huggingface_hub.repocard_data import CardData
from huggingface_hub.utils import ANSI, StatusLine, disable_progress_bars, enable_progress_bars, is_agent, tabulate


class OutputFormat(str, Enum):
    """Output format for CLI commands with auto detection of agent/human mode."""

    agent = "agent"
    auto = "auto"
    human = "human"
    json = "json"
    quiet = "quiet"


def _print_flush(*values: Any, **kwargs: Any) -> None:
    """Like `print`, but always flushed: some CLI flows block on user action right after
    printing (e.g. the device-code login), so output must not stay buffered."""
    print(*values, **kwargs, flush=True)


class Output:
    """Output sink for the `hf` CLI.

    Mode is resolved once at init time based on `is_agent()` auto-detection
    and can be overridden per-command via `set_mode()`.
    """

    mode: OutputFormat
    no_truncate: bool

    def __init__(self) -> None:
        self.no_truncate = False
        self.set_mode()

    def set_mode(self, mode: OutputFormat = OutputFormat.auto) -> None:
        """Override the output mode (called once at startup and again per '--format' flag)."""
        if mode == OutputFormat.auto:
            mode = OutputFormat.agent if is_agent() else OutputFormat.human
        self.mode = mode
        is_human = mode == OutputFormat.human
        ANSI.set_enabled(is_human)
        if constants.HF_HUB_DISABLE_PROGRESS_BARS is None:  # env var has priority
            if is_human:
                enable_progress_bars()
            else:
                disable_progress_bars()

    def set_no_truncate(self, no_truncate: bool) -> None:
        """Toggle off cell truncation for human table output."""
        self.no_truncate = no_truncate

    def is_quiet(self) -> bool:
        return self.mode == OutputFormat.quiet

    def text(self, msg: str | None = None, *, human: str | None = None, agent: str | None = None) -> None:
        """Print a free-form text message to stdout."""
        if msg is not None:
            if human is not None or agent is not None:
                raise ValueError("Cannot mix 'msg' with 'human'/'agent'.")
            human = msg
            agent = _strip_ansi(msg)

        match self.mode:
            case OutputFormat.human:
                if human is not None:
                    _print_flush(human)
            case OutputFormat.agent:
                if agent is not None:
                    _print_flush(agent)
            # json/quiet: no-op

    def table(
        self,
        items: Sequence[dict[str, Any]],
        *,
        headers: list[str] | None = None,
        id_key: str | None = None,
        alignments: dict[str, str] | None = None,
    ) -> None:
        """Print tabular data to stdout.

        Args:
            items: List of dicts. Headers are auto-detected from keys if not provided.
            headers: Explicit column names. If None, derived from dict keys (all-None columns filtered).
            id_key: Key to print in quiet mode. If None, uses the first header.
            alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
        """
        if not items:
            match self.mode:
                case OutputFormat.agent | OutputFormat.human:
                    _print_flush("No results found.")
                case OutputFormat.json:
                    _print_flush("[]")
            return

        if headers is None:
            all_columns = list(items[0].keys())
            headers = [col for col in all_columns if any(item.get(col) is not None for item in items)]
        rows = [[item.get(h) for h in headers] for item in items]

        match self.mode:
            case OutputFormat.human:  # padded table, adaptive truncation, SCREAMING_SNAKE headers
                screaming_headers = [_to_header(h) for h in headers]
                formatted_rows: list[list[str]] = [[_format_table_value_human(v) for v in row] for row in rows]

                is_truncated = _truncate_columns(screaming_headers, formatted_rows, no_truncate=self.no_truncate)

                inferred = {**_infer_alignments(headers, rows), **(alignments or {})}
                screaming_alignments = {_to_header(k): v for k, v in inferred.items()}
                _print_flush(
                    tabulate(
                        cast("list[list[str | int]]", formatted_rows),
                        headers=screaming_headers,
                        alignments=screaming_alignments,
                    ),
                )
                if is_truncated:
                    self.hint("Use `--no-truncate` or `--format json` to display full values.")
            case OutputFormat.agent:  # TSV, no truncation, full timestamps
                _print_flush("\t".join(headers))
                for row in rows:
                    _print_flush("\t".join(_format_table_cell_agent(v) for v in row))
            case OutputFormat.json:  # compact JSON array
                _print_flush(json.dumps(list(items), default=str))
            case OutputFormat.quiet:  # id_key column (or first column), one per line
                quiet_key = id_key or headers[0]
                for item in items:
                    _print_flush(_escape_control_chars(str(item.get(quiet_key, ""))))

    def dict(self, data: Any, *, id_key: str | None = None) -> None:
        """Print structured data as JSON in all modes (indented for human, compact otherwise).

        Accepts a dict or a dataclass.
        """
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            data = _dataclass_to_dict(data)
        if self.mode == OutputFormat.quiet and id_key is not None:
            _print_flush(data.get(id_key, ""))
            return
        indent = 2 if self.mode == OutputFormat.human else None
        _print_flush(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Print a success summary to stdout."""
        match self.mode:
            case OutputFormat.human:  # ✓ message + key: value lines
                parts = [ANSI.green(f"{_ascii_safe('✓', '[OK]')} {message}")]
                for k, v in data.items():
                    if v is not None:
                        parts.append(f"  {k}: {v}")
                _print_flush("\n".join(parts))
            case OutputFormat.agent:  # key=val pairs, space-separated
                parts = [f"{k}={v}" for k, v in data.items() if v is not None]
                _print_flush(" ".join(parts) if parts else message)
            case OutputFormat.json:  # json.dumps(data), message ignored
                _print_flush(json.dumps(data, default=str) if data else "")
            case OutputFormat.quiet:  # first value only
                values = list(data.values())
                if values:
                    _print_flush(values[0])

    def confirm(self, message: str, *, default: bool = False, yes: bool = False, confirm_param: str = "--yes") -> None:
        """
        Ask for confirmation. Raises `ConfirmationError` in non-human modes.
        """
        if yes:
            return
        if self.mode != OutputFormat.human:
            raise ConfirmationError(f"{message} Use {confirm_param} to skip confirmation.")
        click.confirm(message, default=default, abort=True)

    def status(self, message: str | None = None) -> StatusLine:
        """Return a status line that emits only in human mode (no-op otherwise)."""
        status = StatusLine(enabled=self.mode == OutputFormat.human)
        if message is not None:
            status.update(message)
        return status

    def warning(self, message: str) -> None:
        """Print a non-fatal warning to stderr (all modes)."""
        if self.mode == OutputFormat.human:
            _print_flush(ANSI.yellow(f"Warning: {message}"), file=sys.stderr)
        else:
            _print_flush(f"Warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print an error to stderr (all modes)."""
        if self.mode == OutputFormat.human:
            _print_flush(ANSI.red(f"Error: {message}"), file=sys.stderr)
        else:
            _print_flush(f"Error: {message}", file=sys.stderr)

    def log(self, message: str) -> None:
        """Print a text message to stderr (human: gray, json/agent: plain text).

        Suppressed in quiet mode. Kept in json mode (like agent) since agents
        commonly run with ``--format json`` and the message goes to stderr so
        it never pollutes the parsed stdout.
        """
        if self.mode == OutputFormat.quiet:
            return
        if self.mode == OutputFormat.human:
            _print_flush(ANSI.gray(message), file=sys.stderr)
        else:
            _print_flush(message, file=sys.stderr)

    def hint(self, message: str) -> None:
        """Print a helpful hint to stderr (human: gray, json/agent: plain text).

        Suppressed in quiet mode. Kept in json mode (like agent) since agents
        commonly run with ``--format json`` and the next-command hints are useful
        there; hints go to stderr so they never pollute the parsed stdout.
        """
        self.log(f"Hint: {message}")


# HELPERS


@cache
def _ascii_safe(char: str, fallback: str) -> str:
    """Return `char`, or `fallback` if stdout cannot encode it.

    On Windows, `sys.stdout` uses the ANSI code page (e.g. cp1252) instead of the console
    encoding whenever it is not attached to a console (output redirected, piped, captured by
    a CI step). Printing a decoration like "✓" then raises `UnicodeEncodeError` - a
    `ValueError` subclass, so it surfaces as a confusing "Invalid value." error and aborts
    the command. Cached since the stream encoding is fixed for the lifetime of the process.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        char.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return fallback
    return char


def _serialize_value(v: object) -> object:
    """Recursively serialize a value to be JSON-compatible."""
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    elif isinstance(v, dict):
        return {key: _serialize_value(val) for key, val in v.items() if val is not None}
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def _dataclass_to_dict(info: Any) -> dict[str, Any]:
    """Convert a dataclass to a json-serializable dict."""
    data = dataclasses.asdict(info)

    for field in dataclasses.fields(info):
        value = getattr(info, field.name)
        if isinstance(value, CardData):
            data[field.name] = value.to_dict()

    return {k: _serialize_value(v) for k, v in data.items() if v is not None}


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _escape_control_chars(text: str) -> str:
    """Escape non-printable characters (CR, LF, ESC, ...) so Hub-provided strings cannot alter the terminal output."""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)


def _single_line(text: str) -> str:
    return _escape_control_chars(" ".join(text.split()))


def _to_header(name: str) -> str:
    """Convert a camelCase or PascalCase string to SCREAMING_SNAKE_CASE."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return s.upper()


def _infer_alignments(headers: list[str], rows: list[list[Any]]) -> dict[str, str]:
    """Return ``{"col": "right"}`` for columns where every non-None value is numeric."""
    result: dict[str, str] = {}
    for c, h in enumerate(headers):
        if all(row[c] is None or (isinstance(row[c], (int, float)) and not isinstance(row[c], bool)) for row in rows):
            result[h] = "right"
    return result


def _format_table_value_human(value: Any) -> str:
    """Convert a value to string for terminal display."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return _ascii_safe("✔", "yes") if value else ""
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str) and re.search(r"^\d{4}-\d{2}-\d{2}T", value):
        return value[:10]
    if isinstance(value, str):
        return _single_line(value)
    if isinstance(value, list):
        return ", ".join(_format_table_value_human(v) for v in value)
    elif isinstance(value, dict):
        if "name" in value:  # Likely to be a user or org => print name
            return _single_line(str(value["name"]))
        return _single_line(json.dumps(value))
    return _single_line(str(value))


def _truncate_columns(
    headers: list[str],
    rows: list[list[str]],
    *,
    no_truncate: bool,
) -> bool:
    """Truncate cells in-place to fit the current terminal width.

    Returns `True` if any cell was truncated, so the caller can emit a hint.
    `shutil.get_terminal_size` is cross-platform: it honors `$COLUMNS`, then
    queries the OS-native API, then falls back to `(80, 24)`.
    """
    if no_truncate or not rows:
        return False

    n = len(headers)
    # Per-column natural width: longest of header label and cell values.
    natural = [max(len(headers[c]), *(len(rows[r][c]) for r in range(len(rows)))) for c in range(n)]

    # `max(0, n - 1)` accounts for the single-space separator between columns.
    budget = shutil.get_terminal_size().columns - max(0, n - 1)
    if sum(natural) <= budget:
        return False

    # Shrink the widest column 1 char at a time. Floors keep the header label
    # visible; the `4` is the smallest cap that still shows "x..." (one content
    # char plus the "..." marker).
    caps = natural.copy()
    min_widths = [max(len(h), 4) for h in headers]
    while sum(caps) > budget:
        widest = max(
            (i for i, w in enumerate(caps) if w > min_widths[i]),
            key=lambda i: caps[i],
            default=-1,
        )
        if widest < 0:
            break  # everything at floor — table wraps slightly
        caps[widest] -= 1

    truncated = False
    for row in rows:
        for c, cell in enumerate(row):
            if len(cell) > caps[c]:
                truncated = True
                row[c] = cell[: caps[c] - 3] + "..."
    return truncated


def _format_table_cell_agent(value: Any) -> str:
    """Format a cell value for agent TSV output (ISO timestamps, tabs escaped)."""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return _single_line(str(value))


out = Output()
<|endoftext|># Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains helper utilities for hf CLI extensions."""

import errno
import json
import os
import re
import shutil
import subprocess
import venv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from html import unescape
from pathlib import Path
from typing import Annotated, Literal

import click
import httpx2

from huggingface_hub.errors import CLIError, CLIExtensionInstallError, ConfirmationError
from huggingface_hub.utils import get_session, logging

from ._cli_utils import typer_factory
from ._framework import Argument, Option
from ._output import out


DEFAULT_EXTENSION_OWNER = "huggingface"
EXTENSIONS_ROOT = Path("~/.local/share/hf/extensions")
MANIFEST_FILENAME = "manifest.json"
EXTENSIONS_HELP = (
    "Manage hf CLI extensions.\n\n"
    "Security Warning: extensions are third-party executables or Python packages. "
    "Install only from sources you trust."
)
extensions_cli = typer_factory(help=EXTENSIONS_HELP)
_EXTENSIONS_GITHUB_TOPIC = "hf-extension"
_EXTENSIONS_DOWNLOAD_TIMEOUT = 10
_EXTENSIONS_PIP_INSTALL_TIMEOUT = 300

logger = logging.get_logger(__name__)


class _GitHubRateLimitError(CLIError):
    """The GitHub API refused the request because a rate limit is exhausted.

    Distinguished from other `CLIError`s because it says nothing about the extension being handled:
    it is not worth retrying on the next one, and it must not be mistaken for an install failure.
    """


class _ExtensionUpdateStatus(str, Enum):
    UPDATED = "updated"
    UP_TO_DATE = "up_to_date"
    SKIPPED = "skipped"


@dataclass
class ExtensionManifest:
    owner: str
    repo: str
    repo_id: str
    short_name: str
    executable_path: str
    type: Literal["binary", "python"]
    installed_at: datetime
    description: str | None = None
    commit_sha: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ExtensionManifest":
        manifest_path = path / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise CLIError(f"Manifest file not found at {manifest_path}. Your extension may be corrupted.")
        data = json.loads(manifest_path.read_text())
        # Ignore keys not in the dataclass (e.g. fields dropped since the manifest was written).
        data = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        data["installed_at"] = datetime.fromisoformat(data["installed_at"])
        return ExtensionManifest(**data)

    def save(self, path: Path) -> None:
        manifest_path = path / MANIFEST_FILENAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["installed_at"] = self.installed_at.isoformat()
        manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))


@extensions_cli.command(
    "install",
    examples=[
        "hf extensions install hf-claude",
        "hf extensions install hanouticelina/hf-claude",
        "hf extensions install alvarobartt/hf-mem",
    ],
)
def extension_install(
    ctx: click.Context,
    repo_id: Annotated[
        str,
        Argument(help="GitHub extension repository in `[OWNER/]hf-<name>` format."),
    ],
    force: Annotated[bool, Option("--force", help="Overwrite if already installed.")] = False,
) -> None:
    """Install an extension from a public GitHub repository.

    Security warning: this installs a third-party executable or Python package.
    Install only from sources you trust.
    """
    owner, repo_name, short_name = _normalize_repo_id(repo_id)
    root_ctx = ctx.find_root()
    reserved_commands = set(getattr(root_ctx.command, "commands", {}).keys())
    if short_name in reserved_commands:
        raise CLIError(
            f"Cannot install extension '{short_name}' because it conflicts with an existing `hf {short_name}` command."
        )

    extension_dir = _get_extension_dir(short_name)
    if extension_dir.exists() and not force:
        raise CLIError(f"Extension '{short_name}' is already installed. Use --force to overwrite.")

    if not _github_repo_exists(owner=owner, repo_name=repo_name):
        raise CLIError(f"GitHub repository '{owner}/{repo_name}' not found. It must exist and be public.")
    if extension_dir.exists():
        # --force reinstall: only remove the previous install once the repo is known to exist.
        shutil.rmtree(extension_dir)
    manifest = _install_extension(owner=owner, repo_name=repo_name, short_name=short_name)
    ext_type = manifest.type.capitalize()
    out.result(
        f"{ext_type} extension installed",
        source=f"{owner}/{repo_name}",
        command=f"hf {short_name}",
    )
    out.hint(f"Run it with: hf {short_name}")


@extensions_cli.command(
    "update",
    examples=[
        "hf extensions update",
        "hf extensions update hf-claude",
        "hf extensions update alvarobartt/hf-mem",
    ],
)
def extension_update(
    name: Annotated[
        str | None,
        Argument(
            help=(
                "Extension to update (with or without `hf-` prefix, optionally as `OWNER/hf-<name>`). "
                "If omitted, all installed extensions are checked and the outdated ones are updated."
            ),
        ),
    ] = None,
) -> None:
    """Update installed extension(s) to their latest version."""
    if name is not None:
        manifest = _load_installed_extension_for_update(name)
        update_status = _update_installed_extension(manifest)
        match update_status:
            case _ExtensionUpdateStatus.UPDATED:
                out.result("Extension updated", name=manifest.short_name, source=manifest.repo_id)
            case _ExtensionUpdateStatus.UP_TO_DATE:
                out.result(f"Extension '{manifest.short_name}' is already up to date", name=manifest.short_name)
            case _ExtensionUpdateStatus.SKIPPED:
                pass  # warning already emitted by _update_installed_extension
        return

    manifests = _list_installed_extensions()
    if not manifests:
        out.warning("No extensions installed.")
        out.hint("Install one with: hf extensions install <repo_id>")
        return

    updated = []
    up_to_date = []
    for manifest in manifests:
        out.log(f"Checking '{manifest.short_name}' ({manifest.repo_id})...")
        try:
            update_status = _update_installed_extension(manifest)
        except _GitHubRateLimitError:
            # Every remaining extension would fail the same way, so stop rather than repeat it.
            raise
        except Exception as error:
            # Keep updating the other extensions even if one fails.
            out.warning(f"Could not update '{manifest.short_name}' ({manifest.repo_id}): {error}. Skipping.")
            continue
        match update_status:
            case _ExtensionUpdateStatus.UPDATED:
                updated.append(manifest.short_name)
            case _ExtensionUpdateStatus.UP_TO_DATE:
                up_to_date.append(manifest.short_name)
            case _ExtensionUpdateStatus.SKIPPED:
                pass  # warning already emitted by _update_installed_extension
    out.result(
        "Extensions update complete",
        updated=", ".join(updated) if updated else None,
        up_to_date=", ".join(up_to_date) if up_to_date else None,
    )


@extensions_cli.command(
    "exec",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False, "ignore_unknown_options": True},
    examples=[
        "hf extensions exec claude -- --help",
        "hf extensions exec claude --model zai-org/GLM-5",
    ],
)
def extension_exec(
    ctx: click.Context,
    name: Annotated[
        str,
        Argument(help="Extension name (with or without `hf-` prefix)."),
    ],
) -> None:
    """Execute an installed extension."""
    short_name = _normalize_extension_name(name)
    executable_path = _resolve_installed_executable_path(short_name)

    if not executable_path.is_file():
        raise CLIError(f"Extension '{short_name}' is not installed.")

    exit_code = _execute_extension_binary(executable_path=executable_path, args=list(ctx.args))
    raise click.exceptions.Exit(code=exit_code)


@extensions_cli.command("list | ls", examples=["hf extensions list"])
def extension_list() -> None:
    """List installed extension commands."""
    rows = [
        {
            "command": f"hf {manifest.short_name}",
            "source": str(manifest.repo_id),
            "type": str(manifest.type),
            "installed": manifest.installed_at.strftime("%Y-%m-%d"),
            "description": manifest.description,
        }
        for manifest in _list_installed_extensions()
    ]
    out.table(rows, id_key="command")


@extensions_cli.command("search", examples=["hf extensions search"])
def extension_search() -> None:
    """Search extensions available on GitHub (tagged with 'hf-extension' topic)."""
    response = _github_request(
        "GET",
        "https://api.github.com/search/repositories",
        params={"q": f"topic:{_EXTENSIONS_GITHUB_TOPIC}", "sort": "stars", "order": "desc", "per_page": 100},
    )
    data = response.json()

    installed = {m.short_name for m in _list_installed_extensions()}

    rows = []
    for repo in data.get("items", []):
        short_name = repo["name"].removeprefix("hf-")
        rows.append(
            {
                "name": short_name,
                "repo": repo["full_name"],
                "stars": repo.get("stargazers_count", 0),
                "description": repo.get("description") or "",
                "installed": "yes" if short_name in installed else "",
            }
        )

    out.table(rows, id_key="repo")


@extensions_cli.command("remove | rm", examples=["hf extensions remove claude"])
def extension_remove(
    name: Annotated[
        str,
        Argument(help="Extension name to remove (with or without `hf-` prefix)."),
    ],
) -> None:
    """Remove an installed extension."""
    short_name = _normalize_extension_name(name)
    extension_dir = _get_extension_dir(short_name)

    if not extension_dir.is_dir():
        raise CLIError(f"Extension '{short_name}' is not installed.")

    shutil.rmtree(extension_dir)
    out.result("Extension removed", name=short_name)


### HELPER FUNCTIONS


def _list_installed_extensions() -> list[ExtensionManifest]:
    """Return manifests for all validly-installed extensions, sorted by directory name."""
    root_dir = EXTENSIONS_ROOT.expanduser()
    if not root_dir.is_dir():
        return []
    manifests = []
    for extension_dir in sorted(root_dir.iterdir()):
        if not extension_dir.is_dir() or not extension_dir.name.startswith("hf-"):
            continue
        try:
            manifests.append(ExtensionManifest.load(extension_dir))
        except Exception as e:
            logger.debug(f"Failed to load manifest for extension '{extension_dir.name}': {e}")
            continue
    return manifests


def list_installed_extensions_for_help() -> list[tuple[str, str]]:
    entries = []
    for manifest in _list_installed_extensions():
        tag = f"[extension {manifest.repo_id}]"
        help_text = f"{manifest.description} {tag}" if manifest.description is not None else tag
        entries.append((manifest.short_name, help_text))
    return entries


def dispatch_unknown_top_level_extension(args: list[str], known_commands: set[str]) -> int | None:
    if not args:
        return None

    command_name = args[0]
    if command_name.startswith("-"):
        return None
    all_known = {a.strip() for cmd in known_commands for a in cmd.split("|")}
    if command_name in all_known:
        return None

    try:
        short_name = _validate_extension_short_name(command_name.removeprefix("hf-"), original_input=command_name)
    except CLIError:
        return None

    executable_path: Path | None
    try:
        executable_path = _resolve_installed_executable_path(short_name)
    except Exception:
        executable_path = _auto_install_official_extension(short_name)

    if executable_path is None or not executable_path.is_file():
        return None

    return _execute_extension_binary(executable_path=executable_path, args=list(args[1:]))


def _auto_install_official_extension(short_name: str) -> Path | None:
    """Try to auto-install huggingface/hf-<name>. Returns executable path or None."""
    owner, repo_name = DEFAULT_EXTENSION_OWNER, f"hf-{short_name}"
    if _get_extension_dir(short_name).exists():
        return None

    # Every unknown `hf <command>` reaches this point, typos included, so the existence check must
    # not spend REST API quota (see `_github_request`).
    try:
        if not _github_repo_exists(owner=owner, repo_name=repo_name):
            return None
    except Exception:
        return None

    try:
        out.confirm(f"'{short_name}' is an official Hugging Face extension ({owner}/{repo_name}). Install it?")
    except ConfirmationError:
        return None
    manifest = _install_extension(owner=owner, repo_name=repo_name, short_name=short_name)
    return Path(manifest.executable_path).expanduser()


def _load_installed_extension_for_update(name: str) -> ExtensionManifest:
    short_name = _normalize_extension_name(name)
    extension_dir = _get_extension_dir(short_name)
    if not extension_dir.is_dir():
        owner, _, _ = name.strip().rpartition("/")
        install_target = f"{owner}/hf-{short_name}" if owner else f"hf-{short_name}"
        raise CLIError(
            f"Extension '{short_name}' is not installed. Install it first with: hf extensions install {install_target}"
        )
    return ExtensionManifest.load(extension_dir)


def _update_installed_extension(manifest: ExtensionManifest) -> _ExtensionUpdateStatus:
    owner, repo_name, short_name = manifest.owner, manifest.repo, manifest.short_name

    try:
        latest_sha = _fetch_latest_commit_sha(owner=owner, repo_name=repo_name)
    except _GitHubRateLimitError:
        # Says nothing about this extension in particular, so let the caller stop the whole run.
        raise
    except Exception as error:
        out.warning(f"Could not check updates for '{short_name}' ({owner}/{repo_name}): {error}. Skipping.")
        return _ExtensionUpdateStatus.SKIPPED

    if latest_sha == manifest.commit_sha:
        return _ExtensionUpdateStatus.UP_TO_DATE

    _install_extension(owner=owner, repo_name=repo_name, short_name=short_name, commit_sha=latest_sha)
    return _ExtensionUpdateStatus.UPDATED


def _install_extension(
    *,
    owner: str,
    repo_name: str,
    short_name: str,
    commit_sha: str | None = None,
) -> ExtensionManifest:
    """Fetch and install an extension (binary or Python package), then persist its manifest.

    Installs in place: an existing install is overwritten without being removed first, so a failed
    update keeps the previous version working. A failed fresh install is cleaned up entirely.
    """
    extension_dir = _get_extension_dir(short_name)
    fresh_install = not extension_dir.exists()
    installed = False
    try:
        try:
            binary = _fetch_remote_binary(owner=owner, repo_name=repo_name, short_name=short_name)
        except Exception:
            binary = None

        if binary is not None:
            if os.name == "nt":
                raise CLIError(
                    f"'{owner}/{repo_name}' is a shell-script extension, which is not supported on Windows. "
                    "Only Python extensions can be installed on Windows."
                )
            executable_path = _install_binary_extension(
                extension_dir=extension_dir, short_name=short_name, binary=binary
            )
        else:
            executable_path = _install_python_extension(
                extension_dir=extension_dir, owner=owner, repo_name=repo_name, short_name=short_name
            )

        if commit_sha is None:
            # The extension itself is on disk by now, fetched from the CDN. The SHA only feeds update
            # detection, so a failure here must not throw the install away: without it, the next
            # `hf extensions update` sees no known SHA, considers the extension outdated and reinstalls.
            try:
                commit_sha = _fetch_latest_commit_sha(owner=owner, repo_name=repo_name)
            except Exception as error:
                out.warning(f"Could not record a version for '{owner}/{repo_name}', installing anyway: {error}")

        manifest = ExtensionManifest(
            owner=owner,
            repo=repo_name,
            repo_id=f"{owner}/{repo_name}",
            short_name=short_name,
            executable_path=str(executable_path),
            type="binary" if binary is not None else "python",
            installed_at=datetime.now(timezone.utc),
            description=_try_fetch_remote_description(owner=owner, repo_name=repo_name),
            commit_sha=commit_sha,
        )
        manifest.save(extension_dir)
        installed = True
        return manifest
    except CLIError:
        raise
    except subprocess.TimeoutExpired as e:
        raise CLIExtensionInstallError(
            f"Pip install timed out after {_EXTENSIONS_PIP_INSTALL_TIMEOUT}s for '{owner}/{repo_name}'. "
            "See pip output above for details."
        ) from e
    except subprocess.CalledProcessError as e:
        raise CLIExtensionInstallError(
            f"Failed to install pip package from '{owner}/{repo_name}' (exit code {e.returncode}). "
            "See pip output above for details."
        ) from e
    except Exception as e:
        raise CLIExtensionInstallError(f"Failed to install extension from '{owner}/{repo_name}': {e}") from e
    finally:
        if not installed and fresh_install:
            shutil.rmtree(extension_dir, ignore_errors=True)


def _fetch_latest_commit_sha(*, owner: str, repo_name: str) -> str:
    """Fetch the latest commit SHA on the default branch, used to detect available updates.

    The only REST API call left in the install and update paths. Raises on failure: callers decide
    whether a missing SHA is fatal, and report the reason themselves.
    """
    response = _github_request(
        "GET",
        f"https://api.github.com/repos/{owner}/{repo_name}/commits/HEAD",
        headers={"Accept": "application/vnd.github.sha"},
    )
    return response.text.strip()


def _fetch_remote_binary(*, owner: str, repo_name: str, short_name: str) -> bytes:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/HEAD/hf-{short_name}"
    response = _github_request("GET", raw_url)
    return response.content


def _install_binary_extension(*, extension_dir: Path, short_name: str, binary: bytes) -> Path:
    extension_dir.mkdir(parents=True, exist_ok=True)
    executable_path = extension_dir / _get_executable_name(short_name)
    executable_path.write_bytes(binary)

    if os.name != "nt":
        os.chmod(executable_path, 0o755)

    return executable_path


def _install_python_extension(*, extension_dir: Path, owner: str, repo_name: str, short_name: str) -> Path:
    source_url = f"https://github.com/{owner}/{repo_name}/archive/HEAD.zip"
    venv_dir = extension_dir / "venv"
    venv_python = _get_venv_bin_path(venv_dir, "python.exe" if os.name == "nt" else "python")
    uv_path = shutil.which("uv")

    status = out.status()
    if not venv_python.is_file():
        status.update(f"Creating virtual environment in {venv_dir}")
        extension_dir.mkdir(parents=True, exist_ok=True)
        if uv_path:
            subprocess.run([uv_path, "venv", str(venv_dir)], check=True)
        else:
            venv.EnvBuilder(with_pip=True).create(str(venv_dir))
        status.done(f"Virtual environment created in {venv_dir}")

    status.update(f"Installing package from {source_url}")
    if uv_path:
        # --reinstall: the source URL is the same for every commit, so cached data must be refreshed.
        install_cmd = [uv_path, "pip", "install", "--reinstall", "--python", str(venv_python), source_url]
    else:
        install_cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--force-reinstall",
            source_url,
        ]
    subprocess.run(install_cmd, check=True, timeout=_EXTENSIONS_PIP_INSTALL_TIMEOUT)
    status.done(f"Package installed from {source_url}")

    executable_name = _get_executable_name(short_name)
    venv_executable = _get_venv_bin_path(venv_dir, executable_name)
    if not venv_executable.is_file():
        raise CLIError(
            f"Installed package from '{owner}/{repo_name}' does not expose the required console script "
            f"'{executable_name}'."
        )
    return venv_executable.resolve()


_GITHUB_ABOUT_RE = re.compile(r'<meta name="description" content="([^"]*)"')


def _try_fetch_remote_description(owner: str, repo_name: str) -> str | None:
    """Try to fetch project description either from:
    - manifest.json
    - pyproject.toml
    - the repository's GitHub "About" field

    Only best effort, no error handling.
    """
    base = f"https://raw.githubusercontent.com/{owner}/{repo_name}/HEAD"

    # from manifest.json
    try:
        response = _github_request("GET", f"{base}/{MANIFEST_FILENAME}")
        description = response.json().get("description")
        if isinstance(description, str):
            return description
    except Exception:
        pass

    # from pyproject.toml
    try:
        response = _github_request("GET", f"{base}/pyproject.toml")

        # Weak parser but ok for "best effort"
        for line in response.text.splitlines():
            line = line.strip()
            if line.startswith("description"):
                _, _, value = line.partition("=")
                return value.strip().strip("\"'")
    except Exception:
        pass

    # from the repo page's "About" field. Served by github.com, so it costs no REST API quota.
    try:
        response = _github_request("GET", f"https://github.com/{owner}/{repo_name}")
        if (match := _GITHUB_ABOUT_RE.search(response.text)) is not None:
            # GitHub renders "<about> - <owner>/<repo>", and unrelated boilerplate when About is empty.
            content = unescape(match.group(1))
            if (description := content.removesuffix(f" - {owner}/{repo_name}")) != content:
                return description
    except Exception:
        pass

    return None


def _get_extension_dir(short_name: str) -> Path:
    # Callers validate at the parse boundary already; re-validate here as defense-in-depth since
    # this path is rmtree'd on removal.
    _validate_extension_short_name(short_name, original_input=short_name)
    return EXTENSIONS_ROOT.expanduser() / f"hf-{short_name}"


def _github_request(
    method: str, url: str, *, params: dict | None = None, headers: dict | None = None
) -> httpx2.Response:
    """Perform a GitHub request.

    Shared by every GitHub/Raw fetch in this module so the timeout, redirect and rate-limit policy are
    shared.

    Unauthenticated `api.github.com` calls are capped per IP address, and that quota is shared with
    everything else calling GitHub from the same network (NAT, VPN, CI egress). Remote refs are therefore
    always addressed as `HEAD`, which GitHub resolves to the repo's default branch on
    `raw.githubusercontent.com`, on the archive endpoint and on the REST API alike: no
    `GET /repos/{owner}/{repo}` lookup is needed to discover the branch name.
    """
    response = get_session().request(
        method,
        url,
        params=params,
        headers=headers,
        follow_redirects=True,
        timeout=_EXTENSIONS_DOWNLOAD_TIMEOUT,
    )
    # Primary limits report an exhausted quota, secondary ones only ask to back off.
    if response.status_code in (403, 429) and (
        response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers
    ):
        raise _GitHubRateLimitError(_rate_limit_message(response.headers))
    response.raise_for_status()
    return response


def _rate_limit_message(headers: httpx2.Headers) -> str:
    """Message for a GitHub rate-limit rejection, built from the response headers.

    The cap itself is deliberately not quoted: it depends on the endpoint (60/hour on the core API,
    10/minute on search), so only the wait is reported.

    `retry-after` wins over `x-ratelimit-reset`: GitHub rides the primary window's reset timestamp on
    every REST response, secondary rate limits included, so on a back-off it would point at an hourly
    window that is not the one being enforced.
    """
    message = (
        "GitHub API rate limit exceeded. Unauthenticated requests are capped per IP address, and that "
        "quota is shared with anything else calling GitHub from your network."
    )
    if (retry_after := headers.get("retry-after")) is not None:
        message += f" Retry in {retry_after}s."
    elif (reset := headers.get("x-ratelimit-reset")) is not None:
        try:
            reset_at = datetime.fromtimestamp(int(reset), tz=timezone.utc).astimezone()
        except (OverflowError, OSError, ValueError):
            pass
        else:
            message += f" It resets at {reset_at:%H:%M:%S %Z}."
    return message


def _github_repo_exists(*, owner: str, repo_name: str) -> bool:
    """Whether a public GitHub repo exists.

    Resolved against `github.com` rather than `GET /repos/{owner}/{repo}` to keep the check off the
    metered REST API (see `_github_request`). Only a 404 answers `False`: any other failure is reported
    as such, so callers never present an unreachable GitHub as a missing repo.
    """
    try:
        _github_request("HEAD", f"https://github.com/{owner}/{repo_name}")
    except httpx2.HTTPError as error:
        if isinstance(error, httpx2.HTTPStatusError) and error.response.status_code == 404:
            return False
        raise CLIError(f"Could not reach GitHub to check whether '{owner}/{repo_name}' exists: {error}") from error
    return True


def _get_executable_name(short_name: str) -> str:
    name = f"hf-{short_name}"
    if os.name == "nt":
        name += ".exe"
    return name


def _resolve_installed_executable_path(short_name: str) -> Path:
    extension_dir = _get_extension_dir(short_name)
    manifest = ExtensionManifest.load(extension_dir)
    return Path(manifest.executable_path).expanduser()


def _get_venv_bin_path(venv_dir: Path, executable_name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / executable_name
    return venv_dir / "bin" / executable_name


_ALLOWED_EXTENSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_extension_short_name(short_name: str, *, original_input: str) -> str:
    name = short_name.strip()
    if not name:
        raise CLIError("Extension name cannot be empty.")
    if any(sep in name for sep in ("/", "\\")):
        raise CLIError(f"Invalid extension name '{original_input}'.")
    if ".." in name or ":" in name:
        raise CLIError(f"Invalid extension name '{original_input}'.")
    if not _ALLOWED_EXTENSION_NAME.fullmatch(name):
        raise CLIError(
            f"Invalid extension name '{original_input}'. Allowed characters: letters, digits, '.', '_' and '-'."
        )
    return name


def _normalize_repo_id(repo_id: str) -> tuple[str, str, str]:
    if "://" in repo_id:
        raise CLIError("Only GitHub repositories in `[OWNER/]hf-<name>` format are supported.")

    parts = repo_id.split("/")
    if len(parts) == 1:
        owner = DEFAULT_EXTENSION_OWNER
        repo_name = parts[0]
    elif len(parts) == 2 and all(parts):
        owner, repo_name = parts
    else:
        raise CLIError(f"Expected `[OWNER/]REPO` format, got '{repo_id}'.")

    if not repo_name.startswith("hf-"):
        raise CLIError(f"Extension repository name must start with 'hf-', got '{repo_name}'.")

    short_name = repo_name.removeprefix("hf-")
    if not short_name:
        raise CLIError("Invalid extension repository name 'hf-'.")
    _validate_extension_short_name(short_name, original_input=repo_id)

    return owner, repo_name, short_name


def _normalize_extension_name(name: str) -> str:
    repo_name = name.strip().rsplit("/", 1)[-1]
    return _validate_extension_short_name(repo_name.removeprefix("hf-"), original_input=name)


def _execute_extension_binary(executable_path: Path, args: list[str]) -> int:
    try:
        return subprocess.call([str(executable_path)] + args)
    except OSError as e:
        if os.name == "nt" or e.errno != errno.ENOEXEC:
            raise
        return subprocess.call(["sh", str(executable_path)] + args)
<|endoftext|># Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to interact with papers on the Hugging Face Hub."""

import datetime
import enum
from typing import Annotated, get_args

from huggingface_hub.errors import CLIError, HfHubHTTPError
from huggingface_hub.hf_api import DailyPapersSort_T

from ._cli_utils import (
    LimitOpt,
    TokenOpt,
    get_hf_api,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out


_SORT_OPTIONS = get_args(DailyPapersSort_T)
PaperSortEnum = enum.Enum("PaperSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


def _parse_date(value: str | None) -> str | None:
    """Parse date option, converting 'today' to current date."""
    if value is None:
        return None
    if value.lower() == "today":
        return datetime.date.today().isoformat()
    return value


papers_cli = typer_factory(help="Interact with papers on the Hub.")


@papers_cli.command(
    "list | ls",
    examples=[
        "hf papers ls",
        "hf papers ls --sort trending",
        "hf papers ls --date 2025-01-23",
        "hf papers ls --week 2025-W09",
        "hf papers ls --submitter akhaliq",
        "hf papers ls --format json",
    ],
)
def papers_ls(
    date: Annotated[
        str | None,
        Option(
            help="Date in ISO format (YYYY-MM-DD) or 'today'.",
            callback=_parse_date,
        ),
    ] = None,
    week: Annotated[
        str | None,
        Option(help="ISO week to filter by, e.g. '2025-W09'."),
    ] = None,
    month: Annotated[
        str | None,
        Option(help="Month to filter by in ISO format (YYYY-MM), e.g. '2025-02'."),
    ] = None,
    submitter: Annotated[
        str | None,
        Option(help="Filter by username of the submitter."),
    ] = None,
    sort: Annotated[
        PaperSortEnum | None,
        Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 50,
    token: TokenOpt = None,
) -> None:
    """List daily papers on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = []
    for paper_info in api.list_daily_papers(
        date=date,
        week=week,
        month=month,
        submitter=submitter,
        sort=sort_key,
        limit=limit,
    ):
        item = _dataclass_to_dict(paper_info)
        submitted_by = item.get("submitted_by") or {}
        item["submitted_by_name"] = submitted_by.get("fullname") or submitted_by.get("username") or ""
        results.append(item)
    out.table(
        results,
        headers=["id", "title", "upvotes", "comments", "published_at", "submitted_by_name"],
    )


@papers_cli.command(
    "search",
    examples=[
        'hf papers search "vision language"',
        'hf papers search "attention mechanism" --limit 10',
        'hf papers search "diffusion" --format json',
    ],
)
def papers_search(
    query: Annotated[str, Argument(help="Search query string.")],
    limit: LimitOpt = 20,
    token: TokenOpt = None,
) -> None:
    """Search papers on the Hub."""
    api = get_hf_api(token=token)
    results = [_dataclass_to_dict(paper_info) for paper_info in api.list_papers(query=query, limit=limit)]
    out.table(results, headers=["id", "title", "summary", "upvotes", "published_at"])


@papers_cli.command(
    "info",
    examples=[
        "hf papers info 2601.15621",
    ],
)
def papers_info(
    paper_id: Annotated[str, Argument(help="The arXiv paper ID (e.g. '2502.08025').")],
    token: TokenOpt = None,
) -> None:
    """Get info about a paper on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.paper_info(id=paper_id)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' not found on the Hub.") from e
        raise
    out.dict(info)


@papers_cli.command(
    "read",
    examples=[
        "hf papers read 2601.15621",
    ],
)
def papers_read(
    paper_id: Annotated[str, Argument(help="The arXiv paper ID (e.g. '2502.08025').")],
    token: TokenOpt = None,
) -> None:
    """Read a paper as markdown."""
    api = get_hf_api(token=token)
    try:
        content = api.read_paper(id=paper_id)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' not found on the Hub.") from e
        raise
    out.text(content)
<|endoftext|># Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains CLI utilities (styling, helpers)."""

import difflib
import importlib.metadata
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar, cast

import click

from huggingface_hub import __version__, constants
from huggingface_hub.errors import CLIError
from huggingface_hub.utils._dotenv import load_dotenv

from ._framework import Argument, HfCommand, HfGroup, Option, build_command
from ._help_formatter import StyledContext
from ._output import OutputFormat, out


logger = logging.getLogger(__name__)

# Arbitrary default limit for models/datasets/spaces list commands.
REPO_LIST_DEFAULT_LIMIT = 30

if TYPE_CHECKING:
    from huggingface_hub import Volume
    from huggingface_hub.hf_api import HfApi


def installation_method() -> str:
    from huggingface_hub.utils import installation_method as _installation_method

    return _installation_method()


def get_hf_api(token: str | None = None) -> "HfApi":
    # Import here to avoid circular import
    from huggingface_hub.hf_api import HfApi

    return HfApi(token=token, library_name="huggingface-cli", library_version=__version__)


#### TYPER UTILS

CLI_REFERENCE_URL = "https://huggingface.co/docs/huggingface_hub/en/guides/cli"


def generate_epilog(examples: list[str], docs_anchor: str | None = None) -> str:
    """Generate an epilog with examples and a Learn More section.

    Args:
        examples: List of example commands (without the `$ ` prefix).
        docs_anchor: Optional anchor for the docs URL (e.g., "#hf-download").

    Returns:
        Formatted epilog string.
    """
    docs_url = f"{CLI_REFERENCE_URL}{docs_anchor}" if docs_anchor else CLI_REFERENCE_URL
    examples_str = "\n".join(f"  $ {ex}" for ex in examples)
    return f"""\
Examples
{examples_str}

Learn more
  Use `hf <command> --help` for more information about a command.
  Read the documentation at {docs_url}
"""


TOPIC_T = Literal["main", "help"] | str
FallbackHandlerT = Callable[[list[str], set[str]], int | None]
ExpandPropertyT = TypeVar("ExpandPropertyT", bound=str)


def _format_epilog_no_indent(epilog: str | None, ctx: click.Context, formatter: click.HelpFormatter) -> None:
    """Write the epilog without indentation."""
    if epilog:
        formatter.write_paragraph()
        for line in epilog.split("\n"):
            formatter.write_text(line)


_ALIAS_SPLIT = re.compile(r"\s*\|\s*")


class HFCliTyperGroup(HfGroup):
    """
    CLI Group that:
    - lists commands alphabetically within sections.
    - separates commands by topic (main, help, etc.).
    - formats epilog without extra indentation.
    - supports aliases via pipe-separated names (e.g. ``name="list | ls"``).
    - consumes the global formatting flags (``--format``, ``--json``, ``-q`` / ``--quiet``, ``--no-truncate``)
      anywhere in the args of a leaf command and applies them to ``out``, so leaf
      commands don't need to declare these options themselves.
    - rewrites ``spaces/user/repo`` to ``user/repo --type space`` for commands that accept ``--type``.
    - enriches "No such option" / "No such command" errors with available options or commands.
    """

    context_class = StyledContext

    def invoke(self, ctx: click.Context) -> None:
        """Enrich unknown-option errors with available options or subcommands.

        Catches `NoSuchOption` raised during subcommand `make_context()`
        (option parsing).  For leaf commands (e.g. `hf repos create --test`)
        we list the command's options; for groups (e.g. `hf cache --test`)
        we list subcommands since groups have no user-facing options.
        """
        try:
            return super().invoke(ctx)
        except click.NoSuchOption as e:
            if e.ctx is not None and e.ctx.command is not None:
                cmd = e.ctx.command
                if isinstance(cmd, click.Group):
                    # Group has no user-facing options -> show subcommands instead
                    items = [
                        (name, sub.get_short_help_str(limit=80))
                        for name in cmd.list_commands(e.ctx)
                        if (sub := cmd.get_command(e.ctx, name)) is not None and not sub.hidden
                    ]
                    _enrich_usage_error(e, "commands", items)
                else:
                    # Leaf command -> show its options using Click's rich formatting
                    items = [
                        record
                        for p in cmd.get_params(e.ctx)
                        if isinstance(p, click.Option) and not p.hidden and (record := p.get_help_record(e.ctx))
                    ]
                    _enrich_usage_error(e, "options", items)
            raise

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
        cmd_name = args[0] if args and not args[0].startswith("-") else None
        cmd = self.get_command(ctx, cmd_name) if cmd_name else None

        if isinstance(cmd, (LazyHfCommand, LazyHfGroup)):
            loaded_cmd = cmd.materialize()
            for registered_name, registered_cmd in self.commands.items():
                if registered_cmd is cmd:
                    self.commands[registered_name] = loaded_cmd
            cmd = loaded_cmd

        if cmd is not None:
            self._rewrite_repo_type_prefix(cmd, args)

        try:
            name, resolved_cmd, sub_args = super().resolve_command(ctx, args)
        except click.UsageError as e:
            # Unknown subcommand -> add fuzzy suggestions and list available commands.
            if cmd is None and cmd_name is not None:
                # Expand aliases ("list | ls" → ["list", "ls"]) for accurate fuzzy matching.
                visible_names = [
                    alias
                    for key, registered in self.commands.items()
                    if not registered.hidden
                    for alias in _ALIAS_SPLIT.split(key)
                ]
                matches = difflib.get_close_matches(cmd_name, visible_names)
                if matches:
                    suggestions = ", ".join(f"'{m}'" for m in matches)
                    setattr(e, "message", f"{e.message.rstrip('.')}. Did you mean {suggestions}?")
                items = [
                    (name, sub.get_short_help_str(limit=80))
                    for name in self.list_commands(ctx)
                    if (sub := self.get_command(ctx, name)) is not None and not sub.hidden
                ]
                _enrich_usage_error(e, "commands", items)
            raise

        # If we just resolved a leaf command, eagerly consume any global formatting
        # flags (--format / --json / -q / --quiet / --no-truncate) from its args before click parses
        # them.  Group resolution is recursive — leaves (and only leaves) need this.
        if resolved_cmd is not None and not isinstance(resolved_cmd, click.Group):
            _consume_format_flags_for_leaf(resolved_cmd, sub_args)

        return name, resolved_cmd, sub_args

    @staticmethod
    def _rewrite_repo_type_prefix(cmd: click.Command, args: list[str]) -> None:
        """Rewrite prefixed repo IDs (e.g. ``spaces/user/repo``) to ``user/repo --type space``.

        Only applies to commands that have a ``--type`` / ``--repo-type`` option and
        at least one repo-ID positional argument (any ``click.Argument`` whose name
        ends with ``_id``, e.g. ``repo_id``, ``from_id``, ``to_id``).  When the
        token that maps to such an argument matches ``{prefix}/org/repo`` (where
        *prefix* is one of ``spaces``, ``datasets``, or ``models``), the prefix is
        stripped and an implicit ``--type {type}`` is appended.  An error is raised
        if ``--type`` is also provided explicitly or if multiple prefixed arguments
        disagree on the repo type.

        Only repo-ID positional slots are inspected so that other positional
        arguments (filenames, local paths, patterns …) are never misinterpreted as
        prefixed repo IDs.
        """
        has_type_option = any(isinstance(param, click.Option) and "--type" in param.opts for param in cmd.params)
        if not has_type_option:
            return

        # Locate all repo-ID positional arguments and their indices among Arguments.
        repo_id_positions: set[int] = set()
        arg_idx = 0
        for param in cmd.params:
            if isinstance(param, click.Argument):
                if param.name in ("repo_id", "from_id", "to_id"):
                    repo_id_positions.add(arg_idx)
                arg_idx += 1

        if not repo_id_positions:
            return

        # Build a set of option names that consume a following value token.
        value_options: set[str] = set()
        for param in cmd.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in (*param.opts, *param.secondary_opts):
                    value_options.add(opt)

        # Walk through args (skipping args[0] = command name) to map positional
        # slots to their indices in `args`.
        positional_count = 0
        repo_id_arg_indices: list[int] = []
        i = 1
        while i < len(args):
            arg = args[i]
            if arg == "--":
                break  # everything after -- is positional literal; stop rewriting
            if arg.startswith("-"):
                if "=" in arg or arg not in value_options:
                    i += 1  # flag or --opt=val — single token
                else:
                    i += 2  # value-taking option — skip the value too
            else:
                if positional_count in repo_id_positions:
                    repo_id_arg_indices.append(i)
                positional_count += 1
                i += 1

        if not repo_id_arg_indices:
            return

        # Check each repo-ID arg for a type prefix and collect rewrites.
        inferred_type: str | None = None
        first_prefix: str | None = None
        rewrites: list[tuple[int, str]] = []  # (args index, new value without prefix)

        for arg_index in repo_id_arg_indices:
            parts = args[arg_index].split("/", 2)
            if len(parts) != 3 or parts[0] not in constants.REPO_TYPES_MAPPING:
                continue
            prefix = parts[0]
            mapped_type = constants.REPO_TYPES_MAPPING[prefix]
            if inferred_type is not None and mapped_type != inferred_type:
                raise click.UsageError(f"Conflicting repo type prefixes: '{first_prefix}/' and '{prefix}/'.")
            inferred_type = mapped_type
            first_prefix = prefix
            rewrites.append((arg_index, f"{parts[1]}/{parts[2]}"))

        if not rewrites:
            return

        # Error if --type / --repo-type was also provided explicitly.
        if any(
            arg == "--type" or arg.startswith("--type=") or arg == "--repo-type" or arg.startswith("--repo-type=")
            for arg in args
        ):
            raise click.UsageError(
                f"Ambiguous repo type: got prefix '{first_prefix}/' in repo ID and explicit --type. Use one or the other."
            )

        # Apply all rewrites and append --type once.
        for arg_index, new_value in rewrites:
            args[arg_index] = new_value
        args.extend(["--type", inferred_type])  # type: ignore

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # Try exact match first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # Fall back to alias lookup: check if cmd_name matches any alias
        # taken from https://github.com/fastapi/typer/issues/132#issuecomment-2417492805
        for registered_name, registered_cmd in self.commands.items():
            aliases = _ALIAS_SPLIT.split(registered_name)
            if cmd_name in aliases:
                return registered_cmd
        return None

    def _alias_map(self) -> dict[str, list[str]]:
        """Build a mapping from primary command name to its aliases (if any)."""
        result: dict[str, list[str]] = {}
        for registered_name in self.commands:
            parts = _ALIAS_SPLIT.split(registered_name)
            primary = parts[0]
            result[primary] = parts[1:]
        return result

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        topics: dict[str, list] = {}
        alias_map = self._alias_map()

        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            help_text = cmd.get_short_help_str(limit=formatter.width)
            aliases = alias_map.get(name, [])
            if aliases:
                help_text = f"{help_text} [alias: {', '.join(aliases)}]"
            topic = getattr(cmd, "topic", "main")
            topics.setdefault(topic, []).append((name, help_text))

        with formatter.section("Main commands"):
            formatter.write_dl(topics["main"])
        for topic in sorted(topics.keys()):
            if topic == "main":
                continue
            with formatter.section(f"{topic.capitalize()} commands"):
                formatter.write_dl(topics[topic])

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Collect only the first example from each command (to keep group help concise)
        # Full examples are shown in individual subcommand help (e.g. `hf buckets sync --help`)
        all_examples: list[str] = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            cmd_examples = getattr(cmd, "examples", [])
            if cmd_examples:
                all_examples.append(cmd_examples[0])

        if all_examples:
            epilog = generate_epilog(all_examples)
            _format_epilog_no_indent(epilog, ctx, formatter)
        elif self.epilog:
            _format_epilog_no_indent(self.epilog, ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:  # type: ignore[name-defined]
        # For aliased commands ("list | ls"), use the primary name (first entry).
        primary_names: list[str] = []
        for name in self.commands:
            primary = _ALIAS_SPLIT.split(name)[0]
            primary_names.append(primary)
        return sorted(primary_names)

    def command(  # type: ignore  # adds topic/examples on top of HfGroup.command
        self,
        name: str | None = None,
        *,
        topic: TOPIC_T = "main",
        examples: list[str] | None = None,
        epilog: str | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        # Generate the epilog from examples when not provided explicitly, then build the
        # command with a topic/examples-aware command class.
        if epilog is None and examples:
            epilog = generate_epilog(examples)
        return super().command(name, cls=HFCliCommand(topic, examples), epilog=epilog, **kwargs)


_FORMAT_HELP = "Output format. Defaults to 'auto' which picks 'agent' or 'human' based on the terminal."
_JSON_HELP = "JSON output. Equivalent to '--format json'."
_QUIET_HELP = "Quiet output (one ID per line). Equivalent to '--format quiet'."

_FORMATTING_OPTIONS_HELP_RECORDS: list[tuple[str, str]] = [
    ("--format [auto|human|agent|json|quiet]", _FORMAT_HELP),
    ("--json", _JSON_HELP),
    ("-q, --quiet", _QUIET_HELP),
    ("--no-truncate", "Do not truncate scalar values in human tables (list/dict columns stay shortened)."),
]


def _format_formatting_options_section(formatter: click.HelpFormatter) -> None:
    with formatter.section("Formatting options"):
        formatter.write_dl(_FORMATTING_OPTIONS_HELP_RECORDS)


# Formatting options for pass-through commands (`ignore_unknown_options=True`, e.g. `hf jobs run`), which
# `_consume_format_flags_for_leaf` leaves untouched. Such commands must declare the flags explicitly (so click
# parses them instead of forwarding them to the container command) and apply them with `set_output_format()`.
FormatOpt = Annotated[OutputFormat | None, Option("--format", help=_FORMAT_HELP)]
JsonOpt = Annotated[bool, Option("--json", help=_JSON_HELP)]
QuietOpt = Annotated[bool, Option("-q", "--quiet", help=_QUIET_HELP)]


def set_output_format(format: OutputFormat | None, json: bool, quiet: bool) -> None:
    """Apply the formatting flags declared by a pass-through command to the `out` singleton."""
    flags = [flag for flag, given in (("--format", format is not None), ("--json", json), ("-q", quiet)) if given]
    if len(flags) > 1:
        raise click.UsageError(f"'{flags[0]}' and '{flags[1]}' are mutually exclusive.")
    if format is not None:
        out.set_mode(format)
    elif json:
        out.set_mode(OutputFormat.json)
    elif quiet:
        out.set_mode(OutputFormat.quiet)


def _has_local_formatting_option(cmd: click.Command) -> bool:
    """Return True if the command defines its own --format, --json or --quiet / -q.

    Used to skip the global formatting flag pre-processor and the duplicated "Formatting options" help section for
    legacy commands like 'hf jobs ls' that have their own format/quiet options.
    """
    for param in cmd.params:
        if not isinstance(param, click.Option):
            continue
        opts = (*param.opts, *param.secondary_opts)
        if "--format" in opts or "--json" in opts or "--quiet" in opts or "-q" in opts:
            return True
    return False


def _consume_format_flags_for_leaf(cmd: click.Command, args: list[str]) -> None:
    """Apply global formatting flags from 'args' to a leaf command.

    Two modes, depending on the command:

    * **Pass-through commands** (ignore_unknown_options=True, e.g. 'hf extensions exec'):
      args are forwarded verbatim to an external binary; we don't touch them.

    * **Legacy commands with a local --format option** (e.g. 'hf jobs ls' whose '--format' accepts Go templates):
      the global flags are rewritten in-place to the legacy form ('--json' → '--format json', '--quiet'/'-q' → '--format quiet'
      when the cmd has no own '--quiet') so click can parse them locally. This preserves backwards compatibility with the previous shorthand behavior.

    * **Modern commands** (no local format/quiet/json options): the flags '--format <value>' / '--json' / '--quiet' / '-q' are stripped from 'args' and applied to the singleton 'out'.

    '--no-truncate' is stripped for all non-pass-through commands; when present, human table cells are not truncated.

    Raises click.UsageError if multiple conflicting flags are supplied (e.g. '--json' together with '--format table').
    """
    if cmd.context_settings.get("ignore_unknown_options"):
        return

    no_truncate = _consume_no_truncate_flags(args)
    out.set_no_truncate(no_truncate)

    has_local_format = False
    has_local_quiet = False
    has_local_json = False
    for param in cmd.params:
        if not isinstance(param, click.Option):
            continue
        opts = (*param.opts, *param.secondary_opts)
        if "--format" in opts:
            has_local_format = True
        if "--quiet" in opts or "-q" in opts:
            has_local_quiet = True
        if "--json" in opts:
            has_local_json = True

    if has_local_format:
        _rewrite_legacy_shorthands(args, rewrite_json=not has_local_json, rewrite_quiet=not has_local_quiet)
        return

    # Strip --format/--json/-q/--quiet from 'args' and apply to 'out'
    chosen_mode: OutputFormat = OutputFormat.auto
    chosen_flag: str | None = None

    def _check_conflict(new_flag: str) -> None:
        # Reject any second formatting flag before parsing values, so the user gets
        # a "mutually exclusive" error rather than e.g. an "invalid value" error
        # from the second flag's argument.
        if chosen_flag is not None:
            raise click.UsageError(f"'{chosen_flag}' and '{new_flag}' are mutually exclusive.")

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            break  # everything after '--' is a positional literal
        if arg == "--format":
            _check_conflict("--format")
            if i + 1 >= len(args):
                raise click.UsageError("Option '--format' requires a value.")
            chosen_mode = _parse_format_value(args[i + 1])
            chosen_flag = "--format"
            del args[i : i + 2]  # --format value => 2 args removed
            continue
        if arg.startswith("--format="):
            _check_conflict("--format")
            chosen_mode = _parse_format_value(arg[len("--format=") :])
            chosen_flag = "--format"
            del args[i : i + 1]
            continue
        if arg == "--json":
            _check_conflict("--json")
            chosen_mode = OutputFormat.json
            chosen_flag = "--json"
            del args[i : i + 1]
            continue
        if arg in ("-q", "--quiet"):
            _check_conflict(arg)
            chosen_mode = OutputFormat.quiet
            chosen_flag = arg
            del args[i : i + 1]
            continue
        i += 1

    out.set_mode(chosen_mode)


def _consume_no_truncate_flags(args: list[str]) -> bool:
    """Strip all global --no-truncate flags from args and return whether any was provided."""
    no_truncate = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            break  # everything after '--' is a positional literal
        if arg == "--no-truncate":
            no_truncate = True
            del args[i : i + 1]
            continue
        if arg.startswith("--no-truncate="):
            raise click.UsageError("Option '--no-truncate' does not take a value.")
        i += 1
    return no_truncate


def _rewrite_legacy_shorthands(args: list[str], *, rewrite_json: bool, rewrite_quiet: bool) -> None:
    """Rewrite --json / -q / --quiet to --format ... for legacy commands.

    Used for commands like 'hf jobs ls' that still own their '--format' option.
    The rewrite lets users keep using the global shorthand while click parses
    '--format <value>' locally.
    """
    has_format_in_args = any(arg == "--format" or arg.startswith("--format=") for arg in args)

    if rewrite_json and "--json" in args:
        if has_format_in_args:
            raise click.UsageError("'--json' and '--format' are mutually exclusive.")
        idx = args.index("--json")
        args[idx : idx + 1] = ["--format", "json"]
        has_format_in_args = True

    if rewrite_quiet:
        flag = "-q" if "-q" in args else ("--quiet" if "--quiet" in args else None)
        if flag is not None:
            if has_format_in_args:
                raise click.UsageError(f"'{flag}' and '--format' are mutually exclusive.")
            idx = args.index(flag)
            args[idx : idx + 1] = ["--format", "quiet"]


def _parse_format_value(value: str) -> "OutputFormat":
    try:
        return OutputFormat(value)
    except ValueError:
        valid = ", ".join(m.value for m in OutputFormat)
        raise click.UsageError(f"Invalid value for '--format': '{value}'. Valid values: {valid}.") from None


def _enrich_usage_error(error: click.UsageError, label: str, items: list[tuple[str, str]]) -> None:
    """Append a list of available options or commands to a usage error message."""
    if not items or error.ctx is None or f"Available {label} for" in error.message:
        return
    cmd_path = error.ctx.command_path
    lines = [f"\n\nAvailable {label} for '{cmd_path}':"]
    for name, help_text in items:
        lines.append(f"  {name:30s} {help_text}")
    lines.append(f"\nRun '{cmd_path} --help' for full details.")
    if isinstance(error, click.NoSuchOption) and error.possibilities:
        lines.append(f"\nDid you mean: {', '.join(sorted(error.possibilities))}?")
        setattr(error, "possibilities", [])
    setattr(error, "message", error.message + "\n".join(lines))


def fallback_typer_group_factory(
    fallback_handler: FallbackHandlerT,
    extra_commands_provider: Callable[[], list[tuple[str, str]]] | None = None,
) -> type[HFCliTyperGroup]:
    """Return a Typer group class that runs a fallback handler before command resolution."""

    class FallbackTyperGroup(HFCliTyperGroup):
        def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
            fallback_exit_code = fallback_handler(args, set(self.commands.keys()))
            if fallback_exit_code is not None:
                raise SystemExit(fallback_exit_code)
            return super().resolve_command(ctx, args)

        def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
            super().format_commands(ctx, formatter)
            if extra_commands_provider is not None:
                entries = extra_commands_provider()
                if entries:
                    with formatter.section("Extension commands"):
                        formatter.write_dl(entries)

    return FallbackTyperGroup


def HFCliCommand(topic: TOPIC_T, examples: list[str] | None = None) -> type[HfCommand]:
    def format_epilog(self: click.Command, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _format_epilog_no_indent(self.epilog, ctx, formatter)

    def format_options(self: HfCommand, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        HfCommand.format_options(self, ctx, formatter)
        # Skip the section for commands that define their own --format / --quiet / --json,
        # or for pass-through commands that forward args to an external binary.
        if _has_local_formatting_option(self):
            return
        if self.context_settings.get("ignore_unknown_options"):
            return
        _format_formatting_options_section(formatter)

    def parse_args(self: click.Command, ctx: click.Context, args: list[str]) -> list[str]:
        # Show help when a command with required arguments is invoked without any args
        # (mirrors group behavior: `hf jobs` prints help, so `hf download` should too).
        if not args and not ctx.resilient_parsing:
            if any(isinstance(p, click.Argument) and p.required for p in self.params):
                click.echo(ctx.get_help(), color=ctx.color)
                ctx.exit()
        return HfCommand.parse_args(self, ctx, args)

    return type(
        f"HfCommand{topic.capitalize()}",
        (HfCommand,),
        {
            "context_class": StyledContext,
            "topic": topic,
            "examples": examples or [],
            "format_epilog": format_epilog,
            "format_options": format_options,
            "parse_args": parse_args,
        },
    )


class LazyHfCommand(click.Command):
    """A lightweight command placeholder that imports its implementation when selected."""

    def __init__(
        self,
        name: str,
        *,
        module: str,
        attribute: str,
        is_factory: bool = False,
        topic: TOPIC_T = "main",
        examples_attribute: str | None = None,
        hidden: bool = False,
    ) -> None:
        super().__init__(name=name, hidden=hidden)
        self.module = module
        self.attribute = attribute
        self.is_factory = is_factory
        self.topic = topic
        self.examples: list[str] = []
        self.examples_attribute = examples_attribute
        self._loaded: click.Command | None = None

    def materialize(self) -> click.Command:
        if self._loaded is None:
            module = importlib.import_module(self.module)
            callback = getattr(module, self.attribute)
            examples = getattr(module, self.examples_attribute) if self.examples_attribute is not None else []
            if self.is_factory:
                callback = callback()
            if isinstance(callback, click.Command):
                command = callback
                command.name = self.name
                command.hidden = self.hidden
            else:
                epilog = generate_epilog(examples) if examples else None
                command = build_command(
                    callback,
                    name=self.name,
                    cls=HFCliCommand(self.topic, examples),
                    epilog=epilog,
                    hidden=self.hidden,
                )
            self.examples = getattr(command, "examples", examples)
            self._loaded = command
        return self._loaded

    def get_short_help_str(self, limit: int = 45) -> str:
        return self.materialize().get_short_help_str(limit)


class LazyHfGroup(click.Group):
    """A lightweight group placeholder that imports its implementation when selected or inspected."""

    def __init__(self, name: str, *, module: str, attribute: str, hidden: bool = False) -> None:
        super().__init__(name=name, hidden=hidden)
        self.module = module
        self.attribute = attribute
        self._loaded: click.Group | None = None

    def materialize(self) -> click.Group:
        if self._loaded is None:
            group = getattr(importlib.import_module(self.module), self.attribute)
            if not isinstance(group, click.Group):
                raise TypeError(f"{self.module}.{self.attribute} is not a Click group")
            group.name = self.name
            group.hidden = self.hidden
            self._loaded = group
        return self._loaded

    def list_commands(self, ctx: click.Context) -> list[str]:
        return self.materialize().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return self.materialize().get_command(ctx, cmd_name)

    def get_short_help_str(self, limit: int = 45) -> str:
        return self.materialize().get_short_help_str(limit)


def typer_factory(help: str, epilog: str | None = None, cls: type[HFCliTyperGroup] | None = None) -> "HFCliTyperGroup":
    """Create a CLI command group with consistent settings.

    The returned group is the app: register commands with ``@group.command(...)``,
    subgroups with ``group.add_group(sub, name=...)``, and a group-level callback
    with ``@group.callback(...)``.

    Args:
        help: Help text for the group.
        epilog: Optional epilog text (use `generate_epilog` to create one).
        cls: Optional group class to use (defaults to `HFCliTyperGroup`).

    Returns:
        A configured `HFCliTyperGroup` instance.
    """
    if cls is None:
        cls = HFCliTyperGroup
    return cls(
        help=help,
        epilog=epilog,
        no_args_is_help=True,
        # Increase max content width for better readability
        context_settings={
            "max_content_width": 120,
            "help_option_names": ["-h", "--help"],
        },
    )


class SoftChoice(click.Choice):
    """A click Choice that suggests choices for autocompletion/docs but accepts any string.

    Unlike `click.Choice`, unknown values are passed through as-is instead of raising an error.
    This makes CLI options future-compatible when new server-side values are added.

    Accepts either a sequence of strings or an Enum class:
    ```python
    SoftChoice(SpaceHardware)        # from an enum
    SoftChoice(["a", "b", "c"])      # from a list
    ```
    """

    def __init__(self, choices: Sequence[str] | type[Enum]) -> None:
        values = (
            [m.value for m in choices] if isinstance(choices, type) and issubclass(choices, Enum) else list(choices)
        )
        super().__init__(values, case_sensitive=True)

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> str:
        try:
            return super().convert(value, param, ctx)
        except click.exceptions.BadParameter:
            return str(value)


class RepoType(str, Enum):
    model = "model"
    dataset = "dataset"
    space = "space"
    kernel = "kernel"


RepoIdArg = Annotated[
    str,
    Argument(
        help="The ID of the repo (e.g. `username/repo-name` or `spaces/username/repo-name`).",
    ),
]


RepoTypeOpt = Annotated[
    RepoType,
    Option(
        "--type",
        "--repo-type",
        help="The type of repository (model, dataset, space, or kernel).",
    ),
]

# Same as `RepoTypeOpt` but optional (defaults to `None` rather than `model`). Used by commands that
# accept an `hf://` URI as repo id: a `None` default lets us tell apart "user did not pass --repo-type"
# from "user explicitly passed --repo-type model", which is required to detect conflicts with the URI.
RepoTypeOptionalOpt = Annotated[
    RepoType | None,
    Option(
        "--type",
        "--repo-type",
        help="The type of repository (model, dataset, space, or kernel).",
        show_default="model",
    ),
]

# --- Secret hygiene: keep secret material out of argv ---
#
# Anything passed as a flag value lands in the shell history file and, on Linux, in
# `/proc/<pid>/cmdline`, which any process of the same user can read. The CLI already offers
# argv-free alternatives (`hf auth login`, `HF_TOKEN`, bare `--secrets KEY`, `--secrets-file`);
# these warnings point at them. They stay warnings on purpose: erroring out would break
# existing automation.


def _warn_secret_hygiene(message: str) -> None:
    """Emit a secret-hygiene warning on stderr in every output mode."""
    out.warning(message)


def _warn_on_inline_token(token: str | None) -> str | None:
    """Option callback for `TokenOpt`: warn when a token is passed as a flag value."""
    ctx = click.get_current_context(silent=True)
    # `hf auth login --token` is the non-interactive login flow this warning points users *to*,
    # so it is exempt. Matched on a suffix because the root command name depends on how the CLI
    # was invoked (`hf`, `huggingface-cli`, a test runner's prog name...).
    if token is not None and not (ctx is not None and ctx.command_path.endswith("auth login")):
        _warn_secret_hygiene(
            "`--token <value>` leaves your token in your shell history and in process listings. Run "
            "`hf auth login` once, or export `HF_TOKEN`, and drop the flag."
        )
    return token


def _warn_on_inline_secrets(secrets: list[str] | None) -> list[str] | None:
    """Option callback for `SecretsOpt`: warn once if any pair carries an inline value."""
    if secrets and any("=" in secret for secret in secrets):
        _warn_secret_hygiene(
            "`--secrets KEY=value` leaves the value in your shell history and in process listings. Prefer "
            "`--secrets KEY` (value read from your environment), `--secrets-file FILE` or `--secrets-file -`."
        )
    return secrets


TokenOpt = Annotated[
    str | None,
    Option(
        help="A User Access Token generated from https://huggingface.co/settings/tokens.",
        callback=_warn_on_inline_token,
    ),
]

PrivateOpt = Annotated[
    bool | None,
    Option(
        help="Whether to create a private repo if repo doesn't exist on the Hub. Ignored if the repo already exists.",
    ),
]

RevisionOpt = Annotated[
    str | None,
    Option(
        help="Git revision id which can be a branch name, a tag, or a commit hash.",
    ),
]


LimitOpt = Annotated[
    int,
    Option(help="Limit the number of results."),
]

AuthorOpt = Annotated[
    str | None,
    Option(help="Filter by author or organization."),
]

FilterOpt = Annotated[
    list[str] | None,
    Option(help="Filter by tags (e.g. 'text-classification'). Can be used multiple times."),
]

SearchOpt = Annotated[
    str | None,
    Option(help="Search query."),
]


# --- Env / Secrets shared options and parsing helpers (used by jobs, repos, etc.) ---

EnvOpt = Annotated[
    list[str] | None,
    Option(
        "-e",
        "--env",
        help="Set environment variables. E.g. --env ENV=value",
    ),
]

SecretsOpt = Annotated[
    list[str] | None,
    Option(
        "-s",
        "--secrets",
        help=(
            "Set secret environment variables. Prefer `--secrets SECRET` to read the value from your"
            " environment (e.g. `--secrets HF_TOKEN` to pass your Hugging Face token); `--secrets SECRET=value`"
            " puts the value in your shell history."
        ),
        callback=_warn_on_inline_secrets,
    ),
]

EnvFileOpt = Annotated[
    str | None,
    Option(
        "--env-file",
        help="Read in a file of environment variables. Use `-` to read them from stdin.",
    ),
]

SecretsFileOpt = Annotated[
    str | None,
    Option(
        help="Read in a file of secret environment variables. Use `-` to read them from stdin.",
    ),
]


def _get_extended_environ() -> dict[str, str]:
    """Return a copy of ``os.environ`` with the user's HF token injected (if available)."""
    from huggingface_hub import get_token

    extended_environ = os.environ.copy()
    if (token := get_token()) is not None:
        extended_environ["HF_TOKEN"] = token
    return extended_environ


def _read_env_file(env_file: str) -> str:
    """Read an env/secrets file, or stdin when ``env_file`` is ``-``.

    Piping through stdin is the only way to hand the CLI a secret value that touches neither
    argv nor the disk.
    """
    if env_file == "-":
        return sys.stdin.read()
    path = Path(env_file)
    # POSIX only: Windows reports a synthetic 0o666/0o444 mode, so this would always fire there.
    if os.name == "posix" and (mode := path.stat().st_mode & 0o777) & 0o077:
        _warn_secret_hygiene(
            f"'{env_file}' is readable by other users (mode {mode:o}). Run `chmod 600 {env_file}` to restrict it."
        )
    # "utf-8-sig" drops the BOM that Windows tools (e.g. PowerShell 5.1) write at the start of the file
    return path.read_text(encoding="utf-8-sig")


def parse_env_map(
    env: list[str] | None = None,
    env_file: str | None = None,
) -> dict[str, str | None]:
    """Parse ``-e``/``--env``/``-s``/``--secrets`` and ``--env-file``/``--secrets-file`` CLI args into a dict.

    Uses an extended environment that includes the user's HF token so that
    bare ``--secrets HF_TOKEN`` resolves correctly.
    """
    env_map: dict[str, str | None] = {}
    if env_file or env:
        extended_environ = _get_extended_environ()
        if env_file:
            env_map.update(load_dotenv(_read_env_file(env_file), environ=extended_environ))
        for env_value in env or []:
            env_map.update(load_dotenv(env_value, environ=extended_environ))
    return env_map


def env_map_to_key_value_list(env_map: dict[str, str | None]) -> list[dict[str, str]] | None:
    """Convert an env/secrets dict to the ``[{"key": ..., "value": ...}]`` format used by the Hub API."""
    if not env_map:
        return None
    return [{"key": k, "value": v or ""} for k, v in env_map.items()]


VolumesOpt = Annotated[
    list[str] | None,
    Option(
        "-v",
        "--volume",
        help="Mount one or more volumes. Format: hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]. "
        "TYPE is one of: models, datasets, spaces, buckets. "
        "TYPE defaults to models if omitted. "
        "models, datasets and spaces are always mounted read-only. buckets are read+write by default. "
        "E.g. -v hf://org/m:/data or -v hf://datasets/org/ds:/data or -v hf://buckets/org/b:/mnt:ro",
    ),
]


def parse_volumes(volumes: list[str] | None) -> "list[Volume] | None":
    """Parse volume specs from CLI arguments.

    Format: hf://[TYPE/]SOURCE[/PATH]:/MOUNT_PATH[:ro|:rw]
    Where TYPE is one of: models, datasets, spaces, buckets (defaults to models if omitted).
    SOURCE is the repo/bucket identifier (e.g. 'username/my-model').
    PATH is an optional subfolder inside the repo/bucket.
    MOUNT_PATH starts with '/'.
    Optional ':ro' or ':rw' suffix for read-only or read-write.

    Examples:
        hf://my-org/my-model:/data                (model, implicit type)
        hf://models/my-org/my-model:/data         (model, explicit type)
        hf://datasets/my-org/my-dataset:/data:ro
        hf://buckets/my-org/my-bucket:/mnt
        hf://spaces/my-org/my-space:/app
        hf://datasets/org/ds/train:/data          (with path inside repo)
        hf://buckets/org/b/sub/dir:/mnt           (with path inside bucket)
    """
    if not volumes:
        return None

    from huggingface_hub import Volume
    from huggingface_hub.utils import parse_hf_mount

    result: list[Volume] = []
    for raw_spec in volumes:
        mount = parse_hf_mount(raw_spec)
        result.append(
            Volume(
                type=mount.source.type,
                source=mount.source.id,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                path=mount.source.path_in_repo or None,
                revision=mount.source.revision or None,
            )
        )
    return result


def make_expand_properties_parser(valid_properties: Sequence[ExpandPropertyT]):
    """Create a callback to parse and validate comma-separated expand properties."""

    def _parse_expand_properties(value: str | None) -> list[ExpandPropertyT] | None:
        if value is None:
            return None
        properties = [p.strip() for p in value.split(",")]
        for prop in properties:
            if prop not in valid_properties:
                raise click.BadParameter(
                    f"Invalid expand property: '{prop}'. Valid values are: {', '.join(valid_properties)}"
                )
        return [cast(ExpandPropertyT, prop) for prop in properties]

    return _parse_expand_properties


### SSH


SshIdentityFileOpt = Annotated[
    Path | None,
    Option("-i", "--identity-file", help="Path to the SSH identity file (forwarded to `ssh -i`)."),
]

SshDryRunOpt = Annotated[
    bool,
    Option("--dry-run", help="Print the SSH command instead of running it."),
]


def exec_ssh(
    destination: str, *, port: int | None = None, identity_file: Path | None = None, dry_run: bool = False
) -> None:
    """Run an interactive `ssh` command to `destination` (`user@host`) and exit with its return code.

    With `dry_run`, print the command instead of running it.
    """
    cmd = ["ssh"]
    if identity_file is not None:
        cmd += ["-i", str(identity_file)]
    if port is not None:
        cmd += ["-p", str(port)]
    cmd.append(destination)
    if dry_run:
        out.text(shlex.join(cmd))
        return
    out.text(f"Running `{shlex.join(cmd)}`")
    result = subprocess.run(cmd)
    raise click.exceptions.Exit(code=result.returncode)


### PyPI VERSION CHECKER


def check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    """
    Check whether a newer version of a library is available on PyPI.

    If a newer version is found, print a hint pointing at `hf update`.

    If current version is a pre-release (e.g. `1.0.0.rc1`), or a dev version (e.g. `1.0.0.dev1`), no check is performed.
    If `HF_HUB_DISABLE_UPDATE_CHECK` is set, the check is skipped entirely.

    This function is called at the entry point of the CLI. It only performs the check once every 24 hours, and any error
    during the check is caught and logged, to avoid breaking the CLI.

    Args:
        library: The library to check for updates. Currently supports "huggingface_hub" and "transformers".
    """
    try:
        _check_cli_update(library)
    except Exception:
        # We don't want the CLI to fail on version checks, no matter the reason.
        logger.debug("Error while checking for CLI update.", exc_info=True)


def _check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    if constants.HF_HUB_DISABLE_UPDATE_CHECK:
        return

    current_version = importlib.metadata.version(library)

    # Skip if current version is a pre-release or dev version
    if any(tag in current_version for tag in ["rc", "dev"]):
        return

    # Skip if already checked in the last 24 hours
    if os.path.exists(constants.CHECK_FOR_UPDATE_DONE_PATH):
        mtime = os.path.getmtime(constants.CHECK_FOR_UPDATE_DONE_PATH)
        if (time.time() - mtime) < 24 * 3600:
            return

    # Touch the file to mark that we did the check now
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).touch()

    # Check latest version from the appropriate registry
    if library == "huggingface_hub" and installation_method() == "brew":
        latest_version = _fetch_latest_brew_version()
    else:
        latest_version = _fetch_latest_pypi_version(library)
    if latest_version is None or current_version == latest_version:
        return

    if library == "huggingface_hub":
        update_command = _get_huggingface_hub_update_command()
    else:
        update_command = _get_transformers_update_command()

    message = f"A new version of {library} ({latest_version}) is available! You are using version {current_version}."
    if update_command is not None:
        match library:
            case "huggingface_hub":
                message += "\nTo update, run: hf update"
            case _:
                message += f"\nTo update, run: {' '.join(update_command)}"
    out.hint(message)


def _fetch_latest_pypi_version(library: str) -> str | None:
    """Fetch the latest version of a library from PyPI. Returns None if the request fails."""
    try:
        from huggingface_hub.utils import get_session, hf_raise_for_status

        response = get_session().get(f"https://pypi.org/pypi/{library}/json", timeout=2)
        hf_raise_for_status(response)
        return response.json()["info"]["version"]
    except Exception:
        logger.debug("Error while fetching latest version from PyPI.", exc_info=True)
        return None


def _fetch_latest_brew_version() -> str | None:
    """Fetch the latest version of the `hf` formula from the Homebrew registry. Returns None if the request fails."""
    try:
        from huggingface_hub.utils import get_session, hf_raise_for_status

        response = get_session().get("https://formulae.brew.sh/api/formula/hf.json", timeout=2)
        hf_raise_for_status(response)
        return response.json()["versions"]["stable"]
    except Exception:
        logger.debug("Error while fetching latest version from Homebrew.", exc_info=True)
        return None


def run_update(*, exclude_skill: bool = False) -> int:
    """Run the install-method-appropriate update command for the `hf` CLI.

    Set `exclude_skill` to tell the standalone installer not to (re)install the `hf-cli` skill.

    Raises CLIError if the installation method can't be determined.
    Returns the subprocess exit code on success/failure of the update itself.
    """
    cmd = _get_huggingface_hub_update_command(exclude_skill=exclude_skill)
    if cmd is None:
        raise CLIError(
            "Cannot determine how to update huggingface_hub (unknown installation method). Please update manually."
        )
    return subprocess.call(cmd)


def _get_huggingface_hub_update_command(*, exclude_skill: bool = False) -> list[str] | None:
    """Return the command to update huggingface_hub as an argv list, or None if the installation method is unknown."""
    match installation_method():
        case "brew":
            return ["brew", "upgrade", "hf"]
        case "hf_installer" if os.name == "nt":
            # `iwr ... | iex` cannot take parameters: create a scriptblock so flags reach the installer.
            flags = " -ExcludeSkill" if exclude_skill else ""
            script = f"& ([scriptblock]::Create((iwr -useb https://hf.co/cli/install.ps1))){flags}"
            return ["powershell", "-NoProfile", "-Command", script]
        case "hf_installer":
            flags = " -s -- --exclude-skill" if exclude_skill else " -"
            return ["bash", "-c", f"curl -LsSf https://hf.co/cli/install.sh | bash{flags}"]
        case "pip":
            return [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"]
        case _:
            return None


def _get_transformers_update_command() -> list[str] | None:
    """Return the command to update transformers as an argv list, or None if the installation method is unknown."""
    match installation_method():
        case "hf_installer" if os.name == "nt":
            return [
                "powershell",
                "-NoProfile",
                "-Command",
                "& ([scriptblock]::Create((iwr -useb https://hf.co/cli/install.ps1))) -WithTransformers",
            ]
        case "hf_installer":
            return ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash -s -- --with-transformers"]
        case "pip":
            return [sys.executable, "-m", "pip", "install", "-U", "transformers"]
        case _:
            return None
<|endoftext|># Copyright 202-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains command to download files from the Hub with the CLI."""

import warnings
from typing import Annotated

from huggingface_hub import constants
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.errors import CLIError
from huggingface_hub.file_download import DryRunFileInfo, hf_hub_download
from huggingface_hub.utils import _format_size, parse_hf_uri

from ._cli_utils import RepoIdArg, RepoType, RepoTypeOptionalOpt, RevisionOpt, TokenOpt
from ._framework import Argument, Option
from ._output import out


DOWNLOAD_EXAMPLES = [
    "hf download meta-llama/Llama-3.2-1B-Instruct",
    "hf download meta-llama/Llama-3.2-1B-Instruct config.json tokenizer.json",
    'hf download meta-llama/Llama-3.2-1B-Instruct --include "*.safetensors" --exclude "*.bin"',
    "hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama",
    "hf download HuggingFaceM4/FineVision art/ --repo-type dataset",
    "hf download hf://datasets/HuggingFaceH4/ultrachat_200k",
]


def download(
    repo_id: RepoIdArg,
    filenames: Annotated[
        list[str] | None,
        Argument(
            help="Files to download (e.g. `config.json`, `data/metadata.jsonl`).",
        ),
    ] = None,
    repo_type: RepoTypeOptionalOpt = None,
    revision: RevisionOpt = None,
    include: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to include from files to download. eg: *.json",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to exclude from files to download.",
        ),
    ] = None,
    cache_dir: Annotated[
        str | None,
        Option(
            help="Directory where to save files.",
        ),
    ] = None,
    local_dir: Annotated[
        str | None,
        Option(
            help="If set, the downloaded file will be placed under this directory. Check out https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder for more details.",
        ),
    ] = None,
    force_download: Annotated[
        bool,
        Option(
            help="If True, the files will be downloaded even if they are already cached.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="If True, perform a dry run without actually downloading the file.",
        ),
    ] = False,
    token: TokenOpt = None,
    max_workers: Annotated[
        int,
        Option(
            help="Maximum number of workers to use for downloading files. Default is 8.",
        ),
    ] = 8,
) -> None:
    """Download files from the Hub."""
    if local_dir is not None and cache_dir is not None:
        raise CLIError(
            "Cannot use both `--local-dir` and `--cache-dir` at the same time. "
            "Use `--cache-dir` (or set the HF_HOME environment variable) for shared caching, "
            "or `--local-dir` for a one-off download to a specific directory."
        )

    # `repo_id` may be a plain repo id or an `hf://` URI (e.g. `hf://datasets/my-org/my-dataset@v1.0/data/`).
    # When a URI is provided, it is authoritative for the repo type, revision and (optionally) file path,
    # so explicit `--repo-type` / `--revision` options are forbidden alongside it.
    # We branch on the `hf://` prefix (the user's *intent*) rather than on whether the string parses as a
    # valid URI: a malformed URI then surfaces a precise `HfUriError` (formatted globally in `cli/_errors.py`)
    # instead of silently falling through to the plain-repo-id path and failing later with an opaque error.
    if repo_id.startswith(constants.HF_PROTOCOL):
        if repo_type is not None:
            raise CLIError(f"'--repo-type' cannot be used with an 'hf://' URI ('{repo_id}').")
        if revision is not None:
            raise CLIError(f"'--revision' cannot be used with an 'hf://' URI ('{repo_id}').")
        uri = parse_hf_uri(repo_id)
        if uri.is_bucket:
            raise CLIError("Buckets are not supported by `hf download`. Use `hf sync` instead.")
        # The URI parser strips trailing slashes, but `hf download` uses a trailing '/' to denote a subfolder
        # download (e.g. `data/` -> `data/**`). Re-append it when the URI explicitly ended with '/' so a folder
        # URI keeps routing through the subfolder code path below.
        path_in_repo = uri.path_in_repo
        if path_in_repo and repo_id.endswith("/"):
            path_in_repo += "/"
        repo_id, repo_type_str, revision = uri.id, uri.type, uri.revision
        if path_in_repo:
            if filenames:
                raise CLIError(
                    f"Cannot combine a file path in the hf:// URI ('{path_in_repo}') with positional filenames {filenames}."
                )
            filenames = [path_in_repo]
    else:
        repo_type_str = (repo_type or RepoType.model).value

    def run_download() -> str | DryRunFileInfo | list[DryRunFileInfo]:
        filenames_list = filenames if filenames is not None else []

        # Separate subfolder patterns (ending with '/') from regular filenames
        # Subfolders like "art/" are converted to include patterns like "art/**"
        subfolders = [f for f in filenames_list if f.endswith("/")]
        subfolder_patterns = [f"{f.rstrip('/')}/**" for f in subfolders]
        regular_filenames = [f for f in filenames_list if not f.endswith("/")]

        # Error if subfolder patterns are combined with --include/--exclude
        # Guide user to use --include instead of subfolder argument
        if len(subfolder_patterns) > 0:
            if include is not None and len(include) > 0:
                raise CLIError(
                    f"Cannot combine subfolder argument ('{subfolders[0]}') with `--include`. "
                    f'Please use `--include "{subfolders[0]}*"` instead.'
                )
            if exclude is not None and len(exclude) > 0:
                raise CLIError(
                    f"Cannot combine subfolder argument ('{subfolders[0]}') with `--exclude`. "
                    f'Please use `--include "{subfolders[0]}*"` with `--exclude` instead.'
                )

        # Warn user if patterns are ignored (only if regular filenames are provided)
        if len(regular_filenames) > 0:
            if include is not None and len(include) > 0:
                warnings.warn("Ignoring `--include` since filenames have been explicitly set.")
            if exclude is not None and len(exclude) > 0:
                warnings.warn("Ignoring `--exclude` since filenames have been explicitly set.")

        # Single file to download (not a subfolder): use `hf_hub_download`
        if len(regular_filenames) == 1 and len(subfolder_patterns) == 0:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type_str,
                revision=revision,
                filename=regular_filenames[0],
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_dir=local_dir,
                library_name="huggingface-cli",
                dry_run=dry_run,
            )

        # Otherwise: use `snapshot_download` to ensure all files comes from same revision
        if len(regular_filenames) == 0 and len(subfolder_patterns) == 0:
            # No filenames provided: use include/exclude patterns
            allow_patterns = include
            ignore_patterns = exclude
        else:
            # Combine regular filenames and subfolder patterns as allow_patterns
            allow_patterns = regular_filenames + subfolder_patterns
            ignore_patterns = None

        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type_str,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            cache_dir=cache_dir,
            token=token,
            local_dir=local_dir,
            library_name="huggingface-cli",
            max_workers=max_workers,
            dry_run=dry_run,
        )

    def _print_result(result: str | DryRunFileInfo | list[DryRunFileInfo]) -> None:
        if isinstance(result, str):
            out.result("Downloaded", path=result)
            return

        # Print dry run info
        if isinstance(result, DryRunFileInfo):
            result = [result]
        will_download = [r for r in result if r.will_download]
        total_size = (
            None if any(r.file_size is None for r in will_download) else sum(r.file_size or 0 for r in will_download)
        )
        out.text(
            f"[dry-run] Will download {len(will_download)} files"
            f" (out of {len(result)})"
            f" totalling {_format_size(total_size) if total_size is not None else 'an unknown size'}."
        )
        items = [
            {
                "file": info.filename,
                "size": _format_size(info.file_size) if info.will_download and info.file_size is not None else "-",
            }
            for info in sorted(result, key=lambda x: x.filename)
        ]
        out.table(items)

    _print_result(run_download())
<|endoftext|># Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared ``cp`` command to copy files between local paths, repositories and buckets.

This single command backs three identical CLI entry points: ``hf cp`` (top-level),
``hf repos cp`` and ``hf buckets cp``. It supports any source/destination combination
of local file, repo/bucket ``hf://`` URI, and ``-`` (stdin/stdout), with two exceptions:
- bucket-to-repo copies are not supported (server limitation), and
- local-to-local copies (use a regular ``cp`` for that).
"""

import os
import sys
from dataclasses import replace
from typing import Annotated, Literal

import click

from huggingface_hub import HfApi
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import HfUri, SoftTemporaryDirectory, disable_progress_bars, is_hf_uri, parse_hf_uri

from ._cli_utils import TokenOpt, get_hf_api
from ._framework import Argument
from ._output import out


CP_EXAMPLES = [
    # Download (repo or bucket -> local / stdout)
    "hf cp hf://username/my-model/config.json",
    "hf cp hf://username/my-model/config.json ./config.json",
    "hf cp hf://datasets/username/my-dataset/data.csv ./data/",
    "hf cp hf://buckets/username/my-bucket/config.json -",
    # Upload (local / stdin -> repo or bucket)
    "hf cp ./model.safetensors hf://username/my-model/model.safetensors",
    "hf cp ./config.json hf://buckets/username/my-bucket/logs/",
    "hf cp - hf://buckets/username/my-bucket/config.json",
    # Remote to remote (repo/bucket -> repo/bucket, server-side when possible)
    "hf cp hf://username/source-model/ hf://username/dest-model/",
    "hf cp hf://datasets/username/my-dataset/processed/ hf://buckets/username/my-bucket/processed/",
    "hf cp hf://buckets/username/my-bucket/logs/ hf://buckets/username/archive-bucket/  # copies contents only",
]


# Which alias registered the command, used to restrict the remote endpoint type (see `_enforce_context`).
CpContext = Literal["repos", "buckets"]


def make_cp(context: CpContext | None = None):
    """Build the ``cp`` command function for a given alias.

    The three entry points (`hf cp`, `hf repos cp`, `hf buckets cp`) share the exact same logic;
    'context' only adds a guardrail on the remote endpoint type (see `_enforce_context`).
    """

    def cp(
        src: Annotated[
            str,
            Argument(help="Source: local file, hf:// URI (repo or bucket), or - for stdin."),
        ],
        dst: Annotated[
            str | None,
            Argument(help="Destination: local path, hf:// URI (repo or bucket), or - for stdout."),
        ] = None,
        token: TokenOpt = None,
    ) -> None:
        """Copy files between local paths, repositories, and buckets.

        Handles uploads (local/stdin -> repo/bucket), downloads (repo/bucket -> local/stdout) and
        remote-to-remote copies (repo/bucket -> repo/bucket). Bucket-to-repo and local-to-local
        copies are not supported. For directories, use `hf upload`/`hf download` (repos) or
        `hf buckets sync` (buckets). Remote-to-remote copies only work within the same storage
        region (https://huggingface.co/docs/hub/storage-regions).
        """
        _enforce_context(context, src, dst)
        _run_cp(src, dst, token)

    return cp


def _enforce_context(context: CpContext | None, src: str, dst: str | None) -> None:
    """Guardrail for the `hf repos cp` / `hf buckets cp` aliases.

    These aliases are exact duplicates of `hf cp`, so a bare `hf repos cp` could otherwise touch a
    bucket (and vice versa). We validate the type of the remote side: the destination for uploads and
    remote-to-remote copies, or the source when downloading to a local path / stdout. The top-level
    `hf cp` (i.e. 'context' is None) accepts any combination.
    """
    if context is None:
        return
    # The remote endpoint is the destination when it is an hf:// URI, otherwise the source (download).
    remote = dst if (dst is not None and is_hf_uri(dst)) else src
    if not is_hf_uri(remote):
        return
    if context == "repos" and parse_hf_uri(remote).is_bucket:
        raise CLIError("`hf repos cp` only works with repositories. Use `hf cp` or `hf buckets cp` for buckets.")
    if context == "buckets" and not parse_hf_uri(remote).is_bucket:
        raise CLIError("`hf buckets cp` only works with buckets. Use `hf cp` or `hf repos cp` for repositories.")


def _run_cp(src: str, dst: str | None, token: str | None) -> None:
    api = get_hf_api(token=token)

    src_is_stdin = src == "-"
    dst_is_stdout = dst == "-"
    src_is_hf = is_hf_uri(src)
    dst_is_hf = dst is not None and is_hf_uri(dst)

    # --- Remote to remote: delegate to copy_files (repo/bucket -> repo/bucket) ---
    if src_is_hf and dst_is_hf:
        assert dst is not None  # guaranteed by dst_is_hf
        api.copy_files(src, dst)
        out.result("Copied", src=src, dst=dst)
        return

    # --- At least one side must be a remote hf:// URI (rules out local->local, stdin->local, etc.) ---
    if not src_is_hf and not dst_is_hf:
        if dst is None:
            raise click.BadParameter("Missing destination. Provide a repo or bucket hf:// URI as DST.")
        raise click.BadParameter(
            "One of SRC or DST must be a repo (hf://username/...) or bucket (hf://buckets/...) URI."
        )

    # --- Download: repo/bucket -> local file or stdout ---
    if src_is_hf:
        if dst_is_stdout:
            _download_file_to_stdout(api, src)
            return
        _download_file_to_local(api, src, dst)
        return

    # --- Upload: local file or stdin -> repo/bucket ---
    assert dst is not None  # guaranteed: reaching here means dst_is_hf is True
    _upload_file_to_remote(api, src, dst, src_is_stdin=src_is_stdin)


def _download_file_to_stdout(api: HfApi, src: str) -> None:
    uri = parse_hf_uri(src)
    filename = _source_filename(uri, src)
    # Suppress progress bars to avoid polluting the piped output.
    with disable_progress_bars():
        with SoftTemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, filename)
            _download_single(api, uri, tmp_path)
            with open(tmp_path, "rb") as f:
                while chunk := f.read(32_000_000):  # 32MB chunks
                    sys.stdout.buffer.write(chunk)


def _download_file_to_local(api: HfApi, src: str, dst: str | None) -> None:
    uri = parse_hf_uri(src)
    filename = _source_filename(uri, src)

    if dst is None:
        local_path = filename
    elif os.path.isdir(dst) or dst.endswith(os.sep) or dst.endswith("/"):
        local_path = os.path.join(dst, filename)
    else:
        local_path = dst

    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    _download_single(api, uri, local_path)
    out.result("Downloaded", src=src, dst=local_path)


def _download_single(api: HfApi, uri: HfUri, local_path: str) -> None:
    """Download a single file (repo or bucket) to ``local_path``.

    Used by `_download_file_to_local` and `_download_file_to_stdout`.
    """
    if uri.is_bucket:
        api.download_bucket_files(uri.id, [(uri.path_in_repo, local_path)], raise_on_missing_files=True)
    else:
        # Download into a temporary folder next to the destination (rather than the shared cache)
        # so the final move stays on the same filesystem and is instant. The temp folder is
        # cleaned up automatically once the move is complete.
        parent_dir = os.path.dirname(local_path) or "."
        with SoftTemporaryDirectory(prefix=".tmp", dir=parent_dir) as tmp_dir:
            downloaded_path = api.hf_hub_download(
                repo_id=uri.id,
                repo_type=uri.type,
                filename=uri.path_in_repo,
                revision=uri.revision,
                local_dir=tmp_dir,
            )
            os.replace(downloaded_path, local_path)


def _source_filename(uri: HfUri, src: str) -> str:
    if uri.path_in_repo == "" or src.endswith("/"):
        raise click.BadParameter(
            "Source path must include a file name, not just a repo/bucket or directory path."
            " Use `hf download` or `hf buckets sync` to copy directories."
        )
    return uri.path_in_repo.rsplit("/", 1)[-1]


def _upload_file_to_remote(api: HfApi, src: str, dst: str, *, src_is_stdin: bool) -> None:
    uri = parse_hf_uri(dst)

    if src_is_stdin:
        if uri.path_in_repo == "" or dst.endswith("/"):
