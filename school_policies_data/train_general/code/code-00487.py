    #    (command_name : string, predicate : unbound_method | string | None)
    # tuples, where 'predicate' is a method of the parent command that
    # determines whether the corresponding command is applicable in the
    # current situation.  (Eg. we "install_headers" is only applicable if
    # we have any C header files to install.)  If 'predicate' is None,
    # that command is always applicable.
    #
    # 'sub_commands' is usually defined at the *end* of a class, because
    # predicates can be unbound methods, so they must already have been
    # defined.  The canonical example is the "install" command.
    sub_commands: ClassVar[  # Any to work around variance issues
        list[tuple[str, Callable[[Any], bool] | None]]
    ] = []

    user_options: ClassVar[
        # Specifying both because list is invariant. Avoids mypy override assignment issues
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = []

    # -- Creation/initialization methods -------------------------------

    def __init__(self, dist: Distribution) -> None:
        """Create and initialize a new Command object.  Most importantly,
        invokes the 'initialize_options()' method, which is the real
        initializer and depends on the actual command being
        instantiated.
        """
        # late import because of mutual dependence between these classes
        from distutils.dist import Distribution

        if not isinstance(dist, Distribution):
            raise TypeError("dist must be a Distribution instance")
        if self.__class__ is Command:
            raise RuntimeError("Command is an abstract class")

        self.distribution = dist
        self.initialize_options()

        # Per-command versions of the global flags, so that the user can
        # customize Distutils' behaviour command-by-command and let some
        # commands fall back on the Distribution's behaviour. None means
        # "not defined, check self.distribution's copy".

        # verbose is largely ignored, but needs to be set for
        # backwards compatibility (I think)?
        self.verbose = dist.verbose

        # Some commands define a 'self.force' option to ignore file
        # timestamps, but methods defined *here* assume that
        # 'self.force' exists for all commands.  So define it here
        # just to be safe.
        self.force: bool | None = None

        # The 'help' flag is just used for command-line parsing, so
        # none of that complicated bureaucracy is needed.
        self.help = False

        # 'finalized' records whether or not 'finalize_options()' has been
        # called.  'finalize_options()' itself should not pay attention to
        # this flag: it is the business of 'ensure_finalized()', which
        # always calls 'finalize_options()', to respect/update it.
        self.finalized = False

    def ensure_finalized(self) -> None:
        if not self.finalized:
            self.finalize_options()
        self.finalized = True

    # Subclasses must define:
    #   initialize_options()
    #     provide default values for all options; may be customized by
    #     setup script, by options from config file(s), or by command-line
    #     options
    #   finalize_options()
    #     decide on the final values for all options; this is called
    #     after all possible intervention from the outside world
    #     (command-line, option file, etc.) has been processed
    #   run()
    #     run the command: do whatever it is we're here to do,
    #     controlled by the command's various option values

    @abstractmethod
    def initialize_options(self) -> None:
        """Set default values for all the options that this command
        supports.  Note that these defaults may be overridden by other
        commands, by the setup script, by config files, or by the
        command-line.  Thus, this is not the place to code dependencies
        between options; generally, 'initialize_options()' implementations
        are just a bunch of "self.foo = None" assignments.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    @abstractmethod
    def finalize_options(self) -> None:
        """Set final values for all the options that this command supports.
        This is always called as late as possible, ie.  after any option
        assignments from the command-line or from other commands have been
        done.  Thus, this is the place to code option dependencies: if
        'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
        long as 'foo' still has the same value it was assigned in
        'initialize_options()'.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    def dump_options(self, header=None, indent=""):
        from distutils.fancy_getopt import longopt_xlate

        if header is None:
            header = f"command options for '{self.get_command_name()}':"
        self.announce(indent + header, level=logging.INFO)
        indent = indent + "  "
        for option, _, _ in self.user_options:
            option = option.translate(longopt_xlate)
            if option[-1] == "=":
                option = option[:-1]
            value = getattr(self, option)
            self.announce(indent + f"{option} = {value}", level=logging.INFO)

    @abstractmethod
    def run(self) -> None:
        """A command's raison d'etre: carry out the action it exists to
        perform, controlled by the options initialized in
        'initialize_options()', customized by other commands, the setup
        script, the command-line, and config files, and finalized in
        'finalize_options()'.  All terminal output and filesystem
        interaction should be done by 'run()'.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    def announce(self, msg: object, level: int = logging.DEBUG) -> None:
        log.log(level, msg)

    def debug_print(self, msg: object) -> None:
        """Print 'msg' to stdout if the global DEBUG (taken from the
        DISTUTILS_DEBUG environment variable) flag is true.
        """
        from distutils.debug import DEBUG

        if DEBUG:
            print(msg)
            sys.stdout.flush()

    # -- Option validation methods -------------------------------------
    # (these are very handy in writing the 'finalize_options()' method)
    #
    # NB. the general philosophy here is to ensure that a particular option
    # value meets certain type and value constraints.  If not, we try to
    # force it into conformance (eg. if we expect a list but have a string,
    # split the string on comma and/or whitespace).  If we can't force the
    # option into conformance, raise DistutilsOptionError.  Thus, command
    # classes need do nothing more than (eg.)
    #   self.ensure_string_list('foo')
    # and they can be guaranteed that thereafter, self.foo will be
    # a list of strings.

    def _ensure_stringlike(self, option, what, default=None):
        val = getattr(self, option)
        if val is None:
            setattr(self, option, default)
            return default
        elif not isinstance(val, str):
            raise DistutilsOptionError(f"'{option}' must be a {what} (got `{val}`)")
        return val

    def ensure_string(self, option: str, default: str | None = None) -> None:
        """Ensure that 'option' is a string; if not defined, set it to
        'default'.
        """
        self._ensure_stringlike(option, "string", default)

    def ensure_string_list(self, option: str) -> None:
        r"""Ensure that 'option' is a list of strings.  If 'option' is
        currently a string, we split it either on /,\s*/ or /\s+/, so
        "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become
        ["foo", "bar", "baz"].
        """
        val = getattr(self, option)
        if val is None:
            return
        elif isinstance(val, str):
            setattr(self, option, re.split(r',\s*|\s+', val))
        else:
            if isinstance(val, list):
                ok = all(isinstance(v, str) for v in val)
            else:
                ok = False
            if not ok:
                raise DistutilsOptionError(
                    f"'{option}' must be a list of strings (got {val!r})"
                )

    def _ensure_tested_string(self, option, tester, what, error_fmt, default=None):
        val = self._ensure_stringlike(option, what, default)
        if val is not None and not tester(val):
            raise DistutilsOptionError(
                ("error in '%s' option: " + error_fmt) % (option, val)
            )

    def ensure_filename(self, option: str) -> None:
        """Ensure that 'option' is the name of an existing file."""
        self._ensure_tested_string(
            option, os.path.isfile, "filename", "'%s' does not exist or is not a file"
        )

    def ensure_dirname(self, option: str) -> None:
        self._ensure_tested_string(
            option,
            os.path.isdir,
            "directory name",
            "'%s' does not exist or is not a directory",
        )

    # -- Convenience methods for commands ------------------------------

    def get_command_name(self) -> str:
        if hasattr(self, 'command_name'):
            return self.command_name
        else:
            return self.__class__.__name__

    def set_undefined_options(
        self, src_cmd: str, *option_pairs: tuple[str, str]
    ) -> None:
        """Set the values of any "undefined" options from corresponding
        option values in some other command object.  "Undefined" here means
        "is None", which is the convention used to indicate that an option
        has not been changed between 'initialize_options()' and
        'finalize_options()'.  Usually called from 'finalize_options()' for
        options that depend on some other command rather than another
        option of the same command.  'src_cmd' is the other command from
        which option values will be taken (a command object will be created
        for it if necessary); the remaining arguments are
        '(src_option,dst_option)' tuples which mean "take the value of
        'src_option' in the 'src_cmd' command object, and copy it to
        'dst_option' in the current command object".
        """
        # Option_pairs: list of (src_option, dst_option) tuples
        src_cmd_obj = self.distribution.get_command_obj(src_cmd)
        src_cmd_obj.ensure_finalized()
        for src_option, dst_option in option_pairs:
            if getattr(self, dst_option) is None:
                setattr(self, dst_option, getattr(src_cmd_obj, src_option))

    # NOTE: Because distutils is private to Setuptools and not all commands are exposed here,
    # not every possible command is enumerated in the signature.
    def get_finalized_command(self, command: str, create: bool = True) -> Command:
        """Wrapper around Distribution's 'get_command_obj()' method: find
        (create if necessary and 'create' is true) the command object for
        'command', call its 'ensure_finalized()' method, and return the
        finalized command object.
        """
        # TODO: Raise a more descriptive error when create=False or cmd_obj is None ?
        cmd_obj = cast(Command, self.distribution.get_command_obj(command, create))
        cmd_obj.ensure_finalized()
        return cmd_obj

    # XXX rename to 'get_reinitialized_command()'? (should do the
    # same in dist.py, if so)
    @overload
    def reinitialize_command(
        self, command: str, reinit_subcommands: bool = False
    ) -> Command: ...
    @overload
    def reinitialize_command(
        self, command: _CommandT, reinit_subcommands: bool = False
    ) -> _CommandT: ...
    def reinitialize_command(
        self, command: str | Command, reinit_subcommands=False
    ) -> Command:
        return self.distribution.reinitialize_command(command, reinit_subcommands)

    def run_command(self, command: str) -> None:
        """Run some other command: uses the 'run_command()' method of
        Distribution, which creates and finalizes the command object if
        necessary and then invokes its 'run()' method.
        """
        self.distribution.run_command(command)

    def get_sub_commands(self) -> list[str]:
        """Determine the sub-commands that are relevant in the current
        distribution (ie., that need to be run).  This is based on the
        'sub_commands' class attribute: each tuple in that list may include
        a method that we call to determine if the subcommand needs to be
        run for the current distribution.  Return a list of command names.
        """
        commands = []
        for cmd_name, method in self.sub_commands:
            if method is None or method(self):
                commands.append(cmd_name)
        return commands

    # -- External world manipulation -----------------------------------

    def warn(self, msg: object) -> None:
        log.warning("warning: %s: %s\n", self.get_command_name(), msg)

    def execute(
        self,
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        msg: object = None,
        level: int = 1,
    ) -> None:
        util.execute(func, args, msg)

    def mkpath(self, name: str, mode: int = 0o777) -> None:
        dir_util.mkpath(name, mode)

    @overload
    def copy_file(
        self,
        infile: str | os.PathLike[str],
        outfile: _StrPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[_StrPathT | str, bool]: ...
    @overload
    def copy_file(
        self,
        infile: bytes | os.PathLike[bytes],
        outfile: _BytesPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[_BytesPathT | bytes, bool]: ...
    def copy_file(
        self,
        infile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        outfile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[str | os.PathLike[str] | bytes | os.PathLike[bytes], bool]:
        """Copy a file respecting verbose, dry-run and force flags.  (The
        former two default to whatever is in the Distribution object, and
        the latter defaults to false for commands that don't define it.)"""
        return file_util.copy_file(
            infile,
            outfile,
            preserve_mode,
            preserve_times,
            not self.force,
            link,
        )

    def copy_tree(
        self,
        infile: str | os.PathLike[str],
        outfile: str,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        preserve_symlinks: bool = False,
        level: int = 1,
    ) -> list[str]:
        """Copy an entire directory tree respecting verbose, dry-run,
        and force flags.
        """
        return dir_util.copy_tree(
            infile,
            outfile,
            preserve_mode,
            preserve_times,
            preserve_symlinks,
            not self.force,
        )

    @overload
    def move_file(
        self, src: str | os.PathLike[str], dst: _StrPathT, level: int = 1
    ) -> _StrPathT | str: ...
    @overload
    def move_file(
        self, src: bytes | os.PathLike[bytes], dst: _BytesPathT, level: int = 1
    ) -> _BytesPathT | bytes: ...
    def move_file(
        self,
        src: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        dst: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        level: int = 1,
    ) -> str | os.PathLike[str] | bytes | os.PathLike[bytes]:
        """Move a file respecting dry-run flag."""
        return file_util.move_file(src, dst)

    def spawn(
        self, cmd: MutableSequence[str], search_path: bool = True, level: int = 1
    ) -> None:
        """Spawn an external command respecting dry-run flag."""
        from distutils.spawn import spawn

        spawn(cmd)

    @overload
    def make_archive(
        self,
        base_name: str,
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str: ...
    @overload
    def make_archive(
        self,
        base_name: str | os.PathLike[str],
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str: ...
    def make_archive(
        self,
        base_name: str | os.PathLike[str],
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str:
        return archive_util.make_archive(  # type: ignore[misc] # Mypy bailed out
            base_name,
            format,
            root_dir,
            base_dir,
            owner=owner,
            group=group,
        )

    def make_file(
        self,
        infiles: str | list[str] | tuple[str, ...],
        outfile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        exec_msg: object = None,
        skip_msg: object = None,
        level: int = 1,
    ) -> None:
        """Special case of 'execute()' for operations that process one or
        more input files and generate one output file.  Works just like
        'execute()', except the operation is skipped and a different
        message printed if 'outfile' already exists and is newer than all
        files listed in 'infiles'.  If the command defined 'self.force',
        and it is true, then the command is unconditionally run -- does no
        timestamp checks.
        """
        if skip_msg is None:
            skip_msg = f"skipping {outfile!r} (inputs unchanged)"

        # Allow 'infiles' to be a single string
        if isinstance(infiles, str):
            infiles = (infiles,)
        elif not isinstance(infiles, (list, tuple)):
            raise TypeError("'infiles' must be a string, or a list or tuple of strings")

        if exec_msg is None:
            exec_msg = f"generating {outfile!r} from {', '.join(infiles)!r}"

        # If 'outfile' must be regenerated (either because it doesn't
        # exist, is out-of-date, or the 'force' flag is true) then
        # perform the action that presumably regenerates it
        if self.force or _modified.newer_group(infiles, outfile):
            self.execute(func, args, exec_msg, level)
        # Otherwise, print the "skip" message
        else:
            log.debug(skip_msg)
<|endoftext|>"""distutils.util

Miscellaneous utility functions -- anything that doesn't fit into
one of the other *util.py modules.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import re
import string
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from jaraco.functools import pass_none

from ._log import log
from ._modified import newer
from .compilers.platform import detect, macos
from .errors import DistutilsByteCompileError, DistutilsPlatformError
from .spawn import spawn

if TYPE_CHECKING:
    from typing_extensions import TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

# Platform and macOS helpers now live in the compilers package; these names are
# re-exported for backward compatibility.
get_host_platform = detect.get_host_platform
get_platform = detect.get_platform
is_mingw = detect.is_mingw
MACOSX_VERSION_VAR = macos.VERSION_VAR
get_macosx_target_ver = macos.target_ver
get_macosx_target_ver_from_syscfg = macos._target_ver_from_syscfg
_clear_cached_macosx_ver = macos._clear_cached_target_ver


def split_version(s: str) -> list[int]:
    """Convert a dot-separated string into a list of numbers for comparisons"""
    return [int(n) for n in s.split('.')]


@pass_none
def convert_path(pathname: str | os.PathLike[str]) -> str:
    r"""
    Allow for pathlib.Path inputs, coax to a native path string.

    If None is passed, will just pass it through as
    Setuptools relies on this behavior.

    >>> convert_path(None) is None
    True

    Removes empty paths.

    >>> convert_path('foo/./bar').replace('\\', '/')
    'foo/bar'
    """
    return os.fspath(pathlib.PurePath(pathname))


def change_root(
    new_root: str | os.PathLike[str],
    pathname: str | os.PathLike[str],
) -> str:
    """Return 'pathname' with 'new_root' prepended.  If 'pathname' is
    relative, this is equivalent to "os.path.join(new_root,pathname)".
    Otherwise, it requires making 'pathname' relative and then joining the
    two, which is tricky on DOS/Windows and Mac OS.
    """
    if os.name == 'posix':
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            # type-ignore: This makes bytes-based paths unsupported in this branch.
            # Either this or we don't support absolute os.Pathlike, or we complexify this branch.
            return os.path.join(new_root, str(pathname)[1:])

    elif os.name == 'nt':
        (_drive, path) = os.path.splitdrive(pathname)
        if path[0] == os.sep:
            path = path[1:]
        return os.path.join(new_root, path)

    raise DistutilsPlatformError(f"nothing known about platform '{os.name}'")


@functools.lru_cache
def check_environ() -> None:
    """Ensure that 'os.environ' has all the environment variables we
    guarantee that users can use in config files, command-line options,
    etc.  Currently this includes:
      HOME - user's home directory (Unix only)
      PLAT - description of the current platform, including hardware
             and OS (see 'get_platform()')
    """
    if os.name == 'posix' and 'HOME' not in os.environ:
        try:
            import pwd

            os.environ['HOME'] = pwd.getpwuid(os.getuid())[5]
        except (ImportError, KeyError):
            # bpo-10496: if the current user identifier doesn't exist in the
            # password database, do nothing
            pass

    if 'PLAT' not in os.environ:
        os.environ['PLAT'] = get_platform()


def subst_vars(s, local_vars: Mapping[str, object]) -> str:
    """
    Perform variable substitution on 'string'.
    Variables are indicated by format-style braces ("{var}").
    Variable is substituted by the value found in the 'local_vars'
    dictionary or in 'os.environ' if it's not in 'local_vars'.
    'os.environ' is first checked/augmented to guarantee that it contains
    certain values: see 'check_environ()'.  Raise ValueError for any
    variables not found in either 'local_vars' or 'os.environ'.
    """
    check_environ()
    lookup = dict(os.environ)
    lookup.update((name, str(value)) for name, value in local_vars.items())
    try:
        return _subst_compat(s).format_map(lookup)
    except KeyError as var:
        raise ValueError(f"invalid variable {var}")


def _subst_compat(s):
    """
    Replace shell/Perl-style variable substitution with
    format-style. For compatibility.
    """

    def _subst(match):
        return f'{{{match.group(1)}}}'

    repl = re.sub(r'\$([a-zA-Z_][a-zA-Z_0-9]*)', _subst, s)
    if repl != s:
        import warnings

        warnings.warn(
            "shell/Perl-style substitutions are deprecated",
            DeprecationWarning,
        )
    return repl


def grok_environment_error(exc: object, prefix: str = "error: ") -> str:
    # Function kept for backward compatibility.
    # Used to try clever things with EnvironmentErrors,
    # but nowadays str(exception) produces good messages.
    return prefix + str(exc)


# Needed by 'split_quoted()'
_wordchars_re = re.compile(rf'[^\\\'\"{string.whitespace} ]*')
_squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
_dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')


def split_quoted(s: str) -> list[str]:
    """Split a string up according to Unix shell-like rules for quotes and
    backslashes.  In short: words are delimited by spaces, as long as those
    spaces are not escaped by a backslash, or inside a quoted string.
    Single and double quotes are equivalent, and the quote characters can
    be backslash-escaped.  The backslash is stripped from any two-character
    escape sequence, leaving only the escaped character.  The quote
    characters are stripped from any quoted string.  Returns a list of
    words.
    """

    # This is a nice algorithm for splitting up a single string, since it
    # doesn't require character-by-character examination.  It was a little
    # bit of a brain-bender to get it working right, though...

    s = s.strip()
    words = []
    pos = 0

    while s:
        m = _wordchars_re.match(s, pos)
        assert m is not None
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break

        if s[end] in string.whitespace:
            # unescaped, unquoted whitespace: now
            # we definitely have a word delimiter
            words.append(s[:end])
            s = s[end:].lstrip()
            pos = 0

        elif s[end] == '\\':
            # preserve whatever is being escaped;
            # will become part of the current word
            s = s[:end] + s[end + 1 :]
            pos = end + 1

        else:
            if s[end] == "'":  # slurp singly-quoted string
                m = _squote_re.match(s, end)
            elif s[end] == '"':  # slurp doubly-quoted string
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError(f"this can't happen (bad char '{s[end]}')")

            if m is None:
                raise ValueError(f"bad string (mismatched {s[end]} quotes?)")

            (beg, end) = m.span()
            s = s[:beg] + s[beg + 1 : end - 1] + s[end:]
            pos = m.end() - 2

        if pos >= len(s):
            words.append(s)
            break

    return words


def execute(
    func: Callable[[Unpack[_Ts]], object],
    args: tuple[Unpack[_Ts]],
    msg: object = None,
    verbose: bool = False,
) -> None:
    """
    Perform some action that affects the outside world (e.g. by
    writing to the filesystem). Was previously used to deal with
    "dry run" operations, but now runs unconditionally.
    """
    if msg is None:
        msg = f"{func.__name__}{args!r}"
        if msg[-2:] == ',)':  # correct for singleton tuple
            msg = msg[0:-2] + ')'

    log.info(msg)
    func(*args)


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


def byte_compile(  # noqa: C901
    py_files: Iterable[str],
    optimize: int = 0,
    force: bool = False,
    prefix: str | None = None,
    base_dir: str | None = None,
    verbose: bool = True,
    direct: bool | None = None,
) -> None:
    """Byte-compile a collection of Python source files to .pyc
    files in a __pycache__ subdirectory.  'py_files' is a list
    of files to compile; any files that don't end in ".py" are silently
    skipped.  'optimize' must be one of the following:
      0 - don't optimize
      1 - normal optimization (like "python -O")
      2 - extra optimization (like "python -OO")
    If 'force' is true, all files are recompiled regardless of
    timestamps.

    The source filename encoded in each bytecode file defaults to the
    filenames listed in 'py_files'; you can modify these with 'prefix' and
    'basedir'.  'prefix' is a string that will be stripped off of each
    source filename, and 'base_dir' is a directory name that will be
    prepended (after 'prefix' is stripped).  You can supply either or both
    (or neither) of 'prefix' and 'base_dir', as you wish.

    Byte-compilation is either done directly in this interpreter process
    with the standard py_compile module, or indirectly by writing a
    temporary script and executing it.  Normally, you should let
    'byte_compile()' figure out to use direct compilation or not (see
    the source for details).  The 'direct' flag is used by the script
    generated in indirect mode; unless you know what you're doing, leave
    it set to None.
    """

    # nothing is done if sys.dont_write_bytecode is True
    if sys.dont_write_bytecode:
        raise DistutilsByteCompileError('byte-compiling is disabled.')

    # First, if the caller didn't force us into direct or indirect mode,
    # figure out which mode we should be in.  We take a conservative
    # approach: choose direct mode *only* if the current interpreter is
    # in debug mode and optimize is 0.  If we're not in debug mode (-O
    # or -OO), we don't know which level of optimization this
    # interpreter is running with, so we can't do direct
    # byte-compilation and be certain that it's the right thing.  Thus,
    # always compile indirectly if the current interpreter is in either
    # optimize mode, or if either optimization level was requested by
    # the caller.
    if direct is None:
        direct = __debug__ and optimize == 0

    # "Indirect" byte-compilation: write a temporary script and then
    # run it with the appropriate flags.
    if not direct:
        (script_fd, script_name) = tempfile.mkstemp(".py")
        log.info("writing byte-compilation script '%s'", script_name)
        script = os.fdopen(script_fd, "w", encoding='utf-8')

        with script:
            script.write(
                """\
from distutils.util import byte_compile
files = [
"""
            )

            # XXX would be nice to write absolute filenames, just for
            # safety's sake (script should be more robust in the face of
            # chdir'ing before running it).  But this requires abspath'ing
            # 'prefix' as well, and that breaks the hack in build_lib's
            # 'byte_compile()' method that carefully tacks on a trailing
            # slash (os.sep really) to make sure the prefix here is "just
            # right".  This whole prefix business is rather delicate -- the
            # problem is that it's really a directory, but I'm treating it
            # as a dumb string, so trailing slashes and so forth matter.

            script.write(",\n".join(map(repr, py_files)) + "]\n")
            script.write(
                f"""
byte_compile(files, optimize={optimize!r}, force={force!r},
         prefix={prefix!r}, base_dir={base_dir!r},
         verbose={verbose!r},
         direct=True)
"""
            )

        cmd = [sys.executable]
        cmd.extend(subprocess._optim_args_from_interpreter_flags())
        cmd.append(script_name)
        spawn(cmd)
        execute(os.remove, (script_name,), f"removing {script_name}")

    # "Direct" byte-compilation: use the py_compile module to compile
    # right here, right now.  Note that the script generated in indirect
    # mode simply calls 'byte_compile()' in direct mode, a weird sort of
    # cross-process recursion.  Hey, it works!
    else:
        from py_compile import compile

        for file in py_files:
            if file[-3:] != ".py":
                # This lets us be lazy and not filter filenames in
                # the "install_lib" command.
                continue

            # Terminology from the py_compile module:
            #   cfile - byte-compiled file
            #   dfile - purported source filename (same as 'file' by default)
            if optimize >= 0:
                opt = '' if optimize == 0 else optimize
                cfile = importlib.util.cache_from_source(file, optimization=opt)
            else:
                cfile = importlib.util.cache_from_source(file)
            dfile = file
            if prefix:
                if file[: len(prefix)] != prefix:
                    raise ValueError(
                        f"invalid prefix: filename {file!r} doesn't start with {prefix!r}"
                    )
                dfile = dfile[len(prefix) :]
            if base_dir:
                dfile = os.path.join(base_dir, dfile)

            cfile_base = os.path.basename(cfile)
            if direct:
                if force or newer(file, cfile):
                    log.info("byte-compiling %s to %s", file, cfile_base)
                    compile(file, cfile, dfile)
                else:
                    log.debug("skipping byte-compilation of %s to %s", file, cfile_base)


def rfc822_escape(header: str) -> str:
    """Return a version of the string escaped for inclusion in an
    RFC-822 header, by ensuring there are 8 spaces space after each newline.
    """
    indent = 8 * " "
    lines = header.splitlines(keepends=True)

    # Emulate the behaviour of `str.split`
    # (the terminal line break in `splitlines` does not result in an extra line):
    ends_in_newline = lines and lines[-1].splitlines()[0] != lines[-1]
    suffix = indent if ends_in_newline else ""

    return indent.join(lines) + suffix


def is_freethreaded() -> bool:
    """Return True if the Python interpreter is built with free threading support."""
    return bool(sysconfig.get_config_var('Py_GIL_DISABLED'))
<|endoftext|>import warnings

from . import _modified


def __getattr__(name):
    if name not in ['newer', 'newer_group', 'newer_pairwise']:
        raise AttributeError(name)
    warnings.warn(
        "dep_util is Deprecated. Use functions from setuptools instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_modified, name)
<|endoftext|>"""Provide access to Python's configuration information.  The specific
configuration variables available depend heavily on the platform and
configuration.  The values may be retrieved using
get_config_var(name), and the list of variables is available via
get_config_vars().keys().  Additional convenience functions are also
available.

Written by:   Fred L. Drake, Jr.
Email:        <fdrake@acm.org>
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import sys
import sysconfig
from typing import TYPE_CHECKING, Literal, overload

from jaraco.functools import pass_none

from .ccompiler import CCompiler
from .errors import DistutilsPlatformError
from .util import is_mingw

if TYPE_CHECKING:
    from typing_extensions import deprecated
else:

    def deprecated(message):
        return lambda fn: fn


IS_PYPY = '__pypy__' in sys.builtin_module_names

# These are needed in a couple of spots, so just compute them once.
PREFIX = os.path.normpath(sys.prefix)
EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
BASE_PREFIX = os.path.normpath(sys.base_prefix)
BASE_EXEC_PREFIX = os.path.normpath(sys.base_exec_prefix)

# Path to the base directory of the project. On Windows the binary may
# live in project/PCbuild/win32 or project/PCbuild/amd64.
# set for cross builds
if "_PYTHON_PROJECT_BASE" in os.environ:
    project_base = os.path.abspath(os.environ["_PYTHON_PROJECT_BASE"])
else:
    if sys.executable:
        project_base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # sys.executable can be empty if argv[0] has been changed and Python is
        # unable to retrieve the real program name
        project_base = os.getcwd()


def _is_python_source_dir(d):
    """
    Return True if the target directory appears to point to an
    un-installed Python.
    """
    modules = pathlib.Path(d).joinpath('Modules')
    return any(modules.joinpath(fn).is_file() for fn in ('Setup', 'Setup.local'))


_sys_home = getattr(sys, '_home', None)


def _is_parent(dir_a, dir_b):
    """
    Return True if a is a parent of b.
    """
    return os.path.normcase(dir_a).startswith(os.path.normcase(dir_b))


if os.name == 'nt':

    @pass_none
    def _fix_pcbuild(d):
        # In a venv, sys._home will be inside BASE_PREFIX rather than PREFIX.
        prefixes = PREFIX, BASE_PREFIX
        matched = (
            prefix
            for prefix in prefixes
            if _is_parent(d, os.path.join(prefix, "PCbuild"))
        )
        return next(matched, d)

    project_base = _fix_pcbuild(project_base)
    _sys_home = _fix_pcbuild(_sys_home)


def _python_build():
    if _sys_home:
        return _is_python_source_dir(_sys_home)
    return _is_python_source_dir(project_base)


python_build = _python_build()


# Calculate the build qualifier flags if they are defined.  Adding the flags
# to the include and lib directories only makes sense for an installation, not
# an in-source build.
build_flags = ''
try:
    if not python_build:
        build_flags = sys.abiflags
except AttributeError:
    # It's not a configure-based build, so the sys module doesn't have
    # this attribute, which is fine.
    pass


def get_python_version():
    """Return a string containing the major and minor Python version,
    leaving off the patchlevel.  Sample return values could be '1.5'
    or '2.2'.
    """
    return f'{sys.version_info.major}.{sys.version_info.minor}'


def get_python_inc(plat_specific: bool = False, prefix: str | None = None) -> str:
    """Return the directory containing installed Python header files.

    If 'plat_specific' is false (the default), this is the path to the
    non-platform-specific header files, i.e. Python.h and so on;
    otherwise, this is the path to platform-specific header files
    (namely pyconfig.h).

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """
    default_prefix = BASE_EXEC_PREFIX if plat_specific else BASE_PREFIX
    resolved_prefix = prefix if prefix is not None else default_prefix
    # MinGW imitates posix like layout, but os.name != posix
    os_name = "posix" if is_mingw() else os.name
    try:
        getter = globals()[f'_get_python_inc_{os_name}']
    except KeyError:
        raise DistutilsPlatformError(
            "I don't know where Python installs its C header files "
            f"on platform '{os.name}'"
        )
    return getter(resolved_prefix, prefix, plat_specific)


@pass_none
def _extant(path):
    """
    Replace path with None if it doesn't exist.
    """
    return path if os.path.exists(path) else None


def _get_python_inc_posix(prefix, spec_prefix, plat_specific):
    return (
        _get_python_inc_posix_python(plat_specific)
        or _extant(_get_python_inc_from_config(plat_specific, spec_prefix))
        or _get_python_inc_posix_prefix(prefix)
    )


def _get_python_inc_posix_python(plat_specific):
    """
    Assume the executable is in the build directory. The
    pyconfig.h file should be in the same directory. Since
    the build directory may not be the source directory,
    use "srcdir" from the makefile to find the "Include"
    directory.
    """
    if not python_build:
        return
    if plat_specific:
        return _sys_home or project_base
    incdir = os.path.join(get_config_var('srcdir'), 'Include')
    return os.path.normpath(incdir)


def _get_python_inc_from_config(plat_specific, spec_prefix):
    """
    If no prefix was explicitly specified, provide the include
    directory from the config vars. Useful when
    cross-compiling, since the config vars may come from
    the host
    platform Python installation, while the current Python
    executable is from the build platform installation.

    >>> monkeypatch = getfixture('monkeypatch')
    >>> gpifc = _get_python_inc_from_config
    >>> monkeypatch.setitem(gpifc.__globals__, 'get_config_var', str.lower)
    >>> gpifc(False, '/usr/bin/')
    >>> gpifc(False, '')
    >>> gpifc(False, None)
    'includepy'
    >>> gpifc(True, None)
    'confincludepy'
    """
    if spec_prefix is None:
        return get_config_var('CONF' * plat_specific + 'INCLUDEPY')


def _get_python_inc_posix_prefix(prefix):
    implementation = 'pypy' if IS_PYPY else 'python'
    python_dir = implementation + get_python_version() + build_flags
    return os.path.join(prefix, "include", python_dir)


def _get_python_inc_nt(prefix, spec_prefix, plat_specific):
    if python_build:
        # Include both include dirs to ensure we can find pyconfig.h
        return (
            os.path.join(prefix, "include")
            + os.path.pathsep
            + os.path.dirname(sysconfig.get_config_h_filename())
        )
    return os.path.join(prefix, "include")


# allow this behavior to be monkey-patched. Ref pypa/distutils#2.
def _posix_lib(standard_lib, libpython, early_prefix, prefix):
    if standard_lib:
        return libpython
    else:
        return os.path.join(libpython, "site-packages")


def get_python_lib(
    plat_specific: bool = False, standard_lib: bool = False, prefix: str | None = None
) -> str:
    """Return the directory containing the Python library (standard or
    site additions).

    If 'plat_specific' is true, return the directory containing
    platform-specific modules, i.e. any module from a non-pure-Python
    module distribution; otherwise, return the platform-shared library
    directory.  If 'standard_lib' is true, return the directory
    containing standard Python library modules; otherwise, return the
    directory for site-specific modules.

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """

    early_prefix = prefix

    if prefix is None:
        if standard_lib:
            prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
        else:
            prefix = plat_specific and EXEC_PREFIX or PREFIX

    if os.name == "posix" or is_mingw():
        if plat_specific or standard_lib:
            # Platform-specific modules (any module from a non-pure-Python
            # module distribution) or standard Python library modules.
            libdir = getattr(sys, "platlibdir", "lib")
        else:
            # Pure Python
            libdir = "lib"
        implementation = 'pypy' if IS_PYPY else 'python'
        libpython = os.path.join(prefix, libdir, implementation + get_python_version())
        return _posix_lib(standard_lib, libpython, early_prefix, prefix)
    elif os.name == "nt":
        if standard_lib:
            return os.path.join(prefix, "Lib")
        else:
            return os.path.join(prefix, "Lib", "site-packages")
    else:
        raise DistutilsPlatformError(
            f"I don't know where Python installs its library on platform '{os.name}'"
        )


@functools.lru_cache
def customize_compiler(compiler: CCompiler) -> None:
    """Do any platform-specific customization of a CCompiler instance.

    Retained for backward compatibility; the behavior now lives on the
    compiler itself, so this simply delegates to ``configure_system``.
    """
    compiler.configure_system()


def get_config_h_filename() -> str:
    """Return full pathname of installed pyconfig.h file."""
    return sysconfig.get_config_h_filename()


def get_makefile_filename() -> str:
    """Return full pathname of installed Makefile from the Python build."""
    return sysconfig.get_makefile_filename()


def parse_config_h(fp, g=None):
    """Parse a config.h-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    return sysconfig.parse_config_h(fp, vars=g)


# Regexes needed for parsing Makefile (and similar syntaxes,
# like old-style Setup files).
_variable_rx = re.compile(r"([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
_findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
_findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")


def parse_makefile(fn, g=None):  # noqa: C901
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    from distutils.text_file import TextFile

    fp = TextFile(
        fn,
        strip_comments=True,
        skip_blanks=True,
        join_lines=True,
        errors="surrogateescape",
    )

    if g is None:
        g = {}
    done = {}
    notdone = {}

    while True:
        line = fp.readline()
        if line is None:  # eof
            break
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            # `$$' is a literal `$' in make
            tmpv = v.replace('$$', '')

            if "$" in tmpv:
                notdone[n] = v
            else:
                try:
                    v = int(v)
                except ValueError:
                    # insert literal `$'
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v

    # Variables with a 'PY_' prefix in the makefile. These need to
    # be made available without that prefix through sysconfig.
    # Special care is needed to ensure that variable expansion works, even
    # if the expansion uses the name without a prefix.
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')

    # do variable interpolation here
    while notdone:
        for name in list(notdone):
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    # get it on a subsequent round
                    found = False
                elif n in os.environ:
                    # do it like make: fall back to environment
                    item = os.environ[n]

                elif n in renamed_variables:
                    if name.startswith('PY_') and name[3:] in renamed_variables:
                        item = ""

                    elif 'PY_' + n in notdone:
                        found = False

                    else:
                        item = str(done['PY_' + n])
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end() :]
                    value = value[: m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        del notdone[name]

                        if name.startswith('PY_') and name[3:] in renamed_variables:
                            name = name[3:]
                            if name not in done:
                                done[name] = value
            else:
                # bogus variable reference; just drop it since we can't deal
                del notdone[name]

    fp.close()

    # strip spurious spaces
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()

    # save the results in the global dictionary
    g.update(done)
    return g


def expand_makefile_vars(s, vars):
    """Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
    'string' according to 'vars' (a dictionary mapping variable names to
    values).  Variables not present in 'vars' are silently expanded to the
    empty string.  The variable values in 'vars' should not contain further
    variable expansions; if 'vars' is the output of 'parse_makefile()',
    you're fine.  Returns a variable-expanded version of 's'.
    """

    # This algorithm does multiple expansion, so if vars['foo'] contains
    # "${bar}", it will expand ${foo} to ${bar}, and then expand
    # ${bar}... and so forth.  This is fine as long as 'vars' comes from
    # 'parse_makefile()', which takes care of such expansions eagerly,
    # according to make's variable expansion semantics.

    while True:
        m = _findvar1_rx.search(s) or _findvar2_rx.search(s)
        if m:
            (beg, end) = m.span()
            s = s[0:beg] + vars.get(m.group(1)) + s[end:]
        else:
            break
    return s


_config_vars = None


@overload
def get_config_vars() -> dict[str, str | int]: ...
@overload
def get_config_vars(arg: str, /, *args: str) -> list[str | int | None]: ...
def get_config_vars(*args: str) -> list[str | int | None] | dict[str, str | int]:
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.  Generally this includes
    everything needed to build extensions and install both pure modules and
    extensions.  On Unix, this means every variable defined in Python's
    installed Makefile; on Windows it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _config_vars
    if _config_vars is None:
        _config_vars = sysconfig.get_config_vars().copy()

    return [_config_vars.get(name) for name in args] if args else _config_vars


@overload
@deprecated(
    "SO is deprecated, use EXT_SUFFIX. Support will be removed when this module is synchronized with stdlib Python 3.11"
)
def get_config_var(name: Literal["SO"]) -> int | str | None: ...
@overload
def get_config_var(name: str) -> int | str | None: ...
def get_config_var(name: str) -> int | str | None:
    """Return the value of a single variable using the dictionary
    returned by 'get_config_vars()'.  Equivalent to
    get_config_vars().get(name)
    """
    if name == 'SO':
        import warnings

        warnings.warn('SO is deprecated, use EXT_SUFFIX', DeprecationWarning, 2)
    return get_config_vars().get(name)
<|endoftext|># This is a private module, but setuptools has the explicit permission to use it.
from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from functools import wraps
from typing import TypeVar

from .compat.py310 import dataclass_transform

_T = TypeVar("_T", bound=type)


@dataclass_transform()
def lenient_dataclass(**dc_kwargs):
    """
    Build a dataclass whose ``__init__`` ignores unknown keyword arguments.

    Customize ``__init__`` to preserve backwards compatibility and keep
    tolerating arbitrary keywords, but keep the dataclass-generated
    ``__init__`` to avoid redefining the typing for all the arguments.

    Drop this customization once lenient behaviour and backward
    compatibility are no longer needed and use a regular ``dataclass``
    instead.
    """

    @wraps(dataclass)
    def _wrap(cls: _T) -> _T:  # type: ignore[var-annotated]
        cls = dataclass(**dc_kwargs)(cls)
        # Allowed field names in order
        safe = tuple(f.name for f in fields(cls))
        orig_init = cls.__init__

        @wraps(orig_init)
        def _wrapped_init(self, *args, **kwargs):
            extra = {repr(k): kwargs.pop(k) for k in tuple(kwargs) if k not in safe}
            if extra:
                msg = f"""
                Please remove unknown `{cls.__name__}` options: {','.join(extra)}
                this kind of usage is deprecated and may cause errors in the future.
                """
                warnings.warn(msg)

            # Ensure default values (e.g. []) are used instead of None:
            positional = {
                k: v for k, v in zip(safe, args, strict=False) if v is not None
            }
            keywords = {k: v for k, v in kwargs.items() if v is not None}
            return orig_init(self, **positional, **keywords)

        cls.__init__ = _wrapped_init
        return cls

    return _wrap
<|endoftext|>import importlib
import sys

__version__, _, _ = sys.version.partition(' ')


try:
    # Allow Debian and pkgsrc (only) to customize system
    # behavior. Ref pypa/distutils#2 and pypa/distutils#16.
    # This hook is deprecated and no other environments
    # should use it.
    importlib.import_module('_distutils_system_mod')
except ImportError:
    pass
<|endoftext|>"""
A simple log mechanism styled after PEP 282.

Retained for compatibility and should not be used.
"""

import logging
import warnings

from ._log import log as _global_log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARNING
ERROR = logging.ERROR
FATAL = logging.FATAL

log = _global_log.log
debug = _global_log.debug
info = _global_log.info
warn = _global_log.warning
error = _global_log.error
fatal = _global_log.fatal


def set_threshold(level):
    orig = _global_log.level
    _global_log.setLevel(level)
    return orig


def set_verbosity(v):
    if v <= 0:
        set_threshold(logging.WARNING)
    elif v == 1:
        set_threshold(logging.INFO)
    elif v >= 2:
        set_threshold(logging.DEBUG)


class Log(logging.Logger):
    """distutils.log.Log is deprecated, please use an alternative from `logging`."""

    def __init__(self, threshold=WARN):
        warnings.warn(Log.__doc__)  # avoid DeprecationWarning to ensure warn is shown
        super().__init__(__name__, level=threshold)

    @property
    def threshold(self):
        return self.level

    @threshold.setter
    def threshold(self, level):
        self.setLevel(level)

    warn = logging.Logger.warning
<|endoftext|>"""Timestamp comparison of files and groups of files."""

from __future__ import annotations

import functools
import os.path
from collections.abc import Callable, Iterable
from typing import Literal, TypeVar

from jaraco.functools import splat

from .errors import DistutilsFileError

_SourcesT = TypeVar(
    "_SourcesT", bound="str | bytes | os.PathLike[str] | os.PathLike[bytes]"
)
_TargetsT = TypeVar(
    "_TargetsT", bound="str | bytes | os.PathLike[str] | os.PathLike[bytes]"
)


def _newer(source, target):
    return not os.path.exists(target) or (
        os.path.getmtime(source) > os.path.getmtime(target)
    )


def newer(
    source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> bool:
    """
    Is source modified more recently than target.

    Returns True if 'source' is modified more recently than
    'target' or if 'target' does not exist.

    Raises DistutilsFileError if 'source' does not exist.
    """
    if not os.path.exists(source):
        raise DistutilsFileError(f"file {os.path.abspath(source)!r} does not exist")

    return _newer(source, target)


def newer_pairwise(
    sources: Iterable[_SourcesT],
    targets: Iterable[_TargetsT],
    newer: Callable[[_SourcesT, _TargetsT], bool] = newer,
) -> tuple[list[_SourcesT], list[_TargetsT]]:
    """
    Filter filenames where sources are newer than targets.

    Walk two filename iterables in parallel, testing if each source is newer
    than its corresponding target.  Returns a pair of lists (sources,
    targets) where source is newer than target, according to the semantics
    of 'newer()'.
    """
    newer_pairs = filter(splat(newer), zip(sources, targets, strict=True))
    return tuple(map(list, zip(*newer_pairs, strict=False))) or ([], [])


def newer_group(
    sources: Iterable[str | bytes | os.PathLike[str] | os.PathLike[bytes]],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    missing: Literal["error", "ignore", "newer"] = "error",
) -> bool:
    """
    Is target out-of-date with respect to any file in sources.

    Return True if 'target' is out-of-date with respect to any file
    listed in 'sources'. In other words, if 'target' exists and is newer
    than every file in 'sources', return False; otherwise return True.
    ``missing`` controls how to handle a missing source file:

    - error (default): allow the ``stat()`` call to fail.
    - ignore: silently disregard any missing source files.
    - newer: treat missing source files as "target out of date". This
      mode is handy in "dry-run" mode: it will pretend to carry out
      commands that wouldn't work because inputs are missing, but
      that doesn't matter because dry-run won't run the commands.
    """

    def missing_as_newer(source):
        return missing == 'newer' and not os.path.exists(source)

    ignored = os.path.exists if missing == 'ignore' else None
    return not os.path.exists(target) or any(
        missing_as_newer(source) or _newer(source, target)
        for source in filter(ignored, sources)
    )


newer_pairwise_group = functools.partial(newer_pairwise, newer=newer_group)
<|endoftext|>#
# distutils/version.py
#
# Implements multiple version numbering conventions for the
# Python Module Distribution Utilities.
#
# $Id$
#

"""Provides classes to represent module version numbers (one class for
each style of version numbering).  There are currently two such classes
implemented: StrictVersion and LooseVersion.

Every version number class implements the following interface:
  * the 'parse' method takes a string and parses it to some internal
    representation; if the string is an invalid version number,
    'parse' raises a ValueError exception
  * the class constructor takes an optional string argument which,
    if supplied, is passed to 'parse'
  * __str__ reconstructs the string that was passed to 'parse' (or
    an equivalent string -- ie. one that will generate an equivalent
    version number instance)
  * __repr__ generates Python code to recreate the version number instance
  * _cmp compares the current instance with either another instance
    of the same class or a string (which will be parsed to an instance
    of the same class, thus must follow the same rules)
"""

import contextlib
import re
import warnings


@contextlib.contextmanager
def suppress_known_deprecation():
    with warnings.catch_warnings(record=True) as ctx:
        warnings.filterwarnings(
            action='default',
            category=DeprecationWarning,
            message="distutils Version classes are deprecated.",
        )
        yield ctx


class Version:
    """Abstract base class for version numbering classes.  Just provides
    constructor (__init__) and reproducer (__repr__), because those
    seem to be the same for all version numbering classes; and route
    rich comparisons to _cmp.
    """

    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)
        warnings.warn(
            "distutils Version classes are deprecated. Use packaging.version instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def __repr__(self):
        return f"{self.__class__.__name__} ('{self}')"

    def __eq__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0


# Interface for version-number classes -- must be implemented
# by the following classes (the concrete ones -- Version should
# be treated as an abstract class).
#    __init__ (string) - create and take same action as 'parse'
#                        (string parameter is optional)
#    parse (string)    - convert a string representation to whatever
#                        internal representation is appropriate for
#                        this style of version numbering
#    __str__ (self)    - convert back to a string; should be very similar
#                        (if not identical to) the string supplied to parse
#    __repr__ (self)   - generate Python code to recreate
#                        the instance
#    _cmp (self, other) - compare two version numbers ('other' may
#                        be an unparsed version string, or another
#                        instance of your version class)


class StrictVersion(Version):
    """Version numbering for anal retentives and software idealists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of two or three
    dot-separated numeric components, with an optional "pre-release" tag
    on the end.  The pre-release tag consists of the letter 'a' or 'b'
    followed by a number.  If the numeric components of two version
    numbers are equal, then one with a pre-release tag will always
    be deemed earlier (lesser) than one without.

    The following are valid version numbers (shown in the order that
    would be obtained by sorting according to the supplied cmp function):

        0.4       0.4.0  (these two are equivalent)
        0.4.1
        0.5a1
        0.5b3
        0.5
        0.9.6
        1.0
        1.0.4a3
        1.0.4b1
        1.0.4

    The following are examples of invalid version numbers:

        1
        2.7.2.2
        1.3.a4
        1.3pl1
        1.3c4

    The rationale for this version numbering system will be explained
    in the distutils documentation.
    """

    version_re = re.compile(
        r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$', re.VERBOSE | re.ASCII
    )

    def parse(self, vstring):
        match = self.version_re.match(vstring)
        if not match:
            raise ValueError(f"invalid version number '{vstring}'")

        (major, minor, patch, prerelease, prerelease_num) = match.group(1, 2, 4, 5, 6)

        if patch:
            self.version = tuple(map(int, [major, minor, patch]))
        else:
            self.version = tuple(map(int, [major, minor])) + (0,)

        if prerelease:
            self.prerelease = (prerelease[0], int(prerelease_num))
        else:
            self.prerelease = None

    def __str__(self):
        if self.version[2] == 0:
            vstring = '.'.join(map(str, self.version[0:2]))
        else:
            vstring = '.'.join(map(str, self.version))

        if self.prerelease:
            vstring = vstring + self.prerelease[0] + str(self.prerelease[1])

        return vstring

    def _cmp(self, other):
        if isinstance(other, str):
            with suppress_known_deprecation():
                other = StrictVersion(other)
        elif not isinstance(other, StrictVersion):
            return NotImplemented

        if self.version == other.version:
            # versions match; pre-release drives the comparison
            return self._cmp_prerelease(other)

        return -1 if self.version < other.version else 1

    def _cmp_prerelease(self, other):
        """
        case 1: self has prerelease, other doesn't; other is greater
        case 2: self doesn't have prerelease, other does: self is greater
        case 3: both or neither have prerelease: compare them!
        """
        if self.prerelease and not other.prerelease:
            return -1
        elif not self.prerelease and other.prerelease:
            return 1

        if self.prerelease == other.prerelease:
            return 0
        elif self.prerelease < other.prerelease:
            return -1
        else:
            return 1


# end class StrictVersion


# The rules according to Greg Stein:
# 1) a version number has 1 or more numbers separated by a period or by
#    sequences of letters. If only periods, then these are compared
#    left-to-right to determine an ordering.
# 2) sequences of letters are part of the tuple for comparison and are
#    compared lexicographically
# 3) recognize the numeric components may have leading zeroes
#
# The LooseVersion class below implements these rules: a version number
# string is split up into a tuple of integer and string components, and
# comparison is a simple tuple comparison.  This means that version
# numbers behave in a predictable and obvious way, but a way that might
# not necessarily be how people *want* version numbers to behave.  There
# wouldn't be a problem if people could stick to purely numeric version
# numbers: just split on period and compare the numbers as tuples.
# However, people insist on putting letters into their version numbers;
# the most common purpose seems to be:
#   - indicating a "pre-release" version
#     ('alpha', 'beta', 'a', 'b', 'pre', 'p')
#   - indicating a post-release patch ('p', 'pl', 'patch')
# but of course this can't cover all version number schemes, and there's
# no way to know what a programmer means without asking him.
#
# The problem is what to do with letters (and other non-numeric
# characters) in a version number.  The current implementation does the
# obvious and predictable thing: keep them as strings and compare
# lexically within a tuple comparison.  This has the desired effect if
# an appended letter sequence implies something "post-release":
# eg. "0.99" < "0.99pl14" < "1.0", and "5.001" < "5.001m" < "5.002".
#
# However, if letters in a version number imply a pre-release version,
# the "obvious" thing isn't correct.  Eg. you would expect that
# "1.5.1" < "1.5.2a2" < "1.5.2", but under the tuple/lexical comparison
# implemented here, this just isn't so.
#
# Two possible solutions come to mind.  The first is to tie the
# comparison algorithm to a particular set of semantic rules, as has
# been done in the StrictVersion class above.  This works great as long
# as everyone can go along with bondage and discipline.  Hopefully a
# (large) subset of Python module programmers will agree that the
# particular flavour of bondage and discipline provided by StrictVersion
# provides enough benefit to be worth using, and will submit their
# version numbering scheme to its domination.  The free-thinking
# anarchists in the lot will never give in, though, and something needs
# to be done to accommodate them.
#
# Perhaps a "moderately strict" version class could be implemented that
# lets almost anything slide (syntactically), and makes some heuristic
# assumptions about non-digits in version number strings.  This could
# sink into special-case-hell, though; if I was as talented and
# idiosyncratic as Larry Wall, I'd go ahead and implement a class that
# somehow knows that "1.2.1" < "1.2.2a2" < "1.2.2" < "1.2.2pl3", and is
# just as happy dealing with things like "2g6" and "1.13++".  I don't
# think I'm smart enough to do it right though.
#
# In any case, I've coded the test suite for this module (see
# ../test/test_version.py) specifically to fail on things like comparing
# "1.2a2" and "1.2".  That's not because the *code* is doing anything
# wrong, it's because the simple, obvious design doesn't match my
# complicated, hairy expectations for real-world version numbers.  It
# would be a snap to fix the test suite to say, "Yep, LooseVersion does
# the Right Thing" (ie. the code matches the conception).  But I'd rather
# have a conception that matches common notions about version numbers.


class LooseVersion(Version):
    """Version numbering for anarchists and software realists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of a series of numbers,
    separated by either periods or strings of letters.  When comparing
    version numbers, the numeric components will be compared
    numerically, and the alphabetic components lexically.  The following
    are all valid version numbers, in no particular order:

        1.5.1
        1.5.2b2
        161
        3.10a
        8.02
        3.4j
        1996.07.12
        3.2.pl0
        3.1.1.6
        2g6
        11g
        0.960923
        2.2beta29
        1.13++
        5.5.kw
        2.0b1pl0

    In fact, there is no such thing as an invalid version number under
    this scheme; the rules for comparison are simple and predictable,
    but may not always give the results you want (for some definition
    of "want").
    """

    component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)

    def parse(self, vstring):
        # I've given up on thinking I can reconstruct the version string
        # from the parsed tuple -- so I just store the string here for
        # use by __str__
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring) if x and x != '.']
        for i, obj in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass

        self.version = components

    def __str__(self):
        return self.vstring

    def __repr__(self):
        return f"LooseVersion ('{self}')"

    def _cmp(self, other):
        if isinstance(other, str):
            other = LooseVersion(other)
        elif not isinstance(other, LooseVersion):
            return NotImplemented

        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1


# end class LooseVersion
<|endoftext|>"""distutils.core

The only module that needs to be imported to use the Distutils; provides
the 'setup' function (which is to be called from the setup script).  Also
indirectly provides the Distribution and Command classes, although they are
really defined in distutils.dist and distutils.cmd.
"""

from __future__ import annotations

import os
import sys
import tokenize
from collections.abc import Iterable

from .cmd import Command
from .debug import DEBUG

# Mainly import these so setup scripts can "from distutils.core import" them.
from .dist import Distribution
from .errors import (
    CCompilerError,
    DistutilsArgError,
    DistutilsError,
    DistutilsSetupError,
)
from .extension import Extension
from .extension import _safe as extension_keywords  # noqa  # backwards compatibility

__all__ = ['Command', 'Distribution', 'Extension', 'setup']

# This is a barebones help message generated displayed when the user
# runs the setup script with no arguments at all.  More useful help
# is generated with various --help options: global help, list commands,
# and per-command help.
USAGE = """\
usage: %(script)s [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: %(script)s --help [cmd1 cmd2 ...]
   or: %(script)s --help-commands
   or: %(script)s cmd --help
"""


def gen_usage(script_name):
    script = os.path.basename(script_name)
    return USAGE % locals()


# Some mild magic to control the behaviour of 'setup()' from 'run_setup()'.
_setup_stop_after = None
_setup_distribution = None

# Legal keyword arguments for the setup() function
setup_keywords = (
    'distclass',
    'script_name',
    'script_args',
    'options',
    'name',
    'version',
    'author',
    'author_email',
    'maintainer',
    'maintainer_email',
    'url',
    'license',
    'description',
    'long_description',
    'keywords',
    'platforms',
    'classifiers',
    'download_url',
    'requires',
    'provides',
    'obsoletes',
)


def setup(**attrs):  # noqa: C901
    """The gateway to the Distutils: do everything your setup script needs
    to do, in a highly flexible and user-driven way.  Briefly: create a
    Distribution instance; find and parse config files; parse the command
    line; run each Distutils command found there, customized by the options
    supplied to 'setup()' (as keyword arguments), in config files, and on
    the command line.

    The Distribution instance might be an instance of a class supplied via
    the 'distclass' keyword argument to 'setup'; if no such class is
    supplied, then the Distribution class (in dist.py) is instantiated.
    All other arguments to 'setup' (except for 'cmdclass') are used to set
    attributes of the Distribution instance.

    The 'cmdclass' argument, if supplied, is a dictionary mapping command
    names to command classes.  Each command encountered on the command line
    will be turned into a command class, which is in turn instantiated; any
    class found in 'cmdclass' is used in place of the default, which is
    (for command 'foo_bar') class 'foo_bar' in module
    'distutils.command.foo_bar'.  The command class must provide a
    'user_options' attribute which is a list of option specifiers for
    'distutils.fancy_getopt'.  Any command-line options between the current
    and the next command are used to set attributes of the current command
    object.

    When the entire command-line has been successfully parsed, calls the
    'run()' method on each command object in turn.  This method will be
    driven entirely by the Distribution object (which each command object
    has a reference to, thanks to its constructor), and the
    command-specific options that became attributes of each command
    object.
    """

    global _setup_distribution

    # Determine the distribution class -- either caller-supplied or
    # our Distribution (see below).
    klass = attrs.get('distclass')
    if klass:
        attrs.pop('distclass')
    else:
        klass = Distribution

    if 'script_name' not in attrs:
        attrs['script_name'] = os.path.basename(sys.argv[0])
    if 'script_args' not in attrs:
        attrs['script_args'] = sys.argv[1:]

    # Create the Distribution instance, using the remaining arguments
    # (ie. everything except distclass) to initialize it
    try:
        _setup_distribution = dist = klass(attrs)
    except DistutilsSetupError as msg:
        if 'name' not in attrs:
            raise SystemExit(f"error in setup command: {msg}")
        else:
            raise SystemExit("error in {} setup command: {}".format(attrs['name'], msg))

    if _setup_stop_after == "init":
        return dist

    # Find and parse the config file(s): they will override options from
    # the setup script, but be overridden by the command line.
    dist.parse_config_files()

    if DEBUG:
        print("options (after parsing config files):")
        dist.dump_option_dicts()

    if _setup_stop_after == "config":
        return dist

    # Parse the command line and override config files; any
    # command-line errors are the end user's fault, so turn them into
    # SystemExit to suppress tracebacks.
    try:
        ok = dist.parse_command_line()
    except DistutilsArgError as msg:
        raise SystemExit(gen_usage(dist.script_name) + f"\nerror: {msg}")

    if DEBUG:
        print("options (after parsing command line):")
        dist.dump_option_dicts()

    if _setup_stop_after == "commandline":
        return dist

    # And finally, run all the commands found on the command line.
    if ok:
        return run_commands(dist)

    return dist


# setup ()


def run_commands(dist):
    """Given a Distribution object run all the commands,
    raising ``SystemExit`` errors in the case of failure.

    This function assumes that either ``sys.argv`` or ``dist.script_args``
    is already set accordingly.
    """
    try:
        dist.run_commands()
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
    except OSError as exc:
        if DEBUG:
            sys.stderr.write(f"error: {exc}\n")
            raise
        else:
            raise SystemExit(f"error: {exc}")

    except (DistutilsError, CCompilerError) as msg:
        if DEBUG:
            raise
        else:
            raise SystemExit(f"error: {msg}")

    return dist


def run_setup(script_name, script_args: Iterable[str] | None = None, stop_after="run"):
    """Run a setup script in a somewhat controlled environment, and
    return the Distribution instance that drives things.  This is useful
    if you need to find out the distribution meta-data (passed as
    keyword args from 'script' to 'setup()', or the contents of the
    config files or command-line.

    'script_name' is a file that will be read and run with 'exec()';
    'sys.argv[0]' will be replaced with 'script' for the duration of the
    call.  'script_args' is a list of strings; if supplied,
    'sys.argv[1:]' will be replaced by 'script_args' for the duration of
    the call.

    'stop_after' tells 'setup()' when to stop processing; possible
    values:
      init
        stop after the Distribution instance has been created and
        populated with the keyword arguments to 'setup()'
      config
        stop after config files have been parsed (and their data
        stored in the Distribution instance)
      commandline
        stop after the command-line ('sys.argv[1:]' or 'script_args')
        have been parsed (and the data stored in the Distribution)
      run [default]
        stop after all commands have been run (the same as if 'setup()'
        had been called in the usual way

    Returns the Distribution instance, which provides all information
    used to drive the Distutils.
    """
    if stop_after not in ('init', 'config', 'commandline', 'run'):
        raise ValueError(f"invalid value for 'stop_after': {stop_after!r}")

    global _setup_stop_after
    _setup_stop_after = stop_after

    save_argv = sys.argv.copy()
    g = {'__file__': script_name, '__name__': '__main__'}
    try:
        try:
            sys.argv[0] = script_name
            if script_args is not None:
                sys.argv[1:] = script_args
            # tokenize.open supports automatic encoding detection
            with tokenize.open(script_name) as f:
                code = f.read().replace(r'\r\n', r'\n')
                exec(code, g)  # noqa: S102 # executing the setup script is the point
        finally:
            sys.argv = save_argv
            _setup_stop_after = None
    except SystemExit:
        # Hmm, should we do something if exiting with a non-zero code
        # (ie. error)?
        pass

    if _setup_distribution is None:
        raise RuntimeError(
            "'distutils.core.setup()' was never called -- "
            f"perhaps '{script_name}' is not a Distutils setup script?"
        )

    # I wonder if the setup script's namespace -- g and l -- would be of
    # any interest to callers?
    # print "_setup_distribution:", _setup_distribution
    return _setup_distribution


# run_setup ()
<|endoftext|>"""distutils.archive_util

Utility functions for creating archive files (tarballs, zip files,
that sort of thing)."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from types import ModuleType
from typing import Literal, overload

from ._log import log
from .dir_util import mkpath
from .errors import DistutilsExecError
from .spawn import spawn

zipfile: ModuleType | None = None
with contextlib.suppress(ImportError):
    import zipfile

grp: ModuleType | None = None
pwd: ModuleType | None = None
with contextlib.suppress(ImportError):
    import grp
    import pwd


def _get_gid(name):
    """Returns a gid, given a group name."""
    if grp is None or name is None:
        return None
    try:
        result = grp.getgrnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None


def _get_uid(name):
    """Returns an uid, given a user name."""
    if pwd is None or name is None:
        return None
    try:
        result = pwd.getpwnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None


def make_tarball(
    base_name: str,
    base_dir: str | os.PathLike[str],
    compress: Literal["gzip", "bzip2", "xz"] | None = "gzip",
    verbose: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str:
    """Create a (possibly compressed) tar file from all the files under
    'base_dir'.

    'compress' must be "gzip" (the default), "bzip2", "xz", or None.

    'owner' and 'group' can be used to define an owner and a group for the
    archive that is being built. If not provided, the current owner and group
    will be used.

    The output tar file will be named 'base_dir' +  ".tar", possibly plus
    the appropriate compression extension (".gz", ".bz2", ".xz" or ".Z").

    Returns the output filename.
    """
    tar_compression = {
        'gzip': 'gz',
        'bzip2': 'bz2',
        'xz': 'xz',
        None: '',
    }
    compress_ext = {'gzip': '.gz', 'bzip2': '.bz2', 'xz': '.xz'}

    # flags for compression program, each element of list will be an argument
    if compress is not None and compress not in compress_ext:
        raise ValueError(
            "bad value for 'compress': must be None, 'gzip', 'bzip2', 'xz'"
        )

    archive_name = base_name + '.tar'
    archive_name += compress_ext.get(compress, '')

    mkpath(os.path.dirname(archive_name))

    # creating the tarball
    import tarfile  # late import so Python build itself doesn't break

    log.info('Creating tar archive')

    uid = _get_uid(owner)
    gid = _get_gid(group)

    def _set_uid_gid(tarinfo):
        if gid is not None:
            tarinfo.gid = gid
            tarinfo.gname = group
        if uid is not None:
            tarinfo.uid = uid
            tarinfo.uname = owner
        return tarinfo

    with tarfile.open(  # type: ignore[call-overload] # Dynamic mode
        archive_name, f'w|{tar_compression[compress]}'
    ) as tar:
        tar.add(base_dir, filter=_set_uid_gid)

    return archive_name


def make_zipfile(
    base_name: str,
    base_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> str:
    """Create a zip file from all the files under 'base_dir'.

    The output zip file will be named 'base_name' + ".zip".  Uses either the
    "zipfile" Python module (if available) or the InfoZIP "zip" utility
    (if installed and found on the default search path).  If neither tool is
    available, raises DistutilsExecError.  Returns the name of the output zip
    file.
    """
    zip_filename = base_name + ".zip"
    mkpath(os.path.dirname(zip_filename))

    # If zipfile module is not available, try spawning an external
    # 'zip' command.
    if zipfile is None:
        if verbose:
            zipoptions = "-r"
        else:
            zipoptions = "-rq"

        try:
            spawn(["zip", zipoptions, zip_filename, base_dir])
        except DistutilsExecError:
            # XXX really should distinguish between "couldn't find
            # external 'zip' command" and "zip failed".
            raise DistutilsExecError(
                f"unable to create zip file '{zip_filename}': "
                "could neither import the 'zipfile' module nor "
                "find a standalone zip utility"
            )

    else:
        log.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)

        try:
            zip = zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED)
        except RuntimeError:
            zip = zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_STORED)

        with zip:
            if base_dir != os.curdir:
                path = os.path.normpath(os.path.join(base_dir, ''))
                zip.write(path, path)
                log.info("adding '%s'", path)
            for dirpath, dirnames, filenames in os.walk(base_dir):
                for name in dirnames:
                    path = os.path.normpath(os.path.join(dirpath, name, ''))
                    zip.write(path, path)
                    log.info("adding '%s'", path)
                for name in filenames:
                    path = os.path.normpath(os.path.join(dirpath, name))
                    if os.path.isfile(path):
                        zip.write(path, path)
                        log.info("adding '%s'", path)

    return zip_filename


ARCHIVE_FORMATS: dict[
    str, tuple[Callable[..., str], list[tuple[str, str | None]], str]
] = {
    'gztar': (make_tarball, [('compress', 'gzip')], "gzip'ed tar-file"),
    'bztar': (make_tarball, [('compress', 'bzip2')], "bzip2'ed tar-file"),
    'xztar': (make_tarball, [('compress', 'xz')], "xz'ed tar-file"),
    'ztar': (make_tarball, [('compress', 'compress')], "compressed tar file"),
    'tar': (make_tarball, [('compress', None)], "uncompressed tar file"),
    'zip': (make_zipfile, [], "ZIP file"),
}


def check_archive_formats(formats):
    """Returns the first format from the 'format' list that is unknown.

    If all formats are known, returns None
    """
    for format in formats:
        if format not in ARCHIVE_FORMATS:
            return format
    return None


@overload
def make_archive(
    base_name: str,
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
    base_dir: str | None = None,
    verbose: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str: ...
@overload
def make_archive(
    base_name: str | os.PathLike[str],
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes],
    base_dir: str | None = None,
    verbose: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str: ...
def make_archive(
    base_name: str | os.PathLike[str],
    format: str,
    root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
    base_dir: str | None = None,
    verbose: bool = False,
    owner: str | None = None,
    group: str | None = None,
) -> str:
    """Create an archive file (eg. zip or tar).

    'base_name' is the name of the file to create, minus any format-specific
    extension; 'format' is the archive format: one of "zip", "tar", "gztar",
    "bztar", "xztar", or "ztar".

    'root_dir' is a directory that will be the root directory of the
    archive; ie. we typically chdir into 'root_dir' before creating the
    archive.  'base_dir' is the directory where we start archiving from;
    ie. 'base_dir' will be the common prefix of all files and
    directories in the archive.  'root_dir' and 'base_dir' both default
    to the current directory.  Returns the name of the archive file.

    'owner' and 'group' are used when creating a tar archive. By default,
    uses the current owner and group.
    """
    save_cwd = os.getcwd()
    if root_dir is not None:
        log.debug("changing into '%s'", root_dir)
        base_name = os.path.abspath(base_name)
        os.chdir(root_dir)

    if base_dir is None:
        base_dir = os.curdir

    kwargs: dict[str, str | bool | None] = {}

    try:
        format_info = ARCHIVE_FORMATS[format]
    except KeyError:
        raise ValueError(f"unknown archive format '{format}'")

    func = format_info[0]
    kwargs.update(format_info[1])

    if format != 'zip':
        kwargs['owner'] = owner
        kwargs['group'] = group

    try:
        filename = func(base_name, base_dir, **kwargs)
    finally:
        if root_dir is not None:
            log.debug("changing back to '%s'", save_cwd)
            os.chdir(save_cwd)

    return filename
<|endoftext|>"""distutils.dist

Provides the Distribution class, which represents the module distribution
being built/installed/distributed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import re
import sys
import warnings
from collections.abc import Iterable, Mapping, MutableMapping
from email import message_from_file
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    cast,
    overload,
)

from jaraco.text import SeparatedValues
from packaging.utils import canonicalize_name, canonicalize_version

from ._log import log
from .debug import DEBUG
from .errors import (
    DistutilsArgError,
    DistutilsClassError,
    DistutilsModuleError,
    DistutilsOptionError,
)
from .fancy_getopt import FancyGetopt, translate_longopt
from .util import check_environ, rfc822_escape, strtobool

if TYPE_CHECKING:
    from typing import TypeAlias

    from _typeshed import SupportsWrite

    # type-only import because of mutual dependence between these modules
    from .cmd import Command
    from .extension import Extension

_CommandT = TypeVar("_CommandT", bound="Command")
_OptionsList: TypeAlias = list[
    tuple[str, str | None, str, int] | tuple[str, str | None, str]
]


# Regex to define acceptable Distutils command names.  This is not *quite*
# the same as a Python NAME -- I don't allow leading underscores.  The fact
# that they're very similar is no coincidence; the default naming scheme is
# to look for a Python module named after the command.
command_re = re.compile(r'^[a-zA-Z]([a-zA-Z0-9_]*)$')


def _ensure_list(value: str | Iterable[str], fieldname) -> str | list[str]:
    if isinstance(value, str):
        # a string containing comma separated values is okay.  It will
        # be converted to a list by Distribution.finalize_options().
        pass
    elif not isinstance(value, list):
        # passing a tuple or an iterator perhaps, warn and convert
        typename = type(value).__name__
        msg = "Warning: '{fieldname}' should be a list, got type '{typename}'"
        msg = msg.format(**locals())
        log.warning(msg)
        value = list(value)
    return value


def _repair_newlines(value: str) -> str:
    """
    Repair a ``keywords`` or ``platforms`` value that improperly uses
    newlines to separate items.

    These fields separate items with commas (and, in the old specification,
    PEP 345, with spaces). Newlines were never a valid separator and corrupt
    the generated ``PKG-INFO``/``METADATA`` (pypa/setuptools#4887). As a
    forgiving measure, values that supply these fields as newline-separated
    strings (e.g. a triple-quoted string with one item per line) have each
    line treated as a separate item by replacing newlines with commas, but
    the behavior is deprecated. Empty items produced by consecutive or
    trailing separators are dropped by the caller.
    """
    if "\n" not in value:
        return value
    # A plain warning (UserWarning) rather than DeprecationWarning so that
    # it's shown by default; DeprecationWarning is suppressed outside of test
    # runners and this needs to reach setuptools users (cf. distutils.log.Log).
    warnings.warn(
        "Newlines are not a valid separator for the 'keywords' and "
        "'platforms' fields and their use is deprecated. Separate items "
        "with commas instead. This will raise an error in the future.",
        stacklevel=2,
    )
    return value.replace("\n", ",")


class Distribution:
    """The core of the Distutils.  Most of the work hiding behind 'setup'
    is really done within a Distribution instance, which farms the work out
    to the Distutils commands specified on the command line.

    Setup scripts will almost never instantiate Distribution directly,
    unless the 'setup()' function is totally inadequate to their needs.
    However, it is conceivable that a setup script might wish to subclass
    Distribution for some specialized purpose, and then pass the subclass
    to 'setup()' as the 'distclass' keyword argument.  If so, it is
    necessary to respect the expectations that 'setup' has of Distribution.
    See the code for 'setup()', in core.py, for details.
    """

    # 'global_options' describes the command-line options that may be
    # supplied to the setup script prior to any actual commands.
    # Eg. "./setup.py -n" or "./setup.py --quiet" both take advantage of
    # these global options.  This list should be kept to a bare minimum,
    # since every global option is also valid as a command option -- and we
    # don't want to pollute the commands with too many options that they
    # have minimal control over.
    # The fourth entry for verbose means that it can be repeated.
    global_options: ClassVar[_OptionsList] = [
        ('verbose', 'v', "run verbosely (default)", 1),
        ('quiet', 'q', "run quietly (turns verbosity off)"),
        ('help', 'h', "show detailed help message"),
        ('no-user-cfg', None, 'ignore pydistutils.cfg in your home directory'),
    ]

    # 'common_usage' is a short (2-3 line) string describing the common
    # usage of the setup script.
    common_usage: ClassVar[str] = """\
Common commands: (see '--help-commands' for more)

  setup.py build      will build the package underneath 'build/'
  setup.py install    will install the package
"""

    # options that are not propagated to the commands
    display_options: ClassVar[_OptionsList] = [
        ('help-commands', None, "list all available commands"),
        ('name', None, "print package name"),
        ('version', 'V', "print package version"),
        ('fullname', None, "print <package name>-<version>"),
        ('author', None, "print the author's name"),
        ('author-email', None, "print the author's email address"),
        ('maintainer', None, "print the maintainer's name"),
        ('maintainer-email', None, "print the maintainer's email address"),
        ('contact', None, "print the maintainer's name if known, else the author's"),
        (
            'contact-email',
            None,
            "print the maintainer's email address if known, else the author's",
        ),
        ('url', None, "print the URL for this package"),
        ('license', None, "print the license of the package"),
        ('licence', None, "alias for --license"),
        ('description', None, "print the package description"),
        ('long-description', None, "print the long package description"),
        ('platforms', None, "print the list of platforms"),
        ('classifiers', None, "print the list of classifiers"),
        ('keywords', None, "print the list of keywords"),
        ('provides', None, "print the list of packages/modules provided"),
        ('requires', None, "print the list of packages/modules required"),
        ('obsoletes', None, "print the list of packages/modules made obsolete"),
    ]
    display_option_names: ClassVar[list[str]] = [
        translate_longopt(x[0]) for x in display_options
    ]

    # negative options are options that exclude other options
    negative_opt: ClassVar[dict[str, str]] = {'quiet': 'verbose'}

    # -- Creation/initialization methods -------------------------------

    # Can't Unpack a TypedDict with optional properties, so using Any instead
    def __init__(self, attrs: MutableMapping[str, Any] | None = None) -> None:  # noqa: C901
        """Construct a new Distribution instance: initialize all the
        attributes of a Distribution, and then use 'attrs' (a dictionary
        mapping attribute names to values) to assign some of those
        attributes their "real" values.  (Any attributes not mentioned in
        'attrs' will be assigned to some null value: 0, None, an empty list
        or dictionary, etc.)  Most importantly, initialize the
        'command_obj' attribute to the empty dictionary; this will be
        filled in with real command objects by 'parse_command_line()'.
        """

        # Default values for our command-line options
        self.verbose = True
        self.help = False
        for attr in self.display_option_names:
            setattr(self, attr, False)

        # Store the distribution meta-data (name, version, author, and so
        # forth) in a separate object -- we're getting to have enough
        # information here (and enough command-line options) that it's
        # worth it.  Also delegate 'get_XXX()' methods to the 'metadata'
        # object in a sneaky and underhanded (but efficient!) way.
        self.metadata = DistributionMetadata()
        for basename in self.metadata._METHOD_BASENAMES:
            method_name = "get_" + basename
            setattr(self, method_name, getattr(self.metadata, method_name))

        # 'cmdclass' maps command names to class objects, so we
        # can 1) quickly figure out which class to instantiate when
        # we need to create a new command object, and 2) have a way
        # for the setup script to override command classes
        self.cmdclass: dict[str, type[Command]] = {}

        # 'command_packages' is a list of packages in which commands
        # are searched for.  The factory for command 'foo' is expected
        # to be named 'foo' in the module 'foo' in one of the packages
        # named here.  This list is searched from the left; an error
        # is raised if no named package provides the command being
        # searched for.  (Always access using get_command_packages().)
        self.command_packages: str | list[str] | None = None

        # 'script_name' and 'script_args' are usually set to sys.argv[0]
        # and sys.argv[1:], but they can be overridden when the caller is
        # not necessarily a setup script run from the command-line.
        self.script_name: str | os.PathLike[str] | None = None
        self.script_args: list[str] | None = None

        # 'command_options' is where we store command options between
        # parsing them (from config files, the command-line, etc.) and when
        # they are actually needed -- ie. when the command in question is
        # instantiated.  It is a dictionary of dictionaries of 2-tuples:
        #   command_options = { command_name : { option : (source, value) } }
        self.command_options: dict[str, dict[str, tuple[str, str]]] = {}

        # 'dist_files' is the list of (command, pyversion, file) that
        # have been created by any dist commands run so far. This is
        # filled regardless of whether the run is dry or not. pyversion
        # gives sysconfig.get_python_version() if the dist file is
        # specific to a Python version, 'any' if it is good for all
        # Python versions on the target platform, and '' for a source
        # file. pyversion should not be used to specify minimum or
        # maximum required Python versions; use the metainfo for that
        # instead.
        self.dist_files: list[tuple[str, str, str]] = []

        # These options are really the business of various commands, rather
        # than of the Distribution itself.  We provide aliases for them in
        # Distribution as a convenience to the developer.
        self.packages: list[str] | None = None
        self.package_data: dict[str, list[str]] = {}
        self.package_dir: Mapping[str, str] | None = None
        self.py_modules: list[str] | None = None
        self.libraries: list[tuple[str, dict[str, Any]]] | None = None
        self.headers: list[str] | None = None
        self.ext_modules: list[Extension] | None = None
        self.ext_package: str | None = None
        self.include_dirs: list[str] | None = None
        self.extra_path = None
        self.scripts: list[str] | None = None
        self.data_files: list[tuple[str, list[str]]] | None = None
        self.password: str = ''

        # And now initialize bookkeeping stuff that can't be supplied by
        # the caller at all.  'command_obj' maps command names to
        # Command instances -- that's how we enforce that every command
        # class is a singleton.
        self.command_obj: dict[str, Command] = {}

        # 'have_run' maps command names to boolean values; it keeps track
        # of whether we have actually run a particular command, to make it
        # cheap to "run" a command whenever we think we might need to -- if
        # it's already been done, no need for expensive filesystem
        # operations, we just check the 'have_run' dictionary and carry on.
        # It's only safe to query 'have_run' for a command class that has
        # been instantiated -- a false value will be inserted when the
        # command object is created, and replaced with a true value when
        # the command is successfully run.  Thus it's probably best to use
        # '.get()' rather than a straight lookup.
        self.have_run: dict[str, bool] = {}

        # Now we'll use the attrs dictionary (ultimately, keyword args from
        # the setup script) to possibly override any or all of these
        # distribution options.

        if attrs:
            # Pull out the set of command options and work on them
            # specifically.  Note that this order guarantees that aliased
            # command options will override any supplied redundantly
            # through the general options dictionary.
            options: Mapping[str, Mapping[str, str]] | None = attrs.get('options')
            if options is not None:
                del attrs['options']
                for command, cmd_options in options.items():
                    opt_dict = self.get_option_dict(command)
                    for opt, val in cmd_options.items():
                        opt_dict[opt] = ("setup script", val)

            if 'licence' in attrs:
                attrs['license'] = attrs['licence']
                del attrs['licence']
                msg = "'licence' distribution option is deprecated; use 'license'"
                warnings.warn(msg)

            # Now work on the rest of the attributes.  Any attribute that's
            # not already defined is invalid!
            for key, val in attrs.items():
                if hasattr(self.metadata, "set_" + key):
                    getattr(self.metadata, "set_" + key)(val)
                elif hasattr(self.metadata, key):
                    setattr(self.metadata, key, val)
                elif hasattr(self, key):
                    setattr(self, key, val)
                else:
                    msg = f"Unknown distribution option: {key!r}"
                    warnings.warn(msg)

        # no-user-cfg is handled before other command line args
        # because other args override the config files, and this
        # one is needed before we can load the config files.
        # If attrs['script_args'] wasn't passed, assume false.
        #
        # This also make sure we just look at the global options
        self.want_user_cfg = True

        if self.script_args is not None:
            # Coerce any possible iterable from attrs into a list
            self.script_args = list(self.script_args)
            for arg in self.script_args:
                if not arg.startswith('-'):
                    break
                if arg == '--no-user-cfg':
                    self.want_user_cfg = False
                    break

        self.finalize_options()

    def get_option_dict(self, command):
        """Get the option dictionary for a given command.  If that
        command's option dictionary hasn't been created yet, then create it
        and return the new dictionary; otherwise, return the existing
        option dictionary.
        """
        dict = self.command_options.get(command)
        if dict is None:
            dict = self.command_options[command] = {}
        return dict

    def dump_option_dicts(self, header=None, commands=None, indent: str = "") -> None:
        from pprint import pformat

        if commands is None:  # dump all command option dicts
            commands = sorted(self.command_options.keys())

        if header is not None:
            self.announce(indent + header)
            indent = indent + "  "

        if not commands:
            self.announce(indent + "no commands known yet")
            return

        for cmd_name in commands:
            opt_dict = self.command_options.get(cmd_name)
            if opt_dict is None:
                self.announce(indent + f"no option dict for '{cmd_name}' command")
            else:
                self.announce(indent + f"option dict for '{cmd_name}' command:")
                out = pformat(opt_dict)
                for line in out.split('\n'):
                    self.announce(indent + "  " + line)

    # -- Config file finding/parsing methods ---------------------------

    def find_config_files(self):
        """Find as many configuration files as should be processed for this
        platform, and return a list of filenames in the order in which they
        should be parsed.  The filenames returned are guaranteed to exist
        (modulo nasty race conditions).

        There are multiple possible config files:
        - distutils.cfg in the Distutils installation directory (i.e.
          where the top-level Distutils __inst__.py file lives)
        - a file in the user's home directory named .pydistutils.cfg
          on Unix and pydistutils.cfg on Windows/Mac; may be disabled
          with the ``--no-user-cfg`` option
        - setup.cfg in the current directory
        - a file named by an environment variable
        """
        check_environ()
        files = [str(path) for path in self._gen_paths() if os.path.isfile(path)]

        if DEBUG:
            self.announce("using config files: {}".format(', '.join(files)))

        return files

    def _gen_paths(self):
        # The system-wide Distutils config file
        sys_dir = pathlib.Path(sys.modules['distutils'].__file__).parent
        yield sys_dir / "distutils.cfg"

        # The per-user config file
        prefix = '.' * (os.name == 'posix')
        filename = prefix + 'pydistutils.cfg'
        if self.want_user_cfg:
            with contextlib.suppress(RuntimeError):
                yield pathlib.Path('~').expanduser() / filename

        # All platforms support local setup.cfg
        yield pathlib.Path('setup.cfg')

        # Additional config indicated in the environment
        with contextlib.suppress(TypeError):
            yield pathlib.Path(os.getenv("DIST_EXTRA_CONFIG"))

    def parse_config_files(self, filenames=None):  # noqa: C901
        from configparser import ConfigParser

        # Ignore install directory options if we have a venv
        if sys.prefix != sys.base_prefix:
            ignore_options = [
                'install-base',
                'install-platbase',
                'install-lib',
                'install-platlib',
                'install-purelib',
                'install-headers',
                'install-scripts',
                'install-data',
                'prefix',
                'exec-prefix',
                'home',
                'user',
                'root',
            ]
        else:
            ignore_options = []

        ignore_options = frozenset(ignore_options)

        if filenames is None:
            filenames = self.find_config_files()

        if DEBUG:
            self.announce("Distribution.parse_config_files():")

        parser = ConfigParser()
        for filename in filenames:
            if DEBUG:
                self.announce(f"  reading {filename}")
            parser.read(filename, encoding='utf-8')
            for section in parser.sections():
                options = parser.options(section)
                opt_dict = self.get_option_dict(section)

                for opt in options:
                    if opt != '__name__' and opt not in ignore_options:
                        val = parser.get(section, opt)
                        opt = opt.replace('-', '_')
                        opt_dict[opt] = (filename, val)

            # Make the ConfigParser forget everything (so we retain
            # the original filenames that options come from)
            parser.__init__()

        # If there was a "global" section in the config file, use it
        # to set Distribution options.

        if 'global' in self.command_options:
            for opt, (_src, val) in self.command_options['global'].items():
                alias = self.negative_opt.get(opt)
                try:
                    if alias:
                        setattr(self, alias, not strtobool(val))
                    elif opt == 'verbose':
                        setattr(self, opt, strtobool(val))
                    else:
                        setattr(self, opt, val)
                except ValueError as msg:
                    raise DistutilsOptionError(msg)

    # -- Command-line parsing methods ----------------------------------

    def parse_command_line(self):
        """Parse the setup script's command line, taken from the
        'script_args' instance attribute (which defaults to 'sys.argv[1:]'
        -- see 'setup()' in core.py).  This list is first processed for
        "global options" -- options that set attributes of the Distribution
        instance.  Then, it is alternately scanned for Distutils commands
        and options for that command.  Each new command terminates the
        options for the previous command.  The allowed options for a
        command are determined by the 'user_options' attribute of the
        command class -- thus, we have to be able to load command classes
        in order to parse the command line.  Any error in that 'options'
        attribute raises DistutilsGetoptError; any error on the
        command-line raises DistutilsArgError.  If no Distutils commands
        were found on the command line, raises DistutilsArgError.  Return
        true if command-line was successfully parsed and we should carry
        on with executing commands; false if no errors but we shouldn't
        execute commands (currently, this only happens if user asks for
        help).
        """
        #
        # We now have enough information to show the Macintosh dialog
        # that allows the user to interactively specify the "command line".
        #
        toplevel_options = self._get_toplevel_options()

        # We have to parse the command line a bit at a time -- global
        # options, then the first command, then its options, and so on --
        # because each command will be handled by a different class, and
        # the options that are valid for a particular class aren't known
        # until we have loaded the command class, which doesn't happen
        # until we know what the command is.

        self.commands = []
        parser = FancyGetopt(toplevel_options + self.display_options)
        parser.set_negative_aliases(self.negative_opt)
        parser.set_aliases({'licence': 'license'})
        args = parser.getopt(args=self.script_args, object=self)
        option_order = parser.get_option_order()
        logging.getLogger().setLevel(logging.WARNING - 10 * self.verbose)

        # for display options we return immediately
        if self.handle_display_options(option_order):
            return
        while args:
            args = self._parse_command_opts(parser, args)
            if args is None:  # user asked for help (and got it)
                return

        # Handle the cases of --help as a "global" option, ie.
        # "setup.py --help" and "setup.py --help command ...".  For the
        # former, we show global options (--verbose, --dry-run, etc.)
        # and display-only options (--name, --version, etc.); for the
        # latter, we omit the display-only options and show help for
        # each command listed on the command line.
        if self.help:
            self._show_help(
                parser, display_options=len(self.commands) == 0, commands=self.commands
            )
            return

        # Oops, no commands found -- an end-user error
        if not self.commands:
            raise DistutilsArgError("no commands supplied")

        # All is well: return true
        return True

    def _get_toplevel_options(self):
        """Return the non-display options recognized at the top level.

        This includes options that are recognized *only* at the top
        level as well as options recognized for commands.
        """
        return self.global_options + [
            (
                "command-packages=",
                None,
                "list of packages that provide distutils commands",
            ),
        ]

    def _parse_command_opts(self, parser, args):  # noqa: C901
        """Parse the command-line options for a single command.
        'parser' must be a FancyGetopt instance; 'args' must be the list
        of arguments, starting with the current command (whose options
        we are about to parse).  Returns a new version of 'args' with
        the next command at the front of the list; will be the empty
        list if there are no more commands on the command line.  Returns
        None if the user asked for help on this command.
        """
        # late import because of mutual dependence between these modules
        from distutils.cmd import Command

        # Pull the current command from the head of the command line
        command = args[0]
        if not command_re.match(command):
            raise SystemExit(f"invalid command name '{command}'")
        self.commands.append(command)

        # Dig up the command class that implements this command, so we
        # 1) know that it's a valid command, and 2) know which options
        # it takes.
        try:
            cmd_class = self.get_command_class(command)
        except DistutilsModuleError as msg:
            raise DistutilsArgError(msg)

        # Require that the command class be derived from Command -- want
        # to be sure that the basic "command" interface is implemented.
        if not issubclass(cmd_class, Command):
            raise DistutilsClassError(
                f"command class {cmd_class} must subclass Command"
            )

        # Also make sure that the command object provides a list of its
        # known options.
        if not (
            hasattr(cmd_class, 'user_options')
            and isinstance(cmd_class.user_options, list)
        ):
            raise DistutilsClassError(
                f"command class {cmd_class} must provide "
                "'user_options' attribute (a list of tuples)"
            )

        # If the command class has a list of negative alias options,
        # merge it in with the global negative aliases.
        negative_opt = self.negative_opt
        if hasattr(cmd_class, 'negative_opt'):
            negative_opt = negative_opt.copy()
            negative_opt.update(cmd_class.negative_opt)

        # Check for help_options in command class.  They have a different
        # format (tuple of four) so we need to preprocess them here.
        if hasattr(cmd_class, 'help_options') and isinstance(
            cmd_class.help_options, list
        ):
            help_options = fix_help_options(cmd_class.help_options)
        else:
            help_options = []

        # All commands support the global options too, just by adding
        # in 'global_options'.
        parser.set_option_table(
            self.global_options + cmd_class.user_options + help_options
        )
        parser.set_negative_aliases(negative_opt)
        (args, opts) = parser.getopt(args[1:])
        if hasattr(opts, 'help') and opts.help:
            self._show_help(parser, display_options=False, commands=[cmd_class])
            return

        if hasattr(cmd_class, 'help_options') and isinstance(
            cmd_class.help_options, list
        ):
            help_option_found = 0
            for help_option, _short, _desc, func in cmd_class.help_options:
                if hasattr(opts, parser.get_attr_name(help_option)):
                    help_option_found = 1
                    if callable(func):
                        func()
                    else:
                        raise DistutilsClassError(
                            f"invalid help function {func!r} for help option '{help_option}': "
                            "must be a callable object (function, etc.)"
                        )

            if help_option_found:
                return

        # Put the options from the command-line into their official
        # holding pen, the 'command_options' dictionary.
        opt_dict = self.get_option_dict(command)
        for name, value in vars(opts).items():
            opt_dict[name] = ("command line", value)

        return args

    def finalize_options(self) -> None:
        """Set final values for all the options on the Distribution
        instance, analogous to the .finalize_options() method of Command
        objects.
        """
        for attr in ('keywords', 'platforms'):
            value = getattr(self.metadata, attr)
            if value is None:
                continue
            if isinstance(value, str):
                # SeparatedValues strips whitespace and discards empty items.
                value = list(SeparatedValues(_repair_newlines(value)))
                setattr(self.metadata, attr, value)

    def _show_help(
        self, parser, global_options=True, display_options=True, commands: Iterable = ()
    ):
        """Show help for the setup script command-line in the form of
        several lists of command-line options.  'parser' should be a
        FancyGetopt instance; do not expect it to be returned in the
        same state, as its option table will be reset to make it
        generate the correct help text.

        If 'global_options' is true, lists the global options:
        --verbose, --dry-run, etc.  If 'display_options' is true, lists
        the "display-only" options: --name, --version, etc.  Finally,
        lists per-command help for every command name or command class
        in 'commands'.
        """
        # late import because of mutual dependence between these modules
        from distutils.cmd import Command
        from distutils.core import gen_usage

        if global_options:
            if display_options:
                options = self._get_toplevel_options()
            else:
                options = self.global_options
            parser.set_option_table(options)
            parser.print_help(self.common_usage + "\nGlobal options:")
            print()

        if display_options:
            parser.set_option_table(self.display_options)
            parser.print_help(
                "Information display options (just display information, ignore any commands)"
            )
            print()

        for command in commands:
            if isinstance(command, type) and issubclass(command, Command):
                klass = command
            else:
                klass = self.get_command_class(command)
            if hasattr(klass, 'help_options') and isinstance(klass.help_options, list):
                parser.set_option_table(
                    klass.user_options + fix_help_options(klass.help_options)
                )
            else:
                parser.set_option_table(klass.user_options)
            parser.print_help(f"Options for '{klass.__name__}' command:")
            print()

        print(gen_usage(self.script_name))

    def handle_display_options(self, option_order):
        """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
        from distutils.core import gen_usage

        # User just wants a list of commands -- we'll print it out and stop
        # processing now (ie. if they ran "setup --help-commands foo bar",
        # we ignore "foo bar").
        if self.help_commands:
            self.print_commands()
            print()
            print(gen_usage(self.script_name))
            return 1

        # If user supplied any of the "display metadata" options, then
        # display that metadata in the order in which the user supplied the
        # metadata options.
        any_display_options = 0
        is_display_option = set()
        for option in self.display_options:
            is_display_option.add(option[0])

        for opt, val in option_order:
            if val and opt in is_display_option:
                opt = translate_longopt(opt)
                value = getattr(self.metadata, "get_" + opt)()
                if opt in ('keywords', 'platforms'):
                    print(','.join(value))
                elif opt in ('classifiers', 'provides', 'requires', 'obsoletes'):
                    print('\n'.join(value))
                else:
                    print(value)
                any_display_options = 1

        return any_display_options

    def print_command_list(self, commands, header, max_length) -> None:
        """Print a subset of the list of all commands -- used by
        'print_commands()'.
        """
        print(header + ":")

        for cmd in commands:
            klass = self.cmdclass.get(cmd)
            if not klass:
                klass = self.get_command_class(cmd)
            try:
                description = klass.description
            except AttributeError:
                description = "(no description available)"

            print(f"  {cmd:<{max_length}}  {description}")

    def print_commands(self) -> None:
        """Print out a help message listing all available commands with a
        description of each.  The list is divided into "standard commands"
        (listed in distutils.command.__all__) and "extra commands"
        (mentioned in self.cmdclass, but not a standard command).  The
        descriptions come from the command class attribute
        'description'.
        """
        import distutils.command

        std_commands = distutils.command.__all__
        is_std = set(std_commands)

        extra_commands = [cmd for cmd in self.cmdclass if cmd not in is_std]

        max_length = 0
        for cmd in std_commands + extra_commands:
            max_length = max(max_length, len(cmd))

        self.print_command_list(std_commands, "Standard commands", max_length)
        if extra_commands:
            print()
            self.print_command_list(extra_commands, "Extra commands", max_length)

    def get_command_list(self):
        """Get a list of (command, description) tuples.
        The list is divided into "standard commands" (listed in
        distutils.command.__all__) and "extra commands" (mentioned in
        self.cmdclass, but not a standard command).  The descriptions come
        from the command class attribute 'description'.
        """
        # Currently this is only used on Mac OS, for the Mac-only GUI
        # Distutils interface (by Jack Jansen)
        import distutils.command

        std_commands = distutils.command.__all__
        is_std = set(std_commands)

        extra_commands = [cmd for cmd in self.cmdclass if cmd not in is_std]

        rv = []
        for cmd in std_commands + extra_commands:
            klass = self.cmdclass.get(cmd)
            if not klass:
                klass = self.get_command_class(cmd)
            try:
                description = klass.description
            except AttributeError:
                description = "(no description available)"
            rv.append((cmd, description))
        return rv

    # -- Command class/object methods ----------------------------------

    def get_command_packages(self):
        """Return a list of packages from which commands are loaded."""
        pkgs = self.command_packages
        if not isinstance(pkgs, list):
            if pkgs is None:
                pkgs = ''
            pkgs = [pkg.strip() for pkg in pkgs.split(',') if pkg != '']
            if "distutils.command" not in pkgs:
                pkgs.insert(0, "distutils.command")
            self.command_packages = pkgs
        return pkgs

    def get_command_class(self, command: str) -> type[Command]:
        """Return the class that implements the Distutils command named by
        'command'.  First we check the 'cmdclass' dictionary; if the
        command is mentioned there, we fetch the class object from the
        dictionary and return it.  Otherwise we load the command module
        ("distutils.command." + command) and fetch the command class from
        the module.  The loaded class is also stored in 'cmdclass'
        to speed future calls to 'get_command_class()'.

        Raises DistutilsModuleError if the expected module could not be
        found, or if that module does not define the expected class.
        """
        klass = self.cmdclass.get(command)
        if klass:
            return klass

        for pkgname in self.get_command_packages():
            module_name = f"{pkgname}.{command}"
            klass_name = command

            try:
                __import__(module_name)
                module = sys.modules[module_name]
            except ImportError:
                continue

            try:
                klass = cast("type[Command]", getattr(module, klass_name))
            except AttributeError:
                raise DistutilsModuleError(
                    f"invalid command '{command}' (no class '{klass_name}' in module '{module_name}')"
                )

            self.cmdclass[command] = klass
            return klass

        raise DistutilsModuleError(f"invalid command '{command}'")

    @overload
    def get_command_obj(
        self, command: str, create: Literal[True] = True
    ) -> Command: ...
    @overload
    def get_command_obj(self, command: str, create: bool) -> Command | None: ...
    def get_command_obj(self, command: str, create: bool = True) -> Command | None:
        """Return the command object for 'command'.  Normally this object
        is cached on a previous call to 'get_command_obj()'; if no command
        object for 'command' is in the cache, then we either create and
        return it (if 'create' is true) or return None.
        """
        cmd_obj = self.command_obj.get(command)
        if not cmd_obj and create:
            if DEBUG:
                self.announce(
                    "Distribution.get_command_obj(): "
                    f"creating '{command}' command object"
                )

            klass = self.get_command_class(command)
            cmd_obj = self.command_obj[command] = klass(self)
            self.have_run[command] = False

            # Set any options that were supplied in config files
            # or on the command line.  (NB. support for error
            # reporting is lame here: any errors aren't reported
            # until 'finalize_options()' is called, which means
            # we won't report the source of the error.)
            options = self.command_options.get(command)
            if options:
                self._set_command_options(cmd_obj, options)

        return cmd_obj

    def _set_command_options(self, command_obj, option_dict=None):  # noqa: C901
        """Set the options for 'command_obj' from 'option_dict'.  Basically
        this means copying elements of a dictionary ('option_dict') to
        attributes of an instance ('command').

        'command_obj' must be a Command instance.  If 'option_dict' is not
        supplied, uses the standard option dictionary for this command
        (from 'self.command_options').
        """
        command_name = command_obj.get_command_name()
        if option_dict is None:
            option_dict = self.get_option_dict(command_name)

        if DEBUG:
            self.announce(f"  setting options for '{command_name}' command:")
        for option, (source, value) in option_dict.items():
            if DEBUG:
                self.announce(f"    {option} = {value} (from {source})")
            try:
                bool_opts = [translate_longopt(o) for o in command_obj.boolean_options]
            except AttributeError:
                bool_opts = []
            try:
                neg_opt = command_obj.negative_opt
            except AttributeError:
                neg_opt = {}

            try:
                is_string = isinstance(value, str)
                if option in neg_opt and is_string:
                    setattr(command_obj, neg_opt[option], not strtobool(value))
                elif option in bool_opts and is_string:
                    setattr(command_obj, option, strtobool(value))
                elif hasattr(command_obj, option):
                    setattr(command_obj, option, value)
                else:
                    raise DistutilsOptionError(
                        f"error in {source}: command '{command_name}' has no such option '{option}'"
                    )
            except ValueError as msg:
                raise DistutilsOptionError(msg)

    @overload
    def reinitialize_command(
        self, command: str, reinit_subcommands: bool = False
    ) -> Command: ...
    @overload
    def reinitialize_command(
        self, command: _CommandT, reinit_subcommands: bool = False
    ) -> _CommandT: ...
    def reinitialize_command(
        self, command: str | Command, reinit_subcommands=False
    ) -> Command:
        """Reinitializes a command to the state it was in when first
        returned by 'get_command_obj()': ie., initialized but not yet
        finalized.  This provides the opportunity to sneak option
        values in programmatically, overriding or supplementing
        user-supplied values from the config files and command line.
        You'll have to re-finalize the command object (by calling
        'finalize_options()' or 'ensure_finalized()') before using it for
        real.

        'command' should be a command name (string) or command object.  If
        'reinit_subcommands' is true, also reinitializes the command's
        sub-commands, as declared by the 'sub_commands' class attribute (if
        it has one).  See the "install" command for an example.  Only
        reinitializes the sub-commands that actually matter, ie. those
        whose test predicates return true.

        Returns the reinitialized command object.
        """
        from distutils.cmd import Command

        if not isinstance(command, Command):
            command_name = command
            command = self.get_command_obj(command_name)
        else:
            command_name = command.get_command_name()

        if not command.finalized:
            return command
        command.initialize_options()
        command.finalized = False
        self.have_run[command_name] = False
        self._set_command_options(command)

        if reinit_subcommands:
            for sub in command.get_sub_commands():
                self.reinitialize_command(sub, reinit_subcommands)

        return command

    # -- Methods that operate on the Distribution ----------------------

    def announce(self, msg, level: int = logging.INFO) -> None:
        log.log(level, msg)

    def run_commands(self) -> None:
        """Run each command that was seen on the setup script command line.
        Uses the list of commands found and cache of command objects
        created by 'get_command_obj()'.
        """
        for cmd in self.commands:
            self.run_command(cmd)

    # -- Methods that operate on its Commands --------------------------

    def run_command(self, command: str) -> None:
        """Do whatever it takes to run a command (including nothing at all,
        if the command has already been run).  Specifically: if we have
        already created and run the command named by 'command', return
        silently without doing anything.  If the command named by 'command'
        doesn't even have a command object yet, create one.  Then invoke
        'run()' on that command object (or an existing one).
        """
        # Already been here, done that? then return silently.
        if self.have_run.get(command):
            return

        log.info("running %s", command)
        cmd_obj = self.get_command_obj(command)
        cmd_obj.ensure_finalized()
        cmd_obj.run()
        self.have_run[command] = True

    # -- Distribution query methods ------------------------------------

    def has_pure_modules(self) -> bool:
        return bool(self.packages or self.py_modules)

    def has_ext_modules(self) -> bool:
        return bool(self.ext_modules)

    def has_c_libraries(self) -> bool:
        return bool(self.libraries)

    def has_modules(self) -> bool:
        return self.has_pure_modules() or self.has_ext_modules()

    def has_headers(self) -> bool:
        return bool(self.headers)

    def has_scripts(self) -> bool:
        return bool(self.scripts)

    def has_data_files(self) -> bool:
        return bool(self.data_files)

    def is_pure(self) -> bool:
        return (
            self.has_pure_modules()
            and not self.has_ext_modules()
            and not self.has_c_libraries()
        )

    # -- Metadata query methods ----------------------------------------

    # If you're looking for 'get_name()', 'get_version()', and so forth,
    # they are defined in a sneaky way: the constructor binds self.get_XXX
    # to self.metadata.get_XXX.  The actual code is in the
    # DistributionMetadata class, below.
    if TYPE_CHECKING:
        # Unfortunately this means we need to specify them manually or not expose statically
        def _(self) -> None:
            self.get_name = self.metadata.get_name
            self.get_version = self.metadata.get_version
            self.get_fullname = self.metadata.get_fullname
            self.get_author = self.metadata.get_author
            self.get_author_email = self.metadata.get_author_email
            self.get_maintainer = self.metadata.get_maintainer
            self.get_maintainer_email = self.metadata.get_maintainer_email
            self.get_contact = self.metadata.get_contact
            self.get_contact_email = self.metadata.get_contact_email
            self.get_url = self.metadata.get_url
            self.get_license = self.metadata.get_license
            self.get_licence = self.metadata.get_licence
            self.get_description = self.metadata.get_description
            self.get_long_description = self.metadata.get_long_description
            self.get_keywords = self.metadata.get_keywords
            self.get_platforms = self.metadata.get_platforms
            self.get_classifiers = self.metadata.get_classifiers
            self.get_download_url = self.metadata.get_download_url
            self.get_requires = self.metadata.get_requires
            self.get_provides = self.metadata.get_provides
            self.get_obsoletes = self.metadata.get_obsoletes

        # Default attributes generated in __init__ from self.display_option_names
        help_commands: bool
        name: str | Literal[False]
        version: str | Literal[False]
        fullname: str | Literal[False]
        author: str | Literal[False]
        author_email: str | Literal[False]
        maintainer: str | Literal[False]
        maintainer_email: str | Literal[False]
        contact: str | Literal[False]
        contact_email: str | Literal[False]
        url: str | Literal[False]
        license: str | Literal[False]
        licence: str | Literal[False]
        description: str | Literal[False]
        long_description: str | Literal[False]
        platforms: str | list[str] | Literal[False]
        classifiers: str | list[str] | Literal[False]
        keywords: str | list[str] | Literal[False]
        provides: list[str] | Literal[False]
        requires: list[str] | Literal[False]
        obsoletes: list[str] | Literal[False]


class DistributionMetadata:
    """Dummy class to hold the distribution meta-data: name, version,
    author, and so forth.
    """

    _METHOD_BASENAMES = (
        "name",
        "version",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "url",
        "license",
        "description",
        "long_description",
        "keywords",
        "platforms",
        "fullname",
        "contact",
        "contact_email",
        "classifiers",
        "download_url",
        # PEP 314
        "provides",
        "requires",
        "obsoletes",
    )

    def __init__(
        self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None = None
    ) -> None:
        if path is not None:
            with open(path) as f:
                self.read_pkg_file(f)
        else:
            self.name: str | None = None
            self.version: str | None = None
            self.author: str | None = None
            self.author_email: str | None = None
            self.maintainer: str | None = None
            self.maintainer_email: str | None = None
            self.url: str | None = None
            self.license: str | None = None
            self.description: str | None = None
            self.long_description: str | None = None
            self.keywords: str | list[str] | None = None
            self.platforms: str | list[str] | None = None
            self.classifiers: str | list[str] | None = None
            self.download_url: str | None = None
            # PEP 314
            self.provides: str | list[str] | None = None
            self.requires: str | list[str] | None = None
            self.obsoletes: str | list[str] | None = None

    def read_pkg_file(self, file: IO[str]) -> None:
        """Reads the metadata values from a file object."""
        msg = message_from_file(file)

        def _read_field(name: str) -> str | None:
            value = msg[name]
            if value and value != "UNKNOWN":
                return value
            return None

        def _read_list(name):
            values = msg.get_all(name, None)
            if values == []:
                return None
            return values

        metadata_version = msg['metadata-version']
        self.name = _read_field('name')
        self.version = _read_field('version')
        self.description = _read_field('summary')
        # we are filling author only.
        self.author = _read_field('author')
        self.maintainer = None
        self.author_email = _read_field('author-email')
        self.maintainer_email = None
        self.url = _read_field('home-page')
        self.license = _read_field('license')

        if 'download-url' in msg:
            self.download_url = _read_field('download-url')
        else:
            self.download_url = None

        self.long_description = _read_field('description')
        self.description = _read_field('summary')

        if 'keywords' in msg:
            self.keywords = _read_field('keywords').split(',')  # type:ignore[union-attr] # Manually checked

        self.platforms = _read_list('platform')
        self.classifiers = _read_list('classifier')

        # PEP 314 - these fields only exist in 1.1
        if metadata_version == '1.1':
            self.requires = _read_list('requires')
            self.provides = _read_list('provides')
            self.obsoletes = _read_list('obsoletes')
        else:
            self.requires = None
            self.provides = None
            self.obsoletes = None

    def write_pkg_info(self, base_dir: str | os.PathLike[str]) -> None:
        """Write the PKG-INFO file into the release tree."""
        with open(
            os.path.join(base_dir, 'PKG-INFO'), 'w', encoding='UTF-8'
        ) as pkg_info:
            self.write_pkg_file(pkg_info)

    def write_pkg_file(self, file: SupportsWrite[str]) -> None:
        """Write the PKG-INFO format data to a file object."""
        version = '1.0'
        if (
            self.provides
            or self.requires
            or self.obsoletes
            or self.classifiers
            or self.download_url
        ):
            version = '1.1'

        # required fields
        file.write(f'Metadata-Version: {version}\n')
        file.write(f'Name: {self.get_name()}\n')
        file.write(f'Version: {self.get_version()}\n')

        def maybe_write(header, val):
            if val:
                file.write(f"{header}: {val}\n")

        # optional fields
        maybe_write("Summary", self.get_description())
        maybe_write("Home-page", self.get_url())
        maybe_write("Author", self.get_contact())
        maybe_write("Author-email", self.get_contact_email())
        maybe_write("License", self.get_license())
        maybe_write("Download-URL", self.download_url)
        maybe_write("Description", rfc822_escape(self.get_long_description() or ""))
        maybe_write("Keywords", ",".join(self.get_keywords()))

        self._write_list(file, 'Platform', self.get_platforms())
        self._write_list(file, 'Classifier', self.get_classifiers())

        # PEP 314
        self._write_list(file, 'Requires', self.get_requires())
        self._write_list(file, 'Provides', self.get_provides())
        self._write_list(file, 'Obsoletes', self.get_obsoletes())

    def _write_list(self, file, name, values):
        values = values or []
        for value in values:
            file.write(f'{name}: {value}\n')

    # -- Metadata query methods ----------------------------------------

    def get_name(self) -> str:
        return self.name or "UNKNOWN"

    def get_version(self) -> str:
        return self.version or "0.0.0"

    def get_fullname(self) -> str:
        return self._fullname(self.get_name(), self.get_version())

    @staticmethod
    def _fullname(name: str, version: str) -> str:
        """
        >>> DistributionMetadata._fullname('setup.tools', '1.0-2')
        'setup_tools-1.0.post2'
        >>> DistributionMetadata._fullname('setup-tools', '1.2post2')
        'setup_tools-1.2.post2'
        >>> DistributionMetadata._fullname('setup-tools', '1.0-r2')
        'setup_tools-1.0.post2'
        >>> DistributionMetadata._fullname('setup.tools', '1.0.post')
        'setup_tools-1.0.post0'
        >>> DistributionMetadata._fullname('setup.tools', '1.0+ubuntu-1')
        'setup_tools-1.0+ubuntu.1'
        """
        return "{}-{}".format(
            canonicalize_name(name).replace('-', '_'),
            canonicalize_version(version, strip_trailing_zero=False),
        )

    def get_author(self) -> str | None:
        return self.author

    def get_author_email(self) -> str | None:
        return self.author_email

    def get_maintainer(self) -> str | None:
        return self.maintainer

    def get_maintainer_email(self) -> str | None:
        return self.maintainer_email

    def get_contact(self) -> str | None:
        return self.maintainer or self.author

    def get_contact_email(self) -> str | None:
        return self.maintainer_email or self.author_email

    def get_url(self) -> str | None:
        return self.url

    def get_license(self) -> str | None:
        return self.license

    get_licence = get_license

    def get_description(self) -> str | None:
        return self.description

    def get_long_description(self) -> str | None:
        return self.long_description

    def get_keywords(self) -> str | list[str]:
        return self.keywords or []

    def set_keywords(self, value: str | Iterable[str]) -> None:
        self.keywords = _ensure_list(value, 'keywords')

    def get_platforms(self) -> str | list[str] | None:
        return self.platforms

    def set_platforms(self, value: str | Iterable[str]) -> None:
        self.platforms = _ensure_list(value, 'platforms')

    def get_classifiers(self) -> str | list[str]:
        return self.classifiers or []

    def set_classifiers(self, value: str | Iterable[str]) -> None:
        self.classifiers = _ensure_list(value, 'classifiers')

    def get_download_url(self) -> str | None:
        return self.download_url

    # PEP 314
    def get_requires(self) -> str | list[str]:
        return self.requires or []

    def set_requires(self, value: Iterable[str]) -> None:
        import distutils.versionpredicate

        for v in value:
            distutils.versionpredicate.VersionPredicate(v)
        self.requires = list(value)

    def get_provides(self) -> str | list[str]:
        return self.provides or []

    def set_provides(self, value: Iterable[str]) -> None:
        value = [v.strip() for v in value]
        for v in value:
            import distutils.versionpredicate

            distutils.versionpredicate.split_provision(v)
        self.provides = value

    def get_obsoletes(self) -> str | list[str]:
        return self.obsoletes or []

    def set_obsoletes(self, value: Iterable[str]) -> None:
        import distutils.versionpredicate

        for v in value:
            distutils.versionpredicate.VersionPredicate(v)
        self.obsoletes = list(value)


def fix_help_options(options):
    """Convert a 4-tuple 'help_options' list as found in various command
    classes to the 3-tuple form required by FancyGetopt.
    """
    return [opt[0:3] for opt in options]
<|endoftext|>"""distutils.spawn

Provides the 'spawn()' function, a front-end to various platform-
specific functions for launching another program in a sub-process.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import warnings
from collections.abc import MutableSequence

from ._log import log
from .errors import DistutilsExecError


@contextlib.contextmanager
def _translate_errors(cmd):
    """Reraise a subprocess failure running 'cmd' as a DistutilsExecError."""
    try:
        yield
    except OSError as exc:
        raise DistutilsExecError(f"command {cmd[0]!r} failed: {exc.args[-1]}") from exc
    except subprocess.CalledProcessError as err:
        raise DistutilsExecError(
            f"command {cmd[0]!r} failed with exit code {err.returncode}"
        ) from err


def spawn(cmd: MutableSequence[bytes | str | os.PathLike[str]], **kwargs) -> None:
    """Run another program, specified as a command list 'cmd', in a new process.

    'cmd' is just the argument list for the new process, ie.
    cmd[0] is the program to run and cmd[1:] are the rest of its arguments.
    Any keyword arguments are passed through to ``subprocess.check_call``.

    Raise DistutilsExecError if running the program fails in any way; just
    return on success.
    """
    log.info(subprocess.list2cmdline(cmd))
    with _translate_errors(cmd):
        subprocess.check_call(cmd, **kwargs)


def find_executable(executable: str, path: str | None = None) -> str | None:
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    warnings.warn(
        'Use shutil.which instead of find_executable', DeprecationWarning, stacklevel=2
    )
    _, ext = os.path.splitext(executable)
    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'

    if os.path.isfile(executable):
        return executable

    if path is None:
        path = os.environ.get('PATH', None)
        # bpo-35755: Don't fall through if PATH is the empty string
        if path is None:
            try:
                path = os.confstr("CS_PATH")
            except (AttributeError, ValueError):
                # os.confstr() or CS_PATH is not available
                path = os.defpath

    # PATH='' doesn't match, whereas PATH=':' looks in the current directory
    if not path:
        return None

    paths = path.split(os.pathsep)
    for p in paths:
        f = os.path.join(p, executable)
        if os.path.isfile(f):
            # the file exists, we have a shot at spawn working
            return f
    return None
<|endoftext|>import warnings

from .compilers.C import msvc

__all__ = ["MSVCCompiler"]

MSVCCompiler = msvc.Compiler


def __getattr__(name):
    if name == '_get_vc_env':
        warnings.warn(
            "_get_vc_env is private; find an alternative (pypa/distutils#340)"
        )
        return msvc._get_vc_env
    raise AttributeError(name)
<|endoftext|>import importlib
import sys


def bypass_compiler_fixup(cmd, args):
    return cmd


if sys.platform == 'darwin':
    compiler_fixup = importlib.import_module('_osx_support').compiler_fixup
else:
    compiler_fixup = bypass_compiler_fixup
<|endoftext|>"""distutils.filelist

Provides the FileList class, used for poking about the filesystem
and building lists of files.
"""

from __future__ import annotations

import fnmatch
import functools
import os
import re
from collections.abc import Iterable
from typing import Literal, overload

from ._log import log
from .errors import DistutilsInternalError, DistutilsTemplateError
from .util import convert_path


class FileList:
    """A list of files built by on exploring the filesystem and filtered by
    applying various patterns to what we find there.

    Instance attributes:
      dir
        directory from which files will be taken -- only used if
        'allfiles' not supplied to constructor
      files
        list of filenames currently being built/filtered/manipulated
      allfiles
        complete list of files under consideration (ie. without any
        filtering applied)
    """

    def __init__(self, warn: object = None, debug_print: object = None) -> None:
        # ignore argument to FileList, but keep them for backwards
        # compatibility
        self.allfiles: Iterable[str] | None = None
        self.files: list[str] = []

    def set_allfiles(self, allfiles: Iterable[str]) -> None:
        self.allfiles = allfiles

    def findall(self, dir: str | os.PathLike[str] = os.curdir) -> None:
        self.allfiles = findall(dir)

    def debug_print(self, msg: object) -> None:
        """Print 'msg' to stdout if the global DEBUG (taken from the
        DISTUTILS_DEBUG environment variable) flag is true.
        """
        from distutils.debug import DEBUG

        if DEBUG:
            print(msg)

    # Collection methods

    def append(self, item: str) -> None:
        self.files.append(item)

    def extend(self, items: Iterable[str]) -> None:
        self.files.extend(items)

    def sort(self) -> None:
        # Not a strict lexical sort!
        sortable_files = sorted(map(os.path.split, self.files))
        self.files = []
        for sort_tuple in sortable_files:
            self.files.append(os.path.join(*sort_tuple))

    # Other miscellaneous utility methods

    def remove_duplicates(self) -> None:
        # Assumes list has been sorted!
        for i in range(len(self.files) - 1, 0, -1):
            if self.files[i] == self.files[i - 1]:
                del self.files[i]

    # "File template" methods

    def _parse_template_line(self, line):
        words = line.split()
        action = words[0]

        patterns = dir = dir_pattern = None

        if action in ('include', 'exclude', 'global-include', 'global-exclude'):
            if len(words) < 2:
                raise DistutilsTemplateError(
                    f"'{action}' expects <pattern1> <pattern2> ..."
                )
            patterns = [convert_path(w) for w in words[1:]]
        elif action in ('recursive-include', 'recursive-exclude'):
            if len(words) < 3:
                raise DistutilsTemplateError(
                    f"'{action}' expects <dir> <pattern1> <pattern2> ..."
                )
            dir = convert_path(words[1])
            patterns = [convert_path(w) for w in words[2:]]
        elif action in ('graft', 'prune'):
            if len(words) != 2:
                raise DistutilsTemplateError(
                    f"'{action}' expects a single <dir_pattern>"
                )
            dir_pattern = convert_path(words[1])
        else:
            raise DistutilsTemplateError(f"unknown action '{action}'")

        return (action, patterns, dir, dir_pattern)

    def process_template_line(self, line: str) -> None:  # noqa: C901
        # Parse the line: split it up, make sure the right number of words
        # is there, and return the relevant words.  'action' is always
        # defined: it's the first word of the line.  Which of the other
        # three are defined depends on the action; it'll be either
        # patterns, (dir and patterns), or (dir_pattern).
        (action, patterns, dir, dir_pattern) = self._parse_template_line(line)

        # OK, now we know that the action is valid and we have the
        # right number of words on the line for that action -- so we
        # can proceed with minimal error-checking.
        if action == 'include':
            self.debug_print("include " + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=True):
                    log.warning("warning: no files found matching '%s'", pattern)

        elif action == 'exclude':
            self.debug_print("exclude " + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=True):
                    log.warning(
                        "warning: no previously-included files found matching '%s'",
                        pattern,
                    )

        elif action == 'global-include':
            self.debug_print("global-include " + ' '.join(patterns))
            for pattern in patterns:
                if not self.include_pattern(pattern, anchor=False):
                    log.warning(
                        (
                            "warning: no files found matching '%s' "
                            "anywhere in distribution"
                        ),
                        pattern,
                    )

        elif action == 'global-exclude':
            self.debug_print("global-exclude " + ' '.join(patterns))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, anchor=False):
                    log.warning(
                        (
                            "warning: no previously-included files matching "
                            "'%s' found anywhere in distribution"
                        ),
                        pattern,
                    )

        elif action == 'recursive-include':
            self.debug_print("recursive-include {} {}".format(dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.include_pattern(pattern, prefix=dir):
                    msg = "warning: no files found matching '%s' under directory '%s'"
                    log.warning(msg, pattern, dir)

        elif action == 'recursive-exclude':
            self.debug_print("recursive-exclude {} {}".format(dir, ' '.join(patterns)))
            for pattern in patterns:
                if not self.exclude_pattern(pattern, prefix=dir):
                    log.warning(
                        (
                            "warning: no previously-included files matching "
                            "'%s' found under directory '%s'"
                        ),
                        pattern,
                        dir,
                    )

        elif action == 'graft':
            self.debug_print("graft " + dir_pattern)
            if not self.include_pattern(None, prefix=dir_pattern):
                log.warning("warning: no directories found matching '%s'", dir_pattern)

        elif action == 'prune':
            self.debug_print("prune " + dir_pattern)
            if not self.exclude_pattern(None, prefix=dir_pattern):
                log.warning(
                    ("no previously-included directories found matching '%s'"),
                    dir_pattern,
                )
        else:
            raise DistutilsInternalError(
                f"this cannot happen: invalid action '{action}'"
            )

    # Filtering/selection methods
    @overload
    def include_pattern(
        self,
        pattern: str | None,
        anchor: bool = True,
        prefix: str | None = None,
        is_regex: Literal[False] = False,
    ) -> bool: ...
    @overload
    def include_pattern(
        self,
        pattern: str | re.Pattern[str],
        anchor: bool = True,
        prefix: str | None = None,
        *,
        is_regex: Literal[True],
    ) -> bool: ...
    @overload
    def include_pattern(
        self,
        pattern: str | re.Pattern[str],
        anchor: bool,
        prefix: str | None,
        is_regex: Literal[True],
    ) -> bool: ...
    def include_pattern(
        self,
        pattern: str | re.Pattern | None,
        anchor: bool = True,
        prefix: str | None = None,
        is_regex: bool = False,
    ) -> bool:
        """Select strings (presumably filenames) from 'self.files' that
        match 'pattern', a Unix-style wildcard (glob) pattern.  Patterns
        are not quite the same as implemented by the 'fnmatch' module: '*'
        and '?'  match non-special characters, where "special" is platform-
        dependent: slash on Unix; colon, slash, and backslash on
        DOS/Windows; and colon on Mac OS.

        If 'anchor' is true (the default), then the pattern match is more
        stringent: "*.py" will match "foo.py" but not "foo/bar.py".  If
        'anchor' is false, both of these will match.

        If 'prefix' is supplied, then only filenames starting with 'prefix'
        (itself a pattern) and ending with 'pattern', with anything in between
        them, will match.  'anchor' is ignored in this case.

        If 'is_regex' is true, 'anchor' and 'prefix' are ignored, and
        'pattern' is assumed to be either a string containing a regex or a
        regex object -- no translation is done, the regex is just compiled
        and used as-is.

        Selected strings will be added to self.files.

        Return True if files are found, False otherwise.
        """
        # XXX docstring lying about what the special chars are?
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print(f"include_pattern: applying regex r'{pattern_re.pattern}'")

        # delayed loading of allfiles list
        if self.allfiles is None:
            self.findall()
        assert self.allfiles is not None

        for name in self.allfiles:
            if pattern_re.search(name):
                self.debug_print(" adding " + name)
                self.files.append(name)
                files_found = True
        return files_found

    @overload
    def exclude_pattern(
        self,
        pattern: str | None,
        anchor: bool = True,
        prefix: str | None = None,
        is_regex: Literal[False] = False,
    ) -> bool: ...
    @overload
    def exclude_pattern(
        self,
        pattern: str | re.Pattern[str],
        anchor: bool = True,
        prefix: str | None = None,
        *,
        is_regex: Literal[True],
    ) -> bool: ...
    @overload
    def exclude_pattern(
        self,
        pattern: str | re.Pattern[str],
        anchor: bool,
        prefix: str | None,
        is_regex: Literal[True],
    ) -> bool: ...
    def exclude_pattern(
        self,
        pattern: str | re.Pattern | None,
        anchor: bool = True,
        prefix: str | None = None,
        is_regex: bool = False,
    ) -> bool:
        """Remove strings (presumably filenames) from 'files' that match
        'pattern'.  Other parameters are the same as for
        'include_pattern()', above.
        The list 'self.files' is modified in place.
        Return True if files are found, False otherwise.
        """
        files_found = False
        pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
        self.debug_print(f"exclude_pattern: applying regex r'{pattern_re.pattern}'")
        for i in range(len(self.files) - 1, -1, -1):
            if pattern_re.search(self.files[i]):
                self.debug_print(" removing " + self.files[i])
                del self.files[i]
                files_found = True
        return files_found


# Utility functions


def _find_all_simple(path):
    """
    Find all files under 'path'
    """
    all_unique = _UniqueDirs.filter(os.walk(path, followlinks=True))
    results = (
        os.path.join(base, file) for base, dirs, files in all_unique for file in files
    )
    return filter(os.path.isfile, results)


class _UniqueDirs(set):
    """
    Exclude previously-seen dirs from walk results,
    avoiding infinite recursion.
    Ref https://bugs.python.org/issue44497.
    """

    def __call__(self, walk_item):
        """
        Given an item from an os.walk result, determine
        if the item represents a unique dir for this instance
        and if not, prevent further traversal.
        """
        base, dirs, _files = walk_item
        stat = os.stat(base)
        candidate = stat.st_dev, stat.st_ino
        found = candidate in self
        if found:
            del dirs[:]
        self.add(candidate)
        return not found

    @classmethod
    def filter(cls, items):
        return filter(cls(), items)


def findall(dir: str | os.PathLike[str] = os.curdir):
    """
    Find all files under 'dir' and return the list of full filenames.
    Unless dir is '.', return full filenames with dir prepended.
    """
    files = _find_all_simple(dir)
    if dir == os.curdir:
        make_rel = functools.partial(os.path.relpath, start=dir)
        files = map(make_rel, files)
    return list(files)


def glob_to_re(pattern):
    """Translate a shell-like glob pattern to a regular expression; return
    a string containing the regex.  Differs from 'fnmatch.translate()' in
    that '*' does not match "special characters" (which are
    platform-specific).
    """
    pattern_re = fnmatch.translate(pattern)

    # '?' and '*' in the glob pattern become '.' and '.*' in the RE, which
    # IMHO is wrong -- '?' and '*' aren't supposed to match slash in Unix,
    # and by extension they shouldn't match such "special characters" under
    # any OS.  So change all non-escaped dots in the RE to match any
    # character except the special characters (currently: just os.sep).
    sep = os.sep
    if os.sep == '\\':
        # we're using a regex to manipulate a regex, so we need
        # to escape the backslash twice
        sep = r'\\\\'
    escaped = rf'\1[^{sep}]'
    pattern_re = re.sub(r'((?<!\\)(\\\\)*)\.', escaped, pattern_re)
    return pattern_re


def translate_pattern(pattern, anchor=True, prefix=None, is_regex=False):
    """Translate a shell-like wildcard pattern to a compiled regular
    expression.  Return the compiled regex.  If 'is_regex' true,
    then 'pattern' is directly compiled to a regex (if it's a string)
    or just returned as-is (assumes it's a regex object).
    """
    if is_regex:
        if isinstance(pattern, str):
            return re.compile(pattern)
        else:
            return pattern

    # ditch start and end characters
    start, _, end = glob_to_re('_').partition('_')

    if pattern:
        pattern_re = glob_to_re(pattern)
        assert pattern_re.startswith(start) and pattern_re.endswith(end)
    else:
        pattern_re = ''

    if prefix is not None:
        prefix_re = glob_to_re(prefix)
        assert prefix_re.startswith(start) and prefix_re.endswith(end)
        prefix_re = prefix_re[len(start) : len(prefix_re) - len(end)]
        sep = os.sep
        if os.sep == '\\':
            sep = r'\\'
        pattern_re = pattern_re[len(start) : len(pattern_re) - len(end)]
        pattern_re = rf'{start}\A{prefix_re}{sep}.*{pattern_re}{end}'
    else:  # no prefix -- respect anchor flag
        if anchor:
            pattern_re = rf'{start}\A{pattern_re[len(start) :]}'

    return re.compile(pattern_re)
<|endoftext|>"""distutils.file_util

Utility functions for operating on single files.
"""

import os

from ._log import log
from .errors import DistutilsFileError

# for generating verbose output in 'copy_file()'
_copy_action = {None: 'copying', 'hard': 'hard linking', 'sym': 'symbolically linking'}


def _copy_file_contents(src, dst, buffer_size=16 * 1024):  # noqa: C901
    """Copy the file 'src' to 'dst'; both must be filenames.  Any error
    opening either file, reading from 'src', or writing to 'dst', raises
    DistutilsFileError.  Data is read/written in chunks of 'buffer_size'
    bytes (default 16k).  No attempt is made to handle anything apart from
    regular files.
    """
    # Stolen from shutil module in the standard library, but with
    # custom error-handling added.
    fsrc = None
    fdst = None
    try:
        try:
            fsrc = open(src, 'rb')  # noqa: SIM115 # closed in the finally below
        except OSError as e:
            raise DistutilsFileError(f"could not open '{src}': {e.strerror}")

        if os.path.exists(dst):
            try:
                os.unlink(dst)
            except OSError as e:
                raise DistutilsFileError(f"could not delete '{dst}': {e.strerror}")

        try:
            fdst = open(dst, 'wb')  # noqa: SIM115 # closed in the finally below
        except OSError as e:
            raise DistutilsFileError(f"could not create '{dst}': {e.strerror}")

        while True:
            try:
                buf = fsrc.read(buffer_size)
            except OSError as e:
                raise DistutilsFileError(f"could not read from '{src}': {e.strerror}")

            if not buf:
                break

            try:
                fdst.write(buf)
            except OSError as e:
                raise DistutilsFileError(f"could not write to '{dst}': {e.strerror}")
    finally:
        if fdst:
            fdst.close()
        if fsrc:
            fsrc.close()


def copy_file(  # noqa: C901
    src,
    dst,
    preserve_mode=True,
    preserve_times=True,
    update=False,
    link=None,
    verbose=True,
):
    """Copy a file 'src' to 'dst'.  If 'dst' is a directory, then 'src' is
    copied there with the same name; otherwise, it must be a filename.  (If
    the file exists, it will be ruthlessly clobbered.)  If 'preserve_mode'
    is true (the default), the file's mode (type and permission bits, or
    whatever is analogous on the current platform) is copied.  If
    'preserve_times' is true (the default), the last-modified and
    last-access times are copied as well.  If 'update' is true, 'src' will
    only be copied if 'dst' does not exist, or if 'dst' does exist but is
    older than 'src'.

    'link' allows you to make hard links (os.link) or symbolic links
    (os.symlink) instead of copying: set it to "hard" or "sym"; if it is
    None (the default), files are copied.  Don't set 'link' on systems that
    don't support it: 'copy_file()' doesn't check if hard or symbolic
    linking is available. If hardlink fails, falls back to
    _copy_file_contents().

    Under Mac OS, uses the native file copy function in macostools; on
    other systems, uses '_copy_file_contents()' to copy file contents.

    Return a tuple (dest_name, copied): 'dest_name' is the actual name of
    the output file, and 'copied' is true if the file was copied.
    """
    # XXX if the destination file already exists, we clobber it if
    # copying, but blow up if linking.  Hmmm.  And I don't know what
    # macostools.copyfile() does.  Should definitely be consistent, and
    # should probably blow up if destination exists and we would be
    # changing it (ie. it's not already a hard/soft link to src OR
    # (not update) and (src newer than dst).

    from distutils._modified import newer
    from stat import S_IMODE

    if not os.path.isfile(src):
        raise DistutilsFileError(
            f"can't copy '{src}': doesn't exist or not a regular file"
        )

    if os.path.isdir(dst):
        dir = dst
        dst = os.path.join(dst, os.path.basename(src))
    else:
        dir = os.path.dirname(dst)

    if update and not newer(src, dst):
        if verbose >= 1:
            log.debug("not copying %s (output up-to-date)", src)
        return (dst, False)

    try:
        action = _copy_action[link]
    except KeyError:
        raise ValueError(f"invalid value '{link}' for 'link' argument")

    if verbose >= 1:
        if os.path.basename(dst) == os.path.basename(src):
            log.info("%s %s -> %s", action, src, dir)
        else:
            log.info("%s %s -> %s", action, src, dst)

    # If linking (hard or symbolic), use the appropriate system call
    # (Unix only, of course, but that's the caller's responsibility)
    if link == 'hard':
        if not (os.path.exists(dst) and os.path.samefile(src, dst)):
            try:
                os.link(src, dst)
            except OSError:
                # If hard linking fails, fall back on copying file
                # (some special filesystems don't support hard linking
                #  even under Unix, see issue #8876).
                pass
            else:
                return (dst, True)
    elif link == 'sym' and not (os.path.exists(dst) and os.path.samefile(src, dst)):
        os.symlink(src, dst)
        return (dst, True)

    # Otherwise (non-Mac, not linking), copy the file contents and
    # (optionally) copy the times and mode.
    _copy_file_contents(src, dst)
    if preserve_mode or preserve_times:
        st = os.stat(src)

        # According to David Ascher <da@ski.org>, utime() should be done
        # before chmod() (at least under NT).
        if preserve_times:
            os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        if preserve_mode:
            os.chmod(dst, S_IMODE(st.st_mode))

    return (dst, True)


# XXX I suspect this is Unix-specific -- need porting help!
def move_file(src, dst, verbose=True):  # noqa: C901
    """Move a file 'src' to 'dst'.  If 'dst' is a directory, the file will
    be moved into it with the same name; otherwise, 'src' is just renamed
    to 'dst'.  Return the new full name of the file.

    Handles cross-device moves on Unix using 'copy_file()'.  What about
    other systems???
    """
    import errno
    from os.path import basename, dirname, exists, isdir, isfile

    if verbose >= 1:
        log.info("moving %s -> %s", src, dst)

    if not isfile(src):
        raise DistutilsFileError(f"can't move '{src}': not a regular file")

    if isdir(dst):
        dst = os.path.join(dst, basename(src))
    elif exists(dst):
        raise DistutilsFileError(
            f"can't move '{src}': destination '{dst}' already exists"
        )

    if not isdir(dirname(dst)):
        raise DistutilsFileError(
            f"can't move '{src}': destination '{dst}' not a valid path"
        )

    copy_it = False
    try:
        os.rename(src, dst)
    except OSError as e:
        (num, msg) = e.args
        if num == errno.EXDEV:
            copy_it = True
        else:
            raise DistutilsFileError(f"couldn't move '{src}' to '{dst}': {msg}")

    if copy_it:
        copy_file(src, dst, verbose=verbose)
        try:
            os.unlink(src)
        except OSError as e:
            (num, msg) = e.args
            try:
                os.unlink(dst)
            except OSError:
                pass
            raise DistutilsFileError(
                f"couldn't move '{src}' to '{dst}' by copy/delete: "
                f"delete '{src}' failed: {msg}"
            )
    return dst


def write_file(filename, contents):
    """Create a file with the specified name and write 'contents' (a
    sequence of strings without line terminators) to it.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in contents)
<|endoftext|>"""distutils.dir_util

Utility functions for manipulating directories and directory trees."""

import functools
import itertools
import os
import pathlib

from . import file_util
from ._log import log
from .errors import DistutilsFileError, DistutilsInternalError


class SkipRepeatAbsolutePaths(set):
    """
    Cache for mkpath.

    In addition to cheapening redundant calls, eliminates redundant
    "creating /foo/bar/baz" messages in dry-run mode.
    """

    def __init__(self):
        SkipRepeatAbsolutePaths.instance = self

    @classmethod
    def clear(cls):
        super(cls, cls.instance).clear()

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(path, *args, **kwargs):
            if path.absolute() in self:
                return
            result = func(path, *args, **kwargs)
            self.add(path.absolute())
            return result

        return wrapper


# Python 3.8 compatibility
wrapper = SkipRepeatAbsolutePaths().wrap


@functools.singledispatch
@wrapper
def mkpath(name: pathlib.Path, mode=0o777, verbose=True) -> None:
    """Create a directory and any missing ancestor directories.

    If the directory already exists (or if 'name' is the empty string, which
    means the current directory, which of course exists), then do nothing.
    Raise DistutilsFileError if unable to create some directory along the way
    (eg. some sub-path exists, but is a file rather than a directory).
    If 'verbose' is true, log the directory created.
    """
    if verbose and not name.is_dir():
        log.info("creating %s", name)

    try:
        name.mkdir(mode=mode, parents=True, exist_ok=True)
    except OSError as exc:
        raise DistutilsFileError(f"could not create '{name}': {exc.args[-1]}")


@mkpath.register
def _(name: str, *args, **kwargs):
    return mkpath(pathlib.Path(name), *args, **kwargs)


@mkpath.register
def _(name: None, *args, **kwargs):
    """
    Detect a common bug -- name is None.
    """
    raise DistutilsInternalError(f"mkpath: 'name' must be a string (got {name!r})")


def create_tree(base_dir, files, mode=0o777, verbose=True):
    """Create all the empty directories under 'base_dir' needed to put 'files'
    there.

    'base_dir' is just the name of a directory which doesn't necessarily
    exist yet; 'files' is a list of filenames to be interpreted relative to
    'base_dir'.  'base_dir' + the directory portion of every file in 'files'
    will be created if it doesn't already exist.  'mode' and 'verbose'
    flags are as for 'mkpath()'.
    """
    # First get the list of directories to create
    need_dir = {os.path.join(base_dir, os.path.dirname(file)) for file in files}

    # Now create them
    for dir in sorted(need_dir):
        mkpath(dir, mode, verbose=verbose)


def copy_tree(
    src,
    dst,
    preserve_mode=True,
    preserve_times=True,
    preserve_symlinks=False,
    update=False,
    verbose=True,
):
    """Copy an entire directory tree 'src' to a new location 'dst'.

    Both 'src' and 'dst' must be directory names.  If 'src' is not a
    directory, raise DistutilsFileError.  If 'dst' does not exist, it is
    created with 'mkpath()'.  The end result of the copy is that every
    file in 'src' is copied to 'dst', and directories under 'src' are
    recursively copied to 'dst'.  Return the list of files that were
    copied or might have been copied, using their output name.  The
    return value is unaffected by 'update': it is simply
    the list of all files under 'src', with the names changed to be
    under 'dst'.

    'preserve_mode' and 'preserve_times' are the same as for
    'copy_file'; note that they only apply to regular files, not to
    directories.  If 'preserve_symlinks' is true, symlinks will be
    copied as symlinks (on platforms that support them!); otherwise
    (the default), the destination of the symlink will be copied.
    'update' and 'verbose' are the same as for 'copy_file'.
    """
    if not os.path.isdir(src):
        raise DistutilsFileError(f"cannot copy tree '{src}': not a directory")
    try:
        names = os.listdir(src)
    except OSError as e:
        raise DistutilsFileError(f"error listing files in '{src}': {e.strerror}")

    mkpath(dst, verbose=verbose)

    copy_one = functools.partial(
        _copy_one,
        src=src,
        dst=dst,
        preserve_symlinks=preserve_symlinks,
        verbose=verbose,
        preserve_mode=preserve_mode,
        preserve_times=preserve_times,
        update=update,
    )
    return list(itertools.chain.from_iterable(map(copy_one, names)))


def _copy_one(
    name,
    *,
    src,
    dst,
    preserve_symlinks,
    verbose,
    preserve_mode,
    preserve_times,
    update,
):
    src_name = os.path.join(src, name)
    dst_name = os.path.join(dst, name)

    if name.startswith('.nfs'):
        # skip NFS rename files
        return

    if preserve_symlinks and os.path.islink(src_name):
        link_dest = os.readlink(src_name)
        if verbose >= 1:
            log.info("linking %s -> %s", dst_name, link_dest)
        os.symlink(link_dest, dst_name)
        yield dst_name

    elif os.path.isdir(src_name):
        yield from copy_tree(
            src_name,
            dst_name,
            preserve_mode,
            preserve_times,
            preserve_symlinks,
            update,
            verbose=verbose,
        )
    else:
        file_util.copy_file(
            src_name,
            dst_name,
            preserve_mode,
            preserve_times,
            update,
            verbose=verbose,
        )
        yield dst_name


def _build_cmdtuple(path, cmdtuples):
    """Helper for remove_tree()."""
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        if os.path.isdir(real_f) and not os.path.islink(real_f):
            _build_cmdtuple(real_f, cmdtuples)
        else:
            cmdtuples.append((os.remove, real_f))
    cmdtuples.append((os.rmdir, path))


def remove_tree(directory, verbose=True):
    """Recursively remove an entire directory tree.

    Any errors are ignored (apart from being reported to stdout if 'verbose'
    is true).
    """
    if verbose >= 1:
        log.info("removing '%s' (and everything under it)", directory)
    cmdtuples = []
    _build_cmdtuple(directory, cmdtuples)
    for cmd in cmdtuples:
        try:
            cmd[0](cmd[1])
            # Clear the cache
            SkipRepeatAbsolutePaths.clear()
        except OSError as exc:
            log.warning("error removing %s: %s", directory, exc)


def ensure_relative(path):
    """Take the full path 'path', and make it a relative path.

    This is useful to make 'path' the second argument to os.path.join().
    """
    drive, path = os.path.splitdrive(path)
    if path[0:1] == os.sep:
        path = drive + path[1:]
    return path
<|endoftext|>"""Language-agnostic compiler exceptions."""


class Error(Exception):
    """A compiler operation failed."""


class UnknownFileType(Error):
    """Attempt to process an unknown file type."""


class PlatformError(Error):
    """A compiler operation isn't supported on the current platform."""
<|endoftext|>import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger whose name is stable across this package's import
    location, which migrates over time (``distutils.compilers.C`` now,
    ``setuptools._distutils.compilers.C`` when vendored, and eventually a
    standalone ``compilers.C``).

    Pass ``__name__``; everything before ``compilers.`` is stripped.

    >>> get_logger('distutils.compilers.C.base').name
    'compilers.C.base'
    >>> get_logger('setuptools._distutils.compilers.C.msvc').name
    'compilers.C.msvc'
    >>> get_logger('compilers.C.unix').name
    'compilers.C.unix'
    """
    stable_name = 'compilers.' + name.split('compilers.', 1)[1]
    return logging.getLogger(stable_name)
<|endoftext|>"""Timestamp comparison of files and groups of files."""

from __future__ import annotations

import os.path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import TypeAlias

_Path: TypeAlias = "str | bytes | os.PathLike[str] | os.PathLike[bytes]"


def _newer(source, target) -> bool:
    return not os.path.exists(target) or (
        os.path.getmtime(source) > os.path.getmtime(target)
    )


def newer(source: _Path, target: _Path) -> bool:
    """
    Is source modified more recently than target.

    Returns True if 'source' is modified more recently than 'target' or if
    'target' does not exist. Raises FileNotFoundError if 'source' does not
    exist.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"file {os.path.abspath(source)!r} does not exist")

    return _newer(source, target)


def newer_group(
    sources: Iterable[_Path],
    target: _Path,
    missing: Literal["error", "ignore", "newer"] = "error",
) -> bool:
    """
    Is target out-of-date with respect to any file in sources.

    Return True if 'target' is out-of-date with respect to any file listed in
    'sources'. In other words, if 'target' exists and is newer than every file
    in 'sources', return False; otherwise return True. ``missing`` controls how
    to handle a missing source file:

    - error (default): allow the ``stat()`` call to fail.
    - ignore: silently disregard any missing source files.
    - newer: treat missing source files as "target out of date".
    """

    def missing_as_newer(source):
        return missing == 'newer' and not os.path.exists(source)

    ignored = os.path.exists if missing == 'ignore' else None
    return not os.path.exists(target) or any(
        missing_as_newer(source) or _newer(source, target)
        for source in filter(ignored, sources)
    )
<|endoftext|>"""Small stdlib-only utilities used across the compilers package."""

from __future__ import annotations

import re
import string

# Needed by 'split_quoted()'
_wordchars_re = re.compile(rf'[^\\\'\"{string.whitespace} ]*')
_squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
_dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')


def split_quoted(s: str) -> list[str]:
    """Split a string up according to Unix shell-like rules for quotes and
    backslashes.  In short: words are delimited by spaces, as long as those
    spaces are not escaped by a backslash, or inside a quoted string.
    Single and double quotes are equivalent, and the quote characters can
    be backslash-escaped.  The backslash is stripped from any two-character
    escape sequence, leaving only the escaped character.  The quote
    characters are stripped from any quoted string.  Returns a list of
    words.
    """

    # This is a nice algorithm for splitting up a single string, since it
    # doesn't require character-by-character examination.  It was a little
    # bit of a brain-bender to get it working right, though...

    s = s.strip()
    words = []
    pos = 0

    while s:
        m = _wordchars_re.match(s, pos)
        assert m is not None
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break

        if s[end] in string.whitespace:
            # unescaped, unquoted whitespace: now
            # we definitely have a word delimiter
            words.append(s[:end])
            s = s[end:].lstrip()
            pos = 0

        elif s[end] == '\\':
            # preserve whatever is being escaped;
            # will become part of the current word
            s = s[:end] + s[end + 1 :]
            pos = end + 1

        else:
            if s[end] == "'":  # slurp singly-quoted string
                m = _squote_re.match(s, end)
            elif s[end] == '"':  # slurp doubly-quoted string
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError(f"this can't happen (bad char '{s[end]}')")

            if m is None:
                raise ValueError(f"bad string (mismatched {s[end]} quotes?)")

            (beg, end) = m.span()
            s = s[:beg] + s[beg + 1 : end - 1] + s[end:]
            pos = m.end() - 2

        if pos >= len(s):
            words.append(s)
            break

    return words
<|endoftext|>"""macOS-specific compiler support."""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
from collections.abc import Mapping
from typing import TypeVar

from ..errors import PlatformError

_MappingT = TypeVar("_MappingT", bound=Mapping)

VERSION_VAR = 'MACOSX_DEPLOYMENT_TARGET'

if sys.platform == 'darwin':
    from _osx_support import compiler_fixup
else:

    def compiler_fixup(compiler_so, cc_args):
        """Off macOS there's nothing to fix up."""
        return compiler_so


def customize_compiler(config_vars) -> None:
    """Apply macOS SDK/architecture fixups to build config vars, in place.

    A no-op off macOS. Mirrors distutils' historical ``_customize_macos`` to
    support interpreters from binary installers, where the user's build tools
    and OS version may differ from the system Python itself was built on.
    """
    if sys.platform == 'darwin':
        import _osx_support

        _osx_support.customize_compiler(config_vars)


_syscfg_target_ver = None


def _clear_cached_target_ver():
    """For testing only. Do not call."""
    global _syscfg_target_ver
    _syscfg_target_ver = None


def _target_ver_from_syscfg():
    """The deployment target latched into the interpreter's configuration."""
    global _syscfg_target_ver
    if _syscfg_target_ver is None:
        _syscfg_target_ver = sysconfig.get_config_var(VERSION_VAR) or ''
    return _syscfg_target_ver


def _split_version(s: str) -> list[int]:
    return [int(n) for n in s.split('.')]


def target_ver() -> str | None:
    """Return the version of macOS for which we are building.

    Defaults to the version latched in sysconfig when the interpreter was
    built, unless overridden by ``MACOSX_DEPLOYMENT_TARGET``. Returns None if
    neither source has a value.
    """
    syscfg_ver = _target_ver_from_syscfg()
    env_ver = os.environ.get(VERSION_VAR)

    if env_ver:
        # An override below 10.3 while the interpreter was configured for 10.3+
        # would build extensions with an LDSHARED that uses
        # '-undefined dynamic_lookup', which only works on >= 10.3.
        if (
            syscfg_ver
            and _split_version(syscfg_ver) >= [10, 3]
            and _split_version(env_ver) < [10, 3]
        ):
            raise PlatformError(
                f'${VERSION_VAR} mismatch: now "{env_ver}" but "{syscfg_ver}" '
                'during configure; must use 10.3 or later'
            )
        return env_ver
    return syscfg_ver or None


def inject_ver(env: _MappingT | None) -> _MappingT | dict[str, str | int] | None:
    """
    Ensure a subprocess inherits the deployment target the build was
    configured with, so extensions link against a consistent macOS version.
    """
    if platform.system() != 'Darwin':
        return env

    ver = target_ver()
    update = {VERSION_VAR: ver} if ver else {}
    resolved = os.environ if env is None else env
    return {**resolved, **update}
<|endoftext|>"""Platform identification used to select and configure compilers."""

from __future__ import annotations

import os
import sys
import sysconfig


def get_host_platform() -> str:
    """A string identifying the platform the build is running on."""
    return sysconfig.get_platform()


def get_platform() -> str:
    """The platform being built for.

    Matches :func:`get_host_platform` except on Windows, where the MSVC
    target architecture (``VSCMD_ARG_TGT_ARCH``) takes precedence to support
    cross-compilation.
    """
    if os.name == 'nt':
        TARGET_TO_PLAT = {
            'x86': 'win32',
            'x64': 'win-amd64',
            'arm': 'win-arm32',
            'arm64': 'win-arm64',
        }
        target = os.environ.get('VSCMD_ARG_TGT_ARCH')
        return TARGET_TO_PLAT.get(target) or get_host_platform()
    return get_host_platform()


def is_mingw() -> bool:
    """Whether the current platform is mingw.

    Python compiled with Mingw-w64 has ``sys.platform == 'win32'`` and
    ``get_platform()`` starts with ``'mingw'``.
    """
    return sys.platform == 'win32' and get_platform().startswith('mingw')
<|endoftext|>"""Exceptions raised by the C/C++ compilers."""

from ..errors import Error


class PreprocessError(Error):
    """Failure to preprocess one or more C/C++ files."""


class CompileError(Error):
    """Failure to compile one or more C/C++ source files."""


class LibError(Error):
    """Failure to create a static library from one or more C/C++ object
    files."""


class LinkError(Error):
    """Failure to link one or more C/C++ object files into an executable
    or shared library file."""
<|endoftext|>"""distutils.zosccompiler

Contains the selection of the c & c++ compilers on z/OS. There are several
different c compilers on z/OS, all of them are optional, so the correct
one needs to be chosen based on the users input. This is compatible with
the following compilers:

IBM C/C++ For Open Enterprise Languages on z/OS 2.0
IBM Open XL C/C++ 1.1 for z/OS
IBM XL C/C++ V2.4.1 for z/OS 2.4 and 2.5
IBM z/OS XL C/C++
"""

import os
import subprocess
import sysconfig
from typing import ClassVar

from . import unix
from .errors import CompileError

_cc_args = {
    'ibm-openxl': [
        '-m64',
        '-fvisibility=default',
        '-fzos-le-char-mode=ascii',
        '-fno-short-enums',
    ],
    'ibm-xlclang': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
    ],
    'ibm-xlc': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
        '-qlanglvl=extc99',
    ],
}

_cxx_args = {
    'ibm-openxl': [
        '-m64',
        '-fvisibility=default',
        '-fzos-le-char-mode=ascii',
        '-fno-short-enums',
    ],
    'ibm-xlclang': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
    ],
    'ibm-xlc': [
        '-q64',
        '-qexportall',
        '-qascii',
        '-qstrict',
        '-qnocsect',
        '-Wa,asa,goff',
        '-Wa,xplink',
        '-qgonumber',
        '-qenum=int',
        '-Wc,DLL',
        '-qlanglvl=extended0x',
    ],
}

_asm_args = {
    'ibm-openxl': ['-fasm', '-fno-integrated-as', '-Wa,--ASA', '-Wa,--GOFF'],
    'ibm-xlclang': [],
    'ibm-xlc': [],
}

_ld_args = {
    'ibm-openxl': [],
    'ibm-xlclang': ['-Wl,dll', '-q64'],
    'ibm-xlc': ['-Wl,dll', '-q64'],
}


# Python on z/OS is built with no compiler specific options in it's CFLAGS.
# But each compiler requires it's own specific options to build successfully,
# though some of the options are common between them
class Compiler(unix.Compiler):
    compiler_type = 'zos'
    description = "IBM XL C/C++ Compilers"
    src_extensions: ClassVar[list[str] | None] = [
        '.c',
        '.C',
        '.cc',
        '.cxx',
        '.cpp',
        '.m',
        '.s',
    ]
    _cpp_extensions: ClassVar[list[str]] = ['.cc', '.cpp', '.cxx', '.C']
    _asm_extensions: ClassVar[list[str]] = ['.s']

    def _get_zos_compiler_name(self):
        zos_compiler_names = [
            os.path.basename(binary)
            for envvar in ('CC', 'CXX', 'LDSHARED')
            if (binary := os.environ.get(envvar, None))
        ]
        if len(zos_compiler_names) == 0:
            return 'ibm-openxl'

        zos_compilers = {}
        for compiler in (
            'ibm-clang',
            'ibm-clang64',
            'ibm-clang++',
            'ibm-clang++64',
            'clang',
            'clang++',
            'clang-14',
        ):
            zos_compilers[compiler] = 'ibm-openxl'

        for compiler in ('xlclang', 'xlclang++', 'njsc', 'njsc++'):
            zos_compilers[compiler] = 'ibm-xlclang'

        for compiler in ('xlc', 'xlC', 'xlc++'):
            zos_compilers[compiler] = 'ibm-xlc'

        return zos_compilers.get(zos_compiler_names[0], 'ibm-openxl')

    def __init__(self, verbose=False, force=False):
        super().__init__(verbose, force=force)
        self.zos_compiler = self._get_zos_compiler_name()
        self.configure_system()

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        local_args = []
        if ext in self._cpp_extensions:
            compiler = self.compiler_cxx
            local_args.extend(_cxx_args[self.zos_compiler])
        elif ext in self._asm_extensions:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
            local_args.extend(_asm_args[self.zos_compiler])
        else:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
        local_args.extend(cc_args)

        try:
            self.call(compiler + local_args + [src, '-o', obj] + extra_postargs)
        except (subprocess.CalledProcessError, OSError) as msg:
            raise CompileError(msg)

    def runtime_library_dir_option(self, dir):
        return '-L' + dir

    def link(
        self,
        target_desc,
        objects,
        output_filename,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug=False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ):
        # For a built module to use functions from cpython, it needs to use Pythons
        # side deck file. The side deck is located beside the libpython3.xx.so
        ldversion = sysconfig.get_config_var('LDVERSION')
        if sysconfig.is_python_build():
            side_deck_path = os.path.join(
                sysconfig.get_config_var('abs_builddir'),
                f'libpython{ldversion}.x',
            )
        else:
            side_deck_path = os.path.join(
                sysconfig.get_config_var('installed_base'),
                sysconfig.get_config_var('platlibdir'),
                f'libpython{ldversion}.x',
            )

        if os.path.exists(side_deck_path):
            if extra_postargs:
                extra_postargs.append(side_deck_path)
            else:
                extra_postargs = [side_deck_path]

        # Check and replace libraries included side deck files
        if runtime_library_dirs:
            for dir in runtime_library_dirs:
                for library in libraries[:]:
                    library_side_deck = os.path.join(dir, f'{library}.x')
                    if os.path.exists(library_side_deck):
                        libraries.remove(library)
                        extra_postargs.append(library_side_deck)
                        break

        # Any required ld args for the given compiler
        extra_postargs.extend(_ld_args[self.zos_compiler])

        super().link(
            target_desc,
            objects,
            output_filename,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            export_symbols,
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )
<|endoftext|>"""distutils.cygwinccompiler

Provides the CygwinCCompiler class, a subclass of UnixCCompiler that
handles the Cygwin port of the GNU C compiler to Windows.  It also contains
the Mingw32CCompiler class which handles the mingw32 port of GCC (same as
cygwin in no-cygwin mode).
"""

from __future__ import annotations

import copy
import os
import pathlib
import shlex
import subprocess
import sys
import warnings
from sysconfig import get_config_vars

import packaging.version

from ..errors import Error, PlatformError
from ..logging import get_logger
from ..platform.detect import is_mingw
from . import unix
from .errors import CompileError

log = get_logger(__name__)


def get_msvcr():
    """No longer needed, but kept for backward compatibility."""
    return []


_runtime_library_dirs_msg = (
    "Unable to set runtime library search path on Windows, "
    "usually indicated by `runtime_library_dirs` parameter to Extension"
)


class Compiler(unix.Compiler):
    """Handles the Cygwin port of the GNU C compiler to Windows."""

    compiler_type = 'cygwin'
    description = "Cygwin port of GNU C Compiler for Win32"
    obj_extension = ".o"
    static_lib_extension = ".a"
    shared_lib_extension = ".dll.a"
    dylib_lib_extension = ".dll"
    static_lib_format = "lib%s%s"
    shared_lib_format = "lib%s%s"
    dylib_lib_format = "cyg%s%s"
    exe_extension = ".exe"

    def __init__(self, verbose=False, force=False):
        super().__init__(verbose, force=force)

        status, details = check_config_h()
        self.debug_print(f"Python's GCC status: {status} (details: {details})")
        if status is not CONFIG_H_OK:
            self.warn(
                "Python's pyconfig.h doesn't seem to support your compiler. "
                f"Reason: {details}. "
                "Compiling may fail because of undefined preprocessor macros."
            )

        self.cc, self.cxx = get_config_vars('CC', 'CXX')

        # Override 'CC' and 'CXX' environment variables for
        # building using MINGW compiler for MSVC python.
        self.cc = os.environ.get('CC', self.cc or 'gcc')
        self.cxx = os.environ.get('CXX', self.cxx or 'g++')

        self.linker_dll = self.cc
        self.linker_dll_cxx = self.cxx
        shared_option = "-shared"

        self.set_executables(
            compiler=f'{self.cc} -mcygwin -O1 -Wall',
            compiler_so=f'{self.cc} -mcygwin -mdll -O1 -Wall',
            compiler_cxx=f'{self.cxx} -mcygwin -O1 -Wall',
            compiler_so_cxx=f'{self.cxx} -mcygwin -mdll -O1 -Wall',
            linker_exe=f'{self.cc} -mcygwin',
            linker_so=f'{self.linker_dll} -mcygwin {shared_option}',
            linker_exe_cxx=f'{self.cxx} -mcygwin',
            linker_so_cxx=f'{self.linker_dll_cxx} -mcygwin {shared_option}',
        )

        self.dll_libraries = get_msvcr()

    @property
    def gcc_version(self):
        # Older numpy depended on this existing to check for ancient
        # gcc versions. This doesn't make much sense with clang etc so
        # just hardcode to something recent.
        # https://github.com/numpy/numpy/pull/20333
        warnings.warn(
            "gcc_version attribute of CygwinCCompiler is deprecated. "
            "Instead of returning actual gcc version a fixed value 11.2.0 is returned.",
            DeprecationWarning,
            stacklevel=2,
        )
        return packaging.version.Version("11.2.0")

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compiles the source by spawning GCC and windres if needed."""
        if ext in ('.rc', '.res'):
            # gcc needs '.res' and '.rc' compiled to object files !!!
            try:
                self.call(["windres", "-i", src, "-o", obj])
            except (subprocess.CalledProcessError, OSError) as msg:
                raise CompileError(msg)
        else:  # for other files use the C-compiler
            try:
                if self.detect_language(src) == 'c++':
                    self.call(
                        self.compiler_so_cxx
                        + cc_args
                        + [src, '-o', obj]
                        + extra_postargs
                    )
                else:
