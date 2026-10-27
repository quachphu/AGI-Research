    """

    def __init__(
        self,
        template: Template,
        context: Context,
        body_stream: t.Optional[t.Iterable[str]] = None,
    ) -> None:
        if body_stream is None:
            if context.environment.is_async:
                raise RuntimeError(
                    "Async mode requires a body stream to be passed to"
                    " a template module. Use the async methods of the"
                    " API you are using."
                )

            body_stream = list(template.root_render_func(context))

        self._body_stream = body_stream
        self.__dict__.update(context.get_exported())
        self.__name__ = template.name

    def __html__(self) -> Markup:
        return Markup(concat(self._body_stream))

    def __str__(self) -> str:
        return concat(self._body_stream)

    def __repr__(self) -> str:
        if self.__name__ is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.__name__)
        return f"<{type(self).__name__} {name}>"


class TemplateExpression:
    """The :meth:`jinja2.Environment.compile_expression` method returns an
    instance of this object.  It encapsulates the expression-like access
    to the template with an expression it wraps.
    """

    def __init__(self, template: Template, undefined_to_none: bool) -> None:
        self._template = template
        self._undefined_to_none = undefined_to_none

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Optional[t.Any]:
        context = self._template.new_context(dict(*args, **kwargs))
        consume(self._template.root_render_func(context))
        rv = context.vars["result"]
        if self._undefined_to_none and isinstance(rv, Undefined):
            rv = None
        return rv


class TemplateStream:
    """A template stream works pretty much like an ordinary python generator
    but it can buffer multiple items to reduce the number of total iterations.
    Per default the output is unbuffered which means that for every unbuffered
    instruction in the template one string is yielded.

    If buffering is enabled with a buffer size of 5, five items are combined
    into a new string.  This is mainly useful if you are streaming
    big templates to a client via WSGI which flushes after each iteration.
    """

    def __init__(self, gen: t.Iterator[str]) -> None:
        self._gen = gen
        self.disable_buffering()

    def dump(
        self,
        fp: t.Union[str, t.IO[bytes]],
        encoding: t.Optional[str] = None,
        errors: t.Optional[str] = "strict",
    ) -> None:
        """Dump the complete stream into a file or file-like object.
        Per default strings are written, if you want to encode
        before writing specify an `encoding`.

        Example usage::

            Template('Hello {{ name }}!').stream(name='foo').dump('hello.html')
        """
        close = False

        if isinstance(fp, str):
            if encoding is None:
                encoding = "utf-8"

            real_fp: t.IO[bytes] = open(fp, "wb")
            close = True
        else:
            real_fp = fp

        try:
            if encoding is not None:
                iterable = (x.encode(encoding, errors) for x in self)  # type: ignore
            else:
                iterable = self  # type: ignore

            if hasattr(real_fp, "writelines"):
                real_fp.writelines(iterable)
            else:
                for item in iterable:
                    real_fp.write(item)
        finally:
            if close:
                real_fp.close()

    def disable_buffering(self) -> None:
        """Disable the output buffering."""
        self._next = partial(next, self._gen)
        self.buffered = False

    def _buffered_generator(self, size: int) -> t.Iterator[str]:
        buf: t.List[str] = []
        c_size = 0
        push = buf.append

        while True:
            try:
                while c_size < size:
                    c = next(self._gen)
                    push(c)
                    if c:
                        c_size += 1
            except StopIteration:
                if not c_size:
                    return
            yield concat(buf)
            del buf[:]
            c_size = 0

    def enable_buffering(self, size: int = 5) -> None:
        """Enable buffering.  Buffer `size` items before yielding them."""
        if size <= 1:
            raise ValueError("buffer size too small")

        self.buffered = True
        self._next = partial(next, self._buffered_generator(size))

    def __iter__(self) -> "TemplateStream":
        return self

    def __next__(self) -> str:
        return self._next()  # type: ignore


# hook in default template class.  if anyone reads this comment: ignore that
# it's possible to use custom templates ;-)
Environment.template_class = Template
<|endoftext|>"""A sandbox layer that ensures unsafe operations cannot be performed.
Useful when the template itself comes from an untrusted source.
"""

import operator
import types
import typing as t
from _string import formatter_field_name_split  # type: ignore
from collections import abc
from collections import deque
from functools import update_wrapper
from string import Formatter

from markupsafe import EscapeFormatter
from markupsafe import Markup

from .environment import Environment
from .exceptions import SecurityError
from .runtime import Context
from .runtime import Undefined

F = t.TypeVar("F", bound=t.Callable[..., t.Any])

#: maximum number of items a range may produce
MAX_RANGE = 100000

#: Unsafe function attributes.
UNSAFE_FUNCTION_ATTRIBUTES: t.Set[str] = set()

#: Unsafe method attributes. Function attributes are unsafe for methods too.
UNSAFE_METHOD_ATTRIBUTES: t.Set[str] = set()

#: unsafe generator attributes.
UNSAFE_GENERATOR_ATTRIBUTES = {"gi_frame", "gi_code"}

#: unsafe attributes on coroutines
UNSAFE_COROUTINE_ATTRIBUTES = {"cr_frame", "cr_code"}

#: unsafe attributes on async generators
UNSAFE_ASYNC_GENERATOR_ATTRIBUTES = {"ag_code", "ag_frame"}

_mutable_spec: t.Tuple[t.Tuple[t.Type[t.Any], t.FrozenSet[str]], ...] = (
    (
        abc.MutableSet,
        frozenset(
            [
                "add",
                "clear",
                "difference_update",
                "discard",
                "pop",
                "remove",
                "symmetric_difference_update",
                "update",
            ]
        ),
    ),
    (
        abc.MutableMapping,
        frozenset(["clear", "pop", "popitem", "setdefault", "update"]),
    ),
    (
        abc.MutableSequence,
        frozenset(
            ["append", "clear", "pop", "reverse", "insert", "sort", "extend", "remove"]
        ),
    ),
    (
        deque,
        frozenset(
            [
                "append",
                "appendleft",
                "clear",
                "extend",
                "extendleft",
                "pop",
                "popleft",
                "remove",
                "rotate",
            ]
        ),
    ),
)


def safe_range(*args: int) -> range:
    """A range that can't generate ranges with a length of more than
    MAX_RANGE items.
    """
    rng = range(*args)

    if len(rng) > MAX_RANGE:
        raise OverflowError(
            "Range too big. The sandbox blocks ranges larger than"
            f" MAX_RANGE ({MAX_RANGE})."
        )

    return rng


def unsafe(f: F) -> F:
    """Marks a function or method as unsafe.

    .. code-block: python

        @unsafe
        def delete(self):
            pass
    """
    f.unsafe_callable = True  # type: ignore
    return f


def is_internal_attribute(obj: t.Any, attr: str) -> bool:
    """Test if the attribute given is an internal python attribute.  For
    example this function returns `True` for the `func_code` attribute of
    python objects.  This is useful if the environment method
    :meth:`~SandboxedEnvironment.is_safe_attribute` is overridden.

    >>> from jinja2.sandbox import is_internal_attribute
    >>> is_internal_attribute(str, "mro")
    True
    >>> is_internal_attribute(str, "upper")
    False
    """
    if isinstance(obj, types.FunctionType):
        if attr in UNSAFE_FUNCTION_ATTRIBUTES:
            return True
    elif isinstance(obj, types.MethodType):
        if attr in UNSAFE_FUNCTION_ATTRIBUTES or attr in UNSAFE_METHOD_ATTRIBUTES:
            return True
    elif isinstance(obj, type):
        if attr == "mro":
            return True
    elif isinstance(obj, (types.CodeType, types.TracebackType, types.FrameType)):
        return True
    elif isinstance(obj, types.GeneratorType):
        if attr in UNSAFE_GENERATOR_ATTRIBUTES:
            return True
    elif hasattr(types, "CoroutineType") and isinstance(obj, types.CoroutineType):
        if attr in UNSAFE_COROUTINE_ATTRIBUTES:
            return True
    elif hasattr(types, "AsyncGeneratorType") and isinstance(
        obj, types.AsyncGeneratorType
    ):
        if attr in UNSAFE_ASYNC_GENERATOR_ATTRIBUTES:
            return True
    return attr.startswith("__")


def modifies_known_mutable(obj: t.Any, attr: str) -> bool:
    """This function checks if an attribute on a builtin mutable object
    (list, dict, set or deque) or the corresponding ABCs would modify it
    if called.

    >>> modifies_known_mutable({}, "clear")
    True
    >>> modifies_known_mutable({}, "keys")
    False
    >>> modifies_known_mutable([], "append")
    True
    >>> modifies_known_mutable([], "index")
    False

    If called with an unsupported object, ``False`` is returned.

    >>> modifies_known_mutable("foo", "upper")
    False
    """
    for typespec, unsafe in _mutable_spec:
        if isinstance(obj, typespec):
            return attr in unsafe
    return False


class SandboxedEnvironment(Environment):
    """The sandboxed environment.  It works like the regular environment but
    tells the compiler to generate sandboxed code.  Additionally subclasses of
    this environment may override the methods that tell the runtime what
    attributes or functions are safe to access.

    If the template tries to access insecure code a :exc:`SecurityError` is
    raised.  However also other exceptions may occur during the rendering so
    the caller has to ensure that all exceptions are caught.
    """

    sandboxed = True

    #: default callback table for the binary operators.  A copy of this is
    #: available on each instance of a sandboxed environment as
    #: :attr:`binop_table`
    default_binop_table: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "**": operator.pow,
        "%": operator.mod,
    }

    #: default callback table for the unary operators.  A copy of this is
    #: available on each instance of a sandboxed environment as
    #: :attr:`unop_table`
    default_unop_table: t.Dict[str, t.Callable[[t.Any], t.Any]] = {
        "+": operator.pos,
        "-": operator.neg,
    }

    #: a set of binary operators that should be intercepted.  Each operator
    #: that is added to this set (empty by default) is delegated to the
    #: :meth:`call_binop` method that will perform the operator.  The default
    #: operator callback is specified by :attr:`binop_table`.
    #:
    #: The following binary operators are interceptable:
    #: ``//``, ``%``, ``+``, ``*``, ``-``, ``/``, and ``**``
    #:
    #: The default operation form the operator table corresponds to the
    #: builtin function.  Intercepted calls are always slower than the native
    #: operator call, so make sure only to intercept the ones you are
    #: interested in.
    #:
    #: .. versionadded:: 2.6
    intercepted_binops: t.FrozenSet[str] = frozenset()

    #: a set of unary operators that should be intercepted.  Each operator
    #: that is added to this set (empty by default) is delegated to the
    #: :meth:`call_unop` method that will perform the operator.  The default
    #: operator callback is specified by :attr:`unop_table`.
    #:
    #: The following unary operators are interceptable: ``+``, ``-``
    #:
    #: The default operation form the operator table corresponds to the
    #: builtin function.  Intercepted calls are always slower than the native
    #: operator call, so make sure only to intercept the ones you are
    #: interested in.
    #:
    #: .. versionadded:: 2.6
    intercepted_unops: t.FrozenSet[str] = frozenset()

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.globals["range"] = safe_range
        self.binop_table = self.default_binop_table.copy()
        self.unop_table = self.default_unop_table.copy()

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        """The sandboxed environment will call this method to check if the
        attribute of an object is safe to access.  Per default all attributes
        starting with an underscore are considered private as well as the
        special attributes of internal python objects as returned by the
        :func:`is_internal_attribute` function.
        """
        return not (attr.startswith("_") or is_internal_attribute(obj, attr))

    def is_safe_callable(self, obj: t.Any) -> bool:
        """Check if an object is safely callable. By default callables
        are considered safe unless decorated with :func:`unsafe`.

        This also recognizes the Django convention of setting
        ``func.alters_data = True``.
        """
        return not (
            getattr(obj, "unsafe_callable", False) or getattr(obj, "alters_data", False)
        )

    def call_binop(
        self, context: Context, operator: str, left: t.Any, right: t.Any
    ) -> t.Any:
        """For intercepted binary operator calls (:meth:`intercepted_binops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.binop_table[operator](left, right)

    def call_unop(self, context: Context, operator: str, arg: t.Any) -> t.Any:
        """For intercepted unary operator calls (:meth:`intercepted_unops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.unop_table[operator](arg)

    def getitem(
        self, obj: t.Any, argument: t.Union[str, t.Any]
    ) -> t.Union[t.Any, Undefined]:
        """Subscribe an object from sandboxed code."""
        try:
            return obj[argument]
        except (TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        value = getattr(obj, attr)
                    except AttributeError:
                        pass
                    else:
                        fmt = self.wrap_str_format(value)
                        if fmt is not None:
                            return fmt
                        if self.is_safe_attribute(obj, argument, value):
                            return value
                        return self.unsafe_undefined(obj, argument)
        return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Union[t.Any, Undefined]:
        """Subscribe an object from sandboxed code and prefer the
        attribute.  The attribute passed *must* be a bytestring.
        """
        try:
            value = getattr(obj, attribute)
        except AttributeError:
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                pass
        else:
            fmt = self.wrap_str_format(value)
            if fmt is not None:
                return fmt
            if self.is_safe_attribute(obj, attribute, value):
                return value
            return self.unsafe_undefined(obj, attribute)
        return self.undefined(obj=obj, name=attribute)

    def unsafe_undefined(self, obj: t.Any, attribute: str) -> Undefined:
        """Return an undefined object for unsafe attributes."""
        return self.undefined(
            f"access to attribute {attribute!r} of"
            f" {type(obj).__name__!r} object is unsafe.",
            name=attribute,
            obj=obj,
            exc=SecurityError,
        )

    def wrap_str_format(self, value: t.Any) -> t.Optional[t.Callable[..., str]]:
        """If the given value is a ``str.format`` or ``str.format_map`` method,
        return a new function than handles sandboxing. This is done at access
        rather than in :meth:`call`, so that calls made without ``call`` are
        also sandboxed.
        """
        if not isinstance(
            value, (types.MethodType, types.BuiltinMethodType)
        ) or value.__name__ not in ("format", "format_map"):
            return None

        f_self: t.Any = value.__self__

        if not isinstance(f_self, str):
            return None

        str_type: t.Type[str] = type(f_self)
        is_format_map = value.__name__ == "format_map"
        formatter: SandboxedFormatter

        if isinstance(f_self, Markup):
            formatter = SandboxedEscapeFormatter(self, escape=f_self.escape)
        else:
            formatter = SandboxedFormatter(self)

        vformat = formatter.vformat

        def wrapper(*args: t.Any, **kwargs: t.Any) -> str:
            if is_format_map:
                if kwargs:
                    raise TypeError("format_map() takes no keyword arguments")

                if len(args) != 1:
                    raise TypeError(
                        f"format_map() takes exactly one argument ({len(args)} given)"
                    )

                kwargs = args[0]
                args = ()

            return str_type(vformat(f_self, args, kwargs))

        return update_wrapper(wrapper, value)

    def call(
        __self,  # noqa: B902
        __context: Context,
        __obj: t.Any,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Call an object from sandboxed code."""

        # the double prefixes are to avoid double keyword argument
        # errors when proxying the call.
        if not __self.is_safe_callable(__obj):
            raise SecurityError(f"{__obj!r} is not safely callable")
        return __context.call(__obj, *args, **kwargs)


class ImmutableSandboxedEnvironment(SandboxedEnvironment):
    """Works exactly like the regular `SandboxedEnvironment` but does not
    permit modifications on the builtin mutable objects `list`, `set`, and
    `dict` by using the :func:`modifies_known_mutable` function.
    """

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        if not super().is_safe_attribute(obj, attr, value):
            return False

        return not modifies_known_mutable(obj, attr)


class SandboxedFormatter(Formatter):
    def __init__(self, env: Environment, **kwargs: t.Any) -> None:
        self._env = env
        super().__init__(**kwargs)

    def get_field(
        self, field_name: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]
    ) -> t.Tuple[t.Any, str]:
        first, rest = formatter_field_name_split(field_name)
        obj = self.get_value(first, args, kwargs)
        for is_attr, i in rest:
            if is_attr:
                obj = self._env.getattr(obj, i)
            else:
                obj = self._env.getitem(obj, i)
        return obj, first


class SandboxedEscapeFormatter(SandboxedFormatter, EscapeFormatter):
    pass
<|endoftext|>"""The optional bytecode cache system. This is useful if you have very
complex template situations and the compilation of all those templates
slows down your application too much.

Situations where this is useful are often forking web applications that
are initialized on the first request.
"""

import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

    class _MemcachedClient(te.Protocol):
        def get(self, key: str) -> bytes: ...

        def set(
            self, key: str, value: bytes, timeout: t.Optional[int] = None
        ) -> None: ...


bc_version = 5
# Magic bytes to identify Jinja bytecode cache files. Contains the
# Python major and minor version to avoid loading incompatible bytecode
# if a project upgrades its Python version.
bc_magic = (
    b"j2"
    + pickle.dumps(bc_version, 2)
    + pickle.dumps((sys.version_info[0] << 24) | sys.version_info[1], 2)
)


class Bucket:
    """Buckets are used to store the bytecode for one template.  It's created
    and initialized by the bytecode cache and passed to the loading functions.

    The buckets get an internal checksum from the cache assigned and use this
    to automatically reject outdated cache material.  Individual bytecode
    cache subclasses don't have to care about cache invalidation.
    """

    def __init__(self, environment: "Environment", key: str, checksum: str) -> None:
        self.environment = environment
        self.key = key
        self.checksum = checksum
        self.reset()

    def reset(self) -> None:
        """Resets the bucket (unloads the bytecode)."""
        self.code: t.Optional[CodeType] = None

    def load_bytecode(self, f: t.BinaryIO) -> None:
        """Loads bytecode from a file or file like object."""
        # make sure the magic header is correct
        magic = f.read(len(bc_magic))
        if magic != bc_magic:
            self.reset()
            return
        # the source code of the file changed, we need to reload
        checksum = pickle.load(f)
        if self.checksum != checksum:
            self.reset()
            return
        # if marshal_load fails then we need to reload
        try:
            self.code = marshal.load(f)
        except (EOFError, ValueError, TypeError):
            self.reset()
            return

    def write_bytecode(self, f: t.IO[bytes]) -> None:
        """Dump the bytecode into the file or file like object passed."""
        if self.code is None:
            raise TypeError("can't write empty bucket")
        f.write(bc_magic)
        pickle.dump(self.checksum, f, 2)
        marshal.dump(self.code, f)

    def bytecode_from_string(self, string: bytes) -> None:
        """Load bytecode from bytes."""
        self.load_bytecode(BytesIO(string))

    def bytecode_to_string(self) -> bytes:
        """Return the bytecode as bytes."""
        out = BytesIO()
        self.write_bytecode(out)
        return out.getvalue()


class BytecodeCache:
    """To implement your own bytecode cache you have to subclass this class
    and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
    these methods are passed a :class:`~jinja2.bccache.Bucket`.

    A very basic bytecode cache that saves the bytecode on the file system::

        from os import path

        class MyCache(BytecodeCache):

            def __init__(self, directory):
                self.directory = directory

            def load_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                if path.exists(filename):
                    with open(filename, 'rb') as f:
                        bucket.load_bytecode(f)

            def dump_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                with open(filename, 'wb') as f:
                    bucket.write_bytecode(f)

    A more advanced version of a filesystem based bytecode cache is part of
    Jinja.
    """

    def load_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to load bytecode into a
        bucket.  If they are not able to find code in the cache for the
        bucket, it must not do anything.
        """
        raise NotImplementedError()

    def dump_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to write the bytecode
        from a bucket back to the cache.  If it unable to do so it must not
        fail silently but raise an exception.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """Clears the cache.  This method is not used by Jinja but should be
        implemented to allow applications to clear the bytecode cache used
        by a particular environment.
        """

    def get_cache_key(
        self, name: str, filename: t.Optional[t.Union[str]] = None
    ) -> str:
        """Returns the unique hash key for this template name."""
        hash = sha1(name.encode("utf-8"))

        if filename is not None:
            hash.update(f"|{filename}".encode())

        return hash.hexdigest()

    def get_source_checksum(self, source: str) -> str:
        """Returns a checksum for the source."""
        return sha1(source.encode("utf-8")).hexdigest()

    def get_bucket(
        self,
        environment: "Environment",
        name: str,
        filename: t.Optional[str],
        source: str,
    ) -> Bucket:
        """Return a cache bucket for the given template.  All arguments are
        mandatory but filename may be `None`.
        """
        key = self.get_cache_key(name, filename)
        checksum = self.get_source_checksum(source)
        bucket = Bucket(environment, key, checksum)
        self.load_bytecode(bucket)
        return bucket

    def set_bucket(self, bucket: Bucket) -> None:
        """Put the bucket into the cache."""
        self.dump_bytecode(bucket)


class FileSystemBytecodeCache(BytecodeCache):
    """A bytecode cache that stores bytecode on the filesystem.  It accepts
    two arguments: The directory where the cache items are stored and a
    pattern string that is used to build the filename.

    If no directory is specified a default cache directory is selected.  On
    Windows the user's temp directory is used, on UNIX systems a directory
    is created for the user in the system temp directory.

    The pattern can be used to have multiple separate caches operate on the
    same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
    is replaced with the cache key.

    >>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

    This bytecode cache supports clearing of the cache using the clear method.
    """

    def __init__(
        self, directory: t.Optional[str] = None, pattern: str = "__jinja2_%s.cache"
    ) -> None:
        if directory is None:
            directory = self._get_default_cache_dir()
        self.directory = directory
        self.pattern = pattern

    def _get_default_cache_dir(self) -> str:
        def _unsafe_dir() -> "te.NoReturn":
            raise RuntimeError(
                "Cannot determine safe temp directory.  You "
                "need to explicitly provide one."
            )

        tmpdir = tempfile.gettempdir()

        # On windows the temporary directory is used specific unless
        # explicitly forced otherwise.  We can just use that.
        if os.name == "nt":
            return tmpdir
        if not hasattr(os, "getuid"):
            _unsafe_dir()

        dirname = f"_jinja2-cache-{os.getuid()}"
        actual_dir = os.path.join(tmpdir, dirname)

        try:
            os.mkdir(actual_dir, stat.S_IRWXU)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.chmod(actual_dir, stat.S_IRWXU)
            actual_dir_stat = os.lstat(actual_dir)
            if (
                actual_dir_stat.st_uid != os.getuid()
                or not stat.S_ISDIR(actual_dir_stat.st_mode)
                or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
            ):
                _unsafe_dir()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        actual_dir_stat = os.lstat(actual_dir)
        if (
            actual_dir_stat.st_uid != os.getuid()
            or not stat.S_ISDIR(actual_dir_stat.st_mode)
            or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
        ):
            _unsafe_dir()

        return actual_dir

    def _get_cache_filename(self, bucket: Bucket) -> str:
        return os.path.join(self.directory, self.pattern % (bucket.key,))

    def load_bytecode(self, bucket: Bucket) -> None:
        filename = self._get_cache_filename(bucket)

        # Don't test for existence before opening the file, since the
        # file could disappear after the test before the open.
        try:
            f = open(filename, "rb")
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            # PermissionError can occur on Windows when an operation is
            # in progress, such as calling clear().
            return

        with f:
            bucket.load_bytecode(f)

    def dump_bytecode(self, bucket: Bucket) -> None:
        # Write to a temporary file, then rename to the real name after
        # writing. This avoids another process reading the file before
        # it is fully written.
        name = self._get_cache_filename(bucket)
        f = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(name),
            prefix=os.path.basename(name),
            suffix=".tmp",
            delete=False,
        )

        def remove_silent() -> None:
            try:
                os.remove(f.name)
            except OSError:
                # Another process may have called clear(). On Windows,
                # another program may be holding the file open.
                pass

        try:
            with f:
                bucket.write_bytecode(f)
        except BaseException:
            remove_silent()
            raise

        try:
            os.replace(f.name, name)
        except OSError:
            # Another process may have called clear(). On Windows,
            # another program may be holding the file open.
            remove_silent()
        except BaseException:
            remove_silent()
            raise

    def clear(self) -> None:
        # imported lazily here because google app-engine doesn't support
        # write access on the file system and the function does not exist
        # normally.
        from os import remove

        files = fnmatch.filter(os.listdir(self.directory), self.pattern % ("*",))
        for filename in files:
            try:
                remove(os.path.join(self.directory, filename))
            except OSError:
                pass


class MemcachedBytecodeCache(BytecodeCache):
    """This class implements a bytecode cache that uses a memcache cache for
    storing the information.  It does not enforce a specific memcache library
    (tummy's memcache or cmemcache) but will accept any class that provides
    the minimal interface required.

    Libraries compatible with this class:

    -   `cachelib <https://github.com/pallets/cachelib>`_
    -   `python-memcached <https://pypi.org/project/python-memcached/>`_

    (Unfortunately the django cache interface is not compatible because it
    does not support storing binary data, only text. You can however pass
    the underlying cache client to the bytecode cache which is available
    as `django.core.cache.cache._client`.)

    The minimal interface for the client passed to the constructor is this:

    .. class:: MinimalClientInterface

        .. method:: set(key, value[, timeout])

            Stores the bytecode in the cache.  `value` is a string and
            `timeout` the timeout of the key.  If timeout is not provided
            a default timeout or no timeout should be assumed, if it's
            provided it's an integer with the number of seconds the cache
            item should exist.

        .. method:: get(key)

            Returns the value for the cache key.  If the item does not
            exist in the cache the return value must be `None`.

    The other arguments to the constructor are the prefix for all keys that
    is added before the actual cache key and the timeout for the bytecode in
    the cache system.  We recommend a high (or no) timeout.

    This bytecode cache does not support clearing of used items in the cache.
    The clear method is a no-operation function.

    .. versionadded:: 2.7
       Added support for ignoring memcache errors through the
       `ignore_memcache_errors` parameter.
    """

    def __init__(
        self,
        client: "_MemcachedClient",
        prefix: str = "jinja2/bytecode/",
        timeout: t.Optional[int] = None,
        ignore_memcache_errors: bool = True,
    ):
        self.client = client
        self.prefix = prefix
        self.timeout = timeout
        self.ignore_memcache_errors = ignore_memcache_errors

    def load_bytecode(self, bucket: Bucket) -> None:
        try:
            code = self.client.get(self.prefix + bucket.key)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
        else:
            bucket.bytecode_from_string(code)

    def dump_bytecode(self, bucket: Bucket) -> None:
        key = self.prefix + bucket.key
        value = bucket.bytecode_to_string()

        try:
            if self.timeout is not None:
                self.client.set(key, value, self.timeout)
            else:
                self.client.set(key, value)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
<|endoftext|>"""The optimizer tries to constant fold expressions and modify the AST
in place so that it should be faster to evaluate.

Because the AST does not contain all the scoping information and the
compiler has to find that out, we cannot do all the optimizations we
want. For example, loop unrolling doesn't work because unrolled loops
would have a different scope. The solution would be a second syntax tree
that stored the scoping rules.
"""

import typing as t

from . import nodes
from .visitor import NodeTransformer

if t.TYPE_CHECKING:
    from .environment import Environment


def optimize(node: nodes.Node, environment: "Environment") -> nodes.Node:
    """The context hint can be used to perform an static optimization
    based on the context given."""
    optimizer = Optimizer(environment)
    return t.cast(nodes.Node, optimizer.visit(node))


class Optimizer(NodeTransformer):
    def __init__(self, environment: "t.Optional[Environment]") -> None:
        self.environment = environment

    def generic_visit(
        self, node: nodes.Node, *args: t.Any, **kwargs: t.Any
    ) -> nodes.Node:
        node = super().generic_visit(node, *args, **kwargs)

        # Do constant folding. Some other nodes besides Expr have
        # as_const, but folding them causes errors later on.
        if isinstance(node, nodes.Expr):
            try:
                return nodes.Const.from_untrusted(
                    node.as_const(args[0] if args else None),
                    lineno=node.lineno,
                    environment=self.environment,
                )
            except nodes.Impossible:
                pass

        return node
<|endoftext|>import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes

import markupsafe

if t.TYPE_CHECKING:
    import typing_extensions as te

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class _MissingType:
    def __repr__(self) -> str:
        return "missing"

    def __reduce__(self) -> str:
        return "missing"


missing: t.Any = _MissingType()
"""Special singleton representing missing values for the runtime."""

internal_code: t.MutableSet[CodeType] = set()

concat = "".join


def pass_context(f: F) -> F:
    """Pass the :class:`~jinja2.runtime.Context` as the first argument
    to the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    If only ``Context.eval_context`` is needed, use
    :func:`pass_eval_context`. If only ``Context.environment`` is
    needed, use :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``contextfunction`` and ``contextfilter``.
    """
    f.jinja_pass_arg = _PassArg.context  # type: ignore
    return f


def pass_eval_context(f: F) -> F:
    """Pass the :class:`~jinja2.nodes.EvalContext` as the first argument
    to the decorated function when called while rendering a template.
    See :ref:`eval-context`.

    Can be used on functions, filters, and tests.

    If only ``EvalContext.environment`` is needed, use
    :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``evalcontextfunction`` and ``evalcontextfilter``.
    """
    f.jinja_pass_arg = _PassArg.eval_context  # type: ignore
    return f


def pass_environment(f: F) -> F:
    """Pass the :class:`~jinja2.Environment` as the first argument to
    the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    .. versionadded:: 3.0.0
        Replaces ``environmentfunction`` and ``environmentfilter``.
    """
    f.jinja_pass_arg = _PassArg.environment  # type: ignore
    return f


class _PassArg(enum.Enum):
    context = enum.auto()
    eval_context = enum.auto()
    environment = enum.auto()

    @classmethod
    def from_obj(cls, obj: F) -> t.Optional["_PassArg"]:
        if hasattr(obj, "jinja_pass_arg"):
            return obj.jinja_pass_arg  # type: ignore

        return None


def internalcode(f: F) -> F:
    """Marks the function as internally used"""
    internal_code.add(f.__code__)
    return f


def is_undefined(obj: t.Any) -> bool:
    """Check if the object passed is undefined.  This does nothing more than
    performing an instance check against :class:`Undefined` but looks nicer.
    This can be used for custom filters or tests that want to react to
    undefined variables.  For example a custom default filter can look like
    this::

        def default(var, default=''):
            if is_undefined(var):
                return default
            return var
    """
    from .runtime import Undefined

    return isinstance(obj, Undefined)


def consume(iterable: t.Iterable[t.Any]) -> None:
    """Consumes an iterable without doing anything with it."""
    for _ in iterable:
        pass


def clear_caches() -> None:
    """Jinja keeps internal caches for environments and lexers.  These are
    used so that Jinja doesn't have to recreate environments and lexers all
    the time.  Normally you don't have to care about that but if you are
    measuring memory consumption you may want to clean the caches.
    """
    from .environment import get_spontaneous_environment
    from .lexer import _lexer_cache

    get_spontaneous_environment.cache_clear()
    _lexer_cache.clear()


def import_string(import_name: str, silent: bool = False) -> t.Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If the `silent` is True the return value will be `None` if the import
    fails.

    :return: imported object
    """
    try:
        if ":" in import_name:
            module, obj = import_name.split(":", 1)
        elif "." in import_name:
            module, _, obj = import_name.rpartition(".")
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if not silent:
            raise


def open_if_exists(filename: str, mode: str = "rb") -> t.Optional[t.IO[t.Any]]:
    """Returns a file descriptor for the filename if that file exists,
    otherwise ``None``.
    """
    if not os.path.isfile(filename):
        return None

    return open(filename, mode)


def object_type_repr(obj: t.Any) -> str:
    """Returns the name of the object's type.  For some recognized
    singletons the name of the object is returned instead. (For
    example for `None` and `Ellipsis`).
    """
    if obj is None:
        return "None"
    elif obj is Ellipsis:
        return "Ellipsis"

    cls = type(obj)

    if cls.__module__ == "builtins":
        return f"{cls.__name__} object"

    return f"{cls.__module__}.{cls.__name__} object"


def pformat(obj: t.Any) -> str:
    """Format an object using :func:`pprint.pformat`."""
    from pprint import pformat

    return pformat(obj)


_http_re = re.compile(
    r"""
    ^
    (
        (https?://|www\.)  # scheme or www
        (([\w%-]+\.)+)?  # subdomain
        (
            [a-z]{2,63}  # basic tld
        |
            xn--[\w%]{2,59}  # idna tld
        )
    |
        ([\w%-]{2,63}\.)+  # basic domain
        (com|net|int|edu|gov|org|info|mil)  # basic tld
    |
        (https?://)  # scheme
        (
            (([\d]{1,3})(\.[\d]{1,3}){3})  # IPv4
        |
            (\[([\da-f]{0,4}:){2}([\da-f]{0,4}:?){1,6}])  # IPv6
        )
    )
    (?::[\d]{1,5})?  # port
    (?:[/?#]\S*)?  # path, query, and fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
_email_re = re.compile(r"^\S+@\w[\w.-]*\.\w+$")


def urlize(
    text: str,
    trim_url_limit: t.Optional[int] = None,
    rel: t.Optional[str] = None,
    target: t.Optional[str] = None,
    extra_schemes: t.Optional[t.Iterable[str]] = None,
) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param text: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.
    """
    if trim_url_limit is not None:

        def trim_url(x: str) -> str:
            if len(x) > trim_url_limit:
                return f"{x[:trim_url_limit]}..."

            return x

    else:

        def trim_url(x: str) -> str:
            return x

    words = re.split(r"(\s+)", str(markupsafe.escape(text)))
    rel_attr = f' rel="{markupsafe.escape(rel)}"' if rel else ""
    target_attr = f' target="{markupsafe.escape(target)}"' if target else ""

    for i, word in enumerate(words):
        head, middle, tail = "", word, ""
        match = re.match(r"^([(<]|&lt;)+", middle)

        if match:
            head = match.group()
            middle = middle[match.end() :]

        # Unlike lead, which is anchored to the start of the string,
        # need to check that the string ends with any of the characters
        # before trying to match all of them, to avoid backtracking.
        if middle.endswith((")", ">", ".", ",", "\n", "&gt;")):
            match = re.search(r"([)>.,\n]|&gt;)+$", middle)

            if match:
                tail = match.group()
                middle = middle[: match.start()]

        # Prefer balancing parentheses in URLs instead of ignoring a
        # trailing character.
        for start_char, end_char in ("(", ")"), ("<", ">"), ("&lt;", "&gt;"):
            start_count = middle.count(start_char)

            if start_count <= middle.count(end_char):
                # Balanced, or lighter on the left
                continue

            # Move as many as possible from the tail to balance
            for _ in range(min(start_count, tail.count(end_char))):
                end_index = tail.index(end_char) + len(end_char)
                # Move anything in the tail before the end char too
                middle += tail[:end_index]
                tail = tail[end_index:]

        if _http_re.match(middle):
            if middle.startswith("https://") or middle.startswith("http://"):
                middle = (
                    f'<a href="{middle}"{rel_attr}{target_attr}>{trim_url(middle)}</a>'
                )
            else:
                middle = (
                    f'<a href="https://{middle}"{rel_attr}{target_attr}>'
                    f"{trim_url(middle)}</a>"
                )

        elif middle.startswith("mailto:") and _email_re.match(middle[7:]):
            middle = f'<a href="{middle}">{middle[7:]}</a>'

        elif (
            "@" in middle
            and not middle.startswith("www.")
            # ignore values like `@a@b`
            and not middle.startswith("@")
            and ":" not in middle
            and _email_re.match(middle)
        ):
            middle = f'<a href="mailto:{middle}">{middle}</a>'

        elif extra_schemes is not None:
            for scheme in extra_schemes:
                if middle != scheme and middle.startswith(scheme):
                    middle = f'<a href="{middle}"{rel_attr}{target_attr}>{middle}</a>'

        words[i] = f"{head}{middle}{tail}"

    return "".join(words)


def generate_lorem_ipsum(
    n: int = 5, html: bool = True, min: int = 20, max: int = 100
) -> str:
    """Generate some lorem ipsum for the template."""
    from .constants import LOREM_IPSUM_WORDS

    words = LOREM_IPSUM_WORDS.split()
    result = []

    for _ in range(n):
        next_capitalized = True
        last_comma = last_fullstop = 0
        word = None
        last = None
        p = []

        # each paragraph contains out of 20 to 100 words.
        for idx, _ in enumerate(range(randrange(min, max))):
            while True:
                word = choice(words)
                if word != last:
                    last = word
                    break
            if next_capitalized:
                word = word.capitalize()
                next_capitalized = False
            # add commas
            if idx - randrange(3, 8) > last_comma:
                last_comma = idx
                last_fullstop += 2
                word += ","
            # add end of sentences
            if idx - randrange(10, 20) > last_fullstop:
                last_comma = last_fullstop = idx
                word += "."
                next_capitalized = True
            p.append(word)

        # ensure that the paragraph ends with a dot.
        p_str = " ".join(p)

        if p_str.endswith(","):
            p_str = p_str[:-1] + "."
        elif not p_str.endswith("."):
            p_str += "."

        result.append(p_str)

    if not html:
        return "\n\n".join(result)
    return markupsafe.Markup(
        "\n".join(f"<p>{markupsafe.escape(x)}</p>" for x in result)
    )


def url_quote(obj: t.Any, charset: str = "utf-8", for_qs: bool = False) -> str:
    """Quote a string for use in a URL using the given charset.

    :param obj: String or bytes to quote. Other types are converted to
        string then encoded to bytes using the given charset.
    :param charset: Encode text to bytes using this charset.
    :param for_qs: Quote "/" and use "+" for spaces.
    """
    if not isinstance(obj, bytes):
        if not isinstance(obj, str):
            obj = str(obj)

        obj = obj.encode(charset)

    safe = b"" if for_qs else b"/"
    rv = quote_from_bytes(obj, safe)

    if for_qs:
        rv = rv.replace("%20", "+")

    return rv


@abc.MutableMapping.register
class LRUCache:
    """A simple LRU Cache implementation."""

    # this is fast for small capacities (something below 1000) but doesn't
    # scale.  But as long as it's only used as storage for templates this
    # won't do any harm.

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._mapping: t.Dict[t.Any, t.Any] = {}
        self._queue: te.Deque[t.Any] = deque()
        self._postinit()

    def _postinit(self) -> None:
        # alias all queue methods for faster lookup
        self._popleft = self._queue.popleft
        self._pop = self._queue.pop
        self._remove = self._queue.remove
        self._wlock = Lock()
        self._append = self._queue.append

    def __getstate__(self) -> t.Mapping[str, t.Any]:
        return {
            "capacity": self.capacity,
            "_mapping": self._mapping,
            "_queue": self._queue,
        }

    def __setstate__(self, d: t.Mapping[str, t.Any]) -> None:
        self.__dict__.update(d)
        self._postinit()

    def __getnewargs__(self) -> t.Tuple[t.Any, ...]:
        return (self.capacity,)

    def copy(self) -> "te.Self":
        """Return a shallow copy of the instance."""
        rv = self.__class__(self.capacity)
        rv._mapping.update(self._mapping)
        rv._queue.extend(self._queue)
        return rv

    def get(self, key: t.Any, default: t.Any = None) -> t.Any:
        """Return an item from the cache dict or `default`"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: t.Any, default: t.Any = None) -> t.Any:
        """Set `default` if the key is not in the cache otherwise
        leave unchanged. Return the value of this key.
        """
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def clear(self) -> None:
        """Clear the cache."""
        with self._wlock:
            self._mapping.clear()
            self._queue.clear()

    def __contains__(self, key: t.Any) -> bool:
        """Check if a key exists in this cache."""
        return key in self._mapping

    def __len__(self) -> int:
        """Return the current size of the cache."""
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._mapping!r}>"

    def __getitem__(self, key: t.Any) -> t.Any:
        """Get an item from the cache. Moves the item up so that it has the
        highest priority then.

        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            rv = self._mapping[key]

            if self._queue[-1] != key:
                try:
                    self._remove(key)
                except ValueError:
                    # if something removed the key from the container
                    # when we read, ignore the ValueError that we would
                    # get otherwise.
                    pass

                self._append(key)

            return rv

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        """Sets the value for an item. Moves the item up so that it
        has the highest priority then.
        """
        with self._wlock:
            if key in self._mapping:
                self._remove(key)
            elif len(self._mapping) == self.capacity:
                del self._mapping[self._popleft()]

            self._append(key)
            self._mapping[key] = value

    def __delitem__(self, key: t.Any) -> None:
        """Remove an item from the cache dict.
        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            del self._mapping[key]

            try:
                self._remove(key)
            except ValueError:
                pass

    def items(self) -> t.Iterable[t.Tuple[t.Any, t.Any]]:
        """Return a list of items."""
        result = [(key, self._mapping[key]) for key in list(self._queue)]
        result.reverse()
        return result

    def values(self) -> t.Iterable[t.Any]:
        """Return a list of all values."""
        return [x[1] for x in self.items()]

    def keys(self) -> t.Iterable[t.Any]:
        """Return a list of all keys ordered by most recent usage."""
        return list(self)

    def __iter__(self) -> t.Iterator[t.Any]:
        return reversed(tuple(self._queue))

    def __reversed__(self) -> t.Iterator[t.Any]:
        """Iterate over the keys in the cache dict, oldest items
        coming first.
        """
        return iter(tuple(self._queue))

    __copy__ = copy


def select_autoescape(
    enabled_extensions: t.Collection[str] = ("html", "htm", "xml"),
    disabled_extensions: t.Collection[str] = (),
    default_for_string: bool = True,
    default: bool = False,
) -> t.Callable[[t.Optional[str]], bool]:
    """Intelligently sets the initial value of autoescaping based on the
    filename of the template.  This is the recommended way to configure
    autoescaping if you do not want to write a custom function yourself.

    If you want to enable it for all templates created from strings or
    for all templates with `.html` and `.xml` extensions::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            enabled_extensions=('html', 'xml'),
            default_for_string=True,
        ))

    Example configuration to turn it on at all times except if the template
    ends with `.txt`::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            disabled_extensions=('txt',),
            default_for_string=True,
            default=True,
        ))

    The `enabled_extensions` is an iterable of all the extensions that
    autoescaping should be enabled for.  Likewise `disabled_extensions` is
    a list of all templates it should be disabled for.  If a template is
    loaded from a string then the default from `default_for_string` is used.
    If nothing matches then the initial value of autoescaping is set to the
    value of `default`.

    For security reasons this function operates case insensitive.

    .. versionadded:: 2.9
    """
    enabled_patterns = tuple(f".{x.lstrip('.').lower()}" for x in enabled_extensions)
    disabled_patterns = tuple(f".{x.lstrip('.').lower()}" for x in disabled_extensions)

    def autoescape(template_name: t.Optional[str]) -> bool:
        if template_name is None:
            return default_for_string
        template_name = template_name.lower()
        if template_name.endswith(enabled_patterns):
            return True
        if template_name.endswith(disabled_patterns):
            return False
        return default

    return autoescape


def htmlsafe_json_dumps(
    obj: t.Any, dumps: t.Optional[t.Callable[..., str]] = None, **kwargs: t.Any
) -> markupsafe.Markup:
    """Serialize an object to a string of JSON with :func:`json.dumps`,
    then replace HTML-unsafe characters with Unicode escapes and mark
    the result safe with :class:`~markupsafe.Markup`.

    This is available in templates as the ``|tojson`` filter.

    The following characters are escaped: ``<``, ``>``, ``&``, ``'``.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    :param obj: The object to serialize to JSON.
    :param dumps: The ``dumps`` function to use. Defaults to
        ``env.policies["json.dumps_function"]``, which defaults to
        :func:`json.dumps`.
    :param kwargs: Extra arguments to pass to ``dumps``. Merged onto
        ``env.policies["json.dumps_kwargs"]``.

    .. versionchanged:: 3.0
        The ``dumper`` parameter is renamed to ``dumps``.

    .. versionadded:: 2.9
    """
    if dumps is None:
        dumps = json.dumps

    return markupsafe.Markup(
        dumps(obj, **kwargs)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


class Cycler:
    """Cycle through values by yield them one at a time, then restarting
    once the end is reached. Available as ``cycler`` in templates.

    Similar to ``loop.cycle``, but can be used outside loops or across
    multiple loops. For example, render a list of folders and files in a
    list, alternating giving them "odd" and "even" classes.

    .. code-block:: html+jinja

        {% set row_class = cycler("odd", "even") %}
        <ul class="browser">
        {% for folder in folders %}
          <li class="folder {{ row_class.next() }}">{{ folder }}
        {% endfor %}
        {% for file in files %}
          <li class="file {{ row_class.next() }}">{{ file }}
        {% endfor %}
        </ul>

    :param items: Each positional argument will be yielded in the order
        given for each cycle.

    .. versionadded:: 2.1
    """

    def __init__(self, *items: t.Any) -> None:
        if not items:
            raise RuntimeError("at least one item has to be provided")
        self.items = items
        self.pos = 0

    def reset(self) -> None:
        """Resets the current item to the first item."""
        self.pos = 0

    @property
    def current(self) -> t.Any:
        """Return the current item. Equivalent to the item that will be
        returned next time :meth:`next` is called.
        """
        return self.items[self.pos]

    def next(self) -> t.Any:
        """Return the current item, then advance :attr:`current` to the
        next item.
        """
        rv = self.current
        self.pos = (self.pos + 1) % len(self.items)
        return rv

    __next__ = next


class Joiner:
    """A joining helper for templates."""

    def __init__(self, sep: str = ", ") -> None:
        self.sep = sep
        self.used = False

    def __call__(self) -> str:
        if not self.used:
            self.used = True
            return ""
        return self.sep


class Namespace:
    """A namespace object that can hold arbitrary attributes.  It may be
    initialized from a dictionary or with keyword arguments."""

    def __init__(*args: t.Any, **kwargs: t.Any) -> None:  # noqa: B902
        self, args = args[0], args[1:]
        self.__attrs = dict(*args, **kwargs)

    def __getattribute__(self, name: str) -> t.Any:
        # __class__ is needed for the awaitable check in async mode
        if name in {"_Namespace__attrs", "__class__"}:
            return object.__getattribute__(self, name)
        try:
            return self.__attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setitem__(self, name: str, value: t.Any) -> None:
        self.__attrs[name] = value

    def __repr__(self) -> str:
        return f"<Namespace {self.__attrs!r}>"
<|endoftext|>"""Jinja is a template engine written in pure Python. It provides a
non-XML syntax that supports inline expressions and an optional
sandboxed environment.
"""

from .bccache import BytecodeCache as BytecodeCache
from .bccache import FileSystemBytecodeCache as FileSystemBytecodeCache
from .bccache import MemcachedBytecodeCache as MemcachedBytecodeCache
from .environment import Environment as Environment
from .environment import Template as Template
from .exceptions import TemplateAssertionError as TemplateAssertionError
from .exceptions import TemplateError as TemplateError
from .exceptions import TemplateNotFound as TemplateNotFound
from .exceptions import TemplateRuntimeError as TemplateRuntimeError
from .exceptions import TemplatesNotFound as TemplatesNotFound
from .exceptions import TemplateSyntaxError as TemplateSyntaxError
from .exceptions import UndefinedError as UndefinedError
from .loaders import BaseLoader as BaseLoader
from .loaders import ChoiceLoader as ChoiceLoader
from .loaders import DictLoader as DictLoader
from .loaders import FileSystemLoader as FileSystemLoader
from .loaders import FunctionLoader as FunctionLoader
from .loaders import ModuleLoader as ModuleLoader
from .loaders import PackageLoader as PackageLoader
from .loaders import PrefixLoader as PrefixLoader
from .runtime import ChainableUndefined as ChainableUndefined
from .runtime import DebugUndefined as DebugUndefined
from .runtime import make_logging_undefined as make_logging_undefined
from .runtime import StrictUndefined as StrictUndefined
from .runtime import Undefined as Undefined
from .utils import clear_caches as clear_caches
from .utils import is_undefined as is_undefined
from .utils import pass_context as pass_context
from .utils import pass_environment as pass_environment
from .utils import pass_eval_context as pass_eval_context
from .utils import select_autoescape as select_autoescape

__version__ = "3.1.6"
<|endoftext|>"""API for traversing the AST nodes. Implemented by the compiler and
meta introspection.
"""

import typing as t

from .nodes import Node

if t.TYPE_CHECKING:
    import typing_extensions as te

    class VisitCallable(te.Protocol):
        def __call__(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any: ...


class NodeVisitor:
    """Walks the abstract syntax tree and call visitor functions for every
    node found.  The visitor functions may return values which will be
    forwarded by the `visit` method.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `get_visitor` function.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.
    """

    def get_visitor(self, node: Node) -> "t.Optional[VisitCallable]":
        """Return the visitor function for this node or `None` if no visitor
        exists for this node.  In that case the generic visit function is
        used instead.
        """
        return getattr(self, f"visit_{type(node).__name__}", None)

    def visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Visit a node."""
        f = self.get_visitor(node)

        if f is not None:
            return f(node, *args, **kwargs)

        return self.generic_visit(node, *args, **kwargs)

    def generic_visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Called if no explicit visitor function exists for a node."""
        for child_node in node.iter_child_nodes():
            self.visit(child_node, *args, **kwargs)


class NodeTransformer(NodeVisitor):
    """Walks the abstract syntax tree and allows modifications of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor functions to replace or remove the old node.  If the return
    value of the visitor function is `None` the node will be removed
    from the previous location otherwise it's replaced with the return
    value.  The return value may be the original node in which case no
    replacement takes place.
    """

    def generic_visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> Node:
        for field, old_value in node.iter_fields():
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Node):
                        value = self.visit(value, *args, **kwargs)
                        if value is None:
                            continue
                        elif not isinstance(value, Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value, *args, **kwargs)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_list(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.List[Node]:
        """As transformers may return lists in some places this method
        can be used to enforce a list as return value.
        """
        rv = self.visit(node, *args, **kwargs)

        if not isinstance(rv, list):
            return [rv]

        return rv
<|endoftext|>"""API and implementations for loading templates from different data
sources.
"""

import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType

from .exceptions import TemplateNotFound
from .utils import internalcode

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template


def split_template_path(template: str) -> t.List[str]:
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split("/"):
        if (
            os.path.sep in piece
            or (os.path.altsep and os.path.altsep in piece)
            or piece == os.path.pardir
        ):
            raise TemplateNotFound(template)
        elif piece and piece != ".":
            pieces.append(piece)
    return pieces


class BaseLoader:
    """Baseclass for all loaders.  Subclass this and override `get_source` to
    implement a custom loading mechanism.  The environment provides a
    `get_template` method that calls the loader's `load` method to get the
    :class:`Template` object.

    A very basic example for a loader that looks up templates on the file
    system could look like this::

        from jinja2 import BaseLoader, TemplateNotFound
        from os.path import join, exists, getmtime

        class MyLoader(BaseLoader):

            def __init__(self, path):
                self.path = path

            def get_source(self, environment, template):
                path = join(self.path, template)
                if not exists(path):
                    raise TemplateNotFound(template)
                mtime = getmtime(path)
                with open(path) as f:
                    source = f.read()
                return source, path, lambda: mtime == getmtime(path)
    """

    #: if set to `False` it indicates that the loader cannot provide access
    #: to the source of templates.
    #:
    #: .. versionadded:: 2.4
    has_source_access = True

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        """Get the template source, filename and reload helper for a template.
        It's passed the environment and template name and has to return a
        tuple in the form ``(source, filename, uptodate)`` or raise a
        `TemplateNotFound` error if it can't locate the template.

        The source part of the returned tuple must be the source of the
        template as a string. The filename should be the name of the
        file on the filesystem if it was loaded from there, otherwise
        ``None``. The filename is used by Python for the tracebacks
        if no loader extension is used.

        The last item in the tuple is the `uptodate` function.  If auto
        reloading is enabled it's always called to check if the template
        changed.  No arguments are passed so the function must store the
        old state somewhere (for example in a closure).  If it returns `False`
        the template will be reloaded.
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"{type(self).__name__} cannot provide access to the source"
            )
        raise TemplateNotFound(template)

    def list_templates(self) -> t.List[str]:
        """Iterates over all templates.  If the loader does not support that
        it should raise a :exc:`TypeError` which is the default behavior.
        """
        raise TypeError("this loader cannot iterate over all templates")

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Loads a template.  This method looks up the template in the cache
        or loads one by calling :meth:`get_source`.  Subclasses should not
        override this method as loaders working on collections of other
        loaders (such as :class:`PrefixLoader` or :class:`ChoiceLoader`)
        will not call this method but `get_source` directly.
        """
        code = None
        if globals is None:
            globals = {}

        # first we try to get the source for this template together
        # with the filename and the uptodate function.
        source, filename, uptodate = self.get_source(environment, name)

        # try to load the code from the bytecode cache if there is a
        # bytecode cache configured.
        bcc = environment.bytecode_cache
        if bcc is not None:
            bucket = bcc.get_bucket(environment, name, filename, source)
            code = bucket.code

        # if we don't have code so far (not cached, no longer up to
        # date) etc. we compile the template
        if code is None:
            code = environment.compile(source, name, filename)

        # if the bytecode cache is available and the bucket doesn't
        # have a code so far, we give the bucket the new code and put
        # it back to the bytecode cache.
        if bcc is not None and bucket.code is None:
            bucket.code = code
            bcc.set_bucket(bucket)

        return environment.template_class.from_code(
            environment, code, globals, uptodate
        )


class FileSystemLoader(BaseLoader):
    """Load templates from a directory in the file system.

    The path can be relative or absolute. Relative paths are relative to
    the current working directory.

    .. code-block:: python

        loader = FileSystemLoader("templates")

    A list of paths can be given. The directories will be searched in
    order, stopping at the first matching template.

    .. code-block:: python

        loader = FileSystemLoader(["/override/templates", "/default/templates"])

    :param searchpath: A path, or list of paths, to the directory that
        contains the templates.
    :param encoding: Use this encoding to read the text from template
        files.
    :param followlinks: Follow symbolic links in the path.

    .. versionchanged:: 2.8
        Added the ``followlinks`` parameter.
    """

    def __init__(
        self,
        searchpath: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
        encoding: str = "utf-8",
        followlinks: bool = False,
    ) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]

        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, str, t.Callable[[], bool]]:
        pieces = split_template_path(template)

        for searchpath in self.searchpath:
            # Use posixpath even on Windows to avoid "drive:" or UNC
            # segments breaking out of the search directory.
            filename = posixpath.join(searchpath, *pieces)

            if os.path.isfile(filename):
                break
        else:
            plural = "path" if len(self.searchpath) == 1 else "paths"
            paths_str = ", ".join(repr(p) for p in self.searchpath)
            raise TemplateNotFound(
                template,
                f"{template!r} not found in search {plural}: {paths_str}",
            )

        with open(filename, encoding=self.encoding) as f:
            contents = f.read()

        mtime = os.path.getmtime(filename)

        def uptodate() -> bool:
            try:
                return os.path.getmtime(filename) == mtime
            except OSError:
                return False

        # Use normpath to convert Windows altsep to sep.
        return contents, os.path.normpath(filename), uptodate

    def list_templates(self) -> t.List[str]:
        found = set()
        for searchpath in self.searchpath:
            walk_dir = os.walk(searchpath, followlinks=self.followlinks)
            for dirpath, _, filenames in walk_dir:
                for filename in filenames:
                    template = (
                        os.path.join(dirpath, filename)[len(searchpath) :]
                        .strip(os.path.sep)
                        .replace(os.path.sep, "/")
                    )
                    if template[:2] == "./":
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)


if sys.version_info >= (3, 13):

    def _get_zipimporter_files(z: t.Any) -> t.Dict[str, object]:
        try:
            get_files = z._get_files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required"
                " metadata to list templates."
            ) from e
        return get_files()
else:

    def _get_zipimporter_files(z: t.Any) -> t.Dict[str, object]:
        try:
            files = z._files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required"
                " metadata to list templates."
            ) from e
        return files  # type: ignore[no-any-return]


class PackageLoader(BaseLoader):
    """Load templates from a directory in a Python package.

    :param package_name: Import name of the package that contains the
        template directory.
    :param package_path: Directory within the imported package that
        contains the templates.
    :param encoding: Encoding of template files.

    The following example looks up templates in the ``pages`` directory
    within the ``project.ui`` package.

    .. code-block:: python

        loader = PackageLoader("project.ui", "pages")

    Only packages installed as directories (standard pip behavior) or
    zip/egg files (less common) are supported. The Python API for
    introspecting data in packages is too limited to support other
    installation methods the way this loader requires.

    There is limited support for :pep:`420` namespace packages. The
    template directory is assumed to only be in one namespace
    contributor. Zip files contributing to a namespace are not
    supported.

    .. versionchanged:: 3.0
        No longer uses ``setuptools`` as a dependency.

    .. versionchanged:: 3.0
        Limited PEP 420 namespace package support.
    """

    def __init__(
        self,
        package_name: str,
        package_path: "str" = "templates",
        encoding: str = "utf-8",
    ) -> None:
        package_path = os.path.normpath(package_path).rstrip(os.path.sep)

        # normpath preserves ".", which isn't valid in zip paths.
        if package_path == os.path.curdir:
            package_path = ""
        elif package_path[:2] == os.path.curdir + os.path.sep:
            package_path = package_path[2:]

        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding

        # Make sure the package exists. This also makes namespace
        # packages work, otherwise get_loader returns None.
        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, "An import spec was not found for the package."
        loader = spec.loader
        assert loader is not None, "A loader was not found for the package."
        self._loader = loader
        self._archive = None

        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))  # type: ignore
            template_root = os.path.join(pkgdir, package_path).rstrip(os.path.sep)
        else:
            roots: t.List[str] = []

            # One element for regular packages, multiple for namespace
            # packages, or None for single module file.
            if spec.submodule_search_locations:
                roots.extend(spec.submodule_search_locations)
            # A single module file, use the parent directory instead.
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))

            if not roots:
                raise ValueError(
                    f"The {package_name!r} package was not installed in a"
                    " way that PackageLoader understands."
                )

            for root in roots:
                root = os.path.join(root, package_path)

                if os.path.isdir(root):
                    template_root = root
                    break
            else:
                raise ValueError(
                    f"PackageLoader could not find a {package_path!r} directory"
                    f" in the {package_name!r} package."
                )

        self._template_root = template_root

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, str, t.Optional[t.Callable[[], bool]]]:
        # Use posixpath even on Windows to avoid "drive:" or UNC
        # segments breaking out of the search directory. Use normpath to
        # convert Windows altsep to sep.
        p = os.path.normpath(
            posixpath.join(self._template_root, *split_template_path(template))
        )
        up_to_date: t.Optional[t.Callable[[], bool]]

        if self._archive is None:
            # Package is a directory.
            if not os.path.isfile(p):
                raise TemplateNotFound(template)

            with open(p, "rb") as f:
                source = f.read()

            mtime = os.path.getmtime(p)

            def up_to_date() -> bool:
                return os.path.isfile(p) and os.path.getmtime(p) == mtime

        else:
            # Package is a zip file.
            try:
                source = self._loader.get_data(p)  # type: ignore
            except OSError as e:
                raise TemplateNotFound(template) from e

            # Could use the zip's mtime for all template mtimes, but
            # would need to safely reload the module if it's out of
            # date, so just report it as always current.
            up_to_date = None

        return source.decode(self.encoding), p, up_to_date

    def list_templates(self) -> t.List[str]:
        results: t.List[str] = []

        if self._archive is None:
            # Package is a directory.
            offset = len(self._template_root)

            for dirpath, _, filenames in os.walk(self._template_root):
                dirpath = dirpath[offset:].lstrip(os.path.sep)
                results.extend(
                    os.path.join(dirpath, name).replace(os.path.sep, "/")
                    for name in filenames
                )
        else:
            files = _get_zipimporter_files(self._loader)

            # Package is a zip file.
            prefix = (
                self._template_root[len(self._archive) :].lstrip(os.path.sep)
                + os.path.sep
            )
            offset = len(prefix)

            for name in files:
                # Find names under the templates directory that aren't directories.
                if name.startswith(prefix) and name[-1] != os.path.sep:
                    results.append(name[offset:].replace(os.path.sep, "/"))

        results.sort()
        return results


class DictLoader(BaseLoader):
    """Loads a template from a Python dict mapping template names to
    template source.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled by default.
    """

    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, None, t.Callable[[], bool]]:
        if template in self.mapping:
            source = self.mapping[template]
            return source, None, lambda: source == self.mapping.get(template)
        raise TemplateNotFound(template)

    def list_templates(self) -> t.List[str]:
        return sorted(self.mapping)


class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    a string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(
        self,
        load_func: t.Callable[
            [str],
            t.Optional[
                t.Union[
                    str, t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]
                ]
            ],
        ],
    ) -> None:
        self.load_func = load_func

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        rv = self.load_func(template)

        if rv is None:
            raise TemplateNotFound(template)

        if isinstance(rv, str):
            return rv, None, None

        return rv


class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(
        self, mapping: t.Mapping[str, BaseLoader], delimiter: str = "/"
    ) -> None:
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template: str) -> t.Tuple[BaseLoader, str]:
        try:
            prefix, name = template.split(self.delimiter, 1)
            loader = self.mapping[prefix]
        except (ValueError, KeyError) as e:
            raise TemplateNotFound(template) from e
        return loader, name

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(template) from e

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(name) from e

    def list_templates(self) -> t.List[str]:
        result = []
        for prefix, loader in self.mapping.items():
            for template in loader.list_templates():
                result.append(prefix + self.delimiter + template)
        return result


class ChoiceLoader(BaseLoader):
    """This loader works like the `PrefixLoader` just that no prefix is
    specified.  If a template could not be found by one loader the next one
    is tried.

    >>> loader = ChoiceLoader([
    ...     FileSystemLoader('/path/to/user/templates'),
    ...     FileSystemLoader('/path/to/system/templates')
    ... ])

    This is useful if you want to allow users to override builtin templates
    from a different location.
    """

    def __init__(self, loaders: t.Sequence[BaseLoader]) -> None:
        self.loaders = loaders

    def get_source(
        self, environment: "Environment", template: str
    ) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        for loader in self.loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        for loader in self.loaders:
            try:
                return loader.load(environment, name, globals)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(name)

    def list_templates(self) -> t.List[str]:
        found = set()
        for loader in self.loaders:
            found.update(loader.list_templates())
        return sorted(found)


class _TemplateModule(ModuleType):
    """Like a normal module but with support for weak references"""


class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ModuleLoader('/path/to/compiled/templates')

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """

    has_source_access = False

    def __init__(
        self,
        path: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
    ) -> None:
        package_name = f"_jinja2_module_templates_{id(self):x}"

        # create a fake module that looks for the templates in the
        # path given.
        mod = _TemplateModule(package_name)

        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]

        mod.__path__ = [os.fspath(p) for p in path]

        sys.modules[package_name] = weakref.proxy(
            mod, lambda x: sys.modules.pop(package_name, None)
        )

        # the only strong reference, the sys.modules entry is weak
        # so that the garbage collector can remove it once the
        # loader that created it goes out of business.
        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name: str) -> str:
        return "tmpl_" + sha1(name.encode("utf-8")).hexdigest()

    @staticmethod
    def get_module_filename(name: str) -> str:
        return ModuleLoader.get_template_key(name) + ".py"

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        key = self.get_template_key(name)
        module = f"{self.package_name}.{key}"
        mod = getattr(self.module, module, None)

        if mod is None:
            try:
                mod = __import__(module, None, None, ["root"])
            except ImportError as e:
                raise TemplateNotFound(name) from e

            # remove the entry from sys.modules, we only want the attribute
            # on the module object we have stored on the loader.
            sys.modules.pop(module, None)

        if globals is None:
            globals = {}

        return environment.template_class.from_module_dict(
            environment, mod.__dict__, globals
        )
<|endoftext|>"""AST nodes generated by the parser for the compiler. Also provides
some node tree helper functions used by the parser and compiler in order
to normalize nodes.
"""

import inspect
import operator
import typing as t
from collections import deque

from markupsafe import Markup

from .utils import _PassArg

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

_NodeBound = t.TypeVar("_NodeBound", bound="Node")

_binop_to_func: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "**": operator.pow,
    "%": operator.mod,
    "+": operator.add,
    "-": operator.sub,
}

_uaop_to_func: t.Dict[str, t.Callable[[t.Any], t.Any]] = {
    "not": operator.not_,
    "+": operator.pos,
    "-": operator.neg,
}

_cmpop_to_func: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gteq": operator.ge,
    "lt": operator.lt,
    "lteq": operator.le,
    "in": lambda a, b: a in b,
    "notin": lambda a, b: a not in b,
}


class Impossible(Exception):
    """Raised if the node could not perform a requested action."""


class NodeType(type):
    """A metaclass for nodes that handles the field and attribute
    inheritance.  fields and attributes from the parent class are
    automatically forwarded to the child."""

    def __new__(mcs, name, bases, d):  # type: ignore
        for attr in "fields", "attributes":
            storage: t.List[t.Tuple[str, ...]] = []
            storage.extend(getattr(bases[0] if bases else object, attr, ()))
            storage.extend(d.get(attr, ()))
            assert len(bases) <= 1, "multiple inheritance not allowed"
            assert len(storage) == len(set(storage)), "layout conflict"
            d[attr] = tuple(storage)
        d.setdefault("abstract", False)
        return type.__new__(mcs, name, bases, d)


class EvalContext:
    """Holds evaluation time information.  Custom attributes can be attached
    to it in extensions.
    """

    def __init__(
        self, environment: "Environment", template_name: t.Optional[str] = None
    ) -> None:
        self.environment = environment
        if callable(environment.autoescape):
            self.autoescape = environment.autoescape(template_name)
        else:
            self.autoescape = environment.autoescape
        self.volatile = False

    def save(self) -> t.Mapping[str, t.Any]:
        return self.__dict__.copy()

    def revert(self, old: t.Mapping[str, t.Any]) -> None:
        self.__dict__.clear()
        self.__dict__.update(old)


def get_eval_context(node: "Node", ctx: t.Optional[EvalContext]) -> EvalContext:
    if ctx is None:
        if node.environment is None:
            raise RuntimeError(
                "if no eval context is passed, the node must have an"
                " attached environment."
            )
        return EvalContext(node.environment)
    return ctx


class Node(metaclass=NodeType):
    """Baseclass for all Jinja nodes.  There are a number of nodes available
    of different types.  There are four major types:

    -   :class:`Stmt`: statements
    -   :class:`Expr`: expressions
    -   :class:`Helper`: helper nodes
    -   :class:`Template`: the outermost wrapper node

    All nodes have fields and attributes.  Fields may be other nodes, lists,
    or arbitrary values.  Fields are passed to the constructor as regular
    positional arguments, attributes as keyword arguments.  Each node has
    two attributes: `lineno` (the line number of the node) and `environment`.
    The `environment` attribute is set at the end of the parsing process for
    all nodes automatically.
    """

    fields: t.Tuple[str, ...] = ()
    attributes: t.Tuple[str, ...] = ("lineno", "environment")
    abstract = True

    lineno: int
    environment: t.Optional["Environment"]

    def __init__(self, *fields: t.Any, **attributes: t.Any) -> None:
        if self.abstract:
            raise TypeError("abstract nodes are not instantiable")
        if fields:
            if len(fields) != len(self.fields):
                if not self.fields:
                    raise TypeError(f"{type(self).__name__!r} takes 0 arguments")
                raise TypeError(
                    f"{type(self).__name__!r} takes 0 or {len(self.fields)}"
                    f" argument{'s' if len(self.fields) != 1 else ''}"
                )
            for name, arg in zip(self.fields, fields):
                setattr(self, name, arg)
        for attr in self.attributes:
            setattr(self, attr, attributes.pop(attr, None))
        if attributes:
            raise TypeError(f"unknown attribute {next(iter(attributes))!r}")

    def iter_fields(
        self,
        exclude: t.Optional[t.Container[str]] = None,
        only: t.Optional[t.Container[str]] = None,
    ) -> t.Iterator[t.Tuple[str, t.Any]]:
        """This method iterates over all fields that are defined and yields
        ``(key, value)`` tuples.  Per default all fields are returned, but
        it's possible to limit that to some fields by providing the `only`
        parameter or to exclude some using the `exclude` parameter.  Both
        should be sets or tuples of field names.
        """
        for name in self.fields:
            if (
                (exclude is None and only is None)
                or (exclude is not None and name not in exclude)
                or (only is not None and name in only)
            ):
                try:
                    yield name, getattr(self, name)
                except AttributeError:
                    pass

    def iter_child_nodes(
        self,
        exclude: t.Optional[t.Container[str]] = None,
        only: t.Optional[t.Container[str]] = None,
    ) -> t.Iterator["Node"]:
        """Iterates over all direct child nodes of the node.  This iterates
        over all fields and yields the values of they are nodes.  If the value
        of a field is a list all the nodes in that list are returned.
        """
        for _, item in self.iter_fields(exclude, only):
            if isinstance(item, list):
                for n in item:
                    if isinstance(n, Node):
                        yield n
            elif isinstance(item, Node):
                yield item

    def find(self, node_type: t.Type[_NodeBound]) -> t.Optional[_NodeBound]:
        """Find the first node of a given type.  If no such node exists the
        return value is `None`.
        """
        for result in self.find_all(node_type):
            return result

        return None

    def find_all(
        self, node_type: t.Union[t.Type[_NodeBound], t.Tuple[t.Type[_NodeBound], ...]]
    ) -> t.Iterator[_NodeBound]:
        """Find all the nodes of a given type.  If the type is a tuple,
        the check is performed for any of the tuple items.
        """
        for child in self.iter_child_nodes():
            if isinstance(child, node_type):
                yield child  # type: ignore
            yield from child.find_all(node_type)

    def set_ctx(self, ctx: str) -> "Node":
        """Reset the context of a node and all child nodes.  Per default the
        parser will all generate nodes that have a 'load' context as it's the
        most common one.  This method is used in the parser to set assignment
        targets and other nodes to a store context.
        """
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if "ctx" in node.fields:
                node.ctx = ctx  # type: ignore
            todo.extend(node.iter_child_nodes())
        return self

    def set_lineno(self, lineno: int, override: bool = False) -> "Node":
        """Set the line numbers of the node and children."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if "lineno" in node.attributes:
                if node.lineno is None or override:
                    node.lineno = lineno
            todo.extend(node.iter_child_nodes())
        return self

    def set_environment(self, environment: "Environment") -> "Node":
        """Set the environment for all nodes."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            node.environment = environment
            todo.extend(node.iter_child_nodes())
        return self

    def __eq__(self, other: t.Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return tuple(self.iter_fields()) == tuple(other.iter_fields())

    __hash__ = object.__hash__

    def __repr__(self) -> str:
        args_str = ", ".join(f"{a}={getattr(self, a, None)!r}" for a in self.fields)
        return f"{type(self).__name__}({args_str})"

    def dump(self) -> str:
        def _dump(node: t.Union[Node, t.Any]) -> None:
            if not isinstance(node, Node):
                buf.append(repr(node))
                return

            buf.append(f"nodes.{type(node).__name__}(")
            if not node.fields:
                buf.append(")")
                return
            for idx, field in enumerate(node.fields):
                if idx:
                    buf.append(", ")
                value = getattr(node, field)
                if isinstance(value, list):
                    buf.append("[")
                    for idx, item in enumerate(value):
                        if idx:
                            buf.append(", ")
                        _dump(item)
                    buf.append("]")
                else:
                    _dump(value)
            buf.append(")")

        buf: t.List[str] = []
        _dump(self)
        return "".join(buf)


class Stmt(Node):
    """Base node for all statements."""

    abstract = True


class Helper(Node):
    """Nodes that exist in a specific context only."""

    abstract = True


class Template(Node):
    """Node that represents a template.  This must be the outermost node that
    is passed to the compiler.
    """

    fields = ("body",)
    body: t.List[Node]


class Output(Stmt):
    """A node that holds multiple expressions which are then printed out.
    This is used both for the `print` statement and the regular template data.
    """

    fields = ("nodes",)
    nodes: t.List["Expr"]


class Extends(Stmt):
    """Represents an extends statement."""

    fields = ("template",)
    template: "Expr"


class For(Stmt):
    """The for loop.  `target` is the target for the iteration (usually a
    :class:`Name` or :class:`Tuple`), `iter` the iterable.  `body` is a list
    of nodes that are used as loop-body, and `else_` a list of nodes for the
    `else` block.  If no else node exists it has to be an empty list.

    For filtered nodes an expression can be stored as `test`, otherwise `None`.
    """

    fields = ("target", "iter", "body", "else_", "test", "recursive")
    target: Node
    iter: Node
    body: t.List[Node]
    else_: t.List[Node]
    test: t.Optional[Node]
    recursive: bool


class If(Stmt):
    """If `test` is true, `body` is rendered, else `else_`."""

    fields = ("test", "body", "elif_", "else_")
    test: Node
    body: t.List[Node]
    elif_: t.List["If"]
    else_: t.List[Node]


class Macro(Stmt):
    """A macro definition.  `name` is the name of the macro, `args` a list of
    arguments and `defaults` a list of defaults if there are any.  `body` is
    a list of nodes for the macro body.
    """

    fields = ("name", "args", "defaults", "body")
    name: str
    args: t.List["Name"]
    defaults: t.List["Expr"]
    body: t.List[Node]


class CallBlock(Stmt):
    """Like a macro without a name but a call instead.  `call` is called with
    the unnamed macro as `caller` argument this node holds.
    """

    fields = ("call", "args", "defaults", "body")
    call: "Call"
    args: t.List["Name"]
    defaults: t.List["Expr"]
    body: t.List[Node]


class FilterBlock(Stmt):
    """Node for filter sections."""

    fields = ("body", "filter")
    body: t.List[Node]
    filter: "Filter"


class With(Stmt):
    """Specific node for with statements.  In older versions of Jinja the
    with statement was implemented on the base of the `Scope` node instead.

    .. versionadded:: 2.9.3
    """

    fields = ("targets", "values", "body")
    targets: t.List["Expr"]
    values: t.List["Expr"]
    body: t.List[Node]


class Block(Stmt):
    """A node that represents a block.

    .. versionchanged:: 3.0.0
        the `required` field was added.
    """

    fields = ("name", "body", "scoped", "required")
    name: str
    body: t.List[Node]
    scoped: bool
    required: bool


class Include(Stmt):
    """A node that represents the include tag."""

    fields = ("template", "with_context", "ignore_missing")
    template: "Expr"
    with_context: bool
    ignore_missing: bool


class Import(Stmt):
    """A node that represents the import tag."""

    fields = ("template", "target", "with_context")
    template: "Expr"
    target: str
    with_context: bool


class FromImport(Stmt):
    """A node that represents the from import tag.  It's important to not
    pass unsafe names to the name attribute.  The compiler translates the
    attribute lookups directly into getattr calls and does *not* use the
    subscript callback of the interface.  As exported variables may not
    start with double underscores (which the parser asserts) this is not a
    problem for regular Jinja code, but if this node is used in an extension
    extra care must be taken.

    The list of names may contain tuples if aliases are wanted.
    """

    fields = ("template", "names", "with_context")
    template: "Expr"
    names: t.List[t.Union[str, t.Tuple[str, str]]]
    with_context: bool


class ExprStmt(Stmt):
    """A statement that evaluates an expression and discards the result."""

    fields = ("node",)
    node: Node


class Assign(Stmt):
    """Assigns an expression to a target."""

    fields = ("target", "node")
    target: "Expr"
    node: Node


class AssignBlock(Stmt):
    """Assigns a block to a target."""

    fields = ("target", "filter", "body")
    target: "Expr"
    filter: t.Optional["Filter"]
    body: t.List[Node]


class Expr(Node):
    """Baseclass for all expressions."""

    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        """Return the value of the expression as constant or raise
        :exc:`Impossible` if this was not possible.

        An :class:`EvalContext` can be provided, if none is given
        a default context is created which requires the nodes to have
        an attached environment.

        .. versionchanged:: 2.4
           the `eval_ctx` parameter was added.
        """
        raise Impossible()

    def can_assign(self) -> bool:
        """Check if it's possible to assign something to this node."""
        return False


class BinExpr(Expr):
    """Baseclass for all binary expressions."""

    fields = ("left", "right")
    left: Expr
    right: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        # intercepted operators cannot be folded at compile time
        if (
            eval_ctx.environment.sandboxed
            and self.operator in eval_ctx.environment.intercepted_binops  # type: ignore
        ):
            raise Impossible()
        f = _binop_to_func[self.operator]
        try:
            return f(self.left.as_const(eval_ctx), self.right.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e


class UnaryExpr(Expr):
    """Baseclass for all unary expressions."""

    fields = ("node",)
    node: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        # intercepted operators cannot be folded at compile time
        if (
            eval_ctx.environment.sandboxed
            and self.operator in eval_ctx.environment.intercepted_unops  # type: ignore
        ):
            raise Impossible()
        f = _uaop_to_func[self.operator]
        try:
            return f(self.node.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e


class Name(Expr):
    """Looks up a name or stores a value in a name.
    The `ctx` of the node can be one of the following values:

    -   `store`: store a value in the name
    -   `load`: load that name
    -   `param`: like `store` but if the name was defined as function parameter.
    """

    fields = ("name", "ctx")
    name: str
    ctx: str

    def can_assign(self) -> bool:
        return self.name not in {"true", "false", "none", "True", "False", "None"}


class NSRef(Expr):
    """Reference to a namespace value assignment"""

    fields = ("name", "attr")
    name: str
    attr: str

    def can_assign(self) -> bool:
        # We don't need any special checks here; NSRef assignments have a
        # runtime check to ensure the target is a namespace object which will
        # have been checked already as it is created using a normal assignment
        # which goes through a `Name` node.
        return True


class Literal(Expr):
    """Baseclass for literals."""

    abstract = True


class Const(Literal):
    """All constant values.  The parser will return this node for simple
    constants such as ``42`` or ``"foo"`` but it can be used to store more
    complex values such as lists too.  Only constants with a safe
    representation (objects where ``eval(repr(x)) == x`` is true).
    """

    fields = ("value",)
    value: t.Any

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        return self.value

    @classmethod
    def from_untrusted(
        cls,
        value: t.Any,
        lineno: t.Optional[int] = None,
        environment: "t.Optional[Environment]" = None,
    ) -> "Const":
        """Return a const object if the value is representable as
        constant value in the generated code, otherwise it will raise
        an `Impossible` exception.
        """
        from .compiler import has_safe_repr

        if not has_safe_repr(value):
            raise Impossible()
        return cls(value, lineno=lineno, environment=environment)


class TemplateData(Literal):
    """A constant template string."""

    fields = ("data",)
    data: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> str:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        if eval_ctx.autoescape:
            return Markup(self.data)
        return self.data


class Tuple(Literal):
    """For loop unpacking and some other things like multiple arguments
    for subscripts.  Like for :class:`Name` `ctx` specifies if the tuple
    is used for loading the names or storing.
    """

    fields = ("items", "ctx")
    items: t.List[Expr]
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Tuple[t.Any, ...]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return tuple(x.as_const(eval_ctx) for x in self.items)

    def can_assign(self) -> bool:
        for item in self.items:
            if not item.can_assign():
                return False
        return True


class List(Literal):
    """Any list literal such as ``[1, 2, 3]``"""

    fields = ("items",)
    items: t.List[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.List[t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return [x.as_const(eval_ctx) for x in self.items]


class Dict(Literal):
    """Any dict literal such as ``{1: 2, 3: 4}``.  The items must be a list of
    :class:`Pair` nodes.
    """

    fields = ("items",)
    items: t.List["Pair"]

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Dict[t.Any, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return dict(x.as_const(eval_ctx) for x in self.items)


class Pair(Helper):
    """A key, value pair for dicts."""

    fields = ("key", "value")
    key: Expr
    value: Expr

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Tuple[t.Any, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key.as_const(eval_ctx), self.value.as_const(eval_ctx)


class Keyword(Helper):
    """A key, value pair for keyword arguments where key is a string."""

    fields = ("key", "value")
    key: str
    value: Expr

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Tuple[str, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key, self.value.as_const(eval_ctx)


class CondExpr(Expr):
    """A conditional expression (inline if expression).  (``{{
    foo if bar else baz }}``)
    """

    fields = ("test", "expr1", "expr2")
    test: Expr
    expr1: Expr
    expr2: t.Optional[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if self.test.as_const(eval_ctx):
            return self.expr1.as_const(eval_ctx)

        # if we evaluate to an undefined object, we better do that at runtime
        if self.expr2 is None:
            raise Impossible()

        return self.expr2.as_const(eval_ctx)


def args_as_const(
    node: t.Union["_FilterTestCommon", "Call"], eval_ctx: t.Optional[EvalContext]
) -> t.Tuple[t.List[t.Any], t.Dict[t.Any, t.Any]]:
    args = [x.as_const(eval_ctx) for x in node.args]
    kwargs = dict(x.as_const(eval_ctx) for x in node.kwargs)

    if node.dyn_args is not None:
        try:
            args.extend(node.dyn_args.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e

    if node.dyn_kwargs is not None:
        try:
            kwargs.update(node.dyn_kwargs.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e

    return args, kwargs


class _FilterTestCommon(Expr):
    fields = ("node", "name", "args", "kwargs", "dyn_args", "dyn_kwargs")
    node: Expr
    name: str
    args: t.List[Expr]
    kwargs: t.List[Pair]
    dyn_args: t.Optional[Expr]
    dyn_kwargs: t.Optional[Expr]
    abstract = True
    _is_filter = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        if eval_ctx.volatile:
            raise Impossible()

        if self._is_filter:
            env_map = eval_ctx.environment.filters
        else:
            env_map = eval_ctx.environment.tests

        func = env_map.get(self.name)
        pass_arg = _PassArg.from_obj(func)  # type: ignore

        if func is None or pass_arg is _PassArg.context:
            raise Impossible()

        if eval_ctx.environment.is_async and (
            getattr(func, "jinja_async_variant", False) is True
            or inspect.iscoroutinefunction(func)
        ):
            raise Impossible()

        args, kwargs = args_as_const(self, eval_ctx)
        args.insert(0, self.node.as_const(eval_ctx))

        if pass_arg is _PassArg.eval_context:
            args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            args.insert(0, eval_ctx.environment)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise Impossible() from e


class Filter(_FilterTestCommon):
    """Apply a filter to an expression. ``name`` is the name of the
    filter, the other fields are the same as :class:`Call`.

    If ``node`` is ``None``, the filter is being used in a filter block
    and is applied to the content of the block.
    """

    node: t.Optional[Expr]  # type: ignore

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.node is None:
            raise Impossible()

        return super().as_const(eval_ctx=eval_ctx)


class Test(_FilterTestCommon):
    """Apply a test to an expression. ``name`` is the name of the test,
    the other field are the same as :class:`Call`.

    .. versionchanged:: 3.0
        ``as_const`` shares the same logic for filters and tests. Tests
        check for volatile, async, and ``@pass_context`` etc.
        decorators.
    """

    _is_filter = False


class Call(Expr):
    """Calls an expression.  `args` is a list of arguments, `kwargs` a list
    of keyword arguments (list of :class:`Keyword` nodes), and `dyn_args`
    and `dyn_kwargs` has to be either `None` or a node that is used as
    node for dynamic positional (``*args``) or keyword (``**kwargs``)
    arguments.
    """

    fields = ("node", "args", "kwargs", "dyn_args", "dyn_kwargs")
    node: Expr
    args: t.List[Expr]
    kwargs: t.List[Keyword]
    dyn_args: t.Optional[Expr]
    dyn_kwargs: t.Optional[Expr]


class Getitem(Expr):
    """Get an attribute or item from an expression and prefer the item."""

    fields = ("node", "arg", "ctx")
    node: Expr
    arg: Expr
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.ctx != "load":
            raise Impossible()

        eval_ctx = get_eval_context(self, eval_ctx)

        try:
            return eval_ctx.environment.getitem(
                self.node.as_const(eval_ctx), self.arg.as_const(eval_ctx)
            )
        except Exception as e:
            raise Impossible() from e


class Getattr(Expr):
    """Get an attribute or item from an expression that is a ascii-only
    bytestring and prefer the attribute.
    """

    fields = ("node", "attr", "ctx")
    node: Expr
    attr: str
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.ctx != "load":
            raise Impossible()

        eval_ctx = get_eval_context(self, eval_ctx)

        try:
            return eval_ctx.environment.getattr(self.node.as_const(eval_ctx), self.attr)
        except Exception as e:
            raise Impossible() from e


class Slice(Expr):
    """Represents a slice object.  This must only be used as argument for
    :class:`Subscript`.
    """

    fields = ("start", "stop", "step")
    start: t.Optional[Expr]
    stop: t.Optional[Expr]
    step: t.Optional[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> slice:
        eval_ctx = get_eval_context(self, eval_ctx)

        def const(obj: t.Optional[Expr]) -> t.Optional[t.Any]:
            if obj is None:
                return None
            return obj.as_const(eval_ctx)

        return slice(const(self.start), const(self.stop), const(self.step))


class Concat(Expr):
    """Concatenates the list of expressions provided after converting
    them to strings.
    """

    fields = ("nodes",)
    nodes: t.List[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> str:
        eval_ctx = get_eval_context(self, eval_ctx)
        return "".join(str(x.as_const(eval_ctx)) for x in self.nodes)


class Compare(Expr):
    """Compares an expression with some other expressions.  `ops` must be a
    list of :class:`Operand`\\s.
    """

    fields = ("expr", "ops")
    expr: Expr
    ops: t.List["Operand"]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        result = value = self.expr.as_const(eval_ctx)

        try:
            for op in self.ops:
                new_value = op.expr.as_const(eval_ctx)
                result = _cmpop_to_func[op.op](value, new_value)

                if not result:
                    return False

                value = new_value
        except Exception as e:
            raise Impossible() from e

        return result


class Operand(Helper):
    """Holds an operator and an expression."""

    fields = ("op", "expr")
    op: str
    expr: Expr


class Mul(BinExpr):
    """Multiplies the left with the right node."""

    operator = "*"


class Div(BinExpr):
    """Divides the left by the right node."""

    operator = "/"


class FloorDiv(BinExpr):
    """Divides the left by the right node and converts the
    result into an integer by truncating.
    """

    operator = "//"


class Add(BinExpr):
    """Add the left to the right node."""

    operator = "+"


class Sub(BinExpr):
    """Subtract the right from the left node."""

    operator = "-"


class Mod(BinExpr):
    """Left modulo right."""

    operator = "%"


class Pow(BinExpr):
    """Left to the power of right."""

    operator = "**"


class And(BinExpr):
    """Short circuited AND."""

    operator = "and"

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) and self.right.as_const(eval_ctx)


class Or(BinExpr):
    """Short circuited OR."""

    operator = "or"

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) or self.right.as_const(eval_ctx)


class Not(UnaryExpr):
    """Negate the expression."""

    operator = "not"


class Neg(UnaryExpr):
    """Make the expression negative."""

    operator = "-"


class Pos(UnaryExpr):
    """Make the expression positive (noop for most expressions)"""

    operator = "+"


# Helpers for extensions


class EnvironmentAttribute(Expr):
    """Loads an attribute from the environment object.  This is useful for
    extensions that want to call a callback stored on the environment.
    """

    fields = ("name",)
    name: str


class ExtensionAttribute(Expr):
    """Returns the attribute of an extension bound to the environment.
    The identifier is the identifier of the :class:`Extension`.

    This node is usually constructed by calling the
    :meth:`~jinja2.ext.Extension.attr` method on an extension.
    """

    fields = ("identifier", "name")
    identifier: str
    name: str


class ImportedName(Expr):
    """If created with an import name the import name is returned on node
    access.  For example ``ImportedName('cgi.escape')`` returns the `escape`
    function from the cgi module on evaluation.  Imports are optimized by the
    compiler so there is no need to assign them to local variables.
    """

    fields = ("importname",)
    importname: str


class InternalName(Expr):
    """An internal name in the compiler.  You cannot create these nodes
    yourself but the parser provides a
    :meth:`~jinja2.parser.Parser.free_identifier` method that creates
    a new identifier for you.  This identifier is not available from the
    template and is not treated specially by the compiler.
    """

    fields = ("name",)
    name: str

    def __init__(self) -> None:
        raise TypeError(
            "Can't create internal names.  Use the "
            "`free_identifier` method on a parser."
        )


class MarkSafe(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`)."""

    fields = ("expr",)
    expr: Expr

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> Markup:
        eval_ctx = get_eval_context(self, eval_ctx)
        return Markup(self.expr.as_const(eval_ctx))


class MarkSafeIfAutoescape(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`) but
    only if autoescaping is active.

    .. versionadded:: 2.5
    """

    fields = ("expr",)
    expr: Expr

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Union[Markup, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        expr = self.expr.as_const(eval_ctx)
        if eval_ctx.autoescape:
            return Markup(expr)
        return expr


class ContextReference(Expr):
    """Returns the current template context.  It can be used like a
    :class:`Name` node, with a ``'load'`` ctx and will return the
    current :class:`~jinja2.runtime.Context` object.

    Here an example that assigns the current template name to a
    variable named `foo`::

        Assign(Name('foo', ctx='store'),
               Getattr(ContextReference(), 'name'))

    This is basically equivalent to using the
    :func:`~jinja2.pass_context` decorator when using the high-level
    API, which causes a reference to the context to be passed as the
    first argument to a function.
    """


class DerivedContextReference(Expr):
    """Return the current template context including locals. Behaves
    exactly like :class:`ContextReference`, but includes local
    variables, such as from a ``for`` loop.

    .. versionadded:: 2.11
    """


class Continue(Stmt):
    """Continue a loop."""


class Break(Stmt):
    """Break a loop."""


class Scope(Stmt):
    """An artificial scope."""

    fields = ("body",)
    body: t.List[Node]


class OverlayScope(Stmt):
    """An overlay scope for extensions.  This is a largely unoptimized scope
    that however can be used to introduce completely arbitrary variables into
    a sub scope from a dictionary or dictionary like object.  The `context`
    field has to evaluate to a dictionary object.

    Example usage::

        OverlayScope(context=self.call_method('get_context'),
                     body=[...])

    .. versionadded:: 2.10
    """

    fields = ("context", "body")
    context: Expr
    body: t.List[Node]


class EvalContextModifier(Stmt):
    """Modifies the eval context.  For each option that should be modified,
    a :class:`Keyword` has to be added to the :attr:`options` list.

    Example to change the `autoescape` setting::

        EvalContextModifier(options=[Keyword('autoescape', Const(True))])
    """

    fields = ("options",)
    options: t.List[Keyword]


class ScopedEvalContextModifier(EvalContextModifier):
    """Modifies the eval context and reverts it later.  Works exactly like
    :class:`EvalContextModifier` but will only modify the
    :class:`~jinja2.nodes.EvalContext` for nodes in the :attr:`body`.
    """

    fields = ("body",)
    body: t.List[Node]


# make sure nobody creates custom nodes
def _failing_new(*args: t.Any, **kwargs: t.Any) -> "te.NoReturn":
    raise TypeError("can't create custom node types")


NodeType.__new__ = staticmethod(_failing_new)  # type: ignore
del _failing_new
<|endoftext|>"""Built-in template filters used with the ``|`` operator."""

import math
import random
import re
import typing
import typing as t
from collections import abc
from inspect import getattr_static
from itertools import chain
from itertools import groupby

from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str

from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment
    from .nodes import EvalContext
    from .runtime import Context
    from .sandbox import SandboxedEnvironment  # noqa: F401

    class HasHTML(te.Protocol):
        def __html__(self) -> str:
            pass


F = t.TypeVar("F", bound=t.Callable[..., t.Any])
K = t.TypeVar("K")
V = t.TypeVar("V")


def ignore_case(value: V) -> V:
    """For use as a postprocessor for :func:`make_attrgetter`. Converts strings
    to lowercase and returns other types as-is."""
    if isinstance(value, str):
        return t.cast(V, value.lower())

    return value


def make_attrgetter(
    environment: "Environment",
    attribute: t.Optional[t.Union[str, int]],
    postprocess: t.Optional[t.Callable[[t.Any], t.Any]] = None,
    default: t.Optional[t.Any] = None,
) -> t.Callable[[t.Any], t.Any]:
    """Returns a callable that looks up the given attribute from a
    passed object with the rules of the environment.  Dots are allowed
    to access attributes of attributes.  Integer parts in paths are
    looked up as integers.
    """
    parts = _prepare_attribute_parts(attribute)

    def attrgetter(item: t.Any) -> t.Any:
        for part in parts:
            item = environment.getitem(item, part)

            if default is not None and isinstance(item, Undefined):
                item = default

        if postprocess is not None:
            item = postprocess(item)

        return item

    return attrgetter


def make_multi_attrgetter(
    environment: "Environment",
    attribute: t.Optional[t.Union[str, int]],
    postprocess: t.Optional[t.Callable[[t.Any], t.Any]] = None,
) -> t.Callable[[t.Any], t.List[t.Any]]:
    """Returns a callable that looks up the given comma separated
    attributes from a passed object with the rules of the environment.
    Dots are allowed to access attributes of each attribute.  Integer
    parts in paths are looked up as integers.

    The value returned by the returned callable is a list of extracted
    attribute values.

    Examples of attribute: "attr1,attr2", "attr1.inner1.0,attr2.inner2.0", etc.
    """
    if isinstance(attribute, str):
        split: t.Sequence[t.Union[str, int, None]] = attribute.split(",")
    else:
        split = [attribute]

    parts = [_prepare_attribute_parts(item) for item in split]

    def attrgetter(item: t.Any) -> t.List[t.Any]:
        items = [None] * len(parts)

        for i, attribute_part in enumerate(parts):
            item_i = item

            for part in attribute_part:
                item_i = environment.getitem(item_i, part)

            if postprocess is not None:
                item_i = postprocess(item_i)

            items[i] = item_i

        return items

    return attrgetter


def _prepare_attribute_parts(
    attr: t.Optional[t.Union[str, int]],
) -> t.List[t.Union[str, int]]:
    if attr is None:
        return []

    if isinstance(attr, str):
        return [int(x) if x.isdigit() else x for x in attr.split(".")]

    return [attr]


def do_forceescape(value: "t.Union[str, HasHTML]") -> Markup:
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()

    return escape(str(value))


def do_urlencode(
    value: t.Union[str, t.Mapping[str, t.Any], t.Iterable[t.Tuple[str, t.Any]]],
) -> str:
    """Quote data for use in a URL path or query using UTF-8.

    Basic wrapper around :func:`urllib.parse.quote` when given a
    string, or :func:`urllib.parse.urlencode` for a dict or iterable.

    :param value: Data to quote. A string will be quoted directly. A
        dict or iterable of ``(key, value)`` pairs will be joined as a
        query string.

    When given a string, "/" is not quoted. HTTP servers treat "/" and
    "%2F" equivalently in paths. If you need quoted slashes, use the
    ``|replace("/", "%2F")`` filter.

    .. versionadded:: 2.7
    """
    if isinstance(value, str) or not isinstance(value, abc.Iterable):
        return url_quote(value)

    if isinstance(value, dict):
        items: t.Iterable[t.Tuple[str, t.Any]] = value.items()
    else:
        items = value  # type: ignore

    return "&".join(
        f"{url_quote(k, for_qs=True)}={url_quote(v, for_qs=True)}" for k, v in items
    )


@pass_eval_context
def do_replace(
    eval_ctx: "EvalContext", s: str, old: str, new: str, count: t.Optional[int] = None
) -> str:
    """Return a copy of the value with all occurrences of a substring
    replaced with a new one. The first argument is the substring
    that should be replaced, the second is the replacement string.
    If the optional third argument ``count`` is given, only the first
    ``count`` occurrences are replaced:

    .. sourcecode:: jinja

        {{ "Hello World"|replace("Hello", "Goodbye") }}
            -> Goodbye World

        {{ "aaaaargh"|replace("a", "d'oh, ", 2) }}
            -> d'oh, d'oh, aaargh
    """
    if count is None:
        count = -1

    if not eval_ctx.autoescape:
        return str(s).replace(str(old), str(new), count)

    if (
        hasattr(old, "__html__")
        or hasattr(new, "__html__")
        and not hasattr(s, "__html__")
    ):
        s = escape(s)
    else:
        s = soft_str(s)

    return s.replace(soft_str(old), soft_str(new), count)


def do_upper(s: str) -> str:
    """Convert a value to uppercase."""
    return soft_str(s).upper()


def do_lower(s: str) -> str:
    """Convert a value to lowercase."""
    return soft_str(s).lower()


def do_items(value: t.Union[t.Mapping[K, V], Undefined]) -> t.Iterator[t.Tuple[K, V]]:
    """Return an iterator over the ``(key, value)`` items of a mapping.

    ``x|items`` is the same as ``x.items()``, except if ``x`` is
    undefined an empty iterator is returned.

    This filter is useful if you expect the template to be rendered with
    an implementation of Jinja in another programming language that does
    not have a ``.items()`` method on its mapping type.

    .. code-block:: html+jinja

        <dl>
        {% for key, value in my_dict|items %}
            <dt>{{ key }}
            <dd>{{ value }}
        {% endfor %}
        </dl>

    .. versionadded:: 3.1
    """
    if isinstance(value, Undefined):
        return

    if not isinstance(value, abc.Mapping):
        raise TypeError("Can only get item pairs from a mapping.")

    yield from value.items()


# Check for characters that would move the parser state from key to value.
# https://html.spec.whatwg.org/#attribute-name-state
_attr_key_re = re.compile(r"[\s/>=]", flags=re.ASCII)


@pass_eval_context
def do_xmlattr(
    eval_ctx: "EvalContext", d: t.Mapping[str, t.Any], autospace: bool = True
) -> str:
    """Create an SGML/XML attribute string based on the items in a dict.

    **Values** that are neither ``none`` nor ``undefined`` are automatically
    escaped, safely allowing untrusted user input.

    User input should not be used as **keys** to this filter. If any key
    contains a space, ``/`` solidus, ``>`` greater-than sign, or ``=`` equals
    sign, this fails with a ``ValueError``. Regardless of this, user input
    should never be used as keys to this filter, or must be separately validated
    first.

    .. sourcecode:: html+jinja

        <ul{{ {'class': 'my_list', 'missing': none,
                'id': 'list-%d'|format(variable)}|xmlattr }}>
        ...
        </ul>

    Results in something like this:

    .. sourcecode:: html

        <ul class="my_list" id="list-42">
        ...
        </ul>

    As you can see it automatically prepends a space in front of the item
    if the filter returned something unless the second parameter is false.

    .. versionchanged:: 3.1.4
        Keys with ``/`` solidus, ``>`` greater-than sign, or ``=`` equals sign
        are not allowed.

    .. versionchanged:: 3.1.3
        Keys with spaces are not allowed.
    """
    items = []

    for key, value in d.items():
        if value is None or isinstance(value, Undefined):
            continue

        if _attr_key_re.search(key) is not None:
            raise ValueError(f"Invalid character in attribute name: {key!r}")

        items.append(f'{escape(key)}="{escape(value)}"')

    rv = " ".join(items)

    if autospace and rv:
        rv = " " + rv

    if eval_ctx.autoescape:
        rv = Markup(rv)

    return rv


def do_capitalize(s: str) -> str:
    """Capitalize a value. The first character will be uppercase, all others
    lowercase.
    """
    return soft_str(s).capitalize()


_word_beginning_split_re = re.compile(r"([-\s({\[<]+)")


def do_title(s: str) -> str:
    """Return a titlecased version of the value. I.e. words will start with
    uppercase letters, all remaining characters are lowercase.
    """
    return "".join(
        [
            item[0].upper() + item[1:].lower()
            for item in _word_beginning_split_re.split(soft_str(s))
            if item
        ]
    )


def do_dictsort(
    value: t.Mapping[K, V],
    case_sensitive: bool = False,
    by: 'te.Literal["key", "value"]' = "key",
    reverse: bool = False,
) -> t.List[t.Tuple[K, V]]:
    """Sort a dict and yield (key, value) pairs. Python dicts may not
    be in the order you want to display them in, so sort them first.

    .. sourcecode:: jinja

        {% for key, value in mydict|dictsort %}
            sort the dict by key, case insensitive

        {% for key, value in mydict|dictsort(reverse=true) %}
            sort the dict by key, case insensitive, reverse order

        {% for key, value in mydict|dictsort(true) %}
            sort the dict by key, case sensitive

        {% for key, value in mydict|dictsort(false, 'value') %}
            sort the dict by value, case insensitive
    """
    if by == "key":
        pos = 0
    elif by == "value":
        pos = 1
    else:
        raise FilterArgumentError('You can only sort by either "key" or "value"')

    def sort_func(item: t.Tuple[t.Any, t.Any]) -> t.Any:
        value = item[pos]

        if not case_sensitive:
            value = ignore_case(value)

        return value

    return sorted(value.items(), key=sort_func, reverse=reverse)


@pass_environment
def do_sort(
    environment: "Environment",
    value: "t.Iterable[V]",
    reverse: bool = False,
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.List[V]":
    """Sort an iterable using Python's :func:`sorted`.

    .. sourcecode:: jinja

        {% for city in cities|sort %}
            ...
        {% endfor %}

    :param reverse: Sort descending instead of ascending.
    :param case_sensitive: When sorting strings, sort upper and lower
        case separately.
    :param attribute: When sorting objects or dicts, an attribute or
        key to sort by. Can use dot notation like ``"address.city"``.
        Can be a list of attributes like ``"age,name"``.

    The sort is stable, it does not change the relative order of
    elements that compare equal. This makes it is possible to chain
    sorts on different attributes and ordering.

    .. sourcecode:: jinja

        {% for user in users|sort(attribute="name")
            |sort(reverse=true, attribute="age") %}
            ...
        {% endfor %}

    As a shortcut to chaining when the direction is the same for all
    attributes, pass a comma separate list of attributes.

    .. sourcecode:: jinja

        {% for user in users|sort(attribute="age,name") %}
            ...
        {% endfor %}

    .. versionchanged:: 2.11.0
        The ``attribute`` parameter can be a comma separated list of
        attributes, e.g. ``"age,name"``.

    .. versionchanged:: 2.6
       The ``attribute`` parameter was added.
    """
    key_func = make_multi_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return sorted(value, key=key_func, reverse=reverse)


@pass_environment
def sync_do_unique(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Iterator[V]":
    """Returns a list of unique items from the given iterable.

    .. sourcecode:: jinja

        {{ ['foo', 'bar', 'foobar', 'FooBar']|unique|list }}
            -> ['foo', 'bar', 'foobar']

    The unique items are yielded in the same order as their first occurrence in
    the iterable passed to the filter.

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Filter objects with unique values for this attribute.
    """
    getter = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    seen = set()

    for item in value:
        key = getter(item)

        if key not in seen:
            seen.add(key)
            yield item


@async_variant(sync_do_unique)  # type: ignore
async def do_unique(
    environment: "Environment",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Iterator[V]":
    return sync_do_unique(
        environment, await auto_to_list(value), case_sensitive, attribute
    )


def _min_or_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    func: "t.Callable[..., V]",
    case_sensitive: bool,
    attribute: t.Optional[t.Union[str, int]],
) -> "t.Union[V, Undefined]":
    it = iter(value)

    try:
        first = next(it)
    except StopIteration:
        return environment.undefined("No aggregated item, sequence was empty.")

    key_func = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return func(chain([first], it), key=key_func)


@pass_environment
def do_min(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Union[V, Undefined]":
    """Return the smallest item from the sequence.

    .. sourcecode:: jinja

        {{ [1, 2, 3]|min }}
            -> 1

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Get the object with the min value of this attribute.
    """
    return _min_or_max(environment, value, min, case_sensitive, attribute)


@pass_environment
def do_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Union[V, Undefined]":
    """Return the largest item from the sequence.

    .. sourcecode:: jinja

        {{ [1, 2, 3]|max }}
            -> 3

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Get the object with the max value of this attribute.
    """
    return _min_or_max(environment, value, max, case_sensitive, attribute)


def do_default(
    value: V,
    default_value: V = "",  # type: ignore
    boolean: bool = False,
) -> V:
    """If the value is undefined it will return the passed default value,
    otherwise the value of the variable:

    .. sourcecode:: jinja

        {{ my_variable|default('my_variable is not defined') }}

    This will output the value of ``my_variable`` if the variable was
    defined, otherwise ``'my_variable is not defined'``. If you want
    to use default with variables that evaluate to false you have to
    set the second parameter to `true`:

    .. sourcecode:: jinja

        {{ ''|default('the string was empty', true) }}

    .. versionchanged:: 2.11
       It's now possible to configure the :class:`~jinja2.Environment` with
       :class:`~jinja2.ChainableUndefined` to make the `default` filter work
       on nested elements and attributes that may contain undefined values
       in the chain without getting an :exc:`~jinja2.UndefinedError`.
    """
    if isinstance(value, Undefined) or (boolean and not value):
        return default_value

    return value


@pass_eval_context
def sync_do_join(
    eval_ctx: "EvalContext",
    value: t.Iterable[t.Any],
    d: str = "",
    attribute: t.Optional[t.Union[str, int]] = None,
) -> str:
    """Return a string which is the concatenation of the strings in the
    sequence. The separator between elements is an empty string per
    default, you can define it with the optional parameter:

    .. sourcecode:: jinja

        {{ [1, 2, 3]|join('|') }}
            -> 1|2|3

        {{ [1, 2, 3]|join }}
            -> 123

    It is also possible to join certain attributes of an object:

    .. sourcecode:: jinja

        {{ users|join(', ', attribute='username') }}

    .. versionadded:: 2.6
       The `attribute` parameter was added.
    """
    if attribute is not None:
        value = map(make_attrgetter(eval_ctx.environment, attribute), value)

    # no automatic escaping?  joining is a lot easier then
    if not eval_ctx.autoescape:
        return str(d).join(map(str, value))

    # if the delimiter doesn't have an html representation we check
    # if any of the items has.  If yes we do a coercion to Markup
    if not hasattr(d, "__html__"):
        value = list(value)
        do_escape = False

        for idx, item in enumerate(value):
            if hasattr(item, "__html__"):
                do_escape = True
            else:
                value[idx] = str(item)

        if do_escape:
            d = escape(d)
        else:
            d = str(d)

        return d.join(value)

    # no html involved, to normal joining
    return soft_str(d).join(map(soft_str, value))


@async_variant(sync_do_join)  # type: ignore
async def do_join(
    eval_ctx: "EvalContext",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    d: str = "",
    attribute: t.Optional[t.Union[str, int]] = None,
) -> str:
    return sync_do_join(eval_ctx, await auto_to_list(value), d, attribute)


def do_center(value: str, width: int = 80) -> str:
    """Centers the value in a field of a given width."""
    return soft_str(value).center(width)


@pass_environment
def sync_do_first(
    environment: "Environment", seq: "t.Iterable[V]"
) -> "t.Union[V, Undefined]":
    """Return the first item of a sequence."""
    try:
        return next(iter(seq))
    except StopIteration:
        return environment.undefined("No first item, sequence was empty.")


@async_variant(sync_do_first)  # type: ignore
async def do_first(
    environment: "Environment", seq: "t.Union[t.AsyncIterable[V], t.Iterable[V]]"
) -> "t.Union[V, Undefined]":
    try:
        return await auto_aiter(seq).__anext__()
    except StopAsyncIteration:
        return environment.undefined("No first item, sequence was empty.")


@pass_environment
def do_last(
    environment: "Environment", seq: "t.Reversible[V]"
) -> "t.Union[V, Undefined]":
    """Return the last item of a sequence.

    Note: Does not work with generators. You may want to explicitly
    convert it to a list:

    .. sourcecode:: jinja

        {{ data | selectattr('name', '==', 'Jinja') | list | last }}
    """
    try:
        return next(iter(reversed(seq)))
    except StopIteration:
        return environment.undefined("No last item, sequence was empty.")


# No async do_last, it may not be safe in async mode.


@pass_context
def do_random(context: "Context", seq: "t.Sequence[V]") -> "t.Union[V, Undefined]":
    """Return a random item from the sequence."""
    try:
        return random.choice(seq)
    except IndexError:
        return context.environment.undefined("No random item, sequence was empty.")


def do_filesizeformat(value: t.Union[str, float, int], binary: bool = False) -> str:
    """Format the value like a 'human-readable' file size (i.e. 13 kB,
    4.1 MB, 102 Bytes, etc).  Per default decimal prefixes are used (Mega,
    Giga, etc.), if the second parameter is set to `True` the binary
    prefixes are used (Mebi, Gibi).
    """
    bytes = float(value)
    base = 1024 if binary else 1000
    prefixes = [
        ("KiB" if binary else "kB"),
        ("MiB" if binary else "MB"),
        ("GiB" if binary else "GB"),
        ("TiB" if binary else "TB"),
        ("PiB" if binary else "PB"),
        ("EiB" if binary else "EB"),
        ("ZiB" if binary else "ZB"),
        ("YiB" if binary else "YB"),
    ]

    if bytes == 1:
        return "1 Byte"
    elif bytes < base:
        return f"{int(bytes)} Bytes"
    else:
        for i, prefix in enumerate(prefixes):
            unit = base ** (i + 2)

            if bytes < unit:
                return f"{base * bytes / unit:.1f} {prefix}"

        return f"{base * bytes / unit:.1f} {prefix}"


def do_pprint(value: t.Any) -> str:
    """Pretty print a variable. Useful for debugging."""
    return pformat(value)


_uri_scheme_re = re.compile(r"^([\w.+-]{2,}:(/){0,2})$")


@pass_eval_context
def do_urlize(
    eval_ctx: "EvalContext",
    value: str,
    trim_url_limit: t.Optional[int] = None,
    nofollow: bool = False,
    target: t.Optional[str] = None,
    rel: t.Optional[str] = None,
    extra_schemes: t.Optional[t.Iterable[str]] = None,
) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param value: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param nofollow: Add the ``rel=nofollow`` attribute to links.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior. Defaults to
        ``env.policies["urlize.extra_schemes"]``, which defaults to no
        extra schemes.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.

    .. versionchanged:: 2.8
       The ``target`` parameter was added.
    """
    policies = eval_ctx.environment.policies
    rel_parts = set((rel or "").split())

    if nofollow:
        rel_parts.add("nofollow")

    rel_parts.update((policies["urlize.rel"] or "").split())
    rel = " ".join(sorted(rel_parts)) or None

    if target is None:
        target = policies["urlize.target"]

    if extra_schemes is None:
        extra_schemes = policies["urlize.extra_schemes"] or ()

    for scheme in extra_schemes:
        if _uri_scheme_re.fullmatch(scheme) is None:
            raise FilterArgumentError(f"{scheme!r} is not a valid URI scheme prefix.")

    rv = urlize(
        value,
        trim_url_limit=trim_url_limit,
        rel=rel,
        target=target,
        extra_schemes=extra_schemes,
    )

    if eval_ctx.autoescape:
        rv = Markup(rv)

    return rv


def do_indent(
    s: str, width: t.Union[int, str] = 4, first: bool = False, blank: bool = False
) -> str:
    """Return a copy of the string with each line indented by 4 spaces. The
    first line and blank lines are not indented by default.

    :param width: Number of spaces, or a string, to indent by.
    :param first: Don't skip indenting the first line.
    :param blank: Don't skip indenting empty lines.

    .. versionchanged:: 3.0
        ``width`` can be a string.

    .. versionchanged:: 2.10
        Blank lines are not indented by default.

        Rename the ``indentfirst`` argument to ``first``.
    """
    if isinstance(width, str):
        indention = width
    else:
        indention = " " * width

    newline = "\n"

    if isinstance(s, Markup):
        indention = Markup(indention)
        newline = Markup(newline)

    s += newline  # this quirk is necessary for splitlines method

    if blank:
        rv = (newline + indention).join(s.splitlines())
    else:
        lines = s.splitlines()
        rv = lines.pop(0)

        if lines:
            rv += newline + newline.join(
                indention + line if line else line for line in lines
            )

    if first:
        rv = indention + rv

    return rv


@pass_environment
def do_truncate(
    env: "Environment",
    s: str,
    length: int = 255,
    killwords: bool = False,
    end: str = "...",
    leeway: t.Optional[int] = None,
) -> str:
    """Return a truncated copy of the string. The length is specified
    with the first parameter which defaults to ``255``. If the second
    parameter is ``true`` the filter will cut the text at length. Otherwise
    it will discard the last word. If the text was in fact
    truncated it will append an ellipsis sign (``"..."``). If you want a
    different ellipsis sign than ``"..."`` you can specify it using the
    third parameter. Strings that only exceed the length by the tolerance
    margin given in the fourth parameter will not be truncated.

    .. sourcecode:: jinja

        {{ "foo bar baz qux"|truncate(9) }}
            -> "foo..."
        {{ "foo bar baz qux"|truncate(9, True) }}
            -> "foo ba..."
        {{ "foo bar baz qux"|truncate(11) }}
            -> "foo bar baz qux"
        {{ "foo bar baz qux"|truncate(11, False, '...', 0) }}
            -> "foo bar..."

    The default leeway on newer Jinja versions is 5 and was 0 before but
    can be reconfigured globally.
    """
    if leeway is None:
        leeway = env.policies["truncate.leeway"]

    assert length >= len(end), f"expected length >= {len(end)}, got {length}"
    assert leeway >= 0, f"expected leeway >= 0, got {leeway}"

    if len(s) <= length + leeway:
        return s

    if killwords:
        return s[: length - len(end)] + end

    result = s[: length - len(end)].rsplit(" ", 1)[0]
    return result + end


@pass_environment
def do_wordwrap(
    environment: "Environment",
    s: str,
    width: int = 79,
    break_long_words: bool = True,
    wrapstring: t.Optional[str] = None,
    break_on_hyphens: bool = True,
) -> str:
    """Wrap a string to the given width. Existing newlines are treated
    as paragraphs to be wrapped separately.

    :param s: Original text to wrap.
    :param width: Maximum length of wrapped lines.
    :param break_long_words: If a word is longer than ``width``, break
        it across lines.
    :param break_on_hyphens: If a word contains hyphens, it may be split
        across lines.
    :param wrapstring: String to join each wrapped line. Defaults to
        :attr:`Environment.newline_sequence`.

    .. versionchanged:: 2.11
        Existing newlines are treated as paragraphs wrapped separately.

    .. versionchanged:: 2.11
        Added the ``break_on_hyphens`` parameter.

    .. versionchanged:: 2.7
        Added the ``wrapstring`` parameter.
    """
    import textwrap

    if wrapstring is None:
        wrapstring = environment.newline_sequence

    # textwrap.wrap doesn't consider existing newlines when wrapping.
    # If the string has a newline before width, wrap will still insert
    # a newline at width, resulting in a short line. Instead, split and
    # wrap each paragraph individually.
    return wrapstring.join(
        [
            wrapstring.join(
                textwrap.wrap(
                    line,
                    width=width,
                    expand_tabs=False,
                    replace_whitespace=False,
                    break_long_words=break_long_words,
                    break_on_hyphens=break_on_hyphens,
                )
            )
            for line in s.splitlines()
        ]
    )


_word_re = re.compile(r"\w+")


def do_wordcount(s: str) -> int:
    """Count the words in that string."""
    return len(_word_re.findall(soft_str(s)))


def do_int(value: t.Any, default: int = 0, base: int = 10) -> int:
    """Convert the value into an integer. If the
    conversion doesn't work it will return ``0``. You can
    override this default using the first parameter. You
    can also override the default base (10) in the second
    parameter, which handles input with prefixes such as
    0b, 0o and 0x for bases 2, 8 and 16 respectively.
    The base is ignored for decimal numbers and non-string values.
    """
    try:
        if isinstance(value, str):
            return int(value, base)

        return int(value)
    except (TypeError, ValueError):
        # this quirk is necessary so that "42.23"|int gives 42.
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def do_float(value: t.Any, default: float = 0.0) -> float:
    """Convert the value into a floating point number. If the
    conversion doesn't work it will return ``0.0``. You can
    override this default using the first parameter.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def do_format(value: str, *args: t.Any, **kwargs: t.Any) -> str:
    """Apply the given values to a `printf-style`_ format string, like
    ``string % values``.

    .. sourcecode:: jinja

        {{ "%s, %s!"|format(greeting, name) }}
        Hello, World!

    In most cases it should be more convenient and efficient to use the
    ``%`` operator or :meth:`str.format`.

    .. code-block:: text

        {{ "%s, %s!" % (greeting, name) }}
        {{ "{}, {}!".format(greeting, name) }}

    .. _printf-style: https://docs.python.org/library/stdtypes.html
        #printf-style-string-formatting
    """
    if args and kwargs:
        raise FilterArgumentError(
            "can't handle positional and keyword arguments at the same time"
        )

    return soft_str(value) % (kwargs or args)


def do_trim(value: str, chars: t.Optional[str] = None) -> str:
    """Strip leading and trailing characters, by default whitespace."""
    return soft_str(value).strip(chars)


def do_striptags(value: "t.Union[str, HasHTML]") -> str:
    """Strip SGML/XML tags and replace adjacent whitespace by one space."""
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()

    return Markup(str(value)).striptags()


def sync_do_slice(
    value: "t.Collection[V]", slices: int, fill_with: "t.Optional[V]" = None
) -> "t.Iterator[t.List[V]]":
    """Slice an iterator and return a list of lists containing
    those items. Useful if you want to create a div containing
    three ul tags that represent columns:

    .. sourcecode:: html+jinja

        <div class="columnwrapper">
          {%- for column in items|slice(3) %}
            <ul class="column-{{ loop.index }}">
            {%- for item in column %}
              <li>{{ item }}</li>
            {%- endfor %}
            </ul>
          {%- endfor %}
        </div>

    If you pass it a second argument it's used to fill missing
    values on the last iteration.
    """
    seq = list(value)
    length = len(seq)
    items_per_slice = length // slices
    slices_with_extra = length % slices
    offset = 0

    for slice_number in range(slices):
        start = offset + slice_number * items_per_slice

        if slice_number < slices_with_extra:
            offset += 1

        end = offset + (slice_number + 1) * items_per_slice
        tmp = seq[start:end]

        if fill_with is not None and slice_number >= slices_with_extra:
            tmp.append(fill_with)

        yield tmp


@async_variant(sync_do_slice)  # type: ignore
async def do_slice(
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    slices: int,
    fill_with: t.Optional[t.Any] = None,
) -> "t.Iterator[t.List[V]]":
    return sync_do_slice(await auto_to_list(value), slices, fill_with)


def do_batch(
    value: "t.Iterable[V]", linecount: int, fill_with: "t.Optional[V]" = None
) -> "t.Iterator[t.List[V]]":
    """
    A filter that batches items. It works pretty much like `slice`
    just the other way round. It returns a list of lists with the
    given number of items. If you provide a second parameter this
    is used to fill up missing items. See this example:

    .. sourcecode:: html+jinja

        <table>
        {%- for row in items|batch(3, '&nbsp;') %}
          <tr>
          {%- for column in row %}
            <td>{{ column }}</td>
          {%- endfor %}
          </tr>
        {%- endfor %}
        </table>
    """
    tmp: t.List[V] = []

    for item in value:
        if len(tmp) == linecount:
            yield tmp
            tmp = []

        tmp.append(item)

    if tmp:
        if fill_with is not None and len(tmp) < linecount:
            tmp += [fill_with] * (linecount - len(tmp))

        yield tmp


def do_round(
    value: float,
    precision: int = 0,
    method: 'te.Literal["common", "ceil", "floor"]' = "common",
) -> float:
    """Round the number to a given precision. The first
    parameter specifies the precision (default is ``0``), the
    second the rounding method:

    - ``'common'`` rounds either up or down
    - ``'ceil'`` always rounds up
    - ``'floor'`` always rounds down

    If you don't specify a method ``'common'`` is used.

    .. sourcecode:: jinja

        {{ 42.55|round }}
            -> 43.0
        {{ 42.55|round(1, 'floor') }}
            -> 42.5

    Note that even if rounded to 0 precision, a float is returned.  If
    you need a real integer, pipe it through `int`:

    .. sourcecode:: jinja

        {{ 42.55|round|int }}
            -> 43
    """
    if method not in {"common", "ceil", "floor"}:
        raise FilterArgumentError("method must be common, ceil or floor")

    if method == "common":
        return round(value, precision)

    func = getattr(math, method)
    return t.cast(float, func(value * (10**precision)) / (10**precision))


class _GroupTuple(t.NamedTuple):
    grouper: t.Any
    list: t.List[t.Any]

    # Use the regular tuple repr to hide this subclass if users print
    # out the value during debugging.
    def __repr__(self) -> str:
        return tuple.__repr__(self)

    def __str__(self) -> str:
        return tuple.__str__(self)


@pass_environment
def sync_do_groupby(
    environment: "Environment",
    value: "t.Iterable[V]",
    attribute: t.Union[str, int],
    default: t.Optional[t.Any] = None,
    case_sensitive: bool = False,
) -> "t.List[_GroupTuple]":
    """Group a sequence of objects by an attribute using Python's
    :func:`itertools.groupby`. The attribute can use dot notation for
    nested access, like ``"address.city"``. Unlike Python's ``groupby``,
    the values are sorted first so only one group is returned for each
    unique value.

    For example, a list of ``User`` objects with a ``city`` attribute
    can be rendered in groups. In this example, ``grouper`` refers to
    the ``city`` value of the group.

    .. sourcecode:: html+jinja

        <ul>{% for city, items in users|groupby("city") %}
          <li>{{ city }}
            <ul>{% for user in items %}
              <li>{{ user.name }}
            {% endfor %}</ul>
          </li>
        {% endfor %}</ul>

    ``groupby`` yields namedtuples of ``(grouper, list)``, which
    can be used instead of the tuple unpacking above. ``grouper`` is the
    value of the attribute, and ``list`` is the items with that value.

    .. sourcecode:: html+jinja

        <ul>{% for group in users|groupby("city") %}
          <li>{{ group.grouper }}: {{ group.list|join(", ") }}
        {% endfor %}</ul>

    You can specify a ``default`` value to use if an object in the list
    does not have the given attribute.

    .. sourcecode:: jinja

        <ul>{% for city, items in users|groupby("city", default="NY") %}
          <li>{{ city }}: {{ items|map(attribute="name")|join(", ") }}</li>
        {% endfor %}</ul>

    Like the :func:`~jinja-filters.sort` filter, sorting and grouping is
    case-insensitive by default. The ``key`` for each group will have
    the case of the first item in that group of values. For example, if
    a list of users has cities ``["CA", "NY", "ca"]``, the "CA" group
    will have two values. This can be disabled by passing
    ``case_sensitive=True``.

    .. versionchanged:: 3.1
        Added the ``case_sensitive`` parameter. Sorting and grouping is
        case-insensitive by default, matching other filters that do
        comparisons.

    .. versionchanged:: 3.0
        Added the ``default`` parameter.

    .. versionchanged:: 2.6
        The attribute supports dot notation for nested access.
    """
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    out = [
        _GroupTuple(key, list(values))
        for key, values in groupby(sorted(value, key=expr), expr)
    ]

    if not case_sensitive:
        # Return the real key from the first value instead of the lowercase key.
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]

    return out


@async_variant(sync_do_groupby)  # type: ignore
async def do_groupby(
    environment: "Environment",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    attribute: t.Union[str, int],
    default: t.Optional[t.Any] = None,
    case_sensitive: bool = False,
) -> "t.List[_GroupTuple]":
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    out = [
        _GroupTuple(key, await auto_to_list(values))
        for key, values in groupby(sorted(await auto_to_list(value), key=expr), expr)
    ]

    if not case_sensitive:
        # Return the real key from the first value instead of the lowercase key.
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]

    return out


@pass_environment
def sync_do_sum(
    environment: "Environment",
    iterable: "t.Iterable[V]",
    attribute: t.Optional[t.Union[str, int]] = None,
    start: V = 0,  # type: ignore
) -> V:
    """Returns the sum of a sequence of numbers plus the value of parameter
    'start' (which defaults to 0).  When the sequence is empty it returns
    start.

    It is also possible to sum up only certain attributes:

    .. sourcecode:: jinja

        Total: {{ items|sum(attribute='price') }}

    .. versionchanged:: 2.6
       The ``attribute`` parameter was added to allow summing up over
       attributes.  Also the ``start`` parameter was moved on to the right.
    """
    if attribute is not None:
        iterable = map(make_attrgetter(environment, attribute), iterable)

    return sum(iterable, start)  # type: ignore[no-any-return, call-overload]


@async_variant(sync_do_sum)  # type: ignore
async def do_sum(
    environment: "Environment",
    iterable: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    attribute: t.Optional[t.Union[str, int]] = None,
    start: V = 0,  # type: ignore
) -> V:
    rv = start

    if attribute is not None:
        func = make_attrgetter(environment, attribute)
    else:

        def func(x: V) -> V:
            return x

    async for item in auto_aiter(iterable):
        rv += func(item)

    return rv


def sync_do_list(value: "t.Iterable[V]") -> "t.List[V]":
    """Convert the value into a list.  If it was a string the returned list
    will be a list of characters.
    """
    return list(value)


@async_variant(sync_do_list)  # type: ignore
async def do_list(value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]") -> "t.List[V]":
    return await auto_to_list(value)


def do_mark_safe(value: str) -> Markup:
    """Mark the value as safe which means that in an environment with automatic
    escaping enabled this variable will not be escaped.
    """
    return Markup(value)


def do_mark_unsafe(value: str) -> str:
    """Mark a value as unsafe.  This is the reverse operation for :func:`safe`."""
    return str(value)


@typing.overload
def do_reverse(value: str) -> str: ...


@typing.overload
def do_reverse(value: "t.Iterable[V]") -> "t.Iterable[V]": ...


def do_reverse(value: t.Union[str, t.Iterable[V]]) -> t.Union[str, t.Iterable[V]]:
    """Reverse the object or return an iterator that iterates over it the other
    way round.
    """
    if isinstance(value, str):
        return value[::-1]

    try:
        return reversed(value)  # type: ignore
    except TypeError:
        try:
            rv = list(value)
            rv.reverse()
            return rv
        except TypeError as e:
            raise FilterArgumentError("argument must be iterable") from e


@pass_environment
def do_attr(
    environment: "Environment", obj: t.Any, name: str
) -> t.Union[Undefined, t.Any]:
    """Get an attribute of an object. ``foo|attr("bar")`` works like
    ``foo.bar``, but returns undefined instead of falling back to ``foo["bar"]``
    if the attribute doesn't exist.

    See :ref:`Notes on subscriptions <notes-on-subscriptions>` for more details.
    """
    # Environment.getattr will fall back to obj[name] if obj.name doesn't exist.
    # But we want to call env.getattr to get behavior such as sandboxing.
    # Determine if the attr exists first, so we know the fallback won't trigger.
    try:
        # This avoids executing properties/descriptors, but misses __getattr__
        # and __getattribute__ dynamic attrs.
        getattr_static(obj, name)
    except AttributeError:
        # This finds dynamic attrs, and we know it's not a descriptor at this point.
        if not hasattr(obj, name):
            return environment.undefined(obj=obj, name=name)

    return environment.getattr(obj, name)


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    *,
    attribute: str = ...,
    default: t.Optional[t.Any] = None,
) -> t.Iterable[t.Any]: ...


@pass_context
def sync_do_map(
    context: "Context", value: t.Iterable[t.Any], *args: t.Any, **kwargs: t.Any
) -> t.Iterable[t.Any]:
    """Applies a filter on a sequence of objects or looks up an attribute.
    This is useful when dealing with lists of objects but you are really
    only interested in a certain value of it.

    The basic usage is mapping on an attribute.  Imagine you have a list
    of users but you are only interested in a list of usernames:

    .. sourcecode:: jinja

        Users on this page: {{ users|map(attribute='username')|join(', ') }}

    You can specify a ``default`` value to use if an object in the list
    does not have the given attribute.

    .. sourcecode:: jinja

        {{ users|map(attribute="username", default="Anonymous")|join(", ") }}

    Alternatively you can let it invoke a filter by passing the name of the
    filter and the arguments afterwards.  A good example would be applying a
    text conversion filter on a sequence:

    .. sourcecode:: jinja

        Users on this page: {{ titles|map('lower')|join(', ') }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (u.username for u in users)
        (getattr(u, "username", "Anonymous") for u in users)
        (do_lower(x) for x in titles)

    .. versionchanged:: 2.11.0
        Added the ``default`` parameter.

    .. versionadded:: 2.7
    """
    if value:
        func = prepare_map(context, args, kwargs)

        for item in value:
            yield func(item)


@typing.overload
def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    *,
    attribute: str = ...,
    default: t.Optional[t.Any] = None,
) -> t.Iterable[t.Any]: ...


@async_variant(sync_do_map)  # type: ignore
async def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    *args: t.Any,
    **kwargs: t.Any,
) -> t.AsyncIterable[t.Any]:
    if value:
        func = prepare_map(context, args, kwargs)

        async for item in auto_aiter(value):
            yield await auto_await(func(item))


@pass_context
def sync_do_select(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to each object,
    and only selecting the objects with the test succeeding.

    If no test is specified, each object will be evaluated as a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ numbers|select("odd") }}
        {{ numbers|select("odd") }}
        {{ numbers|select("divisibleby", 3) }}
        {{ numbers|select("lessthan", 42) }}
        {{ strings|select("equalto", "mystring") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (n for n in numbers if test_odd(n))
        (n for n in numbers if test_divisibleby(n, 3))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: x, False)


@async_variant(sync_do_select)  # type: ignore
async def do_select(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, False)


@pass_context
def sync_do_reject(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to each object,
    and rejecting the objects with the test succeeding.

    If no test is specified, each object will be evaluated as a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ numbers|reject("odd") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (n for n in numbers if not test_odd(n))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@async_variant(sync_do_reject)  # type: ignore
async def do_reject(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@pass_context
def sync_do_selectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to the specified
    attribute of each object, and only selecting the objects with the
    test succeeding.

    If no test is specified, the attribute's value will be evaluated as
    a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ users|selectattr("is_active") }}
        {{ users|selectattr("email", "none") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (user for user in users if user.is_active)
        (user for user in users if test_none(user.email))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: x, True)


@async_variant(sync_do_selectattr)  # type: ignore
async def do_selectattr(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, True)


@pass_context
def sync_do_rejectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to the specified
    attribute of each object, and rejecting the objects with the test
    succeeding.

    If no test is specified, the attribute's value will be evaluated as
    a boolean.

    .. sourcecode:: jinja

        {{ users|rejectattr("is_active") }}
        {{ users|rejectattr("email", "none") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (user for user in users if not user.is_active)
        (user for user in users if not test_none(user.email))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@async_variant(sync_do_rejectattr)  # type: ignore
async def do_rejectattr(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@pass_eval_context
def do_tojson(
    eval_ctx: "EvalContext", value: t.Any, indent: t.Optional[int] = None
) -> Markup:
    """Serialize an object to a string of JSON, and mark it safe to
    render in HTML. This filter is only for use in HTML documents.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    :param value: The object to serialize to JSON.
    :param indent: The ``indent`` parameter passed to ``dumps``, for
        pretty-printing the value.

    .. versionadded:: 2.9
    """
    policies = eval_ctx.environment.policies
    dumps = policies["json.dumps_function"]
    kwargs = policies["json.dumps_kwargs"]

    if indent is not None:
        kwargs = kwargs.copy()
        kwargs["indent"] = indent

    return htmlsafe_json_dumps(value, dumps=dumps, **kwargs)


def prepare_map(
    context: "Context", args: t.Tuple[t.Any, ...], kwargs: t.Dict[str, t.Any]
) -> t.Callable[[t.Any], t.Any]:
    if not args and "attribute" in kwargs:
        attribute = kwargs.pop("attribute")
        default = kwargs.pop("default", None)

        if kwargs:
            raise FilterArgumentError(
                f"Unexpected keyword argument {next(iter(kwargs))!r}"
            )

        func = make_attrgetter(context.environment, attribute, default=default)
    else:
        try:
            name = args[0]
            args = args[1:]
        except LookupError:
            raise FilterArgumentError("map requires a filter argument") from None

        def func(item: t.Any) -> t.Any:
            return context.environment.call_filter(
                name, item, args, kwargs, context=context
            )

    return func


def prepare_select_or_reject(
    context: "Context",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> t.Callable[[t.Any], t.Any]:
    if lookup_attr:
        try:
            attr = args[0]
        except LookupError:
            raise FilterArgumentError("Missing parameter for attribute name") from None

        transfunc = make_attrgetter(context.environment, attr)
        off = 1
    else:
        off = 0

        def transfunc(x: V) -> V:
            return x

    try:
        name = args[off]
        args = args[1 + off :]

        def func(item: t.Any) -> t.Any:
            return context.environment.call_test(name, item, args, kwargs, context)

    except LookupError:
        func = bool  # type: ignore

    return lambda item: modfunc(func(transfunc(item)))


def select_or_reject(
    context: "Context",
    value: "t.Iterable[V]",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.Iterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)

        for item in value:
            if func(item):
                yield item


async def async_select_or_reject(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.AsyncIterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)

        async for item in auto_aiter(value):
            if func(item):
                yield item


FILTERS = {
    "abs": abs,
    "attr": do_attr,
    "batch": do_batch,
    "capitalize": do_capitalize,
    "center": do_center,
    "count": len,
    "d": do_default,
    "default": do_default,
    "dictsort": do_dictsort,
    "e": escape,
    "escape": escape,
    "filesizeformat": do_filesizeformat,
    "first": do_first,
    "float": do_float,
    "forceescape": do_forceescape,
    "format": do_format,
    "groupby": do_groupby,
    "indent": do_indent,
    "int": do_int,
    "join": do_join,
    "last": do_last,
    "length": len,
    "list": do_list,
    "lower": do_lower,
    "items": do_items,
    "map": do_map,
    "min": do_min,
    "max": do_max,
    "pprint": do_pprint,
    "random": do_random,
    "reject": do_reject,
    "rejectattr": do_rejectattr,
    "replace": do_replace,
    "reverse": do_reverse,
    "round": do_round,
    "safe": do_mark_safe,
    "select": do_select,
    "selectattr": do_selectattr,
    "slice": do_slice,
    "sort": do_sort,
    "string": soft_str,
    "striptags": do_striptags,
    "sum": do_sum,
    "title": do_title,
    "trim": do_trim,
    "truncate": do_truncate,
    "unique": do_unique,
    "upper": do_upper,
    "urlencode": do_urlencode,
    "urlize": do_urlize,
    "wordcount": do_wordcount,
    "wordwrap": do_wordwrap,
    "xmlattr": do_xmlattr,
    "tojson": do_tojson,
}
<|endoftext|>"""The runtime functions and state used by compiled templates."""

import functools
import sys
import typing as t
from collections import abc
from itertools import chain

from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str

from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context

V = t.TypeVar("V")
F = t.TypeVar("F", bound=t.Callable[..., t.Any])

if t.TYPE_CHECKING:
    import logging

    import typing_extensions as te

    from .environment import Environment

    class LoopRenderFunc(te.Protocol):
        def __call__(
            self,
            reciter: t.Iterable[V],
            loop_render_func: "LoopRenderFunc",
            depth: int = 0,
        ) -> str: ...


# these variables are exported to the template runtime
exported = [
    "LoopContext",
    "TemplateReference",
    "Macro",
    "Markup",
    "TemplateRuntimeError",
    "missing",
    "escape",
    "markup_join",
    "str_join",
    "identity",
    "TemplateNotFound",
    "Namespace",
    "Undefined",
    "internalcode",
]
async_exported = [
    "AsyncLoopContext",
    "auto_aiter",
    "auto_await",
]


def identity(x: V) -> V:
    """Returns its argument. Useful for certain things in the
    environment.
    """
    return x


def markup_join(seq: t.Iterable[t.Any]) -> str:
    """Concatenation that escapes if necessary and converts to string."""
    buf = []
    iterator = map(soft_str, seq)
    for arg in iterator:
        buf.append(arg)
        if hasattr(arg, "__html__"):
            return Markup("").join(chain(buf, iterator))
    return concat(buf)


def str_join(seq: t.Iterable[t.Any]) -> str:
    """Simple args to string conversion and concatenation."""
    return concat(map(str, seq))


def new_context(
    environment: "Environment",
    template_name: t.Optional[str],
    blocks: t.Dict[str, t.Callable[["Context"], t.Iterator[str]]],
    vars: t.Optional[t.Dict[str, t.Any]] = None,
    shared: bool = False,
    globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    locals: t.Optional[t.Mapping[str, t.Any]] = None,
) -> "Context":
    """Internal helper for context creation."""
    if vars is None:
        vars = {}
    if shared:
        parent = vars
    else:
        parent = dict(globals or (), **vars)
    if locals:
        # if the parent is shared a copy should be created because
        # we don't want to modify the dict passed
        if shared:
            parent = dict(parent)
        for key, value in locals.items():
            if value is not missing:
                parent[key] = value
    return environment.context_class(
        environment, parent, template_name, blocks, globals=globals
    )


class TemplateReference:
    """The `self` in templates."""

    def __init__(self, context: "Context") -> None:
        self.__context = context

    def __getitem__(self, name: str) -> t.Any:
        blocks = self.__context.blocks[name]
        return BlockReference(name, self.__context, blocks, 0)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.__context.name!r}>"


def _dict_method_all(dict_method: F) -> F:
    @functools.wraps(dict_method)
    def f_all(self: "Context") -> t.Any:
        return dict_method(self.get_all())

    return t.cast(F, f_all)


@abc.Mapping.register
class Context:
    """The template context holds the variables of a template.  It stores the
    values passed to the template and also the names the template exports.
    Creating instances is neither supported nor useful as it's created
    automatically at various stages of the template evaluation and should not
    be created by hand.

    The context is immutable.  Modifications on :attr:`parent` **must not**
    happen and modifications on :attr:`vars` are allowed from generated
    template code only.  Template filters and global functions marked as
    :func:`pass_context` get the active context passed as first argument
    and are allowed to access the context read-only.

    The template context supports read only dict operations (`get`,
    `keys`, `values`, `items`, `iterkeys`, `itervalues`, `iteritems`,
    `__getitem__`, `__contains__`).  Additionally there is a :meth:`resolve`
    method that doesn't fail with a `KeyError` but returns an
    :class:`Undefined` object for missing variables.
    """

    def __init__(
        self,
        environment: "Environment",
        parent: t.Dict[str, t.Any],
        name: t.Optional[str],
        blocks: t.Dict[str, t.Callable[["Context"], t.Iterator[str]]],
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ):
        self.parent = parent
        self.vars: t.Dict[str, t.Any] = {}
        self.environment: Environment = environment
        self.eval_ctx = EvalContext(self.environment, name)
        self.exported_vars: t.Set[str] = set()
        self.name = name
        self.globals_keys = set() if globals is None else set(globals)

        # create the initial mapping of blocks.  Whenever template inheritance
        # takes place the runtime will update this mapping with the new blocks
        # from the template.
        self.blocks = {k: [v] for k, v in blocks.items()}

    def super(
        self, name: str, current: t.Callable[["Context"], t.Iterator[str]]
    ) -> t.Union["BlockReference", "Undefined"]:
        """Render a parent block."""
        try:
            blocks = self.blocks[name]
            index = blocks.index(current) + 1
            blocks[index]
        except LookupError:
            return self.environment.undefined(
                f"there is no parent block called {name!r}.", name="super"
            )
        return BlockReference(name, self, blocks, index)

    def get(self, key: str, default: t.Any = None) -> t.Any:
        """Look up a variable by name, or return a default if the key is
        not found.

        :param key: The variable name to look up.
        :param default: The value to return if the key is not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def resolve(self, key: str) -> t.Union[t.Any, "Undefined"]:
        """Look up a variable by name, or return an :class:`Undefined`
        object if the key is not found.

        If you need to add custom behavior, override
        :meth:`resolve_or_missing`, not this method. The various lookup
        functions use that method, not this one.

        :param key: The variable name to look up.
        """
        rv = self.resolve_or_missing(key)

        if rv is missing:
            return self.environment.undefined(name=key)

        return rv

    def resolve_or_missing(self, key: str) -> t.Any:
        """Look up a variable by name, or return a ``missing`` sentinel
        if the key is not found.

        Override this method to add custom lookup behavior.
        :meth:`resolve`, :meth:`get`, and :meth:`__getitem__` use this
        method. Don't call this method directly.

        :param key: The variable name to look up.
        """
        if key in self.vars:
            return self.vars[key]

        if key in self.parent:
            return self.parent[key]

        return missing

    def get_exported(self) -> t.Dict[str, t.Any]:
        """Get a new dict with the exported variables."""
        return {k: self.vars[k] for k in self.exported_vars}

    def get_all(self) -> t.Dict[str, t.Any]:
        """Return the complete context as dict including the exported
        variables.  For optimizations reasons this might not return an
        actual copy so be careful with using it.
        """
        if not self.vars:
            return self.parent
        if not self.parent:
            return self.vars
        return dict(self.parent, **self.vars)

    @internalcode
    def call(
        __self,
        __obj: t.Callable[..., t.Any],
        *args: t.Any,
        **kwargs: t.Any,  # noqa: B902
    ) -> t.Union[t.Any, "Undefined"]:
        """Call the callable with the arguments and keyword arguments
        provided but inject the active context or environment as first
        argument if the callable has :func:`pass_context` or
        :func:`pass_environment`.
        """
        if __debug__:
            __traceback_hide__ = True  # noqa

        # Allow callable classes to take a context
        if (
            hasattr(__obj, "__call__")  # noqa: B004
            and _PassArg.from_obj(__obj.__call__) is not None
        ):
            __obj = __obj.__call__

        pass_arg = _PassArg.from_obj(__obj)

        if pass_arg is _PassArg.context:
            # the active context should have access to variables set in
            # loops and blocks without mutating the context itself
            if kwargs.get("_loop_vars"):
                __self = __self.derived(kwargs["_loop_vars"])
            if kwargs.get("_block_vars"):
                __self = __self.derived(kwargs["_block_vars"])
            args = (__self,) + args
        elif pass_arg is _PassArg.eval_context:
            args = (__self.eval_ctx,) + args
        elif pass_arg is _PassArg.environment:
            args = (__self.environment,) + args

        kwargs.pop("_block_vars", None)
        kwargs.pop("_loop_vars", None)

        try:
            return __obj(*args, **kwargs)
        except StopIteration:
            return __self.environment.undefined(
                "value was undefined because a callable raised a"
                " StopIteration exception"
            )

    def derived(self, locals: t.Optional[t.Dict[str, t.Any]] = None) -> "Context":
        """Internal helper function to create a derived context.  This is
        used in situations where the system needs a new context in the same
        template that is independent.
        """
        context = new_context(
            self.environment, self.name, {}, self.get_all(), True, None, locals
        )
        context.eval_ctx = self.eval_ctx
        context.blocks.update((k, list(v)) for k, v in self.blocks.items())
        return context

    keys = _dict_method_all(dict.keys)
    values = _dict_method_all(dict.values)
    items = _dict_method_all(dict.items)

    def __contains__(self, name: str) -> bool:
        return name in self.vars or name in self.parent

    def __getitem__(self, key: str) -> t.Any:
        """Look up a variable by name with ``[]`` syntax, or raise a
        ``KeyError`` if the key is not found.
        """
        item = self.resolve_or_missing(key)

        if item is missing:
            raise KeyError(key)

        return item

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.get_all()!r} of {self.name!r}>"


class BlockReference:
    """One block on a template reference."""

    def __init__(
        self,
        name: str,
        context: "Context",
        stack: t.List[t.Callable[["Context"], t.Iterator[str]]],
        depth: int,
    ) -> None:
        self.name = name
        self._context = context
        self._stack = stack
        self._depth = depth

    @property
    def super(self) -> t.Union["BlockReference", "Undefined"]:
        """Super the block."""
        if self._depth + 1 >= len(self._stack):
            return self._context.environment.undefined(
                f"there is no parent block called {self.name!r}.", name="super"
            )
        return BlockReference(self.name, self._context, self._stack, self._depth + 1)

    @internalcode
    async def _async_call(self) -> str:
        rv = self._context.environment.concat(  # type: ignore
            [x async for x in self._stack[self._depth](self._context)]  # type: ignore
        )

        if self._context.eval_ctx.autoescape:
            return Markup(rv)

        return rv

    @internalcode
    def __call__(self) -> str:
        if self._context.environment.is_async:
            return self._async_call()  # type: ignore

        rv = self._context.environment.concat(  # type: ignore
            self._stack[self._depth](self._context)
        )

        if self._context.eval_ctx.autoescape:
            return Markup(rv)

        return rv


class LoopContext:
    """A wrapper iterable for dynamic ``for`` loops, with information
    about the loop and iteration.
    """

    #: Current iteration of the loop, starting at 0.
    index0 = -1

    _length: t.Optional[int] = None
    _after: t.Any = missing
    _current: t.Any = missing
    _before: t.Any = missing
    _last_changed_value: t.Any = missing

    def __init__(
        self,
        iterable: t.Iterable[V],
        undefined: t.Type["Undefined"],
        recurse: t.Optional["LoopRenderFunc"] = None,
        depth0: int = 0,
    ) -> None:
        """
        :param iterable: Iterable to wrap.
        :param undefined: :class:`Undefined` class to use for next and
            previous items.
        :param recurse: The function to render the loop body when the
            loop is marked recursive.
        :param depth0: Incremented when looping recursively.
        """
        self._iterable = iterable
        self._iterator = self._to_iterator(iterable)
        self._undefined = undefined
        self._recurse = recurse
        #: How many levels deep a recursive loop currently is, starting at 0.
        self.depth0 = depth0

    @staticmethod
    def _to_iterator(iterable: t.Iterable[V]) -> t.Iterator[V]:
        return iter(iterable)

    @property
    def length(self) -> int:
        """Length of the iterable.

        If the iterable is a generator or otherwise does not have a
        size, it is eagerly evaluated to get a size.
        """
        if self._length is not None:
            return self._length

        try:
            self._length = len(self._iterable)  # type: ignore
        except TypeError:
            iterable = list(self._iterator)
            self._iterator = self._to_iterator(iterable)
            self._length = len(iterable) + self.index + (self._after is not missing)

        return self._length

    def __len__(self) -> int:
        return self.length

    @property
    def depth(self) -> int:
        """How many levels deep a recursive loop currently is, starting at 1."""
        return self.depth0 + 1

    @property
    def index(self) -> int:
        """Current iteration of the loop, starting at 1."""
        return self.index0 + 1

    @property
    def revindex0(self) -> int:
        """Number of iterations from the end of the loop, ending at 0.

        Requires calculating :attr:`length`.
        """
        return self.length - self.index

    @property
    def revindex(self) -> int:
        """Number of iterations from the end of the loop, ending at 1.

        Requires calculating :attr:`length`.
        """
        return self.length - self.index0

    @property
    def first(self) -> bool:
        """Whether this is the first iteration of the loop."""
        return self.index0 == 0

    def _peek_next(self) -> t.Any:
        """Return the next element in the iterable, or :data:`missing`
        if the iterable is exhausted. Only peeks one item ahead, caching
        the result in :attr:`_last` for use in subsequent checks. The
        cache is reset when :meth:`__next__` is called.
        """
        if self._after is not missing:
            return self._after

        self._after = next(self._iterator, missing)
        return self._after

    @property
    def last(self) -> bool:
        """Whether this is the last iteration of the loop.

        Causes the iterable to advance early. See
        :func:`itertools.groupby` for issues this can cause.
        The :func:`groupby` filter avoids that issue.
        """
        return self._peek_next() is missing

    @property
    def previtem(self) -> t.Union[t.Any, "Undefined"]:
        """The item in the previous iteration. Undefined during the
        first iteration.
        """
        if self.first:
            return self._undefined("there is no previous item")

        return self._before

    @property
    def nextitem(self) -> t.Union[t.Any, "Undefined"]:
        """The item in the next iteration. Undefined during the last
        iteration.

        Causes the iterable to advance early. See
        :func:`itertools.groupby` for issues this can cause.
        The :func:`jinja-filters.groupby` filter avoids that issue.
        """
        rv = self._peek_next()

        if rv is missing:
            return self._undefined("there is no next item")

        return rv

    def cycle(self, *args: V) -> V:
        """Return a value from the given args, cycling through based on
        the current :attr:`index0`.

        :param args: One or more values to cycle through.
        """
        if not args:
            raise TypeError("no items for cycling given")

        return args[self.index0 % len(args)]

    def changed(self, *value: t.Any) -> bool:
        """Return ``True`` if previously called with a different value
        (including when called for the first time).

        :param value: One or more values to compare to the last call.
        """
        if self._last_changed_value != value:
            self._last_changed_value = value
            return True

        return False

    def __iter__(self) -> "LoopContext":
        return self

    def __next__(self) -> t.Tuple[t.Any, "LoopContext"]:
        if self._after is not missing:
            rv = self._after
            self._after = missing
        else:
            rv = next(self._iterator)

        self.index0 += 1
        self._before = self._current
        self._current = rv
        return rv, self

    @internalcode
    def __call__(self, iterable: t.Iterable[V]) -> str:
        """When iterating over nested data, render the body of the loop
        recursively with the given inner iterable data.

        The loop must have the ``recursive`` marker for this to work.
        """
        if self._recurse is None:
            raise TypeError(
                "The loop must have the 'recursive' marker to be called recursively."
            )

        return self._recurse(iterable, self._recurse, depth=self.depth)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.index}/{self.length}>"


class AsyncLoopContext(LoopContext):
    _iterator: t.AsyncIterator[t.Any]  # type: ignore

    @staticmethod
    def _to_iterator(  # type: ignore
        iterable: t.Union[t.Iterable[V], t.AsyncIterable[V]],
    ) -> t.AsyncIterator[V]:
        return auto_aiter(iterable)

    @property
    async def length(self) -> int:  # type: ignore
        if self._length is not None:
            return self._length

        try:
            self._length = len(self._iterable)  # type: ignore
        except TypeError:
            iterable = [x async for x in self._iterator]
            self._iterator = self._to_iterator(iterable)
            self._length = len(iterable) + self.index + (self._after is not missing)

        return self._length

    @property
    async def revindex0(self) -> int:  # type: ignore
        return await self.length - self.index

    @property
    async def revindex(self) -> int:  # type: ignore
        return await self.length - self.index0

    async def _peek_next(self) -> t.Any:
        if self._after is not missing:
            return self._after

        try:
            self._after = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._after = missing

        return self._after

    @property
    async def last(self) -> bool:  # type: ignore
        return await self._peek_next() is missing

    @property
    async def nextitem(self) -> t.Union[t.Any, "Undefined"]:
        rv = await self._peek_next()

        if rv is missing:
            return self._undefined("there is no next item")

        return rv

    def __aiter__(self) -> "AsyncLoopContext":
        return self

    async def __anext__(self) -> t.Tuple[t.Any, "AsyncLoopContext"]:
        if self._after is not missing:
            rv = self._after
            self._after = missing
        else:
            rv = await self._iterator.__anext__()

        self.index0 += 1
        self._before = self._current
        self._current = rv
        return rv, self


class Macro:
    """Wraps a macro function."""

    def __init__(
        self,
