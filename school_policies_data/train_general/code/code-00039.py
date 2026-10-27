        target = other._path if isinstance(other, Path) else other
        return self._path.__le__(target)

    def __gt__(self, other: pathlib.PurePath | Path) -> bool:
        target = other._path if isinstance(other, Path) else other
        return self._path.__gt__(target)

    def __ge__(self, other: pathlib.PurePath | Path) -> bool:
        target = other._path if isinstance(other, Path) else other
        return self._path.__ge__(target)

    def __truediv__(self, other: str | PathLike[str]) -> Self:
        return type(self)(self._path / other, limiter=self._limiter)

    def __rtruediv__(self, other: str | PathLike[str]) -> Self:
        return type(self)(other, limiter=self._limiter) / self

    @property
    def limiter(self) -> CapacityLimiter | None:
        """The capacity limiter used by this path, if not the global limiter."""
        return self._limiter

    @property
    def parts(self) -> tuple[str, ...]:
        return self._path.parts

    @property
    def drive(self) -> str:
        return self._path.drive

    @property
    def root(self) -> str:
        return self._path.root

    @property
    def anchor(self) -> str:
        return self._path.anchor

    @property
    def parents(self) -> Sequence[Self]:
        return tuple(type(self)(p, limiter=self._limiter) for p in self._path.parents)

    @property
    def parent(self) -> Self:
        return type(self)(self._path.parent, limiter=self._limiter)

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def suffixes(self) -> list[str]:
        return self._path.suffixes

    @property
    def stem(self) -> str:
        return self._path.stem

    async def absolute(self) -> Self:
        path = await to_thread.run_sync(self._path.absolute, limiter=self._limiter)
        return type(self)(path, limiter=self._limiter)

    def as_posix(self) -> str:
        return self._path.as_posix()

    def as_uri(self) -> str:
        return self._path.as_uri()

    if sys.version_info >= (3, 13):
        parser: ClassVar[ModuleType] = pathlib.Path.parser

        @classmethod
        def from_uri(cls, uri: str, *, limiter: CapacityLimiter | None = None) -> Self:
            return cls(pathlib.Path.from_uri(uri), limiter=limiter)

        def full_match(
            self, path_pattern: str, *, case_sensitive: bool | None = None
        ) -> bool:
            return self._path.full_match(path_pattern, case_sensitive=case_sensitive)

        def match(
            self, path_pattern: str, *, case_sensitive: bool | None = None
        ) -> bool:
            return self._path.match(path_pattern, case_sensitive=case_sensitive)
    else:

        def match(self, path_pattern: str) -> bool:
            return self._path.match(path_pattern)

    if sys.version_info >= (3, 14):

        @property
        def info(self) -> PathInfo:
            return self._path.info

        async def copy(
            self,
            target: str | os.PathLike[str],
            *,
            follow_symlinks: bool = True,
            preserve_metadata: bool = False,
        ) -> Self:
            func = partial(
                self._path.copy,
                follow_symlinks=follow_symlinks,
                preserve_metadata=preserve_metadata,
            )
            return type(self)(
                await to_thread.run_sync(
                    func, pathlib.Path(target), limiter=self._limiter
                ),
                limiter=self._limiter,
            )

        async def copy_into(
            self,
            target_dir: str | os.PathLike[str],
            *,
            follow_symlinks: bool = True,
            preserve_metadata: bool = False,
        ) -> Self:
            func = partial(
                self._path.copy_into,
                follow_symlinks=follow_symlinks,
                preserve_metadata=preserve_metadata,
            )
            return type(self)(
                await to_thread.run_sync(
                    func, pathlib.Path(target_dir), limiter=self._limiter
                ),
                limiter=self._limiter,
            )

        async def move(self, target: str | os.PathLike[str]) -> Self:
            # Upstream does not handle anyio.Path properly as a PathLike
            target = pathlib.Path(target)
            return type(self)(
                await to_thread.run_sync(
                    self._path.move, target, limiter=self._limiter
                ),
                limiter=self._limiter,
            )

        async def move_into(
            self,
            target_dir: str | os.PathLike[str],
        ) -> Self:
            return type(self)(
                await to_thread.run_sync(
                    self._path.move_into, target_dir, limiter=self._limiter
                ),
                limiter=self._limiter,
            )

    def is_relative_to(self, other: str | PathLike[str]) -> bool:
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False

    async def chmod(self, mode: int, *, follow_symlinks: bool = True) -> None:
        func = partial(os.chmod, follow_symlinks=follow_symlinks)
        return await to_thread.run_sync(func, self._path, mode, limiter=self._limiter)

    @classmethod
    async def cwd(cls, *, limiter: CapacityLimiter | None = None) -> Self:
        path = await to_thread.run_sync(pathlib.Path.cwd, limiter=limiter)
        return cls(path, limiter=limiter)

    if sys.version_info >= (3, 12):

        async def exists(self, *, follow_symlinks: bool = True) -> bool:
            return await to_thread.run_sync(
                partial(self._path.exists, follow_symlinks=follow_symlinks),
                abandon_on_cancel=True,
                limiter=self._limiter,
            )

    else:

        async def exists(self) -> bool:
            return await to_thread.run_sync(
                self._path.exists, abandon_on_cancel=True, limiter=self._limiter
            )

    async def expanduser(self) -> Self:
        return type(self)(
            await to_thread.run_sync(
                self._path.expanduser, abandon_on_cancel=True, limiter=self._limiter
            ),
            limiter=self._limiter,
        )

    if sys.version_info < (3, 12):
        # Python 3.11 and earlier
        def glob(self, pattern: str) -> AsyncIterator[Self]:
            gen = self._path.glob(pattern)
            return _PathIterator(gen, self._limiter, type(self))
    elif (3, 12) <= sys.version_info < (3, 13):
        # changed in Python 3.12:
        # - The case_sensitive parameter was added.
        def glob(
            self,
            pattern: str,
            *,
            case_sensitive: bool | None = None,
        ) -> AsyncIterator[Self]:
            gen = self._path.glob(pattern, case_sensitive=case_sensitive)
            return _PathIterator(gen, self._limiter, type(self))
    elif sys.version_info >= (3, 13):
        # Changed in Python 3.13:
        # - The recurse_symlinks parameter was added.
        # - The pattern parameter accepts a path-like object.
        def glob(  # type: ignore[misc] # mypy doesn't allow for differing signatures in a conditional block
            self,
            pattern: str | PathLike[str],
            *,
            case_sensitive: bool | None = None,
            recurse_symlinks: bool = False,
        ) -> AsyncIterator[Self]:
            gen = self._path.glob(
                pattern,  # type: ignore[arg-type]
                case_sensitive=case_sensitive,
                recurse_symlinks=recurse_symlinks,
            )
            return _PathIterator(gen, self._limiter, type(self))

    if sys.version_info >= (3, 13):

        async def group(self, *, follow_symlinks: bool = True) -> str:
            return await to_thread.run_sync(
                partial(self._path.group, follow_symlinks=follow_symlinks),
                abandon_on_cancel=True,
                limiter=self._limiter,
            )

    else:

        async def group(self) -> str:
            return await to_thread.run_sync(
                self._path.group, abandon_on_cancel=True, limiter=self._limiter
            )

    async def hardlink_to(
        self, target: str | bytes | PathLike[str] | PathLike[bytes]
    ) -> None:
        if isinstance(target, Path):
            target = target._path

        await to_thread.run_sync(os.link, target, self, limiter=self._limiter)

    @classmethod
    async def home(cls, *, limiter: CapacityLimiter | None = None) -> Self:
        home_path = await to_thread.run_sync(pathlib.Path.home, limiter=limiter)
        return cls(home_path, limiter=limiter)

    def is_absolute(self) -> bool:
        return self._path.is_absolute()

    async def is_block_device(self) -> bool:
        return await to_thread.run_sync(
            self._path.is_block_device, abandon_on_cancel=True, limiter=self._limiter
        )

    async def is_char_device(self) -> bool:
        return await to_thread.run_sync(
            self._path.is_char_device, abandon_on_cancel=True, limiter=self._limiter
        )

    if sys.version_info >= (3, 13):

        async def is_dir(self, *, follow_symlinks: bool = True) -> bool:
            return await to_thread.run_sync(
                partial(self._path.is_dir, follow_symlinks=follow_symlinks),
                abandon_on_cancel=True,
                limiter=self._limiter,
            )

    else:

        async def is_dir(self) -> bool:
            return await to_thread.run_sync(
                self._path.is_dir, abandon_on_cancel=True, limiter=self._limiter
            )

    async def is_fifo(self) -> bool:
        return await to_thread.run_sync(
            self._path.is_fifo, abandon_on_cancel=True, limiter=self._limiter
        )

    if sys.version_info >= (3, 13):

        async def is_file(self, *, follow_symlinks: bool = True) -> bool:
            return await to_thread.run_sync(
                partial(self._path.is_file, follow_symlinks=follow_symlinks),
                abandon_on_cancel=True,
                limiter=self._limiter,
            )

    else:

        async def is_file(self) -> bool:
            return await to_thread.run_sync(
                self._path.is_file, abandon_on_cancel=True, limiter=self._limiter
            )

    if sys.version_info >= (3, 12):

        async def is_junction(self) -> bool:
            return await to_thread.run_sync(
                self._path.is_junction, limiter=self._limiter
            )

    async def is_mount(self) -> bool:
        return await to_thread.run_sync(
            os.path.ismount, self._path, abandon_on_cancel=True, limiter=self._limiter
        )

    if sys.version_info < (3, 15):

        def is_reserved(self) -> bool:
            return self._path.is_reserved()

    async def is_socket(self) -> bool:
        return await to_thread.run_sync(
            self._path.is_socket, abandon_on_cancel=True, limiter=self._limiter
        )

    async def is_symlink(self) -> bool:
        return await to_thread.run_sync(
            self._path.is_symlink, abandon_on_cancel=True, limiter=self._limiter
        )

    async def iterdir(self) -> AsyncIterator[Self]:
        gen = (
            self._path.iterdir()
            if sys.version_info < (3, 13)
            else await to_thread.run_sync(
                self._path.iterdir, abandon_on_cancel=True, limiter=self._limiter
            )
        )
        async for path in _PathIterator(gen, self._limiter, type(self)):
            yield path

    def joinpath(self, *args: str | PathLike[str]) -> Self:
        return type(self)(self._path.joinpath(*args), limiter=self._limiter)

    async def lchmod(self, mode: int) -> None:
        await to_thread.run_sync(self._path.lchmod, mode, limiter=self._limiter)

    async def lstat(self) -> os.stat_result:
        return await to_thread.run_sync(
            self._path.lstat, abandon_on_cancel=True, limiter=self._limiter
        )

    async def mkdir(
        self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        await to_thread.run_sync(
            self._path.mkdir, mode, parents, exist_ok, limiter=self._limiter
        )

    @overload
    async def open(
        self,
        mode: OpenBinaryMode,
        buffering: int = ...,
        encoding: str | None = ...,
        errors: str | None = ...,
        newline: str | None = ...,
    ) -> AsyncFile[bytes]: ...

    @overload
    async def open(
        self,
        mode: OpenTextMode = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        errors: str | None = ...,
        newline: str | None = ...,
    ) -> AsyncFile[str]: ...

    async def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> AsyncFile[Any]:
        fp = await to_thread.run_sync(
            self._path.open,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            limiter=self._limiter,
        )
        return AsyncFile(fp, limiter=self._limiter)

    if sys.version_info >= (3, 13):

        async def owner(self, *, follow_symlinks: bool = True) -> str:
            return await to_thread.run_sync(
                partial(self._path.owner, follow_symlinks=follow_symlinks),
                abandon_on_cancel=True,
                limiter=self._limiter,
            )

    else:

        async def owner(self) -> str:
            return await to_thread.run_sync(
                self._path.owner, abandon_on_cancel=True, limiter=self._limiter
            )

    async def read_bytes(self) -> bytes:
        return await to_thread.run_sync(self._path.read_bytes, limiter=self._limiter)

    if sys.version_info >= (3, 13):

        async def read_text(
            self,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> str:
            return await to_thread.run_sync(
                self._path.read_text,
                encoding,
                errors,
                newline,
                limiter=self._limiter,
            )

    else:

        async def read_text(
            self, encoding: str | None = None, errors: str | None = None
        ) -> str:
            return await to_thread.run_sync(
                self._path.read_text, encoding, errors, limiter=self._limiter
            )

    if sys.version_info >= (3, 12):

        def relative_to(
            self, *other: str | PathLike[str], walk_up: bool = False
        ) -> Self:
            # relative_to() should work with any PathLike but it doesn't
            others = [pathlib.Path(other) for other in other]
            return type(self)(
                self._path.relative_to(*others, walk_up=walk_up), limiter=self._limiter
            )

    else:

        def relative_to(self, *other: str | PathLike[str]) -> Self:
            return type(self)(self._path.relative_to(*other), limiter=self._limiter)

    async def readlink(self) -> Self:
        target = await to_thread.run_sync(
            os.readlink, self._path, limiter=self._limiter
        )
        return type(self)(target, limiter=self._limiter)

    async def rename(self, target: str | pathlib.PurePath | Path) -> Self:
        if isinstance(target, Path):
            target = target._path

        await to_thread.run_sync(self._path.rename, target, limiter=self._limiter)
        return type(self)(target, limiter=self._limiter)

    async def replace(self, target: str | pathlib.PurePath | Path) -> Self:
        if isinstance(target, Path):
            target = target._path

        await to_thread.run_sync(self._path.replace, target, limiter=self._limiter)
        return type(self)(target, limiter=self._limiter)

    async def resolve(self, strict: bool = False) -> Self:
        func = partial(self._path.resolve, strict=strict)
        return type(self)(
            await to_thread.run_sync(
                func, abandon_on_cancel=True, limiter=self._limiter
            ),
            limiter=self._limiter,
        )

    if sys.version_info < (3, 12):
        # Pre Python 3.12
        def rglob(self, pattern: str) -> AsyncIterator[Self]:
            gen = self._path.rglob(pattern)
            return _PathIterator(gen, self._limiter, type(self))
    elif (3, 12) <= sys.version_info < (3, 13):
        # Changed in Python 3.12:
        # - The case_sensitive parameter was added.
        def rglob(
            self, pattern: str, *, case_sensitive: bool | None = None
        ) -> AsyncIterator[Self]:
            gen = self._path.rglob(pattern, case_sensitive=case_sensitive)
            return _PathIterator(gen, self._limiter, type(self))
    elif sys.version_info >= (3, 13):
        # Changed in Python 3.13:
        # - The recurse_symlinks parameter was added.
        # - The pattern parameter accepts a path-like object.
        def rglob(  # type: ignore[misc] # mypy doesn't allow for differing signatures in a conditional block
            self,
            pattern: str | PathLike[str],
            *,
            case_sensitive: bool | None = None,
            recurse_symlinks: bool = False,
        ) -> AsyncIterator[Self]:
            gen = self._path.rglob(
                pattern,  # type: ignore[arg-type]
                case_sensitive=case_sensitive,
                recurse_symlinks=recurse_symlinks,
            )
            return _PathIterator(gen, self._limiter, type(self))

    async def rmdir(self) -> None:
        await to_thread.run_sync(self._path.rmdir, limiter=self._limiter)

    async def samefile(self, other_path: str | PathLike[str]) -> bool:
        if isinstance(other_path, Path):
            other_path = other_path._path

        return await to_thread.run_sync(
            self._path.samefile,
            other_path,
            abandon_on_cancel=True,
            limiter=self._limiter,
        )

    async def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        func = partial(os.stat, follow_symlinks=follow_symlinks)
        return await to_thread.run_sync(
            func, self._path, abandon_on_cancel=True, limiter=self._limiter
        )

    async def symlink_to(
        self,
        target: str | bytes | PathLike[str] | PathLike[bytes],
        target_is_directory: bool = False,
    ) -> None:
        if isinstance(target, Path):
            target = target._path

        await to_thread.run_sync(
            self._path.symlink_to, target, target_is_directory, limiter=self._limiter
        )

    async def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        await to_thread.run_sync(
            self._path.touch, mode, exist_ok, limiter=self._limiter
        )

    async def unlink(self, missing_ok: bool = False) -> None:
        try:
            await to_thread.run_sync(self._path.unlink, limiter=self._limiter)
        except FileNotFoundError:
            if not missing_ok:
                raise

    if sys.version_info >= (3, 12):

        async def walk(
            self,
            top_down: bool = True,
            on_error: Callable[[OSError], object] | None = None,
            follow_symlinks: bool = False,
        ) -> AsyncIterator[tuple[Self, list[str], list[str]]]:
            def get_next_value() -> tuple[pathlib.Path, list[str], list[str]] | None:
                try:
                    return next(gen)
                except StopIteration:
                    return None

            gen = self._path.walk(top_down, on_error, follow_symlinks)
            while True:
                value = await to_thread.run_sync(get_next_value, limiter=self._limiter)
                if value is None:
                    return

                root, dirs, paths = value
                yield type(self)(root, limiter=self._limiter), dirs, paths

    def with_name(self, name: str) -> Self:
        return type(self)(self._path.with_name(name), limiter=self._limiter)

    if sys.version_info < (3, 13):
        # Backport pathlib's Python>=3.13 behavior for empty stems on paths with non-empty suffixes.
        # See: https://github.com/python/cpython/pull/114612
        def with_stem(self, stem: str) -> Self:
            suffix = self._path.suffix
            if not suffix:
                return self.with_name(stem)
            elif not stem:
                # If the suffix is non-empty, we can't make the stem empty.
                raise ValueError(f"{self!r} has a non-empty suffix")
            else:
                return self.with_name(stem + suffix)
    else:

        def with_stem(self, stem: str) -> Self:
            return type(self)(self._path.with_stem(stem), limiter=self._limiter)

    def with_suffix(self, suffix: str) -> Self:
        return type(self)(self._path.with_suffix(suffix), limiter=self._limiter)

    def with_segments(self, *pathsegments: str | PathLike[str]) -> Self:
        return type(self)(*pathsegments, limiter=self._limiter)

    async def write_bytes(self, data: ReadableBuffer) -> int:
        return await to_thread.run_sync(
            self._path.write_bytes, data, limiter=self._limiter
        )

    async def write_text(
        self,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        return await to_thread.run_sync(
            self._path.write_text,
            data,
            encoding,
            errors,
            newline,
            limiter=self._limiter,
        )


PathLike.register(Path)
<|endoftext|>from __future__ import annotations

from inspect import iscoroutine

__all__ = ("amap", "as_completed", "gather")

from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar, overload

from ..abc._tasks import get_coro_name
from ._exceptions import BrokenResourceError
from ._streams import create_memory_object_stream
from ._tasks import TaskHandle, create_task_group

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@overload
async def gather(
    coro1: Coroutine[Any, Any, R], coro2: Coroutine[Any, Any, S], /
) -> tuple[R, S]: ...


@overload
async def gather(
    coro1: Coroutine[Any, Any, R],
    coro2: Coroutine[Any, Any, S],
    coro3: Coroutine[Any, Any, T],
    /,
) -> tuple[R, S, T]: ...


@overload
async def gather(
    coro1: Coroutine[Any, Any, R],
    coro2: Coroutine[Any, Any, S],
    coro3: Coroutine[Any, Any, T],
    coro4: Coroutine[Any, Any, U],
    /,
) -> tuple[R, S, T, U]: ...


@overload
async def gather(
    coro1: Coroutine[Any, Any, R],
    coro2: Coroutine[Any, Any, S],
    coro3: Coroutine[Any, Any, T],
    coro4: Coroutine[Any, Any, U],
    coro5: Coroutine[Any, Any, V],
    /,
) -> tuple[R, S, T, U, V]: ...


@overload
async def gather(
    coro1: Coroutine[Any, Any, R],
    coro2: Coroutine[Any, Any, S],
    coro3: Coroutine[Any, Any, T],
    coro4: Coroutine[Any, Any, U],
    coro5: Coroutine[Any, Any, V],
    coro6: Coroutine[Any, Any, W],
    /,
) -> tuple[R, S, T, U, V, W]: ...


# handle arbitrary length if awaitables are all of the same type
@overload
async def gather(*coros: Coroutine[Any, Any, R]) -> tuple[R, ...]: ...


async def gather(*coros: Coroutine[Any, Any, Any]) -> tuple[Any, ...]:
    """
    Run coroutines concurrently in a task group. The order of result values corresponds
    to the order of coroutines passed.

    :param coros: coroutine objects to run as tasks
    :return: task results for each argument in the same order as they were passed

    """
    handles: list[TaskHandle[Any]] = []
    async with create_task_group() as tg:
        handles.extend(tg.create_task(coro) for coro in coros)

    return tuple(h.return_value for h in handles)


@asynccontextmanager
async def as_completed(
    *awaitables: Awaitable[R],
) -> AsyncGenerator[MemoryObjectReceiveStream[TaskHandle[R]]]:
    """
    Run awaitable objects concurrently in a task group, returning an iterator which can
    be used to get finished task handles in the order they complete.

    :param awaitables: awaitable objects to run as tasks
    :return: MemoryObjectReceiveStream for iterating over task handles as tasks complete.

    """
    if not awaitables:
        raise ValueError("as_completed() takes at least one awaitable")

    send, recv = create_memory_object_stream[TaskHandle[R]](len(awaitables))
    task_handles: list[TaskHandle[R]] = []

    async def runner(
        awaitable: Awaitable[R],
        index: int,
        _send: MemoryObjectSendStream[TaskHandle[R]],
    ) -> R:
        async with _send:
            try:
                return await awaitable
            finally:
                try:
                    _send.send_nowait(task_handles[index])
                except BrokenResourceError:
                    pass

    async with recv, create_task_group() as tg:
        async with send:
            for i, awaitable in enumerate(awaitables):
                name = get_coro_name(awaitable) if iscoroutine(awaitable) else None
                task_handles.append(
                    tg.start_soon(runner, awaitable, i, send.clone(), name=name)
                )
        try:
            yield recv
        finally:
            tg.cancel_scope.cancel()


async def amap(
    func: Callable[[T], Coroutine[Any, Any, R]], args: Iterable[T]
) -> list[R]:
    """
    Run the given coroutine function concurrently for multiple argument values.

    :param func: a coroutine function that takes a single argument
    :param args: a sequence of argument values to pass to ``func``
    :return: task results for each argument in the same order as they were passed

    """
    handles: list[TaskHandle[R]] = []
    async with create_task_group() as tg:
        handles.extend(tg.start_soon(func, arg) for arg in args)

    return [h.return_value for h in handles]
<|endoftext|>from __future__ import annotations

import math
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import Token
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar

from ._exceptions import NoEventLoopError

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

sniffio: Any
try:
    import sniffio
except ModuleNotFoundError:
    sniffio = None

if TYPE_CHECKING:
    from ..abc import AsyncBackend

# This must be updated when new backends are introduced
BACKENDS = "asyncio", "trio"

T_Retval = TypeVar("T_Retval")
PosArgsT = TypeVarTuple("PosArgsT")

threadlocals = threading.local()
loaded_backends: dict[str, type[AsyncBackend]] = {}


def run(
    func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
    *args: Unpack[PosArgsT],
    backend: str = "asyncio",
    backend_options: dict[str, Any] | None = None,
) -> T_Retval:
    """
    Run the given coroutine function in an asynchronous event loop.

    The current thread must not be already running an event loop.

    :param func: a coroutine function
    :param args: positional arguments to ``func``
    :param backend: name of the asynchronous event loop implementation – currently
        either ``asyncio`` or ``trio``
    :param backend_options: keyword arguments to call the backend ``run()``
        implementation with (documented :ref:`here <backend options>`)
    :return: the return value of the coroutine function
    :raises RuntimeError: if an asynchronous event loop is already running in this
        thread
    :raises LookupError: if the named backend is not found

    """
    if asynclib_name := current_async_library():
        raise RuntimeError(f"Already running {asynclib_name} in this thread")

    try:
        async_backend = get_async_backend(backend)
    except ImportError as exc:
        if backend in BACKENDS:
            raise LookupError(
                f"Backend {backend!r} is not available. "
                f"Install it with: pip install anyio[{backend}]"
            ) from exc

        raise LookupError(f"No such backend: {backend}") from exc

    token = None
    if asynclib_name is None:
        # Since we're in control of the event loop, we can cache the name of the async
        # library
        token = set_current_async_library(backend)

    try:
        backend_options = backend_options or {}
        return async_backend.run(func, args, {}, backend_options)
    finally:
        reset_current_async_library(token)


async def sleep(delay: float) -> None:
    """
    Pause the current task for the specified duration.

    :param delay: the duration, in seconds

    """
    return await get_async_backend().sleep(delay)


async def sleep_forever() -> None:
    """
    Pause the current task until it's cancelled.

    This is a shortcut for ``sleep(math.inf)``.

    .. versionadded:: 3.1

    """
    await sleep(math.inf)


async def sleep_until(deadline: float) -> None:
    """
    Pause the current task until the given time.

    :param deadline: the absolute time to wake up at (according to the internal
        monotonic clock of the event loop)

    .. versionadded:: 3.1

    """
    now = current_time()
    await sleep(max(deadline - now, 0))


def current_time() -> float:
    """
    Return the current value of the event loop's internal clock.

    :return: the clock value (seconds)
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().current_time()


def get_all_backends() -> tuple[str, ...]:
    """Return a tuple of the names of all built-in backends."""
    return BACKENDS


def get_available_backends() -> tuple[str, ...]:
    """
    Test for the availability of built-in backends.

    :return a tuple of the built-in backend names that were successfully imported

    .. versionadded:: 4.12

    """
    available_backends: list[str] = []
    for backend_name in get_all_backends():
        try:
            get_async_backend(backend_name)
        except ImportError:
            continue

        available_backends.append(backend_name)

    return tuple(available_backends)


def get_cancelled_exc_class() -> type[BaseException]:
    """
    Return the current async library's cancellation exception class.

    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().cancelled_exception_class()


#
# Private API
#


@contextmanager
def claim_worker_thread(
    backend_class: type[AsyncBackend], token: object
) -> Generator[Any, None, None]:
    from ..lowlevel import EventLoopToken

    threadlocals.current_token = EventLoopToken(backend_class, token)
    try:
        yield
    finally:
        del threadlocals.current_token


def get_async_backend(asynclib_name: str | None = None) -> type[AsyncBackend]:
    if asynclib_name is None:
        asynclib_name = current_async_library()
        if not asynclib_name:
            raise NoEventLoopError(
                f"Not currently running on any asynchronous event loop. "
                f"Available async backends: {', '.join(get_all_backends())}"
            )

    # We use our own dict instead of sys.modules to get the already imported back-end
    # class because the appropriate modules in sys.modules could potentially be only
    # partially initialized
    try:
        return loaded_backends[asynclib_name]
    except KeyError:
        module = import_module(f"anyio._backends._{asynclib_name}")
        loaded_backends[asynclib_name] = module.backend_class
        return module.backend_class


def current_async_library() -> str | None:
    if sniffio is None:
        # If sniffio is not installed, we assume we're either running asyncio or nothing
        import asyncio

        try:
            asyncio.get_running_loop()
            return "asyncio"
        except RuntimeError:
            pass
    else:
        try:
            return sniffio.current_async_library()
        except sniffio.AsyncLibraryNotFoundError:
            pass

    return None


def set_current_async_library(asynclib_name: str | None) -> Token | None:
    # no-op if sniffio is not installed
    if sniffio is None:
        return None

    return sniffio.current_async_library_cvar.set(asynclib_name)


def reset_current_async_library(token: Token | None) -> None:
    if token is not None:
        sniffio.current_async_library_cvar.reset(token)
<|endoftext|>from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TypeVar

from ..lowlevel import checkpoint_if_cancelled
from ._eventloop import get_async_backend
from ._exceptions import BusyResourceError, NoEventLoopError
from ._tasks import CancelScope
from ._testing import TaskInfo, get_current_task

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

T = TypeVar("T")


@dataclass(frozen=True)
class EventStatistics:
    """
    :ivar int tasks_waiting: number of tasks waiting on :meth:`~.Event.wait`
    """

    tasks_waiting: int


@dataclass(frozen=True)
class CapacityLimiterStatistics:
    """
    :ivar int borrowed_tokens: number of tokens currently borrowed by tasks
    :ivar float total_tokens: total number of available tokens
    :ivar tuple borrowers: tasks or other objects currently holding tokens borrowed from
        this limiter
    :ivar int tasks_waiting: number of tasks waiting on
        :meth:`~.CapacityLimiter.acquire` or
        :meth:`~.CapacityLimiter.acquire_on_behalf_of`
    """

    borrowed_tokens: int
    total_tokens: float
    borrowers: tuple[object, ...]
    tasks_waiting: int


@dataclass(frozen=True)
class LockStatistics:
    """
    :ivar bool locked: flag indicating if this lock is locked or not
    :ivar ~anyio.TaskInfo owner: task currently holding the lock (or ``None`` if the
        lock is not held by any task)
    :ivar int tasks_waiting: number of tasks waiting on :meth:`~.Lock.acquire`
    """

    locked: bool
    owner: TaskInfo | None
    tasks_waiting: int


@dataclass(frozen=True)
class ConditionStatistics:
    """
    :ivar int tasks_waiting: number of tasks blocked on :meth:`~.Condition.wait`
    :ivar ~anyio.LockStatistics lock_statistics: statistics of the underlying
        :class:`~.Lock`
    """

    tasks_waiting: int
    lock_statistics: LockStatistics


@dataclass(frozen=True)
class SemaphoreStatistics:
    """
    :ivar int tasks_waiting: number of tasks waiting on :meth:`~.Semaphore.acquire`

    """

    tasks_waiting: int


class Event:
    __slots__ = ("__weakref__",)

    def __new__(cls) -> Event:
        try:
            return get_async_backend().create_event()
        except NoEventLoopError:
            return EventAdapter()

    def set(self) -> None:
        """Set the flag, notifying all listeners."""
        raise NotImplementedError

    def is_set(self) -> bool:
        """Return ``True`` if the flag is set, ``False`` if not."""
        raise NotImplementedError

    async def wait(self) -> None:
        """
        Wait until the flag has been set.

        If the flag has already been set when this method is called, it returns
        immediately.

        """
        raise NotImplementedError

    def statistics(self) -> EventStatistics:
        """Return statistics about the current state of this event."""
        raise NotImplementedError


class EventAdapter(Event):
    __slots__ = "_internal_event", "_is_set"

    def __new__(cls) -> Self:
        return object.__new__(cls)

    def __init__(self) -> None:
        self._internal_event: Event | None = None
        self._is_set = False

    @property
    def _event(self) -> Event:
        if self._internal_event is None:
            self._internal_event = get_async_backend().create_event()
            if self._is_set:
                self._internal_event.set()

        return self._internal_event

    def set(self) -> None:
        if self._internal_event is None:
            self._is_set = True
        else:
            self._event.set()

    def is_set(self) -> bool:
        if self._internal_event is None:
            return self._is_set

        return self._internal_event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def statistics(self) -> EventStatistics:
        if self._internal_event is None:
            return EventStatistics(tasks_waiting=0)

        return self._internal_event.statistics()


class Lock:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, fast_acquire: bool = False) -> Lock:
        try:
            return get_async_backend().create_lock(fast_acquire=fast_acquire)
        except NoEventLoopError:
            return LockAdapter(fast_acquire=fast_acquire)

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    async def acquire(self) -> None:
        """Acquire the lock."""
        raise NotImplementedError

    def acquire_nowait(self) -> None:
        """
        Acquire the lock, without blocking.

        :raises ~anyio.WouldBlock: if the operation would block

        """
        raise NotImplementedError

    def release(self) -> None:
        """Release the lock."""
        raise NotImplementedError

    def locked(self) -> bool:
        """Return True if the lock is currently held."""
        raise NotImplementedError

    def statistics(self) -> LockStatistics:
        """
        Return statistics about the current state of this lock.

        .. versionadded:: 3.0
        """
        raise NotImplementedError


class LockAdapter(Lock):
    __slots__ = "_fast_acquire", "_internal_lock"

    def __new__(cls, *, fast_acquire: bool = False) -> Self:
        return object.__new__(cls)

    def __init__(self, *, fast_acquire: bool = False):
        self._internal_lock: Lock | None = None
        self._fast_acquire = fast_acquire

    @property
    def _lock(self) -> Lock:
        if self._internal_lock is None:
            self._internal_lock = get_async_backend().create_lock(
                fast_acquire=self._fast_acquire
            )

        return self._internal_lock

    async def __aenter__(self) -> None:
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._internal_lock is not None:
            self._internal_lock.release()

    async def acquire(self) -> None:
        """Acquire the lock."""
        await self._lock.acquire()

    def acquire_nowait(self) -> None:
        """
        Acquire the lock, without blocking.

        :raises ~anyio.WouldBlock: if the operation would block

        """
        self._lock.acquire_nowait()

    def release(self) -> None:
        """Release the lock."""
        self._lock.release()

    def locked(self) -> bool:
        """Return True if the lock is currently held."""
        return self._lock.locked()

    def statistics(self) -> LockStatistics:
        """
        Return statistics about the current state of this lock.

        .. versionadded:: 3.0

        """
        if self._internal_lock is None:
            return LockStatistics(False, None, 0)

        return self._internal_lock.statistics()


class Condition:
    __slots__ = "__weakref__", "_lock", "_owner_task", "_waiters"

    def __init__(self, lock: Lock | None = None):
        self._owner_task: TaskInfo | None = None
        self._lock = lock or Lock()
        self._waiters: deque[Event] = deque()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def _check_acquired(self) -> None:
        if self._owner_task != get_current_task():
            raise RuntimeError("The current task is not holding the underlying lock")

    async def acquire(self) -> None:
        """Acquire the underlying lock."""
        await self._lock.acquire()
        self._owner_task = get_current_task()

    def acquire_nowait(self) -> None:
        """
        Acquire the underlying lock, without blocking.

        :raises ~anyio.WouldBlock: if the operation would block

        """
        self._lock.acquire_nowait()
        self._owner_task = get_current_task()

    def release(self) -> None:
        """Release the underlying lock."""
        self._lock.release()

    def locked(self) -> bool:
        """Return True if the lock is set."""
        return self._lock.locked()

    def notify(self, n: int = 1) -> None:
        """Notify exactly n listeners."""
        self._check_acquired()
        for _ in range(n):
            try:
                event = self._waiters.popleft()
            except IndexError:
                break

            event.set()

    def notify_all(self) -> None:
        """Notify all the listeners."""
        self._check_acquired()
        for event in self._waiters:
            event.set()

        self._waiters.clear()

    async def wait(self) -> None:
        """Wait for a notification."""
        await checkpoint_if_cancelled()
        self._check_acquired()
        event = Event()
        self._waiters.append(event)
        self.release()
        try:
            await event.wait()
        except BaseException:
            if not event.is_set():
                self._waiters.remove(event)
            elif self._waiters:
                # This task was notified by could not act on it, so pass
                # it on to the next task
                self._waiters.popleft().set()

            raise
        finally:
            with CancelScope(shield=True):
                await self.acquire()

    async def wait_for(self, predicate: Callable[[], T]) -> T:
        """
        Wait until a predicate becomes true.

        :param predicate: a callable that returns a truthy value when the condition is
            met
        :return: the result of the predicate

        .. versionadded:: 4.11.0

        """
        while not (result := predicate()):
            await self.wait()

        return result

    def statistics(self) -> ConditionStatistics:
        """
        Return statistics about the current state of this condition.

        .. versionadded:: 3.0
        """
        return ConditionStatistics(len(self._waiters), self._lock.statistics())


class Semaphore:
    __slots__ = "__weakref__", "_fast_acquire"

    def __new__(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> Semaphore:
        try:
            return get_async_backend().create_semaphore(
                initial_value, max_value=max_value, fast_acquire=fast_acquire
            )
        except NoEventLoopError:
            return SemaphoreAdapter(initial_value, max_value=max_value)

    def __init__(
        self,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ):
        if not isinstance(initial_value, int):
            raise TypeError("initial_value must be an integer")
        if initial_value < 0:
            raise ValueError("initial_value must be >= 0")
        if max_value is not None:
            if not isinstance(max_value, int):
                raise TypeError("max_value must be an integer or None")
            if max_value < initial_value:
                raise ValueError(
                    "max_value must be equal to or higher than initial_value"
                )

        self._fast_acquire = fast_acquire

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    async def acquire(self) -> None:
        """Decrement the semaphore value, blocking if necessary."""
        raise NotImplementedError

    def acquire_nowait(self) -> None:
        """
        Acquire the underlying lock, without blocking.

        :raises ~anyio.WouldBlock: if the operation would block

        """
        raise NotImplementedError

    def release(self) -> None:
        """Increment the semaphore value."""
        raise NotImplementedError

    @property
    def value(self) -> int:
        """The current value of the semaphore."""
        raise NotImplementedError

    @property
    def max_value(self) -> int | None:
        """The maximum value of the semaphore."""
        raise NotImplementedError

    def statistics(self) -> SemaphoreStatistics:
        """
        Return statistics about the current state of this semaphore.

        .. versionadded:: 3.0
        """
        raise NotImplementedError


class SemaphoreAdapter(Semaphore):
    __slots__ = "_initial_value", "_internal_semaphore", "_max_value"

    def __new__(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> Self:
        return object.__new__(cls)

    def __init__(
        self,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> None:
        super().__init__(initial_value, max_value=max_value, fast_acquire=fast_acquire)
        self._internal_semaphore: Semaphore | None = None
        self._initial_value = initial_value
        self._max_value = max_value

    @property
    def _semaphore(self) -> Semaphore:
        if self._internal_semaphore is None:
            self._internal_semaphore = get_async_backend().create_semaphore(
                self._initial_value, max_value=self._max_value
            )

        return self._internal_semaphore

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def acquire_nowait(self) -> None:
        self._semaphore.acquire_nowait()

    def release(self) -> None:
        self._semaphore.release()

    @property
    def value(self) -> int:
        if self._internal_semaphore is None:
            return self._initial_value

        return self._semaphore.value

    @property
    def max_value(self) -> int | None:
        return self._max_value

    def statistics(self) -> SemaphoreStatistics:
        if self._internal_semaphore is None:
            return SemaphoreStatistics(tasks_waiting=0)

        return self._semaphore.statistics()


class CapacityLimiter:
    __slots__ = ("__weakref__",)

    def __new__(cls, total_tokens: float) -> CapacityLimiter:
        try:
            return get_async_backend().create_capacity_limiter(total_tokens)
        except NoEventLoopError:
            return CapacityLimiterAdapter(total_tokens)

    async def __aenter__(self) -> None:
        raise NotImplementedError

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    @property
    def total_tokens(self) -> float:
        """
        The total number of tokens available for borrowing.

        This is a read-write property. If the total number of tokens is increased, the
        proportionate number of tasks waiting on this limiter will be granted their
        tokens.

        .. versionchanged:: 3.0
            The property is now writable.
        .. versionchanged:: 4.12
            The value can now be set to 0.

        """
        raise NotImplementedError

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        raise NotImplementedError

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens that have currently been borrowed."""
        raise NotImplementedError

    @property
    def available_tokens(self) -> float:
        """The number of tokens currently available to be borrowed"""
        raise NotImplementedError

    def acquire_nowait(self) -> None:
        """
        Acquire a token for the current task without waiting for one to become
        available.

        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing

        """
        raise NotImplementedError

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        """
        Acquire a token without waiting for one to become available.

        :param borrower: the entity borrowing a token
        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing

        """
        raise NotImplementedError

    async def acquire(self) -> None:
        """
        Acquire a token for the current task, waiting if necessary for one to become
        available.

        """
        raise NotImplementedError

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """
        Acquire a token, waiting if necessary for one to become available.

        :param borrower: the entity borrowing a token

        """
        raise NotImplementedError

    def release(self) -> None:
        """
        Release the token held by the current task.

        :raises RuntimeError: if the current task has not borrowed a token from this
            limiter.

        """
        raise NotImplementedError

    def release_on_behalf_of(self, borrower: object) -> None:
        """
        Release the token held by the given borrower.

        :raises RuntimeError: if the borrower has not borrowed a token from this
            limiter.

        """
        raise NotImplementedError

    def statistics(self) -> CapacityLimiterStatistics:
        """
        Return statistics about the current state of this limiter.

        .. versionadded:: 3.0

        """
        raise NotImplementedError


class CapacityLimiterAdapter(CapacityLimiter):
    __slots__ = "_internal_limiter", "_total_tokens"

    def __new__(cls, total_tokens: float) -> Self:
        return object.__new__(cls)

    def __init__(self, total_tokens: float) -> None:
        self._internal_limiter: CapacityLimiter | None = None
        self.total_tokens = total_tokens

    @property
    def _limiter(self) -> CapacityLimiter:
        if self._internal_limiter is None:
            self._internal_limiter = get_async_backend().create_capacity_limiter(
                self._total_tokens
            )

        return self._internal_limiter

    async def __aenter__(self) -> None:
        await self._limiter.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return await self._limiter.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def total_tokens(self) -> float:
        if self._internal_limiter is None:
            return self._total_tokens

        return self._internal_limiter.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        if not isinstance(value, int) and not math.isinf(value):
            raise TypeError("total_tokens must be an int or math.inf")
        elif value < 0:
            raise ValueError("total_tokens must be >= 0")

        if self._internal_limiter is None:
            self._total_tokens = value
            return

        self._limiter.total_tokens = value

    @property
    def borrowed_tokens(self) -> int:
        if self._internal_limiter is None:
            return 0

        return self._internal_limiter.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        if self._internal_limiter is None:
            return self._total_tokens

        return self._internal_limiter.available_tokens

    def acquire_nowait(self) -> None:
        self._limiter.acquire_nowait()

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        self._limiter.acquire_on_behalf_of_nowait(borrower)

    async def acquire(self) -> None:
        await self._limiter.acquire()

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        await self._limiter.acquire_on_behalf_of(borrower)

    def release(self) -> None:
        self._limiter.release()

    def release_on_behalf_of(self, borrower: object) -> None:
        self._limiter.release_on_behalf_of(borrower)

    def statistics(self) -> CapacityLimiterStatistics:
        if self._internal_limiter is None:
            return CapacityLimiterStatistics(
                borrowed_tokens=0,
                total_tokens=self.total_tokens,
                borrowers=(),
                tasks_waiting=0,
            )

        return self._internal_limiter.statistics()


class ResourceGuard:
    """
    A context manager for ensuring that a resource is only used by a single task at a
    time.

    Entering this context manager while the previous has not exited it yet will trigger
    :exc:`BusyResourceError`.

    :param action: the action to guard against (visible in the :exc:`BusyResourceError`
        when triggered, e.g. "Another task is already {action} this resource")

    .. versionadded:: 4.1
    """

    __slots__ = "__weakref__", "_guarded", "action"

    def __init__(self, action: str = "using"):
        self.action: str = action
        self._guarded = False

    def __enter__(self) -> None:
        if self._guarded:
            raise BusyResourceError(self.action)

        self._guarded = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._guarded = False
<|endoftext|>from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable
from selectors import EVENT_READ, EVENT_WRITE, DefaultSelector
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike

_selector_lock = threading.Lock()
_selector: Selector | None = None


class Selector:
    def __init__(self) -> None:
        self._thread = threading.Thread(target=self.run, name="AnyIO socket selector")
        self._selector = DefaultSelector()
        self._send, self._receive = socket.socketpair()
        self._send.setblocking(False)
        self._receive.setblocking(False)
        # This somewhat reduces the amount of memory wasted queueing up data
        # for wakeups. With these settings, maximum number of 1-byte sends
        # before getting BlockingIOError:
        #   Linux 4.8: 6
        #   macOS (darwin 15.5): 1
        #   Windows 10: 525347
        # Windows you're weird. (And on Windows setting SNDBUF to 0 makes send
        # blocking, even on non-blocking sockets, so don't do that.)
        self._receive.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        self._send.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        # On Windows this is a TCP socket so this might matter. On other
        # platforms this fails b/c AF_UNIX sockets aren't actually TCP.
        try:
            self._send.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        self._selector.register(self._receive, EVENT_READ)
        self._closed = False

    def start(self) -> None:
        self._thread.start()
        threading._register_atexit(self._stop)  # type: ignore[attr-defined]

    def _stop(self) -> None:
        global _selector
        self._closed = True
        self._notify_self()
        self._send.close()
        self._thread.join()
        self._selector.unregister(self._receive)
        self._receive.close()
        self._selector.close()
        _selector = None
        assert not self._selector.get_map(), (
            "selector still has registered file descriptors after shutdown"
        )

    def _notify_self(self) -> None:
        try:
            self._send.send(b"\x00")
        except BlockingIOError:
            pass

    def add_reader(self, fd: FileDescriptorLike, callback: Callable[[], Any]) -> None:
        loop = asyncio.get_running_loop()
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, EVENT_READ, {EVENT_READ: (loop, callback)})
        else:
            if EVENT_READ in key.data:
                raise ValueError(
                    "this file descriptor is already registered for reading"
                )

            key.data[EVENT_READ] = loop, callback
            self._selector.modify(fd, key.events | EVENT_READ, key.data)

        self._notify_self()

    def add_writer(self, fd: FileDescriptorLike, callback: Callable[[], Any]) -> None:
        loop = asyncio.get_running_loop()
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, EVENT_WRITE, {EVENT_WRITE: (loop, callback)})
        else:
            if EVENT_WRITE in key.data:
                raise ValueError(
                    "this file descriptor is already registered for writing"
                )

            key.data[EVENT_WRITE] = loop, callback
            self._selector.modify(fd, key.events | EVENT_WRITE, key.data)

        self._notify_self()

    def remove_reader(self, fd: FileDescriptorLike) -> bool:
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return False

        if new_events := key.events ^ EVENT_READ:
            del key.data[EVENT_READ]
            self._selector.modify(fd, new_events, key.data)
        else:
            self._selector.unregister(fd)

        return True

    def remove_writer(self, fd: FileDescriptorLike) -> bool:
        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return False

        if new_events := key.events ^ EVENT_WRITE:
            del key.data[EVENT_WRITE]
            self._selector.modify(fd, new_events, key.data)
        else:
            self._selector.unregister(fd)

        return True

    def run(self) -> None:
        while not self._closed:
            for key, events in self._selector.select():
                if key.fileobj is self._receive:
                    try:
                        while self._receive.recv(4096):
                            pass
                    except BlockingIOError:
                        pass

                    continue

                if events & EVENT_READ:
                    loop, callback = key.data[EVENT_READ]
                    self.remove_reader(key.fd)
                    try:
                        loop.call_soon_threadsafe(callback)
                    except RuntimeError:
                        pass  # the loop was already closed

                if events & EVENT_WRITE:
                    loop, callback = key.data[EVENT_WRITE]
                    self.remove_writer(key.fd)
                    try:
                        loop.call_soon_threadsafe(callback)
                    except RuntimeError:
                        pass  # the loop was already closed


def get_selector() -> Selector:
    global _selector

    with _selector_lock:
        if _selector is None:
            _selector = Selector()
            _selector.start()

        return _selector
<|endoftext|>from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from signal import Signals

from ._eventloop import get_async_backend


def open_signal_receiver(
    *signals: Signals,
) -> AbstractContextManager[AsyncIterator[Signals]]:
    """
    Start receiving operating system signals.

    :param signals: signals to receive (e.g. ``signal.SIGINT``)
    :return: an asynchronous context manager for an asynchronous iterator which yields
        signal numbers
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. warning:: Windows does not support signals natively so it is best to avoid
        relying on this in cross-platform applications.

    .. warning:: On asyncio, this permanently replaces any previous signal handler for
        the given signals, as set via :meth:`~asyncio.loop.add_signal_handler`.

    """
    return get_async_backend().open_signal_receiver(*signals)
<|endoftext|>from __future__ import annotations

import math
import sys
from collections.abc import (
    Coroutine,
    Generator,
)
from contextlib import contextmanager
from enum import Enum, auto
from inspect import iscoroutine
from types import TracebackType
from typing import Any, Generic, final

from ..abc import TaskGroup, TaskStatus
from ._eventloop import get_async_backend, get_cancelled_exc_class
from ._exceptions import TaskCancelled, TaskFailed, TaskNotFinished

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 11):
    from typing import Never, Self, TypeVarTuple
else:
    from typing_extensions import Never, Self, TypeVarTuple

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_startval_co = TypeVar("T_startval_co", covariant=True, default=Never)
PosArgsT = TypeVarTuple("PosArgsT")


class _IgnoredTaskStatus(TaskStatus[object]):
    def started(self, value: object = None) -> None:
        pass


TASK_STATUS_IGNORED = _IgnoredTaskStatus()


class CancelScope:
    """
    Wraps a unit of work that can be made separately cancellable.

    :param deadline: The time (clock value) when this scope is cancelled automatically
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread
    """

    __slots__ = ("__weakref__",)

    def __new__(
        cls, *, deadline: float = math.inf, shield: bool = False
    ) -> CancelScope:
        return get_async_backend().create_cancel_scope(shield=shield, deadline=deadline)

    def cancel(self, reason: str | None = None) -> None:
        """
        Cancel this scope immediately.

        :param reason: a message describing the reason for the cancellation

        """
        raise NotImplementedError

    @property
    def deadline(self) -> float:
        """
        The time (clock value) when this scope is cancelled automatically.

        Will be ``float('inf')`` if no timeout has been set.

        """
        raise NotImplementedError

    @deadline.setter
    def deadline(self, value: float) -> None:
        raise NotImplementedError

    @property
    def cancel_called(self) -> bool:
        """``True`` if :meth:`cancel` has been called."""
        raise NotImplementedError

    @property
    def cancelled_caught(self) -> bool:
        """
        ``True`` if this scope suppressed a cancellation exception it itself raised.

        This is typically used to check if any work was interrupted, or to see if the
        scope was cancelled due to its deadline being reached. The value will, however,
        only be ``True`` if the cancellation was triggered by the scope itself (and not
        an outer scope).

        """
        raise NotImplementedError

    @property
    def shield(self) -> bool:
        """
        ``True`` if this scope is shielded from external cancellation.

        While a scope is shielded, it will not receive cancellations from outside.

        """
        raise NotImplementedError

    @shield.setter
    def shield(self, value: bool) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        raise NotImplementedError


@contextmanager
def fail_at(
    deadline: float | None, shield: bool = False, reason: str | None = None
) -> Generator[CancelScope, None, None]:
    """
    Create a context manager which raises a :class:`TimeoutError` if the code in the
    enclosing context does not finish by the given deadline.

    :param deadline: the deadline before raising the exception, or
        ``None`` to disable the timeout
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :param reason: explanation for timeout to add to the message of a raised `TimeoutError`
    :return: a context manager that yields a cancel scope
    :rtype: :class:`~typing.ContextManager`\\[:class:`~anyio.CancelScope`\\]
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. versionadded:: 4.15.0

    """
    current_time = get_async_backend().current_time
    effective_deadline = math.inf if deadline is None else deadline
    with get_async_backend().create_cancel_scope(
        deadline=effective_deadline, shield=shield
    ) as cancel_scope:
        yield cancel_scope

    if cancel_scope.cancelled_caught and current_time() >= cancel_scope.deadline:
        raise TimeoutError(reason) if reason else TimeoutError


@contextmanager
def fail_after(
    delay: float | None, shield: bool = False, reason: str | None = None
) -> Generator[CancelScope, None, None]:
    """
    Create a context manager which raises a :class:`TimeoutError` if the code in the
    enclosing context block does not finish in time.

    :param delay: maximum allowed time (in seconds) before raising the exception, or
        ``None`` to disable the timeout
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :param reason: explanation for timeout to add to the message of a raised `TimeoutError`
    :return: a context manager that yields a cancel scope
    :rtype: :class:`~typing.ContextManager`\\[:class:`~anyio.CancelScope`\\]
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    current_time = get_async_backend().current_time
    deadline = (current_time() + delay) if delay is not None else math.inf
    with fail_at(deadline, shield=shield, reason=reason) as scope:
        yield scope


def move_on_at(deadline: float | None, shield: bool = False) -> CancelScope:
    """
    Create a cancel scope with a deadline that expires after the given delay.

    :param deadline: the deadline before exiting the context block, or
        ``None`` to disable the timeout
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :return: a cancel scope
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. versionadded:: 4.15.0

    """
    return get_async_backend().create_cancel_scope(
        deadline=deadline if deadline is not None else math.inf, shield=shield
    )


def move_on_after(delay: float | None, shield: bool = False) -> CancelScope:
    """
    Create a cancel scope with a deadline that expires after the given delay.

    :param delay: maximum allowed time (in seconds) before exiting the context block, or
        ``None`` to disable the timeout
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :return: a cancel scope
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. note:: Unlike with :func:`fail_after`, the timer starts when this function is
        called, not when the context manager is entered. This will be changed in v5.0.

    """
    deadline = (
        (get_async_backend().current_time() + delay) if delay is not None else math.inf
    )
    return get_async_backend().create_cancel_scope(deadline=deadline, shield=shield)


def current_effective_deadline() -> float:
    """
    Return the nearest deadline among all the cancel scopes effective for the current
    task.

    :return: a clock value from the event loop's internal clock (or ``float('inf')`` if
        there is no deadline in effect, or ``float('-inf')`` if the current scope has
        been cancelled)
    :rtype: float
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().current_effective_deadline()


def create_task_group() -> TaskGroup:
    """
    Create a task group.

    :return: a task group
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().create_task_group()


@final
class TaskHandle(Generic[T_co, T_startval_co]):
    """
    Returned from the task-spawning methods of :class:`TaskGroup`. Can be awaited on to
    get the return value of the task (or the raised exception). If the task was
    terminated by a :exc:`BaseException`, :exc:`TaskFailed` will be raised (or its
    subclass :exc:`TaskCancelled` if the task was cancelled).

    .. versionadded:: 4.14.0
    """

    class Status(Enum):
        """
        The status of a task handle.

        .. attribute:: PENDING

            The task has not finished yet.
        .. attribute:: FINISHED

            The task has finished with a return value.
        .. attribute:: CANCELLING

            The task has been cancelled but has not finished yet.
        .. attribute:: CANCELLED

            The task was cancelled and has finished since.
        .. attribute:: FAILED

            The task raised an exception.
        """

        PENDING = auto()
        FINISHED = auto()
        CANCELLING = auto()
        CANCELLED = auto()
        FAILED = auto()

    __slots__ = (
        "__weakref__",
        "_cancel_scope",
        "_coro",
        "_exception",
        "_finished_event",
        "_name",
        "_return_value",
        "_start_value",
    )

    _return_value: T_co
    _start_value: T_startval_co

    def __init__(
        self,
        coro: Coroutine[Any, Any, T_co],
        name: object,
        cancel_scope: CancelScope | None = None,
    ) -> None:
        from ._synchronization import Event

        self._coro = coro
        self._cancel_scope = cancel_scope if cancel_scope is not None else CancelScope()
        self._finished_event = Event()
        self._exception: BaseException | None = None

        if name is not None:
            self._name = str(name)
        elif iscoroutine(coro):
            self._name = coro.__qualname__
        else:
            self._name = str(coro)  # coroutine-like object (e.g. asend() objects)

    async def _run_coro(self) -> None:
        __tracebackhide__ = True

        with self._cancel_scope:
            try:
                retval = await self._coro
            except BaseException as exc:
                self._exception = exc
                raise
            else:
                self._return_value = retval
            finally:
                self._finished_event.set()
                del self  # Break the reference cycle

    def cancel(self) -> None:
        """
        Set the task to a cancelled state.

        This will interrupt any interruptible asynchronous operation, and will cause
        any further awaits on this task to get immediately cancelled, unless done in
        a shielded cancel scope.

        If the task has already finished, this method has no effect.
        """
        if not self._finished_event.is_set():
            self._cancel_scope.cancel()

    @property
    def coro(self) -> Coroutine[Any, Any, T_co]:
        """
        The coroutine object that was passed to one of the task-spawning methods in
        :class:`TaskGroup`.
        """
        return self._coro

    @property
    def status(self) -> TaskHandle.Status:
        """
        The current status of the task.

        Every task starts in the :attr:`~TaskHandle.Status.PENDING` state.
        If a task is cancelled while in this state, it will transition to the
        :attr:`~TaskHandle.Status.CANCELLING` state. When the task finishes, it will
        transition to one of the three final states (
        :attr:`~TaskHandle.Status.FINISHED`, :attr:`~TaskHandle.Status.FAILED`, or
        :attr:`~TaskHandle.Status.CANCELLING`) depending on the exception the task
        raised, if any. No other status transitions will happen.
        """
        if not self._finished_event.is_set():
            if self._cancel_scope.cancel_called:
                return TaskHandle.Status.CANCELLING
            else:
                return TaskHandle.Status.PENDING
        elif self._exception is not None:
            if isinstance(self._exception, get_cancelled_exc_class()):
                return TaskHandle.Status.CANCELLED
            else:
                return TaskHandle.Status.FAILED
        else:
            return TaskHandle.Status.FINISHED

    @property
    def name(self) -> str:
        """The name of the task."""
        return self._name

    @property
    def exception(self) -> BaseException | None:
        """
        The exception raised by the task, or ``None`` if it finished without raising.

        :raises TaskNotFinished: if the task has not finished yet
        :raises TaskCancelled: if the task was cancelled

        """
        match self.status:
            case TaskHandle.Status.PENDING:
                raise TaskNotFinished("the task has not finished yet")
            case TaskHandle.Status.FINISHED:
                return None
            case TaskHandle.Status.CANCELLING:
                raise TaskCancelled("the task was cancelled")
            case TaskHandle.Status.CANCELLED:
                raise TaskCancelled("the task was cancelled") from self._exception
            case TaskHandle.Status.FAILED:
                return self._exception

    @property
    def return_value(self) -> T_co:
        """
        The return value of the task.

        :raises TaskNotFinished: if the task has not finished yet
        :raises TaskCancelled: if the task was cancelled
        :raises TaskFailed: if the task raised an exception

        """
        match self.status:
            case TaskHandle.Status.PENDING:
                raise TaskNotFinished("the task has not finished yet")
            case TaskHandle.Status.FINISHED:
                return self._return_value
            case TaskHandle.Status.CANCELLING:
                raise TaskCancelled("the task was cancelled")
            case TaskHandle.Status.CANCELLED:
                raise TaskCancelled("the task was cancelled") from self._exception
            case TaskHandle.Status.FAILED:
                raise TaskFailed("the task raised an exception") from self._exception

    @property
    def start_value(self) -> T_startval_co:
        """
        The value passed to :meth:`task_status.started() <.abc.TaskStatus.started>`,

        :raises RuntimeError: if the task was not started with :meth:`TaskGroup.start()
            <.abc.TaskGroup.start>`
        """
        try:
            return self._start_value
        except AttributeError:
            raise RuntimeError(
                "the task was not started with TaskGroup.start()"
            ) from None

    async def wait(self) -> None:
        """
        Wait for the task to finish.

        This method will return as soon as the task has finished, no matter how it
        happened.
        """
        await self._finished_event.wait()

    def __await__(self) -> Generator[Any, Any, T_co]:
        yield from self._finished_event.wait().__await__()
        return self.return_value

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.status.name.lower()} "
            f"name={self._name!r} coro={self._coro!r}>"
        )
<|endoftext|>from __future__ import annotations

import sys
from collections.abc import Generator
from textwrap import dedent
from typing import Any

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


class BrokenResourceError(Exception):
    """
    Raised when trying to use a resource that has been rendered unusable due to external
    causes (e.g. a send stream whose peer has disconnected).
    """


class BrokenWorkerProcess(Exception):
    """
    Raised by :meth:`~anyio.to_process.run_sync` if the worker process terminates abruptly or
    otherwise misbehaves.
    """


class BrokenWorkerInterpreter(Exception):
    """
    Raised by :meth:`~anyio.to_interpreter.run_sync` if an unexpected exception is
    raised in the subinterpreter.
    """

    def __init__(self, excinfo: Any):
        # This was adapted from concurrent.futures.interpreter.ExecutionFailed
        msg = excinfo.formatted
        if not msg:
            if excinfo.type and excinfo.msg:
                msg = f"{excinfo.type.__name__}: {excinfo.msg}"
            else:
                msg = excinfo.type.__name__ or excinfo.msg

        super().__init__(msg)
        self.excinfo = excinfo

    def __str__(self) -> str:
        try:
            formatted = self.excinfo.errdisplay
        except Exception:
            return super().__str__()
        else:
            return dedent(
                f"""
                {super().__str__()}

                Uncaught in the interpreter:

                {formatted}
                """.strip()
            )


class BusyResourceError(Exception):
    """
    Raised when two tasks are trying to read from or write to the same resource
    concurrently.
    """

    def __init__(self, action: str):
        super().__init__(f"Another task is already {action} this resource")


class ClosedResourceError(Exception):
    """Raised when trying to use a resource that has been closed."""


class ConnectionFailed(OSError):
    """
    Raised when a connection attempt fails.

    .. note:: This class inherits from :exc:`OSError` for backwards compatibility.
    """


def iterate_exceptions(
    exception: BaseException,
) -> Generator[BaseException, None, None]:
    if isinstance(exception, BaseExceptionGroup):
        for exc in exception.exceptions:
            yield from iterate_exceptions(exc)
    else:
        yield exception


class DelimiterNotFound(Exception):
    """
    Raised during
    :meth:`~anyio.streams.buffered.BufferedByteReceiveStream.receive_until` if the
    maximum number of bytes has been read without the delimiter being found.
    """

    def __init__(self, max_bytes: int) -> None:
        super().__init__(
            f"The delimiter was not found among the first {max_bytes} bytes"
        )


class EndOfStream(Exception):
    """
    Raised when trying to read from a stream that has been closed from the other end.
    """


class IncompleteRead(Exception):
    """
    Raised during
    :meth:`~anyio.streams.buffered.BufferedByteReceiveStream.receive_exactly` or
    :meth:`~anyio.streams.buffered.BufferedByteReceiveStream.receive_until` if the
    connection is closed before the requested amount of bytes has been read.
    """

    def __init__(self) -> None:
        super().__init__(
            "The stream was closed before the read operation could be completed"
        )


class TypedAttributeLookupError(LookupError):
    """
    Raised by :meth:`~anyio.TypedAttributeProvider.extra` when the given typed attribute
    is not found and no default value has been given.
    """


class WouldBlock(Exception):
    """Raised by ``X_nowait`` functions if ``X()`` would block."""


class NoEventLoopError(RuntimeError):
    """
    Raised by several functions that require an event loop to be running in the current
    thread when there is no running event loop.

    This is also raised by :func:`.from_thread.run` and :func:`.from_thread.run_sync`
    if not calling from an AnyIO worker thread, and no ``token`` was passed.
    """


class RunFinishedError(RuntimeError):
    """
    Raised by :func:`.from_thread.run` and :func:`.from_thread.run_sync` if the event
    loop associated with the explicitly passed token has already finished.
    """

    def __init__(self) -> None:
        super().__init__(
            "The event loop associated with the given token has already finished"
        )


class TaskFailed(Exception):
    """
    Raised when awaiting on, or attempting to access the return value of, a
    :class:`.TaskHandle` that raised an exception.
    """


class TaskCancelled(TaskFailed):
    """
    Raised when awaiting on, or attempting to access the return value of, a
    :class:`.TaskHandle` that was cancelled.
    """


class TaskNotFinished(Exception):
    """
    Raised when attempting to access the return value or exception of a
    :class:`.TaskHandle` that is still pending completion.
    """


class FutureFailed(Exception):
    """
    Raised when awaiting on, or attempting to access the return value of, a
    :class:`.Future` that raised an exception.
    """


class FutureCancelled(FutureFailed):
    """
    Raised when attempting to access the return value or exception of a
    :class:`.Future` that was cancelled.
    """


class FutureNotFinished(Exception):
    """
    Raised when attempting to access the return value or exception of a
    :class:`.Future` that is still pending completion.
    """


class FutureAlreadyFinished(Exception):
    """
    Raised when attempting set a result of or await a
    :class:`.Future` that has already completed.
    """
<|endoftext|>from __future__ import annotations

from collections.abc import Generator
from enum import Enum, auto
from typing import Any, Generic, TypeVar

from ._exceptions import (
    FutureAlreadyFinished,
    FutureCancelled,
    FutureFailed,
    FutureNotFinished,
)
from ._synchronization import Event

T = TypeVar("T")


class Future(Generic[T]):
    """
    An awaitable object that works similar to a :class:`asyncio.Future` but
    with similar characteristics to a :class:`.TaskHandle`.
    """

    class Status(Enum):
        """
        The status of a future handle.

        .. attribute:: PENDING

            The future has not finished yet.
        .. attribute:: FINISHED

            The future has finished with a return value.
        .. attribute:: CANCELLED

            The future was cancelled and has finished since.
        .. attribute:: FAILED

            The future raised an exception.
        """

        PENDING = auto()
        FINISHED = auto()
        CANCELLED = auto()
        FAILED = auto()

    __slots__ = (
        "_cancelled",
        "_exception",
        "_finished_event",
        "_name",
        "_return_value",
    )
    _return_value: T

    def __init__(self, *, name: str | None = None) -> None:
        self._finished_event = Event()
        self._exception: BaseException | None = None
        self._cancelled: bool = False
        self._name = name

    def _check_pending(self) -> None:
        """Shortcut for checking if a Future is pending

        :raises FutureAlreadyFinished: if future was already given a result or exception.
        :raises FutureCancelled: if future has been cancelled previously.
        """
        match self.status:
            case Future.Status.PENDING:
                return
            case Future.Status.FINISHED:
                raise FutureAlreadyFinished("future has already finished")
            case Future.Status.FAILED:
                raise FutureAlreadyFinished("future already failed")
            case Future.Status.CANCELLED:
                raise FutureCancelled("future was cancelled")

    async def wait(self) -> None:
        """
        Waits for the future to finish.

        This method will attempt to wait for a result or exception
        """
        await self._finished_event.wait()

    def cancel(self) -> None:
        """Cancels a pending `.Future` object

        Does nothing if the Future was already finished.
        """
        if self.status is Future.Status.PENDING:
            self._cancelled = True
            self._finished_event.set()

    @property
    def exception(self) -> BaseException | None:
        """
        The exception value of a `.Future`

        :raises FutureNotFinished: if future is still pending
        :raises FutureCancelled: if future was cancelled
        :returns: None if future succeeds with a result sent instead
            otherwise this will be an exception
        """

        match self.status:
            case Future.Status.PENDING:
                raise FutureNotFinished("the future has not finished yet")
            case Future.Status.FINISHED:
                return None
            case Future.Status.CANCELLED:
                raise FutureCancelled("the future was cancelled")
            case Future.Status.FAILED:
                return self._exception

    @exception.setter
    def exception(self, exception: BaseException) -> None:
        """Send exception for a `.Future` object

        :raises FutureAlreadyFinished: if future was already given a result or exception.
        :raises FutureCancelled: if future has been cancelled previously.
        """
        self._check_pending()
        self._exception = exception
        self._finished_event.set()

    @property
    def return_value(self) -> T:
        """
        The return value of the future.

        :raises FutureNotFinished: if the future has not finished yet
        :raises FutureCancelled: if the future was cancelled
        :raises FutureFailed: if the future raised an exception

        """
        match self.status:
            case Future.Status.PENDING:
                raise FutureNotFinished("the future has not finished yet")
            case Future.Status.FINISHED:
                return self._return_value
            case Future.Status.CANCELLED:
                raise FutureCancelled("the future was cancelled")
            case Future.Status.FAILED:
                raise FutureFailed(
                    "the future raised an exception"
                ) from self._exception

    @return_value.setter
    def return_value(self, value: T) -> None:
        """
        Send pending result for a `.Future` object

        :raises FutureAlreadyFinished: if future was already given a result or exception.
        :raises FutureCancelled: if future has been cancelled previously.
        """
        self._check_pending()
        self._return_value = value
        self._finished_event.set()

    @property
    def status(self) -> Future.Status:
        """
        The current status of a future.

        Every future starts in the :attr:`~Future.Status.PENDING` state.
        If a future is cancelled while in this state, it will transition to the
        :attr:`Future.Status.CANCELLED` state. When the task finishes, it will
        transition to one of the three final states (
        :attr:`Future.Status.FINISHED`, :attr:`Future.Status.FAILED`, or
        :attr:`Future.Status.CANCELLED`) depending on the exception the task
        raised, if any. No other status transitions will happen.
        """
        if not self._finished_event.is_set():
            return Future.Status.PENDING
        elif self._cancelled:
            return Future.Status.CANCELLED
        elif self._exception is not None:
            return Future.Status.FAILED
        else:
            return Future.Status.FINISHED

    def __await__(self) -> Generator[Any, Any, T]:
        yield from self.wait().__await__()
        return self.return_value

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.status.name.lower()} "
            f"name={self._name!r}>"
        )
<|endoftext|>from __future__ import annotations

import errno
import os
import socket
import ssl
import stat
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from os import PathLike, chmod
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from .. import ConnectionFailed, to_thread
from ..abc import (
    ByteStreamConnectable,
    ConnectedUDPSocket,
    ConnectedUNIXDatagramSocket,
    IPAddressType,
    IPSockAddrType,
    SocketListener,
    SocketStream,
    UDPSocket,
    UNIXDatagramSocket,
    UNIXSocketStream,
)
from ..streams.stapled import MultiListener
from ..streams.tls import TLSConnectable, TLSStream
from ._eventloop import get_async_backend
from ._resources import aclose_forcefully
from ._synchronization import Event
from ._tasks import create_task_group, move_on_after

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike
else:
    FileDescriptorLike = object

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info < (3, 13):
    from typing_extensions import deprecated
else:
    from warnings import deprecated

IPPROTO_IPV6 = getattr(socket, "IPPROTO_IPV6", 41)  # https://bugs.python.org/issue29515

AnyIPAddressFamily = Literal[
    AddressFamily.AF_UNSPEC, AddressFamily.AF_INET, AddressFamily.AF_INET6
]
IPAddressFamily = Literal[AddressFamily.AF_INET, AddressFamily.AF_INET6]


def idna2008_resolve(host: str) -> bytes:
    try:
        return host.encode("ascii")
    except UnicodeEncodeError:
        import idna

        return idna.encode(host, uts46=True)


# tls_hostname given
@overload
async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = ...,
    local_port: int | None = ...,
    ssl_context: ssl.SSLContext | None = ...,
    tls_standard_compatible: bool = ...,
    tls_hostname: str,
    happy_eyeballs_delay: float = ...,
) -> TLSStream: ...


# ssl_context given
@overload
async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = ...,
    local_port: int | None = ...,
    ssl_context: ssl.SSLContext,
    tls_standard_compatible: bool = ...,
    tls_hostname: str | None = ...,
    happy_eyeballs_delay: float = ...,
) -> TLSStream: ...


# tls=True
@overload
async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = ...,
    local_port: int | None = ...,
    tls: Literal[True],
    ssl_context: ssl.SSLContext | None = ...,
    tls_standard_compatible: bool = ...,
    tls_hostname: str | None = ...,
    happy_eyeballs_delay: float = ...,
) -> TLSStream: ...


# tls=False
@overload
async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = ...,
    local_port: int | None = ...,
    tls: Literal[False],
    ssl_context: ssl.SSLContext | None = ...,
    tls_standard_compatible: bool = ...,
    tls_hostname: str | None = ...,
    happy_eyeballs_delay: float = ...,
) -> SocketStream: ...


# No TLS arguments
@overload
async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = ...,
    local_port: int | None = ...,
    happy_eyeballs_delay: float = ...,
) -> SocketStream: ...


async def connect_tcp(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    local_host: IPAddressType | None = None,
    local_port: int | None = None,
    tls: bool = False,
    ssl_context: ssl.SSLContext | None = None,
    tls_standard_compatible: bool = True,
    tls_hostname: str | None = None,
    happy_eyeballs_delay: float = 0.25,
) -> SocketStream | TLSStream:
    """
    Connect to a host using the TCP protocol.

    This function implements the stateless version of the Happy Eyeballs algorithm (RFC
    6555). If ``remote_host`` is a host name that resolves to multiple IP addresses,
    each one is tried until one connection attempt succeeds. If the first attempt does
    not connected within 250 milliseconds, a second attempt is started using the next
    address in the list, and so on. On IPv6 enabled systems, an IPv6 address (if
    available) is tried first.

    When the connection has been established, a TLS handshake will be done if either
    ``ssl_context`` or ``tls_hostname`` is not ``None``, or if ``tls`` is ``True``.

    :param remote_host: the IP address or host name to connect to
    :param remote_port: port on the target host to connect to
    :param local_host: the interface address or name to bind the socket to before
        connecting
    :param local_port: the local port to bind to (requires ``local_host`` to also be
        set)
    :param tls: ``True`` to do a TLS handshake with the connected stream and return a
        :class:`~anyio.streams.tls.TLSStream` instead
    :param ssl_context: the SSL context object to use (if omitted, a default context is
        created)
    :param tls_standard_compatible: If ``True``, performs the TLS shutdown handshake
        before closing the stream and requires that the server does this as well.
        Otherwise, :exc:`~ssl.SSLEOFError` may be raised during reads from the stream.
        Some protocols, such as HTTP, require this option to be ``False``.
        See :meth:`~ssl.SSLContext.wrap_socket` for details.
    :param tls_hostname: host name to check the server certificate against (defaults to
        the value of ``remote_host``)
    :param happy_eyeballs_delay: delay (in seconds) before starting the next connection
        attempt
    :return: a socket stream object if no TLS handshake was done, otherwise a TLS stream
    :raises ConnectionFailed: if the connection fails

    """
    # Placed here due to https://github.com/python/mypy/issues/7057
    connected_stream: SocketStream | None = None

    async def try_connect(remote_host: str, event: Event) -> None:
        nonlocal connected_stream
        try:
            stream = await asynclib.connect_tcp(remote_host, remote_port, local_address)
        except OSError as exc:
            oserrors.append(exc)
            return
        else:
            if connected_stream is None:
                connected_stream = stream
                tg.cancel_scope.cancel()
            else:
                await stream.aclose()
        finally:
            event.set()

    asynclib = get_async_backend()
    local_address: IPSockAddrType | None = None
    family = socket.AF_UNSPEC
    if local_host:
        gai_res = await getaddrinfo(str(local_host), local_port)
        family, *_, local_address = gai_res[0]

    target_host = str(remote_host)
    try:
        addr_obj = ip_address(remote_host)
    except ValueError:
        addr_obj = None

    if addr_obj is not None:
        if isinstance(addr_obj, IPv6Address):
            target_addrs = [(socket.AF_INET6, addr_obj.compressed)]
        else:
            target_addrs = [(socket.AF_INET, addr_obj.compressed)]
    else:
        # getaddrinfo() will raise an exception if name resolution fails
        gai_res = await getaddrinfo(
            target_host, remote_port, family=family, type=socket.SOCK_STREAM
        )

        # Organize the list so that the first address is an IPv6 address (if available)
        # and the second one is an IPv4 addresses. The rest can be in whatever order.
        v6_found = v4_found = False
        target_addrs = []
        for af, *_, sa in gai_res:
            if af == socket.AF_INET6 and not v6_found:
                v6_found = True
                target_addrs.insert(0, (af, sa[0]))
            elif af == socket.AF_INET and not v4_found and v6_found:
                v4_found = True
                target_addrs.insert(1, (af, sa[0]))
            else:
                target_addrs.append((af, sa[0]))

    oserrors: list[OSError] = []
    try:
        async with create_task_group() as tg:
            for _af, addr in target_addrs:
                event = Event()
                tg.start_soon(try_connect, addr, event)
                with move_on_after(happy_eyeballs_delay):
                    await event.wait()

        if connected_stream is None:
            cause = (
                oserrors[0]
                if len(oserrors) == 1
                else ExceptionGroup("multiple connection attempts failed", oserrors)
            )
            raise OSError("All connection attempts failed") from cause
    finally:
        oserrors.clear()

    if tls or tls_hostname or ssl_context:
        try:
            return await TLSStream.wrap(
                connected_stream,
                server_side=False,
                hostname=tls_hostname or str(remote_host),
                ssl_context=ssl_context,
                standard_compatible=tls_standard_compatible,
            )
        except BaseException:
            await aclose_forcefully(connected_stream)
            raise

    return connected_stream


async def connect_unix(path: str | bytes | PathLike[Any]) -> UNIXSocketStream:
    """
    Connect to the given UNIX socket.

    Not available on Windows.

    :param path: path to the socket
    :return: a socket stream object
    :raises ConnectionFailed: if the connection fails

    """
    path = os.fspath(path)
    return await get_async_backend().connect_unix(path)


async def create_tcp_listener(
    *,
    local_host: IPAddressType | None = None,
    local_port: int = 0,
    family: AnyIPAddressFamily = socket.AddressFamily.AF_UNSPEC,
    backlog: int = 65536,
    reuse_port: bool = False,
) -> MultiListener[SocketStream]:
    """
    Create a TCP socket listener.

    :param local_port: port number to listen on
    :param local_host: IP address of the interface to listen on. If omitted, listen on
        all IPv4 and IPv6 interfaces. To listen on all interfaces on a specific address
        family, use ``0.0.0.0`` for IPv4 or ``::`` for IPv6.
    :param family: address family (used if ``local_host`` was omitted)
    :param backlog: maximum number of queued incoming connections (up to a maximum of
        2**16, or 65536)
    :param reuse_port: ``True`` to allow multiple sockets to bind to the same
        address/port (not supported on Windows)
    :return: a multi-listener object containing one or more socket listeners
    :raises OSError: if there's an error creating a socket, or binding to one or more
        interfaces failed

    """
    asynclib = get_async_backend()
    backlog = min(backlog, 65536)
    local_host = str(local_host) if local_host is not None else None

    def setup_raw_socket(
        fam: AddressFamily,
        bind_addr: tuple[str, int] | tuple[str, int, int, int],
        *,
        v6only: bool = True,
    ) -> socket.socket:
        sock = socket.socket(fam)
        try:
            sock.setblocking(False)

            if fam == AddressFamily.AF_INET6:
                sock.setsockopt(IPPROTO_IPV6, socket.IPV6_V6ONLY, v6only)

            # For Windows, enable exclusive address use. For others, enable address
            # reuse.
            if sys.platform == "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            if reuse_port:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

            # Workaround for #554
            if fam == socket.AF_INET6 and "%" in bind_addr[0]:
                addr, scope_id = bind_addr[0].split("%", 1)
                bind_addr = (addr, bind_addr[1], 0, int(scope_id))

            sock.bind(bind_addr)
            sock.listen(backlog)
        except BaseException:
            sock.close()
            raise

        return sock

    # We passing type=0 on non-Windows platforms as a workaround for a uvloop bug
    # where we don't get the correct scope ID for IPv6 link-local addresses when passing
    # type=socket.SOCK_STREAM to getaddrinfo():
    # https://github.com/MagicStack/uvloop/issues/539
    gai_res = await getaddrinfo(
        local_host,
        local_port,
        family=family,
        type=socket.SOCK_STREAM if sys.platform == "win32" else 0,
        flags=socket.AI_PASSIVE | socket.AI_ADDRCONFIG,
    )

    # The set comprehension is here to work around a glibc bug:
    # https://sourceware.org/bugzilla/show_bug.cgi?id=14969
    sockaddrs = sorted({res for res in gai_res if res[1] == SocketKind.SOCK_STREAM})

    # Special case for dual-stack binding on the "any" interface
    if (
        local_host is None
        and family == AddressFamily.AF_UNSPEC
        and socket.has_dualstack_ipv6()
        and any(fam == AddressFamily.AF_INET6 for fam, *_ in gai_res)
    ):
        raw_socket = setup_raw_socket(
            AddressFamily.AF_INET6, ("::", local_port), v6only=False
        )
        listener = asynclib.create_tcp_listener(raw_socket)
        return MultiListener([listener])

    errors: list[OSError] = []
    try:
        for _ in range(len(sockaddrs)):
            listeners: list[SocketListener] = []
            bound_ephemeral_port = local_port
            try:
                for fam, *_, sockaddr in sockaddrs:
                    sockaddr = sockaddr[0], bound_ephemeral_port, *sockaddr[2:]
                    raw_socket = setup_raw_socket(fam, sockaddr)

                    # Store the assigned port if an ephemeral port was requested, so
                    # we'll bind to the same port on all interfaces
                    if local_port == 0 and len(gai_res) > 1:
                        bound_ephemeral_port = raw_socket.getsockname()[1]

                    listeners.append(asynclib.create_tcp_listener(raw_socket))
            except BaseException as exc:
                for listener in listeners:
                    await listener.aclose()

                # If an ephemeral port was requested but binding the assigned port
                # failed for another interface, rotate the address list and try again
                if (
                    isinstance(exc, OSError)
                    and exc.errno == errno.EADDRINUSE
                    and local_port == 0
                    and bound_ephemeral_port
                ):
                    errors.append(exc)
                    sockaddrs.append(sockaddrs.pop(0))
                    continue

                raise

            return MultiListener(listeners)

        raise OSError(
            f"Could not create {len(sockaddrs)} listeners with a consistent port"
        ) from ExceptionGroup("Several bind attempts failed", errors)
    finally:
        del errors  # Prevent reference cycles


async def create_unix_listener(
    path: str | bytes | PathLike[Any],
    *,
    mode: int | None = None,
    backlog: int = 65536,
) -> SocketListener:
    """
    Create a UNIX socket listener.

    Not available on Windows.

    :param path: path of the socket
    :param mode: permissions to set on the socket
    :param backlog: maximum number of queued incoming connections (up to a maximum of
        2**16, or 65536)
    :return: a listener object

    .. versionchanged:: 3.0
        If a socket already exists on the file system in the given path, it will be
        removed first.

    """
    backlog = min(backlog, 65536)
    raw_socket = await setup_unix_local_socket(path, mode, socket.SOCK_STREAM)
    try:
        raw_socket.listen(backlog)
        return get_async_backend().create_unix_listener(raw_socket)
    except BaseException:
        raw_socket.close()
        raise


async def create_udp_socket(
    family: AnyIPAddressFamily = AddressFamily.AF_UNSPEC,
    *,
    local_host: IPAddressType | None = None,
    local_port: int = 0,
    reuse_port: bool = False,
) -> UDPSocket:
    """
    Create a UDP socket.

    If ``port`` has been given, the socket will be bound to this port on the local
    machine, making this socket suitable for providing UDP based services.

    :param family: address family (``AF_INET`` or ``AF_INET6``) – automatically
        determined from ``local_host`` if omitted
    :param local_host: IP address or host name of the local interface to bind to
    :param local_port: local port to bind to
    :param reuse_port: ``True`` to allow multiple sockets to bind to the same
        address/port (not supported on Windows)
    :return: a UDP socket

    """
    if family is AddressFamily.AF_UNSPEC and not local_host:
        raise ValueError('Either "family" or "local_host" must be given')

    if local_host:
        gai_res = await getaddrinfo(
            str(local_host),
            local_port,
            family=family,
            type=socket.SOCK_DGRAM,
            flags=socket.AI_PASSIVE | socket.AI_ADDRCONFIG,
        )
        family = cast(AnyIPAddressFamily, gai_res[0][0])
        local_address = gai_res[0][-1]
    elif family is AddressFamily.AF_INET6:
        local_address = ("::", 0)
    else:
        local_address = ("0.0.0.0", 0)

    sock = await get_async_backend().create_udp_socket(
        family, local_address, None, reuse_port
    )
    return cast(UDPSocket, sock)


async def create_connected_udp_socket(
    remote_host: IPAddressType,
    remote_port: int,
    *,
    family: AnyIPAddressFamily = AddressFamily.AF_UNSPEC,
    local_host: IPAddressType | None = None,
    local_port: int = 0,
    reuse_port: bool = False,
) -> ConnectedUDPSocket:
    """
    Create a connected UDP socket.

    Connected UDP sockets can only communicate with the specified remote host/port, an
    any packets sent from other sources are dropped.

    :param remote_host: remote host to set as the default target
    :param remote_port: port on the remote host to set as the default target
    :param family: address family (``AF_INET`` or ``AF_INET6``) – automatically
        determined from ``local_host`` or ``remote_host`` if omitted
    :param local_host: IP address or host name of the local interface to bind to
    :param local_port: local port to bind to
    :param reuse_port: ``True`` to allow multiple sockets to bind to the same
        address/port (not supported on Windows)
    :return: a connected UDP socket

    """
    local_address = None
    if local_host:
        gai_res = await getaddrinfo(
            str(local_host),
            local_port,
            family=family,
            type=socket.SOCK_DGRAM,
            flags=socket.AI_PASSIVE | socket.AI_ADDRCONFIG,
        )
        family = cast(AnyIPAddressFamily, gai_res[0][0])
        local_address = gai_res[0][-1]

    gai_res = await getaddrinfo(
        str(remote_host), remote_port, family=family, type=socket.SOCK_DGRAM
    )
    family = cast(AnyIPAddressFamily, gai_res[0][0])
    remote_address = gai_res[0][-1]

    sock = await get_async_backend().create_udp_socket(
        family, local_address, remote_address, reuse_port
    )
    return cast(ConnectedUDPSocket, sock)


async def create_unix_datagram_socket(
    *,
    local_path: None | str | bytes | PathLike[Any] = None,
    local_mode: int | None = None,
) -> UNIXDatagramSocket:
    """
    Create a UNIX datagram socket.

    Not available on Windows.

    If ``local_path`` has been given, the socket will be bound to this path, making this
    socket suitable for receiving datagrams from other processes. Other processes can
    send datagrams to this socket only if ``local_path`` is set.

    If a socket already exists on the file system in the ``local_path``, it will be
    removed first.

    :param local_path: the path on which to bind to
    :param local_mode: permissions to set on the local socket
    :return: a UNIX datagram socket

    """
    raw_socket = await setup_unix_local_socket(
        local_path, local_mode, socket.SOCK_DGRAM
    )
    return await get_async_backend().create_unix_datagram_socket(raw_socket, None)


async def create_connected_unix_datagram_socket(
    remote_path: str | bytes | PathLike[Any],
    *,
    local_path: None | str | bytes | PathLike[Any] = None,
    local_mode: int | None = None,
) -> ConnectedUNIXDatagramSocket:
    """
    Create a connected UNIX datagram socket.

    Connected datagram sockets can only communicate with the specified remote path.

    If ``local_path`` has been given, the socket will be bound to this path, making
    this socket suitable for receiving datagrams from other processes. Other processes
    can send datagrams to this socket only if ``local_path`` is set.

    If a socket already exists on the file system in the ``local_path``, it will be
    removed first.

    :param remote_path: the path to set as the default target
    :param local_path: the path on which to bind to
    :param local_mode: permissions to set on the local socket
    :return: a connected UNIX datagram socket

    """
    remote_path = os.fspath(remote_path)
    raw_socket = await setup_unix_local_socket(
        local_path, local_mode, socket.SOCK_DGRAM
    )
    return await get_async_backend().create_unix_datagram_socket(
        raw_socket, remote_path
    )


async def getaddrinfo(
    host: bytes | str | None,
    port: str | int | None,
    *,
    family: int | AddressFamily = 0,
    type: int | SocketKind = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[tuple[AddressFamily, SocketKind, int, str, tuple[str, int]]]:
    """
    Look up a numeric IP address given a host name.

    Internationalized domain names are translated according to the (non-transitional)
    IDNA 2008 standard.

    .. note:: 4-tuple IPv6 socket addresses are automatically converted to 2-tuples of
        (host, port), unlike what :func:`socket.getaddrinfo` does.

    :param host: host name
    :param port: port number
    :param family: socket family (`'AF_INET``, ...)
    :param type: socket type (``SOCK_STREAM``, ...)
    :param proto: protocol number
    :param flags: flags to pass to upstream ``getaddrinfo()``
    :return: list of tuples containing (family, type, proto, canonname, sockaddr)

    .. seealso:: :func:`socket.getaddrinfo`

    """
    # Handle unicode hostnames
    encoded_host = idna2008_resolve(host) if isinstance(host, str) else host
    gai_res = await get_async_backend().getaddrinfo(
        encoded_host, port, family=family, type=type, proto=proto, flags=flags
    )
    return [
        (family, type, proto, canonname, convert_ipv6_sockaddr(sockaddr))
        for family, type, proto, canonname, sockaddr in gai_res
        # filter out IPv6 results when IPv6 is disabled
        if not isinstance(sockaddr[0], int)
    ]


def getnameinfo(sockaddr: IPSockAddrType, flags: int = 0) -> Awaitable[tuple[str, str]]:
    """
    Look up the host name of an IP address.

    :param sockaddr: socket address (e.g. (ipaddress, port) for IPv4)
    :param flags: flags to pass to upstream ``getnameinfo()``
    :return: a tuple of (host name, service name)
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. seealso:: :func:`socket.getnameinfo`

    """
    return get_async_backend().getnameinfo(sockaddr, flags)


@deprecated("This function is deprecated; use `wait_readable` instead")
def wait_socket_readable(sock: socket.socket) -> Awaitable[None]:
    """
    .. deprecated:: 4.7.0
       Use :func:`wait_readable` instead.

    Wait until the given socket has data to be read.

    .. warning:: Only use this on raw sockets that have not been wrapped by any higher
        level constructs like socket streams!

    :param sock: a socket object
    :raises ~anyio.ClosedResourceError: if the socket was closed while waiting for the
        socket to become readable
    :raises ~anyio.BusyResourceError: if another task is already waiting for the socket
        to become readable
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().wait_readable(sock.fileno())


@deprecated("This function is deprecated; use `wait_writable` instead")
def wait_socket_writable(sock: socket.socket) -> Awaitable[None]:
    """
    .. deprecated:: 4.7.0
       Use :func:`wait_writable` instead.

    Wait until the given socket can be written to.

    This does **NOT** work on Windows when using the asyncio backend with a proactor
    event loop (default on py3.8+).

    .. warning:: Only use this on raw sockets that have not been wrapped by any higher
        level constructs like socket streams!

    :param sock: a socket object
    :raises ~anyio.ClosedResourceError: if the socket was closed while waiting for the
        socket to become writable
    :raises ~anyio.BusyResourceError: if another task is already waiting for the socket
        to become writable
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().wait_writable(sock.fileno())


def wait_readable(obj: FileDescriptorLike) -> Awaitable[None]:
    """
    Wait until the given object has data to be read.

    On Unix systems, ``obj`` must either be an integer file descriptor, or else an
    object with a ``.fileno()`` method which returns an integer file descriptor. Any
    kind of file descriptor can be passed, though the exact semantics will depend on
    your kernel. For example, this probably won't do anything useful for on-disk files.

    On Windows systems, ``obj`` must either be an integer ``SOCKET`` handle, or else an
    object with a ``.fileno()`` method which returns an integer ``SOCKET`` handle. File
    descriptors aren't supported, and neither are handles that refer to anything besides
    a ``SOCKET``.

    On backends where this functionality is not natively provided (asyncio
    ``ProactorEventLoop`` on Windows), it is provided using a separate selector thread
    which is set to shut down when the interpreter shuts down.

    .. warning:: Don't use this on raw sockets that have been wrapped by any higher
        level constructs like socket streams!

    :param obj: an object with a ``.fileno()`` method or an integer handle
    :raises ~anyio.ClosedResourceError: if the object was closed while waiting for the
        object to become readable
    :raises ~anyio.BusyResourceError: if another task is already waiting for the object
        to become readable
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().wait_readable(obj)


def wait_writable(obj: FileDescriptorLike) -> Awaitable[None]:
    """
    Wait until the given object can be written to.

    :param obj: an object with a ``.fileno()`` method or an integer handle
    :raises ~anyio.ClosedResourceError: if the object was closed while waiting for the
        object to become writable
    :raises ~anyio.BusyResourceError: if another task is already waiting for the object
        to become writable
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. seealso:: See the documentation of :func:`wait_readable` for the definition of
       ``obj`` and notes on backend compatibility.

    .. warning:: Don't use this on raw sockets that have been wrapped by any higher
        level constructs like socket streams!

    """
    return get_async_backend().wait_writable(obj)


def notify_closing(obj: FileDescriptorLike) -> None:
    """
    Call this before closing a file descriptor (on Unix) or socket (on
    Windows). This will cause any `wait_readable` or `wait_writable`
    calls on the given object to immediately wake up and raise
    `~anyio.ClosedResourceError`.

    This doesn't actually close the object – you still have to do that
    yourself afterwards. Also, you want to be careful to make sure no
    new tasks start waiting on the object in between when you call this
    and when it's actually closed. So to close something properly, you
    usually want to do these steps in order:

    1. Explicitly mark the object as closed, so that any new attempts
       to use it will abort before they start.
    2. Call `notify_closing` to wake up any already-existing users.
    3. Actually close the object.

    It's also possible to do them in a different order if that's more
    convenient, *but only if* you make sure not to have any checkpoints in
    between the steps. This way they all happen in a single atomic
    step, so other tasks won't be able to tell what order they happened
    in anyway.

    :param obj: an object with a ``.fileno()`` method or an integer handle
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    get_async_backend().notify_closing(obj)


#
# Private API
#


def convert_ipv6_sockaddr(
    sockaddr: tuple[str, int, int, int] | tuple[str, int],
) -> tuple[str, int]:
    """
    Convert a 4-tuple IPv6 socket address to a 2-tuple (address, port) format.

    If the scope ID is nonzero, it is added to the address, separated with ``%``.
    Otherwise the flow id and scope id are simply cut off from the tuple.
    Any other kinds of socket addresses are returned as-is.

    :param sockaddr: the result of :meth:`~socket.socket.getsockname`
    :return: the converted socket address

    """
    # This is more complicated than it should be because of MyPy
    if isinstance(sockaddr, tuple) and len(sockaddr) == 4:
        host, port, _flowinfo, scope_id = sockaddr
        if scope_id:
            # PyPy (as of v7.3.11) leaves the interface name in the result, so
            # we discard it and only get the scope ID from the end
            # (https://foss.heptapod.net/pypy/pypy/-/issues/3938)
            host = host.split("%")[0]

            # Add scope_id to the address
            return f"{host}%{scope_id}", port
        else:
            return host, port
    else:
        return sockaddr


async def setup_unix_local_socket(
    path: None | str | bytes | PathLike[Any],
    mode: int | None,
    socktype: int,
) -> socket.socket:
    """
    Create a UNIX local socket object, deleting the socket at the given path if it
    exists.

    Not available on Windows.

    :param path: path of the socket
    :param mode: permissions to set on the socket
    :param socktype: socket.SOCK_STREAM or socket.SOCK_DGRAM

    """
    path_str: str | None
    if path is not None:
        path_str = os.fsdecode(path)

        # Linux abstract namespace sockets aren't backed by a concrete file so skip stat call
        if not path_str.startswith("\0"):
            # Copied from pathlib...
            try:
                stat_result = os.stat(path)
            except OSError as e:
                if e.errno not in (
                    errno.ENOENT,
                    errno.ENOTDIR,
                    errno.EBADF,
                    errno.ELOOP,
                ):
                    raise
            else:
                if stat.S_ISSOCK(stat_result.st_mode):
                    os.unlink(path)
    else:
        path_str = None

    raw_socket = socket.socket(socket.AF_UNIX, socktype)
    raw_socket.setblocking(False)

    if path_str is not None:
        try:
            await to_thread.run_sync(raw_socket.bind, path_str, abandon_on_cancel=True)
            if mode is not None:
                await to_thread.run_sync(chmod, path_str, mode, abandon_on_cancel=True)
        except BaseException:
            raw_socket.close()
            raise

    return raw_socket


@dataclass
class TCPConnectable(ByteStreamConnectable):
    """
    Connects to a TCP server at the given host and port.

    :param host: host name or IP address of the server
    :param port: TCP port number of the server
    """

    host: str | IPv4Address | IPv6Address
    port: int

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError("TCP port number out of range")

    @override
    async def connect(self) -> SocketStream:
        try:
            return await connect_tcp(self.host, self.port)
        except OSError as exc:
            raise ConnectionFailed(
                f"error connecting to {self.host}:{self.port}: {exc}"
            ) from exc


@dataclass
class UNIXConnectable(ByteStreamConnectable):
    """
    Connects to a UNIX domain socket at the given path.

    :param path: the file system path of the socket
    """

    path: str | bytes | PathLike[str] | PathLike[bytes]

    @override
    async def connect(self) -> UNIXSocketStream:
        try:
            return await connect_unix(self.path)
        except OSError as exc:
            raise ConnectionFailed(f"error connecting to {self.path!r}: {exc}") from exc


def as_connectable(
    remote: ByteStreamConnectable
    | tuple[str | IPv4Address | IPv6Address, int]
    | str
    | bytes
    | PathLike[str],
    /,
    *,
    tls: bool = False,
    ssl_context: ssl.SSLContext | None = None,
    tls_hostname: str | None = None,
    tls_standard_compatible: bool = True,
) -> ByteStreamConnectable:
    """
    Return a byte stream connectable from the given object.

    If a bytestream connectable is given, it is returned unchanged.
    If a tuple of (host, port) is given, a TCP connectable is returned.
    If a string or bytes path is given, a UNIX connectable is returned.

    If ``tls=True``, the connectable will be wrapped in a
    :class:`~.streams.tls.TLSConnectable`.

    :param remote: a connectable, a tuple of (host, port) or a path to a UNIX socket
    :param tls: if ``True``, wrap the plaintext connectable in a
        :class:`~.streams.tls.TLSConnectable`, using the provided TLS settings)
    :param ssl_context: if ``tls=True``, the SSLContext object to use  (if not provided,
        a secure default will be created)
    :param tls_hostname: if ``tls=True``, host name of the server to use for checking
        the server certificate (defaults to the host portion of the address for TCP
        connectables)
    :param tls_standard_compatible: if ``False`` and ``tls=True``, makes the TLS stream
        skip the closing handshake when closing the connection, so it won't raise an
        exception if the server does the same

    """
    connectable: TCPConnectable | UNIXConnectable | TLSConnectable
    if isinstance(remote, ByteStreamConnectable):
        return remote
    elif isinstance(remote, tuple) and len(remote) == 2:
        connectable = TCPConnectable(*remote)
    elif isinstance(remote, (str, bytes, PathLike)):
        connectable = UNIXConnectable(remote)
    else:
        raise TypeError(f"cannot convert {remote!r} to a connectable")

    if tls:
        if not tls_hostname and isinstance(connectable, TCPConnectable):
            tls_hostname = str(connectable.host)

        connectable = TLSConnectable(
            connectable,
            ssl_context=ssl_context,
            hostname=tls_hostname,
            standard_compatible=tls_standard_compatible,
        )

    return connectable
<|endoftext|>from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from inspect import isasyncgen, iscoroutine, isgenerator
from types import TracebackType
from typing import Protocol, TypeVar, cast, final

_T_co = TypeVar("_T_co", covariant=True)
_ExitT_co = TypeVar("_ExitT_co", covariant=True, bound="bool | None")


class _SupportsCtxMgr(Protocol[_T_co, _ExitT_co]):
    def __contextmanager__(self) -> AbstractContextManager[_T_co, _ExitT_co]: ...


class _SupportsAsyncCtxMgr(Protocol[_T_co, _ExitT_co]):
    def __asynccontextmanager__(
        self,
    ) -> AbstractAsyncContextManager[_T_co, _ExitT_co]: ...


class ContextManagerMixin:
    """
    Mixin class providing context manager functionality via a generator-based
    implementation.

    This class allows you to implement a context manager via :meth:`__contextmanager__`
    which should return a generator. The mechanics are meant to mirror those of
    :func:`@contextmanager <contextlib.contextmanager>`.

    .. note:: Classes using this mix-in are not reentrant as context managers, meaning
        that once you enter it, you can't re-enter before first exiting it.

    .. seealso:: :doc:`contextmanagers`
    """

    __cm: AbstractContextManager[object, bool | None] | None = None

    @final
    def __enter__(self: _SupportsCtxMgr[_T_co, bool | None]) -> _T_co:
        # Needed for mypy to assume self still has the __cm member
        assert isinstance(self, ContextManagerMixin)
        if self.__cm is not None:
            raise RuntimeError(
                f"this {self.__class__.__qualname__} has already been entered"
            )

        cm = self.__contextmanager__()
        if not isinstance(cm, AbstractContextManager):
            if isgenerator(cm):
                raise TypeError(
                    "__contextmanager__() returned a generator object instead of "
                    "a context manager. Did you forget to add the @contextmanager "
                    "decorator?"
                )

            raise TypeError(
                f"__contextmanager__() did not return a context manager object, "
                f"but {cm.__class__!r}"
            )

        if cm is self:
            raise TypeError(
                f"{self.__class__.__qualname__}.__contextmanager__() returned "
                f"self. Did you forget to add the @contextmanager decorator and a "
                f"'yield' statement?"
            )

        value = cm.__enter__()
        self.__cm = cm
        return value

    @final
    def __exit__(
        self: _SupportsCtxMgr[object, _ExitT_co],
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> _ExitT_co:
        # Needed for mypy to assume self still has the __cm member
        assert isinstance(self, ContextManagerMixin)
        if self.__cm is None:
            raise RuntimeError(
                f"this {self.__class__.__qualname__} has not been entered yet"
            )

        # Prevent circular references
        cm = self.__cm
        del self.__cm

        return cast(_ExitT_co, cm.__exit__(exc_type, exc_val, exc_tb))

    @abstractmethod
    def __contextmanager__(self) -> AbstractContextManager[object, bool | None]:
        """
        Implement your context manager logic here.

        This method **must** be decorated with
        :func:`@contextmanager <contextlib.contextmanager>`.

        .. note:: Remember that the ``yield`` will raise any exception raised in the
            enclosed context block, so use a ``finally:`` block to clean up resources!

        :return: a context manager object
        """


class AsyncContextManagerMixin:
    """
    Mixin class providing async context manager functionality via a generator-based
    implementation.

    This class allows you to implement a context manager via
    :meth:`__asynccontextmanager__`. The mechanics are meant to mirror those of
    :func:`@asynccontextmanager <contextlib.asynccontextmanager>`.

    .. note:: Classes using this mix-in are not reentrant as context managers, meaning
        that once you enter it, you can't re-enter before first exiting it.

    .. seealso:: :doc:`contextmanagers`
    """

    __cm: AbstractAsyncContextManager[object, bool | None] | None = None

    @final
    async def __aenter__(self: _SupportsAsyncCtxMgr[_T_co, bool | None]) -> _T_co:
        # Needed for mypy to assume self still has the __cm member
        assert isinstance(self, AsyncContextManagerMixin)
        if self.__cm is not None:
            raise RuntimeError(
                f"this {self.__class__.__qualname__} has already been entered"
            )

        cm = self.__asynccontextmanager__()
        if not isinstance(cm, AbstractAsyncContextManager):
            if isasyncgen(cm):
                raise TypeError(
                    "__asynccontextmanager__() returned an async generator instead of "
                    "an async context manager. Did you forget to add the "
                    "@asynccontextmanager decorator?"
                )
            elif iscoroutine(cm):
                cm.close()
                raise TypeError(
                    "__asynccontextmanager__() returned a coroutine object instead of "
                    "an async context manager. Did you forget to add the "
                    "@asynccontextmanager decorator and a 'yield' statement?"
                )

            raise TypeError(
                f"__asynccontextmanager__() did not return an async context manager, "
                f"but {cm.__class__!r}"
            )

        if cm is self:
            raise TypeError(
                f"{self.__class__.__qualname__}.__asynccontextmanager__() returned "
                f"self. Did you forget to add the @asynccontextmanager decorator and a "
                f"'yield' statement?"
            )

        value = await cm.__aenter__()
        self.__cm = cm
        return value

    @final
    async def __aexit__(
        self: _SupportsAsyncCtxMgr[object, _ExitT_co],
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> _ExitT_co:
        assert isinstance(self, AsyncContextManagerMixin)
        if self.__cm is None:
            raise RuntimeError(
                f"this {self.__class__.__qualname__} has not been entered yet"
            )

        # Prevent circular references
        cm = self.__cm
        del self.__cm

        return cast(_ExitT_co, await cm.__aexit__(exc_type, exc_val, exc_tb))

    @abstractmethod
    def __asynccontextmanager__(
        self,
    ) -> AbstractAsyncContextManager[object, bool | None]:
        """
        Implement your async context manager logic here.

        This method **must** be decorated with
        :func:`@asynccontextmanager <contextlib.asynccontextmanager>`.

        .. note:: Remember that the ``yield`` will raise any exception raised in the
            enclosed context block, so use a ``finally:`` block to clean up resources!

        :return: an async context manager object
        """
<|endoftext|>from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Mapping, Sequence
from io import BytesIO
from os import PathLike
from subprocess import PIPE, CalledProcessError, CompletedProcess
from typing import IO, Any, TypeAlias, cast

from ..abc import Process  # noqa: TC001
from ._eventloop import get_async_backend
from ._tasks import create_task_group

StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]


async def run_process(
    command: StrOrBytesPath | Sequence[StrOrBytesPath],
    *,
    input: bytes | None = None,
    stdin: int | IO[Any] | None = None,
    stdout: int | IO[Any] | None = PIPE,
    stderr: int | IO[Any] | None = PIPE,
    check: bool = True,
    cwd: StrOrBytesPath | None = None,
    env: Mapping[str, str] | None = None,
    startupinfo: Any = None,
    creationflags: int = 0,
    start_new_session: bool = False,
    pass_fds: Sequence[int] = (),
    user: str | int | None = None,
    group: str | int | None = None,
    extra_groups: Iterable[str | int] | None = None,
    umask: int = -1,
) -> CompletedProcess[bytes]:
    """
    Run an external command in a subprocess and wait until it completes.

    .. seealso:: :func:`subprocess.run`

    :param command: either a string to pass to the shell, or an iterable of strings
        containing the executable name or path and its arguments
    :param input: bytes passed to the standard input of the subprocess
    :param stdin: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`,
        a file-like object, or `None`; ``input`` overrides this
    :param stdout: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`,
        a file-like object, or `None`
    :param stderr: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`,
        :data:`subprocess.STDOUT`, a file-like object, or `None`
    :param check: if ``True``, raise :exc:`~subprocess.CalledProcessError` if the
        process terminates with a return code other than 0
    :param cwd: If not ``None``, change the working directory to this before running the
        command
    :param env: if not ``None``, this mapping replaces the inherited environment
        variables from the parent process
    :param startupinfo: an instance of :class:`subprocess.STARTUPINFO` that can be used
        to specify process startup parameters (Windows only)
    :param creationflags: flags that can be used to control the creation of the
        subprocess (see :class:`subprocess.Popen` for the specifics)
    :param start_new_session: if ``true`` the setsid() system call will be made in the
        child process prior to the execution of the subprocess. (POSIX only)
    :param pass_fds: sequence of file descriptors to keep open between the parent and
        child processes. (POSIX only)
    :param user: effective user to run the process as (Python >= 3.9, POSIX only)
    :param group: effective group to run the process as (Python >= 3.9, POSIX only)
    :param extra_groups: supplementary groups to set in the subprocess (Python >= 3.9,
        POSIX only)
    :param umask: if not negative, this umask is applied in the child process before
        running the given command (Python >= 3.9, POSIX only)
    :return: an object representing the completed process
    :raises ~subprocess.CalledProcessError: if ``check`` is ``True`` and the process
        exits with a nonzero return code

    """

    async def drain_stream(stream: AsyncIterable[bytes], index: int) -> None:
        buffer = BytesIO()
        async for chunk in stream:
            buffer.write(chunk)

        stream_contents[index] = buffer.getvalue()

    if stdin is not None and input is not None:
        raise ValueError("only one of stdin and input is allowed")

    async with await open_process(
        command,
        stdin=PIPE if input else stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=env,
        startupinfo=startupinfo,
        creationflags=creationflags,
        start_new_session=start_new_session,
        pass_fds=pass_fds,
        user=user,
        group=group,
        extra_groups=extra_groups,
        umask=umask,
    ) as process:
        stream_contents: list[bytes | None] = [None, None]
        async with create_task_group() as tg:
            if process.stdout:
                tg.start_soon(drain_stream, process.stdout, 0)

            if process.stderr:
                tg.start_soon(drain_stream, process.stderr, 1)

            if process.stdin and input:
                await process.stdin.send(input)
                await process.stdin.aclose()

            await process.wait()

    output, errors = stream_contents
    if check and process.returncode != 0:
        raise CalledProcessError(cast(int, process.returncode), command, output, errors)

    return CompletedProcess(command, cast(int, process.returncode), output, errors)


async def open_process(
    command: StrOrBytesPath | Sequence[StrOrBytesPath],
    *,
    stdin: int | IO[Any] | None = PIPE,
    stdout: int | IO[Any] | None = PIPE,
    stderr: int | IO[Any] | None = PIPE,
    cwd: StrOrBytesPath | None = None,
    env: Mapping[str, str] | None = None,
    startupinfo: Any = None,
    creationflags: int = 0,
    start_new_session: bool = False,
    pass_fds: Sequence[int] = (),
    user: str | int | None = None,
    group: str | int | None = None,
    extra_groups: Iterable[str | int] | None = None,
    umask: int = -1,
) -> Process:
    """
    Start an external command in a subprocess.

    .. seealso:: :class:`subprocess.Popen`

    :param command: either a string to pass to the shell, or an iterable of strings
        containing the executable name or path and its arguments
    :param stdin: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`, a
        file-like object, or ``None``
    :param stdout: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`,
        a file-like object, or ``None``
    :param stderr: one of :data:`subprocess.PIPE`, :data:`subprocess.DEVNULL`,
        :data:`subprocess.STDOUT`, a file-like object, or ``None``
    :param cwd: If not ``None``, the working directory is changed before executing
    :param env: If env is not ``None``, it must be a mapping that defines the
        environment variables for the new process
    :param creationflags: flags that can be used to control the creation of the
        subprocess (see :class:`subprocess.Popen` for the specifics)
    :param startupinfo: an instance of :class:`subprocess.STARTUPINFO` that can be used
        to specify process startup parameters (Windows only)
    :param start_new_session: if ``true`` the setsid() system call will be made in the
        child process prior to the execution of the subprocess. (POSIX only)
    :param pass_fds: sequence of file descriptors to keep open between the parent and
        child processes. (POSIX only)
    :param user: effective user to run the process as (POSIX only)
    :param group: effective group to run the process as (POSIX only)
    :param extra_groups: supplementary groups to set in the subprocess (POSIX only)
    :param umask: if not negative, this umask is applied in the child process before
        running the given command (POSIX only)
    :return: an asynchronous process object

    """
    kwargs: dict[str, Any] = {}
    if user is not None:
        kwargs["user"] = user

    if group is not None:
        kwargs["group"] = group

    if extra_groups is not None:
        kwargs["extra_groups"] = extra_groups

    if umask >= 0:
        kwargs["umask"] = umask

    return await get_async_backend().open_process(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=env,
        startupinfo=startupinfo,
        creationflags=creationflags,
        start_new_session=start_new_session,
        pass_fds=pass_fds,
        **kwargs,
    )
<|endoftext|>from __future__ import annotations

import math
from typing import TypeVar
from warnings import warn

from ..streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
    _MemoryObjectStreamState,
)

T_Item = TypeVar("T_Item")


class create_memory_object_stream(
    tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]],
):
    """
    Create a memory object stream.

    The stream's item type can be annotated like
    :func:`create_memory_object_stream[T_Item]`.

    :param max_buffer_size: number of items held in the buffer until ``send()`` starts
        blocking
    :param item_type: old way of marking the streams with the right generic type for
        static typing (does nothing on AnyIO 4)

        .. deprecated:: 4.0
          Use ``create_memory_object_stream[YourItemType](...)`` instead.
    :return: a tuple of (send stream, receive stream)

    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float = 0, item_type: object = None
    ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
        if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        if max_buffer_size < 0:
            raise ValueError("max_buffer_size cannot be negative")
        if item_type is not None:
            warn(
                "The item_type argument has been deprecated in AnyIO 4.0. "
                "Use create_memory_object_stream[YourItemType](...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        state = _MemoryObjectStreamState[T_Item](max_buffer_size)
        return (MemoryObjectSendStream(state), MemoryObjectReceiveStream(state))
<|endoftext|>from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, final, overload

from ._exceptions import TypedAttributeLookupError

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

T_Attr = TypeVar("T_Attr")
T_Default = TypeVar("T_Default")
undefined = sentinel("undefined")


def typed_attribute() -> Any:
    """Return a unique object, used to mark typed attributes."""
    return object()


class TypedAttributeSet:
    """
    Superclass for typed attribute collections.

    Checks that every public attribute of every subclass has a type annotation.
    """

    def __init_subclass__(cls) -> None:
        annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
        for attrname in dir(cls):
            if not attrname.startswith("_") and attrname not in annotations:
                raise TypeError(
                    f"Attribute {attrname!r} is missing its type annotation"
                )

        super().__init_subclass__()


class TypedAttributeProvider:
    """Base class for classes that wish to provide typed extra attributes."""

    @property
    def extra_attributes(self) -> Mapping[T_Attr, Callable[[], T_Attr]]:
        """
        A mapping of the extra attributes to callables that return the corresponding
        values.

        If the provider wraps another provider, the attributes from that wrapper should
        also be included in the returned mapping (but the wrapper may override the
        callables from the wrapped instance).

        """
        return {}

    @overload
    def extra(self, attribute: T_Attr) -> T_Attr: ...

    @overload
    def extra(self, attribute: T_Attr, default: T_Default) -> T_Attr | T_Default: ...

    @final
    def extra(self, attribute: Any, default: object = undefined) -> object:
        """
        extra(attribute, default=undefined)

        Return the value of the given typed extra attribute.

        :param attribute: the attribute (member of a :class:`~TypedAttributeSet`) to
            look for
        :param default: the value that should be returned if no value is found for the
            attribute
        :raises ~anyio.TypedAttributeLookupError: if the search failed and no default
            value was given

        """
        try:
            getter = self.extra_attributes[attribute]
        except KeyError:
            if default is undefined:
                raise TypedAttributeLookupError("Attribute not found") from None
            else:
                return default

        return getter()
<|endoftext|>from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterable
from io import BytesIO, TextIOWrapper
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Generic,
    overload,
)

from .. import to_thread
from .._core._fileio import AsyncFile
from ..lowlevel import checkpoint_if_cancelled
from ._tasks import CancelScope

if TYPE_CHECKING:
    from _typeshed import OpenBinaryMode, OpenTextMode, ReadableBuffer, WriteableBuffer


class TemporaryFile(Generic[AnyStr]):
    """
    An asynchronous temporary file that is automatically created and cleaned up.

    This class provides an asynchronous context manager interface to a temporary file.
    The file is created using Python's standard `tempfile.TemporaryFile` function in a
    background thread, and is wrapped as an asynchronous file using `AsyncFile`.

    :param mode: The mode in which the file is opened. Defaults to "w+b".
    :param buffering: The buffering policy (-1 means the default buffering).
    :param encoding: The encoding used to decode or encode the file. Only applicable in
        text mode.
    :param newline: Controls how universal newlines mode works (only applicable in text
        mode).
    :param suffix: The suffix for the temporary file name.
    :param prefix: The prefix for the temporary file name.
    :param dir: The directory in which the temporary file is created.
    :param errors: The error handling scheme used for encoding/decoding errors.
    """

    _async_file: AsyncFile[AnyStr]

    @overload
    def __init__(
        self: TemporaryFile[bytes],
        mode: OpenBinaryMode = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        *,
        errors: str | None = ...,
    ): ...
    @overload
    def __init__(
        self: TemporaryFile[str],
        mode: OpenTextMode,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        *,
        errors: str | None = ...,
    ): ...

    def __init__(
        self,
        mode: OpenTextMode | OpenBinaryMode = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        *,
        errors: str | None = None,
    ) -> None:
        self.mode = mode
        self.buffering = buffering
        self.encoding = encoding
        self.newline = newline
        self.suffix: str | None = suffix
        self.prefix: str | None = prefix
        self.dir: str | None = dir
        self.errors = errors

    async def __aenter__(self) -> AsyncFile[AnyStr]:
        fp = await to_thread.run_sync(
            lambda: tempfile.TemporaryFile(
                self.mode,
                self.buffering,
                self.encoding,
                self.newline,
                self.suffix,
                self.prefix,
                self.dir,
                errors=self.errors,
            )
        )
        self._async_file = AsyncFile(fp)
        return self._async_file

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._async_file.aclose()


class NamedTemporaryFile(Generic[AnyStr]):
    """
    An asynchronous named temporary file that is automatically created and cleaned up.

    This class provides an asynchronous context manager for a temporary file with a
    visible name in the file system. It uses Python's standard
    :func:`~tempfile.NamedTemporaryFile` function and wraps the file object with
    :class:`AsyncFile` for asynchronous operations.

    :param mode: The mode in which the file is opened. Defaults to "w+b".
    :param buffering: The buffering policy (-1 means the default buffering).
    :param encoding: The encoding used to decode or encode the file. Only applicable in
        text mode.
    :param newline: Controls how universal newlines mode works (only applicable in text
        mode).
    :param suffix: The suffix for the temporary file name.
    :param prefix: The prefix for the temporary file name.
    :param dir: The directory in which the temporary file is created.
    :param delete: Whether to delete the file when it is closed.
    :param errors: The error handling scheme used for encoding/decoding errors.
    :param delete_on_close: (Python 3.12+) Whether to delete the file on close.
    """

    _async_file: AsyncFile[AnyStr]

    @overload
    def __init__(
        self: NamedTemporaryFile[bytes],
        mode: OpenBinaryMode = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        delete: bool = ...,
        *,
        errors: str | None = ...,
        delete_on_close: bool = ...,
    ): ...
    @overload
    def __init__(
        self: NamedTemporaryFile[str],
        mode: OpenTextMode,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        delete: bool = ...,
        *,
        errors: str | None = ...,
        delete_on_close: bool = ...,
    ): ...

    def __init__(
        self,
        mode: OpenBinaryMode | OpenTextMode = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        delete: bool = True,
        *,
        errors: str | None = None,
        delete_on_close: bool = True,
    ) -> None:
        self._params: dict[str, Any] = {
            "mode": mode,
            "buffering": buffering,
            "encoding": encoding,
            "newline": newline,
            "suffix": suffix,
            "prefix": prefix,
            "dir": dir,
            "delete": delete,
            "errors": errors,
        }
        if sys.version_info >= (3, 12):
            self._params["delete_on_close"] = delete_on_close

    async def __aenter__(self) -> AsyncFile[AnyStr]:
        fp = await to_thread.run_sync(
            lambda: tempfile.NamedTemporaryFile(**self._params)
        )
        self._async_file = AsyncFile(fp)
        return self._async_file

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._async_file.aclose()


class SpooledTemporaryFile(AsyncFile[AnyStr]):
    """
    An asynchronous spooled temporary file that starts in memory and is spooled to disk.

    This class provides an asynchronous interface to a spooled temporary file, much like
    Python's standard :class:`~tempfile.SpooledTemporaryFile`. It supports asynchronous
    write operations and provides a method to force a rollover to disk.

    :param max_size: Maximum size in bytes before the file is rolled over to disk.
    :param mode: The mode in which the file is opened. Defaults to "w+b".
    :param buffering: The buffering policy (-1 means the default buffering).
    :param encoding: The encoding used to decode or encode the file (text mode only).
    :param newline: Controls how universal newlines mode works (text mode only).
    :param suffix: The suffix for the temporary file name.
    :param prefix: The prefix for the temporary file name.
    :param dir: The directory in which the temporary file is created.
    :param errors: The error handling scheme used for encoding/decoding errors.
    """

    _rolled: bool = False

    @overload
    def __init__(
        self: SpooledTemporaryFile[bytes],
        max_size: int = ...,
        mode: OpenBinaryMode = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        *,
        errors: str | None = ...,
    ): ...
    @overload
    def __init__(
        self: SpooledTemporaryFile[str],
        max_size: int = ...,
        mode: OpenTextMode = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        newline: str | None = ...,
        suffix: str | None = ...,
        prefix: str | None = ...,
        dir: str | None = ...,
        *,
        errors: str | None = ...,
    ): ...

    def __init__(
        self,
        max_size: int = 0,
        mode: OpenBinaryMode | OpenTextMode = "w+b",
        buffering: int = -1,
        encoding: str | None = None,
        newline: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        *,
        errors: str | None = None,
    ) -> None:
        self._tempfile_params: dict[str, Any] = {
            "mode": mode,
            "buffering": buffering,
            "encoding": encoding,
            "newline": newline,
            "suffix": suffix,
            "prefix": prefix,
            "dir": dir,
            "errors": errors,
        }
        self._max_size = max_size
        if "b" in mode:
            super().__init__(BytesIO())  # type: ignore[arg-type]
        else:
            super().__init__(
                TextIOWrapper(  # type: ignore[arg-type]
                    BytesIO(),
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                    write_through=True,
                )
            )

    async def aclose(self) -> None:
        if not self._rolled:
            self._fp.close()
            return

        await super().aclose()

    async def _check(self) -> None:
        if self._rolled or self._fp.tell() <= self._max_size:
            return

        await self.rollover()

    async def rollover(self) -> None:
        if self._rolled:
            return

        self._rolled = True
        buffer = self._fp
        buffer.seek(0)
        self._fp = await to_thread.run_sync(
            lambda: tempfile.TemporaryFile(**self._tempfile_params)
        )
        await self.write(buffer.read())
        buffer.close()

    @property
    def closed(self) -> bool:
        return self._fp.closed

    async def read(self, size: int = -1) -> AnyStr:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.read(size)

        return await super().read(size)  # type: ignore[return-value]

    async def read1(self: SpooledTemporaryFile[bytes], size: int = -1) -> bytes:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.read1(size)

        return await super().read1(size)

    async def readline(self) -> AnyStr:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.readline()

        return await super().readline()  # type: ignore[return-value]

    async def readlines(self) -> list[AnyStr]:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.readlines()

        return await super().readlines()  # type: ignore[return-value]

    async def readinto(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.readinto(b)

        return await super().readinto(b)

    async def readinto1(self: SpooledTemporaryFile[bytes], b: WriteableBuffer) -> int:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.readinto1(b)

        return await super().readinto1(b)

    async def seek(self, offset: int, whence: int | None = os.SEEK_SET) -> int:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.seek(offset, whence)

        return await super().seek(offset, whence)

    async def tell(self) -> int:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.tell()

        return await super().tell()

    async def truncate(self, size: int | None = None) -> int:
        if not self._rolled:
            await checkpoint_if_cancelled()
            return self._fp.truncate(size)

        return await super().truncate(size)

    @overload
    async def write(self: SpooledTemporaryFile[bytes], b: ReadableBuffer) -> int: ...
    @overload
    async def write(self: SpooledTemporaryFile[str], b: str) -> int: ...

    async def write(self, b: ReadableBuffer | str) -> int:
        """
        Asynchronously write data to the spooled temporary file.

        If the file has not yet been rolled over, the data is written synchronously,
        and a rollover is triggered if the size exceeds the maximum size.

        :param s: The data to write.
        :return: The number of bytes written.
        :raises RuntimeError: If the underlying file is not initialized.

        """
        if not self._rolled:
            await checkpoint_if_cancelled()
            result = self._fp.write(b)
            await self._check()
            return result

        return await super().write(b)  # type: ignore[misc]

    @overload
    async def writelines(
        self: SpooledTemporaryFile[bytes], lines: Iterable[ReadableBuffer]
    ) -> None: ...
    @overload
    async def writelines(
        self: SpooledTemporaryFile[str], lines: Iterable[str]
    ) -> None: ...

    async def writelines(self, lines: Iterable[str] | Iterable[ReadableBuffer]) -> None:
        """
        Asynchronously write a list of lines to the spooled temporary file.

        If the file has not yet been rolled over, the lines are written synchronously,
        and a rollover is triggered if the size exceeds the maximum size.

        :param lines: An iterable of lines to write.
        :raises RuntimeError: If the underlying file is not initialized.

        """
        if not self._rolled:
            await checkpoint_if_cancelled()
            result = self._fp.writelines(lines)
            await self._check()
            return result

        return await super().writelines(lines)  # type: ignore[misc]


class TemporaryDirectory(Generic[AnyStr]):
    """
    An asynchronous temporary directory that is created and cleaned up automatically.

    This class provides an asynchronous context manager for creating a temporary
    directory. It wraps Python's standard :class:`~tempfile.TemporaryDirectory` to
    perform directory creation and cleanup operations in a background thread.

    :param suffix: Suffix to be added to the temporary directory name.
    :param prefix: Prefix to be added to the temporary directory name.
    :param dir: The parent directory where the temporary directory is created.
    :param ignore_cleanup_errors: Whether to ignore errors during cleanup
    :param delete: Whether to delete the directory upon closing (Python 3.12+).
    """

    def __init__(
        self,
        suffix: AnyStr | None = None,
        prefix: AnyStr | None = None,
        dir: AnyStr | None = None,
        *,
        ignore_cleanup_errors: bool = False,
        delete: bool = True,
    ) -> None:
        self.suffix: AnyStr | None = suffix
        self.prefix: AnyStr | None = prefix
        self.dir: AnyStr | None = dir
        self.ignore_cleanup_errors = ignore_cleanup_errors
        self.delete = delete

        self._tempdir: tempfile.TemporaryDirectory | None = None

    async def __aenter__(self) -> str:
        params: dict[str, Any] = {
            "suffix": self.suffix,
            "prefix": self.prefix,
            "dir": self.dir,
            "ignore_cleanup_errors": self.ignore_cleanup_errors,
        }
        if sys.version_info >= (3, 12):
            params["delete"] = self.delete

        self._tempdir = await to_thread.run_sync(
            lambda: tempfile.TemporaryDirectory(**params)
        )
        return await to_thread.run_sync(self._tempdir.__enter__)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._tempdir is not None:
            with CancelScope(shield=True):
                await to_thread.run_sync(
                    self._tempdir.__exit__, exc_type, exc_value, traceback
                )

    async def cleanup(self) -> None:
        if self._tempdir is not None:
            await to_thread.run_sync(self._tempdir.cleanup)


@overload
async def mkstemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
    text: bool = False,
) -> tuple[int, str]: ...


@overload
async def mkstemp(
    suffix: bytes | None = None,
    prefix: bytes | None = None,
    dir: bytes | None = None,
    text: bool = False,
) -> tuple[int, bytes]: ...


async def mkstemp(
    suffix: AnyStr | None = None,
    prefix: AnyStr | None = None,
    dir: AnyStr | None = None,
    text: bool = False,
) -> tuple[int, str | bytes]:
    """
    Asynchronously create a temporary file and return an OS-level handle and the file
    name.

    This function wraps `tempfile.mkstemp` and executes it in a background thread.

    :param suffix: Suffix to be added to the file name.
    :param prefix: Prefix to be added to the file name.
    :param dir: Directory in which the temporary file is created.
    :param text: Whether the file is opened in text mode.
    :return: A tuple containing the file descriptor and the file name.

    """
    return await to_thread.run_sync(tempfile.mkstemp, suffix, prefix, dir, text)


@overload
async def mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
) -> str: ...


@overload
async def mkdtemp(
    suffix: bytes | None = None,
    prefix: bytes | None = None,
    dir: bytes | None = None,
) -> bytes: ...


async def mkdtemp(
    suffix: AnyStr | None = None,
    prefix: AnyStr | None = None,
    dir: AnyStr | None = None,
) -> str | bytes:
    """
    Asynchronously create a temporary directory and return its path.

    This function wraps `tempfile.mkdtemp` and executes it in a background thread.

    :param suffix: Suffix to be added to the directory name.
    :param prefix: Prefix to be added to the directory name.
    :param dir: Parent directory where the temporary directory is created.
    :return: The path of the created temporary directory.

    """
    return await to_thread.run_sync(tempfile.mkdtemp, suffix, prefix, dir)


async def gettempdir() -> str:
    """
    Asynchronously return the name of the directory used for temporary files.

    This function wraps `tempfile.gettempdir` and executes it in a background thread.

    :return: The path of the temporary directory as a string.

    """
    return await to_thread.run_sync(tempfile.gettempdir)


async def gettempdirb() -> bytes:
    """
    Asynchronously return the name of the directory used for temporary files in bytes.

    This function wraps `tempfile.gettempdirb` and executes it in a background thread.

    :return: The path of the temporary directory as bytes.

    """
    return await to_thread.run_sync(tempfile.gettempdirb)
<|endoftext|>from __future__ import annotations

from collections.abc import Awaitable, Generator
from typing import Any, cast

from ._eventloop import get_async_backend


class TaskInfo:
    """
    Represents an asynchronous task.

    :ivar int id: the unique identifier of the task
    :ivar parent_id: the identifier of the parent task, if any
    :vartype parent_id: Optional[int]
    :ivar str name: the description of the task (if any)
    :ivar ~collections.abc.Coroutine coro: the coroutine object of the task
    """

    __slots__ = "_name", "coro", "id", "name", "parent_id"

    def __init__(
        self,
        id: int,
        parent_id: int | None,
        name: str | None,
        coro: Generator[Any, Any, Any] | Awaitable[Any],
    ):
        func = get_current_task
        self._name = f"{func.__module__}.{func.__qualname__}"
        self.id: int = id
        self.parent_id: int | None = parent_id
        self.name: str | None = name
        self.coro: Generator[Any, Any, Any] | Awaitable[Any] = coro

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaskInfo):
            return self.id == other.id

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, name={self.name!r})"

    def has_pending_cancellation(self) -> bool:
        """
        Return ``True`` if the task has a cancellation pending, ``False`` otherwise.

        """
        return False


def get_current_task() -> TaskInfo:
    """
    Return the current task.

    :return: a representation of the current task
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().get_current_task()


def get_running_tasks() -> list[TaskInfo]:
    """
    Return a list of running tasks in the current event loop.

    :return: a list of task info objects
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return cast("list[TaskInfo]", get_async_backend().get_running_tasks())


async def wait_all_tasks_blocked() -> None:
    """Wait until all other tasks are waiting for something."""
    await get_async_backend().wait_all_tasks_blocked()
<|endoftext|>import sys
from typing import Any, Awaitable, Callable, TypeVar

from frozenlist import FrozenList

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

if sys.version_info >= (3, 13):
    from typing import TypeVarTuple
else:
    from typing_extensions import TypeVarTuple

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])

__version__ = "1.4.0"

__all__ = ("Signal",)


class Signal(FrozenList[Callable[[Unpack[_Ts]], Awaitable[object]]]):
    """Coroutine-based signal implementation.

    To connect a callback to a signal, use any list method.

    Signals are fired using the send() coroutine, which takes named
    arguments.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: object):
        super().__init__()
        self._owner = owner

    def __repr__(self) -> str:
        return "<Signal owner={}, frozen={}, {!r}>".format(
            self._owner, self.frozen, list(self)
        )

    async def send(self, *args: Unpack[_Ts], **kwargs: Any) -> None:
        """
        Sends data to all registered receivers.
        """
        if not self.frozen:
            raise RuntimeError("Cannot send non-frozen signal.")

        for receiver in self:
            await receiver(*args, **kwargs)

    def __call__(
        self, func: Callable[[Unpack[_Ts]], Awaitable[_T]]
    ) -> Callable[[Unpack[_Ts]], Awaitable[_T]]:
        """Decorator to add a function to this Signal."""
        self.append(func)
        return func
<|endoftext|># This is a stub package designed to roughly emulate the _yaml
# extension module, which previously existed as a standalone module
# and has been moved into the `yaml` package namespace.
# It does not perfectly mimic its old counterpart, but should get
# close enough for anyone who's relying on it even when they shouldn't.
import yaml

# in some circumstances, the yaml module we imoprted may be from a different version, so we need
# to tread carefully when poking at it here (it may not have the attributes we expect)
if not getattr(yaml, '__with_libyaml__', False):
    from sys import version_info

    exc = ModuleNotFoundError if version_info >= (3, 6) else ImportError
    raise exc("No module named '_yaml'")
else:
    from yaml._yaml import *
    import warnings
    warnings.warn(
        'The _yaml extension module is now located at yaml._yaml'
        ' and its location is subject to change.  To use the'
        ' LibYAML-based parser and emitter, import from `yaml`:'
        ' `from yaml import CLoader as Loader, CDumper as Dumper`.',
        DeprecationWarning
    )
    del warnings
    # Don't `del yaml` here because yaml is actually an existing
    # namespace member of _yaml.

__name__ = '_yaml'
# If the module is top-level (i.e. not a part of any specific package)
# then the attribute should be set to ''.
# https://docs.python.org/3.8/library/types.html
__package__ = ''
<|endoftext|>"""
Cycler
======

Cycling through combinations of values, producing dictionaries.

You can add cyclers::

    from cycler import cycler
    cc = (cycler(color=list('rgb')) +
          cycler(linestyle=['-', '--', '-.']))
    for d in cc:
        print(d)

Results in::

    {'color': 'r', 'linestyle': '-'}
    {'color': 'g', 'linestyle': '--'}
    {'color': 'b', 'linestyle': '-.'}


You can multiply cyclers::

    from cycler import cycler
    cc = (cycler(color=list('rgb')) *
          cycler(linestyle=['-', '--', '-.']))
    for d in cc:
        print(d)

Results in::

    {'color': 'r', 'linestyle': '-'}
    {'color': 'r', 'linestyle': '--'}
    {'color': 'r', 'linestyle': '-.'}
    {'color': 'g', 'linestyle': '-'}
    {'color': 'g', 'linestyle': '--'}
    {'color': 'g', 'linestyle': '-.'}
    {'color': 'b', 'linestyle': '-'}
    {'color': 'b', 'linestyle': '--'}
    {'color': 'b', 'linestyle': '-.'}
"""


from __future__ import annotations

from collections.abc import Hashable, Iterable, Generator
import copy
from functools import reduce
from itertools import product, cycle
from operator import mul, add
# Dict, List, Union required for runtime cast calls
from typing import TypeVar, Generic, Callable, Union, Dict, List, Any, overload, cast

__version__ = "0.12.1"

K = TypeVar("K", bound=Hashable)
L = TypeVar("L", bound=Hashable)
V = TypeVar("V")
U = TypeVar("U")


def _process_keys(
    left: Cycler[K, V] | Iterable[dict[K, V]],
    right: Cycler[K, V] | Iterable[dict[K, V]] | None,
) -> set[K]:
    """
    Helper function to compose cycler keys.

    Parameters
    ----------
    left, right : iterable of dictionaries or None
        The cyclers to be composed.

    Returns
    -------
    keys : set
        The keys in the composition of the two cyclers.
    """
    l_peek: dict[K, V] = next(iter(left)) if left != [] else {}
    r_peek: dict[K, V] = next(iter(right)) if right is not None else {}
    l_key: set[K] = set(l_peek.keys())
    r_key: set[K] = set(r_peek.keys())
    if l_key & r_key:
        raise ValueError("Can not compose overlapping cycles")
    return l_key | r_key


def concat(left: Cycler[K, V], right: Cycler[K, U]) -> Cycler[K, V | U]:
    r"""
    Concatenate `Cycler`\s, as if chained using `itertools.chain`.

    The keys must match exactly.

    Examples
    --------
    >>> num = cycler('a', range(3))
    >>> let = cycler('a', 'abc')
    >>> num.concat(let)
    cycler('a', [0, 1, 2, 'a', 'b', 'c'])

    Returns
    -------
    `Cycler`
        The concatenated cycler.
    """
    if left.keys != right.keys:
        raise ValueError(
            "Keys do not match:\n"
            "\tIntersection: {both!r}\n"
            "\tDisjoint: {just_one!r}".format(
                both=left.keys & right.keys, just_one=left.keys ^ right.keys
            )
        )
    _l = cast(Dict[K, List[Union[V, U]]], left.by_key())
    _r = cast(Dict[K, List[Union[V, U]]], right.by_key())
    return reduce(add, (_cycler(k, _l[k] + _r[k]) for k in left.keys))


class Cycler(Generic[K, V]):
    """
    Composable cycles.

    This class has compositions methods:

    ``+``
      for 'inner' products (zip)

    ``+=``
      in-place ``+``

    ``*``
      for outer products (`itertools.product`) and integer multiplication

    ``*=``
      in-place ``*``

    and supports basic slicing via ``[]``.

    Parameters
    ----------
    left, right : Cycler or None
        The 'left' and 'right' cyclers.
    op : func or None
        Function which composes the 'left' and 'right' cyclers.
    """

    def __call__(self):
        return cycle(self)

    def __init__(
        self,
        left: Cycler[K, V] | Iterable[dict[K, V]] | None,
        right: Cycler[K, V] | None = None,
        op: Any = None,
    ):
        """
        Semi-private init.

        Do not use this directly, use `cycler` function instead.
        """
        if isinstance(left, Cycler):
            self._left: Cycler[K, V] | list[dict[K, V]] = Cycler(
                left._left, left._right, left._op
            )
        elif left is not None:
            # Need to copy the dictionary or else that will be a residual
            # mutable that could lead to strange errors
            self._left = [copy.copy(v) for v in left]
        else:
            self._left = []

        if isinstance(right, Cycler):
            self._right: Cycler[K, V] | None = Cycler(
                right._left, right._right, right._op
            )
        else:
            self._right = None

        self._keys: set[K] = _process_keys(self._left, self._right)
        self._op: Any = op

    def __contains__(self, k):
        return k in self._keys

    @property
    def keys(self) -> set[K]:
        """The keys this Cycler knows about."""
        return set(self._keys)

    def change_key(self, old: K, new: K) -> None:
        """
        Change a key in this cycler to a new name.
        Modification is performed in-place.

        Does nothing if the old key is the same as the new key.
        Raises a ValueError if the new key is already a key.
        Raises a KeyError if the old key isn't a key.
        """
        if old == new:
            return
        if new in self._keys:
            raise ValueError(
                f"Can't replace {old} with {new}, {new} is already a key"
            )
        if old not in self._keys:
            raise KeyError(
                f"Can't replace {old} with {new}, {old} is not a key"
            )

        self._keys.remove(old)
        self._keys.add(new)

        if self._right is not None and old in self._right.keys:
            self._right.change_key(old, new)

        # self._left should always be non-None
        # if self._keys is non-empty.
        elif isinstance(self._left, Cycler):
            self._left.change_key(old, new)
        else:
            # It should be completely safe at this point to
            # assume that the old key can be found in each
            # iteration.
            self._left = [{new: entry[old]} for entry in self._left]

    @classmethod
    def _from_iter(cls, label: K, itr: Iterable[V]) -> Cycler[K, V]:
        """
        Class method to create 'base' Cycler objects
        that do not have a 'right' or 'op' and for which
        the 'left' object is not another Cycler.

        Parameters
        ----------
        label : hashable
            The property key.

        itr : iterable
            Finite length iterable of the property values.

        Returns
        -------
        `Cycler`
            New 'base' cycler.
        """
        ret: Cycler[K, V] = cls(None)
        ret._left = list({label: v} for v in itr)
        ret._keys = {label}
        return ret

    def __getitem__(self, key: slice) -> Cycler[K, V]:
        # TODO : maybe add numpy style fancy slicing
        if isinstance(key, slice):
            trans = self.by_key()
            return reduce(add, (_cycler(k, v[key]) for k, v in trans.items()))
        else:
            raise ValueError("Can only use slices with Cycler.__getitem__")

    def __iter__(self) -> Generator[dict[K, V], None, None]:
        if self._right is None:
            for left in self._left:
                yield dict(left)
        else:
            if self._op is None:
                raise TypeError(
                    "Operation cannot be None when both left and right are defined"
                )
            for a, b in self._op(self._left, self._right):
                out = {}
                out.update(a)
                out.update(b)
                yield out

    def __add__(self, other: Cycler[L, U]) -> Cycler[K | L, V | U]:
        """
        Pair-wise combine two equal length cyclers (zip).

        Parameters
        ----------
        other : Cycler
        """
        if len(self) != len(other):
            raise ValueError(
                f"Can only add equal length cycles, not {len(self)} and {len(other)}"
            )
        return Cycler(
            cast(Cycler[Union[K, L], Union[V, U]], self),
            cast(Cycler[Union[K, L], Union[V, U]], other),
            zip
        )

    @overload
    def __mul__(self, other: Cycler[L, U]) -> Cycler[K | L, V | U]:
        ...

    @overload
    def __mul__(self, other: int) -> Cycler[K, V]:
        ...

    def __mul__(self, other):
        """
        Outer product of two cyclers (`itertools.product`) or integer
        multiplication.

        Parameters
        ----------
        other : Cycler or int
        """
        if isinstance(other, Cycler):
            return Cycler(
                cast(Cycler[Union[K, L], Union[V, U]], self),
                cast(Cycler[Union[K, L], Union[V, U]], other),
                product
            )
        elif isinstance(other, int):
            trans = self.by_key()
            return reduce(
                add, (_cycler(k, v * other) for k, v in trans.items())
            )
        else:
            return NotImplemented

    @overload
    def __rmul__(self, other: Cycler[L, U]) -> Cycler[K | L, V | U]:
        ...

    @overload
    def __rmul__(self, other: int) -> Cycler[K, V]:
        ...

    def __rmul__(self, other):
        return self * other

    def __len__(self) -> int:
        op_dict: dict[Callable, Callable[[int, int], int]] = {zip: min, product: mul}
        if self._right is None:
            return len(self._left)
        l_len = len(self._left)
        r_len = len(self._right)
        return op_dict[self._op](l_len, r_len)

    # iadd and imul do not exapand the the type as the returns must be consistent with
    # self, thus they flag as inconsistent with add/mul
    def __iadd__(self, other: Cycler[K, V]) -> Cycler[K, V]:  # type: ignore[misc]
        """
        In-place pair-wise combine two equal length cyclers (zip).

        Parameters
        ----------
        other : Cycler
        """
        if not isinstance(other, Cycler):
            raise TypeError("Cannot += with a non-Cycler object")
        # True shallow copy of self is fine since this is in-place
        old_self = copy.copy(self)
        self._keys = _process_keys(old_self, other)
        self._left = old_self
        self._op = zip
        self._right = Cycler(other._left, other._right, other._op)
        return self

    def __imul__(self, other: Cycler[K, V] | int) -> Cycler[K, V]:  # type: ignore[misc]
        """
        In-place outer product of two cyclers (`itertools.product`).

        Parameters
        ----------
        other : Cycler
        """
        if not isinstance(other, Cycler):
            raise TypeError("Cannot *= with a non-Cycler object")
        # True shallow copy of self is fine since this is in-place
        old_self = copy.copy(self)
        self._keys = _process_keys(old_self, other)
        self._left = old_self
        self._op = product
        self._right = Cycler(other._left, other._right, other._op)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cycler):
            return False
        if len(self) != len(other):
            return False
        if self.keys ^ other.keys:
            return False
        return all(a == b for a, b in zip(self, other))

    __hash__ = None  # type: ignore

    def __repr__(self) -> str:
        op_map = {zip: "+", product: "*"}
        if self._right is None:
            lab = self.keys.pop()
            itr = list(v[lab] for v in self)
            return f"cycler({lab!r}, {itr!r})"
        else:
            op = op_map.get(self._op, "?")
            msg = "({left!r} {op} {right!r})"
            return msg.format(left=self._left, op=op, right=self._right)

    def _repr_html_(self) -> str:
        # an table showing the value of each key through a full cycle
        output = "<table>"
        sorted_keys = sorted(self.keys, key=repr)
        for key in sorted_keys:
            output += f"<th>{key!r}</th>"
        for d in iter(self):
            output += "<tr>"
            for k in sorted_keys:
                output += f"<td>{d[k]!r}</td>"
            output += "</tr>"
        output += "</table>"
        return output

    def by_key(self) -> dict[K, list[V]]:
        """
        Values by key.

        This returns the transposed values of the cycler.  Iterating
        over a `Cycler` yields dicts with a single value for each key,
        this method returns a `dict` of `list` which are the values
        for the given key.

        The returned value can be used to create an equivalent `Cycler`
        using only `+`.

        Returns
        -------
        transpose : dict
            dict of lists of the values for each key.
        """

        # TODO : sort out if this is a bottle neck, if there is a better way
        # and if we care.

        keys = self.keys
        out: dict[K, list[V]] = {k: list() for k in keys}

        for d in self:
            for k in keys:
                out[k].append(d[k])
        return out

    # for back compatibility
    _transpose = by_key

    def simplify(self) -> Cycler[K, V]:
        """
        Simplify the cycler into a sum (but no products) of cyclers.

        Returns
        -------
        simple : Cycler
        """
        # TODO: sort out if it is worth the effort to make sure this is
        # balanced.  Currently it is is
        # (((a + b) + c) + d) vs
        # ((a + b) + (c + d))
        # I would believe that there is some performance implications
        trans = self.by_key()
        return reduce(add, (_cycler(k, v) for k, v in trans.items()))

    concat = concat


@overload
def cycler(arg: Cycler[K, V]) -> Cycler[K, V]:
    ...


@overload
def cycler(**kwargs: Iterable[V]) -> Cycler[str, V]:
    ...


@overload
def cycler(label: K, itr: Iterable[V]) -> Cycler[K, V]:
    ...


def cycler(*args, **kwargs):
    """
    Create a new `Cycler` object from a single positional argument,
    a pair of positional arguments, or the combination of keyword arguments.

    cycler(arg)
    cycler(label1=itr1[, label2=iter2[, ...]])
    cycler(label, itr)

    Form 1 simply copies a given `Cycler` object.

    Form 2 composes a `Cycler` as an inner product of the
    pairs of keyword arguments. In other words, all of the
    iterables are cycled simultaneously, as if through zip().

    Form 3 creates a `Cycler` from a label and an iterable.
    This is useful for when the label cannot be a keyword argument
    (e.g., an integer or a name that has a space in it).

    Parameters
    ----------
    arg : Cycler
        Copy constructor for Cycler (does a shallow copy of iterables).
    label : name
        The property key. In the 2-arg form of the function,
        the label can be any hashable object. In the keyword argument
        form of the function, it must be a valid python identifier.
    itr : iterable
        Finite length iterable of the property values.
        Can be a single-property `Cycler` that would
        be like a key change, but as a shallow copy.

    Returns
    -------
    cycler : Cycler
        New `Cycler` for the given property

    """
    if args and kwargs:
        raise TypeError(
            "cycler() can only accept positional OR keyword arguments -- not both."
        )

    if len(args) == 1:
        if not isinstance(args[0], Cycler):
            raise TypeError(
                "If only one positional argument given, it must "
                "be a Cycler instance."
            )
        return Cycler(args[0])
    elif len(args) == 2:
        return _cycler(*args)
    elif len(args) > 2:
        raise TypeError(
            "Only a single Cycler can be accepted as the lone "
            "positional argument. Use keyword arguments instead."
        )

    if kwargs:
        return reduce(add, (_cycler(k, v) for k, v in kwargs.items()))

    raise TypeError("Must have at least a positional OR keyword arguments")


def _cycler(label: K, itr: Iterable[V]) -> Cycler[K, V]:
    """
    Create a new `Cycler` object from a property name and iterable of values.

    Parameters
    ----------
    label : hashable
        The property key.
    itr : iterable
        Finite length iterable of the property values.

    Returns
    -------
    cycler : Cycler
        New `Cycler` for the given property
    """
    if isinstance(itr, Cycler):
        keys = itr.keys
        if len(keys) != 1:
            msg = "Can not create Cycler from a multi-property Cycler"
            raise ValueError(msg)

        lab = keys.pop()
        # Doesn't need to be a new list because
        # _from_iter() will be creating that new list anyway.
        itr = (v[lab] for v in itr)

    return Cycler._from_iter(label, itr)
<|endoftext|>"""
requests.compat
~~~~~~~~~~~~~~~

This module previously handled import compatibility issues
between Python 2 and Python 3. It remains for backwards
compatibility until the next major version.
"""

# pyright: reportUnusedImport=false

from __future__ import annotations

import importlib
import sys
from types import ModuleType

# -------
# urllib3
# -------
from urllib3 import (
    __version__ as urllib3_version,  # type: ignore[reportPrivateImportUsage]
)

# Detect which major version of urllib3 is being used.
try:
    is_urllib3_1 = int(urllib3_version.split(".")[0]) == 1
except (TypeError, AttributeError):
    # If we can't discern a version, prefer old functionality.
    is_urllib3_1 = True

# -------------------
# Character Detection
# -------------------


def _resolve_char_detection() -> ModuleType | None:
    """Find supported character detection libraries."""
    chardet = None
    for lib in ("chardet", "charset_normalizer"):
        if chardet is None:
            try:
                chardet = importlib.import_module(lib)
            except ImportError:
                pass
    return chardet


chardet = _resolve_char_detection()

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = _ver[0] == 2

#: Python 3.x?
is_py3 = _ver[0] == 3

# json/simplejson module import resolution
has_simplejson = False
try:
    import simplejson as json  # type: ignore[import-not-found]

    has_simplejson = True
except ImportError:
    import json

if has_simplejson:
    from simplejson import JSONDecodeError  # type: ignore[import-not-found]
else:
    from json import JSONDecodeError

# Keep OrderedDict for backwards compatibility.
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from http import cookiejar as cookielib
from http.cookies import Morsel
from io import StringIO

# --------------
# Legacy Imports
# --------------
from urllib.parse import (
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunparse,
)
from urllib.request import (
    getproxies,
    getproxies_environment,
    parse_http_list,
    proxy_bypass,
    proxy_bypass_environment,  # type: ignore[attr-defined]  # https://github.com/python/cpython/issues/145331
)

builtin_str = str
str = str
bytes = bytes
basestring = (str, bytes)
numeric_types = (int, float)
integer_types = (int,)
<|endoftext|>"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `http.cookiejar.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

from __future__ import annotations

import calendar
import copy
import time
from collections.abc import Iterator, MutableMapping
from http.cookiejar import Cookie, CookieJar, CookiePolicy
from typing import TYPE_CHECKING, Any, TypeVar, overload

from ._internal_utils import to_native_string
from ._types import is_prepared as _is_prepared
from .compat import Morsel, cookielib, urlparse, urlunparse

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

    from .models import PreparedRequest

import threading


class MockRequest:
    """Wraps a `requests.PreparedRequest` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    type: str

    def __init__(self, request: PreparedRequest) -> None:
        assert _is_prepared(request)
        self._r = request
        self._new_headers: dict[str, str] = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self) -> str:
        return self.type

    def get_host(self) -> str:
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self) -> str:
        return self.get_host()

    def get_full_url(self) -> str:
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get("Host"):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = to_native_string(self._r.headers["Host"], encoding="utf-8")
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse(
            [
                parsed.scheme,
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ]
        )

    def is_unverifiable(self) -> bool:
        return True

    def has_header(self, name: str) -> bool:
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name: str, default: str | None = None) -> str | None:
        return self._r.headers.get(name, self._new_headers.get(name, default))  # type: ignore[return-value]

    def add_header(self, key: str, val: str) -> None:
        """cookiejar has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError(
            "Cookie headers should be added with add_unredirected_header()"
        )

    def add_unredirected_header(self, name: str, value: str) -> None:
        self._new_headers[name] = value

    def get_new_headers(self) -> dict[str, str]:
        return self._new_headers

    @property
    def unverifiable(self) -> bool:
        return self.is_unverifiable()

    @property
    def origin_req_host(self) -> str:
        return self.get_origin_req_host()

    @property
    def host(self) -> str:
        return self.get_host()


class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """

    def __init__(self, headers: Any) -> None:
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self) -> Any:
        return self._headers

    def getheaders(self, name: str) -> Any:
        self._headers.getheaders(name)


def extract_cookies_to_jar(
    jar: CookieJar, request: PreparedRequest, response: Any
) -> None:
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, "_original_response") and response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)  # type: ignore[arg-type]


def get_cookie_header(jar: CookieJar, request: PreparedRequest) -> str | None:
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    r = MockRequest(request)
    jar.add_cookie_header(r)  # type: ignore[arg-type]
    return r.get_new_headers().get("Cookie")


def remove_cookie_by_name(
    cookiejar: CookieJar, name: str, domain: str | None = None, path: str | None = None
) -> None:
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables: list[tuple[str, str, str]] = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """


class RequestsCookieJar(CookieJar, MutableMapping[str, str | None]):  # type: ignore[misc]
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """

    _policy: CookiePolicy

    def get(  # type: ignore[override]
        self,
        name: str,
        default: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> str | None:
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(
        self, name: str, value: str | Morsel[dict[str, str]] | None, **kwargs: Any
    ) -> Cookie | None:
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(
                self, name, domain=kwargs.get("domain"), path=kwargs.get("path")
            )
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self) -> Iterator[str]:
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name

    def keys(self) -> list[str]:  # type: ignore[override]
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())

    def itervalues(self) -> Iterator[str | None]:
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value

    def values(self) -> list[str | None]:  # type: ignore[override]
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())

    def iteritems(self) -> Iterator[tuple[str, str | None]]:
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield cookie.name, cookie.value

    def items(self) -> list[tuple[str, str | None]]:  # type: ignore[override]
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())

    def list_domains(self) -> list[str]:
        """Utility method to list all the domains in the jar."""
        domains: list[str] = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self) -> list[str]:
        """Utility method to list all the paths in the jar."""
        paths: list[str] = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self) -> bool:
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains: list[str] = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:  # type: ignore[reportUnnecessaryComparison]  # defensive check
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(
        self, domain: str | None = None, path: str | None = None
    ) -> dict[str, str | None]:
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary: dict[str, str | None] = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (
                path is None or cookie.path == path
            ):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __iter__(self) -> Iterator[Cookie]:  # type: ignore[override]
        """RequestCookieJar's __iter__ comes from CookieJar not MutableMapping."""
        return super().__iter__()

    def __contains__(self, name: object) -> bool:
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name: str) -> str | None:
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

