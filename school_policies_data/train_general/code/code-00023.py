    4. To check whether progress bars are disabled, use `are_progress_bars_disabled()`.

NOTE: Environment variable `HF_HUB_DISABLE_PROGRESS_BARS` has the priority.

Example:
    ```py
    >>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars, tqdm

    # Disable progress bars globally
    >>> disable_progress_bars()

    # Use as normal `tqdm`
    >>> for _ in tqdm(range(5)):
    ...    pass

    # Still not showing progress bars, as `disable=False` is overwritten to `True`.
    >>> for _ in tqdm(range(5), disable=False):
    ...    pass

    >>> are_progress_bars_disabled()
    True

    # Re-enable progress bars globally
    >>> enable_progress_bars()

    # Progress bar will be shown !
    >>> for _ in tqdm(range(5)):
    ...   pass
    100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]
    ```

Group-based control:
    ```python
    # Disable progress bars for a specific group
    >>> disable_progress_bars("peft.foo")

    # Check state of different groups
    >>> assert not are_progress_bars_disabled("peft"))
    >>> assert not are_progress_bars_disabled("peft.something")
    >>> assert are_progress_bars_disabled("peft.foo"))
    >>> assert are_progress_bars_disabled("peft.foo.bar"))

    # Enable progress bars for a subgroup
    >>> enable_progress_bars("peft.foo.bar")

    # Check if enabling a subgroup affects the parent group
    >>> assert are_progress_bars_disabled("peft.foo"))
    >>> assert not are_progress_bars_disabled("peft.foo.bar"))

    # No progress bar for `name="peft.foo"`
    >>> for _ in tqdm(range(5), name="peft.foo"):
    ...     pass

    # Progress bar will be shown for `name="peft.foo.bar"`
    >>> for _ in tqdm(range(5), name="peft.foo.bar"):
    ...     pass
    100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]

    ```
"""

import io
import logging
import os
import threading
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, ContextManager, TypeVar, cast

from tqdm.auto import tqdm as old_tqdm

from ..constants import HF_HUB_DISABLE_PROGRESS_BARS


# The `HF_HUB_DISABLE_PROGRESS_BARS` environment variable can be True, False, or not set (None),
# allowing for control over progress bar visibility. When set, this variable takes precedence
# over programmatic settings, dictating whether progress bars should be shown or hidden globally.
# Essentially, the environment variable's setting overrides any code-based configurations.
#
# If `HF_HUB_DISABLE_PROGRESS_BARS` is not defined (None), it implies that users can manage
# progress bar visibility through code. By default, progress bars are turned on.

progress_bar_states: dict[str, bool] = {}


class disable_progress_bars:
    """
    Disable progress bars either globally or for a specified group.

    This function updates the state of progress bars based on a group name.
    If no group name is provided, all progress bars are disabled. The operation
    respects the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable's setting.

    Works as both a regular call and a context manager:
        disable_progress_bars()           # disables until enable_progress_bars()
        with disable_progress_bars():     # disables for the block, re-enables on exit
            ...

    Args:
        name (`str`, *optional*):
            The name of the group for which to disable the progress bars. If None,
            progress bars are disabled globally.

    Raises:
        Warning: If the environment variable precludes changes.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name

        if HF_HUB_DISABLE_PROGRESS_BARS is False:
            warnings.warn(
                "Cannot disable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=0` is set and has priority."
            )
            self._should_reenable = False
            return

        self._should_reenable = not are_progress_bars_disabled(name)
        if name is None:
            progress_bar_states.clear()
            progress_bar_states["_global"] = False
        else:
            keys_to_remove = [key for key in progress_bar_states if key.startswith(f"{name}.")]
            for key in keys_to_remove:
                del progress_bar_states[key]
            progress_bar_states[name] = False

    def __enter__(self) -> "disable_progress_bars":
        return self

    def __exit__(self, *exc) -> None:
        if self._should_reenable:
            enable_progress_bars(self.name)


def enable_progress_bars(name: str | None = None) -> None:
    """
    Enable progress bars either globally or for a specified group.

    This function sets the progress bars to enabled for the specified group or globally
    if no group is specified. The operation is subject to the `HF_HUB_DISABLE_PROGRESS_BARS`
    environment setting.

    Args:
        name (`str`, *optional*):
            The name of the group for which to enable the progress bars. If None,
            progress bars are enabled globally.

    Raises:
        Warning: If the environment variable precludes changes.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        warnings.warn(
            "Cannot enable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=1` is set and has priority."
        )
        return

    if name is None:
        progress_bar_states.clear()
        progress_bar_states["_global"] = True
    else:
        keys_to_remove = [key for key in progress_bar_states if key.startswith(f"{name}.")]
        for key in keys_to_remove:
            del progress_bar_states[key]
        progress_bar_states[name] = True


def are_progress_bars_disabled(name: str | None = None) -> bool:
    """
    Check if progress bars are disabled globally or for a specific group.

    This function returns whether progress bars are disabled for a given group or globally.
    It checks the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable first, then the programmatic
    settings.

    Args:
        name (`str`, *optional*):
            The group name to check; if None, checks the global setting.

    Returns:
        `bool`: True if progress bars are disabled, False otherwise.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        return True

    if name is None:
        return not progress_bar_states.get("_global", True)

    while name:
        if name in progress_bar_states:
            return not progress_bar_states[name]
        name = ".".join(name.split(".")[:-1])

    return not progress_bar_states.get("_global", True)


def is_tqdm_disabled(log_level: int) -> bool | None:
    """
    Determine if tqdm progress bars should be disabled based on logging level and environment settings.

    see https://github.com/huggingface/huggingface_hub/pull/2000 and https://github.com/huggingface/huggingface_hub/pull/2698.
    """
    if log_level == logging.NOTSET:
        return True
    if os.getenv("TQDM_POSITION") == "-1":
        return False
    return None


class tqdm(old_tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)  # do not pass `name` to `tqdm`
        if are_progress_bars_disabled(name):
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


# Prevent tqdm's default multiprocessing write-lock from spawning a resource
# tracker subprocess via fork_exec(). That path fails when stderr has an invalid
# fd (e.g. Textual TUIs that return -1 from sys.stderr.fileno()). Inter-process
# bar coordination on the HF subclass is not a supported use case. See #4065.
tqdm.set_lock(threading.RLock())


class silent_tqdm:
    """Fake tqdm object that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def update(self, n: int | float | None = 1) -> None:
        pass


@contextmanager
def tqdm_stream_file(path: Path | str) -> Iterator[io.BufferedReader]:
    """
    Open a file as binary and wrap the `read` method to display a progress bar when it's streamed.

    First implemented in `transformers` in 2019 but removed when switched to git-lfs. Used in `huggingface_hub` to show
    progress bar when uploading an LFS file to the Hub. See github.com/huggingface/transformers/pull/2078#discussion_r354739608
    for implementation details.

    Note: currently implementation handles only files stored on disk as it is the most common use case. Could be
          extended to stream any `BinaryIO` object but we might have to debug some corner cases.

    Example:
    ```py
    >>> with tqdm_stream_file("config.json") as f:
    >>>     httpx2.put(url, data=f)
    config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
    ```
    """
    if isinstance(path, str):
        path = Path(path)

    with path.open("rb") as f:
        total_size = path.stat().st_size
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            total=total_size,
            initial=0,
            desc=path.name,
        )

        f_read = f.read

        def _inner_read(size: int | None = -1) -> bytes:
            data = f_read(size)
            pbar.update(len(data))
            return data

        f.read = _inner_read  # type: ignore

        yield f

        pbar.close()


def _create_progress_bar(*, cls: type[old_tqdm], log_level: int, name: str | None = None, **kwargs) -> old_tqdm:
    """Create a progress bar.

    For our `tqdm` subclass (or subclasses of it): respects all disable signals
    (`HF_HUB_DISABLE_PROGRESS_BARS`, `disable_progress_bars()`, log level) and uses
    `disable=None` for TTY auto-detection (see https://github.com/huggingface/huggingface_hub/pull/2000),
    unless `TQDM_POSITION=-1` forces bars on (https://github.com/huggingface/huggingface_hub/pull/2698).

    For other classes: does not inject `disable` or `name`. the custom class is fully
    responsible for its own behavior. Vanilla tqdm defaults to `disable=False` (bar shows).
    Omits `name` which vanilla tqdm rejects with `TqdmKeyError`. See https://github.com/huggingface/huggingface_hub/issues/4050.
    """
    # issubclass() crashes on non-class callables (e.g. functools.partial), guard with isinstance.
    if not (isinstance(cls, type) and issubclass(cls, tqdm)):
        return cls(**kwargs)  # type: ignore[return-value]

    # HF subclass: keep the historical log-level / TTY behavior. Group-based
    # disabling is already handled in `tqdm.__init__`.
    disable = is_tqdm_disabled(log_level)
    return cls(disable=disable, name=name, **kwargs)  # type: ignore[return-value]


def _get_progress_bar_context(
    *,
    desc: str,
    log_level: int,
    total: int | None = None,
    initial: int = 0,
    unit: str = "B",
    unit_scale: bool = True,
    name: str | None = None,
    tqdm_class: type[old_tqdm] | None = None,
    _tqdm_bar: tqdm | None = None,
) -> ContextManager[tqdm]:
    if _tqdm_bar is not None:
        return nullcontext(_tqdm_bar)
        # ^ `contextlib.nullcontext` mimics a context manager that does nothing
        #   Makes it easier to use the same code path for both cases but in the later
        #   case, the progress bar is not closed when exiting the context manager.

    return _create_progress_bar(  # type: ignore
        cls=tqdm_class or tqdm,
        log_level=log_level,
        name=name,
        unit=unit,
        unit_scale=unit_scale,
        total=total,
        initial=initial,
        desc=desc,
    )


T = TypeVar("T")
R = TypeVar("R")


def _make_thread_pool_executor(*, max_workers: int | None, bar_class: type[old_tqdm]) -> ThreadPoolExecutor:
    """Create a `ThreadPoolExecutor` that shares the tqdm lock with its worker threads.

    Like `tqdm.contrib.concurrent.thread_map`, we make sure the bar class has a lock and pass it to the
    worker threads, so that bars created inside the mapped function (e.g. the per-file download bars in
    `snapshot_download`) don't race on concurrent `update()` calls. `get_lock()` creates the shared
    class-level lock if needed. Classes that don't implement tqdm's locking API (e.g. `silent_tqdm`)
    render nothing anyway, so no lock is needed.
    """
    if hasattr(bar_class, "get_lock") and hasattr(bar_class, "set_lock"):
        return ThreadPoolExecutor(
            max_workers=max_workers, initializer=bar_class.set_lock, initargs=(bar_class.get_lock(),)
        )
    return ThreadPoolExecutor(max_workers=max_workers)


def hf_thread_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    max_workers: int | None = None,
    tqdm_class: type[old_tqdm] | None = None,
    **tqdm_kwargs,
) -> list[R]:
    """Drop-in replacement for `tqdm.contrib.concurrent.thread_map`.

    The version shipped by `tqdm` consumes results in submission order and the progress bar only advances when the "next
    item in input order" finishes. A single slow early item pins the bar near zero while later items
    complete, with everything flushed at once at the end. This helper avoids that problem by consuming items in completion
    order.

    See https://github.com/huggingface/huggingface_hub/issues/4518 for more details.

    Args:
        fn (`Callable`):
            Function to apply to each item.
        iterable (`Iterable`):
            Items to process.
        max_workers (`int`, *optional*):
            Maximum number of worker threads. Defaults to tqdm's own default.
        tqdm_class (`type`, *optional*):
            Progress bar class to use. Defaults to `huggingface_hub`'s `tqdm`.
        **tqdm_kwargs:
            Additional keyword arguments forwarded to the progress bar (e.g. `desc`).

    Returns:
        `list`: Results in the same order as `iterable`.
    """
    items = list(iterable)
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    tqdm_kwargs.setdefault("total", len(items))
    bar_class = tqdm_class or tqdm
    results: list[R | None] = [None] * len(items)
    with (
        bar_class(**tqdm_kwargs) as pbar,
        _make_thread_pool_executor(max_workers=max_workers, bar_class=bar_class) as executor,
    ):
        future_to_index = {executor.submit(fn, item): index for index, item in enumerate(items)}
        try:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
                pbar.update(1)
        except BaseException:
            # Cancel not-yet-started tasks so a single failure doesn't keep the whole queue running
            # (mirrors `tqdm.contrib.concurrent.thread_map`, which relies on `Executor.map` for this).
            for future in future_to_index:
                future.cancel()
            raise
    return cast("list[R]", results)
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Handle typing imports based on system compatibility."""

from collections.abc import Callable
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin


UNION_TYPES: list[Any] = [Union, UnionType]


HTTP_METHOD_T = Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]

# type hint meaning "function signature not changed by decorator"
CallableT = TypeVar("CallableT", bound=Callable)

_JSON_SERIALIZABLE_TYPES = (int, float, str, bool, type(None))


def is_jsonable(obj: Any, _visited: set[int] | None = None) -> bool:
    """Check if an object is JSON serializable.

    This is a weak check, as it does not check for the actual JSON serialization, but only for the types of the object.
    It works correctly for basic use cases but do not guarantee an exhaustive check.

    Object is considered to be recursively json serializable if:
    - it is an instance of int, float, str, bool, or NoneType
    - it is a list or tuple and all its items are json serializable
    - it is a dict and all its keys are strings and all its values are json serializable

    Uses a visited set to avoid infinite recursion on circular references. If object has already been visited, it is
    considered not json serializable.
    """
    # Initialize visited set to track object ids and detect circular references
    if _visited is None:
        _visited = set()

    # Detect circular reference
    obj_id = id(obj)
    if obj_id in _visited:
        return False

    # Add current object to visited before recursive checks
    _visited.add(obj_id)
    try:
        if isinstance(obj, _JSON_SERIALIZABLE_TYPES):
            return True
        if isinstance(obj, (list, tuple)):
            return all(is_jsonable(item, _visited) for item in obj)
        if isinstance(obj, dict):
            return all(
                isinstance(key, _JSON_SERIALIZABLE_TYPES) and is_jsonable(value, _visited)
                for key, value in obj.items()
            )
        if hasattr(obj, "__json__"):
            return True
        return False
    except RecursionError:
        return False
    finally:
        # Remove the object id from visited to avoid side‑effects for other branches
        _visited.discard(obj_id)


def is_simple_optional_type(type_: type) -> bool:
    """Check if a type is optional, i.e. Optional[Type] or Union[Type, None] or Type | None, where Type is a non-composite type."""
    if get_origin(type_) in UNION_TYPES:
        union_args = get_args(type_)
        if len(union_args) == 2 and type(None) in union_args:
            return True
    return False


def unwrap_simple_optional_type(optional_type: type) -> type:
    """Unwraps a simple optional type, i.e. returns Type from Optional[Type]."""
    for arg in get_args(optional_type):
        if arg is not type(None):
            return arg
    raise ValueError(f"'{optional_type}' is not an optional type")
<|endoftext|>import contextlib
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path

import yaml
from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout

from .. import constants
from . import logging


logger = logging.get_logger(__name__)

# Wrap `yaml.dump` to set `allow_unicode=True` by default.
#
# Example:
# ```py
# >>> yaml.dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "\\U0001F440"\nsome unicode: "\\u65E5\\u672C\\u304B"\n'
#
# >>> yaml_dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "👀"\nsome unicode: "日本か"\n'
# ```
yaml_dump: Callable[..., str] = partial(yaml.dump, stream=None, allow_unicode=True)  # type: ignore


@contextlib.contextmanager
def SoftTemporaryDirectory(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: Path | str | None = None,
    **kwargs,
) -> Generator[Path, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield Path(tmpdir.name).resolve()

    try:
        # First once with normal cleanup
        shutil.rmtree(tmpdir.name)
    except Exception:
        # If failed, try to set write permission and retry
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass

    # And finally, cleanup the tmpdir.
    # If it fails again, give up but do not throw error
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


@contextlib.contextmanager
def WeakFileLock(lock_file: str | Path, *, timeout: float | None = None) -> Generator[BaseFileLock, None, None]:
    """A filelock with some custom logic.

    This filelock is weaker than the default filelock in that:
    1. It won't raise an exception if release fails.
    2. It will default to a SoftFileLock if the filesystem does not support flock.
    3. Lock files are created with mode 0o664 (group-writable) instead of the default 0o644.
       This allows multiple users sharing a cache directory to wait for locks.

    An INFO log message is emitted every 10 seconds if the lock is not acquired immediately.
    If a timeout is provided, a `filelock.Timeout` exception is raised if the lock is not acquired within the timeout.
    """
    log_interval = constants.FILELOCK_LOG_EVERY_SECONDS
    lock = FileLock(lock_file, timeout=log_interval, mode=0o664)
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            raise Timeout(str(lock_file))

        try:
            lock.acquire(timeout=min(log_interval, timeout - elapsed_time) if timeout else log_interval)
        except Timeout:
            logger.info(
                f"Still waiting to acquire lock on {lock_file} (elapsed: {time.time() - start_time:.1f} seconds)"
            )
        except NotImplementedError as e:
            if "use SoftFileLock instead" in str(e):
                logger.warning(
                    "FileSystem does not appear to support flock. Falling back to SoftFileLock for %s", lock_file
                )
                lock = SoftFileLock(lock_file, timeout=log_interval)
                continue
        else:
            break

    try:
        yield lock
    finally:
        try:
            lock.release()
        except OSError:
            try:
                Path(lock_file).unlink()
            except OSError:
                pass
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to validate argument values in `huggingface_hub`."""

import inspect
import re
import warnings
from functools import wraps
from itertools import chain
from typing import Any

from huggingface_hub.errors import HFValidationError

from ._typing import CallableT


REPO_ID_REGEX = re.compile(
    r"""
    (\b[\w\-.]{1,96}\b/)? # optional namespace (username or organization)
    \b               # starts with a word boundary
    [\w\-.]{1,96}    # repo_name: alphanumeric + . _ -
    \b               # ends with a word boundary
    """,
    # `\w` matches Unicode word characters by default, but the allowed charset is
    # ASCII-only (see the "Rules" docstring below and the internal Hub validation
    # it mirrors), so `café` or `文件夹` would otherwise pass this regex unrejected.
    flags=re.VERBOSE | re.ASCII,
)


def validate_hf_hub_args(fn: CallableT) -> CallableT:
    """Validate values received as argument for any public method of `huggingface_hub`.

    The goal of this decorator is to harmonize validation of arguments reused
    everywhere. By default, all defined validators are tested.

    Validators:
        - [`~utils.validate_repo_id`]: `repo_id` must be `"repo_name"`
          or `"namespace/repo_name"`. Namespace is a username or an organization.
        - [`~utils.smoothly_deprecate_legacy_arguments`]: Ignore `proxies` when downloading files (should be set globally).

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_hf_hub_args

    >>> @validate_hf_hub_args
    ... def my_cool_method(repo_id: str):
    ...     print(repo_id)

    >>> my_cool_method(repo_id="valid_repo_id")
    valid_repo_id

    >>> my_cool_method("other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

    >>> my_cool_method(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
    ```

    Raises:
        [`~utils.HFValidationError`]:
            If an input is not valid.
    """
    # TODO: add an argument to opt-out validation for specific argument?
    signature = inspect.signature(fn)

    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        for arg_name, arg_value in chain(
            zip(signature.parameters, args),  # Args values
            kwargs.items(),  # Kwargs values
        ):
            if arg_name in ["repo_id", "from_id", "to_id"]:
                validate_repo_id(arg_value)

        kwargs = smoothly_deprecate_legacy_arguments(fn_name=fn.__name__, kwargs=kwargs)

        return fn(*args, **kwargs)

    return _inner_fn  # type: ignore


def validate_repo_id(repo_id: str | None) -> None:
    """Validate `repo_id` is valid.

    This is not meant to replace the proper validation made on the Hub but rather to
    avoid local inconsistencies whenever possible (example: passing `repo_type` in the
    `repo_id` is forbidden).

    Rules:
    - Between 1 and 96 characters.
    - Either "repo_name" or "namespace/repo_name"
    - [a-zA-Z0-9] or "-", "_", "."
    - "--" and ".." are forbidden

    Valid: `"foo"`, `"foo/bar"`, `"123"`, `"Foo-BAR_foo.bar123"`

    Not valid: `"datasets/foo/bar"`, `".repo_id"`, `"foo--bar"`, `"foo.git"`

    Example:
    ```py
    >>> from huggingface_hub.utils import validate_repo_id
    >>> validate_repo_id(repo_id="valid_repo_id")
    >>> validate_repo_id(repo_id="other..repo..id")
    huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
    ```

    Discussed in https://github.com/huggingface/huggingface_hub/issues/1008.
    In moon-landing (internal repository):
    - https://github.com/huggingface/moon-landing/blob/main/server/lib/Names.ts#L27
    - https://github.com/huggingface/moon-landing/blob/main/server/views/components/NewRepoForm/NewRepoForm.svelte#L138
    """
    if repo_id is None:
        # Repo id is not always required
        return

    if not isinstance(repo_id, str):
        # Typically, a Path is not a repo_id
        raise HFValidationError(f"Repo id must be a string, not {type(repo_id)}: '{repo_id}'.")

    if repo_id.count("/") > 1:
        raise HFValidationError(
            "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
            f" '{repo_id}'. Use `repo_type` argument if needed."
        )

    if not REPO_ID_REGEX.fullmatch(repo_id):
        raise HFValidationError(
            "Repo id must use alphanumeric chars, '-', '_' or '.'."
            " The name cannot start or end with '-' or '.' and the maximum length is 96:"
            f" '{repo_id}'."
        )

    if "--" in repo_id or ".." in repo_id:
        raise HFValidationError(f"Cannot have -- or .. in repo_id: '{repo_id}'.")

    if repo_id.endswith(".git"):
        raise HFValidationError(f"Repo_id cannot end by '.git': '{repo_id}'.")


def smoothly_deprecate_legacy_arguments(fn_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Smoothly deprecate legacy arguments in the `huggingface_hub` codebase.

    This function ignores some deprecated arguments from the kwargs and warns the user they are ignored.
    The goal is to avoid breaking existing code while guiding the user to the new way of doing things.

    List of deprecated arguments:
        - `proxies`:
            To set up proxies, user must either use the HTTP_PROXY environment variable or configure the `httpx2.Client`
            manually using the [`set_client_factory`] function.

            In huggingface_hub 0.x, `proxies` was a dictionary directly passed to `requests.request`.
            Since huggingface_hub 1.x, per-request `proxies` are no longer supported.
            In particular, it is not possible to configure proxies on a per-request basis. The solution is to configure
            it globally using the [`set_client_factory`] function or using the HTTP_PROXY environment variable.

            For more details, see:
            - https://httpx2.pydantic.dev/advanced/proxies/

        - `resume_download`: deprecated without replacement. `huggingface_hub` always resumes downloads whenever possible.
        - `force_filename`: deprecated without replacement. Filename is always the same as on the Hub.
        - `local_dir_use_symlinks`: deprecated without replacement. Downloading to a local directory does not use symlinks anymore.
    """
    new_kwargs = kwargs.copy()  # do not mutate input !

    # proxies
    proxies = new_kwargs.pop("proxies", None)  # remove from kwargs
    if proxies is not None:
        warnings.warn(
            f"The `proxies` argument is ignored in `{fn_name}`. To set up proxies, use the HTTP_PROXY / HTTPS_PROXY"
            " environment variables or configure the `httpx2.Client` manually using `huggingface_hub.set_client_factory`."
            " See https://httpx2.pydantic.dev/advanced/proxies/ for more details."
        )

    # resume_download
    resume_download = new_kwargs.pop("resume_download", None)  # remove from kwargs
    if resume_download is not None:
        warnings.warn(
            f"The `resume_download` argument is deprecated and ignored in `{fn_name}`. Downloads always resume"
            " whenever possible."
        )

    # force_filename
    force_filename = new_kwargs.pop("force_filename", None)  # remove from kwargs
    if force_filename is not None:
        warnings.warn(
            f"The `force_filename` argument is deprecated and ignored in `{fn_name}`. Filename is always the same "
            "as on the Hub."
        )

    # local_dir_use_symlinks
    local_dir_use_symlinks = new_kwargs.pop("local_dir_use_symlinks", None)  # remove from kwargs
    if local_dir_use_symlinks is not None:
        warnings.warn(
            f"The `local_dir_use_symlinks` argument is deprecated and ignored in `{fn_name}`. Downloading to a local"
            " directory does not use symlinks anymore."
        )

    return new_kwargs
<|endoftext|># AI-generated module (ChatGPT)
import re


# Escape sequences expanded inside quoted values. Double-quoted values additionally
# expand "\$" to "$"; single-quoted values keep it verbatim.
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
_DOUBLE_QUOTE_ESCAPES = {**_ESCAPES, "$": "$"}


def _unescape(value: str, escapes: dict[str, str]) -> str:
    r"""Expand backslash escapes in a single left-to-right pass.

    Processing in one pass (rather than chained `str.replace` calls) ensures an escaped
    backslash (`\\`) is consumed as a unit and cannot merge with the following character,
    e.g. `\\n` is a backslash followed by `n`, not a newline. Unknown escapes are kept as-is.
    """
    return re.sub(r"\\(.)", lambda match: escapes.get(match.group(1), match.group(0)), value)


def load_dotenv(dotenv_str: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """
    Parse a DOTENV-format string and return a dictionary of key-value pairs.
    Handles quoted values, comments, export keyword, and blank lines.
    """
    env: dict[str, str] = {}
    line_pattern = re.compile(
        r"""
        \s*
        (?:export[^\S\n]+)?               # optional export
        ([A-Za-z_][A-Za-z0-9_]*)          # key
        (?:
            [^\S\n]*
            (=)                           # equal sign
            (?:
                [^\S\n]*
                (                         # quoted value
                    '(?:\\'|[^'])*'       # single-quoted
                    | \"(?:\\\"|[^\"])*\" # double-quoted
                )
                [^\S\n]*(?:\#[^\n\r]*)?   # inline comment (needs no preceding whitespace after a quote)
                |
                (?:[^\S\n]+(?!\#))?       # whitespace after '=' also separates an inline comment
                (                         # unquoted value (may contain '#')
                    [^\n\r]*?
                )
                (?:[^\S\n]+\#[^\n\r]*)?   # inline comment (must be preceded by whitespace)
            )
            |
            [^\S\n]*(?:\#[^\n\r]*)?       # bare key (no '='), with an optional inline comment
        )
    """,
        re.VERBOSE,
    )

    for line in dotenv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip comments and empty lines

        match = line_pattern.fullmatch(line)
        if match:
            key = match.group(1)
            val = None
            if match.group(2):  # if there is '='
                raw_val = match.group(3) or match.group(4) or ""
                val = raw_val.strip()
                # Remove surrounding quotes if quoted
                if val.startswith('"') and val.endswith('"'):
                    # Double-quoted values expand escape sequences (\n, \t, \", \\, \$).
                    val = _unescape(val[1:-1], _DOUBLE_QUOTE_ESCAPES)
                elif val.startswith("'") and val.endswith("'"):
                    # Single-quoted values are kept verbatim: no escape expansion.
                    val = val[1:-1]
            elif environ is not None:
                # Get it from the current environment
                val = environ.get(key)

            if val is not None:
                env[key] = val

    return env
<|endoftext|>import functools
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal


FILENAME_T = str
TENSOR_NAME_T = str
DTYPE_T = Literal["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "U3", "BOOL"]


@dataclass
class TensorInfo:
    """Information about a tensor.

    For more details regarding the safetensors format, check out https://huggingface.co/docs/safetensors/index#format.

    Attributes:
        dtype (`str`):
            The data type of the tensor ("F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "U3", "BOOL").
        shape (`list[int]`):
            The shape of the tensor.
        data_offsets (`tuple[int, int]`):
            The offsets of the data in the file as a tuple `[BEGIN, END]`.
        parameter_count (`int`):
            The number of parameters in the tensor.
    """

    dtype: DTYPE_T
    shape: list[int]
    data_offsets: tuple[int, int]
    parameter_count: int = field(init=False)

    def __post_init__(self) -> None:
        # Taken from https://stackoverflow.com/a/13840436
        try:
            self.parameter_count = functools.reduce(operator.mul, self.shape)
        except TypeError:
            self.parameter_count = 1  # scalar value has no shape


@dataclass
class SafetensorsFileMetadata:
    """Metadata for a Safetensors file hosted on the Hub.

    This class is returned by [`parse_safetensors_file_metadata`].

    For more details regarding the safetensors format, check out https://huggingface.co/docs/safetensors/index#format.

    Attributes:
        metadata (`dict`):
            The metadata contained in the file.
        tensors (`dict[str, TensorInfo]`):
            A map of all tensors. Keys are tensor names and values are information about the corresponding tensor, as a
            [`TensorInfo`] object.
        parameter_count (`dict[str, int]`):
            A map of the number of parameters per data type. Keys are data types and values are the number of parameters
            of that data type.
    """

    metadata: dict[str, str]
    tensors: dict[TENSOR_NAME_T, TensorInfo]
    parameter_count: dict[DTYPE_T, int] = field(init=False)

    def __post_init__(self) -> None:
        parameter_count: dict[DTYPE_T, int] = defaultdict(int)
        for tensor in self.tensors.values():
            parameter_count[tensor.dtype] += tensor.parameter_count
        self.parameter_count = dict(parameter_count)


@dataclass
class SafetensorsRepoMetadata:
    """Metadata for a Safetensors repo.

    A repo is considered to be a Safetensors repo if it contains either a 'model.safetensors' weight file (non-shared
    model) or a 'model.safetensors.index.json' index file (sharded model) at its root.

    This class is returned by [`get_safetensors_metadata`].

    For more details regarding the safetensors format, check out https://huggingface.co/docs/safetensors/index#format.

    Attributes:
        metadata (`dict`, *optional*):
            The metadata contained in the 'model.safetensors.index.json' file, if it exists. Only populated for sharded
            models.
        sharded (`bool`):
            Whether the repo contains a sharded model or not.
        weight_map (`dict[str, str]`):
            A map of all weights. Keys are tensor names and values are filenames of the files containing the tensors.
        files_metadata (`dict[str, SafetensorsFileMetadata]`):
            A map of all files metadata. Keys are filenames and values are the metadata of the corresponding file, as
            a [`SafetensorsFileMetadata`] object.
        parameter_count (`dict[str, int]`):
            A map of the number of parameters per data type. Keys are data types and values are the number of parameters
            of that data type.
    """

    metadata: dict | None
    sharded: bool
    weight_map: dict[TENSOR_NAME_T, FILENAME_T]  # tensor name -> filename
    files_metadata: dict[FILENAME_T, SafetensorsFileMetadata]  # filename -> metadata
    parameter_count: dict[DTYPE_T, int] = field(init=False)

    def __post_init__(self) -> None:
        parameter_count: dict[DTYPE_T, int] = defaultdict(int)
        for file_metadata in self.files_metadata.values():
            for dtype, nb_parameters_ in file_metadata.parameter_count.items():
                parameter_count[dtype] += nb_parameters_
        self.parameter_count = dict(parameter_count)
<|endoftext|># Licensed under the Apache License, Version 2.0 (the "License");
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
"""
Helpful utility functions and classes in relation to exploring API endpoints
with the aim for a user-friendly interface.
"""

import math
import re
from typing import TYPE_CHECKING

from ..repocard_data import ModelCardData


if TYPE_CHECKING:
    from ..hf_api import ModelInfo


def _is_emission_within_threshold(model_info: "ModelInfo", minimum_threshold: float, maximum_threshold: float) -> bool:
    """Checks if a model's emission is within a given threshold.

    Args:
        model_info (`ModelInfo`):
            A model info object containing the model's emission information.
        minimum_threshold (`float`):
            A minimum carbon threshold to filter by, such as 1.
        maximum_threshold (`float`):
            A maximum carbon threshold to filter by, such as 10.

    Returns:
        `bool`: Whether the model's emission is within the given threshold.
    """
    if minimum_threshold is None and maximum_threshold is None:
        raise ValueError("Both `minimum_threshold` and `maximum_threshold` cannot both be `None`")
    if minimum_threshold is None:
        minimum_threshold = -1
    if maximum_threshold is None:
        maximum_threshold = math.inf

    card_data = getattr(model_info, "card_data", None)
    if card_data is None or not isinstance(card_data, (dict, ModelCardData)):
        return False

    # Get CO2 emission metadata
    emission = card_data.get("co2_eq_emissions", None)
    if isinstance(emission, dict):
        emission = emission["emissions"]
    if not emission:
        return False

    # Filter out if value is missing or out of range
    matched = re.search(r"\d+\.\d+|\d+", str(emission))
    if matched is None:
        return False

    emission_value = float(matched.group(0))
    return minimum_threshold <= emission_value <= maximum_threshold
<|endoftext|># Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Contains a helper to get the token from machine (env variable, secret or config file)."""

import configparser
import io
import logging
import os
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import TypedDict

from .. import constants
from ..errors import DeviceCodeError, OAuthErrorCode, OIDCError
from ._fixes import WeakFileLock
from ._oauth_device import refresh_access_token
from ._runtime import is_colab_enterprise, is_google_colab


_SECRET_FILE_MODE = 0o600
_SECRET_DIR_MODE = 0o700


def _write_secret(path: Path, content: str) -> None:
    """Write content to file, restricting both the file and its parent directory to owner-only on POSIX systems."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=_SECRET_DIR_MODE)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
    # Token names are arbitrary Unicode (a user-set `displayName`), so write UTF-8 rather
    # than the locale encoding, which cannot represent them on e.g. Windows.
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        path.chmod(_SECRET_FILE_MODE)
        path.parent.chmod(_SECRET_DIR_MODE)
    except (OSError, NotImplementedError):
        # Windows does not support POSIX modes; chmod() will raise. Best-effort.
        pass


_IS_GOOGLE_COLAB_CHECKED = False
_GOOGLE_COLAB_SECRET_LOCK = Lock()
_GOOGLE_COLAB_SECRET: str | None = None

logger = logging.getLogger(__name__)


def get_token() -> str | None:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`huggingface_hub.utils.build_hf_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    If `HF_OIDC_RESOURCE` is set (Trusted Publishers, typically in CI), a short-lived token obtained via OIDC token
    exchange takes precedence. Otherwise the token is retrieved from the `HF_TOKEN` environment variable, then from the
    token file in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
    `hf auth login`.

    OAuth tokens obtained with the browser-based login come with a refresh token: when such a token is close to
    expiry, it is transparently refreshed and persisted before being returned.

    Note: if `HF_OIDC_RESOURCE` is set but the OIDC token exchange fails, this raises instead of returning `None`,
    opting into OIDC is explicit, so a failure surfaces as a clear error rather than a silent fallback.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return (
        _get_token_from_oidc()
        or _get_token_from_environment()
        or _get_token_from_file_refreshed()
        or _get_token_from_google_colab()
    )


def _get_token_from_google_colab() -> str | None:
    """Get token from Google Colab secrets vault using `google.colab.userdata.get(...)`.

    Token is read from the vault only once per session and then stored in a global variable to avoid re-requesting
    access to the vault.
    """
    # If it's not a Google Colab or it's Colab Enterprise, fallback to environment variable or token file authentication
    if not is_google_colab() or is_colab_enterprise():
        return None

    # `google.colab.userdata` is not thread-safe
    # This can lead to a deadlock if multiple threads try to access it at the same time
    # (typically when using `snapshot_download`)
    # => use a lock
    # See https://github.com/huggingface/huggingface_hub/issues/1952 for more details.
    with _GOOGLE_COLAB_SECRET_LOCK:
        global _GOOGLE_COLAB_SECRET
        global _IS_GOOGLE_COLAB_CHECKED

        if _IS_GOOGLE_COLAB_CHECKED:  # request access only once
            return _GOOGLE_COLAB_SECRET

        try:
            from google.colab import userdata  # type: ignore
            from google.colab.errors import Error as ColabError  # type: ignore
        except ImportError:
            return None

        try:
            token = userdata.get("HF_TOKEN")
            _GOOGLE_COLAB_SECRET = _clean_token(token)
        except userdata.NotebookAccessError:
            # Means the user has a secret call `HF_TOKEN` and got a popup "please grand access to HF_TOKEN" and refused it
            # => warn user but ignore error => do not re-request access to user
            warnings.warn(
                "\nAccess to the secret `HF_TOKEN` has not been granted on this notebook."
                "\nYou will not be requested again."
                "\nPlease restart the session if you want to be prompted again."
            )
            _GOOGLE_COLAB_SECRET = None
        except userdata.SecretNotFoundError:
            # No `HF_TOKEN` secret defined: simply not logged in via the Colab vault. Not worth a
            # warning now that `login()` is the primary flow (it would even fire during `login()`
            # itself, telling the user to set up a secret while they are busy authenticating).
            logger.info(
                "The secret `HF_TOKEN` does not exist in your Colab secrets. Run `huggingface_hub.login()` to"
                " authenticate (recommended but still optional to access public models or datasets)."
            )
            _GOOGLE_COLAB_SECRET = None
        except ColabError as e:
            # Something happen but we don't know what => recommend to open a GitHub issue
            warnings.warn(f"\nError while fetching `HF_TOKEN` secret value from your vault: '{str(e)}'.")
            _GOOGLE_COLAB_SECRET = None

        _IS_GOOGLE_COLAB_CHECKED = True
        return _GOOGLE_COLAB_SECRET


def _get_token_from_environment() -> str | None:
    # `HF_TOKEN` has priority (keep `HUGGING_FACE_HUB_TOKEN` for backward compatibility)
    return _clean_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _get_token_from_file() -> str | None:
    try:
        return _clean_token(Path(constants.HF_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


class _OidcTokenCache(TypedDict):
    resource: str
    token: str
    expires_at: float  # monotonic clock value after which the cached token must be re-exchanged


# Cache for the OIDC-exchanged token: re-exchanging on every `get_token()` call would be wasteful,
# and re-exchanging shortly before expiry transparently keeps long-running jobs authenticated.
_OIDC_TOKEN_LOCK = Lock()
_OIDC_TOKEN_CACHE: _OidcTokenCache | None = None
_OIDC_REFRESH_MARGIN = 300  # re-exchange this many seconds before the token actually expires


def _get_token_from_oidc() -> str | None:
    """Get a short-lived OIDC token in CI (Trusted Publishers).

    Enabled by setting `HF_OIDC_RESOURCE`, which scopes the token to a repo or user.
    The ID token is read from `HF_OIDC_ID_TOKEN` if available, or minted from a supported CI provider (e.g. GitHub Actions).

    Returns `None` when OIDC is not enabled.
    If enabled, any failure is raised explicitly rather than falling back silently.

    See `huggingface_hub._oidc` and https://huggingface.co/docs/hub/trusted-publishers.
    """
    resource = os.environ.get("HF_OIDC_RESOURCE")
    if not resource:
        return None

    from .._oidc import detect_provider, oidc_login

    global _OIDC_TOKEN_CACHE
    with _OIDC_TOKEN_LOCK:
        now = time.monotonic()
        if (
            _OIDC_TOKEN_CACHE is not None
            and _OIDC_TOKEN_CACHE["resource"] == resource
            and now < _OIDC_TOKEN_CACHE["expires_at"]
        ):
            return _OIDC_TOKEN_CACHE["token"]

        # An explicit id token (any provider) takes precedence; otherwise mint from a detected one.
        subject_token = os.environ.get("HF_OIDC_ID_TOKEN") or None
        if subject_token is None and detect_provider() is None:
            raise OIDCError(
                "HF_OIDC_RESOURCE is set but no OIDC id token is available: not running in a supported "
                "CI provider (github) and HF_OIDC_ID_TOKEN is not set. Set HF_OIDC_ID_TOKEN to the id "
                "token minted by your CI provider, or unset HF_OIDC_RESOURCE."
            )

        result = oidc_login(resource=resource, subject_token=subject_token)
        token = result["access_token"]
        expires_in = int(result.get("expires_in", 3600))
        # A pre-supplied HF_OIDC_ID_TOKEN can't be re-minted, so refreshing early is pointless (the id
        # token is likely already expired by then): cache for the full lifetime. Only the auto-minted
        # path can refresh, so only it gets the safety margin.
        margin = 0 if subject_token is not None else _OIDC_REFRESH_MARGIN
        _OIDC_TOKEN_CACHE = {
            "resource": resource,
            "token": token,
            "expires_at": now + max(expires_in - margin, 0),
        }
        return token


class _OAuthRefreshCache(TypedDict):
    file_token: str  # token as read from HF_TOKEN_PATH (cache key)
    resolved_token: str  # token to return (refreshed, or identical if no refresh was needed)
    recheck_at: float  # wall-clock timestamp after which the expiry must be re-evaluated


# Cache the refresh decision in-process: `get_token()` is called on every HTTP request and must not
# re-read the stored tokens file (let alone hit the network) each time.
_OAUTH_REFRESH_LOCK = Lock()
_OAUTH_REFRESH_CACHE: _OAuthRefreshCache | None = None
_OAUTH_REFRESH_MARGIN = 24 * 3600  # refresh when less than 1 day of validity remains
_OAUTH_RECHECK_INTERVAL = 300  # re-check interval when there is no metadata or the refresh failed
_OAUTH_REFRESH_WARNED = False  # warn at most once per process on refresh failure


def _get_token_from_file_refreshed() -> str | None:
    """Get the token from `HF_TOKEN_PATH`, transparently refreshing it if close to expiry."""
    token = _get_token_from_file()
    if token is None:
        return None
    return _refresh_oauth_token_if_needed(token)


def _refresh_oauth_token_if_needed(token: str) -> str:
    """Refresh an OAuth access token if it is close to expiry. Best-effort: never raises.

    OAuth tokens obtained with the browser-based login are stored with a `refresh_token` and an
    `expires_at` timestamp (see `_save_token`). When the active token is one of them and about to
    expire, exchange the refresh token for a new access token and persist it. Any other token is
    returned unchanged.
    """
    global _OAUTH_REFRESH_CACHE
    with _OAUTH_REFRESH_LOCK:
        now = time.time()
        cache = _OAUTH_REFRESH_CACHE
        if cache is not None and cache["file_token"] == token and now < cache["recheck_at"]:
            return cache["resolved_token"]

        token_name, fields = next(
            ((name, fields) for name, fields in _read_stored_tokens_full().items() if fields.get("hf_token") == token),
            (None, {}),
        )
        refresh_token = fields.get("refresh_token")
        expires_at = _parse_expires_at(fields)
        if token_name is None or refresh_token is None or expires_at is None:
            # `token` may have just been replaced by a concurrent refresh (in which case it no
            # longer appears in the stored tokens): serve the fresh file token without caching.
            current_file_token = _get_token_from_file()
            if current_file_token is not None and current_file_token != token:
                return current_file_token
            # Not a refreshable OAuth token (or its metadata was lost): nothing to do.
            _OAUTH_REFRESH_CACHE = {
                "file_token": token,
                "resolved_token": token,
                "recheck_at": now + _OAUTH_RECHECK_INTERVAL,
            }
            return token

        if expires_at - _OAUTH_REFRESH_MARGIN > now:
            _OAUTH_REFRESH_CACHE = {
                "file_token": token,
                "resolved_token": token,
                "recheck_at": expires_at - _OAUTH_REFRESH_MARGIN,
            }
            return token

        try:
            # Cross-process file lock: if the server rotates refresh tokens, two processes
            # refreshing concurrently would invalidate each other's refresh token.
            with WeakFileLock(constants.HF_STORED_TOKENS_PATH + ".lock", timeout=30):
                # Re-read under the lock: another process may have refreshed in the meantime.
                fields = _read_stored_tokens_full().get(token_name, {})
                if fields.get("hf_token") != token:
                    # Another process already refreshed this token: adopt its result.
                    new_token = fields.get("hf_token") or token
                    new_expires_at = _parse_expires_at(fields)
                else:
                    response = refresh_access_token(refresh_token)
                    new_token = response["access_token"]
                    new_expires_at = int(now) + int(response["expires_in"]) if "expires_in" in response else None
                    _save_token(
                        token=new_token,
                        token_name=token_name,
                        # The server may rotate the refresh token; keep the old one if it doesn't.
                        refresh_token=response.get("refresh_token") or refresh_token,
                        expires_at=new_expires_at,
                    )
                    # Update the active token file, unless another process switched to a different token meanwhile.
                    if _get_token_from_file() == token:
                        _write_secret(Path(constants.HF_TOKEN_PATH), new_token)
                    logger.info(f"Access token `{token_name}` has been refreshed.")
        except Exception as e:
            if isinstance(e, DeviceCodeError) and e.error_code == OAuthErrorCode.INVALID_GRANT:
                # Refresh token expired or revoked: retrying is pointless, a re-login is required.
                # Warned unconditionally (the inf recheck guarantees it fires at most once): an
                # earlier transient warning must not suppress this actionable message.
                logger.warning(
                    "Your Hugging Face access token has expired and could not be refreshed "
                    f"(session expired or revoked). Run `hf auth login` to re-authenticate. ({e})"
                )
                recheck_at = float("inf")
            else:
                # Transient failure (offline, server error, ...): retry later.
                _warn_refresh_failure_once(f"Could not refresh your Hugging Face access token: {e}. Will retry later.")
                recheck_at = now + _OAUTH_RECHECK_INTERVAL
            # Return the existing token: if it's truly expired, the API will reject it with a clear error.
            _OAUTH_REFRESH_CACHE = {"file_token": token, "resolved_token": token, "recheck_at": recheck_at}
            return token

        _OAUTH_REFRESH_CACHE = {
            "file_token": new_token,
            "resolved_token": new_token,
            # The floor guards against a token lifetime shorter than the refresh margin, which
            # would otherwise put `recheck_at` in the past and trigger a refresh on every call.
            "recheck_at": max(
                now + _OAUTH_RECHECK_INTERVAL,
                new_expires_at - _OAUTH_REFRESH_MARGIN if new_expires_at else 0,
            ),
        }
        return new_token


def _warn_refresh_failure_once(message: str) -> None:
    global _OAUTH_REFRESH_WARNED
    if not _OAUTH_REFRESH_WARNED:
        logger.warning(message)
        _OAUTH_REFRESH_WARNED = True


def _parse_expires_at(fields: dict[str, str]) -> int | None:
    """Parse the `expires_at` field of a stored-tokens section, `None` if missing or corrupt."""
    try:
        return int(fields["expires_at"])
    except (KeyError, ValueError):
        return None


def get_stored_tokens() -> dict[str, str]:
    """
    Returns the parsed INI file containing the access tokens.
    The file is located at `HF_STORED_TOKENS_PATH`, defaulting to `~/.cache/huggingface/stored_tokens`.
    If the file does not exist, an empty dictionary is returned.

    Returns: `dict[str, str]`
        Key is the token name and value is the token.
    """
    return {token_name: fields.get("hf_token", "") for token_name, fields in _read_stored_tokens_full().items()}


def _read_stored_tokens_full() -> dict[str, dict[str, str]]:
    """Read all sections of the stored tokens INI file, with all their fields.

    Beside `hf_token`, sections for OAuth tokens also carry `refresh_token` and `expires_at`
    (unix timestamp), used by [`get_token`] to transparently refresh them.
    """
    tokens_path = Path(constants.HF_STORED_TOKENS_PATH)
    if not tokens_path.exists():
        return {}
    # interpolation=None: token values are opaque strings, a `%` must not be interpreted.
    # Read as UTF-8 (token names are arbitrary Unicode). A file that is not valid UTF-8 is treated
    # as unparseable, so that the user can log in again to overwrite it.
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(tokens_path, encoding="utf-8")
        return {token_name: dict(config.items(token_name)) for token_name in config.sections()}
    except (configparser.Error, UnicodeDecodeError) as e:
        logger.error(f"Error parsing stored tokens file: {e}")
        return {}


def _save_stored_tokens_full(stored_tokens: dict[str, dict[str, str]]) -> None:
    """Write all sections and their fields to the stored tokens INI file."""
    config = configparser.ConfigParser(interpolation=None)
    for token_name in sorted(stored_tokens.keys()):
        config.add_section(token_name)
        for key, value in stored_tokens[token_name].items():
            config.set(token_name, key, value)

    buf = io.StringIO()
    config.write(buf)
    _write_secret(Path(constants.HF_STORED_TOKENS_PATH), buf.getvalue())


def _get_token_by_name(token_name: str) -> str | None:
    """
    Get the token by name.

    Args:
        token_name (`str`):
            The name of the token to get.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.

    """
    stored_tokens = get_stored_tokens()
    if token_name not in stored_tokens:
        return None
    return _clean_token(stored_tokens[token_name])


def _save_token(
    token: str, token_name: str, *, refresh_token: str | None = None, expires_at: int | None = None
) -> None:
    """
    Save the given token.

    If the stored tokens file does not exist, it will be created.
    Args:
        token (`str`):
            The token to save.
        token_name (`str`):
            The name of the token.
        refresh_token (`str`, *optional*):
            OAuth refresh token used to renew the access token when it expires.
        expires_at (`int`, *optional*):
            Unix timestamp at which the access token expires.
    """
    stored_tokens = _read_stored_tokens_full()
    fields = {"hf_token": token}
    if refresh_token is not None:
        fields["refresh_token"] = refresh_token
    if expires_at is not None:
        fields["expires_at"] = str(expires_at)
    # Replace the whole section: re-logging in under the same name must drop stale metadata.
    stored_tokens[token_name] = fields
    _save_stored_tokens_full(stored_tokens)
    logger.info(f"The token `{token_name}` has been saved to {constants.HF_STORED_TOKENS_PATH}")


def _clean_token(token: str | None) -> str | None:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None
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
"""Detect whether the process is being invoked by an AI coding agent.

Detection is based on environment variables that AI agents set in their shell
sessions. `AI_AGENT` and `AGENT` are treated as a universal standard (any
tool can set its harness id there); the remaining checks are tool-specific and
ordered by priority (first match wins).

The list of known harnesses is maintained on the Hub and exposed at
`{ENDPOINT}/api/agent-harnesses`. We fetch it at most once a day and cache it
locally so the list can be updated without requiring a new client release.

Detection is entirely best-effort: there is no hardcoded list of harnesses. When
the registry cannot be fetched (and no cached copy is available), detection simply
reports "no agent". Any error while fetching/reading the registry is swallowed —
detection must never make a process fail.

More details: https://huggingface.co/docs/hub/agents-overview#register-your-agent-harness
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, TypedDict

from .. import constants
from . import logging


logger = logging.get_logger(__name__)

# Refresh the cached registry at most once every 24 hours.
_REGISTRY_TTL_SECONDS = 24 * 3600

# Short timeout: fetching the registry is best-effort telemetry, never block the caller for long.
_REGISTRY_FETCH_TIMEOUT = 3


class HarnessInfo(TypedDict, total=False):
    """A single harness entry. `envVars` maps an env var name to a match pattern (see `_env_vars_match`)."""

    envVars: dict[str, str]


class Registry(TypedDict):
    """The agent harness registry, as served by `{ENDPOINT}/api/agent-harnesses`."""

    standardEnvVars: list[str]
    harnesses: dict[str, HarnessInfo]


# Empty registry: detection is disabled (no agent ever detected). Used when the
# Hub is unreachable and no cached copy is available.
_EMPTY_REGISTRY: Registry = {"standardEnvVars": [], "harnesses": {}}

# In-process cache of the resolved registry. Populated lazily on first detection.
_registry: Registry | None = None


def detect_agent() -> Optional[str]:
    """Return the id of the detected AI agent harness or `None`.

    Harnesses are checked in registry order; for each one we match its env var
    pattern(s) and, failing that, the standard `AI_AGENT` / `AGENT` vars
    against the harness id. The first match wins. When a standard var is set to
    an unrecognized value, `"unknown"` is returned.
    """
    registry = _get_registry()
    standard_vars = registry.get("standardEnvVars") or []
    harnesses = registry.get("harnesses") or {}

    for harness_id, info in harnesses.items():
        env_vars = (info or {}).get("envVars")
        if env_vars and _env_vars_match(env_vars):
            return harness_id
        for var in standard_vars:
            if os.environ.get(var, "").strip() == harness_id:
                return harness_id

    # No harness matched but a standard var is set => unrecognized agent.
    lowercased_harnesses = {k.lower() for k in harnesses.keys()}
    for var in standard_vars:
        if value := os.environ.get(var, "").strip().lower():
            if value in lowercased_harnesses:
                return value
            return "unknown"

    return None


def is_agent() -> bool:
    """Return `True` if the process is being invoked by an AI coding agent."""
    return detect_agent() is not None


def _env_vars_match(env_vars: dict[str, str]) -> bool:
    """Return `True` if any `(var, pattern)` from the harness matches the environment.

    Supported patterns:
      - `"*"`: the variable is set to any non-empty value
      - `"<value>"`: the variable equals this exact value
    """
    for var, pattern in env_vars.items():
        value = os.environ.get(var)
        if not value:
            continue
        if pattern == "*":
            return True
        if value == pattern:
            return True
    return False


def _get_registry() -> Registry:
    """Return the harness registry, loading (and caching in-process) on first call.

    Best-effort: any unexpected error degrades to an empty registry so detection
    never raises.
    """
    global _registry
    if _registry is None:
        try:
            _registry = _load_registry()
        except Exception:
            logger.debug("Could not resolve agent harnesses registry.", exc_info=True)
            _registry = _EMPTY_REGISTRY
    return _registry


def _load_registry() -> Registry:
    """Resolve the registry from the local cache or the Hub.

    No hardcoded list: if the Hub is unreachable and no cached copy exists, an
    empty registry is returned (i.e. no agent is detected).
    """
    path = constants.AGENT_HARNESSES_PATH

    # 1. Use the cached file if it was refreshed within the last 24 hours.
    if cached := _read_cached_registry(path, max_age=_REGISTRY_TTL_SECONDS):
        return cached

    # 2. Otherwise refresh it from the Hub and persist it for next time.
    if (fetched := _fetch_registry()) is not None:
        _write_cached_registry(path, fetched)
        return fetched

    # 3. Fetch failed: reuse a stale cache if available, else give up (no detection).
    if stale := _read_cached_registry(path, max_age=None):
        return stale
    return _EMPTY_REGISTRY


def _read_cached_registry(path: str, max_age: int | None) -> Registry | None:
    """Return the cached registry, or `None` if missing/stale/unreadable."""
    try:
        if not os.path.exists(path):
            return None
        if max_age is not None and (time.time() - os.path.getmtime(path)) >= max_age:
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.debug("Could not read cached agent harnesses registry.", exc_info=True)
        return None


def _write_cached_registry(path: str, registry: Registry) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f)
    except Exception:
        logger.debug("Could not cache agent harnesses registry.", exc_info=True)


def _fetch_registry() -> Registry | None:
    """Fetch the registry from the Hub. Returns `None` when offline or on any error."""
    if constants.HF_HUB_OFFLINE:
        return None
    try:
        from ._http import get_session

        response = get_session().get(
            f"{constants.ENDPOINT}/api/agent-harnesses",
            timeout=_REGISTRY_FETCH_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.debug("Could not fetch agent harnesses registry from the Hub.", exc_info=True)
        return None
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to handle paths in Huggingface Hub."""

import os
from collections.abc import Callable, Generator, Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")

# Always ignore `.git` and `.cache/huggingface` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/*",
    "*/.cache/huggingface",
    "**/.cache/huggingface/**",
]
# Forbidden to commit these folders
FORBIDDEN_FOLDERS = [".git", ".cache"]


def as_extended_path(path: str | Path, max_length: int = 255) -> str:
    r"""Return `path` in its Windows extended-length form if it is too long, unchanged otherwise.

    Some Windows versions do not allow for paths longer than 255 characters (247 for directories, i.e. MAX_PATH minus
    room for an 8.3 file name). In this case, we must specify them as extended paths by using the `\\?\` prefix, which
    only works on absolute paths. Network shares take the `\\?\UNC\server\share\...` form: prefixing them verbatim
    would produce an invalid `\\?\\\server\...` path.

    Args:
        path (`str` or `Path`):
            The path to convert. Returned unchanged on non-Windows platforms, if it is short enough, or if it is
            already an extended path.
        max_length (`int`, *optional*):
            Length above which the path must be converted. Defaults to 255, the limit for files. Pass 247 for paths
            whose parent directories are created by the caller.
    """
    path = str(path)
    if os.name != "nt":
        return path
    absolute_path = os.path.abspath(path)
    if len(absolute_path) <= max_length or absolute_path.startswith("\\\\?\\"):
        return path
    if absolute_path.startswith("\\\\"):  # UNC share: `\\server\share\...` => `\\?\UNC\server\share\...`
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


def filter_repo_objects(
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    """Filter repo objects based on an allowlist and a denylist.

    Input must be a list of paths (`str` or `Path`) or a list of arbitrary objects.
    In the later case, `key` must be provided and specifies a function of one argument
    that is used to extract a path from each element in iterable.

    Patterns are Standard Wildcards (globbing patterns), NOT regular expressions. The pattern matching is based on
    Python's `fnmatch.fnmatchcase`, so it is case-sensitive on every platform. Backslashes are treated as path separators
    and normalized to forward slashes in both patterns and paths before matching, so patterns built with `os.path.join`
    on Windows work as expected.

    Note that it matches `*` across path boundaries, unlike traditional Unix shell globbing. For example, `"data/*.json"`
    will match both `data/file.json` and `data/subdir/file.json`.

    See https://docs.python.org/3/library/fnmatch.html for more details.

    Args:
        items (`Iterable`):
            List of items to filter.
        allow_patterns (`str` or `list[str]`, *optional*):
            Patterns constituting the allowlist. If provided, item paths must match at
            least one pattern from the allowlist.
        ignore_patterns (`str` or `list[str]`, *optional*):
            Patterns constituting the denylist. If provided, item paths must not match
            any patterns from the denylist.
        key (`Callable[[T], str]`, *optional*):
            Single-argument function to extract a path from each item. If not provided,
            the `items` must already be `str` or `Path`.

    Returns:
        Filtered list of objects, as a generator.

    Raises:
        :class:`ValueError`:
            If `key` is not provided and items are not `str` or `Path`.

    Example usage with paths:
    ```python
    >>> # Filter only PDFs that are not hidden.
    >>> list(filter_repo_objects(
    ...     ["aaa.pdf", "bbb.jpg", ".ccc.pdf", ".ddd.png"],
    ...     allow_patterns=["*.pdf"],
    ...     ignore_patterns=[".*"],
    ... ))
    ["aaa.pdf"]
    ```

    Example usage with objects:
    ```python
    >>> list(filter_repo_objects(
    ... [
    ...     CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/bbb.jpg", path_in_repo="bbb.jpg")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ccc.pdf", path_in_repo=".ccc.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ddd.png", path_in_repo=".ddd.png")
    ... ],
    ... allow_patterns=["*.pdf"],
    ... ignore_patterns=[".*"],
    ... key=lambda x: x.path_in_repo
    ... ))
    [CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")]
    ```
    """
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]

    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]

    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(_normalize_separators(p)) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(_normalize_separators(p)) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string.")

        key = _identity  # Items must be `str` or `Path`, otherwise raise ValueError

    for item in items:
        path = _normalize_separators(key(item))

        # Skip if there's an allowlist and path doesn't match any
        if allow_patterns is not None and not any(fnmatchcase(path, r) for r in allow_patterns):
            continue

        # Skip if there's a denylist and path matches any
        if ignore_patterns is not None and any(fnmatchcase(path, r) for r in ignore_patterns):
            continue

        yield item


def _normalize_separators(value: str | Path) -> str:
    # Repo paths always use `/` and `\` is not an fnmatch escape character, so treating backslashes as path separators
    # (e.g. patterns built with `os.path.join` on Windows) is safe.
    # `value` can be a `Path` if a custom `key` returns one; coerce to `str` first.
    return str(value).replace("\\", "/")


def _add_wildcard_to_directories(pattern: str) -> str:
    if pattern.endswith("/"):
        return pattern + "*"
    return pattern
<|endoftext|># Copyright 2020 Optuna, Hugging Face
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
"""Logging utilities."""

import logging
import os
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)

from .. import constants


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _get_default_logging_level():
    """
    If `HF_HUB_VERBOSITY` env var is set to one of the valid choices return that as the new default level. If it is not
    - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("HF_HUB_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option HF_HUB_VERBOSITY={env_level_str}, has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _configure_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.addHandler(logging.StreamHandler())
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: str | None = None) -> logging.Logger:
    """
        Returns a logger with the specified name. This function is not supposed
        to be directly accessed by library users.

        Args:
            name (`str`, *optional*):
                The name of the logger to get, usually the filename

        Example:

    ```python
    >>> from huggingface_hub import get_logger

    >>> logger = get_logger(__file__)
    >>> logger.set_verbosity_info()
    ```
    """

    if name is None:
        name = _get_library_name()

    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the HuggingFace Hub's root logger.

    Returns:
        Logging level, e.g., `huggingface_hub.logging.DEBUG` and
        `huggingface_hub.logging.INFO`.

    > [!TIP]
    > HuggingFace Hub has following logging levels:
    >
    > - `huggingface_hub.logging.CRITICAL`, `huggingface_hub.logging.FATAL`
    > - `huggingface_hub.logging.ERROR`
    > - `huggingface_hub.logging.WARNING`, `huggingface_hub.logging.WARN`
    > - `huggingface_hub.logging.INFO`
    > - `huggingface_hub.logging.DEBUG`
    """
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Sets the level for the HuggingFace Hub's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., `huggingface_hub.logging.DEBUG` and
            `huggingface_hub.logging.INFO`.
    """
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """
    Sets the verbosity to `logging.INFO`.
    """
    return set_verbosity(INFO)


def set_verbosity_warning():
    """
    Sets the verbosity to `logging.WARNING`.
    """
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """
    Sets the verbosity to `logging.DEBUG`.
    """
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """
    Sets the verbosity to `logging.ERROR`.
    """
    return set_verbosity(ERROR)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is
    disabled by default.
    """
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the
    HuggingFace Hub's default handler to prevent double logging if the root
    logger has been configured.
    """
    _get_library_root_logger().propagate = True


_configure_library_root_logger()

if constants.HF_DEBUG:
    # If `HF_DEBUG` environment variable is set, set the verbosity of `huggingface_hub` logger to `DEBUG`.
    set_verbosity_debug()
<|endoftext|>import warnings
from collections.abc import Iterable
from functools import wraps
from inspect import Parameter, signature


def _deprecate_positional_args(*, version: str):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Args:
        version (`str`):
            The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)
            # extra_args > 0
            args_msg = [
                f"{name}='{arg}'" if isinstance(arg, str) else f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                f"Deprecated positional argument(s) used in '{f.__name__}': pass"
                f" {args_msg} as keyword args. From version {version} passing these"
                " as positional arguments will result in an error,",
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    return _inner_deprecate_positional_args


def _deprecate_arguments(
    *,
    version: str,
    deprecated_args: Iterable[str],
    custom_message: str | None = None,
):
    """Decorator to issue warnings when using deprecated arguments.

    TODO: could be useful to be able to set a custom error message.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        deprecated_args (`list[str]`):
            List of the arguments to be deprecated.
        custom_message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)

        @wraps(f)
        def inner_f(*args, **kwargs):
            # Check for used deprecated arguments
            used_deprecated_args = []
            for _, parameter in zip(args, sig.parameters.values()):
                if parameter.name in deprecated_args:
                    used_deprecated_args.append(parameter.name)
            for kwarg_name, kwarg_value in kwargs.items():
                if (
                    # If argument is deprecated but still used
                    kwarg_name in deprecated_args
                    # And then the value is not the default value
                    and kwarg_value != sig.parameters[kwarg_name].default
                ):
                    used_deprecated_args.append(kwarg_name)

            # Warn and proceed
            if len(used_deprecated_args) > 0:
                message = (
                    f"Deprecated argument(s) used in '{f.__name__}':"
                    f" {', '.join(used_deprecated_args)}. Will not be supported from"
                    f" version '{version}'."
                )
                if custom_message is not None:
                    message += "\n\n" + custom_message
                warnings.warn(message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_positional_args


def _deprecate_method(*, version: str, message: str | None = None):
    """Decorator to issue warnings when using a deprecated method.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_method(f):
        name = f.__name__
        if name == "__init__":
            name = f.__qualname__.split(".")[0]  # class name instead of method name

        @wraps(f)
        def inner_f(*args, **kwargs):
            warning_message = (
                f"'{name}' (from '{f.__module__}') is deprecated and will be removed from version '{version}'."
            )
            if message is not None:
                warning_message += " " + message
            warnings.warn(warning_message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_method
<|endoftext|>"""Utilities to efficiently compute the SHA 256 hash of a bunch of bytes."""

from typing import BinaryIO

from .insecure_hashlib import sha1, sha256


def sha_fileobj(fileobj: BinaryIO, chunk_size: int | None = None) -> bytes:
    """
    Computes the sha256 hash of the given file object, by chunks of size `chunk_size`.

    Args:
        fileobj (file-like object):
            The File object to compute sha256 for, typically obtained with `open(path, "rb")`
        chunk_size (`int`, *optional*):
            The number of bytes to read from `fileobj` at once, defaults to 1MB.

    Returns:
        `bytes`: `fileobj`'s sha256 hash as bytes
    """
    chunk_size = chunk_size if chunk_size is not None else 1024 * 1024

    sha = sha256()
    while True:
        chunk = fileobj.read(chunk_size)
        sha.update(chunk)
        if not chunk:
            break
    return sha.digest()


def git_hash(data: bytes) -> str:
    """
    Computes the git-sha1 hash of the given bytes, using the same algorithm as git.

    This is equivalent to running `git hash-object`. See https://git-scm.com/docs/git-hash-object
    for more details.

    Note: this method is valid for regular files. For LFS files, the proper git hash is supposed to be computed on the
          pointer file content, not the actual file content. However, for simplicity, we directly compare the sha256 of
          the LFS file content when we want to compare LFS files.

    Args:
        data (`bytes`):
            The data to compute the git-hash for.

    Returns:
        `str`: the git-hash of `data` as an hexadecimal string.

    Example:
    ```python
    >>> from huggingface_hub.utils.sha import git_hash
    >>> git_hash(b"Hello, World!")
    'b45ef6fec89518d314f546fd6c3025367b721684'
    ```
    """
    # Taken from https://gist.github.com/msabramo/763200
    # Note: no need to optimize by reading the file in chunks as we're not supposed to hash huge files (5MB maximum).
    sha = sha1()
    sha.update(b"blob ")
    sha.update(str(len(data)).encode())
    sha.update(b"\0")
    sha.update(data)
    return sha.hexdigest()
<|endoftext|># Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License
"""Contains utilities to easily handle subprocesses in `huggingface_hub`."""

import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import IO

from .logging import get_logger


logger = get_logger(__name__)


@contextmanager
def capture_output() -> Generator[StringIO, None, None]:
    """Capture output that is printed to terminal.

    Taken from https://stackoverflow.com/a/34738440

    Example:
    ```py
    >>> with capture_output() as output:
    ...     print("hello world")
    >>> assert output.getvalue() == "hello world\n"
    ```
    """
    output = StringIO()
    previous_output = sys.stdout
    sys.stdout = output
    try:
        yield output
    finally:
        sys.stdout = previous_output


def run_subprocess(
    command: str | list[str],
    folder: str | Path | None = None,
    check=True,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Method to run subprocesses. Calling this will capture the `stderr` and `stdout`,
    please call `subprocess.run` manually in case you would like for them not to
    be captured.

    Args:
        command (`str` or `list[str]`):
            The command to execute as a string or list of strings.
        folder (`str`, *optional*):
            The folder in which to run the command. Defaults to current working
            directory (from `os.getcwd()`).
        check (`bool`, *optional*, defaults to `True`):
            Setting `check` to `True` will raise a `subprocess.CalledProcessError`
            when the subprocess has a non-zero exit code.
        kwargs (`dict[str]`):
            Keyword arguments to be passed to the `subprocess.run` underlying command.

    Returns:
        `subprocess.CompletedProcess`: The completed process.
    """
    if isinstance(command, str):
        command = command.split()

    if isinstance(folder, Path):
        folder = str(folder)

    return subprocess.run(
        command,
        capture_output=True,
        check=check,
        encoding="utf-8",
        errors="replace",  # if not utf-8, replace char by �
        cwd=folder or os.getcwd(),
        **kwargs,
    )


@contextmanager
def run_interactive_subprocess(
    command: str | list[str],
    folder: str | Path | None = None,
    **kwargs,
) -> Generator[tuple[IO[str], IO[str]], None, None]:
    """Run a subprocess in an interactive mode in a context manager.

    Args:
        command (`str` or `list[str]`):
            The command to execute as a string or list of strings.
        folder (`str`, *optional*):
            The folder in which to run the command. Defaults to current working
            directory (from `os.getcwd()`).
        kwargs (`dict[str]`):
            Keyword arguments to be passed to the `subprocess.run` underlying command.

    Returns:
        `tuple[IO[str], IO[str]]`: A tuple with `stdin` and `stdout` to interact
        with the process (input and output are utf-8 encoded).

    Example:
    ```python
    with _interactive_subprocess("git credential-store get") as (stdin, stdout):
        # Write to stdin
        stdin.write("url=hf.co\nusername=obama\n".encode("utf-8"))
        stdin.flush()

        # Read from stdout
        output = stdout.read().decode("utf-8")
    ```
    """
    if isinstance(command, str):
        command = command.split()

    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",  # if not utf-8, replace char by �
        cwd=folder or os.getcwd(),
        **kwargs,
    ) as process:
        assert process.stdin is not None, "subprocess is opened as subprocess.PIPE"
        assert process.stdout is not None, "subprocess is opened as subprocess.PIPE"
        yield process.stdin, process.stdout
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to handle headers to send in calls to Huggingface Hub."""

from huggingface_hub.errors import LocalTokenNotFoundError

from .. import constants
from ._auth import get_token
from ._detect_agent import detect_agent
from ._runtime import (
    get_hf_hub_version,
    get_python_version,
    get_torch_version,
    is_torch_available,
)
from ._validators import validate_hf_hub_args


@validate_hf_hub_args
def build_hf_headers(
    *,
    token: bool | str | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build headers dictionary to send in a HF Hub call.

    By default, authorization token is always provided either from argument (explicit
    use) or retrieved from the cache (implicit use). To explicitly avoid sending the
    token to the Hub, set `token=False` or set the `HF_HUB_DISABLE_IMPLICIT_TOKEN`
    environment variable.

    In case of an API call that requires write access, an error is thrown if token is
    `None` or token is an organization token (starting with `"api_org***"`).

    In addition to the auth header, a user-agent is added to provide information about
    the installed packages (versions of python, huggingface_hub, torch).

    Args:
        token (`str`, `bool`, *optional*):
            The token to be sent in authorization header for the Hub call:
                - if a string, it is used as the Hugging Face token
                - if `True`, the token is read from the machine (cache or env variable)
                - if `False`, authorization header is not set
                - if `None`, the token is read from the machine only except if
                  `HF_HUB_DISABLE_IMPLICIT_TOKEN` env variable is set.
        library_name (`str`, *optional*):
            The name of the library that is making the HTTP request. Will be added to
            the user-agent header.
        library_version (`str`, *optional*):
            The version of the library that is making the HTTP request. Will be added
            to the user-agent header.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string. It will
            be completed with information about the installed packages.
        headers (`dict`, *optional*):
            Additional headers to include in the request. Those headers take precedence
            over the ones generated by this function.

    Returns:
        A `dict` of headers to pass in your API call.

    Example:
    ```py
        >>> build_hf_headers(token="hf_***") # explicit token
        {"authorization": "Bearer hf_***", "user-agent": ""}

        >>> build_hf_headers(token=True) # explicitly use cached token
        {"authorization": "Bearer hf_***",...}

        >>> build_hf_headers(token=False) # explicitly don't use cached token
        {"user-agent": ...}

        >>> build_hf_headers() # implicit use of the cached token
        {"authorization": "Bearer hf_***",...}

        # HF_HUB_DISABLE_IMPLICIT_TOKEN=True # to set as env variable
        >>> build_hf_headers() # token is not sent
        {"user-agent": ...}

        >>> build_hf_headers(library_name="transformers", library_version="1.2.3")
        {"authorization": ..., "user-agent": "transformers/1.2.3; hf_hub/0.10.2; python/3.10.4; tensorflow/1.55"}
    ```

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If organization token is passed and "write" access is required.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If "write" access is required but token is not passed and not saved locally.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `token=True` but token is not saved locally.
    """
    # Get auth token to send
    token_to_send = get_token_to_send(token)

    # Combine headers
    hf_headers = {
        "user-agent": _http_user_agent(
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )
    }
    if token_to_send is not None:
        hf_headers["authorization"] = f"Bearer {token_to_send}"
    if headers is not None:
        hf_headers.update(headers)
    return hf_headers


def get_token_to_send(token: bool | str | None) -> str | None:
    """Select the token to send from either `token` or the cache."""
    # Case token is explicitly provided
    if isinstance(token, str):
        return token

    # Case token is explicitly forbidden
    if token is False:
        return None

    # Case token is explicitly required
    if token is True:
        cached_token = get_token()
        if cached_token is None:
            raise LocalTokenNotFoundError(
                "Token is required (`token=True`), but no token found. You"
                " need to provide a token or be logged in to Hugging Face with"
                " `hf auth login` or `huggingface_hub.login`. See"
                " https://huggingface.co/settings/tokens."
            )
        return cached_token

    # Case implicit use of the token is forbidden by env variable. Checked before resolving the
    # cached token: `get_token()` may refresh an OAuth token (network call + file writes), which
    # must not happen when the resolved token wouldn't be used anyway.
    if constants.HF_HUB_DISABLE_IMPLICIT_TOKEN:
        return None

    # Otherwise: we use the cached token as the user has not explicitly forbidden it
    return get_token()


def _http_user_agent(
    *,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
) -> str:
    """Format a user-agent string containing information about the installed packages.

    Args:
        library_name (`str`, *optional*):
            The name of the library that is making the HTTP request.
        library_version (`str`, *optional*):
            The version of the library that is making the HTTP request.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string.

    Returns:
        The formatted user-agent string.
    """
    if library_name is not None:
        ua = f"{library_name}/{library_version}"
    else:
        ua = "unknown/None"
    ua += f"; hf_hub/{get_hf_hub_version()}"
    ua += f"; python/{get_python_version()}"

    if not constants.HF_HUB_DISABLE_TELEMETRY:
        if is_torch_available():
            ua += f"; torch/{get_torch_version()}"

        agent = detect_agent()
        if agent:
            ua += f"; agent/{agent}"

    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent

    # Retrieve user-agent origin headers from environment variable
    origin = constants.HF_HUB_USER_AGENT_ORIGIN
    if origin is not None:
        ua += "; origin/" + origin

    return _deduplicate_user_agent(ua)


def _deduplicate_user_agent(user_agent: str) -> str:
    """Deduplicate redundant information in the generated user-agent."""
    # Split around ";" > Strip whitespaces > Store as dict keys (ensure unicity) > format back as string
    # Order is implicitly preserved by dictionary structure (see https://stackoverflow.com/a/53657523).
    return "; ".join({key.strip(): None for key in user_agent.split(";")}.keys())
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to manage Git credentials."""

import re
import subprocess

from ..constants import ENDPOINT
from ._subprocess import run_interactive_subprocess, run_subprocess


GIT_CREDENTIAL_REGEX = re.compile(
    r"""
        ^\s* # start of line
        credential\.helper # credential.helper value
        \s*=\s* # separator
        ([\w\-\/]+) # the helper name or absolute path (group 1)
        (\s|$) # whitespace or end of line
    """,
    flags=re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)


def list_credential_helpers(folder: str | None = None) -> list[str]:
    """Return the list of git credential helpers configured.

    See https://git-scm.com/docs/gitcredentials.

    Credentials are saved in all configured helpers (store, cache, macOS keychain,...).
    Calls "`git credential approve`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    try:
        output = run_subprocess("git config --list", folder=folder).stdout
        parsed = _parse_credential_output(output)
        return parsed
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr)


def set_git_credential(token: str, username: str = "hf_user", folder: str | None = None) -> None:
    """Save a username/token pair in git credential for HF Hub registry.

    Credentials are saved in all configured helpers (store, cache, macOS keychain,...).
    Calls "`git credential approve`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        token (`str`, defaults to `"hf_user"`):
            A git password. In practice, the User Access Token for the Hub.
            See https://huggingface.co/settings/tokens.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess("git credential approve", folder=folder) as (
        stdin,
        _,
    ):
        stdin.write(f"url={ENDPOINT}\nusername={username.lower()}\npassword={token}\n\n")
        stdin.flush()


def unset_git_credential(username: str = "hf_user", folder: str | None = None) -> None:
    """Erase credentials from git credential for HF Hub registry.

    Credentials are erased from the configured helpers (store, cache, macOS
    keychain,...), if any. If `username` is not provided, any credential configured for
    HF Hub endpoint is erased.
    Calls "`git credential erase`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess("git credential reject", folder=folder) as (
        stdin,
        _,
    ):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()


def _parse_credential_output(output: str) -> list[str]:
    """Parse the output of `git credential fill` to extract the password.

    Args:
        output (`str`):
            The output of `git credential fill`.
    """
    # NOTE: If user has set a helper for a custom URL, it will not be caught here.
    #       Example: `credential.https://huggingface.co.helper=store`
    #       See: https://github.com/huggingface/huggingface_hub/pull/1138#discussion_r1013324508
    return sorted(  # Sort for nice printing
        {  # Might have some duplicates
            match[0] for match in GIT_CREDENTIAL_REGEX.findall(output)
        }
    )
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains a utility to iterate by chunks over an iterator."""

import itertools
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterable[Iterable[T]]:
    """Iterates over an iterator chunk by chunk.

    Taken from https://stackoverflow.com/a/8998040.
    See also https://github.com/huggingface/huggingface_hub/pull/920#discussion_r938793088.

    Args:
        iterable (`Iterable`):
            The iterable on which we want to iterate.
        chunk_size (`int`):
            Size of the chunks. Must be a strictly positive integer (e.g. >0).

    Example:

    ```python
    >>> from huggingface_hub.utils import chunk_iterable

    >>> for items in chunk_iterable(range(17), chunk_size=8):
    ...     print(items)
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # [8, 9, 10, 11, 12, 13, 14, 15]
    # [16] # smaller last chunk
    ```

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `chunk_size` <= 0.

    > [!WARNING]
    > The last chunk can be smaller than `chunk_size`.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("`chunk_size` must be a strictly positive integer (>0).")

    iterator = iter(iterable)
    while True:
        try:
            next_item = next(iterator)
        except StopIteration:
            return
        yield itertools.chain((next_item,), itertools.islice(iterator, chunk_size - 1))
<|endoftext|># Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

"""Public utility helpers, loaded on first use.

Most library modules import a small subset of this package. Keeping the facade lazy
prevents every such import from also loading unrelated cache, terminal, subprocess,
and optional-dependency helpers.
"""

# ruff: noqa: F401, F403

import importlib
import sys
import types
from typing import TYPE_CHECKING


_SUBMODULES = {
    "_auth",
    "_detect_agent",
    "_http",
    "_tqdm",
    "logging",
}

_SUBMOD_ATTRS = {
    "_auth": [
        "get_stored_tokens",
        "get_token",
    ],
    "_cache_assets": [
        "cached_assets_path",
    ],
    "_cache_manager": [
        "CachedFileInfo",
        "CachedIncompleteFileInfo",
        "CachedRepoInfo",
        "CachedRevisionInfo",
        "DeleteCacheStrategy",
        "HFCacheInfo",
        "_format_size",
        "scan_cache_dir",
    ],
    "_chunk_utils": [
        "chunk_iterable",
    ],
    "_datetime": [
        "parse_datetime",
    ],
    "_detect_agent": [
        "detect_agent",
        "is_agent",
    ],
    "_experimental": [
        "experimental",
    ],
    "_fixes": [
        "SoftTemporaryDirectory",
        "WeakFileLock",
        "yaml_dump",
    ],
    "_git_credential": [
        "list_credential_helpers",
        "set_git_credential",
        "unset_git_credential",
    ],
    "_headers": [
        "build_hf_headers",
        "get_token_to_send",
    ],
    "_hf_uris": [
        "HfMount",
        "HfUri",
        "is_hf_uri",
        "parse_hf_mount",
        "parse_hf_uri",
    ],
    "_http": [
        "ASYNC_CLIENT_FACTORY_T",
        "CLIENT_FACTORY_T",
        "RateLimitInfo",
        "close_session",
        "fix_hf_endpoint_in_url",
        "get_async_session",
        "get_session",
        "hf_raise_for_status",
        "http_backoff",
        "http_stream_backoff",
        "parse_ratelimit_headers",
        "set_async_client_factory",
        "set_client_factory",
    ],
    "_pagination": [
        "paginate",
    ],
    "_paths": [
        "DEFAULT_IGNORE_PATTERNS",
        "FORBIDDEN_FOLDERS",
        "filter_repo_objects",
    ],
    "_runtime": [
        "dump_environment_info",
        "get_aiohttp_version",
        "get_fastai_version",
        "get_fastapi_version",
        "get_fastcore_version",
        "get_gradio_version",
        "get_graphviz_version",
        "get_hf_hub_version",
        "get_jinja_version",
        "get_numpy_version",
        "get_pillow_version",
        "get_pydantic_version",
        "get_pydot_version",
        "get_python_version",
        "get_tensorboard_version",
        "get_tf_version",
        "get_torch_version",
        "installation_method",
        "is_aiohttp_available",
        "is_colab_enterprise",
        "is_fastai_available",
        "is_fastapi_available",
        "is_fastcore_available",
        "is_google_colab",
        "is_gradio_available",
        "is_graphviz_available",
        "is_jinja_available",
        "is_notebook",
        "is_numpy_available",
        "is_package_available",
        "is_pillow_available",
        "is_pydantic_available",
        "is_pydot_available",
        "is_safetensors_available",
        "is_tensorboard_available",
        "is_tf_available",
        "is_torch_available",
    ],
    "_safetensors": [
        "SafetensorsFileMetadata",
        "SafetensorsRepoMetadata",
        "TensorInfo",
    ],
    "_subprocess": [
        "capture_output",
        "run_interactive_subprocess",
        "run_subprocess",
    ],
    "_telemetry": [
        "send_telemetry",
    ],
    "_terminal": [
        "ANSI",
        "StatusLine",
        "select_choice",
        "tabulate",
    ],
    "_typing": [
        "is_jsonable",
        "is_simple_optional_type",
        "unwrap_simple_optional_type",
    ],
    "_validators": [
        "validate_hf_hub_args",
        "validate_repo_id",
    ],
    "_xet": [
        "XetFileData",
        "XetTokenType",
        "parse_xet_file_data_from_response",
    ],
    "huggingface_hub.errors": [
        "BadRequestError",
        "BucketNotFoundError",
        "CacheNotFound",
        "CorruptedCacheException",
        "DisabledRepoError",
        "EntryNotFoundError",
        "FileMetadataError",
        "GatedRepoError",
        "HFValidationError",
        "HfHubHTTPError",
        "JobNotFoundError",
        "LocalEntryNotFoundError",
        "LocalTokenNotFoundError",
        "NotASafetensorsRepoError",
        "OfflineModeIsEnabled",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "SafetensorsParsingError",
    ],
    "tqdm": [
        "are_progress_bars_disabled",
        "disable_progress_bars",
        "enable_progress_bars",
        "hf_thread_map",
        "is_tqdm_disabled",
        "silent_tqdm",
        "tqdm",
        "tqdm_stream_file",
    ],
}

__all__ = sorted(["httpx", *(attr for attrs in _SUBMOD_ATTRS.values() for attr in attrs if not attr.startswith("_"))])

_ATTR_TO_MODULE = {attr: module for module, attrs in _SUBMOD_ATTRS.items() for attr in attrs}


def __getattr__(name: str):
    if name == "httpx":  # Keep the same export for downstream libraries across major versions.
        return importlib.import_module("httpx2")
    if name in _SUBMODULES:
        module_name = "tqdm" if name == "_tqdm" else name
        return importlib.import_module(f"{__name__}.{module_name}")
    if attr_module_name := _ATTR_TO_MODULE.get(name):
        if attr_module_name.startswith("huggingface_hub."):
            module = importlib.import_module(attr_module_name)
        else:
            module = importlib.import_module(f"{__name__}.{attr_module_name}")
        return getattr(module, name)
    raise AttributeError(f"No {__name__} attribute {name}")


def __dir__() -> list[str]:
    return [*__all__, "httpx"]


class _LazyUtilsModule(types.ModuleType):
    def __getattribute__(self, name: str):
        # Import machinery stores the `tqdm` submodule on this package. The public
        # `utils.tqdm` attribute has historically referred to the class instead.
        if name == "tqdm":
            return importlib.import_module(f"{__name__}.tqdm").tqdm
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _LazyUtilsModule


if TYPE_CHECKING:  # pragma: no cover
    import httpx2 as httpx  # noqa: F401

    from huggingface_hub.errors import (
        BadRequestError,  # noqa: F401
        BucketNotFoundError,  # noqa: F401
        CacheNotFound,  # noqa: F401
        CorruptedCacheException,  # noqa: F401
        DisabledRepoError,  # noqa: F401
        EntryNotFoundError,  # noqa: F401
        FileMetadataError,  # noqa: F401
        GatedRepoError,  # noqa: F401
        HfHubHTTPError,  # noqa: F401
        HFValidationError,  # noqa: F401
        JobNotFoundError,  # noqa: F401
        LocalEntryNotFoundError,  # noqa: F401
        LocalTokenNotFoundError,  # noqa: F401
        NotASafetensorsRepoError,  # noqa: F401
        OfflineModeIsEnabled,  # noqa: F401
        RepositoryNotFoundError,  # noqa: F401
        RevisionNotFoundError,  # noqa: F401
        SafetensorsParsingError,  # noqa: F401
    )

    from . import _auth as _auth  # noqa: F401
    from . import _detect_agent as _detect_agent  # noqa: F401
    from . import _http as _http  # noqa: F401
    from . import logging as logging  # noqa: F401
    from . import tqdm as _tqdm  # noqa: F401
    from ._auth import (
        get_stored_tokens,  # noqa: F401
        get_token,  # noqa: F401
    )
    from ._cache_assets import cached_assets_path  # noqa: F401
    from ._cache_manager import (
        CachedFileInfo,  # noqa: F401
        CachedIncompleteFileInfo,  # noqa: F401
        CachedRepoInfo,  # noqa: F401
        CachedRevisionInfo,  # noqa: F401
        DeleteCacheStrategy,  # noqa: F401
        HFCacheInfo,  # noqa: F401
        _format_size,  # noqa: F401
        scan_cache_dir,  # noqa: F401
    )
    from ._chunk_utils import chunk_iterable  # noqa: F401
    from ._datetime import parse_datetime  # noqa: F401
    from ._detect_agent import (
        detect_agent,  # noqa: F401
        is_agent,  # noqa: F401
    )
    from ._experimental import experimental  # noqa: F401
    from ._fixes import (
        SoftTemporaryDirectory,  # noqa: F401
        WeakFileLock,  # noqa: F401
        yaml_dump,  # noqa: F401
    )
    from ._git_credential import (
        list_credential_helpers,  # noqa: F401
        set_git_credential,  # noqa: F401
        unset_git_credential,  # noqa: F401
    )
    from ._headers import (
        build_hf_headers,  # noqa: F401
        get_token_to_send,  # noqa: F401
    )
    from ._hf_uris import (
        HfMount,  # noqa: F401
        HfUri,  # noqa: F401
        is_hf_uri,  # noqa: F401
        parse_hf_mount,  # noqa: F401
        parse_hf_uri,  # noqa: F401
    )
    from ._http import (
        ASYNC_CLIENT_FACTORY_T,  # noqa: F401
        CLIENT_FACTORY_T,  # noqa: F401
        RateLimitInfo,  # noqa: F401
        close_session,  # noqa: F401
        fix_hf_endpoint_in_url,  # noqa: F401
        get_async_session,  # noqa: F401
        get_session,  # noqa: F401
        hf_raise_for_status,  # noqa: F401
        http_backoff,  # noqa: F401
        http_stream_backoff,  # noqa: F401
        parse_ratelimit_headers,  # noqa: F401
        set_async_client_factory,  # noqa: F401
        set_client_factory,  # noqa: F401
    )
    from ._pagination import paginate  # noqa: F401
    from ._paths import (
        DEFAULT_IGNORE_PATTERNS,  # noqa: F401
        FORBIDDEN_FOLDERS,  # noqa: F401
        filter_repo_objects,  # noqa: F401
    )
    from ._runtime import (
        dump_environment_info,  # noqa: F401
        get_aiohttp_version,  # noqa: F401
        get_fastai_version,  # noqa: F401
        get_fastapi_version,  # noqa: F401
        get_fastcore_version,  # noqa: F401
        get_gradio_version,  # noqa: F401
        get_graphviz_version,  # noqa: F401
        get_hf_hub_version,  # noqa: F401
        get_jinja_version,  # noqa: F401
        get_numpy_version,  # noqa: F401
        get_pillow_version,  # noqa: F401
        get_pydantic_version,  # noqa: F401
        get_pydot_version,  # noqa: F401
        get_python_version,  # noqa: F401
        get_tensorboard_version,  # noqa: F401
        get_tf_version,  # noqa: F401
        get_torch_version,  # noqa: F401
        installation_method,  # noqa: F401
        is_aiohttp_available,  # noqa: F401
        is_colab_enterprise,  # noqa: F401
        is_fastai_available,  # noqa: F401
        is_fastapi_available,  # noqa: F401
        is_fastcore_available,  # noqa: F401
        is_google_colab,  # noqa: F401
        is_gradio_available,  # noqa: F401
        is_graphviz_available,  # noqa: F401
        is_jinja_available,  # noqa: F401
        is_notebook,  # noqa: F401
        is_numpy_available,  # noqa: F401
        is_package_available,  # noqa: F401
        is_pillow_available,  # noqa: F401
        is_pydantic_available,  # noqa: F401
        is_pydot_available,  # noqa: F401
        is_safetensors_available,  # noqa: F401
        is_tensorboard_available,  # noqa: F401
        is_tf_available,  # noqa: F401
        is_torch_available,  # noqa: F401
    )
    from ._safetensors import (
        SafetensorsFileMetadata,  # noqa: F401
        SafetensorsRepoMetadata,  # noqa: F401
        TensorInfo,  # noqa: F401
    )
    from ._subprocess import (
        capture_output,  # noqa: F401
        run_interactive_subprocess,  # noqa: F401
        run_subprocess,  # noqa: F401
    )
    from ._telemetry import send_telemetry  # noqa: F401
    from ._terminal import (
        ANSI,  # noqa: F401
        StatusLine,  # noqa: F401
        select_choice,  # noqa: F401
        tabulate,  # noqa: F401
    )
    from ._typing import (
        is_jsonable,  # noqa: F401
        is_simple_optional_type,  # noqa: F401
        unwrap_simple_optional_type,  # noqa: F401
    )
    from ._validators import (
        validate_hf_hub_args,  # noqa: F401
        validate_repo_id,  # noqa: F401
    )
    from ._xet import (
        XetFileData,  # noqa: F401
        XetTokenType,  # noqa: F401
        parse_xet_file_data_from_response,  # noqa: F401
    )
    from .tqdm import (
        are_progress_bars_disabled,  # noqa: F401
        disable_progress_bars,  # noqa: F401
        enable_progress_bars,  # noqa: F401
        hf_thread_map,  # noqa: F401
        is_tqdm_disabled,  # noqa: F401
        silent_tqdm,  # noqa: F401
        tqdm,  # noqa: F401
        tqdm_stream_file,  # noqa: F401
    )
<|endoftext|># Copyright 2019-present, the HuggingFace Inc. team.
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
from pathlib import Path

from ..constants import HF_ASSETS_CACHE


def cached_assets_path(
    library_name: str,
    namespace: str = "default",
    subfolder: str = "default",
    *,
    assets_dir: str | Path | None = None,
) -> Path:
    """Return a folder path to cache arbitrary files.

    `huggingface_hub` provides a canonical folder path to store assets. This is the
    recommended way to integrate cache in a downstream library as it will benefit from
    the builtins tools to scan and delete the cache properly.

    The distinction is made between files cached from the Hub and assets. Files from the
    Hub are cached in a git-aware manner and entirely managed by `huggingface_hub`. See
    [related documentation](https://huggingface.co/docs/huggingface_hub/how-to-cache).
    All other files that a downstream library caches are considered to be "assets"
    (files downloaded from external sources, extracted from a .tar archive, preprocessed
    for training,...).

    Once the folder path is generated, it is guaranteed to exist and to be a directory.
    The path is based on 3 levels of depth: the library name, a namespace and a
    subfolder. Those 3 levels grants flexibility while allowing `huggingface_hub` to
    expect folders when scanning/deleting parts of the assets cache. Within a library,
    it is expected that all namespaces share the same subset of subfolder names but this
    is not a mandatory rule. The downstream library has then full control on which file
    structure to adopt within its cache. Namespace and subfolder are optional (would
    default to a `"default/"` subfolder) but library name is mandatory as we want every
    downstream library to manage its own cache.

    Expected tree:
    ```text
        assets/
        └── datasets/
        │   ├── SQuAD/
        │   │   ├── downloaded/
        │   │   ├── extracted/
        │   │   └── processed/
        │   ├── Helsinki-NLP--tatoeba_mt/
        │       ├── downloaded/
        │       ├── extracted/
        │       └── processed/
        └── transformers/
            ├── default/
            │   ├── something/
            ├── bert-base-cased/
            │   ├── default/
            │   └── training/
        hub/
        └── models--julien-c--EsperBERTo-small/
            ├── blobs/
            │   ├── (...)
            │   ├── (...)
            ├── refs/
            │   └── (...)
            └── [ 128]  snapshots/
                ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
                │   ├── (...)
                └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                    └── (...)
    ```


    Args:
        library_name (`str`):
            Name of the library that will manage the cache folder. Example: `"dataset"`.
        namespace (`str`, *optional*, defaults to "default"):
            Namespace to which the data belongs. Example: `"SQuAD"`.
        subfolder (`str`, *optional*, defaults to "default"):
            Subfolder in which the data will be stored. Example: `extracted`.
        assets_dir (`str`, `Path`, *optional*):
            Path to the folder where assets are cached. This must not be the same folder
            where Hub files are cached. Defaults to `HF_HOME / "assets"` if not provided.
            Can also be set with `HF_ASSETS_CACHE` environment variable.

    Returns:
        Path to the cache folder (`Path`).

    Example:
    ```py
    >>> from huggingface_hub import cached_assets_path

    >>> cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
    PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/SQuAD/download')

    >>> cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="extracted")
    PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/SQuAD/extracted')

    >>> cached_assets_path(library_name="datasets", namespace="Helsinki-NLP/tatoeba_mt")
    PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/Helsinki-NLP--tatoeba_mt/default')

    >>> cached_assets_path(library_name="datasets", assets_dir="/tmp/tmp123456")
    PosixPath('/tmp/tmp123456/datasets/default/default')
    ```
    """
    # Resolve assets_dir
    if assets_dir is None:
        assets_dir = HF_ASSETS_CACHE
    assets_dir = Path(assets_dir).expanduser().resolve()

    # Avoid names that could create path issues
    for part in (" ", "/", "\\"):
        library_name = library_name.replace(part, "--")
        namespace = namespace.replace(part, "--")
        subfolder = subfolder.replace(part, "--")

    # Path to subfolder is created
    path = assets_dir / library_name / namespace / subfolder
    try:
        path.mkdir(exist_ok=True, parents=True)
    except (FileExistsError, NotADirectoryError):
        raise ValueError(f"Corrupted assets folder: cannot create directory because of an existing file ({path}).")

    # Return
    return path
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
"""Cache-wide shared blob store, deduplicating Xet files across cached repos.

Store entries at `<cache_dir>/blobs/<2-hex-prefix>/<xet_hash>` are regular files.
Per-repo `blobs/<etag>` entries are relative symlinks to them. A companion
`<xet_hash>.refs` manifest lists those symlinks so cache deletion only needs to inspect
the blobs it touches instead of scanning the entire store.

The manifest is a hint, not trusted metadata: every entry is validated against the
filesystem before use, and any missing or unreadable manifest makes garbage collection
leak the shared blob rather than risk deleting referenced content.
"""

import os
import re
import secrets
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from filelock import FileLock, SoftFileLock

from .. import constants
from . import logging


logger = logging.get_logger(__name__)

_XET_HASH_REGEX = re.compile(r"[0-9a-f]{64}")
_REPO_DIR_REGEX = re.compile(rf"(?:{'|'.join(sorted(constants.REPO_TYPES_MAPPING))})--.+")

SHARED_BLOBS_DIR_NAME = "blobs"
SHARED_BLOBS_MARKER_NAME = ".huggingface-shared-blobs"
SHARED_BLOBS_LAYOUT_VERSION = "1"
_MANIFEST_SUFFIX = ".refs"
_LOCK_SUFFIX = ".lock"
_SOFT_LOCK_TIMEOUT = 10
_MARKER_TMP_REGEX = re.compile(rf"{re.escape(SHARED_BLOBS_MARKER_NAME)}\.[0-9a-f]{{8}}\.tmp")


def shared_blobs_dir(cache_dir: str | Path) -> Path:
    """Return the path of the shared blob store inside a cache directory."""
    return Path(cache_dir) / SHARED_BLOBS_DIR_NAME


def shared_blob_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the store path for a given Xet file hash.

    Raises `ValueError` if `xet_hash` is not a valid Xet hash.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        raise ValueError(f"Invalid Xet file hash: '{xet_hash}'.")
    return shared_blobs_dir(cache_dir) / xet_hash[:2] / xet_hash


def shared_blob_manifest_path(cache_dir: str | Path, xet_hash: str) -> Path:
    """Return the reference manifest path for a given Xet file hash."""
    return shared_blob_path(cache_dir, xet_hash).with_name(f"{xet_hash}{_MANIFEST_SUFFIX}")


def _is_regular_file(path: Path) -> bool:
    # checks if path is a regular file without following symlinks
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _is_directory(path: Path) -> bool:
    # checks if path is a regular directory without following symlinks
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_usable_store_entry(store_path: Path, expected_size: int | None) -> bool:
    """Return whether `store_path` is a regular, readable payload of the expected size."""
    if expected_size is None:
        # Size unknown (HEAD without a content length): the entry cannot be validated.
        return False
    try:
        store_stat = store_path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(store_stat.st_mode):
        return False
    if store_stat.st_size != expected_size:
        logger.warning(
            f"Shared blob '{store_path}' has an unexpected size ({store_stat.st_size} instead of {expected_size}). "
            "Not using it."
        )
        return False
    if not os.access(store_path, os.R_OK):
        logger.warning(f"Shared blob '{store_path}' is not readable. Not using it.")
        return False
    return True


def is_shared_blobs_dir(path: str | Path) -> bool:
    """Return whether `path` is an owned, supported shared blob store."""
    store_dir = Path(path)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    if not _is_directory(store_dir) or not _is_regular_file(marker_path):
        return False
    try:
        return marker_path.read_text() == f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    except OSError:
        return False


def _cleanup_abandoned_marker_temps(store_dir: Path) -> bool:
    """Remove leftover marker temporaries.

    Returns whether the directory is empty afterwards, i.e. safe to mark. Any foreign content returns False.
    """
    expected_content = f"{SHARED_BLOBS_LAYOUT_VERSION}\n"
    entries = list(store_dir.iterdir())
    for entry in entries:
        if _MARKER_TMP_REGEX.fullmatch(entry.name) is None or not _is_regular_file(entry):
            return False
        try:
            if not expected_content.startswith(entry.read_text()):
                return False
        except (OSError, UnicodeError):
            return False
    for entry in entries:
        entry.unlink(missing_ok=True)
    return not any(store_dir.iterdir())


def _ensure_shared_blobs_dir(cache_dir: str | Path) -> bool:
    """Create and mark the shared store, refusing to adopt an unmarked directory."""
    store_dir = shared_blobs_dir(cache_dir)
    marker_path = store_dir / SHARED_BLOBS_MARKER_NAME
    try:
        store_dir.mkdir(exist_ok=True)
        if is_shared_blobs_dir(store_dir):
            return True
        if not _is_directory(store_dir) or not _cleanup_abandoned_marker_temps(store_dir):
            logger.debug(f"Refusing to use unmarked shared blob directory '{store_dir}'.")
            return False
        _repair_shared_directory_mode(store_dir, cache_dir)

        tmp_marker = marker_path.with_name(f"{marker_path.name}.{secrets.token_hex(4)}.tmp")
        try:
            tmp_marker.write_text(f"{SHARED_BLOBS_LAYOUT_VERSION}\n")
            tmp_marker.chmod(_shared_blob_mode(cache_dir))
            os.replace(tmp_marker, marker_path)
        finally:
            tmp_marker.unlink(missing_ok=True)
        return is_shared_blobs_dir(store_dir)
    except OSError as e:
        logger.debug(f"Could not initialize shared blob directory '{store_dir}': {e}")
        return False


def _ensure_prefix_dir(cache_dir: str | Path, xet_hash: str) -> Path | None:
    if not _ensure_shared_blobs_dir(cache_dir):
        return None
    prefix_dir = shared_blob_path(cache_dir, xet_hash).parent
    try:
        prefix_dir.mkdir(exist_ok=True)
    except OSError as e:
        logger.debug(f"Could not create shared blob prefix directory '{prefix_dir}': {e}")
        return None
    if not _is_directory(prefix_dir):
        logger.debug(f"Refusing to use non-directory shared blob prefix '{prefix_dir}'.")
        return None
    try:
        _repair_shared_directory_mode(prefix_dir, cache_dir)
    except OSError as e:
        logger.debug(f"Could not set shared blob prefix permissions on '{prefix_dir}': {e}")
        return None
    return prefix_dir


def _shared_blob_mode(cache_dir: str | Path) -> int:
    """Return a read-only mode accessible to users who can traverse the cache root."""
    if os.name == "nt":
        # Windows uses ACLs; 0444 would set the read-only attribute and block replacement and GC.
        return 0o666
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o400
    return 0o400 | (0o040 if cache_mode & stat.S_IXGRP else 0) | (0o004 if cache_mode & stat.S_IXOTH else 0)


def _shared_directory_mode(cache_dir: str | Path) -> int:
    """Mirror cache-root access and inheritance bits on newly created store directories."""
    try:
        cache_mode = stat.S_IMODE(Path(cache_dir).stat().st_mode)
    except OSError:
        return 0o700
    return cache_mode & (0o777 | stat.S_ISGID | stat.S_ISVTX)


def _repair_shared_directory_mode(path: Path, cache_dir: str | Path) -> None:
    expected_mode = _shared_directory_mode(cache_dir)
    if stat.S_IMODE(path.lstat().st_mode) != expected_mode:
        path.chmod(expected_mode)


def _prepare_shared_blob_permissions(blob_path: Path, prefix_dir: Path, cache_dir: str | Path) -> None:
    """Make a payload immutable and readable according to the shared cache policy."""
    blob_mode = _shared_blob_mode(cache_dir)
    os.chmod(blob_path, blob_mode)
    if os.name == "nt" or not hasattr(os, "chown"):
        return
    target_gid = prefix_dir.stat().st_gid
    if blob_path.stat().st_gid == target_gid:
        return
    try:
        os.chown(blob_path, -1, target_gid)
    except OSError:
        # Without other-read, a wrong group makes the entry unreadable to other users: fall back to repo-local.
        if blob_mode & stat.S_IRGRP and not blob_mode & stat.S_IROTH:
            raise


def _path_for_comparison(path: str | Path) -> Path:
    """Return an absolute path without the Windows extended-length prefix."""
    path_str = os.fspath(path)
    if path_str[:8].lower() == "\\\\?\\unc\\":
        path_str = f"\\\\{path_str[8:]}"
    elif path_str.startswith("\\\\?\\"):
        path_str = path_str[4:]
    return Path(os.path.abspath(path_str))


def _relative_blob_path(blob_path: str | Path, cache_dir: str | Path) -> str | None:
    blob_path = _path_for_comparison(blob_path)
    cache_dir = _path_for_comparison(cache_dir)
    try:
        relative_path = blob_path.relative_to(cache_dir)
    except ValueError:
        return None
    if (
        len(relative_path.parts) != 3
        or _REPO_DIR_REGEX.fullmatch(relative_path.parts[0]) is None
        or relative_path.parts[1] != "blobs"
        or not relative_path.parts[2]
    ):
        return None
    relative_str = relative_path.as_posix()
    return None if "\n" in relative_str or "\r" in relative_str else relative_str


def _shared_blob_target(
    blob_path: str | Path, cache_dir: str | Path, *, raise_on_io_error: bool = False
) -> Path | None:
    """Return the canonical store target if `blob_path` is a managed shared symlink."""
    blob_path = Path(blob_path)
    if _relative_blob_path(blob_path, cache_dir) is None:
        return None
    try:
        if not stat.S_ISLNK(blob_path.lstat().st_mode):
            return None
        link_target = Path(os.readlink(blob_path))
    except FileNotFoundError:
        return None
    except OSError:
        if raise_on_io_error:
            raise
        return None

    comparable_blob_path = _path_for_comparison(blob_path)
    target = link_target if link_target.is_absolute() else comparable_blob_path.parent / link_target
    target = _path_for_comparison(target)
    store_dir = _path_for_comparison(shared_blobs_dir(cache_dir))
    try:
        relative_target = target.relative_to(store_dir)
    except ValueError:
        return None
    if (
        len(relative_target.parts) != 2
        or _XET_HASH_REGEX.fullmatch(relative_target.parts[1]) is None
        or relative_target.parts[0] != relative_target.parts[1][:2]
    ):
        return None
    return shared_blobs_dir(cache_dir) / relative_target


def shared_blob_target(blob_path: str | Path, cache_dir: str | Path) -> Path | None:
    """Return the canonical store target if `blob_path` is a valid shared symlink.

    Callers check `is_shared_blobs_dir` once before resolving many blobs.
    """
    target = _shared_blob_target(blob_path, cache_dir)
    return target if target is not None and _is_regular_file(target) else None


def _manifest_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_MANIFEST_SUFFIX}")


def _lock_path_for_store_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.name}{_LOCK_SUFFIX}")


@contextmanager
def _shared_blob_lock(store_path: Path) -> Generator[None, None, None]:
    """Lock publication, reference creation, and GC for one content hash."""
    lock_path = _lock_path_for_store_path(store_path)
    # Create the lock file ourselves: `O_NOFOLLOW` refuses a planted symlink and 0o666 lets other users of a
    # shared cache take the same lock. `FileLock` alone would follow symlinks and apply the umask.
    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob lock is not a regular file: '{lock_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
    finally:
        os.close(fd)
    lock = FileLock(lock_path, mode=0o666)
    try:
        lock.acquire()
    except NotImplementedError:
        # SoftFileLock uses file existence as the lock, so it cannot reuse the flock file above.
        lock = SoftFileLock(f"{lock_path}.soft", mode=0o666)
        lock.acquire(timeout=_SOFT_LOCK_TIMEOUT)
    try:
        yield
    finally:
        try:
            lock.release()
        except OSError:
            pass


def _append_manifest_reference(manifest_path: Path, relative_blob_path: str) -> None:
    """Append and flush one reference before its symlink is made visible."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(manifest_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"Shared blob manifest is not a regular file: '{manifest_path}'.")
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
        line = f"{relative_blob_path}\n".encode()
        if os.write(fd, line) != len(line):
            raise OSError(f"Could not append a complete reference to '{manifest_path}'.")
        os.fsync(fd)
    finally:
        os.close(fd)


def _make_temporary_symlink(blob_path: Path, store_path: Path) -> Path:
    tmp_link = blob_path.with_name(f".{blob_path.name}.{secrets.token_hex(4)}.shared")
    relative_target = os.path.relpath(_path_for_comparison(store_path), start=_path_for_comparison(blob_path).parent)
    os.symlink(relative_target, tmp_link)
    return tmp_link


def shared_blobs_enabled() -> bool:
    """Return whether the shared blob store may be used."""
    return not constants.HF_HUB_DISABLE_SHARED_BLOBS and not constants.HF_HUB_DISABLE_XET


def try_link_from_shared_store(
    *, blob_path: str, xet_hash: str, cache_dir: str | Path, expected_size: int | None
) -> bool:
    """Materialize `blobs/<etag>` as a symlink to an existing store entry, if any.

    The reference manifest is flushed before the symlink becomes visible. Failures are
    best-effort misses and leave the regular download path untouched.
    """
    if _XET_HASH_REGEX.fullmatch(xet_hash) is None or not is_shared_blobs_dir(shared_blobs_dir(cache_dir)):
        return False
    store_path = shared_blob_path(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if relative_blob_path is None:
        return False

    tmp_link: Path | None = None
    try:
        with _shared_blob_lock(store_path):
            if not _is_usable_store_entry(store_path, expected_size):
                return False

            tmp_link = _make_temporary_symlink(Path(blob_path), store_path)
            _append_manifest_reference(_manifest_path_for_store_path(store_path), relative_blob_path)
            os.replace(tmp_link, blob_path)
    except OSError as e:
        logger.debug(f"Could not symlink '{blob_path}' from shared blob store: {e}")
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' reused from shared blob store (no download needed).")
    return True


def has_shared_blob(*, xet_hash: str, cache_dir: str | Path, expected_size: int | None) -> bool:
    """Return whether a usable store entry exists, without touching the cache."""
    if (
        not shared_blobs_enabled()
        or _XET_HASH_REGEX.fullmatch(xet_hash) is None
        or not is_shared_blobs_dir(shared_blobs_dir(cache_dir))
    ):
        return False
    return _is_usable_store_entry(shared_blob_path(cache_dir, xet_hash), expected_size)


def publish_blob_to_shared_store(
    *,
    blob_path: str,
    xet_hash: str,
    cache_dir: str | Path,
    expected_size: int | None,
    replace_existing: bool = False,
) -> bool:
    """Move a fresh Xet download into the store and replace its repo blob with a symlink.

    Best-effort: on failure the repo blob remains (or is restored as) a regular local
    file. Returns whether the repo blob was successfully shared.
    """
    # Publishing a payload whose size is unknown would create an entry no reader can validate.
    if expected_size is None or _XET_HASH_REGEX.fullmatch(xet_hash) is None:
        return False
    prefix_dir = _ensure_prefix_dir(cache_dir, xet_hash)
    relative_blob_path = _relative_blob_path(blob_path, cache_dir)
    if prefix_dir is None or relative_blob_path is None:
        return False

    blob_path_obj = Path(blob_path)
    store_path = shared_blob_path(cache_dir, xet_hash)
    manifest_path = _manifest_path_for_store_path(store_path)
    tmp_link: Path | None = None
    blob_moved = False
    try:
        with _shared_blob_lock(store_path):
            tmp_link = _make_temporary_symlink(blob_path_obj, store_path)
            store_is_usable = not replace_existing and _is_usable_store_entry(store_path, expected_size)
            _append_manifest_reference(manifest_path, relative_blob_path)
            if not store_is_usable:
                _prepare_shared_blob_permissions(blob_path_obj, prefix_dir, cache_dir)
                os.replace(blob_path_obj, store_path)
                blob_moved = True
            try:
                os.replace(tmp_link, blob_path_obj)
            except OSError:
                if blob_moved:
                    shutil.copyfile(store_path, blob_path_obj)  # restore under the lock, before GC can run
                    blob_moved = False
                raise
    except OSError as e:
        logger.debug(f"Could not publish '{blob_path}' to shared blob store: {e}")
        if blob_moved:
            raise OSError(f"Could not restore repo blob '{blob_path}' after shared-store failure") from e
        return False
    finally:
        if tmp_link is not None:
            tmp_link.unlink(missing_ok=True)

    logger.debug(f"Blob '{blob_path}' published to shared blob store.")
    return True


def _read_valid_manifest_references(store_path: Path, cache_dir: str | Path) -> set[Path] | None:
    manifest_path = _manifest_path_for_store_path(store_path)
    if not _is_regular_file(manifest_path):
        return None
    try:
        lines = manifest_path.read_text().splitlines()
    except (OSError, UnicodeError):
        return None

    references: set[Path] = set()
    cache_dir = Path(os.path.abspath(cache_dir))
    for line in lines:
        relative_path = Path(line)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        candidate = cache_dir / relative_path
        if _relative_blob_path(candidate, cache_dir) is None:
            continue
        try:
            target = _shared_blob_target(candidate, cache_dir, raise_on_io_error=True)
        except OSError:
            return None
        if target == store_path:
            references.add(candidate)
    return references


def expected_shared_blob_freed_size(
    store_path: Path, *, blob_paths_to_delete: set[Path], cache_dir: str | Path
) -> int:
    """Predict bytes freed if all known references are part of a deletion plan."""
    references = _read_valid_manifest_references(store_path, cache_dir)
    if references is None or not references or not references.issubset(blob_paths_to_delete):
        return 0
    try:
        return store_path.lstat().st_size if _is_regular_file(store_path) else 0
    except OSError:
        return 0


def shared_store_blob_paths(cache_dir: str | Path) -> set[Path]:
    """Return all valid canonical payload paths in a marked shared store."""
    store_dir = shared_blobs_dir(cache_dir)
    if not is_shared_blobs_dir(store_dir):
        return set()
    paths: set[Path] = set()
    try:
        prefix_dirs = list(store_dir.iterdir())
    except OSError:
        return paths
    for prefix_dir in prefix_dirs:
        if re.fullmatch(r"[0-9a-f]{2}", prefix_dir.name) is None or not _is_directory(prefix_dir):
            continue
        try:
            entries = list(prefix_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if (
                _XET_HASH_REGEX.fullmatch(entry.name) is not None
                and entry.name.startswith(prefix_dir.name)
                and _is_regular_file(entry)
            ):
                paths.add(entry)
    return paths


def unreferenced_shared_blobs(cache_dir: str | Path) -> dict[Path, int]:
    """Return store payloads without any valid manifest reference, with their sizes.

    Payloads with a missing or unreadable manifest are not listed, consistent with `sweep_shared_blob`.
    """
    unreferenced: dict[Path, int] = {}
    for store_path in shared_store_blob_paths(cache_dir):
        references = _read_valid_manifest_references(store_path, cache_dir)
        if references is not None and not references:
            try:
                unreferenced[store_path] = store_path.lstat().st_size
            except OSError:
                pass
    return unreferenced


def _rewrite_manifest(manifest_path: Path, references: set[Path], cache_dir: str | Path) -> None:
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{secrets.token_hex(4)}.tmp")
    cache_dir = Path(os.path.abspath(cache_dir))
    try:
        content = "".join(f"{path.relative_to(cache_dir).as_posix()}\n" for path in sorted(references))
        tmp_path.write_text(content)
        tmp_path.chmod(0o666)
        with tmp_path.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(tmp_path, manifest_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def sweep_shared_blob(store_path: Path, *, cache_dir: str | Path) -> int:
    """Collect one shared blob if its manifest has no remaining valid references.

    Missing, invalid, or unreadable metadata is never grounds for deletion. Returns the
    number of payload bytes removed.
    """
    store_path = Path(os.path.abspath(store_path))
    cache_dir = Path(os.path.abspath(cache_dir))
    expected_store_dir = Path(os.path.abspath(shared_blobs_dir(cache_dir)))
    try:
        relative_store_path = store_path.relative_to(expected_store_dir)
    except ValueError:
        return 0
    if (
        not is_shared_blobs_dir(expected_store_dir)
        or len(relative_store_path.parts) != 2
        or _XET_HASH_REGEX.fullmatch(store_path.name) is None
        or store_path.parent.name != store_path.name[:2]
    ):
        return 0

    try:
        with _shared_blob_lock(store_path):
            references = _read_valid_manifest_references(store_path, cache_dir)
            if references is None:
                return 0
            manifest_path = _manifest_path_for_store_path(store_path)
            if references:
                _rewrite_manifest(manifest_path, references, cache_dir)
                return 0
            if not _is_regular_file(store_path):
                return 0
            size = store_path.lstat().st_size
            store_path.unlink()
            manifest_path.unlink(missing_ok=True)
            return size
    except OSError as e:
        logger.debug(f"Could not collect shared blob '{store_path}': {e}")
        return 0
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to manage the HF cache directory."""

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub.errors import CacheNotFound, CorruptedCacheException

from ..constants import HF_HUB_CACHE, REPO_TYPES_MAPPING
from . import _shared_blobs, logging
from ._parsing import format_timesince
from ._terminal import tabulate


logger = logging.get_logger(__name__)

REPO_TYPE_T = Literal["model", "dataset", "space", "kernel"]

# List of OS-created helper files that need to be ignored
FILES_TO_IGNORE = [".DS_Store", "Thumbs.db", "desktop.ini"]


@dataclass(frozen=True)
class CachedFileInfo:
    """Frozen data structure holding information about a single cached file.

    Args:
        file_name (`str`):
            Name of the file. Example: `config.json`.
        file_path (`Path`):
            Path of the file in the `snapshots` directory. The file path is a symlink
            referring to a blob in the `blobs` folder.
        blob_path (`Path`):
            Path of the repo-local blob entry referenced by `file_path`. The blob entry
            can itself be a symlink to the cache-wide shared blob store.
        size_on_disk (`int`):
            Size of the blob file in bytes.
        blob_last_accessed (`float`):
            Timestamp of the last time the blob file has been accessed (from any
            revision).
        blob_last_modified (`float`):
            Timestamp of the last time the blob file has been modified/created.

    > [!WARNING]
    > `blob_last_accessed` and `blob_last_modified` reliability can depend on the OS you
    > are using. See [python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
    > for more details.
    """

    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int

    blob_last_accessed: float
    blob_last_modified: float

    @property
    def blob_last_accessed_str(self) -> str:
        """
        (property) Timestamp of the last time the blob file has been accessed (from any
        revision), returned as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.blob_last_accessed)

    @property
    def blob_last_modified_str(self) -> str:
        """
        (property) Timestamp of the last time the blob file has been modified, returned
        as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.blob_last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Size of the blob file as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)


@dataclass(frozen=True)
class CachedRevisionInfo:
    """Frozen data structure holding information about a revision.

    A revision correspond to a folder in the `snapshots` folder and is populated with
    the exact tree structure as the repo on the Hub but contains only symlinks. A
    revision can be either referenced by 1 or more `refs` or be "detached" (no refs).

    Args:
        commit_hash (`str`):
            Hash of the revision (unique).
            Example: `"9338f7b671827df886678df2bdd7cc7b4f36dffd"`.
        snapshot_path (`Path`):
            Path to the revision directory in the `snapshots` folder. It contains the
            exact tree structure as the repo on the Hub.
        files: (`frozenset[CachedFileInfo]`):
            Set of [`~CachedFileInfo`] describing all files contained in the snapshot.
        refs (`frozenset[str]`):
            Set of `refs` pointing to this revision. If the revision has no `refs`, it
            is considered detached.
            Example: `{"main", "2.4.0"}` or `{"refs/pr/1"}`.
        size_on_disk (`int`):
            Sum of the blob file sizes that are symlink-ed by the revision.
        last_modified (`float`):
            Timestamp of the last time the revision has been created/modified.

    > [!WARNING]
    > `last_accessed` cannot be determined correctly on a single revision as blob files
    > are shared across revisions.

    > [!WARNING]
    > `size_on_disk` is not necessarily the sum of all file sizes because of possible
    > duplicated files. Besides, only blobs are taken into account, not the (negligible)
    > size of folders and symlinks.
    """

    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: frozenset[CachedFileInfo]
    refs: frozenset[str]

    last_modified: float

    @property
    def last_modified_str(self) -> str:
        """
        (property) Timestamp of the last time the revision has been modified, returned
        as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        """
        (property) Total number of files in the revision.
        """
        return len(self.files)


@dataclass(frozen=True)
class CachedRepoInfo:
    """Frozen data structure holding information about a cached repository.

    Args:
        repo_id (`str`):
            Repo id of the repo on the Hub. Example: `"google/fleurs"`.
        repo_type (`Literal["dataset", "model", "space", "kernel"]`):
            Type of the cached repo.
        repo_path (`Path`):
            Local path to the cached repo.
        size_on_disk (`int`):
            Sum of the blob file sizes in the cached repo.
        nb_files (`int`):
            Total number of blob files in the cached repo.
        revisions (`frozenset[CachedRevisionInfo]`):
            Set of [`~CachedRevisionInfo`] describing all revisions cached in the repo.
        last_accessed (`float`):
            Timestamp of the last time a blob file of the repo has been accessed.
        last_modified (`float`):
            Timestamp of the last time a blob file of the repo has been modified/created.

    > [!WARNING]
    > `size_on_disk` is not necessarily the sum of all revisions sizes because of
    > duplicated files. Besides, only blobs are taken into account, not the (negligible)
    > size of folders and symlinks.

    > [!WARNING]
    > `last_accessed` and `last_modified` reliability can depend on the OS you are using.
    > See [python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
    > for more details.
    """

    repo_id: str
    repo_type: REPO_TYPE_T
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: frozenset[CachedRevisionInfo]

    last_accessed: float
    last_modified: float

    @property
    def last_accessed_str(self) -> str:
        """
        (property) Last time a blob file of the repo has been accessed, returned as a
        human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_accessed)

    @property
    def last_modified_str(self) -> str:
        """
        (property) Last time a blob file of the repo has been modified, returned as a
        human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    @property
    def cache_id(self) -> str:
        """Canonical `type/id` identifier used across cache tooling."""
        return f"{self.repo_type}/{self.repo_id}"

    @property
    def refs(self) -> dict[str, CachedRevisionInfo]:
        """
        (property) Mapping between `refs` and revision data structures.
        """
        return {ref: revision for revision in self.revisions for ref in revision.refs}


@dataclass(frozen=True)
class DeleteCacheStrategy:
    """Frozen data structure holding the strategy to delete cached revisions or files.

    This object is not meant to be instantiated programmatically but to be returned by
    [`~utils.HFCacheInfo.delete_revisions`] or [`~utils.HFCacheInfo.delete_files`]. See
    documentation for usage example.

    Args:
        expected_freed_size (`float`):
            Expected freed size once strategy is executed.
        blobs (`frozenset[Path]`):
            Set of blob file paths to be deleted.
        refs (`frozenset[Path]`):
            Set of reference file paths to be deleted.
        repos (`frozenset[Path]`):
            Set of entire repo paths to be deleted.
        snapshots (`frozenset[Path]`):
            Set of snapshots to be deleted (directory of symlinks).
        files (`frozenset[Path]`, *optional*):
            Set of individual snapshot entries to be deleted. Their blobs are deleted only if
            listed in `blobs`.
        cache_dir (`Path` or `None`):
            Cache directory the strategy was computed from. Used to collect shared
            blobs referenced by repo-local symlinks removed by the deletion.
    """

    expected_freed_size: int
    blobs: frozenset[Path]
    refs: frozenset[Path]
    repos: frozenset[Path]
    snapshots: frozenset[Path]
    files: frozenset[Path] = frozenset()
    cache_dir: Path | None = None

    @property
    def expected_freed_size_str(self) -> str:
        """
        (property) Expected size that will be freed as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.expected_freed_size)

    def execute(self) -> None:
        """Execute the defined strategy.

        > [!WARNING]
        > If this method is interrupted, the cache might get corrupted. Deletion order is
        > implemented so that references and symlinks are deleted before the actual blob
        > files.

        > [!WARNING]
        > This method is irreversible. If executed, cached files are erased and must be
        > downloaded again.
        """
        # Record shared targets before their repo symlinks are deleted; only those manifests are swept below.
        shared_store_paths: set[Path] = set()
        if self.cache_dir is not None and _shared_blobs.is_shared_blobs_dir(
            _shared_blobs.shared_blobs_dir(self.cache_dir)
        ):
            blob_paths = set(self.blobs)
            for repo_path in self.repos:
                blobs_dir = repo_path / "blobs"
                try:
                    blob_paths.update(blobs_dir.iterdir())
                except OSError:
                    pass
            for blob_path in blob_paths:
                if store_path := _shared_blobs.shared_blob_target(blob_path, self.cache_dir):
                    shared_store_paths.add(store_path)

        # Deletion order matters. Blobs are deleted in last so that the user can't end
        # up in a state where a `ref`` refers to a missing snapshot or a snapshot
        # symlink refers to a deleted blob.

        # Unlink snapshot entries first: if it fails, their blobs must not be deleted.
        for path in self.files:
            path.unlink(missing_ok=True)

        # Delete entire repos
        for path in self.repos:
            _try_delete_path(path, path_type="repo")

        # Delete snapshot directories
        for path in self.snapshots:
            _try_delete_path(path, path_type="snapshot")

        # Delete refs files
        for path in self.refs:
            _try_delete_path(path, path_type="ref")

        # Delete blob files
        for path in self.blobs:
            _try_delete_path(path, path_type="blob")

        # Manifest entries are validated against the filesystem; failures leak data rather than delete it.
        if self.cache_dir is not None:
            for store_path in shared_store_paths:
                _shared_blobs.sweep_shared_blob(store_path, cache_dir=self.cache_dir)

        logger.info(f"Cache deletion done. Saved {self.expected_freed_size_str}.")


@dataclass(frozen=True)
class CachedIncompleteFileInfo:
    """Frozen data structure holding information about a single incomplete download.

    Interrupted downloads leave `<cache>/<repo>/blobs/<etag>.incomplete` files behind.
    These are not part of any committed revision, so they are surfaced separately by
    [`scan_cache_dir`].

    Args:
        file_path (`Path`):
            Path of the `.incomplete` file in the `blobs` folder.
        size_on_disk (`int`):
            Size of the partially-downloaded file in bytes.
    """

    file_path: Path
    size_on_disk: int


@dataclass(frozen=True)
class HFCacheInfo:
    """Frozen data structure holding information about the entire cache-system.

    This data structure is returned by [`scan_cache_dir`] and is immutable.

    Args:
        size_on_disk (`int`):
            Physical size of the cache-system in bytes. A blob shared across repos is
            counted once, and store-only shared blobs no longer referenced by any repo are
            included. Can be smaller than the sum of the per-repo sizes.
        repos (`frozenset[CachedRepoInfo]`):
            Set of [`~CachedRepoInfo`] describing all valid cached repos found on the
            cache-system while scanning.
        incomplete_files (`frozenset[CachedIncompleteFileInfo]`):
            Set of [`~CachedIncompleteFileInfo`] describing orphaned `*.incomplete`
            files left behind by interrupted downloads.
        warnings (`list[CorruptedCacheException]`):
            List of [`~CorruptedCacheException`] that occurred while scanning the cache.
            Those exceptions are captured so that the scan can continue. Corrupted repos
            are skipped from the scan.
        cache_dir (`Path` or `None`):
            Cache directory that was scanned.

    > [!WARNING]
    > `size_on_disk` only accounts for blobs, and corrupted repos are skipped. It is a
    > physical size: unlike `CachedRepoInfo.size_on_disk`, which attributes a shared blob
    > to every repo referencing it, a shared blob is counted here only once.
    """

    size_on_disk: int
    repos: frozenset[CachedRepoInfo]
    incomplete_files: frozenset[CachedIncompleteFileInfo]
    warnings: list[CorruptedCacheException]
    cache_dir: Path | None = None

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Physical size of the cache-system as a human-readable string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    @property
    def incomplete_size_on_disk(self) -> int:
        """(property) Sum of all incomplete download sizes in bytes."""
        return sum(file.size_on_disk for file in self.incomplete_files)

    def delete_revisions(self, *revisions: str) -> DeleteCacheStrategy:
        """Prepare the strategy to delete one or more revisions cached locally.

        Input revisions can be any revision hash. If a revision hash is not found in the
        local cache, a warning is thrown but no error is raised. Revisions can be from
        different cached repos since hashes are unique across repos,

        Examples:
        ```py
        >>> from huggingface_hub import scan_cache_dir
        >>> cache_info = scan_cache_dir()
        >>> delete_strategy = cache_info.delete_revisions(
        ...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
        ... )
        >>> print(f"Will free {delete_strategy.expected_freed_size_str}.")
        Will free 7.9K.
        >>> delete_strategy.execute()
        Cache deletion done. Saved 7.9K.
        ```

        ```py
        >>> from huggingface_hub import scan_cache_dir
        >>> scan_cache_dir().delete_revisions(
        ...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa",
        ...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
        ...     "6c0e6080953db56375760c0471a8c5f2929baf11",
        ... ).execute()
        Cache deletion done. Saved 8.6G.
        ```

        > [!WARNING]
        > `delete_revisions` returns a [`~utils.DeleteCacheStrategy`] object that needs to
        > be executed. The [`~utils.DeleteCacheStrategy`] is not meant to be modified but
        > allows having a dry run before actually executing the deletion.
        """
        hashes_to_delete: set[str] = set(revisions)

        repos_with_revisions: dict[CachedRepoInfo, set[CachedRevisionInfo]] = defaultdict(set)

        for repo in self.repos:
            for revision in repo.revisions:
                if revision.commit_hash in hashes_to_delete:
                    repos_with_revisions[repo].add(revision)
                    hashes_to_delete.remove(revision.commit_hash)

        if len(hashes_to_delete) > 0:
            logger.warning(f"Revision(s) not found - cannot delete them: {', '.join(hashes_to_delete)}")

        delete_strategy_blobs: set[Path] = set()
        delete_strategy_refs: set[Path] = set()
        delete_strategy_repos: set[Path] = set()
        delete_strategy_snapshots: set[Path] = set()

        # Blob entries the strategy will unlink (path -> scanned size).
        blobs_to_unlink: dict[Path, int] = {}

        for affected_repo, revisions_to_delete in repos_with_revisions.items():
            other_revisions = affected_repo.revisions - revisions_to_delete

            # If no other revisions, it means all revisions are deleted
            # -> delete the entire cached repo
            if len(other_revisions) == 0:
                delete_strategy_repos.add(affected_repo.repo_path)
                for revision in affected_repo.revisions:
                    for file in revision.files:
                        blobs_to_unlink[file.blob_path] = file.size_on_disk
                continue

            # Some revisions of the repo will be deleted but not all. We need to filter
            # which blob files will not be linked anymore.
            other_blob_paths = {rev_file.blob_path for revision in other_revisions for rev_file in revision.files}
            for revision_to_delete in revisions_to_delete:
                # Snapshot dir
                delete_strategy_snapshots.add(revision_to_delete.snapshot_path)

                # Refs dir
                for ref in revision_to_delete.refs:
                    delete_strategy_refs.add(affected_repo.repo_path / "refs" / ref)

                # Blobs dir
                for file in revision_to_delete.files:
                    # Some files are not symlinks to the blobs dir: files copied instead of symlinked
                    # (Windows, filesystems without symlink support) or files created by the user
                    # inside the snapshot dir (e.g. build artifacts from a `pip install`). They are
                    # already removed when deleting the snapshot dir -> count them but don't delete
                    # them a second time.
                    if revision_to_delete.snapshot_path in file.blob_path.parents:
                        blobs_to_unlink[file.blob_path] = file.size_on_disk
                        continue

                    # Blob file not referenced by remaining revisions -> delete
                    if file.blob_path not in other_blob_paths:
                        delete_strategy_blobs.add(file.blob_path)
                        blobs_to_unlink[file.blob_path] = file.size_on_disk

        # Return the strategy instead of executing it.
        return DeleteCacheStrategy(
            blobs=frozenset(delete_strategy_blobs),
            refs=frozenset(delete_strategy_refs),
            repos=frozenset(delete_strategy_repos),
            snapshots=frozenset(delete_strategy_snapshots),
            expected_freed_size=_expected_freed_size(blobs_to_unlink, self.cache_dir),
            cache_dir=self.cache_dir,
        )

    def delete_files(self, *files: CachedFileInfo) -> DeleteCacheStrategy:
        """Prepare the strategy to delete individual cached files.

        Only the snapshot entries are removed. A blob is deleted only if no other cached file
        references it, so blobs shared with other revisions are kept. Refs and snapshot
        directories are kept as well, even if a snapshot ends up empty. To delete whole
        revisions or repos, use [`~utils.HFCacheInfo.delete_revisions`].

        Example:
        ```py
        >>> from huggingface_hub import scan_cache_dir
        >>> cache_info = scan_cache_dir()
        >>> files = [
        ...     file
        ...     for repo in cache_info.repos
        ...     if repo.repo_id == "org/model"
        ...     for revision in repo.revisions
        ...     for file in revision.files
        ...     if file.file_name == "model-Q4_K_M.gguf"
        ... ]
        >>> delete_strategy = cache_info.delete_files(*files)
        >>> print(f"Will free {delete_strategy.expected_freed_size_str}.")
        Will free 4.9G.
        >>> delete_strategy.execute()
        Cache deletion done. Saved 4.9G.
        ```
        """
        selected = set(files)
        retained_blobs = {
            file.blob_path
            for repo in self.repos
            for revision in repo.revisions
            for file in revision.files
            if file not in selected
        }
        blobs_to_unlink = {
            file.blob_path: file.size_on_disk for file in selected if file.blob_path not in retained_blobs
        }
        file_paths = frozenset(file.file_path for file in selected)
        return DeleteCacheStrategy(
            expected_freed_size=_expected_freed_size(blobs_to_unlink, self.cache_dir),
            # Files stored directly in the snapshot (no symlink) are already removed with the entry.
            blobs=frozenset(blobs_to_unlink.keys() - file_paths),
            refs=frozenset(),
            repos=frozenset(),
            snapshots=frozenset(),
            files=file_paths,
            cache_dir=self.cache_dir,
        )

    def export_as_table(self, *, verbosity: int = 0) -> str:
        """Generate a table from the [`HFCacheInfo`] object.

        Pass `verbosity=0` to get a table with a single row per repo, with columns
        "repo_id", "repo_type", "size_on_disk", "nb_files", "last_accessed", "last_modified", "refs", "local_path".

        Pass `verbosity=1` to get a table with a row per repo and revision (thus multiple rows can appear for a single repo), with columns
        "repo_id", "repo_type", "revision", "size_on_disk", "nb_files", "last_modified", "refs", "local_path".

        Example:
        ```py
        >>> from huggingface_hub.utils import scan_cache_dir

        >>> hf_cache_info = scan_cache_dir()
        HFCacheInfo(...)

        >>> print(hf_cache_info.export_as_table())
        REPO ID                                             REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS LOCAL PATH
        --------------------------------------------------- --------- ------------ -------- ------------- ------------- ---- --------------------------------------------------------------------------------------------------
        roberta-base                                        model             2.7M        5 1 day ago     1 week ago    main ~/.cache/huggingface/hub/models--roberta-base
        suno/bark                                           model             8.8K        1 1 week ago    1 week ago    main ~/.cache/huggingface/hub/models--suno--bark
        t5-base                                             model           893.8M        4 4 days ago    7 months ago  main ~/.cache/huggingface/hub/models--t5-base
        t5-large                                            model             3.0G        4 5 weeks ago   5 months ago  main ~/.cache/huggingface/hub/models--t5-large

        >>> print(hf_cache_info.export_as_table(verbosity=1))
        REPO ID                                             REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED REFS LOCAL PATH
        --------------------------------------------------- --------- ---------------------------------------- ------------ -------- ------------- ---- -----------------------------------------------------------------------------------------------------------------------------------------------------
        roberta-base                                        model     e2da8e2f811d1448a5b465c236feacd80ffbac7b         2.7M        5 1 week ago    main ~/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b
        suno/bark                                           model     70a8a7d34168586dc5d028fa9666aceade177992         8.8K        1 1 week ago    main ~/.cache/huggingface/hub/models--suno--bark/snapshots/70a8a7d34168586dc5d028fa9666aceade177992
        t5-base                                             model     a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1       893.8M        4 7 months ago  main ~/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
        t5-large                                            model     150ebc2c4b72291e770f58e6057481c8d2ed331a         3.0G        4 5 months ago  main ~/.cache/huggingface/hub/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a
        ```

        Args:
            verbosity (`int`, *optional*):
                The verbosity level. Defaults to 0.

        Returns:
            `str`: The table as a string.
        """
        if verbosity == 0:
            return tabulate(
                rows=[
                    [
                        repo.repo_id,
                        repo.repo_type,
                        f"{repo.size_on_disk_str:>12}",
                        repo.nb_files,
                        repo.last_accessed_str,
                        repo.last_modified_str,
                        ", ".join(sorted(repo.refs)),
                        str(repo.repo_path),
                    ]
                    for repo in sorted(self.repos, key=lambda repo: repo.repo_path)
                ],
                headers=[
                    "REPO ID",
                    "REPO TYPE",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_ACCESSED",
                    "LAST_MODIFIED",
                    "REFS",
                    "LOCAL PATH",
                ],
            )
        else:
            return tabulate(
                rows=[
                    [
                        repo.repo_id,
                        repo.repo_type,
                        revision.commit_hash,
                        f"{revision.size_on_disk_str:>12}",
                        revision.nb_files,
                        revision.last_modified_str,
                        ", ".join(sorted(revision.refs)),
                        str(revision.snapshot_path),
                    ]
                    for repo in sorted(self.repos, key=lambda repo: repo.repo_path)
                    for revision in sorted(repo.revisions, key=lambda revision: revision.commit_hash)
                ],
                headers=[
                    "REPO ID",
                    "REPO TYPE",
                    "REVISION",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_MODIFIED",
                    "REFS",
                    "LOCAL PATH",
                ],
            )


def scan_cache_dir(cache_dir: str | Path | None = None) -> HFCacheInfo:
    """Scan the entire HF cache-system and return a [`~HFCacheInfo`] structure.

    Use `scan_cache_dir` in order to programmatically scan your cache-system. The cache
    will be scanned repo by repo. If a repo is corrupted, a [`~CorruptedCacheException`]
    will be thrown internally but captured and returned in the [`~HFCacheInfo`]
    structure. Only valid repos get a proper report.

    ```py
    >>> from huggingface_hub import scan_cache_dir

    >>> hf_cache_info = scan_cache_dir()
    HFCacheInfo(
        size_on_disk=3398085269,
        repos=frozenset({
            CachedRepoInfo(
                repo_id='t5-small',
                repo_type='model',
                repo_path=PosixPath(...),
                size_on_disk=970726914,
                nb_files=11,
                revisions=frozenset({
                    CachedRevisionInfo(
                        commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                        size_on_disk=970726339,
                        snapshot_path=PosixPath(...),
                        files=frozenset({
                            CachedFileInfo(
                                file_name='config.json',
                                size_on_disk=1197
                                file_path=PosixPath(...),
                                blob_path=PosixPath(...),
                            ),
                            CachedFileInfo(...),
                            ...
                        }),
                    ),
                    CachedRevisionInfo(...),
                    ...
                }),
            ),
            CachedRepoInfo(...),
            ...
        }),
        warnings=[
            CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
            CorruptedCacheException(...),
            ...
        ],
    )
    ```

    You can also print a detailed report directly from the `hf` command line using:
    ```text
    > hf cache ls
    ID                          SIZE     LAST_ACCESSED LAST_MODIFIED REFS
    --------------------------- -------- ------------- ------------- -----------
    dataset/nyu-mll/glue          157.4M 2 days ago    2 days ago    main script
    model/LiquidAI/LFM2-VL-1.6B     3.2G 4 days ago    4 days ago    main
    model/microsoft/UserLM-8b      32.1G 4 days ago    4 days ago    main

    Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
    Got 1 warning(s) while scanning. Use -vvv to print details.
    ```

    Args:
        cache_dir (`str` or `Path`, `optional`):
            Cache directory to cache. Defaults to the default HF cache directory.

    > [!WARNING]
    > Raises:
    >
    >     `CacheNotFound`
    >       If the cache directory does not exist.
    >
    >     [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
    >       If the cache directory is a file, instead of a directory.

    Returns: a [`~HFCacheInfo`] object.
    """
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise CacheNotFound(
            f"Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.",
            cache_dir=cache_dir,
        )

    if cache_dir.is_file():
        raise ValueError(
            f"Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable."
        )

    repos: set[CachedRepoInfo] = set()
    warnings: list[CorruptedCacheException] = []
    for repo_path in cache_dir.iterdir():
        if repo_path.name in FILES_TO_IGNORE:
            continue
        if repo_path.name == ".locks":  # skip './.locks/' folder
            continue
        if repo_path.name == "CACHEDIR.TAG":  # skip CACHEDIR.TAG file
            continue
        if repo_path.name == _shared_blobs.SHARED_BLOBS_DIR_NAME and _shared_blobs.is_shared_blobs_dir(repo_path):
            continue
        try:
            repos.add(_scan_cached_repo(repo_path))
        except CorruptedCacheException as e:
            warnings.append(e)

    # Per-repo sizes stay logical. The cache-wide total counts each shared payload once, including
    # store-only payloads no repo references anymore.
    store_marked = _shared_blobs.is_shared_blobs_dir(_shared_blobs.shared_blobs_dir(cache_dir))
    repo_blob_paths = {
        file.blob_path
        for repo in repos
        for revision in repo.revisions
        for file in revision.files
        if not store_marked or _shared_blobs.shared_blob_target(file.blob_path, cache_dir) is None
    }
    physical_blob_paths = repo_blob_paths | _shared_blobs.shared_store_blob_paths(cache_dir)
    physical_size = 0
    for blob_path in physical_blob_paths:
        try:
            physical_size += blob_path.stat().st_size
        except OSError:
            pass

    return HFCacheInfo(
        repos=frozenset(repos),
        size_on_disk=physical_size,
        incomplete_files=_scan_incomplete_files(cache_dir),
        warnings=warnings,
        cache_dir=cache_dir,
    )


def _scan_incomplete_files(cache_dir: Path) -> frozenset[CachedIncompleteFileInfo]:
    """Find orphaned `*.incomplete` partial-download files in the cache.

    Interrupted downloads leave `<cache>/<repo>/blobs/<etag>.incomplete` files behind.
    These are not part of any committed revision, so they are reported separately in the
    [`~HFCacheInfo`] returned by [`scan_cache_dir`].
    """
    files: set[CachedIncompleteFileInfo] = set()
    for path in cache_dir.glob("*/blobs/*.incomplete"):
        try:
            size_on_disk = path.stat().st_size
        except OSError:
            continue
        files.add(CachedIncompleteFileInfo(file_path=path, size_on_disk=size_on_disk))
    return frozenset(files)


def _scan_cached_repo(repo_path: Path) -> CachedRepoInfo:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a [`~CorruptedCacheException`].
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException(f"Repo path is not a directory: {repo_path}")

    if "--" not in repo_path.name:
        raise CorruptedCacheException(f"Repo path is not a valid HuggingFace cache directory: {repo_path}")

    repo_type_prefix, repo_id = repo_path.name.split("--", maxsplit=1)
    repo_id = repo_id.replace("--", "/")  # google/fleurs -> "google/fleurs"

    if repo_type_prefix not in REPO_TYPES_MAPPING:
        raise CorruptedCacheException(
            f"Repo type must be one of {sorted(REPO_TYPES_MAPPING)}, found `{repo_type_prefix}` ({repo_path})."
        )
    repo_type = REPO_TYPES_MAPPING[repo_type_prefix]  # "models" -> "model"

    blob_stats: dict[Path, os.stat_result] = {}  # Key is blob_path, value is blob stats

    snapshots_path = repo_path / "snapshots"
    refs_path = repo_path / "refs"

    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException(f"Snapshots dir doesn't exist in cached repo: {snapshots_path}")

    # Scan over `refs` directory

    # key is revision hash, value is set of refs
    refs_by_hash: dict[str, set[str]] = defaultdict(set)
    if refs_path.exists():
        # Example of `refs` directory
        # ── refs
        #     ├── main
        #     └── refs
        #         └── pr
        #             └── 1
        if refs_path.is_file():
            raise CorruptedCacheException(f"Refs directory cannot be a file: {refs_path}")

        for ref_path in refs_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if ref_path.is_dir() or ref_path.name in FILES_TO_IGNORE:
                continue

            ref_name = ref_path.relative_to(refs_path).as_posix()
            with ref_path.open() as f:
                commit_hash = f.read()

            refs_by_hash[commit_hash].add(ref_name)

    # Scan snapshots directory
    cached_revisions: set[CachedRevisionInfo] = set()
    for revision_path in snapshots_path.iterdir():
        # Ignore OS-created helper files
        if revision_path.name in FILES_TO_IGNORE:
            continue
        if revision_path.is_file():
            raise CorruptedCacheException(f"Snapshots folder corrupted. Found a file: {revision_path}")

        cached_files = set()
        for file_path in revision_path.glob("**/*"):
            # glob("**/*") iterates over all files and directories -> skip directories
            if file_path.is_dir():
                continue

            # Keep the first symlink hop: a repo blob may itself point into the shared store, and
            # deletion must unlink the repo reference, not the shared payload.
            if file_path.is_symlink():
                blob_path = Path(os.path.abspath(file_path.parent / os.readlink(file_path)))
            else:
                blob_path = file_path
            if not blob_path.exists():
                raise CorruptedCacheException(f"Blob missing (broken symlink): {blob_path}")

            if blob_path not in blob_stats:
                blob_stats[blob_path] = blob_path.stat()

            cached_files.add(
                CachedFileInfo(
                    file_name=file_path.name,
                    file_path=file_path,
                    size_on_disk=blob_stats[blob_path].st_size,
                    blob_path=blob_path,
                    blob_last_accessed=blob_stats[blob_path].st_atime,
                    blob_last_modified=blob_stats[blob_path].st_mtime,
                )
            )

        # Last modified is either the last modified blob file or the revision folder
        # itself if it is empty
        if len(cached_files) > 0:
            revision_last_modified = max(blob_stats[file.blob_path].st_mtime for file in cached_files)
        else:
            revision_last_modified = revision_path.stat().st_mtime

        cached_revisions.add(
            CachedRevisionInfo(
                commit_hash=revision_path.name,
                files=frozenset(cached_files),
                refs=frozenset(refs_by_hash.pop(revision_path.name, set())),
                size_on_disk=sum(
                    blob_stats[blob_path].st_size for blob_path in {file.blob_path for file in cached_files}
                ),
                snapshot_path=revision_path,
                last_modified=revision_last_modified,
            )
        )

    # Check that all refs referred to an existing revision
    if len(refs_by_hash) > 0:
        raise CorruptedCacheException(
            f"Reference(s) refer to missing commit hashes: {dict(refs_by_hash)} ({repo_path})."
        )

    # Last modified is either the last modified blob file or the repo folder itself if
    # no blob files has been found. Same for last accessed.
    if len(blob_stats) > 0:
        repo_last_accessed = max(stat.st_atime for stat in blob_stats.values())
        repo_last_modified = max(stat.st_mtime for stat in blob_stats.values())
    else:
        repo_stats = repo_path.stat()
        repo_last_accessed = repo_stats.st_atime
        repo_last_modified = repo_stats.st_mtime

    # Build and return frozen structure
    return CachedRepoInfo(
        nb_files=len(blob_stats),
        repo_id=repo_id,
        repo_path=repo_path,
        repo_type=repo_type,  # type: ignore
        revisions=frozenset(cached_revisions),
        size_on_disk=sum(stat.st_size for stat in blob_stats.values()),
        last_accessed=repo_last_accessed,
        last_modified=repo_last_modified,
    )


def _format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    num_f = float(num)
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_f) < 1000.0:
            return f"{num_f:3.1f}{unit}"
        num_f /= 1000.0
    return f"{num_f:.1f}Y"


def _expected_freed_size(blobs_to_unlink: dict[Path, int], cache_dir: Path | None) -> int:
    """Expected number of bytes freed by unlinking a set of blob files.

    Regular repo blobs retain the historical logical accounting. Shared symlinks count
    their canonical payload once only when every currently valid manifest reference is
    included in the deletion plan.
    """
    if cache_dir is None or not _shared_blobs.is_shared_blobs_dir(_shared_blobs.shared_blobs_dir(cache_dir)):
        return sum(blobs_to_unlink.values())
    expected_freed_size = 0
    shared_store_paths: set[Path] = set()
    for blob_path, size_on_disk in blobs_to_unlink.items():
        if store_path := _shared_blobs.shared_blob_target(blob_path, cache_dir):
            shared_store_paths.add(store_path)
        else:
            expected_freed_size += size_on_disk
    blob_paths_to_delete = set(blobs_to_unlink)
    for store_path in shared_store_paths:
        expected_freed_size += _shared_blobs.expected_shared_blob_freed_size(
            store_path, blob_paths_to_delete=blob_paths_to_delete, cache_dir=cache_dir
        )
    return expected_freed_size


def _try_delete_path(path: Path, path_type: str) -> None:
    """Try to delete a local file or folder.

    If the path does not exist, error is logged as a warning and then ignored.

    Args:
        path (`Path`):
            Path to delete. Can be a file or a folder.
        path_type (`str`):
            What path are we deleting ? Only for logging purposes. Example: "snapshot".
    """
    logger.debug(f"Delete {path_type}: {path}")
    try:
        if path.is_file() or path.is_symlink():  # `is_file()` is False for a broken symlink
            os.remove(path)
        else:
            shutil.rmtree(path)
    except FileNotFoundError:
        # Nothing to delete: the path is already gone. Log without traceback to avoid flooding the
        # terminal when deleting many paths at once.
        logger.warning(f"Couldn't delete {path_type}: file not found ({path})")
    except PermissionError:
        logger.warning(f"Couldn't delete {path_type}: permission denied ({path})", exc_info=True)
<|endoftext|>import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx2

from .. import constants
from . import hf_raise_for_status, http_backoff, validate_hf_hub_args


_REGEX_XET_HASH = re.compile(r"^[0-9a-fA-F]{64}$")

XET_CONNECTION_INFO_SAFETY_PERIOD = 60  # seconds
XET_CONNECTION_INFO_CACHE_SIZE = 1_000
XET_CONNECTION_INFO_CACHE: dict[str, "XetConnectionInfo"] = {}
_XET_CONNECTION_INFO_LOCK = threading.Lock()  # guards the cache dict and the per-key lock registry
_XET_CONNECTION_INFO_KEY_LOCKS: dict[str, threading.Lock] = {}


def is_valid_xet_hash(xet_hash: str) -> bool:
    """Return whether `xet_hash` looks like a real Xet content hash (64 hex chars)."""
    return bool(_REGEX_XET_HASH.fullmatch(xet_hash))


class XetTokenType(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class XetFileData:
    file_hash: str
    refresh_route: str


@dataclass(frozen=True)
class XetConnectionInfo:
    access_token: str
    expiration_unix_epoch: int
    endpoint: str


def parse_xet_file_data_from_response(response: httpx2.Response, endpoint: str | None = None) -> XetFileData | None:
    """
    Parse XET file metadata from an HTTP response.

    This function extracts XET file metadata from the HTTP headers or HTTP links
    of a given response object. If the required metadata is not found, it returns `None`.

    Args:
        response (`httpx2.Response`):
            The HTTP response object containing headers dict and links dict to extract the XET metadata from.
    Returns:
        `Optional[XetFileData]`:
            An instance of `XetFileData` containing the file hash and refresh route if the metadata
            is found. Returns `None` if the required metadata is missing.
    """
    if response is None:
        return None
    try:
        file_hash = response.headers[constants.HUGGINGFACE_HEADER_X_XET_HASH]

        if constants.HUGGINGFACE_HEADER_LINK_XET_AUTH_KEY in response.links:
            refresh_route = response.links[constants.HUGGINGFACE_HEADER_LINK_XET_AUTH_KEY]["url"]
        else:
            refresh_route = response.headers[constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE]
    except KeyError:
        return None
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    if refresh_route.startswith(constants.HUGGINGFACE_CO_URL_HOME):
        refresh_route = refresh_route.replace(constants.HUGGINGFACE_CO_URL_HOME.rstrip("/"), endpoint.rstrip("/"))
    return XetFileData(
        file_hash=file_hash,
        refresh_route=refresh_route,
    )


def parse_xet_connection_info_from_headers(headers: Mapping[str, str]) -> XetConnectionInfo | None:
    """
    Parse XET connection info from the HTTP headers or return None if not found.
    Args:
        headers (`dict`):
           HTTP headers to extract the XET metadata from.
    Returns:
        `XetConnectionInfo` or `None`:
            The information needed to connect to the XET storage service.
            Returns `None` if the headers do not contain the XET connection info.
    """
    try:
        endpoint = headers[constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT]
        access_token = headers[constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN]
        expiration_unix_epoch = int(headers[constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION])
    except (KeyError, ValueError, TypeError):
        return None

    return XetConnectionInfo(
        endpoint=endpoint,
        access_token=access_token,
        expiration_unix_epoch=expiration_unix_epoch,
    )


@validate_hf_hub_args
def refresh_xet_connection_info(
    *,
    file_data: XetFileData,
    headers: dict[str, str],
) -> XetConnectionInfo:
    """
    Utilizes the information in the parsed metadata to request the Hub xet connection information.
    This includes the access token, expiration, and XET service URL.
    Args:
        file_data: (`XetFileData`):
            The file data needed to refresh the xet connection information.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
    Returns:
        `XetConnectionInfo`:
            The connection information needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    if file_data.refresh_route is None:
        raise ValueError("The provided xet metadata does not contain a refresh endpoint.")
    return _fetch_xet_connection_info_with_url(file_data.refresh_route, headers)


@validate_hf_hub_args
def _fetch_xet_connection_info_with_url(
    url: str,
    headers: dict[str, str],
) -> XetConnectionInfo:
    """
    Requests the xet connection info from the supplied URL. This includes the
    access token, expiration time, and endpoint to use for the xet storage service.

    Result is cached to avoid redundant requests.

    Args:
        url: (`str`):
            The access token endpoint URL.
        headers (`dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
    Returns:
        `XetConnectionInfo`:
            The connection information needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    # Check cache first
    cache_key = _cache_key(url, headers)
    cached_info = XET_CONNECTION_INFO_CACHE.get(cache_key)
    if cached_info is not None:
        if not _is_expired(cached_info):
            return cached_info

