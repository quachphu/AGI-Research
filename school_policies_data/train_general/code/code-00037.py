                posix_pipe(stdin, stdout, delim, buf_size, callback, callback_len)
<|endoftext|>"""
IO/concurrency helpers for `tqdm.contrib`.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ..auto import tqdm as tqdm_auto

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['MonoWorker']


class MonoWorker:
    """
    Supports one running task and one waiting task.
    The waiting task is the most recent submitted (others are discarded).
    """
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = deque([], 2)

    def submit(self, func, *args, **kwargs):
        """`func(*args, **kwargs)` may replace currently waiting task."""
        futures = self.futures
        if len(futures) == futures.maxlen:
            running = futures.popleft()
            if not running.done():
                if len(futures):  # clear waiting
                    waiting = futures.pop()
                    waiting.cancel()
                futures.appendleft(running)  # re-insert running
        try:
            waiting = self.pool.submit(func, *args, **kwargs)
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            futures.append(waiting)
            return waiting
<|endoftext|>"""
Helper functionality for interoperability with stdlib `logging`.
"""
import logging
import sys
from contextlib import contextmanager

try:
    from typing import Iterator, List, Optional, Type  # noqa: F401, pylint: disable=unused-import
except ImportError:
    pass

from ..std import tqdm as std_tqdm


class _TqdmLoggingHandler(logging.StreamHandler):
    def __init__(
        self,
        tqdm_class=std_tqdm  # type: Type[std_tqdm]
    ):
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa pylint: disable=bare-except
            self.handleError(record)


def _is_console_logging_handler(handler):
    return (isinstance(handler, logging.StreamHandler)
            and handler.stream in {sys.stdout, sys.stderr})


def _get_first_found_console_logging_handler(handlers):
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler


@contextmanager
def logging_redirect_tqdm(
    loggers=None,  # type: Optional[List[logging.Logger]],
    tqdm_class=std_tqdm  # type: Type[std_tqdm]
):
    # type: (...) -> Iterator[None]
    """
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.stream = orig_handler.stream
                for log_filter in orig_handler.filters:
                    tqdm_handler.addFilter(log_filter)
            logger.handlers = [
                handler for handler in logger.handlers
                if not _is_console_logging_handler(handler)] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers


@contextmanager
def tqdm_logging_redirect(
    *args,
    # loggers=None,  # type: Optional[List[logging.Logger]]
    # tqdm=None,  # type: Optional[Type[tqdm.tqdm]]
    **kwargs
):
    # type: (...) -> Iterator[None]
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    tqdm_kwargs = kwargs.copy()
    loggers = tqdm_kwargs.pop('loggers', None)
    tqdm_class = tqdm_kwargs.pop('tqdm_class', std_tqdm)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
<|endoftext|>"""
Thin wrappers around `concurrent.futures`.
"""
import sys
from contextlib import contextmanager
from operator import length_hint

from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['thread_map', 'process_map', 'interpreter_map']


class _InterpreterLock:
    """Reentrant lock backed by a cross-interpreter queue."""
    from threading import get_ident
    from time import monotonic as _time

    def __init__(self, queue):
        from threading import RLock
        self._queue = queue
        self._lock = RLock()
        self._owner = None
        self._depth = 0

    def acquire(self, blocking=True, timeout=-1):
        from queue import Empty
        start = self._time()
        if timeout == -1:
            acquired = self._lock.acquire(blocking)
        else:
            acquired = self._lock.acquire(blocking, timeout)
        if not acquired:
            return False
        if self._depth:
            self._depth += 1
            return True
        try:
            if not blocking:
                self._queue.get_nowait()
            elif timeout == -1:
                self._queue.get()
            else:
                remaining = max(0, timeout - (self._time() - start))
                self._queue.get(timeout=remaining)
        except Empty:
            self._lock.release()
            return False
        self._owner = self.get_ident()
        self._depth = 1
        return True

    def release(self):
        if self._owner != self.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._depth -= 1
        if not self._depth:
            self._owner = None
            self._queue.put(None)
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


@contextmanager
def ensure_lock(tqdm_class, lock_name="", lock=None):
    """get (create if necessary) and then restore `tqdm_class`'s lock"""
    old_lock = getattr(tqdm_class, '_lock', None)  # don't create a new lock
    if lock is None:
        lock = old_lock or tqdm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tqdm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tqdm_class._lock
    else:
        tqdm_class.set_lock(old_lock)


def _get_interpreter_init(tqdm_class, lock_queue_id):
    """Return an initializer which bootstraps the parent import path and lock."""
    code = (
        "import sys\n"
        f"sys.path[:] = {sys.path!r}\n"
        "from concurrent import interpreters\n"
        "from importlib import import_module\n"
        "from tqdm.contrib.concurrent import _InterpreterLock\n"
        f"tqdm_class = import_module({tqdm_class.__module__!r})\n"
        f"for name in {tqdm_class.__qualname__.split('.')!r}:\n"
        "    tqdm_class = getattr(tqdm_class, name)\n"
        "tqdm_class.monitor_interval = 0\n"
        f"tqdm_class.set_lock(_InterpreterLock(interpreters.Queue({lock_queue_id!r})))")
    return exec, (code,)


def _min_map_len(iterables):
    """min(map(length_hint, iterables), default=0)"""
    return min((n for it in iterables if (n := length_hint(it, -1)) >= 0), default=0)


def _executor_map(
    PoolExecutor, fn, *iterables, max_workers=None, timeout=None, chunksize=1, lock_name="",
    tqdm_class=tqdm_auto, smoothing=0.0, _lock=None, _initializer=None, _initargs=None,
    **tqdm_kwargs
):
    """
    Implementation of `thread_map`, `process_map` and `interpreter_map`.

    Parameters
    ----------
    max_workers  : int
    timeout  : int
    buffersize  : int
        Requires Python>=3.14.
    thread_name_prefix  : str
    max_tasks_per_child  : int
    mp_context  : str
    """
    kwargs = tqdm_kwargs.copy()
    if 'total' not in kwargs:
        kwargs['total'] = _min_map_len(iterables)
    map_kwargs = {}
    if 'buffersize' in kwargs:
        map_kwargs['buffersize'] = kwargs.pop('buffersize')
    pool_kwargs = {}
    for k in ('thread_name_prefix', 'max_tasks_per_child', 'mp_context'):
        if k in kwargs:
            pool_kwargs[k] = kwargs.pop(k)
    dynamic_miniters = None
    if kwargs['total'] and 'miniters' not in kwargs:
        try:
            from os import process_cpu_count as cpu_count
        except ImportError:
            from os import cpu_count
        # thread & process pools have different default workers, but we KISS here
        rough_max = max_workers or min(32, (cpu_count() or 1) + 4)
        if kwargs['total'] > rough_max:
            kwargs['miniters'] = rough_max
            dynamic_miniters = True
    with ensure_lock(tqdm_class, lock_name=lock_name, lock=_lock) as lk:
        # share lock in case workers are already using `tqdm`
        if _initializer is None:
            _initializer = tqdm_class.set_lock
            _initargs = (lk,)
        with PoolExecutor(max_workers=max_workers, initializer=_initializer, initargs=_initargs,
                          **pool_kwargs) as ex:
            with tqdm_class(smoothing=smoothing, **kwargs) as pbar:
                if dynamic_miniters is not None:
                    pbar.dynamic_miniters = True
                orisubmit = ex.submit

                def patchsubmit(*args, **kwargs):
                    fut = orisubmit(*args, **kwargs)
                    fut.add_done_callback(lambda _: pbar.update())
                    return fut
                ex.submit = patchsubmit
                return list(ex.map(
                    fn, *iterables, timeout=timeout, chunksize=chunksize, **map_kwargs))


def thread_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to `concurrent.futures.ThreadPoolExecutor`.
    thread_name_prefix  : str, optional
        Passed to `concurrent.futures.ThreadPoolExecutor` [default: ''].
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    buffersize  : int, optional
        Requires Python>=3.14 [default: None].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    smoothing  : float, optional
        Passed to `tqdm_class`; the [default: 0] is average (due to erratic update frequency).
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: ''].
    """
    from concurrent.futures import ThreadPoolExecutor
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)


def interpreter_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.InterpreterPoolExecutor` (Python 3.14+).

    Parameters
    ----------
    Same as `thread_map`.

    Notes
    -----
    `fn`, its arguments, and its return values must be pickleable.
    Worker progress bars using the same `tqdm_class` share a cross-interpreter write lock.
    """
    from concurrent import interpreters
    from concurrent.futures import InterpreterPoolExecutor
    lock_queue = interpreters.create_queue()
    lock_queue.put(None)
    tqdm_class = tqdm_kwargs.get("tqdm_class", tqdm_auto)
    initializer, initargs = _get_interpreter_init(tqdm_class, lock_queue.id)
    return _executor_map(
        InterpreterPoolExecutor, fn, *iterables, _lock=_InterpreterLock(lock_queue),
        _initializer=initializer, _initargs=initargs, **tqdm_kwargs)


def process_map(fn, *iterables, lock_name="mp_lock", **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to `concurrent.futures.ProcessPoolExecutor`.
    timeout  : int or float, optional
        Seconds to wait before raising `TimeoutError` if `__next__` is called and the
        result isn't available. [default: None].
    chunksize  : int, optional
        Approximate size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    buffersize  : int, optional
        Requires Python>=3.14 [default: None].
    max_tasks_per_child  : int, optional
        Maximum number of tasks a worker process can complete before being replaced
        with a new process; passed to `concurrent.futures.ProcessPoolExecutor`.
    mp_context  : multiprocessing.BaseContext, optional
        Multiprocessing context to use, e.g. `multiprocessing.get_context('fork')`.
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    smoothing  : float, optional
        Passed to `tqdm_class`; the [default: 0] is average (due to erratic update frequency).
    """
    from concurrent.futures import ProcessPoolExecutor
    if iterables and 'chunksize' not in tqdm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        shortest_iterable_len = _min_map_len(iterables)
        if shortest_iterable_len > 1000:
            from warnings import warn
            warn("Iterable length %d > 1000 but `chunksize` is not set."
                 " This may seriously degrade multiprocess performance."
                 " Set `chunksize=1` or more." % shortest_iterable_len,
                 TqdmWarning, stacklevel=2)
    return _executor_map(ProcessPoolExecutor, fn, *iterables, lock_name=lock_name, **tqdm_kwargs)
<|endoftext|>"""
Sends updates to a Discord bot.

Usage:
>>> from tqdm.contrib.discord import tqdm, trange
>>> for i in trange(10, token='{token}', channel_id='{channel_id}'):
...     ...

![screenshot](https://tqdm.github.io/img/screenshot-discord.png)
"""
from warnings import warn

from requests import Session
from requests.utils import default_user_agent

from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
from ..utils import envwrap
from ..version import __version__
from .utils_worker import MonoWorker

__author__ = {"github.com/": ["casperdcl", "guigoruiz1"]}
__all__ = ['DiscordIO', 'tqdm_discord', 'tdrange', 'tqdm', 'trange']


class DiscordIO(MonoWorker):
    """Non-blocking file-like IO using a Discord Bot."""
    API = 'https://discord.com/api/v10'
    UA = f"tqdm (https://tqdm.github.io, {__version__}) {default_user_agent()}"

    def __init__(self, token, channel_id):
        """Creates a new message in the given `channel_id`."""
        super().__init__()
        self.token = token
        self.channel_id = channel_id
        self.session = Session()
        self.text = self.__class__.__name__
        self.message_id  # pylint: disable=pointless-statement

    @property
    def message_id(self):
        if hasattr(self, '_message_id'):
            return self._message_id  # pylint: disable=access-member-before-definition
        try:
            req = self.session.post(
                f'{self.API}/channels/{self.channel_id}/messages',
                headers={'Authorization': f'Bot {self.token}', 'User-Agent': self.UA},
                json={'content': f"`{self.text}`"})
            res = req.json()
            req.raise_for_status()
        except Exception as e:
            if req.status_code == 429:
                warn("Creation rate limit: try increasing `mininterval`.",
                     TqdmWarning, stacklevel=2)
            else:
                tqdm_auto.write(str(e))
        else:
            self._message_id = res['id']
            return self._message_id

    def write(self, s):
        """Replaces internal `message_id`'s text with `s`."""
        if not s:
            s = "..."
        s = s.replace('\r', '').strip()
        if s == self.text:
            return  # avoid duplicate message Bot error
        message_id = self.message_id
        if message_id is None:
            return
        self.text = s
        try:
            future = self.submit(
                self.session.patch,
                f'{self.API}/channels/{self.channel_id}/messages/{message_id}',
                headers={'Authorization': f'Bot {self.token}', 'User-Agent': self.UA},
                json={'content': f"`{self.text}`"})
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future

    def delete(self):
        """Deletes internal `message_id`."""
        try:
            future = self.submit(
                self.session.delete,
                f'{self.API}/channels/{self.channel_id}/messages/{self.message_id}',
                headers={'Authorization': f'Bot {self.token}', 'User-Agent': self.UA})
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future


class tqdm_discord(tqdm_auto):  # pylint: disable=inconsistent-mro
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Discord Bot.
    May take a few seconds to create (`__init__`).

    - create a discord bot (not public, no requirement of OAuth2 code
      grant, only send message permissions) & invite it to a channel:
      <https://discordpy.readthedocs.io/en/latest/discord.html>
    - copy the bot `{token}` & `{channel_id}` and paste below

    >>> from tqdm.contrib.discord import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', channel_id='{channel_id}'):
    ...     ...
    """
    @envwrap("tqdm", "discord", is_method=True)
    def __init__(self, *args, token=None, channel_id=None, **kwargs):
        """
        Parameters
        ----------
        token  : str, required. Discord bot token
            [default: ${TQDM_DISCORD_TOKEN}].
        channel_id  : int, required. Discord channel ID
            [default: ${TQDM_DISCORD_CHANNEL_ID}].

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()
            self.dio = DiscordIO(token, channel_id)
        super().__init__(*args, **kwargs)

    def display(self, **kwargs):  # pylint: disable=arguments-differ
        super().display(**kwargs)
        fmt = self.format_dict
        if fmt.get('bar_format', None):
            fmt['bar_format'] = fmt['bar_format'].replace(
                '<bar/>', '{bar:10u}').replace('{bar}', '{bar:10u}')
        self.dio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super().clear(*args, **kwargs)
        if not self.disable:
            self.dio.write("")

    def close(self):
        if self.disable:
            return
        super().close()
        if not (self.leave or (self.leave is None and self.pos == 0)):
            self.dio.delete()


def tdrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.discord.tqdm(range(*args), **kwargs)`."""
    return tqdm_discord(range(*args), **kwargs)


# Aliases
tqdm = tqdm_discord
trange = tdrange
<|endoftext|>"""
Sends updates to a Telegram bot.

Usage:
>>> from tqdm.contrib.telegram import tqdm, trange
>>> for i in trange(10, token='{token}', chat_id='{chat_id}'):
...     ...

![screenshot](https://tqdm.github.io/img/screenshot-telegram.gif)
"""
from warnings import warn

from requests import Session

from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
from ..utils import envwrap
from .utils_worker import MonoWorker

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['TelegramIO', 'tqdm_telegram', 'ttgrange', 'tqdm', 'trange']


class TelegramIO(MonoWorker):
    """Non-blocking file-like IO using a Telegram Bot."""
    API = 'https://api.telegram.org/bot'

    def __init__(self, token, chat_id):
        """Creates a new message in the given `chat_id`."""
        super().__init__()
        self.token = token
        self.chat_id = chat_id
        self.session = Session()
        self.text = self.__class__.__name__
        self.message_id  # pylint: disable=pointless-statement

    @property
    def message_id(self):
        if hasattr(self, '_message_id'):
            return self._message_id  # pylint: disable=access-member-before-definition
        try:
            req = self.session.post(
                f'{self.API}{self.token}/sendMessage',
                data={'text': f"`{self.text}`", 'chat_id': self.chat_id,
                      'parse_mode': 'MarkdownV2'})
            res = req.json()
            req.raise_for_status()
        except Exception as e:
            if req.status_code == 429:
                warn("Creation rate limit: try increasing `mininterval`.",
                     TqdmWarning, stacklevel=2)
            else:
                tqdm_auto.write(str(e))
        else:
            self._message_id = res['result']['message_id']
            return self._message_id

    def write(self, s):
        """Replaces internal `message_id`'s text with `s`."""
        if not s:
            s = "..."
        s = s.replace('\r', '').strip()
        if s == self.text:
            return  # avoid duplicate message Bot error
        message_id = self.message_id
        if message_id is None:
            return
        self.text = s
        try:
            future = self.submit(
                self.session.post, f'{self.API}{self.token}/editMessageText',
                data={'text': f"`{s}`", 'chat_id': self.chat_id,
                      'message_id': message_id, 'parse_mode': 'MarkdownV2'})
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future

    def delete(self):
        """Deletes internal `message_id`."""
        try:
            future = self.submit(
                self.session.post, '{self.API}{self.token}/deleteMessage',
                data={'chat_id': self.chat_id, 'message_id': self.message_id})
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future


class tqdm_telegram(tqdm_auto):  # pylint: disable=inconsistent-mro
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Telegram Bot.
    May take a few seconds to create (`__init__`).

    - create a bot <https://core.telegram.org/bots#6-botfather>
    - copy its `{token}`
    - add the bot to a chat and send it a message such as `/start`
    - go to <https://api.telegram.org/bot`{token}`/getUpdates> to find out
      the `{chat_id}`
    - paste the `{token}` & `{chat_id}` below

    >>> from tqdm.contrib.telegram import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', chat_id='{chat_id}'):
    ...     ...
    """
    @envwrap("tqdm", "telegram", is_method=True)
    def __init__(self, *args, token=None, chat_id=None, **kwargs):
        """
        Parameters
        ----------
        token  : str, required. Telegram token
            [default: ${TQDM_TELEGRAM_TOKEN}].
        chat_id  : str, required. Telegram chat ID
            [default: ${TQDM_TELEGRAM_CHAT_ID}].

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()
            self.tgio = TelegramIO(token, chat_id)
        super().__init__(*args, **kwargs)

    def display(self, **kwargs):  # pylint: disable=arguments-differ
        super().display(**kwargs)
        fmt = self.format_dict
        if fmt.get('bar_format', None):
            fmt['bar_format'] = fmt['bar_format'].replace(
                '<bar/>', '{bar:10u}').replace('{bar}', '{bar:10u}')
        self.tgio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super().clear(*args, **kwargs)
        if not self.disable:
            self.tgio.write("")

    def close(self):
        if self.disable:
            return
        super().close()
        if not (self.leave or (self.leave is None and self.pos == 0)):
            self.tgio.delete()


def ttgrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.telegram.tqdm(range(*args), **kwargs)`."""
    return tqdm_telegram(range(*args), **kwargs)


# Aliases
tqdm = tqdm_telegram
trange = ttgrange
<|endoftext|>"""
Thin wrappers around common functions.

Subpackages contain potentially unstable extensions.
"""
from warnings import warn

from ..auto import tqdm as tqdm_auto
from ..std import TqdmDeprecationWarning, tqdm
from ..utils import ObjectWrapper

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['tenumerate', 'tzip', 'tmap']


class DummyTqdmFile(ObjectWrapper):
    """Dummy file-like that will write to tqdm"""

    def __init__(self, wrapped):
        super().__init__(wrapped)
        self._buf = []

    def write(self, x, nolock=False):
        nl = b"\n" if isinstance(x, bytes) else "\n"
        pre, sep, post = x.rpartition(nl)
        if sep:
            blank = type(nl)()
            tqdm.write(blank.join(self._buf + [pre, sep]),
                       end=blank, file=self._wrapped, nolock=nolock)
            self._buf = [post]
        else:
            self._buf.append(x)

    def __del__(self):
        if self._buf:
            blank = type(self._buf[0])()
            try:
                tqdm.write(blank.join(self._buf), end=blank, file=self._wrapped)
            except (OSError, ValueError):
                pass


def builtin_iterable(func):
    """Returns `func`"""
    warn("This function has no effect, and will be removed in tqdm==5.0.0",
         TqdmDeprecationWarning, stacklevel=2)
    return func


def tenumerate(iterable, start=0, total=None, tqdm_class=tqdm_auto, **tqdm_kwargs):
    """
    Equivalent of `numpy.ndenumerate` or builtin `enumerate`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(iterable, np.ndarray):
            return tqdm_class(np.ndenumerate(iterable), total=total or iterable.size,
                              **tqdm_kwargs)
    return enumerate(tqdm_class(iterable, total=total, **tqdm_kwargs), start)


def tzip(iter1, *iter2plus, **tqdm_kwargs):
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    kwargs = tqdm_kwargs.copy()
    tqdm_class = kwargs.pop("tqdm_class", tqdm_auto)
    yield from zip(tqdm_class(iter1, **kwargs), *iter2plus)


def tmap(function, *sequences, **tqdm_kwargs):
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    for i in tzip(*sequences, **tqdm_kwargs):
        yield function(*i)
<|endoftext|>"""
Sends updates to a Slack app.

Usage:
>>> from tqdm.contrib.slack import tqdm, trange
>>> for i in trange(10, token='{token}', channel='{channel}'):
...     ...

![screenshot](https://tqdm.github.io/img/screenshot-slack.png)
"""
import logging

try:
    from slack_sdk import WebClient
except ImportError:
    raise ImportError("Please `pip install slack-sdk`")

from ..auto import tqdm as tqdm_auto
from ..utils import envwrap
from .utils_worker import MonoWorker

__author__ = {"github.com/": ["0x2b3bfa0", "casperdcl"]}
__all__ = ['SlackIO', 'tqdm_slack', 'tsrange', 'tqdm', 'trange']


class SlackIO(MonoWorker):
    """Non-blocking file-like IO using a Slack app."""
    def __init__(self, token, channel):
        """Creates a new message in the given `channel`."""
        super().__init__()
        self.client = WebClient(token=token)
        self.text = self.__class__.__name__
        try:
            self.message = self.client.chat_postMessage(channel=channel, text=self.text)
        except Exception as e:
            tqdm_auto.write(str(e))
            self.message = None

    def write(self, s):
        """Replaces internal `message`'s text with `s`."""
        if not s:
            s = "..."
        s = s.replace('\r', '').strip()
        if s == self.text:
            return  # skip duplicate message
        message = self.message
        if message is None:
            return
        self.text = s
        try:
            future = self.submit(self.client.chat_update, channel=message['channel'],
                                 ts=message['ts'], text='`' + s + '`')
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future


class tqdm_slack(tqdm_auto):  # pylint: disable=inconsistent-mro
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Slack app.
    May take a few seconds to create (`__init__`).

    - create a Slack app with the `chat:write` scope & invite it to a
      channel: <https://api.slack.com/authentication/basics>
    - copy the bot `{token}` & `{channel}` and paste below
    >>> from tqdm.contrib.slack import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', channel='{channel}'):
    ...     ...
    """
    @envwrap("tqdm", "slack", is_method=True)
    def __init__(self, *args, token=None, channel=None, **kwargs):
        """
        Parameters
        ----------
        token  : str, required. Slack token
            [default: ${TQDM_SLACK_TOKEN}].
        channel  : int, required. Slack channel
            [default: ${TQDM_SLACK_CHANNEL}].
        mininterval  : float, optional.
          Minimum of [default: 1.5] to avoid rate limit.

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        if not kwargs.get('disable'):
            kwargs = kwargs.copy()
            logging.getLogger("HTTPClient").setLevel(logging.WARNING)
            self.sio = SlackIO(token, channel)
            kwargs['mininterval'] = max(1.5, kwargs.get('mininterval', 1.5))
        super().__init__(*args, **kwargs)

    def display(self, **kwargs):  # pylint: disable=arguments-differ
        super().display(**kwargs)
        fmt = self.format_dict
        if fmt.get('bar_format', None):
            fmt['bar_format'] = fmt['bar_format'].replace(
                '<bar/>', '`{bar:10}`').replace('{bar}', '`{bar:10u}`')
        elif self.total:
            fmt['bar_format'] = '{l_bar}`{bar:10}`{r_bar}'
        if fmt['ascii'] is False:
            fmt['ascii'] = [":black_square:", ":small_blue_diamond:", ":large_blue_diamond:",
                            ":large_blue_square:"]
            fmt['ncols'] = 336
        self.sio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super().clear(*args, **kwargs)
        if not self.disable:
            self.sio.write("")


def tsrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.slack.tqdm(range(*args), **kwargs)`."""
    return tqdm_slack(range(*args), **kwargs)


# Aliases
tqdm = tqdm_slack
trange = tsrange
<|endoftext|>"""
Thin wrappers around `itertools`.
"""
import itertools
import math

from ..auto import tqdm as tqdm_auto

__author__ = {"github.com/": ["casperdcl"]}
__all__ = [
    'chain', 'product', 'permutations', 'combinations', 'combinations_with_replacement', 'batched']


def chain(*iterables, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.chain`."""
    if total is None:
        try:
            total = sum(map(len, iterables))
        except (TypeError, AttributeError):
            pass
    return tqdm_class(itertools.chain(*iterables), total=total, **kwargs)


def product(*iterables, repeat=1, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.product`."""
    if total is None:
        try:
            lens = list(map(len, iterables))
        except (TypeError, AttributeError):
            pass
        else:
            total = math.prod(lens) ** repeat
    yield from tqdm_class(itertools.product(*iterables, repeat=repeat), total=total, **kwargs)


def permutations(iterable, r=None, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.permutations`."""
    if total is None:
        try:
            n = len(iterable)
        except (TypeError, AttributeError):
            pass
        else:
            r = n if r is None else r
            if r > n:
                total = 0
            else:
                total = math.perm(n, r)
    return tqdm_class(itertools.permutations(iterable, r), total=total, **kwargs)


def combinations(iterable, r, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.combinations`."""
    if total is None:
        try:
            n = len(iterable)
        except (TypeError, AttributeError):
            pass
        else:
            if r > n:
                total = 0
            else:
                total = math.comb(n, r)
    return tqdm_class(itertools.combinations(iterable, r), total=total, **kwargs)


def combinations_with_replacement(iterable, r, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.combinations_with_replacement`."""
    if total is None:
        try:
            n = len(iterable)
        except (TypeError, AttributeError):
            pass
        else:
            total = 1
            for i in range(n+r-1, n-1, -1):
                total *= i
            for i in range(1, r+1):
                total //= i
    return tqdm_class(itertools.combinations_with_replacement(iterable, r), total=total, **kwargs)


def batched(iterable, n, total=None, tqdm_class=tqdm_auto, **kwargs):
    """Equivalent of `itertools.batched`."""
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            pass
    return tqdm_class(itertools.batched(iterable, n), unit_scale=n,
                      total=(total+n-1) // n if total is not None else None,
                      **kwargs)
<|endoftext|>"""
Even more features than `tqdm.auto` (all the bells & whistles):

- `tqdm.auto`
- `tqdm.tqdm.pandas`
- `tqdm.contrib.slack`
    + uses `${TQDM_SLACK_TOKEN}` and `${TQDM_SLACK_CHANNEL}`
- `tqdm.contrib.telegram`
    + uses `${TQDM_TELEGRAM_TOKEN}` and `${TQDM_TELEGRAM_CHAT_ID}`
- `tqdm.contrib.discord`
    + uses `${TQDM_DISCORD_TOKEN}` and `${TQDM_DISCORD_CHANNEL_ID}`
"""
__all__ = ['tqdm', 'trange']
import warnings
from os import getenv

if getenv("TQDM_SLACK_TOKEN") and getenv("TQDM_SLACK_CHANNEL"):
    from .slack import tqdm, trange
elif getenv("TQDM_TELEGRAM_TOKEN") and getenv("TQDM_TELEGRAM_CHAT_ID"):
    from .telegram import tqdm, trange
elif getenv("TQDM_DISCORD_TOKEN") and getenv("TQDM_DISCORD_CHANNEL_ID"):
    from .discord import tqdm, trange
else:
    from ..auto import tqdm, trange

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    tqdm.pandas()
<|endoftext|>from __future__ import annotations

import dataclasses
import socket
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, contextmanager
from inspect import isasyncgenfunction, iscoroutinefunction, ismethod
from typing import TYPE_CHECKING, Any, cast

import pytest
from _pytest.fixtures import FuncFixtureInfo, SubRequest
from _pytest.outcomes import Exit
from _pytest.scope import Scope

from . import get_available_backends
from ._core._eventloop import (
    current_async_library,
    get_async_backend,
    reset_current_async_library,
    set_current_async_library,
)
from ._core._exceptions import iterate_exceptions

if TYPE_CHECKING:
    from .abc import TestRunner

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

if TYPE_CHECKING:
    # pytest >= 9.2 keeps CallSpec2 as a TYPE_CHECKING-only alias of CallSpec
    from _pytest.python import CallSpec2 as CallSpec
else:
    try:
        # c.f. https://github.com/pytest-dev/pytest/pull/14742
        # pytest >= 9.2
        from _pytest.python import CallSpec
    except ImportError:
        from _pytest.python import CallSpec2 as CallSpec

_current_runner: TestRunner | None = None
_runner_stack: ExitStack | None = None
_runner_leases = 0


def extract_backend_and_options(backend: object) -> tuple[str, dict[str, Any]]:
    if isinstance(backend, str):
        return backend, {}
    elif isinstance(backend, tuple) and len(backend) == 2:
        if isinstance(backend[0], str) and isinstance(backend[1], dict):
            return cast(tuple[str, dict[str, Any]], backend)

    raise TypeError("anyio_backend must be either a string or tuple of (string, dict)")


@contextmanager
def get_runner(
    backend_name: str, backend_options: dict[str, Any]
) -> Iterator[TestRunner]:
    global _current_runner, _runner_leases, _runner_stack
    if _current_runner is None:
        asynclib = get_async_backend(backend_name)
        _runner_stack = ExitStack()
        if current_async_library() is None:
            # Since we're in control of the event loop, we can cache the name of the
            # async library
            token = set_current_async_library(backend_name)
            _runner_stack.callback(reset_current_async_library, token)

        backend_options = backend_options or {}
        _current_runner = _runner_stack.enter_context(
            asynclib.create_test_runner(backend_options)
        )

    _runner_leases += 1
    try:
        yield _current_runner
    finally:
        _runner_leases -= 1
        if not _runner_leases:
            assert _runner_stack is not None
            _runner_stack.close()
            _runner_stack = _current_runner = None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("anyio")
    parser.addini(
        "anyio_mode",
        default="strict",
        help='AnyIO plugin mode (either "strict" or "auto")',
    )
    group.addoption(
        "--anyio-mode",
        default=None,
        dest="anyio_mode",
        help="""
        'auto'   - All async test functions will be handled by AnyIO pytest plugin
        'strict' - Disabling autoprocessing(useful when anyio tests
            need to coexist with other async test plugins)
        """,
    )


def _is_auto_mode(config: pytest.Config, mode: str) -> bool:
    return (
        config.getoption(mode, default=None) == "auto" or config.getini(mode) == "auto"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "anyio: mark the (coroutine function) test to be run asynchronously via anyio.",
    )
    if (
        _is_auto_mode(config, "anyio_mode")
        and config.pluginmanager.has_plugin("asyncio")
        and _is_auto_mode(config, "asyncio_mode")
    ):
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(
                "AnyIO auto mode has been enabled together with pytest-asyncio auto "
                "mode. This may cause unexpected behavior."
            ),
            1,
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Generator[Any]:
    def wrapper(anyio_backend: Any, request: SubRequest, **kwargs: Any) -> Any:
        # Rebind any fixture methods to the request instance
        if (
            request.instance
            and ismethod(func)
            and type(func.__self__) is type(request.instance)
        ):
            local_func = func.__func__.__get__(request.instance)
        else:
            local_func = func

        backend_name, backend_options = extract_backend_and_options(anyio_backend)
        if has_backend_arg:
            kwargs["anyio_backend"] = anyio_backend

        if has_request_arg:
            kwargs["request"] = request

        with get_runner(backend_name, backend_options) as runner:
            # re-entrant call into the test runner detected. this happens when an async fixture
            # is dynamically requested via request.getfixturevalue() from inside a running async
            # test or fixture. on asyncio this raises RuntimeError: This event loop is already
            # running, on trio the runner deadlocks - the host loop blocks waiting for the
            # coroutine to return, but the coroutine is waiting for the host loop. raising here
            # prevents the hang and gives a consistent error across backends.
            if runner.is_running():
                raise RuntimeError(
                    "Cannot schedule a coroutine in the test runner while another is already running; "
                    "likely caused by request.getfixturevalue() on an async fixture."
                )

            if isasyncgenfunction(local_func):
                yield from runner.run_asyncgen_fixture(local_func, kwargs)
            else:
                yield runner.run_fixture(local_func, kwargs)

    # Only apply this to coroutine functions and async generator functions in requests
    # that involve the anyio_backend fixture
    func = fixturedef.func
    if isasyncgenfunction(func) or iscoroutinefunction(func):
        if "anyio_backend" in request.fixturenames:
            fixturedef.func = wrapper
            original_argname = fixturedef.argnames

            if not (has_backend_arg := "anyio_backend" in fixturedef.argnames):
                fixturedef.argnames += ("anyio_backend",)

            if not (has_request_arg := "request" in fixturedef.argnames):
                fixturedef.argnames += ("request",)

            try:
                return (yield)
            finally:
                fixturedef.func = func
                fixturedef.argnames = original_argname

    return (yield)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makeitem(
    collector: pytest.Module | pytest.Class, name: str, obj: object
) -> None:
    if collector.istestfunction(obj, name):
        inner_func = obj.hypothesis.inner_test if hasattr(obj, "hypothesis") else obj
        if iscoroutinefunction(inner_func):
            anyio_auto_mode = _is_auto_mode(collector.config, "anyio_mode")
            marker = collector.get_closest_marker("anyio")
            own_markers = getattr(obj, "pytestmark", ())
            if (
                anyio_auto_mode
                or marker
                or any(marker.name == "anyio" for marker in own_markers)
            ):
                pytest.mark.usefixtures("anyio_backend")(obj)


def pytest_collection_finish(session: pytest.Session) -> None:
    for i, item in reversed(list(enumerate(session.items))):
        if (
            isinstance(item, pytest.Function)
            and iscoroutinefunction(item.function)
            and item.get_closest_marker("anyio") is not None
            and "anyio_backend" not in item.fixturenames
        ):
            new_items = []
            try:
                cs_fields = {f.name for f in dataclasses.fields(CallSpec)}
            except TypeError:
                cs_fields = set()

            for param_index, backend in enumerate(get_available_backends()):
                if "_arg2scope" in cs_fields:  # pytest >= 8
                    callspec = CallSpec(
                        params={"anyio_backend": backend},
                        indices={"anyio_backend": param_index},
                        _arg2scope={"anyio_backend": Scope.Module},
                        _idlist=[backend],
                        marks=[],
                    )
                else:  # pytest 7.x
                    callspec = CallSpec(  # type: ignore[call-arg]
                        funcargs={},
                        params={"anyio_backend": backend},
                        indices={"anyio_backend": param_index},
                        arg2scope={"anyio_backend": Scope.Module},
                        idlist=[backend],
                        marks=[],
                    )

                fi = item._fixtureinfo
                new_names_closure = list(fi.names_closure)
                if "anyio_backend" not in new_names_closure:
                    new_names_closure.append("anyio_backend")

                new_fixtureinfo = FuncFixtureInfo(
                    argnames=fi.argnames,
                    initialnames=fi.initialnames,
                    names_closure=new_names_closure,
                    name2fixturedefs=fi.name2fixturedefs,
                )
                new_item = pytest.Function.from_parent(
                    item.parent,
                    name=f"{item.originalname}[{backend}]",
                    callspec=callspec,
                    callobj=item.obj,
                    fixtureinfo=new_fixtureinfo,
                    keywords=item.keywords,
                    originalname=item.originalname,
                )
                new_items.append(new_item)

            session.items[i : i + 1] = new_items


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: Any) -> bool | None:
    def run_with_hypothesis(**kwargs: Any) -> None:
        with get_runner(backend_name, backend_options) as runner:
            runner.run_test(original_func, kwargs)

    backend = pyfuncitem.funcargs.get("anyio_backend")
    if backend:
        backend_name, backend_options = extract_backend_and_options(backend)

        if hasattr(pyfuncitem.obj, "hypothesis"):
            # Wrap the inner test function unless it's already wrapped
            original_func = pyfuncitem.obj.hypothesis.inner_test
            if original_func.__qualname__ != run_with_hypothesis.__qualname__:
                if iscoroutinefunction(original_func):
                    pyfuncitem.obj.hypothesis.inner_test = run_with_hypothesis

            return None

        if iscoroutinefunction(pyfuncitem.obj):
            funcargs = pyfuncitem.funcargs
            testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
            with get_runner(backend_name, backend_options) as runner:
                try:
                    runner.run_test(pyfuncitem.obj, testargs)
                except ExceptionGroup as excgrp:
                    for exc in iterate_exceptions(excgrp):
                        if isinstance(exc, (Exit, KeyboardInterrupt, SystemExit)):
                            raise exc from excgrp

                    raise

            return True

    return None


@pytest.fixture(scope="module", params=get_available_backends())
def anyio_backend(request: Any) -> Any:
    return request.param


@pytest.fixture
def anyio_backend_name(anyio_backend: Any) -> str:
    if isinstance(anyio_backend, str):
        return anyio_backend
    else:
        return anyio_backend[0]


@pytest.fixture
def anyio_backend_options(anyio_backend: Any) -> dict[str, Any]:
    if isinstance(anyio_backend, str):
        return {}
    else:
        return anyio_backend[1]


class FreePortFactory:
    """
    Manages port generation based on specified socket kind, ensuring no duplicate
    ports are generated.

    This class provides functionality for generating available free ports on the
    system. It is initialized with a specific socket kind and can generate ports
    for given address families while avoiding reuse of previously generated ports.

    Users should not instantiate this class directly, but use the
    ``free_tcp_port_factory`` and ``free_udp_port_factory`` fixtures instead. For simple
    uses cases, ``free_tcp_port`` and ``free_udp_port`` can be used instead.
    """

    def __init__(self, kind: socket.SocketKind) -> None:
        self._kind = kind
        self._generated = set[int]()

    @property
    def kind(self) -> socket.SocketKind:
        """
        The type of socket connection (e.g., :data:`~socket.SOCK_STREAM` or
        :data:`~socket.SOCK_DGRAM`) used to bind for checking port availability

        """
        return self._kind

    def __call__(self, family: socket.AddressFamily | None = None) -> int:
        """
        Return an unbound port for the given address family.

        :param family: if omitted, both IPv4 and IPv6 addresses will be tried
        :return: a port number

        """
        if family is not None:
            families = [family]
        else:
            families = [socket.AF_INET]
            if socket.has_ipv6:
                families.append(socket.AF_INET6)

        while True:
            port = 0
            with ExitStack() as stack:
                for fam in families:
                    sock = stack.enter_context(socket.socket(fam, self._kind))
                    addr = "::1" if fam == socket.AF_INET6 else "127.0.0.1"
                    try:
                        sock.bind((addr, port))
                    except OSError:
                        break

                    if not port:
                        port = sock.getsockname()[1]
                else:
                    if port not in self._generated:
                        self._generated.add(port)
                        return port


@pytest.fixture(scope="session")
def free_tcp_port_factory() -> FreePortFactory:
    return FreePortFactory(socket.SOCK_STREAM)


@pytest.fixture(scope="session")
def free_udp_port_factory() -> FreePortFactory:
    return FreePortFactory(socket.SOCK_DGRAM)


@pytest.fixture
def free_tcp_port(free_tcp_port_factory: Callable[[], int]) -> int:
    return free_tcp_port_factory()


@pytest.fixture
def free_udp_port(free_udp_port_factory: Callable[[], int]) -> int:
    return free_udp_port_factory()
<|endoftext|>from __future__ import annotations

__all__ = (
    "current_default_interpreter_limiter",
    "run_sync",
)

import atexit
import os
import sys
from collections import deque
from collections.abc import Callable
from typing import Any, Final, TypeVar

from . import current_time, to_thread
from ._core._exceptions import BrokenWorkerInterpreter
from ._core._synchronization import CapacityLimiter
from .lowlevel import RunVar

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

if sys.version_info >= (3, 14):
    from concurrent.interpreters import ExecutionFailed, create

    def _interp_call(
        func: Callable[..., Any], args: tuple[Any, ...]
    ) -> tuple[Any, bool]:
        try:
            retval = func(*args)
        except BaseException as exc:
            return exc, True
        else:
            return retval, False

    class _Worker:
        last_used: float = 0

        def __init__(self) -> None:
            self._interpreter = create()

        def destroy(self) -> None:
            self._interpreter.close()

        def call(
            self,
            func: Callable[..., T_Retval],
            args: tuple[Any, ...],
        ) -> T_Retval:
            try:
                res, is_exception = self._interpreter.call(_interp_call, func, args)
            except ExecutionFailed as exc:
                raise BrokenWorkerInterpreter(exc.excinfo) from exc

            if is_exception:
                raise res

            return res
elif sys.version_info >= (3, 13):
    import _interpqueues
    import _interpreters

    UNBOUND: Final = 2  # I have no clue how this works, but it was used in the stdlib
    FMT_UNPICKLED: Final = 0
    FMT_PICKLED: Final = 1
    QUEUE_PICKLE_ARGS: Final = (FMT_PICKLED, UNBOUND)
    QUEUE_UNPICKLE_ARGS: Final = (FMT_UNPICKLED, UNBOUND)

    _run_func = compile(
        """
import _interpqueues
from _interpreters import NotShareableError
from pickle import loads, dumps, HIGHEST_PROTOCOL

QUEUE_PICKLE_ARGS = (1, 2)
QUEUE_UNPICKLE_ARGS = (0, 2)

item = _interpqueues.get(queue_id)[0]
try:
    func, args = loads(item)
    retval = func(*args)
except BaseException as exc:
    is_exception = True
    retval = exc
else:
    is_exception = False

try:
    _interpqueues.put(queue_id, (retval, is_exception), *QUEUE_UNPICKLE_ARGS)
except NotShareableError:
    retval = dumps(retval, HIGHEST_PROTOCOL)
    _interpqueues.put(queue_id, (retval, is_exception), *QUEUE_PICKLE_ARGS)
    """,
        "<string>",
        "exec",
    )

    class _Worker:
        last_used: float = 0

        def __init__(self) -> None:
            self._interpreter_id = _interpreters.create()
            self._queue_id = _interpqueues.create(1, *QUEUE_UNPICKLE_ARGS)
            _interpreters.set___main___attrs(
                self._interpreter_id, {"queue_id": self._queue_id}
            )

        def destroy(self) -> None:
            _interpqueues.destroy(self._queue_id)
            _interpreters.destroy(self._interpreter_id)

        def call(
            self,
            func: Callable[..., T_Retval],
            args: tuple[Any, ...],
        ) -> T_Retval:
            import pickle

            item = pickle.dumps((func, args), pickle.HIGHEST_PROTOCOL)
            _interpqueues.put(self._queue_id, item, *QUEUE_PICKLE_ARGS)
            exc_info = _interpreters.exec(self._interpreter_id, _run_func)
            if exc_info:
                raise BrokenWorkerInterpreter(exc_info)

            res = _interpqueues.get(self._queue_id)
            (res, is_exception), fmt = res[:2]
            if fmt == FMT_PICKLED:
                res = pickle.loads(res)

            if is_exception:
                raise res

            return res
else:

    class _Worker:
        last_used: float = 0

        def __init__(self) -> None:
            raise RuntimeError("subinterpreters require at least Python 3.13")

        def call(
            self,
            func: Callable[..., T_Retval],
            args: tuple[Any, ...],
        ) -> T_Retval:
            raise NotImplementedError

        def destroy(self) -> None:
            pass


DEFAULT_CPU_COUNT: Final = 8  # this is just an arbitrarily selected value
MAX_WORKER_IDLE_TIME = (
    30  # seconds a subinterpreter can be idle before becoming eligible for pruning
)

T_Retval = TypeVar("T_Retval")
PosArgsT = TypeVarTuple("PosArgsT")

_idle_workers = RunVar[deque[_Worker]]("_available_workers")
_default_interpreter_limiter = RunVar[CapacityLimiter]("_default_interpreter_limiter")


def _stop_workers(workers: deque[_Worker]) -> None:
    for worker in workers:
        worker.destroy()

    workers.clear()


async def run_sync(
    func: Callable[[Unpack[PosArgsT]], T_Retval],
    *args: Unpack[PosArgsT],
    limiter: CapacityLimiter | None = None,
) -> T_Retval:
    """
    Call the given function with the given arguments in a subinterpreter.

    .. warning:: On Python 3.13, the :mod:`concurrent.interpreters` module was not yet
        available, so the code path for that Python version relies on an undocumented,
        private API. As such, it is recommended to not rely on this function for anything
        mission-critical on Python 3.13.

    :param func: a callable
    :param args: the positional arguments for the callable
    :param limiter: capacity limiter to use to limit the total number of subinterpreters
        running (if omitted, the default limiter is used)
    :return: the result of the call
    :raises BrokenWorkerInterpreter: if there's an internal error in a subinterpreter

    """
    if limiter is None:
        limiter = current_default_interpreter_limiter()

    try:
        idle_workers = _idle_workers.get()
    except LookupError:
        idle_workers = deque()
        _idle_workers.set(idle_workers)
        atexit.register(_stop_workers, idle_workers)

    async with limiter:
        try:
            worker = idle_workers.pop()
        except IndexError:
            worker = _Worker()

    try:
        return await to_thread.run_sync(
            worker.call,
            func,
            args,
            limiter=limiter,
        )
    finally:
        # Prune workers that have been idle for too long
        now = current_time()
        while idle_workers:
            if now - idle_workers[0].last_used <= MAX_WORKER_IDLE_TIME:
                break

            await to_thread.run_sync(idle_workers.popleft().destroy, limiter=limiter)

        worker.last_used = current_time()
        idle_workers.append(worker)


def current_default_interpreter_limiter() -> CapacityLimiter:
    """
    Return the capacity limiter used by default to limit the number of concurrently
    running subinterpreters.

    Defaults to the number of CPU cores.

    :return: a capacity limiter object

    """
    try:
        return _default_interpreter_limiter.get()
    except LookupError:
        limiter = CapacityLimiter(os.cpu_count() or DEFAULT_CPU_COUNT)
        _default_interpreter_limiter.set(limiter)
        return limiter
<|endoftext|>from __future__ import annotations

__all__ = (
    "BlockingPortal",
    "BlockingPortalProvider",
    "check_cancelled",
    "run",
    "run_sync",
    "start_blocking_portal",
)

import sys
from collections.abc import Awaitable, Callable, Coroutine, Generator
from concurrent.futures import Future
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    contextmanager,
)
from contextvars import Context
from dataclasses import dataclass, field
from functools import partial
from inspect import isawaitable
from threading import Lock, Thread, current_thread, get_ident
from types import TracebackType
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

from ._core._eventloop import (
    get_cancelled_exc_class,
    threadlocals,
)
from ._core._eventloop import run as run_eventloop
from ._core._exceptions import NoEventLoopError
from ._core._synchronization import Event
from ._core._tasks import CancelScope, create_task_group
from .abc._tasks import TaskStatus
from .lowlevel import EventLoopToken, current_token

if sys.version_info >= (3, 11):
    from typing import Self, TypeVarTuple, Unpack
else:
    from typing_extensions import Self, TypeVarTuple, Unpack

T_Retval = TypeVar("T_Retval")
T_co = TypeVar("T_co", covariant=True)
PosArgsT = TypeVarTuple("PosArgsT")


def _token_or_error(token: EventLoopToken | None) -> EventLoopToken:
    if token is not None:
        return token

    try:
        return threadlocals.current_token
    except AttributeError:
        raise NoEventLoopError(
            "Not running inside an AnyIO worker thread, and no event loop token was "
            "provided"
        ) from None


def run(
    func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
    *args: Unpack[PosArgsT],
    token: EventLoopToken | None = None,
) -> T_co:
    """
    Call a coroutine function from a worker thread.

    :param func: a coroutine function
    :param args: positional arguments for the callable
    :param token: an event loop token to use to get back to the event loop thread
        (required if calling this function from outside an AnyIO worker thread)
    :return: the return value of the coroutine function
    :raises MissingTokenError: if no token was provided and called from outside an
        AnyIO worker thread
    :raises RunFinishedError: if the event loop tied to ``token`` is no longer running

    .. versionchanged:: 4.11.0
        Added the ``token`` parameter.

    """
    explicit_token = token is not None
    token = _token_or_error(token)
    return token.backend_class.run_async_from_thread(
        func, args, token=token.native_token if explicit_token else None
    )


def run_sync(
    func: Callable[[Unpack[PosArgsT]], T_Retval],
    *args: Unpack[PosArgsT],
    token: EventLoopToken | None = None,
) -> T_Retval:
    """
    Call a function in the event loop thread from a worker thread.

    :param func: a callable
    :param args: positional arguments for the callable
    :param token: an event loop token to use to get back to the event loop thread
        (required if calling this function from outside an AnyIO worker thread)
    :return: the return value of the callable
    :raises MissingTokenError: if no token was provided and called from outside an
        AnyIO worker thread
    :raises RunFinishedError: if the event loop tied to ``token`` is no longer running

    .. versionchanged:: 4.11.0
        Added the ``token`` parameter.

    """
    explicit_token = token is not None
    token = _token_or_error(token)
    return token.backend_class.run_sync_from_thread(
        func, args, token=token.native_token if explicit_token else None
    )


class _BlockingAsyncContextManager(AbstractContextManager, Generic[T_co]):
    _enter_future: Future[T_co]
    _exit_future: Future[bool | None]
    _exit_event: Event
    _exit_exc_info: tuple[
        type[BaseException] | None, BaseException | None, TracebackType | None
    ] = (None, None, None)

    def __init__(
        self, async_cm: AbstractAsyncContextManager[T_co], portal: BlockingPortal
    ):
        self._async_cm = async_cm
        self._portal = portal

    async def run_async_cm(self) -> bool | None:
        try:
            self._exit_event = Event()
            value = await self._async_cm.__aenter__()
        except BaseException as exc:
            self._enter_future.set_exception(exc)
            raise
        else:
            self._enter_future.set_result(value)

        try:
            # Wait for the sync context manager to exit.
            # This next statement can raise `get_cancelled_exc_class()` if
            # something went wrong in a task group in this async context
            # manager.
            await self._exit_event.wait()
        finally:
            # In case of cancellation, it could be that we end up here before
            # `_BlockingAsyncContextManager.__exit__` is called, and an
            # `_exit_exc_info` has been set.
            result = await self._async_cm.__aexit__(*self._exit_exc_info)

        return result

    def __enter__(self) -> T_co:
        self._enter_future = Future()
        self._exit_future = self._portal.start_task_soon(self.run_async_cm)
        return self._enter_future.result()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        self._exit_exc_info = exc_type, exc_value, traceback
        self._portal.call(self._exit_event.set)
        return self._exit_future.result()


class _BlockingPortalTaskStatus(TaskStatus):
    def __init__(self, future: Future):
        self._future = future

    def started(self, value: object = None) -> None:
        self._future.set_result(value)


class BlockingPortal:
    """
    An object that lets external threads run code in an asynchronous event loop.

    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread
    """

    def __init__(self) -> None:
        self._token = current_token()
        self._event_loop_thread_id: int | None = get_ident()
        self._stop_event = Event()
        self._task_group = create_task_group()

    async def __aenter__(self) -> Self:
        await self._task_group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        await self.stop()
        return await self._task_group.__aexit__(exc_type, exc_val, exc_tb)

    def _check_running(self) -> None:
        if self._event_loop_thread_id is None:
            raise RuntimeError("This portal is not running")
        if self._event_loop_thread_id == get_ident():
            raise RuntimeError(
                "This method cannot be called from the event loop thread"
            )

    async def sleep_until_stopped(self) -> None:
        """Sleep until :meth:`stop` is called."""
        await self._stop_event.wait()

    async def stop(self, cancel_remaining: bool = False) -> None:
        """
        Signal the portal to shut down.

        This marks the portal as no longer accepting new calls and exits from
        :meth:`sleep_until_stopped`.

        :param cancel_remaining: ``True`` to cancel all the remaining tasks, ``False``
            to let them finish before returning

        """
        self._event_loop_thread_id = None
        self._stop_event.set()
        if cancel_remaining:
            self._task_group.cancel_scope.cancel("the blocking portal is shutting down")

    async def _call_func(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        future: Future[T_Retval],
    ) -> None:
        event_loop_thread_id = self._event_loop_thread_id

        def callback(f: Future[T_Retval]) -> None:
            if f.cancelled():
                if event_loop_thread_id == get_ident():
                    scope.cancel("the future was cancelled")
                elif event_loop_thread_id is not None:
                    run_sync(
                        scope.cancel, "the future was cancelled", token=self._token
                    )

        try:
            retval_or_awaitable = func(*args, **kwargs)
            if isawaitable(retval_or_awaitable):
                with CancelScope() as scope:
                    future.add_done_callback(callback)
                    retval = await retval_or_awaitable
            else:
                retval = retval_or_awaitable
        except get_cancelled_exc_class():
            future.cancel()
            future.set_running_or_notify_cancel()
        except BaseException as exc:
            if not future.cancelled():
                future.set_exception(exc)

            # Let base exceptions fall through
            if not isinstance(exc, Exception):
                raise
        else:
            if not future.cancelled():
                future.set_result(retval)
        finally:
            scope = None  # type: ignore[assignment]

    def _spawn_task_from_thread(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        name: object,
        future: Future[T_Retval],
    ) -> None:
        """
        Spawn a new task using the given callable.

        :param func: a callable
        :param args: positional arguments to be passed to the callable
        :param kwargs: keyword arguments to be passed to the callable
        :param name: name of the task (will be coerced to a string if not ``None``)
        :param future: a future that will resolve to the return value of the callable,
            or the exception raised during its execution

        """
        run_sync(
            partial(self._task_group.start_soon, name=name),
            self._call_func,
            func,
            args,
            kwargs,
            future,
            token=self._token,
        )

    @overload
    def call(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        *args: Unpack[PosArgsT],
    ) -> T_Retval: ...

    @overload
    def call(
        self, func: Callable[[Unpack[PosArgsT]], T_Retval], *args: Unpack[PosArgsT]
    ) -> T_Retval: ...

    def call(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        *args: Unpack[PosArgsT],
    ) -> T_Retval:
        """
        Call the given function in the event loop thread.

        If the callable returns a coroutine object, it is awaited on.

        :param func: any callable
        :raises RuntimeError: if the portal is not running or if this method is called
            from within the event loop thread

        """
        return cast(T_Retval, self.start_task_soon(func, *args).result())

    @overload
    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]: ...

    @overload
    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]: ...

    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]:
        """
        Start a task in the portal's task group.

        The task will be run inside a cancel scope which can be cancelled by cancelling
        the returned future.

        :param func: the target function
        :param args: positional arguments passed to ``func``
        :param name: name of the task (will be coerced to a string if not ``None``)
        :return: a future that resolves with the return value of the callable if the
            task completes successfully, or with the exception raised in the task
        :raises RuntimeError: if the portal is not running or if this method is called
            from within the event loop thread
        :rtype: concurrent.futures.Future[T_Retval]

        .. versionadded:: 3.0

        """
        self._check_running()
        f: Future[T_Retval] = Future()
        self._spawn_task_from_thread(func, args, {}, name, f)
        return f

    def start_task(
        self,
        func: Callable[..., Awaitable[T_Retval]],
        *args: object,
        name: object = None,
    ) -> tuple[Future[T_Retval], Any]:
        """
        Start a task in the portal's task group and wait until it signals for readiness.

        This method works the same way as :meth:`.abc.TaskGroup.start`.

        :param func: the target function
        :param args: positional arguments passed to ``func``
        :param name: name of the task (will be coerced to a string if not ``None``)
        :return: a tuple of (future, task_status_value) where the ``task_status_value``
            is the value passed to ``task_status.started()`` from within the target
            function
        :rtype: tuple[concurrent.futures.Future[T_Retval], Any]

        .. versionadded:: 3.0

        """

        def task_done(future: Future[T_Retval]) -> None:
            if not task_status_future.done():
                if future.cancelled():
                    task_status_future.cancel()
                elif future.exception():
                    task_status_future.set_exception(future.exception())
                else:
                    exc = RuntimeError(
                        "Task exited without calling task_status.started()"
                    )
                    task_status_future.set_exception(exc)

        self._check_running()
        task_status_future: Future = Future()
        task_status = _BlockingPortalTaskStatus(task_status_future)
        f: Future = Future()
        f.add_done_callback(task_done)
        self._spawn_task_from_thread(func, args, {"task_status": task_status}, name, f)
        return f, task_status_future.result()

    def wrap_async_context_manager(
        self, cm: AbstractAsyncContextManager[T_co]
    ) -> AbstractContextManager[T_co]:
        """
        Wrap an async context manager as a synchronous context manager via this portal.

        Spawns a task that will call both ``__aenter__()`` and ``__aexit__()``, stopping
        in the middle until the synchronous context manager exits.

        :param cm: an asynchronous context manager
        :return: a synchronous context manager

        .. versionadded:: 2.1

        """
        return _BlockingAsyncContextManager(cm, self)


@dataclass
class BlockingPortalProvider:
    """
    A manager for a blocking portal. Used as a context manager. The first thread to
    enter this context manager causes a blocking portal to be started with the specific
    parameters, and the last thread to exit causes the portal to be shut down. Thus,
    there will be exactly one blocking portal running in this context as long as at
    least one thread has entered this context manager.

    The parameters are the same as for :func:`~anyio.run`.

    :param backend: name of the backend
    :param backend_options: backend options

    .. versionadded:: 4.4
    """

    backend: str = "asyncio"
    backend_options: dict[str, Any] | None = None
    _lock: Lock = field(init=False, default_factory=Lock)
    _leases: int = field(init=False, default=0)
    _portal: BlockingPortal = field(init=False)
    _portal_cm: AbstractContextManager[BlockingPortal] | None = field(
        init=False, default=None
    )

    def __enter__(self) -> BlockingPortal:
        with self._lock:
            if self._portal_cm is None:
                self._portal_cm = start_blocking_portal(
                    self.backend, self.backend_options
                )
                self._portal = self._portal_cm.__enter__()

            self._leases += 1
            return self._portal

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        portal_cm: AbstractContextManager[BlockingPortal] | None = None
        with self._lock:
            assert self._portal_cm
            assert self._leases > 0
            self._leases -= 1
            if not self._leases:
                portal_cm = self._portal_cm
                self._portal_cm = None
                del self._portal

        if portal_cm:
            portal_cm.__exit__(None, None, None)


@contextmanager
def start_blocking_portal(
    backend: str = "asyncio",
    backend_options: dict[str, Any] | None = None,
    *,
    name: str | None = None,
) -> Generator[BlockingPortal, Any, None]:
    """
    Start a new event loop in a new thread and run a blocking portal in its main task.

    The parameters are the same as for :func:`~anyio.run`.

    :param backend: name of the backend
    :param backend_options: backend options
    :param name: name of the thread
    :return: a context manager that yields a blocking portal

    .. versionchanged:: 3.0
        Usage as a context manager is now required.

    """

    async def run_portal() -> None:
        async with BlockingPortal() as portal_:
            if name is None:
                current_thread().name = f"{backend}-portal-{id(portal_):x}"

            future.set_result(portal_)
            await portal_.sleep_until_stopped()

    def run_blocking_portal() -> None:
        if future.set_running_or_notify_cancel():
            try:
                run_eventloop(
                    run_portal, backend=backend, backend_options=backend_options
                )
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)

    future: Future[BlockingPortal] = Future()
    kwargs: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        kwargs["context"] = Context()

    thread = Thread(target=run_blocking_portal, daemon=True, name=name, **kwargs)
    thread.start()
    try:
        cancel_remaining_tasks = False
        portal = future.result()
        try:
            yield portal
        except BaseException:
            cancel_remaining_tasks = True
            raise
        finally:
            try:
                portal.call(portal.stop, cancel_remaining_tasks)
            except RuntimeError:
                pass
    finally:
        thread.join()


def check_cancelled() -> None:
    """
    Check if the cancel scope of the host task's running the current worker thread has
    been cancelled.

    If the host task's current cancel scope has indeed been cancelled, the
    backend-specific cancellation exception will be raised.

    :raises RuntimeError: if the current thread was not spawned by
        :func:`.to_thread.run_sync`

    """
    try:
        token: EventLoopToken = threadlocals.current_token
    except AttributeError:
        raise NoEventLoopError(
            "This function can only be called inside an AnyIO worker thread"
        ) from None

    token.backend_class.check_cancelled()
<|endoftext|>from __future__ import annotations

__all__ = (
    "EventLoopToken",
    "RunVar",
    "RunvarToken",
    "cancel_shielded_checkpoint",
    "checkpoint",
    "checkpoint_if_cancelled",
    "current_token",
)

import enum
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, final, overload
from weakref import WeakKeyDictionary

from ._core._eventloop import get_async_backend

if TYPE_CHECKING:
    from .abc import AsyncBackend

T = TypeVar("T")
D = TypeVar("D")


async def checkpoint() -> None:
    """
    Check for cancellation and allow the scheduler to switch to another task.

    Equivalent to (but more efficient than)::

        await checkpoint_if_cancelled()
        await cancel_shielded_checkpoint()

    .. versionadded:: 3.0

    """
    await get_async_backend().checkpoint()


async def checkpoint_if_cancelled() -> None:
    """
    Enter a checkpoint if the enclosing cancel scope has been cancelled.

    This does not allow the scheduler to switch to a different task.

    .. versionadded:: 3.0

    """
    await get_async_backend().checkpoint_if_cancelled()


async def cancel_shielded_checkpoint() -> None:
    """
    Allow the scheduler to switch to another task but without checking for cancellation.

    Equivalent to (but potentially more efficient than)::

        with CancelScope(shield=True):
            await checkpoint()

    .. versionadded:: 3.0

    """
    await get_async_backend().cancel_shielded_checkpoint()


@final
@dataclass(frozen=True, repr=False)
class EventLoopToken:
    """
    An opaque object that holds a reference to an event loop.

    .. versionadded:: 4.11.0
    """

    backend_class: type[AsyncBackend]
    native_token: object


def current_token() -> EventLoopToken:
    """
    Return a token object that can be used to call code in the current event loop from
    another thread.

    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    .. versionadded:: 4.11.0

    """
    backend_class = get_async_backend()
    raw_token = backend_class.current_token()
    return EventLoopToken(backend_class, raw_token)


_run_vars: WeakKeyDictionary[object, dict[RunVar[Any], Any]] = WeakKeyDictionary()


class _NoValueSet(enum.Enum):
    NO_VALUE_SET = enum.auto()


class RunvarToken(Generic[T]):
    """
    A token that can be used to restore a :class:`RunVar` to its previous value.

    Returned by :meth:`RunVar.set`. Can be used as a context manager to automatically
    reset the variable on exit, or passed directly to :meth:`RunVar.reset`.
    """

    __slots__ = "_redeemed", "_value", "_var"

    def __init__(self, var: RunVar[T], value: T | Literal[_NoValueSet.NO_VALUE_SET]):
        self._var = var
        self._value: T | Literal[_NoValueSet.NO_VALUE_SET] = value
        self._redeemed = False

    def __enter__(self) -> RunvarToken[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._var.reset(self)


class RunVar(Generic[T]):
    """
    Like a :class:`~contextvars.ContextVar`, except scoped to the running event loop.

    Can be used as a context manager, Just like :class:`~contextvars.ContextVar`, that
    will reset the variable to its previous value when the context block is exited.
    """

    __slots__ = "_default", "_name"

    NO_VALUE_SET: Literal[_NoValueSet.NO_VALUE_SET] = _NoValueSet.NO_VALUE_SET

    def __init__(
        self, name: str, default: T | Literal[_NoValueSet.NO_VALUE_SET] = NO_VALUE_SET
    ):
        self._name = name
        self._default = default

    @property
    def _current_vars(self) -> dict[RunVar[T], T]:
        native_token = current_token().native_token
        try:
            return _run_vars[native_token]
        except KeyError:
            run_vars = _run_vars[native_token] = {}
            return run_vars

    @overload
    def get(self, default: D) -> T | D: ...

    @overload
    def get(self) -> T: ...

    def get(
        self, default: D | Literal[_NoValueSet.NO_VALUE_SET] = NO_VALUE_SET
    ) -> T | D:
        """
        Return the current value of this run variable.

        :param default: a fallback value to return if no value has been set
        :return: the current value, the provided default, or the variable's own default
        :raises LookupError: if no value is set and no default is available

        """
        try:
            return self._current_vars[self]
        except KeyError:
            if default is not RunVar.NO_VALUE_SET:
                return default
            elif self._default is not RunVar.NO_VALUE_SET:
                return self._default

        raise LookupError(
            f'Run variable "{self._name}" has no value and no default set'
        )

    def set(self, value: T) -> RunvarToken[T]:
        """
        Set the value of this run variable for the current event loop.

        :param value: the new value
        :return: a token that can be used to restore the previous value

        """
        current_vars = self._current_vars
        token = RunvarToken(self, current_vars.get(self, RunVar.NO_VALUE_SET))
        current_vars[self] = value
        return token

    def reset(self, token: RunvarToken[T]) -> None:
        """
        Restore this run variable to the value it held before the matching :meth:`set`.

        :param token: the token returned by :meth:`set`
        :raises ValueError: if the token belongs to a different :class:`RunVar` or the token
            has already been used

        """
        if token._var is not self:
            raise ValueError("This token does not belong to this RunVar")

        if token._redeemed:
            raise ValueError("This token has already been used")

        if token._value is _NoValueSet.NO_VALUE_SET:
            try:
                del self._current_vars[self]
            except KeyError:
                pass
        else:
            self._current_vars[self] = token._value

        token._redeemed = True

    def __repr__(self) -> str:
        return f"<RunVar name={self._name!r}>"
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from ._lazyimport import (
    fix_package_names,
    install_lazy_importer,
    set_deprecated_aliases,
)

if TYPE_CHECKING or not install_lazy_importer():
    from ._core._concurrency_utils import amap as amap
    from ._core._concurrency_utils import as_completed as as_completed
    from ._core._concurrency_utils import gather as gather
    from ._core._contextmanagers import (
        AsyncContextManagerMixin as AsyncContextManagerMixin,
    )
    from ._core._contextmanagers import ContextManagerMixin as ContextManagerMixin
    from ._core._eventloop import current_time as current_time
    from ._core._eventloop import get_all_backends as get_all_backends
    from ._core._eventloop import get_available_backends as get_available_backends
    from ._core._eventloop import get_cancelled_exc_class as get_cancelled_exc_class
    from ._core._eventloop import run as run
    from ._core._eventloop import sleep as sleep
    from ._core._eventloop import sleep_forever as sleep_forever
    from ._core._eventloop import sleep_until as sleep_until
    from ._core._exceptions import BrokenResourceError as BrokenResourceError
    from ._core._exceptions import BrokenWorkerInterpreter as BrokenWorkerInterpreter
    from ._core._exceptions import BrokenWorkerProcess as BrokenWorkerProcess
    from ._core._exceptions import BusyResourceError as BusyResourceError
    from ._core._exceptions import ClosedResourceError as ClosedResourceError
    from ._core._exceptions import ConnectionFailed as ConnectionFailed
    from ._core._exceptions import DelimiterNotFound as DelimiterNotFound
    from ._core._exceptions import EndOfStream as EndOfStream
    from ._core._exceptions import FutureAlreadyFinished as FutureAlreadyFinished
    from ._core._exceptions import FutureCancelled as FutureCancelled
    from ._core._exceptions import FutureFailed as FutureFailed
    from ._core._exceptions import FutureNotFinished as FutureNotFinished
    from ._core._exceptions import IncompleteRead as IncompleteRead
    from ._core._exceptions import NoEventLoopError as NoEventLoopError
    from ._core._exceptions import RunFinishedError as RunFinishedError
    from ._core._exceptions import TaskCancelled as TaskCancelled
    from ._core._exceptions import TaskFailed as TaskFailed
    from ._core._exceptions import TaskNotFinished as TaskNotFinished
    from ._core._exceptions import (
        TypedAttributeLookupError as TypedAttributeLookupError,
    )
    from ._core._exceptions import WouldBlock as WouldBlock
    from ._core._fileio import AsyncFile as AsyncFile
    from ._core._fileio import Path as Path
    from ._core._fileio import open_file as open_file
    from ._core._fileio import wrap_file as wrap_file
    from ._core._futures import Future as Future
    from ._core._resources import aclose_forcefully as aclose_forcefully
    from ._core._signals import open_signal_receiver as open_signal_receiver
    from ._core._sockets import TCPConnectable as TCPConnectable
    from ._core._sockets import UNIXConnectable as UNIXConnectable
    from ._core._sockets import as_connectable as as_connectable
    from ._core._sockets import connect_tcp as connect_tcp
    from ._core._sockets import connect_unix as connect_unix
    from ._core._sockets import (
        create_connected_udp_socket as create_connected_udp_socket,
    )
    from ._core._sockets import (
        create_connected_unix_datagram_socket as create_connected_unix_datagram_socket,
    )
    from ._core._sockets import create_tcp_listener as create_tcp_listener
    from ._core._sockets import create_udp_socket as create_udp_socket
    from ._core._sockets import (
        create_unix_datagram_socket as create_unix_datagram_socket,
    )
    from ._core._sockets import create_unix_listener as create_unix_listener
    from ._core._sockets import getaddrinfo as getaddrinfo
    from ._core._sockets import getnameinfo as getnameinfo
    from ._core._sockets import notify_closing as notify_closing
    from ._core._sockets import wait_readable as wait_readable
    from ._core._sockets import wait_socket_readable as wait_socket_readable
    from ._core._sockets import wait_socket_writable as wait_socket_writable
    from ._core._sockets import wait_writable as wait_writable
    from ._core._streams import (
        create_memory_object_stream as create_memory_object_stream,
    )
    from ._core._subprocesses import open_process as open_process
    from ._core._subprocesses import run_process as run_process
    from ._core._synchronization import CapacityLimiter as CapacityLimiter
    from ._core._synchronization import (
        CapacityLimiterStatistics as CapacityLimiterStatistics,
    )
    from ._core._synchronization import Condition as Condition
    from ._core._synchronization import ConditionStatistics as ConditionStatistics
    from ._core._synchronization import Event as Event
    from ._core._synchronization import EventStatistics as EventStatistics
    from ._core._synchronization import Lock as Lock
    from ._core._synchronization import LockStatistics as LockStatistics
    from ._core._synchronization import ResourceGuard as ResourceGuard
    from ._core._synchronization import Semaphore as Semaphore
    from ._core._synchronization import SemaphoreStatistics as SemaphoreStatistics
    from ._core._tasks import TASK_STATUS_IGNORED as TASK_STATUS_IGNORED
    from ._core._tasks import CancelScope as CancelScope
    from ._core._tasks import TaskHandle as TaskHandle
    from ._core._tasks import create_task_group as create_task_group
    from ._core._tasks import current_effective_deadline as current_effective_deadline
    from ._core._tasks import fail_after as fail_after
    from ._core._tasks import fail_at as fail_at
    from ._core._tasks import move_on_after as move_on_after
    from ._core._tasks import move_on_at as move_on_at
    from ._core._tempfile import NamedTemporaryFile as NamedTemporaryFile
    from ._core._tempfile import SpooledTemporaryFile as SpooledTemporaryFile
    from ._core._tempfile import TemporaryDirectory as TemporaryDirectory
    from ._core._tempfile import TemporaryFile as TemporaryFile
    from ._core._tempfile import gettempdir as gettempdir
    from ._core._tempfile import gettempdirb as gettempdirb
    from ._core._tempfile import mkdtemp as mkdtemp
    from ._core._tempfile import mkstemp as mkstemp
    from ._core._testing import TaskInfo as TaskInfo
    from ._core._testing import get_current_task as get_current_task
    from ._core._testing import get_running_tasks as get_running_tasks
    from ._core._testing import wait_all_tasks_blocked as wait_all_tasks_blocked
    from ._core._typedattr import TypedAttributeProvider as TypedAttributeProvider
    from ._core._typedattr import TypedAttributeSet as TypedAttributeSet
    from ._core._typedattr import typed_attribute as typed_attribute

    # ruff: isort: off
    from . import (
        abc as abc,
    )
    from . import (
        from_thread as from_thread,
    )
    from . import (
        functools as functools,
    )
    from . import (
        itertools as itertools,
    )
    from . import (
        lowlevel as lowlevel,
    )
    from . import (
        to_interpreter as to_interpreter,
    )
    from . import (
        to_process as to_process,
    )
    from . import (
        to_thread as to_thread,
    )

    fix_package_names()
    set_deprecated_aliases(
        {
            "BrokenWorkerIntepreter": "anyio.BrokenWorkerInterpreter",
        }
    )
<|endoftext|>from __future__ import annotations

__all__ = (
    "fix_package_names",
    "install_lazy_importer",
    "set_deprecated_aliases",
)

import ast
import inspect
import sys
import warnings
from importlib import import_module
from types import ModuleType
from typing import Any


def install_lazy_importer() -> bool:
    module_globals = sys._getframe(1).f_globals
    module_name = module_globals["__name__"]
    module_prefix = module_name + "."
    module = sys.modules[module_name]
    lazy_map, deprecated_aliases, submodule_names = _build_lazy_map(module)
    names = sorted(lazy_map)

    # Delete symbols that are not part of the API
    del module_globals["TYPE_CHECKING"]
    del module_globals["install_lazy_importer"]

    if not lazy_map and not deprecated_aliases:
        return False

    def __getattr__(name: str) -> Any:
        if new_name := deprecated_aliases.get(name):
            emit_deprecation_warning(module_name, name, new_name)
            target_mod, target_attr = new_name.rsplit(".", 1)
        elif name in submodule_names:
            target_mod, target_attr = "." + name, ""
        else:
            try:
                target_mod, target_attr = lazy_map[name]
            except KeyError:
                raise AttributeError(
                    f"module {module_name!r} has no attribute {name!r}"
                ) from None

        imported = import_module(target_mod, module_name)
        value = getattr(imported, target_attr) if target_attr else imported

        # patch the module name to match
        if (
            getattr(value, "__module__", "").startswith(module_prefix)
            and name not in deprecated_aliases
        ):
            value.__module__ = module_name

        module_globals[name] = value
        return value

    def __dir__() -> list[str]:
        return names

    module_globals["__dir__"] = __dir__
    module_globals["__getattr__"] = __getattr__
    module_globals.pop("fix_package_names", None)
    module_globals.pop("set_deprecated_aliases", None)
    return True


def fix_package_names() -> None:
    module_globals = sys._getframe(1).f_globals
    module_prefix = module_globals["__name__"] + "."
    del module_globals[fix_package_names.__name__]
    for value in module_globals.values():
        if modname := getattr(value, "__module__", ""):
            if modname.startswith(module_prefix):
                parts = modname.split(".")
                value.__module__ = ".".join(
                    part for part in parts if not part.startswith("_")
                )


def emit_deprecation_warning(module_name: str, name: str, target: str) -> None:
    warnings.warn(
        f"The {module_name}.{name} alias is deprecated, use {target} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def set_deprecated_aliases(aliases: dict[str, str]) -> None:
    module_globals = sys._getframe(1).f_globals
    module_name = module_globals["__name__"]
    del module_globals[set_deprecated_aliases.__name__]

    def __getattr__(name: str) -> Any:
        try:
            target = aliases[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None

        emit_deprecation_warning(module_name, name, target)
        target_modname, attrname = target.rsplit(".", 1)
        module = import_module(target_modname)
        return getattr(module, attrname)

    sys.modules[module_name].__dict__["__getattr__"] = __getattr__


def _build_lazy_map(
    module: ModuleType,
) -> tuple[dict[str, tuple[str, str]], dict[str, str], list[str]]:
    try:
        source = inspect.getsource(module)
    except OSError:
        return {}, {}, []

    tree = compile(source, module.__file__ or "", "exec", ast.PyCF_ONLY_AST)
    assert isinstance(tree, ast.Module)
    out: dict[str, tuple[str, str]] = {}
    deprecated_aliases: dict[str, str] = {}
    submodule_names: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.If) or not _is_type_checking_block(node.test):
            continue

        for stmt in node.body:
            match stmt:
                case ast.ImportFrom():
                    if stmt.module is None:
                        submodule_names.extend(alias.name for alias in stmt.names)
                    else:
                        base = "." * stmt.level + (stmt.module or "")
                        for alias in stmt.names:
                            if alias.name == "*":
                                raise RuntimeError("star imports not supported")

                            exported = alias.asname or alias.name
                            out[exported] = (base, alias.name)
                case ast.Expr() if isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "set_deprecated_aliases"
                    ):
                        arg0 = call.args[0]
                        assert isinstance(arg0, ast.Dict)
                        for key, value in zip(arg0.keys, arg0.values, strict=True):
                            assert isinstance(key, ast.Constant)
                            assert isinstance(key.value, str)
                            assert isinstance(value, ast.Constant)
                            assert isinstance(value.value, str)
                            deprecated_aliases[key.value] = value.value

    return out, deprecated_aliases, submodule_names


def _is_type_checking_block(test: ast.AST) -> bool:
    if not isinstance(test, ast.BoolOp):
        return False

    subtest = test.values[0]
    match subtest:
        case ast.Name():
            return subtest.id == "TYPE_CHECKING"
        case ast.Attribute():
            return (
                isinstance(subtest.value, ast.Name)
                and subtest.value.id == "typing"
                and subtest.attr == "TYPE_CHECKING"
            )
        case _:
            return False
<|endoftext|>from __future__ import annotations

__all__ = (
    "Chain",
    "accumulate",
    "batched",
    "combinations",
    "combinations_with_replacement",
    "compress",
    "count",
    "cycle",
    "dropwhile",
    "filterfalse",
    "groupby",
    "islice",
    "pairwise",
    "permutations",
    "product",
    "repeat",
    "starmap",
    "takewhile",
    "tee",
    "zip_longest",
)

import itertools
import operator
import sys
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast, overload

from ._core._synchronization import Lock
from ._core._tasks import CancelScope
from .lowlevel import cancel_shielded_checkpoint, checkpoint, checkpoint_if_cancelled

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

T = TypeVar("T")
R = TypeVar("R")
_tee_end = sentinel("_tee_end")


@dataclass(eq=False)
class _IterableAsyncIterator(AsyncIterator[T]):
    iterator: Iterator[T]

    async def __anext__(self) -> T:
        await checkpoint_if_cancelled()
        try:
            result = next(self.iterator)
        except StopIteration:
            await cancel_shielded_checkpoint()
            raise StopAsyncIteration from None

        await cancel_shielded_checkpoint()
        return result


def _iterate(iterable: Iterable[T] | AsyncIterable[T]) -> AsyncIterator[T]:
    if isinstance(iterable, AsyncIterator):
        return iterable

    if isinstance(iterable, AsyncIterable):
        return iterable.__aiter__()

    return _IterableAsyncIterator(iter(iterable))


@dataclass(eq=False)
class _TeeLink(Generic[T]):
    value: object | None = None
    next: _TeeLink[T] | None = None
    filled: bool = False


@dataclass(eq=False)
class _TeeState(Generic[T]):
    iterator: AsyncIterator[T]
    lock: Lock = field(default_factory=Lock)

    async def fill(self, link: _TeeLink[T]) -> bool:
        if link.filled:
            return False

        async with self.lock:
            if link.filled:
                return True

            link.value = await anext(self.iterator, _tee_end)
            if link.value is not _tee_end:
                link.next = _TeeLink()

            link.filled = True
            return True


class _TeeAsyncIterator(AsyncIterator[T]):
    _state: _TeeState[T]
    _link: _TeeLink[T]
    _element_yielded: bool

    def __init__(
        self, iterable: Iterable[T] | AsyncIterable[T] | _TeeAsyncIterator[T]
    ) -> None:
        if isinstance(iterable, _TeeAsyncIterator):
            self._state = iterable._state
            self._link = iterable._link
        else:
            self._state = _TeeState(_iterate(iterable))
            self._link = _TeeLink()

        self._element_yielded = False

    async def __anext__(self) -> T:
        had_yieldpoint = await self._state.fill(self._link)
        if self._link.value is _tee_end:
            if not self._element_yielded:
                await checkpoint()

            raise StopAsyncIteration

        if not had_yieldpoint:
            await checkpoint_if_cancelled()

        self._element_yielded = True
        value = cast(T, self._link.value)
        next_link = self._link.next
        assert next_link is not None
        self._link = next_link
        if not had_yieldpoint:
            await cancel_shielded_checkpoint()

        return value


async def _operator_add(x: T, y: T) -> T:
    return operator.add(x, y)


async def accumulate(
    iterable: Iterable[T] | AsyncIterable[T],
    function: Callable[[T, T], Awaitable[T]] = _operator_add,
    *,
    initial: T | None = None,
) -> AsyncGenerator[T, None]:
    iterator = _iterate(iterable)
    if initial is None:
        try:
            total = await anext(iterator)
        except StopAsyncIteration:
            await checkpoint()
            return
    else:
        await checkpoint_if_cancelled()
        total = initial
        await cancel_shielded_checkpoint()

    yield total

    async for element in iterator:
        total = await function(total, element)
        yield total


async def batched(
    iterable: Iterable[T] | AsyncIterable[T], n: int, *, strict: bool = False
) -> AsyncGenerator[tuple[T, ...], None]:
    if n < 1:
        raise ValueError("n must be at least one")

    iterator = _iterate(iterable)

    while True:
        batch: list[T] = []
        for _ in range(n):
            try:
                batch.append(await anext(iterator))
            except StopAsyncIteration:
                if not batch:
                    await checkpoint()
                    return
                if strict:
                    raise ValueError("batched(): incomplete batch") from None

                yield tuple(batch)
                return

        yield tuple(batch)


class Chain:
    def __call__(
        self, *iterables: Iterable[T] | AsyncIterable[T]
    ) -> AsyncGenerator[T, None]:
        return self.from_iterable(iterables)

    async def from_iterable(
        self,
        iterables: (
            Iterable[Iterable[T] | AsyncIterable[T]]
            | AsyncIterable[Iterable[T] | AsyncIterable[T]]
        ),
    ) -> AsyncGenerator[T, None]:
        element_yielded = False
        outer_iter = _iterate(iterables)

        try:
            async for iterable in outer_iter:
                async for element in _iterate(iterable):
                    element_yielded = True
                    yield element
        finally:
            aclose = getattr(outer_iter, "aclose", None)
            if aclose is not None:
                with CancelScope(shield=True):
                    await aclose()

        if not element_yielded:
            await checkpoint()


chain: Chain = Chain()


async def combinations(
    iterable: Iterable[T] | AsyncIterable[T], r: int
) -> AsyncGenerator[tuple[T, ...], None]:
    pool: list[T] = [element async for element in _iterate(iterable)]
    async for combination in _iterate(itertools.combinations(pool, r)):
        yield combination


async def combinations_with_replacement(
    iterable: Iterable[T] | AsyncIterable[T], r: int
) -> AsyncGenerator[tuple[T, ...], None]:
    pool: list[T] = [element async for element in _iterate(iterable)]
    async for combination in _iterate(itertools.combinations_with_replacement(pool, r)):
        yield combination


async def compress(
    data: Iterable[T] | AsyncIterable[T],
    selectors: Iterable[object] | AsyncIterable[object],
) -> AsyncGenerator[T, None]:
    data_iterator = _iterate(data)
    selector_iterator = _iterate(selectors)
    element_yielded = False

    while True:
        try:
            datum = await anext(data_iterator)
            selector = await anext(selector_iterator)
        except StopAsyncIteration:
            if not element_yielded:
                await checkpoint()

            return

        if selector:
            element_yielded = True
            yield datum


async def count(start: int = 0, step: int = 1) -> AsyncGenerator[int, None]:
    n = start
    while True:
        await checkpoint_if_cancelled()
        value = n
        n += step
        await cancel_shielded_checkpoint()
        yield value


async def cycle(
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[T, None]:
    saved: list[T] = []
    async for element in _iterate(iterable):
        saved.append(element)
        yield element

    if not saved:
        await checkpoint()
        return

    while True:
        for element in saved:
            await checkpoint()
            yield element


async def dropwhile(
    predicate: Callable[[T], Awaitable[object]],
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[T, None]:
    element_yielded = False
    dropping = True

    async for element in _iterate(iterable):
        if dropping and await predicate(element):
            continue

        dropping = False
        element_yielded = True
        yield element

    if not element_yielded:
        await checkpoint()


async def filterfalse(
    predicate: Callable[[T], Awaitable[object]],
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[T, None]:
    element_yielded = False

    async for element in _iterate(iterable):
        if not await predicate(element):
            element_yielded = True
            yield element

    if not element_yielded:
        await checkpoint()


@overload
def groupby(
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[tuple[T, list[T]], None]: ...


@overload
def groupby(
    iterable: Iterable[T] | AsyncIterable[T],
    key: Callable[[T], Awaitable[R]],
) -> AsyncGenerator[tuple[R, list[T]], None]: ...


async def groupby(
    iterable: Iterable[T] | AsyncIterable[T],
    key: Callable[[T], Awaitable[object]] | None = None,
) -> AsyncGenerator[tuple[object, list[T]], None]:
    iterator = _iterate(iterable)
    try:
        element = await anext(iterator)
    except StopAsyncIteration:
        await checkpoint()
        return

    group_key = element if key is None else await key(element)
    values = [element]

    async for element in iterator:
        next_key = element if key is None else await key(element)
        if next_key != group_key:
            completed_group = group_key, values
            group_key = next_key
            values = [element]
            yield completed_group
        else:
            values.append(element)

    yield group_key, values


@overload
def islice(
    iterable: Iterable[T] | AsyncIterable[T],
    stop: int | None,
    /,
) -> AsyncGenerator[T, None]: ...


@overload
def islice(
    iterable: Iterable[T] | AsyncIterable[T],
    start: int | None,
    stop: int | None,
    step: int | None = 1,
    /,
) -> AsyncGenerator[T, None]: ...


async def islice(
    iterable: Iterable[T] | AsyncIterable[T],
    *args: int | None,
) -> AsyncGenerator[T, None]:
    if not args:
        raise TypeError("islice expected at least 2 arguments, got 1")
    if len(args) > 3:
        raise TypeError(f"islice expected at most 4 arguments, got {len(args) + 1}")

    slice_args = slice(*args)

    start_message = (
        "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize."
    )
    stop_message = (
        "Stop argument for islice() must be None or an integer: 0 <= x <= sys.maxsize."
    )
    step_message = "Step for islice() must be a positive integer or None."

    def normalize_index(value: object, message: str) -> int:
        try:
            index = operator.index(cast(Any, value))
        except TypeError:
            raise ValueError(message) from None

        if index < 0 or index > sys.maxsize:
            raise ValueError(message)

        return index

    start = (
        0
        if slice_args.start is None
        else normalize_index(slice_args.start, start_message)
    )
    stop = (
        None
        if slice_args.stop is None
        else normalize_index(slice_args.stop, stop_message)
    )
    step = (
        1 if slice_args.step is None else normalize_index(slice_args.step, step_message)
    )

    if step <= 0:
        raise ValueError(step_message)

    if stop == 0 or start == stop:
        await checkpoint()
        return

    iterator = _iterate(iterable)
    index = 0
    element_yielded = False

    while stop is None or index < stop:
        try:
            element = await anext(iterator)
        except StopAsyncIteration:
            if not element_yielded:
                await checkpoint()

            return

        if index >= start and (index - start) % step == 0:
            index += 1
            element_yielded = True
            yield element
        else:
            index += 1

    if not element_yielded:
        await checkpoint()


async def pairwise(
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[tuple[T, T], None]:
    iterator = _iterate(iterable)
    try:
        previous = await anext(iterator)
    except StopAsyncIteration:
        await checkpoint()
        return

    element_yielded = False
    async for element in iterator:
        element_yielded = True
        pair = (previous, element)
        previous = element
        yield pair

    if not element_yielded:
        await checkpoint()


async def permutations(
    iterable: Iterable[T] | AsyncIterable[T], r: int | None = None
) -> AsyncGenerator[tuple[T, ...], None]:
    pool: list[T] = [element async for element in _iterate(iterable)]
    n = len(pool)
    if r is None:
        r = n
    elif not isinstance(r, int):
        raise TypeError("Expected int as r")
    elif r < 0:
        raise ValueError("r must be non-negative")

    async for permutation in _iterate(itertools.permutations(pool, r)):
        yield permutation


async def product(
    *iterables: Iterable[T] | AsyncIterable[T], repeat: int = 1
) -> AsyncGenerator[tuple[T, ...], None]:
    repeat = operator.index(repeat)
    if repeat < 0:
        raise ValueError("repeat argument cannot be negative")

    pools: list[tuple[T, ...]] = []
    for iterable in iterables:
        pool: list[T] = [element async for element in _iterate(iterable)]
        pools.append(tuple(pool))

    async for value in _iterate(itertools.product(*pools, repeat=repeat)):
        yield value


async def repeat(element: T, times: int | None = None) -> AsyncGenerator[T, None]:
    if times is None:
        while True:
            await checkpoint()
            yield element

    remaining = operator.index(cast(Any, times))
    if remaining <= 0:
        await checkpoint()
        return

    while remaining > 0:
        await checkpoint_if_cancelled()
        remaining -= 1
        await cancel_shielded_checkpoint()
        yield element


async def starmap(
    function: Callable[..., Awaitable[R]],
    iterable: (
        Iterable[Iterable[object] | AsyncIterable[object]]
        | AsyncIterable[Iterable[object] | AsyncIterable[object]]
    ),
) -> AsyncGenerator[R, None]:
    result_yielded = False

    async for args_iterable in _iterate(iterable):
        args = [element async for element in _iterate(args_iterable)]
        result_yielded = True
        yield await function(*args)

    if not result_yielded:
        await checkpoint()


def tee(
    iterable: Iterable[T] | AsyncIterable[T], n: int = 2
) -> tuple[AsyncIterator[T], ...]:
    n = operator.index(cast(Any, n))
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return ()

    iterator = _TeeAsyncIterator(iterable)
    iterators: list[AsyncIterator[T]] = [iterator]
    iterators.extend(_TeeAsyncIterator(iterator) for _ in range(n - 1))
    return tuple(iterators)


async def takewhile(
    predicate: Callable[[T], Awaitable[object]],
    iterable: Iterable[T] | AsyncIterable[T],
) -> AsyncGenerator[T, None]:
    element_yielded = False

    async for element in _iterate(iterable):
        if not await predicate(element):
            if not element_yielded:
                await checkpoint()

            return

        element_yielded = True
        yield element

    if not element_yielded:
        await checkpoint()


async def zip_longest(
    *iterables: Iterable[object] | AsyncIterable[object],
    fillvalue: object = None,
) -> AsyncGenerator[tuple[object, ...], None]:
    iterators = [_iterate(iterable) for iterable in iterables]
    num_active = len(iterators)
    if not num_active:
        await checkpoint()
        return

    active = [True] * num_active
    tuple_yielded = False

    while True:
        values: list[object] = []
        for index, iterator in enumerate(iterators):
            if not active[index]:
                values.append(fillvalue)
                continue

            try:
                value = await anext(iterator)
            except StopAsyncIteration:
                active[index] = False
                num_active -= 1
                if not num_active:
                    if not tuple_yielded:
                        await checkpoint()

                    return

                value = fillvalue

            values.append(value)

        tuple_yielded = True
        yield tuple(values)
<|endoftext|>from __future__ import annotations

__all__ = (
    "AsyncCacheInfo",
    "AsyncCacheParameters",
    "AsyncLRUCacheWrapper",
    "cache",
    "lru_cache",
    "reduce",
)

import functools
from collections import OrderedDict
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
)
from functools import update_wrapper
from inspect import iscoroutinefunction
from typing import (
    Any,
    Generic,
    NamedTuple,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
    final,
    overload,
)
from weakref import WeakKeyDictionary

from ._core._eventloop import current_time
from ._core._synchronization import Lock
from .lowlevel import RunVar, checkpoint

T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")
lru_cache_items: RunVar[
    WeakKeyDictionary[
        AsyncLRUCacheWrapper[Any, Any],
        OrderedDict[
            Hashable,
            tuple[_InitialMissingType, Lock, float | None]
            | tuple[Any, None, float | None],
        ],
    ]
] = RunVar("lru_cache_items")


class _InitialMissingType:
    pass


initial_missing: _InitialMissingType = _InitialMissingType()


class AsyncCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int
    ttl: int | None


class AsyncCacheParameters(TypedDict):
    maxsize: int | None
    typed: bool
    always_checkpoint: bool
    ttl: int | None


class _LRUMethodWrapper(Generic[T]):
    def __init__(self, wrapper: AsyncLRUCacheWrapper[..., T], instance: object):
        self.__wrapper = wrapper
        self.__instance = instance

    def cache_info(self) -> AsyncCacheInfo:
        return self.__wrapper.cache_info()

    def cache_parameters(self) -> AsyncCacheParameters:
        return self.__wrapper.cache_parameters()

    def cache_clear(self) -> None:
        self.__wrapper.cache_clear()

    async def __call__(self, *args: Any, **kwargs: Any) -> T:
        if self.__instance is None:
            return await self.__wrapper(*args, **kwargs)

        return await self.__wrapper(self.__instance, *args, **kwargs)


@final
class AsyncLRUCacheWrapper(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, Awaitable[T]],
        maxsize: int | None,
        typed: bool,
        always_checkpoint: bool,
        ttl: int | None,
    ):
        self.__wrapped__ = func
        self._hits: int = 0
        self._misses: int = 0
        self._maxsize = max(maxsize, 0) if maxsize is not None else None
        self._currsize: int = 0
        self._typed = typed
        self._always_checkpoint = always_checkpoint
        self._ttl = ttl
        update_wrapper(self, func)

    def cache_info(self) -> AsyncCacheInfo:
        return AsyncCacheInfo(
            self._hits, self._misses, self._maxsize, self._currsize, self._ttl
        )

    def cache_parameters(self) -> AsyncCacheParameters:
        return {
            "maxsize": self._maxsize,
            "typed": self._typed,
            "always_checkpoint": self._always_checkpoint,
            "ttl": self._ttl,
        }

    def cache_clear(self) -> None:
        if cache := lru_cache_items.get(None):
            cache.pop(self, None)
            self._hits = self._misses = self._currsize = 0

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        # Easy case first: if maxsize == 0, no caching is done
        if self._maxsize == 0:
            value = await self.__wrapped__(*args, **kwargs)
            self._misses += 1
            return value

        # The key is constructed as a flat tuple to avoid memory overhead
        key: tuple[Any, ...] = args
        if kwargs:
            # initial_missing is used as a separator
            key += (initial_missing,) + sum(kwargs.items(), ())

        if self._typed:
            key += tuple(type(arg) for arg in args)
            if kwargs:
                key += (initial_missing,) + tuple(type(val) for val in kwargs.values())

        try:
            cache = lru_cache_items.get()
        except LookupError:
            cache = WeakKeyDictionary()
            lru_cache_items.set(cache)

        try:
            cache_entry = cache[self]
        except KeyError:
            cache_entry = cache[self] = OrderedDict()

        cached_value: T | _InitialMissingType
        try:
            cached_value, lock, expires_at = cache_entry[key]
        except KeyError:
            # We're the first task to call this function
            cached_value, lock, expires_at = (
                initial_missing,
                Lock(fast_acquire=not self._always_checkpoint),
                None,
            )
            cache_entry[key] = cached_value, lock, expires_at

        if lock is None:
            if expires_at is not None and current_time() >= expires_at:
                self._currsize -= 1
                cached_value, lock, expires_at = (
                    initial_missing,
                    Lock(fast_acquire=not self._always_checkpoint),
                    None,
                )
                cache_entry[key] = cached_value, lock, expires_at
            else:
                # The value was already cached
                self._hits += 1
                cache_entry.move_to_end(key)
                if self._always_checkpoint:
                    await checkpoint()

                return cast(T, cached_value)

        async with lock:
            # Check if another task filled the cache while we acquired the lock
            if (cached_value := cache_entry[key][0]) is initial_missing:
                self._misses += 1
                if self._maxsize is not None and self._currsize >= self._maxsize:
                    cache_entry.popitem(last=False)
                else:
                    self._currsize += 1

                value = await self.__wrapped__(*args, **kwargs)
                expires_at = (
                    current_time() + self._ttl if self._ttl is not None else None
                )
                cache_entry[key] = value, None, expires_at
            else:
                # Another task filled the cache while we were waiting for the lock
                self._hits += 1
                cache_entry.move_to_end(key)
                value = cast(T, cached_value)

        return value

    def __get__(
        self, instance: object, owner: type | None = None
    ) -> _LRUMethodWrapper[T]:
        wrapper = _LRUMethodWrapper(self, instance)
        update_wrapper(wrapper, self.__wrapped__)
        return wrapper


class _LRUCacheWrapper:
    def __init__(
        self, maxsize: int | None, typed: bool, always_checkpoint: bool, ttl: int | None
    ):
        self._maxsize = maxsize
        self._typed = typed
        self._always_checkpoint = always_checkpoint
        self._ttl = ttl

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Coroutine[Any, Any, T]], /
    ) -> AsyncLRUCacheWrapper[P, T]: ...

    @overload
    def __call__(
        self, func: Callable[..., T], /
    ) -> functools._lru_cache_wrapper[T]: ...

    def __call__(
        self, f: Callable[P, Coroutine[Any, Any, T]] | Callable[..., T], /
    ) -> AsyncLRUCacheWrapper[P, T] | functools._lru_cache_wrapper[T]:
        if iscoroutinefunction(f):
            return AsyncLRUCacheWrapper(
                f, self._maxsize, self._typed, self._always_checkpoint, self._ttl
            )

        return functools.lru_cache(maxsize=self._maxsize, typed=self._typed)(f)  # type: ignore[arg-type]


@overload
def cache(  # type: ignore[overload-overlap]
    func: Callable[P, Coroutine[Any, Any, T]], /
) -> AsyncLRUCacheWrapper[P, T]: ...


@overload
def cache(func: Callable[..., T], /) -> functools._lru_cache_wrapper[T]: ...


def cache(func: Callable[..., Any] | Callable[P, Coroutine[Any, Any, Any]], /) -> Any:
    """
    A convenient shortcut for :func:`lru_cache` with ``maxsize=None``.

    This is the asynchronous equivalent to :func:`functools.cache`.

    """
    return lru_cache(maxsize=None)(func)


@overload
def lru_cache(
    *,
    maxsize: int | None = ...,
    typed: bool = ...,
    always_checkpoint: bool = ...,
    ttl: int | None = ...,
) -> _LRUCacheWrapper: ...


@overload
def lru_cache(  # type: ignore[overload-overlap]
    func: Callable[P, Coroutine[Any, Any, T]], /
) -> AsyncLRUCacheWrapper[P, T]: ...


@overload
def lru_cache(func: Callable[..., T], /) -> functools._lru_cache_wrapper[T]: ...


def lru_cache(
    func: Callable[..., Coroutine[Any, Any, Any]] | Callable[..., Any] | None = None,
    /,
    *,
    maxsize: int | None = 128,
    typed: bool = False,
    always_checkpoint: bool = False,
    ttl: int | None = None,
) -> Any:
    """
    An asynchronous version of :func:`functools.lru_cache`.

    If a synchronous function is passed, the standard library
    :func:`functools.lru_cache` is applied instead.

    :param always_checkpoint: if ``True``, every call to the cached function will be
        guaranteed to yield control to the event loop at least once
    :param ttl: time in seconds after which to invalidate cache entries

    .. note:: Caches and locks are managed on a per-event loop basis.

    """
    if func is None:
        return _LRUCacheWrapper(maxsize, typed, always_checkpoint, ttl)

    if not callable(func):
        raise TypeError("the first argument must be callable")

    return _LRUCacheWrapper(maxsize, typed, always_checkpoint, ttl)(func)


@overload
async def reduce(
    function: Callable[[T, S], Awaitable[T]],
    iterable: Iterable[S] | AsyncIterable[S],
    /,
    initial: T,
) -> T: ...


@overload
async def reduce(
    function: Callable[[T, T], Awaitable[T]],
    iterable: Iterable[T] | AsyncIterable[T],
    /,
) -> T: ...


async def reduce(  # type: ignore[misc]
    function: Callable[[T, T], Awaitable[T]] | Callable[[T, S], Awaitable[T]],
    iterable: Iterable[T] | Iterable[S] | AsyncIterable[T] | AsyncIterable[S],
    /,
    initial: T | _InitialMissingType = initial_missing,
) -> T:
    """
    Asynchronous version of :func:`functools.reduce`.

    :param function: a coroutine function that takes two arguments: the accumulated
        value and the next element from the iterable
    :param iterable: an iterable or async iterable
    :param initial: the initial value (if missing, the first element of the iterable is
        used as the initial value)

    """
    element: Any
    function_called = False
    if isinstance(iterable, AsyncIterable):
        async_it = iterable.__aiter__()
        if initial is initial_missing:
            try:
                value = cast(T, await async_it.__anext__())
            except StopAsyncIteration:
                raise TypeError(
                    "reduce() of empty sequence with no initial value"
                ) from None
        else:
            value = cast(T, initial)

        async for element in async_it:
            value = await function(value, element)
            function_called = True
    elif isinstance(iterable, Iterable):
        it = iter(iterable)
        if initial is initial_missing:
            try:
                value = cast(T, next(it))
            except StopIteration:
                raise TypeError(
                    "reduce() of empty sequence with no initial value"
                ) from None
        else:
            value = cast(T, initial)

        for element in it:
            value = await function(value, element)
            function_called = True
    else:
        raise TypeError("reduce() argument 2 must be an iterable or async iterable")

    # Make sure there is at least one checkpoint, even if an empty iterable and an
    # initial value were given
    if not function_called:
        await checkpoint()

    return value
<|endoftext|>from __future__ import annotations

__all__ = (
    "current_default_process_limiter",
    "process_worker",
    "run_sync",
)

import os
import pickle
import runpy
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from types import ModuleType
from typing import TypeVar, cast

from ._core._eventloop import current_time, get_async_backend, get_cancelled_exc_class
from ._core._exceptions import BrokenWorkerProcess
from ._core._subprocesses import open_process
from ._core._synchronization import CapacityLimiter
from ._core._tasks import CancelScope, fail_after
from .abc import ByteReceiveStream, ByteSendStream, Process
from .lowlevel import RunVar, checkpoint_if_cancelled
from .streams.buffered import BufferedByteReceiveStream

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

WORKER_MAX_IDLE_TIME = 300  # 5 minutes

T_Retval = TypeVar("T_Retval")
PosArgsT = TypeVarTuple("PosArgsT")

_process_pool_workers: RunVar[set[Process]] = RunVar("_process_pool_workers")
_process_pool_idle_workers: RunVar[deque[tuple[Process, float]]] = RunVar(
    "_process_pool_idle_workers"
)
_default_process_limiter: RunVar[CapacityLimiter] = RunVar("_default_process_limiter")


async def run_sync(  # type: ignore[return]
    func: Callable[[Unpack[PosArgsT]], T_Retval],
    *args: Unpack[PosArgsT],
    cancellable: bool = False,
    limiter: CapacityLimiter | None = None,
) -> T_Retval:
    """
    Call the given function with the given arguments in a worker process.

    If the ``cancellable`` option is enabled and the task waiting for its completion is
    cancelled, the worker process running it will be abruptly terminated using SIGKILL
    (or ``terminateProcess()`` on Windows).

    :param func: a callable
    :param args: positional arguments for the callable
    :param cancellable: ``True`` to allow cancellation of the operation while it's
        running
    :param limiter: capacity limiter to use to limit the total amount of processes
        running (if omitted, the default limiter is used)
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread
    :return: an awaitable that yields the return value of the function.

    """

    async def send_raw_command(pickled_cmd: bytes) -> object:
        try:
            await stdin.send(pickled_cmd)
            response = await buffered.receive_until(b"\n", 50)
            status, length = response.split(b" ")
            if status not in (b"RETURN", b"EXCEPTION"):
                raise RuntimeError(
                    f"Worker process returned unexpected response: {response!r}"
                )

            pickled_response = await buffered.receive_exactly(int(length))
        except BaseException as exc:
            workers.discard(process)
            try:
                process.kill()
                with CancelScope(shield=True):
                    await process.aclose()
            except ProcessLookupError:
                pass

            if isinstance(exc, get_cancelled_exc_class()):
                raise
            else:
                raise BrokenWorkerProcess from exc

        retval = pickle.loads(pickled_response)
        if status == b"EXCEPTION":
            assert isinstance(retval, BaseException)
            raise retval
        else:
            return retval

    # First pickle the request before trying to reserve a worker process
    await checkpoint_if_cancelled()
    request = pickle.dumps(("run", func, args), protocol=pickle.HIGHEST_PROTOCOL)

    # If this is the first run in this event loop thread, set up the necessary variables
    try:
        workers = _process_pool_workers.get()
        idle_workers = _process_pool_idle_workers.get()
    except LookupError:
        workers = set()
        idle_workers = deque()
        _process_pool_workers.set(workers)
        _process_pool_idle_workers.set(idle_workers)
        get_async_backend().setup_process_pool_exit_at_shutdown(workers)

    async with limiter or current_default_process_limiter():
        # Pop processes from the pool (starting from the most recently used) until we
        # find one that hasn't exited yet
        process: Process
        while idle_workers:
            process, _idle_since = idle_workers.pop()
            if process.returncode is None:
                stdin = cast(ByteSendStream, process.stdin)
                buffered = BufferedByteReceiveStream(
                    cast(ByteReceiveStream, process.stdout)
                )

                # Prune any other workers that have been idle for WORKER_MAX_IDLE_TIME
                # seconds or longer
                now = current_time()
                killed_processes: list[Process] = []
                while idle_workers:
                    if now - idle_workers[0][1] < WORKER_MAX_IDLE_TIME:
                        break

                    process_to_kill, _idle_since = idle_workers.popleft()
                    process_to_kill.kill()
                    workers.remove(process_to_kill)
                    killed_processes.append(process_to_kill)

                with CancelScope(shield=True):
                    for killed_process in killed_processes:
                        await killed_process.aclose()

                break

            workers.remove(process)
        else:
            command = [sys.executable, "-u", "-m", __name__]
            process = await open_process(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
            try:
                stdin = cast(ByteSendStream, process.stdin)
                buffered = BufferedByteReceiveStream(
                    cast(ByteReceiveStream, process.stdout)
                )
                with fail_after(20):
                    message = await buffered.receive(6)

                if message != b"READY\n":
                    raise BrokenWorkerProcess(
                        f"Worker process returned unexpected response: {message!r}"
                    )

                main_module_path = getattr(sys.modules["__main__"], "__file__", None)
                pickled = pickle.dumps(
                    ("init", sys.path, main_module_path),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                await send_raw_command(pickled)
            except (BrokenWorkerProcess, get_cancelled_exc_class()):
                raise
            except BaseException as exc:
                process.kill()
                raise BrokenWorkerProcess(
                    "Error during worker process initialization"
                ) from exc

            workers.add(process)

        with CancelScope(shield=not cancellable):
            try:
                return cast(T_Retval, await send_raw_command(request))
            finally:
                if process in workers:
                    idle_workers.append((process, current_time()))


def current_default_process_limiter() -> CapacityLimiter:
    """
    Return the capacity limiter that is used by default to limit the number of worker
    processes.

    :return: a capacity limiter object

    """
    try:
        return _default_process_limiter.get()
    except LookupError:
        limiter = CapacityLimiter(os.cpu_count() or 2)
        _default_process_limiter.set(limiter)
        return limiter


def process_worker() -> None:
    # Redirect standard streams to os.devnull so that user code won't interfere with the
    # parent-worker communication
    stdin = sys.stdin
    stdout = sys.stdout
    sys.stdin = open(os.devnull)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    stdout.buffer.write(b"READY\n")
    while True:
        retval = exception = None
        try:
            command, *args = pickle.load(stdin.buffer)
        except EOFError:
            return
        except BaseException as exc:
            exception = exc
        else:
            if command == "run":
                func, args = args
                try:
                    retval = func(*args)
                except BaseException as exc:
                    exception = exc
            elif command == "init":
                main_module_path: str | None
                sys.path, main_module_path = args
                del sys.modules["__main__"]
                if main_module_path and os.path.isfile(main_module_path):
                    # Load the parent's main module but as __mp_main__ instead of
                    # __main__ (like multiprocessing does) to avoid infinite recursion
                    try:
                        main = ModuleType("__mp_main__")
                        main_content = runpy.run_path(
                            main_module_path, run_name="__mp_main__"
                        )
                        main.__dict__.update(main_content)
                        sys.modules["__main__"] = sys.modules["__mp_main__"] = main
                    except BaseException as exc:
                        exception = exc
        try:
            if exception is not None:
                status = b"EXCEPTION"
                pickled = pickle.dumps(exception, pickle.HIGHEST_PROTOCOL)
            else:
                status = b"RETURN"
                pickled = pickle.dumps(retval, pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            exception = exc
            status = b"EXCEPTION"
            pickled = pickle.dumps(exc, pickle.HIGHEST_PROTOCOL)

        stdout.buffer.write(b"%s %d\n" % (status, len(pickled)))
        stdout.buffer.write(pickled)

        # Respect SIGTERM
        if isinstance(exception, SystemExit):
            raise exception


if __name__ == "__main__":
    process_worker()
<|endoftext|>from __future__ import annotations

__all__ = (
    "current_default_thread_limiter",
    "run_sync",
)

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar
from warnings import warn

from ._core._eventloop import get_async_backend

if TYPE_CHECKING:
    from ._core._synchronization import CapacityLimiter

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

T_Retval = TypeVar("T_Retval")
PosArgsT = TypeVarTuple("PosArgsT")


async def run_sync(
    func: Callable[[Unpack[PosArgsT]], T_Retval],
    *args: Unpack[PosArgsT],
    abandon_on_cancel: bool = False,
    cancellable: bool | None = None,
    limiter: CapacityLimiter | None = None,
) -> T_Retval:
    """
    Call the given function with the given arguments in a worker thread.

    If the ``abandon_on_cancel`` option is enabled and the task waiting for its
    completion is cancelled, the thread will still run its course but its
    return value (or any raised exception) will be ignored.

    :param func: a callable
    :param args: positional arguments for the callable
    :param abandon_on_cancel: ``True`` to abandon the thread (leaving it to run
        unchecked on own) if the host task is cancelled, ``False`` to ignore
        cancellations in the host task until the operation has completed in the worker
        thread
    :param cancellable: deprecated alias of ``abandon_on_cancel``; will override
        ``abandon_on_cancel`` if both parameters are passed
    :param limiter: capacity limiter to use to limit the total amount of threads running
        (if omitted, the default limiter is used)
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread
    :return: an awaitable that yields the return value of the function.

    """
    if cancellable is not None:
        abandon_on_cancel = cancellable
        warn(
            "The `cancellable=` keyword argument to `anyio.to_thread.run_sync` is "
            "deprecated since AnyIO 4.1.0; use `abandon_on_cancel=` instead",
            DeprecationWarning,
            stacklevel=2,
        )

    return await get_async_backend().run_sync_in_worker_thread(
        func, args, abandon_on_cancel=abandon_on_cancel, limiter=limiter
    )


def current_default_thread_limiter() -> CapacityLimiter:
    """
    Return the capacity limiter that is used by default to limit the number of
    concurrent threads.

    :return: a capacity limiter object
    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread

    """
    return get_async_backend().current_default_thread_limiter()
<|endoftext|>from __future__ import annotations

import array
import math
import os
import socket
import sys
import types
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Sequence,
)
from contextlib import AbstractContextManager
from contextvars import Context
from dataclasses import dataclass
from functools import partial, wraps
from io import IOBase
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind
from types import TracebackType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

import trio.from_thread
import trio.lowlevel
from outcome import Error, Outcome, Value
from trio.lowlevel import (
    current_root_task,
    current_task,
    notify_closing,
    wait_readable,
    wait_writable,
)
from trio.socket import SocketType as TrioSocketType
from trio.to_thread import run_sync

from .. import (
    CapacityLimiterStatistics,
    EventStatistics,
    LockStatistics,
    RunFinishedError,
    TaskInfo,
    WouldBlock,
    abc,
)
from .._core._eventloop import claim_worker_thread
from .._core._exceptions import (
    BrokenResourceError,
    BusyResourceError,
    ClosedResourceError,
    EndOfStream,
)
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import (
    CapacityLimiter as BaseCapacityLimiter,
)
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import Lock as BaseLock
from .._core._synchronization import (
    ResourceGuard,
    SemaphoreStatistics,
)
from .._core._synchronization import Semaphore as BaseSemaphore
from .._core._tasks import CancelScope as BaseCancelScope
from .._core._tasks import TaskHandle
from ..abc import IPSockAddrType, UDPPacketType, UNIXDatagramPacketType
from ..abc._eventloop import AsyncBackend, StrOrBytesPath
from ..abc._tasks import T_contra, call_for_coroutine, get_callable_name, get_coro_name

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike

    from ..streams.memory import MemoryObjectSendStream

if sys.version_info < (3, 15):
    from typing_extensions import sentinel

if sys.version_info >= (3, 11):
    from typing import Self, TypeVarTuple, Unpack
else:
    from exceptiongroup import BaseExceptionGroup
    from typing_extensions import Self, TypeVarTuple, Unpack

T = TypeVar("T")
T_Retval = TypeVar("T_Retval")
T_co = TypeVar("T_co", covariant=True)
T_SockAddr = TypeVar("T_SockAddr", str, IPSockAddrType)
PosArgsT = TypeVarTuple("PosArgsT")
P = ParamSpec("P")


def ensure_returns_coro(
    func: Callable[P, Awaitable[T_Retval]],
) -> Callable[P, Coroutine[Any, Any, T_Retval]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, T_Retval]:
        awaitable = func(*args, **kwargs)
        # Check the common case first.
        if isinstance(awaitable, Coroutine):
            return awaitable
        elif not isinstance(awaitable, Awaitable):
            # The user violated the type annotations. Still, we should pass this on to
            # Trio so it can raise with an appropriate message.
            return awaitable
        else:

            @wraps(func)
            async def inner_wrapper() -> T_Retval:
                return await awaitable

            return inner_wrapper()

    return wrapper


#
# Event loop
#

RunVar = trio.lowlevel.RunVar


#
# Timeouts and cancellation
#


class CancelScope(BaseCancelScope):
    __slots__ = ("__original",)

    def __new__(
        cls, original: trio.CancelScope | None = None, **kwargs: object
    ) -> Self:
        return object.__new__(cls)

    def __init__(self, original: trio.CancelScope | None = None, **kwargs: Any) -> None:
        self.__original = original or trio.CancelScope(**kwargs)

    def __enter__(self) -> Self:
        self.__original.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self.__original.__exit__(exc_type, exc_val, exc_tb)

    def cancel(self, reason: str | None = None) -> None:
        self.__original.cancel(reason)

    @property
    def deadline(self) -> float:
        return self.__original.deadline

    @deadline.setter
    def deadline(self, value: float) -> None:
        self.__original.deadline = value

    @property
    def cancel_called(self) -> bool:
        return self.__original.cancel_called

    @property
    def cancelled_caught(self) -> bool:
        return self.__original.cancelled_caught

    @property
    def shield(self) -> bool:
        return self.__original.shield

    @shield.setter
    def shield(self, value: bool) -> None:
        self.__original.shield = value


#
# Task groups
#

empty_start_value = sentinel("empty_start_value")


class _TrioTaskStatus(abc.TaskStatus[T_contra], Generic[T_contra]):
    early_start_value: T_contra | object = empty_start_value
    real_task_status: trio.TaskStatus[T_contra | None] | None = None

    def started(self, value: T_contra | None = None) -> None:
        if self.real_task_status is None:
            if self.early_start_value is not empty_start_value:
                raise RuntimeError("called 'started' twice on the same task status")

            self.early_start_value = value
        else:
            self.real_task_status.started(value)


class TaskGroup(abc.TaskGroup):
    def __init__(self) -> None:
        self._entered = False
        self._active = False
        self._nursery_manager = trio.open_nursery(strict_exception_groups=True)
        self.cancel_scope = None  # type: ignore[assignment]

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("TaskGroup cannot be entered more than once")

        self._entered = True
        self._active = True
        self._nursery = await self._nursery_manager.__aenter__()
        self.cancel_scope = CancelScope(self._nursery.cancel_scope)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            # trio.Nursery.__exit__ returns bool; .open_nursery has wrong type
            return await self._nursery_manager.__aexit__(exc_type, exc_val, exc_tb)  # type: ignore[return-value]
        except BaseExceptionGroup as exc:
            if not exc.split(trio.Cancelled)[1]:
                raise trio.Cancelled._create() from exc

            raise
        finally:
            del exc_val, exc_tb
            self._active = False

    def _check_active(self, coro: Coroutine | None = None) -> None:
        if not self._active:
            if coro is not None:
                coro.close()

            raise RuntimeError(
                "This task group is not active; no new tasks can be started."
            )

    def create_task(
        self,
        coro: Coroutine[Any, Any, T_co],
        *,
        name: object = None,
        context: Context | None = None,
    ) -> TaskHandle[T_co]:
        if not isinstance(coro, Coroutine):
            raise TypeError(f"expected a coroutine, got {coro.__class__.__qualname__}")

        self._check_active(coro)
        final_name = get_coro_name(coro, name)
        handle = TaskHandle(coro, final_name)
        if context is not None:
            context.run(
                partial(self._nursery.start_soon, handle._run_coro, name=handle.name)
            )
        else:
            self._nursery.start_soon(handle._run_coro, name=handle.name)

        return handle

    async def start(
        self,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        *args: Unpack[PosArgsT],
        name: object = None,
        return_handle: Literal[False, True] = False,
    ) -> Any:
        handle: TaskHandle[T_co]

        async def run_coro_with_task_status(
            *, task_status: trio.TaskStatus[Any]
        ) -> None:
            nonlocal handle
            wrapper_task_status = _TrioTaskStatus()
            coro = call_for_coroutine(func, args, task_status=wrapper_task_status)
            if wrapper_task_status.early_start_value is not empty_start_value:
                task_status.started(wrapper_task_status.early_start_value)
            else:
                wrapper_task_status.real_task_status = task_status

            handle = TaskHandle(coro, final_name)
            await handle._run_coro()

        self._check_active()
        final_name = get_callable_name(func, name)
        start_value = await self._nursery.start(
            run_coro_with_task_status, name=final_name
        )
        if return_handle:
            handle._start_value = start_value
            return handle
        else:
            return start_value


#
# Subprocesses
#


@dataclass(eq=False)
class ReceiveStreamWrapper(abc.ByteReceiveStream):
    _stream: trio.abc.ReceiveStream

    async def receive(self, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        try:
            data = await self._stream.receive_some(max_bytes)
        except trio.ClosedResourceError as exc:
            raise ClosedResourceError from exc.__cause__
        except trio.BrokenResourceError as exc:
            raise BrokenResourceError from exc.__cause__

        if data:
            return bytes(data)
        else:
            raise EndOfStream

    async def aclose(self) -> None:
        await self._stream.aclose()


@dataclass(eq=False)
class SendStreamWrapper(abc.ByteSendStream):
    _stream: trio.abc.SendStream

    async def send(self, item: bytes) -> None:
        try:
            await self._stream.send_all(item)
        except trio.ClosedResourceError as exc:
            raise ClosedResourceError from exc.__cause__
        except trio.BrokenResourceError as exc:
            raise BrokenResourceError from exc.__cause__

    async def aclose(self) -> None:
        await self._stream.aclose()


@dataclass(eq=False)
class Process(abc.Process):
    _process: trio.Process
    _stdin: abc.ByteSendStream | None
    _stdout: abc.ByteReceiveStream | None
    _stderr: abc.ByteReceiveStream | None

    async def aclose(self) -> None:
        with CancelScope(shield=True):
            if self._stdin:
                await self._stdin.aclose()
            if self._stdout:
                await self._stdout.aclose()
            if self._stderr:
                await self._stderr.aclose()

        try:
            await self.wait()
        except BaseException:
            self.kill()
            with CancelScope(shield=True):
                await self.wait()
            raise

    async def wait(self) -> int:
        return await self._process.wait()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def send_signal(self, signal: Signals) -> None:
        self._process.send_signal(signal)

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    @property
    def stdin(self) -> abc.ByteSendStream | None:
        return self._stdin

    @property
    def stdout(self) -> abc.ByteReceiveStream | None:
        return self._stdout

    @property
    def stderr(self) -> abc.ByteReceiveStream | None:
        return self._stderr


class _ProcessPoolShutdownInstrument(trio.abc.Instrument):
    def after_run(self) -> None:
        super().after_run()


current_default_worker_process_limiter: trio.lowlevel.RunVar = RunVar(
    "current_default_worker_process_limiter"
)


async def _shutdown_process_pool(workers: set[abc.Process]) -> None:
    try:
        await trio.sleep(math.inf)
    except trio.Cancelled:
        for process in workers:
            if process.returncode is None:
                process.kill()

        with CancelScope(shield=True):
            for process in workers:
                await process.aclose()


#
# Sockets and networking
#


class _TrioSocketMixin(Generic[T_SockAddr]):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        self._trio_socket = trio_socket
        self._closed = False

    def _check_closed(self) -> None:
        if self._closed:
            raise ClosedResourceError
        if self._trio_socket.fileno() < 0:
            raise BrokenResourceError

    @property
    def _raw_socket(self) -> socket.socket:
        return self._trio_socket._sock  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        if self._trio_socket.fileno() >= 0:
            self._closed = True
            self._trio_socket.close()

    def _convert_socket_error(self, exc: BaseException) -> NoReturn:
        if isinstance(exc, trio.ClosedResourceError):
            raise ClosedResourceError from exc
        elif self._trio_socket.fileno() < 0 and self._closed:
            raise ClosedResourceError from None
        elif isinstance(exc, OSError):
            raise BrokenResourceError from exc
        else:
            raise exc


class SocketStream(_TrioSocketMixin, abc.SocketStream):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        super().__init__(trio_socket)
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        with self._receive_guard:
            try:
                data = await self._trio_socket.recv(max_bytes)
            except BaseException as exc:
                self._convert_socket_error(exc)

            if data:
                return data
            else:
                raise EndOfStream

    async def send(self, item: bytes) -> None:
        with self._send_guard:
            view = memoryview(item)
            while view:
                try:
                    bytes_sent = await self._trio_socket.send(view)
                except BaseException as exc:
                    self._convert_socket_error(exc)

                view = view[bytes_sent:]

    async def send_eof(self) -> None:
        self._trio_socket.shutdown(socket.SHUT_WR)


class UNIXSocketStream(SocketStream, abc.UNIXSocketStream):
    async def receive_fds(self, msglen: int, maxfds: int) -> tuple[bytes, list[int]]:
        if not isinstance(msglen, int) or msglen < 0:
            raise ValueError("msglen must be a non-negative integer")
        if not isinstance(maxfds, int) or maxfds < 1:
            raise ValueError("maxfds must be a positive integer")

        fds = array.array("i")
        await trio.lowlevel.checkpoint()
        with self._receive_guard:
            while True:
                try:
                    message, ancdata, _flags, _addr = await self._trio_socket.recvmsg(
                        msglen, socket.CMSG_LEN(maxfds * fds.itemsize)
                    )
                except BaseException as exc:
                    self._convert_socket_error(exc)
                else:
                    if not message and not ancdata:
                        raise EndOfStream

                    break

        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level != socket.SOL_SOCKET or cmsg_type != socket.SCM_RIGHTS:
                raise RuntimeError(
                    f"Received unexpected ancillary data; message = {message!r}, "
                    f"cmsg_level = {cmsg_level}, cmsg_type = {cmsg_type}"
                )

            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])

        return message, list(fds)

    async def send_fds(self, message: bytes, fds: Collection[int | IOBase]) -> None:
        if not message:
            raise ValueError("message must not be empty")
        if not fds:
            raise ValueError("fds must not be empty")

        filenos: list[int] = []
        for fd in fds:
            if isinstance(fd, int):
                filenos.append(fd)
            elif isinstance(fd, IOBase):
                filenos.append(fd.fileno())

        fdarray = array.array("i", filenos)
        await trio.lowlevel.checkpoint()
        with self._send_guard:
            while True:
                try:
                    await self._trio_socket.sendmsg(
                        [message],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                fdarray,
                            )
                        ],
                    )
                    break
                except BaseException as exc:
                    self._convert_socket_error(exc)


class TCPSocketListener(_TrioSocketMixin, abc.SocketListener):
    def __init__(self, raw_socket: socket.socket):
        super().__init__(trio.socket.from_stdlib_socket(raw_socket))
        self._accept_guard = ResourceGuard("accepting connections from")

    async def accept(self) -> SocketStream:
        with self._accept_guard:
            try:
                trio_socket, _addr = await self._trio_socket.accept()
            except BaseException as exc:
                self._convert_socket_error(exc)

        trio_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SocketStream(trio_socket)


class UNIXSocketListener(_TrioSocketMixin, abc.SocketListener):
    def __init__(self, raw_socket: socket.socket):
        super().__init__(trio.socket.from_stdlib_socket(raw_socket))
        self._accept_guard = ResourceGuard("accepting connections from")

    async def accept(self) -> UNIXSocketStream:
        with self._accept_guard:
            try:
                trio_socket, _addr = await self._trio_socket.accept()
            except BaseException as exc:
                self._convert_socket_error(exc)

        return UNIXSocketStream(trio_socket)


class UDPSocket(_TrioSocketMixin[IPSockAddrType], abc.UDPSocket):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        super().__init__(trio_socket)
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    async def receive(self) -> tuple[bytes, IPSockAddrType]:
        with self._receive_guard:
            try:
                data, addr = await self._trio_socket.recvfrom(65536)
                return data, convert_ipv6_sockaddr(addr)
            except BaseException as exc:
                self._convert_socket_error(exc)

    async def send(self, item: UDPPacketType) -> None:
        with self._send_guard:
            try:
                await self._trio_socket.sendto(*item)
            except BaseException as exc:
                self._convert_socket_error(exc)


class ConnectedUDPSocket(_TrioSocketMixin[IPSockAddrType], abc.ConnectedUDPSocket):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        super().__init__(trio_socket)
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    async def receive(self) -> bytes:
        with self._receive_guard:
            try:
                return await self._trio_socket.recv(65536)
            except BaseException as exc:
                self._convert_socket_error(exc)

    async def send(self, item: bytes) -> None:
        with self._send_guard:
            try:
                await self._trio_socket.send(item)
            except BaseException as exc:
                self._convert_socket_error(exc)


class UNIXDatagramSocket(_TrioSocketMixin[str], abc.UNIXDatagramSocket):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        super().__init__(trio_socket)
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    async def receive(self) -> UNIXDatagramPacketType:
        with self._receive_guard:
            try:
                data, addr = await self._trio_socket.recvfrom(65536)
                return data, addr
            except BaseException as exc:
                self._convert_socket_error(exc)

    async def send(self, item: UNIXDatagramPacketType) -> None:
        with self._send_guard:
            try:
                await self._trio_socket.sendto(*item)
            except BaseException as exc:
                self._convert_socket_error(exc)


class ConnectedUNIXDatagramSocket(
    _TrioSocketMixin[str], abc.ConnectedUNIXDatagramSocket
):
    def __init__(self, trio_socket: TrioSocketType) -> None:
        super().__init__(trio_socket)
        self._receive_guard = ResourceGuard("reading from")
        self._send_guard = ResourceGuard("writing to")

    async def receive(self) -> bytes:
        with self._receive_guard:
            try:
                return await self._trio_socket.recv(65536)
            except BaseException as exc:
                self._convert_socket_error(exc)

    async def send(self, item: bytes) -> None:
        with self._send_guard:
            try:
                await self._trio_socket.send(item)
            except BaseException as exc:
                self._convert_socket_error(exc)


#
# Synchronization
#


class Event(BaseEvent):
    __slots__ = ("__original",)

    def __new__(cls) -> Self:
        return object.__new__(cls)

    def __init__(self) -> None:
        self.__original = trio.Event()

    def is_set(self) -> bool:
        return self.__original.is_set()

    async def wait(self) -> None:
        return await self.__original.wait()

    def statistics(self) -> EventStatistics:
        orig_statistics = self.__original.statistics()
        return EventStatistics(tasks_waiting=orig_statistics.tasks_waiting)

    def set(self) -> None:
        self.__original.set()


class Lock(BaseLock):
    __slots__ = "__original", "_fast_acquire"

    def __new__(cls, *, fast_acquire: bool = False) -> Self:
        return object.__new__(cls)

    def __init__(self, *, fast_acquire: bool = False) -> None:
        self._fast_acquire = fast_acquire
        self.__original = trio.Lock()

    @staticmethod
    def _convert_runtime_error_msg(exc: RuntimeError) -> None:
        if exc.args == ("attempt to re-acquire an already held Lock",):
            exc.args = ("Attempted to acquire an already held Lock",)

    async def acquire(self) -> None:
        if not self._fast_acquire:
            try:
                await self.__original.acquire()
            except RuntimeError as exc:
                self._convert_runtime_error_msg(exc)
                raise

            return

        # This is the "fast path" where we don't let other tasks run
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.__original.acquire_nowait()
        except trio.WouldBlock:
            await self.__original._lot.park()
        except RuntimeError as exc:
            self._convert_runtime_error_msg(exc)
            raise

    def acquire_nowait(self) -> None:
        try:
            self.__original.acquire_nowait()
        except trio.WouldBlock:
            raise WouldBlock from None
        except RuntimeError as exc:
            self._convert_runtime_error_msg(exc)
            raise

    def locked(self) -> bool:
        return self.__original.locked()

    def release(self) -> None:
        self.__original.release()

    def statistics(self) -> LockStatistics:
        orig_statistics = self.__original.statistics()
        owner = TrioTaskInfo(orig_statistics.owner) if orig_statistics.owner else None
        return LockStatistics(
            orig_statistics.locked, owner, orig_statistics.tasks_waiting
        )


class Semaphore(BaseSemaphore):
    __slots__ = ("__original",)

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
        self.__original = trio.Semaphore(initial_value, max_value=max_value)

    async def acquire(self) -> None:
        if not self._fast_acquire:
            await self.__original.acquire()
            return

        # This is the "fast path" where we don't let other tasks run
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.__original.acquire_nowait()
        except trio.WouldBlock:
            await self.__original._lot.park()

    def acquire_nowait(self) -> None:
        try:
            self.__original.acquire_nowait()
        except trio.WouldBlock:
            raise WouldBlock from None

    @property
    def max_value(self) -> int | None:
        return self.__original.max_value

    @property
    def value(self) -> int:
        return self.__original.value

    def release(self) -> None:
        self.__original.release()

    def statistics(self) -> SemaphoreStatistics:
        orig_statistics = self.__original.statistics()
        return SemaphoreStatistics(orig_statistics.tasks_waiting)


class CapacityLimiter(BaseCapacityLimiter):
    __slots__ = ("__original",)

    def __new__(
        cls,
        total_tokens: float | None = None,
        *,
        original: trio.CapacityLimiter | None = None,
    ) -> Self:
        return object.__new__(cls)

    def __init__(
        self,
        total_tokens: float | None = None,
        *,
        original: trio.CapacityLimiter | None = None,
    ) -> None:
        if original is not None:
            self.__original = original
        else:
            assert total_tokens is not None
            self.__original = trio.CapacityLimiter(total_tokens)

    async def __aenter__(self) -> None:
        return await self.__original.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.__original.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def total_tokens(self) -> float:
        return self.__original.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        self.__original.total_tokens = value

    @property
    def borrowed_tokens(self) -> int:
        return self.__original.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        return self.__original.available_tokens

    def acquire_nowait(self) -> None:
        try:
            self.__original.acquire_nowait()
        except trio.WouldBlock:
            raise WouldBlock from None

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        try:
            self.__original.acquire_on_behalf_of_nowait(borrower)
        except trio.WouldBlock:
            raise WouldBlock from None

    async def acquire(self) -> None:
        await self.__original.acquire()

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        await self.__original.acquire_on_behalf_of(borrower)

    def release(self) -> None:
        return self.__original.release()

    def release_on_behalf_of(self, borrower: object) -> None:
        return self.__original.release_on_behalf_of(borrower)

    def statistics(self) -> CapacityLimiterStatistics:
        orig = self.__original.statistics()
        return CapacityLimiterStatistics(
            borrowed_tokens=orig.borrowed_tokens,
            total_tokens=orig.total_tokens,
            borrowers=tuple(orig.borrowers),
            tasks_waiting=orig.tasks_waiting,
        )


_capacity_limiter_wrapper: trio.lowlevel.RunVar = RunVar("_capacity_limiter_wrapper")


#
# Signal handling
#


class _SignalReceiver:
    _iterator: AsyncIterator[int]

    def __init__(self, signals: tuple[Signals, ...]):
        self._signals = signals

    def __enter__(self) -> Self:
        self._cm = trio.open_signal_receiver(*self._signals)
        self._iterator = self._cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def __aiter__(self) -> _SignalReceiver:
        return self

    async def __anext__(self) -> Signals:
        signum = await self._iterator.__anext__()
        return Signals(signum)


#
# Testing and debugging
#


class TestRunner(abc.TestRunner):
    def __init__(self, **options: Any) -> None:
        from queue import Queue

        self._call_queue: Queue[Callable[[], object]] = Queue()
        self._send_stream: (
            MemoryObjectSendStream[tuple[Awaitable[Any], list[Outcome]]] | None
        ) = None
        self._options = options

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._send_stream:
            self._send_stream.close()
            while self._send_stream is not None:
                self._call_queue.get()()

    def is_running(self) -> bool:
        return trio.lowlevel.in_trio_task()

    async def _run_tests_and_fixtures(self) -> None:
        self._send_stream, receive_stream = create_memory_object_stream[
            tuple[Awaitable[Any], list[Outcome]]
        ](1)
        with receive_stream:
            async for awaitable, outcome_holder in receive_stream:
                try:
                    retval = await awaitable
                except BaseException as exc:
                    outcome_holder.append(Error(exc))
                else:
                    outcome_holder.append(Value(retval))

    def _main_task_finished(self, outcome: object) -> None:
        self._send_stream = None

    def _call_in_runner_task(
        self,
        func: Callable[P, Awaitable[T_Retval]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T_Retval:
        if self._send_stream is None:
            trio.lowlevel.start_guest_run(
                self._run_tests_and_fixtures,
                run_sync_soon_threadsafe=self._call_queue.put,
                done_callback=self._main_task_finished,
                **self._options,
            )
            while self._send_stream is None:
                self._call_queue.get()()

        outcome_holder: list[Outcome] = []
        self._send_stream.send_nowait((func(*args, **kwargs), outcome_holder))
        while not outcome_holder:
            self._call_queue.get()()

        return outcome_holder[0].unwrap()

    def run_asyncgen_fixture(
        self,
        fixture_func: Callable[..., AsyncGenerator[T_Retval, Any]],
        kwargs: dict[str, Any],
    ) -> Iterable[T_Retval]:
        asyncgen = fixture_func(**kwargs)
        fixturevalue: T_Retval = self._call_in_runner_task(asyncgen.asend, None)

        yield fixturevalue

        try:
            self._call_in_runner_task(asyncgen.asend, None)
        except StopAsyncIteration:
            pass
        else:
            self._call_in_runner_task(asyncgen.aclose)
            raise RuntimeError("Async generator fixture did not stop")

    def run_fixture(
        self,
        fixture_func: Callable[..., Coroutine[Any, Any, T_Retval]],
        kwargs: dict[str, Any],
    ) -> T_Retval:
        return self._call_in_runner_task(fixture_func, **kwargs)

    def run_test(
        self, test_func: Callable[..., Coroutine[Any, Any, Any]], kwargs: dict[str, Any]
    ) -> None:
        self._call_in_runner_task(test_func, **kwargs)


class TrioTaskInfo(TaskInfo):
    def __init__(self, task: trio.lowlevel.Task):
        parent_id = None
        if task.parent_nursery and task.parent_nursery.parent_task:
            parent_id = id(task.parent_nursery.parent_task)

        super().__init__(id(task), parent_id, task.name, task.coro)
        self._task = weakref.proxy(task)

    def has_pending_cancellation(self) -> bool:
        try:
            return self._task._cancel_status.effectively_cancelled
        except ReferenceError:
            # If the task is no longer around, it surely doesn't have a cancellation
            # pending
            return False


class TrioBackend(AsyncBackend):
    @classmethod
    def run(
        cls,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        options: dict[str, Any],
    ) -> T_Retval:
        assert not kwargs, "unreachable, and not supported by Trio"
        return trio.run(ensure_returns_coro(func), *args, **options)

    @classmethod
    def current_token(cls) -> object:
        return trio.lowlevel.current_trio_token()

    @classmethod
    def current_time(cls) -> float:
        return trio.current_time()

    @classmethod
    def cancelled_exception_class(cls) -> type[BaseException]:
        return trio.Cancelled

    @classmethod
    async def checkpoint(cls) -> None:
        await trio.lowlevel.checkpoint()

    @classmethod
    async def checkpoint_if_cancelled(cls) -> None:
        await trio.lowlevel.checkpoint_if_cancelled()

    @classmethod
    async def cancel_shielded_checkpoint(cls) -> None:
        await trio.lowlevel.cancel_shielded_checkpoint()

    @classmethod
    async def sleep(cls, delay: float) -> None:
        await trio.sleep(delay)

    @classmethod
    def create_cancel_scope(
        cls, *, deadline: float = math.inf, shield: bool = False
    ) -> BaseCancelScope:
        return CancelScope(deadline=deadline, shield=shield)

    @classmethod
    def current_effective_deadline(cls) -> float:
        return trio.current_effective_deadline()

    @classmethod
    def create_task_group(cls) -> abc.TaskGroup:
        return TaskGroup()

    @classmethod
    def create_event(cls) -> BaseEvent:
        return Event()

    @classmethod
    def create_lock(cls, *, fast_acquire: bool) -> Lock:
        return Lock(fast_acquire=fast_acquire)

    @classmethod
    def create_semaphore(
        cls,
        initial_value: int,
        *,
        max_value: int | None = None,
        fast_acquire: bool = False,
    ) -> BaseSemaphore:
        return Semaphore(initial_value, max_value=max_value, fast_acquire=fast_acquire)

    @classmethod
    def create_capacity_limiter(cls, total_tokens: float) -> CapacityLimiter:
        return CapacityLimiter(total_tokens)

    @classmethod
    async def run_sync_in_worker_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        abandon_on_cancel: bool = False,
        limiter: BaseCapacityLimiter | None = None,
    ) -> T_Retval:
        def wrapper() -> T_Retval:
            with claim_worker_thread(TrioBackend, token):
                return func(*args)

        token = TrioBackend.current_token()
        return await run_sync(
            wrapper,
            abandon_on_cancel=abandon_on_cancel,
            limiter=cast(trio.CapacityLimiter, limiter),
        )

    @classmethod
    def check_cancelled(cls) -> None:
        trio.from_thread.check_cancelled()

    @classmethod
    def run_async_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_co:
        trio_token = cast("trio.lowlevel.TrioToken | None", token)
        try:
            return trio.from_thread.run(func, *args, trio_token=trio_token)
        except trio.RunFinishedError:
            raise RunFinishedError from None

    @classmethod
    def run_sync_from_thread(
        cls,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        args: tuple[Unpack[PosArgsT]],
        token: object,
    ) -> T_Retval:
        trio_token = cast("trio.lowlevel.TrioToken | None", token)
        try:
            return trio.from_thread.run_sync(func, *args, trio_token=trio_token)
        except trio.RunFinishedError:
            raise RunFinishedError from None

    @classmethod
    async def open_process(
        cls,
        command: StrOrBytesPath | Sequence[StrOrBytesPath],
        *,
        stdin: int | IO[Any] | None,
        stdout: int | IO[Any] | None,
        stderr: int | IO[Any] | None,
        **kwargs: Any,
    ) -> Process:
        def convert_item(item: StrOrBytesPath) -> str:
            str_or_bytes = os.fspath(item)
            if isinstance(str_or_bytes, str):
                return str_or_bytes
            else:
                return os.fsdecode(str_or_bytes)

        if isinstance(command, (str, bytes, PathLike)):
            process = await trio.lowlevel.open_process(
                convert_item(command),
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=True,
                **kwargs,
            )
        else:
            process = await trio.lowlevel.open_process(
                [convert_item(item) for item in command],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                **kwargs,
            )

        stdin_stream = SendStreamWrapper(process.stdin) if process.stdin else None
        stdout_stream = ReceiveStreamWrapper(process.stdout) if process.stdout else None
        stderr_stream = ReceiveStreamWrapper(process.stderr) if process.stderr else None
        return Process(process, stdin_stream, stdout_stream, stderr_stream)

    @classmethod
    def setup_process_pool_exit_at_shutdown(cls, workers: set[abc.Process]) -> None:
        trio.lowlevel.spawn_system_task(_shutdown_process_pool, workers)

    @classmethod
    async def connect_tcp(
        cls, host: str, port: int, local_address: IPSockAddrType | None = None
    ) -> SocketStream:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        trio_socket = trio.socket.socket(family)
        trio_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if local_address:
            await trio_socket.bind(local_address)

        try:
            await trio_socket.connect((host, port))
        except BaseException:
            trio_socket.close()
            raise

        return SocketStream(trio_socket)

    @classmethod
    async def connect_unix(cls, path: str | bytes) -> abc.UNIXSocketStream:
        trio_socket = trio.socket.socket(socket.AF_UNIX)
        try:
            await trio_socket.connect(path)
        except BaseException:
            trio_socket.close()
            raise

        return UNIXSocketStream(trio_socket)

    @classmethod
    def create_tcp_listener(cls, sock: socket.socket) -> abc.SocketListener:
        return TCPSocketListener(sock)

    @classmethod
    def create_unix_listener(cls, sock: socket.socket) -> abc.SocketListener:
        return UNIXSocketListener(sock)

    @classmethod
    async def create_udp_socket(
        cls,
        family: socket.AddressFamily,
        local_address: IPSockAddrType | None,
        remote_address: IPSockAddrType | None,
        reuse_port: bool,
    ) -> UDPSocket | ConnectedUDPSocket:
        trio_socket = trio.socket.socket(family=family, type=socket.SOCK_DGRAM)

        if reuse_port:
            trio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        if local_address:
            await trio_socket.bind(local_address)

        if remote_address:
            await trio_socket.connect(remote_address)
            return ConnectedUDPSocket(trio_socket)
        else:
            return UDPSocket(trio_socket)

    @classmethod
    @overload
    async def create_unix_datagram_socket(
        cls, raw_socket: socket.socket, remote_path: None
    ) -> abc.UNIXDatagramSocket: ...

    @classmethod
    @overload
    async def create_unix_datagram_socket(
        cls, raw_socket: socket.socket, remote_path: str | bytes
    ) -> abc.ConnectedUNIXDatagramSocket: ...

    @classmethod
    async def create_unix_datagram_socket(
        cls, raw_socket: socket.socket, remote_path: str | bytes | None
    ) -> abc.UNIXDatagramSocket | abc.ConnectedUNIXDatagramSocket:
        trio_socket = trio.socket.from_stdlib_socket(raw_socket)

        if remote_path:
            await trio_socket.connect(remote_path)
            return ConnectedUNIXDatagramSocket(trio_socket)
        else:
            return UNIXDatagramSocket(trio_socket)

    @classmethod
    async def getaddrinfo(
        cls,
        host: bytes | str | None,
        port: str | int | None,
        *,
        family: int | AddressFamily = 0,
        type: int | SocketKind = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> Sequence[
        tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        return await trio.socket.getaddrinfo(host, port, family, type, proto, flags)

    @classmethod
    async def getnameinfo(
        cls, sockaddr: IPSockAddrType, flags: int = 0
    ) -> tuple[str, str]:
        return await trio.socket.getnameinfo(sockaddr, flags)

    @classmethod
    async def wait_readable(cls, obj: FileDescriptorLike) -> None:
        try:
            await wait_readable(obj)
        except trio.ClosedResourceError as exc:
            raise ClosedResourceError().with_traceback(exc.__traceback__) from None
        except trio.BusyResourceError:
            raise BusyResourceError("reading from") from None

    @classmethod
    async def wait_writable(cls, obj: FileDescriptorLike) -> None:
        try:
            await wait_writable(obj)
        except trio.ClosedResourceError as exc:
            raise ClosedResourceError().with_traceback(exc.__traceback__) from None
        except trio.BusyResourceError:
            raise BusyResourceError("writing to") from None

    @classmethod
    def notify_closing(cls, obj: FileDescriptorLike) -> None:
        notify_closing(obj)

    @classmethod
    async def wrap_listener_socket(cls, sock: socket.socket) -> abc.SocketListener:
        if hasattr(socket, "AF_UNIX") and sock.family == socket.AF_UNIX:
            return UNIXSocketListener(sock)

        return TCPSocketListener(sock)

    @classmethod
    async def wrap_stream_socket(cls, sock: socket.socket) -> SocketStream:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return SocketStream(trio_sock)

    @classmethod
    async def wrap_unix_stream_socket(cls, sock: socket.socket) -> UNIXSocketStream:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return UNIXSocketStream(trio_sock)

    @classmethod
    async def wrap_udp_socket(cls, sock: socket.socket) -> UDPSocket:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return UDPSocket(trio_sock)

    @classmethod
    async def wrap_connected_udp_socket(cls, sock: socket.socket) -> ConnectedUDPSocket:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return ConnectedUDPSocket(trio_sock)

    @classmethod
    async def wrap_unix_datagram_socket(cls, sock: socket.socket) -> UNIXDatagramSocket:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return UNIXDatagramSocket(trio_sock)

    @classmethod
    async def wrap_connected_unix_datagram_socket(
        cls, sock: socket.socket
    ) -> ConnectedUNIXDatagramSocket:
        trio_sock = trio.socket.from_stdlib_socket(sock)
        return ConnectedUNIXDatagramSocket(trio_sock)

    @classmethod
    def current_default_thread_limiter(cls) -> CapacityLimiter:
        try:
            return _capacity_limiter_wrapper.get()
        except LookupError:
            limiter = CapacityLimiter(
                original=trio.to_thread.current_default_thread_limiter()
            )
            _capacity_limiter_wrapper.set(limiter)
            return limiter

    @classmethod
    def open_signal_receiver(
        cls, *signals: Signals
    ) -> AbstractContextManager[AsyncIterator[Signals]]:
        return _SignalReceiver(signals)

    @classmethod
    def get_current_task(cls) -> TaskInfo:
        task = current_task()
        return TrioTaskInfo(task)

    @classmethod
    def get_running_tasks(cls) -> Sequence[TaskInfo]:
        root_task = current_root_task()
        assert root_task
        task_infos = [TrioTaskInfo(root_task)]
        nurseries = root_task.child_nurseries
        while nurseries:
            new_nurseries: list[trio.Nursery] = []
            for nursery in nurseries:
                for task in nursery.child_tasks:
                    task_infos.append(TrioTaskInfo(task))
                    new_nurseries.extend(task.child_nurseries)

            nurseries = new_nurseries

        return task_infos

    @classmethod
    async def wait_all_tasks_blocked(cls) -> None:
        from trio.testing import wait_all_tasks_blocked

        await wait_all_tasks_blocked()

    @classmethod
    def create_test_runner(cls, options: dict[str, Any]) -> TestRunner:
        return TestRunner(**options)


backend_class = TrioBackend
<|endoftext|>from __future__ import annotations

import array
import asyncio
import concurrent.futures
import math
import os
import socket
import sys
import threading
import weakref
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    all_tasks,
    create_task,
    current_task,
    get_running_loop,
    sleep,
)
from asyncio.base_events import _run_until_complete_cb  # type: ignore[attr-defined]
from collections import OrderedDict, deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Sequence,
)
from concurrent.futures import Future
from contextlib import AbstractContextManager, suppress
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from functools import partial, wraps
from inspect import (
    CORO_RUNNING,
    CORO_SUSPENDED,
    getcoroutinestate,
)
from io import IOBase
from os import PathLike
from queue import Queue
from signal import Signals
from socket import AddressFamily, SocketKind
from threading import Thread
from types import CodeType, TracebackType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)
from weakref import WeakKeyDictionary

from .. import (
    CapacityLimiterStatistics,
    EventStatistics,
    LockStatistics,
    TaskInfo,
    abc,
)
from .._core._eventloop import (
    claim_worker_thread,
    set_current_async_library,
    threadlocals,
)
from .._core._exceptions import (
    BrokenResourceError,
    BusyResourceError,
    ClosedResourceError,
    EndOfStream,
    RunFinishedError,
    WouldBlock,
)
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import (
    CapacityLimiter as BaseCapacityLimiter,
)
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import Lock as BaseLock
from .._core._synchronization import (
    ResourceGuard,
    SemaphoreStatistics,
)
from .._core._synchronization import Semaphore as BaseSemaphore
from .._core._tasks import CancelScope as BaseCancelScope
from .._core._tasks import TaskHandle
from ..abc import (
    AsyncBackend,
    IPSockAddrType,
    SocketListener,
    UDPPacketType,
    UNIXDatagramPacketType,
)
from ..abc._tasks import call_for_coroutine, get_callable_name, get_coro_name
from ..lowlevel import RunVar, _run_vars

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike

    from ..abc._eventloop import StrOrBytesPath
    from ..streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
else:
    FileDescriptorLike = object

if sys.version_info >= (3, 11):
    from asyncio import Runner
    from typing import Self, TypeVarTuple, Unpack
else:
    import contextvars
    import enum
    import signal
    from asyncio import coroutines, events, exceptions, tasks

    from exceptiongroup import BaseExceptionGroup
    from typing_extensions import Self, TypeVarTuple, Unpack

    class _State(enum.Enum):
        CREATED = "created"
        INITIALIZED = "initialized"
        CLOSED = "closed"

    class Runner:
        # Copied from CPython 3.11
        def __init__(
            self,
            *,
            debug: bool | None = None,
            loop_factory: Callable[[], AbstractEventLoop] | None = None,
        ):
            self._state = _State.CREATED
            self._debug = debug
            self._loop_factory = loop_factory
            self._loop: AbstractEventLoop | None = None
            self._context = None
            self._interrupt_count = 0
            self._set_event_loop = False

        def __enter__(self) -> Self:
            self._lazy_init()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            self.close()

        def close(self) -> None:
            """Shutdown and close event loop."""
            loop = self._loop
            if self._state is not _State.INITIALIZED or loop is None:
                return
            try:
                _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
                else:
                    loop.run_until_complete(_shutdown_default_executor(loop))
            finally:
                if self._set_event_loop:
                    events.set_event_loop(None)
                loop.close()
                self._loop = None
                self._state = _State.CLOSED

        def get_loop(self) -> AbstractEventLoop:
            """Return embedded event loop."""
            self._lazy_init()
            return self._loop

        def run(self, coro: Coroutine[T_Retval], *, context=None) -> T_Retval:
            """Run a coroutine inside the embedded event loop."""
            if not coroutines.iscoroutine(coro):
                raise ValueError(f"a coroutine was expected, got {coro!r}")

            if events._get_running_loop() is not None:
                # fail fast with short traceback
                raise RuntimeError(
                    "Runner.run() cannot be called from a running event loop"
                )

            self._lazy_init()

            if context is None:
                context = self._context
            task = context.run(self._loop.create_task, coro)

            if (
                threading.current_thread() is threading.main_thread()
                and signal.getsignal(signal.SIGINT) is signal.default_int_handler
            ):
                sigint_handler = partial(self._on_sigint, main_task=task)
                try:
                    signal.signal(signal.SIGINT, sigint_handler)
                except ValueError:
                    # `signal.signal` may throw if `threading.main_thread` does
                    # not support signals (e.g. embedded interpreter with signals
                    # not registered - see gh-91880)
                    sigint_handler = None
            else:
                sigint_handler = None

            self._interrupt_count = 0
            try:
                return self._loop.run_until_complete(task)
            except exceptions.CancelledError:
                if self._interrupt_count > 0:
                    uncancel = getattr(task, "uncancel", None)
                    if uncancel is not None and uncancel() == 0:
                        raise KeyboardInterrupt  # noqa: B904
                raise  # CancelledError
            finally:
                if (
                    sigint_handler is not None
                    and signal.getsignal(signal.SIGINT) is sigint_handler
                ):
                    signal.signal(signal.SIGINT, signal.default_int_handler)

        def _lazy_init(self) -> None:
            if self._state is _State.CLOSED:
                raise RuntimeError("Runner is closed")
            if self._state is _State.INITIALIZED:
                return
            if self._loop_factory is None:
                self._loop = events.new_event_loop()
                if not self._set_event_loop:
                    # Call set_event_loop only once to avoid calling
                    # attach_loop multiple times on child watchers
                    events.set_event_loop(self._loop)
                    self._set_event_loop = True
            else:
                self._loop = self._loop_factory()
            if self._debug is not None:
                self._loop.set_debug(self._debug)
            self._context = contextvars.copy_context()
            self._state = _State.INITIALIZED

        def _on_sigint(self, signum, frame, main_task: asyncio.Task) -> None:
            self._interrupt_count += 1
            if self._interrupt_count == 1 and not main_task.done():
                main_task.cancel()
                # wakeup loop if it is blocked by select() with long timeout
                self._loop.call_soon_threadsafe(lambda: None)
                return
            raise KeyboardInterrupt()

    def _cancel_all_tasks(loop: AbstractEventLoop) -> None:
        to_cancel = tasks.all_tasks(loop)
        if not to_cancel:
            return

        for task in to_cancel:
            task.cancel()

        loop.run_until_complete(tasks.gather(*to_cancel, return_exceptions=True))

        for task in to_cancel:
            if task.cancelled():
                continue
            if task.exception() is not None:
                loop.call_exception_handler(
                    {
                        "message": "unhandled exception during asyncio.run() shutdown",
                        "exception": task.exception(),
                        "task": task,
                    }
                )

    async def _shutdown_default_executor(loop: AbstractEventLoop) -> None:
        """Schedule the shutdown of the default executor."""

        def _do_shutdown(future: asyncio.futures.Future) -> None:
            try:
                loop._default_executor.shutdown(wait=True)  # type: ignore[attr-defined]
                loop.call_soon_threadsafe(future.set_result, None)
            except Exception as ex:
                loop.call_soon_threadsafe(future.set_exception, ex)

        loop._executor_shutdown_called = True
        if loop._default_executor is None:
            return
        future = loop.create_future()
        thread = threading.Thread(target=_do_shutdown, args=(future,))
        thread.start()
        try:
            await future
        finally:
            thread.join()


T_Retval = TypeVar("T_Retval")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
PosArgsT = TypeVarTuple("PosArgsT")
P = ParamSpec("P")

_root_task: RunVar[asyncio.Task[Any] | None] = RunVar("_root_task")


def find_root_task() -> asyncio.Task:
    root_task = _root_task.get(None)
    if root_task is not None and not root_task.done():
        return root_task

    # Look for a task that has been started via run_until_complete()
    for task in all_tasks():
        if task._callbacks and not task.done():
            for cb, context in task._callbacks:
                if (
                    cb is _run_until_complete_cb
                    or getattr(cb, "__module__", None) == "uvloop.loop"
                ):
                    _root_task.set(task)

                    def _unset(t: asyncio.Task[Any]) -> None:
                        if vars := _run_vars.get(t.get_loop()):
                            vars.pop(_root_task, None)

                    # Register a callback to break the task -> loop -> _run_var[loop][_root_task] -> task cycle
                    # Also run it in its own context to not create another reference.
                    # We can't use RunVar.reset() here since these are called synchronously
                    # and thus lowlevel.current_token() (which RunVar.reset() depends on) fails.
                    task.add_done_callback(_unset, context=context)
                    return task

    # Look up the topmost task in the AnyIO task tree, if possible
    task = cast(asyncio.Task, current_task())
    state = _task_states.get(task)
    if state:
        cancel_scope = state.cancel_scope
        while cancel_scope and cancel_scope._parent_scope is not None:
            cancel_scope = cancel_scope._parent_scope

        if cancel_scope is not None:
            return cast(asyncio.Task, cancel_scope._host_task)

    return task


def _task_started(task: asyncio.Task) -> bool:
    """Return ``True`` if the task has been started and has not finished."""
    # The task coro should never be None here, as we never add finished tasks to the
    # task list
    coro = task.get_coro()
    assert coro is not None
    return getcoroutinestate(coro) in (CORO_RUNNING, CORO_SUSPENDED)


#
# Timeouts and cancellation
#


def is_anyio_cancellation(exc: CancelledError) -> bool:
    # Sometimes third party frameworks catch a CancelledError and raise a new one, so as
    # a workaround we have to look at the previous ones in __context__ too for a
    # matching cancel message
    while True:
        if (
            exc.args
            and isinstance(exc.args[0], str)
            and exc.args[0].startswith("Cancelled via cancel scope ")
        ):
            return True

        if isinstance(exc.__context__, CancelledError):
            exc = exc.__context__
            continue

        return False


class CancelScope(BaseCancelScope):
    __slots__ = (
        "_active",
        "_cancel_called",
        "_cancel_handle",
        "_cancel_reason",
        "_cancelled_caught",
        "_child_scopes",
        "_deadline",
        "_host_task",
        "_parent_scope",
        "_pending_uncancellations",
        "_shield",
        "_tasks",
        "_timeout_handle",
    )

    def __new__(cls, *, deadline: float = math.inf, shield: bool = False) -> Self:
        return object.__new__(cls)

    def __init__(self, deadline: float = math.inf, shield: bool = False):
        self._deadline = deadline
        self._shield = shield
        self._parent_scope: CancelScope | None = None
        self._child_scopes: set[CancelScope] = set()
        self._cancel_called = False
        self._cancel_reason: str | None = None
        self._cancelled_caught = False
        self._active = False
        self._timeout_handle: asyncio.TimerHandle | None = None
        self._cancel_handle: asyncio.Handle | None = None
        self._tasks: set[asyncio.Task] = set()
        self._host_task: asyncio.Task | None = None
        if sys.version_info >= (3, 11):
            self._pending_uncancellations: int | None = 0
        else:
            self._pending_uncancellations = None

    def __enter__(self) -> Self:
        if self._active:
            raise RuntimeError(
                "Each CancelScope may only be used for a single 'with' block"
            )

        self._host_task = host_task = cast(asyncio.Task, current_task())
        self._tasks.add(host_task)
        try:
            task_state = _task_states[host_task]
        except KeyError:
            task_state = TaskState(None, self)
            _task_states[host_task] = task_state
        else:
            self._parent_scope = task_state.cancel_scope
            task_state.cancel_scope = self
            if self._parent_scope is not None:
                # If using an eager task factory, the parent scope may not even contain
                # the host task
                self._parent_scope._child_scopes.add(self)
                self._parent_scope._tasks.discard(host_task)

        self._timeout()
        self._active = True

        # Start cancelling the host task if the scope was cancelled before entering
        if self._cancel_called:
            self._deliver_cancellation(self)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        del exc_tb

        if not self._active:
            raise RuntimeError("This cancel scope is not active")
        if current_task() is not self._host_task:
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than it was "
                "entered in"
            )

        assert self._host_task is not None
        host_task_state = _task_states.get(self._host_task)
        if host_task_state is None or host_task_state.cancel_scope is not self:
            raise RuntimeError(
                "Attempted to exit a cancel scope that isn't the current tasks's "
                "current cancel scope"
            )

        try:
            self._active = False
            if self._timeout_handle:
                self._timeout_handle.cancel()
                self._timeout_handle = None

            self._tasks.remove(self._host_task)
            if self._parent_scope is not None:
                self._parent_scope._child_scopes.remove(self)
                self._parent_scope._tasks.add(self._host_task)

            host_task_state.cancel_scope = self._parent_scope

            # Restart the cancellation effort in the closest visible, cancelled parent
            # scope if necessary
