        Wrap all synchronous methods of the underlying filesystem with asynchronous versions.
        """
        excluded_methods = {"open"}
        for method_name in dir(self.sync_fs):
            if method_name.startswith("_") or method_name in excluded_methods:
                continue

            attr = inspect.getattr_static(self.sync_fs, method_name)
            if isinstance(attr, property):
                continue

            method = getattr(self.sync_fs, method_name)
            if callable(method) and not inspect.iscoroutinefunction(method):
                async_method = async_wrapper(method, obj=self, semaphore=self.semaphore)
                setattr(self, f"_{method_name}", async_method)

    @classmethod
    def wrap_class(cls, sync_fs_class):
        """
        Create a new class that can be used to instantiate an AsyncFileSystemWrapper
        with lazy instantiation of the underlying synchronous filesystem.

        Parameters
        ----------
        sync_fs_class : type
            The class of the synchronous filesystem to wrap.

        Returns
        -------
        type
            A new class that wraps the provided synchronous filesystem class.
        """

        class GeneratedAsyncFileSystemWrapper(cls):
            def __init__(self, *args, **kwargs):
                sync_fs = sync_fs_class(*args, **kwargs)
                super().__init__(sync_fs)

        GeneratedAsyncFileSystemWrapper.__name__ = (
            f"Async{sync_fs_class.__name__}Wrapper"
        )
        return GeneratedAsyncFileSystemWrapper
<|endoftext|>from __future__ import annotations

import inspect
import logging
import os
import tempfile
import time
import weakref
from collections.abc import Callable
from shutil import rmtree
from typing import TYPE_CHECKING, Any, ClassVar

from fsspec import filesystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.implementations.chained import ChainedFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile
from fsspec.transaction import Transaction
from fsspec.utils import infer_compression

if TYPE_CHECKING:
    from fsspec.implementations.cache_mapper import AbstractCacheMapper

logger = logging.getLogger("fsspec.cached")


class WriteCachedTransaction(Transaction):
    def complete(self, commit=True):
        rpaths = [f.path for f in self.files]
        lpaths = [f.fn for f in self.files]
        if commit:
            self.fs.put(lpaths, rpaths)
        self.files.clear()
        self.fs._intrans = False
        self.fs._transaction = None
        self.fs = None  # break cycle


class CachingFileSystem(ChainedFileSystem):
    """Locally caching filesystem, layer over any other FS

    This class implements chunk-wise local storage of remote files, for quick
    access after the initial download. The files are stored in a given
    directory with hashes of URLs for the filenames. If no directory is given,
    a temporary one is used, which should be cleaned up by the OS after the
    process ends. The files themselves are sparse (as implemented in
    :class:`~fsspec.caching.MMapCache`), so only the data which is accessed
    takes up space.

    Restrictions:

    - the block-size must be the same for each access of a given file, unless
      all blocks of the file have already been read
    - caching can only be applied to file-systems which produce files
      derived from fsspec.spec.AbstractBufferedFile ; LocalFileSystem is also
      allowed, for testing
    """

    protocol: ClassVar[str | tuple[str, ...]] = ("blockcache", "cached")
    _strip_tokenize_options = ("fo",)

    def __init__(
        self,
        target_protocol=None,
        cache_storage="TMP",
        cache_check=10,
        check_files=False,
        expiry_time=604800,
        target_options=None,
        fs=None,
        same_names: bool | None = None,
        compression=None,
        cache_mapper: AbstractCacheMapper | None = None,
        cache_storage_mode=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        target_protocol: str (optional)
            Target filesystem protocol. Provide either this or ``fs``.
        cache_storage: str or list(str)
            Location to store files. If "TMP", this is a temporary directory,
            and will be cleaned up by the OS when this process ends (or later).
            If a list, each location will be tried in the order given, but
            only the last will be considered writable.
        cache_check: int
            Number of seconds between reload of cache metadata
        check_files: bool
            Whether to explicitly see if the UID of the remote file matches
            the stored one before using. Warning: some file systems such as
            HTTP cannot reliably give a unique hash of the contents of some
            path, so be sure to set this option to False.
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            Set to falsy to prevent expiry. The default is equivalent to one
            week.
        target_options: dict or None
            Passed to the instantiation of the FS, if fs is None.
        fs: filesystem instance
            The target filesystem to run against. Provide this or ``protocol``.
        same_names: bool (optional)
            By default, target URLs are hashed using a ``HashCacheMapper`` so
            that files from different backends with the same basename do not
            conflict. If this argument is ``true``, a ``BasenameCacheMapper``
            is used instead. Other cache mapper options are available by using
            the ``cache_mapper`` keyword argument. Only one of this and
            ``cache_mapper`` should be specified.
        compression: str (optional)
            To decompress on download. Can be 'infer' (guess from the URL name),
            one of the entries in ``fsspec.compression.compr``, or None for no
            decompression.
        cache_mapper: AbstractCacheMapper (optional)
            The object use to map from original filenames to cached filenames.
            Only one of this and ``same_names`` should be specified.
        cache_storage_mode: int (optional)
            Permission mode used when creating the cache storage directory,
            e.g. ``0o700``. The default of ``None`` leaves the directory at
            the system default (governed by the umask), which is the existing
            behaviour. Pass an octal mode such as ``0o700`` to keep the cache
            directory, and so the cached data files within it, readable by the
            owner only.
        """
        super().__init__(**kwargs)
        if fs is None and target_protocol is None:
            raise ValueError(
                "Please provide filesystem instance(fs) or target_protocol"
            )
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Both filesystems (fs) and target_protocol may not be both given."
            )
        if cache_storage == "TMP":
            tempdir = tempfile.mkdtemp()
            storage = [tempdir]
            weakref.finalize(self, self._remove_tempdir, tempdir)
        else:
            if isinstance(cache_storage, str):
                storage = [cache_storage]
            else:
                storage = cache_storage
        self.cache_storage_mode = cache_storage_mode
        self._makedirs(storage[-1])
        self.storage = storage
        self.kwargs = target_options or {}
        self.cache_check = cache_check
        self.check_files = check_files
        self.expiry = expiry_time
        self.compression = compression

        # Size of cache in bytes. If None then the size is unknown and will be
        # recalculated the next time cache_size() is called. On writes to the
        # cache this is reset to None.
        self._cache_size = None

        if same_names is not None and cache_mapper is not None:
            raise ValueError(
                "Cannot specify both same_names and cache_mapper in "
                "CachingFileSystem.__init__"
            )
        if cache_mapper is not None:
            self._mapper = cache_mapper
        else:
            self._mapper = create_cache_mapper(
                same_names if same_names is not None else False
            )

        self.target_protocol = (
            target_protocol
            if isinstance(target_protocol, str)
            else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
        )
        self._metadata = CacheMetadata(self.storage)
        self.load_cache()
        self.fs = fs if fs is not None else filesystem(target_protocol, **self.kwargs)

        def _strip_protocol(path):
            # acts as a method, since each instance has a difference target
            return self.fs._strip_protocol(type(self)._strip_protocol(path))

        self._strip_protocol: Callable = _strip_protocol

    @staticmethod
    def _remove_tempdir(tempdir):
        try:
            rmtree(tempdir)
        except Exception:
            pass

    def _makedirs(self, path):
        if self.cache_storage_mode is None:
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True, mode=self.cache_storage_mode)

    def _mkcache(self):
        self._makedirs(self.storage[-1])

    def cache_size(self):
        """Return size of cache in bytes.

        If more than one cache directory is in use, only the size of the last
        one (the writable cache directory) is returned.
        """
        if self._cache_size is None:
            cache_dir = self.storage[-1]
            self._cache_size = filesystem("file").du(cache_dir, withdirs=True)
        return self._cache_size

    def load_cache(self):
        """Read set of stored blocks from file"""
        self._metadata.load()
        self._mkcache()
        self.last_cache = time.time()

    def save_cache(self):
        """Save set of stored blocks from file"""
        self._mkcache()
        self._metadata.save()
        self.last_cache = time.time()
        self._cache_size = None

    def _check_cache(self):
        """Reload caches if time elapsed or any disappeared"""
        self._mkcache()
        if not self.cache_check:
            # explicitly told not to bother checking
            return
        timecond = time.time() - self.last_cache > self.cache_check
        existcond = all(os.path.exists(storage) for storage in self.storage)
        if timecond or not existcond:
            self.load_cache()

    def _check_file(self, path):
        """Is path in cache and still valid"""
        path = self._strip_protocol(path)
        self._check_cache()
        return self._metadata.check_file(path, self)

    def clear_cache(self):
        """Remove all files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.
        """
        rmtree(self.storage[-1])
        self.load_cache()
        self._cache_size = None

    def clear_expired_cache(self, expiry_time=None):
        """Remove all expired files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.

        Parameters
        ----------
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            If not defined the default is equivalent to the attribute from the
            file caching instantiation.
        """

        if not expiry_time:
            expiry_time = self.expiry

        self._check_cache()

        expired_files, writable_cache_empty = self._metadata.clear_expired(expiry_time)
        for fn in expired_files:
            if os.path.exists(fn):
                os.remove(fn)

        if writable_cache_empty:
            rmtree(self.storage[-1])
            self.load_cache()

        self._cache_size = None

    def pop_from_cache(self, path):
        """Remove cached version of given file

        Deletes local copy of the given (remote) path. If it is found in a cache
        location which is not the last, it is assumed to be read-only, and
        raises PermissionError
        """
        path = self._strip_protocol(path)
        fn = self._metadata.pop_file(path)
        if fn is not None:
            os.remove(fn)
        self._cache_size = None

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """Wrap the target _open

        If the whole file exists in the cache, just open it locally and
        return that.

        Otherwise, open the file on the target FS, and make it have a mmap
        cache pointing to the location which we determine, in our cache.
        The ``blocks`` instance is shared, so as the mmap cache instance
        updates, so does the entry in our ``cached_files`` attribute.
        We monkey-patch this file, so that when it closes, we call
        ``close_and_update`` to save the state of the blocks.
        """
        path = self._strip_protocol(path)

        path = self.fs._strip_protocol(path)
        if "r" not in mode:
            return self.fs._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
        detail = self._check_file(path)
        if detail:
            # file is in cache
            detail, fn = detail
            hash, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                # stored file is complete
                logger.debug("Opening local copy of %s", path)
                return open(fn, mode)
            # TODO: action where partial file exists in read-only cache
            logger.debug("Opening partially cached copy of %s", path)
        else:
            hash = self._mapper(path)
            fn = os.path.join(self.storage[-1], hash)
            blocks = set()
            detail = {
                "original": path,
                "fn": hash,
                "blocks": blocks,
                "time": time.time(),
                "uid": self.fs.ukey(path),
            }
            self._metadata.update_file(path, detail)
            logger.debug("Creating local sparse file for %s", path)

        # explicitly submitting the size to the open call will avoid extra
        # operations when opening. This is particularly relevant
        # for any file that is read over a network, e.g. S3.
        size = detail.get("size")

        # call target filesystems open
        self._mkcache()
        f = self.fs._open(
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            cache_type="none",
            size=size,
            **kwargs,
        )

        # set size if not already set
        if size is None:
            detail["size"] = f.size
            self._metadata.update_file(path, detail)

        if self.compression:
            comp = (
                infer_compression(path)
                if self.compression == "infer"
                else self.compression
            )
            f = compr[comp](f, mode="rb")
        if "blocksize" in detail:
            if detail["blocksize"] != f.blocksize:
                raise BlocksizeMismatchError(
                    f"Cached file must be reopened with same block"
                    f" size as original (old: {detail['blocksize']},"
                    f" new {f.blocksize})"
                )
        else:
            detail["blocksize"] = f.blocksize

        def _fetch_ranges(ranges):
            return self.fs.cat_ranges(
                [path] * len(ranges),
                [r[0] for r in ranges],
                [r[1] for r in ranges],
                **kwargs,
            )

        multi_fetcher = None if self.compression else _fetch_ranges
        f.cache = MMapCache(
            f.blocksize, f._fetch_range, f.size, fn, blocks, multi_fetcher=multi_fetcher
        )
        close = f.close
        f.close = lambda: self.close_and_update(f, close)
        self.save_cache()
        return f

    def _parent(self, path):
        return self.fs._parent(path)

    def hash_name(self, path: str, *args: Any) -> str:
        # Kept for backward compatibility with downstream libraries.
        # Ignores extra arguments, previously same_name boolean.
        return self._mapper(path)

    def close_and_update(self, f, close):
        """Called when a file is closing, so store the set of blocks"""
        if f.closed:
            return
        path = self._strip_protocol(f.path)
        self._metadata.on_close_cached_file(f, path)
        try:
            logger.debug("going to save")
            self.save_cache()
            logger.debug("saved")
        except OSError:
            logger.debug("Cache saving failed while closing file")
        except NameError:
            logger.debug("Cache save failed due to interpreter shutdown")
        close()
        f.closed = True

    def ls(self, path, detail=True, **kwargs):
        return self.fs.ls(path, detail, **kwargs)

    def __getattribute__(self, item):
        if item in {
            "load_cache",
            "_get_cached_file_before_open",
            "_open",
            "save_cache",
            "close_and_update",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_make_local_details",
            "open",
            "cat",
            "cat_file",
            "_cat_file",
            "cat_ranges",
            "_cat_ranges",
            "get",
            "read_block",
            "tail",
            "head",
            "info",
            "ls",
            "exists",
            "isfile",
            "isdir",
            "_check_file",
            "_check_cache",
            "_makedirs",
            "_mkcache",
            "clear_cache",
            "clear_expired_cache",
            "pop_from_cache",
            "local_file",
            "_paths_from_path",
            "get_mapper",
            "open_many",
            "commit_many",
            "hash_name",
            "__hash__",
            "__eq__",
            "to_json",
            "to_dict",
            "cache_size",
            "pipe_file",
            "pipe",
            "start_transaction",
            "end_transaction",
        }:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            if hasattr(type(self), item):
                return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                    *args, **kw
                )
            # method is in the whitelist but not defined on this subclass;
            # fall through to delegate to the wrapped filesystem below
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            # property
            return type(self).transaction.__get__(self)
        if item in {"_cache", "transaction_type", "protocol"}:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)

    def __eq__(self, other):
        """Test for equality."""
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        return (
            self.storage == other.storage
            and self.kwargs == other.kwargs
            and self.cache_check == other.cache_check
            and self.check_files == other.check_files
            and self.expiry == other.expiry
            and self.compression == other.compression
            and self._mapper == other._mapper
            and self.target_protocol == other.target_protocol
        )

    def __hash__(self):
        """Calculate hash."""
        return (
            hash(tuple(self.storage))
            ^ hash(str(self.kwargs))
            ^ hash(self.cache_check)
            ^ hash(self.check_files)
            ^ hash(self.expiry)
            ^ hash(self.compression)
            ^ hash(self._mapper)
            ^ hash(self.target_protocol)
        )


class WholeFileCacheFileSystem(CachingFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This is similar to ``CachingFileSystem``, but without
    the block-wise functionality and so can work even when sparse files
    are not allowed. See its docstring for definition of the init
    arguments.

    The class still needs access to the remote store for listing files,
    and may refresh cached files.
    """

    protocol = "filecache"
    local_file = True

    def open_many(self, open_files, **kwargs):
        paths = [of.path for of in open_files]
        if "r" in open_files.mode:
            self._mkcache()
        else:
            return [
                LocalTempFile(
                    self.fs,
                    path,
                    mode=open_files.mode,
                    fn=os.path.join(self.storage[-1], self._mapper(path)),
                    **kwargs,
                )
                for path in paths
            ]

        if self.compression:
            raise NotImplementedError
        details = [self._check_file(sp) for sp in paths]
        downpath = [p for p, d in zip(paths, details) if not d]
        downfn0 = [
            os.path.join(self.storage[-1], self._mapper(p))
            for p, d in zip(paths, details)
        ]  # keep these path names for opening later
        downfn = [fn for fn, d in zip(downfn0, details) if not d]
        if downpath:
            # skip if all files are already cached and up to date
            self.fs.get(downpath, downfn)

            # update metadata - only happens when downloads are successful
            newdetail = [
                {
                    "original": path,
                    "fn": self._mapper(path),
                    "blocks": True,
                    "time": time.time(),
                    "uid": self.fs.ukey(path),
                }
                for path in downpath
            ]
            for path, detail in zip(downpath, newdetail):
                self._metadata.update_file(path, detail)
            self.save_cache()

        def firstpart(fn):
            # helper to adapt both whole-file and simple-cache
            return fn[1] if isinstance(fn, tuple) else fn

        return [
            open(firstpart(fn0) if fn0 else fn1, mode=open_files.mode)
            for fn0, fn1 in zip(details, downfn0)
        ]

    def commit_many(self, open_files):
        self.fs.put([f.fn for f in open_files], [f.path for f in open_files])
        [f.close() for f in open_files]
        for f in open_files:
            # in case autocommit is off, and so close did not already delete
            try:
                os.remove(f.name)
            except FileNotFoundError:
                pass
        self._cache_size = None

    def _make_local_details(self, path):
        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        detail = {
            "original": path,
            "fn": hash,
            "blocks": True,
            "time": time.time(),
            "uid": self.fs.ukey(path),
        }
        self._metadata.update_file(path, detail)
        logger.debug("Copying %s to local cache", path)
        return fn

    def cat(
        self,
        path,
        recursive=False,
        on_error="raise",
        callback=DEFAULT_CALLBACK,
        **kwargs,
    ):
        paths = self.expand_path(
            path, recursive=recursive, maxdepth=kwargs.get("maxdepth")
        )
        getpaths = []
        storepaths = []
        fns = []
        out = {}
        for p in paths.copy():
            try:
                detail = self._check_file(p)
                if not detail:
                    fn = self._make_local_details(p)
                    getpaths.append(p)
                    storepaths.append(fn)
                else:
                    detail, fn = detail if isinstance(detail, tuple) else (None, detail)
                fns.append(fn)
            except Exception as e:
                if on_error == "raise":
                    raise
                if on_error == "return":
                    out[p] = e
                paths.remove(p)

        if getpaths:
            self.fs.get(getpaths, storepaths)
            self.save_cache()

        callback.set_size(len(paths))
        for p, fn in zip(paths, fns):
            with open(fn, "rb") as f:
                out[p] = f.read()
            callback.relative_update(1)
        if isinstance(path, str) and len(paths) == 1 and recursive is False:
            out = out[paths[0]]
        return out

    def _get_cached_file_before_open(self, path, **kwargs):
        fn = self._make_local_details(path)
        # call target filesystems open
        self._mkcache()
        if self.compression:
            with self.fs._open(path, mode="rb", **kwargs) as f, open(fn, "wb") as f2:
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = (
                    infer_compression(path)
                    if self.compression == "infer"
                    else self.compression
                )
                f = compr[comp](f, mode="rb")
                data = True
                while data:
                    block = getattr(f, "blocksize", 5 * 2**20)
                    data = f.read(block)
                    f2.write(data)
        else:
            self.fs.get_file(path, fn)
        self.save_cache()

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        # For read (or append), (try) download from remote
        if "r" in mode or "a" in mode:
            if not self._check_file(path):
                if self.fs.exists(path):
                    self._get_cached_file_before_open(path, **kwargs)
                elif "r" in mode:
                    raise FileNotFoundError(path)

            detail, fn = self._check_file(path)
            _, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                logger.debug("Opening local copy of %s", path)
            else:
                raise ValueError(
                    f"Attempt to open partially cached file {path}"
                    f" as a wholly cached file"
                )

        # Just reading does not need special file handling
        if "r" in mode and "+" not in mode:
            # In order to support downstream filesystems to be able to
            # infer the compression from the original filename, like
            # the `TarFileSystem`, let's extend the `io.BufferedReader`
            # fileobject protocol by adding a dedicated attribute
            # `original`.
            f = open(fn, mode)
            f.original = detail.get("original")
            return f

        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        user_specified_kwargs = {
            k: v
            for k, v in kwargs.items()
            # those kwargs were added by open(), we don't want them
            if k not in ["autocommit", "block_size", "cache_options"]
        }
        return LocalTempFile(self, path, mode=mode, fn=fn, **user_specified_kwargs)

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        logger.debug("async cat_file %s", path)
        path = self._strip_protocol(path)
        sha = self._mapper(path)
        fn = self._check_file(path)

        if not fn:
            fn = os.path.join(self.storage[-1], sha)
            await self.fs._get_file(path, fn, **kwargs)

        with open(fn, "rb") as f:  # noqa ASYNC230
            if start:
                f.seek(start)
            size = -1 if end is None else end - f.tell()
            return f.read(size)

    async def _cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        logger.debug("async cat ranges %s", paths)
        lpaths = []
        rset = set()
        download = []
        rpaths = []
        for p in paths:
            fn = self._check_file(p)
            if fn is None and p not in rset:
                sha = self._mapper(p)
                fn = os.path.join(self.storage[-1], sha)
                download.append(fn)
                rset.add(p)
                rpaths.append(p)
            lpaths.append(fn)
        if download:
            await self.fs._get(rpaths, download, on_error=on_error)

        return LocalFileSystem().cat_ranges(
            lpaths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs
        )


class SimpleCacheFileSystem(WholeFileCacheFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files, and
    does not keep any metadata about the download time or file details.
    It is therefore safer to use in multi-threaded/concurrent situations.

    This is the only of the caching filesystems that supports write: you will
    be given a real local open file, and upon close and commit, it will be
    uploaded to the target filesystem; the writability or the target URL is
    not checked until that time.

    """

    protocol = "simplecache"
    local_file = True
    transaction_type = WriteCachedTransaction

    def __init__(self, **kwargs):
        kw = kwargs.copy()
        for key in ["cache_check", "expiry_time", "check_files"]:
            kw[key] = False
        super().__init__(**kw)
        for storage in self.storage:
            if not os.path.exists(storage):
                self._makedirs(storage)

    def _check_file(self, path):
        self._check_cache()
        sha = self._mapper(path)
        for storage in self.storage:
            fn = os.path.join(storage, sha)
            if os.path.exists(fn):
                return fn

    def save_cache(self):
        pass

    def load_cache(self):
        pass

    def pipe_file(self, path, value=None, **kwargs):
        if self._intrans:
            with self.open(path, "wb") as f:
                f.write(value)
        else:
            super().pipe_file(path, value)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        details = []
        try:
            details = self.fs.ls(
                path, detail=True, **kwargs
            ).copy()  # don't edit original!
        except FileNotFoundError as e:
            ex = e
        else:
            ex = None
        if self._intrans:
            path1 = path.rstrip("/") + "/"
            for f in self.transaction.files:
                if f.path == path:
                    details.append(
                        {"name": path, "size": f.size or f.tell(), "type": "file"}
                    )
                elif f.path.startswith(path1):
                    if f.path.count("/") == path1.count("/"):
                        details.append(
                            {"name": f.path, "size": f.size or f.tell(), "type": "file"}
                        )
                    else:
                        dname = "/".join(f.path.split("/")[: path1.count("/") + 1])
                        details.append({"name": dname, "size": 0, "type": "directory"})
        if ex is not None and not details:
            raise ex
        if detail:
            return details
        return sorted(_["name"] for _ in details)

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        if self._intrans:
            f = [_ for _ in self.transaction.files if _.path == path]
            if f:
                size = os.path.getsize(f[0].fn) if f[0].closed else f[0].tell()
                return {"name": path, "size": size, "type": "file"}
            f = any(_.path.startswith(path + "/") for _ in self.transaction.files)
            if f:
                return {"name": path, "size": 0, "type": "directory"}
        return self.fs.info(path, **kwargs)

    def pipe(self, path, value=None, **kwargs):
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError("path must be str or dict")

    def cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        logger.debug("cat ranges %s", paths)
        lpaths = [self._check_file(p) for p in paths]
        rpaths = [p for l, p in zip(lpaths, paths) if l is False]
        lpaths = [l for l, p in zip(lpaths, paths) if l is False]
        self.fs.get(rpaths, lpaths)
        paths = [self._check_file(p) for p in paths]
        return LocalFileSystem().cat_ranges(
            paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs
        )

    def _get_cached_file_before_open(self, path, **kwargs):
        sha = self._mapper(path)
        fn = os.path.join(self.storage[-1], sha)
        logger.debug("Copying %s to local cache", path)

        self._mkcache()
        self._cache_size = None

        if self.compression:
            with self.fs._open(path, mode="rb", **kwargs) as f, open(fn, "wb") as f2:
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = (
                    infer_compression(path)
                    if self.compression == "infer"
                    else self.compression
                )
                f = compr[comp](f, mode="rb")
                data = True
                while data:
                    block = getattr(f, "blocksize", 5 * 2**20)
                    data = f.read(block)
                    f2.write(data)
        else:
            self.fs.get_file(path, fn)

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        sha = self._mapper(path)

        # For read (or append), (try) download from remote
        if "r" in mode or "a" in mode:
            if not self._check_file(path):
                # append does not require an existing file but read does
                if self.fs.exists(path):
                    self._get_cached_file_before_open(path, **kwargs)
                elif "r" in mode:
                    raise FileNotFoundError(path)

        fn = self._check_file(path)
        # Just reading does not need special file handling
        if "r" in mode and "+" not in mode:
            return open(fn, mode)

        fn = os.path.join(self.storage[-1], sha)
        user_specified_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["autocommit", "block_size", "cache_options"]
        }  # those were added by open()
        return LocalTempFile(
            self,
            path,
            mode=mode,
            autocommit=not self._intrans,
            fn=fn,
            **user_specified_kwargs,
        )


class LocalTempFile:
    """A temporary local file, which will be uploaded on commit"""

    def __init__(self, fs, path, fn, mode="wb", autocommit=True, seek=0, **kwargs):
        self.fn = fn
        self.fh = open(fn, mode)
        self.mode = mode
        if seek:
            self.fh.seek(seek)
        self.path = path
        self.size = None
        self.fs = fs
        self.closed = False
        self.autocommit = autocommit
        self.kwargs = kwargs

    def __reduce__(self):
        # always open in r+b to allow continuing writing at a location
        return (
            LocalTempFile,
            (self.fs, self.path, self.fn, "r+b", self.autocommit, self.tell()),
        )

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # self.size = self.fh.tell()
        if self.closed:
            return
        self.fh.close()
        self.closed = True
        if self.autocommit:
            self.commit()

    def discard(self):
        self.fh.close()
        os.remove(self.fn)

    def commit(self):
        # calling put() with list arguments avoids path expansion and additional operations
        # like isdir()
        self.fs.put([self.fn], [self.path], **self.kwargs)
        # we do not delete the local copy, it's still in the cache.

    @property
    def name(self):
        return self.fn

    def __repr__(self) -> str:
        return f"LocalTempFile: {self.path}"

    def __getattr__(self, item):
        return getattr(self.fh, item)
<|endoftext|>import os

import pygit2

from fsspec.spec import AbstractFileSystem

from .memory import MemoryFile


class GitFileSystem(AbstractFileSystem):
    """Browse the files of a local git repo at any hash/tag/branch

    (experimental backend)
    """

    root_marker = ""
    cachable = True

    def __init__(self, path=None, fo=None, ref=None, **kwargs):
        """

        Parameters
        ----------
        path: str (optional)
            Local location of the repo (uses current directory if not given).
            May be deprecated in favour of ``fo``. When used with a higher
            level function such as fsspec.open(), may be of the form
            "git://[path-to-repo[:]][ref@]path/to/file" (but the actual
            file path should not contain "@" or ":").
        fo: str (optional)
            Same as ``path``, but passed as part of a chained URL. This one
            takes precedence if both are given.
        ref: str (optional)
            Reference to work with, could be a hash, tag or branch name. Defaults
            to current working tree. Note that ``ls`` and ``open`` also take hash,
            so this becomes the default for those operations
        kwargs
        """
        super().__init__(**kwargs)
        self.repo = pygit2.Repository(fo or path or os.getcwd())
        self.ref = ref or "master"

    @classmethod
    def _strip_protocol(cls, path):
        path = super()._strip_protocol(path).lstrip("/")
        if ":" in path:
            path = path.split(":", 1)[1]
        if "@" in path:
            path = path.split("@", 1)[1]
        return path.lstrip("/")

    def _path_to_object(self, path, ref):
        comm, ref = self.repo.resolve_refish(ref or self.ref)
        parts = path.split("/")
        tree = comm.tree
        for part in parts:
            if part and isinstance(tree, pygit2.Tree):
                if part not in tree:
                    raise FileNotFoundError(path)
                tree = tree[part]
        return tree

    @staticmethod
    def _get_kwargs_from_urls(path):
        path = path.removeprefix("git://")
        out = {}
        if ":" in path:
            out["path"], path = path.split(":", 1)
        if "@" in path:
            out["ref"], path = path.split("@", 1)
        return out

    @staticmethod
    def _object_to_info(obj, path=None):
        # obj.name and obj.filemode are None for the root tree!
        is_dir = isinstance(obj, pygit2.Tree)
        return {
            "type": "directory" if is_dir else "file",
            "name": (
                "/".join([path, obj.name or ""]).lstrip("/") if path else obj.name
            ),
            "hex": str(obj.id),
            "mode": "100644" if obj.filemode is None else f"{obj.filemode:o}",
            "size": 0 if is_dir else obj.size,
        }

    def ls(self, path, detail=True, ref=None, **kwargs):
        tree = self._path_to_object(self._strip_protocol(path), ref)
        return [
            GitFileSystem._object_to_info(obj, path)
            if detail
            else GitFileSystem._object_to_info(obj, path)["name"]
            for obj in (tree if isinstance(tree, pygit2.Tree) else [tree])
        ]

    def info(self, path, ref=None, **kwargs):
        tree = self._path_to_object(self._strip_protocol(path), ref)
        return GitFileSystem._object_to_info(tree, path)

    def ukey(self, path, ref=None):
        return self.info(path, ref=ref)["hex"]

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        ref=None,
        **kwargs,
    ):
        obj = self._path_to_object(path, ref or self.ref)
        return MemoryFile(data=obj.data)
<|endoftext|>import asyncio
import io
import logging
import re
import weakref
from copy import copy
from urllib.parse import urlparse

import aiohttp
import yarl

from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, sync, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import (
    DEFAULT_BLOCK_SIZE,
    glob_translate,
    isfilelike,
    nullcontext,
    tokenize,
)

from ..caching import AllBytes

# https://stackoverflow.com/a/15926317/3821154
ex = re.compile(r"""<(a|A)\s+(?:[^>]*?\s+)?(href|HREF)=["'](?P<url>[^"']+)""")
ex2 = re.compile(r"""(?P<url>http[s]?://[-a-zA-Z0-9@:%_+.~#?&/=]+)""")
logger = logging.getLogger("fsspec.http")


async def get_client(**kwargs):
    return aiohttp.ClientSession(**kwargs)


class HTTPFileSystem(AsyncFileSystem):
    """
    Simple File-System for fetching data via HTTP(S)

    ``ls()`` is implemented by loading the parent page and doing a regex
    match on the result. If simple_link=True, anything of the form
    "http(s)://server.com/stuff?thing=other"; otherwise only links within
    HTML href tags will be used.

    URLs are passed unfiltered to aiohttp, so all addresses are accessible. Where URLs are
    supplied by a user, the calling application may wish to filter to prevent scanning.
    """

    protocol = ("http", "https")
    sep = "/"

    def __init__(
        self,
        simple_links=True,
        block_size=None,
        same_scheme=True,
        size_policy=None,
        cache_type="bytes",
        cache_options=None,
        asynchronous=False,
        loop=None,
        client_kwargs=None,
        get_client=get_client,
        encoded=False,
        **storage_options,
    ):
        """
        NB: if this is called async, you must await set_client

        Parameters
        ----------
        block_size: int
            Blocks to read bytes; if 0, will default to raw requests file-like
            objects instead of HTTPFile instances
        simple_links: bool
            If True, will consider both HTML <a> tags and anything that looks
            like a URL; if False, will consider only the former.
        same_scheme: True
            When doing ls/glob, if this is True, only consider paths that have
            http/https matching the input URLs.
        size_policy: this argument is deprecated
        client_kwargs: dict
            Passed to aiohttp.ClientSession, see
            https://docs.aiohttp.org/en/stable/client_reference.html
            For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
        get_client: Callable[..., aiohttp.ClientSession]
            A callable, which takes keyword arguments and constructs
            an aiohttp.ClientSession. Its state will be managed by
            the HTTPFileSystem class.
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open()
        """
        super().__init__(self, asynchronous=asynchronous, loop=loop, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.client_kwargs = client_kwargs or {}
        self.get_client = get_client
        self.encoded = encoded
        self.kwargs = storage_options
        self._session = None

        # Clean caching-related parameters from `storage_options`
        # before propagating them as `request_options` through `self.kwargs`.
        # TODO: Maybe rename `self.kwargs` to `self.request_options` to make
        #       it clearer.
        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop("use_listings_cache", False)
        request_options.pop("listings_expiry_time", None)
        request_options.pop("max_paths", None)
        request_options.pop("skip_instance_cache", None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return "http"

    def encode_url(self, url):
        return yarl.URL(url, encoded=self.encoded)

    @staticmethod
    def close_session(loop, session):
        if loop is not None and loop.is_running():
            try:
                sync(loop, session.close, timeout=0.1)
                return
            except (TimeoutError, FSTimeoutError, NotImplementedError):
                pass
        connector = getattr(session, "_connector", None)
        if connector is not None:
            # close after loop is dead
            connector._close()

    async def set_session(self):
        if self._session is None:
            self._session = await self.get_client(loop=self.loop, **self.client_kwargs)
            if not self.asynchronous:
                weakref.finalize(self, self.close_session, self.loop, self._session)
        return self._session

    @classmethod
    def _strip_protocol(cls, path):
        """For HTTP, we always want to keep the full URL"""
        return path

    @classmethod
    def _parent(cls, path):
        # override, since _strip_protocol is different for URLs
        par = super()._parent(path)
        if len(par) > 7:  # "http://..."
            return par
        return ""

    async def _ls_real(self, url, detail=True, **kwargs):
        # ignoring URL-encoded arguments
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        session = await self.set_session()
        async with session.get(self.encode_url(url), **self.kwargs) as r:
            self._raise_not_found_for_status(r, url)

            if "Content-Type" in r.headers:
                mimetype = r.headers["Content-Type"].partition(";")[0]
            else:
                mimetype = None

            if mimetype in ("text/html", None):
                try:
                    text = await r.text(errors="ignore")
                    if self.simple_links:
                        links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
                    else:
                        links = [u[2] for u in ex.findall(text)]
                except UnicodeDecodeError:
                    links = []  # binary, not HTML
            else:
                links = []

        out = set()
        parts = urlparse(url)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith("/") and len(l) > 1:
                # absolute URL on this server
                l = f"{parts.scheme}://{parts.netloc}{l}"
            if l.startswith("http"):
                if self.same_schema and l.startswith(url.rstrip("/") + "/"):
                    out.add(l)
                elif l.replace("https", "http").startswith(
                    url.replace("https", "http").rstrip("/") + "/"
                ):
                    # allowed to cross http <-> https
                    out.add(l)
            else:
                if l not in ["..", "../"]:
                    # Ignore FTP-like "parent"
                    out.add("/".join([url.rstrip("/"), l.lstrip("/")]))
        if not out and url.endswith("/"):
            out = await self._ls_real(url.rstrip("/"), detail=False)
        if detail:
            return [
                {
                    "name": u,
                    "size": None,
                    "type": "directory" if u.endswith("/") else "file",
                }
                for u in out
            ]
        else:
            return sorted(out)

    async def _ls(self, url, detail=True, **kwargs):
        if self.use_listings_cache and url in self.dircache:
            out = self.dircache[url]
        else:
            out = await self._ls_real(url, detail=detail, **kwargs)
            self.dircache[url] = out
        return out

    ls = sync_wrapper(_ls)

    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()

    async def _cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = await self._process_limits(url, start, end)
            kw["headers"] = headers
        session = await self.set_session()
        async with session.get(self.encode_url(url), **kw) as r:
            out = await r.read()
            self._raise_not_found_for_status(r, url)
        return out

    async def _get_file(
        self, rpath, lpath, chunk_size=5 * 2**20, callback=DEFAULT_CALLBACK, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(rpath)
        session = await self.set_session()
        async with session.get(self.encode_url(rpath), **kw) as r:
            try:
                size = int(r.headers["content-length"])
            except (ValueError, KeyError):
                size = None

            callback.set_size(size)
            self._raise_not_found_for_status(r, rpath)
            if isfilelike(lpath):
                outfile = lpath
            else:
                outfile = open(lpath, "wb")  # noqa: ASYNC230

            try:
                chunk = True
                while chunk:
                    chunk = await r.content.read(chunk_size)
                    outfile.write(chunk)
                    callback.relative_update(len(chunk))
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    async def _put_file(
        self,
        lpath,
        rpath,
        chunk_size=5 * 2**20,
        callback=DEFAULT_CALLBACK,
        method="post",
        mode="overwrite",
        **kwargs,
    ):
        if mode != "overwrite":
            raise NotImplementedError("Exclusive write")

        async def gen_chunks():
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")  # noqa: ASYNC230
                use_seek = True

            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, "size", None))

                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)

        kw = self.kwargs.copy()
        kw.update(kwargs)
        session = await self.set_session()

        method = method.lower()
        if method not in ("post", "put"):
            raise ValueError(
                f"method has to be either 'post' or 'put', not: {method!r}"
            )

        meth = getattr(session, method)
        async with meth(self.encode_url(rpath), data=gen_chunks(), **kw) as resp:
            self._raise_not_found_for_status(resp, rpath)

    async def _exists(self, path, strict=False, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            logger.debug(path)
            session = await self.set_session()
            r = await session.get(self.encode_url(path), **kw)
            async with r:
                if strict:
                    self._raise_not_found_for_status(r, path)
                return r.status < 400
        except FileNotFoundError:
            return False
        except aiohttp.ClientError:
            if strict:
                raise
            return False

    async def _isfile(self, path, **kwargs):
        return await self._exists(path, **kwargs)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type=None,
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """Make a file-like object

        Parameters
        ----------
        path: str
            Full URL with protocol
        mode: string
            must be "rb"
        block_size: int or None
            Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
        kwargs: key-value
            Any other parameters, passed to requests calls
        """
        if mode != "rb":
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw["asynchronous"] = self.asynchronous
        kw.update(kwargs)
        info = {}
        size = size or info.update(self.info(path, **kwargs)) or info["size"]
        session = sync(self.loop, self.set_session)
        if block_size and size and info.get("partial", True):
            return HTTPFile(
                self,
                path,
                session=session,
                block_size=block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                loop=self.loop,
                **kw,
            )
        else:
            return HTTPStreamFile(
                self,
                path,
                mode=mode,
                loop=self.loop,
                session=session,
                **kw,
            )

    async def open_async(self, path, mode="rb", size=None, **kwargs):
        session = await self.set_session()
        if size is None:
            try:
                size = (await self._info(path, **kwargs))["size"]
            except FileNotFoundError:
                pass
        return AsyncStreamFile(
            self,
            path,
            loop=self.loop,
            session=session,
            size=size,
            **kwargs,
        )

    def ukey(self, url):
        """Unique identifier; assume HTTP files are static, unchanging"""
        return tokenize(url, self.kwargs, self.protocol)

    async def _info(self, url, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        session = await self.set_session()

        for policy in ["head", "get"]:
            try:
                info.update(
                    await _file_info(
                        self.encode_url(url),
                        size_policy=policy,
                        session=session,
                        **self.kwargs,
                        **kwargs,
                    )
                )
                if info.get("size") is not None:
                    break
            except Exception as exc:
                if policy == "get":
                    # If get failed, then raise a FileNotFoundError
                    raise FileNotFoundError(url) from exc
                logger.debug("", exc_info=exc)

        return {"name": url, "size": None, **info, "type": "file"}

    async def _glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        This implementation is idntical to the one in AbstractFileSystem,
        but "?" is not considered as a character for globbing, because it is
        so common in URLs, often identifying the "query" part.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")
        import re

        ends_with_slash = path.endswith("/")  # _strip_protocol strips trailing slash
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_slash or path.endswith(("/**", "/*"))
        idx_star = path.find("*") if path.find("*") >= 0 else len(path)
        idx_brace = path.find("[") if path.find("[") >= 0 else len(path)

        min_idx = min(idx_star, idx_brace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            if await self._exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: await self._info(path, **kwargs)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:min_idx]:
            min_idx = path[:min_idx].rindex("/")
            root = path[: min_idx + 1]
            depth = path[min_idx + 1 :].count("/") + 1
        else:
            root = ""
            depth = path[min_idx + 1 :].count("/") + 1

        if "**" in path:
            if maxdepth is not None:
                idx_double_stars = path.find("**")
                depth_double_stars = path[idx_double_stars:].count("/") + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None

        allpaths = await self._find(
            root, maxdepth=depth, withdirs=True, detail=True, **kwargs
        )

        pattern = glob_translate(path + ("/" if ends_with_slash else ""))
        pattern = re.compile(pattern)

        out = {
            (
                p.rstrip("/")
                if not append_slash_to_dirname
                and info["type"] == "directory"
                and p.endswith("/")
                else p
            ): info
            for p, info in sorted(allpaths.items())
            if pattern.match(p.rstrip("/"))
        }

        if detail:
            return out
        else:
            return list(out)

    async def _isdir(self, path):
        # override, since all URLs are (also) files
        try:
            return bool(await self._ls(path))
        except (FileNotFoundError, ValueError):
            return False

    async def _pipe_file(self, path, value, mode="overwrite", **kwargs):
        """
        Write bytes to a remote file over HTTP.

        Parameters
        ----------
        path : str
            Target URL where the data should be written
        value : bytes
            Data to be written
        mode : str
            How to write to the file - 'overwrite' or 'append'
        **kwargs : dict
            Additional parameters to pass to the HTTP request
        """
        url = self._strip_protocol(path)
        headers = kwargs.pop("headers", {})
        headers["Content-Length"] = str(len(value))

        session = await self.set_session()

        async with session.put(
            self.encode_url(url), data=value, headers=headers, **kwargs
        ) as r:
            r.raise_for_status()


class HTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remote HTTP(S) resource

    Supports only reading, with read-ahead of a predetermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    session: aiohttp.ClientSession or None
        All calls will be made within this session, to avoid restarting
        connections where the server allows this
    block_size: int or None
        The amount of read-ahead to do, in bytes. Default is 5MB, or the value
        configured for the FileSystem creating this file
    size: None or int
        If given, this is the size of the file in bytes, and we don't attempt
        to call the server to find the value.
    kwargs: all other key-values are passed to requests calls.
    """

    def __init__(
        self,
        fs,
        url,
        session=None,
        block_size=None,
        mode="rb",
        cache_type="bytes",
        cache_options=None,
        size=None,
        loop=None,
        asynchronous=False,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError("File mode not supported")
        self.asynchronous = asynchronous
        self.loop = loop
        self.url = url
        self.session = session
        self.details = {"name": url, "size": size, "type": "file"}
        super().__init__(
            fs=fs,
            path=url,
            mode=mode,
            block_size=block_size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )

    def read(self, length=-1):
        """Read bytes from file

        Parameters
        ----------
        length: int
            Read up to this many bytes. If negative, read all content to end of
            file. If the server has not supplied the filesize, attempting to
            read only part of the data will raise a ValueError.
        """
        if (
            (length < 0 and self.loc == 0)  # explicit read all
            # but not when the size is known and fits into a block anyways
            and not (self.size is not None and self.size <= self.blocksize)
        ):
            self._fetch_all()
        if self.size is None:
            if length < 0:
                self._fetch_all()
        else:
            length = min(self.size - self.loc, length)
        return super().read(length)

    async def async_fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f"Fetch all for {self}")
        if not isinstance(self.cache, AllBytes):
            r = await self.session.get(self.fs.encode_url(self.url), **self.kwargs)
            async with r:
                r.raise_for_status()
                out = await r.read()
                self.cache = AllBytes(
                    size=len(out), fetcher=None, blocksize=None, data=out
                )
                self.size = len(out)

    _fetch_all = sync_wrapper(async_fetch_all)

    def _parse_content_range(self, headers):
        """Parse the Content-Range header"""
        s = headers.get("Content-Range", "")
        m = re.match(r"bytes (\d+-\d+|\*)/(\d+|\*)", s)
        if not m:
            return None, None, None

        if m[1] == "*":
            start = end = None
        else:
            start, end = [int(x) for x in m[1].split("-")]
        total = None if m[2] == "*" else int(m[2])
        return start, end, total

    async def async_fetch_range(self, start, end):
        """Download a block of data

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f"Fetch range for {self}: {start}-{end}")
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        logger.debug(f"{self.url} : {headers['Range']}")
        r = await self.session.get(
            self.fs.encode_url(self.url), headers=headers, **kwargs
        )
        async with r:
            if r.status == 416:
                # range request outside file
                return b""
            r.raise_for_status()

            # If the server has handled the range request, it should reply
            # with status 206 (partial content). But we'll guess that a suitable
            # Content-Range header or a Content-Length no more than the
            # requested range also mean we have got the desired range.
            response_is_range = (
                r.status == 206
                or self._parse_content_range(r.headers)[0] == start
                or int(r.headers.get("Content-Length", end + 1)) <= end - start
            )

            if response_is_range:
                # partial content, as expected
                out = await r.read()
            elif start > 0:
                raise ValueError(
                    "The HTTP server doesn't appear to support range requests. "
                    "Only reading this file from the beginning is supported. "
                    "Open with block_size=0 for a streaming file interface."
                )
            else:
                # Response is not a range, but we want the start of the file,
                # so we can read the required amount anyway.
                cl = 0
                out = []
                while True:
                    chunk = await r.content.read(2**20)
                    # data size unknown, let's read until we have enough
                    if chunk:
                        out.append(chunk)
                        cl += len(chunk)
                        if cl > end - start:
                            break
                    else:
                        break
                out = b"".join(out)[: end - start]
            return out

    _fetch_range = sync_wrapper(async_fetch_range)


magic_check = re.compile("([*[])")


def has_magic(s):
    match = magic_check.search(s)
    return match is not None


class HTTPStreamFile(AbstractBufferedFile):
    def __init__(self, fs, url, mode="rb", loop=None, session=None, **kwargs):
        self.asynchronous = kwargs.pop("asynchronous", False)
        self.url = url
        self.loop = loop
        self.session = session
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        super().__init__(fs=fs, path=url, mode=mode, cache_type="none", **kwargs)

        async def cor():
            r = await self.session.get(self.fs.encode_url(url), **kwargs).__aenter__()
            self.fs._raise_not_found_for_status(r, url)
            return r

        self.r = sync(self.loop, cor)
        self.loop = fs.loop

    def seek(self, loc, whence=0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError("Cannot seek streaming HTTP file")

    async def _read(self, num=-1):
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out

    read = sync_wrapper(_read)

    async def _close(self):
        self.r.close()

    def close(self):
        asyncio.run_coroutine_threadsafe(self._close(), self.loop)
        super().close()


class AsyncStreamFile(AbstractAsyncStreamedFile):
    def __init__(
        self, fs, url, mode="rb", loop=None, session=None, size=None, **kwargs
    ):
        self.url = url
        self.session = session
        self.r = None
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        self.kwargs = kwargs
        super().__init__(fs=fs, path=url, mode=mode, cache_type="none")
        self.size = size

    async def read(self, num=-1):
        if self.r is None:
            r = await self.session.get(
                self.fs.encode_url(self.url), **self.kwargs
            ).__aenter__()
            self.fs._raise_not_found_for_status(r, self.url)
            self.r = r
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out

    async def close(self):
        if self.r is not None:
            self.r.close()
            self.r = None
        await super().close()


async def get_range(session, url, start, end, file=None, **kwargs):
    # explicit get a range when we know it must be safe
    kwargs = kwargs.copy()
    headers = kwargs.pop("headers", {}).copy()
    headers["Range"] = f"bytes={start}-{end - 1}"
    r = await session.get(url, headers=headers, **kwargs)
    r.raise_for_status()
    async with r:
        out = await r.read()
    if file:
        with open(file, "r+b") as f:  # noqa: ASYNC230
            f.seek(start)
            f.write(out)
    else:
        return out


async def _file_info(url, session, size_policy="head", **kwargs):
    """Call HEAD on the server to get details about the file (size/checksum etc.)

    Default operation is to explicitly allow redirects and use encoding
    'identity' (no compression) to get the true size of the target.
    """
    logger.debug("Retrieve file size for %s", url)
    kwargs = kwargs.copy()
    ar = kwargs.pop("allow_redirects", True)
    head = kwargs.get("headers", {}).copy()
    head["Accept-Encoding"] = "identity"
    kwargs["headers"] = head

    info = {}
    if size_policy == "head":
        r = await session.head(url, allow_redirects=ar, **kwargs)
    elif size_policy == "get":
        r = await session.get(url, allow_redirects=ar, **kwargs)
    else:
        raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    async with r:
        r.raise_for_status()

        if "Content-Length" in r.headers:
            # Some servers may choose to ignore Accept-Encoding and return
            # compressed content, in which case the returned size is unreliable.
            if "Content-Encoding" not in r.headers or r.headers["Content-Encoding"] in [
                "identity",
                "",
            ]:
                info["size"] = int(r.headers["Content-Length"])
        elif "Content-Range" in r.headers:
            info["size"] = int(r.headers["Content-Range"].split("/")[1])

        if "Content-Type" in r.headers:
            info["mimetype"] = r.headers["Content-Type"].partition(";")[0]

        if r.headers.get("Accept-Ranges") == "none":
            # Some servers may explicitly discourage partial content requests, but
            # the lack of "Accept-Ranges" does not always indicate they would fail
            info["partial"] = False

        info["url"] = str(r.url)

        for checksum_field in ["ETag", "Content-MD5", "Digest", "Last-Modified"]:
            if r.headers.get(checksum_field):
                info[checksum_field] = r.headers[checksum_field]

    return info


async def _file_size(url, session=None, *args, **kwargs):
    if session is None:
        session = await get_client()
    info = await _file_info(url, session=session, *args, **kwargs)
    return info.get("size")


file_size = sync_wrapper(_file_size)
<|endoftext|>"""This file is largely copied from http.py"""

import io
import logging
import re
import urllib.error
import urllib.parse
from copy import copy
from json import dumps, loads
from urllib.parse import urlparse

try:
    import yarl
except (ImportError, ModuleNotFoundError, OSError):
    yarl = False

from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.registry import register_implementation
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE, isfilelike, nullcontext, tokenize

from ..caching import AllBytes

# https://stackoverflow.com/a/15926317/3821154
ex = re.compile(r"""<(a|A)\s+(?:[^>]*?\s+)?(href|HREF)=["'](?P<url>[^"']+)""")
ex2 = re.compile(r"""(?P<url>http[s]?://[-a-zA-Z0-9@:%_+.~#?&/=]+)""")
logger = logging.getLogger("fsspec.http")


class JsHttpException(urllib.error.HTTPError): ...


class StreamIO(io.BytesIO):
    # fake class, so you can set attributes on it
    # will eventually actually stream
    ...


class ResponseProxy:
    """Looks like a requests response"""

    def __init__(self, req, stream=False):
        self.request = req
        self.stream = stream
        self._data = None
        self._headers = None

    @property
    def raw(self):
        if self._data is None:
            b = self.request.response.to_bytes()
            if self.stream:
                self._data = StreamIO(b)
            else:
                self._data = b
        return self._data

    def close(self):
        if hasattr(self, "_data"):
            del self._data

    @property
    def headers(self):
        if self._headers is None:
            self._headers = dict(
                [
                    _.split(": ")
                    for _ in self.request.getAllResponseHeaders().strip().split("\r\n")
                ]
            )
        return self._headers

    @property
    def status_code(self):
        return int(self.request.status)

    def raise_for_status(self):
        if not self.ok:
            raise JsHttpException(
                self.url, self.status_code, self.reason, self.headers, None
            )

    def iter_content(self, chunksize, *_, **__):
        while True:
            out = self.raw.read(chunksize)
            if out:
                yield out
            else:
                break

    @property
    def reason(self):
        return self.request.statusText

    @property
    def ok(self):
        return self.status_code < 400

    @property
    def url(self):
        return self.request.response.responseURL

    @property
    def text(self):
        # TODO: encoding from headers
        return self.content.decode()

    @property
    def content(self):
        self.stream = False
        return self.raw

    def json(self):
        return loads(self.text)


class RequestsSessionShim:
    def __init__(self):
        self.headers = {}

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=None,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):
        from js import Blob, XMLHttpRequest

        logger.debug("JS request: %s %s", method, url)

        if cert or verify or proxies or files or cookies or hooks:
            raise NotImplementedError
        if data and json:
            raise ValueError("Use json= or data=, not both")
        req = XMLHttpRequest.new()
        extra = auth if auth else ()
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req.open(method, url, False, *extra)
        if timeout:
            req.timeout = timeout
        if headers:
            for k, v in headers.items():
                req.setRequestHeader(k, v)

        req.setRequestHeader("Accept", "application/octet-stream")
        req.responseType = "arraybuffer"
        if json:
            blob = Blob.new([dumps(data)], {type: "application/json"})
            req.send(blob)
        elif data:
            if isinstance(data, io.IOBase):
                data = data.read()
            blob = Blob.new([data], {type: "application/octet-stream"})
            req.send(blob)
        else:
            req.send(None)
        return ResponseProxy(req, stream=stream)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST}", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


class HTTPFileSystem(AbstractFileSystem):
    """
    Simple File-System for fetching data via HTTP(S)

    This is the BLOCKING version of the normal HTTPFileSystem. It uses
    requests in normal python and the JS runtime in pyodide.

    ***This implementation is extremely experimental, do not use unless
    you are testing pyodide/pyscript integration***
    """

    protocol = ("http", "https", "sync-http", "sync-https")
    sep = "/"

    def __init__(
        self,
        simple_links=True,
        block_size=None,
        same_scheme=True,
        cache_type="readahead",
        cache_options=None,
        client_kwargs=None,
        encoded=False,
        **storage_options,
    ):
        """

        Parameters
        ----------
        block_size: int
            Blocks to read bytes; if 0, will default to raw requests file-like
            objects instead of HTTPFile instances
        simple_links: bool
            If True, will consider both HTML <a> tags and anything that looks
            like a URL; if False, will consider only the former.
        same_scheme: True
            When doing ls/glob, if this is True, only consider paths that have
            http/https matching the input URLs.
        size_policy: this argument is deprecated
        client_kwargs: dict
            Passed to aiohttp.ClientSession, see
            https://docs.aiohttp.org/en/stable/client_reference.html
            For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open
        """
        super().__init__(self, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.client_kwargs = client_kwargs or {}
        self.encoded = encoded
        self.kwargs = storage_options

        try:
            import js  # noqa: F401

            logger.debug("Starting JS session")
            self.session = RequestsSessionShim()
            self.js = True
        except Exception as e:
            import requests

            logger.debug("Starting cpython session because of: %s", e)
            self.session = requests.Session(**(client_kwargs or {}))
            self.js = False

        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop("use_listings_cache", False)
        request_options.pop("listings_expiry_time", None)
        request_options.pop("max_paths", None)
        request_options.pop("skip_instance_cache", None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return "sync-http"

    def encode_url(self, url):
        if yarl:
            return yarl.URL(url, encoded=self.encoded)
        return url

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        """For HTTP, we always want to keep the full URL"""
        path = path.replace("sync-http://", "http://").replace(
            "sync-https://", "https://"
        )
        return path

    @classmethod
    def _parent(cls, path):
        # override, since _strip_protocol is different for URLs
        par = super()._parent(path)
        if len(par) > 7:  # "http://..."
            return par
        return ""

    def _ls_real(self, url, detail=True, **kwargs):
        # ignoring URL-encoded arguments
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        r = self.session.get(self.encode_url(url), **self.kwargs)
        self._raise_not_found_for_status(r, url)
        text = r.text
        if self.simple_links:
            links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
        else:
            links = [u[2] for u in ex.findall(text)]
        out = set()
        parts = urlparse(url)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith("/") and len(l) > 1:
                # absolute URL on this server
                l = parts.scheme + "://" + parts.netloc + l
            if l.startswith("http"):
                if self.same_schema and l.startswith(url.rstrip("/") + "/"):
                    out.add(l)
                elif l.replace("https", "http").startswith(
                    url.replace("https", "http").rstrip("/") + "/"
                ):
                    # allowed to cross http <-> https
                    out.add(l)
            else:
                if l not in ["..", "../"]:
                    # Ignore FTP-like "parent"
                    out.add("/".join([url.rstrip("/"), l.lstrip("/")]))
        if not out and url.endswith("/"):
            out = self._ls_real(url.rstrip("/"), detail=False)
        if detail:
            return [
                {
                    "name": u,
                    "size": None,
                    "type": "directory" if u.endswith("/") else "file",
                }
                for u in out
            ]
        else:
            return sorted(out)

    def ls(self, url, detail=True, **kwargs):
        if self.use_listings_cache and url in self.dircache:
            out = self.dircache[url]
        else:
            out = self._ls_real(url, detail=detail, **kwargs)
            self.dircache[url] = out
        return out

    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status_code == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()

    def cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = self._process_limits(url, start, end)
            kw["headers"] = headers
        r = self.session.get(self.encode_url(url), **kw)
        self._raise_not_found_for_status(r, url)
        return r.content

    def get_file(
        self, rpath, lpath, chunk_size=5 * 2**20, callback=_DEFAULT_CALLBACK, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(rpath)
        r = self.session.get(self.encode_url(rpath), **kw)
        try:
            size = int(
                r.headers.get("content-length", None)
                or r.headers.get("Content-Length", None)
            )
        except (ValueError, KeyError, TypeError):
            size = None

        callback.set_size(size)
        self._raise_not_found_for_status(r, rpath)
        if not isfilelike(lpath):
            lpath = open(lpath, "wb")
        for chunk in r.iter_content(chunk_size, decode_unicode=False):
            lpath.write(chunk)
            callback.relative_update(len(chunk))

    def put_file(
        self,
        lpath,
        rpath,
        chunk_size=5 * 2**20,
        callback=_DEFAULT_CALLBACK,
        method="post",
        **kwargs,
    ):
        def gen_chunks():
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")
                use_seek = True

            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, "size", None))

                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)

        kw = self.kwargs.copy()
        kw.update(kwargs)

        method = method.lower()
        if method not in ("post", "put"):
            raise ValueError(
                f"method has to be either 'post' or 'put', not: {method!r}"
            )

        meth = getattr(self.session, method)
        resp = meth(rpath, data=gen_chunks(), **kw)
        self._raise_not_found_for_status(resp, rpath)

    def _process_limits(self, url, start, end):
        """Helper for "Range"-based _cat_file"""
        size = None
        suff = False
        if start is not None and start < 0:
            # if start is negative and end None, end is the "suffix length"
            if end is None:
                end = -start
                start = ""
                suff = True
            else:
                size = size or self.info(url)["size"]
                start = size + start
        elif start is None:
            start = 0
        if not suff:
            if end is not None and end < 0:
                if start is not None:
                    size = size or self.info(url)["size"]
                    end = size + end
            elif end is None:
                end = ""
            if isinstance(end, int):
                end -= 1  # bytes range is inclusive
        return f"bytes={start}-{end}"

    def exists(self, path, strict=False, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            logger.debug(path)
            r = self.session.get(self.encode_url(path), **kw)
            if strict:
                self._raise_not_found_for_status(r, path)
            return r.status_code < 400
        except FileNotFoundError:
            return False
        except Exception:
            if strict:
                raise
            return False

    def isfile(self, path, **kwargs):
        return self.exists(path, **kwargs)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type=None,
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """Make a file-like object

        Parameters
        ----------
        path: str
            Full URL with protocol
        mode: string
            must be "rb"
        block_size: int or None
            Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
        kwargs: key-value
            Any other parameters, passed to requests calls
        """
        if mode != "rb":
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw.update(kwargs)
        size = size or self.info(path, **kwargs)["size"]
        if block_size and size:
            return HTTPFile(
                self,
                path,
                session=self.session,
                block_size=block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                **kw,
            )
        else:
            return HTTPStreamFile(
                self,
                path,
                mode=mode,
                session=self.session,
                **kw,
            )

    def ukey(self, url):
        """Unique identifier; assume HTTP files are static, unchanging"""
        return tokenize(url, self.kwargs, self.protocol)

    def info(self, url, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        for policy in ["head", "get"]:
            try:
                info.update(
                    _file_info(
                        self.encode_url(url),
                        size_policy=policy,
                        session=self.session,
                        **self.kwargs,
                        **kwargs,
                    )
                )
                if info.get("size") is not None:
                    break
            except Exception as exc:
                if policy == "get":
                    # If get failed, then raise a FileNotFoundError
                    raise FileNotFoundError(url) from exc
                logger.debug(str(exc))

        return {"name": url, "size": None, **info, "type": "file"}

    def glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        This implementation is idntical to the one in AbstractFileSystem,
        but "?" is not considered as a character for globbing, because it is
        so common in URLs, often identifying the "query" part.
        """
        import re

        ends = path.endswith("/")
        path = self._strip_protocol(path)
        indstar = path.find("*") if path.find("*") >= 0 else len(path)
        indbrace = path.find("[") if path.find("[") >= 0 else len(path)

        ind = min(indstar, indbrace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            root = path
            depth = 1
            if ends:
                path += "/*"
            elif self.exists(path):
                if not detail:
                    return [path]
                else:
                    return {path: self.info(path)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:ind]:
            ind2 = path[:ind].rindex("/")
            root = path[: ind2 + 1]
            depth = None if "**" in path else path[ind2 + 1 :].count("/") + 1
        else:
            root = ""
            depth = None if "**" in path else path[ind + 1 :].count("/") + 1

        allpaths = self.find(
            root, maxdepth=maxdepth or depth, withdirs=True, detail=True, **kwargs
        )
        # Escape characters special to python regex, leaving our supported
        # special characters in place.
        # See https://www.gnu.org/software/bash/manual/html_node/Pattern-Matching.html
        # for shell globbing details.
        pattern = (
            "^"
            + (
                path.replace("\\", r"\\")
                .replace(".", r"\.")
                .replace("+", r"\+")
                .replace("//", "/")
                .replace("(", r"\(")
                .replace(")", r"\)")
                .replace("|", r"\|")
                .replace("^", r"\^")
                .replace("$", r"\$")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .rstrip("/")
            )
            + "$"
        )
        pattern = re.sub("[*]{2}", "=PLACEHOLDER=", pattern)
        pattern = re.sub("[*]", "[^/]*", pattern)
        pattern = re.compile(pattern.replace("=PLACEHOLDER=", ".*"))
        out = {
            p: allpaths[p]
            for p in sorted(allpaths)
            if pattern.match(p.replace("//", "/").rstrip("/"))
        }
        if detail:
            return out
        else:
            return list(out)

    def isdir(self, path):
        # override, since all URLs are (also) files
        try:
            return bool(self.ls(path))
        except (FileNotFoundError, ValueError):
            return False


class HTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remove HTTP(S) resource

    Supports only reading, with read-ahead of a predermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    session: requests.Session or None
        All calls will be made within this session, to avoid restarting
        connections where the server allows this
    block_size: int or None
        The amount of read-ahead to do, in bytes. Default is 5MB, or the value
        configured for the FileSystem creating this file
    size: None or int
        If given, this is the size of the file in bytes, and we don't attempt
        to call the server to find the value.
    kwargs: all other key-values are passed to requests calls.
    """

    def __init__(
        self,
        fs,
        url,
        session=None,
        block_size=None,
        mode="rb",
        cache_type="bytes",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError("File mode not supported")
        self.url = url
        self.session = session
        self.details = {"name": url, "size": size, "type": "file"}
        super().__init__(
            fs=fs,
            path=url,
            mode=mode,
            block_size=block_size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )

    def read(self, length=-1):
        """Read bytes from file

        Parameters
        ----------
        length: int
            Read up to this many bytes. If negative, read all content to end of
            file. If the server has not supplied the filesize, attempting to
            read only part of the data will raise a ValueError.
        """
        if (
            (length < 0 and self.loc == 0)  # explicit read all
            # but not when the size is known and fits into a block anyways
            and not (self.size is not None and self.size <= self.blocksize)
        ):
            self._fetch_all()
        if self.size is None:
            if length < 0:
                self._fetch_all()
        else:
            length = min(self.size - self.loc, length)
        return super().read(length)

    def _fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f"Fetch all for {self}")
        if not isinstance(self.cache, AllBytes):
            r = self.session.get(self.fs.encode_url(self.url), **self.kwargs)
            r.raise_for_status()
            out = r.content
            self.cache = AllBytes(size=len(out), fetcher=None, blocksize=None, data=out)
            self.size = len(out)

    def _parse_content_range(self, headers):
        """Parse the Content-Range header"""
        s = headers.get("Content-Range", "")
        m = re.match(r"bytes (\d+-\d+|\*)/(\d+|\*)", s)
        if not m:
            return None, None, None

        if m[1] == "*":
            start = end = None
        else:
            start, end = [int(x) for x in m[1].split("-")]
        total = None if m[2] == "*" else int(m[2])
        return start, end, total

    def _fetch_range(self, start, end):
        """Download a block of data

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f"Fetch range for {self}: {start}-{end}")
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        logger.debug("%s : %s", self.url, headers["Range"])
        r = self.session.get(self.fs.encode_url(self.url), headers=headers, **kwargs)
        if r.status_code == 416:
            # range request outside file
            return b""
        r.raise_for_status()

        # If the server has handled the range request, it should reply
        # with status 206 (partial content). But we'll guess that a suitable
        # Content-Range header or a Content-Length no more than the
        # requested range also mean we have got the desired range.
        cl = r.headers.get("Content-Length", r.headers.get("content-length", end + 1))
        response_is_range = (
            r.status_code == 206
            or self._parse_content_range(r.headers)[0] == start
            or int(cl) <= end - start
        )

        if response_is_range:
            # partial content, as expected
            out = r.content
        elif start > 0:
            raise ValueError(
                "The HTTP server doesn't appear to support range requests. "
                "Only reading this file from the beginning is supported. "
                "Open with block_size=0 for a streaming file interface."
            )
        else:
            # Response is not a range, but we want the start of the file,
            # so we can read the required amount anyway.
            cl = 0
            out = []
            for chunk in r.iter_content(2**20, False):
                out.append(chunk)
                cl += len(chunk)
            out = b"".join(out)[: end - start]
        return out


magic_check = re.compile("([*[])")


def has_magic(s):
    match = magic_check.search(s)
    return match is not None


class HTTPStreamFile(AbstractBufferedFile):
    def __init__(self, fs, url, mode="rb", session=None, **kwargs):
        self.url = url
        self.session = session
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        super().__init__(fs=fs, path=url, mode=mode, cache_type="readahead", **kwargs)

        r = self.session.get(self.fs.encode_url(url), stream=True, **kwargs)
        self.fs._raise_not_found_for_status(r, url)
        self.it = r.iter_content(1024, False)
        self.leftover = b""

        self.r = r

    def seek(self, *args, **kwargs):
        raise ValueError("Cannot seek streaming HTTP file")

    def read(self, num=-1):
        bufs = [self.leftover]
        leng = len(self.leftover)
        while leng < num or num < 0:
            try:
                out = self.it.__next__()
            except StopIteration:
                break
            if out:
                bufs.append(out)
            else:
                break
            leng += len(out)
        out = b"".join(bufs)
        if num >= 0:
            self.leftover = out[num:]
            out = out[:num]
        else:
            self.leftover = b""
        self.loc += len(out)
        return out

    def close(self):
        self.r.close()
        self.closed = True


def get_range(session, url, start, end, **kwargs):
    # explicit get a range when we know it must be safe
    kwargs = kwargs.copy()
    headers = kwargs.pop("headers", {}).copy()
    headers["Range"] = f"bytes={start}-{end - 1}"
    r = session.get(url, headers=headers, **kwargs)
    r.raise_for_status()
    return r.content


def _file_info(url, session, size_policy="head", **kwargs):
    """Call HEAD on the server to get details about the file (size/checksum etc.)

    Default operation is to explicitly allow redirects and use encoding
    'identity' (no compression) to get the true size of the target.
    """
    logger.debug("Retrieve file size for %s", url)
    kwargs = kwargs.copy()
    ar = kwargs.pop("allow_redirects", True)
    head = kwargs.get("headers", {}).copy()
    # TODO: not allowed in JS
    # head["Accept-Encoding"] = "identity"
    kwargs["headers"] = head

    info = {}
    if size_policy == "head":
        r = session.head(url, allow_redirects=ar, **kwargs)
    elif size_policy == "get":
        r = session.get(url, allow_redirects=ar, **kwargs)
    else:
        raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    r.raise_for_status()

    # TODO:
    #  recognise lack of 'Accept-Ranges',
    #                 or 'Accept-Ranges': 'none' (not 'bytes')
    #  to mean streaming only, no random access => return None
    if "Content-Length" in r.headers:
        info["size"] = int(r.headers["Content-Length"])
    elif "Content-Range" in r.headers:
        info["size"] = int(r.headers["Content-Range"].split("/")[1])
    elif "content-length" in r.headers:
        info["size"] = int(r.headers["content-length"])
    elif "content-range" in r.headers:
        info["size"] = int(r.headers["content-range"].split("/")[1])

    for checksum_field in ["ETag", "Content-MD5", "Digest"]:
        if r.headers.get(checksum_field):
            info[checksum_field] = r.headers[checksum_field]

    return info


# importing this is enough to register it
def register():
    register_implementation("http", HTTPFileSystem, clobber=True)
    register_implementation("https", HTTPFileSystem, clobber=True)
    register_implementation("sync-http", HTTPFileSystem, clobber=True)
    register_implementation("sync-https", HTTPFileSystem, clobber=True)


register()


def unregister():
    from fsspec.implementations.http import HTTPFileSystem

    register_implementation("http", HTTPFileSystem, clobber=True)
    register_implementation("https", HTTPFileSystem, clobber=True)
<|endoftext|>import logging
import re
import tarfile

import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.compression import compr
from fsspec.utils import infer_compression

typemap = {b"0": "file", b"5": "directory"}

logger = logging.getLogger("tar")


class TarFileSystem(AbstractArchiveFileSystem):
    """Compressed Tar archives as a file-system (read-only)

    Supports the following formats:
    tar.gz, tar.bz2, tar.xz
    """

    root_marker = ""
    protocol = "tar"
    cachable = False

    def __init__(
        self,
        fo="",
        index_store=None,
        target_options=None,
        target_protocol=None,
        compression=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        target_options = target_options or {}

        if isinstance(fo, str):
            self.of = fsspec.open(fo, protocol=target_protocol, **target_options)
            fo = self.of.open()  # keep the reference

        # Try to infer compression.
        if compression is None:
            name = None

            # Try different ways to get hold of the filename. `fo` might either
            # be a `fsspec.LocalFileOpener`, an `io.BufferedReader` or an
            # `fsspec.AbstractFileSystem` instance.
            try:
                # Amended io.BufferedReader or similar.
                # This uses a "protocol extension" where original filenames are
                # propagated to archive-like filesystems in order to let them
                # infer the right compression appropriately.
                if hasattr(fo, "original"):
                    name = fo.original

                # fsspec.LocalFileOpener
                elif hasattr(fo, "path"):
                    name = fo.path

                # io.BufferedReader
                elif hasattr(fo, "name"):
                    name = fo.name

                # fsspec.AbstractFileSystem
                elif hasattr(fo, "info"):
                    name = fo.info()["name"]

            except Exception as ex:
                logger.warning(
                    f"Unable to determine file name, not inferring compression: {ex}"
                )

            if name is not None:
                compression = infer_compression(name)
                logger.info(f"Inferred compression {compression} from file name {name}")

        if compression is not None:
            # TODO: tarfile already implements compression with modes like "'r:gz'",
            #  but then would seek to offset in the file work?
            fo = compr[compression](fo)

        self._fo_ref = fo
        self.fo = fo  # the whole instance is a context
        self.tar = tarfile.TarFile(fileobj=self.fo)
        self.dir_cache = None

        self.index_store = index_store
        self.index = None
        self._index()

    def _index(self):
        # TODO: load and set saved index, if exists
        out = {}
        for ti in self.tar:
            info = ti.get_info()
            info["type"] = typemap.get(info["type"], "file")
            orig_name = info["name"].rstrip("/")
            # Collapse duplicate slashes in the filesystem-facing name.
            name = re.sub("/+", "/", orig_name)
            info["name"] = name
            out[name] = (info, ti.offset_data, orig_name)

        self.index = out
        # TODO: save index to self.index_store here, if set

    def _get_dirs(self):
        if self.dir_cache is not None:
            return

        # This enables ls to get directories as children as well as files
        self.dir_cache = {
            dirname: {"name": dirname, "size": 0, "type": "directory"}
            for dirname in self._all_dirnames(self.index)
        }
        self.dir_cache.update(
            {info["name"]: info for info, _, _ in self.index.values()}
        )

    def _open(self, path, mode="rb", **kwargs):
        if mode != "rb":
            raise ValueError("Read-only filesystem implementation")
        # Accept paths containing the archive's duplicate slashes too.
        path = re.sub("/+", "/", path)
        details, _, orig_name = self.index[path]
        if details["type"] != "file":
            raise ValueError("Can only handle regular files")
        return self.tar.extractfile(orig_name)

    def close(self):
        """Commits any write changes to the file. Done on ``del`` too."""
        self.tar.close()

    def __del__(self):
        if hasattr(self, "tar"):
            self.close()
            del self.tar
        if hasattr(self, "of") and hasattr(self.of, "__exit__"):
            self.of.__exit__(None, None, None)
<|endoftext|>"""
This module contains SMBFileSystem class responsible for handling access to
Windows Samba network shares by using package smbprotocol
"""

import datetime
import re
import uuid
from stat import S_ISDIR, S_ISLNK

import smbclient
import smbprotocol.exceptions

from .. import AbstractFileSystem
from ..utils import infer_storage_options

# ! pylint: disable=bad-continuation


class SMBFileSystem(AbstractFileSystem):
    """Allow reading and writing to Windows and Samba network shares.

    When using `fsspec.open()` for getting a file-like object the URI
    should be specified as this format:
    ``smb://workgroup;user:password@server:port/share/folder/file.csv``.

    Example::

        >>> import fsspec
        >>> with fsspec.open(
        ...     'smb://myuser:mypassword@myserver.com/' 'share/folder/file.csv'
        ... ) as smbfile:
        ...     df = pd.read_csv(smbfile, sep='|', header=None)

    Note that you need to pass in a valid hostname or IP address for the host
    component of the URL. Do not use the Windows/NetBIOS machine name for the
    host component.

    The first component of the path in the URL points to the name of the shared
    folder. Subsequent path components will point to the directory/folder/file.

    The URL components ``workgroup`` , ``user``, ``password`` and ``port`` may be
    optional.

    .. note::

        For working this source require `smbprotocol`_ to be installed, e.g.::

            $ pip install smbprotocol
            # or
            # pip install smbprotocol[kerberos]

    .. _smbprotocol: https://github.com/jborean93/smbprotocol#requirements

    Note: if using this with the ``open`` or ``open_files``, with full URLs,
    there is no way to tell if a path is relative, so all paths are assumed
    to be absolute.
    """

    protocol = "smb"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        host,
        port=None,
        username=None,
        password=None,
        timeout=60,
        encrypt=None,
        share_access=None,
        register_session_retries=4,
        register_session_retry_wait=1,
        register_session_retry_factor=10,
        auto_mkdir=False,
        **kwargs,
    ):
        """
        You can use _get_kwargs_from_urls to get some kwargs from
        a reasonable SMB url.

        Authentication will be anonymous or integrated if username/password are not
        given.

        Parameters
        ----------
        host: str
            The remote server name/ip to connect to
        port: int or None
            Port to connect with. Usually 445, sometimes 139.
        username: str or None
            Username to connect with. Required if Kerberos auth is not being used.
        password: str or None
            User's password on the server, if using username
        timeout: int
            Connection timeout in seconds
        encrypt: bool
            Whether to force encryption or not, once this has been set to True
            the session cannot be changed back to False.
        share_access: str or None
            Specifies the default access applied to file open operations
            performed with this file system object.
            This affects whether other processes can concurrently open a handle
            to the same file.

            - None (the default): exclusively locks the file until closed.
            - 'r': Allow other handles to be opened with read access.
            - 'w': Allow other handles to be opened with write access.
            - 'd': Allow other handles to be opened with delete access.
        register_session_retries: int
            Number of retries to register a session with the server. Retries are not performed
            for authentication errors, as they are considered as invalid credentials and not network
            issues. If set to negative value, no register attempts will be performed.
        register_session_retry_wait: int
            Time in seconds to wait between each retry. Number must be non-negative.
        register_session_retry_factor: int
            Base factor for the wait time between each retry. The wait time
            is calculated using exponential function. For factor=1 all wait times
            will be equal to `register_session_retry_wait`. For any number of retries,
            the last wait time will be equal to `register_session_retry_wait` and for retries>1
            the first wait time will be equal to `register_session_retry_wait / factor`.
            Number must be equal to or greater than 1. Optimal factor is 10.
        auto_mkdir: bool
            Whether, when opening a file, the directory containing it should
            be created (if it doesn't already exist). This is assumed by pyarrow
            and zarr-python code.
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.encrypt = encrypt
        self.temppath = kwargs.pop("temppath", "")
        self.share_access = share_access
        self.register_session_retries = register_session_retries
        if register_session_retry_wait < 0:
            raise ValueError(
                "register_session_retry_wait must be a non-negative integer"
            )
        self.register_session_retry_wait = register_session_retry_wait
        if register_session_retry_factor < 1:
            raise ValueError(
                "register_session_retry_factor must be a positive "
                "integer equal to or greater than 1"
            )
        self.register_session_retry_factor = register_session_retry_factor
        self.auto_mkdir = auto_mkdir
        self._connect()

    @property
    def _port(self):
        return 445 if self.port is None else self.port

    def _connect(self):
        import time

        if self.register_session_retries <= -1:
            return

        retried_errors = []

        wait_time = self.register_session_retry_wait
        n_waits = (
            self.register_session_retries - 1
        )  # -1 = No wait time after the last retry
        factor = self.register_session_retry_factor

        # Generate wait times for each retry attempt.
        # Wait times are calculated using exponential function. For factor=1 all wait times
        # will be equal to `wait`. For any number of retries the last wait time will be
        # equal to `wait` and for retries>2 the first wait time will be equal to `wait / factor`.
        wait_times = iter(
            factor ** (n / n_waits - 1) * wait_time for n in range(0, n_waits + 1)
        )

        for attempt in range(self.register_session_retries + 1):
            try:
                smbclient.register_session(
                    self.host,
                    username=self.username,
                    password=self.password,
                    port=self._port,
                    encrypt=self.encrypt,
                    connection_timeout=self.timeout,
                )
                return
            except (
                smbprotocol.exceptions.SMBAuthenticationError,
                smbprotocol.exceptions.LogonFailure,
            ):
                # These exceptions should not be repeated, as they clearly indicate
                # that the credentials are invalid and not a network issue.
                raise
            except ValueError as exc:
                if re.findall(r"\[Errno -\d+]", str(exc)):
                    # This exception is raised by the smbprotocol.transport:Tcp.connect
                    # and originates from socket.gaierror (OSError). These exceptions might
                    # be raised due to network instability. We will retry to connect.
                    retried_errors.append(exc)
                else:
                    # All another ValueError exceptions should be raised, as they are not
                    # related to network issues.
                    raise
            except Exception as exc:
                # Save the exception and retry to connect. This except might be dropped
                # in the future, once all exceptions suited for retry are identified.
                retried_errors.append(exc)

            if attempt < self.register_session_retries:
                time.sleep(next(wait_times))

        # Raise last exception to inform user about the connection issues.
        # Note: Should we use ExceptionGroup to raise all exceptions?
        raise retried_errors[-1]

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(path):
        # smb://workgroup;user:password@host:port/share/folder/file.csv
        out = infer_storage_options(path)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def mkdir(self, path, create_parents=True, **kwargs):
        wpath = _as_unc_path(self.host, path)
        if create_parents:
            smbclient.makedirs(wpath, exist_ok=False, port=self._port, **kwargs)
        else:
            smbclient.mkdir(wpath, port=self._port, **kwargs)

    def makedirs(self, path, exist_ok=False):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.makedirs(wpath, exist_ok=exist_ok, port=self._port)

    def rmdir(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.rmdir(wpath, port=self._port)

    def info(self, path, **kwargs):
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port, **kwargs)
        if S_ISDIR(stats.st_mode):
            stype = "directory"
        elif S_ISLNK(stats.st_mode):
            stype = "link"
        else:
            stype = "file"
        res = {
            "name": path + "/" if stype == "directory" else path,
            "size": stats.st_size,
            "type": stype,
            "uid": stats.st_uid,
            "gid": stats.st_gid,
            "time": stats.st_atime,
            "mtime": stats.st_mtime,
        }
        return res

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_ctime, tz=datetime.timezone.utc)

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_mtime, tz=datetime.timezone.utc)

    def ls(self, path, detail=True, **kwargs):
        unc = _as_unc_path(self.host, path)
        listed = smbclient.listdir(unc, port=self._port, **kwargs)
        dirs = ["/".join([path.rstrip("/"), p]) for p in listed]
        if detail:
            dirs = [self.info(d) for d in dirs]
        return dirs

    # pylint: disable=too-many-arguments
    def _open(
        self,
        path,
        mode="rb",
        block_size=-1,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """
        block_size: int or None
            If 0, no buffering, 1, line buffering, >1, buffer that many bytes

        Notes
        -----
        By specifying 'share_access' in 'kwargs' it is possible to override the
        default shared access setting applied in the constructor of this object.
        """
        if self.auto_mkdir and "w" in mode:
            self.makedirs(self._parent(path), exist_ok=True)
        bls = block_size if block_size is not None and block_size >= 0 else -1
        wpath = _as_unc_path(self.host, path)
        share_access = kwargs.pop("share_access", self.share_access)
        if "w" in mode and autocommit is False:
            temp = _as_temp_path(self.host, path, self.temppath)
            return SMBFileOpener(
                wpath, temp, mode, port=self._port, block_size=bls, **kwargs
            )
        return smbclient.open_file(
            wpath,
            mode,
            buffering=bls,
            share_access=share_access,
            port=self._port,
            **kwargs,
        )

    def copy(self, path1, path2, **kwargs):
        """Copy within two locations in the same filesystem"""
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        if self.auto_mkdir:
            self.makedirs(self._parent(path2), exist_ok=True)
        smbclient.copyfile(wpath1, wpath2, port=self._port, **kwargs)

    def _rm(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            stats = smbclient.stat(wpath, port=self._port)
            if S_ISDIR(stats.st_mode):
                smbclient.rmdir(wpath, port=self._port)
            else:
                smbclient.remove(wpath, port=self._port)

    def mv(self, path1, path2, recursive=None, maxdepth=None, **kwargs):
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        smbclient.rename(wpath1, wpath2, port=self._port, **kwargs)


def _as_unc_path(host, path):
    rpath = path.replace("/", "\\")
    unc = f"\\\\{host}{rpath}"
    return unc


def _as_temp_path(host, path, temppath):
    share = path.split("/")[1]
    temp_file = f"/{share}{temppath}/{uuid.uuid4()}"
    unc = _as_unc_path(host, temp_file)
    return unc


def _share_has_path(path):
    parts = path.count("/")
    if path.endswith("/"):
        return parts > 2
    return parts > 1


class SMBFileOpener:
    """writes to remote temporary file, move on commit"""

    def __init__(self, path, temp, mode, port=445, block_size=-1, **kwargs):
        self.path = path
        self.temp = temp
        self.mode = mode
        self.block_size = block_size
        self.kwargs = kwargs
        self.smbfile = None
        self._incontext = False
        self.port = port
        self._open()

    def _open(self):
        if self.smbfile is None or self.smbfile.closed:
            self.smbfile = smbclient.open_file(
                self.temp,
                self.mode,
                port=self.port,
                buffering=self.block_size,
                **self.kwargs,
            )

    def commit(self):
        """Move temp file to definitive on success."""
        # TODO: use transaction support in SMB protocol
        smbclient.replace(self.temp, self.path, port=self.port)

    def discard(self):
        """Remove the temp file on failure."""
        smbclient.remove(self.temp, port=self.port)

    def __fspath__(self):
        return self.path

    def __iter__(self):
        return self.smbfile.__iter__()

    def __getattr__(self, item):
        return getattr(self.smbfile, item)

    def __enter__(self):
        self._incontext = True
        return self.smbfile.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._incontext = False
        self.smbfile.__exit__(exc_type, exc_value, traceback)
<|endoftext|>from __future__ import annotations

import abc
import hashlib

from fsspec.implementations.local import make_path_posix


class AbstractCacheMapper(abc.ABC):
    """Abstract super-class for mappers from remote URLs to local cached
    basenames.
    """

    @abc.abstractmethod
    def __call__(self, path: str) -> str: ...

    def __eq__(self, other: object) -> bool:
        # Identity only depends on class. When derived classes have attributes
        # they will need to be included.
        return isinstance(other, type(self))

    def __hash__(self) -> int:
        # Identity only depends on class. When derived classes have attributes
        # they will need to be included.
        return hash(type(self))


class BasenameCacheMapper(AbstractCacheMapper):
    """Cache mapper that uses the basename of the remote URL and a fixed number
    of directory levels above this.

    The default is zero directory levels, meaning different paths with the same
    basename will have the same cached basename.
    """

    def __init__(self, directory_levels: int = 0):
        if directory_levels < 0:
            raise ValueError(
                "BasenameCacheMapper requires zero or positive directory_levels"
            )
        self.directory_levels = directory_levels

        # Separator for directories when encoded as strings.
        self._separator = "_@_"

    def __call__(self, path: str) -> str:
        path = make_path_posix(path)
        prefix, *bits = path.rsplit("/", self.directory_levels + 1)
        if bits:
            return self._separator.join(bits)
        else:
            return prefix  # No separator found, simple filename

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self.directory_levels == other.directory_levels

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.directory_levels)


class HashCacheMapper(AbstractCacheMapper):
    """Cache mapper that uses a hash of the remote URL."""

    def __call__(self, path: str) -> str:
        return hashlib.sha256(path.encode()).hexdigest()


def create_cache_mapper(same_names: bool) -> AbstractCacheMapper:
    """Factory method to create cache mapper for backward compatibility with
    ``CachingFileSystem`` constructor using ``same_names`` kwarg.
    """
    if same_names:
        return BasenameCacheMapper()
    else:
        return HashCacheMapper()
<|endoftext|>from typing import ClassVar

from fsspec import AbstractFileSystem

__all__ = ("ChainedFileSystem",)


class ChainedFileSystem(AbstractFileSystem):
    """Chained filesystem base class.

    A chained filesystem is designed to be layered over another FS.
    This is useful to implement things like caching.

    This base class does very little on its own, but is used as a marker
    that the class is designed for chaining.

    Right now this is only used in `url_to_fs` to provide the path argument
    (`fo`) to the chained filesystem from the underlying filesystem.

    Additional functionality may be added in the future.
    """

    protocol: ClassVar[str] = "chained"
<|endoftext|>import os
import ssl
import uuid
from ftplib import FTP, FTP_TLS, Error, error_perm
from typing import Any

from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, isfilelike

SECURITY_PROTOCOL_MAP = {
    "tls": ssl.PROTOCOL_TLS,
    "sslv23": ssl.PROTOCOL_SSLv23,
}
for protocol in ["TLSv1", "TLSv1_1", "TLSv1_2"]:
    if hasattr(ssl, f"PROTOCOL_{protocol}"):
        SECURITY_PROTOCOL_MAP[protocol.lower()] = getattr(ssl, f"PROTOCOL_{protocol}")


class ImplicitFTPTLS(FTP_TLS):
    """
    FTP_TLS subclass that automatically wraps sockets in SSL
    to support implicit FTPS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sock = None

    @property
    def sock(self):
        """Return the socket."""
        return self._sock

    @sock.setter
    def sock(self, value):
        """When modifying the socket, ensure that it is ssl wrapped."""
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value)
        self._sock = value


class FTPFileSystem(AbstractFileSystem):
    """A filesystem over classic FTP"""

    root_marker = "/"
    cachable = False
    protocol = "ftp"

    def __init__(
        self,
        host,
        port=21,
        username=None,
        password=None,
        acct=None,
        block_size=None,
        tempdir=None,
        timeout=30,
        encoding="utf-8",
        tls=False,
        **kwargs,
    ):
        """
        You can use _get_kwargs_from_urls to get some kwargs from
        a reasonable FTP url.

        Authentication will be anonymous if username/password are not
        given.

        Parameters
        ----------
        host: str
            The remote server name/ip to connect to
        port: int
            Port to connect with
        username: str or None
            If authenticating, the user's identifier
        password: str of None
            User's password on the server, if using
        acct: str or None
            Some servers also need an "account" string for auth
        block_size: int or None
            If given, the read-ahead or write buffer size.
        tempdir: str
            Directory on remote to put temporary files when in a transaction
        timeout: int
            Timeout of the ftp connection in seconds
        encoding: str
            Encoding to use for directories and filenames in FTP connection
        tls: bool or str
            Enable FTP-TLS for secure connections:
                - False: Plain FTP (default)
                - True: Explicit TLS (FTPS with AUTH TLS command)
                - "tls": Auto-negotiate highest protocol
                - "tlsv1": TLS v1.0
                - "tlsv1_1": TLS v1.1
                - "tlsv1_2": TLS v1.2
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.tempdir = tempdir or "/tmp"
        self.cred = username or "", password or "", acct or ""
        self.timeout = timeout
        self.encoding = encoding
        if block_size is not None:
            self.blocksize = block_size
        else:
            self.blocksize = 2**16
        self.tls = tls
        self._connect()
        if isinstance(self.tls, bool) and self.tls:
            self.ftp.prot_p()

    def _connect(self):
        security = None
        if self.tls:
            if isinstance(self.tls, str):
                ftp_cls = ImplicitFTPTLS
                security = SECURITY_PROTOCOL_MAP.get(
                    self.tls,
                    f"Not supported {self.tls} protocol",
                )
                if isinstance(security, str):
                    raise ValueError(security)
            else:
                ftp_cls = FTP_TLS
        else:
            ftp_cls = FTP
        self.ftp = ftp_cls(timeout=self.timeout, encoding=self.encoding)
        if security:
            self.ftp.ssl_version = security
        self.ftp.connect(self.host, self.port)
        self.ftp.login(*self.cred)

    @classmethod
    def _strip_protocol(cls, path):
        return "/" + infer_storage_options(path)["path"].lstrip("/").rstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        out = []
        if path not in self.dircache:
            try:
                try:
                    out = [
                        (fn, details)
                        for (fn, details) in self.ftp.mlsd(path)
                        if fn not in [".", ".."]
                        and details["type"] not in ["pdir", "cdir"]
                    ]
                except error_perm:
                    out = _mlsd2(self.ftp, path)  # Not platform independent
                for fn, details in out:
                    details["name"] = "/".join(
                        ["" if path == "/" else path, fn.lstrip("/")]
                    )
                    if details["type"] == "file":
                        details["size"] = int(details["size"])
                    else:
                        details["size"] = 0
                    if details["type"] == "dir":
                        details["type"] = "directory"
                self.dircache[path] = out
            except Error:
                try:
                    info = self.info(path)
                    if info["type"] == "file":
                        out = [(path, info)]
                except (Error, IndexError) as exc:
                    raise FileNotFoundError(path) from exc
        files = self.dircache.get(path, out)
        if not detail:
            return sorted([fn for fn, details in files])
        return [details for fn, details in files]

    def info(self, path, **kwargs):
        # implement with direct method
        path = self._strip_protocol(path)
        if path == "/":
            # special case, since this dir has no real entry
            return {"name": "/", "size": 0, "type": "directory"}
        files = self.ls(self._parent(path).lstrip("/"), True)
        try:
            out = next(f for f in files if f["name"] == path)
        except StopIteration as exc:
            raise FileNotFoundError(path) from exc
        return out

    def get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            if not os.path.exists(lpath):
                os.mkdir(lpath)
            return
        if isfilelike(lpath):
            outfile = lpath
        else:
            outfile = open(lpath, "wb")

        def cb(x):
            outfile.write(x)

        self.ftp.retrbinary(
            f"RETR {rpath}",
            blocksize=self.blocksize,
            callback=cb,
        )
        if not isfilelike(lpath):
            outfile.close()

    def cat_file(self, path, start=None, end=None, **kwargs):
        if end is not None:
            return super().cat_file(path, start, end, **kwargs)
        out = []

        def cb(x):
            out.append(x)

        try:
            self.ftp.retrbinary(
                f"RETR {path}",
                blocksize=self.blocksize,
                rest=start,
                callback=cb,
            )
        except (Error, error_perm) as orig_exc:
            raise FileNotFoundError(path) from orig_exc
        return b"".join(out)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        autocommit=True,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        block_size = block_size or self.blocksize
        return FTPFile(
            self,
            path,
            mode=mode,
            block_size=block_size,
            tempdir=self.tempdir,
            autocommit=autocommit,
            cache_options=cache_options,
        )

    def _rm(self, path):
        path = self._strip_protocol(path)
        self.ftp.delete(path)
        self.invalidate_cache(self._parent(path))

    def rm(self, path, recursive=False, maxdepth=None):
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            else:
                self.rmdir(p)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        parent = self._parent(path)
        if parent != self.root_marker and not self.exists(parent) and create_parents:
            self.mkdir(parent, create_parents=create_parents)

        self.ftp.mkd(path)
        self.invalidate_cache(self._parent(path))

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._strip_protocol(path)
        if self.exists(path):
            # NB: "/" does not "exist" as it has no directory entry
            if not exist_ok:
                raise FileExistsError(f"{path} exists without `exist_ok`")
            # exists_ok=True -> no-op
        else:
            self.mkdir(path, create_parents=True)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        self.ftp.rmd(path)
        self.invalidate_cache(self._parent(path))

    def mv(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        self.ftp.rename(path1, path2)
        self.invalidate_cache(self._parent(path1))
        self.invalidate_cache(self._parent(path2))

    def __del__(self):
        self.ftp.close()

    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path, None)
        super().invalidate_cache(path)


class TransferDone(Exception):
    """Internal exception to break out of transfer"""

    pass


class FTPFile(AbstractBufferedFile):
    """Interact with a remote FTP file with read/write buffering"""

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        **kwargs,
    ):
        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )
        if not autocommit:
            self.target = self.path
            self.path = "/".join([kwargs["tempdir"], str(uuid.uuid4())])

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)

    def _fetch_range(self, start, end):
        """Get bytes between given byte limits

        Implemented by raising an exception in the fetch callback when the
        number of bytes received reaches the requested amount.

        Will fail if the server does not respect the REST command on
        retrieve requests.
        """
        out = []
        total = [0]

        def callback(x):
            total[0] += len(x)
            if total[0] > end - start:
                out.append(x[: (end - start) - total[0]])
                if end < self.size:
                    raise TransferDone
            else:
                out.append(x)

            if total[0] == end - start and end < self.size:
                raise TransferDone

        try:
            self.fs.ftp.retrbinary(
                f"RETR {self.path}",
                blocksize=self.blocksize,
                rest=start,
                callback=callback,
            )
        except TransferDone:
            try:
                # stop transfer, we got enough bytes for this block
                self.fs.ftp.abort()
                self.fs.ftp.getmultiline()
            except Error:
                self.fs._connect()

        return b"".join(out)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        self.fs.ftp.storbinary(
            f"STOR {self.path}", self.buffer, blocksize=self.blocksize, rest=self.offset
        )
        return True


def _mlsd2(ftp, path="."):
    """
    Fall back to using `dir` instead of `mlsd` if not supported.

    This parses a Linux style `ls -l` response to `dir`, but the response may
    be platform dependent.

    Parameters
    ----------
    ftp: ftplib.FTP
    path: str
        Expects to be given path, but defaults to ".".
    """
    lines = []
    minfo = []
    ftp.dir(path, lines.append)
    for line in lines:
        split_line = line.split(maxsplit=8)
        if len(split_line) < 9:
            continue
        name = split_line[8]
        unix_mode = split_line[0]
        if unix_mode[0] == "l" and " -> " in name:
            # Symbolic link: "<name> -> <target>"; keep only the link name.
            name = name.split(" -> ", 1)[0]
        this = (
            name,
            {
                "modify": " ".join(split_line[5:8]),
                "unix.owner": split_line[2],
                "unix.group": split_line[3],
                "unix.mode": unix_mode,
                "size": split_line[4],
            },
        )
        if this[1]["unix.mode"][0] == "d":
            this[1]["type"] = "dir"
        else:
            this[1]["type"] = "file"
        minfo.append(this)
    return minfo
<|endoftext|># This module contains abstractions for the input stream. You don't have to
# looks further, there are no pretty code.
#
# We define two classes here.
#
#   Mark(source, line, column)
# It's just a record and its only use is producing nice error messages.
# Parser does not use it for any other purposes.
#
#   Reader(source, data)
# Reader determines the encoding of `data` and converts it to unicode.
# Reader provides the following methods and attributes:
#   reader.peek(length=1) - return the next `length` characters
#   reader.forward(length=1) - move the current position to `length` characters.
#   reader.index - the number of the current character.
#   reader.line, stream.column - the line and the column of the current character.

__all__ = ['Reader', 'ReaderError']

from .error import YAMLError, Mark

import codecs, re

class ReaderError(YAMLError):

    def __init__(self, name, position, character, encoding, reason):
        self.name = name
        self.character = character
        self.position = position
        self.encoding = encoding
        self.reason = reason

    def __str__(self):
        if isinstance(self.character, bytes):
            return "'%s' codec can't decode byte #x%02x: %s\n"  \
                    "  in \"%s\", position %d"    \
                    % (self.encoding, ord(self.character), self.reason,
                            self.name, self.position)
        else:
            return "unacceptable character #x%04x: %s\n"    \
                    "  in \"%s\", position %d"    \
                    % (self.character, self.reason,
                            self.name, self.position)

class Reader(object):
    # Reader:
    # - determines the data encoding and converts it to a unicode string,
    # - checks if characters are in allowed range,
    # - adds '\0' to the end.

    # Reader accepts
    #  - a `bytes` object,
    #  - a `str` object,
    #  - a file-like object with its `read` method returning `str`,
    #  - a file-like object with its `read` method returning `unicode`.

    # Yeah, it's ugly and slow.

    def __init__(self, stream):
        self.name = None
        self.stream = None
        self.stream_pointer = 0
        self.eof = True
        self.buffer = ''
        self.pointer = 0
        self.raw_buffer = None
        self.raw_decode = None
        self.encoding = None
        self.index = 0
        self.line = 0
        self.column = 0
        if isinstance(stream, str):
            self.name = "<unicode string>"
            self.check_printable(stream)
            self.buffer = stream+'\0'
        elif isinstance(stream, bytes):
            self.name = "<byte string>"
            self.raw_buffer = stream
            self.determine_encoding()
        else:
            self.stream = stream
            self.name = getattr(stream, 'name', "<file>")
            self.eof = False
            self.raw_buffer = None
            self.determine_encoding()

    def peek(self, index=0):
        try:
            return self.buffer[self.pointer+index]
        except IndexError:
            self.update(index+1)
            return self.buffer[self.pointer+index]

    def prefix(self, length=1):
        if self.pointer+length >= len(self.buffer):
            self.update(length)
        return self.buffer[self.pointer:self.pointer+length]

    def forward(self, length=1):
        if self.pointer+length+1 >= len(self.buffer):
            self.update(length+1)
        while length:
            ch = self.buffer[self.pointer]
            self.pointer += 1
            self.index += 1
            if ch in '\n\x85\u2028\u2029'  \
                    or (ch == '\r' and self.buffer[self.pointer] != '\n'):
                self.line += 1
                self.column = 0
            elif ch != '\uFEFF':
                self.column += 1
            length -= 1

    def get_mark(self):
        if self.stream is None:
            return Mark(self.name, self.index, self.line, self.column,
                    self.buffer, self.pointer)
        else:
            return Mark(self.name, self.index, self.line, self.column,
                    None, None)

    def determine_encoding(self):
        while not self.eof and (self.raw_buffer is None or len(self.raw_buffer) < 2):
            self.update_raw()
        if isinstance(self.raw_buffer, bytes):
            if self.raw_buffer.startswith(codecs.BOM_UTF16_LE):
                self.raw_decode = codecs.utf_16_le_decode
                self.encoding = 'utf-16-le'
            elif self.raw_buffer.startswith(codecs.BOM_UTF16_BE):
                self.raw_decode = codecs.utf_16_be_decode
                self.encoding = 'utf-16-be'
            else:
                self.raw_decode = codecs.utf_8_decode
                self.encoding = 'utf-8'
        self.update(1)

    NON_PRINTABLE = re.compile('[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]')
    def check_printable(self, data):
        match = self.NON_PRINTABLE.search(data)
        if match:
            character = match.group()
            position = self.index+(len(self.buffer)-self.pointer)+match.start()
            raise ReaderError(self.name, position, ord(character),
                    'unicode', "special characters are not allowed")

    def update(self, length):
        if self.raw_buffer is None:
            return
        self.buffer = self.buffer[self.pointer:]
        self.pointer = 0
        while len(self.buffer) < length:
            if not self.eof:
                self.update_raw()
            if self.raw_decode is not None:
                try:
                    data, converted = self.raw_decode(self.raw_buffer,
                            'strict', self.eof)
                except UnicodeDecodeError as exc:
                    character = self.raw_buffer[exc.start]
                    if self.stream is not None:
                        position = self.stream_pointer-len(self.raw_buffer)+exc.start
                    else:
                        position = exc.start
                    raise ReaderError(self.name, position, character,
                            exc.encoding, exc.reason)
            else:
                data = self.raw_buffer
                converted = len(data)
            self.check_printable(data)
            self.buffer += data
            self.raw_buffer = self.raw_buffer[converted:]
            if self.eof:
                self.buffer += '\0'
                self.raw_buffer = None
                break

    def update_raw(self, size=4096):
        data = self.stream.read(size)
        if self.raw_buffer is None:
            self.raw_buffer = data
        else:
            self.raw_buffer += data
        self.stream_pointer += len(data)
        if not data:
            self.eof = True
<|endoftext|>
# Abstract classes.

class Event(object):
    def __init__(self, start_mark=None, end_mark=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
    def __repr__(self):
        attributes = [key for key in ['anchor', 'tag', 'implicit', 'value']
                if hasattr(self, key)]
        arguments = ', '.join(['%s=%r' % (key, getattr(self, key))
                for key in attributes])
        return '%s(%s)' % (self.__class__.__name__, arguments)

class NodeEvent(Event):
    def __init__(self, anchor, start_mark=None, end_mark=None):
        self.anchor = anchor
        self.start_mark = start_mark
        self.end_mark = end_mark

class CollectionStartEvent(NodeEvent):
    def __init__(self, anchor, tag, implicit, start_mark=None, end_mark=None,
            flow_style=None):
        self.anchor = anchor
        self.tag = tag
        self.implicit = implicit
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.flow_style = flow_style

class CollectionEndEvent(Event):
    pass

# Implementations.

class StreamStartEvent(Event):
    def __init__(self, start_mark=None, end_mark=None, encoding=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.encoding = encoding

class StreamEndEvent(Event):
    pass

class DocumentStartEvent(Event):
    def __init__(self, start_mark=None, end_mark=None,
            explicit=None, version=None, tags=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.explicit = explicit
        self.version = version
        self.tags = tags

class DocumentEndEvent(Event):
    def __init__(self, start_mark=None, end_mark=None,
            explicit=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.explicit = explicit

class AliasEvent(NodeEvent):
    pass

class ScalarEvent(NodeEvent):
    def __init__(self, anchor, tag, implicit, value,
            start_mark=None, end_mark=None, style=None):
        self.anchor = anchor
        self.tag = tag
        self.implicit = implicit
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style

class SequenceStartEvent(CollectionStartEvent):
    pass

class SequenceEndEvent(CollectionEndEvent):
    pass

class MappingStartEvent(CollectionStartEvent):
    pass

class MappingEndEvent(CollectionEndEvent):
    pass

<|endoftext|>
__all__ = ['Composer', 'ComposerError']

from .error import MarkedYAMLError
from .events import *
from .nodes import *

class ComposerError(MarkedYAMLError):
    pass

class Composer:

    def __init__(self):
        self.anchors = {}

    def check_node(self):
        # Drop the STREAM-START event.
        if self.check_event(StreamStartEvent):
            self.get_event()

        # If there are more documents available?
        return not self.check_event(StreamEndEvent)

    def get_node(self):
        # Get the root node of the next document.
        if not self.check_event(StreamEndEvent):
            return self.compose_document()

    def get_single_node(self):
        # Drop the STREAM-START event.
        self.get_event()

        # Compose a document if the stream is not empty.
        document = None
        if not self.check_event(StreamEndEvent):
            document = self.compose_document()

        # Ensure that the stream contains no more documents.
        if not self.check_event(StreamEndEvent):
            event = self.get_event()
            raise ComposerError("expected a single document in the stream",
                    document.start_mark, "but found another document",
                    event.start_mark)

        # Drop the STREAM-END event.
        self.get_event()

        return document

    def compose_document(self):
        # Drop the DOCUMENT-START event.
        self.get_event()

        # Compose the root node.
        node = self.compose_node(None, None)

        # Drop the DOCUMENT-END event.
        self.get_event()

        self.anchors = {}
        return node

    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            event = self.get_event()
            anchor = event.anchor
            if anchor not in self.anchors:
                raise ComposerError(None, None, "found undefined alias %r"
                        % anchor, event.start_mark)
            return self.anchors[anchor]
        event = self.peek_event()
        anchor = event.anchor
        if anchor is not None:
            if anchor in self.anchors:
                raise ComposerError("found duplicate anchor %r; first occurrence"
                        % anchor, self.anchors[anchor].start_mark,
                        "second occurrence", event.start_mark)
        self.descend_resolver(parent, index)
        if self.check_event(ScalarEvent):
            node = self.compose_scalar_node(anchor)
        elif self.check_event(SequenceStartEvent):
            node = self.compose_sequence_node(anchor)
        elif self.check_event(MappingStartEvent):
            node = self.compose_mapping_node(anchor)
        self.ascend_resolver()
        return node

    def compose_scalar_node(self, anchor):
        event = self.get_event()
        tag = event.tag
        if tag is None or tag == '!':
            tag = self.resolve(ScalarNode, event.value, event.implicit)
        node = ScalarNode(tag, event.value,
                event.start_mark, event.end_mark, style=event.style)
        if anchor is not None:
            self.anchors[anchor] = node
        return node

    def compose_sequence_node(self, anchor):
        start_event = self.get_event()
        tag = start_event.tag
        if tag is None or tag == '!':
            tag = self.resolve(SequenceNode, None, start_event.implicit)
        node = SequenceNode(tag, [],
                start_event.start_mark, None,
                flow_style=start_event.flow_style)
        if anchor is not None:
            self.anchors[anchor] = node
        index = 0
        while not self.check_event(SequenceEndEvent):
            node.value.append(self.compose_node(node, index))
            index += 1
        end_event = self.get_event()
        node.end_mark = end_event.end_mark
        return node

    def compose_mapping_node(self, anchor):
        start_event = self.get_event()
        tag = start_event.tag
        if tag is None or tag == '!':
            tag = self.resolve(MappingNode, None, start_event.implicit)
        node = MappingNode(tag, [],
                start_event.start_mark, None,
                flow_style=start_event.flow_style)
        if anchor is not None:
            self.anchors[anchor] = node
        while not self.check_event(MappingEndEvent):
            #key_event = self.peek_event()
            item_key = self.compose_node(node, None)
            #if item_key in node.value:
            #    raise ComposerError("while composing a mapping", start_event.start_mark,
            #            "found duplicate key", key_event.start_mark)
            item_value = self.compose_node(node, item_key)
            #node.value[item_key] = item_value
            node.value.append((item_key, item_value))
        end_event = self.get_event()
        node.end_mark = end_event.end_mark
        return node

<|endoftext|>
__all__ = [
    'BaseConstructor',
    'SafeConstructor',
    'FullConstructor',
    'UnsafeConstructor',
    'Constructor',
    'ConstructorError'
]

from .error import *
from .nodes import *

import collections.abc, datetime, base64, binascii, re, sys, types

class ConstructorError(MarkedYAMLError):
    pass

class BaseConstructor:

    yaml_constructors = {}
    yaml_multi_constructors = {}

    def __init__(self):
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.state_generators = []
        self.deep_construct = False

    def check_data(self):
        # If there are more documents available?
        return self.check_node()

    def check_state_key(self, key):
        """Block special attributes/methods from being set in a newly created
        object, to prevent user-controlled methods from being called during
        deserialization"""
        if self.get_state_keys_blacklist_regexp().match(key):
            raise ConstructorError(None, None,
                "blacklisted key '%s' in instance state found" % (key,), None)

    def get_data(self):
        # Construct and return the next document.
        if self.check_node():
            return self.construct_document(self.get_node())

    def get_single_data(self):
        # Ensure that the stream contains a single document and construct it.
        node = self.get_single_node()
        if node is not None:
            return self.construct_document(node)
        return None

    def construct_document(self, node):
        data = self.construct_object(node)
        while self.state_generators:
            state_generators = self.state_generators
            self.state_generators = []
            for generator in state_generators:
                for dummy in generator:
                    pass
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.deep_construct = False
        return data

    def construct_object(self, node, deep=False):
        if node in self.constructed_objects:
            return self.constructed_objects[node]
        if deep:
            old_deep = self.deep_construct
            self.deep_construct = True
        if node in self.recursive_objects:
            raise ConstructorError(None, None,
                    "found unconstructable recursive node", node.start_mark)
        self.recursive_objects[node] = None
        constructor = None
        tag_suffix = None
        if node.tag in self.yaml_constructors:
            constructor = self.yaml_constructors[node.tag]
        else:
            for tag_prefix in self.yaml_multi_constructors:
                if tag_prefix is not None and node.tag.startswith(tag_prefix):
                    tag_suffix = node.tag[len(tag_prefix):]
                    constructor = self.yaml_multi_constructors[tag_prefix]
                    break
            else:
                if None in self.yaml_multi_constructors:
                    tag_suffix = node.tag
                    constructor = self.yaml_multi_constructors[None]
                elif None in self.yaml_constructors:
                    constructor = self.yaml_constructors[None]
                elif isinstance(node, ScalarNode):
                    constructor = self.__class__.construct_scalar
                elif isinstance(node, SequenceNode):
                    constructor = self.__class__.construct_sequence
                elif isinstance(node, MappingNode):
                    constructor = self.__class__.construct_mapping
        if tag_suffix is None:
            data = constructor(self, node)
        else:
            data = constructor(self, tag_suffix, node)
        if isinstance(data, types.GeneratorType):
            generator = data
            data = next(generator)
            if self.deep_construct:
                for dummy in generator:
                    pass
            else:
                self.state_generators.append(generator)
        self.constructed_objects[node] = data
        del self.recursive_objects[node]
        if deep:
            self.deep_construct = old_deep
        return data

    def construct_scalar(self, node):
        if not isinstance(node, ScalarNode):
            raise ConstructorError(None, None,
                    "expected a scalar node, but found %s" % node.id,
                    node.start_mark)
        return node.value

    def construct_sequence(self, node, deep=False):
        if not isinstance(node, SequenceNode):
            raise ConstructorError(None, None,
                    "expected a sequence node, but found %s" % node.id,
                    node.start_mark)
        return [self.construct_object(child, deep=deep)
                for child in node.value]

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                    "expected a mapping node, but found %s" % node.id,
                    node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, collections.abc.Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark,
                        "found unhashable key", key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    def construct_pairs(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                    "expected a mapping node, but found %s" % node.id,
                    node.start_mark)
        pairs = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            pairs.append((key, value))
        return pairs

    @classmethod
    def add_constructor(cls, tag, constructor):
        if not 'yaml_constructors' in cls.__dict__:
            cls.yaml_constructors = cls.yaml_constructors.copy()
        cls.yaml_constructors[tag] = constructor

    @classmethod
    def add_multi_constructor(cls, tag_prefix, multi_constructor):
        if not 'yaml_multi_constructors' in cls.__dict__:
            cls.yaml_multi_constructors = cls.yaml_multi_constructors.copy()
        cls.yaml_multi_constructors[tag_prefix] = multi_constructor

class SafeConstructor(BaseConstructor):

    def construct_scalar(self, node):
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if key_node.tag == 'tag:yaml.org,2002:value':
                    return self.construct_scalar(value_node)
        return super().construct_scalar(node)

    def flatten_mapping(self, node):
        merge = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == 'tag:yaml.org,2002:merge':
                del node.value[index]
                if isinstance(value_node, MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, SequenceNode):
                    submerge = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, MappingNode):
                            raise ConstructorError("while constructing a mapping",
                                    node.start_mark,
                                    "expected a mapping for merging, but found %s"
                                    % subnode.id, subnode.start_mark)
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    submerge.reverse()
                    for value in submerge:
                        merge.extend(value)
                else:
                    raise ConstructorError("while constructing a mapping", node.start_mark,
                            "expected a mapping or list of mappings for merging, but found %s"
                            % value_node.id, value_node.start_mark)
            elif key_node.tag == 'tag:yaml.org,2002:value':
                key_node.tag = 'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1
        if merge:
            node.value = merge + node.value

    def construct_mapping(self, node, deep=False):
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        return super().construct_mapping(node, deep=deep)

    def construct_yaml_null(self, node):
        self.construct_scalar(node)
        return None

    bool_values = {
        'yes':      True,
        'no':       False,
        'true':     True,
        'false':    False,
        'on':       True,
        'off':      False,
    }

    def construct_yaml_bool(self, node):
        value = self.construct_scalar(node)
        return self.bool_values[value.lower()]

    def construct_yaml_int(self, node):
        value = self.construct_scalar(node)
        value = value.replace('_', '')
        sign = +1
        if value[0] == '-':
            sign = -1
        if value[0] in '+-':
            value = value[1:]
        if value == '0':
            return 0
        elif value.startswith('0b'):
            return sign*int(value[2:], 2)
        elif value.startswith('0x'):
            return sign*int(value[2:], 16)
        elif value[0] == '0':
            return sign*int(value, 8)
        elif ':' in value:
            digits = [int(part) for part in value.split(':')]
            digits.reverse()
            base = 1
            value = 0
            for digit in digits:
                value += digit*base
                base *= 60
            return sign*value
        else:
            return sign*int(value)

    inf_value = 1e300
    while inf_value != inf_value*inf_value:
        inf_value *= inf_value
    nan_value = -inf_value/inf_value   # Trying to make a quiet NaN (like C99).

    def construct_yaml_float(self, node):
        value = self.construct_scalar(node)
        value = value.replace('_', '').lower()
        sign = +1
        if value[0] == '-':
            sign = -1
        if value[0] in '+-':
            value = value[1:]
        if value == '.inf':
            return sign*self.inf_value
        elif value == '.nan':
            return self.nan_value
        elif ':' in value:
            digits = [float(part) for part in value.split(':')]
            digits.reverse()
            base = 1
            value = 0.0
            for digit in digits:
                value += digit*base
                base *= 60
            return sign*value
        else:
            return sign*float(value)

    def construct_yaml_binary(self, node):
        try:
            value = self.construct_scalar(node).encode('ascii')
        except UnicodeEncodeError as exc:
            raise ConstructorError(None, None,
                    "failed to convert base64 data into ascii: %s" % exc,
                    node.start_mark)
        try:
            if hasattr(base64, 'decodebytes'):
                return base64.decodebytes(value)
            else:
                return base64.decodestring(value)
        except binascii.Error as exc:
            raise ConstructorError(None, None,
                    "failed to decode base64 data: %s" % exc, node.start_mark)

    timestamp_regexp = re.compile(
            r'''^(?P<year>[0-9][0-9][0-9][0-9])
                -(?P<month>[0-9][0-9]?)
                -(?P<day>[0-9][0-9]?)
                (?:(?:[Tt]|[ \t]+)
                (?P<hour>[0-9][0-9]?)
                :(?P<minute>[0-9][0-9])
                :(?P<second>[0-9][0-9])
                (?:\.(?P<fraction>[0-9]*))?
                (?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)
                (?::(?P<tz_minute>[0-9][0-9]))?))?)?$''', re.X)

    def construct_yaml_timestamp(self, node):
        value = self.construct_scalar(node)
        match = self.timestamp_regexp.match(node.value)
        values = match.groupdict()
        year = int(values['year'])
        month = int(values['month'])
        day = int(values['day'])
        if not values['hour']:
            return datetime.date(year, month, day)
        hour = int(values['hour'])
        minute = int(values['minute'])
        second = int(values['second'])
        fraction = 0
        tzinfo = None
        if values['fraction']:
            fraction = values['fraction'][:6]
            while len(fraction) < 6:
                fraction += '0'
            fraction = int(fraction)
        if values['tz_sign']:
            tz_hour = int(values['tz_hour'])
            tz_minute = int(values['tz_minute'] or 0)
            delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
            if values['tz_sign'] == '-':
                delta = -delta
            tzinfo = datetime.timezone(delta)
        elif values['tz']:
            tzinfo = datetime.timezone.utc
        return datetime.datetime(year, month, day, hour, minute, second, fraction,
                                 tzinfo=tzinfo)

    def construct_yaml_omap(self, node):
        # Note: we do not check for duplicate keys, because it's too
        # CPU-expensive.
        omap = []
        yield omap
        if not isinstance(node, SequenceNode):
            raise ConstructorError("while constructing an ordered map", node.start_mark,
                    "expected a sequence, but found %s" % node.id, node.start_mark)
        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError("while constructing an ordered map", node.start_mark,
                        "expected a mapping of length 1, but found %s" % subnode.id,
                        subnode.start_mark)
            if len(subnode.value) != 1:
                raise ConstructorError("while constructing an ordered map", node.start_mark,
                        "expected a single mapping item, but found %d items" % len(subnode.value),
                        subnode.start_mark)
            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            value = self.construct_object(value_node)
            omap.append((key, value))

    def construct_yaml_pairs(self, node):
        # Note: the same code as `construct_yaml_omap`.
        pairs = []
        yield pairs
        if not isinstance(node, SequenceNode):
            raise ConstructorError("while constructing pairs", node.start_mark,
                    "expected a sequence, but found %s" % node.id, node.start_mark)
        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError("while constructing pairs", node.start_mark,
                        "expected a mapping of length 1, but found %s" % subnode.id,
                        subnode.start_mark)
            if len(subnode.value) != 1:
                raise ConstructorError("while constructing pairs", node.start_mark,
                        "expected a single mapping item, but found %d items" % len(subnode.value),
                        subnode.start_mark)
            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            value = self.construct_object(value_node)
            pairs.append((key, value))

    def construct_yaml_set(self, node):
        data = set()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_yaml_str(self, node):
        return self.construct_scalar(node)

    def construct_yaml_seq(self, node):
        data = []
        yield data
        data.extend(self.construct_sequence(node))

    def construct_yaml_map(self, node):
        data = {}
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_yaml_object(self, node, cls):
        data = cls.__new__(cls)
        yield data
        if hasattr(data, '__setstate__'):
            state = self.construct_mapping(node, deep=True)
            data.__setstate__(state)
        else:
            state = self.construct_mapping(node)
            data.__dict__.update(state)

    def construct_undefined(self, node):
        raise ConstructorError(None, None,
                "could not determine a constructor for the tag %r" % node.tag,
                node.start_mark)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:null',
        SafeConstructor.construct_yaml_null)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:bool',
        SafeConstructor.construct_yaml_bool)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:int',
        SafeConstructor.construct_yaml_int)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:float',
        SafeConstructor.construct_yaml_float)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:binary',
        SafeConstructor.construct_yaml_binary)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:timestamp',
        SafeConstructor.construct_yaml_timestamp)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:omap',
        SafeConstructor.construct_yaml_omap)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:pairs',
        SafeConstructor.construct_yaml_pairs)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:set',
        SafeConstructor.construct_yaml_set)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:str',
        SafeConstructor.construct_yaml_str)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:seq',
        SafeConstructor.construct_yaml_seq)

SafeConstructor.add_constructor(
        'tag:yaml.org,2002:map',
        SafeConstructor.construct_yaml_map)

SafeConstructor.add_constructor(None,
        SafeConstructor.construct_undefined)

class FullConstructor(SafeConstructor):
    # 'extend' is blacklisted because it is used by
    # construct_python_object_apply to add `listitems` to a newly generate
    # python instance
    def get_state_keys_blacklist(self):
        return ['^extend$', '^__.*__$']

    def get_state_keys_blacklist_regexp(self):
        if not hasattr(self, 'state_keys_blacklist_regexp'):
            self.state_keys_blacklist_regexp = re.compile('(' + '|'.join(self.get_state_keys_blacklist()) + ')')
        return self.state_keys_blacklist_regexp

    def construct_python_str(self, node):
        return self.construct_scalar(node)

    def construct_python_unicode(self, node):
        return self.construct_scalar(node)

    def construct_python_bytes(self, node):
        try:
            value = self.construct_scalar(node).encode('ascii')
        except UnicodeEncodeError as exc:
            raise ConstructorError(None, None,
                    "failed to convert base64 data into ascii: %s" % exc,
                    node.start_mark)
        try:
            if hasattr(base64, 'decodebytes'):
                return base64.decodebytes(value)
            else:
                return base64.decodestring(value)
        except binascii.Error as exc:
            raise ConstructorError(None, None,
                    "failed to decode base64 data: %s" % exc, node.start_mark)

    def construct_python_long(self, node):
        return self.construct_yaml_int(node)

    def construct_python_complex(self, node):
       return complex(self.construct_scalar(node))

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    def find_python_module(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError("while constructing a Python module", mark,
                    "expected non-empty name appended to the tag", mark)
        if unsafe:
            try:
                __import__(name)
            except ImportError as exc:
                raise ConstructorError("while constructing a Python module", mark,
                        "cannot find module %r (%s)" % (name, exc), mark)
        if name not in sys.modules:
            raise ConstructorError("while constructing a Python module", mark,
                    "module %r is not imported" % name, mark)
        return sys.modules[name]

    def find_python_name(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError("while constructing a Python object", mark,
                    "expected non-empty name appended to the tag", mark)
        if '.' in name:
            module_name, object_name = name.rsplit('.', 1)
        else:
            module_name = 'builtins'
            object_name = name
        if unsafe:
            try:
                __import__(module_name)
            except ImportError as exc:
                raise ConstructorError("while constructing a Python object", mark,
                        "cannot find module %r (%s)" % (module_name, exc), mark)
        if module_name not in sys.modules:
            raise ConstructorError("while constructing a Python object", mark,
                    "module %r is not imported" % module_name, mark)
        module = sys.modules[module_name]
        if not hasattr(module, object_name):
            raise ConstructorError("while constructing a Python object", mark,
                    "cannot find %r in the module %r"
                    % (object_name, module.__name__), mark)
        return getattr(module, object_name)

    def construct_python_name(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError("while constructing a Python name", node.start_mark,
                    "expected the empty value, but found %r" % value, node.start_mark)
        return self.find_python_name(suffix, node.start_mark)

    def construct_python_module(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError("while constructing a Python module", node.start_mark,
                    "expected the empty value, but found %r" % value, node.start_mark)
        return self.find_python_module(suffix, node.start_mark)

    def make_python_instance(self, suffix, node,
            args=None, kwds=None, newobj=False, unsafe=False):
        if not args:
            args = []
        if not kwds:
            kwds = {}
        cls = self.find_python_name(suffix, node.start_mark)
        if not (unsafe or isinstance(cls, type)):
            raise ConstructorError("while constructing a Python instance", node.start_mark,
                    "expected a class, but found %r" % type(cls),
                    node.start_mark)
        if newobj and isinstance(cls, type):
            return cls.__new__(cls, *args, **kwds)
        else:
            return cls(*args, **kwds)

    def set_python_instance_state(self, instance, state, unsafe=False):
        if hasattr(instance, '__setstate__'):
            instance.__setstate__(state)
        else:
            slotstate = {}
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if hasattr(instance, '__dict__'):
                if not unsafe and state:
                    for key in state.keys():
                        self.check_state_key(key)
                instance.__dict__.update(state)
            elif state:
                slotstate.update(state)
            for key, value in slotstate.items():
                if not unsafe:
                    self.check_state_key(key)
                setattr(instance, key, value)

    def construct_python_object(self, suffix, node):
        # Format:
        #   !!python/object:module.name { ... state ... }
        instance = self.make_python_instance(suffix, node, newobj=True)
        yield instance
        deep = hasattr(instance, '__setstate__')
        state = self.construct_mapping(node, deep=deep)
        self.set_python_instance_state(instance, state)

    def construct_python_object_apply(self, suffix, node, newobj=False):
        # Format:
        #   !!python/object/apply       # (or !!python/object/new)
        #   args: [ ... arguments ... ]
        #   kwds: { ... keywords ... }
        #   state: ... state ...
        #   listitems: [ ... listitems ... ]
        #   dictitems: { ... dictitems ... }
        # or short format:
        #   !!python/object/apply [ ... arguments ... ]
        # The difference between !!python/object/apply and !!python/object/new
        # is how an object is created, check make_python_instance for details.
        if isinstance(node, SequenceNode):
            args = self.construct_sequence(node, deep=True)
            kwds = {}
            state = {}
            listitems = []
            dictitems = {}
        else:
            value = self.construct_mapping(node, deep=True)
            args = value.get('args', [])
            kwds = value.get('kwds', {})
            state = value.get('state', {})
            listitems = value.get('listitems', [])
            dictitems = value.get('dictitems', {})
        instance = self.make_python_instance(suffix, node, args, kwds, newobj)
        if state:
            self.set_python_instance_state(instance, state)
        if listitems:
            instance.extend(listitems)
        if dictitems:
            for key in dictitems:
                instance[key] = dictitems[key]
        return instance

    def construct_python_object_new(self, suffix, node):
        return self.construct_python_object_apply(suffix, node, newobj=True)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/none',
    FullConstructor.construct_yaml_null)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/bool',
    FullConstructor.construct_yaml_bool)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/str',
    FullConstructor.construct_python_str)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/unicode',
    FullConstructor.construct_python_unicode)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/bytes',
    FullConstructor.construct_python_bytes)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/int',
    FullConstructor.construct_yaml_int)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/long',
    FullConstructor.construct_python_long)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/float',
    FullConstructor.construct_yaml_float)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/complex',
    FullConstructor.construct_python_complex)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/list',
    FullConstructor.construct_yaml_seq)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    FullConstructor.construct_python_tuple)

FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/dict',
    FullConstructor.construct_yaml_map)

FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/name:',
    FullConstructor.construct_python_name)

class UnsafeConstructor(FullConstructor):

    def find_python_module(self, name, mark):
        return super(UnsafeConstructor, self).find_python_module(name, mark, unsafe=True)

    def find_python_name(self, name, mark):
        return super(UnsafeConstructor, self).find_python_name(name, mark, unsafe=True)

    def make_python_instance(self, suffix, node, args=None, kwds=None, newobj=False):
        return super(UnsafeConstructor, self).make_python_instance(
            suffix, node, args, kwds, newobj, unsafe=True)

    def set_python_instance_state(self, instance, state):
        return super(UnsafeConstructor, self).set_python_instance_state(
            instance, state, unsafe=True)

UnsafeConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/module:',
    UnsafeConstructor.construct_python_module)

UnsafeConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object:',
    UnsafeConstructor.construct_python_object)

UnsafeConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object/new:',
    UnsafeConstructor.construct_python_object_new)

UnsafeConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object/apply:',
    UnsafeConstructor.construct_python_object_apply)

# Constructor is same as UnsafeConstructor. Need to leave this in place in case
# people have extended it directly.
class Constructor(UnsafeConstructor):
    pass
<|endoftext|>
__all__ = ['BaseResolver', 'Resolver']

from .error import *
from .nodes import *

import re

class ResolverError(YAMLError):
    pass

class BaseResolver:

    DEFAULT_SCALAR_TAG = 'tag:yaml.org,2002:str'
    DEFAULT_SEQUENCE_TAG = 'tag:yaml.org,2002:seq'
    DEFAULT_MAPPING_TAG = 'tag:yaml.org,2002:map'

    yaml_implicit_resolvers = {}
    yaml_path_resolvers = {}

    def __init__(self):
        self.resolver_exact_paths = []
        self.resolver_prefix_paths = []

    @classmethod
    def add_implicit_resolver(cls, tag, regexp, first):
        if not 'yaml_implicit_resolvers' in cls.__dict__:
            implicit_resolvers = {}
            for key in cls.yaml_implicit_resolvers:
                implicit_resolvers[key] = cls.yaml_implicit_resolvers[key][:]
            cls.yaml_implicit_resolvers = implicit_resolvers
        if first is None:
            first = [None]
        for ch in first:
            cls.yaml_implicit_resolvers.setdefault(ch, []).append((tag, regexp))

    @classmethod
    def add_path_resolver(cls, tag, path, kind=None):
        # Note: `add_path_resolver` is experimental.  The API could be changed.
        # `new_path` is a pattern that is matched against the path from the
        # root to the node that is being considered.  `node_path` elements are
        # tuples `(node_check, index_check)`.  `node_check` is a node class:
        # `ScalarNode`, `SequenceNode`, `MappingNode` or `None`.  `None`
        # matches any kind of a node.  `index_check` could be `None`, a boolean
        # value, a string value, or a number.  `None` and `False` match against
        # any _value_ of sequence and mapping nodes.  `True` matches against
        # any _key_ of a mapping node.  A string `index_check` matches against
        # a mapping value that corresponds to a scalar key which content is
        # equal to the `index_check` value.  An integer `index_check` matches
        # against a sequence value with the index equal to `index_check`.
        if not 'yaml_path_resolvers' in cls.__dict__:
            cls.yaml_path_resolvers = cls.yaml_path_resolvers.copy()
        new_path = []
        for element in path:
            if isinstance(element, (list, tuple)):
                if len(element) == 2:
                    node_check, index_check = element
                elif len(element) == 1:
                    node_check = element[0]
                    index_check = True
                else:
                    raise ResolverError("Invalid path element: %s" % element)
            else:
                node_check = None
                index_check = element
            if node_check is str:
                node_check = ScalarNode
            elif node_check is list:
                node_check = SequenceNode
            elif node_check is dict:
                node_check = MappingNode
            elif node_check not in [ScalarNode, SequenceNode, MappingNode]  \
                    and not isinstance(node_check, str) \
                    and node_check is not None:
                raise ResolverError("Invalid node checker: %s" % node_check)
            if not isinstance(index_check, (str, int))  \
                    and index_check is not None:
                raise ResolverError("Invalid index checker: %s" % index_check)
            new_path.append((node_check, index_check))
        if kind is str:
            kind = ScalarNode
        elif kind is list:
            kind = SequenceNode
        elif kind is dict:
            kind = MappingNode
        elif kind not in [ScalarNode, SequenceNode, MappingNode]    \
                and kind is not None:
            raise ResolverError("Invalid node kind: %s" % kind)
        cls.yaml_path_resolvers[tuple(new_path), kind] = tag

    def descend_resolver(self, current_node, current_index):
        if not self.yaml_path_resolvers:
            return
        exact_paths = {}
        prefix_paths = []
        if current_node:
            depth = len(self.resolver_prefix_paths)
            for path, kind in self.resolver_prefix_paths[-1]:
                if self.check_resolver_prefix(depth, path, kind,
                        current_node, current_index):
                    if len(path) > depth:
                        prefix_paths.append((path, kind))
                    else:
                        exact_paths[kind] = self.yaml_path_resolvers[path, kind]
        else:
            for path, kind in self.yaml_path_resolvers:
                if not path:
                    exact_paths[kind] = self.yaml_path_resolvers[path, kind]
                else:
                    prefix_paths.append((path, kind))
        self.resolver_exact_paths.append(exact_paths)
        self.resolver_prefix_paths.append(prefix_paths)

    def ascend_resolver(self):
        if not self.yaml_path_resolvers:
            return
        self.resolver_exact_paths.pop()
        self.resolver_prefix_paths.pop()

    def check_resolver_prefix(self, depth, path, kind,
            current_node, current_index):
        node_check, index_check = path[depth-1]
        if isinstance(node_check, str):
            if current_node.tag != node_check:
                return
        elif node_check is not None:
            if not isinstance(current_node, node_check):
                return
        if index_check is True and current_index is not None:
            return
        if (index_check is False or index_check is None)    \
                and current_index is None:
            return
        if isinstance(index_check, str):
            if not (isinstance(current_index, ScalarNode)
                    and index_check == current_index.value):
                return
        elif isinstance(index_check, int) and not isinstance(index_check, bool):
            if index_check != current_index:
                return
        return True

    def resolve(self, kind, value, implicit):
        if kind is ScalarNode and implicit[0]:
            if value == '':
                resolvers = self.yaml_implicit_resolvers.get('', [])
            else:
                resolvers = self.yaml_implicit_resolvers.get(value[0], [])
            wildcard_resolvers = self.yaml_implicit_resolvers.get(None, [])
            for tag, regexp in resolvers + wildcard_resolvers:
                if regexp.match(value):
                    return tag
            implicit = implicit[1]
        if self.yaml_path_resolvers:
            exact_paths = self.resolver_exact_paths[-1]
            if kind in exact_paths:
                return exact_paths[kind]
            if None in exact_paths:
                return exact_paths[None]
        if kind is ScalarNode:
            return self.DEFAULT_SCALAR_TAG
        elif kind is SequenceNode:
            return self.DEFAULT_SEQUENCE_TAG
        elif kind is MappingNode:
            return self.DEFAULT_MAPPING_TAG

class Resolver(BaseResolver):
    pass

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:bool',
        re.compile(r'''^(?:yes|Yes|YES|no|No|NO
                    |true|True|TRUE|false|False|FALSE
                    |on|On|ON|off|Off|OFF)$''', re.X),
        list('yYnNtTfFoO'))

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                    |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$''', re.X),
        list('-+0123456789.'))

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:int',
        re.compile(r'''^(?:[-+]?0b[0-1_]+
                    |[-+]?0[0-7_]+
                    |[-+]?(?:0|[1-9][0-9_]*)
                    |[-+]?0x[0-9a-fA-F_]+
                    |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$''', re.X),
        list('-+0123456789'))

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:merge',
        re.compile(r'^(?:<<)$'),
        ['<'])

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:null',
        re.compile(r'''^(?: ~
                    |null|Null|NULL
                    | )$''', re.X),
        ['~', 'n', 'N', ''])

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:timestamp',
        re.compile(r'''^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
                    |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
                     (?:[Tt]|[ \t]+)[0-9][0-9]?
                     :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
                     (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$''', re.X),
        list('0123456789'))

Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:value',
        re.compile(r'^(?:=)$'),
        ['='])

# The following resolver is only for documentation purposes. It cannot work
# because plain scalars cannot start with '!', '&', or '*'.
Resolver.add_implicit_resolver(
        'tag:yaml.org,2002:yaml',
        re.compile(r'^(?:!|&|\*)$'),
        list('!&*'))

<|endoftext|>
__all__ = ['BaseLoader', 'FullLoader', 'SafeLoader', 'Loader', 'UnsafeLoader']

from .reader import *
from .scanner import *
from .parser import *
from .composer import *
from .constructor import *
from .resolver import *

class BaseLoader(Reader, Scanner, Parser, Composer, BaseConstructor, BaseResolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        BaseConstructor.__init__(self)
        BaseResolver.__init__(self)

class FullLoader(Reader, Scanner, Parser, Composer, FullConstructor, Resolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        FullConstructor.__init__(self)
        Resolver.__init__(self)

class SafeLoader(Reader, Scanner, Parser, Composer, SafeConstructor, Resolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)

class Loader(Reader, Scanner, Parser, Composer, Constructor, Resolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        Constructor.__init__(self)
        Resolver.__init__(self)

# UnsafeLoader is the same as Loader (which is and was always unsafe on
# untrusted input). Use of either Loader or UnsafeLoader should be rare, since
# FullLoad should be able to load almost all YAML safely. Loader is left intact
# to ensure backwards compatibility.
class UnsafeLoader(Reader, Scanner, Parser, Composer, Constructor, Resolver):

    def __init__(self, stream):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        Constructor.__init__(self)
        Resolver.__init__(self)
<|endoftext|>
# Scanner produces tokens of the following types:
# STREAM-START
# STREAM-END
# DIRECTIVE(name, value)
# DOCUMENT-START
# DOCUMENT-END
# BLOCK-SEQUENCE-START
# BLOCK-MAPPING-START
# BLOCK-END
# FLOW-SEQUENCE-START
# FLOW-MAPPING-START
# FLOW-SEQUENCE-END
# FLOW-MAPPING-END
# BLOCK-ENTRY
# FLOW-ENTRY
# KEY
# VALUE
# ALIAS(value)
# ANCHOR(value)
# TAG(value)
# SCALAR(value, plain, style)
#
# Read comments in the Scanner code for more details.
#

__all__ = ['Scanner', 'ScannerError']

from .error import MarkedYAMLError
from .tokens import *

class ScannerError(MarkedYAMLError):
    pass

class SimpleKey:
    # See below simple keys treatment.

    def __init__(self, token_number, required, index, line, column, mark):
        self.token_number = token_number
        self.required = required
        self.index = index
        self.line = line
        self.column = column
        self.mark = mark

class Scanner:

    def __init__(self):
        """Initialize the scanner."""
        # It is assumed that Scanner and Reader will have a common descendant.
        # Reader do the dirty work of checking for BOM and converting the
        # input data to Unicode. It also adds NUL to the end.
        #
        # Reader supports the following methods
        #   self.peek(i=0)       # peek the next i-th character
        #   self.prefix(l=1)     # peek the next l characters
        #   self.forward(l=1)    # read the next l characters and move the pointer.

        # Had we reached the end of the stream?
        self.done = False

        # The number of unclosed '{' and '['. `flow_level == 0` means block
        # context.
        self.flow_level = 0

        # List of processed tokens that are not yet emitted.
        self.tokens = []

        # Add the STREAM-START token.
        self.fetch_stream_start()

        # Number of tokens that were emitted through the `get_token` method.
        self.tokens_taken = 0

        # The current indentation level.
        self.indent = -1

        # Past indentation levels.
        self.indents = []

        # Variables related to simple keys treatment.

        # A simple key is a key that is not denoted by the '?' indicator.
        # Example of simple keys:
        #   ---
        #   block simple key: value
        #   ? not a simple key:
        #   : { flow simple key: value }
        # We emit the KEY token before all keys, so when we find a potential
        # simple key, we try to locate the corresponding ':' indicator.
        # Simple keys should be limited to a single line and 1024 characters.

        # Can a simple key start at the current position? A simple key may
        # start:
        # - at the beginning of the line, not counting indentation spaces
        #       (in block context),
        # - after '{', '[', ',' (in the flow context),
        # - after '?', ':', '-' (in the block context).
        # In the block context, this flag also signifies if a block collection
        # may start at the current position.
        self.allow_simple_key = True

        # Keep track of possible simple keys. This is a dictionary. The key
        # is `flow_level`; there can be no more that one possible simple key
        # for each level. The value is a SimpleKey record:
        #   (token_number, required, index, line, column, mark)
        # A simple key may start with ALIAS, ANCHOR, TAG, SCALAR(flow),
        # '[', or '{' tokens.
        self.possible_simple_keys = {}

    # Public methods.

    def check_token(self, *choices):
        # Check if the next token is one of the given types.
        while self.need_more_tokens():
            self.fetch_more_tokens()
        if self.tokens:
            if not choices:
                return True
            for choice in choices:
                if isinstance(self.tokens[0], choice):
                    return True
        return False

    def peek_token(self):
        # Return the next token, but do not delete if from the queue.
        # Return None if no more tokens.
        while self.need_more_tokens():
            self.fetch_more_tokens()
        if self.tokens:
            return self.tokens[0]
        else:
            return None

    def get_token(self):
        # Return the next token.
        while self.need_more_tokens():
            self.fetch_more_tokens()
        if self.tokens:
            self.tokens_taken += 1
            return self.tokens.pop(0)

    # Private methods.

    def need_more_tokens(self):
        if self.done:
            return False
        if not self.tokens:
            return True
        # The current token may be a potential simple key, so we
        # need to look further.
        self.stale_possible_simple_keys()
        if self.next_possible_simple_key() == self.tokens_taken:
            return True

    def fetch_more_tokens(self):

        # Eat whitespaces and comments until we reach the next token.
        self.scan_to_next_token()

        # Remove obsolete possible simple keys.
        self.stale_possible_simple_keys()

        # Compare the current indentation and column. It may add some tokens
        # and decrease the current indentation level.
        self.unwind_indent(self.column)

        # Peek the next character.
        ch = self.peek()

        # Is it the end of stream?
        if ch == '\0':
            return self.fetch_stream_end()

        # Is it a directive?
        if ch == '%' and self.check_directive():
            return self.fetch_directive()

        # Is it the document start?
        if ch == '-' and self.check_document_start():
            return self.fetch_document_start()

        # Is it the document end?
        if ch == '.' and self.check_document_end():
            return self.fetch_document_end()

