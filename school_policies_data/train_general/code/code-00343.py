This caching system is critical for performance as it eliminates the need to re-run
expensive autotuning operations when the same kernels are compiled multiple times.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import os.path
import re
import threading
import weakref
from typing import Any
from typing_extensions import override

import torch
from torch._guards import CompileContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from torch.utils._triton import has_triton

from ..cache_key import AUTOTUNE_CACHE_KEY_STRATEGY
from ..remote_cache import (
    create_cache,
    JsonDataTy,
    LocalAutotuneCache,
    LocalCacheBackend,
    RemoteCache,
    RemoteCacheJsonSerde,
)
from .hints import InductorMeta
from .triton_compat import Config, HAS_WARP_SPEC


log = logging.getLogger(__name__)


_InductorMetaTy = InductorMeta


def inductor_meta_from_config() -> _InductorMetaTy:
    from torch._inductor import config

    backend_hash = None
    if has_triton():
        try:
            backend_hash = torch.utils._triton.triton_hash_with_backend()
        except RuntimeError:
            # This can get the error:
            #   RuntimeError: 0 active drivers ([]). There should only be one.
            pass

    is_hip = None
    if torch.version.hip is not None:
        is_hip = True

    return {
        "autotune_local_cache": config.autotune_local_cache,
        "autotune_remote_cache": config.autotune_remote_cache,
        "backend_hash": backend_hash,
        "bundled_autotune_remote_cache": config.bundled_autotune_remote_cache,
        "coordinate_descent_tuning": config.coordinate_descent_tuning,
        "is_fbcode": config.is_fbcode(),
        "is_hip": is_hip,
    }


@CacheArtifactFactory.register
class AutotuneCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        autotune_cache = LocalCacheBackend()
        key = os.path.join(cache_dir(), self.key)
        autotune_cache._put(key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "autotune"

    @override
    @staticmethod
    def encode(content: JsonDataTy) -> bytes:
        if isinstance(content, bytes):
            raise AssertionError("content must not be bytes before encoding")
        serde = RemoteCacheJsonSerde()
        content_bytes = serde.encode(content)
        if not isinstance(content_bytes, bytes):
            raise AssertionError(
                f"Expected bytes after encoding, got {type(content_bytes)}"
            )
        return content_bytes


@dataclasses.dataclass
class AutotuneCache:
    """Coordinates local and remote autotune cache lookups for one kernel."""

    configs_hash: str
    local_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    remote_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    artifact_recorder: CacheArtifactRecorder | None = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> AutotuneCache | None:
        cache = AutotuneCache(configs_hash)
        key = AutotuneCache._prepare_key(filename)
        local_cache_key = AutotuneCache._make_local_cache_key(
            os.path.dirname(filename), key
        )
        cache.artifact_recorder = CacheArtifactRecorder(
            AutotuneCacheArtifact.type(),
            AutotuneCache._artifact_key_from_local_cache_key(local_cache_key),
        )

        cache._setup_local_cache(inductor_meta, local_cache_key)
        cache._setup_remote_autotune_cache(inductor_meta, key)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    @staticmethod
    def _prepare_key(filename: str) -> str:
        from torch.compiler import config as cconfig

        # base of filename is already sha256 hash the source contents
        key = f"{os.path.basename(filename)}:{cconfig.cache_key_tag}"
        return AUTOTUNE_CACHE_KEY_STRATEGY.key(key)

    @staticmethod
    def _make_local_cache_key(dirname: str, cache_key: str) -> str:
        """
        [Note: torch_key in autotune cache key]
        Include torch_key() in the cache key so that different versions
        of torch result in cache invalidation. This is important in case
        of changes to the best_config format or other code changes that
        are not backward compatible w.r.t. the cache.
        """
        from ..codecache import torch_key

        updated_cache_key = AUTOTUNE_CACHE_KEY_STRATEGY.key(cache_key, torch_key())
        return os.path.join(dirname, f"{updated_cache_key}.best_config")

    @staticmethod
    def _artifact_key_from_local_cache_key(local_cache_key: str) -> str:
        return os.path.join(*local_cache_key.split(os.sep)[-2:])

    def _record_artifact(self, data: JsonDataTy) -> None:
        # Older pickled AutotuneCache instances may not have this field.
        if recorder := getattr(self, "artifact_recorder", None):
            recorder.record(data)

    # Read the best config options from the most local cache and return it.
    def _read(self) -> dict[str, JsonDataTy] | None:
        if local_cache := self.local_cache:
            cache, key = local_cache
            AutotuneCacheBundler.sync()
            best_config = cache.get(key)
            if best_config is not None:
                if not isinstance(best_config, dict):
                    raise AssertionError(
                        f"Expected dict for best_config, got {type(best_config)}"
                    )
                # Imagine we have a new model that reuses some existing kernels that
                # have already been compiled. If we didn't put() here on cache hit,
                # then the new model would only bundle newly compiled kernels, not
                # existing kernels that were already compiled and cached.
                AutotuneCacheBundler.put(key, best_config)
                self._record_artifact(best_config)
                if best_config:
                    return best_config

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    self._record_artifact(best_config)
                    return best_config

        return None

    # Read the best config options from the most local cache and figure out
    # which `configs` represents that option.
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: list[Config]
    ) -> Config | None:
        if best := self._read():
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        if not inductor_meta.get("autotune_local_cache", True):
            return

        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        if local_cache is None:
            return
        self.local_cache = (local_cache, cache_key)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        if (backend_hash := inductor_meta.get("backend_hash", None)) is None:
            log.debug(
                "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
            )
            return
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )

        from ..codecache import torch_key

        is_fbcode = bool(inductor_meta.get("is_fbcode", False))

        salt = "autotune-best-config-v2"
        # re: torch_key - see [Note: torch_key in autotune cache key]
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(
            torch_key().hex(), backend_hash, self.configs_hash, salt
        )

        remote_cache = create_cache(
            key,
            is_fbcode,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
        )
        if not remote_cache:
            return

        # Save the args passed to create_cache
        # in case AutotuneCache needs to be pickled
        self.remote_cache_full_key = key
        self.is_fbcode = is_fbcode
        self.remote_cache = (remote_cache, cache_key)

    # The AutotuneCache may be serialized/deserialized if we're using
    # AsyncCompile worker processes to run triton compilation.
    # This is because AutotuneCache instances are created on the worker
    # process, but we need to run AutotuneCache.save on the parent process
    # when actually doing autotuning.
    def __getstate__(self) -> dict[str, Any]:
        # The remote cache handles themselves may not be serializable
        # So clear it and reconstruct it on setstate
        remote_cache = getattr(self, "remote_cache", None)
        return {
            **self.__dict__,
            # Save the cache_key portion
            "remote_cache": remote_cache and remote_cache[1],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Reconstruct the remote cache on the parent class
        self.__dict__.update(state)
        if self.remote_cache is not None:
            if not isinstance(self.remote_cache, str):
                raise AssertionError(
                    f"Expected str for remote_cache after deserialization, got {type(self.remote_cache)}"
                )
            if not hasattr(self, "remote_cache_full_key"):
                raise AssertionError(
                    "Missing remote_cache_full_key attribute after deserialization"
                )
            if not hasattr(self, "is_fbcode"):
                raise AssertionError(
                    "Missing is_fbcode attribute after deserialization"
                )
            cache_key = self.remote_cache
            remote_cache = create_cache(
                self.remote_cache_full_key,
                self.is_fbcode,
                "FbRemoteAutotuneCache",
                "RemoteAutotuneCache",
            )
            if remote_cache is not None:
                self.remote_cache = (remote_cache, cache_key)
            else:
                log.warning("Warning, failed to recreate remote cache after pickling")
                self.remote_cache = None

    # Save the config in the caches
    def save(
        self,
        config: Config,
        time_taken_ns: int,
        found_by_coordesc: bool = False,
        triton_cache_hash: str | None = None,
    ) -> None:
        data: dict[str, JsonDataTy] = {
            # pyrefly: ignore [missing-attribute]
            **config.kwargs,
            # pyrefly: ignore [missing-attribute]
            "num_warps": config.num_warps,
            # pyrefly: ignore [missing-attribute]
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
            "triton_cache_hash": triton_cache_hash,
        }
        # Save extra_options if present on the config. This allows third-party
        # backends to store custom tuned options alongside the standard config.
        if extra_options := getattr(config, "extra_options", None):
            data["extra_options"] = extra_options
        if HAS_WARP_SPEC:
            data.update(
                {
                    "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                    "num_buffers_warp_spec": getattr(
                        config, "num_buffers_warp_spec", 0
                    ),
                }
            )

        self._record_artifact(data)

        if local_cache := self.local_cache:
            cache, key = local_cache
            # pyrefly: ignore [bad-argument-type]
            cache.put(key, data)
            # pyrefly: ignore [bad-argument-type]
            AutotuneCacheBundler.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            # pyrefly: ignore [bad-argument-type]
            cache.put(key, data)


class _AutotuneCacheBundlerImpl:
    """
    Caches a set of local autotune entries together in a single cache.
    """

    _key: str
    _cache: RemoteCache[JsonDataTy]

    # All known entries from local autotune cache writes.
    _entries: dict[str, JsonDataTy]

    def end_compile(self) -> None:
        # TODO: Do we need to compute time_taken_ms and encode that somehow?
        if self._entries:
            self._cache.put(self._key, self._entries)

    def put(self, basename: str, data: JsonDataTy) -> None:
        # Do we need to worry about duplicates? We only have a single local fs
        # entry - so probably not.
        self._entries[basename] = data

    def __init__(self, key: str, cache: RemoteCache[JsonDataTy]) -> None:
        self._key = key
        self._cache = cache
        self._entries = {}

    def sync(self) -> None:
        # We don't currently use this - but we could async load starting at
        # `begin_compile` and wait for the load to be finished here.
        pass

    @classmethod
    def _should_use_bundled_autotune_remote_cache(
        cls, inductor_meta: _InductorMetaTy
    ) -> bool:
        # The bundled autotune cache is only available if you've also got local
        # caching enabled (because we feed the bundled data to the local cache).
        if not inductor_meta.get("autotune_local_cache", True):
            return False

        # Check if we're enabled via config
        if (
            bundled_autotune_remote_cache := inductor_meta.get(
                "bundled_autotune_remote_cache"
            )
        ) is not None:
            return bool(bundled_autotune_remote_cache)

        if not cls._get_is_fbcode(inductor_meta):
            return False
        if torch._utils_internal.is_fb_unit_test():
            return False
        if inductor_meta.get("is_hip"):
            return False

        try:
            from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
        except ModuleNotFoundError:
            return False

        jk = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:bundled_autotune_remote_cache_version"
        )
        return REMOTE_CACHE_VERSION >= jk

    def _load_cache(self) -> bool:
        from torch._inductor import codecache

        # The single key is defined on construction of the cache.
        entries = self._cache.get(self._key)
        if entries is None or not isinstance(entries, dict):
            # We couldn't load the cache - so mark _entries as non-None so we
            # store local cache values.
            return False

        # Go through the entries we got from the cache and save them locally.
        time_saved_ns = 0
        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        for basename, data in entries.items():
            # Reconstruct the final filename (see put())
            root, ext = _splitext_nodot(basename)
            _, _, filename = codecache.get_path(root, ext)
            if isinstance(data, dict) and (tsns := data.get("time_saved_ns")):
                time_saved_ns += int(tsns)  # type: ignore[arg-type]
            if local_cache is not None:
                local_cache.put(filename, data)

        codecache.add_ephemeral_timeout_increase_for_distributed(time_saved_ns)

        return True

    @staticmethod
    def _get_is_fbcode(inductor_meta: _InductorMetaTy) -> bool:
        return bool(inductor_meta.get("is_fbcode", False))

    @staticmethod
    def _get_backend_hash(inductor_meta: _InductorMetaTy) -> str:
        backend_hash = inductor_meta["backend_hash"]
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )
        return backend_hash


class AutotuneCacheBundler:
    """Manages one bundled autotune cache write scope for a compile context."""

    _context_bundlers: weakref.WeakKeyDictionary[
        CompileContext, AutotuneCacheBundler
    ] = weakref.WeakKeyDictionary()
    _context_bundlers_lock = threading.Lock()

    def __init__(self) -> None:
        self._bundler: _AutotuneCacheBundlerImpl | None = None

    @classmethod
    def _get_context_bundler(
        cls,
        ctx: CompileContext | None = None,
        *,
        create: bool,
    ) -> AutotuneCacheBundler | None:
        if ctx is None:
            return None

        with cls._context_bundlers_lock:
            bundler = cls._context_bundlers.get(ctx)
            if bundler is None and create:
                bundler = cls()
                cls._context_bundlers[ctx] = bundler
            if bundler is not None and not isinstance(bundler, cls):
                raise AssertionError(
                    f"Expected AutotuneCacheBundler or None, got {type(bundler)}"
                )
            return bundler

    @classmethod
    def has_active_compile(cls, compile_context: CompileContext | None) -> bool:
        context_bundler = cls._get_context_bundler(compile_context, create=False)
        return context_bundler is not None and context_bundler._bundler is not None

    # Call this before we start any autotune computation for an inductor python
    # file. On a cache hit it copies the individual results into the local
    # autotune caches.
    @classmethod
    def begin_compile(
        cls,
        inductor_meta: _InductorMetaTy,
        *,
        code: str | None = None,
        code_hash: str | None = None,
    ) -> None:
        if code is not None:
            if code_hash is not None:
                raise AssertionError("Cannot specify both code and code_hash")
            code_hash = _comment_stripped_hash(code)
        if code_hash is None:
            raise AssertionError("Either code or code_hash must be provided")

        if not _AutotuneCacheBundlerImpl._should_use_bundled_autotune_remote_cache(
            inductor_meta
        ):
            return

        context_bundler = cls._get_context_bundler(
            CompileContext.try_get(), create=True
        )
        if context_bundler is None:
            log.debug(
                "Skipping bundled autotune cache because compile_context is not set"
            )
            return
        if context_bundler._bundler is not None:
            raise AssertionError(
                "begin_compile called while a bundler is already active"
            )

        cache = create_cache(
            "bundled-autotune-v1",
            _AutotuneCacheBundlerImpl._get_is_fbcode(inductor_meta),
            "FbRemoteBundledAutotuneCache",
            "RemoteBundledAutotuneCache",
        )
        if not cache:
            return

        # We're starting a compilation phase. We have a cache key for the code
        # we're compiling. We'll get the individual autotune bundles later (via
        # self.put()). For now create the AutotuneCacheBundler and try to load
        # from the cache.

        salt = "bundled-autotune-best-configs-v1"
        backend_hash = _AutotuneCacheBundlerImpl._get_backend_hash(inductor_meta)
        # TODO: The autotune cache includes configs_hash in the key. The problem
        # is that the configs_hash includes info from the individual pointwise()
        # calls (size_hints, for example) which we can't know yet. I *think*
        # that info is basically present in the `code_hash` (since it's a
        # parameter to the pointwise decorator) - but is there other info we
        # need to include from inductor_meta?
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(code_hash, backend_hash, salt)

        bundler = _AutotuneCacheBundlerImpl(key, cache)
        if not bundler._load_cache():
            # We couldn't load from the cache - so save the data so we can store
            # the saved autotunes.
            context_bundler._bundler = bundler

        # If we get a cache hit don't bother saving any of the individual
        # autotune results.

    # Call this after all individual autotune results are finished for a
    # inductor python file. If we gathered any individual results then we bundle
    # those and put it into the cache.
    @classmethod
    def end_compile(cls) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            context_bundler._bundler = None
            bundler.end_compile()

    @classmethod
    def sync(cls) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            bundler.sync()

    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None:
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            # The filename comes in as something like
            # "/tmp/tmp{random}/{aa}/{basename}.py" (where aa is
            # basename[1:3]). Strip it down and make sure that it looks like a path
            # we could reconstruct (because it's possible for the caller to
            # customize the path).
            basename = os.path.basename(filename)

            # TODO: check cache_dir() vs filename, then strip dirname
            bundler.put(basename, data)


# Remove the comments from the code (which include things like run ids and file
# paths) and then hash the result.
def _comment_stripped_hash(code: str) -> str:
    code = re.sub(r"#.*$", "", code, count=0, flags=re.MULTILINE)
    return torch._inductor.codecache.code_hash(code)


def _should_use_remote_autotune_cache(inductor_meta: _InductorMetaTy) -> bool:
    if (config := inductor_meta.get("autotune_remote_cache")) is not None:
        return bool(config)
    if not inductor_meta.get("is_fbcode"):
        return False
    if torch._utils_internal.is_fb_unit_test():
        return False
    if inductor_meta.get("is_hip"):
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:autotune_memcache_version"
    )


def _reconstruct_triton_config(
    best_config: dict[str, Any],
    extra_options: JsonDataTy | None,
) -> Config:
    num_warps = best_config.pop("num_warps")
    num_stages = best_config.pop("num_stages")
    config_args: dict[str, Any] = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if HAS_WARP_SPEC:
        config_args.update(
            {
                "num_consumer_groups": best_config.pop("num_consumer_groups", 0),
                "num_buffers_warp_spec": best_config.pop("num_buffers_warp_spec", 0),
            }
        )
    # pyrefly: ignore [bad-argument-count, unexpected-keyword]
    triton_config = Config(best_config, **config_args)
    # pyrefly: ignore [missing-attribute]
    triton_config.extra_options = extra_options
    return triton_config


def _load_cached_autotuning(
    best_config: dict[str, JsonDataTy],
    configs_hash: str,
    configs: list[Config],
    inductor_meta: _InductorMetaTy,
) -> Config | None:
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    best_config.pop("triton_cache_hash", None)

    # Extract extra_options if present. This allows third-party backends
    # to restore custom tuned options from the cache.
    extra_options = best_config.pop("extra_options", None)

    found_by_coordesc = inductor_meta.get(
        "coordinate_descent_tuning"
    ) and best_config.pop("found_by_coordesc", False)

    if not found_by_coordesc:
        matching_configs = [
            cfg
            for cfg in configs
            # pyrefly: ignore [missing-attribute]
            if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
            # pyrefly: ignore [missing-attribute]
            and cfg.num_warps == best_config.get("num_warps")
            # pyrefly: ignore [missing-attribute]
            and cfg.num_stages == best_config.get("num_stages")
        ]
        if len(matching_configs) == 1:
            matched_config = matching_configs[0]
            # pyrefly: ignore [missing-attribute]
            matched_config.extra_options = extra_options
            return matched_config

    # Reconstruct Config from cached data. This handles both coordesc
    # configs and dynamically added configs (e.g. _dynamic_scale_rblock)
    # that aren't in the original config list.
    best_config.pop("found_by_coordesc", None)
    triton_config = _reconstruct_triton_config(best_config, extra_options)
    if found_by_coordesc:
        # pyrefly: ignore [missing-attribute]
        triton_config.found_by_coordesc = True
    return triton_config


def _splitext_nodot(basename: str) -> tuple[str, str]:
    root, ext = os.path.splitext(basename)
    if ext:
        ext = ext[1:]
    return root, ext
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import typing
from enum import auto, Enum

import torch
from torch.utils._triton import has_triton_package


# The following maximums only apply to runtime autotuning, when using FixedTritonConfig one may see larger values
# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 8192 if torch.version.hip else 4096,
    "Y": 1024,
    "Z": 1024,
    "R0_": 4096 * 16,  # * 16 is multi-kernel only
    "R1_": 2048 * 16,  # * 16 is multi-kernel only
}
TRITON_MAX_RSPLIT = 64
TRITON_MAX_TENSOR_NUMEL = 1 << 20
TRITON_DOT_MIN_BLOCK = 16
TRITON_DEFAULT_BLOCK_SIZES = {
    "XBLOCK": 128,
    "YBLOCK": 1,
    "ZBLOCK": 1,
    "R0_BLOCK": 1,
}
TRITON_DEFAULT_RSPLIT = 1
TRITON_DEFAULT_RSPLIT_SIZE = 1


def native_matmul_block_numel(
    kwargs: typing.Mapping[str, int], r0_block: int | None = None
) -> int:
    return (
        kwargs.get("XBLOCK", 1)
        * kwargs.get("YBLOCK", 1)
        * kwargs.get("ZBLOCK", 1)
        * (kwargs.get("R0_BLOCK", 1) if r0_block is None else r0_block)
    )


def native_matmul_persistent_rblock(r0_block: int) -> int:
    return max(r0_block, TRITON_DOT_MIN_BLOCK)


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


# Define `AttrsDescriptorWrapper` function with clear conditional handling
if has_triton_package():
    import triton
    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        # Triton 3.2.0 - the second implementation
        from triton.backends.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "tt.divisibility": divisible_by_16,
                "tt.equal_to": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            res = AttrsDescriptor.from_dict(
                {"arg_properties": kwargs, "cls": AttrsDescriptor.__name__}
            )
            if res.property_values["tt.divisibility"] != 16:
                raise AssertionError(
                    f"Expected tt.divisibility == 16, got {res.property_values['tt.divisibility']}"
                )
            if res.property_values["tt.equal_to"] != 1:
                raise AssertionError(
                    f"Expected tt.equal_to == 1, got {res.property_values['tt.equal_to']}"
                )
            return res

    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        # Triton 3.0.0 - the original implementation
        from triton.compiler.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "divisible_by_16": divisible_by_16,
                "equal_to_1": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            return AttrsDescriptor(**kwargs)

    else:
        # Triton in 2025:
        # note: there's also a range of triton commits not currently supported
        # from ~Dec 9, 2024 to Jan 1 2025, in which AttrsDescriptors are still
        # used, but the contents are different.

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # pyrefly: ignore [not-iterable]
            # Build attr dict merging divisibility and pointer_range per arg index,
            # since a single arg can carry both attributes.
            result = {(x,): [["tt.divisibility", 16]] for x in (divisible_by_16 or ())}
            for x in pointer_range_32 or ():
                key = (x,)
                if key in result:
                    result[key].append(["tt.pointer_range", 32])
                else:
                    result[key] = [["tt.pointer_range", 32]]
            return result

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    AttrsDescriptorWrapper = collections.namedtuple(  # type: ignore[no-redef, name-match]
        # pyrefly: ignore [invalid-argument]
        "AttrsDescriptor",
        ["divisible_by_16", "equal_to_1", "pointer_range_32"],
        defaults=[(), (), ()],
    )


class HeuristicType(Enum):
    PERSISTENT_REDUCTION = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    SPLIT_SCAN = auto()
    TEMPLATE = auto()
    USER_AUTOTUNE = auto()
    FIXED = auto()


class AutotuneHint(Enum):
    ONE_ELEMENT_PER_THREAD = 0

    # Triton codegen tries to codegen set of AutotuneHints.
    # Enum.__repr__ looks like "<AutotuneHint.ELEMENTS_PER_WARP_32: 0>""
    # which isn't valid python.
    # Enum.__str__ will just return "AutotuneHint.ELEMENTS_PER_WARP_32".
    __repr__ = Enum.__str__


class DeviceProperties(typing.NamedTuple):
    """Copy device properties into a data structure not requiring torch to be imported"""

    type: str  # type: ignore[assignment]
    index: int  # type: ignore[assignment]
    multi_processor_count: int
    cc: int
    major: int | None = None
    regs_per_multiprocessor: int | None = None
    max_threads_per_multi_processor: int | None = None
    max_threads_per_block: int | None = None
    warp_size: int | None = None

    @property
    def warp_size_or_default(self) -> int:
        if self.warp_size is not None:
            return self.warp_size
        if self.type in ("cuda", "hip"):
            raise RuntimeError(f"{self.type} device properties must report warp_size")
        return 32

    @classmethod
    @functools.cache
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)
        try:
            multi_processor_count = props.multi_processor_count
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            elif device_type == "mtia":
                multi_processor_count = 64
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
            warp_size=getattr(props, "warp_size", None),
        )


def get_warp_size(device) -> int:
    """Return the wave/warp size in threads for the given device.

    Reads from torch.cuda.get_device_properties(device).warp_size via the cached
    DeviceProperties.create(). Correct on both AMD (64 for CDNA/gfx9, 32 for
    RDNA/gfx10+) and NVIDIA (always 32). Missing cuda/hip warp_size metadata is
    treated as an error rather than silently falling back.
    """
    return DeviceProperties.create(device).warp_size_or_default


class TritonMeta(typing.TypedDict, total=False):
    """Metadata bag threaded from Triton codegen into the runtime launcher.

    total=False because the key set is populated incrementally across codegen
    (signature/device/constants/configs first, then backend/ROCm/tlx extras)
    and the whole bag is forwarded verbatim to external Triton APIs, which
    tolerate and ignore keys they do not recognize. `device` is typed as the
    codegen-time DeviceProperties; CachingAutotuner rewrites it to the integer
    device index before reaching Triton, and the runtime read sites that expect
    that int narrow it with an explicit cast.
    """

    signature: dict[str, typing.Any]
    device: DeviceProperties
    device_type: str
    constants: dict[str, typing.Any]
    configs: list[typing.Any]
    native_matmul: bool
    launch_cooperative_grid: bool
    enable_fp_fusion: bool
    launch_pdl: bool
    disable_ftz: bool
    matrix_instr_nonkdim: int
    waves_per_eu: int
    kpack: int
    restore_value: tuple[str, ...]
    reset_to_zero: tuple[str, ...]
    backend_options: dict[str, typing.Any]


class InductorMeta(typing.TypedDict, total=False):
    """The inductor kernel-config / heuristics metadata bag.

    Produced on the codegen side (TritonKernel.inductor_meta_common /
    inductor_meta_per_kernel plus the codegen_kernel literal) and consumed by
    the runtime autotuning machinery in triton_heuristics.py,
    coordinate_descent_tuner.py, and autotune_cache.py. Every key is optional
    (total=False): consumers read via .get(...) with defaults, and several keys
    are injected only on specific paths (reductions, combo kernels, fixed grids).
    Dynamically keyed nested bags (e.g. combo_grid_meta) stay typed dict[str,
    Any] because their keys are computed at runtime. The codegen-side producers
    still build this bag as a plain dict[str, Any], so the write side is not yet
    checked against this TypedDict; typing the producers is left as a follow-up.
    """

    # Global inductor config snapshot (inductor_meta_common / inductor_meta_from_config)
    backend_hash: str | None
    assert_indirect_indexing: bool
    autotune_local_cache: bool
    autotune_pointwise: bool
    autotune_remote_cache: bool | None
    bundled_autotune_remote_cache: bool | None
    force_disable_caches: bool
    dynamic_scale_rblock: bool
    incremental_autotune: bool
    max_autotune: bool
    max_autotune_pointwise: bool
    min_split_scan_rblock: int
    spill_threshold: int
    store_cubin: bool
    deterministic: bool
    batch_invariant: bool
    force_filter_reduction_configs: bool
    mix_order_reduction_allow_multi_stages: bool
    dynamic_disable_pipelining: bool
    are_deterministic_algorithms_enabled: bool
    is_hip: bool | None
    is_fbcode: bool
    profile_bandwidth: bool
    profile_bandwidth_regex: str
    profile_bandwidth_output: str
    profile_bandwidth_with_do_bench_using_profiling: bool
    coordinate_descent_tuning: bool
    coordinate_descent_search_radius: int
    coordinate_descent_check_all_directions: bool

    # Per-kernel metadata (inductor_meta_per_kernel)
    no_x_dim: bool
    atomic_add_found: bool
    num_load: int
    num_store: int
    num_reduction: int
    autotune_hints: typing.Any
    RSPLIT_SIZE: int
    has_loadstore_with_contiguous_rdim: bool
    tma_min_block_sizes: dict[str, int]
    uses_tma: bool
    uses_device_tma: bool
    host_tma_descriptor_args: dict[str, dict[str, typing.Any]]
    tiling_scores: typing.Any
    min_xblock: int
    min_rblock: int
    persistent_reduction: bool
    native_matmul_persistent_rblock: int
    add_persistent_rblock: bool
    max_persistent_rblock: int
    strict_sum_rblock: int
    kernel_num_gb: float
    kernel_flop: int

    # Kernel identity / launch (codegen_kernel literal)
    grid_type: str
    kernel_name: str
    mutated_arg_names: typing.Any
    optimize_mem: bool

    # Injected at runtime (CachingAutotuner, reduction heuristics, combo kernels)
    warp_size: int | None
    max_threads_per_block: int | None
    reduction_hint: typing.Any
    combo_grid_meta: dict[str, typing.Any]
    combo_tuning_groups: typing.Any
    combo_coordesc_field_order: list[str]
    combo_coordesc_field_limits: dict[str, int]
    combo_warp_stage_candidates: typing.Any
    extra_launcher_args: typing.Any
    fixed_grid: typing.Any
    precomputed_grids: typing.Any
    config_args: typing.Any
    use_fast_triton_launcher: bool


class HalideInputSpec(typing.NamedTuple):
    ctype: str
    name: str
    shape: list[str] | None = None
    stride: list[str] | None = None
    offset: str | None = None
    alias_of: str | None = None

    def bindings_type(self) -> str:
        if self.ctype in ("at::Half*", "at::BFloat16*"):
            return "uint16_t*"  # half not defined
        return self.ctype

    def halide_type(self) -> str:
        if self.ctype == "at::Half*":
            return "halide_type_t(halide_type_float, 16)"  # half not defined
        if self.ctype == "at::BFloat16*":
            return "halide_type_t(halide_type_bfloat, 16)"  # half not defined
        return f"halide_type_of<{self.ctype.replace('*', '')}>()"

    def is_scalar(self) -> bool:
        return self.shape is None

    def is_buffer(self) -> bool:
        return self.shape is not None


class HalideMeta(typing.NamedTuple):
    argtypes: list[HalideInputSpec]
    target: str
    scheduler: str | None = None
    scheduler_flags: dict[str, int | str] | None = None
    cuda_device: int | None = None

    def args(self) -> list[str]:
        """Command line args to pass to halide generator"""
        args = [f"target={self.target}"]
        if self.scheduler:
            args.append(f"autoscheduler={self.scheduler}")
        if self.scheduler_flags:
            if not self.scheduler:
                raise AssertionError("scheduler_flags requires scheduler to be set")
            for k, v in self.scheduler_flags.items():
                args.append(f"autoscheduler.{k}={v}")
        return args

    def is_cuda(self) -> bool:
        return self.cuda_device is not None
<|endoftext|># mypy: allow-untyped-defs
import math


try:
    import halide as hl  # type: ignore[import-untyped, import-not-found]
except ImportError:
    hl = None

PHILOX_N_ROUNDS_DEFAULT = 10  # Default number of rounds for philox

if hl is not None:
    PHILOX_KEY_A_U32 = hl.u32(0x9E3779B9)
    PHILOX_KEY_B_U32 = hl.u32(0xBB67AE85)
    PHILOX_ROUND_A_U32 = hl.u32(0xD2511F53)
    PHILOX_ROUND_B_U32 = hl.u32(0xCD9E8D57)
else:
    PHILOX_KEY_A_U32 = None
    PHILOX_KEY_B_U32 = None
    PHILOX_ROUND_A_U32 = None
    PHILOX_ROUND_B_U32 = None


def _pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = hl.max(hl.f32(1.0e-7), u1)
    th = hl.f32(math.tau) * u2
    r = hl.sqrt(hl.f32(-2.0) * hl.log(u1))
    return r * hl.cos(th), r * hl.sin(th)


def _uint_to_uniform_float(x):
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """

    # TODO:
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    # https://github.com/triton-lang/triton/blob/e4a0d93ff1a367c7d4eeebbcd7079ed267e6b06f/python/triton/language/random.py#L116-L132.
    if x.type() != hl.UInt(32) and x.type() != hl.Int(32):
        raise AssertionError(f"Expected UInt(32) or Int(32), got {x.type()}")
    x = hl.cast(hl.Int(32), x)
    scale = hl.f64(4.6566127342e-10)
    x = hl.select(x < 0, -x - 1, x)
    return x * scale


def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds):
    def umulhi(a, b):
        a = hl.cast(hl.UInt(64), a)
        b = hl.cast(hl.UInt(64), b)
        return hl.cast(hl.UInt(32), ((a * b) >> 32) & hl.u64(0xFFFFFFFF))

    for _ in range(n_rounds):
        _c0, _c2 = c0, c2

        c0 = umulhi(PHILOX_ROUND_B_U32, _c2) ^ c1 ^ k0
        c2 = umulhi(PHILOX_ROUND_A_U32, _c0) ^ c3 ^ k1
        c1 = PHILOX_ROUND_B_U32 * _c2
        c3 = PHILOX_ROUND_A_U32 * _c0
        # raise key
        k0 = k0 + PHILOX_KEY_A_U32
        k1 = k1 + PHILOX_KEY_B_U32

    return c0, c1, c2, c3


def halide_philox(seed, c0, c1, c2, c3, n_rounds):
    seed = hl.cast(hl.UInt(64), seed)

    if c0.type().bits() != 32:
        raise AssertionError(f"Expected 32-bit type, got {c0.type().bits()}-bit")

    seed_hi = hl.cast(hl.UInt(32), (seed >> 32) & hl.u64(0xFFFFFFFF))
    seed_lo = hl.cast(hl.UInt(32), seed & hl.u64(0xFFFFFFFF))

    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)


def randint4x(seed, offset, n_rounds):
    offset = hl.cast(hl.UInt(32), offset)
    _0 = hl.u32(0)
    return halide_philox(seed, offset, _0, _0, _0, n_rounds)


def rand4x(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    i1, i2, i3, i4 = randint4x(seed, offset, n_rounds)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    u3 = _uint_to_uniform_float(i3)
    u4 = _uint_to_uniform_float(i4)
    return u1, u2, u3, u4


def randint(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret


def rand(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    source = randint(seed, offset, n_rounds)
    return _uint_to_uniform_float(source)


def rand_eager_kernel(seed, offset_blocks, tid, VEC, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    inv = hl.cast(hl.Float(32), 1.0 / 4294967296.0)  # 2^-32
    half = hl.cast(hl.Float(32), 0.5) * inv

    tid_u64 = hl.cast(hl.UInt(64), tid)
    VEC_u64 = hl.cast(hl.UInt(64), VEC)
    subseq = tid_u64 // VEC_u64
    which4 = (tid_u64 % VEC_u64) // hl.cast(hl.UInt(64), 4)
    lane = tid_u64 % hl.cast(hl.UInt(64), 4)

    offblk = hl.cast(hl.UInt(64), offset_blocks) + which4

    c0 = hl.cast(hl.UInt(32), offblk & hl.cast(hl.UInt(64), 0xFFFFFFFF))
    c1 = hl.cast(
        hl.UInt(32),
        (offblk >> hl.cast(hl.UInt(64), 32)) & hl.cast(hl.UInt(64), 0xFFFFFFFF),
    )
    c2 = hl.cast(hl.UInt(32), subseq & hl.cast(hl.UInt(64), 0xFFFFFFFF))
    c3 = hl.cast(
        hl.UInt(32),
        (subseq >> hl.cast(hl.UInt(64), 32)) & hl.cast(hl.UInt(64), 0xFFFFFFFF),
    )

    u0, u1, u2, u3 = halide_philox(seed, c0, c1, c2, c3, n_rounds)

    v01 = hl.select(lane == hl.cast(hl.UInt(64), 0), u0, u1)
    v23 = hl.select(lane == hl.cast(hl.UInt(64), 2), u2, u3)
    rand_int = hl.select(
        (lane == hl.cast(hl.UInt(64), 0)) | (lane == hl.cast(hl.UInt(64), 1)), v01, v23
    )

    return hl.cast(hl.Float(32), 1.0) - (hl.cast(hl.Float(32), rand_int) * inv + half)


def randn(seed, offset):
    i1, i2, _, _ = randint4x(seed, offset, PHILOX_N_ROUNDS_DEFAULT)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    n1, _ = _pair_uniform_to_normal(u1, u2)
    return n1


def randint64(seed, offset, low, high):
    r0, r1, _r2, _r3 = randint4x(seed, offset, PHILOX_N_ROUNDS_DEFAULT)
    r0 = hl.cast(hl.UInt(64), r0)
    r1 = hl.cast(hl.UInt(64), r1)

    result = r0 | (r1 << 32)
    size = high - low
    result = result % hl.cast(hl.UInt(64), size)
    result = hl.cast(hl.Int(64), result) + low
    return result
<|endoftext|>"""Lock acquisition utilities

This module provides safe and unsafe lock acquisition functions for both threading.Lock
and FileLock objects, with configurable timeout behaviors. It supports three timeout modes:
blocking (infinite wait), non-blocking (immediate), and blocking with timeout (finite wait).

The module offers both context manager and manual acquisition patterns:
- Safe acquisition: Uses context managers that automatically handle lock release
- Unsafe acquisition: Manual acquisition that requires explicit release by the caller
"""

from __future__ import annotations

from contextlib import _GeneratorContextManager, contextmanager, ExitStack
from typing import TYPE_CHECKING, TypeAlias
from typing_extensions import Protocol

from filelock import BaseFileLock, Timeout

from . import exceptions


if TYPE_CHECKING:
    from collections.abc import Generator
    from threading import Lock

    from .implementations import _CacheImpl


_LockContextManager: TypeAlias = _GeneratorContextManager[None, None, None]


class _LockProtocol(Protocol):  # noqa: PYI046
    def __call__(self, timeout: float | None = None) -> _LockContextManager: ...


# Infinite timeout - blocks indefinitely until lock is acquired.
_BLOCKING: float = -1
# No timeout - returns immediately if lock cannot be acquired.
_NON_BLOCKING: float = 0
# Finite timeout - blocks for a specified duration before raising a timeout error.
_BLOCKING_WITH_TIMEOUT: float = 60.0
# Default timeout for lock acquisition.
_DEFAULT_TIMEOUT: float = _BLOCKING_WITH_TIMEOUT


@contextmanager
def _acquire_lock_with_timeout(
    lock: Lock,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    """Context manager that safely acquires a threading.Lock with timeout and automatically releases it.

    This function provides a safe way to acquire a lock with timeout support, ensuring
    the lock is always released even if an exception occurs during execution.

    Args:
        lock: The threading.Lock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Yields:
        None: Yields control to the caller while holding the lock

    Raises:
        LockTimeoutError: If the lock cannot be acquired within the timeout period

    Example:
        with _acquire_lock_with_timeout(my_lock, timeout=30.0):
            # Critical section - lock is held
            perform_critical_operation()
        # Lock is automatically released here
    """
    _unsafe_acquire_lock_with_timeout(lock, timeout=timeout)

    try:
        yield
    finally:
        lock.release()


def _unsafe_acquire_lock_with_timeout(lock: Lock, timeout: float | None = None) -> None:
    """Acquire a threading.Lock with timeout without automatic release (unsafe).

    This function acquires a lock with timeout support but does NOT automatically
    release it. The caller is responsible for releasing the lock explicitly.
    Use this only when you need manual control over lock lifetime.

    Args:
        lock: The threading.Lock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Raises:
        LockTimeoutError: If the lock cannot be acquired within the timeout period

    Warning:
        This is an "unsafe" function because it does not automatically release
        the lock. Always call lock.release() when done, preferably in a try/finally
        block or use the safe _acquire_lock_with_timeout context manager instead.

    Example:
        lock = Lock()
        try:
            _unsafe_acquire_lock_with_timeout(lock, timeout=30.0)
            # Critical section - lock is held
            perform_critical_operation()
        finally:
            lock.release()  # Must manually release!
    """
    _timeout: float = timeout if timeout is not None else _DEFAULT_TIMEOUT
    if not lock.acquire(timeout=_timeout):
        raise exceptions.LockTimeoutError(lock, _timeout)


@contextmanager
def _acquire_flock_with_timeout(
    flock: BaseFileLock,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    """Context manager that safely acquires a FileLock with timeout and automatically releases it.

    This function provides a safe way to acquire a file lock with timeout support, ensuring
    the lock is always released even if an exception occurs during execution.

    Args:
        flock: The FileLock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Yields:
        None: Yields control to the caller while holding the file lock

    Raises:
        FileLockTimeoutError: If the file lock cannot be acquired within the timeout period

    Example:
        flock = FileLock("/tmp/my_process.lock")
        with _acquire_flock_with_timeout(flock, timeout=30.0):
            # Critical section - file lock is held
            perform_exclusive_file_operation()
        # File lock is automatically released here
    """
    _unsafe_acquire_flock_with_timeout(flock, timeout=timeout)

    try:
        yield
    finally:
        flock.release()


def _unsafe_acquire_flock_with_timeout(
    flock: BaseFileLock,
    timeout: float | None,
) -> None:
    """Acquire a FileLock with timeout without automatic release (unsafe).

    This function acquires a file lock with timeout support but does NOT automatically
    release it. The caller is responsible for releasing the lock explicitly.
    Use this only when you need manual control over lock lifetime.

    Args:
        flock: The FileLock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Raises:
        FileLockTimeoutError: If the file lock cannot be acquired within the timeout period

    Warning:
        This is an "unsafe" function because it does not automatically release
        the lock. Always call flock.release() when done, preferably in a try/finally
        block or use the safe _acquire_flock_with_timeout context manager instead.

    Example:
        flock = FileLock("/tmp/my_process.lock")
        try:
            _unsafe_acquire_flock_with_timeout(flock, timeout=30.0)
            # Critical section - file lock is held
            perform_exclusive_file_operation()
        finally:
            flock.release()  # Must manually release!
    """
    _timeout: float = timeout if timeout is not None else _DEFAULT_TIMEOUT
    try:
        _ = flock.acquire(timeout=_timeout)
    except Timeout as err:
        raise exceptions.FileLockTimeoutError(flock, _timeout) from err


@contextmanager
def _acquire_many_impl_locks_with_timeout(
    *impls: _CacheImpl,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    with ExitStack() as stack:
        for impl in impls:
            stack.enter_context(impl.lock(timeout))
        yield
<|endoftext|>"""Context management for PyTorch Inductor runtime caching.

This module provides context classes for collecting configuration and environment
information used in caching decisions for PyTorch's Inductor runtime.
"""

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from typing_extensions import override, TypedDict

import torch


class _Context(ABC):
    """Abstract base class for context providers.

    Context providers collect specific configuration and environment information
    that affects compilation and runtime behavior.
    """

    @staticmethod
    @abstractmethod
    def forms_of_context() -> Sequence[str]:
        """Return a sequence of context form names provided by this context class.

        Returns:
            A sequence of strings representing the available context forms.
        """


class _RuntimeContext(_Context):
    """Context provider for runtime configuration and environment settings.

    Collects configuration settings that affect runtime behavior but not
    compilation, such as Inductor configs, determinism settings, and CUDA
    matmul precision configurations.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the runtime context forms provided by this class.

        Returns:
            A sequence containing the available runtime context forms:
            - "inductor_configs": PyTorch Inductor configuration settings
            - "torch_determinism_configs": Deterministic algorithm settings
            - "cuda_matmul_precision_configs": CUDA matrix multiplication precision settings
        """
        return (
            "inductor_configs",
            "torch_determinism_configs",
            "cuda_matmul_precision_configs",
        )

    @staticmethod
    def inductor_configs() -> dict[str, object]:
        """Get portable Inductor configuration settings.

        Returns:
            A dictionary containing Inductor configuration settings,
            including private configs.
        """
        from torch._inductor import config

        return config.save_config_portable(ignore_private_configs=False)

    @staticmethod
    def torch_determinism_configs() -> dict[str, object]:
        """Get PyTorch deterministic algorithm configuration settings.

        Returns:
            A dictionary containing deterministic algorithm settings:
            - Whether deterministic algorithms are enabled
            - Whether deterministic algorithm warnings are enabled
            - Fill uninitialized memory setting
        """
        return {
            "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch.is_deterministic_algorithms_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "torch.utils.deterministic.fill_uninitialized_memory": getattr(
                torch.utils.deterministic, "fill_uninitialized_memory", None
            ),
        }

    @staticmethod
    def cuda_matmul_precision_configs() -> dict[str, object]:
        """Get CUDA matrix multiplication precision configuration settings.

        Returns:
            A dictionary containing CUDA matmul precision settings:
            - FP32 precision setting
            - FP16 reduced precision reduction allowance
            - BF16 reduced precision reduction allowance
        """
        return {
            "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }


class _CompileContext(_Context):
    """Context provider for compilation-related configuration and environment settings.

    Collects information that affects compilation behavior, such as PyTorch and Triton
    versions, runtime environment, and accelerator properties.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the compile context forms provided by this class.

        Returns:
            A sequence containing the available compile context forms:
            - "torch_version_hash": PyTorch version hash
            - "triton_version_hash": Triton version hash (if available)
            - "runtime": Runtime type (CUDA/HIP/None)
            - "runtime_version": Runtime version string
            - "accelerator_properties": GPU/accelerator properties
        """
        return (
            "torch_version_hash",
            "triton_version_hash",
            "runtime",
            "runtime_version",
            "accelerator_properties",
        )

    @cache
    @staticmethod
    def torch_version_hash() -> str:
        """Get base64-encoded PyTorch version hash.

        Returns:
            A base64-encoded string representing the PyTorch version hash.
        """
        from torch._inductor.codecache import torch_key

        return b64encode(torch_key()).decode()

    @cache
    @staticmethod
    def triton_version_hash() -> str | None:
        """Get Triton version key if Triton is available.

        Returns:
            Triton version key if Triton is available, None otherwise.
        """
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        return triton_key() if HAS_TRITON else None

    @cache
    @staticmethod
    def runtime() -> str | None:
        """Determine the runtime type based on available backends.

        Returns:
            "CUDA" if CUDA is available, "HIP" if HIP is available, None otherwise.
        """
        return "CUDA" if torch.version.cuda else "HIP" if torch.version.hip else None

    @cache
    @staticmethod
    def runtime_version() -> str | None:
        """Get the version string for the detected runtime.

        Returns:
            Version string for the current runtime (CUDA or HIP), or None if
            no supported runtime is detected.
        """
        return {
            "CUDA": torch.version.cuda,
            "HIP": torch.version.hip,
            "None": None,
        }.get(_CompileContext.runtime() or "None")

    @cache
    @staticmethod
    def accelerator_properties() -> str | None:
        """Get string representation of CUDA device properties.

        Returns:
            String representation of CUDA device properties if a runtime is
            available, None otherwise.
        """
        return (
            repr(torch.cuda.get_device_properties())
            if _CompileContext.runtime() and torch.cuda.is_available()
            else None
        )


class SelectedRuntimeContext(TypedDict):
    inductor_configs: bool
    torch_determinism_configs: bool
    cuda_matmul_precision_configs: bool


class SelectedCompileContext(TypedDict):
    torch_version_hash: bool
    triton_version_hash: bool
    runtime: bool
    runtime_version: bool
    accelerator_properties: bool


class IsolationSchema(TypedDict):
    """Schema for specifying which context forms to include in cache isolation.

    Attributes:
        runtime_context: Either True (include all runtime context), False (exclude all),
                        or a SelectedRuntimeContext dict specifying which forms to include.
        compile_context: Either True (include all compile context), False (exclude all),
                        or a SelectedCompileContext dict specifying which forms to include.
    """

    runtime_context: SelectedRuntimeContext | bool
    compile_context: SelectedCompileContext | bool


_DEFAULT_ISOLATION_SCHEMA: IsolationSchema = IsolationSchema(
    runtime_context=True, compile_context=True
)


def _collect_runtime_context(
    selection: SelectedRuntimeContext | bool,
) -> dict[str, object] | None:
    """Collect runtime context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_RuntimeContext, form)()
        for form in _RuntimeContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _collect_compile_context(
    selection: SelectedCompileContext | bool,
) -> dict[str, object] | None:
    """Collect compile context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_CompileContext, form)()
        for form in _CompileContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _isolation_context(
    ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA,
) -> dict[str, object]:
    """Generate context data based on the isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A dictionary containing the selected context data with keys
        "runtime_context" and "compile_context", where each value is
        either None (if excluded) or a dict of context form data.
    """
    return {
        "runtime_context": _collect_runtime_context(ischema["runtime_context"]),
        "compile_context": _collect_compile_context(ischema["compile_context"]),
    }


def _isolation_key(ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA) -> str:
    """Generate a unique key for the given isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A 32-character hexadecimal string that uniquely identifies
        the context specified by the isolation schema.
    """
    return sha256(
        json.dumps(_isolation_context(ischema), sort_keys=True).encode()
    ).hexdigest()[:32]
<|endoftext|>"""Utility functions

This module provides helper functions for LRU caching decorators used
throughout the caching system.
"""

from collections.abc import Callable
from functools import lru_cache, wraps
from typing_extensions import ParamSpec, TypedDict, TypeVar

from torch import Tensor


# Type specification for function parameters
P = ParamSpec("P")
# Type variable for function return values
R = TypeVar("R")


def _lru_cache(fn: Callable[P, R]) -> Callable[P, R]:
    """LRU cache decorator with TypeError fallback.

    Provides LRU caching with a fallback mechanism that calls the original
    function if caching fails due to unhashable arguments. Uses a cache
    size of 64 with typed comparison.

    Args:
        fn: The function to be cached.

    Returns:
        A wrapper function that attempts caching with fallback to original function.
    """
    cached_fn = lru_cache(maxsize=64, typed=True)(fn)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return cached_fn(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)

    return wrapper


class EncodedTensor(TypedDict):
    """TypedDict for encoded tensor metadata."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


def _encode_tensor(t: Tensor) -> EncodedTensor:
    """Encode a tensor's metadata into a JSON-serializable dict.

    Args:
        t: PyTorch tensor to encode

    Returns:
        Dict containing shape, stride, and dtype information
    """
    return EncodedTensor(
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        dtype=str(t.dtype),
    )
<|endoftext|>"""Public interfaces for PyTorch Inductor runtime caching.

This module provides high-level caching interfaces for memoization and
result caching functionality.
"""

import atexit
import functools
import json
import logging
import pickle
import shutil
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import cast, Generic, Protocol, TypedDict
from typing_extensions import ParamSpec, TypeVar

from filelock import FileLock

import torch
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import clear_on_fresh_cache

from . import config, implementations, locks


logger = torch._logging.getArtifactLogger(__name__, "caching")

# Type variable for function parameters
_P = ParamSpec("_P")
# Type variable for function return type
_R = TypeVar("_R")
# Type variable for encoded result type
_EncodedR = TypeVar("_EncodedR")


class CacheDumpEntry(TypedDict):
    """A single cache entry in the dump format.

    Attributes:
        params: The encoded function parameters.
        result: The encoded function result.
    """

    params: object
    result: object


class CacheDump(TypedDict):
    """The structure of the memoizer cache dump file.

    Cache entries are organized by sub_key in the collections field.
    Memoizers without a sub_key use None as their key in collections.

    Multiple Memoizer instances with different sub_keys can coexist in the same file.

    Attributes:
        collections: Dictionary mapping sub_keys (or None for root entries)
            to their cache entries.
        cache_size: The total number of cache entries across all collections.
    """

    collections: dict[str | None, dict[str, CacheDumpEntry]]
    cache_size: int


@dataclass
class CacheEntry:
    """A cache entry containing encoded parameters and result.

    This dataclass stores the encoded form of function parameters and
    the (possibly encoded) result for human-readable cache dumps.

    Attributes:
        encoded_params: The encoded function parameters used for debugging/inspection.
        encoded_result: The (possibly encoded) result of the function call.
    """

    encoded_params: object
    encoded_result: object


@dataclass
class InterimResult(Generic[_R]):
    """Wrapper for an available interim result from a DeferredRecording.

    When get_interim_result() returns an InterimResult, it means the
    make_interim_result callable was provided and has been invoked.
    The wrapped value is the result of that invocation.

    This allows distinguishing between:
    - No interim result configured: get_interim_result() returns None
    - Interim result available: get_interim_result() returns InterimResult(value)
      where value may itself be None if the callable returns None.

    Attributes:
        value: The value returned by calling make_interim_result().
               May legitimately be None if the callable returns None.
    """

    value: _R


@dataclass
class DeferredRecording(Generic[_R, _EncodedR]):
    """Signals that recording should happen at a later time.

    When returned from a custom_result_encoder, the memoizer will:
    1. Skip immediate caching
    2. Register a completion callback on this object

    The encoder is responsible for calling `finalize(encoded_result)`
    when the actual result is ready to be cached. This is useful when
    the function returns a value whose final encoded form is not yet
    available (e.g., an object containing a pending computation),
    allowing the expensive work to remain hidden while still enabling caching.

    This class handles the race condition where the computation might complete
    before the memoizer has registered its callback. If finalize() is called
    first, the result is stored and the callbacks are invoked when registered.

    Multiple callbacks can be registered - they will all be invoked when
    finalize() is called (or immediately if already completed). Callbacks are
    guaranteed to be invoked in registration order.

    Optionally, a `make_interim_result` callable can be provided to handle
    subsequent calls while the deferred recording is pending. When a memoized
    function is called again with the same parameters while a deferred recording
    is still pending, the memoizer can use this callable to construct an
    appropriate return value instead of re-executing the function. The callable
    does not necessarily return the same object - it provides a way to derive
    a suitable response based on the pending deferred recording's context.

    This class is thread-safe: concurrent calls to finalize() and
    register_callback() are properly synchronized.

    WARNING: Callbacks are executed while holding the internal lock to ensure
    ordering guarantees. Callbacks MUST NOT call back into this DeferredRecording
    instance (e.g., calling finalize() or register_callback() from within a
    callback) as this will cause a deadlock.

    Example usage in an encoder:
        def my_encoder(*args, **kwargs):
            def encode(result: _R) -> DeferredRecording[_R, _EncodedR]:
                deferred: DeferredRecording[_R, _EncodedR] = DeferredRecording(
                        # Construct an appropriate return for subsequent calls while pending
                        make_interim_result=lambda: result
                    )

                def on_complete():
                    encoded_result: _EncodedR = _encode_result(result)
                    deferred.finalize(encoded_result)

                result.add_completion_callback(on_complete)
                return deferred

            return encode

    Attributes:
        _callbacks: List of callbacks to invoke when finalize() is called.
                   Set to None after finalize() is called to indicate completion.
        _encoded_result: The encoded result passed to finalize().
        _lock: Lock for thread-safe access to mutable state.
        make_interim_result: Optional callable that constructs a return value for subsequent
                           calls while the deferred recording is pending. If set, the memoizer
                           calls this instead of re-executing the function. The callable may
                           return the same object or construct a new appropriate response.
    """

    _callbacks: list[Callable[[_EncodedR], None]] | None = field(
        default_factory=list, repr=False
    )
    _encoded_result: _EncodedR | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    make_interim_result: Callable[[], _R] | None = field(default=None, repr=False)

    def finalize(self, encoded_result: _EncodedR) -> None:
        """Finalize the deferred recording with the encoded result.

        This method should be called by the encoder (typically via a completion
        callback) when the actual result is ready to be cached.

        All registered callbacks are invoked with the encoded result in
        registration order. If no callbacks are registered yet, the result is
        stored and callbacks will be invoked when registered.

        This method is thread-safe.

        WARNING: Callbacks are executed while holding the internal lock.
        Do not call finalize() or register_callback() from within a callback,
        as this will cause a deadlock.

        Args:
            encoded_result: The encoded result to cache.
        """
        with self._lock:
            if self._callbacks is None:
                raise RuntimeError(
                    "finalize() called multiple times on DeferredRecording"
                )

            # Store the result and get callbacks to invoke
            self._encoded_result = encoded_result
            callbacks_to_invoke = self._callbacks
            self._callbacks = None

            # Execute callbacks inside the lock to ensure ordering:
            # - All pre-finalize callbacks execute in registration order
            # - Any callback registered during/after finalize will see
            #   _callbacks=None and execute immediately (also under lock)
            for callback in callbacks_to_invoke:
                callback(encoded_result)

    def register_callback(self, callback: Callable[[_EncodedR], None]) -> None:
        """Register a completion callback.

        This method is called by memoizers to register callbacks that
        will insert cache entries. Multiple callbacks can be registered.

        If finalize() has already been called (e.g., the computation was fast),
        the callback is invoked immediately with the stored result.

        This method is thread-safe. Callbacks are guaranteed to be invoked
        in registration order.

        WARNING: Callbacks are executed while holding the internal lock.
        Do not call finalize() or register_callback() from within a callback,
        as this will cause a deadlock.

        Args:
            callback: The function to call with the encoded result.
        """
        with self._lock:
            if self._callbacks is None:
                # Already finalized - invoke immediately while holding lock
                # to maintain ordering with respect to other callbacks
                # we need to cast here, we can't assert because the encoded
                # result may genuinely be None
                callback(cast(_EncodedR, self._encoded_result))
            else:
                self._callbacks.append(callback)

    def get_interim_result(self) -> InterimResult[_R] | None:
        """Try to get an interim result for use while deferred recording is pending.

        When a memoized function is called again with the same parameters while
        a deferred recording is still pending, the memoizer uses this method to
        check if an interim result can be returned instead of re-executing the
        function.

        This method provides thread-safe access to the make_interim_result
        callable. It atomically checks if the callable was provided and invokes
        it if so.

        This method is thread-safe.

        Returns:
            InterimResult wrapping the value if make_interim_result was provided,
            or None if no interim result is available.

        Note:
            The callable is invoked each time this method is called, so if
            make_interim_result returns a new object each time, callers will
            receive different objects. To return the same object, the callable
            should capture and return a reference to a single object.
        """
        with self._lock:
            if self.make_interim_result is not None:
                return InterimResult(self.make_interim_result())
            return None


class ResultEncoderFactory(Protocol[_P, _R, _EncodedR]):
    """Protocol for custom result encoder factories.

    A result encoder factory is a callable with the following signature:
        factory(fn) -> params_to_encoder(*args, **kwargs) -> encoder(result) -> encoded

    The three levels are:
    1. factory(fn): Takes the underlying unwrapped function, returns a params_to_encoder
    2. params_to_encoder(*args, **kwargs): Takes the memoized function's arguments,
       returns an encoder function
    3. encoder(result): Converts the result R -> _EncodedR (or DeferredRecording)

    The `fn` parameter allows the encoder to call the underlying function without
    triggering memoization, which is useful when the encoder needs to re-execute
    the function (e.g., to get a fresh result for DeferredRecording).

    Example:
        def my_encoder_factory(fn: Callable) -> Callable:
            def params_to_encoder(*args, **kwargs) -> Callable:
                def encode(result: R) -> EncodedR:
                    return {"encoded": result}
                return encode
            return params_to_encoder

        @memoizer.record(custom_result_encoder=my_encoder_factory)
        def compute(x: int) -> int:
            return x * 2
    """

    def __call__(
        self,
        fn: Callable[_P, _R],
    ) -> Callable[_P, Callable[[_R], _EncodedR | DeferredRecording[_R, _EncodedR]]]: ...


# pyrefly: ignore [variance-mismatch]
class ResultDecoderFactory(Protocol[_P, _R, _EncodedR]):
    """Protocol for custom result decoder factories.

    A result decoder factory is a callable with the following signature:
        factory(fn) -> params_to_decoder(*args, **kwargs) -> decoder(encoded) -> result

    The three levels are:
    1. factory(fn): Takes the underlying unwrapped function, returns a params_to_decoder
    2. params_to_decoder(*args, **kwargs): Takes the memoized function's arguments,
       returns a decoder function
    3. decoder(encoded): Converts the encoded result _EncodedR -> R

    The `fn` parameter allows the decoder to call the underlying function without
    triggering memoization, which is useful for fallback paths when the cached
    result cannot be fully decoded.

    Example:
        def my_decoder_factory(fn: Callable) -> Callable:
            def params_to_decoder(*args, **kwargs) -> Callable:
                def decode(encoded_result: EncodedR) -> R:
                    return encoded_result["value"]
                return decode
            return params_to_decoder

        @memoizer.replay(custom_result_decoder=my_decoder_factory)
        def compute(x: int) -> int:
            return x * 2
    """

    def __call__(
        self,
        fn: Callable[_P, _R],
    ) -> Callable[_P, Callable[[_EncodedR], _R]]: ...


class _BaseMemoizer:
    """Base class for memoization interfaces.

    This class provides the common memoize method that orchestrates
    record and replay functionality.
    """

    @staticmethod
    def _make_key(
        custom_params_encoder: Callable[..., object] | None,
        *args: object,
        **kwargs: object,
    ) -> str:
        """Generate a cache key from function parameters.

        Args:
            custom_params_encoder: Optional encoder to apply to function parameters.
                                  If None, params are pickled directly.
            *args: Positional arguments to encode.
            **kwargs: Keyword arguments to encode.

        Returns:
            A 32-character hex string suitable for use as a cache key.
        """
        if custom_params_encoder is None:
            # Pickle the parameters directly
            pickled_params: bytes = pickle.dumps((args, kwargs))
        else:
            # Encode the parameters using the custom encoder
            encoded_params = custom_params_encoder(*args, **kwargs)
            # Pickle the encoded output
            pickled_params = pickle.dumps(encoded_params)

        # Hash the pickled bytes with SHA256
        hash_obj = sha256(pickled_params)

        # Get hex digest and truncate to 32 characters
        return hash_obj.hexdigest()[:32]

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result. Must be implemented by subclasses.

        This can be used standalone to cache results without necessarily
        replaying them, for example for logging or analytics purposes.

        See memoize() for a description of the arguments.
        """
        raise NotImplementedError

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result. Must be implemented by subclasses.

        This can be used standalone to use cached values that are otherwise
        too hard to automatically record.

        See memoize() for a description of the arguments.
        """
        raise NotImplementedError

    def memoize(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Memoize a function with record and replay functionality.

        This is a decorator that attempts to replay cached results first.
        If a cache miss occurs, it records the result by executing the wrapped function.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.
            custom_result_decoder: Optional decoder factory for cached results.
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.memoize(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization with replay and record.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Create decorated versions using record and replay
            replay_fn = self.replay(
                custom_params_encoder,
                custom_result_decoder,
            )(fn)
            record_fn = self.record(
                custom_params_encoder,
                custom_result_encoder,
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Attempt to replay from cache, or record on cache miss.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result from cache (if hit) or from executing the function (if miss).
                """
                # Try to replay first
                try:
                    return replay_fn(*args, **kwargs)
                except KeyError:
                    # Cache miss - record the result
                    return record_fn(*args, **kwargs)

            return inner

        return wrapper


class Memoizer(_BaseMemoizer):
    """In-memory memoization interface for caching function results.

    This class provides methods for recording, retrieving, and managing
    cached function results in memory with custom encoding/decoding logic.

    Cache entries are stored as CacheEntry objects containing both the encoded
    parameters and the encoded result. This makes debugging easier since entries
    can be inspected with full context about what inputs produced each result.

    On memoizer destruction, the cache is automatically dumped to a shared JSON file
    in the cache directory for debugging and inspection purposes. Multiple Memoizer
    instances contribute to the same file additively.

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, or when you need custom encoding/decoding
    of results.
    """

    def __init__(self, sub_key: str | None = None) -> None:
        """Initialize the Memoizer instance with an in-memory cache.

        Args:
            sub_key: Optional key for organizing cache entries in the JSON dump.
                    If provided, cache entries are stored under cache_entries[sub_key].
                    If None, cache entries are merged directly into root cache_entries.
        """
        self._cache: implementations._InMemoryCacheImpl[CacheEntry] = (
            implementations._InMemoryCacheImpl()
        )
        # Optional sub_key for nested cache structure
        self._sub_key: str | None = sub_key
        # Track pending deferred recordings by cache key.
        # Uses WeakValueDictionary to prevent memory leaks: if the DeferredRecording
        # is no longer referenced elsewhere (e.g., the async computation was cancelled
        # or failed and garbage collected), the entry is automatically removed.
        self._pending_deferred: weakref.WeakValueDictionary[str, DeferredRecording] = (
            weakref.WeakValueDictionary()
        )
        # Register atexit handler to dump cache on program exit
        if config.IS_DUMP_MEMOIZER_CACHE_ENABLED():
            atexit.register(self._dump_to_disk)
        # Pre-populate cache from dump file if configured (with sub_key now set)
        self._maybe_prepopulate_from_dump()
        # Register with clear_on_fresh_cache for fresh_cache() integration
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        """Clear the in-memory cache.

        This method resets the in-memory cache to empty. It is called by
        the fresh_cache() context manager via clear_on_fresh_cache registration.
        """
        self._cache._memory.clear()

    def _finalize_deferred_recording(
        self,
        cache_key: str,
        encoded_params: object,
        encoded_result: _EncodedR,
    ) -> None:
        """Finalize a deferred recording by inserting the cache entry.

        This method is called when a DeferredRecording's finalize() method is invoked.
        It creates the cache entry and inserts it into the in-memory cache.

        Args:
            cache_key: The cache key for the entry.
            encoded_params: The encoded function parameters.
            encoded_result: The final encoded result to cache.
        """
        # Remove from pending tracking
        self._pending_deferred.pop(cache_key, None)
        # Insert into cache
        cache_entry = CacheEntry(
            encoded_params=encoded_params,
            encoded_result=encoded_result,
        )
        self._cache.insert(cache_key, cache_entry)

    @functools.cached_property
    def _shared_cache_filepath(self) -> Path:
        """Get the shared cache filepath for memoizer cache dumps.

        Returns:
            The path to the shared memoizer cache JSON file.
        """
        return Path(cache_dir()) / "memoizer_cache.json"

    @functools.cached_property
    def _shared_cache_lockfile(self) -> Path:
        """Get the lock file path for the shared memoizer cache.

        Returns:
            The path to the lock file for the shared cache.
        """
        return Path(cache_dir()) / "memoizer_cache.lock"

    def _read_dump_from_disk(self, filepath: Path | None = None) -> CacheDump | None:
        """Read a cache dump from disk.

        Attempts to read and parse a cache JSON file.

        Args:
            filepath: Path to the dump file to read. If None, uses the
                     shared cache filepath (self._shared_cache_filepath).

        Returns:
            The cache dump if the file exists and is valid JSON, None otherwise.
        """
        target_path = filepath if filepath is not None else self._shared_cache_filepath
        try:
            with open(target_path) as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None

    def _write_dump_to_disk(self, dump: CacheDump) -> None:
        """Write the cache dump to disk.

        Writes the provided dump to the shared cache JSON file and logs the result.

        Args:
            dump: The cache dump to write.
        """
        try:
            with open(self._shared_cache_filepath, "w") as f:
                json.dump(dump, f, indent=2)

            # Log the filepath
            if self._sub_key:
                logger.log(
                    logging.INFO,
                    "Memoizer cache (sub_key=%s) dumped to: %s",
                    self._sub_key,
                    self._shared_cache_filepath,
                )
            else:
                logger.log(
                    logging.INFO,
                    "Memoizer cache dumped to: %s",
                    self._shared_cache_filepath,
                )
        except Exception as e:
            # If dumping fails, just log it and don't crash the program
            logger.log(
                logging.WARNING,
                "Warning: Failed to dump memoizer cache: %s",
                e,
            )

    def _prepare_dump(self, existing_dump: CacheDump | None) -> CacheDump:
        """Prepare a cache dump from the current Memoizer state.

        Takes the existing dump (if any) and merges it with the current
        in-memory cache entries.

        Args:
            existing_dump: The existing dump to merge with, or None if starting fresh.

        Returns:
            The prepared cache dump ready to be written to disk.
        """
        # Start with existing data or empty structure
        if existing_dump is not None:
            dump = existing_dump
        else:
            dump: CacheDump = {"collections": {}, "cache_size": 0}

        # Ensure collections field exists
        if "collections" not in dump:
            dump["collections"] = {}

        # JSON serializes None keys as "null" string, so we need to handle both cases
        # When reading from JSON, the key will be "null" (string), not None
        lookup_key = "null" if self._sub_key is None else self._sub_key

        # Get existing entries for this sub_key to merge with
        existing_entries = dump["collections"].get(lookup_key, {})

        # Format cache entries as {"params": ..., "result": ...}
        formatted_cache: dict[str, CacheDumpEntry] = dict(existing_entries)
        for key, value in self._cache._memory.items():
            entry = value
            formatted_cache[key] = CacheDumpEntry(
                params=entry.encoded_params,
                result=entry.encoded_result,
            )

        # Store under sub_key in collections (use lookup_key to maintain JSON compatibility)
        dump["collections"][lookup_key] = formatted_cache

        # Calculate total cache size
        total_size = sum(len(collection) for collection in dump["collections"].values())
        dump["cache_size"] = total_size

        return dump

    def _dump_to_disk(self) -> None:
        """Dump the in-memory cache to a shared JSON file.

        This method is automatically called on program exit via atexit.
        It reads any existing cache data, merges it with this instance's cache,
        and writes the combined result back. Multiple Memoizer instances
        contribute to the same file additively.

        If self._sub_key is set and non-empty, cache entries are stored under
        cache_entries[sub_key]. Otherwise, they're merged directly into cache_entries.

        Cache entries are formatted as {"params": <encoded_params>, "result": <encoded_result>}
        for better human readability.

        The filepath where the cache was dumped is logged.
        """
        # Skip if cache is empty
        if not self._cache._memory:
            return

        # Ensure parent directory exists
        self._shared_cache_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Acquire file lock to ensure thread/process safety
        flock = FileLock(str(self._shared_cache_lockfile))
        with locks._acquire_flock_with_timeout(flock):
            existing_dump = self._read_dump_from_disk()
            dump = self._prepare_dump(existing_dump)
            self._write_dump_to_disk(dump)

    def _maybe_prepopulate_from_dump(self) -> None:
        """Pre-populate cache entries from a dump file if configured.

        Checks the CACHE_DUMP_FILE_PATH config option for a path to a JSON dump file
        produced by IS_DUMP_MEMOIZER_CACHE_ENABLED. If a valid path is provided and
        the file exists, loads cache entries from it into the in-memory cache.

        For Memoizer instances without a sub_key, loads entries from the root cache_entries.
        For Memoizer instances with a sub_key, loads entries from cache_entries[sub_key].

        This method is called during __init__ to pre-populate the cache.
        """
        dump_file_path = config.CACHE_DUMP_FILE_PATH()

        # Skip if no dump file configured
        if not dump_file_path:
            return

        # Read the dump file using the helper method
        dump = self._read_dump_from_disk(Path(dump_file_path))
        if dump is None:
            return

        # Extract entries to load from the dump
        entries_to_load = self._extract_entries_from_dump(dump)
        if not entries_to_load:
            return

        # Populate the cache
        self._populate_cache_from_entries(entries_to_load)

        # Log the result
        if self._sub_key:
            logger.log(
                logging.INFO,
                "Loaded %d cache entries from %s (sub_key=%s)",
                len(entries_to_load),
                dump_file_path,
                self._sub_key,
            )
        else:
            logger.log(
                logging.INFO,
                "Loaded %d cache entries from %s",
                len(entries_to_load),
                dump_file_path,
            )

    def _extract_entries_from_dump(self, dump: CacheDump) -> dict[str, CacheDumpEntry]:
        """Extract cache entries from a dump based on sub_key.

        Args:
            dump: The cache dump to extract entries from.

        Returns:
            Dictionary of cache entries to load.
        """
        collections = dump.get("collections", {})
        # JSON serializes None keys as "null" string, so we need to handle both cases
        # When reading from JSON, the key will be "null" (string), not None
        lookup_key = "null" if self._sub_key is None else self._sub_key
        return collections.get(lookup_key, {})

    def _populate_cache_from_entries(self, entries: dict[str, CacheDumpEntry]) -> None:
        """Populate the in-memory cache from dump entries.

        Args:
            entries: Dictionary of cache entries to load.
        """
        for key, entry in entries.items():
            cache_entry = CacheEntry(
                encoded_params=entry["params"],
                encoded_result=entry["result"],
            )
            self._cache.insert(key, cache_entry)

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  First receives the underlying function, then takes
                                  function parameters and returns an encoder function
                                  that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the function to compute the result
                result = fn(*args, **kwargs)

                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Encode params for human-readable dump
                if custom_params_encoder is not None:
                    encoded_params = custom_params_encoder(*args, **kwargs)
                else:
                    encoded_params = {
                        "args": args,
                        "kwargs": kwargs,
                    }

                # Encode the result if encoder is provided
                if custom_result_encoder is not None:
                    # Get the encoder function by calling the factory:
                    # 1. First call with fn to get params_to_encoder factory
                    # 2. Second call with params to get the encoder function
                    params_to_encoder = custom_result_encoder(fn)
                    encoder_fn = params_to_encoder(*args, **kwargs)
                    encoded_result = encoder_fn(result)
                else:
                    encoded_result = result

                # Check for deferred recording
                if isinstance(encoded_result, DeferredRecording):
                    # Track the pending deferred recording
                    self._pending_deferred[cache_key] = encoded_result
                    # Register the callback - handles race condition if encoded_result already completed
                    encoded_result.register_callback(
                        functools.partial(
                            self._finalize_deferred_recording,
                            cache_key,
                            encoded_params,
                        )
                    )
                    return result  # Return without caching

                # Store CacheEntry in cache
                cache_entry = CacheEntry(
                    encoded_params=encoded_params,
                    encoded_result=encoded_result,
                )
                self._cache.insert(cache_key, cache_entry)

                # Return the original result (not the encoded version)
                return result

            return inner

        return wrapper

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results instead of executing
        the function. Raises an exception if no cached result exists.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  First receives the underlying function, then takes
                                  function parameters and returns a decoder function
                                  that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.replay(
                custom_params_encoder=my_param_encoder,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to retrieve from cache.

            Args:
                fn: The function to wrap (not actually called).

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, always raise KeyError (cache miss)
            if not config.IS_CACHING_MODULE_ENABLED():

                @functools.wraps(fn)
                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Retrieve the cached result without calling the function.

                Args:
                    *args: Positional arguments to generate the cache key.
                    **kwargs: Keyword arguments to generate the cache key.

                Returns:
                    The cached result (decoded if decoder is provided).

                Raises:
                    KeyError: If no cached result exists for the given parameters.
                """
                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Check if result is cached
                cached_hit = self._cache.get(cache_key)
                if cached_hit is not None:
                    # Extract the cached value
                    cache_entry = cached_hit.value

                    # Decode and return the cached result
                    if custom_result_decoder is not None:
                        # Get the decoder function by calling the factory:
                        # 1. First call with fn to get params_to_decoder factory
                        # 2. Second call with params to get the decoder function
                        params_to_decoder = custom_result_decoder(fn)
                        decoder_fn = params_to_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cache_entry.encoded_result))
                    return cast(_R, cache_entry.encoded_result)

                # Check for pending deferred recording with interim result
                pending = self._pending_deferred.get(cache_key)
                if pending is not None:
                    interim = pending.get_interim_result()
                    if interim is not None:
                        return interim.value

                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper


class PersistentMemoizer(_BaseMemoizer):
    """Persistent memoization interface for caching function results to disk.

    This class provides methods for recording, retrieving, and managing
    cached function results using a two-level cache strategy:
    1. In-memory cache (fast, checked first) - via Memoizer instance
    2. On-disk cache (persistent, checked on memory miss)

    Results are persisted across process restarts.

    On program exit, the in-memory cache entries are automatically dumped to
    the shared JSON file. If sub_dir is non-empty, entries are stored under
    a nested structure based on the sub_dir. If sub_dir is empty, entries are
    merged directly into the root cache_entries.

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, custom result encoding and/or decoding,
    or when you need disk caching to persist results across program boundaries.
    """

    def __init__(self, sub_dir: PathLike[str] | None = None) -> None:
        """Initialize the PersistentMemoizer with two-level caching.

        Args:
            sub_dir: Optional subdirectory within the cache directory for
                    organizing cached results. Defaults to empty string if not specified.
                    If non-empty, cache entries will be stored under cache_entries[sub_dir].
                    If empty, cache entries are merged into root cache_entries.
        """
        # Use a Memoizer instance for in-memory caching
        self._memoizer: Memoizer = Memoizer(sub_key=str(sub_dir) if sub_dir else None)
        # Store on-disk cache as a separate attribute
        self._disk_cache: implementations._OnDiskCacheImpl = (
            implementations._OnDiskCacheImpl(sub_dir=sub_dir)
        )
        # Register with clear_on_fresh_cache for fresh_cache() integration
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        """Clear the on-disk cache.

        This method removes the on-disk cache directory. The in-memory cache
        is cleared automatically by the underlying Memoizer instance, which
        registers itself with clear_on_fresh_cache.

        This is called by the fresh_cache() context manager via clear_on_fresh_cache
        registration.
        """
        # Clear on-disk cache by removing the cache directory
        # Note: in-memory cache is cleared by the Memoizer's own cache_clear
        if self._disk_cache._cache_dir.exists():
            shutil.rmtree(self._disk_cache._cache_dir, ignore_errors=True)

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding to both caches.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic. Results are stored in both
        the in-memory cache and the on-disk cache.

        This method delegates to the underlying Memoizer for memory caching,
        then adds disk persistence. For deferred recordings, it registers
        an additional callback to persist to disk when the recording completes.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  First receives the underlying function, then takes
                                  function parameters and returns an encoder function
                                  that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Get the memory-cached version from the memoizer
            memory_record_fn = self._memoizer.record(
                custom_params_encoder, custom_result_encoder
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result in both caches.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the memory-cached version (which calls fn and caches in memory)
                result = memory_record_fn(*args, **kwargs)

                # Also store in disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Check if this was a deferred recording
                pending = self._memoizer._pending_deferred.get(cache_key)
                if pending is not None:
                    # Deferred recording - register our own callback for disk persistence
                    # By the time our callback runs, Memoizer's callback will have
                    # already inserted the cache entry, so we just read and persist it
                    pending.register_callback(
                        functools.partial(self._persist_to_disk, cache_key)
                    )
                    return result

                # Persist the cache entry to disk
                self._persist_to_disk(cache_key)

                return result

            return inner

        return wrapper

    def _persist_to_disk(
        self,
        cache_key: str,
        _callback_result: object = None,
    ) -> None:
        """Persist a cache entry to disk.

        This method handles disk persistence for both immediate and deferred recordings:
        - For immediate recordings: called directly after memory caching
        - For deferred recordings: registered as a callback that runs after the
          Memoizer's callback has inserted the entry into memory

        Args:
            cache_key: The cache key for the entry.
            _callback_result: Unused. When called as a callback from DeferredRecording,
                             this receives the encoded result, but we ignore it and
                             read the full CacheEntry from the Memoizer's cache instead.
        """
        # Always read from memory cache - Memoizer's callback has already inserted it
        cached_hit = self._memoizer._cache.get(cache_key)
        if not cached_hit:
            raise AssertionError("Cache entry must exist in memory cache")
        cache_entry = cached_hit.value

        # Store the full CacheEntry in disk cache for easier debugging
        pickled_entry: bytes = pickle.dumps(cache_entry)
        self._disk_cache.insert(cache_key, pickled_entry)

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results using a two-level
        cache strategy. It checks the in-memory cache first (fast), then
        falls back to the on-disk cache. If found on disk, the result is
        cached in memory for future access.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  First receives the underlying function, then takes
                                  function parameters and returns a decoder function
                                  that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.replay(
                custom_params_encoder=my_param_encoder,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to retrieve from cache.

            Args:
                fn: The function to wrap (not actually called).

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, always raise KeyError (cache miss)
            if not config.IS_CACHING_MODULE_ENABLED():

                @functools.wraps(fn)
                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            # Get the memory replay function
            memory_replay_fn = self._memoizer.replay(
                custom_params_encoder, custom_result_decoder
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Retrieve the cached result without calling the function.

                Checks memory cache first, then disk cache. Populates memory
                cache from disk on a disk hit.

                Args:
                    *args: Positional arguments to generate the cache key.
                    **kwargs: Keyword arguments to generate the cache key.

                Returns:
                    The cached result (decoded if decoder is provided).

                Raises:
                    KeyError: If no cached result exists for the given parameters.
                """
                # Try memory cache first via memoizer
                try:
                    return memory_replay_fn(*args, **kwargs)
                except KeyError:
                    pass  # Memory miss, check disk

                # Memory miss - check disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)
                disk_hit = self._disk_cache.get(cache_key)
                if disk_hit is not None:
                    # Disk cache hit - unpickle the CacheEntry
                    pickled_value = disk_hit.value
                    cache_entry = cast(CacheEntry, pickle.loads(pickled_value))

                    # Populate memory cache for future access
                    self._memoizer._cache.insert(cache_key, cache_entry)

                    # Decode and return
                    if custom_result_decoder is not None:
                        # Get the decoder function by calling the factory:
                        # 1. First call with fn to get params_to_decoder factory
                        # 2. Second call with params to get the decoder function
                        params_to_decoder = custom_result_decoder(fn)
                        decoder_fn = params_to_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cache_entry.encoded_result))
                    return cast(_R, cache_entry.encoded_result)

                # Complete miss
                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper
<|endoftext|>import os
from collections.abc import Callable
from functools import cache, partial
from typing import TypeVar

import torch
from torch._environment import is_fbcode


T = TypeVar("T")


def _is_force_disable_caches() -> bool:
    """Check if caching is force disabled via inductor config.

    This defers importing torch._inductor.config to avoid circular imports.

    Returns:
        True if force_disable_caches is set in inductor config, False otherwise.
    """
    from torch._inductor import config as inductor_config

    return inductor_config.force_disable_caches


@cache
def _env_var_val(env_var: str, default: T) -> str | T:
    """Get the value of an environment variable or return the default.

    Args:
        env_var: Environment variable name to check
        default: Default value to return if environment variable is not set

    Returns:
        The value from the environment variable as a string, or the default if not set
    """
    return os.environ.get(env_var, default)


@cache
def _env_var_config(env_var: str, default: bool) -> bool:
    env_val = _env_var_val(env_var, None)
    if env_val is not None:
        return env_val == "1"
    return default


@cache
def _versioned_config(
    jk_name: str,
    this_version: int,
    oss_default: bool,
    env_var_override: str | None = None,
) -> bool:
    """
    A versioned configuration utility that determines boolean settings based on:
    1. Environment variable override (highest priority)
    2. JustKnobs version comparison in fbcode environments
    3. OSS default fallback

    This function enables gradual rollouts of features in fbcode by comparing
    a local version against a JustKnobs-controlled remote version, while
    allowing environment variable overrides for testing and OSS defaults
    for non-fbcode environments.

    Args:
        jk_name: JustKnobs key name (e.g., "pytorch/inductor:feature_version")
        this_version: Local version number to compare against JustKnobs version
        oss_default: Default value to use in non-fbcode environments
        env_var_override: Optional environment variable name that, when set,
                         overrides all other logic

    Returns:
        bool: Configuration value determined by the priority order above
    """
    if (
        env_var_override
        and (env_var_value := os.environ.get(env_var_override)) is not None
    ):
        return env_var_value == "1"
    elif is_fbcode():
        # this method returns 0 on failure, which we should check for specifically.
        # in the case of JK failure, the safe bet is to simply disable the config
        jk_version: int = torch._utils_internal.justknobs_getval_int(jk_name)
        return (this_version >= jk_version) and (jk_version != 0)
    return oss_default


# toggles the entire caching module, but only when calling through the
# public facing interfaces. get/insert operations become no-ops in the sense
# that get will always miss and insert will never insert; record becomes a
# no-op in the sense that the function will always be called and the cache
# will never be accessed
_CACHING_MODULE_VERSION: int = 0
_CACHING_MODULE_VERSION_JK: str = "pytorch/inductor:caching_module_version"
_CACHING_MODULE_OSS_DEFAULT: bool = False
_CACHING_MODULE_ENV_VAR_OVERRIDE: str = "TORCHINDUCTOR_ENABLE_CACHING_MODULE"


def _is_caching_module_enabled_base() -> bool:
    """Base check for caching module enablement via versioned config."""
    return _versioned_config(
        _CACHING_MODULE_VERSION_JK,
        _CACHING_MODULE_VERSION,
        _CACHING_MODULE_OSS_DEFAULT,
        _CACHING_MODULE_ENV_VAR_OVERRIDE,
    )


def IS_CACHING_MODULE_ENABLED() -> bool:
    """Check if the caching module is enabled.

    Returns False if:
    - The versioned config disables it
    - force_disable_caches is set in inductor config

    Returns:
        True if caching module is enabled, False otherwise.
    """
    if not _is_caching_module_enabled_base():
        return False
    if _is_force_disable_caches():
        return False
    return True


# Controls whether the Memoizer dumps its cache to a JSON file on destruction.
# When enabled, the Memoizer will write its in-memory cache to a JSON file
# (memoizer_cache.json in the cache directory) on destruction. This dump file
# can later be used to pre-populate Memoizers via CACHE_DUMP_FILE_PATH.
#
# This is useful for:
# - Debugging and inspection of cached values
# - Creating cache snapshots that can be reused across runs
# - Pre-warming caches for subsequent executions
#
# Set via environment variable: TORCHINDUCTOR_DUMP_MEMOIZER_CACHE=1
_DUMP_MEMOIZER_CACHE_ENV_VAR: str = "TORCHINDUCTOR_DUMP_MEMOIZER_CACHE"
_DUMP_MEMOIZER_CACHE_DEFAULT: bool = False
IS_DUMP_MEMOIZER_CACHE_ENABLED: Callable[[], bool] = partial(
    _env_var_config,
    _DUMP_MEMOIZER_CACHE_ENV_VAR,
    _DUMP_MEMOIZER_CACHE_DEFAULT,
)


# Path to a cache dump file to pre-populate Memoizers on initialization.
# This should point to a JSON file produced by IS_DUMP_MEMOIZER_CACHE_ENABLED.
#
# When set, Memoizers will load cached entries from this file on initialization,
# allowing cache values to be reused across separate runs without recomputation.
# This is particularly useful for:
# - Pre-warming caches with known-good values
# - Reproducing behavior from a previous run
# - Sharing cached computations across different environments
#
# The dump file format is produced by the Memoizer's _dump_to_disk method when
# IS_DUMP_MEMOIZER_CACHE_ENABLED is set to true.
#
# Set via environment variable: TORCHINDUCTOR_CACHE_DUMP_FILE_PATH=/path/to/dump.json
_CACHE_DUMP_FILE_PATH_ENV_VAR: str = "TORCHINDUCTOR_CACHE_DUMP_FILE_PATH"
_CACHE_DUMP_FILE_PATH_DEFAULT: str | None = None
CACHE_DUMP_FILE_PATH: Callable[[], str | None] = partial(
    _env_var_val,
    _CACHE_DUMP_FILE_PATH_ENV_VAR,
    _CACHE_DUMP_FILE_PATH_DEFAULT,
)
<|endoftext|>from . import config, encoders, memoizers
from .context import IsolationSchema, SelectedCompileContext, SelectedRuntimeContext
from .exceptions import (
    CacheError,
    FileLockTimeoutError,
    KeyEncodingError,
    LockTimeoutError,
    SystemError,
    UserError,
    ValueDecodingError,
    ValueEncodingError,
)
from .interfaces import Memoizer, PersistentMemoizer


__all__ = [
    "CacheError",
    "FileLockTimeoutError",
    "IsolationSchema",
    "KeyEncodingError",
    "LockTimeoutError",
    "Memoizer",
    "PersistentMemoizer",
    "SelectedCompileContext",
    "SelectedRuntimeContext",
    "SystemError",
    "UserError",
    "ValueDecodingError",
    "ValueEncodingError",
    "encoders",
    "memoizers",
]
<|endoftext|>"""
Custom encoder functions

This module provides reusable encoder functions that convert function parameters
into JSON-serializable dictionaries for caching purposes.
"""

from typing_extensions import TypedDict

import torch
from torch import Tensor
from torch._inductor.pattern_matcher import Match
from torch._inductor.runtime.caching.utils import _encode_tensor, EncodedTensor


class ShouldPadEncodedParams(TypedDict):
    """TypedDict for encoded should_pad parameters."""

    mat1: EncodedTensor
    mat2: EncodedTensor
    op: str
    input: EncodedTensor | None
    mat1_exclude_padding_time: bool
    mat2_exclude_padding_time: bool
    tf32: bool


def should_pad_params_encoder(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> ShouldPadEncodedParams:
    """Encode parameters for _should_pad into a human-readable dict.

    This encoder extracts only the information needed for caching:
    - Tensor shape, stride, and dtype (not the actual data)
    - Whether padding time should be excluded for mat1 and mat2
    - The operation as a string

    Args:
        match: The pattern match object
        mat1: First matrix tensor
        mat2: Second matrix tensor
        op: The operation being performed
        input: Optional input tensor for addmm

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    # Import here to avoid circular dependency
    from torch._inductor.fx_passes.pad_mm import should_exclude_padding_time

    return ShouldPadEncodedParams(
        mat1=_encode_tensor(mat1),
        mat2=_encode_tensor(mat2),
        op=str(op),
        input=_encode_tensor(input) if input is not None else None,
        mat1_exclude_padding_time=should_exclude_padding_time(match, "mat1"),
        mat2_exclude_padding_time=should_exclude_padding_time(match, "mat2"),
        tf32=False
        if mat1.dtype != torch.float32
        else bool(
            torch.backends.cuda.matmul.fp32_precision == "tf32"
            or torch.backends.mkldnn.fp32_precision == "tf32"
        ),
    )
<|endoftext|>"""Cache implementation classes

This module provides concrete implementations of caching backends including
in-memory, on-disk, and remote caching strategies. Each implementation follows
the abstract _CacheImpl interface and provides thread-safe operations with
appropriate locking mechanisms.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BufferedReader, BufferedWriter
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Generic, TypeVar
from typing_extensions import override

from filelock import BaseFileLock, FileLock

from . import locks


# Type variable for cache value types
_V = TypeVar("_V")


@dataclass
class Hit(Generic[_V]):
    """Result wrapper for hits on cache get operations.

    Allows distinguishing between a cache miss and a cached None value.

    Attributes:
        value: The cached value. The type depends on the cache implementation:
              - InMemoryCache: object (any Python object)
              - OnDiskCache: bytes
              - RemoteCache: bytes
    """

    value: _V


class Miss:
    """Sentinel class representing a cache miss.

    Used to distinguish between a cached None value and a cache miss
    when None is a valid cached value.
    """


# Singleton instance for cache miss sentinel
miss = Miss()


class _CacheImpl(ABC, Generic[_V]):
    """Abstract base class for cache implementations.

    This class defines the interface that all cache implementations must follow.
    It provides thread-safe operations through a locking mechanism and supports
    both get and insert operations.

    Type Parameters:
        _V: The type of values stored in the cache. Each implementation specifies
           its own value type (e.g., object for in-memory, bytes for on-disk).
    """

    def __init__(self) -> None:
        """Initialize the cache implementation with a threading lock."""
        self._lock: Lock = Lock()

    @property
    def lock(self) -> locks._LockProtocol:
        """Get a context manager for acquiring the cache lock.

        Locking of the cache is not done by the implementation itself, but by the
        interface that uses it. The interface may want to hold the lock for longer
        than a single cache operation, for example when dealing with multiple
        cache implementations at once, so we leave that decision up to the interface.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the lock.

        Returns:
            A callable that returns a context manager for the lock.
        """

        def _lock_with_timeout(
            timeout: float | None = None,
        ) -> locks._LockContextManager:
            return locks._acquire_lock_with_timeout(self._lock, timeout)

        return _lock_with_timeout

    @abstractmethod
    def get(self, key: str) -> Hit[_V] | None:
        """Retrieve a value from the cache.

        Args:
            key: The key to look up in the cache (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value,
            or None on cache miss.
        """

    @abstractmethod
    def insert(self, key: str, value: _V) -> bool:
        """Insert a key-value pair into the cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key.

        Returns:
            True if the insertion was successful, False if not inserted.
        """


class _InMemoryCacheImpl(_CacheImpl[_V], Generic[_V]):
    """In-memory cache implementation using a dictionary.

    This implementation stores key-value pairs in a Python dictionary,
    with keys being pickled for consistent hashing. It provides fast
    access but is limited by available memory and process lifetime.

    Type Parameters:
        _V: The type of values stored in the cache.
    """

    def __init__(self) -> None:
        """Initialize the in-memory cache with an empty dictionary."""
        super().__init__()
        self._memory: dict[str, _V] = {}

    @override
    def get(self, key: str) -> Hit[_V] | None:
        """Retrieve a value from the in-memory cache.

        Args:
            key: The key to look up (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value,
            or None on cache miss.
        """
        if key in self._memory:
            return Hit(value=self._memory[key])
        return None

    @override
    def insert(self, key: str, value: _V) -> bool:
        """Insert a key-value pair into the in-memory cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key.

        Returns:
            True if the insertion was successful (key was new),
            False if not inserted (key already existed).
        """
        if key not in self._memory:
            self._memory[key] = value
            return True
        return False


class _OnDiskCacheImpl(_CacheImpl[bytes]):
    """On-disk cache implementation using file system storage.

    This implementation stores cached data as files on disk, with version
    headers to handle cache invalidation. It uses file locking to ensure
    thread safety across processes and provides persistent storage that
    survives process restarts.

    The value type is 'bytes' since data must be serialized to bytes for disk storage.

    Attributes:
        _version: _Version number for cache format compatibility.
        _version_header_length: Length of the version header in bytes.
    """

    _version: int = 0
    _version_header_length: int = 4

    def __init__(self, sub_dir: PathLike[str] | None = None) -> None:
        """Initialize the on-disk cache with a specified subdirectory.

        Args:
            sub_dir: Subdirectory name within the cache directory.
                    Defaults to empty string if not specified.
        """
        self._cache_dir: Path = self._base_dir / (sub_dir or "")
        self._flock: BaseFileLock = FileLock(str(self._cache_dir / "dir.lock"))

    @property
    def _base_dir(self) -> Path:
        """Get the base directory for cache storage.

        Returns:
            Path to the cache directory based on the default cache dir
            and the specified subdirectory.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache")

    def _fpath_from_key(self, key: str) -> Path:
        """Generate a file path from a cache key.

        Args:
            key: The cache key to convert to a file path (must be str).

        Returns:
            A Path object representing the file location for this key.
        """
        return self._cache_dir / key

    @classmethod
    def _version_header(cls) -> bytes:
        """Generate the version header bytes.

        Returns:
            A byte string representing the current cache version header.
        """
        return sha256(str(cls._version).encode()).digest()[: cls._version_header_length]

    def _version_header_matches(self, fp: BufferedReader) -> bool:
        """Check if the file's version header matches the current version.

        Args:
            fp: File pointer positioned at the start of the file.

        Returns:
            True if the version header matches, False otherwise.
        """
        return fp.read(self._version_header_length) == self._version_header()

    def _write_version_header(self, fp: BufferedWriter) -> None:
        """Write the version header to a file.

        Args:
            fp: File pointer where the version header should be written.
        """
        fp.write(self._version_header())

    @override
    @property
    def lock(self) -> locks._LockProtocol:
        """Get a context manager for acquiring the file lock.

        Uses file locking to ensure thread safety across processes.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the file lock.

        Returns:
            A callable that returns a context manager for the file lock.
        """

        def _lock_with_timeout(
            timeout: float | None = None,
        ) -> locks._LockContextManager:
            return locks._acquire_flock_with_timeout(self._flock, timeout)

        return _lock_with_timeout

    @override
    def get(self, key: str) -> Hit[bytes] | None:
        """Retrieve a value from the on-disk cache.

        Args:
            key: The key to look up in the cache (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value (bytes),
            or None on cache miss or version mismatch.
        """
        fpath: Path = self._fpath_from_key(key)

        if not fpath.is_file():
            return None

        pickled_value: bytes | None = None
        with open(fpath, "rb") as fp:
            if self._version_header_matches(fp):
                pickled_value = fp.read()

        if not pickled_value:
            # if pickled_value is still None, even though the file exists, then
            # we know that the version header did not match. in this case implementation
            # is up to preference, we choose to remove entries that do not match
            # the version header so that the key can be re-cached later with the correct
            # version header
            fpath.unlink()
            return None

        return Hit(value=pickled_value)

    @override
    def insert(self, key: str, value: bytes) -> bool:
        """Insert a key-value pair into the on-disk cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key (must be bytes).

        Returns:
            True if successfully inserted, False if the key already exists
            with a valid version.
        """
        fpath: Path = self._fpath_from_key(key)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        r_fp, w_fp, inserted = None, None, False
        try:
            w_fp = open(fpath, "xb")  # noqa: SIM115
        except FileExistsError:
            is_stale: bool = False
            with open(fpath, "rb") as r_fp:
                is_stale = not self._version_header_matches(r_fp)

            if is_stale:
                # same story as above, in this case the version header doesn't
                # match so we choose to remove the old entry so that the new
                # k/v pair can be cached
                fpath.unlink()
                w_fp = open(fpath, "xb")  # noqa: SIM115
            else:
                w_fp = None
        finally:
            if w_fp:
                try:
                    self._write_version_header(w_fp)
                    w_fp.write(value)
                    inserted = True
                finally:
                    w_fp.close()

        return inserted


try:
    from .fb.implementations import _RemoteCacheImpl
except ModuleNotFoundError:

    class _RemoteCacheImpl(_CacheImpl[bytes]):
        """Fallback remote cache implementation for non-Facebook environments.

        This is a no-op implementation that always raises NotImplementedError.
        The actual remote cache implementation is provided in the `.fb` module
        for Facebook-specific environments.

        The value type is 'bytes' for consistency with the FB implementation.

        Attributes:
            _version: _Version number for cache format compatibility.
            has_strong_consistency: Whether the remote cache provides strong
                                   consistency guarantees.
        """

        _version: int = 0
        has_strong_consistency: bool = False

        def __init__(self) -> None:
            """Initialize the fallback remote cache implementation.

            Note: We don't need to initialize any form of lock since this
            implementation provides a pseudo-lock context manager.
            """

        @override
        @property
        def lock(self) -> locks._LockProtocol:
            """Get a pseudo lock that does nothing.

            Most remote cache implementations don't have an ability to implement
            any form of locking, so we provide a no-op pseudo-lock for consistency
            with the interface.

            Args:
                timeout: Optional timeout in seconds (float). Ignored in this

            Returns:
                A callable that returns a no-op context manager.
            """

            @contextmanager
            def pseudo_lock(
                timeout: float | None = None,
            ) -> Generator[None, None, None]:
                yield

            return pseudo_lock

        @override
        def get(self, key: str) -> Hit[bytes] | None:
            """Raise NotImplementedError for remote cache get operations.

            Args:
                key: The key to look up (must be str, ignored).

            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError

        @override
        def insert(self, key: str, value: bytes) -> bool:
            """Raise NotImplementedError for remote cache insert operations.

            Args:
                key: The key to insert (must be str, ignored).
                value: The value to insert (must be bytes, ignored).

            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError
<|endoftext|>"""Exception classes for PyTorch Inductor runtime caching.

This module defines a hierarchy of exceptions used throughout the caching system.
All custom exceptions inherit from CacheError, with UserError serving as a base
for user-facing errors that also inherit from TypeError for compatibility.
"""

from threading import Lock

from filelock import BaseFileLock


class CacheError(Exception):
    """Base class for all caching-related errors.

    This is the root exception class for all custom exceptions raised by the caching
    module, providing a common interface for error handling and logging.
    """


class SystemError(CacheError, RuntimeError):
    """Base class for system-level caching errors.

    This class represents errors that occur during cache operations, such as
    storage or retrieval failures. It inherits from RuntimeError to indicate
    that the error is not caused by user input.
    """


class LockTimeoutError(SystemError):
    """Error raised when a lock operation times out.

    This exception is raised when a lock operation exceeds the specified timeout
    limit, indicating that the lock could not be acquired within the allotted time.
    """

    def __init__(self, lock: Lock, timeout: float) -> None:
        """Initialize the lock timeout error with detailed lock information.

        Args:
            lock: The lock object that timed out.
            timeout: The timeout limit that was exceeded.
        """
        super().__init__(f"Failed to acquire lock {lock} within {timeout} seconds.")


class FileLockTimeoutError(SystemError):
    """Error raised when a file lock operation times out.

    This exception is raised when a file lock operation exceeds the specified timeout
    limit, indicating that the lock could not be acquired within the allotted time.
    """

    def __init__(self, flock: BaseFileLock, timeout: float) -> None:
        """Initialize the file lock timeout error with detailed lock information.

        Args:
            flock: The file lock object that timed out.
            timeout: The timeout limit that was exceeded.
        """
        super().__init__(
            f"Failed to acquire file lock {flock} within {timeout} seconds."
        )


class UserError(CacheError, TypeError):
    """Base class for user-facing cache errors that also inherit from TypeError.

    This class combines CacheError with TypeError to provide compatibility
    with existing exception handling patterns while maintaining the cache
    error hierarchy. All user-facing cache errors should inherit from this class.
    """


class KeyEncodingError(UserError):
    """Base class for errors that occur during cache key encoding operations.

    Raised when cache keys cannot be properly encoded for storage or transmission.
    This includes serialization, hashing, or other encoding-related failures.
    """


class ValueEncodingError(UserError):
    """Base class for errors that occur during cache value encoding operations.

    Raised when cache values cannot be properly encoded for storage or transmission.
    This includes serialization, compression, or other encoding-related failures.
    """


class ValueDecodingError(UserError):
    """Base class for errors that occur during cache value decoding operations.

    Raised when cached values cannot be properly decoded during retrieval.
    This includes deserialization, decompression, or other decoding-related failures.
    """
<|endoftext|>"""
Heuristic registry system for PyTorch Inductor.

Provides centralized registration for both:
- Template heuristics (compile-time, keyed by template_name/device_type/op_name)
- Codegen heuristics (runtime, keyed by name/device_type)

Both share one underlying registry dict and cascading fallback lookup.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from .template.base import TemplateConfigHeuristics


_HEURISTIC_REGISTRY: dict[tuple[str | None, ...], Any] = {}
# Alias so tests can snapshot/restore the shared registry via the old name.
# This intentionally covers both template and codegen entries.
_TEMPLATE_HEURISTIC_REGISTRY = _HEURISTIC_REGISTRY
_HEURISTIC_CACHE: dict[tuple[str | None, ...], Any] = {}

log = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Shared internals
# -----------------------------------------------------------------


def _register(
    key: tuple[str | None, ...],
    target: Any,
) -> None:
    _HEURISTIC_REGISTRY[key] = target


def _lookup(name: str, device_type: str, op_name: str | None) -> Any | None:
    """Cascading fallback lookup: (name, device, op) -> (name, None, op) -> (name, device, None) -> (name, None, None)."""
    keys = [
        (name, device_type, op_name),
        (name, None, op_name),
        (name, device_type, None),
        (name, None, None),
    ]
    for key in keys:
        if key in _HEURISTIC_REGISTRY:
            return _HEURISTIC_REGISTRY[key]
    return None


# -----------------------------------------------------------------
# Template heuristic API
# -----------------------------------------------------------------


def register_template_heuristic(
    template_name: str,
    device_type: str | None,
    register: bool = True,
    op_name: str | None = None,
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu").
            None means applicable to all device types.
        register: Whether to register this heuristic.
        op_name: Operator name for per-op specialization.

    Example:
        @register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
        class CUDAMMTemplateConfigHeuristic(CUDAConfigHeuristic):
            pass
    """

    def decorator(
        cls: type[TemplateConfigHeuristics],
    ) -> type[TemplateConfigHeuristics]:
        if register:
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            _register(key, cls)
            log.info(
                "Registered template heuristic: %s for template_name=%s, device_type=%s, op_name=%s",
                cls.__name__,
                template_name,
                device_type,
                op_name,
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    """
    Retrieve a template heuristic instance for the given template and device type.

    Returns a cached instance. Falls back to TemplateConfigHeuristics() if not found.
    """
    cache_key: tuple[str | None, ...] = (template_name, device_type, op_name)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(template_name, device_type, op_name)

    if heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, op_name=%s. "
            "Available: %s. Using fallback.",
            template_name,
            device_type,
            op_name,
            list(_HEURISTIC_REGISTRY.keys()),
        )
        return _Base()

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


def get_registered_heuristic_class(
    template_name: str, device_type: str, op_name: str
) -> None | type[TemplateConfigHeuristics]:
    """Get the registered class (not instance) for the given template/device/op."""
    return _lookup(template_name, device_type, op_name)


# -----------------------------------------------------------------
# Codegen heuristic API
# -----------------------------------------------------------------


class CodegenConfigHeuristics:
    """Base class for codegen heuristics (pointwise, reduction, etc.)."""

    def get_configs(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError


def register_codegen_heuristic(
    name: str,
    device_type: str | None = None,
    register: bool = True,
) -> Any:
    """
    Decorator to register a codegen heuristic class.

    Args:
        name: Heuristic name (e.g., "pointwise", "reduction").
        device_type: Device type ("hip", "xpu", "cuda").
            None means default/fallback for all devices.
        register: Whether to actually register.

    Example:
        @register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
        class ROCmPointwiseHeuristic(CodegenConfigHeuristics):
            ...
    """

    def decorator(cls: type[CodegenConfigHeuristics]) -> type[CodegenConfigHeuristics]:
        if register:
            key: tuple[str | None, ...] = (name, device_type, None)
            _register(key, cls)
            log.info(
                "Registered codegen heuristic: %s for name=%s, device_type=%s",
                cls.__name__,
                name,
                device_type,
            )
        return cls

    return decorator


def get_codegen_heuristic(name: str, device_type: str) -> CodegenConfigHeuristics:
    """
    Retrieve a codegen heuristic instance for the given name and device type.

    Returns a cached instance. Uses cascading fallback:
        (name, device_type, None) -> (name, None, None)
    """
    cache_key: tuple[str | None, ...] = (name, device_type, None)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        # Lazily import codegen heuristics to trigger registration
        import torch._inductor.heuristics.triton_codegen  # noqa: F401

        heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        raise ValueError(
            f"No codegen heuristic found - name={name}, device_type={device_type}. "
            f"Available: {list(_HEURISTIC_REGISTRY.keys())}"
        )

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


# -----------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------


def clear_registry() -> None:
    """Clear all registered heuristics. Primarily for testing."""
    _HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
    override_heuristic_class: type[TemplateConfigHeuristics] | None = None,
) -> Iterator[None]:
    """
    Context manager to temporarily override template heuristics.

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
        override_heuristic_class: Heuristic class to use for the override.
    """
    if override_heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        override_heuristic_class = _Base
    original_entries: dict[tuple[str | None, ...], Any] = {}
    new_keys: list[tuple[str | None, ...]] = []
    # Clears the shared cache (both template and codegen instances).
    # Codegen instances are stateless so re-creation is cheap.
    _HEURISTIC_CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            if op_name is None:
                raise AssertionError("op_name must not be None")
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            if key in _HEURISTIC_REGISTRY:
                original_entries[key] = _HEURISTIC_REGISTRY[key]
            _HEURISTIC_REGISTRY[key] = override_heuristic_class
            new_keys.append(key)
        yield
    finally:
        for key in new_keys:
            _HEURISTIC_REGISTRY.pop(key, None)
            if key in original_entries:
                _HEURISTIC_REGISTRY[key] = original_entries[key]
        # Same shared-cache clear on exit; see entry comment above.
        _HEURISTIC_CACHE.clear()
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor import config as inductor_config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ...kernel.mm import (
    aten__fp8_mm,
    aten__int_mm,
    aten_addmm,
    aten_bias_addmm,
    aten_mm,
    aten_mm_dtype,
)
from ...kernel.mm_plus_mm import aten_mm_plus_mm
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    from ...kernel_inputs import KernelInputs


# These are all labeled as device type None to indicate that they
# are valid for all device types
@register_template_heuristic(aten_mm.uid, None)
@register_template_heuristic(aten_mm_dtype.uid, "cuda")
@register_template_heuristic(aten_mm_dtype.uid, "xpu")
@register_template_heuristic(aten__fp8_mm.uid, None)
@register_template_heuristic(aten__int_mm.uid, None)
@register_template_heuristic(aten_bmm.uid, None)
@register_template_heuristic(aten_mm_plus_mm.uid, None)
# bmm dtype is only valid on cuda
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
@register_template_heuristic(aten_bmm_dtype.uid, "xpu")
class ATenConfigHeuristics(TemplateConfigHeuristics):
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        yield dict()


# None here indicates that this is valid for all device types on that op
# Note (None, op) takes precedence over (device_type, None)
@register_template_heuristic(aten_addmm.uid, None, op_name="addmm")
@register_template_heuristic(aten_baddbmm.uid, None, op_name="baddbmm")
class ATenAddMMConfigHeuristics(ATenConfigHeuristics):
    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "alpha": alpha,
            "beta": beta,
        }


@register_template_heuristic(aten_bias_addmm.uid, None, op_name="addmm")
class ATenBiasAddMMConfigHeuristics(
    ATenAddMMConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics
):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if not (
            len(bias.get_size()) == 2
            and bias.get_stride()[0] == 0
            and inductor_config.triton.autotune_cublasLt
        ):
            raise AssertionError(
                "Expected 2D bias with stride[0]==0 and autotune_cublasLt enabled"
            )
        yield from super()._get_template_configs_impl(kernel_inputs, op_name)
<|endoftext|>from dataclasses import dataclass
from enum import auto, Enum
from itertools import product

import torch._inductor.config as config


class TensorMapUpdateMode(Enum):
    """Enum mirroring cutlass.utils.TensorMapUpdateMode to decouple this file from a cutlass dependency."""

    SMEM = auto()
    GMEM = auto()


@dataclass(frozen=True)
class CuTeGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the exhaustive configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    For information regarding valid config sets, see:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py
    """

    # Tile_n is always the same regardless of 2cta
    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # Valid clusters
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    for use_2cta, cluster_set, tile_m_range in [
        (False, clusters_no_2cta, [64, 128]),
        (True, clusters_2cta, [128, 256]),
    ]:
        for tensormap_update_mode, tile_m, tile_n, (cluster_m, cluster_n) in product(
            [TensorMapUpdateMode.SMEM, TensorMapUpdateMode.GMEM],
            tile_m_range,
            tile_n_vals,
            cluster_set,
        ):
            configs.append(
                CuTeGemmConfig(
                    tile_m,
                    tile_n,
                    cluster_m,
                    cluster_n,
                    USE_2_CTA=use_2cta,
                    TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                )
            )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the default configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    """

    config_tuples = [
        (128, 256, 2, 1, False, TensorMapUpdateMode.SMEM),
        (256, 160, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.GMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
        (256, 256, 8, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 192, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.SMEM),
        (128, 96, 1, 2, False, TensorMapUpdateMode.SMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.SMEM),
        (64, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.GMEM),
        (128, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 160, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
    ]

    return [CuTeGemmConfig(*args) for args in config_tuples]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.

    Note: CuTeDSL autotuning is still experimental — enabling it may trigger kernel launch failures
    or unstable results. By default, autotuning is disabled and we return only
    a single baseline config.
    """
    if (
        config.cutedsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import use_contiguous
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip contiguous mm on not cuda"""


@register_template_heuristic(
    mm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="mm",
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="addmm",
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return
        mat2 = kernel_inputs.mat1mat2()[1]
        if mat2.get_layout().is_contiguous():
            # no need for contiguous decomposition
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        if not use_contiguous(m, n, k):
            return
        yield {}
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import fields
from functools import cache
from typing import Any, TYPE_CHECKING, TypeAlias

import sympy

import torch
import torch._vendor.quack.gemm_config as quack_gemm_config
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Sequence

GemmConfigKey: TypeAlias = tuple[tuple[str, Any], ...]


def gemm_config_key(config: quack_gemm_config.GemmConfig) -> GemmConfigKey:
    """Project a QuACK GEMM config using the dataclass schema as the contract."""
    return tuple(
        (field.name, getattr(config, field.name))
        for field in fields(quack_gemm_config.GemmConfig)
    )


def gemm_config_from_key(config_key: GemmConfigKey) -> quack_gemm_config.GemmConfig:
    """Reconstruct a QuACK GEMM config from its generated-code cache key."""
    return quack_gemm_config.GemmConfig(**dict(config_key))


def explicit_gemm_configs_for_device(
    config: dict[str, Any], device: torch.device
) -> tuple[quack_gemm_config.GemmConfig, ...]:
    """Return device configs matching every explicitly pinned field.

    Exact type matching prevents bool/int aliases from selecting a different key.
    """
    field_names = tuple(field.name for field in fields(quack_gemm_config.GemmConfig))
    unexpected = [name for name in config if name not in field_names]
    if unexpected:
        raise NotImplementedError(
            "FlexGEMM explicit QUACK config contains unexpected GemmConfig fields: "
            f"{unexpected}"
        )

    candidates = candidate_gemm_configs_for_device(device)
    expected_device_capacity = candidates[0].device_capacity
    requested_device_capacity = config.get("device_capacity")
    if (
        type(requested_device_capacity) is type(expected_device_capacity)
        and requested_device_capacity != expected_device_capacity
    ):
        raise NotImplementedError(
            f"FlexGEMM explicit QUACK config targets SM{requested_device_capacity}0, "
            f"but {device} uses SM{expected_device_capacity}0 configs"
        )
    matches = tuple(
        candidate
        for candidate in candidates
        if all(
            type(value) is type(getattr(candidate, name))
            and value == getattr(candidate, name)
            for name, value in config.items()
        )
    )
    if matches:
        return matches
    raise NotImplementedError(
        f"FlexGEMM explicit QUACK config constraints are not supported on {device}: "
        f"{config}"
    )


@cache
def dense_gemm_config_priority_keys() -> tuple[GemmConfigKey, ...]:
    """Return the measured dense FlexGEMM QuACK preference order."""
    configs = (
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=224,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=160,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
    )
    return tuple(gemm_config_key(config) for config in configs)


def candidate_gemm_configs_for_device(device: torch.device):
    """Return all device-compatible QuACK configs before shape-specific ranking."""
    device_capacity = torch.cuda.get_device_capability(device)[0]
    if device_capacity == 11:
        device_capacity = 10
    priority_map = {
        key: priority for priority, key in enumerate(dense_gemm_config_priority_keys())
    }
    configs = sorted(
        (
            config
            for config in quack_gemm_config.get_all_configs()
            if config.device_capacity == device_capacity and not config.use_tma_gather
        ),
        key=lambda config: (
            priority_map.get(gemm_config_key(config), len(priority_map)),
            config.tile_m,
            config.tile_n,
            config.cluster_m,
            config.cluster_n,
            int(config.is_dynamic_persistent),
        ),
    )
    if not configs:
        raise RuntimeError(
            f"FlexGEMM found no QuACK configs for CUDA device capability "
            f"SM{device_capacity}0"
        )
    return configs


def default_gemm_config_key(
    device: torch.device,
    m,
    n,
    configs: Sequence[quack_gemm_config.GemmConfig] | None = None,
) -> GemmConfigKey:
    """Return the untuned default QuACK config key for generated code."""
    configs = candidate_gemm_configs_for_device(device) if configs is None else configs
    config_keys = OrderedSet([gemm_config_key(config) for config in configs])
    default_key, skinny_key, large_rect_key, large_key = (
        dense_gemm_config_priority_keys()[:4]
    )

    from torch._inductor.virtualized import V

    guard_or_false = V.graph.sizevars.guard_or_false
    if guard_or_false(sympy.Le(m, n)):
        min_dim, max_dim = m, n
    elif guard_or_false(sympy.Lt(n, m)):
        min_dim, max_dim = n, m
    else:
        return (
            default_key if default_key in config_keys else gemm_config_key(configs[0])
        )

    if guard_or_false(sympy.Lt(min_dim, 512)):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(sympy.And(sympy.Eq(min_dim, 1024), sympy.Eq(max_dim, 1024))):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(
        sympy.And(
            sympy.Ge(max_dim, 4096), sympy.Ge(min_dim, 768), sympy.Lt(min_dim, 1024)
        )
    ):
        preferred_keys = (large_key, default_key)
    elif guard_or_false(sympy.And(sympy.Ge(max_dim, 4096), sympy.Eq(min_dim, 1024))):
        preferred_keys = (large_rect_key, default_key)
    elif guard_or_false(sympy.Ge(min_dim, 2048)):
        preferred_keys = (large_key, default_key)
    else:
        preferred_keys = (default_key,)

    for key in preferred_keys:
        if key in config_keys:
            return key
    return gemm_config_key(configs[0])
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from ... import config as inductor_config
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs


class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs) -> bool:
        """
        simple base override for GEMM family templates that run only in max-autotune
        """
        return inductor_config.max_autotune or inductor_config.max_autotune_gemm
<|endoftext|>"""
Template heuristics for pre-written template kernels (mm, conv, flex_attention...).

These heuristics are used at compile time to generate autotuning configs
for template-based kernels (Triton, ATen, CuTe DSL, etc.).
"""

from __future__ import annotations

# expose the entry function
from torch._inductor.heuristics.registry import get_template_heuristic

# NOTE: add new template heuristics here, so they get imported and registered
from . import (
    aten,
    base,
    contiguous_mm,
    decompose_k,
    nv_universal_gemm,
    registry,
    tlx,
    triton,
)


__all__ = [
    "get_template_heuristic",
]
<|endoftext|>from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import os
from functools import partial
from threading import Lock
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.heuristics.registry import register_template_heuristic
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Min, Mod
from torch.utils._triton import has_triton_stable_tma_api

from ... import config
from ...autows_utils import meta_ws_enabled
from ...kernel.bmm import bmm_template
from ...kernel.mm import (
    blackwell_ws_persistent_device_tma_mm_template,
    get_scaling_options,
    get_tile_size,
    mm_template,
    persistent_mm_template,
    persistent_tma_mm_template,
    scaled_mm_device_tma_epilogue_scaling_template,
    scaled_mm_device_tma_main_loop_scaling_template,
)
from ...kernel.mm_plus_mm import mm_plus_mm_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import (
    get_backend_num_stages,
    get_default_kpack,
    get_num_sms,
    get_tma_workspace_arg,
    TMA_DESCRIPTOR_SIZE,
    triton_type,
    using_b200,
    using_rocm_rdna3,
)
from ...virtualized import V
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .triton_addmm import AddMMConfigMixin


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from triton import Config as TritonConfig

else:
    from torch._inductor.runtime.triton_compat import Config as TritonConfig
log = logging.getLogger(__name__)


def _origami_enabled() -> bool:
    """Check if origami GEMM optimization is enabled."""
    return config.rocm.origami


USE_META_WS = meta_ws_enabled()


def _use_template_autows() -> bool:
    """Whether to expand the Blackwell GEMM search space with Meta Triton autoWS
    configs: opt-in flag plus an enabled meta-WS Triton build."""
    return config.triton.enable_template_autows and USE_META_WS


# Check if running on ROCm
IS_ROCM = torch.version.hip is not None


# rocm-origami pip pkg is only available on ROCm builds and is only used when
# both max_autotune and config.rocm.origami are enabled (env-var driven, set once
# at config import). Cache the import here so the hot path never pays an exception
# and CUDA/CPU/origami-disabled processes never attempt the import.
if IS_ROCM and config.max_autotune and config.rocm.origami:
    try:
        import origami  # type: ignore[import-not-found]
    except ImportError:
        origami = None
        log.info(
            "rocm-origami not installed; ROCm origami GEMM selection disabled. "
            "Install via pip install rocm-origami to enable."
        )
else:
    origami = None


# TODO(rocm-origami): replace these wrappers with public accessors when the
# rocm-origami package exposes them. Centralizing the private-attribute access
# here keeps the noqa suppressions localized and makes the migration a one-line
# change per helper instead of ~10 scattered call sites.
def _origami_problem(selector):  # type: ignore[no-untyped-def]
    return selector._problem


def _origami_hardware(selector):  # type: ignore[no-untyped-def]
    return selector._hardware


def _origami_configs(selector):  # type: ignore[no-untyped-def]
    return selector._configs


# Gemm Configs
@dataclasses.dataclass
class BaseConfig:
    """
    Base Gemm configuration used for most backends (CPU, CUDA)
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: int | None = dataclasses.field(kw_only=True, default=None)


@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """
    Gemm configuration used for most backends (CPU, CUDA)
    """

    group_m: int = dataclasses.field(kw_only=True, default=8)


ConvConfig = BaseConfig


@dataclasses.dataclass
class DepthwiseConvConfig:
    """
    Configuration for depthwise conv1d Triton template.
    Uses BLOCK_N x BLOCK_L x BLOCK_C tiling (channels-last NLC layout).
    Matches the hand-written NLC kernel from depthwise_conv1d_benchmark.py.
    """

    block_n: int
    block_l: int
    block_c: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class BlackwellGPUGemmConfig(GemmConfig):
    """
    Gemm configuration used for templates with features explicitly
    targeting Nvidia Blackwell GPUs
    """

    # epilogue_subtile must be a power of 2 (1 means no subtiling)
    epilogue_subtile: int = dataclasses.field(kw_only=True, default=1)
    warp_specialize: bool = dataclasses.field(kw_only=True, default=True)
    flatten: bool = dataclasses.field(kw_only=True, default=True)
    # Meta Triton autoWS knobs (meta-WS builds only)
    use_meta_ws: bool = dataclasses.field(kw_only=True, default=False)
    data_partition_factor: int = dataclasses.field(kw_only=True, default=1)
    separate_epilogue_store: bool = dataclasses.field(kw_only=True, default=False)


# FlexAttention Configs
@dataclasses.dataclass
class FlexConfig:
    """
    Base Config class for flex attention
    - FlexAttn forward and backward will use this. For flex decoding,
      please use FlexDecodingConfig.

    NOTE:
    For flex_attn bwd block_m and block_n are reused for block_m1, block_m2, block_n1, block_n2

    """

    block_m: int
    block_n: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class FlexBwDConfig:
    """
    Base Config class for flex attention backward
    - FlexAttn backward will use this.

    Note: flex bwd configs

    Kernel Constraints:
      * BLOCK_N1 % BLOCK_M1 == 0
      * BLOCK_M2 % BLOCK_N2 == 0

    Pattern 1 - Symmetric Pairing (M, N, N, M):
    - Used in autotune configs
    - block_m1=M, block_n1=N, block_m2=N, block_n2=M
    - Only requires checking BLOCK_N % BLOCK_M == 0
    - Second constraint (BLOCK_M2 % BLOCK_N2) automatically satisfied

    Pattern 2 - Independent Parameters (M1, N1, M2, N2):
    - Used in exhaustive search for maximum flexibility
    - All four parameters can be set independently
    - Requires checking both constraints

    """

    block_m1: int
    block_n1: int
    block_m2: int
    block_n2: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class FlexDecodeConfig:
    """
    Config class for flex decoding
    """

    block_n: int
    num_stages: int
    num_warps: int


# ROCm classes
@dataclasses.dataclass
class ROCmGemmConfig(GemmConfig):
    """
    ROCm subclass for GEMMs, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 16
    waves_per_eu: int = 0
    kpack: int = 1


@dataclasses.dataclass
class ROCmConvConfig(ConvConfig):
    """
    ROCm subclass for Conv, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 16
    waves_per_eu: int = 0
    kpack: int = 1


@dataclasses.dataclass
class ROCmFlexConfig(FlexConfig):
    """
    ROCm subclass for FlexAttn, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 1


@dataclasses.dataclass
class ROCmFlexBwDConfig(FlexBwDConfig):
    """
    ROCm subclass for FlexAttn backward, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 1


@dataclasses.dataclass
class ROCmFlexDecodeConfig(FlexDecodeConfig):
    """
    ROCm subclass for FlexDecode, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 1


class BaseHeuristicSingleton(type):
    """
    Thread-safe implementation of singleton to be used in the config heuristic subclasses
    to ensure heavy __init__ calls are not repeatedly run
    """

    _instances: dict[type[Any], Any] = {}
    _lock: Lock = Lock()

    def __call__(
        cls: BaseHeuristicSingleton, *args: Any, **kwargs: Any
    ) -> BaseConfigHeuristic:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__()
                cls._instances[cls] = instance
            return cls._instances[cls]


class BaseConfigHeuristic(metaclass=BaseHeuristicSingleton):
    """
    Base class for mm_configs, device specific triton kernels config inherit from here
    """

    def __init__(self) -> None:
        # Whether the heuristic is used for int8. Use this when the heuristic is int8 exclusive
        # but prefer the preprocess_mm_configs argument when it's used for both
        self.has_int8_tensor: bool = False
        # Whether to scale configs at all
        # TODO(coconutruben): remove this once mm_plus_mm and tests support scaling
        self.should_scale_configs: bool = True
        # List of dictionaries to store the kernel configs. Configs that evaluate to true
        # will be utilised on the target platform. The configs are as follows:
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
        self.mm_configs: list[BaseConfig] = [
            GemmConfig(32, 32, 16, 1, 2),
            GemmConfig(32, 32, 128, 2, 4),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(64, 32, 128, 5, 4),
            GemmConfig(64, 64, 16, 2, 4),
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(64, 64, 128, 5, 4),
            GemmConfig(64, 128, 32, 3, 4),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(64, 128, 64, 3, 4),
            GemmConfig(64, 128, 128, 4, 4),
            GemmConfig(128, 64, 32, 3, 4),
            GemmConfig(128, 64, 32, 4, 8),
            GemmConfig(128, 128, 32, 2, 8),
            GemmConfig(128, 128, 32, 3, 4),
            GemmConfig(128, 128, 64, 3, 4),
            GemmConfig(128, 128, 64, 5, 8),
            GemmConfig(128, 128, 128, 4, 8),
            # 128x256x64 NS=4: larger N-tile that wins on Hopper-class matmul
            GemmConfig(128, 256, 64, 4, 8),
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs: list[BaseConfig] = [
            GemmConfig(
                BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, group_m=group_m
            )
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, 2, 3, 4, 5]
            for num_warps in [2, 4, 8]
            for group_m in [8]
        ]

        # these are only used in tuned_mm when AutoHeuristic is enabled
        # the idea is that when AutoHeuristic collects data to learn a heuristic, more configs are autotuned
        # when the learned heuristic is used, the learned heuristic reduces the number of configs down to 10
        # which saves compilation time (since less configs are autotuned) and potentially increase performance
        # because the learned heuristic might predict a config that is not part mm_configs
        self.extra_mm_configs: list[BaseConfig] = [
            GemmConfig(16, 32, 16, 3, 2),
            GemmConfig(16, 32, 32, 4, 2),
            GemmConfig(16, 32, 32, 5, 2),
            GemmConfig(64, 64, 128, 3, 4),
            GemmConfig(128, 64, 32, 2, 2),
            GemmConfig(128, 64, 64, 3, 8),
            GemmConfig(128, 64, 128, 4, 8),
            GemmConfig(128, 128, 32, 4, 4),
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 64, 5, 4),
        ]

        self.int8_mm_configs: list[BaseConfig] = [
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 128, 32, 3, 4),
            GemmConfig(128, 64, 32, 3, 4),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(128, 64, 32, 4, 8),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(128, 128, 32, 2, 8),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(128, 256, 128, 3, 8),
            GemmConfig(256, 128, 128, 3, 8),
        ]

        self.mixed_mm_configs: list[BaseConfig] = [
            GemmConfig(16, 128, 256, 3, 4),
            GemmConfig(16, 128, 256, 5, 8),
        ]

        self.persistent_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 256, 64, 3, 8),
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 128, 3, 8),
            GemmConfig(128, 128, 128, 3, 4),
            GemmConfig(128, 128, 64, 4, 8),
            GemmConfig(128, 128, 64, 5, 8),
            GemmConfig(256, 128, 64, 4, 8),
            GemmConfig(128, 128, 64, 5, 4),
        ]

        self.blackwell_persistent_mm_configs: list[BaseConfig] = [
            BlackwellGPUGemmConfig(
                128,
                256,
                64,
                4,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                256,
                128,
                2,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                256,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                4,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            # Include no-subtiling. Always required for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                8,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            # Include subtile=4. Always required for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                4,
                8,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                4,
                8,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
        ]

        self.blackwell_persistent_addmm_configs: list[BaseConfig] = [
            # Include each subtiling factor for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
        ]

        self.scaled_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 256, 32, 3, 8),
            GemmConfig(256, 128, 32, 3, 8),
            GemmConfig(256, 64, 32, 4, 4),
            GemmConfig(64, 256, 32, 4, 4),
            GemmConfig(128, 128, 32, 4, 4),
            GemmConfig(128, 64, 32, 4, 4),
            GemmConfig(64, 128, 32, 4, 4),
            GemmConfig(128, 32, 32, 4, 4),
            GemmConfig(64, 32, 32, 5, 2),
            GemmConfig(256, 128, 128, 3, 8),
            GemmConfig(256, 64, 128, 4, 4),
            GemmConfig(64, 256, 128, 4, 4),
            GemmConfig(128, 128, 128, 4, 4),
            GemmConfig(128, 64, 64, 4, 4),
            GemmConfig(64, 128, 64, 4, 4),
            GemmConfig(128, 32, 64, 4, 4),
            GemmConfig(64, 32, 64, 5, 2),
            GemmConfig(16, 32, 32, 2, 2),
            GemmConfig(16, 64, 32, 2, 2),
            GemmConfig(16, 128, 32, 2, 4),
            GemmConfig(16, 256, 32, 2, 4),
            GemmConfig(16, 32, 64, 2, 2),
            GemmConfig(16, 64, 64, 2, 2),
            GemmConfig(16, 128, 64, 2, 4),
            GemmConfig(16, 256, 64, 2, 4),
            GemmConfig(32, 32, 32, 2, 2),
            GemmConfig(32, 64, 32, 2, 2),
            GemmConfig(32, 128, 32, 2, 4),
            GemmConfig(32, 256, 32, 2, 4),
            GemmConfig(32, 32, 64, 2, 2),
            GemmConfig(32, 64, 64, 2, 2),
            GemmConfig(32, 128, 64, 2, 4),
            GemmConfig(32, 256, 64, 2, 4),
            GemmConfig(16, 32, 32, 3, 2),
            GemmConfig(16, 64, 32, 3, 2),
            GemmConfig(16, 128, 32, 3, 4),
            GemmConfig(16, 256, 32, 3, 4),
            GemmConfig(16, 32, 64, 3, 2),
            GemmConfig(16, 64, 64, 3, 2),
            GemmConfig(16, 128, 64, 3, 4),
            GemmConfig(16, 256, 64, 3, 4),
            GemmConfig(32, 32, 32, 3, 2),
            GemmConfig(32, 64, 32, 3, 2),
            GemmConfig(32, 128, 32, 3, 4),
            GemmConfig(32, 256, 32, 3, 4),
            GemmConfig(32, 32, 64, 3, 2),
            GemmConfig(32, 64, 64, 3, 2),
            GemmConfig(32, 128, 64, 3, 4),
            GemmConfig(32, 256, 64, 3, 4),
            GemmConfig(16, 32, 32, 4, 2),
            GemmConfig(16, 64, 32, 4, 2),
            GemmConfig(16, 128, 32, 4, 4),
            GemmConfig(16, 256, 32, 4, 4),
            GemmConfig(16, 32, 64, 4, 2),
            GemmConfig(16, 64, 64, 4, 2),
            GemmConfig(16, 128, 64, 4, 4),
            GemmConfig(16, 256, 64, 4, 4),
            GemmConfig(32, 32, 32, 4, 2),
            GemmConfig(32, 64, 32, 4, 2),
            GemmConfig(32, 128, 32, 4, 4),
            GemmConfig(32, 256, 32, 4, 4),
            GemmConfig(32, 32, 64, 4, 2),
            GemmConfig(32, 64, 64, 4, 2),
            GemmConfig(32, 128, 64, 4, 4),
            GemmConfig(32, 256, 64, 4, 4),
            GemmConfig(16, 32, 32, 5, 2),
            GemmConfig(16, 64, 32, 5, 2),
            GemmConfig(16, 128, 32, 5, 4),
            GemmConfig(16, 256, 32, 5, 4),
            GemmConfig(16, 32, 64, 5, 2),
            GemmConfig(16, 64, 64, 5, 2),
            GemmConfig(16, 128, 64, 5, 4),
            GemmConfig(16, 256, 64, 5, 4),
            GemmConfig(32, 32, 32, 5, 2),
            GemmConfig(32, 64, 32, 5, 2),
            GemmConfig(32, 128, 32, 5, 4),
            GemmConfig(32, 256, 32, 5, 4),
            GemmConfig(32, 32, 64, 5, 2),
            GemmConfig(32, 64, 64, 5, 2),
            GemmConfig(32, 128, 64, 5, 4),
            GemmConfig(32, 256, 64, 5, 4),
            GemmConfig(16, 32, 32, 6, 2),
            GemmConfig(16, 64, 32, 6, 2),
            GemmConfig(16, 128, 32, 6, 4),
            GemmConfig(16, 256, 32, 6, 4),
            GemmConfig(16, 32, 64, 6, 2),
            GemmConfig(16, 64, 64, 6, 2),
            GemmConfig(16, 128, 64, 6, 4),
            GemmConfig(16, 256, 64, 6, 4),
            GemmConfig(32, 32, 32, 6, 2),
            GemmConfig(32, 64, 32, 6, 2),
            GemmConfig(32, 128, 32, 6, 4),
            GemmConfig(32, 256, 32, 6, 4),
            GemmConfig(32, 32, 64, 6, 2),
            GemmConfig(32, 64, 64, 6, 2),
            GemmConfig(32, 128, 64, 6, 4),
            GemmConfig(32, 256, 64, 6, 4),
            GemmConfig(64, 16, 256, 5, 4),
            GemmConfig(64, 32, 256, 5, 4),
            GemmConfig(64, 128, 128, 2, 4),
            GemmConfig(64, 128, 128, 3, 4),
            GemmConfig(128, 128, 128, 2, 4),
            GemmConfig(128, 256, 128, 4, 8),
            GemmConfig(256, 128, 128, 2, 4),
            GemmConfig(256, 128, 128, 2, 8),
        ]

        self.scaled_persistent_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 128, 3, 8),
            GemmConfig(128, 128, 128, 4, 8),
            GemmConfig(128, 128, 128, 4, 4),
            GemmConfig(128, 128, 128, 3, 4),
            GemmConfig(128, 128, 128, 5, 4),
            GemmConfig(128, 128, 128, 5, 8),
            GemmConfig(128, 128, 128, 6, 8),
            GemmConfig(128, 128, 64, 4, 8),
            GemmConfig(64, 32, 256, 5, 4),
            GemmConfig(128, 256, 128, 3, 8),
            GemmConfig(64, 128, 256, 4, 4),
            GemmConfig(64, 256, 128, 4, 4),
        ]

        self.blackwell_scaled_persistent_mm_configs: list[BaseConfig] = [
            BlackwellGPUGemmConfig(
                block_m=c.block_m,
                block_n=c.block_n,
                block_k=c.block_k,
                num_stages=c.num_stages,
                num_warps=c.num_warps,
                hint_override=c.hint_override,
                group_m=8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            )
            for c in self.scaled_persistent_mm_configs
        ]

        # TODO: Unify with other gemm patterns, mm_plus_mm currently follows
        # slightly different pattern than rest
        self.mm_plus_mm_configs: list[BaseConfig] = [
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 64, 32, 3, 8),
            GemmConfig(64, 64, 32, 4, 16),
            GemmConfig(64, 32, 32, 4, 8),
            GemmConfig(32, 64, 32, 4, 8),
            GemmConfig(128, 128, 32, 1, 8),
            GemmConfig(64, 64, 64, 1, 8),
            GemmConfig(32, 32, 128, 1, 8),
            GemmConfig(64, 64, 16, 2, 4),
            GemmConfig(32, 32, 16, 1, 2),
        ]

        self.conv_configs: list[BaseConfig] = [
            # BLOCK_K=16 configs
            ConvConfig(64, 256, 16, 2, 4),
            ConvConfig(256, 64, 16, 2, 4),
            ConvConfig(1024, 16, 16, 1, 8),
            # BLOCK_K=32 configs
            ConvConfig(128, 128, 32, 2, 8),
            ConvConfig(64, 64, 32, 2, 4),
            ConvConfig(64, 256, 32, 2, 8),
            ConvConfig(256, 64, 32, 2, 8),
            # BLOCK_K=64 configs
            ConvConfig(128, 128, 64, 3, 8),
            ConvConfig(64, 128, 64, 4, 4),
            ConvConfig(128, 64, 64, 4, 4),
            ConvConfig(256, 128, 64, 2, 8),
            ConvConfig(128, 256, 64, 2, 8),
            # BLOCK_K=128 configs - optimal when IN_C=128 (single iteration over channels)
            ConvConfig(128, 128, 128, 2, 8),
            ConvConfig(128, 128, 128, 3, 8),
            ConvConfig(64, 128, 128, 4, 4),
            ConvConfig(256, 128, 128, 2, 8),
            ConvConfig(128, 256, 128, 2, 8),
        ]

        # Depthwise conv1d configs: BLOCK_N x BLOCK_L x BLOCK_C tiling
        # Derived from autotuning results on H100 for depthwise conv1d
        # channels-last (NLC) layout with shape x=[3072, 128, 202]
        # Matches _nlc_autotune_configs from depthwise_conv1d_benchmark.py
        self.depthwise_conv_configs: list[DepthwiseConvConfig] = [
            # BLOCK_C=32, BLOCK_L=32
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=32, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=32, block_c=32, num_stages=5, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=32, block_c=32, num_stages=4, num_warps=4
            ),
            # BLOCK_C=32, BLOCK_L=64
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=32, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=64, block_c=32, num_stages=3, num_warps=8
            ),
            # BLOCK_C=32, BLOCK_L=256
            DepthwiseConvConfig(
                block_n=16, block_l=256, block_c=32, num_stages=5, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=256, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=256, block_c=32, num_stages=3, num_warps=8
            ),
            # BLOCK_C=64
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=64, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=64, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=64, num_stages=3, num_warps=8
            ),
            # BLOCK_C=128
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=128, num_stages=3, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=128, num_stages=3, num_warps=4
            ),
        ]

        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(128, 64, 3, 4),
            FlexConfig(128, 128, 3, 4),
            FlexConfig(128, 128, 2, 8),
            FlexConfig(128, 128, 1, 8),
            FlexConfig(64, 128, 3, 4),
            FlexConfig(64, 64, 3, 4),
        ]

        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M, BLOCK_N, BLOCK_N, BLOCK_M, s, w)
            for BLOCK_M in [32, 64]
            for BLOCK_N in [32, 64, 128]
            for s in [1, 3, 4, 5]  # num_stages
            for w in ([4, 8] if BLOCK_M >= 128 or BLOCK_N >= 128 else [4])
            if BLOCK_N % BLOCK_M == 0
        ]

        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = [
            FlexDecodeConfig(64, 3, 2),
            FlexDecodeConfig(32, 3, 2),
            FlexDecodeConfig(128, 3, 2),
        ]

        self.exhaustive_flex_attn_fwd_configs: list[FlexConfig] = [
            FlexConfig(BLOCK_M, BLOCK_N, num_stages, num_warps)
            for BLOCK_M in [16, 32, 64, 128]
            for BLOCK_N in [32, 64, 128]
            for num_stages in [1, 3, 4, 5]
            for num_warps in [2, 4, 8]
        ]

        self.exhaustive_flex_attn_bwd_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_stages, num_warps)
            for BLOCK_M1 in [16, 32, 64, 128]
            for BLOCK_N1 in [16, 32, 64, 128]
            for BLOCK_M2 in [16, 32, 64, 128]
            for BLOCK_N2 in [16, 32, 64, 128]
            for num_stages in [1, 3, 4]
            for num_warps in [2, 4, 8]
            if BLOCK_N1 % BLOCK_M1 == 0
            and BLOCK_M2 % BLOCK_N2 == 0  # kernel static assertions
        ]

        self.exhaustive_flex_decode_configs: list[FlexDecodeConfig] = [
            FlexDecodeConfig(block_n, num_stages, num_warps)
            for block_n in [16, 32, 64, 128]
            for num_stages in [1, 3, 4, 5]
            for num_warps in [2, 4, 8]
        ]

    def _get_extra_config_key_and_kwargs(
        self, conf: BaseConfig
    ) -> tuple[tuple[int | None, ...], dict[str, Any]]:
        """Hook for subclasses to extend config dedup key and kwargs."""
        return (), {}

    def _finalize_mm_configs(
        self,
        configs: list[BaseConfig],
