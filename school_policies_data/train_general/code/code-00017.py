    print(source_name)
    raise ValueError(f"Unrecognized file extension: {source_name}")


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str], language: str = "c") -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, f'{name}{suffix}')
    cc = _find_compiler(language)
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    if language == "c++":
        cc_cmd.insert(3, "-std=c++17")
    cc_cmd += [_library_flag(lib) for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


def _library_flag(lib: str) -> str:
    # Match .so files with optional version numbers (e.g., .so, .so.1, .so.513.50.1)
    if re.search(r'\.so(\.\d+)*$', lib) or lib.endswith(".a"):
        return f"-l:{lib}"
    return f"-l{lib}"


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _get_file_extension(language):
    if language == "c":
        return ".c"
    if language == "c++":
        return ".cpp"
    raise ValueError(f"Unexpected languange: {language}")


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_cache_manager(src: bytes, config: dict[str, list[str] | None]):
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(platform_key().encode("utf-8"))
    for k, vs in config.items():
        if vs is None:
            continue
        digest.update(k.encode("utf-8"))
        for v in vs:
            digest.update(v.encode("utf-8"))
            digest.update(b":")
    key = digest.hexdigest()
    return get_cache_manager(key)


def _compile_so(src: bytes, src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                libraries: list[str] | None, ccflags: list[str] | None, load_module: bool, language: str):
    config = dict(language=[language], library_dirs=library_dirs, include_dirs=include_dirs, libraries=libraries,
                  ccflags=ccflags)
    cache = _get_cache_manager(src, config=config)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is not None:
        if not load_module:
            return cache_path
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [],
                    language=language)
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    return _load_module_from_path(name, cache_path) if load_module else cache_path


def _compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                          libraries: list[str] | None, ccflags: list[str] | None, load_module: bool):
    src_path = os.path.abspath(src_path)
    src_name = os.path.basename(src_path)
    with open(src_path, "rb") as f:
        src = f.read()

    language = _language_from_filename(src_name)
    return _compile_so(src=src, src_path=src_path, name=name, library_dirs=library_dirs, include_dirs=include_dirs,
                       libraries=libraries, ccflags=ccflags, language=language, load_module=load_module)


def _compile_so_from_src(src: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                         libraries: list[str] | None, ccflags: list[str] | None, language, load_module: bool):
    src_bytes = src.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}{_get_file_extension(language)}")
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        return _compile_so(src=src_bytes, src_path=src_path, name=name, library_dirs=library_dirs,
                           include_dirs=include_dirs, libraries=libraries, ccflags=ccflags, language=language,
                           load_module=load_module)


def compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                         include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                         ccflags: list[str] | None = None) -> str:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=False)


def compile_so_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                        include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                        ccflags: list[str] | None = None, language="c") -> str:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=False)


def compile_module_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                             include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                             ccflags: list[str] | None = None) -> ModuleType:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=True)


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None, language="c") -> ModuleType:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=True)
<|endoftext|>import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import base64
import hashlib
import functools
import sysconfig

from triton import __version__, knobs


class CacheManager(ABC):

    def __init__(self, key, override=False, dump=False):
        pass

    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, data, filename, binary=True) -> str:
        pass

    @abstractmethod
    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put_group(self, filename: str, group: Dict[str, str]):
        pass


class FileCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = knobs.cache.dump_dir
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = knobs.cache.override_dir
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = knobs.cache.dir
            if self.cache_dir:
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{filename}"
        if not self.has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        try:
            with open(grp_filepath) as f:
                grp_data = json.load(f)
        except Exception:
            # exit on corrupted cache.
            return None
        child_paths = grp_data.get("child_paths", None)
        # Invalid group data.
        if child_paths is None:
            return None
        result = {}
        for c, p in child_paths.items():
            if not os.path.exists(p):
                return None
            result[c] = p
        return result

    # Note a group of pushed files as being part of a group
    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        grp_contents = json.dumps({"child_paths": group})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename, binary=False)

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.
    """

    def __init__(self, key: str):
        pass

    @abstractmethod
    def get(self, filenames: List[str]) -> Dict[str, bytes]:
        pass

    @abstractmethod
    def put(self, filename: str, data: bytes):
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):

    def __init__(self, key):
        import redis
        self._key = key
        self._key_fmt = knobs.cache.redis.key_format
        self._redis = redis.Redis(
            host=knobs.cache.redis.host,
            port=knobs.cache.redis.port,
        )

    def _get_key(self, filename: str) -> str:
        return self._key_fmt.format(key=self._key, filename=filename)

    def get(self, filenames: List[str]) -> Dict[str, str]:
        results = self._redis.mget([self._get_key(f) for f in filenames])
        return {filename: result for filename, result in zip(filenames, results) if result is not None}

    def put(self, filename: str, data: bytes) -> Dict[str, bytes]:
        self._redis.set(self._get_key(filename), data)


class RemoteCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        # Setup backend pointed too by `TRITON_REMOTE_CACHE_BACKEND`.
        remote_cache_cls = knobs.cache.remote_manager_class
        if not remote_cache_cls:
            raise RuntimeError(
                "Unable to instantiate RemoteCacheManager, TRITON_REMOTE_CACHE_BACKEND doesn't point to a valid class")
        self._backend = remote_cache_cls(key)

        self._override = override
        self._dump = dump

        # Use a `FileCacheManager` to materialize remote cache paths locally.
        self._file_cache_manager = FileCacheManager(key, override=override, dump=dump)

    def _materialize(self, filename: str, data: bytes):
        # We use a backing `FileCacheManager` to provide the materialized data.
        return self._file_cache_manager.put(data, filename, binary=True)

    def get_file(self, filename: str) -> Optional[str]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_file(filename)

        # We always check the remote cache backend -- even if our internal file-
        # based cache has the item -- to make sure LRU accounting works as
        # expected.
        results = self._backend.get([filename])
        if len(results) == 0:
            return None
        (_, data), = results.items()
        return self._materialize(filename, data)

    def put(self, data, filename: str, binary=True) -> str:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put(data, filename, binary=binary)

        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        self._backend.put(filename, data)
        return self._materialize(filename, data)

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_group(filename)

        grp_filename = f"__grp__{filename}"
        grp_filepath = self.get_file(grp_filename)
        if grp_filepath is None:
            return None
        try:
            with open(grp_filepath) as f:
                grp_data = json.load(f)
        except Exception:
            # exit on corrupted cache.
            return None
        child_paths = grp_data.get("child_paths", None)

        result = None

        # Found group data.
        if child_paths is not None:
            child_data = self._backend.get(child_paths)
            if len(child_data) != len(child_paths):
                return None
            result = {}
            for child_path, data in child_data.items():
                result[child_path] = self._materialize(child_path, data)

        return result

    def put_group(self, filename: str, group: Dict[str, str]):
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put_group(filename, group)

        grp_contents = json.dumps({"child_paths": sorted(list(group.keys()))})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename)


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


def get_cache_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key))


def get_override_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key), override=True)


def get_dump_manager(key) -> CacheManager:
    cls = knobs.cache.manager_class or FileCacheManager
    return cls(_base32(key), dump=True)


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _base32(key)


@functools.lru_cache()
def triton_key():
    import pkgutil
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]
    # compiler
    path_prefixes = [
        (os.path.join(TRITON_PATH, "compiler"), "triton.compiler."),
        (os.path.join(TRITON_PATH, "backends"), "triton.backends."),
    ]
    for path, prefix in path_prefixes:
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.sha256(f.read()).hexdigest()]

    # backend
    libtriton_hash = hashlib.sha256()
    ext = sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]
    with open(os.path.join(TRITON_PATH, "_C", f"libtriton.{ext}"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.walk_packages([language_path], prefix="triton.language."):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    return f'{__version__}' + '-'.join(contents)


def get_cache_key(src, backend, backend_options, env_vars):
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
    return key
<|endoftext|>from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from types import ModuleType


@dataclass(frozen=True)
class GPUTarget(object):
    # Target backend, e.g., cuda, hip
    backend: str
    # Target architecture, e.g., 90 (for cuda compute capability), gfx940 (for hip)
    arch: Union[int, str]
    warp_size: int


class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1


class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget):
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        raise NotImplementedError

    @abstractmethod
    def parse_options(self, options: dict) -> object:
        """
        Converts an `options` dictionary into an arbitrary object and returns it.
        This function may contain target-specific heuristics and check the legality of the provided options
        """
        raise NotImplementedError

    @abstractmethod
    def add_stages(self, stages: dict, options: object) -> None:
        """
        Populates `stages` dictionary with entries of the form:
        ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
        The value of each entry may populate a `metadata` dictionary.
        Stages will be run sequentially (in inseriton order) and can communicate using `metadata`.
        All stages are expected to return a `str` object, except for the last stage which returns
        a `bytes` object for execution by the launcher.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dialects(self, context):
        """
        Load additional MLIR dialects into the provided `context`
        """
        raise NotImplementedError

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations
        """
        raise NotImplementedError

    @staticmethod
    def parse_attr(desc):
        assert isinstance(desc, str)
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 16]]
        return ret

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        if arg % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        if arg.data_ptr() % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""
<|endoftext|>import importlib
import os
import inspect
import sys
from dataclasses import dataclass
from typing import Type, TypeVar, Union
from types import ModuleType
from .driver import DriverBase
from .compiler import BaseBackend

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

T = TypeVar("T", bound=Union[BaseBackend, DriverBase])


def _find_concrete_subclasses(module: ModuleType, base_class: Type[T]) -> Type[T]:
    ret: list[Type[T]] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]


def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    # Fast path: optionally skip entry point discovery (which can be slow) and
    # discover only in-tree backends under the `triton.backends` namespace.
    skip_entrypoints_env = os.environ.get("TRITON_BACKENDS_IN_TREE", "")

    if skip_entrypoints_env == "1":
        root = os.path.dirname(__file__)
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if name.startswith('__'):
                continue
            compiler = importlib.import_module(f"triton.backends.{name}.compiler")
            driver = importlib.import_module(f"triton.backends.{name}.driver")
            backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                     _find_concrete_subclasses(driver, DriverBase))
        return backends

    # Default path: discover via entry points for out-of-tree/downstream plugins.
    for ep in entry_points().select(group="triton.backends"):
        compiler = importlib.import_module(f"{ep.value}.compiler")
        driver = importlib.import_module(f"{ep.value}.driver")
        backends[ep.name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),  # type: ignore
                                    _find_concrete_subclasses(driver, DriverBase))  # type: ignore
    return backends


backends: dict[str, Backend] = _discover_backends()
<|endoftext|>from abc import ABCMeta, abstractmethod
import re
from typing import Callable, List, Protocol, Sequence

from triton._utils import find_paths_if
from triton._C.libtriton import make_tensordesc_args


def decompose_descriptor(arg):
    # Currently host-side tensor descriptors are passed as tensor desc + shape + strides.
    # We still need to pass shape/strides after descriptor lowering, so they appear twice.
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", arg.round_f32_to_tf32, *arg.shape, *arg.strides]


def _is_descriptor(arg):
    return isinstance(arg, str) and arg.startswith("tensordesc")


def wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg):
    signature = tuple(signature.values())
    tensordesc_paths = find_paths_if(signature, lambda _, x: _is_descriptor(x))
    if len(tensordesc_paths) == 0:
        return launcher

    # Build a tree to speed up tensordesc type checking, e.g.
    # signature = ('tensordesc', 'i32', ('i32', 'tensordesc'))
    # relevant_paths = {0: {}, 2: {1: {}}}
    relevant_paths = {}
    for path in tensordesc_paths:
        cur = relevant_paths
        for step in path:
            cur = cur.setdefault(step, {})

    def inner(*args):
        base_args = args[:-1]
        kernel_args = args[-1]
        wrapped = make_tensordesc_args(
            kernel_args,
            signature,
            relevant_paths,
            tensordesc_meta,
            base_args,
            make_tensordesc_arg,
        )
        return launcher(*base_args, wrapped)

    return inner


def _parse_descriptor(descriptor):
    match = re.match(r"tensordesc(?:_im2col)?<([^[>]*)\[([^\]]*)\]", descriptor)
    assert match, f"Malformed tensor descriptor type: {descriptor}"

    dtype = match.group(1)
    block_shape = match.group(2)
    block_ndim = block_shape.count(",") + 1

    rank_match = re.search(r",input_rank=(\d+)", descriptor)
    ndim = int(rank_match.group(1)) if rank_match else block_ndim
    return (dtype, ndim)


def _expand_descriptor(descriptor, has_tensordesc_meta, descriptor_type):
    dtype, ndim = _parse_descriptor(descriptor)
    expanded = []

    # If there is no descriptor metadata, the descriptor was decomposed to:
    # base pointer, shape, strides, padding, round_f32_to_tf32.
    if not has_tensordesc_meta:
        expanded.append("*" + dtype)
        for _ in range(2 * ndim):
            expanded.append("i64")
        expanded.append("i1")
        expanded.append("i1")
    else:
        expanded.append(descriptor_type)

    for _ in range(ndim):
        expanded.append("i32")
    for _ in range(ndim):
        expanded.append("i64")
    return expanded


def expand_signature(signature, tensordesc_meta, descriptor_type):
    has_tensordesc_meta = bool(tensordesc_meta)

    result = []

    def visit(signature, result):
        if _is_descriptor(signature):
            result.extend(_expand_descriptor(signature, has_tensordesc_meta, descriptor_type))
            return
        elif isinstance(signature, tuple):
            inner = []
            for s in signature:
                visit(s, inner)
            result.append(tuple(inner))
        else:
            result.append(signature)

    result = []
    for s in signature:
        visit(s, result)
    return result


class Benchmarker(Protocol):

    def __call__(self, kernel_call: Callable, *, quantiles: List[float], **kwargs) -> Sequence[float]:
        pass


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for this backend.

        Args:
            ty (str): The Triton type string. e.g., 'i32', '*fp16', 'fp32'.

        Returns:
            str: The C++ type string.
        """
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """
        Return the benchmarking function that this backend should use by default.
        """
        raise NotImplementedError

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        """
        Allocate profile scratch when no explicit profile allocator override was installed.
        """
        raise NotImplementedError

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        import torch
        device = self.get_active_torch_device()
        device_interface = self.get_device_interface()
        if stream is None:
            return torch.zeros(size, dtype=torch.int8, device=device)
        if stream == 0:
            with device_interface.stream(device_interface.default_stream(device)):
                return torch.zeros(size, dtype=torch.int8, device=device)
        launch_stream = device_interface.ExternalStream(stream, device=device)
        with device_interface.stream(launch_stream):
            scratch = torch.zeros(size, dtype=torch.int8, device=device)
        scratch.record_stream(launch_stream)
        return scratch
<|endoftext|>from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        elif lhs_bitwidth == 64:
            return (1, 1, 4)
        elif lhs_bitwidth == 32:
            return (1, 1, 8)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version(arch: int = 80):
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch).path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas(arch).version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm 979132a0.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(90, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    # TODO: Handle non-"a" sms
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    ptx_version: int = None
    ptx_options: Optional[str] = knobs.nvidia.ptxas_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')
        if "gsan" in self.instrumentation_mode:
            gsan_lib = default_libdir / "gsan.ll"
            if not gsan_lib.exists():
                raise FileNotFoundError(f"GSan runtime is missing at {gsan_lib}. "
                                        "Rebuild Triton to generate it.")
            extern_libs['gsan'] = str(gsan_lib)

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @property
    def enable_iisan(self):
        return "iisan" in self.instrumentation_mode


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan", "iisan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emuTF32 = (capability // 10 >= 8)
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_optimize_partition_warps(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 == 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        nvidia.passes.ttnvgpuir.add_tmem_load_reduce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        if "fpsan" in opt.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        if "fpsan" in options.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
        if any(mode in options.instrumentation_mode for mode in ["consan", "fpsan"]):
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)

        mod = src
        # Work around ptxas rejecting PTX generated by the LLVM SLP vectorizer on sm_80.
        disable_slp_vectorizer = capability == 80
        if capability == 90:
            # Work around llvm/llvm-project#203741, where SLP reorders ordered fadd
            # reductions. This release's LLVM pin predates the upstream fix.
            ttgir = mod.str_nodebug()
            disable_slp_vectorizer = "bf16" in ttgir and "tt.reduce" in ttgir
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        if "gsan" in options.instrumentation_mode:
            # GSan introduces layout conversions, so must come before shared memory allocation
            passes.ttgpuir.add_global_sanitizer(pm)

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm, "consan" in options.instrumentation_mode)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        if "consan" in options.instrumentation_mode:
            passes.ttgpuir.add_prepare_consan_captures(pm, "nvidia")
        nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_tmem_barrier_insertion(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version, "consan" in options.instrumentation_mode)
        nvidia.passes.ttnvgpuir.add_initialize_ws_cluster_barriers(pm, capability, ptx_version)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)

        if capability == 107:
            cap_llvm = 100
        else:
            cap_llvm = capability

        proc = sm_arch_from_capability(cap_llvm)
        features = get_features(options, cap_llvm)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        if options.enable_reflect_ftz:
            nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, disable_slp_vectorizer=disable_slp_vectorizer)

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = 'nvptx64-nvidia-cuda'

        if capability == 107:
            cap_llvm = 100
        else:
            cap_llvm = capability

        proc = sm_arch_from_capability(cap_llvm)
        features = get_features(opt, cap_llvm)
        flags = ["nvptx-mad-wide-opt"]
        canonicalize_gep = "fpsan" in opt.instrumentation_mode
        ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False, canonicalize_gep)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        if not knobs.compilation.dump_ir_extract_di_local_variables:
            # Remove the debug flag that prevents ptxas from optimizing the code
            # Note: if this flag is removed, the source var name and type info will be lost when ptx was compiled into cubin
            #           and we may not be able to see them in cuda-gdb
            ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if knobs.nvidia.dump_nvptx:
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas(self.target.arch).path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            # -Ofc mid miscompiles some large ConSan kernels into invalid global
            # accesses; -O1 keeps compile time reasonable without that ptxas bug.
            if (not knobs.nvidia.disable_ptxas_opt
                    and any(mode in opt.instrumentation_mode for mode in ["consan", "fpsan"])):
                ptx_extra_options += ["--opt-level", "1"]

            # Add --regAllocOptLevel=2 to work around ptxas 13.x bug
            reg_alloc = ['--regAllocOptLevel=2']

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *reg_alloc, *ptx_extra_options, f'--gpu-name={arch}',
                fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version(self.target.arch)
        return f'{version}-{self.target.arch}'
<|endoftext|>import functools
import os
import subprocess
import triton
import ctypes
import sys
from triton import knobs
from triton.runtime.build import compile_module_from_file
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, decompose_descriptor, expand_signature, wrap_handle_tensordesc_impl

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['libcuda.so.1']
PyCUtensorMap = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None
GSAN_PER_DEVICE_STATE_STRIDE = 1 << 30


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


def _cuda_driver_is_active():
    candidates = ["libcuda.so.1"]
    try:
        candidates.extend([os.path.join(path, "libcuda.so.1") for path in libcuda_dirs()])
    except Exception:
        pass

    libcuda = None
    for candidate in candidates:
        try:
            libcuda = ctypes.CDLL(candidate)
            break
        except OSError:
            continue

    if libcuda is None:
        return False

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    if cu_init(0) != 0:
        return False

    cu_device_get_count = libcuda.cuDeviceGetCount
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if cu_device_get_count(ctypes.byref(count)) != 0:
        return False

    return count.value > 0


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_file(
            src_path=os.path.join(dirname, "driver.c"),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyCUtensorMap = mod.PyCUtensorMap
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.unload_module = mod.unload_module
        self.get_current_device = mod.get_current_device
        self.set_current_device = mod.set_current_device
        self.get_default_stream = mod.get_default_stream
        self.get_device_capability = mod.get_device_capability
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor_tiled = mod.fill_tma_descriptor_tiled
        self.fill_tma_descriptor_im2col = mod.fill_tma_descriptor_im2col
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9
TMA_TF32 = 11


def make_tensordesc_arg(arg, metadata, _):
    if metadata is None:
        return decompose_descriptor(arg)

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]
    is_im2col = metadata.get("is_im2col", False)

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        expanded_shape = list(shape)
        expanded_shape[-1] *= 2
    else:
        expanded_shape = shape

    if arg.round_f32_to_tf32:
        elem_type = TMA_TF32

    if is_im2col:
        # Im2col mode - use im2col descriptor fill function
        # block_size from metadata is [pixelsPerColumn, channelsPerPixel] (possibly clamped)
        element_strides = arg.element_strides if arg.element_strides is not None else [1] * len(shape)
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_im2col(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
            arg.pixel_box_lower_corner,
            arg.pixel_box_upper_corner,
            element_strides,
        )
    else:
        # Tiled mode - use existing tiled descriptor fill function
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
        )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg)


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        self.gsan_enabled = "gsan" in getattr(metadata, "instrumentation_mode", "")
        if self.gsan_enabled:
            signature["_gsan_globals_ptr"] = "*i8"

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta, "nvTmaDesc")
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        active_driver = triton.runtime.driver.active

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        def allocate_default_profile_scratch(size, align):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                return active_driver.allocate_default_profile_scratch(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        if _allocation.has_profile_allocator():
            profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                               _allocation._profile_allocator)
        else:
            profile_scratch = allocate_default_profile_scratch(self.profile_scratch_size, self.profile_scratch_align)

        kernel_args = args
        if self.gsan_enabled:
            import triton.experimental.gsan._allocator as gsan_allocator
            device = triton.runtime.driver.active.get_current_device()
            device_rank = gsan_allocator.get_device_rank(device)
            gsan_state_ptr = gsan_allocator.get_global_state_pointer() + device_rank * GSAN_PER_DEVICE_STATE_STRIDE
            kernel_args = (*args, gsan_state_ptr)

        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
                    profile_scratch, self.arg_annotations, self.kernel_signature, kernel_args)
        if self.gsan_enabled:
            import triton.experimental.gsan._stream_sync as gsan_stream_sync
            gsan_stream_sync.synchronize_launch_stream(device)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        if sys.modules.get("torch") is not None:
            super().__init__()
        else:
            self.get_device_capability = self._get_device_capability
            self.get_current_stream = self._get_current_stream
            self.get_current_device = self._get_current_device
            self.set_current_device = self._set_current_device

    def _get_device_capability(self, device):
        return self.utils.get_device_capability(device)

    def _get_current_stream(self, device):
        # The CUDA driver API does not expose PyTorch's notion of the current
        # stream. In torch-free launches we fall back to the device's default
        # stream after making that device's primary context current.
        return self.utils.get_default_stream(device)

    def _get_current_device(self):
        return self.utils.get_current_device()

    def _set_current_device(self, device):
        self.utils.set_current_device(device)

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        # Temporary workaround for torch-loaded processes: cuInit(0) is not
        # fork-safe after CUDA initialization and can make Triton report no
        # active driver from forked workers.
        torch = sys.modules.get("torch")
        if torch is not None:
            return torch.cuda.is_available() and torch.version.hip is None
        return _cuda_driver_is_active()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
<|endoftext|>from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, amd
from triton import knobs
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import os
import hashlib
import tempfile
import re
import functools
import warnings
from pathlib import Path


def get_min_dot_size(target: GPUTarget):
    # We fallback to use FMA and cast arguments if certain configurations is
    # not supported natively by matrix core units.
    return lambda lhs_type, rhs_type: (1, 1, 1)


def is_pingpong_schedule_enabled(arch, use_async_copy):
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)) \
        if knobs.amd.use_block_pingpong is None else knobs.amd.use_block_pingpong


def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942" or "gfx110" in arch or "gfx115" in arch or "gfx120" in arch) \
        if knobs.amd.use_in_thread_transpose is None else knobs.amd.use_in_thread_transpose


def is_async_copy_enabled(arch):
    return (arch in ["gfx950", "gfx1250"]) if knobs.amd.use_async_copy is None else knobs.amd.use_async_copy


def is_coexec_scheduler_supported(arch):
    return arch in ["gfx1250"]


def is_coexec_scheduler_enabled(arch):
    if knobs.amd.use_coexec_scheduler is not None:
        return knobs.amd.use_coexec_scheduler
    return arch in ["gfx1250"]


def is_expert_scheduling_enabled(arch):
    if knobs.amd.use_expert_scheduling is not None:
        return knobs.amd.use_expert_scheduling
    return arch in ["gfx1250"]


def is_fpsan_supported(arch):
    return arch in ["gfx942", "gfx950", "gfx1250"]


def is_consan_supported(arch):
    return arch in ["gfx1250"]


def disable_real_true16_feature(arch):
    return '-real-true16' if arch.startswith('gfx11') else ''


def _parse_llvm_fn_attrs(attrs):
    if not isinstance(attrs, str):
        return tuple(attrs)
    parsed = []
    for attr in attrs.split(","):
        name, sep, value = attr.partition("=")
        name = name.strip()
        if name:
            parsed.append((name, value.strip() if sep else ""))
    return tuple(parsed)


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 0
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    # We have native support for OCP fp8 variants since CDNA4/RDNA4. For earlier generations,
    # we software emulate the support for them.
    # UZ fp8 variants (fp8e4b8 and fp8e5b16) are natively supported for CDNA3. For other
    # architectures they are software emulated.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", 'bf16x3', 'bf16x6')
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'
    instrumentation_mode: str = ""

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. Experimental; right now no effect.
    schedule_hint: str = ''

    # Experimental: intended for development and debugging; may change or be removed without notice.
    # Comma-separated LLVM function attributes; bare names are emitted as valueless attributes.
    # Example: llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp,noinline"
    llvm_fn_attrs: str | Tuple[Tuple[str, str], ...] = ""

    def __post_init__(self):
        gfx_major = int(self.arch[3:-2])  # Drop "gfx" prefix and minor/patch number
        warp_size = 32 if gfx_major >= 10 else 64
        object.__setattr__(self, 'warp_size', warp_size)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

        if (self.arch == 'gfx950') and (self.kpack != 1):
            warnings.warn(
                f"kpack is deprecated starting from gfx950 and will be removed in later releases. So for now kpack = {self.kpack} will be overwritten to 1 to make transitioning easier."
            )
            object.__setattr__(self, 'kpack', 1)

        object.__setattr__(self, 'llvm_fn_attrs', _parse_llvm_fn_attrs(self.llvm_fn_attrs))

        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        for lib in ["ocml", "ockl"]:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):
    instrumentation = None
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"hip:{options.arch}"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {'arch': knobs.runtime.override_arch or self.target.arch}

        if opts.get("num_ctas", 1) > 1 and not amd.supports_multi_cta_launch(self.target.arch):
            raise ValueError(f"num_ctas > 1 not supported on {self.target.arch}")

        # Enable XF32 (TF32) for CDNA3 GPUs
        if self.target.arch == 'gfx942':
            allowed_dot_input_precisions = set(HIPOptions.allowed_dot_input_precisions)
            allowed_dot_input_precisions.update({'tf32'})
            args["allowed_dot_input_precisions"] = tuple(sorted(allowed_dot_input_precisions))

        if "supported_fp8_dtypes" not in opts:
            args["supported_fp8_dtypes"] = tuple(sorted(HIPOptions.supported_fp8_dtypes))

        if self.target.arch == 'gfx950':
            deprecated_fp8_dot_operand_dtypes = set(HIPOptions.deprecated_fp8_dot_operand_dtypes)
            deprecated_fp8_dot_operand_dtypes.update({"fp8e5b16", "fp8e4b8"})
            args["deprecated_fp8_dot_operand_dtypes"] = tuple(sorted(deprecated_fp8_dot_operand_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts and opts[k] is not None})
        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": get_min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.load_dialects(ctx)

    # is_within_2gb() needs to check for a torch subobject and this var tracks torch
    # availability state: None - not tested, True - torch is present. Anything else -
    # no torch available. First call to is_within_2gb() checks torch availability
    # and caches it.
    _torch_available: None | bool = None

    @staticmethod
    def is_within_2gb(arg):
        if HIPBackend._torch_available is None:
            try:
                import torch
                HIPBackend._torch_available = True
            except ImportError:
                HIPBackend._torch_available = False
        elif HIPBackend._torch_available:
            import torch

        MAX_INT_32 = 2**31 - 1
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= MAX_INT_32
        if HIPBackend._torch_available and isinstance(arg, torch.Tensor) and hasattr(arg, "untyped_storage"):
            return arg.untyped_storage().size() <= MAX_INT_32
        return False

    @staticmethod
    def parse_attr(desc):
        ret = BaseBackend.parse_attr(desc)
        if "S" in desc:
            ret += [["tt.pointer_range", 32]]
        return ret

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
        if knobs.amd.use_buffer_ops and HIPBackend.is_within_2gb(arg):
            ret += "S"
        return ret

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        if not amd.supports_tdm(options.arch):
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod, 'make_ttgir_early')
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        emuTF32 = False
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_optimize_epilogue(pm)
        amd.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)
        amd.passes.ttgpuir.add_hoist_layout_conversions(pm)
        amd.passes.ttgpuir.add_sink_layout_conversions(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)

        use_async_copy = is_async_copy_enabled(options.arch)
        use_block_pingpong = is_pingpong_schedule_enabled(options.arch, use_async_copy)
        amd.passes.ttgpuir.add_optimize_descriptor_encoding(pm)
        amd.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)
        amd.passes.ttgpuir.add_pipeline(pm, use_async_copy, use_block_pingpong)
        if use_async_copy:
            amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
        amd.passes.ttgpuir.add_convert_to_tensor_ops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if is_in_thread_transpose_enabled(options.arch):
            amd.passes.ttgpuir.add_in_thread_transpose(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_move_up_prologue_loads(pm)
        if use_block_pingpong and options.num_stages > 1:
            amd.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)

        if knobs.amd.use_buffer_ops:
            amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            amd.passes.ttgpuir.add_convert_to_buffer_ops(
                pm,
                options.arch,
                knobs.amd.use_buffer_atomics,
                knobs.amd.buffer_ops_analyze_small_tensor_range,
            )
            amd.passes.ttgpuir.add_optimize_buffer_op_ptr(pm)

        amd.passes.ttgpuir.add_fold_true_cmpi(pm)
        amd.passes.ttgpuir.add_prepare_if_combining(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if options.instrumentation_mode == "fpsan" and is_fpsan_supported(options.arch):
            amd.passes.ttgpuir.add_fp_sanitizer(pm)
            passes.ttgpuir.add_fp_sanitizer(pm)
        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def gluon_to_ttgir(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        amd.passes.ttgpuir.add_warp_pipeline(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)

        if options.instrumentation_mode == "fpsan" and is_fpsan_supported(options.arch):
            amd.passes.ttgpuir.add_fp_sanitizer(pm)
            passes.ttgpuir.add_fp_sanitizer(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        amd.passes.ttgpuir.add_warp_pipeline_conversion(pm, options.arch)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.convert.add_index_to_llvmir(pm)

        # Reserve LDS space for ConSan captures before allocation computes offsets.
        if "consan" in options.instrumentation_mode and is_consan_supported(options.arch):
            passes.ttgpuir.add_prepare_consan_captures(pm, "amd")
        amd.passes.ttgpuir.add_allocate_shared_memory(pm, options.arch)
        # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after shared
        if "consan" in options.instrumentation_mode and is_consan_supported(options.arch):
            passes.ttgpuir.add_concurrency_sanitizer(pm)
            passes.gluon.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        amd.passes.ttgpuir.add_warp_specialize_to_llvm(pm, options.arch)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        # This can not be moved below the di_scope pass
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, options.arch, __HIP_FTZ)
        # Cleanup leftover unrealized_conversion_casts before converted to LLVM.
        passes.convert.add_reconcile_unrealized_casts(pm)
        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        target_features = ''
        if knobs.compilation.enable_asan:
            target_features = '+xnack'
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 500)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes first given this may affect later optimizations.
        # The kernel is the only non-declaration function with external linkage;
        # instrumentation helpers (e.g. ConSan) use internal linkage.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        kernel_fn = next((fn for fn in fns if fn.is_external_linkage()), None)
        if not kernel_fn:
            raise RuntimeError("Could not find kernel function")
        kernel_fn.set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)

        cluster_dim = metadata["num_ctas"]
        kernel_fn.add_fn_attr("amdgpu-cluster-dims", f"{cluster_dim},1,1")
        # warp-specialization mutates num_warps
        total_warps_num = options.num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            total_warps_num = total_num_warps
        kernel_fn.add_fn_attr("amdgpu-flat-work-group-size", f"1,{total_warps_num*options.warp_size}")
        kernel_fn.add_fn_attr("uniform-work-group-size", "true")
        # "amdgpu-waves-per-eu"="<min>[,<max>]" allows controlling the minimal and (optional)
        # maximal number of waves per SIMD. If <max> is omitted, then no restriction other than
        # whatever allowed by the hardware for occupancy.
        # Specifying "N,N" forces LLVM to focus on a single register budget, simplifies some
        # heuristics and may improve scheduling.
        if options.waves_per_eu != 0:
            kernel_fn.add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu},{options.waves_per_eu}")

        if is_coexec_scheduler_enabled(options.arch) and options.num_warps <= 4:
            kernel_fn.add_fn_attr("amdgpu-sched-strategy", "coexec")

        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        kernel_fn.add_fn_attr("denormal-fp-math-f32", denormal_mode)

        if knobs.compilation.enable_asan:
            kernel_fn.add_fn_target_feature("+xnack")
            kernel_fn.add_fn_asan_attr()
        for name, value in options.llvm_fn_attrs:
            kernel_fn.remove_fn_attr(name)
            kernel_fn.add_fn_attr(name, value)

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        #
        # TODO(tyb0807): Disabled when using MIR swap/dump because the value is
        # not serializable to/from MIR YAML
        if options.arch != "gfx1250" and not (knobs.amd.swap_mir or knobs.amd.dump_mir):
            amd.set_all_fn_arg_inreg(kernel_fn)

        if knobs.compilation.enable_asan:
            default_libdir = Path(__file__).parent / 'lib'
            paths = [
                str(default_libdir / 'asanrtl.bc'),
                str(default_libdir / "ocml.bc"),
                str(default_libdir / "ockl.bc")
            ]
            llvm.link_extern_libs(llvm_mod, paths)
        elif options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            if len(paths) > 0:
                llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion,
                             disable_vector_combine=True)

        # Architectures with architected SGPRs store the workgroup id in ttmp9 (X) and ttmp7 (Y[15:0], Z[31:16]).
        # These attributes are used to determine if Z should be masked out when loading Y. They are inferred during
        # optimize_module from calls to @llvm.amdgcn.workgroup.id.x/y/z(). We cannot rely on this because a
        # dispatch dimensions might be used even if there is no program_id() call for it.
        if amd.has_architected_sgprs(options.arch):
            kernel_fn.remove_fn_attr("amdgpu-no-workgroup-id-x")
            kernel_fn.remove_fn_attr("amdgpu-no-workgroup-id-y")
            kernel_fn.remove_fn_attr("amdgpu-no-workgroup-id-z")

        if knobs.amd.scalarize_packed_fops:
            amd.add_scalarize_packed_fops_llvm_pass(kernel_fn)

        # Get some metadata
        metadata["num_warps"] = total_warps_num
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        amd.cleanup_bitcode_metadata(llvm_mod)
        # Disable inlining of print related functions,
        # because inlining of these function could slow down compilation significantly
        amd.disable_print_inline(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accommodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        flags = []
        if is_expert_scheduling_enabled(options.arch):
            flags.append("amdgpu-expert-scheduling-mode")
        features = disable_real_true16_feature(options.arch)
        ir_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()
        dump_file_id = names[0] + '_' + ir_hash
        _ = llvm.translate_to_mir(src, amd.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,
                                  dump_file_id)
        llvm.dump_sched_dag(src, amd.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,
                            dump_file_id)
        if knobs.amd.swap_mir_enable_misched and not knobs.amd.swap_mir:
            raise ValueError("TRITON_SWAP_MIR_ENABLE_MISCHED requires TRITON_SWAP_MIR to be set")
        if knobs.amd.swap_mir:
            amdgcn = llvm.translate_mir_to_asm(os.path.join(knobs.amd.swap_mir, dump_file_id + '.txt'),
                                               amd.TARGET_TRIPLE, options.arch, features, flags,
                                               options.enable_fp_fusion, False, knobs.amd.swap_mir_enable_misched)
        else:
            amdgcn = llvm.translate_to_asm(src, amd.TARGET_TRIPLE, options.arch, features, flags,
                                           options.enable_fp_fusion, False, False)
        if knobs.amd.dump_amdgcn:
            print("// -----// AMDGCN Dump //----- //")
            print(amdgcn)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        target_features = []
        if knobs.compilation.enable_asan:
            target_features.append('+xnack')
        if true16 := disable_real_true16_feature(options.arch):
            target_features.append(true16)
        hsaco = amd.assemble_amdgcn(src, options.arch, ','.join(target_features))
        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, "wb") as fd_in:
                    fd_in.write(hsaco)
                amd.link_hsaco(tmp_in.name, tmp_out.name)
            with open(tmp_out.name, "rb") as fd_out:
                ret = fd_out.read()
        return ret

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f'{self.target}'
<|endoftext|>import functools
import os
import subprocess
import triton
from pathlib import Path
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, decompose_descriptor, expand_signature, wrap_handle_tensordesc_impl
from triton.runtime import _allocation
from triton.runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
PyTDMDescriptor = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


def _find_already_mmapped_dylib_on_linux(lib_name):
    import platform
    if platform.system() != 'Linux':
        return None

    # Use dl_iterate_phdr to walk through the list of shared libraries at runtime.
    # See https://www.man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html for details.

    import ctypes
    from ctypes import c_char, c_int, c_size_t, c_void_p, c_char_p, POINTER

    class DlPhdrInfo(ctypes.Structure):
        _fields_ = [
            ('dlpi_addr', c_void_p),
            ('dlpi_name', c_char_p),
            # We don't care about the remaining fields.
        ]

    # callback_t must use POINTER(c_char) to avoid copying.
    callback_t = ctypes.CFUNCTYPE(c_int, POINTER(DlPhdrInfo), POINTER(c_size_t), POINTER(c_char))

    # Load libc and get the dl_iterate_phdr symbol.
    try:
        dl_iterate_phdr = ctypes.CDLL('libc.so.6').dl_iterate_phdr
    except Exception:
        return None
    # argtypes must use c_char_p to accept create_string_buffer.
    dl_iterate_phdr.argtypes = [callback_t, c_char_p]
    dl_iterate_phdr.restype = c_int

    max_path_length = 4096
    path = ctypes.create_string_buffer(max_path_length + 1)

    # Define callback to get the loaded dylib path.
    def callback(info, size, data):
        dlpi_name = info.contents.dlpi_name
        p = Path(os.fsdecode(dlpi_name))
        if lib_name in p.name:
            # Found the dylib; get its path.
            ctypes.memmove(data, dlpi_name, min(max_path_length, len(dlpi_name)))
            return 1
        return 0

    if dl_iterate_phdr(callback_t(callback), path):
        return os.fsdecode(ctypes.string_at(path))
    return None


@functools.lru_cache()
def _get_path_to_hip_runtime_dylib():
    lib_name = "libamdhip64.so"

    # If we are told explicitly what HIP runtime dynamic library to use, obey that.
    if env_libhip_path := knobs.amd.libhip_path:
        if env_libhip_path.endswith(lib_name) and os.path.exists(env_libhip_path):
            return env_libhip_path
        raise RuntimeError(f"TRITON_LIBHIP_PATH '{env_libhip_path}' does not point to a valid {lib_name}")

    # If the shared object is already mmapped to address space, use it.
    mmapped_path = _find_already_mmapped_dylib_on_linux(lib_name)
    if mmapped_path:
        if os.path.exists(mmapped_path):
            return mmapped_path
        raise RuntimeError(f"memory mapped '{mmapped_path}' in process does not point to a valid {lib_name}")

    paths = []

    # Check backend
    local_lib = os.path.join(os.path.dirname(__file__), "lib", lib_name)
    if os.path.exists(local_lib):
        return local_lib
    paths.append(local_lib)

    import site
    # First search the HIP runtime dynamic library packaged with PyTorch. It's very likely
    # that we run Triton together with PyTorch. This makes sure we use the same dynamic
    # library to avoid version mismatch.
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE:  # ENABLE_USER_SITE is initialized in getusersitepackages()
        site_packages = [user_site] + site_packages
    for path in site_packages:
        path = os.path.join(path, "torch", "lib", lib_name)
        if os.path.exists(path):
            return path
        paths.append(path)

    # Then try to see if developer provides a HIP runtime dynamic library using LD_LIBARAY_PATH.
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path:
        for d in env_ld_library_path.split(":"):
            f = os.path.join(d, lib_name)
            if os.path.exists(f):
                return f
            paths.append(f)

    # HIP_PATH should point to HIP SDK root if set
    env_hip_path = os.getenv("HIP_PATH")
    if env_hip_path:
        hip_lib_path = os.path.join(env_hip_path, "lib", lib_name)
        if os.path.exists(hip_lib_path):
            return hip_lib_path
        paths.append(hip_lib_path)

    # if available, `hipconfig --path` prints the HIP SDK root
    try:
        hip_root = subprocess.check_output(["hipconfig", "--path"]).decode().strip()
        if hip_root:
            hip_lib_path = os.path.join(hip_root, "lib", lib_name)
            if os.path.exists(hip_lib_path):
                return hip_lib_path
            paths.append(hip_lib_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # hipconfig may not be available
        pass

    # ROCm lib dir based on env var
    env_rocm_path = os.getenv("ROCM_PATH")
    if env_rocm_path:
        rocm_lib_path = os.path.join(env_rocm_path, "lib", lib_name)
        if os.path.exists(rocm_lib_path):
            return rocm_lib_path
        paths.append(rocm_lib_path)

    # Afterwards try to search the loader dynamic library resolution paths.
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libamdhip64.so.6 (libc6,x86-64) => /opt/rocm-6.0.2/lib/libamdhip64.so.6
    # libamdhip64.so (libc6,x86-64) => /opt/rocm-6.0.2/lib/libamdhip64.so
    locs = [line.split()[-1] for line in libs.splitlines() if line.strip().endswith(lib_name)]
    for loc in locs:
        if os.path.exists(loc):
            return loc
        paths.append(loc)

    # As a last resort, guess if we have it in some common installation path.
    common_install_path = os.path.join('/opt/rocm/lib/', lib_name)
    if os.path.exists(common_install_path):
        return common_install_path
    paths.append(common_install_path)

    raise RuntimeError(f"cannot locate {lib_name} after attempted paths {paths}")


class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        libhip_path = _get_path_to_hip_runtime_dylib()
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        # Just do a simple search and replace here instead of templates or format strings.
        # This way we don't need to escape-quote C code curly brackets and we can replace
        # exactly once.
        src = src.replace('/*py_libhip_search_path*/', libhip_path, 1)
        mod = compile_module_from_src(src=src, name="hip_utils", include_dirs=include_dirs)
        self.load_binary = mod.load_binary
        self.unload_module = mod.unload_module
        self.get_device_properties = mod.get_device_properties
        self.create_tdm_descriptor = mod.create_tdm_descriptor
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        global PyTDMDescriptor
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyTDMDescriptor = mod.PyTDMDescriptor
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE


# -------------------- Launcher ----------------------------
def ty_to_cpp(ty):
    if ty.startswith('*'):
        return "hipDeviceptr_t"
    if ty == "tensordesc":
        return "TDMDescriptor"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


def make_tensordesc_arg(arg, tensordesc_metadata, base_args):
    """
    Translate a tensor descriptor argument into the appropriate list of kernel
    arguments. If `tensordesc_metadata` is provided, we will create a
    TDMDescriptor object. Otherwise, we decompose the tensor descriptor into
    base pointer, shape, strides, and padding flag. In both cases, we append the
    shape and strides at the end to match the expected kernel signature.
    """

    if tensordesc_metadata is None:
        return decompose_descriptor(arg)

    kernel_metadata = base_args[8]
    shape = arg.shape
    strides = arg.strides
    base = arg.base.data_ptr()

    assert "elem_bits" in tensordesc_metadata and "block_size" in tensordesc_metadata
    elem_bits = tensordesc_metadata["elem_bits"]
    block_size = tensordesc_metadata["block_size"]
    pad_interval, pad_amount = 0, 0
    interval_padding_pairs = tensordesc_metadata.get("interval_padding_pairs", [])
    if interval_padding_pairs:
        assert len(interval_padding_pairs) == 1 and len(interval_padding_pairs[0]) == 2
        pad_interval, pad_amount = interval_padding_pairs[0]
    num_warps = kernel_metadata[0]

    driver = triton.runtime.driver.active
    assert isinstance(driver, HIPDriver)

    desc = driver.utils.create_tdm_descriptor(elem_bits, block_size, num_warps, pad_interval, pad_amount, shape,
                                              strides, base)

    return [desc, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    """
    Wrap a kernel launcher function to handle tensor descriptor arguments.
    Use the provided `tensordesc_metadata` to determine whether to create
    TDMDescriptor objects or decompose the tensor descriptors.

    Args:
        launcher (callable): The original kernel launcher function.
        signature (Dict[int, str]): The kernel signature mapping argument indices to types.
        tensordesc_metadata (List[Dict] or None): The list of tensor descriptor metadata, following the order
                                                  of tensor descriptor arguments. If None, decompose tensor descriptors.
    Returns:
        launcher (callable): The wrapped kernel launcher function.
    """

    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg)


class HIPLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta, "tensordesc")
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.warp_size = metadata.warp_size
        # Check if cooperative groups are supported on the device.
        if self.launch_cooperative_grid:
            driver = triton.runtime.driver.active
            assert isinstance(driver, HIPDriver)
            device = driver.get_current_device()
            device_properties = driver.utils.get_device_properties(device)
            assert device_properties['cooperativeLaunch'], \
                "Cooperative launch requested but not supported by device"
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        active_driver = triton.runtime.driver.active

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        def allocate_default_profile_scratch(size, align):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * size
                return active_driver.allocate_default_profile_scratch(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        if _allocation.has_profile_allocator():
            profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                               _allocation._profile_allocator)
        else:
            profile_scratch = allocate_default_profile_scratch(self.profile_scratch_size, self.profile_scratch_align)

        self.launch(self.launch_cooperative_grid, gridX, gridY, gridZ, stream, function, global_scratch,
                    profile_scratch, kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook,
                    self.warp_size, self.arg_annotations, self.kernel_signature, args)


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.launcher_cls = HIPLauncher

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is not None)
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties['arch']
        warp_size = device_properties['warpSize']
        return GPUTarget("hip", arch.split(':')[0], warp_size)

    def get_active_torch_device(self):
        import torch
        # when using hip devices, the device string in pytorch is "cuda"
        return torch.device("cuda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
<|endoftext|># Copyright (c) 2017-present, Gregory Szorc
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license. See the LICENSE file for details.

# ruff: noqa: F403, F405

"""Python interface to the Zstandard (zstd) compression library."""

from __future__ import absolute_import, unicode_literals

# This module serves 2 roles:
#
# 1) Export the C or CFFI "backend" through a central module.
# 2) Implement additional functionality built on top of C or CFFI backend.
import builtins
import io
import os
import platform
import sys

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing import ByteString as Buffer

# Some Python implementations don't support C extensions. That's why we have
# a CFFI implementation in the first place. The code here import one of our
# "backends" then re-exports the symbols from this module. For convenience,
# we support falling back to the CFFI backend if the C extension can't be
# imported. But for performance reasons, we only do this on unknown Python
# implementation. Notably, for CPython we require the C extension by default.
# Because someone will inevitably want special behavior, the behavior is
# configurable via an environment variable. A potentially better way to handle
# this is to import a special ``__importpolicy__`` module or something
# defining a variable and `setup.py` could write the file with whatever
# policy was specified at build time. Until someone needs it, we go with
# the hacky but simple environment variable approach.
_module_policy = os.environ.get(
    "PYTHON_ZSTANDARD_IMPORT_POLICY", "default"
).strip()

if _module_policy == "default":
    if platform.python_implementation() in ("CPython",):
        from .backend_c import *  # type: ignore

        backend = "cext"
    elif platform.python_implementation() in ("PyPy",):
        from .backend_cffi import *  # type: ignore

        backend = "cffi"
    else:
        try:
            from .backend_c import *

            backend = "cext"
        except ImportError:
            from .backend_cffi import *

            backend = "cffi"
elif _module_policy == "cffi_fallback":
    try:
        from .backend_c import *

        backend = "cext"
    except ImportError:
        from .backend_cffi import *

        backend = "cffi"
elif _module_policy == "rust":
    from .backend_rust import *  # type: ignore

    backend = "rust"
elif _module_policy == "cext":
    from .backend_c import *

    backend = "cext"
elif _module_policy == "cffi":
    from .backend_cffi import *

    backend = "cffi"
else:
    raise ImportError(
        "unknown module import policy: %s; use default, cffi_fallback, "
        "cext, or cffi" % _module_policy
    )

# Keep this in sync with python-zstandard.h, rust-ext/src/lib.rs, and debian/changelog.
__version__ = "0.25.0"

_MODE_CLOSED = 0
_MODE_READ = 1
_MODE_WRITE = 2


def open(
    filename,
    mode="rb",
    cctx=None,
    dctx=None,
    encoding=None,
    errors=None,
    newline=None,
    closefd=None,
):
    """Create a file object with zstd (de)compression.

    The object returned from this function will be a
    :py:class:`ZstdDecompressionReader` if opened for reading in binary mode,
    a :py:class:`ZstdCompressionWriter` if opened for writing in binary mode,
    or an ``io.TextIOWrapper`` if opened for reading or writing in text mode.

    :param filename:
       ``bytes``, ``str``, or ``os.PathLike`` defining a file to open or a
       file object (with a ``read()`` or ``write()`` method).
    :param mode:
       ``str`` File open mode. Accepts any of the open modes recognized by
       ``open()``.
    :param cctx:
       ``ZstdCompressor`` to use for compression. If not specified and file
       is opened for writing, the default ``ZstdCompressor`` will be used.
    :param dctx:
       ``ZstdDecompressor`` to use for decompression. If not specified and file
       is opened for reading, the default ``ZstdDecompressor`` will be used.
    :param encoding:
        ``str`` that defines text encoding to use when file is opened in text
        mode.
    :param errors:
       ``str`` defining text encoding error handling mode.
    :param newline:
       ``str`` defining newline to use in text mode.
    :param closefd:
       ``bool`` whether to close the file when the returned object is closed.
        Only used if a file object is passed. If a filename is specified, the
        opened file is always closed when the returned object is closed.
    """
    normalized_mode = mode.replace("t", "")

    if normalized_mode in ("r", "rb"):
        dctx = dctx or ZstdDecompressor()
        open_mode = "r"
        raw_open_mode = "rb"
    elif normalized_mode in ("w", "wb", "a", "ab", "x", "xb"):
        cctx = cctx or ZstdCompressor()
        open_mode = "w"
        raw_open_mode = normalized_mode
        if not raw_open_mode.endswith("b"):
            raw_open_mode = raw_open_mode + "b"
    else:
        raise ValueError("Invalid mode: {!r}".format(mode))

    if hasattr(os, "PathLike"):
        types = (str, bytes, os.PathLike)
    else:
        types = (str, bytes)

    if isinstance(filename, types):  # type: ignore
        inner_fh = builtins.open(filename, raw_open_mode)
        closefd = True
    elif hasattr(filename, "read") or hasattr(filename, "write"):
        inner_fh = filename
        closefd = bool(closefd)
    else:
        raise TypeError(
            "filename must be a str, bytes, file or PathLike object"
        )

    if open_mode == "r":
        fh = dctx.stream_reader(inner_fh, closefd=closefd)
    elif open_mode == "w":
        fh = cctx.stream_writer(inner_fh, closefd=closefd)
    else:
        raise RuntimeError("logic error in zstandard.open() handling open mode")

    if "b" not in normalized_mode:
        return io.TextIOWrapper(
            fh, encoding=encoding, errors=errors, newline=newline
        )
    else:
        return fh


def compress(data: Buffer, level: int = 3) -> bytes:
    """Compress source data using the zstd compression format.

    This performs one-shot compression using basic/default compression
    settings.

    This method is provided for convenience and is equivalent to calling
    ``ZstdCompressor(level=level).compress(data)``.

    If you find yourself calling this function in a tight loop,
    performance will be greater if you construct a single ``ZstdCompressor``
    and repeatedly call ``compress()`` on it.
    """
    cctx = ZstdCompressor(level=level)

    return cctx.compress(data)


def decompress(data: Buffer, max_output_size: int = 0) -> bytes:
    """Decompress a zstd frame into its original data.

    This performs one-shot decompression using basic/default compression
    settings.

    This method is provided for convenience and is equivalent to calling
    ``ZstdDecompressor().decompress(data, max_output_size=max_output_size)``.

    If you find yourself calling this function in a tight loop, performance
    will be greater if you construct a single ``ZstdDecompressor`` and
    repeatedly call ``decompress()`` on it.
    """
    dctx = ZstdDecompressor()

    return dctx.decompress(data, max_output_size=max_output_size)
<|endoftext|># Copyright (c) 2016-present, Gregory Szorc
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license. See the LICENSE file for details.

"""Python interface to the Zstandard (zstd) compression library."""

from __future__ import absolute_import, unicode_literals

# This should match what the C extension exports.
__all__ = [
    "BufferSegment",
    "BufferSegments",
    "BufferWithSegments",
    "BufferWithSegmentsCollection",
    "ZstdCompressionChunker",
    "ZstdCompressionDict",
    "ZstdCompressionObj",
    "ZstdCompressionParameters",
    "ZstdCompressionReader",
    "ZstdCompressionWriter",
    "ZstdCompressor",
    "ZstdDecompressionObj",
    "ZstdDecompressionReader",
    "ZstdDecompressionWriter",
    "ZstdDecompressor",
    "ZstdError",
    "FrameParameters",
    "backend_features",
    "estimate_decompression_context_size",
    "frame_content_size",
    "frame_header_size",
    "get_frame_parameters",
    "train_dictionary",
    # Constants.
    "FLUSH_BLOCK",
    "FLUSH_FRAME",
    "COMPRESSOBJ_FLUSH_FINISH",
    "COMPRESSOBJ_FLUSH_BLOCK",
    "ZSTD_VERSION",
    "FRAME_HEADER",
    "CONTENTSIZE_UNKNOWN",
    "CONTENTSIZE_ERROR",
    "MAX_COMPRESSION_LEVEL",
    "COMPRESSION_RECOMMENDED_INPUT_SIZE",
    "COMPRESSION_RECOMMENDED_OUTPUT_SIZE",
    "DECOMPRESSION_RECOMMENDED_INPUT_SIZE",
    "DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE",
    "MAGIC_NUMBER",
    "BLOCKSIZELOG_MAX",
    "BLOCKSIZE_MAX",
    "WINDOWLOG_MIN",
    "WINDOWLOG_MAX",
    "CHAINLOG_MIN",
    "CHAINLOG_MAX",
    "HASHLOG_MIN",
    "HASHLOG_MAX",
    "MINMATCH_MIN",
    "MINMATCH_MAX",
    "SEARCHLOG_MIN",
    "SEARCHLOG_MAX",
    "SEARCHLENGTH_MIN",
    "SEARCHLENGTH_MAX",
    "TARGETLENGTH_MIN",
    "TARGETLENGTH_MAX",
    "LDM_MINMATCH_MIN",
    "LDM_MINMATCH_MAX",
    "LDM_BUCKETSIZELOG_MAX",
    "STRATEGY_FAST",
    "STRATEGY_DFAST",
    "STRATEGY_GREEDY",
    "STRATEGY_LAZY",
    "STRATEGY_LAZY2",
    "STRATEGY_BTLAZY2",
    "STRATEGY_BTOPT",
    "STRATEGY_BTULTRA",
    "STRATEGY_BTULTRA2",
    "DICT_TYPE_AUTO",
    "DICT_TYPE_RAWCONTENT",
    "DICT_TYPE_FULLDICT",
    "FORMAT_ZSTD1",
    "FORMAT_ZSTD1_MAGICLESS",
]

import io
import os

from ._cffi import (  # type: ignore
    ffi,
    lib,
)

backend_features = set()  # type: ignore

COMPRESSION_RECOMMENDED_INPUT_SIZE = lib.ZSTD_CStreamInSize()
COMPRESSION_RECOMMENDED_OUTPUT_SIZE = lib.ZSTD_CStreamOutSize()
DECOMPRESSION_RECOMMENDED_INPUT_SIZE = lib.ZSTD_DStreamInSize()
DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE = lib.ZSTD_DStreamOutSize()

new_nonzero = ffi.new_allocator(should_clear_after_alloc=False)

MAX_COMPRESSION_LEVEL = lib.ZSTD_maxCLevel()
MAGIC_NUMBER = lib.ZSTD_MAGICNUMBER
FRAME_HEADER = b"\x28\xb5\x2f\xfd"
CONTENTSIZE_UNKNOWN = lib.ZSTD_CONTENTSIZE_UNKNOWN
CONTENTSIZE_ERROR = lib.ZSTD_CONTENTSIZE_ERROR
ZSTD_VERSION = (
    lib.ZSTD_VERSION_MAJOR,
    lib.ZSTD_VERSION_MINOR,
    lib.ZSTD_VERSION_RELEASE,
)

BLOCKSIZELOG_MAX = lib.ZSTD_BLOCKSIZELOG_MAX
BLOCKSIZE_MAX = lib.ZSTD_BLOCKSIZE_MAX
WINDOWLOG_MIN = lib.ZSTD_WINDOWLOG_MIN
WINDOWLOG_MAX = lib.ZSTD_WINDOWLOG_MAX
CHAINLOG_MIN = lib.ZSTD_CHAINLOG_MIN
CHAINLOG_MAX = lib.ZSTD_CHAINLOG_MAX
HASHLOG_MIN = lib.ZSTD_HASHLOG_MIN
HASHLOG_MAX = lib.ZSTD_HASHLOG_MAX
MINMATCH_MIN = lib.ZSTD_MINMATCH_MIN
MINMATCH_MAX = lib.ZSTD_MINMATCH_MAX
SEARCHLOG_MIN = lib.ZSTD_SEARCHLOG_MIN
SEARCHLOG_MAX = lib.ZSTD_SEARCHLOG_MAX
SEARCHLENGTH_MIN = lib.ZSTD_MINMATCH_MIN
SEARCHLENGTH_MAX = lib.ZSTD_MINMATCH_MAX
TARGETLENGTH_MIN = lib.ZSTD_TARGETLENGTH_MIN
TARGETLENGTH_MAX = lib.ZSTD_TARGETLENGTH_MAX
LDM_MINMATCH_MIN = lib.ZSTD_LDM_MINMATCH_MIN
LDM_MINMATCH_MAX = lib.ZSTD_LDM_MINMATCH_MAX
LDM_BUCKETSIZELOG_MAX = lib.ZSTD_LDM_BUCKETSIZELOG_MAX

STRATEGY_FAST = lib.ZSTD_fast
STRATEGY_DFAST = lib.ZSTD_dfast
STRATEGY_GREEDY = lib.ZSTD_greedy
STRATEGY_LAZY = lib.ZSTD_lazy
STRATEGY_LAZY2 = lib.ZSTD_lazy2
STRATEGY_BTLAZY2 = lib.ZSTD_btlazy2
STRATEGY_BTOPT = lib.ZSTD_btopt
STRATEGY_BTULTRA = lib.ZSTD_btultra
STRATEGY_BTULTRA2 = lib.ZSTD_btultra2

DICT_TYPE_AUTO = lib.ZSTD_dct_auto
DICT_TYPE_RAWCONTENT = lib.ZSTD_dct_rawContent
DICT_TYPE_FULLDICT = lib.ZSTD_dct_fullDict

FORMAT_ZSTD1 = lib.ZSTD_f_zstd1
FORMAT_ZSTD1_MAGICLESS = lib.ZSTD_f_zstd1_magicless

FLUSH_BLOCK = 0
FLUSH_FRAME = 1

COMPRESSOBJ_FLUSH_FINISH = 0
COMPRESSOBJ_FLUSH_BLOCK = 1


def _cpu_count():
    # os.cpu_count() was introducd in Python 3.4.
    try:
        return os.cpu_count() or 0
    except AttributeError:
        pass

    # Linux.
    try:
        return os.sysconf("SC_NPROCESSORS_ONLN")
    except (AttributeError, ValueError):
        pass

    # TODO implement on other platforms.
    return 0


class BufferSegment:
    """Represents a segment within a ``BufferWithSegments``.

    This type is essentially a reference to N bytes within a
    ``BufferWithSegments``.

    The object conforms to the buffer protocol.
    """

    @property
    def offset(self):
        """The byte offset of this segment within its parent buffer."""
        raise NotImplementedError()

    def __len__(self):
        """Obtain the length of the segment, in bytes."""
        raise NotImplementedError()

    def tobytes(self):
        """Obtain bytes copy of this segment."""
        raise NotImplementedError()


class BufferSegments:
    """Represents an array of ``(offset, length)`` integers.

    This type is effectively an index used by :py:class:`BufferWithSegments`.

    The array members are 64-bit unsigned integers using host/native bit order.

    Instances conform to the buffer protocol.
    """


class BufferWithSegments:
    """A memory buffer containing N discrete items of known lengths.

    This type is essentially a fixed size memory address and an array
    of 2-tuples of ``(offset, length)`` 64-bit unsigned native-endian
    integers defining the byte offset and length of each segment within
    the buffer.

    Instances behave like containers.

    Instances also conform to the buffer protocol. So a reference to the
    backing bytes can be obtained via ``memoryview(o)``. A *copy* of the
    backing bytes can be obtained via ``.tobytes()``.

    This type exists to facilitate operations against N>1 items without
    the overhead of Python object creation and management. Used with
    APIs like :py:meth:`ZstdDecompressor.multi_decompress_to_buffer`, it
    is possible to decompress many objects in parallel without the GIL
    held, leading to even better performance.
    """

    @property
    def size(self):
        """Total sizein bytes of the backing buffer."""
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        """Obtains a segment within the buffer.

        The returned object references memory within this buffer.

        :param i:
           Integer index of segment to retrieve.
        :return:
           :py:class:`BufferSegment`
        """
        raise NotImplementedError()

    def segments(self):
        """Obtain the array of ``(offset, length)`` segments in the buffer.

        :return:
           :py:class:`BufferSegments`
        """
        raise NotImplementedError()

    def tobytes(self):
        """Obtain bytes copy of this instance."""
        raise NotImplementedError()


class BufferWithSegmentsCollection:
    """A virtual spanning view over multiple BufferWithSegments.

    Instances are constructed from 1 or more :py:class:`BufferWithSegments`
    instances. The resulting object behaves like an ordered sequence whose
    members are the segments within each ``BufferWithSegments``.

    If the object is composed of 2 ``BufferWithSegments`` instances with the
    first having 2 segments and the second have 3 segments, then ``b[0]``
    and ``b[1]`` access segments in the first object and ``b[2]``, ``b[3]``,
    and ``b[4]`` access segments from the second.
    """

    def __len__(self):
        """The number of segments within all ``BufferWithSegments``."""
        raise NotImplementedError()

    def __getitem__(self, i):
        """Obtain the ``BufferSegment`` at an offset."""
        raise NotImplementedError()


class ZstdError(Exception):
    pass


def _zstd_error(zresult):
    # Resolves to bytes on Python 2 and 3. We use the string for formatting
    # into error messages, which will be literal unicode. So convert it to
    # unicode.
    return ffi.string(lib.ZSTD_getErrorName(zresult)).decode("utf-8")


def _make_cctx_params(params):
    res = lib.ZSTD_createCCtxParams()
    if res == ffi.NULL:
        raise MemoryError()

    res = ffi.gc(res, lib.ZSTD_freeCCtxParams)

    attrs = [
        (lib.ZSTD_c_format, params.format),
        (lib.ZSTD_c_compressionLevel, params.compression_level),
        (lib.ZSTD_c_windowLog, params.window_log),
        (lib.ZSTD_c_hashLog, params.hash_log),
        (lib.ZSTD_c_chainLog, params.chain_log),
        (lib.ZSTD_c_searchLog, params.search_log),
        (lib.ZSTD_c_minMatch, params.min_match),
        (lib.ZSTD_c_targetLength, params.target_length),
        (lib.ZSTD_c_strategy, params.strategy),
        (lib.ZSTD_c_contentSizeFlag, params.write_content_size),
        (lib.ZSTD_c_checksumFlag, params.write_checksum),
        (lib.ZSTD_c_dictIDFlag, params.write_dict_id),
        (lib.ZSTD_c_nbWorkers, params.threads),
        (lib.ZSTD_c_jobSize, params.job_size),
        (lib.ZSTD_c_overlapLog, params.overlap_log),
        (lib.ZSTD_c_forceMaxWindow, params.force_max_window),
        (lib.ZSTD_c_enableLongDistanceMatching, params.enable_ldm),
        (lib.ZSTD_c_ldmHashLog, params.ldm_hash_log),
        (lib.ZSTD_c_ldmMinMatch, params.ldm_min_match),
        (lib.ZSTD_c_ldmBucketSizeLog, params.ldm_bucket_size_log),
        (lib.ZSTD_c_ldmHashRateLog, params.ldm_hash_rate_log),
    ]

    for param, value in attrs:
        _set_compression_parameter(res, param, value)

    return res


class ZstdCompressionParameters(object):
    """Low-level zstd compression parameters.

    This type represents a collection of parameters to control how zstd
    compression is performed.

    Instances can be constructed from raw parameters or derived from a
    base set of defaults specified from a compression level (recommended)
    via :py:meth:`ZstdCompressionParameters.from_level`.

    >>> # Derive compression settings for compression level 7.
    >>> params = zstandard.ZstdCompressionParameters.from_level(7)

    >>> # With an input size of 1MB
    >>> params = zstandard.ZstdCompressionParameters.from_level(7, source_size=1048576)

    Using ``from_level()``, it is also possible to override individual compression
    parameters or to define additional settings that aren't automatically derived.
    e.g.:

    >>> params = zstandard.ZstdCompressionParameters.from_level(4, window_log=10)
    >>> params = zstandard.ZstdCompressionParameters.from_level(5, threads=4)

    Or you can define low-level compression settings directly:

    >>> params = zstandard.ZstdCompressionParameters(window_log=12, enable_ldm=True)

    Once a ``ZstdCompressionParameters`` instance is obtained, it can be used to
    configure a compressor:

    >>> cctx = zstandard.ZstdCompressor(compression_params=params)

    Some of these are very low-level settings. It may help to consult the official
    zstandard documentation for their behavior. Look for the ``ZSTD_p_*`` constants
    in ``zstd.h`` (https://github.com/facebook/zstd/blob/dev/lib/zstd.h).
    """

    @staticmethod
    def from_level(level, source_size=0, dict_size=0, **kwargs):
        """Create compression parameters from a compression level.

        :param level:
           Integer compression level.
        :param source_size:
           Integer size in bytes of source to be compressed.
        :param dict_size:
           Integer size in bytes of compression dictionary to use.
        :return:
           :py:class:`ZstdCompressionParameters`
        """
        params = lib.ZSTD_getCParams(level, source_size, dict_size)

        args = {
            "window_log": "windowLog",
            "chain_log": "chainLog",
            "hash_log": "hashLog",
            "search_log": "searchLog",
            "min_match": "minMatch",
            "target_length": "targetLength",
            "strategy": "strategy",
        }

        for arg, attr in args.items():
            if arg not in kwargs:
                kwargs[arg] = getattr(params, attr)

        return ZstdCompressionParameters(**kwargs)

    def __init__(
        self,
        format=0,
        compression_level=0,
        window_log=0,
        hash_log=0,
        chain_log=0,
        search_log=0,
        min_match=0,
        target_length=0,
        strategy=-1,
        write_content_size=1,
        write_checksum=0,
        write_dict_id=0,
        job_size=0,
        overlap_log=-1,
        force_max_window=0,
        enable_ldm=0,
        ldm_hash_log=0,
        ldm_min_match=0,
        ldm_bucket_size_log=0,
        ldm_hash_rate_log=-1,
        threads=0,
    ):
        params = lib.ZSTD_createCCtxParams()
        if params == ffi.NULL:
            raise MemoryError()

        params = ffi.gc(params, lib.ZSTD_freeCCtxParams)

        self._params = params

        if threads < 0:
            threads = _cpu_count()

        # We need to set ZSTD_c_nbWorkers before ZSTD_c_jobSize and ZSTD_c_overlapLog
        # because setting ZSTD_c_nbWorkers resets the other parameters.
        _set_compression_parameter(params, lib.ZSTD_c_nbWorkers, threads)

        _set_compression_parameter(params, lib.ZSTD_c_format, format)
        _set_compression_parameter(
            params, lib.ZSTD_c_compressionLevel, compression_level
        )
        _set_compression_parameter(params, lib.ZSTD_c_windowLog, window_log)
        _set_compression_parameter(params, lib.ZSTD_c_hashLog, hash_log)
        _set_compression_parameter(params, lib.ZSTD_c_chainLog, chain_log)
        _set_compression_parameter(params, lib.ZSTD_c_searchLog, search_log)
        _set_compression_parameter(params, lib.ZSTD_c_minMatch, min_match)
        _set_compression_parameter(
            params, lib.ZSTD_c_targetLength, target_length
        )

        if strategy == -1:
            strategy = 0

        _set_compression_parameter(params, lib.ZSTD_c_strategy, strategy)
        _set_compression_parameter(
            params, lib.ZSTD_c_contentSizeFlag, write_content_size
        )
        _set_compression_parameter(
            params, lib.ZSTD_c_checksumFlag, write_checksum
        )
        _set_compression_parameter(params, lib.ZSTD_c_dictIDFlag, write_dict_id)
        _set_compression_parameter(params, lib.ZSTD_c_jobSize, job_size)

        if overlap_log == -1:
            overlap_log = 0

        _set_compression_parameter(params, lib.ZSTD_c_overlapLog, overlap_log)
        _set_compression_parameter(
            params, lib.ZSTD_c_forceMaxWindow, force_max_window
        )
        _set_compression_parameter(
            params, lib.ZSTD_c_enableLongDistanceMatching, enable_ldm
        )
        _set_compression_parameter(params, lib.ZSTD_c_ldmHashLog, ldm_hash_log)
        _set_compression_parameter(
            params, lib.ZSTD_c_ldmMinMatch, ldm_min_match
        )
        _set_compression_parameter(
            params, lib.ZSTD_c_ldmBucketSizeLog, ldm_bucket_size_log
        )

        if ldm_hash_rate_log == -1:
            ldm_hash_rate_log = 0

        _set_compression_parameter(
            params, lib.ZSTD_c_ldmHashRateLog, ldm_hash_rate_log
        )

    @property
    def format(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_format)

    @property
    def compression_level(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_compressionLevel
        )

    @property
    def window_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_windowLog)

    @property
    def hash_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_hashLog)

    @property
    def chain_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_chainLog)

    @property
    def search_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_searchLog)

    @property
    def min_match(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_minMatch)

    @property
    def target_length(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_targetLength)

    @property
    def strategy(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_strategy)

    @property
    def write_content_size(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_contentSizeFlag
        )

    @property
    def write_checksum(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_checksumFlag)

    @property
    def write_dict_id(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_dictIDFlag)

    @property
    def job_size(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_jobSize)

    @property
    def overlap_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_overlapLog)

    @property
    def force_max_window(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_forceMaxWindow
        )

    @property
    def enable_ldm(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_enableLongDistanceMatching
        )

    @property
    def ldm_hash_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmHashLog)

    @property
    def ldm_min_match(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmMinMatch)

    @property
    def ldm_bucket_size_log(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_ldmBucketSizeLog
        )

    @property
    def ldm_hash_rate_log(self):
        return _get_compression_parameter(
            self._params, lib.ZSTD_c_ldmHashRateLog
        )

    @property
    def threads(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_nbWorkers)

    def estimated_compression_context_size(self):
        """Estimated size in bytes needed to compress with these parameters."""
        return lib.ZSTD_estimateCCtxSize_usingCCtxParams(self._params)


def estimate_decompression_context_size():
    """Estimate the memory size requirements for a decompressor instance.

    :return:
       Integer number of bytes.
    """
    return lib.ZSTD_estimateDCtxSize()


def _set_compression_parameter(params, param, value):
    zresult = lib.ZSTD_CCtxParams_setParameter(params, param, value)
    if lib.ZSTD_isError(zresult):
        raise ZstdError(
            "unable to set compression context parameter: %s"
            % _zstd_error(zresult)
        )


def _get_compression_parameter(params, param):
    result = ffi.new("int *")

    zresult = lib.ZSTD_CCtxParams_getParameter(params, param, result)
    if lib.ZSTD_isError(zresult):
        raise ZstdError(
            "unable to get compression context parameter: %s"
            % _zstd_error(zresult)
        )

    return result[0]


class ZstdCompressionWriter(object):
    """Writable compressing stream wrapper.

    ``ZstdCompressionWriter`` is a write-only stream interface for writing
    compressed data to another stream.

    This type conforms to the ``io.RawIOBase`` interface and should be usable
    by any type that operates against a *file-object* (``typing.BinaryIO``
    in Python type hinting speak). Only methods that involve writing will do
    useful things.

    As data is written to this stream (e.g. via ``write()``), that data
    is sent to the compressor. As compressed data becomes available from
    the compressor, it is sent to the underlying stream by calling its
    ``write()`` method.

    Both ``write()`` and ``flush()`` return the number of bytes written to the
    object's ``write()``. In many cases, small inputs do not accumulate enough
    data to cause a write and ``write()`` will return ``0``.

    Calling ``close()`` will mark the stream as closed and subsequent I/O
    operations will raise ``ValueError`` (per the documented behavior of
    ``io.RawIOBase``). ``close()`` will also call ``close()`` on the underlying
    stream if such a method exists and the instance was constructed with
    ``closefd=True``

    Instances are obtained by calling :py:meth:`ZstdCompressor.stream_writer`.

    Typically usage is as follows:

    >>> cctx = zstandard.ZstdCompressor(level=10)
    >>> compressor = cctx.stream_writer(fh)
    >>> compressor.write(b"chunk 0\\n")
    >>> compressor.write(b"chunk 1\\n")
    >>> compressor.flush()
    >>> # Receiver will be able to decode ``chunk 0\\nchunk 1\\n`` at this point.
    >>> # Receiver is also expecting more data in the zstd *frame*.
    >>>
    >>> compressor.write(b"chunk 2\\n")
    >>> compressor.flush(zstandard.FLUSH_FRAME)
    >>> # Receiver will be able to decode ``chunk 0\\nchunk 1\\nchunk 2``.
    >>> # Receiver is expecting no more data, as the zstd frame is closed.
    >>> # Any future calls to ``write()`` at this point will construct a new
    >>> # zstd frame.

    Instances can be used as context managers. Exiting the context manager is
    the equivalent of calling ``close()``, which is equivalent to calling
    ``flush(zstandard.FLUSH_FRAME)``:

    >>> cctx = zstandard.ZstdCompressor(level=10)
    >>> with cctx.stream_writer(fh) as compressor:
    ...     compressor.write(b'chunk 0')
    ...     compressor.write(b'chunk 1')
    ...     ...

    .. important::

       If ``flush(FLUSH_FRAME)`` is not called, emitted data doesn't
       constitute a full zstd *frame* and consumers of this data may complain
       about malformed input. It is recommended to use instances as a context
       manager to ensure *frames* are properly finished.

    If the size of the data being fed to this streaming compressor is known,
    you can declare it before compression begins:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh, size=data_len) as compressor:
    ...     compressor.write(chunk0)
    ...     compressor.write(chunk1)
    ...     ...

    Declaring the size of the source data allows compression parameters to
    be tuned. And if ``write_content_size`` is used, it also results in the
    content size being written into the frame header of the output data.

    The size of chunks being ``write()`` to the destination can be specified:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh, write_size=32768) as compressor:
    ...     ...

    To see how much memory is being used by the streaming compressor:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh) as compressor:
    ...     ...
    ...     byte_size = compressor.memory_size()

    Thte total number of bytes written so far are exposed via ``tell()``:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh) as compressor:
    ...     ...
    ...     total_written = compressor.tell()

    ``stream_writer()`` accepts a ``write_return_read`` boolean argument to
    control the return value of ``write()``. When ``False`` (the default),
    ``write()`` returns the number of bytes that were ``write()``'en to the
    underlying object. When ``True``, ``write()`` returns the number of bytes
    read from the input that were subsequently written to the compressor.
    ``True`` is the *proper* behavior for ``write()`` as specified by the
    ``io.RawIOBase`` interface and will become the default value in a future
    release.
    """

    def __init__(
        self,
        compressor,
        writer,
        source_size,
        write_size,
        write_return_read,
        closefd=True,
    ):
        self._compressor = compressor
        self._writer = writer
        self._write_size = write_size
        self._write_return_read = bool(write_return_read)
        self._closefd = bool(closefd)
        self._entered = False
        self._closing = False
        self._closed = False
        self._bytes_compressed = 0

        self._dst_buffer = ffi.new("char[]", write_size)
        self._out_buffer = ffi.new("ZSTD_outBuffer *")
        self._out_buffer.dst = self._dst_buffer
        self._out_buffer.size = len(self._dst_buffer)
        self._out_buffer.pos = 0

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(compressor._cctx, source_size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

    def __enter__(self):
        if self._closed:
            raise ValueError("stream is closed")

        if self._entered:
            raise ZstdError("cannot __enter__ multiple times")

        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self.close()
        self._compressor = None

        return False

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    def memory_size(self):
        return lib.ZSTD_sizeof_CCtx(self._compressor._cctx)

    def fileno(self):
        f = getattr(self._writer, "fileno", None)
        if f:
            return f()
        else:
            raise OSError("fileno not available on underlying writer")

    def close(self):
        if self._closed:
            return

        try:
            self._closing = True
            self.flush(FLUSH_FRAME)
        finally:
            self._closing = False
            self._closed = True

        # Call close() on underlying stream as well.
        f = getattr(self._writer, "close", None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def isatty(self):
        return False

    def readable(self):
        return False

    def readline(self, size=-1):
        raise io.UnsupportedOperation()

    def readlines(self, hint=-1):
        raise io.UnsupportedOperation()

    def seek(self, offset, whence=None):
        raise io.UnsupportedOperation()

    def seekable(self):
        return False

    def truncate(self, size=None):
        raise io.UnsupportedOperation()

    def writable(self):
        return True

    def writelines(self, lines):
        raise NotImplementedError("writelines() is not yet implemented")

    def read(self, size=-1):
        raise io.UnsupportedOperation()

    def readall(self):
        raise io.UnsupportedOperation()

    def readinto(self, b):
        raise io.UnsupportedOperation()

    def write(self, data):
        """Send data to the compressor and possibly to the inner stream."""
        if self._closed:
            raise ValueError("stream is closed")

        total_write = 0

        data_buffer = ffi.from_buffer(data)

        in_buffer = ffi.new("ZSTD_inBuffer *")
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0

        out_buffer = self._out_buffer
        out_buffer.pos = 0

        while in_buffer.pos < in_buffer.size:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx,
                out_buffer,
                in_buffer,
                lib.ZSTD_e_continue,
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if out_buffer.pos:
                self._writer.write(
                    ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                )
                total_write += out_buffer.pos
                self._bytes_compressed += out_buffer.pos
                out_buffer.pos = 0

        if self._write_return_read:
            return in_buffer.pos
        else:
            return total_write

    def flush(self, flush_mode=FLUSH_BLOCK):
        """Evict data from compressor's internal state and write it to inner stream.

        Calling this method may result in 0 or more ``write()`` calls to the
        inner stream.

        This method will also call ``flush()`` on the inner stream, if such a
        method exists.

        :param flush_mode:
           How to flush the zstd compressor.

           ``zstandard.FLUSH_BLOCK`` will flush data already sent to the
           compressor but not emitted to the inner stream. The stream is still
           writable after calling this. This is the default behavior.

           See documentation for other ``zstandard.FLUSH_*`` constants for more
           flushing options.
        :return:
           Integer number of bytes written to the inner stream.
        """

        if flush_mode == FLUSH_BLOCK:
            flush = lib.ZSTD_e_flush
        elif flush_mode == FLUSH_FRAME:
            flush = lib.ZSTD_e_end
        else:
            raise ValueError("unknown flush_mode: %r" % flush_mode)

        if self._closed:
            raise ValueError("stream is closed")

        total_write = 0

        out_buffer = self._out_buffer
        out_buffer.pos = 0

        in_buffer = ffi.new("ZSTD_inBuffer *")
        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0

        while True:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, out_buffer, in_buffer, flush
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if out_buffer.pos:
                self._writer.write(
                    ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                )
                total_write += out_buffer.pos
                self._bytes_compressed += out_buffer.pos
                out_buffer.pos = 0

            if not zresult:
                break

        f = getattr(self._writer, "flush", None)
        if f and not self._closing:
            f()

        return total_write

    def tell(self):
        return self._bytes_compressed


class ZstdCompressionObj(object):
    """A compressor conforming to the API in Python's standard library.

    This type implements an API similar to compression types in Python's
    standard library such as ``zlib.compressobj`` and ``bz2.BZ2Compressor``.
    This enables existing code targeting the standard library API to swap
    in this type to achieve zstd compression.

    .. important::

       The design of this API is not ideal for optimal performance.

       The reason performance is not optimal is because the API is limited to
       returning a single buffer holding compressed data. When compressing
       data, we don't know how much data will be emitted. So in order to
       capture all this data in a single buffer, we need to perform buffer
       reallocations and/or extra memory copies. This can add significant
       overhead depending on the size or nature of the compressed data how
       much your application calls this type.

       If performance is critical, consider an API like
       :py:meth:`ZstdCompressor.stream_reader`,
       :py:meth:`ZstdCompressor.stream_writer`,
       :py:meth:`ZstdCompressor.chunker`, or
       :py:meth:`ZstdCompressor.read_to_iter`, which result in less overhead
       managing buffers.

    Instances are obtained by calling :py:meth:`ZstdCompressor.compressobj`.

    Here is how this API should be used:

    >>> cctx = zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj()
    >>> data = cobj.compress(b"raw input 0")
    >>> data = cobj.compress(b"raw input 1")
    >>> data = cobj.flush()

    Or to flush blocks:

    >>> cctx.zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj()
    >>> data = cobj.compress(b"chunk in first block")
    >>> data = cobj.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK)
    >>> data = cobj.compress(b"chunk in second block")
    >>> data = cobj.flush()

    For best performance results, keep input chunks under 256KB. This avoids
    extra allocations for a large output object.

    It is possible to declare the input size of the data that will be fed
    into the compressor:

    >>> cctx = zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj(size=6)
    >>> data = cobj.compress(b"foobar")
    >>> data = cobj.flush()
    """

    def __init__(
        self, compressor, write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE
    ):
        self._compressor = compressor
        self._out = ffi.new("ZSTD_outBuffer *")
        self._dst_buffer = ffi.new("char[]", write_size)
        self._out.dst = self._dst_buffer
        self._out.size = write_size
        self._out.pos = 0
        self._finished = False

    def compress(self, data):
        """Send data to the compressor.

        This method receives bytes to feed to the compressor and returns
        bytes constituting zstd compressed data.

        The zstd compressor accumulates bytes and the returned bytes may be
        substantially smaller or larger than the size of the input data on
        any given call. The returned value may be the empty byte string
        (``b""``).

        :param data:
           Data to write to the compressor.
        :return:
           Compressed data.
        """
        if self._finished:
            raise ZstdError("cannot call compress() after compressor finished")

        data_buffer = ffi.from_buffer(data)
        source = ffi.new("ZSTD_inBuffer *")
        source.src = data_buffer
        source.size = len(data_buffer)
        source.pos = 0

        chunks = []

        while source.pos < len(data):
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, self._out, source, lib.ZSTD_e_continue
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if self._out.pos:
                chunks.append(ffi.buffer(self._out.dst, self._out.pos)[:])
                self._out.pos = 0

        return b"".join(chunks)

    def flush(self, flush_mode=COMPRESSOBJ_FLUSH_FINISH):
        """Emit data accumulated in the compressor that hasn't been outputted yet.

        The ``flush_mode`` argument controls how to end the stream.

        ``zstandard.COMPRESSOBJ_FLUSH_FINISH`` (the default) ends the
        compression stream and finishes a zstd frame. Once this type of flush
        is performed, ``compress()`` and ``flush()`` can no longer be called.
        This type of flush **must** be called to end the compression context. If
        not called, the emitted data may be incomplete and may not be readable
        by a decompressor.

        ``zstandard.COMPRESSOBJ_FLUSH_BLOCK`` will flush a zstd block. This
        ensures that all data fed to this instance will have been omitted and
        can be decoded by a decompressor. Flushes of this type can be performed
        multiple times. The next call to ``compress()`` will begin a new zstd
        block.

        :param flush_mode:
           How to flush the zstd compressor.
        :return:
           Compressed data.
        """
        if flush_mode not in (
            COMPRESSOBJ_FLUSH_FINISH,
            COMPRESSOBJ_FLUSH_BLOCK,
        ):
            raise ValueError("flush mode not recognized")

        if self._finished:
            raise ZstdError("compressor object already finished")

        if flush_mode == COMPRESSOBJ_FLUSH_BLOCK:
            z_flush_mode = lib.ZSTD_e_flush
        elif flush_mode == COMPRESSOBJ_FLUSH_FINISH:
            z_flush_mode = lib.ZSTD_e_end
            self._finished = True
        else:
            raise ZstdError("unhandled flush mode")

        assert self._out.pos == 0

        in_buffer = ffi.new("ZSTD_inBuffer *")
        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0

        chunks = []

        while True:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, self._out, in_buffer, z_flush_mode
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "error ending compression stream: %s" % _zstd_error(zresult)
                )

            if self._out.pos:
                chunks.append(ffi.buffer(self._out.dst, self._out.pos)[:])
                self._out.pos = 0

            if not zresult:
                break

        return b"".join(chunks)


class ZstdCompressionChunker(object):
    """Compress data to uniformly sized chunks.

    This type allows you to iteratively feed chunks of data into a compressor
    and produce output chunks of uniform size.

    ``compress()``, ``flush()``, and ``finish()`` all return an iterator of
    ``bytes`` instances holding compressed data. The iterator may be empty.
    Callers MUST iterate through all elements of the returned iterator before
    performing another operation on the object or else the compressor's
    internal state may become confused. This can result in an exception being
    raised or malformed data being emitted.

    All chunks emitted by ``compress()`` will have a length of the configured
    chunk size.

    ``flush()`` and ``finish()`` may return a final chunk smaller than
    the configured chunk size.

    Instances are obtained by calling :py:meth:`ZstdCompressor.chunker`.

    Here is how the API should be used:

    >>> cctx = zstandard.ZstdCompressor()
    >>> chunker = cctx.chunker(chunk_size=32768)
    >>>
    >>> with open(path, 'rb') as fh:
    ...     while True:
    ...         in_chunk = fh.read(32768)
    ...         if not in_chunk:
    ...             break
    ...
    ...         for out_chunk in chunker.compress(in_chunk):
    ...             # Do something with output chunk of size 32768.
    ...
    ...     for out_chunk in chunker.finish():
    ...         # Do something with output chunks that finalize the zstd frame.

    This compressor type is often a better alternative to
    :py:class:`ZstdCompressor.compressobj` because it has better performance
    properties.

    ``compressobj()`` will emit output data as it is available. This results
    in a *stream* of output chunks of varying sizes. The consistency of the
    output chunk size with ``chunker()`` is more appropriate for many usages,
    such as sending compressed data to a socket.

    ``compressobj()`` may also perform extra memory reallocations in order
    to dynamically adjust the sizes of the output chunks. Since ``chunker()``
    output chunks are all the same size (except for flushed or final chunks),
    there is less memory allocation/copying overhead.
    """

    def __init__(self, compressor, chunk_size):
        self._compressor = compressor
        self._out = ffi.new("ZSTD_outBuffer *")
        self._dst_buffer = ffi.new("char[]", chunk_size)
        self._out.dst = self._dst_buffer
        self._out.size = chunk_size
        self._out.pos = 0

        self._in = ffi.new("ZSTD_inBuffer *")
        self._in.src = ffi.NULL
        self._in.size = 0
        self._in.pos = 0
        self._finished = False

    def compress(self, data):
        """Feed new input data into the compressor.

        :param data:
           Data to feed to compressor.
        :return:
           Iterator of ``bytes`` representing chunks of compressed data.
        """
        if self._finished:
            raise ZstdError("cannot call compress() after compression finished")

        if self._in.src != ffi.NULL:
            raise ZstdError(
                "cannot perform operation before consuming output "
                "from previous operation"
            )

        data_buffer = ffi.from_buffer(data)

        if not len(data_buffer):
            return

        self._in.src = data_buffer
        self._in.size = len(data_buffer)
        self._in.pos = 0

        while self._in.pos < self._in.size:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, self._out, self._in, lib.ZSTD_e_continue
            )

            if self._in.pos == self._in.size:
                self._in.src = ffi.NULL
                self._in.size = 0
                self._in.pos = 0

            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if self._out.pos == self._out.size:
                yield ffi.buffer(self._out.dst, self._out.pos)[:]
                self._out.pos = 0

    def flush(self):
        """Flushes all data currently in the compressor.

        :return:
           Iterator of ``bytes`` of compressed data.
        """
        if self._finished:
            raise ZstdError("cannot call flush() after compression finished")

        if self._in.src != ffi.NULL:
            raise ZstdError(
                "cannot call flush() before consuming output from "
                "previous operation"
            )

        while True:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, self._out, self._in, lib.ZSTD_e_flush
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if self._out.pos:
                yield ffi.buffer(self._out.dst, self._out.pos)[:]
                self._out.pos = 0

            if not zresult:
                return

    def finish(self):
        """Signals the end of input data.

        No new data can be compressed after this method is called.

        This method will flush buffered data and finish the zstd frame.

        :return:
           Iterator of ``bytes`` of compressed data.
        """
        if self._finished:
            raise ZstdError("cannot call finish() after compression finished")

        if self._in.src != ffi.NULL:
            raise ZstdError(
                "cannot call finish() before consuming output from "
                "previous operation"
            )

        while True:
            zresult = lib.ZSTD_compressStream2(
                self._compressor._cctx, self._out, self._in, lib.ZSTD_e_end
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "zstd compress error: %s" % _zstd_error(zresult)
                )

            if self._out.pos:
                yield ffi.buffer(self._out.dst, self._out.pos)[:]
                self._out.pos = 0

            if not zresult:
                self._finished = True
                return


class ZstdCompressionReader(object):
    """Readable compressing stream wrapper.

    ``ZstdCompressionReader`` is a read-only stream interface for obtaining
    compressed data from a source.

    This type conforms to the ``io.RawIOBase`` interface and should be usable
    by any type that operates against a *file-object* (``typing.BinaryIO``
    in Python type hinting speak).

    Instances are neither writable nor seekable (even if the underlying
    source is seekable). ``readline()`` and ``readlines()`` are not implemented
    because they don't make sense for compressed data. ``tell()`` returns the
    number of compressed bytes emitted so far.

    Instances are obtained by calling :py:meth:`ZstdCompressor.stream_reader`.

    In this example, we open a file for reading and then wrap that file
    handle with a stream from which compressed data can be ``read()``.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     reader = cctx.stream_reader(fh)
    ...     while True:
    ...         chunk = reader.read(16384)
    ...         if not chunk:
    ...             break
    ...
    ...         # Do something with compressed chunk.

    Instances can also be used as context managers:

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     with cctx.stream_reader(fh) as reader:
    ...         while True:
    ...             chunk = reader.read(16384)
    ...             if not chunk:
    ...                 break
    ...
    ...             # Do something with compressed chunk.

    When the context manager exits or ``close()`` is called, the stream is
    closed, underlying resources are released, and future operations against
    the compression stream will fail.

    ``stream_reader()`` accepts a ``size`` argument specifying how large the
    input stream is. This is used to adjust compression parameters so they are
    tailored to the source size. e.g.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     with cctx.stream_reader(fh, size=os.stat(path).st_size) as reader:
    ...         ...

    If the ``source`` is a stream, you can specify how large ``read()``
    requests to that stream should be via the ``read_size`` argument.
    It defaults to ``zstandard.COMPRESSION_RECOMMENDED_INPUT_SIZE``. e.g.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     # Will perform fh.read(8192) when obtaining data to feed into the
    ...     # compressor.
    ...     with cctx.stream_reader(fh, read_size=8192) as reader:
    ...         ...
    """

    def __init__(self, compressor, source, read_size, closefd=True):
        self._compressor = compressor
        self._source = source
        self._read_size = read_size
        self._closefd = closefd
        self._entered = False
        self._closed = False
        self._bytes_compressed = 0
        self._finished_input = False
        self._finished_output = False

        self._in_buffer = ffi.new("ZSTD_inBuffer *")
        # Holds a ref so backing bytes in self._in_buffer stay alive.
        self._source_buffer = None

    def __enter__(self):
        if self._entered:
            raise ValueError("cannot __enter__ multiple times")

        if self._closed:
            raise ValueError("stream is closed")

        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self._compressor = None
        self.close()
        self._source = None

        return False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def readline(self):
        raise io.UnsupportedOperation()

    def readlines(self):
        raise io.UnsupportedOperation()

    def write(self, data):
        raise OSError("stream is not writable")

    def writelines(self, ignored):
        raise OSError("stream is not writable")

    def isatty(self):
        return False

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return

        self._closed = True

        f = getattr(self._source, "close", None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def tell(self):
        return self._bytes_compressed

    def readall(self):
        chunks = []

        while True:
            chunk = self.read(1048576)
            if not chunk:
                break

            chunks.append(chunk)

        return b"".join(chunks)

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    next = __next__

    def _read_input(self):
        if self._finished_input:
            return

        if hasattr(self._source, "read"):
            data = self._source.read(self._read_size)

            if not data:
                self._finished_input = True
                return

            self._source_buffer = ffi.from_buffer(data)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0
        else:
            self._source_buffer = ffi.from_buffer(self._source)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0

    def _compress_into_buffer(self, out_buffer):
        if self._in_buffer.pos >= self._in_buffer.size:
            return

        old_pos = out_buffer.pos

        zresult = lib.ZSTD_compressStream2(
            self._compressor._cctx,
            out_buffer,
            self._in_buffer,
            lib.ZSTD_e_continue,
        )

        self._bytes_compressed += out_buffer.pos - old_pos

        if self._in_buffer.pos == self._in_buffer.size:
            self._in_buffer.src = ffi.NULL
            self._in_buffer.pos = 0
            self._in_buffer.size = 0
            self._source_buffer = None

            if not hasattr(self._source, "read"):
                self._finished_input = True

        if lib.ZSTD_isError(zresult):
            raise ZstdError("zstd compress error: %s", _zstd_error(zresult))

        return out_buffer.pos and out_buffer.pos == out_buffer.size

    def read(self, size=-1):
        if self._closed:
            raise ValueError("stream is closed")

        if size < -1:
            raise ValueError("cannot read negative amounts less than -1")

        if size == -1:
            return self.readall()

        if self._finished_output or size == 0:
            return b""

        # Need a dedicated ref to dest buffer otherwise it gets collected.
        dst_buffer = ffi.new("char[]", size)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0

        if self._compress_into_buffer(out_buffer):
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        while not self._finished_input:
            self._read_input()

            if self._compress_into_buffer(out_buffer):
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        # EOF
        old_pos = out_buffer.pos

        zresult = lib.ZSTD_compressStream2(
            self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end
        )

        self._bytes_compressed += out_buffer.pos - old_pos

        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error ending compression stream: %s", _zstd_error(zresult)
            )

        if zresult == 0:
            self._finished_output = True

        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def read1(self, size=-1):
        if self._closed:
            raise ValueError("stream is closed")

        if size < -1:
            raise ValueError("cannot read negative amounts less than -1")

        if self._finished_output or size == 0:
            return b""

        # -1 returns arbitrary number of bytes.
        if size == -1:
            size = COMPRESSION_RECOMMENDED_OUTPUT_SIZE

        dst_buffer = ffi.new("char[]", size)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0

        # read1() dictates that we can perform at most 1 call to the
        # underlying stream to get input. However, we can't satisfy this
        # restriction with compression because not all input generates output.
        # It is possible to perform a block flush in order to ensure output.
        # But this may not be desirable behavior. So we allow multiple read()
        # to the underlying stream. But unlike read(), we stop once we have
        # any output.

        self._compress_into_buffer(out_buffer)
        if out_buffer.pos:
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        while not self._finished_input:
            self._read_input()

            # If we've filled the output buffer, return immediately.
            if self._compress_into_buffer(out_buffer):
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

            # If we've populated the output buffer and we're not at EOF,
            # also return, as we've satisfied the read1() limits.
            if out_buffer.pos and not self._finished_input:
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

            # Else if we're at EOS and we have room left in the buffer,
            # fall through to below and try to add more data to the output.

        # EOF.
        old_pos = out_buffer.pos

        zresult = lib.ZSTD_compressStream2(
            self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end
        )

        self._bytes_compressed += out_buffer.pos - old_pos

        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error ending compression stream: %s" % _zstd_error(zresult)
            )

        if zresult == 0:
            self._finished_output = True

        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto(self, b):
        if self._closed:
            raise ValueError("stream is closed")

        if self._finished_output:
            return 0

        # TODO use writable=True once we require CFFI >= 1.12.
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b"", 0)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0

        if self._compress_into_buffer(out_buffer):
            return out_buffer.pos

        while not self._finished_input:
            self._read_input()
            if self._compress_into_buffer(out_buffer):
                return out_buffer.pos

        # EOF.
        old_pos = out_buffer.pos
        zresult = lib.ZSTD_compressStream2(
            self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end
        )

        self._bytes_compressed += out_buffer.pos - old_pos

        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error ending compression stream: %s", _zstd_error(zresult)
            )

        if zresult == 0:
            self._finished_output = True

        return out_buffer.pos

    def readinto1(self, b):
        if self._closed:
            raise ValueError("stream is closed")

        if self._finished_output:
            return 0

        # TODO use writable=True once we require CFFI >= 1.12.
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b"", 0)

        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0

        self._compress_into_buffer(out_buffer)
        if out_buffer.pos:
            return out_buffer.pos

        while not self._finished_input:
            self._read_input()

            if self._compress_into_buffer(out_buffer):
                return out_buffer.pos

            if out_buffer.pos and not self._finished_input:
                return out_buffer.pos

        # EOF.
        old_pos = out_buffer.pos

        zresult = lib.ZSTD_compressStream2(
            self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end
        )

        self._bytes_compressed += out_buffer.pos - old_pos

        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error ending compression stream: %s" % _zstd_error(zresult)
            )

        if zresult == 0:
            self._finished_output = True

        return out_buffer.pos


class ZstdCompressor(object):
    """
    Create an object used to perform Zstandard compression.

    Each instance is essentially a wrapper around a ``ZSTD_CCtx`` from
    zstd's C API.

    An instance can compress data various ways. Instances can be used
    multiple times. Each compression operation will use the compression
    parameters defined at construction time.

    .. note:

       When using a compression dictionary and multiple compression
       operations are performed, the ``ZstdCompressionParameters`` derived
       from an integer compression ``level`` and the first compressed data's
       size will be reused for all subsequent operations. This may not be
       desirable if source data sizes vary significantly.

    ``compression_params`` is mutually exclusive with ``level``,
    ``write_checksum``, ``write_content_size``, ``write_dict_id``, and
    ``threads``.

    Assume that each ``ZstdCompressor`` instance can only handle a single
    logical compression operation at the same time. i.e. if you call a method
    like ``stream_reader()`` to obtain multiple objects derived from the same
    ``ZstdCompressor`` instance and attempt to use them simultaneously, errors
    will likely occur.

    If you need to perform multiple logical compression operations and you
    can't guarantee those operations are temporally non-overlapping, you need
    to obtain multiple ``ZstdCompressor`` instances.

    Unless specified otherwise, assume that no two methods of
    ``ZstdCompressor`` instances can be called from multiple Python
    threads simultaneously. In other words, assume instances are not thread safe
    unless stated otherwise.

    :param level:
       Integer compression level. Valid values are all negative integers
       through 22. Lower values generally yield faster operations with lower
       compression ratios. Higher values are generally slower but compress
       better. The default is 3, which is what the ``zstd`` CLI uses. Negative
       levels effectively engage ``--fast`` mode from the ``zstd`` CLI.
    :param dict_data:
       A ``ZstdCompressionDict`` to be used to compress with dictionary
        data.
    :param compression_params:
       A ``ZstdCompressionParameters`` instance defining low-level compression
       parameters. If defined, this will overwrite the ``level`` argument.
    :param write_checksum:
       If True, a 4 byte content checksum will be written with the compressed
       data, allowing the decompressor to perform content verification.
    :param write_content_size:
       If True (the default), the decompressed content size will be included
       in the header of the compressed data. This data will only be written if
       the compressor knows the size of the input data.
    :param write_dict_id:
       Determines whether the dictionary ID will be written into the compressed
       data. Defaults to True. Only adds content to the compressed data if
       a dictionary is being used.
    :param threads:
       Number of threads to use to compress data concurrently. When set,
       compression operations are performed on multiple threads. The default
       value (0) disables multi-threaded compression. A value of ``-1`` means
       to set the number of threads to the number of detected logical CPUs.
    """

    def __init__(
        self,
        level=3,
        dict_data=None,
        compression_params=None,
        write_checksum=None,
        write_content_size=None,
        write_dict_id=None,
        threads=0,
    ):
        if level > lib.ZSTD_maxCLevel():
            raise ValueError(
                "level must be less than %d" % lib.ZSTD_maxCLevel()
            )

        if threads < 0:
            threads = _cpu_count()

        if compression_params and write_checksum is not None:
            raise ValueError(
                "cannot define compression_params and write_checksum"
            )

        if compression_params and write_content_size is not None:
            raise ValueError(
                "cannot define compression_params and write_content_size"
            )

        if compression_params and write_dict_id is not None:
            raise ValueError(
                "cannot define compression_params and write_dict_id"
            )

        if compression_params and threads:
            raise ValueError("cannot define compression_params and threads")

        if compression_params:
            self._params = _make_cctx_params(compression_params)
        else:
            if write_dict_id is None:
                write_dict_id = True

            params = lib.ZSTD_createCCtxParams()
            if params == ffi.NULL:
                raise MemoryError()

            self._params = ffi.gc(params, lib.ZSTD_freeCCtxParams)

            _set_compression_parameter(
                self._params, lib.ZSTD_c_compressionLevel, level
            )

            _set_compression_parameter(
                self._params,
                lib.ZSTD_c_contentSizeFlag,
                write_content_size if write_content_size is not None else 1,
            )

            _set_compression_parameter(
                self._params,
                lib.ZSTD_c_checksumFlag,
                1 if write_checksum else 0,
            )

            _set_compression_parameter(
                self._params, lib.ZSTD_c_dictIDFlag, 1 if write_dict_id else 0
            )

            if threads:
                _set_compression_parameter(
                    self._params, lib.ZSTD_c_nbWorkers, threads
                )

        cctx = lib.ZSTD_createCCtx()
        if cctx == ffi.NULL:
            raise MemoryError()

        self._cctx = cctx
        self._dict_data = dict_data

        # We defer setting up garbage collection until after calling
        # _setup_cctx() to ensure the memory size estimate is more accurate.
        try:
            self._setup_cctx()
        finally:
            self._cctx = ffi.gc(
                cctx, lib.ZSTD_freeCCtx, size=lib.ZSTD_sizeof_CCtx(cctx)
            )

    def _setup_cctx(self):
        zresult = lib.ZSTD_CCtx_setParametersUsingCCtxParams(
            self._cctx, self._params
        )
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "could not set compression parameters: %s"
                % _zstd_error(zresult)
            )

        dict_data = self._dict_data

        if dict_data:
            if dict_data._cdict:
                zresult = lib.ZSTD_CCtx_refCDict(self._cctx, dict_data._cdict)
            else:
                zresult = lib.ZSTD_CCtx_loadDictionary_advanced(
                    self._cctx,
                    dict_data.as_bytes(),
                    len(dict_data),
                    lib.ZSTD_dlm_byRef,
                    dict_data._dict_type,
                )

            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "could not load compression dictionary: %s"
                    % _zstd_error(zresult)
                )

    def memory_size(self):
        """Obtain the memory usage of this compressor, in bytes.

        >>> cctx = zstandard.ZstdCompressor()
        >>> memory = cctx.memory_size()
        """
        return lib.ZSTD_sizeof_CCtx(self._cctx)

    def compress(self, data):
        """
        Compress data in a single operation.

        This is the simplest mechanism to perform compression: simply pass in a
        value and get a compressed value back. It is almost the most prone to
        abuse.

        The input and output values must fit in memory, so passing in very large
        values can result in excessive memory usage. For this reason, one of the
        streaming based APIs is preferred for larger values.

        :param data:
           Source data to compress
        :return:
           Compressed data

        >>> cctx = zstandard.ZstdCompressor()
        >>> compressed = cctx.compress(b"data to compress")
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        data_buffer = ffi.from_buffer(data)

        dest_size = lib.ZSTD_compressBound(len(data_buffer))
        out = new_nonzero("char[]", dest_size)

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, len(data_buffer))
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        out_buffer = ffi.new("ZSTD_outBuffer *")
        in_buffer = ffi.new("ZSTD_inBuffer *")

        out_buffer.dst = out
        out_buffer.size = dest_size
        out_buffer.pos = 0

        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0

        zresult = lib.ZSTD_compressStream2(
            self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end
        )

        if lib.ZSTD_isError(zresult):
            raise ZstdError("cannot compress: %s" % _zstd_error(zresult))
        elif zresult:
            raise ZstdError("unexpected partial frame flush")

        return ffi.buffer(out, out_buffer.pos)[:]

    def compressobj(self, size=-1):
        """
        Obtain a compressor exposing the Python standard library compression API.

        See :py:class:`ZstdCompressionObj` for the full documentation.

        :param size:
           Size in bytes of data that will be compressed.
        :return:
           :py:class:`ZstdCompressionObj`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        cobj = ZstdCompressionObj(self, COMPRESSION_RECOMMENDED_OUTPUT_SIZE)

        return cobj

    def chunker(self, size=-1, chunk_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE):
        """
        Create an object for iterative compressing to same-sized chunks.

        This API is similar to :py:meth:`ZstdCompressor.compressobj` but has
        better performance properties.

        :param size:
           Size in bytes of data that will be compressed.
        :param chunk_size:
           Size of compressed chunks.
        :return:
           :py:class:`ZstdCompressionChunker`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        return ZstdCompressionChunker(self, chunk_size=chunk_size)

    def copy_stream(
        self,
        ifh,
        ofh,
        size=-1,
        read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE,
        write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE,
    ):
        """
        Copy data between 2 streams while compressing it.

        Data will be read from ``ifh``, compressed, and written to ``ofh``.
        ``ifh`` must have a ``read(size)`` method. ``ofh`` must have a
        ``write(data)``
        method.

        >>> cctx = zstandard.ZstdCompressor()
        >>> with open(input_path, "rb") as ifh, open(output_path, "wb") as ofh:
        ...     cctx.copy_stream(ifh, ofh)

        It is also possible to declare the size of the source stream:

        >>> cctx = zstandard.ZstdCompressor()
        >>> cctx.copy_stream(ifh, ofh, size=len_of_input)

        You can also specify how large the chunks that are ``read()``
        and ``write()`` from and to the streams:

        >>> cctx = zstandard.ZstdCompressor()
        >>> cctx.copy_stream(ifh, ofh, read_size=32768, write_size=16384)

        The stream copier returns a 2-tuple of bytes read and written:

        >>> cctx = zstandard.ZstdCompressor()
        >>> read_count, write_count = cctx.copy_stream(ifh, ofh)

        :param ifh:
           Source stream to read from
        :param ofh:
           Destination stream to write to
        :param size:
           Size in bytes of the source stream. If defined, compression
           parameters will be tuned for this size.
        :param read_size:
           Chunk sizes that source stream should be ``read()`` from.
        :param write_size:
           Chunk sizes that destination stream should be ``write()`` to.
        :return:
           2-tuple of ints of bytes read and written, respectively.
        """

        if not hasattr(ifh, "read"):
            raise ValueError("first argument must have a read() method")
        if not hasattr(ofh, "write"):
            raise ValueError("second argument must have a write() method")

        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        in_buffer = ffi.new("ZSTD_inBuffer *")
        out_buffer = ffi.new("ZSTD_outBuffer *")

        dst_buffer = ffi.new("char[]", write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = write_size
        out_buffer.pos = 0

        total_read, total_write = 0, 0

        while True:
            data = ifh.read(read_size)
            if not data:
                break

            data_buffer = ffi.from_buffer(data)
            total_read += len(data_buffer)
            in_buffer.src = data_buffer
            in_buffer.size = len(data_buffer)
            in_buffer.pos = 0

            while in_buffer.pos < in_buffer.size:
                zresult = lib.ZSTD_compressStream2(
                    self._cctx, out_buffer, in_buffer, lib.ZSTD_e_continue
                )
                if lib.ZSTD_isError(zresult):
                    raise ZstdError(
                        "zstd compress error: %s" % _zstd_error(zresult)
                    )

                if out_buffer.pos:
                    ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                    total_write += out_buffer.pos
                    out_buffer.pos = 0

        # We've finished reading. Flush the compressor.
        while True:
            zresult = lib.ZSTD_compressStream2(
                self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end
            )
            if lib.ZSTD_isError(zresult):
                raise ZstdError(
                    "error ending compression stream: %s" % _zstd_error(zresult)
                )

            if out_buffer.pos:
                ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                total_write += out_buffer.pos
                out_buffer.pos = 0

            if zresult == 0:
                break

        return total_read, total_write

    def stream_reader(
        self,
        source,
        size=-1,
        read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE,
        closefd=True,
    ):
        """
        Wrap a readable source with a stream that can read compressed data.

        This will produce an object conforming to the ``io.RawIOBase``
        interface which can be ``read()`` from to retrieve compressed data
        from a source.

        The source object can be any object with a ``read(size)`` method
        or an object that conforms to the buffer protocol.

        See :py:class:`ZstdCompressionReader` for type documentation and usage
        examples.

        :param source:
           Object to read source data from
        :param size:
           Size in bytes of source object.
        :param read_size:
           How many bytes to request when ``read()``'ing from the source.
        :param closefd:
           Whether to close the source stream when the returned stream is
           closed.
        :return:
           :py:class:`ZstdCompressionReader`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        try:
            size = len(source)
        except Exception:
            pass

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        return ZstdCompressionReader(self, source, read_size, closefd=closefd)

    def stream_writer(
        self,
        writer,
        size=-1,
        write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE,
        write_return_read=True,
        closefd=True,
    ):
        """
        Create a stream that will write compressed data into another stream.

        The argument to ``stream_writer()`` must have a ``write(data)`` method.
        As compressed data is available, ``write()`` will be called with the
        compressed data as its argument. Many common Python types implement
        ``write()``, including open file handles and ``io.BytesIO``.

        See :py:class:`ZstdCompressionWriter` for more documentation, including
        usage examples.

        :param writer:
           Stream to write compressed data to.
        :param size:
           Size in bytes of data to be compressed. If set, it will be used
           to influence compression parameter tuning and could result in the
           size being written into the header of the compressed data.
        :param write_size:
           How much data to ``write()`` to ``writer`` at a time.
        :param write_return_read:
           Whether ``write()`` should return the number of bytes that were
           consumed from the input.
        :param closefd:
           Whether to ``close`` the ``writer`` when this stream is closed.
        :return:
           :py:class:`ZstdCompressionWriter`
        """
        if not hasattr(writer, "write"):
            raise ValueError("must pass an object with a write() method")

        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        return ZstdCompressionWriter(
            self, writer, size, write_size, write_return_read, closefd=closefd
        )

    def read_to_iter(
        self,
        reader,
        size=-1,
        read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE,
        write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE,
    ):
        """
        Read uncompressed data from a reader and return an iterator

        Returns an iterator of compressed data produced from reading from
        ``reader``.

        This method provides a mechanism to stream compressed data out of a
        source as an iterator of data chunks.

        Uncompressed data will be obtained from ``reader`` by calling the
        ``read(size)`` method of it or by reading a slice (if ``reader``
        conforms to the *buffer protocol*). The source data will be streamed
        into a compressor. As compressed data is available, it will be exposed
        to the iterator.

        Data is read from the source in chunks of ``read_size``. Compressed
        chunks are at most ``write_size`` bytes. Both values default to the
        zstd input and and output defaults, respectively.

        If reading from the source via ``read()``, ``read()`` will be called
        until it raises or returns an empty bytes (``b""``). It is perfectly
        valid for the source to deliver fewer bytes than were what requested
        by ``read(size)``.

        The caller is partially in control of how fast data is fed into the
        compressor by how it consumes the returned iterator. The compressor
        will not consume from the reader unless the caller consumes from the
        iterator.

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh):
        ...     # Do something with emitted data.

        ``read_to_iter()`` accepts a ``size`` argument declaring the size of
        the input stream:

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh, size=some_int):
        >>>     pass

        You can also control the size that data is ``read()`` from the source
        and the ideal size of output chunks:

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh, read_size=16384, write_size=8192):
        >>>     pass

        ``read_to_iter()`` does not give direct control over the sizes of chunks
        fed into the compressor. Instead, chunk sizes will be whatever the object
        being read from delivers. These will often be of a uniform size.

        :param reader:
           Stream providing data to be compressed.
        :param size:
           Size in bytes of input data.
        :param read_size:
           Controls how many bytes are ``read()`` from the source.
        :param write_size:
           Controls the output size of emitted chunks.
        :return:
           Iterator of ``bytes``.
        """

        if hasattr(reader, "read"):
            have_read = True
        elif hasattr(reader, "__getitem__"):
            have_read = False
            buffer_offset = 0
            size = len(reader)
        else:
            raise ValueError(
                "must pass an object with a read() method or "
                "conforms to buffer protocol"
            )

        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)

        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN

        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError(
                "error setting source size: %s" % _zstd_error(zresult)
            )

        in_buffer = ffi.new("ZSTD_inBuffer *")
