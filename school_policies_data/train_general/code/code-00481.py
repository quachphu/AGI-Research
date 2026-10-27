        raise NonMatchingSplitsSizesError(str(bad_splits))
    logger.info("All the splits matched successfully.")


def get_size_checksum_dict(path: str, record_checksum: bool = False) -> dict:
    """Compute the file size and the sha256 checksum of a file"""
    if record_checksum:
        m = insecure_hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                m.update(chunk)
            checksum = m.hexdigest()
    else:
        checksum = None
    return {"num_bytes": os.path.getsize(path), "checksum": checksum}


def is_small_dataset(dataset_size):
    """Check if `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.

    Args:
        dataset_size (int): Dataset size in bytes.

    Returns:
        bool: Whether `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.
    """
    if dataset_size and config.IN_MEMORY_MAX_SIZE:
        return dataset_size < config.IN_MEMORY_MAX_SIZE
    else:
        return False
<|endoftext|>"""Contains utilities to flag a feature as "experimental" in datasets."""

import warnings
from functools import wraps
from typing import Callable


def experimental(fn: Callable) -> Callable:
    """Decorator to flag a feature as experimental.

    An experimental feature trigger a warning when used as it might be subject to breaking changes in the future.

    Args:
        fn (`Callable`):
            The function to flag as experimental.

    Returns:
        `Callable`: The decorated function.

    Example:

    ```python
    >>> from datasets.utils import experimental

    >>> @experimental
    ... def my_function():
    ...     print("Hello world!")

    >>> my_function()
    UserWarning: 'my_function' is experimental and might be subject to breaking changes in the future.
    Hello world!
    ```
    """

    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        warnings.warn(
            (f"'{fn.__name__}' is experimental and might be subject to breaking changes in the future."),
            UserWarning,
        )
        return fn(*args, **kwargs)

    return _inner_fn
<|endoftext|>import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from typing import Any, ClassVar, Optional

import yaml
from huggingface_hub import DatasetCardData

from ..config import METADATA_CONFIGS_FIELD
from ..features import Features
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger


logger = get_logger(__name__)


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    def _check_no_duplicates_on_constructed_node(self, node):
        keys = [self.constructed_objects[key_node] for key_node, _ in node.value]
        keys = [tuple(key) if isinstance(key, list) else key for key in keys]
        counter = Counter(keys)
        duplicate_keys = [key for key in counter if counter[key] > 1]
        if duplicate_keys:
            raise TypeError(f"Got duplicate yaml keys: {duplicate_keys}")

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        self._check_no_duplicates_on_constructed_node(node)
        return mapping


def _split_yaml_from_readme(readme_content: str) -> tuple[Optional[str], str]:
    full_content = list(readme_content.splitlines())
    if full_content and full_content[0] == "---" and "---" in full_content[1:]:
        sep_idx = full_content[1:].index("---") + 1
        yamlblock = "\n".join(full_content[1:sep_idx])
        return yamlblock, "\n".join(full_content[sep_idx + 1 :])

    return None, "\n".join(full_content)


class MetadataConfigs(dict[str, dict[str, Any]]):
    """Should be in format {config_name: {**config_params}}."""

    FIELD_NAME: ClassVar[str] = METADATA_CONFIGS_FIELD

    @staticmethod
    def _raise_if_data_files_field_not_valid(metadata_config: dict):
        yaml_data_files = metadata_config.get("data_files")
        if yaml_data_files is not None:
            yaml_error_message = textwrap.dedent(
                f"""
                Expected data_files in YAML to be either a string or a list of strings
                or a list of dicts with two keys: 'split' and 'path', but got {yaml_data_files}
                Examples of data_files in YAML:

                   data_files: data.csv

                   data_files: data/*.png

                   data_files:
                    - part0/*
                    - part1/*

                   data_files:
                    - split: train
                      path: train/*
                    - split: test
                      path: test/*

                   data_files:
                    - split: train
                      path:
                      - train/part1/*
                      - train/part2/*
                    - split: test
                      path: test/*

                PS: some symbols like dashes '-' are not allowed in split names
                """
            )
            if not isinstance(yaml_data_files, (list, str)):
                raise ValueError(yaml_error_message)
            if isinstance(yaml_data_files, list):
                for yaml_data_files_item in yaml_data_files:
                    if (
                        not isinstance(yaml_data_files_item, (str, dict))
                        or isinstance(yaml_data_files_item, dict)
                        and not (
                            len(yaml_data_files_item) == 2
                            and "split" in yaml_data_files_item
                            and re.match(_split_re, yaml_data_files_item["split"])
                            and isinstance(yaml_data_files_item.get("path"), (str, list))
                        )
                    ):
                        raise ValueError(yaml_error_message)

    @classmethod
    def _from_exported_parquet_files_and_dataset_infos(
        cls,
        parquet_commit_hash: str,
        exported_parquet_files: list[dict[str, Any]],
        dataset_infos: DatasetInfosDict,
    ) -> "MetadataConfigs":
        metadata_configs = {
            config_name: {
                "data_files": [
                    {
                        "split": split_name,
                        "path": [
                            parquet_file["url"].replace("refs%2Fconvert%2Fparquet", parquet_commit_hash)
                            for parquet_file in parquet_files_for_split
                        ],
                    }
                    for split_name, parquet_files_for_split in groupby(parquet_files_for_config, itemgetter("split"))
                ],
                "version": str(dataset_infos.get(config_name, DatasetInfo()).version or "0.0.0"),
            }
            for config_name, parquet_files_for_config in groupby(exported_parquet_files, itemgetter("config"))
        }
        if dataset_infos:
            # Preserve order of configs and splits
            metadata_configs = {
                config_name: {
                    "data_files": [
                        data_file
                        for split_name in dataset_info.splits
                        for data_file in metadata_configs[config_name]["data_files"]
                        if data_file["split"] == split_name
                    ],
                    "version": metadata_configs[config_name]["version"],
                }
                for config_name, dataset_info in dataset_infos.items()
            }
        return cls(metadata_configs)

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "MetadataConfigs":
        if dataset_card_data.get(cls.FIELD_NAME):
            metadata_configs = dataset_card_data[cls.FIELD_NAME]
            if not isinstance(metadata_configs, list):
                raise ValueError(f"Expected {cls.FIELD_NAME} to be a list, but got '{metadata_configs}'")
            for metadata_config in metadata_configs:
                if "config_name" not in metadata_config:
                    raise ValueError(
                        f"Each config must include `config_name` field with a string name of a config, "
                        f"but got {metadata_config}. "
                    )
                cls._raise_if_data_files_field_not_valid(metadata_config)
            return cls(
                {
                    config.pop("config_name"): {
                        param: value if param != "features" else Features._from_yaml_list(value)
                        for param, value in config.items()
                    }
                    for metadata_config in metadata_configs
                    if (config := metadata_config.copy())
                }
            )
        return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            for metadata_config in self.values():
                self._raise_if_data_files_field_not_valid(metadata_config)
            current_metadata_configs = self.from_dataset_card_data(dataset_card_data)
            total_metadata_configs = dict(sorted({**current_metadata_configs, **self}.items()))
            for config_name, config_metadata in total_metadata_configs.items():
                config_metadata.pop("config_name", None)
            dataset_card_data[self.FIELD_NAME] = [
                {"config_name": config_name, **config_metadata}
                for config_name, config_metadata in total_metadata_configs.items()
            ]

    def get_default_config_name(self) -> Optional[str]:
        default_config_name = None
        for config_name, metadata_config in self.items():
            if len(self) == 1 or config_name == "default" or metadata_config.get("default"):
                if default_config_name is None:
                    default_config_name = config_name
                else:
                    raise ValueError(
                        f"Dataset has several default configs: '{default_config_name}' and '{config_name}'."
                    )
        return default_config_name


# DEPRECATED - just here to support old versions of evaluate like 0.2.2
# To support new tasks on the Hugging Face Hub, please open a PR for this file:
# https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/pipelines.ts
known_task_ids = {
    "image-classification": [],
    "translation": [],
    "image-segmentation": [],
    "fill-mask": [],
    "automatic-speech-recognition": [],
    "token-classification": [],
    "sentence-similarity": [],
    "audio-classification": [],
    "question-answering": [],
    "summarization": [],
    "zero-shot-classification": [],
    "table-to-text": [],
    "feature-extraction": [],
    "other": [],
    "multiple-choice": [],
    "text-classification": [],
    "text-to-image": [],
    "text2text-generation": [],
    "zero-shot-image-classification": [],
    "tabular-classification": [],
    "tabular-regression": [],
    "image-to-image": [],
    "tabular-to-text": [],
    "unconditional-image-generation": [],
    "text-retrieval": [],
    "text-to-speech": [],
    "object-detection": [],
    "audio-to-audio": [],
    "text-generation": [],
    "conversational": [],
    "table-question-answering": [],
    "visual-question-answering": [],
    "image-to-text": [],
    "reinforcement-learning": [],
    "voice-activity-detection": [],
    "time-series-forecasting": [],
    "document-question-answering": [],
}
<|endoftext|>import contextlib
from multiprocessing import Pool, RLock

from tqdm.auto import tqdm

from ..utils import experimental, logging


logger = logging.get_logger(__name__)


class ParallelBackendConfig:
    backend_name = None


@experimental
def parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func):
    """
    **Experimental.** Apply a function to iterable elements in parallel, where the implementation uses either
    multiprocessing.Pool or joblib for parallelization.

    Args:
        function (`Callable[[Any], Any]`): Function to be applied to `iterable`.
        iterable (`list`, `tuple` or `np.ndarray`): Iterable elements to apply function to.
        num_proc (`int`): Number of processes (if no backend specified) or jobs (using joblib).
        types (`tuple`): Additional types (besides `dict` values) to apply `function` recursively to their elements.
        disable_tqdm (`bool`): Whether to disable the tqdm progressbar.
        desc (`str`): Prefix for the tqdm progressbar.
        single_map_nested_func (`Callable`): Map function that applies `function` to an element from `iterable`.
            Takes a tuple of function, data_struct, types, rank, disable_tqdm, desc as input, where data_struct is an
            element of `iterable`, and `rank` is used for progress bar.
    """
    if ParallelBackendConfig.backend_name is None:
        return _map_with_multiprocessing_pool(
            function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
        )

    return _map_with_joblib(
        function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
    )


def _map_with_multiprocessing_pool(
    function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
):
    num_proc = num_proc if num_proc <= len(iterable) else len(iterable)
    split_kwds = []  # We organize the splits ourselve (contiguous splits)
    for index in range(num_proc):
        div = len(iterable) // num_proc
        mod = len(iterable) % num_proc
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        split_kwds.append((function, iterable[start:end], batched, batch_size, types, index, disable_tqdm, desc))

    if len(iterable) != sum(len(i[1]) for i in split_kwds):
        raise ValueError(
            f"Error dividing inputs iterable among processes. "
            f"Total number of objects {len(iterable)}, "
            f"length: {sum(len(i[1]) for i in split_kwds)}"
        )

    logger.info(
        f"Spawning {num_proc} processes for {len(iterable)} objects in slices of {[len(i[1]) for i in split_kwds]}"
    )
    initargs, initializer = None, None
    if not disable_tqdm:
        initargs, initializer = (RLock(),), tqdm.set_lock
    with Pool(num_proc, initargs=initargs, initializer=initializer) as pool:
        mapped = pool.map(single_map_nested_func, split_kwds)
    logger.info(f"Finished {num_proc} processes")
    mapped = [obj for proc_res in mapped for obj in proc_res]
    logger.info(f"Unpacked {len(mapped)} objects")

    return mapped


def _map_with_joblib(
    function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
):
    # progress bar is not yet supported for _map_with_joblib, because tqdm couldn't accurately be applied to joblib,
    # and it requires monkey-patching joblib internal classes which is subject to change
    import joblib

    with joblib.parallel_backend(ParallelBackendConfig.backend_name, n_jobs=num_proc):
        return joblib.Parallel()(
            joblib.delayed(single_map_nested_func)((function, obj, batched, batch_size, types, None, True, None))
            for obj in iterable
        )


@experimental
@contextlib.contextmanager
def parallel_backend(backend_name: str):
    """
    **Experimental.**  Configures the parallel backend for parallelized dataset loading, which uses the parallelization
    implemented by joblib.

    Args:
        backend_name (str): Name of backend for parallelization implementation, has to be supported by joblib.

     Example usage:
     ```py
     with parallel_backend('spark'):
       dataset = load_dataset(..., num_proc=2)
     ```
    """
    ParallelBackendConfig.backend_name = backend_name

    if backend_name == "spark":
        from joblibspark import register_spark

        register_spark()

        # TODO: call create_cache_and_write_probe if "download" in steps
        # TODO: raise NotImplementedError when Dataset.map etc is called

    try:
        yield
    finally:
        ParallelBackendConfig.backend_name = None
<|endoftext|>import importlib
import io
import shutil
import warnings
from typing import List

import fsspec
import fsspec.asyn
from fsspec.implementations.local import LocalFileSystem

from . import compression


COMPRESSION_FILESYSTEMS: list[compression.BaseCompressedFileFileSystem] = [
    compression.Bz2FileSystem,
    compression.GzipFileSystem,
    compression.Lz4FileSystem,
    compression.XzFileSystem,
    compression.ZstdFileSystem,
]

EXTENSION_TO_COMPRESSION_FS_FILE_CLS: dict[str, type[io.BytesIO]] = {}
# Register custom filesystems
for fs_class in COMPRESSION_FILESYSTEMS:
    if fs_class.protocol in fsspec.registry and fsspec.registry[fs_class.protocol] is not fs_class:
        warnings.warn(f"A filesystem protocol was already set for {fs_class.protocol} and will be overwritten.")
    fsspec.register_implementation(fs_class.protocol, fs_class, clobber=True)
    for extension in fs_class.extensions:
        if fs_class.compression in fsspec.compression.compr:
            EXTENSION_TO_COMPRESSION_FS_FILE_CLS[extension] = fsspec.compression.compr[fs_class.compression]


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """
    Checks if `fs` is a remote filesystem.

    Args:
        fs (`fsspec.spec.AbstractFileSystem`):
            An abstract super-class for pythonic file-systems, e.g. `fsspec.filesystem(\'file\')` or `s3fs.S3FileSystem`.
    """
    return not isinstance(fs, LocalFileSystem)


def rename(fs: fsspec.AbstractFileSystem, src: str, dst: str):
    """
    Renames the file `src` in `fs` to `dst`.
    """
    if not is_remote_filesystem(fs):
        # LocalFileSystem.mv does copy + rm, it is more efficient to simply move a local directory
        shutil.move(fs._strip_protocol(src), fs._strip_protocol(dst))
    else:
        fs.mv(src, dst, recursive=True)
<|endoftext|>import os
from functools import partial
from typing import Optional

import fsspec
from fsspec.archive import AbstractArchiveFileSystem


class BaseCompressedFileFileSystem(AbstractArchiveFileSystem):
    """Read contents of compressed file as a filesystem with one file inside."""

    root_marker = ""
    protocol: str = (
        None  # protocol passed in prefix to the url. ex: "gzip", for gzip://file.txt::http://foo.bar/file.txt.gz
    )
    compression: str = None  # compression type in fsspec. ex: "gzip"
    extensions: list[str] = None  # extensions of the filename to strip. ex: ".gz" to get file.txt from file.txt.gz

    def __init__(
        self, fo: str = "", target_protocol: Optional[str] = None, target_options: Optional[dict] = None, **kwargs
    ):
        """
        The compressed file system can be instantiated from any compressed file.
        It reads the contents of compressed file as a filesystem with one file inside, as if it was an archive.

        The single file inside the filesystem is named after the compresssed file,
        without the compression extension at the end of the filename.

        Args:
            fo (:obj:``str``): Path to compressed file. Will fetch file using ``fsspec.open()``
            mode (:obj:``str``): Currently, only 'rb' accepted
            target_protocol(:obj:``str``, optional): To override the FS protocol inferred from a URL.
            target_options (:obj:``dict``, optional): Kwargs passed when instantiating the target FS.
        """
        super().__init__(self, **kwargs)
        self.fo = fo.__fspath__() if hasattr(fo, "__fspath__") else fo
        # always open as "rb" since fsspec can then use the TextIOWrapper to make it work for "r" mode
        self._open_with_fsspec = partial(
            fsspec.open,
            self.fo,
            mode="rb",
            protocol=target_protocol,
            compression=self.compression,
            client_kwargs={
                "requote_redirect_url": False,  # see https://github.com/huggingface/datasets/pull/5459
                "trust_env": True,  # Enable reading proxy env variables.
                **(target_options or {}).pop("client_kwargs", {}),  # To avoid issues if it was already passed.
            },
            **(target_options or {}),
        )
        self.compressed_name = os.path.basename(self.fo.split("::")[0])
        self.uncompressed_name = (
            self.compressed_name[: self.compressed_name.rindex(".")]
            if "." in self.compressed_name
            else self.compressed_name
        )
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        # compressed file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def _get_dirs(self):
        if self.dir_cache is None:
            f = {**self._open_with_fsspec().fs.info(self.fo), "name": self.uncompressed_name}
            self.dir_cache = {f["name"]: f}

    def cat(self, path: str):
        with self._open_with_fsspec().open() as f:
            return f.read()

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if mode != "rb":
            raise ValueError(f"Tried to read with mode {mode} on file {self.fo} opened with mode 'rb'")
        return self._open_with_fsspec().open()


class Bz2FileSystem(BaseCompressedFileFileSystem):
    """Read contents of BZ2 file as a filesystem with one file inside."""

    protocol = "bz2"
    compression = "bz2"
    extensions = [".bz2"]


class GzipFileSystem(BaseCompressedFileFileSystem):
    """Read contents of GZIP file as a filesystem with one file inside."""

    protocol = "gzip"
    compression = "gzip"
    extensions = [".gz", ".gzip"]


class Lz4FileSystem(BaseCompressedFileFileSystem):
    """Read contents of LZ4 file as a filesystem with one file inside."""

    protocol = "lz4"
    compression = "lz4"
    extensions = [".lz4"]


class XzFileSystem(BaseCompressedFileFileSystem):
    """Read contents of .xz (LZMA) file as a filesystem with one file inside."""

    protocol = "xz"
    compression = "xz"
    extensions = [".xz"]


class ZstdFileSystem(BaseCompressedFileFileSystem):
    """
    Read contents of .zstd file as a filesystem with one file inside.
    """

    protocol = "zstd"
    compression = "zstd"
    extensions = [".zst", ".zstd"]
<|endoftext|>from argparse import ArgumentParser
from typing import Optional

from datasets.commands import BaseDatasetsCLICommand
from datasets.hub import delete_from_hub


def _command_factory(args):
    return DeleteFromHubCommand(
        args.dataset_id,
        args.config_name,
        args.token,
        args.revision,
    )


class DeleteFromHubCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser):
        parser: ArgumentParser = parser.add_parser("delete_from_hub", help="Delete dataset config from the Hub")
        parser.add_argument(
            "dataset_id", help="source dataset ID, e.g. USERNAME/DATASET_NAME or ORGANIZATION/DATASET_NAME"
        )
        parser.add_argument("config_name", help="config name to delete")
        parser.add_argument("--token", help="access token to the Hugging Face Hub")
        parser.add_argument("--revision", help="source revision")
        parser.set_defaults(func=_command_factory)

    def __init__(
        self,
        dataset_id: str,
        config_name: str,
        token: Optional[str],
        revision: Optional[str],
    ):
        self._dataset_id = dataset_id
        self._config_name = config_name
        self._token = token
        self._revision = revision

    def run(self) -> None:
        _ = delete_from_hub(self._dataset_id, self._config_name, revision=self._revision, token=self._token)
<|endoftext|>import platform
from argparse import ArgumentParser

import fsspec
import huggingface_hub
import pandas
import pyarrow

from datasets import __version__ as version
from datasets.commands import BaseDatasetsCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env", help="Print relevant system environment info.")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        info = {
            "`datasets` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "`huggingface_hub` version": huggingface_hub.__version__,
            "PyArrow version": pyarrow.__version__,
            "Pandas version": pandas.__version__,
            "`fsspec` version": fsspec.__version__,
        }

        print("\nCopy-and-paste the text below in your GitHub issue.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
<|endoftext|>from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseDatasetsCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
<|endoftext|>#!/usr/bin/env python
from argparse import ArgumentParser

from datasets.commands.delete_from_hub import DeleteFromHubCommand
from datasets.commands.env import EnvironmentCommand
from datasets.commands.test import TestCommand
from datasets.utils.logging import set_verbosity_info


def parse_unknown_args(unknown_args):
    return {key.lstrip("-"): value for key, value in zip(unknown_args[::2], unknown_args[1::2])}


def main():
    parser = ArgumentParser(
        "HuggingFace Datasets CLI tool", usage="datasets-cli <command> [<args>]", allow_abbrev=False
    )
    commands_parser = parser.add_subparsers(help="datasets-cli command helpers")
    set_verbosity_info()

    # Register commands
    EnvironmentCommand.register_subcommand(commands_parser)
    TestCommand.register_subcommand(commands_parser)
    DeleteFromHubCommand.register_subcommand(commands_parser)

    # Parse args
    args, unknown_args = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)
    kwargs = parse_unknown_args(unknown_args)

    # Run
    service = args.func(args, **kwargs)
    service.run()


if __name__ == "__main__":
    main()
<|endoftext|>import logging
import os
from argparse import ArgumentParser
from collections.abc import Generator
from shutil import rmtree

import datasets.config
from datasets.builder import DatasetBuilder
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_manager import DownloadMode
from datasets.info import DatasetInfosDict
from datasets.load import dataset_module_factory, get_dataset_builder_class
from datasets.utils.info_utils import VerificationMode
from datasets.utils.logging import ERROR, get_logger


logger = get_logger(__name__)


def _test_command_factory(args):
    return TestCommand(
        args.dataset,
        args.name,
        args.cache_dir,
        args.data_dir,
        args.all_configs,
        args.save_info or args.save_infos,
        args.ignore_verifications,
        args.force_redownload,
        args.clear_cache,
        args.num_proc,
    )


class TestCommand(BaseDatasetsCLICommand):
    __test__ = False  # to tell pytest it's not a test class

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        test_parser = parser.add_parser("test", help="Test dataset loading.")
        test_parser.add_argument("--name", type=str, default=None, help="Dataset processing name")
        test_parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Cache directory where the datasets are stored.",
        )
        test_parser.add_argument(
            "--data_dir",
            type=str,
            default=None,
            help="Can be used to specify a manual directory to get the files from.",
        )
        test_parser.add_argument("--all_configs", action="store_true", help="Test all dataset configurations")
        test_parser.add_argument(
            "--save_info", action="store_true", help="Save the dataset infos in the dataset card (README.md)"
        )
        test_parser.add_argument(
            "--ignore_verifications",
            action="store_true",
            help="Run the test without checksums and splits checks.",
        )
        test_parser.add_argument("--force_redownload", action="store_true", help="Force dataset redownload")
        test_parser.add_argument(
            "--clear_cache",
            action="store_true",
            help="Remove downloaded files and cached datasets after each config test",
        )
        test_parser.add_argument("--num_proc", type=int, default=None, help="Number of processes")
        # aliases
        test_parser.add_argument("--save_infos", action="store_true", help="alias to save_info")
        test_parser.add_argument("dataset", type=str, help="Name of the dataset to download")
        test_parser.set_defaults(func=_test_command_factory)

    def __init__(
        self,
        dataset: str,
        name: str,
        cache_dir: str,
        data_dir: str,
        all_configs: bool,
        save_infos: bool,
        ignore_verifications: bool,
        force_redownload: bool,
        clear_cache: bool,
        num_proc: int,
    ):
        self._dataset = dataset
        self._name = name
        self._cache_dir = cache_dir
        self._data_dir = data_dir
        self._all_configs = all_configs
        self._save_infos = save_infos
        self._ignore_verifications = ignore_verifications
        self._force_redownload = force_redownload
        self._clear_cache = clear_cache
        self._num_proc = num_proc
        if clear_cache and not cache_dir:
            print(
                "When --clear_cache is used, specifying a cache directory is mandatory.\n"
                "The 'download' folder of the cache directory and the dataset builder cache will be deleted after each configuration test.\n"
                "Please provide a --cache_dir that will be used to test the dataset."
            )
            exit(1)
        if save_infos:
            self._ignore_verifications = True

    def run(self):
        logging.getLogger("filelock").setLevel(ERROR)
        if self._name is not None and self._all_configs:
            print("Both parameters `config` and `all_configs` can't be used at once.")
            exit(1)
        path, config_name = self._dataset, self._name
        module = dataset_module_factory(path)
        builder_cls = get_dataset_builder_class(module)
        n_builders = len(builder_cls.BUILDER_CONFIGS) if self._all_configs and builder_cls.BUILDER_CONFIGS else 1

        def get_builders() -> Generator[DatasetBuilder, None, None]:
            if self._all_configs and builder_cls.BUILDER_CONFIGS:
                for i, config in enumerate(builder_cls.BUILDER_CONFIGS):
                    if "config_name" in module.builder_kwargs:
                        yield builder_cls(
                            cache_dir=self._cache_dir,
                            data_dir=self._data_dir,
                            **module.builder_kwargs,
                        )
                    else:
                        yield builder_cls(
                            config_name=config.name,
                            cache_dir=self._cache_dir,
                            data_dir=self._data_dir,
                            **module.builder_kwargs,
                        )
            else:
                if "config_name" in module.builder_kwargs:
                    yield builder_cls(cache_dir=self._cache_dir, data_dir=self._data_dir, **module.builder_kwargs)
                else:
                    yield builder_cls(
                        config_name=config_name,
                        cache_dir=self._cache_dir,
                        data_dir=self._data_dir,
                        **module.builder_kwargs,
                    )

        for j, builder in enumerate(get_builders()):
            print(f"Testing builder '{builder.config.name}' ({j + 1}/{n_builders})")
            builder._record_checksums = os.path.exists(
                os.path.join(builder.get_imported_module_dir(), datasets.config.DATASETDICT_INFOS_FILENAME)
            )  # record checksums only if we need to update a (deprecated) dataset_infos.json
            builder.download_and_prepare(
                download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS
                if not self._force_redownload
                else DownloadMode.FORCE_REDOWNLOAD,
                verification_mode=VerificationMode.NO_CHECKS
                if self._ignore_verifications
                else VerificationMode.ALL_CHECKS,
                num_proc=self._num_proc,
            )
            builder.as_dataset()

            # If save_infos=True, we create the dataset card (README.md)
            # The dataset_infos are saved in the YAML part of the README.md
            # This is to allow the user to upload them on HF afterwards.
            if self._save_infos:
                save_infos_dir = os.path.basename(path) if not os.path.isdir(path) else path
                os.makedirs(save_infos_dir, exist_ok=True)
                DatasetInfosDict(**{builder.config.name: builder.info}).write_to_directory(save_infos_dir)
                print(f"Dataset card saved at {os.path.join(save_infos_dir, datasets.config.REPOCARD_FILENAME)}")

            # If clear_cache=True, the download folder and the dataset builder cache directory are deleted
            if self._clear_cache:
                if os.path.isdir(builder._cache_dir):
                    logger.warning(f"Clearing cache at {builder._cache_dir}")
                    rmtree(builder._cache_dir)
                download_dir = os.path.join(self._cache_dir, datasets.config.DOWNLOADED_DATASETS_DIR)
                if os.path.isdir(download_dir):
                    logger.warning(f"Clearing cache at {download_dir}")
                    rmtree(download_dir)

        print("Test successful.")
<|endoftext|>from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NamedTuple, TypeAlias, cast
from weakref import WeakValueDictionary

from ._api import (
    AcquireReturnProxy,
    _canonical,
    _ensure_current_process,
    _fork_transition,
    _raise_chained_errors,
    _register_fork_class,
    _register_fork_object,
    _register_owned_descriptor,
    _resolve_read_write_timeout,
    _seconds,
    _timeout_behavior,
    _unregister_owned_descriptor,
)
from ._error import Timeout
from ._identity import host_name, owner_is_stale, process_start_token

if sys.platform == "win32":  # pragma: win32 cover
    from ._windows import _open_non_reparse_fd

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from _typeshed import Unused

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

_LOGGER: Final[logging.Logger] = logging.getLogger("filelock")
_GETPID: Final[Callable[[], int]] = os.getpid
_IS_PYPY: Final[bool] = sys.implementation.name == "pypy"
_NEEDS_CONNECTION_ESCROW: Final[bool] = (
    hasattr(os, "register_at_fork") and sys.implementation.name == "cpython" and sys.version_info < (3, 12)
)
_ConnectionParameter: TypeAlias = (
    str | bytes | os.PathLike[str] | os.PathLike[bytes] | float | int | type[sqlite3.Connection] | None
)
_DatabaseIdentity: TypeAlias = tuple[int, int]

# sqlite3_busy_timeout() accepts a C int, max 2_147_483_647 on 32-bit. Use a lower value to be safe (~23 days).
_MAX_SQLITE_TIMEOUT_MS: Final[int] = 2_000_000_000 - 1
_UNSAFE_FORK_EXIT_STATUS: Final[int] = 70
# O_NONBLOCK keeps an open from blocking on a FIFO planted at the path; the regular-file check then rejects it.
_DB_OPEN_FLAGS: Final[int] = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
# Linux names an open descriptor under /proc/self/fd, macOS and the BSDs under /dev/fd. SQLite connects through that
# name, so it reopens the inode already validated, and journal_mode=MEMORY keeps it from deriving an on-disk journal
# name from the synthetic path.
_FD_DIR: Final[str] = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
# Header bytes 18 and 19 hold the file format versions, 2 in WAL mode: https://www.sqlite.org/fileformat.html
_SQLITE_MAGIC: Final[bytes] = b"SQLite format 3\x00"
_FORMAT_VERSIONS: Final[slice] = slice(18, 20)
_HEADER_SIZE: Final[int] = 20
_WAL_FORMAT_VERSION: Final[int] = 2


class _SQLiteTransitionContext(threading.local):
    depth: int = 0


_SQLITE_TRANSITION_CONTEXT: Final = _SQLiteTransitionContext()


class _ConnectionEscrow:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._functions: tuple[Callable[[sqlite3.Connection], None], Callable[[sqlite3.Connection], None]] | None = None

    def functions(
        self,
    ) -> tuple[Callable[[sqlite3.Connection], None], Callable[[sqlite3.Connection], None]] | None:
        if not _NEEDS_CONNECTION_ESCROW:
            return None  # pragma: >=3.12 cover
        with self._lock:  # pragma: <3.12 cover  # pragma: needs fork
            if self._functions is None:
                import ctypes  # ruff:ignore[import-outside-top-level]  # keep optional ctypes and its audited dlsym out of ordinary imports

                function_type = ctypes.PYFUNCTYPE(None, ctypes.py_object)
                increment_address = ctypes.cast(ctypes.pythonapi.Py_IncRef, ctypes.c_void_p).value
                decrement_address = ctypes.cast(ctypes.pythonapi.Py_DecRef, ctypes.c_void_p).value
                if increment_address is None or decrement_address is None:  # pragma: no cover - resolved CPython API
                    msg = "CPython reference functions have no address"
                    raise RuntimeError(msg)
                self._functions = (
                    cast("Callable[[sqlite3.Connection], None]", function_type(increment_address)),
                    cast("Callable[[sqlite3.Connection], None]", function_type(decrement_address)),
                )
            return self._functions

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        self._lock = threading.RLock()


_CONNECTION_ESCROW: Final = _ConnectionEscrow()


class _ForkedDatabaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paths: set[pathlib.Path] = set()
        self._identities: set[_DatabaseIdentity] = set()
        self._sqlite_used = False
        self._all_paths_poisoned = False

    def raise_if_poisoned(self, path: pathlib.Path) -> None:
        identity = self.identity(path)
        with self._lock:
            all_paths_poisoned = self._all_paths_poisoned
            poisoned = (
                all_paths_poisoned or path in self._paths or (identity is not None and identity in self._identities)
            )
        if poisoned:  # pragma: needs fork
            msg = (
                "ReadWriteLock is unavailable in a PyPy fork child; exec or exit before using it"
                if all_paths_poisoned
                else f"SQLite database {path!s} was active across fork(); exec or exit before using it in the child"
            )
            raise RuntimeError(msg)

    def poison_after_fork(self, path: pathlib.Path, identity: _DatabaseIdentity | None) -> None:
        self._paths.add(path)
        if identity is not None:
            self._identities.add(identity)

    def note_sqlite_use(self) -> None:
        if _IS_PYPY:
            with self._lock:
                self._sqlite_used = True

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        self._lock = threading.RLock()
        self._all_paths_poisoned = self._all_paths_poisoned or (_IS_PYPY and self._sqlite_used)
        self._sqlite_used = False

    @staticmethod
    def identity(path: pathlib.Path) -> _DatabaseIdentity | None:
        try:
            stat_result = path.stat()
        except OSError:
            return None
        return stat_result.st_dev, stat_result.st_ino


_FORKED_DATABASES: Final = _ForkedDatabaseRegistry()

_LINK_PREFIX: Final[str] = ".filelock-"
_SWEPT_DIRECTORIES: Final[set[str]] = set()


# POSIX locks belong to the process and the inode, so closing any descriptor on the inode drops the locks every other
# connection of this process holds there. SQLite defers its own closes for this reason (section 2.2 of
# https://www.sqlite.org/howtocorrupt.html); the probes stay open until the last connection on their inode closes.
class _ProbeDescriptors:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inodes: dict[_DatabaseIdentity, _InodeProbes] = {}
        self._releases: deque[_Probe] = deque()
        self._leases = count()

    def adopt(self, database: str, claimed: list[_Probe]) -> _Probe | None:  # pragma: win32 no cover
        # Reuse a probe no live connection holds: each connect would otherwise keep another descriptor open until the
        # inode's last close, and two live connections through one /dev/fd name make macOS SQLite fail to open.
        try:
            stat_result: Final = os.lstat(database)
        except OSError:
            return None
        identity: Final = stat_result.st_dev, stat_result.st_ino
        with self._lock:
            if not stat.S_ISREG(stat_result.st_mode) or not (probes := self._inodes.get(identity)) or not probes.idle:
                return None
            claimed.append(adopted := _Probe(identity, probes.idle.pop(), next(self._leases)))
            probes.leases.add(adopted.lease)
        return adopted

    def retain(self, fd: int, claimed: list[_Probe]) -> _Probe:  # pragma: win32 no cover
        stat_result: Final = os.fstat(fd)
        identity: Final = stat_result.st_dev, stat_result.st_ino
        token: Final = _register_owned_descriptor(fd, identity)
        with self._lock:
            (probes := self._inodes.setdefault(identity, _InodeProbes())).descriptors.append((fd, token))
            claimed.append(retained := _Probe(identity, fd, next(self._leases)))
            probes.leases.add(retained.lease)
        return retained

    def shared(self, probe: _Probe) -> bool:  # pragma: win32 no cover
        with self._lock:
            return len(self._inodes[probe.identity].leases) > 1

    def release(self, probe: _Probe) -> None:  # pragma: win32 no cover
        # A finalizer may run this on a thread already inside the lock, so it never waits; the lock holder drains.
        self._releases.append(probe)
        self.drain()

    def drain(self) -> None:  # pragma: win32 no cover
        while self._releases and self._lock.acquire(blocking=False):
            try:
                # Dequeued only once applied, so an exception leaves the rest of the release for the next drain.
                while self._releases:
                    self._release_locked(self._releases[0])
                    self._releases.popleft()
            finally:
                self._lock.release()

    def _release_locked(self, probe: _Probe) -> None:  # pragma: win32 no cover
        if (probes := self._inodes.get(probe.identity)) is None:
            return
        if probe.lease in probes.leases:
            probes.leases.remove(probe.lease)
            if probes.leases:
                probes.idle.append(probe.fd)
        if probes.leases:
            return
        # No connect may adopt a probe about to close, even one an interrupt leaves listed.
        probes.idle.clear()
        # Closing under the lock keeps a concurrent connect from taking a lock on this inode before the close.
        while probes.descriptors:
            fd, token = probes.descriptors[-1]
            if token is not None:  # pragma: needs fork
                _unregister_owned_descriptor(token)
            # Forgotten before the close: after an interrupt, a leaked descriptor beats closing a reused number.
            probes.descriptors.pop()
            # The descriptor is gone even when close reports an error, and no lock of this process depends on it.
            with suppress(OSError):
                os.close(fd)
        del self._inodes[probe.identity]

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        # The child holds no POSIX locks, and the owned-descriptor registry closes the inherited probes.
        self._lock = threading.Lock()
        self._inodes = {}
        self._releases = deque()


class _Probe(NamedTuple):
    identity: _DatabaseIdentity
    fd: int
    # One per connection, so a release applied again after an interrupt kept it queued counts once.
    lease: int


@dataclass(slots=True)
class _InodeProbes:
    leases: set[int] = field(default_factory=set)
    descriptors: list[tuple[int, int | None]] = field(default_factory=list)
    idle: list[int] = field(default_factory=list)


_PROBE_DESCRIPTORS: Final = _ProbeDescriptors()


class _ForkSafeConnection(sqlite3.Connection):
    _creator_pid: int
    _decrement_escrow: Callable[[sqlite3.Connection], None] | None
    _database_directory: pathlib.Path | None
    database_identity: _DatabaseIdentity | None = None
    _probe: _Probe | None = None

    def __new__(
        cls,
        *_args: _ConnectionParameter,
        **_kwargs: _ConnectionParameter,
    ) -> Self:
        connection = super().__new__(cls)
        connection._creator_pid = _GETPID()
        connection._decrement_escrow = None
        connection._database_directory = None
        return connection

    def close(self) -> None:
        with _sqlite_transition():
            if _GETPID() != self._creator_pid:  # pragma: needs fork
                return
            with _fork_transition():
                sqlite3.Connection.close(self)
                try:
                    if (directory := self._database_directory) is not None:  # pragma: needs posix-hard-link
                        self._database_directory = None
                        # Leftover private links are plain directories someone may already have removed.
                        with suppress(FileNotFoundError):
                            shutil.rmtree(directory)
                finally:
                    if (probe := self._probe) is not None:  # pragma: win32 no cover
                        self._probe = self.database_identity = None
                        _PROBE_DESCRIPTORS.release(probe)
                if (decrement := self._decrement_escrow) is not None:  # pragma: <3.12 cover  # pragma: needs fork
                    self._decrement_escrow = None
                    decrement(self)

    def retain_until_close(self, directory: pathlib.Path | None, probe: _Probe) -> None:  # pragma: win32 no cover
        """Keep the private link and this connection's probe descriptor until SQLite no longer uses them."""
        self._database_directory = directory
        self._probe = probe
        self.database_identity = probe.identity

    def acquire_escrow(  # pragma: <3.12 cover  # pragma: needs fork
        self,
        functions: tuple[Callable[[sqlite3.Connection], None], Callable[[sqlite3.Connection], None]] | None,
    ) -> None:
        # The caller only reaches here holding the escrow functions; it skips the call entirely without them.
        if functions is not None:  # pragma: no branch
            increment, decrement = functions
            increment(self)
            self._decrement_escrow = decrement

    def __del__(self) -> None:
        with suppress(sqlite3.Error, RuntimeError, OSError):
            self.close()


class _ReadWriteLockMeta(type):
    """
    Resolve singleton instances for ``is_singleton=True`` construction.

    This logic lives here rather than in ReadWriteLock.get_lock so ``ReadWriteLock(path)`` returns cached instances
    without a 2-arg ``super()`` call that type checkers cannot verify.

    """

    _instances: WeakValueDictionary[pathlib.Path, ReadWriteLock]
    _instances_lock: threading.RLock
    _instances_pid: int
    _instances_under_construction: set[pathlib.Path]

    def __call__(
        cls,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,
    ) -> ReadWriteLock:
        _ensure_current_process()
        timeout = _seconds("timeout", timeout)
        if cls._instances_pid != _GETPID():
            cls._reset_class_after_fork()
        construction_pid = _GETPID()
        if not is_singleton:
            instance = super().__call__(lock_file, timeout, blocking=blocking, is_singleton=is_singleton)
            if _GETPID() != construction_pid:  # pragma: forked child
                msg = "ReadWriteLock construction cannot continue after fork"
                raise RuntimeError(msg)
            return instance

        # Resolve only the parent, so a symlink at the lock path stays its own key instead of aliasing its target.
        normalized = pathlib.Path(_canonical(lock_file))
        with cls._instances_lock:
            if normalized not in cls._instances:
                if normalized in cls._instances_under_construction:  # pragma: no cover - exercised in an audit callback
                    msg = f"Singleton lock construction is already active for {lock_file!s}"
                    raise RuntimeError(msg)
                construction_registry = cls._instances_under_construction
                construction_registry.add(normalized)
                try:
                    instance = super().__call__(lock_file, timeout, blocking=blocking, is_singleton=is_singleton)
                finally:
                    if _GETPID() == construction_pid:
                        construction_registry.discard(normalized)
                if _GETPID() != construction_pid:
                    msg = "ReadWriteLock construction cannot continue after fork"
                    raise RuntimeError(msg)
                cls._instances[normalized] = instance
            else:
                instance = cls._instances[normalized]

            if (
                _timeout_behavior(instance.timeout, any_negative=False)
                != _timeout_behavior(timeout, any_negative=False)
                or instance.blocking != blocking
            ):
                msg = (
                    f"Singleton lock created with timeout={instance.timeout}, blocking={instance.blocking},"
                    f" cannot be changed to timeout={timeout}, blocking={blocking}"
                )
                raise ValueError(msg)
            return instance

    def _reset_class_after_fork(cls) -> None:  # pragma: forked child
        cls._instances = WeakValueDictionary()
        cls._instances_lock = threading.RLock()
        cls._instances_pid = _GETPID()
        cls._instances_under_construction = set()


class ReadWriteLock(metaclass=_ReadWriteLockMeta):
    """
    Cross-process read-write lock backed by SQLite.

    Allows concurrent shared readers or a single exclusive writer. The lock is reentrant within the same mode (multiple
    ``acquire_read`` calls nest, as do multiple ``acquire_write`` calls from the same thread), but upgrading from read
    to write or downgrading from write to read raises :class:`RuntimeError`. Write locks are pinned to the thread that
    acquired them, and each thread sharing a read lock can release only the levels it acquired.

    An instance coordinates processes, not threads: a thread asking for the opposite mode of another thread's hold
    raises :class:`RuntimeError` instead of waiting. Threads that must wait for each other each need their own
    instance (``is_singleton=False``).

    By default, ``is_singleton=True``: calling ``ReadWriteLock(path)`` with the same resolved path returns the same
    instance. The path is handed to :func:`sqlite3.connect` as given, so a ``.db`` extension is a convention rather
    than a requirement; the filesystem must be one the active SQLite VFS supports. Give the lock a file of its own: a
    WAL-mode database raises :class:`ValueError`.

    :param lock_file: path to the SQLite database file used as the lock
    :param timeout: maximum wait time in seconds; ``-1`` waits up to SQLite's busy-timeout cap of about 23 days
    :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable
    :param is_singleton: if ``True``, reuse existing instances for the same resolved path

    .. versionadded:: 3.21.0

    """

    _instances: WeakValueDictionary[pathlib.Path, ReadWriteLock] = WeakValueDictionary()
    _instances_lock = threading.RLock()
    _instances_pid = _GETPID()
    _instances_under_construction: ClassVar[set[pathlib.Path]] = set()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._instances = WeakValueDictionary()
        cls._instances_lock = threading.RLock()
        cls._instances_pid = _GETPID()
        cls._instances_under_construction = set()
        _register_fork_class(cls)

    @classmethod
    def get_lock(
        cls, lock_file: str | os.PathLike[str], timeout: float = -1, *, blocking: bool = True
    ) -> ReadWriteLock:
        """
        Return the singleton :class:`ReadWriteLock` for *lock_file*.

        :param lock_file: path to the SQLite database file used as the lock
        :param timeout: maximum wait time in seconds; ``-1`` waits up to SQLite's busy-timeout cap of about 23 days
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable

        :returns: the singleton lock instance

        :raises ValueError: if an instance already exists for this path with different *timeout* or *blocking* values

        """
        return cls(lock_file, timeout, blocking=blocking)

    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,  # ruff:ignore[unused-method-argument]  # consumed by _ReadWriteLockMeta.__call__
    ) -> None:
        self.lock_file = os.fspath(lock_file)
        self._canonical_path = pathlib.Path(_canonical(lock_file))
        _FORKED_DATABASES.raise_if_poisoned(self._canonical_path)
        self.timeout = timeout
        self.blocking = blocking
        self._transaction_lock = threading.Lock()  # serializes the (possibly blocking) SQLite transaction work
        self._internal_lock = threading.Lock()  # protects _lock_level / _current_mode updates and rollback
        self._lock_level = 0
        self._current_mode: Literal["read", "write"] | None = None
        self._write_thread_id: int | None = None
        self._read_depths: dict[int, int] = {}
        self._acquisition_thread_ids: set[int] = set()
        self._con: _ForkSafeConnection | None = None
        self._connection_transaction_released = True
        self._connection_identity: _DatabaseIdentity | None = None
        self._closed = False
        self._creator_pid = _GETPID()
        self._fork_invalidated = False
        _register_fork_object(self)
        with _fork_transition(), _sqlite_transition():
            validation_connection = self._open_connection(sqlite_timeout=5.0)
            validation_connection.close()

    def acquire_read(self, timeout: float | None = None, *, blocking: bool | None = None) -> AcquireReturnProxy:
        """
        Acquire a shared read lock.

        If this instance already holds a read lock, the lock level is incremented (reentrant). Attempting to acquire a
        read lock while holding a write lock raises :class:`RuntimeError` (downgrade not allowed).

        :param timeout: seconds to wait; ``None`` uses the instance setting; ``-1`` waits up to SQLite's
            busy-timeout cap of about 23 days
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` on contention;
            ``None`` uses the instance setting

        :returns: a proxy that can be used as a context manager to release the lock

        :raises RuntimeError: if a write lock is already held on this instance
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        return self._acquire(
            "read",
            self.timeout if timeout is None else timeout,
            blocking=self.blocking if blocking is None else blocking,
        )

    def acquire_write(self, timeout: float | None = None, *, blocking: bool | None = None) -> AcquireReturnProxy:
        """
        Acquire an exclusive write lock.

        If this instance already holds a write lock from the same thread, the lock level is incremented (reentrant).
        Attempting to acquire a write lock while holding a read lock raises :class:`RuntimeError` (upgrade not allowed).
        Write locks are pinned to the acquiring thread: a different thread trying to re-enter also raises
        :class:`RuntimeError`.

        :param timeout: seconds to wait; ``None`` uses the instance setting; ``-1`` waits up to SQLite's
            busy-timeout cap of about 23 days
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` on contention;
            ``None`` uses the instance setting

        :returns: a proxy that can be used as a context manager to release the lock

        :raises RuntimeError: if a read lock is already held, or a write lock is held by a different thread
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        return self._acquire(
            "write",
            self.timeout if timeout is None else timeout,
            blocking=self.blocking if blocking is None else blocking,
        )

    @contextmanager
    def read_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> Generator[None]:
        """
        Context manager that acquires and releases a shared read lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        self.acquire_read(timeout, blocking=blocking)
        try:
            yield
        finally:
            self.release()

    @contextmanager
    def write_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> Generator[None]:
        """
        Context manager that acquires and releases an exclusive write lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        self.acquire_write(timeout, blocking=blocking)
        try:
            yield
        finally:
            self.release()

    def release(self, *, force: bool = False) -> None:
        """
        Release one level of the current lock.

        When the lock level reaches zero the underlying SQLite transaction is rolled back, releasing the database lock.

        :param force: if ``True``, release the lock completely regardless of the current lock level

        :raises RuntimeError: if no lock is currently held and *force* is ``False``, or if the calling thread holds
            no level of the current lock and *force* is ``False``

        """
        with _fork_transition():
            _ensure_current_process()
            if self._inherited:  # pragma: needs fork
                return
            self._raise_if_acquiring("release")
            self._release(force=force, close=False)

    def close(self) -> None:
        """
        Release the lock (if held) and close the underlying SQLite connection.

        After calling this method, the lock instance is no longer usable. Unlike :meth:`release`, it ends every
        thread's hold on this instance, whichever thread calls it.

        """
        with _fork_transition():
            _ensure_current_process()
            if self._inherited:  # pragma: needs fork
                return
            self._raise_if_acquiring("close")
            self._release(force=True, close=True)

    def _release_level_of(self, thread_id: int) -> None:
        with _fork_transition():
            _ensure_current_process()
            self._raise_if_acquiring("release")
            self._release(force=False, close=False, thread_id=thread_id)

    def _release(self, *, force: bool, close: bool, thread_id: int | None = None) -> None:
        thread_id = threading.get_ident() if thread_id is None else thread_id
        with self._transaction_lock, self._internal_lock:
            if not force:
                self._raise_if_not_holder(thread_id)
            if self._lock_level == 0:
                if force and self._con is None:
                    if close:
                        self._closed = True
                    return
                if not force:
                    msg = f"Cannot release a lock on {self.lock_file} (lock id: {id(self)}) that is not held"
                    raise RuntimeError(msg)
            if not force and self._lock_level > 1:
                self._leave_one_level(thread_id)
                return
            try:
                self._finish_connection()
            except sqlite3.Error:
                if self._connection_transaction_released:
                    self._clear_lock_state()
                raise
            self._clear_lock_state()
            if close:
                self._closed = True

    def _raise_if_not_holder(self, thread_id: int) -> None:
        # The connection runs with check_same_thread=False, so a release from another thread would end the owner's
        # transaction under it and let a second writer in.
        if self._current_mode == "write" and thread_id != self._write_thread_id:
            msg = (
                f"Cannot release write lock on {self.lock_file} (lock id: {id(self)}) "
                f"from thread {thread_id} while it is held by thread {self._write_thread_id}"
            )
            raise RuntimeError(msg)
        # Readers share that one transaction too, so a thread that holds no read level would end another's hold.
        if self._current_mode == "read" and thread_id not in self._read_depths:
            msg = (
                f"Cannot release read lock on {self.lock_file} (lock id: {id(self)}) "
                f"from thread {thread_id}, which does not hold it"
            )
            raise RuntimeError(msg)

    def _leave_one_level(self, thread_id: int) -> None:
        self._lock_level -= 1
        if self._current_mode == "read" and (depth := self._read_depths.pop(thread_id) - 1):
            self._read_depths[thread_id] = depth

    def _clear_lock_state(self) -> None:
        self._lock_level = 0
        self._current_mode = None
        self._write_thread_id = None
        self._read_depths = {}

    def __del__(self) -> None:
        if _GETPID() == getattr(self, "_creator_pid", None) and (connection := getattr(self, "_con", None)) is not None:
            with suppress(sqlite3.Error, RuntimeError):
                connection.close()

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        if self._con is not None:
            _FORKED_DATABASES.poison_after_fork(self._canonical_path, self._connection_identity)
        self._con = None
        self._connection_transaction_released = True
        self._connection_identity = None
        self._transaction_lock = threading.Lock()
        self._internal_lock = threading.Lock()
        self._clear_lock_state()
        self._acquisition_thread_ids = set()
        self._fork_invalidated = True

    @property
    def _inherited(self) -> bool:
        return self._fork_invalidated or _GETPID() != self._creator_pid

    def _raise_if_unusable(self) -> None:
        _ensure_current_process()
        if self._inherited:  # pragma: needs fork
            msg = f"ReadWriteLock on {self.lock_file} was invalidated by fork(); construct a new instance"
            raise RuntimeError(msg)
        if self._closed:
            msg = "Cannot operate on a closed database."
            raise sqlite3.ProgrammingError(msg)

    def _acquire(self, mode: Literal["read", "write"], timeout: float, *, blocking: bool) -> AcquireReturnProxy:
        with _fork_transition():
            self._raise_if_unusable()
            timeout = _resolve_read_write_timeout(timeout, blocking=blocking)
            operation_pid = _GETPID()
            thread_id = threading.get_ident()
            with self._internal_lock:
                if self._lock_level > 0:
                    return self._validate_reentrant(mode)
                if thread_id in self._acquisition_thread_ids:  # pragma: no cover - exercised in an audit callback
                    msg = f"Cannot acquire ReadWriteLock on {self.lock_file} while acquisition is active in this thread"
                    raise RuntimeError(msg)
                self._acquisition_thread_ids.add(thread_id)
            try:
                start_time = time.perf_counter()
                self._acquire_transaction_lock(blocking=blocking, timeout=timeout)
                try:
                    self._raise_if_unusable()
                    return self._do_acquire_inner(
                        mode,
                        timeout,
                        blocking=blocking,
                        operation_pid=operation_pid,
                        start_time=start_time,
                    )
                finally:
                    self._transaction_lock.release()
            finally:
                with self._internal_lock:
                    self._acquisition_thread_ids.discard(thread_id)

    def _do_acquire_inner(
        self,
        mode: Literal["read", "write"],
        timeout: float,
        *,
        blocking: bool,
        operation_pid: int,
        start_time: float,
    ) -> AcquireReturnProxy:
        # Double-check: another thread may have acquired the lock while we waited on _transaction_lock.
        with self._internal_lock:
            if self._lock_level > 0:
                return self._validate_reentrant(mode)
        if self._con is not None:
            self._finish_connection()
        try:
            self._open_for_acquisition(
                timeout,
                blocking=blocking,
                operation_pid=operation_pid,
                start_time=start_time,
            )
            self._configure_and_begin(
                mode,
                timeout,
                blocking=blocking,
                operation=(operation_pid, start_time),
            )
            self._raise_if_process_changed(operation_pid)
        except BaseException as error:
            acquisition_error: BaseException
            if isinstance(error, sqlite3.OperationalError) and "database is locked" in str(error):
                acquisition_error = Timeout(self.lock_file)
            else:
                acquisition_error = error
            try:
                self._finish_connection()
            except sqlite3.Error as cleanup_error:
                _raise_chained_errors(acquisition_error, cleanup_error)
            if acquisition_error is not error:
                raise acquisition_error from None
            raise
        with self._internal_lock:
            self._raise_if_process_changed(operation_pid)
            self._current_mode = mode
            self._lock_level = 1
            if mode == "write":
                self._write_thread_id = threading.get_ident()
            else:
                self._read_depths = {threading.get_ident(): 1}
        return AcquireReturnProxy(lock=self)

    def _open_for_acquisition(self, timeout: float, *, blocking: bool, operation_pid: int, start_time: float) -> None:
        with _sqlite_transition():
            sqlite_timeout = (
                timeout_for_sqlite(
                    timeout,
                    blocking=blocking,
                    already_waited=time.perf_counter() - start_time,
                )
                / 1000
            )
            connection = self._open_connection(sqlite_timeout=sqlite_timeout)
            self._con, self._connection_transaction_released, self._connection_identity = (
                connection,
                False,
                _FORKED_DATABASES.identity(self._canonical_path),
            )
            self._raise_if_process_changed(operation_pid)

    def _configure_and_begin(
        self,
        mode: Literal["read", "write"],
        timeout: float,
        *,
        blocking: bool,
        operation: tuple[int, float],
    ) -> None:
        with _sqlite_transition():
            operation_pid, start_time = operation
            connection = cast("_ForkSafeConnection", self._con)
            waited = time.perf_counter() - start_time
            timeout_ms = timeout_for_sqlite(timeout, blocking=blocking, already_waited=waited)
            self._raise_if_process_changed(operation_pid)
            connection.executescript(f"PRAGMA busy_timeout={timeout_ms}; PRAGMA journal_mode=MEMORY;").close()
            # Use legacy journal mode (not WAL) because WAL does not block readers while a concurrent EXCLUSIVE
            # write transaction is active, which makes read-write locking impossible without modifying table data.
            # MEMORY is safe here since no writes happen, so a crash cannot corrupt the DB.
            # See https://sqlite.org/lang_transaction.html#deferred_immediate_and_exclusive_transactions
            #
            # Recompute the remaining timeout after the blocking journal_mode pragma.
            waited = time.perf_counter() - start_time
            recomputed = timeout_for_sqlite(timeout, blocking=blocking, already_waited=waited)
            self._raise_if_process_changed(operation_pid)
            statements = f"PRAGMA busy_timeout={recomputed}; " if recomputed != timeout_ms else ""
            statements += "BEGIN EXCLUSIVE TRANSACTION;" if mode == "write" else "BEGIN TRANSACTION;"
            if mode == "read":
                # SQLite takes the SHARED lock only when a statement reads; BEGIN alone stays deferred.
                # https://www.sqlite.org/lockingv3.html#transaction_control
                # sqlite_schema is an alias added in SQLite 3.33.0; sqlite_master works on every version.
                statements += " SELECT name FROM sqlite_master LIMIT 1;"
            connection.executescript(statements).close()
            self._raise_if_database_replaced(connection)

    def _raise_if_database_replaced(self, connection: _ForkSafeConnection) -> None:
        # Windows pins the name: SQLite opens the database without delete sharing, so it cannot be unlinked or renamed.
        if (identity := connection.database_identity) is None:  # pragma: win32 cover
            return
        try:  # pragma: win32 no cover
            current = self._canonical_path.stat(follow_symlinks=False)
        except FileNotFoundError:  # pragma: win32 no cover
            current = None
        # The lock lives on the inode, so one taken after the path moved on excludes no later opener.
        if current is None or (current.st_dev, current.st_ino) != identity:  # pragma: win32 no cover
            msg = f"lock database {self.lock_file!r} was unlinked or replaced while acquiring; leave it in place"
            raise OSError(msg)

    def _open_connection(self, *, sqlite_timeout: float) -> _ForkSafeConnection:
        with _sqlite_transition():
            creator_pid = _GETPID()
            functions = _CONNECTION_ESCROW.functions()
            if _GETPID() != creator_pid:  # pragma: forked child
                msg = "SQLite connection construction cannot continue after fork"
                raise RuntimeError(msg)
            connection = _connect(
                os.fspath(self._canonical_path),
                factory=_ForkSafeConnection,
                timeout=sqlite_timeout,
            )
            if functions is not None:  # pragma: <3.12 cover  # pragma: needs fork
                connection.acquire_escrow(functions)
            if _GETPID() != creator_pid:  # pragma: forked child
                _FORKED_DATABASES.poison_after_fork(
                    self._canonical_path,
                    _FORKED_DATABASES.identity(self._canonical_path),
                )
                msg = "SQLite connection construction cannot continue after fork"
                raise RuntimeError(msg)
            return connection

    def _finish_connection(self) -> None:
        with _sqlite_transition():
            if (connection := self._con) is None:
                return
            rollback_error: sqlite3.Error | None = None
            if not self._connection_transaction_released:
                if connection.in_transaction:
                    try:
                        connection.rollback()
                    except sqlite3.Error as error:
                        if connection.in_transaction:
                            raise
                        self._connection_transaction_released = True
                        rollback_error = error
                    else:
                        self._connection_transaction_released = True
                else:
                    self._connection_transaction_released = True
            try:
                connection.close()
            except sqlite3.Error as close_error:
                if rollback_error is not None:
                    _raise_chained_errors(rollback_error, close_error)
                raise
            self._con = None
            self._connection_transaction_released = True
            self._connection_identity = None
            if rollback_error is not None:
                raise rollback_error

    def _validate_reentrant(self, mode: Literal["read", "write"]) -> AcquireReturnProxy:
        cur = threading.get_ident()
        if self._current_mode != mode:
            opposite = "write" if mode == "read" else "read"
            if cur == self._write_thread_id or cur in self._read_depths:
                direction = "downgrade" if mode == "read" else "upgrade"
                reason = f"already holding a {opposite} lock ({direction} not allowed)"
            else:
                reason = (
                    f"another thread holds a {opposite} lock on this instance, which coordinates processes rather "
                    "than threads; give each thread its own instance with is_singleton=False"
                )
            msg = f"Cannot acquire {mode} lock on {self.lock_file} (lock id: {id(self)}): {reason}"
            raise RuntimeError(msg)
        if mode == "write" and cur != self._write_thread_id:
            msg = (
                f"Cannot acquire write lock on {self.lock_file} (lock id: {id(self)}) "
                f"from thread {cur} while it is held by thread {self._write_thread_id}"
            )
            raise RuntimeError(msg)
        if mode == "read":
            self._read_depths[cur] = self._read_depths.get(cur, 0) + 1
        self._lock_level += 1
        return AcquireReturnProxy(lock=self)

    def _acquire_transaction_lock(self, *, blocking: bool, timeout: float) -> None:
        if not blocking:
            acquired = self._transaction_lock.acquire(blocking=False)
        elif timeout == -1:
            acquired = self._transaction_lock.acquire(blocking=True)
        else:
            # Lock.acquire raises OverflowError past TIMEOUT_MAX, which is 49.7 days on Windows.
            acquired = self._transaction_lock.acquire(blocking=True, timeout=min(timeout, threading.TIMEOUT_MAX))
        if not acquired:
            raise Timeout(self.lock_file) from None

    def _raise_if_acquiring(self, operation: Literal["acquire", "close", "release"]) -> None:
        with self._internal_lock:
            active_in_current_thread = threading.get_ident() in self._acquisition_thread_ids
        if active_in_current_thread:  # pragma: no cover - exercised in an audit callback
            msg = f"Cannot {operation} ReadWriteLock on {self.lock_file} while acquisition is active in this thread"
            raise RuntimeError(msg)

    def _raise_if_process_changed(self, operation_pid: int) -> None:
        if _GETPID() != operation_pid or self._inherited:  # pragma: forked child
            msg = f"ReadWriteLock on {self.lock_file} was invalidated by fork(); construct a new instance"
            raise RuntimeError(msg)


def _connect(database: str, *, factory: type[_ForkSafeConnection], timeout: float) -> _ForkSafeConnection:
    _FORKED_DATABASES.note_sqlite_use()
    # A symlink at the path would make SQLite open, lock, or create its target, so connect through a descriptor that
    # refuses one.
    if sys.platform == "win32":  # pragma: win32 cover
        fd: Final = _open_lock_database(database)
        # A Windows lock belongs to the handle that took it, so closing this one leaves SQLite's locks in place.
        try:
            _raise_if_wal(fd, database)
            return sqlite3.connect(
                database, check_same_thread=False, factory=factory, cached_statements=0, timeout=timeout
            )
        finally:
            os.close(fd)
    return _connect_through_descriptor(database, factory=factory, timeout=timeout)  # pragma: win32 no cover


def _connect_through_descriptor(  # pragma: win32 no cover
    database: str, *, factory: type[_ForkSafeConnection], timeout: float
) -> _ForkSafeConnection:
    # Listed before its lease is taken and cleared once a connection owns it, so any exception between, an interrupt
    # included, releases the lease instead of pinning the inode's probes.
    claimed: Final[list[_Probe]] = []
    directory: pathlib.Path | None = None
    try:
        if (probe := _PROBE_DESCRIPTORS.adopt(database, claimed)) is None:
            probe = _PROBE_DESCRIPTORS.retain(_open_lock_database(database), claimed)
        identity, fd, _ = probe
        # Releases a finalizer queued while the table was locked.
        _PROBE_DESCRIPTORS.drain()
        _raise_if_wal(fd, database)
        target = pathlib.Path(f"{_FD_DIR}/{fd}")
        reachable: Final = os.access(target, os.F_OK)
        # SQLite reuses a descriptor it deferred closing for a peer's locks only when stat of the name it opened reports
        # the file, which macOS's /dev/fd does not, so every connect beside a holder would keep one more open. NetBSD's
        # static /dev/fd exposes only descriptors 0-63. A private hard link names the file and pins the validated inode.
        if not reachable or (  # pragma: needs posix-hard-link
            _PROBE_DESCRIPTORS.shared(probe) and ((named := target.stat()).st_dev, named.st_ino) != identity
        ):
            try:
                directory = _link_beside(database, identity)
            except OSError:
                # The descriptor path still pins the validated inode; only the deferred descriptors pile up.
                if not reachable:
                    raise
            else:
                target = directory / "lock.db"
        connection = sqlite3.connect(
            os.fspath(target), check_same_thread=False, factory=factory, cached_statements=0, timeout=timeout
        )
        connection.retain_until_close(directory, probe)
        claimed.clear()
        return connection
    finally:
        for unowned in claimed:
            _PROBE_DESCRIPTORS.release(unowned)
        if claimed and directory is not None:  # pragma: needs posix-hard-link
            # A connection that already took the link removes it too when it closes.
            with suppress(FileNotFoundError):
                shutil.rmtree(directory)


def _link_beside(database: str, identity: _DatabaseIdentity) -> pathlib.Path:  # pragma: needs posix-hard-link
    # Beside the database because a link cannot cross filesystems, and TMPDIR is often on another one.
    parent: Final = pathlib.Path(database).parent
    _sweep_links_of_the_dead(parent)
    directory: Final = pathlib.Path(tempfile.mkdtemp(prefix=_link_prefix(), dir=parent))
    try:
        os.link(database, link := directory / "lock.db", follow_symlinks=False)
        linked: Final = link.stat(follow_symlinks=False)
    except BaseException:
        shutil.rmtree(directory)
        raise
    if (linked.st_dev, linked.st_ino) != identity:
        shutil.rmtree(directory)
        msg = f"lock database changed while opening: {database!r}"
        raise OSError(msg)
    return directory


def _sweep_links_of_the_dead(parent: pathlib.Path) -> None:  # pragma: needs posix-hard-link
    # A process killed while it held a link leaves the directory, and an extra link to the database, behind for good.
    # Two threads sweeping at once only both try the same removals, which a lock would cost a fork-safety reset for.
    if (key := os.fspath(parent)) in _SWEPT_DIRECTORIES:
        return
    _SWEPT_DIRECTORIES.add(key)
    try:
        with os.scandir(parent) as entries:
            directories: Final = [
                pathlib.Path(entry.path)
                for entry in entries
                if entry.name.startswith(_LINK_PREFIX) and entry.is_dir(follow_symlinks=False)
            ]
    except OSError:
        return
    for directory in directories:
        if _owner_is_gone(directory.name):
            # A peer sweeping alongside, or a directory this user cannot remove, must not fail the connect.
            with suppress(OSError):
                shutil.rmtree(directory)


def _owner_is_gone(name: str) -> bool:  # pragma: needs posix-hard-link
    # Fails closed like a soft lock's marker: a name it cannot read, or an owner on another host, stays.
    match name.removeprefix(_LINK_PREFIX).split("-"):
        case [pid, start, digest, _] if pid.isdigit() and (start.isdigit() or not start):
            host: Final = host_name()
            return digest == _host_digest(host) and owner_is_stale(int(pid), host, int(start) if start else None)
        case _:
            return False


def _link_prefix() -> str:  # pragma: needs posix-hard-link
    # The owner rides in the name, which mkdtemp creates in one step, so no directory ever exists without it.
    start: Final = process_start_token(pid := os.getpid())
    return f"{_LINK_PREFIX}{pid}-{'' if start is None else start}-{_host_digest(host_name())}-"


def _host_digest(host: str) -> str:  # pragma: needs posix-hard-link
    # SHA-256 as host_name does: some runtimes, GraalPy's among them, have no BLAKE2 with a custom digest size.
    return hashlib.sha256(host.encode()).hexdigest()[:16]


def _open_lock_database(database: str) -> int:
    if sys.platform == "win32":  # pragma: win32 cover
        # The handle refuses a reparse point and omits delete sharing, so the name cannot be swapped while it is open.
        if (fd := _open_non_reparse_fd(database, 0o600)) is None:
            msg = f"lock database {database!r} is being deleted or held without sharing"
            raise PermissionError(msg)
        return fd
    fd = os.open(database, _DB_OPEN_FLAGS, 0o600)  # pragma: win32 no cover
    if not stat.S_ISREG(os.fstat(fd).st_mode):  # pragma: win32 no cover
        os.close(fd)
        msg = f"refusing a non-regular lock database: {database!r}"
        raise OSError(msg)
    return fd  # pragma: win32 no cover


def _raise_if_wal(fd: int, database: str) -> None:
    # SQLite derives the -wal name from the path it connects through, a descriptor name on POSIX, and WAL never blocks
    # readers behind a writer, so the lock cannot work on such a database.
    # pread: an adopted probe is shared, so its file offset is not this connect's to move.
    header: Final = os.pread(fd, _HEADER_SIZE, 0) if hasattr(os, "pread") else os.read(fd, _HEADER_SIZE)
    if header.startswith(_SQLITE_MAGIC) and _WAL_FORMAT_VERSION in header[_FORMAT_VERSIONS]:
        msg = f"refusing a WAL-mode lock database: {database!r}; give ReadWriteLock a file of its own"
        raise ValueError(msg)


@contextmanager
def _sqlite_transition() -> Generator[None]:
    _SQLITE_TRANSITION_CONTEXT.depth += 1
    try:
        yield
    finally:
        _SQLITE_TRANSITION_CONTEXT.depth -= 1


def _abort_forked_sqlite_transition() -> None:  # pragma: forked child
    if _SQLITE_TRANSITION_CONTEXT.depth:
        os._exit(_UNSAFE_FORK_EXIT_STATUS)  # inherited SQLite handles cannot be used or closed safely


def _track_sqlite_use(event: str, _args: Unused) -> None:
    if event == "sqlite3.connect":
        _FORKED_DATABASES.note_sqlite_use()


def timeout_for_sqlite(timeout: float, *, blocking: bool, already_waited: float) -> int:
    if blocking is False:
        return 0

    if timeout == -1:
        return _MAX_SQLITE_TIMEOUT_MS

    if timeout < 0:
        msg = "timeout must be a non-negative number or -1"
        raise ValueError(msg)

    timeout_ms = int((max(timeout - already_waited, 0) if timeout > 0 else timeout) * 1000)
    if timeout_ms > _MAX_SQLITE_TIMEOUT_MS or timeout_ms < 0:
        _LOGGER.warning("timeout %s is too large for SQLite, using %s ms instead", timeout, _MAX_SQLITE_TIMEOUT_MS)
        return _MAX_SQLITE_TIMEOUT_MS
    return timeout_ms


_register_fork_object(_CONNECTION_ESCROW)
_register_fork_object(_FORKED_DATABASES)
_register_fork_object(_PROBE_DESCRIPTORS)
_register_fork_class(ReadWriteLock)
if _IS_PYPY:
    sys.addaudithook(_track_sqlite_use)  # pragma: pypy cover
if hasattr(os, "register_at_fork"):  # pragma: needs fork
    os.register_at_fork(after_in_child=_abort_forked_sqlite_transition)

__all__ = [
    "ReadWriteLock",
    "timeout_for_sqlite",
]
<|endoftext|>from __future__ import annotations

import os
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Final, cast

from ._api import BaseFileLock
from ._util import ensure_directory_exists, raise_on_not_writable_file

if sys.platform == "win32":  # pragma: win32 cover
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _GENERIC_READ: Final[int] = 0x80000000
    _GENERIC_WRITE: Final[int] = 0x40000000
    _SYNCHRONIZE: Final[int] = 0x00100000
    _DESIRED_ACCESS: Final[int] = _GENERIC_READ | _GENERIC_WRITE | _SYNCHRONIZE
    _FILE_SHARE_READ_WRITE: Final[int] = (
        0x00000001 | 0x00000002
    )  # read | write; matches os.open (_SH_DENYNO), no delete
    _FILE_OPEN_IF: Final[int] = 3  # open the file if it exists, create it otherwise; the NtCreateFile OPEN_ALWAYS
    _FILE_ATTRIBUTE_READONLY: Final[int] = 0x00000001
    _FILE_ATTRIBUTE_NORMAL: Final[int] = 0x00000080
    _FILE_ATTRIBUTE_REPARSE_POINT: Final[int] = 0x00000400
    # CreateOptions: keep the handle synchronous (the CRT and msvcrt.locking rely on a maintained file position),
    # refuse a directory, and open a reparse point rather than following it so the check below acts on the link itself.
    _FILE_SYNCHRONOUS_IO_NONALERT: Final[int] = 0x00000020
    _FILE_NON_DIRECTORY_FILE: Final[int] = 0x00000040
    _FILE_OPEN_REPARSE_POINT: Final[int] = 0x00200000
    _CREATE_OPTIONS: Final[int] = _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_NON_DIRECTORY_FILE | _FILE_OPEN_REPARSE_POINT
    _OBJ_CASE_INSENSITIVE: Final[int] = 0x00000040  # Win32 name lookups are case-insensitive
    _OWNER_WRITE: Final[int] = 0o200

    # LockFileEx locks a byte range at an offset carried in OVERLAPPED, independent of the descriptor's file position.
    # msvcrt.locking starts at the current position instead, so a metadata write between lock and unlock could shift
    # the byte a later unlock targets; the explicit offset removes that hazard for both the path lock and #608's
    # descriptor lock.
    _LOCKFILE_FAIL_IMMEDIATELY: Final[int] = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK: Final[int] = 0x00000002
    _ERROR_LOCK_VIOLATION: Final[int] = 33  # another handle holds the byte range
    _ERROR_NOT_LOCKED: Final[int] = 158  # the byte range is not locked

    # NtCreateFile returns the raw NTSTATUS as its value, where CreateFileW collapses several of these into one
    # ERROR_ACCESS_DENIED. Telling them apart is the point (#604): a name pending deletion or a share conflict is
    # transient and worth a retry, a real access denial is not.
    _STATUS_SUCCESS: Final[int] = 0x00000000
    _STATUS_ACCESS_DENIED: Final[int] = 0xC0000022
    _STATUS_SHARING_VIOLATION: Final[int] = 0xC0000043
    _STATUS_DELETE_PENDING: Final[int] = 0xC0000056
    #: How long an open waits out a STATUS_ACCESS_DENIED before treating it as a real denial: NTFS answers it for a
    #: moment while a peer's unlink tears the name down, so a bounded wait keeps a real denial failing fast (#604).
    _ACCESS_DENIED_GRACE: Final[float] = 0.5
    _ACCESS_DENIED_RETRY: Final[float] = 0.002

    _ntdll: Final[ctypes.WinDLL] = ctypes.WinDLL("ntdll")
    _kernel32: Final[ctypes.WinDLL] = ctypes.WinDLL("kernel32", use_last_error=True)

    class _UNICODE_STRING(ctypes.Structure):  # ruff:ignore[invalid-class-name]  # mirrors the Win32 struct name
        _fields_ = (
            ("Length", wintypes.USHORT),  # byte length, not character count
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        )

    class _OBJECT_ATTRIBUTES(ctypes.Structure):  # ruff:ignore[invalid-class-name]  # mirrors the Win32 struct name
        _fields_ = (
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", ctypes.c_void_p),
            ("SecurityQualityOfService", ctypes.c_void_p),
        )

    class _IO_STATUS_BLOCK(ctypes.Structure):  # ruff:ignore[invalid-class-name]  # mirrors the Win32 struct name
        _fields_ = (
            ("Status", ctypes.c_void_p),  # a union of NTSTATUS and PVOID, so it is pointer-sized
            ("Information", ctypes.c_void_p),
        )

    class _OVERLAPPED(ctypes.Structure):  # mirrors the Win32 struct name
        _fields_ = (
            ("Internal", ctypes.c_void_p),  # ULONG_PTR: pointer-sized, not DWORD, or the x64 layout corrupts Offset
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),  # the DUMMYUNIONNAME struct, flattened: low 32 bits of the byte offset
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):  # ruff:ignore[invalid-class-name]  # mirrors the Win32 struct name
        _fields_ = (
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        )

    _ntdll.NtCreateFile.restype = wintypes.LONG  # NTSTATUS
    _ntdll.NtCreateFile.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_OBJECT_ATTRIBUTES),
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.POINTER(ctypes.c_longlong),  # PLARGE_INTEGER AllocationSize
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
    ]
    _ntdll.RtlDosPathNameToNtPathName_U_WithStatus.restype = wintypes.LONG  # NTSTATUS
    _ntdll.RtlDosPathNameToNtPathName_U_WithStatus.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(_UNICODE_STRING),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _ntdll.RtlFreeUnicodeString.restype = None
    _ntdll.RtlFreeUnicodeString.argtypes = [ctypes.POINTER(_UNICODE_STRING)]
    _ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
    _ntdll.RtlNtStatusToDosError.argtypes = [wintypes.LONG]

    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)]
    _kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_OVERLAPPED),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_OVERLAPPED),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    def _lock_fd_nonblocking(fd: int) -> bool:
        # One nonblocking exclusive LockFileEx attempt shared by WindowsFileLock and lock_descriptor, over the one-byte
        # range at offset 0. True on acquisition, False on contention, raise otherwise. The caller owns fd; the handle
        # from get_osfhandle belongs to the CRT descriptor and must not be closed here.
        overlapped = _OVERLAPPED()  # zero-initialized, so Offset/OffsetHigh/hEvent are 0
        flags = _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY
        if _kernel32.LockFileEx(msvcrt.get_osfhandle(fd), flags, 0, 1, 0, ctypes.byref(overlapped)):
            return True
        err = ctypes.get_last_error()
        if err == _ERROR_LOCK_VIOLATION:
            return False
        # A non-contention LockFileEx failure is not reproducible in-process.
        raise ctypes.WinError(err)  # pragma: no cover

    def _unlock_fd(fd: int) -> None:
        overlapped = _OVERLAPPED()  # the same offset 0 and one-byte length the lock used
        if (  # pragma: no cover  # only an interrupted earlier release leaves our own range unlocked
            not _kernel32.UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(overlapped))
            and (err := ctypes.get_last_error()) != _ERROR_NOT_LOCKED
        ):
            raise ctypes.WinError(err)

    class WindowsFileLock(BaseFileLock):
        """
        Uses ``LockFileEx`` to hard lock a byte range of the lock file on Windows systems.

        Lock file cleanup: Windows attempts to delete the lock file after release, but deletion is
        not guaranteed in multi-threaded scenarios where another thread holds an open handle. The lock
        file may persist on disk, which does not affect lock correctness.
        """

        def _acquire(self) -> None:
            raise_on_not_writable_file(self.lock_file)
            ensure_directory_exists(self.lock_file)

            # The reparse test is bound to the opened handle, so a symlink or junction swapped in cannot defeat it
            # through a check-then-open TOCTOU race.
            fd = _open_non_reparse_fd(self.lock_file, self._open_mode())
            if fd is None:
                return  # open contention (share conflict or a name pending deletion); let the retry loop try again
            try:
                locked = _lock_fd_nonblocking(fd)
                if locked:
                    self._mark_descriptor_owned(fd)
            except BaseException:  # pragma: no cover  # cleanup only if the lock attempt itself raises
                os.close(fd)
                raise
            if not locked:
                os.close(fd)  # another holder owns the byte-range lock; let the retry loop try again

        def _release(self) -> None:
            fd = cast("int", self._context.lock_file_fd)
            # Retain the descriptor until the OS unlock succeeds: if UnlockFileEx raises, the byte-range lock is still
            # held, so is_locked must keep reporting held rather than losing the fd. Only after the unlock commits do
            # close and unlink run as post-unlock cleanup; their failure cannot make the lock held again.
            _unlock_fd(fd)
            self._mark_descriptor_released()
            self._close_released_fd(fd, default_suppresses=False)
            if not self._preserve_lock_file:  # preserve_lock_file keeps a stable file identity for the caller (#605)
                with suppress(OSError):
                    Path(self.lock_file).unlink()

    def _open_non_reparse_fd(path: str, mode: int) -> int | None:
        """
        Open *path* for locking while refusing reparse points, bound to the handle actually locked.

        The file is opened through ``NtCreateFile`` with ``FILE_OPEN_REPARSE_POINT`` so a symlink or junction planted
        at the path is not followed, and the reparse decision is read from *that* handle via
        ``GetFileInformationByHandle`` rather than from a prior pathname query. Reading the held handle closes the
        check-then-open race: an attacker cannot swap the path between validation and use because both act on the same
        handle. Share mode omits delete so a peer cannot unlink or rename the file out from under a live holder,
        matching ``os.open``'s ``_SH_DENYNO``.

        ``NtCreateFile`` is used instead of ``CreateFileW`` because its return value carries the raw ``NTSTATUS``.
        Windows collapses a transient delete-pending name and a permanent access denial into the same Win32
        ``ERROR_ACCESS_DENIED``; the status keeps them apart, so a real denial fails fast instead of spinning until the
        caller's timeout (#604).

        The reparse option only guards the final path component; Windows still follows reparse points in intermediate
        directories. This assumes the lock file sits in a lock directory untrusted users cannot modify. A path with
        attacker-controlled parent directories would need component-by-component handle validation.

        :param path: the lock file path.
        :param mode: the permission mode; as ``os.open`` does on Windows, a cleared owner-write bit creates the file
            read-only. The attribute only takes effect when the file is created, not when an existing one is opened.

        :returns: a file descriptor owning the opened handle, or ``None`` on a sharing violation or a delete-pending
            name the caller should treat as contention and retry.

        :raises OSError: if the path resolves to a reparse point, or the open fails for any other reason, raised with
            the Win32 error the status maps to. An access denial is raised only once it outlasts a short grace, since
            a peer unlinking the file as it releases can make the open answer ``STATUS_ACCESS_DENIED`` for a moment.

        """
        # Emit the audit event os.open would, so consumers watching "open" still see the path-level open and can veto.
        sys.audit("open", path, None, os.O_RDWR | os.O_CREAT)
        read_only = not mode & _OWNER_WRITE
        handle, status = _nt_open(path, read_only=read_only)
        if status == _STATUS_ACCESS_DENIED:
            deadline = time.monotonic() + _ACCESS_DENIED_GRACE
            while status == _STATUS_ACCESS_DENIED and time.monotonic() < deadline:
                time.sleep(_ACCESS_DENIED_RETRY)
                handle, status = _nt_open(path, read_only=read_only)
        if status != _STATUS_SUCCESS:
            if status in {_STATUS_SHARING_VIOLATION, _STATUS_DELETE_PENDING}:
                return None
            winerror = _ntdll.RtlNtStatusToDosError(status)
            raise OSError(None, ctypes.FormatError(winerror).strip(), path, winerror)

        info = _BY_HANDLE_FILE_INFORMATION()
        # Querying an open handle we just created does not fail.
        if not _kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):  # pragma: no cover
            err = ctypes.get_last_error()
            _kernel32.CloseHandle(handle)
            raise ctypes.WinError(err)
        if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            _kernel32.CloseHandle(handle)
            msg = f"Lock file is a reparse point (symlink/junction): {path}"
            raise OSError(msg)

        try:
            # O_NOINHERIT mirrors os.open on Windows: the lock fd must not leak into child processes.
            return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_NOINHERIT)
        except BaseException:  # pragma: no cover  # open_osfhandle audits too; a hook raising must not leak the handle
            _kernel32.CloseHandle(handle)
            raise

    def _nt_open(path: str, *, read_only: bool) -> tuple[int, int]:
        """
        Open *path* through ``NtCreateFile`` and return ``(handle, status)``.

        ``RtlDosPathNameToNtPathName_U_WithStatus`` translates the Win32 path to the NT namespace, handling relative,
        drive, UNC and extended-length path forms as Win32 itself would, and allocates a buffer that
        ``RtlFreeUnicodeString`` releases. The handle is ``0`` unless the status is ``STATUS_SUCCESS``.
        """
        nt_name = _UNICODE_STRING()
        status = _ntdll.RtlDosPathNameToNtPathName_U_WithStatus(path, ctypes.byref(nt_name), None, None) & 0xFFFFFFFF
        if status != _STATUS_SUCCESS:
            return 0, status
        try:
            attributes = _OBJECT_ATTRIBUTES()
            attributes.Length = ctypes.sizeof(_OBJECT_ATTRIBUTES)
            attributes.ObjectName = ctypes.pointer(nt_name)
            attributes.Attributes = _OBJ_CASE_INSENSITIVE
            handle = wintypes.HANDLE()
            io_status = _IO_STATUS_BLOCK()
            status = (
                _ntdll.NtCreateFile(
                    ctypes.byref(handle),
                    _DESIRED_ACCESS,
                    ctypes.byref(attributes),
                    ctypes.byref(io_status),
                    None,
                    _FILE_ATTRIBUTE_READONLY if read_only else _FILE_ATTRIBUTE_NORMAL,
                    _FILE_SHARE_READ_WRITE,
                    _FILE_OPEN_IF,
                    _CREATE_OPTIONS,
                    None,
                    0,
                )
                & 0xFFFFFFFF
            )
        finally:
            _ntdll.RtlFreeUnicodeString(ctypes.byref(nt_name))
        if status != _STATUS_SUCCESS:
            return 0, status
        return handle.value or 0, status

else:  # pragma: win32 no cover

    class WindowsFileLock(BaseFileLock):
        """Uses ``LockFileEx`` to hard lock a byte range of the lock file on Windows systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError


__all__ = [
    "WindowsFileLock",
    "_open_non_reparse_fd",
]
<|endoftext|>from __future__ import annotations

from typing import Final


class Timeout(TimeoutError):  # ruff:ignore[error-suffix-on-exception-name]  # public exception name; renaming breaks the API
    """Raised when the lock could not be acquired in *timeout* seconds."""

    def __init__(self, lock_file: str, holder: str | None = None) -> None:
        super().__init__()
        self._lock_file = lock_file
        self._holder = holder

    def __reduce__(self) -> tuple[type[Timeout], tuple[str], dict[str, object]]:
        # holder rides in the state so a subclass taking only lock_file, and a 4.0.12 reader, can still unpickle it.
        return self.__class__, (self._lock_file,), self.__dict__

    def __str__(self) -> str:  # pragma: needs hard-link
        message: Final = f"The file lock '{self._lock_file}' could not be acquired."
        if self._holder is None:
            return message
        return f"{message} It is held by {self._holder}, which this host cannot probe."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.lock_file!r})"

    @property
    def lock_file(self) -> str:
        """The path of the file lock."""
        return self._lock_file


class SoftFileLockLifetimeWarning(FutureWarning):
    """
    The configured soft-lock lifetime permits overlapping live holders after expiry.

    A :class:`FutureWarning` shows by default wherever the lock is configured; a :class:`DeprecationWarning` shows only
    in ``__main__``.

    """


class LeaseSettingsMismatch(ValueError):  # ruff:ignore[error-suffix-on-exception-name]  # public exception name; renaming breaks the API
    """A lease contender disagrees with the published claim about how long the lease lasts."""


class SoftFileLockProtocolError(OSError):
    """Raised when strict soft-lock state cannot be interpreted without risking overlap."""

    def __init__(self, lock_file: str, claim_name: str | None, reason: str) -> None:
        self._lock_file = lock_file
        self._claim_name = claim_name
        self._reason = reason
        super().__init__(self.__str__())

    def __reduce__(
        self,
    ) -> tuple[
        type[SoftFileLockProtocolError], tuple[str, str | None, str], dict[str, object]
    ]:  # pragma: needs hard-link
        return self.__class__, (self._lock_file, self._claim_name, self._reason), self.__dict__

    def __str__(self) -> str:
        location = self._lock_file if self._claim_name is None else f"{self._lock_file}: claim {self._claim_name!r}"
        return f"Invalid strict soft-lock state at {location}: {self._reason}"

    @property
    def lock_file(self) -> str:  # pragma: needs hard-link
        """The requested lock path."""
        return self._lock_file

    @property
    def claim_name(self) -> str | None:  # pragma: needs hard-link
        """The claim that caused the error, if scanning identified one."""
        return self._claim_name

    @property
    def reason(self) -> str:  # pragma: needs hard-link
        """The protocol validation failure."""
        return self._reason


__all__ = [
    "LeaseSettingsMismatch",
    "SoftFileLockLifetimeWarning",
    "SoftFileLockProtocolError",
    "Timeout",
]
<|endoftext|>from __future__ import annotations

import os
import stat
import sys
import time
from contextlib import suppress
from errno import EACCES, EEXIST, EPERM
from itertools import islice
from pathlib import Path
from typing import Final

from ._api import BaseFileLock, _raise_grouped_errors
from ._identity import host_name, owner_is_current_process, owner_is_stale, process_start_token
from ._soft_protocol import STRICT_SOFT_SENTINEL_RECORD
from ._util import break_lock_file, ensure_directory_exists, raise_on_not_writable_file, write_all

MALFORMED_LOCK_AGE_THRESHOLD: Final[float] = 2.0
_MAX_LOCK_FILE_SIZE: Final[int] = 1024
_UNLINK_MAX_RETRIES: Final[int] = 10
_MARKER_WITH_START_TOKEN_LINE_COUNT: Final[int] = 3


class SoftFileLock(BaseFileLock):
    """
    Cooperative file lock based on a shared existence marker.

    Unlike :class:`UnixFileLock <filelock.UnixFileLock>` and :class:`WindowsFileLock <filelock.WindowsFileLock>`, this
    lock does not use OS-level locking primitives. Instead, it creates the lock file with ``O_CREAT | O_EXCL`` and
    treats its existence as the lock indicator. The filesystem must provide coherent exclusive creation and directory
    updates to each participating process. A crash can leave the marker behind.

    The marker contains the holder's PID and hostname. A contender may remove it when it can no longer find a same-host
    process with that PID. A configured :attr:`~filelock.BaseFileLock.lifetime` also permits removal based on marker
    age, including while the holder remains alive. Age-based expiry can overlap protected operations and does not
    provide strict mutual exclusion.

    """

    #: Existence locks reclaim by unlinking a pathname, so an age-based lease may break one; a native inode lock cannot.
    _lifetime_supported: bool = True

    #: Age-based expiry preserves historical behavior but does not provide strict mutual exclusion.
    _lifetime_replacements: tuple[str, str] | None = ("StrictSoftFileLock", "SoftFileLease")

    #: An existence lock unlinks its marker to release, so it cannot promise to keep the pathname.
    _preserve_lock_file_supported: bool = False

    #: An existence lock keeps protocol state in its marker, so it cannot lend the descriptor to an on_acquired hook.
    _on_acquired_supported: bool = False

    #: The marker's (mtime, inode) when this instance last saw it change, and when that was on its monotonic clock.
    _marker_seen: tuple[tuple[float, int], float] | None = None

    #: The identity of a marker a Windows release relinquished but has not unlinked yet.
    _marker_cleanup: tuple[int, int] | None = None

    #: The descriptor of that marker while its close has not run yet, as a list so one C call can take it.
    _marker_fds: list[int]

    def _acquire(self) -> None:
        self._finish_marker_cleanup()
        raise_on_not_writable_file(self.lock_file)
        ensure_directory_exists(self.lock_file)
        # O_CREAT | O_EXCL makes the create fail with EEXIST when the file already exists, so a successful open
        # means this process now holds the lock.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC
        if (o_nofollow := getattr(os, "O_NOFOLLOW", None)) is not None:  # pragma: needs o-nofollow
            flags |= o_nofollow
        try:
            fd = os.open(self.lock_file, flags, self._open_mode())
        except OSError as exception:
            if not (
                exception.errno == EEXIST or (exception.errno == EACCES and sys.platform == "win32")
            ):  # pragma: win32 no cover
                raise
            self._try_break_stale_lock()
            return
        self._mark_descriptor_pending(fd)
        self._publish_held_marker(fd)

    def _publish_held_marker(self, fd: int) -> None:
        # Publish held state only once the record is fully on disk. On any failure, including cancellation, close the
        # descriptor and unlink the path only while it still names the file we opened, so a rollback never deletes a
        # successor's marker that replaced ours at the same path after our lease expired.
        identity: tuple[int, int] | None = None
        try:
            identity = _file_identity(os.fstat(fd))
            self._apply_explicit_mode(fd)
            self._write_lock_info(fd)
        except BaseException:
            self._mark_descriptor_released()
            try:
                os.close(fd)
            finally:
                self._unlink_held_marker(identity)
            raise
        self._mark_descriptor_owned(fd, identity)

    def _apply_explicit_mode(self, fd: int) -> None:
        # os.open filters the mode through the umask, so an explicit mode lands as given only through fchmod, the way
        # UnixFileLock applies it. Windows maps the mode to a read-only bit that os.open already set.
        if self.has_explicit_mode and sys.platform != "win32":  # pragma: win32 no cover
            with suppress(PermissionError):
                os.fchmod(fd, self._context.mode)

    def _try_break_stale_lock(self) -> None:
        with suppress(OSError, ValueError):
            content, mtime, ino = _read_lock_file(self.lock_file)
            if content == STRICT_SOFT_SENTINEL_RECORD:  # pragma: needs hard-link
                return
            holder = _parse_lock_holder(content)

            if holder is None:
                # Unparsable: wrong line count, a non-integer PID or start token, empty, oversized or not UTF-8.
                # Self-heal only once the file is clearly not a half-written fresh lock (a peer between O_EXCL and
                # _write_lock_info), so the brief create-then-write window is never mistaken for a stale lock.
                if self._marker_unchanged_for(mtime, ino) >= MALFORMED_LOCK_AGE_THRESHOLD:
                    break_lock_file(self.lock_file, mtime, ino)
                return

            if owner_is_stale(*holder):
                break_lock_file(self.lock_file, mtime, ino)

    def _marker_unchanged_for(self, mtime: float, ino: int) -> float:
        # The mtime is the writer's or file server's clock, so against time.time() a contender whose clock runs ahead
        # would expire a fresh marker. Only watch it for change, timed on this process's monotonic clock.
        now: Final = time.monotonic()
        if (seen := self._marker_seen) is None or seen[0] != (mtime, ino):
            self._marker_seen = (mtime, ino), now
            return 0.0
        return now - seen[1]

    def _unprobeable_holder(self) -> str | None:
        # This host cannot prove such a holder dead, so whoever decides whether it is gone needs its name.
        with suppress(OSError, ValueError):
            if (holder := self._recorded_holder(_read_lock_file(self.lock_file)[0])) and holder[1] != host_name():
                return f"pid {holder[0]} on {holder[1]}"
        return None

    @classmethod
    def _recorded_holder(cls, content: str | None) -> tuple[int, str] | None:
        return None if (holder := _parse_lock_holder(content)) is None else holder[:2]

    @staticmethod
    def _write_lock_info(fd: int) -> None:
        # No suppression: a write failure must reach the acquisition rollback so it never publishes a half-written
        # marker as held state. The optional third line is this process's start token, absent when the platform
        # exposes no proven start time, in which case a reader falls back to PID-only liveness.
        info = f"{os.getpid()}\n{host_name()}\n"
        if (token := process_start_token(os.getpid())) is not None:
            info += f"{token}\n"
        write_all(fd, info.encode())

    @property
    def pid(self) -> int | None:
        """
        The PID of the process holding this lock, read from the lock file.

        :returns: the PID as an integer, or ``None`` if the lock file does not exist or cannot be parsed

        """
        with suppress(OSError, ValueError):
            holder = _parse_lock_holder(_read_lock_file(self.lock_file)[0])
            if holder is not None:
                return holder[0]
        return None

    @property
    def is_lock_held_by_us(self) -> bool:
        """
        Whether this lock is held by the current process.

        :returns: ``True`` if the lock file exists and names the current process's PID, hostname and, when recorded,
            start token

        """
        with suppress(OSError, ValueError):
            if (holder := _parse_lock_holder(_read_lock_file(self.lock_file)[0])) is not None:
                return owner_is_current_process(*holder)
        return False

    def break_lock(self) -> None:
        """Forcibly break the lock by removing the lock file, regardless of who holds it."""
        with suppress(OSError):
            Path(self.lock_file).unlink()

    def _release_current(self, *, force: bool) -> None:
        super()._release_current(force=force)
        self._finish_marker_cleanup()

    def _release(self) -> None:
        fd = self._context.lock_file_fd
        assert fd is not None  # ruff:ignore[assert]  # _release runs only while held, so the descriptor is set
        # Capture the held file's identity before closing so cleanup can refuse to unlink a successor's marker. A
        # supported lifetime lease lets a peer break our expired marker and create its own at this path before we
        # release; unlinking by path alone would then delete the successor's lock.
        identity: tuple[int, int] | None = None
        with suppress(OSError):
            identity = _file_identity(os.fstat(fd))
        if sys.platform == "win32":  # pragma: win32 cover
            self._release_then_unlink(fd, identity)
        else:  # pragma: win32 no cover
            # Our open descriptor keeps the inode allocated, so its number cannot name a successor's marker yet, and an
            # interrupt before the unlink lands leaves the lock held for the retried release to finish.
            unlinked: Final = self._unlink_open_marker(identity)
            # A failed close may already have released and recycled the descriptor number. Relinquish it before the
            # one close attempt so no later release can close an unrelated descriptor that reused the same integer.
            self._mark_descriptor_released()
            try:
                self._close_released_fd(fd, default_suppresses=False)
            finally:
                # A share that refuses to delete an open file gets the unlink after the close; the marker still exists
                # until then, so its inode cannot be reused either.
                if not unlinked:
                    self._unlink_held_marker(identity)

    def _release_then_unlink(self, fd: int, identity: tuple[int, int] | None) -> None:  # pragma: win32 cover
        # Windows cannot unlink an open file, so the unlink outlives the hold until it lands, here or on a later acquire
        # or release. Only the gate's holder touches the pending record, so two threads never overwrite each other's.
        with self._transition_lock:
            self._marker_fds = [fd]
            self._marker_cleanup = identity
            self._mark_descriptor_released()
            try:
                self._close_marker_fd()
            # Marker cleanup must also run for control-flow exceptions, and both failures must remain observable.
            except BaseException as close_error:
                try:
                    self._finish_marker_cleanup()
                except BaseException as cleanup_error:  # ruff:ignore[blind-except]  # keep control-flow failures
                    _raise_grouped_errors(
                        "lock descriptor close and marker cleanup both failed",
                        close_error,
                        cleanup_error,
                    )
                raise
            self._finish_marker_cleanup()

    def _unlink_open_marker(self, identity: tuple[int, int] | None) -> bool:  # pragma: win32 no cover
        try:
            with suppress(FileNotFoundError):
                if identity is not None and _file_identity(os.lstat(self.lock_file)) == identity:
                    Path(self.lock_file).unlink()
        except OSError:
            return False
        return True

    def _finish_marker_cleanup(self) -> None:
        # Never waits: whoever holds the gate is acquiring or releasing this lock and finishes the cleanup itself.
        if self._marker_cleanup is not None and self._transition_lock.acquire(blocking=False):  # pragma: win32 cover
            try:
                # A close error here belongs to the release an interrupt cut short, not to this caller.
                with suppress(OSError):
                    self._close_marker_fd()
                if (identity := self._marker_cleanup) is not None:  # pragma: no branch  # the gate's holder clears it
                    self._windows_unlink_if_ours(identity)
            finally:
                self._transition_lock.release()

    def _close_marker_fd(self) -> None:  # pragma: win32 cover
        # Popping and closing is one chain of C calls that no signal handler can split, so a retried release closes the
        # descriptor at most once; a second close could hit an unrelated file that reused the number.
        try:
            list(map(os.close, islice(iter(vars(self).setdefault("_marker_fds", []).pop, None), 1)))
        except IndexError:
            return
        except OSError:
            if not self._suppresses_close_error(default_suppresses=False):
                raise

    def _unlink_held_marker(self, identity: tuple[int, int] | None) -> None:
        if identity is None:
            return
        if sys.platform == "win32":  # pragma: win32 cover
            self._windows_unlink_if_ours(identity)
        else:  # pragma: win32 no cover
            with suppress(OSError):
                if _file_identity(os.lstat(self.lock_file)) == identity:
                    Path(self.lock_file).unlink()

    def _windows_unlink_if_ours(self, identity: tuple[int, int]) -> None:  # pragma: win32 cover
        path: Final = self.lock_file
        retry_delay = 0.001
        for attempt in range(_UNLINK_MAX_RETRIES):
            # Windows doesn't immediately release file handles after close, causing EACCES/EPERM on unlink. Recheck
            # identity each attempt: a failed unlink leaves a window for a successor to replace the marker at this path.
            try:
                if _file_identity(os.lstat(path)) == identity:
                    # Forgotten before the unlink: Windows does not promise unique file IDs, so no retry may reach a
                    # successor that reused ours.
                    self._marker_cleanup = None
                    os.unlink(path)  # ruff:ignore[os-unlink]  # Path.unlink runs Python code a handler can interrupt
            except OSError as exc:
                if exc.errno in {EACCES, EPERM}:
                    # The file is still ours, so the next acquire or release tries again if these attempts run out.
                    self._marker_cleanup = identity
                    if attempt < _UNLINK_MAX_RETRIES - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    continue
            self._marker_cleanup = None
            return


def _file_identity(st: os.stat_result) -> tuple[int, int]:
    # (st_dev, st_ino) names the concrete inode behind a path, so a marker recreated at the same pathname after an
    # expired lease reads as a different file. CPython populates both on Windows from the volume serial and file index.
    return st.st_dev, st.st_ino


def _read_lock_file(path: str) -> tuple[str | None, float, int]:
    # A legitimate lock file is always a regular file. Classify the path with lstat first, so any other node (symlink,
    # FIFO, socket, device) is reported as a malformed lock the caller can evict, without an os.open that would follow
    # a symlink, stall on a FIFO, or fail on a socket and leave acquisition wedged. The mtime and inode still flow back
    # for the identity-checked stale break. lstat, not stat, so a hostile symlink is never followed onto its target.
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):  # pragma: needs fifo
        return None, st.st_mtime, st.st_ino
    # Re-check on the opened handle: O_NOFOLLOW refuses a symlink swapped in after the lstat, O_NONBLOCK stops a FIFO
    # swapped in from stalling the open, and the fstat catches any other non-regular replacement race before we read.
    # The capped read stops a huge regular file (e.g. one filled from /dev/zero) from exhausting memory.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):  # pragma: no cover  # only a non-regular node swapped in after the lstat
            return None, st.st_mtime, st.st_ino
        data = os.read(fd, _MAX_LOCK_FILE_SIZE + 1)
    finally:
        os.close(fd)
    if len(data) <= _MAX_LOCK_FILE_SIZE:
        with suppress(UnicodeDecodeError):
            return data.decode("utf-8"), st.st_mtime, st.st_ino
    return None, st.st_mtime, st.st_ino


def _parse_lock_holder(content: str | None) -> tuple[int, str, int | None] | None:
    # A well-formed lock file is "<pid>\n<hostname>\n" with an optional "<start_token>\n" third line naming the
    # holder's process start instant (a filelock 3.29 marker wrote this only on Windows; every platform writes it now).
    # Anything else (wrong line count, a non-integer PID or start token, empty or unreadable content) is unparsable;
    # returning None lets the caller treat it as a malformed lock to self-heal rather than a holder.
    # Every writer since 3.22 ends each line with "\n", so a record without one was cut by a crash or read mid-write. As
    # a holder it would name a foreign host no one reclaims, or a start token that lets a contender break a live lock.
    if not content or not content.endswith("\n") or len(lines := content.strip().splitlines()) not in {2, 3}:
        return None
    try:
        pid = parse_decimal(lines[0])
        start_token = parse_decimal(lines[2]) if len(lines) == _MARKER_WITH_START_TOKEN_LINE_COUNT else None
    except ValueError:
        return None
    # A pid outside the valid range is a malformed lock, not a holder. Without this, a non-positive pid
    # reaches os.kill() where 0 / -1 mean "the caller's own process group / every process" so a dead
    # holder reads as alive and the lock is never reclaimed, while an oversized pid raises OverflowError
    # (not OSError/ValueError) out of the self-heal path. _parse_marker_bytes already enforces this range.
    if not 1 <= pid <= 2**31 - 1:
        return None
    return pid, lines[1], start_token


def parse_decimal(text: str) -> int:
    """
    Read a PID or start token as a writer formats it: ASCII digits only.

    :raises ValueError: for anything else ``int()`` would accept, such as ``1_000``, surrounding whitespace, a sign or
        non-ASCII digits; no filelock writes those

    """
    if not (text.isascii() and text.isdigit()):
        msg = f"not a decimal integer: {text!r}"
        raise ValueError(msg)
    return int(text)


__all__ = [
    "MALFORMED_LOCK_AGE_THRESHOLD",
    "SoftFileLock",
    "parse_decimal",
]
<|endoftext|>"""A minimal native lock over a caller-owned file descriptor, contending with :class:`FileLock` on the same file."""

from __future__ import annotations

import sys
import time
from threading import TIMEOUT_MAX
from typing import Final

from ._api import _seconds

if sys.platform == "win32":  # pragma: win32 cover
    from ._windows import _lock_fd_nonblocking, _unlock_fd
else:  # pragma: win32 no cover
    from ._unix import _lock_fd_nonblocking, _unlock_fd


def lock_descriptor(fd: int, *, blocking: bool = True, poll_interval: float = 0.05) -> bool:
    """
    Take the native OS lock on *fd*, a file descriptor the caller opened and owns.

    This is the same one-byte exclusive lock :class:`FileLock` uses, so a descriptor lock and a path lock on the same
    file contend with each other. Unlike :class:`FileLock` it adds no path handling: it never opens, truncates, closes,
    unlinks, chmods, canonicalizes, or falls back. The caller owns *fd* before, during, and after the call, and must
    close it. On Windows *fd* must be a synchronous descriptor (its handle not opened with ``FILE_FLAG_OVERLAPPED``).

    For timeout, reentrancy, singleton, lifetime, or stale-break behavior, use :class:`FileLock`. There is no async
    wrapper. Run this in an executor, or drive ``blocking=False`` from your own polling loop.

    :param fd: an open file descriptor the caller owns.
    :param blocking: when ``True`` (default), retry the nonblocking attempt every *poll_interval* seconds until it
        succeeds; when ``False``, make one attempt.
    :param poll_interval: positive seconds between attempts while blocking, at most :data:`threading.TIMEOUT_MAX`;
        ignored when *blocking* is ``False``.

    :returns: ``True`` once the lock is held, or ``False`` on contention when ``blocking`` is ``False``.

    :raises OSError: for a permanent native failure, such as an invalid descriptor, or with ``errno.ENOSYS`` when the
        Python build lacks the native locking primitive. The descriptor is left open.
    :raises ValueError: if a blocking call receives a *poll_interval* outside that range, including ``nan``.

    .. versionadded:: 3.30.0

    """
    if not blocking:
        return _lock_fd_nonblocking(fd)
    # time.sleep raises OverflowError past TIMEOUT_MAX; nan fails both comparisons.
    if not 0 < (poll_interval := _seconds("poll_interval", poll_interval)) <= TIMEOUT_MAX:
        msg: Final[str] = f"poll_interval must be greater than 0 and at most {TIMEOUT_MAX}, got {poll_interval}"
        raise ValueError(msg)
    while not _lock_fd_nonblocking(fd):
        time.sleep(poll_interval)
    return True


def unlock_descriptor(fd: int) -> None:
    """
    Release the native OS lock on *fd* without touching the descriptor.

    :param fd: the descriptor a prior :func:`lock_descriptor` locked; the caller still owns and must close it.

    :raises OSError: if the native unlock fails, including ``errno.ENOSYS`` when the Python build lacks the native
        locking primitive; the caller may retry on the same descriptor.

    .. versionadded:: 3.30.0

    """
    _unlock_fd(fd)


__all__ = [
    "lock_descriptor",
    "unlock_descriptor",
]
<|endoftext|>"""An asyncio-based implementation of the file lock."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import sys
import time
from concurrent import futures
from dataclasses import dataclass, fields
from inspect import iscoroutinefunction
from threading import RLock
from typing import TYPE_CHECKING, Final, NoReturn, TypeVar, cast

from ._api import (
    _UNSET_FILE_MODE,
    BaseFileLock,
    CloseErrorPolicy,
    ContextErrorPolicy,
    FileLockContext,
    FileLockMeta,
    ThreadLocalFileContext,
    _append_exception_context,
    _canonical,
    _delegate_to_current,
    _ExtraValue,
    _fork_transition,
    _grouped_errors,
    _raise_body_and_release,
    _raise_chained_errors,
    _raise_cleanup_errors,
    _raise_grouped_errors,
    _register_fork_object,
    _resolve_poll_interval,
    _resolve_timeout,
)
from ._async import (
    _AsyncTransitionGate,
    _AsyncTransitionUnavailableError,
    _BackendOutcome,
    _capture_awaitable,
    _capture_call,
    _drain_future,
    _future_result,
    _run_in_executor,
    _wait_until_done,
)
from ._error import Timeout
from ._lease import SoftFileLease
from ._soft import SoftFileLock
from ._strict import StrictSoftFileLock
from ._unix import UnixFileLock
from ._windows import WindowsFileLock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Hashable, Mapping
    from types import TracebackType

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Self
    else:  # pragma: no cover (<py311)
        from typing_extensions import Self


_LOGGER: Final[logging.Logger] = logging.getLogger("filelock")
_ASYNC_RELEASE_CANCELLATION_ERRORS: Final[str] = "lock release cancellation and backend release both failed"
_ASYNC_CONTEXT_RELEASE_ERRORS: Final[str] = "context body, release cancellation, and backend release failed"
_ASYNC_RELEASE_CANCELLATION_MARKER_ATTR: Final[str] = "_filelock_async_release_cancellation"
_ASYNC_RELEASE_CANCELLATION_MARKER: Final[list[None]] = []
_FINALIZER_RELEASES: Final[set[asyncio.Task[None] | futures.Future[None]]] = set()

_AT = TypeVar("_AT", bound="BaseAsyncFileLock")
_T = TypeVar("_T")


class AsyncFileLockMeta(FileLockMeta):
    def __call__(  # ruff:ignore[too-many-arguments]  # forwards the public constructor's documented parameters
        cls: type[_AT],  # ruff:ignore[invalid-first-argument-name-for-method]  # metaclass __call__ receives the class being constructed
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = _UNSET_FILE_MODE,
        thread_local: bool = False,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        poll_interval: float = 0.05,
        lifetime: float | None = None,
        context_error_policy: ContextErrorPolicy = "chain",
        close_error_policy: CloseErrorPolicy = "default",
        fallback_to_soft: bool = True,
        preserve_lock_file: bool = False,
        on_acquired: Callable[[int], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        run_in_executor: bool = True,
        executor: futures.Executor | None = None,
        **kwargs: _ExtraValue,
    ) -> _AT:
        if thread_local and run_in_executor:
            msg = "run_in_executor is not supported when thread_local is True"
            raise ValueError(msg)
        return super().__call__(  # a subclass may add options of its own, as AsyncSoftFileLease does
            **kwargs,
            lock_file=lock_file,
            timeout=timeout,
            mode=mode,
            thread_local=thread_local,
            blocking=blocking,
            is_singleton=is_singleton,
            poll_interval=poll_interval,
            lifetime=lifetime,
            context_error_policy=context_error_policy,
            close_error_policy=close_error_policy,
            fallback_to_soft=fallback_to_soft,
            preserve_lock_file=preserve_lock_file,
            on_acquired=on_acquired,
            loop=loop,
            run_in_executor=run_in_executor,
            executor=executor,
        )


class BaseAsyncFileLock(BaseFileLock, metaclass=AsyncFileLockMeta):
    """
    Base class for asynchronous file locks.

    .. versionadded:: 3.15.0

    """

    _deadlock_holder_desc: str = "BaseAsyncFileLock instance in this task"
    _constructor_lifetime_warning_stacklevel: int = 4

    @staticmethod
    def _deadlock_scope() -> Hashable | None:
        # One event loop thread runs every task, so a thread-scoped registry cannot tell a same-task reacquire
        # (a real deadlock: the polling task never reaches its own release) from another task queuing behind the
        # holder (no deadlock: each poll yields, so the holder runs on and releases). Only the first may fail
        # fast, so scope holders to the task.
        return asyncio.current_task()

    def __init__(  # ruff:ignore[too-many-arguments]  # public constructor: one parameter per documented lock option
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = _UNSET_FILE_MODE,
        thread_local: bool = False,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        poll_interval: float = 0.05,
        lifetime: float | None = None,
        context_error_policy: ContextErrorPolicy = "chain",
        close_error_policy: CloseErrorPolicy = "default",
        fallback_to_soft: bool = True,
        preserve_lock_file: bool = False,
        on_acquired: Callable[[int], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        run_in_executor: bool = True,
        executor: futures.Executor | None = None,
    ) -> None:
        """
        Create a new lock object.

        :param lock_file: path to the file
        :param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in the
            acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it to a
            negative value. A timeout of 0 means that there is exactly one attempt to acquire the file lock.
        :param mode: file permissions for the lockfile. When not specified, the OS controls permissions via umask and
            default ACLs, preserving POSIX default ACL inheritance in shared directories.
        :param thread_local: Whether this object's internal context should be thread local or not. If this is set to
            ``False`` (the default) then the lock will be reentrant across threads. When ``True``, **all fields of the
            lock's internal context are per-thread**, including the configuration values ``poll_interval``, ``timeout``,
            ``blocking``, ``mode``, and ``lifetime``. Setting one of these properties from one thread does not change
            the value seen by another thread; threads that did not perform the write continue to see the value supplied
            at construction time. ``mode`` has no setter, so construction is the only place it is ever set. If you need
            configuration values to be visible across threads, construct the lock with ``thread_local=False``.
        :param blocking: whether the lock should be blocking or not
        :param is_singleton: If this is set to ``True`` then only one instance of this class will be created per lock
            file. This is useful if you want to use the lock object for reentrant locking without needing to pass the
            same object around.
        :param poll_interval: default interval for polling the lock file, in seconds. It will be used as fallback value
            in the acquire method, if no poll_interval value (``None``) is given.
        :param lifetime: for :class:`AsyncSoftFileLock`, the age in seconds after which a waiting process may delete
            the marker, even while its holder remains alive. This legacy expiry mode does not provide strict mutual
            exclusion. ``None`` (the default) disables age-based expiry. Native OS locks (:class:`AsyncFileLock`)
            cannot be revoked by file age and ignore a non-``None`` ``lifetime`` with a warning.
        :param context_error_policy: how a context manager reconciles a failure in its body with a failure while
            releasing on exit. ``"chain"`` (the default) keeps Python's behavior: the release error propagates with the
            body error in its ``__context__``. ``"group"`` raises a :class:`BaseExceptionGroup` holding the body error
            first and the release error second, so neither hides the other.
        :param close_error_policy: for native locks (:class:`AsyncFileLock`), what to do with an ``os.close`` failure
            after the OS unlock has already committed. ``"default"`` keeps each platform's historical behavior,
            ``"raise"`` always propagates the ``OSError``, and ``"suppress"`` always ignores it.
        :param fallback_to_soft: for :class:`AsyncFileLock`, whether to fall back to soft existence locking when
            ``flock`` returns ``ENOSYS``. ``True`` (default) keeps the fallback; ``False`` propagates the error.
        :param preserve_lock_file: for native locks (:class:`AsyncFileLock`), whether filelock promises not to unlink
            the lock pathname on release. ``False`` (default) keeps each backend's cleanup; ``True`` keeps a stable file
            identity (Windows skips its unlink, Unix refuses the ``ENOSYS`` soft fallback). :class:`AsyncSoftFileLock`
            rejects ``True``.
        :param on_acquired: for native locks (:class:`AsyncFileLock`), a callable invoked with the borrowed lock
            descriptor once per physical acquisition, after the lock is held but before
            :meth:`~BaseAsyncFileLock.acquire` returns. With ``run_in_executor=True`` (the default) it runs in the
            backend executor. It must not close or unlock the descriptor; a raise rolls the acquisition back.
            :class:`AsyncSoftFileLock` rejects it.
        :param loop: the event loop a garbage-collected lock schedules its release on when no loop runs in the
            collecting thread. :meth:`~BaseAsyncFileLock.acquire` and :meth:`~BaseAsyncFileLock.release` always run
            on the loop that awaits them.
        :param run_in_executor: If this is set to ``True`` then the lock will be acquired in an executor.
        :param executor: The executor to use. If not specified, the default executor will be used.

        """
        self._creator_pid = os.getpid()
        self._is_thread_local = thread_local
        self._is_singleton = is_singleton
        self._context_error_policy = context_error_policy  # already validated by the metaclass
        self._close_error_policy = close_error_policy  # already validated by the metaclass
        self._fallback_to_soft = fallback_to_soft
        self._preserve_lock_file = preserve_lock_file  # already validated by the metaclass
        self._on_acquired = on_acquired  # already validated by the metaclass
        self._transition_gate: Final[_AsyncTransitionGate] = _AsyncTransitionGate()
        # BaseFileLock.__init__ never runs here, and the finalizer releases thread-local holds through these.
        self._transition_lock = RLock()
        self._thread_holds = {}
        self._thread_pending = {}

        self._context: AsyncFileLockContext = (AsyncThreadLocalFileContext if thread_local else AsyncFileLockContext)(
            lock_file=os.fspath(lock_file),
            timeout=timeout,
            mode=mode,
            blocking=blocking,
            poll_interval=poll_interval,
            lifetime=lifetime,
            loop=loop,
            run_in_executor=run_in_executor,
            executor=executor,
        )
        _register_fork_object(self)

    def _singleton_extra_mismatches(self, kwargs: Mapping[str, _ExtraValue], /) -> dict[str, tuple[str, str]]:
        mismatches = super()._singleton_extra_mismatches(kwargs)
        # Loops and executors compare by identity: a different one schedules the lock's work elsewhere.
        mismatches.update(
            (name, (str(kwargs[name]), str(value)))
            for name, value in (("loop", self.loop), ("executor", self.executor))
            if kwargs[name] is not value
        )
        if kwargs["run_in_executor"] != self.run_in_executor:
            mismatches["run_in_executor"] = (str(kwargs["run_in_executor"]), str(self.run_in_executor))
        return mismatches

    @property
    def run_in_executor(self) -> bool:
        """Whether run in executor."""
        return self._context.run_in_executor

    @property
    def executor(self) -> futures.Executor | None:
        """The executor."""
        return self._context.executor

    @executor.setter
    def executor(self, value: futures.Executor | None) -> None:  # pragma: no cover
        """
        Change the executor.

        :param futures.Executor | None value: the new executor or ``None``

        """
        self._context.executor = value

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop."""
        return self._context.loop

    async def acquire(  # ty: ignore[invalid-method-override]
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        blocking: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> AsyncAcquireReturnProxy:
        """
        Try to acquire the file lock.

        :param timeout: maximum wait time for acquiring the lock, ``None`` means use the default
            :attr:`~BaseFileLock.timeout` is and if ``timeout < 0``, there is no timeout and this method will block
            until the lock could be acquired
        :param poll_interval: interval of trying to acquire the lock file, ``None`` means use the default
            :attr:`~BaseFileLock.poll_interval`
        :param blocking: whether to retry until this call holds the lock or the timeout expires, ``None`` means use
            the default :attr:`~BaseFileLock.blocking`. ``False`` makes one attempt and raises :class:`~Timeout` on
            failure.
        :param cancel_check: a callable returning ``True`` when the acquisition should be canceled. Checked on each poll
            iteration. When triggered, raises :class:`~Timeout` just like an expired timeout.

        :returns: a context object that will unlock the file when the context is exited

        :raises Timeout: if fails to acquire lock within the timeout period
        :raises ValueError: if a blocking call gets a ``nan`` timeout, or *poll_interval* exceeds
            :data:`threading.TIMEOUT_MAX`

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            async with await lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            await lock.acquire()
            try:
                pass
            finally:
                await lock.release()

        """
        self._raise_if_inherited()
        if timeout is None:
            timeout = self._context.timeout

        if blocking is None:
            blocking = self._context.blocking

        if poll_interval is None:
            poll_interval = self._context.poll_interval
        poll_interval = _resolve_poll_interval(poll_interval)
        timeout = _resolve_timeout(timeout, blocking=blocking)

        start_time = time.perf_counter()
        try:
            return await self._acquire_with_admission(
                blocking=blocking,
                cancel_check=cancel_check,
                timeout=timeout,
                poll_interval=poll_interval,
                start_time=start_time,
            )
        except _AsyncTransitionUnavailableError:
            raise Timeout(self.lock_file) from None

    async def _acquire_with_admission(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        timeout: float,
        poll_interval: float,
        start_time: float,
    ) -> AsyncAcquireReturnProxy:
        async with self._transition_gate.hold_for_acquire(
            blocking=blocking,
            cancel_check=cancel_check,
            deadline=None if timeout < 0 else start_time + timeout,
            poll_interval=poll_interval,
        ):
            # Re-entry only counts, so it skips the executor round trip that resolving the path costs.
            if self.is_locked:
                self._context.lock_counter += 1
                return AsyncAcquireReturnProxy(lock=self)
            # A canceled provisional acquire must finish rollback before another caller can claim its descriptor.
            canonical = await self._off_loop(functools.partial(_canonical, self.lock_file))
            self._context.lock_counter += 1
            self._raise_if_would_deadlock(canonical, timeout=timeout, blocking=blocking)
            self._context.claim_root = canonical
            try:
                await self._async_poll_until_acquired(
                    blocking=blocking,
                    cancel_check=cancel_check,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    start_time=start_time,
                )
            except BaseException:
                self._reconcile_failed_acquire(canonical)
                raise
            finally:
                self._context.claim_root = None
            self._commit_acquire(canonical)
            return AsyncAcquireReturnProxy(lock=self)

    async def _off_loop(self, call: Callable[[], _T]) -> _T:
        # A slow mount stalls a filesystem call, and the loop with it; a coroutine backend runs on the loop anyway.
        if not self.run_in_executor or iscoroutinefunction(self._acquire):
            return call()
        future = _run_in_executor(asyncio.get_running_loop(), self.executor, call)
        await _wait_until_done(future)
        return _future_result(future)

    async def _async_poll_until_acquired(
        self,
        *,
        blocking: bool,
        cancel_check: Callable[[], bool] | None,
        timeout: float,
        poll_interval: float,
        start_time: float,
    ) -> None:
        lock_id = id(self)
        lock_filename = self.lock_file
        while True:
            self._raise_if_inherited()
            _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
            await self._run_acquire_attempt()
            self._raise_if_inherited()
            if self.is_locked:
                _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                return
            if self._check_give_up(
                blocking=blocking,
                cancel_check=cancel_check,
                timeout=timeout,
                start_time=start_time,
            ):
                raise Timeout(lock_filename, await self._off_loop(self._unprobeable_holder))
            delay = poll_interval
            if timeout >= 0:
                delay = min(delay, max(start_time + timeout - time.perf_counter(), 0.0))
            _LOGGER.debug("Lock %s not acquired on %s, waiting %s seconds ...", lock_id, lock_filename, delay)
            await asyncio.sleep(delay)

    async def _run_acquire_attempt(self) -> None:
        acquire_future = self._start_internal_method(
            self._acquire_with_fork_tracking_async
            if iscoroutinefunction(self._acquire)
            else self._break_expired_and_acquire
        )
        try:
            await _wait_until_done(acquire_future)
        except asyncio.CancelledError as cancellation:
            acquire_error: BaseException | None = None
            try:
                await _drain_future(acquire_future)
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                acquire_error = error

            rollback_error: BaseException | None = None
            if self.is_locked:
                try:
                    await _drain_future(self._start_tracked_release())
                except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                    rollback_error = error

            if acquire_error is not None:
                if rollback_error is not None:  # pragma: needs fcntl
                    self._raise_cancelled_errors(
                        "lock acquisition cancellation, backend attempt, and rollback failed",
                        cancellation,
                        acquire_error,
                        rollback_error,
                    )
                self._raise_cancelled_errors(
                    "lock acquisition cancellation and backend attempt both failed", cancellation, acquire_error
                )
            if rollback_error is not None:
                self._raise_cancelled_errors(
                    "lock acquisition cancellation and rollback both failed", cancellation, rollback_error
                )
            raise
        try:
            _future_result(acquire_future)
        except asyncio.CancelledError as acquire_error:
            await self._rollback_backend_cancelled_acquire(acquire_error)

    async def _rollback_backend_cancelled_acquire(self, acquire_error: asyncio.CancelledError) -> NoReturn:
        if self.is_locked:
            try:
                await _drain_future(self._start_tracked_release())
            except BaseException as rollback_error:  # ruff:ignore[blind-except]  # both backend errors must surface
                self._raise_acquire_rollback_errors(acquire_error, rollback_error)
        raise acquire_error

    def _raise_acquire_rollback_errors(self, acquire_error: BaseException, rollback_error: BaseException) -> NoReturn:
        if self._context_error_policy == "group":
            _raise_grouped_errors("lock acquisition backend and rollback both failed", acquire_error, rollback_error)
        _raise_chained_errors(acquire_error, rollback_error)

    def _raise_cancelled_errors(
        self,
        message: str,
        cancellation: asyncio.CancelledError,
        first_error: BaseException,
        second_error: BaseException | None = None,
    ) -> NoReturn:
        if self._context_error_policy == "group":
            marker = (
                (_ASYNC_RELEASE_CANCELLATION_MARKER_ATTR, _ASYNC_RELEASE_CANCELLATION_MARKER)
                if message == _ASYNC_RELEASE_CANCELLATION_ERRORS
                else None
            )
            if second_error is None:
                _raise_grouped_errors(message, cancellation, first_error, marker=marker)
            _raise_grouped_errors(message, cancellation, first_error, second_error, marker=marker)
        if (context := first_error.__context__) is not None and context is not cancellation:
            if (cancellation_context := cancellation.__context__) is not None:
                _append_exception_context(context, cancellation_context)
            cancellation.__context__ = context
        first_error.__context__ = cancellation
        _raise_chained_errors(first_error, second_error)

    async def release(self, force: bool = False) -> None:  # ty: ignore[invalid-method-override]  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        """
        Release the file lock. The lock is only completely released when the lock counter reaches 0. The lock file
        itself may be deleted automatically, the behavior is platform-specific.

        :param force: If true, the lock counter is ignored and the lock is released in every case.

        """
        if self._creator_pid != os.getpid() or not self.is_locked:
            return
        async with self._transition_gate.hold():
            await self._release_serialized(force=force)

    async def _release_serialized(self, *, force: bool) -> None:
        if not self.is_locked:
            return
        if not force and self._context.lock_counter > 1:
            self._context.lock_counter -= 1
            return

        lock_id, lock_filename = id(self), self.lock_file
        _LOGGER.debug("Attempting to release lock %s on %s", lock_id, lock_filename)
        release_future = self._start_tracked_release()
        try:
            await _wait_until_done(release_future)
        except asyncio.CancelledError as cancellation:
            try:
                await _drain_future(release_future)
            except BaseException as release_error:  # ruff:ignore[blind-except]  # cancellation and backend failure must both surface
                self._commit_release_if_released()
                self._raise_cancelled_errors(_ASYNC_RELEASE_CANCELLATION_ERRORS, cancellation, release_error)
            self._commit_release()
            raise
        try:
            _future_result(release_future)
        except BaseException:  # state follows the backend for control-flow exceptions too
            self._commit_release_if_released()
            raise
        self._commit_release()
        _LOGGER.debug("Lock %s released on %s", lock_id, lock_filename)

    def _commit_release_if_released(self) -> None:
        # Commit only when the backend actually unlocked (close or unlink failed after the OS unlock). If the lock is
        # still held, keep the counter so a later release can retry.
        if not self.is_locked:
            self._commit_release()

    def _start_internal_method(
        self, method: Callable[[], None] | Callable[[], Coroutine[None, None, None]]
    ) -> asyncio.Future[_BackendOutcome[None]]:
        if iscoroutinefunction(method):
            return asyncio.create_task(_capture_awaitable(cast("Callable[[], Coroutine[None, None, None]]", method)()))
        loop = asyncio.get_running_loop()
        sync_method = cast("Callable[[], None]", method)
        if self.run_in_executor:
            return _run_in_executor(loop, self.executor, sync_method)
        future: asyncio.Future[_BackendOutcome[None]] = loop.create_future()
        future.set_result(_capture_call(sync_method))
        return future

    def _start_tracked_release(self) -> asyncio.Future[_BackendOutcome[None]]:
        return self._start_internal_method(
            self._release_with_fork_tracking_async
            if iscoroutinefunction(self._release)
            else self._release_with_fork_tracking
        )

    def _break_expired_and_acquire(self) -> None:
        # Runs as one backend call so the lifetime check's lstat and unlink stay off the event loop with the acquire.
        self._try_break_expired_lock()
        self._acquire_with_fork_tracking()

    async def _acquire_with_fork_tracking_async(self) -> None:
        self._try_break_expired_lock()
        with _fork_transition(self):
            try:
                await cast("Callable[[], Awaitable[None]]", self._acquire)()
            except BaseException as acquisition_error:
                await self._rollback_failed_acquire_async(acquisition_error)
                raise
            try:
                self._register_context_descriptor()
            except BaseException as registration_error:  # cancellation must roll back the descriptor
                await self._rollback_failed_registration_async(registration_error)
                raise
        if self.is_locked:
            await self._invoke_on_acquired_async()

    async def _rollback_failed_acquire_async(self, acquisition_error: BaseException) -> None:
        if not self.is_locked:
            return
        registration_error: BaseException | None = None
        tracking_error: BaseException | None = None
        try:
            self._register_context_descriptor()
        except BaseException as error:  # ruff:ignore[blind-except]  # preserve registration and acquisition failures
            registration_error = error
            try:
                # Rollback may fail too; retain the fd so a child can close it without another identity probe.
                self._register_unverified_context_descriptor()
            except BaseException as error:  # ruff:ignore[blind-except]  # pragma: no cover - allocation/control-flow during fallback
                tracking_error = error
        try:
            await _drain_future(self._start_tracked_release())
        except BaseException as rollback_error:  # ruff:ignore[blind-except]  # preserve rollback and acquisition failures
            if registration_error is None and tracking_error is None:
                self._raise_acquire_rollback_errors(acquisition_error, rollback_error)
            _raise_cleanup_errors(
                "lock acquisition cleanup failed",
                acquisition_error,
                registration_error,
                tracking_error,
                rollback_error,
            )
        if registration_error is not None:
            _raise_cleanup_errors(
                "lock acquisition cleanup failed", acquisition_error, registration_error, tracking_error
            )

    async def _rollback_failed_registration_async(self, registration_error: BaseException) -> None:
        tracking_error: BaseException | None = None
        try:
            # Rollback may fail too; retain the fd so a child can close it without another identity probe.
            self._register_unverified_context_descriptor()
        except BaseException as error:  # ruff:ignore[blind-except]  # pragma: no cover - allocation/control-flow during fallback
            tracking_error = error
        try:
            await _drain_future(self._start_tracked_release())
        except BaseException as rollback_error:  # ruff:ignore[blind-except]  # preserve rollback and registration failures
            _raise_cleanup_errors(
                "descriptor registration cleanup failed", registration_error, tracking_error, rollback_error
            )
        if tracking_error is not None:  # pragma: no cover - requires failed in-memory fallback
            _raise_cleanup_errors("descriptor registration cleanup failed", registration_error, tracking_error)

    async def _invoke_on_acquired_async(self) -> None:
        if self._on_acquired is None or self._context.lock_counter != 1:
            return
        try:
            self._on_acquired(cast("int", self._context.lock_file_fd))
        except BaseException as callback_error:  # caller control-flow errors must release the lock
            callback_context = callback_error.__context__
            try:
                await _drain_future(self._start_tracked_release())
            except BaseException as release_error:  # ruff:ignore[blind-except]  # both errors surface via the group below
                _raise_body_and_release(callback_error, release_error)
            callback_error.__context__ = callback_context
            raise

    async def _release_with_fork_tracking_async(self) -> None:
        with _fork_transition(self):
            try:
                await cast("Callable[[], Awaitable[None]]", self._release)()
            finally:
                self._unregister_released_descriptor()

    def __enter__(self) -> NoReturn:
        """Sync context manager entry is not supported because lock acquisition is a coroutine."""
        msg = "Use `async with`: acquire/release are coroutines and cannot be awaited in a sync context manager."
        raise NotImplementedError(msg)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Sync context manager exit is not supported because lock release is a coroutine."""
        msg = "Use `async with`: acquire/release are coroutines and cannot be awaited in a sync context manager."
        raise NotImplementedError(msg)

    async def __aenter__(self) -> Self:
        """
        Acquire the lock.

        :returns: the lock object

        """
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the lock, reconciling a release failure with any body failure per :attr:`context_error_policy`.

        :param exc_type: the exception type if raised
        :param exc_value: the exception value if raised
        :param traceback: the exception traceback if raised

        """
        await self._release_in_context(exc_value)

    async def _release_in_context(  # ty: ignore[invalid-method-override]
        self, body_error: BaseException | None
    ) -> None:
        # The async counterpart of BaseFileLock._release_in_context: await release, then apply the same policy.
        try:
            await self.release()
        except BaseException as release_error:
            if body_error is None:
                raise
            if self._context_error_policy == "chain":
                _append_exception_context(release_error, body_error)
                raise
            if (
                errors := _grouped_errors(
                    release_error,
                    _ASYNC_RELEASE_CANCELLATION_ERRORS,
                    (_ASYNC_RELEASE_CANCELLATION_MARKER_ATTR, _ASYNC_RELEASE_CANCELLATION_MARKER),
                )
            ) is not None:
                match errors:
                    # The marker is only ever set on a (CancelledError, backend_error) pair, so this always matches.
                    case (asyncio.CancelledError() as cancellation, backend_error):  # pragma: no branch
                        _raise_grouped_errors(_ASYNC_CONTEXT_RELEASE_ERRORS, body_error, cancellation, backend_error)
            _raise_body_and_release(body_error, release_error)

    def __del__(self) -> None:
        """Release on deletion; safe to call during GC even when no event loop is running."""
        if vars(self).get("_creator_pid") != os.getpid():
            return  # pragma: forked child
        with contextlib.suppress(Exception):
            # A thread-local lock never uses an executor, so an in-place release matches the loop's and also reaches
            # threads whose loops it cannot schedule on.
            if self._is_thread_local and not iscoroutinefunction(self._release):
                self._release_without_loop()
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # At interpreter exit daemon threads no longer run, so a loop running in one never runs the release.
                if (loop := self._context.loop) is None or loop.is_closed() or sys.is_finalizing():
                    self._release_without_loop()
                    return
                if not loop.is_running():  # pragma: no cover
                    loop.run_until_complete(self.release(force=True))
                    return
                self._leave_shared_registries()
                # The stored loop runs in another thread, and only its own thread may create tasks on it.
                pending = asyncio.run_coroutine_threadsafe(self.release(force=True), loop)
            else:
                self._leave_shared_registries()
                pending = loop.create_task(self.release(force=True))
            # The loop keeps only a weak reference to a task, so hold one until the release finishes.
            _FINALIZER_RELEASES.add(pending)
            pending.add_done_callback(self._settle_finalizer_release)

    def _leave_shared_registries(self) -> None:
        # CPython finalizes before clearing weak references and the scheduled release keeps this lock alive, so the
        # singleton cache would hand it to a new caller whose hold the forced release then drops. Off the deadlock
        # registry, a new lock in this task waits for the release instead of failing fast. The cache lock is never
        # awaited: a constructor holds it while taking the fork locks, inside which the collector can run this.
        if self._instances_lock.acquire(blocking=False):
            try:
                for key in [key for key, instance in self._instances.items() if instance is self]:
                    del self._instances[key]
            finally:
                self._instances_lock.release()
        self._drop_registry_entry()

    def _settle_finalizer_release(self, pending: asyncio.Task[None] | futures.Future[None]) -> None:
        _FINALIZER_RELEASES.discard(pending)
        # Nothing awaits this release, so a failure or cancellation falls back to the sync release here.
        with contextlib.suppress(asyncio.CancelledError, futures.CancelledError):
            if pending.exception() is None:
                return
        with contextlib.suppress(Exception):
            self._release_without_loop()

    def _release_without_loop(self) -> None:
        # No loop can run a release here, so call a sync backend directly as BaseFileLock.__del__ does; a coroutine
        # backend cannot run and keeps its lock.
        if iscoroutinefunction(self._release):
            return
        if self.is_locked:
            try:
                self._release_with_fork_tracking()
            finally:
                self._commit_release_if_released()
        # A thread-local lock also holds for the threads that ran its loops, and nothing else releases those now.
        self._release_thread_holds(exited_only=False)


@dataclass
class AsyncFileLockContext(FileLockContext):
    """A dataclass which holds the context for a ``BaseAsyncFileLock`` object."""

    #: Whether run in executor
    run_in_executor: bool = True

    #: The executor
    executor: futures.Executor | None = None

    #: The loop
    loop: asyncio.AbstractEventLoop | None = None


class AsyncThreadLocalFileContext(AsyncFileLockContext, ThreadLocalFileContext):
    """A thread local version of the ``FileLockContext`` class."""

    # threading.local runs __init__ again with these arguments the first time each thread touches the context.
    def __init__(  # ruff:ignore[too-many-arguments]  # mirrors the AsyncFileLockContext fields set at construction
        self,
        *,
        lock_file: str,
        timeout: float,
        mode: int,
        blocking: bool,
        poll_interval: float,
        lifetime: float | None = None,
        run_in_executor: bool = True,
        executor: futures.Executor | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.current = AsyncFileLockContext(
            lock_file=lock_file,
            timeout=timeout,
            mode=mode,
            blocking=blocking,
            poll_interval=poll_interval,
            lifetime=lifetime,
            run_in_executor=run_in_executor,
            executor=executor,
            loop=loop,
        )


for _field in fields(AsyncFileLockContext):
    setattr(AsyncThreadLocalFileContext, _field.name, _delegate_to_current(_field.name))


class AsyncAcquireReturnProxy:
    """A context-aware object that will release the lock file when exiting."""

    def __init__(self, lock: BaseAsyncFileLock) -> None:  # ruff:ignore[undocumented-public-init]  # trivial release-on-exit proxy
        self.lock = lock

    async def __aenter__(self) -> BaseAsyncFileLock:  # ruff:ignore[undocumented-magic-method]  # returns the wrapped lock
        return self.lock

    async def __aexit__(  # ruff:ignore[undocumented-magic-method]  # releases the wrapped lock on exit
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.lock._release_in_context(exc_value)  # ruff:ignore[private-member-access]  # releases the wrapped lock's context


class AsyncSoftFileLock(SoftFileLock, BaseAsyncFileLock):
    """Simply watches the existence of the lock file."""

    _lifetime_replacements: tuple[str, str] | None = ("AsyncStrictSoftFileLock", "AsyncSoftFileLease")


class AsyncStrictSoftFileLock(StrictSoftFileLock, BaseAsyncFileLock):
    """Run strict owner-claim locking without blocking the event loop."""


class AsyncSoftFileLease(SoftFileLease, BaseAsyncFileLock):
    """Existence lock whose claim expires, so a peer may take it while the previous holder still runs."""


class AsyncUnixFileLock(UnixFileLock, BaseAsyncFileLock):
    """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""


class AsyncWindowsFileLock(WindowsFileLock, BaseAsyncFileLock):
    """Uses ``LockFileEx`` to hard lock a byte range of the lock file on Windows systems."""


__all__ = [
    "AsyncAcquireReturnProxy",
    "AsyncSoftFileLease",
    "AsyncSoftFileLock",
    "AsyncStrictSoftFileLock",
    "AsyncUnixFileLock",
    "AsyncWindowsFileLock",
    "BaseAsyncFileLock",
]


def __dir__() -> list[str]:
    # Hide the sync bases and helpers imported at top level.
    return list(__all__)
<|endoftext|>"""Async wrapper around :class:`ReadWriteLock` for use with ``asyncio``."""

from __future__ import annotations

import asyncio
import functools
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar, cast

from ._api import (
    _append_exception_context,
    _ensure_current_process,
    _fork_transition,
    _register_fork_object,
)
from ._async import (
    _BackendOutcome,
    _capture_call,
    _drain_future,
    _future_result,
    _raise_cancelled_error,
    _task_owners_for,
    _wait_until_done,
)
from ._read_write import ReadWriteLock

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from concurrent import futures
    from types import TracebackType


_P = ParamSpec("_P")
_R = TypeVar("_R")


class AsyncReadWriteLock:
    """
    Async wrapper around :class:`ReadWriteLock` for use in ``asyncio`` applications.

    This wrapper dispatches every blocking SQLite operation to a thread pool via ``loop.run_in_executor()`` because
    Python's :mod:`sqlite3` module has no async API. Each ``asyncio`` task owns its own hold. Tasks share a read lock,
    and a task asking for the write lock waits until the other holders release, with waiting writers ahead of new
    readers. As with :class:`ReadWriteLock`, upgrading or downgrading a held lock raises :class:`RuntimeError`.
    Singleton wrappers over one path share their task holds.

    :param lock_file: path to the SQLite database file used as the lock
    :param timeout: maximum wait time in seconds; ``-1`` waits up to SQLite's busy-timeout cap of about 23 days
    :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable
    :param is_singleton: if ``True``, reuse existing :class:`ReadWriteLock` instances for the same resolved path
    :param loop: event loop for ``run_in_executor``; ``None`` uses the running loop
    :param executor: executor for ``run_in_executor``. When ``None`` this lock creates and owns a dedicated
        single-thread executor so every operation runs on the same thread (SQLite affinity requires this) and shuts it
        down in :meth:`close`. This lock uses a caller-supplied executor as-is and never shuts it down, so after passing
        no executor call :meth:`close` to release the owned one.

    .. versionadded:: 3.21.0

    """

    def __init__(  # ruff:ignore[too-many-arguments]  # public constructor: one parameter per documented lock option
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ) -> None:
        creator_pid = os.getpid()
        self._creator_pid = creator_pid
        self._fork_invalidated = False
        self._closed = False
        _register_fork_object(self)
        with _fork_transition():
            self._lock = ReadWriteLock(lock_file, timeout, blocking=blocking, is_singleton=is_singleton)
            self._owners = _task_owners_for(self._lock)
            self._loop = loop
            self._owns_executor = executor is None
            self._executor = executor or ThreadPoolExecutor(max_workers=1)
            if os.getpid() != creator_pid:  # pragma: forked child
                msg = "AsyncReadWriteLock construction cannot continue after fork"
                raise RuntimeError(msg)

    @property
    def lock_file(self) -> str:
        """The path to the lock file."""
        return self._lock.lock_file

    @property
    def timeout(self) -> float:
        """The default timeout."""
        return self._lock.timeout

    @property
    def blocking(self) -> bool:
        """Whether blocking is enabled by default."""
        return self._lock.blocking

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop (or ``None`` for the running loop)."""
        return self._loop

    @property
    def executor(self) -> futures.Executor:
        """The executor used for ``run_in_executor`` (a dedicated single-thread one if none was supplied)."""
        return self._executor

    @asynccontextmanager
    async def read_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases a shared read lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        await self.acquire_read(timeout, blocking=blocking)
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            await self._release_in_context(body_error)

    @asynccontextmanager
    async def write_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases an exclusive write lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        await self.acquire_write(timeout, blocking=blocking)
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            await self._release_in_context(body_error)

    async def _release_in_context(self, body_error: BaseException | None) -> None:
        try:
            await self.release()
        except BaseException as release_error:
            if body_error is not None:
                _append_exception_context(release_error, body_error)
            raise

    async def acquire_read(
        self, timeout: float | None = None, *, blocking: bool | None = None
    ) -> AsyncAcquireReadWriteReturnProxy:
        """
        Acquire a shared read lock.

        See :meth:`ReadWriteLock.acquire_read` for full semantics.

        :param timeout: seconds to wait; ``None`` uses the instance setting; ``-1`` waits up to SQLite's
            busy-timeout cap of about 23 days
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` on contention;
            ``None`` uses the instance setting

        :returns: a proxy that can be used as an async context manager to release the lock

        :raises RuntimeError: if the calling task already holds the write lock
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self._raise_if_unusable()
        await self._acquire("read", timeout, blocking=blocking)
        return AsyncAcquireReadWriteReturnProxy(lock=self)

    async def acquire_write(
        self, timeout: float | None = None, *, blocking: bool | None = None
    ) -> AsyncAcquireReadWriteReturnProxy:
        """
        Acquire an exclusive write lock.

        See :meth:`ReadWriteLock.acquire_write` for full semantics.

        :param timeout: seconds to wait; ``None`` uses the instance setting; ``-1`` waits up to SQLite's
            busy-timeout cap of about 23 days
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` on contention;
            ``None`` uses the instance setting

        :returns: a proxy that can be used as an async context manager to release the lock

        :raises RuntimeError: if the calling task already holds the read lock
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds
        :raises TypeError: if *timeout* is a :class:`bool` or not a real number
        :raises ValueError: if a blocking call gets a ``nan`` *timeout* or a negative one other than ``-1``

        """
        self._raise_if_unusable()
        await self._acquire("write", timeout, blocking=blocking)
        return AsyncAcquireReadWriteReturnProxy(lock=self)

    async def _acquire(self, mode: Literal["read", "write"], timeout: float | None, *, blocking: bool | None) -> None:
        timeout = self.timeout if timeout is None else timeout
        blocking = self.blocking if blocking is None else blocking
        sync_acquire = self._lock.acquire_read if mode == "read" else self._lock.acquire_write

        def enter_backend(remaining: float) -> None:
            sync_acquire(remaining, blocking=blocking)
            self._owners.backend_thread = threading.get_ident()

        async def enter(remaining: float) -> None:
            await self._run_acquire(functools.partial(enter_backend, remaining))

        await self._owners.acquire(mode, timeout=timeout, blocking=blocking, lock_file=self.lock_file, enter=enter)

    async def release(self, *, force: bool = False) -> None:
        """
        Release one level of the calling task's hold.

        The wrapper releases the database lock once no task holds it.

        :param force: if ``True``, drop the calling task's whole hold at any nesting level

        :raises RuntimeError: if the calling task holds no lock and *force* is ``False``

        """
        _ensure_current_process()
        if self._inherited:  # pragma: needs fork
            return
        await self._owners.release(
            force=force, lock_file=self.lock_file, leave=functools.partial(self._submit, self._leave_backend)
        )

    async def close(self) -> None:
        """
        Release the lock (if held) and close the underlying SQLite connection.

        After calling this method, the lock instance is no longer usable.

        """
        _ensure_current_process()
        if self._inherited:  # pragma: needs fork
            return
        if self._closed:
            return
        close_future = self._submit(self._lock.close)
        try:
            await _wait_until_done(close_future)
        except asyncio.CancelledError as cancellation:
            try:
                await _drain_future(close_future)
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            self._closed = True
            self._owners.reset()
            self._shutdown_owned_executor()
            raise
        _future_result(close_future)
        self._closed = True
        self._owners.reset()
        # Wait for the worker to exit rather than letting it drain in the background: a caller that forks right
        # after closing deserves a single-threaded process, and os.fork warns about any surviving thread.
        if self._owns_executor:
            await asyncio.to_thread(functools.partial(self._executor.shutdown, wait=True))

    async def _run_acquire(self, acquire: Callable[[], None]) -> None:
        acquire_future = self._submit(acquire)
        try:
            await _wait_until_done(acquire_future)
        except asyncio.CancelledError as cancellation:
            try:
                await _drain_future(acquire_future)
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            try:
                await _drain_future(self._submit(self._leave_backend))
            except BaseException as error:  # ruff:ignore[blind-except]  # reported with the cancellation below
                _raise_cancelled_error(cancellation, error)
            raise
        _future_result(acquire_future)

    def _leave_backend(self) -> None:
        # Only the entering worker's level: a sync user of the same singleton keeps its own.
        self._lock._release_level_of(  # ruff:ignore[private-member-access]  # the wrapper owns this backend hold
            cast("int", self._owners.backend_thread)
        )

    def _submit(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> asyncio.Future[_BackendOutcome[_R]]:
        return (self._loop or asyncio.get_running_loop()).run_in_executor(
            self._executor,
            _capture_call,
            functools.partial(func, *args, **kwargs),
        )

    def _shutdown_owned_executor(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False)

    @property
    def _inherited(self) -> bool:
        return self._fork_invalidated or os.getpid() != self._creator_pid

    def _raise_if_unusable(self) -> None:
        _ensure_current_process()
        if self._inherited:  # pragma: needs fork
            msg = f"AsyncReadWriteLock on {self.lock_file} was invalidated by fork(); construct a new instance"
            raise RuntimeError(msg)
        if self._closed:
            msg = "Cannot operate on a closed database."
            raise sqlite3.ProgrammingError(msg)

    def _reset_after_fork_in_child(self) -> None:  # pragma: forked child
        self._fork_invalidated = True

    def __del__(self) -> None:
        # Safety net when close() was never called: shut down the executor we own so its worker thread does not
        # outlive the lock. shutdown(wait=False) never blocks.
        if os.getpid() == getattr(self, "_creator_pid", None) and getattr(self, "_owns_executor", False):
            self._executor.shutdown(wait=False)


class AsyncAcquireReadWriteReturnProxy:
    """Context-aware object that releases the async read/write lock on exit."""

    def __init__(self, lock: AsyncReadWriteLock) -> None:
        self.lock = lock

    async def __aenter__(self) -> AsyncReadWriteLock:
        return self.lock

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.lock.release()


__all__ = [
    "AsyncAcquireReadWriteReturnProxy",
    "AsyncReadWriteLock",
]
<|endoftext|>"""
A platform independent file lock that supports the with-statement.

.. autodata:: filelock.__version__
    :no-value:

.. autodata:: filelock.OwnerMode

.. autodata:: filelock.CompromiseReason

"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Final

from ._api import AcquireReturnProxy, BaseFileLock, CloseErrorPolicy, ContextErrorPolicy, LockOptions
from ._descriptor import lock_descriptor, unlock_descriptor
from ._error import LeaseSettingsMismatch, SoftFileLockLifetimeWarning, SoftFileLockProtocolError, Timeout
from ._lease import CompromiseReason, LeaseCompromise, SoftFileLease
from ._marker import MarkerSoftFileLock, OwnerMode, OwnerRecord

if TYPE_CHECKING:
    from ._async_read_write import (
        AsyncAcquireReadWriteReturnProxy,
        AsyncReadWriteLock,
    )
    from ._read_write import ReadWriteLock
else:
    try:
        from ._async_read_write import AsyncAcquireReadWriteReturnProxy, AsyncReadWriteLock
        from ._read_write import ReadWriteLock
    except ImportError:  # pragma: lacks sqlite3
        AsyncAcquireReadWriteReturnProxy = None
        AsyncReadWriteLock = None
        ReadWriteLock = None

from ._soft import SoftFileLock
from ._soft_rw import AsyncAcquireSoftReadWriteReturnProxy, AsyncSoftReadWriteLock, SoftReadWriteLock
from ._strict import StrictSoftFileClaim, StrictSoftFileClaimState, StrictSoftFileLock
from ._unix import UnixFileLock
from ._unix import has_fcntl as _has_fcntl
from ._windows import WindowsFileLock
from .asyncio import (
    AsyncAcquireReturnProxy,
    AsyncSoftFileLease,
    AsyncSoftFileLock,
    AsyncStrictSoftFileLock,
    AsyncUnixFileLock,
    AsyncWindowsFileLock,
    BaseAsyncFileLock,
)

# Keeps ``filelock.version`` the string 4.0.12 exposed rather than the submodule.
from .version import version

#: version of the project as a string
__version__: Final[str] = version

#: ``True`` when :func:`fcntl.flock` is available. Off Windows, :data:`FileLock` then resolves to
#: :class:`UnixFileLock`, otherwise to :class:`SoftFileLock`. Always ``False`` on Windows.
#:
#: :meta hide-value:
has_fcntl: Final[bool] = _has_fcntl


if sys.platform == "win32":  # pragma: win32 cover
    _FileLock: type[BaseFileLock] = WindowsFileLock
    _AsyncFileLock: type[BaseAsyncFileLock] = AsyncWindowsFileLock
else:  # pragma: win32 no cover # ruff:ignore[collapsible-else-if]  # the else carries the win32 no-cover pragma
    if has_fcntl:
        _FileLock: type[BaseFileLock] = UnixFileLock
        _AsyncFileLock: type[BaseAsyncFileLock] = AsyncUnixFileLock
    else:
        _FileLock = SoftFileLock
        _AsyncFileLock = AsyncSoftFileLock
        warnings.warn("only soft file lock is available", stacklevel=2)

if TYPE_CHECKING:
    FileLock = SoftFileLock
    AsyncFileLock = AsyncSoftFileLock
else:
    #: Alias for the lock, which should be used for the current platform.
    FileLock = _FileLock
    AsyncFileLock = _AsyncFileLock


__all__ = [
    "AcquireReturnProxy",
    "AsyncAcquireReadWriteReturnProxy",
    "AsyncAcquireReturnProxy",
    "AsyncAcquireSoftReadWriteReturnProxy",
    "AsyncFileLock",
    "AsyncReadWriteLock",
    "AsyncSoftFileLease",
    "AsyncSoftFileLock",
    "AsyncSoftReadWriteLock",
    "AsyncStrictSoftFileLock",
    "AsyncUnixFileLock",
    "AsyncWindowsFileLock",
    "BaseAsyncFileLock",
    "BaseFileLock",
    "CloseErrorPolicy",
    "CompromiseReason",
    "ContextErrorPolicy",
    "FileLock",
    "LeaseCompromise",
    "LeaseSettingsMismatch",
    "LockOptions",
    "MarkerSoftFileLock",
    "OwnerMode",
    "OwnerRecord",
    "ReadWriteLock",
    "SoftFileLease",
    "SoftFileLock",
    "SoftFileLockLifetimeWarning",
    "SoftFileLockProtocolError",
    "SoftReadWriteLock",
    "StrictSoftFileClaim",
    "StrictSoftFileClaimState",
    "StrictSoftFileLock",
    "Timeout",
    "UnixFileLock",
    "WindowsFileLock",
    "__version__",
    "has_fcntl",
    "lock_descriptor",
    "unlock_descriptor",
]
<|endoftext|>from __future__ import annotations

import os
import sys
import warnings
from contextlib import suppress
from errno import EACCES, EAGAIN, ENOSYS, EWOULDBLOCK
from pathlib import Path
from typing import Final, cast

from ._api import BaseFileLock
from ._util import ensure_directory_exists

has_fcntl = False
if sys.platform == "win32":  # pragma: win32 cover

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError

else:  # pragma: win32 no cover
    try:
        import fcntl

        _ = (fcntl.flock, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN)
    except (ImportError, AttributeError):
        _FCNTL_UNAVAILABLE: Final[str] = "fcntl is unavailable"

        def _lock_fd_nonblocking(_fd: int) -> bool:
            raise OSError(ENOSYS, _FCNTL_UNAVAILABLE)

        def _unlock_fd(_fd: int) -> None:
            raise OSError(ENOSYS, _FCNTL_UNAVAILABLE)

    else:
        has_fcntl = True
        # Contention errnos for a nonblocking flock. EAGAIN/EWOULDBLOCK are the usual "held elsewhere" codes; some
        # filesystems report EACCES instead, so treat it as contention too rather than a permanent error.
        _CONTENTION_ERRNOS: Final[frozenset[int]] = frozenset({EACCES, EAGAIN, EWOULDBLOCK})

        def _lock_fd_nonblocking(fd: int) -> bool:
            # One nonblocking exclusive flock attempt shared by UnixFileLock and lock_descriptor, so both contend on
            # the same lock and classify errors identically. The caller owns fd; this never closes it.
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exception:
                if exception.errno in _CONTENTION_ERRNOS:
                    return False
                raise
            return True

        def _unlock_fd(fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_UN)

    class UnixFileLock(BaseFileLock):
        """
        Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.

        We leave the lock file in place after release. Unlinking a locked file on Unix splits
        waiters across inodes and breaks mutual exclusion for processes that coordinate via the
        same path.
        """

        def _acquire(self) -> None:
            if (missing_flock := self._acquire_native()) is not None:
                fd, exception, created = missing_flock
                self._switch_to_soft_lock(fd, exception, created=created)

        def _acquire_native(self) -> tuple[int, OSError, bool] | None:
            ensure_directory_exists(self.lock_file)
            if (opened := self._open_lock_file()) is None:
                return None
            fd, created = opened
            self._mark_descriptor_pending(fd)
            try:
                locked = _lock_fd_nonblocking(fd)
            except OSError as exception:
                if exception.errno != ENOSYS:
                    self._mark_descriptor_released()
                    os.close(fd)
                    raise  # contention returns False from _lock_fd_nonblocking, so any raise here is a real failure
                return fd, exception, created
            if locked:
                self._finalize_locked_fd(fd)
            else:
                self._mark_descriptor_released()
                os.close(fd)  # contention; let the retry loop try again
            return None

        def _open_lock_file(self) -> tuple[int, bool] | None:
            # No O_TRUNC: a contender that loses the flock must not truncate the holder's file or change its mode; the
            # winner does both once it owns the lock (#591). O_EXCL tells the ENOSYS fallback whether this attempt made
            # the inode, so it removes only its own placeholder, never a peer's marker.
            open_flags = os.O_RDWR
            if (o_nofollow := getattr(os, "O_NOFOLLOW", None)) is not None:
                open_flags |= o_nofollow
            open_mode = self._open_mode()
            try:
                return os.open(self.lock_file, open_flags | os.O_CREAT | os.O_EXCL, open_mode), True
            except FileExistsError:
                pass
            except FileNotFoundError:
                # On FUSE/NFS a split LOOKUP + CREATE lets a concurrent unlink() race the create, so ENOENT is transient
                # contention. An empty or separator-terminated path (Path() strips the separator) fails this way on
                # every attempt on macOS, so re-raise rather than retry forever.
                if self.lock_file and not self.lock_file.endswith(os.sep) and Path(self.lock_file).parent.exists():
                    return None
                raise
            except PermissionError:
                # Sticky-bit dirs (e.g. /tmp): O_CREAT fails if another user owns the file (#317).
                if not Path(self.lock_file).exists():
                    raise
            try:
                return os.open(self.lock_file, open_flags, open_mode), False
            except FileNotFoundError:
                return None  # unlinked since the create attempt; let the retry loop start over

        def _switch_to_soft_lock(self, fd: int, missing_flock: OSError, *, created: bool) -> None:
            # The filesystem does not implement flock. Capture the opened file's identity before closing so the cleanup
            # below removes only this attempt's placeholder, not a peer's replacement.
            identity: tuple[int, int] | None = None
            with suppress(OSError):
                identity = (fstat := os.fstat(fd)).st_dev, fstat.st_ino
            self._mark_descriptor_released()
            os.close(fd)
            if not self._fallback_to_soft or self._preserve_lock_file or self._on_acquired is not None:
                # Fail closed: the caller opted out of existence-lock semantics (#603), asked to preserve the pathname
                # (#605), or set an on_acquired hook (#607), none of which a soft lock can honor.
                raise missing_flock
            # A file this attempt did not create may be a peer's live marker; the soft lock ages out a stale empty one.
            if created:
                with suppress(OSError):
                    current = os.lstat(self.lock_file)
                    if identity == (current.st_dev, current.st_ino):
                        Path(self.lock_file).unlink()
            self._fallback_to_soft_lock()
            self._acquire()

        def _finalize_locked_fd(self, fd: int) -> None:
            # Runs with the flock held. Truncate and normalize mode under a guard so any failure closes fd rather than
            # leaking it and its lock. A concurrent _release() may have unlinked the inode between our open() and
            # flock() (st_nlink 0), leaving a useless dead-inode lock; drop it and let the retry loop start fresh.
            keep = False
            try:
                stat_result = os.fstat(fd)
                if stat_result.st_nlink != 0:
                    os.ftruncate(fd, 0)
                    self._apply_explicit_mode(fd)
                    keep = True
            except OSError:
                self._mark_descriptor_released()
                os.close(fd)
                raise
            if keep:
                self._mark_descriptor_owned(fd, (stat_result.st_dev, stat_result.st_ino))
            else:
                self._mark_descriptor_released()
                os.close(fd)

        def _apply_explicit_mode(self, fd: int) -> None:
            if self.has_explicit_mode:
                with suppress(PermissionError):
                    os.fchmod(fd, self._context.mode)

        def _fallback_to_soft_lock(self) -> None:
            # Import lazily: this runs only on the rare flock fallback, and asyncio imports _unix, so a
            # module-level import of it here would cycle.
            from ._soft import SoftFileLock  # ruff:ignore[import-outside-top-level]

            warnings.warn("flock not supported on this filesystem, falling back to SoftFileLock", stacklevel=2)
            from .asyncio import AsyncSoftFileLock, BaseAsyncFileLock  # ruff:ignore[import-outside-top-level]

            self.__class__ = AsyncSoftFileLock if isinstance(self, BaseAsyncFileLock) else SoftFileLock

        def _release(self) -> None:
            fd = cast("int", self._context.lock_file_fd)
            # Retain the descriptor until flock succeeds: a failed unlock leaves the kernel lock held, so is_locked
            # must keep reporting held for a retry. Once flock commits, clear held state and close as post-unlock
            # cleanup; a close failure (EIO on FUSE/Docker bind mounts) does not make the kernel lock held again.
            _unlock_fd(fd)
            self._mark_descriptor_released()
            self._close_released_fd(fd, default_suppresses=True)


if sys.platform == "win32":  # pragma: win32 cover
    __all__ = ["UnixFileLock", "has_fcntl"]
else:  # pragma: win32 no cover
    __all__ = ["UnixFileLock", "_lock_fd_nonblocking", "_unlock_fd", "has_fcntl"]
<|endoftext|>from __future__ import annotations

import hashlib
import os
import socket
import sys
from errno import EPERM, ESRCH
from pathlib import Path
from typing import Final

#: Every marker format in this package caps the hostname at the RFC 1123 limit and reads a longer one as malformed.
_HOST_NAME_LIMIT: Final[int] = 253
#: The bytes those formats carry verbatim: printable non-space ASCII, less the ``?`` reserved for the escape.
_HOST_NAME_VERBATIM: Final[frozenset[int]] = frozenset(range(0x21, 0x7F)) - {ord("?")}
