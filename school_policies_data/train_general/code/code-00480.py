        logger.info(f"Downloading took {duration.total_seconds() // 60} min")
        url_or_urls = NestedDataStructure(url_or_urls)
        downloaded_path_or_paths = NestedDataStructure(downloaded_path_or_paths)
        self.downloaded_paths.update(dict(zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten())))

        start_time = datetime.now()
        self._record_sizes_checksums(url_or_urls, downloaded_path_or_paths)
        duration = datetime.now() - start_time
        logger.info(f"Checksum Computation took {duration.total_seconds() // 60} min")

        return downloaded_path_or_paths.data

    def _download_batched(
        self,
        url_or_filenames: list[str],
        download_config: DownloadConfig,
    ) -> list[str]:
        if len(url_or_filenames) >= 16:
            download_config = download_config.copy()
            download_config.disable_tqdm = True
            download_func = partial(self._download_single, download_config=download_config)

            fs: fsspec.AbstractFileSystem
            path = str(url_or_filenames[0])
            if is_relative_path(path):
                # append the relative path to the base_path
                path = url_or_path_join(self._base_path, path)
            fs, path = url_to_fs(path, **download_config.storage_options)
            size = 0
            try:
                size = fs.info(path).get("size", 0)
            except Exception:
                pass
            max_workers = (
                config.HF_DATASETS_MULTITHREADING_MAX_WORKERS if size < (20 << 20) else 1
            )  # enable multithreading if files are small

            return thread_map(
                download_func,
                url_or_filenames,
                desc=download_config.download_desc or "Downloading",
                unit="files",
                position=multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
                if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
                and multiprocessing.current_process()._identity
                else None,
                max_workers=max_workers,
                tqdm_class=tqdm,
            )
        else:
            return [
                self._download_single(url_or_filename, download_config=download_config)
                for url_or_filename in url_or_filenames
            ]

    def _download_single(self, url_or_filename: str, download_config: DownloadConfig) -> str:
        url_or_filename = str(url_or_filename)
        if is_relative_path(url_or_filename):
            # append the relative path to the base_path
            url_or_filename = url_or_path_join(self._base_path, url_or_filename)
        out = cached_path(url_or_filename, download_config=download_config)
        out = tracked_str(out)
        out.set_origin(url_or_filename)
        return out

    def iter_archive(self, path_or_buf: Union[str, io.BufferedReader]):
        """Iterate over files within an archive.

        Args:
            path_or_buf (`str` or `io.BufferedReader`):
                Archive path or archive binary file object.

        Yields:
            `tuple[str, io.BufferedReader]`:
                2-tuple (path_within_archive, file_object).
                File object is opened in binary mode.

        Example:

        ```py
        >>> archive = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        >>> files = dl_manager.iter_archive(archive)
        ```
        """

        if hasattr(path_or_buf, "read"):
            return ArchiveIterable.from_buf(path_or_buf)
        else:
            return ArchiveIterable.from_urlpath(path_or_buf)

    def iter_files(self, paths: Union[str, list[str]]):
        """Iterate over file paths.

        Args:
            paths (`str` or `list` of `str`):
                Root paths.

        Yields:
            `str`: File path.

        Example:

        ```py
        >>> files = dl_manager.download_and_extract('https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip')
        >>> files = dl_manager.iter_files(files)
        ```
        """
        return FilesIterable.from_urlpaths(paths)

    def extract(self, path_or_paths):
        """Extract given path(s).

        Args:
            path_or_paths (path or `list` or `dict`):
                Path of file to extract. Each path is a `str`.

        Returns:
            extracted_path(s): `str`, The extracted paths matching the given input
            path_or_paths.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        >>> extracted_files = dl_manager.extract(downloaded_files)
        ```
        """
        download_config = self.download_config.copy()
        download_config.extract_compressed_file = True
        extract_func = partial(self._download_single, download_config=download_config)
        extracted_paths = map_nested(
            extract_func,
            path_or_paths,
            num_proc=download_config.num_proc,
            desc="Extracting data files",
        )
        path_or_paths = NestedDataStructure(path_or_paths)
        extracted_paths = NestedDataStructure(extracted_paths)
        self.extracted_paths.update(dict(zip(path_or_paths.flatten(), extracted_paths.flatten())))
        return extracted_paths.data

    def download_and_extract(self, url_or_urls):
        """Download and extract given `url_or_urls`.

        Is roughly equivalent to:

        ```
        extracted_paths = dl_manager.extract(dl_manager.download(url_or_urls))
        ```

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL or `list` or `dict` of URLs to download and extract. Each URL is a `str`.

        Returns:
            extracted_path(s): `str`, extracted paths of given URL(s).
        """
        return self.extract(self.download(url_or_urls))

    def get_recorded_sizes_checksums(self):
        return self._recorded_sizes_checksums.copy()

    def delete_extracted_files(self):
        paths_to_delete = set(self.extracted_paths.values()) - set(self.downloaded_paths.values())
        for key, path in list(self.extracted_paths.items()):
            if path in paths_to_delete and os.path.isfile(path):
                os.remove(path)
                del self.extracted_paths[key]

    def manage_extracted_files(self):
        if self.download_config.delete_extracted:
            self.delete_extracted_files()
<|endoftext|>import enum
import inspect
import warnings
from functools import wraps
from typing import Callable, Optional

from .logging import get_logger


_emitted_deprecation_warnings = set()
logger = get_logger(__name__)


def deprecated(help_message: Optional[str] = None):
    """Decorator to mark a class or a function as deprecated.

    Args:
        help_message (:obj:`str`, optional): An optional message to guide the user on how to
            switch to non-deprecated usage of the library.
    """

    def decorator(deprecated_class_or_function: Callable):
        global _emitted_deprecation_warnings

        if inspect.isclass(deprecated_class_or_function):
            deprecated_function = deprecated_class_or_function.__init__
            name = deprecated_class_or_function.__name__
        else:
            deprecated_function = deprecated_class_or_function
            name = deprecated_function.__name__
            # Support deprecating __init__ class method: class name instead
            name = name if name != "__init__" else deprecated_function.__qualname__.split(".")[-2]

        warning_msg = (
            f"{name} is deprecated and will be removed in the next major version of datasets." + f" {help_message}"
            if help_message
            else ""
        )

        @wraps(deprecated_function)
        def wrapper(*args, **kwargs):
            func_hash = hash(deprecated_function)
            if func_hash not in _emitted_deprecation_warnings:
                warnings.warn(warning_msg, category=FutureWarning, stacklevel=2)
                _emitted_deprecation_warnings.add(func_hash)
            return deprecated_function(*args, **kwargs)

        wrapper._decorator_name_ = "deprecated"

        if inspect.isclass(deprecated_class_or_function):
            deprecated_class_or_function.__init__ = wrapper
            return deprecated_class_or_function
        else:
            return wrapper

    return decorator


class OnAccess(enum.EnumMeta):
    """
    Enum metaclass that calls a user-specified function whenever a member is accessed.
    """

    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, enum.Enum) and obj._on_access:
            obj._on_access()
        return obj

    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._on_access:
            member._on_access()
        return member

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(value, names, module=module, qualname=qualname, type=type, start=start)
        if isinstance(obj, enum.Enum) and obj._on_access:
            obj._on_access()
        return obj


class DeprecatedEnum(enum.Enum, metaclass=OnAccess):
    """
    Enum class that calls `deprecate` method whenever a member is accessed.
    """

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member._on_access = member.deprecate
        return member

    @property
    def help_message(self):
        return ""

    def deprecate(self):
        help_message = f" {self.help_message}" if self.help_message else ""
        warnings.warn(
            f"'{self.__objclass__.__name__}' is deprecated and will be removed in the next major version of datasets."
            + help_message,
            FutureWarning,
            stacklevel=3,
        )
<|endoftext|>from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from ..features.features import FeatureType


def ujson_dumps(*args, **kwargs):
    try:
        return pd.io.json.ujson_dumps(*args, **kwargs)
    except AttributeError:
        # Before pandas-2.2.0, ujson_dumps was renamed to dumps: import ujson_dumps as dumps
        return pd.io.json.dumps(*args, **kwargs)


def ujson_loads(*args, **kwargs):
    try:
        return pd.io.json.ujson_loads(*args, **kwargs)
    except AttributeError:
        # Before pandas-2.2.0, ujson_loads was renamed to loads: import ujson_loads as loads
        return pd.io.json.loads(*args, **kwargs)


def json_encode_field(example: Any, json_field_path: list[str]) -> Any:
    if json_field_path:
        field, *json_field_path = json_field_path
        if example is None:
            return None
        elif field == 0:
            return [json_encode_field(x, json_field_path) for x in example]
        else:
            return {**example, field: json_encode_field(example.get(field), json_field_path)}
    elif example is None:
        # keep missing values as real nulls instead of the JSON string "null"
        return None
    else:
        try:
            ujson_loads(example)
        except Exception:
            return ujson_dumps(example)
        else:
            return example


def json_decode_field(example: Any, json_field_path: str) -> Any:
    if json_field_path:
        field, *json_field_path = json_field_path
        if example is None:
            return None
        elif field == 0:
            return [json_decode_field(x, json_field_path) for x in example]
        else:
            return {**example, field: json_decode_field(example.get(field), json_field_path)}
    elif example is None:
        return None
    else:
        try:
            return ujson_loads(example)
        except Exception:
            return example


def find_mixed_struct_types_field_paths(examples: list, allow_root=False) -> list[list[str]]:
    mixed_struct_types_field_paths = []
    examples = [example for example in examples if example is not None]
    if not examples:
        return []
    paths_and_content_to_check = [([], examples)]
    while paths_and_content_to_check:
        path, content = paths_and_content_to_check.pop(0)
        if all(isinstance(x, dict) for x in content):
            if (allow_root or path) and (any(set(x) != set(content[0]) for x in content) or not content[0]):
                mixed_struct_types_field_paths.append(path)
            else:
                for subfield in {field for x in content for field in x}:
                    examples = [x[subfield] for x in content if subfield in x and x[subfield] is not None]
                    if not examples:
                        continue
                    paths_and_content_to_check.append((path + [subfield], examples))
        elif all(isinstance(x, list) for x in content):
            examples = [x for sublist in content for x in sublist if x is not None]
            if not examples:
                continue
            paths_and_content_to_check.append((path + [0], examples))
        elif any(isinstance(x, (dict, list)) for x in content):
            mixed_struct_types_field_paths.append(path)
    return mixed_struct_types_field_paths


def get_json_field_path_from_pyarrow_json_error(err_str: str) -> list[str]:
    # e.g. json_field_path_str = "col/subfield_containing_a_list/[]/subsubfield_in_item_in_the_list"
    json_field_path_str = err_str.split("Column(", 1)[1].rsplit(") changed from", 1)[0].strip("/")
    # e.g. json_field_path = ["col", "subfield_containing_a_list", 0, "subsubfield_in_item_in_the_list"]
    json_field_path = [0 if seg == "[]" else seg for seg in json_field_path_str.split("/")]
    return json_field_path


def insert_json_field_path(json_field_paths: list[list[str]], json_field_path: list[str]) -> None:
    # Add to list of json_field_paths and check if other share a common path
    for i in range(len(json_field_paths)):
        if json_field_paths[i][: len(json_field_path)] == json_field_path:
            json_field_paths[i] = json_field_path
            break
    else:
        json_field_paths.append(json_field_path)


def json_encode_fields_in_json_lines(original_batch: bytes, json_field_paths: list[list[str]]) -> bytes:
    examples = [ujson_loads(line) for line in original_batch.splitlines()]
    for json_field_path in json_field_paths:
        examples = [json_encode_field(example, json_field_path) for example in examples]
    batch = "\n".join([ujson_dumps(example) for example in examples]).encode()
    return batch


def get_json_field_paths_from_feature(feature: "FeatureType") -> list[list[str]]:
    from datasets.features.features import Json, _visit_with_path

    json_field_paths = []

    def get_json_type_path(_feature, feature_path):
        if isinstance(_feature, Json):
            json_field_paths.append(feature_path)
        return _feature

    _visit_with_path(feature, get_json_type_path)
    return json_field_paths


def set_json_types_in_feature(feature: "FeatureType", json_field_paths: list[list[str]]) -> None:
    from datasets.features.features import Json, _visit_with_path

    def set_json_type(feature, feature_path):
        return Json() if feature_path in json_field_paths else feature

    feature = _visit_with_path(feature, set_json_type)
    return feature
<|endoftext|>"""Utility helpers to handle progress bars in `datasets`.

Example:
    1. Use `datasets.utils.tqdm` as you would use `tqdm.tqdm` or `tqdm.auto.tqdm`.
    2. To disable progress bars, either use `disable_progress_bars()` helper or set the
       environment variable `HF_DATASETS_DISABLE_PROGRESS_BARS` to 1.
    3. To re-enable progress bars, use `enable_progress_bars()`.
    4. To check whether progress bars are disabled, use `are_progress_bars_disabled()`.

NOTE: Environment variable `HF_DATASETS_DISABLE_PROGRESS_BARS` has the priority.

Example:
    ```py
    from datasets.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
        tqdm,
    )

    # Disable progress bars globally
    disable_progress_bars()

    # Use as normal `tqdm`
    for _ in tqdm(range(5)):
       do_something()

    # Still not showing progress bars, as `disable=False` is overwritten to `True`.
    for _ in tqdm(range(5), disable=False):
       do_something()

    are_progress_bars_disabled() # True

    # Re-enable progress bars globally
    enable_progress_bars()

    # Progress bar will be shown !
    for _ in tqdm(range(5)):
       do_something()
    ```
"""

import os
import warnings

from tqdm.auto import tqdm as old_tqdm

from ..config import HF_DATASETS_DISABLE_PROGRESS_BARS


# `HF_DATASETS_DISABLE_PROGRESS_BARS` is `Optional[bool]` while `_hf_datasets_progress_bars_disabled`
# is a `bool`. If `HF_DATASETS_DISABLE_PROGRESS_BARS` is set to True or False, it has priority.
# If `HF_DATASETS_DISABLE_PROGRESS_BARS` is None, it means the user have not set the
# environment variable and is free to enable/disable progress bars programmatically.
# TL;DR: env variable has priority over code.
#
# By default, progress bars are enabled.
_hf_datasets_progress_bars_disabled: bool = HF_DATASETS_DISABLE_PROGRESS_BARS or False


def disable_progress_bars() -> None:
    """
    Disable globally progress bars used in `datasets` except if `HF_DATASETS_DISABLE_PROGRESS_BARS` environment
    variable has been set.

    Use [`~utils.enable_progress_bars`] to re-enable them.
    """
    if HF_DATASETS_DISABLE_PROGRESS_BARS is False:
        warnings.warn(
            "Cannot disable progress bars: environment variable `HF_DATASETS_DISABLE_PROGRESS_BARS=0` is set and has"
            " priority."
        )
        return
    global _hf_datasets_progress_bars_disabled
    _hf_datasets_progress_bars_disabled = True


def enable_progress_bars() -> None:
    """
    Enable globally progress bars used in `datasets` except if `HF_DATASETS_DISABLE_PROGRESS_BARS` environment
    variable has been set.

    Use [`~utils.disable_progress_bars`] to disable them.
    """
    if HF_DATASETS_DISABLE_PROGRESS_BARS is True:
        warnings.warn(
            "Cannot enable progress bars: environment variable `HF_DATASETS_DISABLE_PROGRESS_BARS=1` is set and has"
            " priority."
        )
        return
    global _hf_datasets_progress_bars_disabled
    _hf_datasets_progress_bars_disabled = False


def are_progress_bars_disabled() -> bool:
    """Return whether progress bars are globally disabled or not.

    Progress bars used in `datasets` can be enable or disabled globally using [`~utils.enable_progress_bars`]
    and [`~utils.disable_progress_bars`] or by setting `HF_DATASETS_DISABLE_PROGRESS_BARS` as environment variable.
    """
    global _hf_datasets_progress_bars_disabled
    return _hf_datasets_progress_bars_disabled


class tqdm(old_tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        if are_progress_bars_disabled():
            kwargs["disable"] = True
        elif kwargs.get("disable") is None and os.getenv("TQDM_POSITION") == "-1":
            # Force-enable progress bars in cloud environments when disable=None
            kwargs["disable"] = False
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/datasets/issues/6066"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


# backward compatibility
enable_progress_bar = enable_progress_bars
disable_progress_bar = disable_progress_bars


def is_progress_bar_enabled():
    return not are_progress_bars_disabled()
<|endoftext|># Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

"""TF-specific utils import."""

import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4

import numpy as np
import pyarrow as pa
from multiprocess import get_context


try:
    from multiprocess.shared_memory import SharedMemory
except ImportError:
    SharedMemory = None  # Version checks should prevent this being called on older Python versions

from .. import config


def minimal_tf_collate_fn(features):
    if isinstance(features, dict):  # case batch_size=None: nothing to collate
        return features
    elif config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

    first = features[0]
    batch = {}
    for k, v in first.items():
        if isinstance(v, np.ndarray):
            batch[k] = np.stack([f[k] for f in features])
        elif isinstance(v, tf.Tensor):
            batch[k] = tf.stack([f[k] for f in features])
        else:
            batch[k] = np.array([f[k] for f in features])
    return batch


def minimal_tf_collate_fn_with_renaming(features):
    batch = minimal_tf_collate_fn(features)
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    return batch


def is_numeric_pa_type(pa_type):
    if pa.types.is_list(pa_type):
        return is_numeric_pa_type(pa_type.value_type)
    return pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_decimal(pa_type)


def np_get_batch(
    indices, dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, return_dict=False
):
    if not isinstance(indices, np.ndarray):
        indices = indices.numpy()

    is_batched = True
    # Optimization - if we're loading a sequential batch, do it with slicing instead of a list of indices
    if isinstance(indices, np.integer):
        batch = dataset[indices.item()]
        is_batched = False
    elif np.all(np.diff(indices) == 1):
        batch = dataset[indices[0] : indices[-1] + 1]
    elif isinstance(indices, np.ndarray):
        batch = dataset[indices]
    else:
        raise RuntimeError(f"Unexpected type for indices: {type(indices)}")

    if cols_to_retain is not None:
        batch = {
            key: value
            for key, value in batch.items()
            if key in cols_to_retain or key in ("label", "label_ids", "labels")
        }

    if is_batched:
        actual_size = len(list(batch.values())[0])  # Get the length of one of the arrays, assume all same
        # Our collators expect a list of dicts, not a dict of lists/arrays, so we invert
        batch = [{key: value[i] for key, value in batch.items()} for i in range(actual_size)]
    batch = collate_fn(batch, **collate_fn_args)

    if return_dict:
        out_batch = {}
        for col, cast_dtype in columns_to_np_types.items():
            # In case the collate_fn returns something strange
            array = np.array(batch[col])
            array = array.astype(cast_dtype)
            out_batch[col] = array
    else:
        out_batch = []
        for col, cast_dtype in columns_to_np_types.items():
            # In case the collate_fn returns something strange
            array = np.array(batch[col])
            array = array.astype(cast_dtype)
            out_batch.append(array)
    return out_batch


def dataset_to_tf(
    dataset,
    cols_to_retain,
    collate_fn,
    collate_fn_args,
    columns_to_np_types,
    output_signature,
    shuffle,
    batch_size,
    drop_remainder,
):
    """Create a tf.data.Dataset from the underlying Dataset. This is a single-process method - the multiprocess
    equivalent is multiprocess_dataset_to_tf.

    Args:
        dataset (`Dataset`): Dataset to wrap with tf.data.Dataset.
        cols_to_retain (`List[str]`): Dataset column(s) to load in the
            tf.data.Dataset. It is acceptable to include column names that are created by the `collate_fn` and
            that do not exist in the original dataset.
        collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
            lists of samples into a batch.
        collate_fn_args (`Dict`): A  `dict` of keyword arguments to be passed to the
            `collate_fn`. Can be empty.
        columns_to_np_types (`Dict[str, np.dtype]`): A `dict` mapping column names to numpy dtypes.
        output_signature (`Dict[str, tf.TensorSpec]`): A `dict` mapping column names to
            `tf.TensorSpec` objects.
        shuffle(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
            validation/evaluation.
        batch_size (`int`, default `None`): Size of batches to load from the dataset. Defaults to `None`, which implies that
            the dataset won't be batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
        drop_remainder(`bool`, default `None`): Drop the last incomplete batch when loading. If not provided,
            defaults to the same setting as shuffle.

    Returns:
        `tf.data.Dataset`
    """
    if config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

    # TODO Matt: When our minimum Python version is 3.8 or higher, we can delete all of this and move everything
    #            to the NumPy multiprocessing path.
    if hasattr(tf, "random_index_shuffle"):
        random_index_shuffle = tf.random_index_shuffle
    elif hasattr(tf.random.experimental, "index_shuffle"):
        random_index_shuffle = tf.random.experimental.index_shuffle
    else:
        if len(dataset) > 10_000_000:
            warnings.warn(
                "to_tf_dataset() can be memory-inefficient on versions of TensorFlow older than 2.9. "
                "If you are iterating over a dataset with a very large number of samples, consider "
                "upgrading to TF >= 2.9."
            )
        random_index_shuffle = None

    getter_fn = partial(
        np_get_batch,
        dataset=dataset,
        cols_to_retain=cols_to_retain,
        collate_fn=collate_fn,
        collate_fn_args=collate_fn_args,
        columns_to_np_types=columns_to_np_types,
        return_dict=False,
    )

    # This works because dictionaries always output in the same order
    tout = [tf.dtypes.as_dtype(dtype) for dtype in columns_to_np_types.values()]

    @tf.function(input_signature=[tf.TensorSpec(None, tf.int64)])
    def fetch_function(indices):
        output = tf.py_function(
            getter_fn,
            inp=[indices],
            Tout=tout,
        )
        return {key: output[i] for i, key in enumerate(columns_to_np_types.keys())}

    tf_dataset = tf.data.Dataset.range(len(dataset))

    if shuffle and random_index_shuffle is not None:
        base_seed = tf.fill((3,), value=tf.cast(-1, dtype=tf.int64))

        def scan_random_index(state, index):
            if tf.reduce_all(state == -1):
                # This generates a new random seed once per epoch only,
                # to ensure that we iterate over each sample exactly once per epoch
                state = tf.random.uniform(shape=(3,), maxval=2**62, dtype=tf.int64)
            shuffled_index = random_index_shuffle(index=index, seed=state, max_index=len(dataset) - 1)
            return state, shuffled_index

        tf_dataset = tf_dataset.scan(base_seed, scan_random_index)
    elif shuffle:
        tf_dataset = tf_dataset.shuffle(tf_dataset.cardinality())

    if batch_size is not None:
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)

    tf_dataset = tf_dataset.map(fetch_function)

    if batch_size is not None:

        def ensure_shapes(input_dict):
            return {key: tf.ensure_shape(val, output_signature[key].shape) for key, val in input_dict.items()}

    else:
        # Ensure shape but remove batch dimension of output_signature[key].shape
        def ensure_shapes(input_dict):
            return {key: tf.ensure_shape(val, output_signature[key].shape[1:]) for key, val in input_dict.items()}

    return tf_dataset.map(ensure_shapes)


class SharedMemoryContext:
    # This is a context manager for creating shared memory that ensures cleanup happens even if a process is interrupted
    # The process that creates shared memory is always the one responsible for unlinking it in the end
    def __init__(self):
        self.created_shms = []
        self.opened_shms = []

    def get_shm(self, name, size, create):
        shm = SharedMemory(size=int(size), name=name, create=create)
        if create:
            # We only unlink the ones we created in this context
            self.created_shms.append(shm)
        else:
            # If we didn't create it, we only close it when done, we don't unlink it
            self.opened_shms.append(shm)
        return shm

    def get_array(self, name, shape, dtype, create):
        shm = self.get_shm(name=name, size=np.prod(shape) * np.dtype(dtype).itemsize, create=create)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for shm in self.created_shms:
            shm.close()
            shm.unlink()
        for shm in self.opened_shms:
            shm.close()


class NumpyMultiprocessingGenerator:
    def __init__(
        self,
        dataset,
        cols_to_retain,
        collate_fn,
        collate_fn_args,
        columns_to_np_types,
        output_signature,
        shuffle,
        batch_size,
        drop_remainder,
        num_workers,
    ):
        self.dataset = dataset
        self.cols_to_retain = cols_to_retain
        self.collate_fn = collate_fn
        self.collate_fn_args = collate_fn_args
        self.string_columns = [col for col, dtype in columns_to_np_types.items() if dtype is np.str_]
        # Strings will be converted to arrays of single unicode chars, so that we can have a constant itemsize
        self.columns_to_np_types = {
            col: dtype if col not in self.string_columns else np.dtype("U1")
            for col, dtype in columns_to_np_types.items()
        }
        self.output_signature = output_signature
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.num_workers = num_workers
        # Because strings are converted to characters, we need to add one extra dimension to the shape
        self.columns_to_ranks = {
            col: int(spec.shape.rank) if col not in self.string_columns else int(spec.shape.rank) + 1
            for col, spec in output_signature.items()
        }

    def __iter__(self):
        # Make sure we only spawn workers if they have work to do
        num_workers = min(self.num_workers, int(ceil(len(self.dataset) / self.batch_size)))
        # Do the shuffling in iter so that it's done at the start of each epoch
        per_worker_batches, final_batch, final_batch_worker = self.distribute_batches(
            self.dataset, self.batch_size, self.drop_remainder, num_workers, self.shuffle
        )
        ctx = get_context("spawn")
        names = []
        shape_arrays = []
        workers = []
        array_ready_events = [ctx.Event() for _ in range(num_workers)]
        array_loaded_events = [ctx.Event() for _ in range(num_workers)]

        base_args = {
            "dataset": self.dataset,
            "cols_to_retain": self.cols_to_retain,
            "collate_fn": self.collate_fn,
            "collate_fn_args": self.collate_fn_args,
            "columns_to_np_types": self.columns_to_np_types,
            "columns_to_ranks": self.columns_to_ranks,
            "string_columns": self.string_columns,
        }
        with SharedMemoryContext() as shm_ctx:
            for i in range(num_workers):
                worker_random_id = str(uuid4())
                worker_name = f"dw_{i}_{worker_random_id}"[:10]
                names.append(worker_name)

                worker_shape_arrays = {
                    col: shm_ctx.get_array(f"{worker_name}_{col}_shape", shape=(rank,), dtype=np.int64, create=True)
                    for col, rank in self.columns_to_ranks.items()
                }
                shape_arrays.append(worker_shape_arrays)

                worker_indices = per_worker_batches[i]
                if i == final_batch_worker and final_batch is not None:
                    final_batch_arg = final_batch
                else:
                    final_batch_arg = None
                worker_kwargs = {
                    "worker_name": worker_name,
                    "indices": worker_indices,
                    "extra_batch": final_batch_arg,
                    "array_ready_event": array_ready_events[i],
                    "array_loaded_event": array_loaded_events[i],
                    **base_args,
                }
                worker = ctx.Process(target=self.worker_loop, kwargs=worker_kwargs, daemon=True)
                worker.start()
                workers.append(worker)

            end_signal_received = False
            while not end_signal_received:
                for i in range(num_workers):
                    if not array_ready_events[i].wait(timeout=60):
                        raise TimeoutError("Data loading worker timed out!")
                    array_ready_events[i].clear()
                    array_shapes = shape_arrays[i]
                    if any(np.any(shape < 0) for shape in array_shapes.values()):
                        # Child processes send negative array shapes to indicate
                        # that no more data is going to be sent
                        end_signal_received = True
                        break
                    # Matt: Because array shapes are variable we recreate the shared memory each iteration.
                    #       I suspect repeatedly opening lots of shared memory is the bottleneck for the parent process.
                    #       A future optimization, at the cost of some code complexity, could be to reuse shared memory
                    #       between iterations, but this would require knowing in advance the maximum size, or having
                    #       a system to only create a new memory block when a new maximum size is seen.
                    #       Another potential optimization would be to figure out which memory copies are necessary,
                    #       or whether we can yield objects straight out of shared memory.
                    with SharedMemoryContext() as batch_shm_ctx:
                        # This memory context only lasts long enough to copy everything out of the batch
                        arrays = {
                            col: batch_shm_ctx.get_array(
                                f"{names[i]}_{col}",
                                shape=shape,
                                dtype=self.columns_to_np_types[col],
                                create=False,
                            )
                            for col, shape in array_shapes.items()
                        }
                        # Copy everything out of shm because the memory
                        # will be unlinked by the child process at some point
                        arrays = {col: np.copy(arr) for col, arr in arrays.items()}
                        # Now we convert any unicode char arrays to strings
                        for string_col in self.string_columns:
                            arrays[string_col] = (
                                arrays[string_col].view(f"U{arrays[string_col].shape[-1]}").squeeze(-1)
                            )
                    yield arrays
                    array_loaded_events[i].set()
            # Now we just do some cleanup
            # Shared memory is cleaned up by the context manager, so we just make sure workers finish
            for worker in workers:
                worker.join()

    def __call__(self):
        return self

    @staticmethod
    def worker_loop(
        dataset,
        cols_to_retain,
        collate_fn,
        collate_fn_args,
        columns_to_np_types,
        columns_to_ranks,
        string_columns,
        indices,
        extra_batch,
        worker_name,
        array_ready_event,
        array_loaded_event,
    ):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        tf.config.set_visible_devices([], "GPU")  # Make sure workers don't try to allocate GPU memory

        def send_batch_to_parent(indices):
            batch = np_get_batch(
                indices=indices,
                dataset=dataset,
                cols_to_retain=cols_to_retain,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                columns_to_np_types=columns_to_np_types,
                return_dict=True,
            )

            # Now begins the fun part where we start shovelling shared memory at the parent process
            out_arrays = {}
            with SharedMemoryContext() as batch_shm_ctx:
                # The batch shared memory context exists only as long as it takes for the parent process
                # to read everything, after which it cleans everything up again
                for col, cast_dtype in columns_to_np_types.items():
                    # Everything has to be np.array for this to work, even if the collate_fn is giving us tf.Tensor
                    array = batch[col]
                    if col in string_columns:
                        # We can't send unicode arrays over shared memory, so we convert to single chars ("U1")
                        # which have a fixed width of 4 bytes. The parent process will convert these back to strings.
                        array = array.view("U1").reshape(array.shape + (-1,))
                    shape_arrays[col][:] = array.shape
                    out_arrays[col] = batch_shm_ctx.get_array(
                        f"{worker_name}_{col}", shape=array.shape, dtype=cast_dtype, create=True
                    )
                    out_arrays[col][:] = array

                array_ready_event.set()
                array_loaded_event.wait()
                array_loaded_event.clear()

        with SharedMemoryContext() as shm_ctx:
            shape_arrays = {
                col: shm_ctx.get_array(f"{worker_name}_{col}_shape", shape=(rank,), dtype=np.int64, create=False)
                for col, rank in columns_to_ranks.items()
            }

            for batch in indices:
                send_batch_to_parent(batch)
            if extra_batch is not None:
                send_batch_to_parent(extra_batch)
            # Now we send a batsignal to the parent process that we're done
            for col, array in shape_arrays.items():
                array[:] = -1
            array_ready_event.set()

    @staticmethod
    def distribute_batches(dataset, batch_size, drop_remainder, num_workers, shuffle):
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        num_samples = len(indices)
        # We distribute the batches so that reading from the workers in round-robin order yields the exact
        # order specified in indices. This is only important when shuffle is False, but we do it regardless.
        incomplete_batch_cutoff = num_samples - (num_samples % batch_size)
        indices, last_incomplete_batch = np.split(indices, [incomplete_batch_cutoff])
        if drop_remainder or len(last_incomplete_batch) == 0:
            last_incomplete_batch = None

        indices = indices.reshape(-1, batch_size)
        num_batches = len(indices)
        final_batches_cutoff = num_batches - (num_batches % num_workers)
        indices, final_batches = np.split(indices, [final_batches_cutoff])
        indices = indices.reshape(-1, num_workers, batch_size)

        per_worker_indices = np.split(indices, indices.shape[1], axis=1)
        per_worker_indices = [np.squeeze(worker_indices, 1) for worker_indices in per_worker_indices]
        # Distribute the final batches to the first workers
        for i in range(len(final_batches)):
            # len(final_batches) can be zero, and is always less than num_workers
            per_worker_indices[i] = np.concatenate([per_worker_indices[i], final_batches[i].reshape(1, -1)], axis=0)
        # Add the last incomplete batch to the next worker, which might be the first worker
        if last_incomplete_batch is not None:
            incomplete_batch_worker_idx = len(final_batches)
        else:
            incomplete_batch_worker_idx = None
        return per_worker_indices, last_incomplete_batch, incomplete_batch_worker_idx


def multiprocess_dataset_to_tf(
    dataset,
    cols_to_retain,
    collate_fn,
    collate_fn_args,
    columns_to_np_types,
    output_signature,
    shuffle,
    batch_size,
    drop_remainder,
    num_workers,
):
    """Create a tf.data.Dataset from the underlying Dataset. This is a multi-process method - the single-process
    equivalent is dataset_to_tf.

    Args:
        dataset (`Dataset`): Dataset to wrap with tf.data.Dataset.
        cols_to_retain (`List[str]`): Dataset column(s) to load in the
            tf.data.Dataset. It is acceptable to include column names that are created by the `collate_fn` and
            that do not exist in the original dataset.
        collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
            lists of samples into a batch.
        collate_fn_args (`Dict`): A  `dict` of keyword arguments to be passed to the
            `collate_fn`. Can be empty.
        columns_to_np_types (`Dict[str, np.dtype]`): A `dict` mapping column names to numpy dtypes.
        output_signature (`Dict[str, tf.TensorSpec]`): A `dict` mapping column names to
            `tf.TensorSpec` objects.
        shuffle(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
            validation/evaluation.
        batch_size (`int`, default `None`): Size of batches to load from the dataset. Defaults to `None`, which implies that
            the dataset won't be batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
        drop_remainder(`bool`, default `None`): Drop the last incomplete batch when loading. If not provided,
            defaults to the same setting as shuffle.
        num_workers (`int`): Number of workers to use for loading the dataset. Should be >= 1.

    Returns:
        `tf.data.Dataset`
    """
    if config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

    data_generator = NumpyMultiprocessingGenerator(
        dataset=dataset,
        cols_to_retain=cols_to_retain,
        collate_fn=collate_fn,
        collate_fn_args=collate_fn_args,
        columns_to_np_types=columns_to_np_types,
        output_signature=output_signature,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        num_workers=num_workers,
    )

    tf_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
    if drop_remainder:
        dataset_length = int(len(dataset) // batch_size)
    else:
        dataset_length = int(ceil(len(dataset) / batch_size))
    return tf_dataset.apply(tf.data.experimental.assert_cardinality(dataset_length))
<|endoftext|>from importlib import import_module

from .logging import get_logger


logger = get_logger(__name__)


class _PatchedModuleObj:
    """Set all the modules components as attributes of the _PatchedModuleObj object."""

    def __init__(self, module, attrs=None):
        attrs = attrs or []
        if module is not None:
            for key in module.__dict__:
                if key in attrs or not key.startswith("__"):
                    setattr(self, key, getattr(module, key))
        self._original_module = module._original_module if isinstance(module, _PatchedModuleObj) else module


class patch_submodule:
    """
    Patch a submodule attribute of an object, by keeping all other submodules intact at all levels.

    Example::

        >>> import importlib
        >>> from datasets.load import dataset_module_factory
        >>> from datasets.streaming import patch_submodule, xjoin
        >>>
        >>> dataset_module = dataset_module_factory("stanfordnlp/snli")
        >>> snli_module = importlib.import_module(dataset_module.module_path)
        >>> patcher = patch_submodule(snli_module, "os.path.join", xjoin)
        >>> patcher.start()
        >>> assert snli_module.os.path.join is xjoin
    """

    _active_patches = []

    def __init__(self, obj, target: str, new, attrs=None):
        self.obj = obj
        self.target = target
        self.new = new
        self.key = target.split(".")[0]
        self.original = {}
        self.attrs = attrs or []

    def __enter__(self):
        *submodules, target_attr = self.target.split(".")

        # Patch modules:
        # it's used to patch attributes of submodules like "os.path.join";
        # in this case we need to patch "os" and "os.path"

        for i in range(len(submodules)):
            try:
                submodule = import_module(".".join(submodules[: i + 1]))
            except ModuleNotFoundError:
                continue
            # We iterate over all the globals in self.obj in case we find "os" or "os.path"
            for attr in self.obj.__dir__():
                obj_attr = getattr(self.obj, attr)
                # We don't check for the name of the global, but rather if its value *is* "os" or "os.path".
                # This allows to patch renamed modules like "from os import path as ospath".
                if obj_attr is submodule or (
                    isinstance(obj_attr, _PatchedModuleObj) and obj_attr._original_module is submodule
                ):
                    self.original[attr] = obj_attr
                    # patch at top level
                    setattr(self.obj, attr, _PatchedModuleObj(obj_attr, attrs=self.attrs))
                    patched = getattr(self.obj, attr)
                    # construct lower levels patches
                    for key in submodules[i + 1 :]:
                        setattr(patched, key, _PatchedModuleObj(getattr(patched, key, None), attrs=self.attrs))
                        patched = getattr(patched, key)
                    # finally set the target attribute
                    setattr(patched, target_attr, self.new)

        # Patch attribute itself:
        # it's used for builtins like "open",
        # and also to patch "os.path.join" we may also need to patch "join"
        # itself if it was imported as "from os.path import join".

        if submodules:  # if it's an attribute of a submodule like "os.path.join"
            try:
                attr_value = getattr(import_module(".".join(submodules)), target_attr)
            except (AttributeError, ModuleNotFoundError):
                return
            # We iterate over all the globals in self.obj in case we find "os.path.join"
            for attr in self.obj.__dir__():
                # We don't check for the name of the global, but rather if its value *is* "os.path.join".
                # This allows to patch renamed attributes like "from os.path import join as pjoin".
                if getattr(self.obj, attr) is attr_value:
                    self.original[attr] = getattr(self.obj, attr)
                    setattr(self.obj, attr, self.new)
        elif target_attr in globals()["__builtins__"]:  # if it'a s builtin like "open"
            self.original[target_attr] = globals()["__builtins__"][target_attr]
            setattr(self.obj, target_attr, self.new)
        else:
            raise RuntimeError(f"Tried to patch attribute {target_attr} instead of a submodule.")

    def __exit__(self, *exc_info):
        for attr in list(self.original):
            setattr(self.obj, attr, self.original.pop(attr))

    def start(self):
        """Activate a patch."""
        self.__enter__()
        self._active_patches.append(self)

    def stop(self):
        """Stop an active patch."""
        try:
            self._active_patches.remove(self)
        except ValueError:
            # If the patch hasn't been started this will fail
            return None

        return self.__exit__()
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
"""Extends `dill` to support pickling more types and produce more consistent dumps."""

import os
import sys
from io import BytesIO
from types import CodeType, FunctionType

import dill
import pyarrow as pa
from packaging import version

from .. import config


class Pickler(dill.Pickler):
    dispatch = dill._dill.MetaCatchingDict(dill.Pickler.dispatch.copy())
    _legacy_no_dict_keys_sorting = False

    def save(self, obj, save_persistent_id=True):
        obj_type = type(obj)
        if obj_type not in self.dispatch:
            if "regex" in sys.modules:
                import regex  # type: ignore

                if obj_type is regex.Pattern:
                    pklregister(obj_type)(_save_regexPattern)
            if "spacy" in sys.modules:
                import spacy  # type: ignore

                if issubclass(obj_type, spacy.Language):
                    pklregister(obj_type)(_save_spacyLanguage)
            if "tiktoken" in sys.modules:
                import tiktoken  # type: ignore

                if obj_type is tiktoken.Encoding:
                    pklregister(obj_type)(_save_tiktokenEncoding)
            if "torch" in sys.modules:
                import torch  # type: ignore

                if issubclass(obj_type, torch.Tensor):
                    pklregister(obj_type)(_save_torchTensor)

                if obj_type is torch.Generator:
                    pklregister(obj_type)(_save_torchGenerator)

                # Unwrap `torch.compile`-ed modules
                if issubclass(obj_type, torch.nn.Module):
                    obj = getattr(obj, "_orig_mod", obj)
            if "transformers" in sys.modules:
                import transformers  # type: ignore

                if issubclass(obj_type, transformers.PreTrainedTokenizerBase):
                    pklregister(obj_type)(_save_transformersPreTrainedTokenizerBase)

        # Unwrap `torch.compile`-ed functions
        if obj_type is FunctionType:
            obj = getattr(obj, "_torchdynamo_orig_callable", obj)
        dill.Pickler.save(self, obj, save_persistent_id=save_persistent_id)

    def _batch_setitems(self, items, *args, **kwargs):
        # Ignore the order of keys in a dict
        try:
            # Faster, but fails for unorderable elements
            items = sorted(items)
        except Exception:  # TypeError, decimal.InvalidOperation, etc.
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        return super()._batch_setitems(items, *args, **kwargs)

    def memoize(self, obj):
        # Don't memoize strings since two identical strings can have different Python ids
        if type(obj) is not str:  # noqa: E721
            dill.Pickler.memoize(self, obj)


def pklregister(t):
    """Register a custom reducer for the type."""

    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy


def _is_supported_dill_version():
    """Check if the current dill version is in the supported range."""
    return config.DILL_VERSION.release[:3] in [
        version.parse("0.3.6").release,
        version.parse("0.3.7").release,
        version.parse("0.3.8").release,
        version.parse("0.3.9").release,
        version.parse("0.4.0").release,
        version.parse("0.4.1").release,
    ]


def dump(obj, file):
    """Pickle an object to a file."""
    Pickler(file, recurse=True).dump(obj)


def dumps(obj):
    """Pickle an object to a string."""
    file = BytesIO()
    dump(obj, file)
    return file.getvalue()


if config.DILL_VERSION < version.parse("0.3.6"):

    def log(pickler, msg):
        dill._dill.log.info(msg)

elif _is_supported_dill_version():

    def log(pickler, msg):
        dill._dill.logger.trace(pickler, msg)


@pklregister(set)
def _save_set(pickler, obj):
    log(pickler, f"Se: {obj}")
    try:
        # Faster, but fails for unorderable elements
        args = (sorted(obj),)
    except Exception:  # TypeError, decimal.InvalidOperation, etc.
        from datasets.fingerprint import Hasher

        args = (sorted(obj, key=Hasher.hash),)

    pickler.save_reduce(set, args, obj=obj)
    log(pickler, "# Se")


@pklregister(pa.Table)
def _save_arrowTable(pickler, obj):
    # pyarrow's default pickle serializes each chunk's buffers separately, so the
    # pickled size (and therefore the fingerprint cost) scales with the number of
    # chunks rather than the amount of data. Serialize a chunk-count-independent
    # form instead: combine each column's chunks one at a time (bounded memory,
    # never a full-table copy) so identical data produces identical bytes
    # regardless of chunking. See
    # https://github.com/huggingface/datasets/issues/8327.
    def create_arrowTable(schema, columns):
        return pa.Table.from_arrays(columns, schema=schema)

    log(pickler, f"Ta: {obj}")
    args = (obj.schema, [column.combine_chunks() for column in obj.columns])
    pickler.save_reduce(create_arrowTable, args, obj=obj)
    log(pickler, "# Ta")


@pklregister(pa.ChunkedArray)
def _save_arrowChunkedArray(pickler, obj):
    # Same rationale as _save_arrowTable: hash a chunk-count-independent form by
    # combining the chunks into a single array.
    def create_arrowChunkedArray(array):
        return pa.chunked_array([array])

    log(pickler, f"Ca: {obj}")
    args = (obj.combine_chunks(),)
    pickler.save_reduce(create_arrowChunkedArray, args, obj=obj)
    log(pickler, "# Ca")


def _save_regexPattern(pickler, obj):
    import regex  # type: ignore

    log(pickler, f"Re: {obj}")
    args = (obj.pattern, obj.flags)
    pickler.save_reduce(regex.compile, args, obj=obj)
    log(pickler, "# Re")


def _save_tiktokenEncoding(pickler, obj):
    import tiktoken  # type: ignore

    log(pickler, f"Enc: {obj}")
    args = (obj.name, obj._pat_str, obj._mergeable_ranks, obj._special_tokens)
    pickler.save_reduce(tiktoken.Encoding, args, obj=obj)
    log(pickler, "# Enc")


def _save_torchTensor(pickler, obj):
    import torch  # type: ignore

    # `torch.from_numpy` is not picklable in `torch>=1.11.0`
    def create_torchTensor(np_array, dtype=None):
        tensor = torch.from_numpy(np_array)
        if dtype:
            tensor = tensor.type(dtype)
        return tensor

    log(pickler, f"To: {obj}")
    if obj.dtype == torch.bfloat16:
        args = (obj.detach().to(torch.float).cpu().numpy(), torch.bfloat16)
    else:
        args = (obj.detach().cpu().numpy(),)
    pickler.save_reduce(create_torchTensor, args, obj=obj)
    log(pickler, "# To")


def _save_torchGenerator(pickler, obj):
    import torch  # type: ignore

    def create_torchGenerator(state):
        generator = torch.Generator()
        generator.set_state(state)
        return generator

    log(pickler, f"Ge: {obj}")
    args = (obj.get_state(),)
    pickler.save_reduce(create_torchGenerator, args, obj=obj)
    log(pickler, "# Ge")


def _save_spacyLanguage(pickler, obj):
    import spacy  # type: ignore

    def create_spacyLanguage(config, bytes):
        lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
        lang_inst = lang_cls.from_config(config)
        return lang_inst.from_bytes(bytes)

    log(pickler, f"Sp: {obj}")
    args = (obj.config, obj.to_bytes())
    pickler.save_reduce(create_spacyLanguage, args, obj=obj)
    log(pickler, "# Sp")


def _save_transformersPreTrainedTokenizerBase(pickler, obj):
    log(pickler, f"Tok: {obj}")
    # Ignore the `cache` attribute and make hashing stable.
    #
    # Some tokenizers backed by the `tokenizers` library mutate their internal `_tokenizer` state when called
    # (e.g. by enabling truncation/padding). This can change the serialized bytes across runs and make dataset
    # fingerprints unstable, which prevents `.map(load_from_cache_file=True)` from reusing cache files.
    #
    # For hashing/fingerprinting, we temporarily disable backend truncation/padding to avoid these runtime settings
    # affecting the fingerprint, then restore the original settings.
    state = obj.__dict__.copy()
    if "cache" in state and isinstance(state["cache"], dict):
        state["cache"] = {}
    if "deprecation_warnings" in state and isinstance(state["deprecation_warnings"], dict):
        state["deprecation_warnings"] = {}

    backend_tokenizer = obj.__dict__.get("_tokenizer")
    truncation = padding = None
    if (
        backend_tokenizer is not None
        and hasattr(backend_tokenizer, "truncation")
        and hasattr(backend_tokenizer, "padding")
    ):
        truncation = backend_tokenizer.truncation
        padding = backend_tokenizer.padding
        try:
            if truncation is not None and hasattr(backend_tokenizer, "no_truncation"):
                backend_tokenizer.no_truncation()
            if padding is not None and hasattr(backend_tokenizer, "no_padding"):
                backend_tokenizer.no_padding()
        except Exception:
            truncation = padding = None

    try:
        pickler.save_reduce(type(obj), (), state=state, obj=obj)
    finally:
        try:
            if backend_tokenizer is not None:
                if truncation is not None and hasattr(backend_tokenizer, "enable_truncation"):
                    backend_tokenizer.enable_truncation(**truncation)
                if padding is not None and hasattr(backend_tokenizer, "enable_padding"):
                    backend_tokenizer.enable_padding(**padding)
        except Exception:
            pass
    log(pickler, "# Tok")


if config.DILL_VERSION < version.parse("0.3.6"):

    @pklregister(CodeType)
    def _save_code(pickler, obj):
        """
        From dill._dill.save_code
        This is a modified version that removes the origin (filename + line no.)
        of functions created in notebooks or shells for example.
        """
        dill._dill.log.info(f"Co: {obj}")
        # The filename of a function is the .py file where it is defined.
        # Filenames of functions created in notebooks or shells start with '<'
        # ex: <ipython-input-13-9ed2afe61d25> for ipython, and <stdin> for shell
        # Filenames of functions created in ipykernel the filename
        # look like f"{tempdir}/ipykernel_{id1}/{id2}.py"
        # Moreover lambda functions have a special name: '<lambda>'
        # ex: (lambda x: x).__code__.co_name == "<lambda>"  # True
        #
        # For the hashing mechanism we ignore where the function has been defined
        # More specifically:
        # - we ignore the filename of special functions (filename starts with '<')
        # - we always ignore the line number
        # - we only use the base name of the file instead of the whole path,
        # to be robust in case a script is moved for example.
        #
        # Only those two lines are different from the original implementation:
        co_filename = (
            ""
            if obj.co_filename.startswith("<")
            or (
                len(obj.co_filename.split(os.path.sep)) > 1
                and obj.co_filename.split(os.path.sep)[-2].startswith("ipykernel_")
            )
            or obj.co_name == "<lambda>"
            else os.path.basename(obj.co_filename)
        )
        co_firstlineno = 1
        # The rest is the same as in the original dill implementation (with also a version check for 3.10)
        if dill._dill.PY3:
            if hasattr(obj, "co_posonlyargcount"):  # python 3.8 (16 args)
                args = (
                    obj.co_argcount,
                    obj.co_posonlyargcount,
                    obj.co_kwonlyargcount,
                    obj.co_nlocals,
                    obj.co_stacksize,
                    obj.co_flags,
                    obj.co_code,
                    obj.co_consts,
                    obj.co_names,
                    obj.co_varnames,
                    co_filename,
                    obj.co_name,
                    co_firstlineno,
                    obj.co_linetable if sys.version_info >= (3, 10) else obj.co_lnotab,
                    obj.co_freevars,
                    obj.co_cellvars,
                )
            else:  # python 3.7 (15 args)
                args = (
                    obj.co_argcount,
                    obj.co_kwonlyargcount,
                    obj.co_nlocals,
                    obj.co_stacksize,
                    obj.co_flags,
                    obj.co_code,
                    obj.co_consts,
                    obj.co_names,
                    obj.co_varnames,
                    co_filename,
                    obj.co_name,
                    co_firstlineno,
                    obj.co_lnotab,
                    obj.co_freevars,
                    obj.co_cellvars,
                )
        else:
            args = (
                obj.co_argcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,
                obj.co_name,
                co_firstlineno,
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )
        pickler.save_reduce(CodeType, args, obj=obj)
        dill._dill.log.info("# Co")
        return

elif _is_supported_dill_version():
    # From: https://github.com/uqfoundation/dill/blob/dill-0.3.6/dill/_dill.py#L1104
    @pklregister(CodeType)
    def save_code(pickler, obj):
        dill._dill.logger.trace(pickler, "Co: %s", obj)

        ############################################################################################################
        # Modification here for huggingface/datasets
        # The filename of a function is the .py file where it is defined.
        # Filenames of functions created in notebooks or shells start with '<'
        # ex: <ipython-input-13-9ed2afe61d25> for ipython, and <stdin> for shell
        # Filenames of functions created in ipykernel the filename
        # look like f"{tempdir}/ipykernel_{id1}/{id2}.py"
        # Moreover lambda functions have a special name: '<lambda>'
        # ex: (lambda x: x).__code__.co_name == "<lambda>"  # True
        #
        # For the hashing mechanism we ignore where the function has been defined
        # More specifically:
        # - we ignore the filename of special functions (filename starts with '<')
        # - we always ignore the line number
        # - we only use the base name of the file instead of the whole path,
        # to be robust in case a script is moved for example.
        #
        # Only those two lines are different from the original implementation:
        co_filename = (
            ""
            if obj.co_filename.startswith("<")
            or (
                len(obj.co_filename.split(os.path.sep)) > 1
                and obj.co_filename.split(os.path.sep)[-2].startswith("ipykernel_")
            )
            or obj.co_name == "<lambda>"
            else os.path.basename(obj.co_filename)
        )
        co_firstlineno = 1
        # The rest is the same as in the original dill implementation, except for the replacements:
        # - obj.co_filename => co_filename
        # - obj.co_firstlineno => co_firstlineno
        # - obj.co_lnotab => obj.co_linetable for >= 3.10 since co_lnotab was deprecated
        ############################################################################################################

        if hasattr(obj, "co_endlinetable"):  # python 3.11a (20 args)
            args = (
                obj.co_linetable,  # Modification for huggingface/datasets ############################################
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                obj.co_qualname,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_endlinetable,
                obj.co_columntable,
                obj.co_exceptiontable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_exceptiontable"):  # python 3.11 (18 args)
            args = (
                obj.co_linetable,  # Modification for huggingface/datasets #######################################
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                obj.co_qualname,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_exceptiontable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_linetable"):  # python 3.10 (16 args)
            args = (
                obj.co_linetable,  # Modification for huggingface/datasets #######################################
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_posonlyargcount"):  # python 3.8 (16 args)
            args = (
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )
        else:  # python 3.7 (15 args)
            args = (
                obj.co_argcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )

        pickler.save_reduce(dill._dill._create_code, args, obj=obj)
        dill._dill.logger.trace(pickler, "# Co")
        return
<|endoftext|>from collections.abc import Iterable, Iterator


class tracked_str(str):
    origins = {}

    def set_origin(self, origin: str):
        if super().__repr__() not in self.origins:
            self.origins[super().__repr__()] = origin

    def get_origin(self):
        return self.origins.get(super().__repr__(), str(self))

    def __repr__(self) -> str:
        if super().__repr__() not in self.origins or self.origins[super().__repr__()] == self:
            return super().__repr__()
        else:
            return f"{str(self)} (origin={self.origins[super().__repr__()]})"


class tracked_list(list):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_item = None

    def __iter__(self) -> Iterator:
        for x in super().__iter__():
            self.last_item = x
            yield x
        self.last_item = None

    def __repr__(self) -> str:
        if self.last_item is None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}(current={self.last_item})"


class TrackedIterableFromGenerator(Iterable):
    """Utility class to create an iterable from a generator function, in order to reset the generator when needed."""

    def __init__(self, generator, *args):
        super().__init__()
        self.generator = generator
        self.args = args
        self.last_item = None

    def __iter__(self):
        for x in self.generator(*self.args):
            self.last_item = x
            yield x
        self.last_item = None

    def __repr__(self) -> str:
        if self.last_item is None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}(current={self.last_item})"

    def __reduce__(self):
        return (self.__class__, (self.generator, *self.args))
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

# Lint as: python3
"""Some python utils function and classes."""

import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Optional, TypeVar, Union

import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm

from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
    Pickler,
    dump,
    dumps,
    pklregister,
)


try:  # pragma: no branch
    from typing import Final

    import typing_extensions as _typing_extensions
    from typing_extensions import Literal
except ImportError:
    _typing_extensions = Literal = Final = None


logger = logging.get_logger(__name__)


# NOTE: When used on an instance method, the cache is shared across all
# instances and IS NOT per-instance.
# See
# https://stackoverflow.com/questions/14946264/python-lru-cache-decorator-per-instance
# For @property methods, use @memoized_property below.
memoize = functools.lru_cache


def size_str(size_in_bytes):
    """Returns a human readable size string.

    If size_in_bytes is None, then returns "Unknown size".

    For example `size_str(1.5 * datasets.units.GiB) == "1.50 GiB"`.

    Args:
        size_in_bytes: `int` or `None`, the size, in bytes, that we want to
            format as a human-readable size string.
    """
    if not size_in_bytes:
        return "Unknown size"

    _NAME_LIST = [("PiB", 2**50), ("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]

    size_in_bytes = float(size_in_bytes)
    for name, size_bytes in _NAME_LIST:
        value = size_in_bytes / size_bytes
        if value >= 1.0:
            return f"{value:.2f} {name}"
    return f"{int(size_in_bytes)} bytes"


def convert_file_size_to_int(size: Union[int, str]) -> int:
    """
    Converts a size expressed as a string with digits an unit (like `"50MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("PIB"):
        return int(size[:-3]) * (2**50)
    if size.upper().endswith("TIB"):
        return int(size[:-3]) * (2**40)
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("PB"):
        int_size = int(size[:-2]) * (10**15)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("TB"):
        int_size = int(size[:-2]) * (10**12)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(f"`size={size}` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def glob_pattern_to_regex(pattern):
    # partially taken from fsspec:
    # https://github.com/fsspec/filesystem_spec/blob/697d0f8133d8a5fbc3926e4761d7ecd51337ce50/fsspec/asyn.py#L735
    return (
        pattern.replace("\\", r"\\")
        .replace(".", r"\.")
        .replace("*", ".*")
        .replace("+", r"\+")
        .replace("//", "/")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("|", r"\|")
        .replace("^", r"\^")
        .replace("$", r"\$")
        .rstrip("/")
        .replace("?", ".")
    )


def string_to_dict(string: str, pattern: str) -> Optional[dict[str, str]]:
    """Un-format a string using a python f-string pattern.
    From https://stackoverflow.com/a/36838374

    Example::

        >>> p = 'hello, my name is {name} and I am a {age} year old {what}'
        >>> s = p.format(name='cody', age=18, what='quarterback')
        >>> s
        'hello, my name is cody and I am a 18 year old quarterback'
        >>> string_to_dict(s, p)
        {'age': '18', 'name': 'cody', 'what': 'quarterback'}

    Args:
        string (str): input string
        pattern (str): pattern formatted like a python f-string
            This can be a regex - so in case of un-formatting paths you should use posix paths.
            Otherwise backslashes for windows paths can cause issues.

    Returns:
        Optional[dict[str, str]]: dictionary of variable -> value, retrieved from the input using the pattern, or
        `None` if the string does not match the pattern.
    """
    pattern = re.sub(r"{([^:}]+)(?::[^}]+)?}", r"{\1}", pattern)  # remove format specifiers, e.g. {rank:05d} -> {rank}
    regex = re.sub(r"{(.+?)}", r"(?P<_\1>.+)", pattern)
    result = re.search(regex, string)
    if result is None:
        return None
    values = list(result.groups())
    keys = re.findall(r"{(.+?)}", pattern)
    _dict = dict(zip(keys, values))
    return _dict


def asdict(obj):
    """Convert an object to its dictionary representation recursively.

    <Added version="2.4.0"/>
    """

    # Implementation based on https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict

    def _is_dataclass_instance(obj):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj):
        if _is_dataclass_instance(obj):
            result = {}
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name))
                if not f.init or value != f.default or f.metadata.get("include_in_asdict_even_if_is_default", False):
                    result[f.name] = value
            return result
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # obj is a namedtuple
            return type(obj)(*[_asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(_asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj)


@contextmanager
def temporary_assignment(obj, attr, value):
    """Temporarily assign obj.attr to value."""
    original = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


@contextmanager
def temp_seed(seed: int, set_pytorch=False, set_tensorflow=False):
    """Temporarily set the random seed. This works for python numpy, pytorch and tensorflow."""
    np_state = np.random.get_state()
    np.random.seed(seed)

    if set_pytorch and config.TORCH_AVAILABLE:
        import torch

        torch_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)

        if torch.cuda.is_available():
            torch_cuda_states = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(seed)

    if set_tensorflow and config.TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.python.eager import context as tfpycontext

        tf_state = tf.random.get_global_generator()
        temp_gen = tf.random.Generator.from_seed(seed)
        tf.random.set_global_generator(temp_gen)

        if not tf.executing_eagerly():
            raise ValueError("Setting random seed for TensorFlow is only available in eager mode")

        tf_context = tfpycontext.context()  # eager mode context
        tf_seed = tf_context._seed
        tf_rng_initialized = hasattr(tf_context, "_rng")
        if tf_rng_initialized:
            tf_rng = tf_context._rng
        tf_context._set_global_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)

        if set_pytorch and config.TORCH_AVAILABLE:
            torch.random.set_rng_state(torch_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_states)

        if set_tensorflow and config.TF_AVAILABLE:
            tf.random.set_global_generator(tf_state)

            tf_context._seed = tf_seed
            if tf_rng_initialized:
                tf_context._rng = tf_rng
            else:
                delattr(tf_context, "_rng")


def unique_values(values):
    """Iterate over iterable and return only unique values in order."""
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value


def no_op_if_value_is_null(func):
    """If the value is None, return None, else call `func`."""

    def wrapper(value):
        return func(value) if value is not None else None

    return wrapper


def first_non_null_value(iterable):
    """Return the index and the value of the first non-null value in the iterable. If all values are None, return -1 as index."""
    for i, value in enumerate(iterable):
        if value is not None:
            return i, value
    return -1, None


def first_non_null_non_empty_value(iterable):
    """Return the index and the value of the first non-null non-empty value in the iterable. If all values are None or empty, return -1 as index."""
    for i, value in enumerate(iterable):
        if value is not None and not (isinstance(value, (dict, list)) and len(value) == 0):
            return i, value
    return -1, None


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in unique_values(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


class NonMutableDict(dict):
    """Dict where keys can only be added but not modified.

    Will raise an error if the user try to overwrite one key. The error message
    can be customized during construction. It will be formatted using {key} for
    the overwritten key.
    """

    def __init__(self, *args, **kwargs):
        self._error_msg = kwargs.pop(
            "error_msg",
            "Try to overwrite existing key: {key}",
        )
        if kwargs:
            raise ValueError("NonMutableDict cannot be initialized with kwargs.")
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(self._error_msg.format(key=key))
        return super().__setitem__(key, value)

    def update(self, other):
        if any(k in self for k in other):
            raise ValueError(self._error_msg.format(key=set(self) & set(other)))
        return super().update(other)


class classproperty(property):  # pylint: disable=invalid-name
    """Descriptor to be used as decorator for @classmethods."""

    def __get__(self, obj, objtype=None):
        return self.fget.__get__(None, objtype)()


def _single_map_nested(args):
    """Apply a function recursively to each element of a nested data struct."""
    function, data_struct, batched, batch_size, types, rank, disable_tqdm, desc = args

    # Singleton first to spare some computation
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            return function([data_struct])[0]
        else:
            return function(data_struct)
    if (
        batched
        and not isinstance(data_struct, dict)
        and isinstance(data_struct, types)
        and all(not isinstance(v, (dict, types)) for v in data_struct)
    ):
        return [mapped_item for batch in iter_batched(data_struct, batch_size) for mapped_item in function(batch)]

    # Reduce logging to keep things readable in multiprocessing with tqdm
    if rank is not None and logging.get_verbosity() < logging.WARNING:
        logging.set_verbosity_warning()
    # Print at least one thing to fix tqdm in notebooks in multiprocessing
    # see https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
    if rank is not None and not disable_tqdm and any("notebook" in tqdm_cls.__name__ for tqdm_cls in tqdm.__mro__):
        print(" ", end="", flush=True)

    # Loop over single examples or batches and write to buffer/file if examples are to be updated
    pbar_iterable = data_struct.items() if isinstance(data_struct, dict) else data_struct
    pbar_desc = (desc + " " if desc is not None else "") + "#" + str(rank) if rank is not None else desc
    with hf_tqdm(pbar_iterable, disable=disable_tqdm, position=rank, unit="obj", desc=pbar_desc) as pbar:
        if isinstance(data_struct, dict):
            return {
                k: _single_map_nested((function, v, batched, batch_size, types, None, True, None)) for k, v in pbar
            }
        else:
            mapped = [_single_map_nested((function, v, batched, batch_size, types, None, True, None)) for v in pbar]
            if isinstance(data_struct, list):
                return mapped
            elif isinstance(data_struct, tuple):
                return tuple(mapped)
            else:
                return np.array(mapped)


def map_nested(
    function: Callable[[Any], Any],
    data_struct: Any,
    dict_only: bool = False,
    map_list: bool = True,
    map_tuple: bool = False,
    map_numpy: bool = False,
    num_proc: Optional[int] = None,
    parallel_min_length: int = 2,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    types: Optional[tuple] = None,
    disable_tqdm: bool = True,
    desc: Optional[str] = None,
) -> Any:
    """Apply a function recursively to each element of a nested data struct.

    Use multiprocessing if num_proc > 1 and the length of data_struct is greater than or equal to
    `parallel_min_length`.

    <Changed version="2.5.0">

    Before version 2.5.0, multiprocessing was not used if `num_proc` was greater than or equal to ``len(iterable)``.

    Now, if `num_proc` is greater than or equal to ``len(iterable)``, `num_proc` is set to ``len(iterable)`` and
    multiprocessing is used.

    </Changed>

    Args:
        function (`Callable`): Function to be applied to `data_struct`.
        data_struct (`Any`): Data structure to apply `function` to.
        dict_only (`bool`, default `False`): Whether only apply `function` recursively to `dict` values in
            `data_struct`.
        map_list (`bool`, default `True`): Whether also apply `function` recursively to `list` elements (besides `dict`
            values).
        map_tuple (`bool`, default `False`): Whether also apply `function` recursively to `tuple` elements (besides
            `dict` values).
        map_numpy (`bool, default `False`): Whether also apply `function` recursively to `numpy.array` elements (besides
            `dict` values).
        num_proc (`int`, *optional*): Number of processes.
            The level in the data struct used for multiprocessing is the first level that has smaller sub-structs,
            starting from the root.
        parallel_min_length (`int`, default `2`): Minimum length of `data_struct` required for parallel
            processing.
            <Added version="2.5.0"/>
        batched (`bool`, defaults to `False`):
            Provide batch of items to `function`.
            <Added version="2.19.0"/>
        batch_size (`int`, *optional*, defaults to `1000`):
            Number of items per batch provided to `function` if `batched=True`.
            If `batch_size <= 0` or `batch_size == None`, provide the full iterable as a single batch to `function`.
            <Added version="2.19.0"/>
        types (`tuple`, *optional*): Additional types (besides `dict` values) to apply `function` recursively to their
            elements.
        disable_tqdm (`bool`, default `True`): Whether to disable the tqdm progressbar.
        desc (`str`, *optional*): Prefix for the tqdm progressbar.

    Returns:
        `Any`
    """
    if types is None:
        types = []
        if not dict_only:
            if map_list:
                types.append(list)
            if map_tuple:
                types.append(tuple)
            if map_numpy:
                types.append(np.ndarray)
        types = tuple(types)

    # Singleton
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            data_struct = [data_struct]
        mapped = function(data_struct)
        if batched:
            mapped = mapped[0]
        return mapped

    iterable = list(data_struct.values()) if isinstance(data_struct, dict) else data_struct

    if num_proc is None:
        num_proc = 1
    if any(isinstance(v, types) and len(v) > len(iterable) for v in iterable):
        mapped = [
            map_nested(
                function=function,
                data_struct=obj,
                num_proc=num_proc,
                parallel_min_length=parallel_min_length,
                batched=batched,
                batch_size=batch_size,
                types=types,
            )
            for obj in iterable
        ]
    elif num_proc != -1 and num_proc <= 1 or len(iterable) < parallel_min_length:
        if batched:
            if batch_size is None or batch_size <= 0:
                batch_size = max(len(iterable) // num_proc + int(len(iterable) % num_proc > 0), 1)
            iterable = list(iter_batched(iterable, batch_size))
        mapped = [
            _single_map_nested((function, obj, batched, batch_size, types, None, True, None))
            for obj in hf_tqdm(iterable, disable=disable_tqdm, desc=desc)
        ]
        if batched:
            mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".* is experimental and might be subject to breaking changes in the future\\.$",
                category=UserWarning,
            )
            if batched:
                if batch_size is None or batch_size <= 0:
                    batch_size = len(iterable) // num_proc + int(len(iterable) % num_proc > 0)
                iterable = list(iter_batched(iterable, batch_size))
            mapped = parallel_map(
                function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, _single_map_nested
            )
            if batched:
                mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]

    if isinstance(data_struct, dict):
        return dict(zip(data_struct.keys(), mapped))
    else:
        if isinstance(data_struct, list):
            return mapped
        elif isinstance(data_struct, tuple):
            return tuple(mapped)
        else:
            return np.array(mapped)


class NestedDataStructure:
    def __init__(self, data=None):
        self.data = data if data is not None else []

    def flatten(self, data=None):
        data = data if data is not None else self.data
        if isinstance(data, dict):
            return self.flatten(list(data.values()))
        elif isinstance(data, (list, tuple)):
            return [flattened for item in data for flattened in self.flatten(item)]
        else:
            return [data]


def has_sufficient_disk_space(needed_bytes, directory="."):
    try:
        free_bytes = disk_usage(os.path.abspath(directory)).free
    except OSError:
        return True
    return needed_bytes < free_bytes


def copyfunc(func):
    result = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    result.__kwdefaults__ = func.__kwdefaults__
    return result


Y = TypeVar("Y")


def _write_generator_to_queue(queue: queue.Queue, func: Callable[..., Iterable[Y]], kwargs: dict) -> int:
    for i, result in enumerate(func(**kwargs)):
        queue.put(result)
    return i


def _get_pool_pid(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool]) -> set[int]:
    return {f.pid for f in pool._pool}


def iflatmap_unordered(
    pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool],
    func: Callable[..., Iterable[Y]],
    *,
    kwargs_iterable: Iterable[dict],
) -> Iterable[Y]:
    initial_pool_pid = _get_pool_pid(pool)
    pool_changed = False
    with pool._ctx.Manager() as manager:
        queue = manager.Queue()
        async_results = [
            pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_iterable
        ]
        try:
            while True:
                try:
                    yield queue.get(timeout=0.05)
                except Empty:
                    if all(async_result.ready() for async_result in async_results) and queue.empty():
                        break
                if _get_pool_pid(pool) != initial_pool_pid:
                    pool_changed = True
                    # One of the subprocesses has died. We should not wait forever.
                    raise RuntimeError(
                        "One of the subprocesses has abruptly died during map operation."
                        "To debug the error, disable multiprocessing."
                    )
        finally:
            if not pool_changed:
                # we get the result in case there's an error to raise
                [async_result.get(timeout=0.05) for async_result in async_results]


T = TypeVar("T")


def iter_batched(iterable: Iterable[T], n: int) -> Iterable[list[T]]:
    if n < 1:
        raise ValueError(f"Invalid batch size {n}")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
<|endoftext|>from typing import Any, Optional, Union

from huggingface_hub.utils import get_session

from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
    get_authentication_headers_for_url,
)
from .logging import get_logger


logger = get_logger(__name__)


class DatasetViewerError(DatasetsError):
    """Dataset viewer error.

    Raised when trying to use the dataset viewer HTTP API and when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    - unavailable /parquet or /info responses
    """


def get_exported_parquet_files(
    dataset: str, commit_hash: str, token: Optional[Union[str, bool]]
) -> list[dict[str, Any]]:
    """
    Get the dataset exported parquet files
    Docs: https://huggingface.co/docs/datasets-server/parquet
    """
    dataset_viewer_parquet_url = config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/parquet?dataset="
    try:
        parquet_data_files_response = get_session().get(
            url=dataset_viewer_parquet_url + dataset,
            headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
        )
        parquet_data_files_response.raise_for_status()
        if "X-Revision" in parquet_data_files_response.headers:
            if parquet_data_files_response.headers["X-Revision"] == commit_hash or commit_hash is None:
                parquet_data_files_response_json = parquet_data_files_response.json()
                if (
                    parquet_data_files_response_json.get("partial") is False
                    and not parquet_data_files_response_json.get("pending", True)
                    and not parquet_data_files_response_json.get("failed", True)
                    and "parquet_files" in parquet_data_files_response_json
                ):
                    return parquet_data_files_response_json["parquet_files"]
                else:
                    logger.debug(f"Parquet export for {dataset} is not completely ready yet.")
            else:
                logger.debug(
                    f"Parquet export for {dataset} is available but outdated (commit_hash='{parquet_data_files_response.headers['X-Revision']}')"
                )
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the parquet export doesn't exist
        logger.debug(f"No parquet export for {dataset} available ({type(e).__name__}: {e})")
    raise DatasetViewerError("No exported Parquet files available.")


def get_exported_dataset_infos(
    dataset: str, commit_hash: str, token: Optional[Union[str, bool]]
) -> dict[str, dict[str, Any]]:
    """
    Get the dataset information, can be useful to get e.g. the dataset features.
    Docs: https://huggingface.co/docs/datasets-server/info
    """
    dataset_viewer_info_url = config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/info?dataset="
    try:
        info_response = get_session().get(
            url=dataset_viewer_info_url + dataset,
            headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
        )
        info_response.raise_for_status()
        if "X-Revision" in info_response.headers:
            if info_response.headers["X-Revision"] == commit_hash or commit_hash is None:
                info_response = info_response.json()
                if (
                    info_response.get("partial") is False
                    and not info_response.get("pending", True)
                    and not info_response.get("failed", True)
                    and "dataset_info" in info_response
                ):
                    return info_response["dataset_info"]
                else:
                    logger.debug(f"Dataset info for {dataset} is not completely ready yet.")
            else:
                logger.debug(
                    f"Dataset info for {dataset} is available but outdated (commit_hash='{info_response.headers['X-Revision']}')"
                )
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the dataset info doesn't exist
        logger.debug(f"No dataset info for {dataset} available ({type(e).__name__}: {e})")
    raise DatasetViewerError("No exported dataset infos available.")
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
from typing import Optional

from .tqdm import (  # noqa: F401 # imported for backward compatibility
    disable_progress_bar,
    enable_progress_bar,
    is_progress_bar_enabled,
    tqdm,
)


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level():
    """
    If DATASETS_VERBOSITY env var is set to one of the valid choices return that as the new default level.
    If it is not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("DATASETS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option DATASETS_VERBOSITY={env_level_str}, has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    # Apply our default configuration to the library root logger.
    library_root_logger = _get_library_root_logger()
    library_root_logger.addHandler(logging.StreamHandler())
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.
    This function can be used in dataset builders.
    """
    if name is None:
        name = _get_library_name()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the HuggingFace datasets library's root logger.
    Returns:
        Logging level, e.g., `datasets.logging.DEBUG` and `datasets.logging.INFO`.

    > [!TIP]
    > HuggingFace datasets library has following logging levels:
    >     - `datasets.logging.CRITICAL`, `datasets.logging.FATAL`
    >     - `datasets.logging.ERROR`
    >     - `datasets.logging.WARNING`, `datasets.logging.WARN`
    >     - `datasets.logging.INFO`
    >     - `datasets.logging.DEBUG`
    """
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the level for the Hugging Face Datasets library's root logger.
    Args:
        verbosity:
            Logging level, e.g., `datasets.logging.DEBUG` and `datasets.logging.INFO`.
    """
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the level for the Hugging Face datasets library's root logger to `INFO`.

    This will display most of the logging information and tqdm bars.

    Shortcut to `datasets.logging.set_verbosity(datasets.logging.INFO)`.
    """
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the level for the Hugging Face datasets library's root logger to `WARNING`.

    This will display only the warning and errors logging information and tqdm bars.

    Shortcut to `datasets.logging.set_verbosity(datasets.logging.WARNING)`.
    """
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the level for the Hugging Face datasets library's root logger to `DEBUG`.

    This will display all the logging information and tqdm bars.

    Shortcut to `datasets.logging.set_verbosity(datasets.logging.DEBUG)`.
    """
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the level for the Hugging Face datasets library's root logger to `ERROR`.

    This will display only the errors logging information and tqdm bars.

    Shortcut to `datasets.logging.set_verbosity(datasets.logging.ERROR)`.
    """
    return set_verbosity(ERROR)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the Hugging Face datasets library's default handler to prevent double logging if the root logger has
    been configured.
    """
    _get_library_root_logger().propagate = True


# Configure the library root logger at the module level (singleton-like)
_configure_library_root_logger()
<|endoftext|>import numpy as np


def approximate_mode(class_counts, n_draws, rng):
    """Computes approximate mode of multivariate hypergeometric.
    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.
    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.
    Args
    ----------
    class_counts : ndarray of int
        Population per class.
    n_draws : int
        Number of draws (samples to draw) from the overall population.
    rng : random state
        Used to break ties.
    Returns
    -------
    sampled_classes : ndarray of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws

    """
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = n_draws * class_counts / class_counts.sum()
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = np.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = np.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int64)


def stratified_shuffle_split_generate_indices(y, n_train, n_test, rng, n_splits=10):
    """

    Provides train/test indices to split data in train/test sets.
    It's reference is taken from StratifiedShuffleSplit implementation
    of scikit-learn library.

    Args
    ----------

    n_train : int,
        represents the absolute number of train samples.

    n_test : int,
        represents the absolute number of test samples.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError("Minimum class count error")
    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or equal to the number of classes = %d" % (n_train, n_classes)
        )
    if n_test < n_classes:
        raise ValueError(
            "The test_size = %d should be greater or equal to the number of classes = %d" % (n_test, n_classes)
        )
    class_indices = np.split(np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1])
    for _ in range(n_splits):
        n_i = approximate_mode(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = approximate_mode(class_counts_remaining, n_test, rng)

        train = []
        test = []

        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
            train.extend(perm_indices_class_i[: n_i[i]])
            test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])
        train = rng.permutation(train)
        test = rng.permutation(test)

        yield train, test
<|endoftext|>#!/usr/bin/env python
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities to handle file locking in `datasets`."""

import os

from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from packaging import version


class FileLock(FileLock_):
    """
    A `filelock.FileLock` initializer that handles long paths.
    It also uses the current umask for lock files.
    """

    MAX_FILENAME_LENGTH = 255

    def __init__(self, lock_file, *args, **kwargs):
        # The "mode" argument is required if we want to use the current umask in filelock >= 3.10
        # In previous previous it was already using the current umask.
        if "mode" not in kwargs and version.parse(_filelock_version) >= version.parse("3.10.0"):
            umask = os.umask(0o666)
            os.umask(umask)
            kwargs["mode"] = 0o666 & ~umask
        lock_file = self.hash_filename_if_too_long(lock_file)
        super().__init__(lock_file, *args, **kwargs)

    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        filename = os.path.basename(path)
        max_filename_length = cls.MAX_FILENAME_LENGTH
        if issubclass(cls, UnixFileLock):
            max_filename_length = min(max_filename_length, os.statvfs(os.path.dirname(path)).f_namemax)
        if len(filename) > max_filename_length:
            dirname = os.path.dirname(path)
            hashed_filename = str(hash(filename))
            new_filename = (
                filename[: max_filename_length - len(hashed_filename) - 8] + "..." + hashed_filename + ".lock"
            )
            return os.path.join(dirname, new_filename)
        else:
            return path
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

from . import tqdm as _tqdm  # _tqdm is the module
from .experimental import experimental
from .info_utils import VerificationMode
from .logging import disable_progress_bar, enable_progress_bar, is_progress_bar_enabled
from .tqdm import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    tqdm,
)
from .version import Version
<|endoftext|>"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import asyncio
import glob
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import tarfile
import time
import xml.dom.minidom
import zipfile
from collections.abc import Generator
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import fsspec
import huggingface_hub
import huggingface_hub.errors
from fsspec.core import strip_protocol, url_to_fs
from fsspec.utils import can_be_local
from huggingface_hub.utils import get_session, httpx, insecure_hashlib
from packaging import version

from .. import __version__, config
from ..download.download_config import DownloadConfig
from ..filesystems import COMPRESSION_FILESYSTEMS
from . import _tqdm, logging
from ._filelock import FileLock
from .extract import ExtractManager
from .track import TrackedIterableFromGenerator


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INCOMPLETE_SUFFIX = ".incomplete"

T = TypeVar("T", str, Path)

CONNECTION_ERRORS_TO_RETRY = (
    asyncio.TimeoutError,
    httpx.RequestError,
)
SERVER_UNAVAILABLE_CODE = 504
RATE_LIMIT_CODE = 429


def _get_connection_errors_to_retry() -> tuple[type[Exception], ...]:
    # aiohttp is slow to import, and it can't have raised an error if it was never imported
    if "aiohttp" in sys.modules:
        from aiohttp import ClientError

        return CONNECTION_ERRORS_TO_RETRY + (ClientError,)
    return CONNECTION_ERRORS_TO_RETRY


def is_remote_url(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme != "" and not os.path.ismount(urlparse(url_or_filename).scheme + ":/")


def is_local_path(url_or_filename: str) -> bool:
    # On unix the scheme of a local path is empty (for both absolute and relative),
    # while on windows the scheme is the drive name (ex: "c") for absolute paths.
    # for details on the windows behavior, see https://bugs.python.org/issue42215
    return urlparse(url_or_filename).scheme == "" or os.path.ismount(urlparse(url_or_filename).scheme + ":/")


def is_relative_path(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme == "" and not os.path.isabs(url_or_filename)


def relative_to_absolute_path(path: T) -> T:
    """Convert relative path to absolute path."""
    abs_path_str = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    return Path(abs_path_str) if isinstance(path, Path) else abs_path_str


def url_or_path_join(base_name: str, *pathnames: str) -> str:
    if is_remote_url(base_name):
        return posixpath.join(base_name, *(str(pathname).replace(os.sep, "/").lstrip("/") for pathname in pathnames))
    else:
        return Path(base_name, *pathnames).as_posix()


def url_or_path_parent(url_or_path: str) -> str:
    if is_remote_url(url_or_path):
        return url_or_path[: url_or_path.rindex("/")]
    else:
        return os.path.dirname(url_or_path)


def hash_url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".py"):
        filename += ".py"

    return filename


def cached_path(
    url_or_filename,
    download_config=None,
    **download_kwargs,
) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
        ValueError: if it couldn't parse the url or filename correctly
        httpx.NetworkError: in case of internet connection issue
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)

    cache_dir = download_config.cache_dir or config.DOWNLOADED_DATASETS_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    # Convert fsspec URL in the format "file://local/path" to "local/path"
    if can_be_local(url_or_filename):
        url_or_filename = strip_protocol(url_or_filename)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        url_or_filename, storage_options = _prepare_path_and_storage_options(
            url_or_filename, download_config=download_config
        )
        # Download files from Hugging Face.
        # Note: no need to check for https://huggingface.co file URLs since _prepare_path_and_storage_options
        # prepares Hugging Face HTTP URLs as hf:// paths already
        if url_or_filename.startswith("hf://") and not url_or_filename.startswith("hf://buckets/"):
            resolved_path = huggingface_hub.HfFileSystem(
                endpoint=config.HF_ENDPOINT, token=download_config.token
            ).resolve_path(url_or_filename)
            try:
                output_path = huggingface_hub.HfApi(
                    endpoint=config.HF_ENDPOINT,
                    token=download_config.token,
                    library_name="datasets",
                    library_version=__version__,
                    user_agent=get_datasets_user_agent(download_config.user_agent),
                ).hf_hub_download(
                    repo_id=resolved_path.repo_id,
                    repo_type=resolved_path.repo_type,
                    revision=resolved_path.revision,
                    filename=resolved_path.path_in_repo,
                    force_download=download_config.force_download,
                )
            except (
                huggingface_hub.utils.RepositoryNotFoundError,
                huggingface_hub.utils.EntryNotFoundError,
                huggingface_hub.utils.RevisionNotFoundError,
                huggingface_hub.utils.GatedRepoError,
            ) as e:
                raise FileNotFoundError(str(e)) from e
        # Download external files
        else:
            output_path = get_from_cache(
                url_or_filename,
                cache_dir=cache_dir,
                force_download=download_config.force_download,
                user_agent=download_config.user_agent,
                use_etag=download_config.use_etag,
                token=download_config.token,
                storage_options=storage_options,
                download_desc=download_config.download_desc,
                disable_tqdm=download_config.disable_tqdm,
            )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif is_local_path(url_or_filename):
        # File, but it doesn't exist.
        raise FileNotFoundError(f"Local file {url_or_filename} doesn't exist")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if output_path is None:
        return output_path

    if download_config.extract_compressed_file:
        if download_config.extract_on_the_fly:
            # Add a compression prefix to the compressed file so that it can be extracted
            # as it's being read using xopen.
            protocol = _get_extraction_protocol(output_path, download_config=download_config)
            extension = _get_path_extension(url_or_filename.split("::")[0])
            if (
                protocol
                and extension not in ["tgz", "tar"]
                and not url_or_filename.split("::")[0].endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
            ):
                output_path = relative_to_absolute_path(output_path)
                if protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
                    # there is one single file which is the uncompressed file
                    inner_file = os.path.basename(output_path)
                    inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
                    output_path = f"{protocol}://{inner_file}::{output_path}"
                else:
                    output_path = f"{protocol}://::{output_path}"
                return output_path

        # Eager extraction
        output_path = ExtractManager(cache_dir=download_config.cache_dir).extract(
            output_path, force_extract=download_config.force_extract
        )
    return relative_to_absolute_path(output_path)


def get_datasets_user_agent(user_agent: Optional[Union[str, dict]] = None) -> str:
    ua = f"datasets/{__version__}"
    ua += f"; python/{config.PY_VERSION}"
    ua += f"; hf_hub/{huggingface_hub.__version__}"
    ua += f"; pyarrow/{config.PYARROW_VERSION}"
    if config.TORCH_AVAILABLE:
        ua += f"; torch/{config.TORCH_VERSION}"
    if config.TF_AVAILABLE:
        ua += f"; tensorflow/{config.TF_VERSION}"
    if config.JAX_AVAILABLE:
        ua += f"; jax/{config.JAX_VERSION}"
    if isinstance(user_agent, dict):
        ua += f"; {'; '.join(f'{k}/{v}' for k, v in user_agent.items())}"
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def get_authentication_headers_for_url(url: str, token: Optional[Union[str, bool]] = None) -> dict:
    """Handle the HF authentication"""
    if url.startswith(config.HF_ENDPOINT):
        return huggingface_hub.utils.build_hf_headers(
            token=token, library_name="datasets", library_version=__version__
        )
    else:
        return {}


def _raise_if_offline_mode_is_enabled(msg: Optional[str] = None):
    """Raise an OfflineModeIsEnabled error (subclass of ConnectionError) if HF_HUB_OFFLINE is True."""
    if config.HF_HUB_OFFLINE:
        raise huggingface_hub.errors.OfflineModeIsEnabled(
            "Offline mode is enabled." if msg is None else "Offline mode is enabled. " + str(msg)
        )


def fsspec_head(url, storage_options=None):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    return fs.info(path)


def stack_multiprocessing_download_progress_bars():
    # Stack downloads progress bars automatically using HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS=1
    # We use environment variables since the download may happen in a subprocess
    return patch.dict(os.environ, {"HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS": "1"})


class TqdmCallback(fsspec.callbacks.TqdmCallback):
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        if config.FSSPEC_VERSION < version.parse("2024.2.0"):
            super().__init__(tqdm_kwargs, *args, **kwargs)
            self._tqdm = _tqdm  # replace tqdm module by datasets.utils.tqdm module
        else:
            kwargs["tqdm_cls"] = _tqdm.tqdm
            super().__init__(tqdm_kwargs, *args, **kwargs)


def fsspec_get(url, temp_file, storage_options=None, desc=None, disable_tqdm=False):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    callback = TqdmCallback(
        tqdm_kwargs={
            "desc": desc or "Downloading",
            "unit": "B",
            "unit_scale": True,
            "position": multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
            if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
            and multiprocessing.current_process()._identity
            else None,
            "disable": disable_tqdm,
        }
    )
    fs.get_file(path, temp_file.name, callback=callback)


def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    user_agent=None,
    use_etag=True,
    token=None,
    storage_options=None,
    download_desc=None,
    disable_tqdm=False,
) -> str:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
    """
    if storage_options is None:
        storage_options = {}
    if cache_dir is None:
        cache_dir = config.HF_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    response = None
    etag = None

    # Try a first time to file the file on the local file system without eTag (None)
    # if we don't ask for 'force_download' then we spare a request
    filename = hash_url_to_filename(url, etag=None)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download and not use_etag:
        return cache_path

    # Prepare headers for authentication
    headers = get_authentication_headers_for_url(url, token=token)
    if user_agent is not None:
        headers["user-agent"] = user_agent

    response = fsspec_head(url, storage_options=storage_options)
    etag = (response.get("ETag", None) or response.get("etag", None)) if use_etag else None

    # Try a second time
    filename = hash_url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # Retry in case previously locked processes just enter after the precedent process releases the lock
        if os.path.exists(cache_path) and not force_download:
            return cache_path

        incomplete_path = cache_path + ".incomplete"

        # Download to temporary file, then copy to cache path once finished.
        # Otherwise, you get corrupt cache entries if the download gets interrupted.
        with open(incomplete_path, "w+b") as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")
            # GET file object
            fsspec_get(url, temp_file, storage_options=storage_options, desc=download_desc, disable_tqdm=disable_tqdm)

        logger.info(f"storing {url} in cache at {cache_path}")
        shutil.move(temp_file.name, cache_path)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + "\n\n" + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "\n\n" + "".join(docstr)
        return fn

    return docstring_decorator


def estimate_dataset_size(paths):
    return sum(path.stat().st_size for path in paths)


def readline(f: io.RawIOBase):
    # From: https://github.com/python/cpython/blob/d27e2f4d118e7a9909b6a3e5da06c5ff95806a85/Lib/_pyio.py#L525
    res = bytearray()
    while True:
        b = f.read(1)
        if not b:
            break
        res += b
        if res.endswith(b"\n"):
            break
    return bytes(res)


#######################
# Streaming utilities #
#######################

BASE_KNOWN_EXTENSIONS = [
    "txt",
    "csv",
    "json",
    "jsonl",
    "tsv",
    "conll",
    "conllu",
    "orig",
    "parquet",
    "pkl",
    "pickle",
    "rel",
    "xml",
    "arrow",
]
COMPRESSION_EXTENSION_TO_PROTOCOL = {
    # single file compression
    **{
        extension.lstrip("."): fs_class.protocol
        for fs_class in COMPRESSION_FILESYSTEMS
        for extension in fs_class.extensions
    },
    # archive compression
    "zip": "zip",
    "eval": "zip",
}
SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL = {
    extension.lstrip("."): fs_class.protocol
    for fs_class in COMPRESSION_FILESYSTEMS
    for extension in fs_class.extensions
}
SINGLE_FILE_COMPRESSION_PROTOCOLS = {fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS}
SINGLE_SLASH_AFTER_PROTOCOL_PATTERN = re.compile(r"(?<!:):/")


MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL = {
    bytes.fromhex("504B0304"): "zip",
    bytes.fromhex("504B0506"): "zip",  # empty archive
    bytes.fromhex("504B0708"): "zip",  # spanned archive
    bytes.fromhex("425A68"): "bz2",
    bytes.fromhex("1F8B"): "gzip",
    bytes.fromhex("FD377A585A00"): "xz",
    bytes.fromhex("04224D18"): "lz4",
    bytes.fromhex("28B52FFD"): "zstd",
}
MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL = {
    b"Rar!": "rar",
}
MAGIC_NUMBER_MAX_LENGTH = max(
    len(magic_number)
    for magic_number in chain(MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL, MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL)
)


class NonStreamableDatasetError(Exception):
    pass


def _get_path_extension(path: str) -> str:
    # Get extension: https://foo.bar/train.json.gz -> gz
    extension = path.split(".")[-1]
    # Remove query params ("dl=1", "raw=true"): gz?dl=1 -> gz
    # Remove shards infos (".txt_1", ".txt-00000-of-00100"): txt_1 -> txt
    for symb in "?-_":
        extension = extension.split(symb)[0]
    return extension


def _get_extraction_protocol_with_magic_number(f) -> Optional[str]:
    """read the magic number from a file-like object and return the compression protocol"""
    # Check if the file object is seekable even before reading the magic number (to avoid https://bugs.python.org/issue26440)
    try:
        f.seek(0)
    except (AttributeError, io.UnsupportedOperation):
        return None
    magic_number = f.read(MAGIC_NUMBER_MAX_LENGTH)
    f.seek(0)
    for i in range(MAGIC_NUMBER_MAX_LENGTH):
        compression = MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            return compression
        compression = MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            raise NotImplementedError(f"Compression protocol '{compression}' not implemented.")


def _get_extraction_protocol(urlpath: str, download_config: Optional[DownloadConfig] = None) -> Optional[str]:
    # get inner file: zip://train-00000.json.gz::https://foo.bar/data.zip -> zip://train-00000.json.gz
    urlpath = str(urlpath)
    path = urlpath.split("::")[0]
    extension = _get_path_extension(path)
    if (
        extension in BASE_KNOWN_EXTENSIONS
        or extension in ["tgz", "tar"]
        or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    ):
        return None
    elif extension in COMPRESSION_EXTENSION_TO_PROTOCOL:
        return COMPRESSION_EXTENSION_TO_PROTOCOL[extension]
    urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
    try:
        with fsspec.open(urlpath, **(storage_options or {})) as f:
            return _get_extraction_protocol_with_magic_number(f)
    except FileNotFoundError:
        if urlpath.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(
                urlpath + "\nIf the repo is private or gated, make sure to log in with `hf auth login`."
            ) from None
        else:
            raise


def xjoin(a, *p):
    """
    This function extends os.path.join to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xjoin function allows you to apply the join on the first path of the chain.

    Example::

        >>> xjoin("zip://folder1::https://host.com/archive.zip", "file.txt")
        zip://folder1/file.txt::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.join(a, *p)
    else:
        a = posixpath.join(a, *p)
        return "::".join([a] + b)


def xdirname(a):
    """
    This function extends os.path.dirname to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xdirname function allows you to apply the dirname on the first path of the chain.

    Example::

        >>> xdirname("zip://folder1/file.txt::https://host.com/archive.zip")
        zip://folder1::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        a = os.path.dirname(Path(a).as_posix())
    else:
        a = posixpath.dirname(a)
    # if we end up at the root of the protocol, we get for example a = 'http:'
    # so we have to fix it by adding the '//' that was removed:
    if a.endswith(":"):
        a += "//"
    return "::".join([a] + b)


def xexists(urlpath: str, download_config: Optional[DownloadConfig] = None):
    """Extend `os.path.exists` function to support both local and remote files.

    Args:
        urlpath (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """

    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return os.path.exists(main_hop)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        return fs.exists(main_hop)


def xbasename(a):
    """
    This function extends os.path.basename to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xbasename function allows you to apply the basename on the first path of the chain.

    Example::

        >>> xbasename("zip://folder1/file.txt::https://host.com/archive.zip")
        file.txt
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.basename(Path(a).as_posix())
    else:
        return posixpath.basename(a)


def xsplit(a):
    """
    This function extends os.path.split to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplit function allows you to apply the xsplit on the first path of the chain.

    Example::

        >>> xsplit("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1::https://host.com/archive.zip', 'file.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.split(Path(a).as_posix())
    else:
        a, tail = posixpath.split(a)
        return "::".join([a + "//" if a.endswith(":") else a] + b), tail


def xsplitext(a):
    """
    This function extends os.path.splitext to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplitext function allows you to apply the splitext on the first path of the chain.

    Example::

        >>> xsplitext("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1/file::https://host.com/archive.zip', '.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.splitext(Path(a).as_posix())
    else:
        a, ext = posixpath.splitext(a)
        return "::".join([a] + b), ext


def xisfile(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isfile` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isfile(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        return fs.isfile(main_hop)


def xgetsize(path, download_config: Optional[DownloadConfig] = None) -> int:
    """Extend `os.path.getsize` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `int`: optional
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.getsize(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        try:
            size = fs.size(main_hop)
        except huggingface_hub.utils.EntryNotFoundError:
            raise FileNotFoundError(f"No such file: {path}")
        if size is None:
            # use xopen instead of fs.open to make data fetching more robust
            with xopen(path, download_config=download_config) as f:
                size = len(f.read())
        return size


def xisdir(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if not inner_path.strip("/"):
            return True
        return fs.isdir(inner_path)


def xrelpath(path, start=None):
    """Extend `os.path.relpath` function to support remote files.

    Args:
        path (`str`): URL path.
        start (`str`): Start URL directory path.

    Returns:
        `str`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.relpath(main_hop, start=start) if start else os.path.relpath(main_hop)
    else:
        return posixpath.relpath(main_hop, start=str(start).split("::")[0]) if start else os.path.relpath(main_hop)


class _OverridableIOWrapper(io.RawIOBase):
    def __init__(self, f):
        self._overrides = {}
        self.f = f

    def __getattribute__(self, attr):
        if attr == "_overrides":
            return object.__getattribute__(self, attr)
        elif attr in self._overrides:
            return self._overrides[attr]
        return getattr(self.f, attr)

    def __setattr__(self, attr, value):
        if attr == "_overrides":
            object.__setattr__(self, attr, value)
        else:
            self._overrides[attr] = value


def _add_retries_to_file_obj_read_method(file_obj):
    read = file_obj.read
    max_retries = config.STREAMING_READ_MAX_RETRIES

    def read_with_retries(*args, **kwargs):
        disconnect_err = None
        for retry in range(1, max_retries + 1):
            try:
                out = read(*args, **kwargs)
                break
            except _get_connection_errors_to_retry() as err:
                disconnect_err = err
                logger.warning(
                    f"Got disconnected from remote data host. Retrying in {config.STREAMING_READ_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
                )
                time.sleep(config.STREAMING_READ_RETRY_INTERVAL)
            except huggingface_hub.errors.HfHubHTTPError as err:
                if err.response is not None and err.response.status_code == SERVER_UNAVAILABLE_CODE:
                    disconnect_err = err
                    logger.warning(
                        f"Got disconnected from remote data host. Retrying in {config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
                    )
                    time.sleep(config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL)
                elif err.response is not None and err.response.status_code == RATE_LIMIT_CODE:
                    disconnect_err = err
                    logger.warning(str(err))
                    logger.warning(
                        f"Got disconnected from remote data host. Retrying in {config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
                    )
                    time.sleep(config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL)
                else:
                    raise
        else:
            raise ConnectionError("Server Disconnected") from disconnect_err
        return out

    try:
        file_obj.read = read_with_retries
    except AttributeError:  # read-only attribute
        file_obj = _OverridableIOWrapper(file_obj)
        file_obj.read = read_with_retries
    return file_obj


def _prepare_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> tuple[str, dict[str, dict[str, Any]]]:
    prepared_urlpath = []
    prepared_storage_options = {}
    for hop in urlpath.split("::"):
        hop, storage_options = _prepare_single_hop_path_and_storage_options(hop, download_config=download_config)
        prepared_urlpath.append(hop)
        prepared_storage_options.update(storage_options)
    return "::".join(prepared_urlpath), prepared_storage_options


def _prepare_single_hop_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    Prepare the URL and the kwargs that must be passed to the HttpFileSystem or HfFileSystem

    In particular it resolves google drive URLs
    It also adds the authentication headers for the Hugging Face Hub, for both https:// and hf:// paths.

    Storage options are formatted in the form {protocol: storage_options_for_protocol}
    """
    token = None if download_config is None else download_config.token
    if urlpath.startswith(config.HF_ENDPOINT) and "/resolve/" in urlpath:
        urlpath = "hf://" + urlpath[len(config.HF_ENDPOINT) + 1 :].replace("/resolve/", "@", 1)
    protocol = urlpath.split("://")[0] if "://" in urlpath else "file"
    if download_config is not None and protocol in download_config.storage_options:
        storage_options = download_config.storage_options[protocol].copy()
    elif download_config is not None and protocol not in download_config.storage_options:
        storage_options = {
            option_name: option_value
            for option_name, option_value in download_config.storage_options.items()
            if option_name not in fsspec.available_protocols()
        }
    else:
        storage_options = {}
    if protocol in {"http", "https"}:
        client_kwargs = storage_options.pop("client_kwargs", {})
        storage_options["client_kwargs"] = {"trust_env": True, **client_kwargs}  # Enable reading proxy env variables
        if "drive.google.com" in urlpath:
            response = get_session().head(urlpath, timeout=10)
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    urlpath += "&confirm=" + v
                    cookies = response.cookies
                    storage_options = {"cookies": cookies, **storage_options}
            # Fix Google Drive URL to avoid Virus scan warning
            if "confirm=" not in urlpath:
                urlpath += "&confirm=t"
        if urlpath.startswith("https://raw.githubusercontent.com/"):
            # Workaround for served data with gzip content-encoding: https://github.com/fsspec/filesystem_spec/issues/389
            headers = storage_options.pop("headers", {})
            storage_options["headers"] = {"Accept-Encoding": "identity", **headers}
    elif protocol == "hf":
        storage_options = {
            "endpoint": config.HF_ENDPOINT,
            "token": token,
            **storage_options,
        }
    if storage_options:
        storage_options = {protocol: storage_options}
    return urlpath, storage_options


def xopen(file: str, mode="r", *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `open` function to support remote files using `fsspec`.

    It also has a retry mechanism in case connection fails.
    The `args` and `kwargs` are passed to `fsspec.open`, except `token` which is used for queries to private repos on huggingface.co

    Args:
        file (`str`): Path name of the file to be opened.
        mode (`str`, *optional*, default "r"): Mode in which the file is opened.
        *args: Arguments to be passed to `fsspec.open`.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Keyword arguments to be passed to `fsspec.open`.

    Returns:
        file object
    """
    # This works as well for `xopen(str(Path(...)))`
    file_str = _as_str(file)
    main_hop, *rest_hops = file_str.split("::")
    if is_local_path(main_hop):
        # ignore fsspec-specific kwargs
        kwargs.pop("block_size", None)
        return open(main_hop, mode, *args, **kwargs)
    # add headers and cookies for authentication on the HF Hub and for Google Drive
    file, storage_options = _prepare_path_and_storage_options(file_str, download_config=download_config)
    kwargs = {**kwargs, **(storage_options or {})}

    max_retries = config.STREAMING_OPEN_MAX_RETRIES

    disconnect_err = None
    for retry in range(1, max_retries + 1):
        try:
            fs, fs_token, paths = fsspec.get_fs_token_paths(
                file,
                mode,
                storage_options=kwargs,
            )
            file_obj = fs.open(paths[0], mode)
            if hasattr(fs, "of") and hasattr(fs.of, "__exit__"):
                file_obj._fs = fs  # keep a reference or the fs might close the file on gc
            break
        except _get_connection_errors_to_retry() as err:
            disconnect_err = err
            logger.warning(
                f"Failed to connect to remote data host. Retrying in {config.STREAMING_OPEN_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
            )
            time.sleep(config.STREAMING_OPEN_RETRY_INTERVAL)
        except ValueError as e:
            if str(e) == "Cannot seek streaming HTTP file":
                raise NonStreamableDatasetError(
                    "Streaming is not possible for this dataset because data host server doesn't support HTTP range "
                    "requests. You can still load this dataset in non-streaming mode by passing `streaming=False` (default)"
                ) from e
            else:
                raise
        except FileNotFoundError:
            if file.startswith(config.HF_ENDPOINT):
                raise FileNotFoundError(
                    file + "\nIf the repo is private or gated, make sure to log in with `hf auth login`."
                ) from None
            else:
                raise
    else:
        raise ConnectionError("Server Disconnected") from disconnect_err
    file_obj = _add_retries_to_file_obj_read_method(file_obj)
    return file_obj


def xlistdir(path: str, download_config: Optional[DownloadConfig] = None) -> list[str]:
    """Extend `os.listdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(path).split("::")
    if is_local_path(main_hop):
        return os.listdir(path)
    else:
        # globbing inside a zip in a private repo requires authentication
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            raise FileNotFoundError(f"Directory doesn't exist: {path}")
        paths = fs.listdir(inner_path, detail=False)
        return [os.path.basename(path.rstrip("/")) for path in paths]


def xglob(urlpath, *, recursive=False, download_config: Optional[DownloadConfig] = None):
    """Extend `glob.glob` function to support remote files.

    Args:
        urlpath (`str`): URL path with shell-style wildcard patterns.
        recursive (`bool`, default `False`): Whether to match the "**" pattern recursively to zero or more
            directories or subdirectories.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return glob.glob(main_hop, recursive=recursive)
    else:
        # globbing inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[1]
        globbed_paths = fs.glob(inner_path)
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        return ["::".join([f"{protocol}://{globbed_path}"] + rest_hops) for globbed_path in globbed_paths]


def xwalk(urlpath, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `os.walk` function to support remote files.

    Args:
        urlpath (`str`): URL root path.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Additional keyword arguments forwarded to the underlying filesystem.


    Yields:
        `tuple`: 3-tuple (dirpath, dirnames, filenames).
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        yield from os.walk(main_hop, **kwargs)
    else:
        # walking inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            return []
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        for dirpath, dirnames, filenames in fs.walk(inner_path, **kwargs):
            yield "::".join([f"{protocol}://{dirpath}"] + rest_hops), dirnames, filenames


class xPath(type(Path())):
    """Extension of `pathlib.Path` to support both local paths and remote URLs."""

    def __str__(self):
        path_str = super().__str__()
        main_hop, *rest_hops = path_str.split("::")
        if is_local_path(main_hop):
            return main_hop
        path_as_posix = path_str.replace("\\", "/")
        path_as_posix = SINGLE_SLASH_AFTER_PROTOCOL_PATTERN.sub("://", path_as_posix)
        path_as_posix += "//" if path_as_posix.endswith(":") else ""  # Add slashes to root of the protocol
        return path_as_posix

    def exists(self, download_config: Optional[DownloadConfig] = None):
        """Extend `pathlib.Path.exists` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xexists(str(self), download_config=download_config)

    def is_file(self, download_config: Optional[DownloadConfig] = None):
        """Extend `pathlib.Path.is_file` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xisfile(str(self), download_config=download_config)

    def is_dir(self, download_config: Optional[DownloadConfig] = None):
        """Extend `pathlib.Path.is_dir` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xisdir(str(self), download_config=download_config)

    def glob(self, pattern, download_config: Optional[DownloadConfig] = None):
        """Glob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Yields:
            [`xPath`]
        """
        posix_path = self.as_posix()
        main_hop, *rest_hops = posix_path.split("::")
        if is_local_path(main_hop):
            yield from Path(main_hop).glob(pattern)
        else:
            # globbing inside a zip in a private repo requires authentication
            if rest_hops:
                urlpath = rest_hops[0]
                urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
                storage_options = {urlpath.split("://")[0]: storage_options}
                posix_path = "::".join([main_hop, urlpath, *rest_hops[1:]])
            else:
                storage_options = None
            fs, *_ = url_to_fs(xjoin(posix_path, pattern), **(storage_options or {}))
            globbed_paths = fs.glob(xjoin(main_hop, pattern))
            for globbed_path in globbed_paths:
                yield type(self)("::".join([f"{fs.protocol}://{globbed_path}"] + rest_hops))

    def rglob(self, pattern, **kwargs):
        """Rglob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.

        Yields:
            [`xPath`]
        """
        return self.glob("**/" + pattern, **kwargs)

    @property
    def parent(self) -> "xPath":
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            [`xPath`]
        """
        return type(self)(xdirname(self.as_posix()))

    @property
    def name(self) -> str:
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).name

    @property
    def stem(self) -> str:
        """Stem function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).stem

    @property
    def suffix(self) -> str:
        """Suffix function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).suffix

    def open(self, *args, **kwargs):
        """Extend :func:`xopen` to support argument of type :obj:`~pathlib.Path`.

        Args:
            **args: Arguments passed to :func:`fsspec.open`.
            **kwargs: Keyword arguments passed to :func:`fsspec.open`.

        Returns:
            `io.FileIO`: File-like object.
        """
        return xopen(str(self), *args, **kwargs)

    def joinpath(self, *p: tuple[str, ...]) -> "xPath":
        """Extend :func:`xjoin` to support argument of type :obj:`~pathlib.Path`.

        Args:
            *p (`tuple` of `str`): Other path components.

        Returns:
            [`xPath`]
        """
        return type(self)(xjoin(self.as_posix(), *p))

    def __truediv__(self, p: str) -> "xPath":
        return self.joinpath(p)

    def with_suffix(self, suffix):
        main_hop, *rest_hops = str(self).split("::")
        if is_local_path(main_hop):
            return type(self)(str(super().with_suffix(suffix)))
        return type(self)("::".join([type(self)(PurePosixPath(main_hop).with_suffix(suffix)).as_posix()] + rest_hops))


def _as_str(path: Union[str, Path, xPath]):
    return str(path) if isinstance(path, xPath) else str(xPath(str(path)))


def xgzip_open(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    import gzip

    if hasattr(filepath_or_buffer, "read"):
        return gzip.open(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return gzip.open(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)


def xnumpy_load(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    import numpy as np

    if hasattr(filepath_or_buffer, "read"):
        return np.load(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return np.load(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)


def xpandas_read_csv(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pandas as pd

    if hasattr(filepath_or_buffer, "read"):
        return pd.read_csv(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        if kwargs.get("compression", "infer") == "infer":
            kwargs["compression"] = _get_extraction_protocol(filepath_or_buffer, download_config=download_config)
        return pd.read_csv(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)


def xpandas_read_excel(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pandas as pd

    if hasattr(filepath_or_buffer, "read"):
        try:
            return pd.read_excel(filepath_or_buffer, **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(BytesIO(filepath_or_buffer.read()), **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        try:
            return pd.read_excel(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(
                BytesIO(xopen(filepath_or_buffer, "rb", download_config=download_config).read()), **kwargs
            )


def xpyarrow_parquet_read_table(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pyarrow.parquet as pq

    if hasattr(filepath_or_buffer, "read"):
        return pq.read_table(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return pq.read_table(xopen(filepath_or_buffer, mode="rb", download_config=download_config), **kwargs)


def xsio_loadmat(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import scipy.io as sio

    if hasattr(filepath_or_buffer, "read"):
        return sio.loadmat(filepath_or_buffer, **kwargs)
    else:
        return sio.loadmat(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)


def xet_parse(source, parser=None, download_config: Optional[DownloadConfig] = None):
    """Extend `xml.etree.ElementTree.parse` function to support remote files.

    Args:
        source: File path or file object.
        parser (`XMLParser`, *optional*, default `XMLParser`): Parser instance.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `xml.etree.ElementTree.Element`: Root element of the given source document.
    """
    if hasattr(source, "read"):
        return ET.parse(source, parser=parser)
    else:
        with xopen(source, "rb", download_config=download_config) as f:
            return ET.parse(f, parser=parser)


def xxml_dom_minidom_parse(filename_or_file, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `xml.dom.minidom.parse` function to support remote files.

    Args:
        filename_or_file (`str` or file): File path or file object.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs (optional): Additional keyword arguments passed to `xml.dom.minidom.parse`.

    Returns:
        :obj:`xml.dom.minidom.Document`: Parsed document.
    """
    if hasattr(filename_or_file, "read"):
        return xml.dom.minidom.parse(filename_or_file, **kwargs)
    else:
        with xopen(filename_or_file, "rb", download_config=download_config) as f:
            return xml.dom.minidom.parse(f, **kwargs)


class ArchiveIterable(TrackedIterableFromGenerator):
    """An iterable of (path, fileobj) from a TAR archive, used by `iter_archive`"""

    @staticmethod
    def _iter_tar(f):
        stream = tarfile.open(fileobj=f, mode="r|*")
        for tarinfo in stream:
            file_path = tarinfo.name
            if not tarinfo.isreg():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = stream.extractfile(tarinfo)
            yield file_path, file_obj
            stream.members = []
        del stream

    @staticmethod
    def _iter_zip(f):
        zipf = zipfile.ZipFile(f)
        for member in zipf.infolist():
            file_path = member.filename
            if member.is_dir():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = zipf.open(member)
            yield file_path, file_obj

    @classmethod
    def _iter_from_fileobj(cls, f) -> Generator[tuple, None, None]:
        compression = _get_extraction_protocol_with_magic_number(f)
        if compression == "zip":
            yield from cls._iter_zip(f)
        else:
            yield from cls._iter_tar(f)

    @classmethod
    def _iter_from_urlpath(
        cls, urlpath: str, download_config: Optional[DownloadConfig] = None
    ) -> Generator[tuple, None, None]:
        compression = _get_extraction_protocol(urlpath, download_config=download_config)
        # Set block_size=0 to get faster streaming
        # (e.g. for hf:// and https:// it uses streaming Requests file-like instances)
        with xopen(urlpath, "rb", download_config=download_config, block_size=0) as f:
            if compression == "zip":
                yield from cls._iter_zip(f)
            else:
                yield from cls._iter_tar(f)

    @classmethod
    def from_buf(cls, fileobj) -> "ArchiveIterable":
        return cls(cls._iter_from_fileobj, fileobj)

    @classmethod
    def from_urlpath(cls, urlpath_or_buf, download_config: Optional[DownloadConfig] = None) -> "ArchiveIterable":
        return cls(cls._iter_from_urlpath, urlpath_or_buf, download_config)


class FilesIterable(TrackedIterableFromGenerator):
    """An iterable of paths from a list of directories or files"""

    @classmethod
    def _iter_from_urlpaths(
        cls, urlpaths: Union[str, list[str]], download_config: Optional[DownloadConfig] = None
    ) -> Generator[str, None, None]:
        if not isinstance(urlpaths, list):
            urlpaths = [urlpaths]
        for urlpath in urlpaths:
            if xisfile(urlpath, download_config=download_config):
                yield urlpath
            elif xisdir(urlpath, download_config=download_config):
                for dirpath, dirnames, filenames in xwalk(urlpath, download_config=download_config):
                    # in-place modification to prune the search
                    dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith((".", "__"))])
                    if xbasename(dirpath).startswith((".", "__")):
                        # skipping hidden directories
                        continue
                    for filename in sorted(filenames):
                        if filename.startswith((".", "__")):
                            # skipping hidden files
                            continue
                        yield xjoin(dirpath, filename)
            else:
                raise FileNotFoundError(urlpath)

    @classmethod
    def from_urlpaths(cls, urlpaths, download_config: Optional[DownloadConfig] = None) -> "FilesIterable":
        return cls(cls._iter_from_urlpaths, urlpaths, download_config)
<|endoftext|>import numpy as np


def _number_of_shards_in_gen_kwargs(gen_kwargs: dict) -> int:
    """Return the number of possible shards according to the input gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    # until we decide how to define sharding without ambiguity for users
    lists_lengths = {key: len(value) for key, value in gen_kwargs.items() if isinstance(value, list)}
    if len(set(lists_lengths.values())) > 1:
        raise RuntimeError(
            "Sharding is ambiguous for this dataset: "
            + "we found several data sources lists of different lengths, and we don't know over which list we should parallelize:\n"
            + "\n".join(f"\t- key {key} has length {length}" for key, length in lists_lengths.items())
            + "\nTo fix this, check the 'gen_kwargs' and make sure to use lists only for data sources, "
            + "and use tuples otherwise. In the end there should only be one single list, or several lists with the same length."
        )
    max_length = max(lists_lengths.values(), default=0)
    return max(1, max_length)


def _distribute_shards(num_shards: int, max_num_jobs: int) -> list[range]:
    """
    Get the range of shard indices per job.
    If num_shards<max_num_jobs, then num_shards jobs are given a range of one shard.
    The shards indices order is preserved: e.g. all the first shards are given the first job.
    Moreover all the jobs are given approximately the same number of shards.

    Example:

    ```python
    >>> _distribute_shards(2, max_num_jobs=4)
    [range(0, 1), range(1, 2)]
    >>> _distribute_shards(10, max_num_jobs=3)
    [range(0, 4), range(4, 7), range(7, 10)]
    ```
    """
    shards_indices_per_group = []
    for group_idx in range(max_num_jobs):
        num_shards_to_add = num_shards // max_num_jobs + (group_idx < (num_shards % max_num_jobs))
        if num_shards_to_add == 0:
            break
        start = shards_indices_per_group[-1].stop if shards_indices_per_group else 0
        shard_indices = range(start, start + num_shards_to_add)
        shards_indices_per_group.append(shard_indices)
    return shards_indices_per_group


def _split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int) -> list[dict]:
    """Split the gen_kwargs into `max_num_job` gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
    if num_shards == 1:
        return [dict(gen_kwargs)]
    else:
        shard_indices_per_group = _distribute_shards(num_shards=num_shards, max_num_jobs=max_num_jobs)
        return [
            {
                key: [value[shard_idx] for shard_idx in shard_indices_per_group[group_idx]]
                if isinstance(value, list)
                else value
                for key, value in gen_kwargs.items()
            }
            for group_idx in range(len(shard_indices_per_group))
        ]


def _merge_gen_kwargs(gen_kwargs_list: list[dict]) -> dict:
    return {
        key: [value for gen_kwargs in gen_kwargs_list for value in gen_kwargs[key]]
        if isinstance(gen_kwargs_list[0][key], list)
        else gen_kwargs_list[0][key]
        for key in gen_kwargs_list[0]
    }


def _shuffle_gen_kwargs(rng: np.random.Generator, gen_kwargs: dict) -> dict:
    """Return a shuffled copy of the input gen_kwargs"""
    # We must shuffle all the lists, and lists of the same size must have the same shuffling.
    # This way entangled lists of (shard, shard_metadata) are still in the right order.

    # First, let's generate the shuffled indices per list size
    list_sizes = {len(value) for value in gen_kwargs.values() if isinstance(value, list)}
    indices_per_size = {}
    for size in list_sizes:
        indices_per_size[size] = list(range(size))
        rng.shuffle(indices_per_size[size])
    # Now let's copy the gen_kwargs and shuffle the lists based on their sizes
    shuffled_kwargs = dict(gen_kwargs)
    for key, value in shuffled_kwargs.items():
        if isinstance(value, list):
            shuffled_kwargs[key] = [value[i] for i in indices_per_size[len(value)]]
    return shuffled_kwargs
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

# Lint as: python3
"""Version utils."""

import dataclasses
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Optional, Union


_VERSION_REG = re.compile(r"^(?P<major>\d+)" r"\.(?P<minor>\d+)" r"\.(?P<patch>\d+)$")


@total_ordering
@dataclass
class Version:
    """Dataset version `MAJOR.MINOR.PATCH`.

    Args:
        version_str (`str`):
            The dataset version.
        description (`str`):
            A description of what is new in this version.
        major (`str`):
        minor (`str`):
        patch (`str`):

    Example:

    ```py
    >>> VERSION = datasets.Version("1.0.0")
    ```
    """

    version_str: str
    description: Optional[str] = None
    major: Optional[Union[str, int]] = None
    minor: Optional[Union[str, int]] = None
    patch: Optional[Union[str, int]] = None

    def __post_init__(self):
        self.major, self.minor, self.patch = _str_to_version_tuple(self.version_str)

    def __repr__(self):
        return f"{self.tuple[0]}.{self.tuple[1]}.{self.tuple[2]}"

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    def _validate_operand(self, other):
        if isinstance(other, str):
            return Version(other)
        elif isinstance(other, Version):
            return other
        raise TypeError(f"{other} (type {type(other)}) cannot be compared to version.")

    def __eq__(self, other):
        try:
            other = self._validate_operand(other)
        except (TypeError, ValueError):
            return False
        else:
            return self.tuple == other.tuple

    def __lt__(self, other):
        other = self._validate_operand(other)
        return self.tuple < other.tuple

    def __hash__(self):
        return hash(_version_tuple_to_str(self.tuple))

    @classmethod
    def from_dict(cls, dic):
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dic.items() if k in field_names})

    def _to_yaml_string(self) -> str:
        return self.version_str


def _str_to_version_tuple(version_str):
    """Return the tuple (major, minor, patch) version extracted from the str."""
    res = _VERSION_REG.match(version_str)
    if not res:
        raise ValueError(f"Invalid version '{version_str}'. Format should be x.y.z with {{x,y,z}} being digits.")
    return tuple(int(v) for v in [res.group("major"), res.group("minor"), res.group("patch")])


def _version_tuple_to_str(version_tuple):
    """Return the str version from the version tuple (major, minor, patch)."""
    return ".".join(str(v) for v in version_tuple)
<|endoftext|>from typing import Callable


def is_documented_by(function_with_docstring: Callable):
    """Decorator to share docstrings across common functions.

    Args:
        function_with_docstring (`Callable`): Name of the function with the docstring.
    """

    def wrapper(target_function):
        target_function.__doc__ = function_with_docstring.__doc__
        return target_function

    return wrapper
<|endoftext|>import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .. import config
from ._filelock import FileLock
from .logging import get_logger


if TYPE_CHECKING:
    import py7zr
    import rarfile


logger = get_logger(__name__)


class ExtractManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.extract_dir = (
            os.path.join(cache_dir, config.EXTRACTED_DATASETS_DIR) if cache_dir else config.EXTRACTED_DATASETS_PATH
        )
        self.extractor = Extractor

    def _get_output_path(self, path: str) -> str:
        from .file_utils import hash_url_to_filename

        # Path where we extract compressed archives
        # We extract in the cache dir, and get the extracted path name by hashing the original path"
        abs_path = os.path.abspath(path)
        return os.path.join(self.extract_dir, hash_url_to_filename(abs_path))

    def _do_extract(self, output_path: str, force_extract: bool) -> bool:
        return force_extract or (
            not os.path.isfile(output_path) and not (os.path.isdir(output_path) and os.listdir(output_path))
        )

    def extract(self, input_path: str, force_extract: bool = False) -> str:
        extractor_format = self.extractor.infer_extractor_format(input_path)
        if not extractor_format:
            return input_path
        output_path = self._get_output_path(input_path)
        if self._do_extract(output_path, force_extract):
            self.extractor.extract(input_path, output_path, extractor_format)
        return output_path


class BaseExtractor(ABC):
    @classmethod
    @abstractmethod
    def is_extractable(cls, path: Union[Path, str], **kwargs) -> bool: ...

    @staticmethod
    @abstractmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None: ...


class MagicNumberBaseExtractor(BaseExtractor, ABC):
    magic_numbers: list[bytes] = []

    @staticmethod
    def read_magic_number(path: Union[Path, str], magic_number_length: int):
        with open(path, "rb") as f:
            return f.read(magic_number_length)

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes = b"") -> bool:
        if not magic_number:
            magic_number_length = max(len(cls_magic_number) for cls_magic_number in cls.magic_numbers)
            try:
                magic_number = cls.read_magic_number(path, magic_number_length)
            except OSError:
                return False
        return any(magic_number.startswith(cls_magic_number) for cls_magic_number in cls.magic_numbers)


class TarExtractor(BaseExtractor):
    @classmethod
    def is_extractable(cls, path: Union[Path, str], **kwargs) -> bool:
        return tarfile.is_tarfile(path)

    @staticmethod
    def safemembers(members: tarfile.TarFile, output_path: Union[Path, str]):
        """
        Fix for CVE-2007-4559
        Desc:
            Directory traversal vulnerability in the (1) extract and (2) extractall functions in the tarfile
            module in Python allows user-assisted remote attackers to overwrite arbitrary files via a .. (dot dot)
            sequence in filenames in a TAR archive, a related issue to CVE-2001-1267.
        See: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-4559
        From: https://stackoverflow.com/a/10077309
        """

        def resolved(path: Union[Path, str]) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            # joinpath will ignore base if path is absolute
            # Also, a plain startswith(base) would still match sibling directories
            # whose name starts with the base name (e.g. "../base_evil/x" when
            # extracting into ".../base"), so require the base plus a separator
            target = resolved(os.path.join(base, path))
            return target != base and not target.startswith(base + os.sep)

        def badlink(info: tarfile.TarInfo, base: str) -> bool:
            # Links are interpreted relative to the directory containing the link
            tip = resolved(os.path.join(base, os.path.dirname(info.name)))
            return badpath(info.linkname, base=tip)

        base = resolved(output_path)

        for finfo in members:
            if badpath(finfo.name, base):
                logger.error(f"Extraction of {finfo.name} is blocked (illegal path)")
            elif finfo.issym() and badlink(finfo, base):
                logger.error(f"Extraction of {finfo.name} is blocked: Symlink to {finfo.linkname}")
            elif finfo.islnk() and badlink(finfo, base):
                logger.error(f"Extraction of {finfo.name} is blocked: Hard link to {finfo.linkname}")
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        tar_file = tarfile.open(input_path)
        tar_file.extractall(output_path, members=TarExtractor.safemembers(tar_file, output_path))
        tar_file.close()


class GzipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x1f\x8b"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with gzip.open(input_path, "rb") as gzip_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(gzip_file, extracted_file)


class ZipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [
        b"PK\x03\x04",
        b"PK\x05\x06",  # empty archive
        b"PK\x07\x08",  # spanned archive
    ]

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes = b"") -> bool:
        if super().is_extractable(path, magic_number=magic_number):
            return True
        try:
            # Alternative version of zipfile.is_zipfile that has less false positives, but misses executable zip archives.
            # From: https://github.com/python/cpython/pull/5053
            from zipfile import (
                _CD_SIGNATURE,
                _ECD_DISK_NUMBER,
                _ECD_DISK_START,
                _ECD_ENTRIES_TOTAL,
                _ECD_OFFSET,
                _ECD_SIZE,
                _EndRecData,
                sizeCentralDir,
                stringCentralDir,
                structCentralDir,
            )

            with open(path, "rb") as fp:
                endrec = _EndRecData(fp)
                if endrec:
                    if endrec[_ECD_ENTRIES_TOTAL] == 0 and endrec[_ECD_SIZE] == 0 and endrec[_ECD_OFFSET] == 0:
                        return True  # Empty zipfiles are still zipfiles
                    elif endrec[_ECD_DISK_NUMBER] == endrec[_ECD_DISK_START]:
                        fp.seek(endrec[_ECD_OFFSET])  # Central directory is on the same disk
                        if fp.tell() == endrec[_ECD_OFFSET] and endrec[_ECD_SIZE] >= sizeCentralDir:
                            data = fp.read(sizeCentralDir)  # CD is where we expect it to be
                            if len(data) == sizeCentralDir:
                                centdir = struct.unpack(structCentralDir, data)  # CD is the right size
                                if centdir[_CD_SIGNATURE] == stringCentralDir:
                                    return True  # First central directory entry  has correct magic number
            return False
        except Exception:  # catch all errors in case future python versions change the zipfile internals
            return False

    @staticmethod
    def safemembers(members: list[zipfile.ZipInfo], output_path: Union[Path, str]):
        """
        Fix for CVE-2007-4559
        Desc:
            Directory traversal vulnerability in the (1) extract and (2) extractall functions in the tarfile
            module in Python allows user-assisted remote attackers to overwrite arbitrary files via a .. (dot dot)
            sequence in filenames in a TAR archive, a related issue to CVE-2001-1267.
        See: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-4559
        From: https://stackoverflow.com/a/10077309

        This additional mitigation is applied for zipfile as well.
        """

        def resolved(path: Union[Path, str]) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            # joinpath will ignore base if path is absolute
            # Also, a plain startswith(base) would still match sibling directories
            # whose name starts with the base name (e.g. "../base_evil/x" when
            # extracting into ".../base"), so require the base plus a separator
            target = resolved(os.path.join(base, path))
            return target != base and not target.startswith(base + os.sep)

        base = resolved(output_path)

        for finfo in members:
            if badpath(finfo.filename, base):
                logger.error(f"Extraction of {finfo.filename} is blocked (illegal path)")
            # zipfile doesn't support symlinks
            # elif finfo.is_symlink and badlink(finfo, base):
            #     logger.error(f"Extraction of {finfo.name} is blocked: Symlink to {finfo.linkname}")
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        os.makedirs(output_path, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zip_file:
            zip_file.extractall(output_path, members=ZipExtractor.safemembers(zip_file.filelist, output_path))
            zip_file.close()


class XzExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\xfd\x37\x7a\x58\x5a\x00"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with lzma.open(input_path) as compressed_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)


class RarExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00"]  # RAR_ID  # RAR5_ID

    @staticmethod
    def safemembers(members: list["rarfile.RarInfo"], output_path: Union[Path, str]):
        """
        Fix for CVE-2007-4559
        Desc:
            Directory traversal vulnerability in the (1) extract and (2) extractall functions in the tarfile
            module in Python allows user-assisted remote attackers to overwrite arbitrary files via a .. (dot dot)
            sequence in filenames in a TAR archive, a related issue to CVE-2001-1267.
        See: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-4559
        From: https://stackoverflow.com/a/10077309

        This additional mitigation is applied for rarfile as well.
        """

        def resolved(path: Union[Path, str]) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            # joinpath will ignore base if path is absolute
            # Also, a plain startswith(base) would still match sibling directories
            # whose name starts with the base name (e.g. "../base_evil/x" when
            # extracting into ".../base"), so require the base plus a separator
            target = resolved(os.path.join(base, path))
            return target != base and not target.startswith(base + os.sep)

        def badlink(info: "rarfile.RarInfo", base: str) -> bool:
            # Links are interpreted relative to the directory containing the link
            tip = resolved(os.path.join(base, os.path.dirname(info.filename)))
            redir_type, redir_flags, link_name = info.file_redir
            return badpath(link_name, base=tip)

        base = resolved(output_path)

        for finfo in members:
            if badpath(finfo.filename, base):
                logger.error(f"Extraction of {finfo.filename} is blocked (illegal path)")
            elif finfo.is_symlink() and badlink(finfo, base):
                logger.error(f"Extraction of {finfo.filename} is blocked: Symlink to {finfo.file_redir}")
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.RARFILE_AVAILABLE:
            raise ImportError("Please pip install rarfile")
        import rarfile

        os.makedirs(output_path, exist_ok=True)
        rf = rarfile.RarFile(input_path)
        rf.extractall(output_path, members=RarExtractor.safemembers(rf.infolist(), output_path))
        rf.close()


class ZstdExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x28\xb5\x2f\xfd"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.ZSTANDARD_AVAILABLE:
            raise ImportError("Please pip install zstandard")
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with open(input_path, "rb") as ifh, open(output_path, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)


class Bzip2Extractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x42\x5a\x68"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with bz2.open(input_path, "rb") as compressed_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)


class SevenZipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x37\x7a\xbc\xaf\x27\x1c"]

    @staticmethod
    def safemembers(members: list["py7zr.FileInfo"], output_path: Union[Path, str]):
        """
        Fix for CVE-2007-4559
        Desc:
            Directory traversal vulnerability in the (1) extract and (2) extractall functions in the tarfile
            module in Python allows user-assisted remote attackers to overwrite arbitrary files via a .. (dot dot)
            sequence in filenames in a TAR archive, a related issue to CVE-2001-1267.
        See: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-4559
        From: https://stackoverflow.com/a/10077309

        This additional mitigation is applied for py7zr as well.
        """

        def resolved(path: Union[Path, str]) -> str:
            return os.path.realpath(os.path.abspath(path))

        def badpath(path: str, base: str) -> bool:
            # joinpath will ignore base if path is absolute
            # Also, a plain startswith(base) would still match sibling directories
            # whose name starts with the base name (e.g. "../base_evil/x" when
            # extracting into ".../base"), so require the base plus a separator
            target = resolved(os.path.join(base, path))
            return target != base and not target.startswith(base + os.sep)

        def badlink(info: "py7zr.FileInfo", base: str) -> bool:
            # Links are interpreted relative to the directory containing the link
            tip = resolved(os.path.join(base, os.path.dirname(info.filename)))
            return badpath(os.path.basename(info.filename), base=tip)

        base = resolved(output_path)

        for finfo in members:
            if badpath(finfo.filename, base):
                logger.error(f"Extraction of {finfo.filename} is blocked (illegal path)")
            # py7zr already checks symlinks validity
            # elif finfo.is_symlink and badlink(finfo, base):
            #     logger.error(f"Extraction of {finfo.name} is blocked: Symlink to {finfo.linkname}")
            else:
                yield finfo

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.PY7ZR_AVAILABLE:
            raise ImportError("Please pip install py7zr")
        import py7zr

        os.makedirs(output_path, exist_ok=True)
        with py7zr.SevenZipFile(input_path, "r") as archive:
            targets = [finfo.filename for finfo in SevenZipExtractor.safemembers(archive.list(), output_path)]
            archive.extract(output_path, targets=targets)


class Lz4Extractor(MagicNumberBaseExtractor):
    magic_numbers = [b"\x04\x22\x4d\x18"]

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        if not config.LZ4_AVAILABLE:
            raise ImportError("Please pip install lz4")
        import lz4.frame

        with lz4.frame.open(input_path, "rb") as compressed_file:
            with open(output_path, "wb") as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)


class Extractor:
    #  Put zip file to the last, b/c it is possible wrongly detected as zip (I guess it means: as tar or gzip)
    extractors: dict[str, type[BaseExtractor]] = {
        "tar": TarExtractor,
        "gzip": GzipExtractor,
        "zip": ZipExtractor,
        "xz": XzExtractor,
        "rar": RarExtractor,
        "zstd": ZstdExtractor,
        "bz2": Bzip2Extractor,
        "7z": SevenZipExtractor,  # <Added version="2.4.0"/>
        "lz4": Lz4Extractor,  # <Added version="2.4.0"/>
    }

    @classmethod
    def _get_magic_number_max_length(cls):
        return max(
            len(extractor_magic_number)
            for extractor in cls.extractors.values()
            if issubclass(extractor, MagicNumberBaseExtractor)
            for extractor_magic_number in extractor.magic_numbers
        )

    @staticmethod
    def _read_magic_number(path: Union[Path, str], magic_number_length: int):
        try:
            return MagicNumberBaseExtractor.read_magic_number(path, magic_number_length=magic_number_length)
        except OSError:
            return b""

    @classmethod
    def is_extractable(cls, path: Union[Path, str], return_extractor: bool = False) -> bool:
        warnings.warn(
            "Method 'is_extractable' was deprecated in version 2.4.0 and will be removed in 3.0.0. "
            "Use 'infer_extractor_format' instead.",
            category=FutureWarning,
        )
        extractor_format = cls.infer_extractor_format(path)
        if extractor_format:
            return True if not return_extractor else (True, cls.extractors[extractor_format])
        return False if not return_extractor else (False, None)

    @classmethod
    def infer_extractor_format(cls, path: Union[Path, str]) -> Optional[str]:  # <Added version="2.4.0"/>
        magic_number_max_length = cls._get_magic_number_max_length()
        magic_number = cls._read_magic_number(path, magic_number_max_length)
        for extractor_format, extractor in cls.extractors.items():
            if extractor.is_extractable(path, magic_number=magic_number):
                return extractor_format

    @classmethod
    def extract(
        cls,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        extractor_format: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Prevent parallel extractions
        lock_path = str(Path(output_path).with_suffix(".lock"))
        with FileLock(lock_path):
            if os.path.islink(output_path):
                os.unlink(output_path)
            shutil.rmtree(output_path, ignore_errors=True)
            extractor = cls.extractors[extractor_format]
            return extractor.extract(input_path, output_path)
<|endoftext|>import enum
import os
from typing import Optional

from huggingface_hub.utils import insecure_hashlib

from .. import config
from ..exceptions import (
    ExpectedMoreDownloadedFilesError,
    ExpectedMoreSplitsError,
    NonMatchingChecksumError,
    NonMatchingSplitsSizesError,
    UnexpectedDownloadedFileError,
    UnexpectedSplitsError,
)
from .logging import get_logger


logger = get_logger(__name__)


class VerificationMode(enum.Enum):
    """`Enum` that specifies which verification checks to run.

    The default mode is `BASIC_CHECKS`, which will perform only rudimentary checks to avoid slowdowns
    when generating/downloading a dataset for the first time.

    The verification modes:

    |                           | Verification checks                                                           |
    |---------------------------|------------------------------------------------------------------------------ |
    | `ALL_CHECKS`              | Split checks and validity (number of files, checksums) of downloaded files    |
    | `BASIC_CHECKS` (default)  | Same as `ALL_CHECKS` but without checking downloaded files                    |
    | `NO_CHECKS`               | None                                                                          |

    """

    ALL_CHECKS = "all_checks"
    BASIC_CHECKS = "basic_checks"
    NO_CHECKS = "no_checks"


def verify_checksums(expected_checksums: Optional[dict], recorded_checksums: dict, verification_name=None):
    if expected_checksums is None:
        logger.info("Unable to verify checksums.")
        return
    if len(set(expected_checksums) - set(recorded_checksums)) > 0:
        raise ExpectedMoreDownloadedFilesError(str(set(expected_checksums) - set(recorded_checksums)))
    if len(set(recorded_checksums) - set(expected_checksums)) > 0:
        raise UnexpectedDownloadedFileError(str(set(recorded_checksums) - set(expected_checksums)))
    bad_urls = [url for url in expected_checksums if expected_checksums[url] != recorded_checksums[url]]
    for_verification_name = " for " + verification_name if verification_name is not None else ""
    if len(bad_urls) > 0:
        raise NonMatchingChecksumError(
            f"Checksums didn't match{for_verification_name}:\n"
            f"{bad_urls}\n"
            "Set `verification_mode='no_checks'` to skip checksums verification and ignore this error"
        )
    logger.info("All the checksums matched successfully" + for_verification_name)


def verify_splits(expected_splits: Optional[dict], recorded_splits: dict):
    if expected_splits is None:
        logger.info("Unable to verify splits sizes.")
        return
    if len(set(expected_splits) - set(recorded_splits)) > 0:
        raise ExpectedMoreSplitsError(str(set(expected_splits) - set(recorded_splits)))
    if len(set(recorded_splits) - set(expected_splits)) > 0:
        raise UnexpectedSplitsError(str(set(recorded_splits) - set(expected_splits)))
    bad_splits = [
        {"expected": expected_splits[name], "recorded": recorded_splits[name]}
        for name in expected_splits
        if expected_splits[name].num_examples != recorded_splits[name].num_examples
    ]
    if len(bad_splits) > 0:
