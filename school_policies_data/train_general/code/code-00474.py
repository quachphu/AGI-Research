def extend_module_for_streaming(module_path, download_config: Optional[DownloadConfig] = None):
    """Extend the module to support streaming.

    We patch some functions in the module to use `fsspec` to support data streaming:
    - We use `fsspec.open` to open and read remote files. We patch the module function:
      - `open`
    - We use the "::" hop separator to join paths and navigate remote compressed/archive files. We patch the module
      functions:
      - `os.path.join`
      - `pathlib.Path.joinpath` and `pathlib.Path.__truediv__` (called when using the "/" operator)

    The patched functions are replaced with custom functions defined to work with the
    :class:`~download.streaming_download_manager.StreamingDownloadManager`.

    Args:
        module_path: Path to the module to be extended.
        download_config: Mainly use `token` or `storage_options` to support different platforms and auth types.
    """

    module = importlib.import_module(module_path)

    # TODO(QL): always update the module to add subsequent new authentication without removing old ones
    if hasattr(module, "_patched_for_streaming") and module._patched_for_streaming:
        if isinstance(module._patched_for_streaming, DownloadConfig):
            module._patched_for_streaming.token = download_config.token
            module._patched_for_streaming.storage_options = download_config.storage_options
        return

    def wrap_auth(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, download_config=download_config, **kwargs)

        wrapper._decorator_name_ = "wrap_auth"
        return wrapper

    # open files in a streaming fashion
    patch_submodule(module, "open", wrap_auth(xopen)).start()
    patch_submodule(module, "os.listdir", wrap_auth(xlistdir)).start()
    patch_submodule(module, "os.walk", wrap_auth(xwalk)).start()
    patch_submodule(module, "glob.glob", wrap_auth(xglob)).start()
    # allow to navigate in remote zip files
    patch_submodule(module, "os.path.join", xjoin).start()
    patch_submodule(module, "os.path.dirname", xdirname).start()
    patch_submodule(module, "os.path.basename", xbasename).start()
    patch_submodule(module, "os.path.relpath", xrelpath).start()
    patch_submodule(module, "os.path.split", xsplit).start()
    patch_submodule(module, "os.path.splitext", xsplitext).start()
    # allow checks on paths
    patch_submodule(module, "os.path.exists", wrap_auth(xexists)).start()
    patch_submodule(module, "os.path.isdir", wrap_auth(xisdir)).start()
    patch_submodule(module, "os.path.isfile", wrap_auth(xisfile)).start()
    patch_submodule(module, "os.path.getsize", wrap_auth(xgetsize)).start()
    patch_submodule(module, "pathlib.Path", xPath).start()
    # file readers
    patch_submodule(module, "gzip.open", wrap_auth(xgzip_open)).start()
    patch_submodule(module, "numpy.load", wrap_auth(xnumpy_load)).start()
    patch_submodule(module, "pandas.read_csv", wrap_auth(xpandas_read_csv), attrs=["__version__"]).start()
    patch_submodule(module, "pandas.read_excel", wrap_auth(xpandas_read_excel), attrs=["__version__"]).start()
    patch_submodule(module, "scipy.io.loadmat", wrap_auth(xsio_loadmat), attrs=["__version__"]).start()
    patch_submodule(module, "xml.etree.ElementTree.parse", wrap_auth(xet_parse)).start()
    patch_submodule(module, "xml.dom.minidom.parse", wrap_auth(xxml_dom_minidom_parse)).start()
    # pyarrow: do not patch pyarrow attribute in packaged modules
    if not module.__name__.startswith("datasets.packaged_modules."):
        patch_submodule(module, "pyarrow.parquet.read_table", wrap_auth(xpyarrow_parquet_read_table)).start()
    module._patched_for_streaming = download_config


def extend_dataset_builder_for_streaming(builder: "DatasetBuilder"):
    """Extend the dataset builder module and the modules imported by it to support streaming.

    Args:
        builder (:class:`DatasetBuilder`): Dataset builder instance.
    """
    # this extends the open and os.path.join functions for data streaming
    download_config = DownloadConfig(storage_options=builder.storage_options, token=builder.token)
    extend_module_for_streaming(builder.__module__, download_config=download_config)

    # builders can inherit from other builders that might use streaming functionality
    # (for example, ImageFolder and AudioFolder inherit from FolderBuilder which implements examples generation)
    # but these parents builders are not patched automatically as they are not instantiated, so we patch them here
    from .builder import DatasetBuilder

    parent_builder_modules = [
        cls.__module__
        for cls in type(builder).__mro__[1:]  # make sure it's not the same module we've already patched
        if issubclass(cls, DatasetBuilder) and cls.__module__ != DatasetBuilder.__module__
    ]  # check it's not a standard builder from datasets.builder
    for module in parent_builder_modules:
        extend_module_for_streaming(module, download_config=download_config)
<|endoftext|>import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import fsspec
import numpy as np

from .features import List
from .utils import logging
from .utils import tqdm as hf_tqdm


if TYPE_CHECKING:
    from .arrow_dataset import Dataset  # noqa: F401

    try:
        from elasticsearch import Elasticsearch  # noqa: F401

    except ImportError:
        pass
    try:
        import faiss  # noqa: F401

    except ImportError:
        pass

_has_elasticsearch = importlib.util.find_spec("elasticsearch") is not None
_has_faiss = importlib.util.find_spec("faiss") is not None


logger = logging.get_logger(__name__)


class MissingIndex(Exception):
    pass


class SearchResults(NamedTuple):
    scores: list[float]
    indices: list[int]


class BatchedSearchResults(NamedTuple):
    total_scores: list[list[float]]
    total_indices: list[list[int]]


class NearestExamplesResults(NamedTuple):
    scores: list[float]
    examples: dict


class BatchedNearestExamplesResults(NamedTuple):
    total_scores: list[list[float]]
    total_examples: list[dict]


class BaseIndex:
    """Base class for indexing"""

    def search(self, query, k: int = 10, **kwargs) -> SearchResults:
        """
        To implement.
        This method has to return the scores and the indices of the retrieved examples given a certain query.
        """
        raise NotImplementedError

    def search_batch(self, queries, k: int = 10, **kwargs) -> BatchedSearchResults:
        """Find the nearest examples indices to the query.

        Args:
            queries (`Union[List[str], np.ndarray]`): The queries as a list of strings if `column` is a text index or as a numpy array if `column` is a vector index.
            k (`int`): The number of examples to retrieve per query.

        Output:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.search(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return BatchedSearchResults(total_scores, total_indices)

    def save(self, file: Union[str, PurePath]):
        """Serialize the index on disk"""
        raise NotImplementedError

    @classmethod
    def load(cls, file: Union[str, PurePath]) -> "BaseIndex":
        """Deserialize the index from disk"""
        raise NotImplementedError


class ElasticSearchIndex(BaseIndex):
    """
    Sparse index using Elasticsearch. It is used to index text and run queries based on BM25 similarity.
    An Elasticsearch server needs to be accessible, and a python client is declared with
    ```
    es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    ```
    for example.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_name: Optional[str] = None,
        es_index_config: Optional[dict] = None,
    ):
        if not _has_elasticsearch:
            raise ImportError(
                "You must install ElasticSearch to use ElasticSearchIndex. To do so you can run `pip install elasticsearch==7.7.1 for example`"
            )
        if es_client is not None and (host is not None or port is not None):
            raise ValueError("Please specify either `es_client` or `(host, port)`, but not both.")
        host = host or "localhost"
        port = port or 9200

        import elasticsearch.helpers  # noqa: F401 - need this to properly load all the es features
        from elasticsearch import Elasticsearch  # noqa: F811

        self.es_client = es_client if es_client is not None else Elasticsearch([{"host": host, "port": str(port)}])
        self.es_index_name = (
            es_index_name
            if es_index_name is not None
            else "huggingface_datasets_" + os.path.basename(tempfile.NamedTemporaryFile().name)
        )
        self.es_index_config = (
            es_index_config
            if es_index_config is not None
            else {
                "settings": {
                    "number_of_shards": 1,
                    "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                },
                "mappings": {"properties": {"text": {"type": "text", "analyzer": "standard", "similarity": "BM25"}}},
            }
        )

    def add_documents(self, documents: Union[list[str], "Dataset"], column: Optional[str] = None):
        """
        Add documents to the index.
        If the documents are inside a certain column, you can specify it using the `column` argument.
        """
        index_name = self.es_index_name
        index_config = self.es_index_config
        self.es_client.indices.create(index=index_name, body=index_config)
        number_of_docs = len(documents)
        progress = hf_tqdm(unit="docs", total=number_of_docs)
        successes = 0

        def passage_generator():
            if column is not None:
                for i, example in enumerate(documents):
                    yield {"text": example[column], "_id": i}
            else:
                for i, example in enumerate(documents):
                    yield {"text": example, "_id": i}

        # create the ES index
        import elasticsearch as es

        for ok, action in es.helpers.streaming_bulk(
            client=self.es_client,
            index=index_name,
            actions=passage_generator(),
        ):
            progress.update(1)
            successes += ok
        if successes != len(documents):
            logger.warning(
                f"Some documents failed to be added to ElasticSearch. Failures: {len(documents) - successes}/{len(documents)}"
            )
        logger.info(f"Indexed {successes:d} documents")

    def search(self, query: str, k=10, **kwargs) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`str`): The query as a string.
            k (`int`): The number of examples to retrieve.

        Output:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        response = self.es_client.search(
            index=self.es_index_name,
            body={"query": {"multi_match": {"query": query, "fields": ["text"], "type": "cross_fields"}}, "size": k},
            **kwargs,
        )
        hits = response["hits"]["hits"]
        return SearchResults([hit["_score"] for hit in hits], [int(hit["_id"]) for hit in hits])

    def search_batch(self, queries, k: int = 10, max_workers=10, **kwargs) -> BatchedSearchResults:
        import concurrent.futures

        total_scores, total_indices = [None] * len(queries), [None] * len(queries)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.search, query, k, **kwargs): i for i, query in enumerate(queries)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results: SearchResults = future.result()
                total_scores[index] = results.scores
                total_indices[index] = results.indices
        return BatchedSearchResults(total_indices=total_indices, total_scores=total_scores)


class FaissIndex(BaseIndex):
    """
    Dense index using Faiss. It is used to index vectors.
    Faiss is a library for efficient similarity search and clustering of dense vectors.
    It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
    You can find more information about Faiss here:
    - For index types and the string factory: https://github.com/facebookresearch/faiss/wiki/The-index-factory
    - For GPU settings: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
    """

    def __init__(
        self,
        device: Optional[Union[int, list[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
    ):
        """
        Create a Dense index using Faiss. You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory
        """
        if string_factory is not None and custom_index is not None:
            raise ValueError("Please specify either `string_factory` or `custom_index` but not both.")
        if device is not None and custom_index is not None:
            raise ValueError(
                "Cannot pass both 'custom_index' and 'device'. "
                "Pass 'custom_index' already transferred to the target device instead."
            )
        self.device = device
        self.string_factory = string_factory
        self.metric_type = metric_type
        self.faiss_index = custom_index
        if not _has_faiss:
            raise ImportError(
                "You must install Faiss to use FaissIndex. To do so you can run `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`. "
                "A community supported package is also available on pypi: `pip install faiss-cpu` or `pip install faiss-gpu`. "
                "Note that pip may not have the latest version of FAISS, and thus, some of the latest features and bug fixes may not be available."
            )

    def add_vectors(
        self,
        vectors: Union[np.array, "Dataset"],
        column: Optional[str] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: Optional[bool] = None,
    ):
        """
        Add vectors to the index.
        If the arrays are inside a certain column, you can specify it using the `column` argument.
        """
        import faiss  # noqa: F811

        if column and not isinstance(vectors.features[column], List):
            raise ValueError(
                f"Wrong feature type for column '{column}'. Expected 1d array, got {vectors.features[column]}"
            )

        # Create index
        if self.faiss_index is None:
            size = len(vectors[0]) if column is None else len(vectors[0][column])
            if self.string_factory is not None:
                if self.metric_type is None:
                    index = faiss.index_factory(size, self.string_factory)
                else:
                    index = faiss.index_factory(size, self.string_factory, self.metric_type)
            else:
                if self.metric_type is None:
                    index = faiss.IndexFlat(size)
                else:
                    index = faiss.IndexFlat(size, self.metric_type)

            self.faiss_index = self._faiss_index_to_device(index, self.device)
            logger.info(f"Created faiss index of type {type(self.faiss_index)}")

        # Set verbosity level
        if faiss_verbose is not None:
            self.faiss_index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "index") and self.faiss_index.index is not None:
                self.faiss_index.index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "quantizer") and self.faiss_index.quantizer is not None:
                self.faiss_index.quantizer.verbose = faiss_verbose
            if hasattr(self.faiss_index, "clustering_index") and self.faiss_index.clustering_index is not None:
                self.faiss_index.clustering_index.verbose = faiss_verbose

        # Train
        if train_size is not None:
            train_vecs = vectors[:train_size] if column is None else vectors[:train_size][column]
            logger.info(f"Training the index with the first {len(train_vecs)} vectors")
            self.faiss_index.train(train_vecs)
        else:
            logger.info("Ignored the training step of the faiss index as `train_size` is None.")

        # Add vectors
        logger.info(f"Adding {len(vectors)} vectors to the faiss index")
        for i in hf_tqdm(range(0, len(vectors), batch_size)):
            vecs = vectors[i : i + batch_size] if column is None else vectors[i : i + batch_size][column]
            self.faiss_index.add(vecs)

    @staticmethod
    def _faiss_index_to_device(index: "faiss.Index", device: Optional[Union[int, list[int]]] = None) -> "faiss.Index":
        """
        Sends a faiss index to a device.
        A device can either be a positive integer (GPU id), a negative integer (all GPUs),
            or a list of positive integers (select GPUs to use), or `None` for CPU.
        """

        # If device is not specified, then it runs on CPU.
        if device is None:
            return index

        import faiss  # noqa: F811

        # If the device id is given as an integer
        if isinstance(device, int):
            # Positive integers are directly mapped to GPU ids
            if device > -1:
                faiss_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(faiss_res, device, index)
            # And negative integers mean using all GPUs
            else:
                index = faiss.index_cpu_to_all_gpus(index)
        # Device ids given as a list mean mapping to those devices specified.
        elif isinstance(device, (list, tuple)):
            index = faiss.index_cpu_to_gpus_list(index, gpus=list(device))
        else:
            raise TypeError(
                f"The argument type: {type(device)} is not expected. "
                + "Please pass in either nothing, a positive int, a negative int, or a list of positive ints."
            )

        return index

    def search(self, query: np.array, k=10, **kwargs) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`np.array`): The query as a numpy array.
            k (`int`): The number of examples to retrieve.

        Output:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        if len(query.shape) != 1 and (len(query.shape) != 2 or query.shape[0] != 1):
            raise ValueError("Shape of query is incorrect, it has to be either a 1D array or 2D (1, N)")

        queries = query.reshape(1, -1)
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k, **kwargs)
        return SearchResults(scores[0], indices[0].astype(int))

    def search_batch(self, queries: np.array, k=10, **kwargs) -> BatchedSearchResults:
        """Find the nearest examples indices to the queries.

        Args:
            queries (`np.array`): The queries as a numpy array.
            k (`int`): The number of examples to retrieve.

        Output:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        if len(queries.shape) != 2:
            raise ValueError("Shape of query must be 2D")
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k, **kwargs)
        return BatchedSearchResults(scores, indices.astype(int))

    def save(self, file: Union[str, PurePath], storage_options: Optional[dict] = None):
        """Serialize the FaissIndex on disk"""
        import faiss  # noqa: F811

        if self.device is not None and isinstance(self.device, (int, list, tuple)):
            index = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            index = self.faiss_index

        with fsspec.open(str(file), "wb", **(storage_options or {})) as f:
            faiss.write_index(index, faiss.BufferedIOWriter(faiss.PyCallbackIOWriter(f.write)))

    @classmethod
    def load(
        cls,
        file: Union[str, PurePath],
        device: Optional[Union[int, list[int]]] = None,
        storage_options: Optional[dict] = None,
    ) -> "FaissIndex":
        """Deserialize the FaissIndex from disk"""
        import faiss  # noqa: F811

        # Instances of FaissIndex is essentially just a wrapper for faiss indices.
        faiss_index = cls(device=device)
        with fsspec.open(str(file), "rb", **(storage_options or {})) as f:
            index = faiss.read_index(faiss.BufferedIOReader(faiss.PyCallbackIOReader(f.read)))
        faiss_index.faiss_index = faiss_index._faiss_index_to_device(index, faiss_index.device)
        return faiss_index


class IndexableMixin:
    """Add indexing features to `datasets.Dataset`"""

    def __init__(self):
        self._indexes: dict[str, BaseIndex] = {}

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def is_index_initialized(self, index_name: str) -> bool:
        return index_name in self._indexes

    def _check_index_is_initialized(self, index_name: str):
        if not self.is_index_initialized(index_name):
            raise MissingIndex(
                f"Index with index_name '{index_name}' not initialized yet. Please make sure that you call `add_faiss_index` or `add_elasticsearch_index` first."
            )

    def list_indexes(self) -> list[str]:
        """List the `colindex_nameumns`/identifiers of all the attached indexes."""
        return list(self._indexes)

    def get_index(self, index_name: str) -> BaseIndex:
        """List the `index_name`/identifiers of all the attached indexes.

        Args:
            index_name (`str`): Index name.

        Returns:
            [`BaseIndex`]
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name]

    def add_faiss_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        device: Optional[Union[int, list[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of the specified column.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index, see more below).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            column (`str`): The column of the vectors to add to the index.
            index_name (Optional `str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
                By default it corresponds to `column`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        index_name = index_name if index_name is not None else column
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            self, column=column, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def add_faiss_index_from_external_arrays(
        self,
        external_arrays: np.array,
        index_name: str,
        device: Optional[Union[int, list[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of `external_arrays`.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            external_arrays (`np.array`): If you want to use arrays from outside the lib for the index, you can set `external_arrays`.
                It will use `external_arrays` to create the Faiss index instead of the arrays in the given `column`.
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            external_arrays, column=None, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def save_faiss_index(self, index_name: str, file: Union[str, PurePath], storage_options: Optional[dict] = None):
        """Save a FaissIndex on disk.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
        index = self.get_index(index_name)
        if not isinstance(index, FaissIndex):
            raise ValueError(f"Index '{index_name}' is not a FaissIndex but a '{type(index)}'")
        index.save(file, storage_options=storage_options)
        logger.info(f"Saved FaissIndex {index_name} at {file}")

    def load_faiss_index(
        self,
        index_name: str,
        file: Union[str, PurePath],
        device: Optional[Union[int, list[int]]] = None,
        storage_options: Optional[dict] = None,
    ):
        """Load a FaissIndex from disk.

        If you want to do additional configurations, you can have access to the faiss index object by doing
        `.get_index(index_name).faiss_index` to make it fit your needs.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to
                call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
        index = FaissIndex.load(file, device=device, storage_options=storage_options)
        if index.faiss_index.ntotal != len(self):
            raise ValueError(
                f"Index size should match Dataset size, but Index '{index_name}' at {file} has {index.faiss_index.ntotal} elements while the dataset has {len(self)} examples."
            )
        self._indexes[index_name] = index
        logger.info(f"Loaded FaissIndex {index_name} from {file}")

    def add_elasticsearch_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_name: Optional[str] = None,
        es_index_config: Optional[dict] = None,
    ):
        """Add a text index using ElasticSearch for fast retrieval.

        Args:
            column (`str`): The column of the documents to add to the index.
            index_name (Optional `str`): The index_name/identifier of the index. This is the index name that is used to call `.get_nearest` or `.search`.
                By default it corresponds to `column`.
            host (Optional `str`, defaults to localhost):
                host of where ElasticSearch is running
            port (Optional `str`, defaults to 9200):
                port of where ElasticSearch is running
            es_client (Optional `elasticsearch.Elasticsearch`):
                The elasticsearch client used to create the index if host and port are None.
            es_index_name (Optional `str`): The elasticsearch index name used to create the index.
            es_index_config (Optional `dict`):
                The configuration of the elasticsearch index.
                Default config is:

        Config::

            {
                "settings": {
                    "number_of_shards": 1,
                    "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "similarity": "BM25"
                        },
                    }
                },
            }
        """
        index_name = index_name if index_name is not None else column
        es_index = ElasticSearchIndex(
            host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config
        )
        es_index.add_documents(self, column=column)
        self._indexes[index_name] = es_index

    def load_elasticsearch_index(
        self,
        index_name: str,
        es_index_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_config: Optional[dict] = None,
    ):
        """Load an existing text index using ElasticSearch for fast retrieval.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index. This is the index name that is used to call `get_nearest` or `search`.
            es_index_name (`str`):
                The name of elasticsearch index to load.
            host (`str`, *optional*, defaults to `localhost`):
                Host of where ElasticSearch is running.
            port (`str`, *optional*, defaults to `9200`):
                Port of where ElasticSearch is running.
            es_client (`elasticsearch.Elasticsearch`, *optional*):
                The elasticsearch client used to create the index if host and port are `None`.
            es_index_config (`dict`, *optional*):
                The configuration of the elasticsearch index.
                Default config is:
                    ```
                    {
                        "settings": {
                            "number_of_shards": 1,
                            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                        },
                        "mappings": {
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard",
                                    "similarity": "BM25"
                                },
                            }
                        },
                    }
                    ```
        """
        self._indexes[index_name] = ElasticSearchIndex(
            host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config
        )

    def drop_index(self, index_name: str):
        """Drop the index with the specified column.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
        """
        del self._indexes[index_name]

    def search(self, index_name: str, query: Union[str, np.array], k: int = 10, **kwargs) -> SearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            `(scores, indices)`:
                A tuple of `(scores, indices)` where:
                - **scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples
                - **indices** (`List[List[int]]`): the indices of the retrieved examples
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search(query, k, **kwargs)

    def search_batch(
        self, index_name: str, queries: Union[list[str], np.array], k: int = 10, **kwargs
    ) -> BatchedSearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            `(total_scores, total_indices)`:
                A tuple of `(total_scores, total_indices)` where:
                - **total_scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples per query
                - **total_indices** (`List[List[int]]`): the indices of the retrieved examples per query
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search_batch(queries, k, **kwargs)

    def get_nearest_examples(
        self, index_name: str, query: Union[str, np.array], k: int = 10, **kwargs
    ) -> NearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The index_name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            `(scores, examples)`:
                A tuple of `(scores, examples)` where:
                - **scores** (`List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples
                - **examples** (`dict`): the retrieved examples
        """
        self._check_index_is_initialized(index_name)
        scores, indices = self.search(index_name, query, k, **kwargs)
        top_indices = [i for i in indices if i >= 0]
        return NearestExamplesResults(scores[: len(top_indices)], self[top_indices])

    def get_nearest_examples_batch(
        self, index_name: str, queries: Union[list[str], np.array], k: int = 10, **kwargs
    ) -> BatchedNearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            `(total_scores, total_examples)`:
                A tuple of `(total_scores, total_examples)` where:
                - **total_scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples per query
                - **total_examples** (`List[dict]`): the retrieved examples per query
        """
        self._check_index_is_initialized(index_name)
        total_scores, total_indices = self.search_batch(index_name, queries, k, **kwargs)
        total_scores = [
            scores_i[: len([i for i in indices_i if i >= 0])]
            for scores_i, indices_i in zip(total_scores, total_indices)
        ]
        total_samples = [self[[i for i in indices if i >= 0]] for indices in total_indices]
        return BatchedNearestExamplesResults(total_scores, total_samples)
<|endoftext|>import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import huggingface_hub
from fsspec.core import url_to_fs
from huggingface_hub import HfFileSystem
from packaging import version
from tqdm.contrib.concurrent import thread_map

from . import config
from .download import DownloadConfig
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import _prepare_path_and_storage_options, is_local_path, is_relative_path, xbasename, xjoin
from .utils.py_utils import string_to_dict


SingleOriginMetadata = Union[tuple[str, str], tuple[str], tuple[()]]


SANITIZED_DEFAULT_SPLIT = str(Split.TRAIN)


logger = logging.get_logger(__name__)


class Url(str):
    pass


class EmptyDatasetError(FileNotFoundError):
    pass


SPLIT_PATTERN_SHARDED = "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"

SPLIT_KEYWORDS = {
    Split.TRAIN: ["train", "training"],
    Split.VALIDATION: ["validation", "valid", "dev", "val"],
    Split.TEST: ["test", "testing", "eval", "evaluation"],
}
NON_WORDS_CHARS = "-._ 0-9"
if config.FSSPEC_VERSION < version.parse("2023.9.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**",
        "{keyword}[{sep}]*/**",
        "**[{sep}/]{keyword}/**",
        "**[{sep}/]{keyword}[{sep}]*/**",
    ]
elif config.FSSPEC_VERSION < version.parse("2023.12.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/*[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**/*",
        "{keyword}[{sep}]*/**/*",
        "**/*[{sep}/]{keyword}/**/*",
        "**/*[{sep}/]{keyword}[{sep}]*/**/*",
    ]
else:
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/{keyword}[{sep}]*", "**/*[{sep}]{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "**/{keyword}/**",
        "**/{keyword}[{sep}]*/**",
        "**/*[{sep}]{keyword}/**",
        "**/*[{sep}]{keyword}[{sep}]*/**",
    ]

DEFAULT_SPLITS = [Split.TRAIN, Split.VALIDATION, Split.TEST]
DEFAULT_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}
DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}


DEFAULT_PATTERNS_ALL = {
    Split.TRAIN: ["**"],
}

DEFAULT_PATTERNS_LOGS = {"logs": ["**/*.eval"]}
DEFAULT_PATTERNS_HARBOR = {"test": ["**/task.toml", "**/instruction.md"]}

ALL_SPLIT_PATTERNS = [SPLIT_PATTERN_SHARDED]
ALL_DEFAULT_PATTERNS = [
    DEFAULT_PATTERNS_LOGS,
    DEFAULT_PATTERNS_HARBOR,
    DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_PATTERNS_ALL,
]
WILDCARD_CHARACTERS = "*[]"
FILES_TO_IGNORE = [
    "README.md",
    "config.json",
    "dataset_info.json",
    "dataset_infos.json",
    "dummy_data.zip",
    "dataset_dict.json",
]


def contains_wildcards(pattern: str) -> bool:
    return any(wildcard_character in pattern for wildcard_character in WILDCARD_CHARACTERS)


def sanitize_patterns(patterns: Union[dict, list, str]) -> dict[str, Union[list[str], "DataFilesList"]]:
    """
    Take the data_files patterns from the user, and format them into a dictionary.
    Each key is the name of the split, and each value is a list of data files patterns (paths or urls).
    The default split is "train".

    Returns:
        patterns: dictionary of split_name -> list of patterns
    """
    if isinstance(patterns, dict):
        return {str(key): value if isinstance(value, list) else [value] for key, value in patterns.items()}
    elif isinstance(patterns, str):
        return {SANITIZED_DEFAULT_SPLIT: [patterns]}
    elif isinstance(patterns, list):
        if any(isinstance(pattern, dict) for pattern in patterns):
            for pattern in patterns:
                if not (
                    isinstance(pattern, dict)
                    and len(pattern) == 2
                    and "split" in pattern
                    and isinstance(pattern.get("path"), (str, list))
                ):
                    raise ValueError(
                        "Invalid format for data_files entry. "
                        "Each item must be a dictionary with the structure "
                        "{'split': <split_name>, 'path': <path_or_list_of_paths>}.\n"
                        f"Received: {pattern}"
                    )
            splits = [pattern["split"] for pattern in patterns]
            if len(set(splits)) != len(splits):
                raise ValueError(f"Some splits are duplicated in data_files: {splits}")
            return {
                str(pattern["split"]): pattern["path"] if isinstance(pattern["path"], list) else [pattern["path"]]
                for pattern in patterns
            }
        else:
            return {SANITIZED_DEFAULT_SPLIT: patterns}
    else:
        return sanitize_patterns(list(patterns))


def _is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionally check if it's inside a special directory
    we ignore by default (if it starts with a double underscore).

    Users can still explicitly request a filepath inside such a directory if "__pycache__" is
    mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── __pycache__
            └── b.txt

    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "**")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "*/b.txt")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__pycache__/*")
    False
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__*/*")
    False
    """
    # We just need to check if every special directories from the path is present explicitly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the parent path and the parent pattern have the same number of special directories.
    data_dirs_to_ignore_in_path = [part for part in PurePath(matched_rel_path).parent.parts if part.startswith("__")]
    data_dirs_to_ignore_in_pattern = [part for part in PurePath(pattern).parent.parts if part.startswith("__")]
    return len(data_dirs_to_ignore_in_path) != len(data_dirs_to_ignore_in_pattern)


def _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionally check if it's a hidden file or if it's inside
    a hidden directory we ignore by default, i.e. if the file name or a parent directory name starts with a dot.

    Users can still explicitly request a filepath that is hidden or is inside a hidden directory
    if the hidden part is mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", ".*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── a.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".*/*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".hidden_dir/*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/.*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/.*")
    False
    """
    # We just need to check if every hidden part from the path is present explicitly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the path and the pattern have the same number of hidden parts.
    hidden_directories_in_path = [
        part for part in PurePath(matched_rel_path).parts if part.startswith(".") and not set(part) == {"."}
    ]
    hidden_directories_in_pattern = [
        part for part in PurePath(pattern).parts if part.startswith(".") and not set(part) == {"."}
    ]
    return len(hidden_directories_in_path) != len(hidden_directories_in_pattern)


def _get_data_files_patterns(pattern_resolver: Callable[[str], list[str]]) -> dict[str, list[str]]:
    """
    Get the default pattern from a directory or repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    # first check the split patterns like data/{split}-00000-of-00001.parquet
    for split_pattern in ALL_SPLIT_PATTERNS:
        pattern = split_pattern.replace("{split}", "*")
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: set[str] = set()
            for p in data_files:
                p_parts = string_to_dict(xbasename(p), xbasename(split_pattern))
                assert p_parts is not None
                splits.add(p_parts["split"])

            if any(not re.match(_split_re, split) for split in splits):
                raise ValueError(f"Split name should match '{_split_re}'' but got '{splits}'.")
            sorted_splits = [str(split) for split in DEFAULT_SPLITS if split in splits] + sorted(
                splits - {str(split) for split in DEFAULT_SPLITS}
            )
            return {split: [split_pattern.format(split=split)] for split in sorted_splits}
    # then check the default patterns based on train/valid/test splits
    for patterns_dict in ALL_DEFAULT_PATTERNS:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")


def resolve_pattern(
    pattern: str,
    base_path: str,
    allowed_extensions: Optional[list[str]] = None,
    download_config: Optional[DownloadConfig] = None,
) -> list[str]:
    """
    Resolve the paths and URLs of the data files from the pattern passed by the user.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob. In fsspec>=2023.12.0 this is equivalent to
    Python's glob.glob, Path.glob, Path.match and fnmatch where ** is unsupported with a prefix/suffix
    other than a forward slash /.

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicitly mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"

    Example::

        >>> from datasets.data_files import resolve_pattern
        >>> base_path = "."
        >>> resolve_pattern("docs/**/*.py", base_path)
        [/Users/mariosasko/Desktop/projects/datasets/docs/source/_config.py']

    Args:
        pattern (str): Unix pattern or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
            Remote filesystems using fsspec are supported, e.g. with the hf:// protocol.
        base_path (str): Base path to use when resolving relative paths.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=[".csv", ".json", ".txt", ".parquet"]
        download_config ([`DownloadConfig`], *optional*): Specific download configuration parameters.
    Returns:
        List[str]: List of paths or URLs to the local or remote files that match the patterns.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ""
    pattern, storage_options = _prepare_path_and_storage_options(pattern, download_config=download_config)
    fs, fs_pattern = url_to_fs(pattern, **storage_options)
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = (
        pattern.split("://")[0]
        if "://" in pattern
        else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
    )
    protocol_prefix = protocol + "://" if protocol != "file" else ""
    glob_kwargs = {}
    if protocol == "hf":
        # 10 times faster glob with detail=True (ignores costly info like lastCommit)
        glob_kwargs["expand_info"] = False

    # if the pattern contains hops like "zip://csv/*.csv::data.zip", we need to keep them after globbing
    _, *rest_hops = pattern.split("::")
    matched_paths = []
    for filepath, info in fs.glob(fs_pattern, detail=True, **glob_kwargs).items():
        if not (info["type"] == "file" or (info.get("islink") and os.path.isfile(os.path.realpath(filepath)))) or (
            xbasename(filepath) in files_to_ignore
        ):
            continue
        if _is_inside_unrequested_special_dir(filepath, fs_pattern):
            continue
        if _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(filepath, fs_pattern):
            continue
        filepath = filepath if "://" in filepath else protocol_prefix + filepath
        if rest_hops:
            filepath = "::".join([filepath] + rest_hops)
        matched_paths.append(filepath)
    # ignore .ipynb and __pycache__, but keep /../
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any("." + suffix in allowed_extensions for suffix in xbasename(filepath).split(".")[1:])
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(
                f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: {invalid_matched_files}"
            )
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return out


def get_data_patterns(base_path: str, download_config: Optional[DownloadConfig] = None) -> dict[str, list[str]]:
    """
    Get the default pattern from a directory testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    Some examples of supported patterns:

    Input:

        my_dataset_repository/
        ├── README.md
        └── dataset.csv

    Output:

        {'train': ['**']}

    Input:

        my_dataset_repository/
        ├── README.md
        ├── train.csv
        └── test.csv

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train.csv
            └── test.csv

        my_dataset_repository/
        ├── README.md
        ├── train_0.csv
        ├── train_1.csv
        ├── train_2.csv
        ├── train_3.csv
        ├── test_0.csv
        └── test_1.csv

    Output:

        {'train': ['**/train[-._ 0-9]*', '**/*[-._ 0-9]train[-._ 0-9]*', '**/training[-._ 0-9]*', '**/*[-._ 0-9]training[-._ 0-9]*'],
         'test': ['**/test[-._ 0-9]*', '**/*[-._ 0-9]test[-._ 0-9]*', '**/testing[-._ 0-9]*', '**/*[-._ 0-9]testing[-._ 0-9]*', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train/
            │   ├── shard_0.csv
            │   ├── shard_1.csv
            │   ├── shard_2.csv
            │   └── shard_3.csv
            └── test/
                ├── shard_0.csv
                └── shard_1.csv

    Output:

        {'train': ['**/train/**', '**/train[-._ 0-9]*/**', '**/*[-._ 0-9]train/**', '**/*[-._ 0-9]train[-._ 0-9]*/**', ...],
         'test': ['**/test/**', '**/test[-._ 0-9]*/**', '**/*[-._ 0-9]test/**', '**/*[-._ 0-9]test[-._ 0-9]*/**', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train-00000-of-00003.csv
            ├── train-00001-of-00003.csv
            ├── train-00002-of-00003.csv
            ├── test-00000-of-00001.csv
            ├── random-00000-of-00003.csv
            ├── random-00001-of-00003.csv
            └── random-00002-of-00003.csv

    Output:

        {'train': ['data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'test': ['data/test-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'random': ['data/random-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*']}

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(f"The directory at {base_path} doesn't contain any data files") from None


def _get_single_origin_metadata(
    data_file: str,
    download_config: Optional[DownloadConfig] = None,
) -> SingleOriginMetadata:
    if data_file.startswith(config.HF_ENDPOINT):
        fs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
        data_file = "hf://" + data_file[len(config.HF_ENDPOINT) + 1 :]
        data_file = data_file.replace("/resolve/", "/" if data_file.startswith("hf://buckets/") else "@", 1)
        fs_path = data_file
    else:
        data_file, storage_options = _prepare_path_and_storage_options(data_file, download_config=download_config)
        fs, fs_path = url_to_fs(data_file, **storage_options)
    if isinstance(fs, HfFileSystem):
        resolved_path = fs.resolve_path(fs_path)
        if hasattr(resolved_path, "revision"):  # no revision for buckets
            return resolved_path.repo_id, resolved_path.revision
    info = fs.info(fs_path)
    # s3fs uses "ETag", gcsfs uses "etag", and for local we simply check mtime
    for key in ["ETag", "etag", "mtime"]:
        if key in info:
            return (str(info[key]),)
    return ()


def _get_origin_metadata(
    data_files: list[str],
    download_config: Optional[DownloadConfig] = None,
    max_workers: Optional[int] = None,
) -> list[SingleOriginMetadata]:
    max_workers = max_workers if max_workers is not None else config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
    if all("hf://" in data_file for data_file in data_files):
        # No need for multithreading here since the origin metadata of HF files
        # is (repo_id, revision) and is cached after first .info() call.
        return [
            _get_single_origin_metadata(data_file, download_config=download_config)
            for data_file in hf_tqdm(
                data_files,
                desc="Resolving data files",
                # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
                disable=len(data_files) <= 16 or None,
            )
        ]
    return thread_map(
        partial(_get_single_origin_metadata, download_config=download_config),
        data_files,
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
        desc="Resolving data files",
        # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
        disable=len(data_files) <= 16 or None,
    )


class DataFilesList(list[str]):
    """
    List of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns:
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, DataFilesList has an additional attribute ``origin_metadata``.
    It can store:
    - the last modified time of local files
    - ETag of remote files
    - commit sha of a dataset repository

    Thanks to this additional attribute, it is possible to hash the list
    and get a different hash if and only if at least one file changed.
    This is useful for caching Dataset objects that are obtained from a list of data files.
    """

    def __init__(self, data_files: list[str], origin_metadata: list[SingleOriginMetadata]) -> None:
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    def __add__(self, other: "DataFilesList") -> "DataFilesList":
        return DataFilesList([*self, *other], self.origin_metadata + other.origin_metadata)

    @classmethod
    def from_hf_repo(
        cls,
        patterns: list[str],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = f"hf://datasets/{dataset_info.id}@{dataset_info.sha}/{base_path or ''}".rstrip("/")
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_patterns(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern in patterns:
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)

    def filter(
        self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None
    ) -> "DataFilesList":
        patterns = []
        if extensions:
            ext_pattern = "|".join(re.escape(ext) for ext in extensions)
            patterns.append(re.compile(f".*({ext_pattern})(\\..+)?$"))
        if file_names:
            fn_pattern = "|".join(re.escape(fn) for fn in file_names)
            patterns.append(re.compile(rf".*[\/]?({fn_pattern})$"))
        if patterns:
            return DataFilesList(
                [data_file for data_file in self if any(pattern.match(data_file) for pattern in patterns)],
                origin_metadata=self.origin_metadata,
            )
        else:
            return DataFilesList(list(self), origin_metadata=self.origin_metadata)


class DataFilesDict(dict[str, DataFilesList]):
    """
    Dict of split_name -> list of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns :
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, each list is a DataFilesList. It is possible to hash the dictionary
    and get a different hash if and only if at least one file changed.
    For more info, see [`DataFilesList`].

    This is useful for caching Dataset objects that are obtained from a list of data files.

    Changing the order of the keys of this dictionary also doesn't change its hash.
    """

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_local_or_remote(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_hf_repo(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_hf_repo(
                    patterns_for_key,
                    dataset_info=dataset_info,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_patterns(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_patterns(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    def filter(
        self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None
    ) -> "DataFilesDict":
        out = type(self)()
        for key, data_files_list in self.items():
            out[key] = data_files_list.filter(extensions=extensions, file_names=file_names)
        return out


class DataFilesPatternsList(list[str]):
    """
    List of data files patterns (absolute local paths or URLs).
    For each pattern there should also be a list of allowed extensions
    to keep, or a None ot keep all the files for the pattern.
    """

    def __init__(
        self,
        patterns: list[str],
        allowed_extensions: list[Optional[list[str]]],
    ):
        super().__init__(patterns)
        self.allowed_extensions = allowed_extensions

    def __add__(self, other):
        return DataFilesList([*self, *other], self.allowed_extensions + other.allowed_extensions)

    @classmethod
    def from_patterns(
        cls, patterns: list[str], allowed_extensions: Optional[list[str]] = None
    ) -> "DataFilesPatternsList":
        return cls(patterns, [allowed_extensions] * len(patterns))

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern, allowed_extensions in zip(self, self.allowed_extensions):
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return DataFilesList(data_files, origin_metadata)

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsList":
        return DataFilesPatternsList(
            self, [allowed_extensions + extensions for allowed_extensions in self.allowed_extensions]
        )


class DataFilesPatternsDict(dict[str, DataFilesPatternsList]):
    """
    Dict of split_name -> list of data files patterns (absolute local paths or URLs).
    """

    @classmethod
    def from_patterns(
        cls, patterns: dict[str, list[str]], allowed_extensions: Optional[list[str]] = None
    ) -> "DataFilesPatternsDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesPatternsList)
                else DataFilesPatternsList.from_patterns(
                    patterns_for_key,
                    allowed_extensions=allowed_extensions,
                )
            )
        return out

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = DataFilesDict()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.resolve(base_path, download_config)
        return out

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsDict":
        out = type(self)()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.filter_extensions(extensions)
        return out
<|endoftext|>import contextlib
import copy
import itertools
import json
import math
import posixpath
import random
import re
import time
from collections.abc import Sequence
from functools import partial
from typing import Callable, Literal, Optional, Union

import fsspec
import numpy as np
from fsspec.core import url_to_fs
from fsspec.implementations.dirfs import DirFileSystem
from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    HfFileSystem,
)
from huggingface_hub.errors import BucketNotFoundError
from huggingface_hub.hf_file_system import HfFileSystemResolvedBucketPath, HfFileSystemResolvedRepositoryPath
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from . import __version__, config
from .arrow_dataset import (
    Dataset,
    _get_updated_dataset_card,
)
from .features import Features
from .features.features import FeatureType
from .iterable_dataset import IterableDataset
from .naming import _split_re
from .splits import NamedSplit, Split, SplitInfo
from .table import Table
from .utils import logging
from .utils.doc_utils import is_documented_by
from .utils.typing import PathLike


logger = logging.get_logger(__name__)


class bind(partial):
    def __call__(self, *fn_args, **fn_kwargs):
        return self.func(*fn_args, *self.args, **fn_kwargs)


class DatasetDict(dict[Union[str, NamedSplit], "Dataset"]):
    """A dictionary (dict of str: datasets.Dataset) with dataset transforms methods (map, filter, etc.)"""

    def _check_values_type(self):
        for dataset in self.values():
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Values in `DatasetDict` should be of type `Dataset` but got type '{type(dataset)}'")

    def _check_values_features(self):
        items = list(self.items())
        for item_a, item_b in zip(items[:-1], items[1:]):
            if item_a[1].features != item_b[1].features:
                raise ValueError(
                    f"All datasets in `DatasetDict` should have the same features but features for '{item_a[0]}' and '{item_b[0]}' don't match: {item_a[1].features} != {item_b[1].features}"
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Here `del` is used to del the pyarrow tables. This properly closes the files used for memory mapped tables
        for dataset in self.values():
            if hasattr(dataset, "_data"):
                del dataset._data
            if hasattr(dataset, "_indices"):
                del dataset._indices

    def __getitem__(self, k) -> Dataset:
        if isinstance(k, (str, NamedSplit)) or len(self) == 0:
            return super().__getitem__(k)
        else:
            available_suggested_splits = [
                split for split in (Split.TRAIN, Split.TEST, Split.VALIDATION) if split in self
            ]
            suggested_split = available_suggested_splits[0] if available_suggested_splits else list(self)[0]
            raise KeyError(
                f"Invalid key: {k}. Please first select a split. For example: "
                f"`my_dataset_dictionary['{suggested_split}'][{k}]`. "
                f"Available splits: {sorted(self)}"
            )

    @property
    def data(self) -> dict[str, Table]:
        """The Apache Arrow tables backing each split.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.data
        ```
        """
        self._check_values_type()
        return {k: dataset.data for k, dataset in self.items()}

    @property
    def cache_files(self) -> dict[str, dict]:
        """The cache files containing the Apache Arrow table backing each split.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.cache_files
        {'test': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-test.arrow'}],
         'train': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-train.arrow'}],
         'validation': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-validation.arrow'}]}
        ```
        """
        self._check_values_type()
        return {k: dataset.cache_files for k, dataset in self.items()}

    @property
    def num_columns(self) -> dict[str, int]:
        """Number of columns in each split of the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.num_columns
        {'test': 2, 'train': 2, 'validation': 2}
        ```
        """
        self._check_values_type()
        return {k: dataset.num_columns for k, dataset in self.items()}

    @property
    def num_rows(self) -> dict[str, int]:
        """Number of rows in each split of the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.num_rows
        {'test': 1066, 'train': 8530, 'validation': 1066}
        ```
        """
        self._check_values_type()
        return {k: dataset.num_rows for k, dataset in self.items()}

    @property
    def column_names(self) -> dict[str, list[str]]:
        """Names of the columns in each split of the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.column_names
        {'test': ['text', 'label'],
         'train': ['text', 'label'],
         'validation': ['text', 'label']}
        ```
        """
        self._check_values_type()
        return {k: dataset.column_names for k, dataset in self.items()}

    @property
    def shape(self) -> dict[str, tuple[int]]:
        """Shape of each split of the dataset (number of rows, number of columns).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.shape
        {'test': (1066, 2), 'train': (8530, 2), 'validation': (1066, 2)}
        ```
        """
        self._check_values_type()
        return {k: dataset.shape for k, dataset in self.items()}

    def flatten(self, max_depth=16) -> "DatasetDict":
        """Flatten the Apache Arrow Table of each split (nested features are flatten).
        Each column with a struct type is flattened into one column per struct field.
        Other columns are left unchanged.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rajpurkar/squad")
        >>> ds["train"].features
        {'id': Value('string'),
         'title': Value('string'),
         'context': Value('string'),
         'question': Value('string'),
         'answers.text': List(Value('string')),
         'answers.answer_start': List(Value('int32'))}
        >>> ds.flatten()
        DatasetDict({
            train: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
                num_rows: 87599
            })
            validation: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
                num_rows: 10570
            })
        })
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.flatten(max_depth=max_depth) for k, dataset in self.items()})

    def unique(self, column: str) -> dict[str, list]:
        """Return a list of the unique elements in a column for each split.

        This is implemented in the low-level backend and as such, very fast.

        Args:
            column (`str`):
                column name (list all the column names with [`~datasets.DatasetDict.column_names`])

        Returns:
            Dict[`str`, `list`]: Dictionary of unique elements in the given column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.unique("label")
        {'test': [1, 0], 'train': [1, 0], 'validation': [1, 0]}
        ```
        """
        self._check_values_type()
        return {k: dataset.unique(column) for k, dataset in self.items()}

    def cleanup_cache_files(self) -> dict[str, int]:
        """Clean up all cache files in the dataset cache directory, excepted the currently used cache file if there is one.
        Be careful when running this command that no other process is currently using other cache files.

        Return:
            `Dict` with the number of removed files for each split

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.cleanup_cache_files()
        {'test': 0, 'train': 0, 'validation': 0}
        ```
        """
        self._check_values_type()
        return {k: dataset.cleanup_cache_files() for k, dataset in self.items()}

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, count=0, flags=re.M)
        return f"DatasetDict({{\n{repr}\n}})"

    def cast(self, features: Features) -> "DatasetDict":
        """
        Cast the dataset to a new set of features.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            features ([`Features`]):
                New features to cast the dataset to.
                The name and order of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`~DatasetDict.map`] to update the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset, ClassLabel, Value
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds["train"].features
        {'label': ClassLabel(names=['neg', 'pos']),
         'text': Value('string')}
        >>> new_features = ds["train"].features.copy()
        >>> new_features['label'] = ClassLabel(names=['bad', 'good'])
        >>> new_features['text'] = Value('large_string')
        >>> ds = ds.cast(new_features)
        >>> ds["train"].features
        {'label': ClassLabel(names=['bad', 'good']),
         'text': Value('large_string')}
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})

    def cast_column(self, column: str, feature) -> "DatasetDict":
        """Cast column to feature for decoding.

        Args:
            column (`str`):
                Column name.
            feature ([`Feature`]):
                Target feature.

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import load_dataset, ClassLabel
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds["train"].features
        {'label': ClassLabel(names=['neg', 'pos']),
         'text': Value('string')}
        >>> ds = ds.cast_column('label', ClassLabel(names=['bad', 'good']))
        >>> ds["train"].features
        {'label': ClassLabel(names=['bad', 'good']),
         'text': Value('string')}
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()})

    def remove_columns(self, column_names: Union[str, list[str]]) -> "DatasetDict":
        """
        Remove one or several column(s) from each split in the dataset
        and the features associated to the column(s).

        The transformation is applied to all the splits of the dataset dictionary.

        You can also remove a column using [`~DatasetDict.map`] with `remove_columns` but the present method
        doesn't copy the data of the remaining columns and is thus faster.

        Args:
            column_names (`Union[str, list[str]]`):
                Name of the column(s) to remove.

        Returns:
            [`DatasetDict`]: A copy of the dataset object without the columns to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds = ds.remove_columns("label")
        DatasetDict({
            train: Dataset({
                features: ['text'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text'],
                num_rows: 1066
            })
        })
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.remove_columns(column_names=column_names) for k, dataset in self.items()})

    def rename_column(self, original_column_name: str, new_column_name: str) -> "DatasetDict":
        """
        Rename a column in the dataset and move the features associated to the original column under the new column name.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can also rename a column using [`~DatasetDict.map`] with `remove_columns` but the present method:
            - takes care of moving the original features under the new column name.
            - doesn't copy the data to a new dataset and is thus much faster.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds = ds.rename_column("label", "label_new")
        DatasetDict({
            train: Dataset({
                features: ['text', 'label_new'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text', 'label_new'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text', 'label_new'],
                num_rows: 1066
            })
        })
        ```
        """
        self._check_values_type()
        return DatasetDict(
            {
                k: dataset.rename_column(
                    original_column_name=original_column_name,
                    new_column_name=new_column_name,
                )
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: dict[str, str]) -> "DatasetDict":
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names.

        Returns:
            [`DatasetDict`]: A copy of the dataset with renamed columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.rename_columns({'text': 'text_new', 'label': 'label_new'})
        DatasetDict({
            train: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 1066
            })
        })
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, list[str]]) -> "DatasetDict":
        """Select one or several column(s) from each split in the dataset and
        the features associated to the column(s).

        The transformation is applied to all the splits of the dataset
        dictionary.

        Args:
            column_names (`Union[str, list[str]]`):
                Name of the column(s) to keep.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.select_columns("text")
        DatasetDict({
            train: Dataset({
                features: ['text'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text'],
                num_rows: 1066
            })
        })
        ```
        """
        self._check_values_type()
        return DatasetDict({k: dataset.select_columns(column_names=column_names) for k, dataset in self.items()})

    def class_encode_column(self, column: str, include_nulls: bool = False) -> "DatasetDict":
        """Casts the given column as [`~datasets.features.ClassLabel`] and updates the tables.

        Args:
            column (`str`):
                The name of the column to cast.
            include_nulls (`bool`, defaults to `False`):
                Whether to include null values in the class labels. If `True`, the null values will be encoded as the `"None"` class label.

                <Added version="1.14.2"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("google/boolq")
        >>> ds["train"].features
        {'answer': Value('bool'),
         'passage': Value('string'),
         'question': Value('string')}
        >>> ds = ds.class_encode_column("answer")
        >>> ds["train"].features
        {'answer': ClassLabel(num_classes=2, names=['False', 'True']),
         'passage': Value('string'),
         'question': Value('string')}
        ```
        """
        self._check_values_type()
        return DatasetDict(
            {k: dataset.class_encode_column(column=column, include_nulls=include_nulls) for k, dataset in self.items()}
        )

    @contextlib.contextmanager
    def formatted_as(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """To be used in a `with` statement. Set `__getitem__` return format (type and columns).
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'jax', 'arrow', 'pandas', 'polars']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`list[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """
        self._check_values_type()
        old_format_type = {k: dataset._format_type for k, dataset in self.items()}
        old_format_kwargs = {k: dataset._format_kwargs for k, dataset in self.items()}
        old_format_columns = {k: dataset._format_columns for k, dataset in self.items()}
        old_output_all_columns = {k: dataset._output_all_columns for k, dataset in self.items()}
        try:
            self.set_format(type, columns, output_all_columns, **format_kwargs)
            yield
        finally:
            for k, dataset in self.items():
                dataset.set_format(
                    old_format_type[k],
                    old_format_columns[k],
                    old_output_all_columns[k],
                    **old_format_kwargs[k],
                )

    def set_format(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """Set `__getitem__` return format (type and columns).
        The format is set for every dataset in the dataset dictionary.

        Args:
            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'jax', 'arrow', 'pandas', 'polars']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`list[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects),
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        It is possible to call `map` after calling `set_format`. Since `map` may add new columns, then the list of formatted columns
        gets updated. In this case, if you apply `map` on a dataset to add a new column, then this column will be formatted:

            `new formatted columns = (all columns - previously unformatted columns)`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type="numpy", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds["train"].format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'numpy'}
        ```
        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format(
                type=type,
                columns=columns,
                output_all_columns=output_all_columns,
                **format_kwargs,
            )

    def reset_format(self):
        """Reset `__getitem__` return format to python objects and all columns.
        The transformation is applied to all the datasets of the dataset dictionary.

        Same as `self.set_format()`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type="numpy", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds["train"].format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'numpy'}
        >>> ds.reset_format()
        >>> ds["train"].format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        ```
        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format()

    def set_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[list] = None,
        output_all_columns: bool = False,
    ):
        """Set ``__getitem__`` return format using this transform. The transform is applied on-the-fly on batches when ``__getitem__`` is called.
        The transform is set for every dataset in the dataset dictionary
        As :func:`datasets.Dataset.set_format`, this can be reset using :func:`datasets.Dataset.reset_format`

        Args:
            transform (`Callable`, optional): user-defined formatting transform, replaces the format defined by :func:`datasets.Dataset.set_format`
                A formatting function is a callable that takes a batch (as a dict) as input and returns a batch.
                This function is applied right before returning the objects in ``__getitem__``.
            columns (`list[str]`, optional): columns to format in the output
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, default to False): keep un-formatted columns as well in the output (as python objects)
                If set to True, then the other un-formatted columns are kept with the output of the transform.

        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format(
                "custom",
                columns=columns,
                output_all_columns=output_all_columns,
                transform=transform,
            )

    def with_format(
        self,
        type: Optional[str] = None,
        columns: Optional[list] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ) -> "DatasetDict":
        """Set `__getitem__` return format (type and columns). The data formatting is applied on-the-fly.
        The format `type` (for example "numpy") is used to format batches when using `__getitem__`.
        The format is set for every dataset in the dataset dictionary.

        It's also possible to use custom transforms for formatting using [`~datasets.Dataset.with_transform`].

        Contrary to [`~datasets.DatasetDict.set_format`], `with_format` returns a new [`DatasetDict`] object with new [`Dataset`] objects.

        Args:
            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'jax', 'arrow', 'pandas', 'polars']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`list[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds["train"].format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        >>> ds = ds.with_format("torch")
        >>> ds["train"].format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'torch'}
        >>> ds["train"][0]
        {'text': 'compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .',
         'label': tensor(1),
         'input_ids': tensor([  101, 18027, 16310, 16001,  1103,  9321,   178, 11604,  7235,  6617,
                1742,  2165,  2820,  1206,  6588, 22572, 12937,  1811,  2153,  1105,
                1147, 12890, 19587,  6463,  1105, 15026,  1482,   119,   102,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
        ```
        """
        dataset = copy.deepcopy(self)
        dataset.set_format(
            type=type,
            columns=columns,
            output_all_columns=output_all_columns,
            **format_kwargs,
        )
        return dataset

    def with_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[list] = None,
        output_all_columns: bool = False,
    ) -> "DatasetDict":
        """Set `__getitem__` return format using this transform. The transform is applied on-the-fly on batches when `__getitem__` is called.
        The transform is set for every dataset in the dataset dictionary

        As [`~datasets.Dataset.set_format`], this can be reset using [`~datasets.Dataset.reset_format`].

        Contrary to [`~datasets.DatasetDict.set_transform`], `with_transform` returns a new [`DatasetDict`] object with new [`Dataset`] objects.

        Args:
            transform (`Callable`, *optional*):
                User-defined formatting transform, replaces the format defined by [`~datasets.Dataset.set_format`].
                A formatting function is a callable that takes a batch (as a dict) as input and returns a batch.
                This function is applied right before returning the objects in `__getitem__`.
            columns (`list[str]`, *optional*):
                Columns to format in the output.
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects).
                If set to `True`, then the other un-formatted columns are kept with the output of the transform.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> def encode(example):
        ...     return tokenizer(example['text'], truncation=True, padding=True, return_tensors="pt")
        >>> ds = ds.with_transform(encode)
        >>> ds["train"][0]
        {'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]),
         'input_ids': tensor([  101,  1103,  2067,  1110, 17348,  1106,  1129,  1103,  6880,  1432,
                112,   188,  1207,   107, 14255,  1389,   107,  1105,  1115,  1119,
                112,   188,  1280,  1106,  1294,   170, 24194,  1256,  3407,  1190,
                170, 11791,  5253,   188,  1732,  7200, 10947, 12606,  2895,   117,
                179,  7766,   118,   172, 15554,  1181,  3498,  6961,  3263,  1137,
                188,  1566,  7912, 14516,  6997,   119,   102]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0])}
        ```
        """
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        with_split: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
        try_original_type: Optional[bool] = True,
        on_mixed_types: Optional[Literal["use_json"]] = None,
    ) -> "DatasetDict":
        """
        Apply a function to all the examples in the table (individually or in batches) and update the table.
        If your function returns a column that already exists, then it overwrites it.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
        - If batched is `True` and `batch_size` is `n > 1`, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.

        If the function is asynchronous, then `map` will run your function in parallel, with up to one thousand simultaneous calls.
        It is recommended to use a `asyncio.Semaphore` in your function if you want to set a maximum number of operations that can run at the same time.

        Args:
            function (`callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False`
                - `function(example: Dict[str, Any], indices: int) -> Dict[str, Any]` if `batched=False` and `with_indices=True`
                - `function(batch: Dict[str, list]) -> Dict[str, list]` if `batched=True` and `with_indices=False`
                - `function(batch: Dict[str, list], indices: list[int]) -> Dict[str, list]` if `batched=True` and `with_indices=True`

                For advanced usage, the function can also return a `pyarrow.Table`.
                If the function is asynchronous, then `map` will run your function in parallel.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: `lambda x: x`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            with_split (`bool`, defaults to `False`):
                Provide process split to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], split): ...`.
            input_columns (`[Union[str, list[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`,
                `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to `function`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`[Union[str, list[str]]]`, *optional*, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, default `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`[datasets.Features]`, *optional*, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Disallow null values in the table.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`
            num_proc (`int`, *optional*, defaults to `None`):
                 The number of processes to use for multiprocessing.
                - If `None` or `0`, no multiprocessing is used and the operation runs in the main process.
                - If greater than `1`, one or multiple worker processes are used to process data in parallel.
                 Note: The function passed to `map()` must be picklable for multiprocessing to work correctly
                 (i.e., prefer functions defined at the top level of a module, not inside another function or class).
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while mapping examples.
            try_original_type (`Optional[bool]`, defaults to `True`):
                Try to keep the types of the original columns (e.g. int32 -> int32).
                Set to False if you want to always infer new types.
            on_mixed_types (`Literal["use_json"]`, *optional*, defaults to `None`):
                If "use_json", use the Json() type for mixed-types fields,
                i.e. unstructured fields that contain data without a predefined schema.
                In this case, a field with mixed type is set to Json().

                This allow loading lists with a mix of strings/integers/floats
                for example, or dictionaries with arbitrary value types.

                <Added version="4.7.0"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> ds["train"][0:3]["text"]
        ['Review: the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
         'Review: the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
         'Review: effective but too-tepid biopic']

        # process a batch of examples
        >>> ds = ds.map(lambda example: tokenizer(example["text"]), batched=True)
        # set number of processors
        >>> ds = ds.map(add_prefix, num_proc=4)
        ```
        """
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)

        dataset_dict = {}
        for split, dataset in self.items():
            if with_split:
                function = bind(function, split)

            dataset_dict[split] = dataset.map(
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[split],
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc=desc,
                try_original_type=try_original_type,
                on_mixed_types=on_mixed_types,
            )

            if with_split:
                function = function.func

        return DatasetDict(dataset_dict)

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "DatasetDict":
        """Apply a filter function to all the elements in the table in batches
        and update the table so that the dataset only includes examples according to the filter function.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            function (`Callable`): Callable with one of the following signatures:

                - `function(example: Dict[str, Any]) -> bool` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> bool` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, list]) -> list[bool]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, list], *extra_args) -> list[bool]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                If no function is provided, defaults to an always `True` function: `lambda x: True`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the
                signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`[Union[str, list[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to `function`.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`
            num_proc (`int`, *optional*, defaults to `None`):
                 The number of processes to use for multiprocessing.
                - If `None` or `0`, no multiprocessing is used and the operation runs in the main process.
                - If greater than `1`, one or multiple worker processes are used to process data in parallel.
                 Note: The function passed to `map()` must be picklable for multiprocessing to work correctly
                 (i.e., prefer functions defined at the top level of a module, not inside another function or class).
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while filtering examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.filter(lambda x: x["label"] == 1)
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 4265
            })
            validation: Dataset({
                features: ['text', 'label'],
                num_rows: 533
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 533
            })
        })
        ```
        """
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.filter(
                    function=function,
                    with_indices=with_indices,
                    with_rank=with_rank,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    fn_kwargs=fn_kwargs,
                    num_proc=num_proc,
                    desc=desc,
                )
                for k, dataset in self.items()
            }
        )

    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        """Create and cache a new Dataset by flattening the indices mapping.

        Args:
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            cache_file_names (`Dict[str, str]`, *optional*, default `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Allow null values in the table.
            num_proc (`int`, optional, default `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
        """
        self._check_values_type()
        if cache_file_names is None:
            cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.flatten_indices(
                    keep_in_memory=keep_in_memory,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    features=features,
                    disable_nullable=disable_nullable,
                    num_proc=num_proc,
                    new_fingerprint=new_fingerprint,
                )
                for k, dataset in self.items()
            }
        )

    def sort(
        self,
        column_names: Union[str, Sequence[str]],
        reverse: Union[bool, Sequence[bool]] = False,
        null_placement: str = "at_end",
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        """Create a new dataset sorted according to a single or multiple columns.

        Args:
            column_names (`Union[str, Sequence[str]]`):
                Column name(s) to sort by.
            reverse (`Union[bool, Sequence[bool]]`, defaults to `False`):
                If `True`, sort by descending order rather than ascending. If a single bool is provided,
                the value is applied to the sorting of all column names. Otherwise a list of bools with the
                same length and order as column_names must be provided.
            null_placement (`str`, defaults to `at_end`):
                Put `None` values at the beginning if `at_start` or `first` or at the end if `at_end` or `last`
            keep_in_memory (`bool`, defaults to `False`):
                Keep the sorted indices in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the sorted indices
                can be identified, use it instead of recomputing.
            indices_cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                Higher value gives smaller cache files, lower value consume less temporary memory.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes')
        >>> ds['train']['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> sorted_ds = ds.sort('label')
        >>> sorted_ds['train']['label'][:10]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> another_sorted_ds = ds.sort(['label', 'text'], reverse=[True, False])
        >>> another_sorted_ds['train']['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ```
        """
        self._check_values_type()
        if indices_cache_file_names is None:
            indices_cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.sort(
                    column_names=column_names,
                    reverse=reverse,
                    null_placement=null_placement,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def shuffle(
        self,
        seeds: Optional[Union[int, dict[str, Optional[int]]]] = None,
        seed: Optional[int] = None,
        generators: Optional[dict[str, np.random.Generator]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        """Create a new Dataset where the rows are shuffled.

        The transformation is applied to all the datasets of the dataset dictionary.

        Currently shuffling uses numpy random generators.
        You can either supply a NumPy BitGenerator to use, or a seed to initiate NumPy's default random generator (PCG64).

        Args:
            seeds (`Dict[str, int]` or `int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`.
                If `None`, then fresh, unpredictable entropy will be pulled from the OS.
                If an `int` or `array_like[ints]` is passed, then it will be passed to SeedSequence to derive the initial BitGenerator state.
                You can provide one `seed` per dataset in the dataset dictionary.
            seed (`int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`. Alias for seeds (a `ValueError` is raised if both are provided).
            generators (`Dict[str, *optional*, np.random.Generator]`):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
                You have to provide one `generator` per dataset in the dataset dictionary.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            indices_cache_file_names (`Dict[str, str]`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                indices mappings instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds["train"]["label"][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # set a seed
        >>> shuffled_ds = ds.shuffle(seed=42)
        >>> shuffled_ds["train"]["label"][:10]
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        ```
        """
        self._check_values_type()
        if seed is not None and seeds is not None:
            raise ValueError("Please specify seed or seeds, but not both")
        seeds = seed if seed is not None else seeds
        if seeds is None:
            seeds = dict.fromkeys(self)
        elif not isinstance(seeds, dict):
            seeds = dict.fromkeys(self, seeds)
        if generators is None:
            generators = dict.fromkeys(self)
        if indices_cache_file_names is None:
            indices_cache_file_names = dict.fromkeys(self)
        return DatasetDict(
            {
                k: dataset.shuffle(
                    seed=seeds[k],
                    generator=generators[k],
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        """
        Saves a dataset dict to a filesystem using `fsspec.spec.AbstractFileSystem`.

        For [`Image`], [`Audio`] and [`Video`] data:

        All the Image(), Audio() and Video() data are stored in the arrow files.
        If you want to store paths or urls, please use the Value("string") type.

        Args:
            dataset_dict_path (`path-like`):
                Path (e.g. `dataset/train`) or remote URI (e.g. `s3://my-bucket/dataset/train`)
                of the dataset dict directory where the dataset dict will be saved to.
            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be saved to the filesystem. If expressed as a string, needs to be digits followed by a unit
                (like `"50MB"`).
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.
                You need to provide the number of shards for each dataset in the dataset dictionary.
                Use a dictionary to define a different num_shards for each split.

                <Added version="2.8.0"/>
            num_proc (`int`, *optional*, default `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Example:

        ```python
        >>> dataset_dict.save_to_disk("path/to/dataset/directory")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", max_shard_size="1GB")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", num_shards={"train": 1024, "test": 8})
        ```
        """
        fs: fsspec.AbstractFileSystem
        fs, _ = url_to_fs(dataset_dict_path, **(storage_options or {}))

        if num_shards is None:
            num_shards = dict.fromkeys(self)
        elif not isinstance(num_shards, dict):
            raise ValueError(
                "Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}"
            )

        fs.makedirs(dataset_dict_path, exist_ok=True)

        with fs.open(
            posixpath.join(dataset_dict_path, config.DATASETDICT_JSON_FILENAME),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({"splits": list(self)}, f)
        for k, dataset in self.items():
            dataset.save_to_disk(
                posixpath.join(dataset_dict_path, k),
                num_shards=num_shards.get(k),
                max_shard_size=max_shard_size,
                num_proc=num_proc,
                storage_options=storage_options,
            )

    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "DatasetDict":
        """
        Load a dataset that was previously saved using [`save_to_disk`] from a filesystem using `fsspec.spec.AbstractFileSystem`.

        Args:
            dataset_dict_path (`path-like`):
                Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3//my-bucket/dataset/train"`)
                of the dataset dict directory where the dataset dict will be loaded from.
            keep_in_memory (`bool`, defaults to `None`):
                Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                [improve performance](../cache#improve-performance) section.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> ds = load_from_disk('path/to/dataset/directory')
        ```
        """
        fs: fsspec.AbstractFileSystem
        fs, dataset_dict_path = url_to_fs(dataset_dict_path, **(storage_options or {}))

        dataset_dict_json_path = posixpath.join(dataset_dict_path, config.DATASETDICT_JSON_FILENAME)
        dataset_state_json_path = posixpath.join(dataset_dict_path, config.DATASET_STATE_JSON_FILENAME)
        dataset_info_path = posixpath.join(dataset_dict_path, config.DATASET_INFO_FILENAME)
        if not fs.isfile(dataset_dict_json_path):
            if fs.isfile(dataset_info_path) and fs.isfile(dataset_state_json_path):
                raise FileNotFoundError(
                    f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but got a `Dataset`. Please use either `datasets.load_from_disk` or `Dataset.load_from_disk` instead."
                )
            raise FileNotFoundError(
                f"No such file: '{dataset_dict_json_path}'. Expected to load a `DatasetDict` object, but provided path is not a `DatasetDict`."
            )

        with fs.open(dataset_dict_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]

        dataset_dict = DatasetDict()
        for k in splits:
            dataset_dict_split_path = posixpath.join(fs.unstrip_protocol(dataset_dict_path), k)
            dataset_dict[k] = Dataset.load_from_disk(
                dataset_dict_split_path,
                keep_in_memory=keep_in_memory,
                storage_options=storage_options,
            )
        return dataset_dict

    @staticmethod
    def from_csv(
        path_or_paths: dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from CSV file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (str, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`pandas.read_csv`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_csv({'train': 'path/to/dataset.csv'})
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.csv import CsvDatasetReader

        return CsvDatasetReader(
            path_or_paths,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            **kwargs,
        ).read()

    @staticmethod
    def from_json(
        path_or_paths: dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from JSON Lines file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the JSON Lines file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (str, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`JsonConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_json({'train': 'path/to/dataset.json'})
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.json import JsonDatasetReader

        return JsonDatasetReader(
            path_or_paths,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            **kwargs,
        ).read()

    @staticmethod
    def from_parquet(
        path_or_paths: dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[list[str]] = None,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from Parquet file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            columns (`list[str]`, *optional*):
                If not `None`, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`ParquetConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_parquet({'train': 'path/to/dataset/parquet'})
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.parquet import ParquetDatasetReader

        return ParquetDatasetReader(
            path_or_paths,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            columns=columns,
            **kwargs,
        ).read()

    @staticmethod
    def from_text(
        path_or_paths: dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from text file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the text file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`TextConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_text({'train': 'path/to/dataset.txt'})
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.text import TextDatasetReader

        return TextDatasetReader(
            path_or_paths,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            **kwargs,
        ).read()

    @is_documented_by(Dataset.align_labels_with_mapping)
    def align_labels_with_mapping(self, label2id: dict, label_column: str) -> "DatasetDict":
        self._check_values_type()
        return DatasetDict(
            {
                k: dataset.align_labels_with_mapping(label2id=label2id, label_column=label_column)
                for k, dataset in self.items()
            }
        )

    def push_to_hub(
        self,
        repo_id: str,
        config_name: str = "default",
        set_default: Optional[bool] = None,
        data_dir: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[dict[str, int]] = None,
        embed_external_files: bool = True,
        num_proc: Optional[int] = None,
    ) -> Optional[CommitInfo]:
        """Pushes the [`DatasetDict`] to the hub as a Parquet dataset.
        The [`DatasetDict`] is pushed using HTTP requests and does not need to have neither git or git-lfs installed.

        Each dataset split will be pushed independently. The pushed dataset will keep the original split names.

        The resulting Parquet files are self-contained by default: if your dataset contains [`Image`] or [`Audio`]
        data, the Parquet files will store the bytes of your images or audio files.
        You can disable this by setting `embed_external_files` to False.

        Args:
            repo_id (`str`):
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.

                It could also be a location inside a bucket, e.g. `buckets/<user_or_org>/<bucket_name>/...`
            config_name (`str`):
                Configuration name of a dataset. Defaults to "default".
            set_default (`bool`, *optional*):
                Whether to set this configuration as the default one. Otherwise, the default configuration is the one
                named "default".
            data_dir (`str`, *optional*):
                Directory name that will contain the uploaded data files. Defaults to the `config_name` if different
                from "default", else "data".

                <Added version="2.17.0"/>
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload dataset"`.
            commit_description (`str`, *optional*):
                Description of the commit that will be created.
                Additionally, description of the PR if a PR is created (`create_pr` is True).

                <Added version="2.16.0"/>
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the
                organization's default is private. This value is ignored if the repo already exists.
            token (`str`, *optional*):
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `hf auth login`. Will raise an error
                if no token is passed and the user is not logged-in.
            revision (`str`, *optional*):
                Branch to push the uploaded files to. Defaults to the `"main"` branch.

                <Added version="2.15.0"/>
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.

                <Added version="2.15.0"/>
            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                (like `"500MB"` or `"1GB"`).
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. By default, the number of shards depends on `max_shard_size`.
                Use a dictionary to define a different num_shards for each split.

                <Added version="2.8.0"/>
            embed_external_files (`bool`, defaults to `True`):
                Whether to embed file bytes in the shards.
                In particular, this will do the following before the push for the fields of type:

                - [`Audio`] and [`Image`] removes local path information and embed file content in the Parquet files.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when preparing and uploading the dataset.
                This is helpful if the dataset is made of many samples or media files to embed.
                I uses "spawn" context to work with hf_xet, the rust client for fast uploads to HF.
                Multiprocessing is disabled by default.

                <Added version="4.0.0"/>

        Return:
            huggingface_hub.CommitInfo

        Example:

        ```python
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", private=True)
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", max_shard_size="1GB")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", num_shards={"train": 1024, "test": 8})
        ```

        If you want to add a new configuration (or subset) to a dataset (e.g. if the dataset has multiple tasks/versions/languages):

        ```python
        >>> english_dataset.push_to_hub("<organization>/<dataset_id>", "en")
        >>> french_dataset.push_to_hub("<organization>/<dataset_id>", "fr")
        >>> # later
        >>> english_dataset = load_dataset("<organization>/<dataset_id>", "en")
        >>> french_dataset = load_dataset("<organization>/<dataset_id>", "fr")
        ```
        """
        if num_shards is None:
            num_shards = dict.fromkeys(self)
        elif not isinstance(num_shards, dict):
            raise ValueError(
                "Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}"
            )

        self._check_values_type()
        self._check_values_features()
        if config_name == "data":
            raise ValueError("`config_name` cannot be 'data'. Please, choose another name for configuration.")

        if max_shard_size is not None and num_shards is not None and any(x is not None for x in num_shards.values()):
            raise ValueError(
                "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
            )

        for split in self:
            if not re.match(_split_re, split):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split}'.")

        if not data_dir:
            data_dir = config_name if config_name != "default" else "data"  # for backward compatibility

        api = HfApi(endpoint=config.HF_ENDPOINT, token=token, library_name="datasets", library_version=__version__)
        if repo_id.startswith("buckets/"):
            _, _namespace, _bucket_name, *_path_segments = repo_id.split("/")
            try:
                bucket_id = api.bucket_info(_namespace + "/" + _bucket_name).id
            except BucketNotFoundError:
                bucket_url = api.create_bucket(_namespace + "/" + _bucket_name, private=private, exist_ok=True)
                bucket_id = bucket_url.bucket_id
            path = "/".join(s for s in _path_segments if s)
            return _push_to_bucket(
                self,
                bucket_id=bucket_id,
                path=path,
                config_name=config_name,
                set_default=set_default,
                data_dir=data_dir,
                token=token,
                max_shard_size=max_shard_size,
                num_shards=num_shards,
                embed_external_files=embed_external_files,
                num_proc=num_proc,
            )
        else:
            try:
                repo_id = api.repo_info(repo_id, repo_type="dataset").id
            except RepositoryNotFoundError:
                repo_url = api.create_repo(
                    repo_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True,
                )
                repo_id = repo_url.repo_id

            if revision is not None and not revision.startswith("refs/pr/"):
                # We do not call create_branch for a PR reference: 400 Bad Request
                api.create_branch(repo_id, branch=revision, repo_type="dataset", exist_ok=True)
            return _push_to_repo(
                self,
                repo_id=repo_id,
                config_name=config_name,
                set_default=set_default,
                data_dir=data_dir,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                revision=revision,
                create_pr=create_pr,
                max_shard_size=max_shard_size,
                num_shards=num_shards,
                embed_external_files=embed_external_files,
                num_proc=num_proc,
            )


class IterableDatasetDict(dict[Union[str, NamedSplit], IterableDataset]):
    def _check_values_type(self):
        for dataset in self.values():
            if not isinstance(dataset, IterableDataset):
                raise TypeError(f"Values in `DatasetDict` should be of type `Dataset` but got type '{type(dataset)}'")

    def _check_values_features(self):
        items = [(key, dataset._resolve_features()) for key, dataset in self.items()]
        for item_a, item_b in zip(items[:-1], items[1:]):
            if item_a[1].features != item_b[1].features:
                raise ValueError(
                    f"All datasets in `DatasetDict` should have the same features but features for '{item_a[0]}' and '{item_b[0]}' don't match: {item_a[1].features} != {item_b[1].features}"
                )

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, count=0, flags=re.M)
        return f"IterableDatasetDict({{\n{repr}\n}})"

    @property
    def num_columns(self) -> dict[str, Optional[int]]:
        """Number of columns in each split of the dataset.
        This can contain None valies if some splits have unknown features (e.g. after a map() operation).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.num_columns
        {'test': 2, 'train': 2, 'validation': 2}
        ```
        """
        self._check_values_type()
        return {k: dataset.num_columns for k, dataset in self.items()}

    @property
    def column_names(self) -> dict[str, Optional[list[str]]]:
        """Names of the columns in each split of the dataset.
        This can contain None valies if some splits have unknown features (e.g. after a map() operation).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        >>> ds.column_names
        {'test': ['text', 'label'],
         'train': ['text', 'label'],
         'validation': ['text', 'label']}
        ```
        """
        self._check_values_type()
        return {k: dataset.column_names for k, dataset in self.items()}

    def with_format(
        self,
        type: Optional[str] = None,
    ) -> "IterableDatasetDict":
        """
        Return a dataset with the specified format.

        Args:

            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'jax', 'arrow', 'pandas', 'polars']`.
                `None` means it returns python objects (default).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation", streaming=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds = ds.with_format("torch")
        >>> next(iter(ds))
        {'text': 'compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .',
         'label': tensor(1),
         'input_ids': tensor([  101, 18027, 16310, 16001,  1103,  9321,   178, 11604,  7235,  6617,
                1742,  2165,  2820,  1206,  6588, 22572, 12937,  1811,  2153,  1105,
                1147, 12890, 19587,  6463,  1105, 15026,  1482,   119,   102,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
        ```
        """
        return IterableDatasetDict({k: dataset.with_format(type=type) for k, dataset in self.items()})

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_split: bool = False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, list[str]]] = None,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDatasetDict":
        """
        Apply a function to all the examples in the iterable dataset (individually or in batches) and update them.
        If your function returns a column that already exists, then it overwrites it.
        The function is applied on-the-fly on the examples when iterating over the dataset.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
        - If batched is `True` and `batch_size` is `n` > 1, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.

        If the function is asynchronous, then `map` will run your function in parallel, with up to one thousand simultaneous calls.
        It is recommended to use a `asyncio.Semaphore` in your function if you want to set a maximum number of operations that can run at the same time.

        Args:
            function (`Callable`, *optional*, defaults to `None`):
                Function applied on-the-fly on the examples when you iterate on the dataset.
                It must have one of the following signatures:

                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False`
                - `function(example: Dict[str, Any], idx: int) -> Dict[str, Any]` if `batched=False` and `with_indices=True`
                - `function(batch: Dict[str, list]) -> Dict[str, list]` if `batched=True` and `with_indices=False`
                - `function(batch: Dict[str, list], indices: list[int]) -> Dict[str, list]` if `batched=True` and `with_indices=True`

                For advanced usage, the function can also return a `pyarrow.Table`.
                If the function is asynchronous, then `map` will run your function in parallel.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: `lambda x: x`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            input_columns (`[Union[str, list[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function`
                as positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the `batch_size` should be
                dropped instead of being processed by the function.
            remove_columns (`[list[str]]`, *optional*, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> next(iter(ds["train"]))
        {'label': 1,
         'text': 'Review: the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """

        dataset_dict = {}
        for split, dataset in self.items():
            if with_split:
                function = bind(function, split)

            dataset_dict[split] = dataset.map(
                function=function,
                with_indices=with_indices,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=remove_columns,
                fn_kwargs=fn_kwargs,
            )

            if with_split:
                function = function.func

        return IterableDatasetDict(dataset_dict)

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, list[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
    ) -> "IterableDatasetDict":
        """Apply a filter function to all the elements so that the dataset only includes examples according to the filter function.
        The filtering is done on-the-fly when iterating over the dataset.
        The filtering is applied to all the datasets of the dataset dictionary.

        Args:
            function (`Callable`):
                Callable with one of the following signatures:

                - `function(example: Dict[str, Any]) -> bool` if `with_indices=False, batched=False`
                - `function(example: Dict[str, Any], indices: int) -> bool` if `with_indices=True, batched=False`
                - `function(example: Dict[str, list]) -> list[bool]` if `with_indices=False, batched=True`
                - `function(example: Dict[str, list], indices: list[int]) -> list[bool]` if `with_indices=True, batched=True`

                If no function is provided, defaults to an always True function: `lambda x: True`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx): ...`.
            input_columns (`str` or `list[str]`, *optional*):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds = ds.filter(lambda x: x["label"] == 0)
        >>> list(ds["train"].take(3))
        [{'label': 0, 'text': 'Review: simplistic , silly and tedious .'},
         {'label': 0,
         'text': "Review: it's so laddish and juvenile , only teenage boys could possibly find it funny ."},
         {'label': 0,
         'text': 'Review: exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .'}]
        ```
        """
        return IterableDatasetDict(
            {
                k: dataset.filter(
                    function=function,
                    with_indices=with_indices,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    fn_kwargs=fn_kwargs,
                )
                for k, dataset in self.items()
            }
        )

    def shuffle(
        self,
        seed=None,
        generator: Optional[np.random.Generator] = None,
        buffer_size: int = 1000,
        max_buffer_input_shards: int = 10,
    ) -> "IterableDatasetDict":
        """
        Randomly shuffles the elements of this dataset.
        The shuffling is applied to all the datasets of the dataset dictionary.

        This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer,
        replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or
        equal to the full size of the dataset is required.

        For instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1000, then `shuffle` will
        initially select a random element from only the first 1000 elements in the buffer. Once an element is
        selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element,
        maintaining the 1000 element buffer.

        If the dataset is made of several shards, it also does `shuffle` the order of the shards.
        However if the order has been fixed by using [`~datasets.IterableDataset.skip`] or [`~datasets.IterableDataset.take`]
        then the order of the shards is kept unchanged.

        Args:
            seed (`int`, *optional*, defaults to `None`):
                Random seed that will be used to shuffle the dataset.
                It is used to sample from the shuffle buffer and also to shuffle the data shards.
            generator (`numpy.random.Generator`, *optional*):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
            buffer_size (`int`, defaults to `1000`):
                Size of the buffer.
            max_buffer_input_shards (`int`, defaults to `10`):
                Maximum number of shards to use to feed the buffer at a time.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> list(ds["train"].take(3))
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'}]
        >>> ds = ds.shuffle(seed=42)
        >>> list(ds["train"].take(3))
        [{'label': 1,
         'text': "a sports movie with action that's exciting on the field and a story you care about off it ."},
         {'label': 1,
         'text': 'at its best , the good girl is a refreshingly adult take on adultery . . .'},
         {'label': 1,
         'text': "sam jones became a very lucky filmmaker the day wilco got dropped from their record label , proving that one man's ruin may be another's fortune ."}]
        ```
        """
        return IterableDatasetDict(
            {
                k: dataset.shuffle(
                    seed=seed,
                    generator=generator,
                    buffer_size=buffer_size,
                    max_buffer_input_shards=max_buffer_input_shards,
                )
                for k, dataset in self.items()
            }
        )

    def rename_column(self, original_column_name: str, new_column_name: str) -> "IterableDatasetDict":
        """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with a renamed column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_column("text", "movie_review")
        >>> next(iter(ds["train"]))
        {'label': 1,
         'movie_review': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict(
            {
                k: dataset.rename_column(
                    original_column_name=original_column_name,
                    new_column_name=new_column_name,
                )
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: dict[str, str]) -> "IterableDatasetDict":
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with renamed columns

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_columns({"text": "movie_review", "label": "rating"})
        >>> next(iter(ds["train"]))
        {'movie_review': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
         'rating': 1}
        ```
        """
        return IterableDatasetDict(
            {k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()}
        )

    def remove_columns(self, column_names: Union[str, list[str]]) -> "IterableDatasetDict":
        """
        Remove one or several column(s) in the dataset and the features associated to them.
        The removal is done on-the-fly on the examples when iterating over the dataset.
        The removal is applied to all the datasets of the dataset dictionary.


        Args:
            column_names (`Union[str, list[str]]`):
                Name of the column(s) to remove.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset object without the columns to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds = ds.remove_columns("label")
        >>> next(iter(ds["train"]))
        {'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.remove_columns(column_names) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, list[str]]) -> "IterableDatasetDict":
        """Select one or several column(s) in the dataset and the features
        associated to them. The selection is done on-the-fly on the examples
        when iterating over the dataset. The selection is applied to all the
        datasets of the dataset dictionary.


        Args:
            column_names (`Union[str, list[str]]`):
                Name of the column(s) to keep.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset object with only selected columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds = ds.select("text")
        >>> next(iter(ds["train"]))
        {'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.select_columns(column_names) for k, dataset in self.items()})

    def cast_column(self, column: str, feature: FeatureType) -> "IterableDatasetDict":
        """Cast column to feature for decoding.
        The type casting is applied to all the datasets of the dataset dictionary.

        Args:
            column (`str`):
                Column name.
            feature ([`Feature`]):
                Target feature.

        Returns:
            [`IterableDatasetDict`]

        Example:

        ```py
        >>> from datasets import load_dataset, ClassLabel
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds["train"].features
        {'label': ClassLabel(names=['neg', 'pos']),
         'text': Value('string')}
        >>> ds = ds.cast_column('label', ClassLabel(names=['bad', 'good']))
        >>> ds["train"].features
        {'label': ClassLabel(names=['bad', 'good']),
         'text': Value('string')}
        ```
        """
        return IterableDatasetDict(
            {k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()}
        )

    def cast(
        self,
        features: Features,
    ) -> "IterableDatasetDict":
        """
        Cast the dataset to a new set of features.
        The type casting is applied to all the datasets of the dataset dictionary.

        Args:
            features (`Features`):
                New features to cast the dataset to.
                The name of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`map`] to update the Dataset.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with casted features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", streaming=True)
        >>> ds["train"].features
        {'label': ClassLabel(names=['neg', 'pos']),
         'text': Value('string')}
        >>> new_features = ds["train"].features.copy()
        >>> new_features['label'] = ClassLabel(names=['bad', 'good'])
        >>> new_features['text'] = Value('large_string')
        >>> ds = ds.cast(new_features)
        >>> ds["train"].features
        {'label': ClassLabel(names=['bad', 'good']),
         'text': Value('large_string')}
        ```
        """
        return IterableDatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})

    def push_to_hub(
        self,
        repo_id: str,
        config_name: str = "default",
        set_default: Optional[bool] = None,
        data_dir: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[dict[str, int]] = None,
        embed_external_files: bool = True,
        num_proc: Optional[int] = None,
    ) -> CommitInfo:
        """Pushes the [`IterableDatasetDict`] to the hub as a Parquet dataset.
        The [`IterableDatasetDict`] is pushed using HTTP requests and does not need to have neither git or git-lfs installed.

        Each dataset split will be pushed independently. The pushed dataset will keep the original split names.

        The resulting Parquet files are self-contained by default: if your dataset contains [`Image`] or [`Audio`]
        data, the Parquet files will store the bytes of your images or audio files.
        You can disable this by setting `embed_external_files` to False.

        Args:
            repo_id (`str`):
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.

                It could also be a location inside a bucket, e.g. `buckets/<user_or_org>/<bucket_name>/...`
            config_name (`str`):
                Configuration name of a dataset. Defaults to "default".
            set_default (`bool`, *optional*):
                Whether to set this configuration as the default one. Otherwise, the default configuration is the one
                named "default".
            data_dir (`str`, *optional*):
                Directory name that will contain the uploaded data files. Defaults to the `config_name` if different
                from "default", else "data".

                <Added version="2.17.0"/>
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload dataset"`.
            commit_description (`str`, *optional*):
                Description of the commit that will be created.
                Additionally, description of the PR if a PR is created (`create_pr` is True).

                <Added version="2.16.0"/>
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the
                organization's default is private. This value is ignored if the repo already exists.
            token (`str`, *optional*):
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `hf auth login`. Will raise an error
                if no token is passed and the user is not logged-in.
            revision (`str`, *optional*):
                Branch to push the uploaded files to. Defaults to the `"main"` branch.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
            max_shard_size (`int` or `str`, *optional*):
                Optional maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed
                by a unit (like `"500MB"` or `"1GB"`). If not provided, each split keeps its dataset-native shard count.
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. Use a dictionary to define a different `num_shards` for each split.
                If `max_shard_size` is provided and a split's `num_shards` is not, then the number of shards for that split is estimated
                from `max_shard_size`.
            embed_external_files (`bool`, defaults to `True`):
                Whether to embed file bytes in the shards.
                In particular, this will do the following before the push for the fields of type:

                - [`Audio`] and [`Image`] removes local path information and embed file content in the Parquet files.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when preparing and uploading the dataset.
                This is helpful if the dataset is made of many samples or media files to embed.
                I uses "spawn" context to work with hf_xet, the rust client for fast uploads to HF.
                Multiprocessing is disabled by default.

                <Added version="4.0.0"/>

        Return:
            huggingface_hub.CommitInfo

        Example:

        ```python
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", private=True)
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", max_shard_size="1GB")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", num_shards={"train": 1024, "test": 8})
        ```

        If you want to add a new configuration (or subset) to a dataset (e.g. if the dataset has multiple tasks/versions/languages):

        ```python
        >>> english_dataset.push_to_hub("<organization>/<dataset_id>", "en")
        >>> french_dataset.push_to_hub("<organization>/<dataset_id>", "fr")
        >>> # later
        >>> english_dataset = load_dataset("<organization>/<dataset_id>", "en")
        >>> french_dataset = load_dataset("<organization>/<dataset_id>", "fr")
        ```
        """
        # check to make sure that the user doesnt specify the # of shards and max shard sdize at same time, since these are 2 different ways to specify the same thing
        if max_shard_size is not None and num_shards is not None:
            raise ValueError(
                "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
            )

        if num_shards is None:
            num_shards = dict.fromkeys(self)
        elif not isinstance(num_shards, dict):
            raise ValueError(
                "Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}"
            )

        self._check_values_type()
        self._check_values_features()
        if config_name == "data":
            raise ValueError("`config_name` cannot be 'data'. Please, choose another name for configuration.")

        # if max_shard_size is not None and num_shards is not None and any(x is not None for x in num_shards.values()):
        #     raise ValueError(
        #         "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
        #     )

        for split in self:
            if not re.match(_split_re, split):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split}'.")

        if not data_dir:
            data_dir = config_name if config_name != "default" else "data"  # for backward compatibility

        api = HfApi(endpoint=config.HF_ENDPOINT, token=token, library_name="datasets", library_version=__version__)
        if repo_id.startswith("buckets/"):
            _, _namespace, _bucket_name, *_path_segments = repo_id.split("/")
            try:
                bucket_id = api.bucket_info(_namespace + "/" + _bucket_name).id
            except BucketNotFoundError:
                bucket_url = api.create_bucket(_namespace + "/" + _bucket_name, private=private, exist_ok=True)
                bucket_id = bucket_url.bucket_id
            path = "/".join(s for s in _path_segments if s)
            return _push_to_bucket(
                self,
                bucket_id=bucket_id,
                path=path,
                config_name=config_name,
                set_default=set_default,
                data_dir=data_dir,
                token=token,
                max_shard_size=max_shard_size,
                num_shards=num_shards.get(split),
                embed_external_files=embed_external_files,
                num_proc=num_proc,
            )
        else:
            try:
                repo_id = api.repo_info(repo_id, repo_type="dataset").id
            except RepositoryNotFoundError:
                repo_url = api.create_repo(
                    repo_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True,
                )
                repo_id = repo_url.repo_id

            if revision is not None and not revision.startswith("refs/pr/"):
                # We do not call create_branch for a PR reference: 400 Bad Request
                api.create_branch(repo_id, branch=revision, repo_type="dataset", exist_ok=True)
            return _push_to_repo(
                self,
                repo_id=repo_id,
                config_name=config_name,
                set_default=set_default,
                data_dir=data_dir,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                create_pr=create_pr,
                revision=revision,
                max_shard_size=max_shard_size,
                num_shards=num_shards,
                embed_external_files=embed_external_files,
                num_proc=num_proc,
            )


def _push_to_repo(
    dset_dict: Union["DatasetDict", "IterableDatasetDict"],
    repo_id: str,
    config_name: str = "default",
    set_default: Optional[bool] = None,
    data_dir: Optional[str] = None,
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    create_pr: Optional[bool] = False,
    max_shard_size: Optional[Union[int, str]] = None,
    num_shards: Optional[dict[str, Optional[int]]] = None,
    embed_external_files: bool = True,
    num_proc: Optional[int] = None,
) -> CommitInfo:
    api = HfApi(endpoint=config.HF_ENDPOINT, token=token, library_name="datasets", library_version=__version__)
    resolved_output_path = HfFileSystemResolvedRepositoryPath(
        repo_id=repo_id, repo_type="dataset", revision=revision or "main", path_in_repo=""
    )

    additions: list[CommitOperationAdd] = []
    new_parquet_paths: list[str] = []
    splits_info: list[SplitInfo] = []
    uploaded_sizes: list[int] = []

    for split in dset_dict:
        split_additions, split_new_parquet_paths, features, split_info, uploaded_size = dset_dict[
            split
        ]._push_parquet_shards_to_hub(
            resolved_output_path=resolved_output_path,
            data_dir=data_dir,
            split=split,
            token=token,
            max_shard_size=max_shard_size,
            num_shards=(num_shards or {}).get(split),
            create_pr=create_pr,
            embed_external_files=embed_external_files,
            num_proc=num_proc,
        )
        additions += split_additions
        new_parquet_paths += split_new_parquet_paths
        splits_info.append(split_info)
        uploaded_sizes.append(uploaded_size)

    commit_message = commit_message if commit_message is not None else "Upload dataset"
    if len(additions) > config.UPLOADS_MAX_NUMBER_PER_COMMIT:
        logger.info(
            f"Number of files to upload is larger than {config.UPLOADS_MAX_NUMBER_PER_COMMIT}. Splitting the push into multiple commits."
        )
        num_commits = math.ceil(len(additions) / config.UPLOADS_MAX_NUMBER_PER_COMMIT)
        for i in range(0, num_commits):
            operations = additions[
                i * config.UPLOADS_MAX_NUMBER_PER_COMMIT : (i + 1) * config.UPLOADS_MAX_NUMBER_PER_COMMIT
            ]
            for retry, sleep_time in enumerate(itertools.chain(range(10), itertools.repeat(30)), start=1):
                # We need to retry if another commit happens at the same time
                sleep_time *= 1 + random.random()
                try:
                    commit_info = api.create_commit(
                        repo_id,
                        operations=operations,
                        commit_message=commit_message + f" (part {i:05d}-of-{num_commits:05d})",
                        commit_description=commit_description,
                        repo_type="dataset",
                        revision=revision,
                        create_pr=create_pr,
                    )
                except HfHubHTTPError as err:
                    if (
                        err.__context__
                        and isinstance(err.__context__, HfHubHTTPError)
                        and err.__context__.response.status_code == 409
                    ):
                        # 409 is Conflict (another commit is in progress)
                        time.sleep(sleep_time)
                        logger.info(
                            f"Retrying intermediate commit for {repo_id}, {config_name} ({retry}/n with status_code {err.__context__.response.status_code})"
                        )
                        continue
                    else:
                        raise
                break
            logger.info(
                f"Commit #{i + 1} completed"
                + (f" (still {num_commits - i - 1} to go)" if num_commits - i - 1 else "")
                + "."
            )
        last_commit_additions = []
    else:
        last_commit_additions = additions

    for retry, sleep_time in enumerate(itertools.chain(range(10), itertools.repeat(30)), start=1):
        # We need to retry if there was a commit in between in case it touched the dataset card data
        sleep_time *= 1 + random.random()

        # We make sure to get info from this commit
        parent_commit = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
        hf_path = HfFileSystemResolvedRepositoryPath(
            repo_type=resolved_output_path.repo_type,
            repo_id=resolved_output_path.repo_id,
            revision=parent_commit,
            path_in_repo=resolved_output_path.path_in_repo,
        ).unresolve()
        hffs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=token)
        dirfs = DirFileSystem(fs=hffs, path=hf_path)

        # Check the files to delete
        files_to_delete = list(dirfs.glob(f"{data_dir}/*"))

        # Don't delete the new files
        deletions = [
            CommitOperationDelete(path_in_repo=file_to_delete)
            for file_to_delete in files_to_delete
            if file_to_delete not in new_parquet_paths
        ]

        # Update the dataset card
        new_dataset_card, new_legacy_dataset_infos = _get_updated_dataset_card(
            fs=dirfs,
            config_name=config_name,
            splits_info=splits_info,
            features=features,
            data_dir=data_dir,
            set_default=set_default,
            uploaded_sizes=uploaded_sizes,
            deleted_sizes=[],
            remove_other_splits=True,
        )
        dataset_card_additions = [
            CommitOperationAdd(path_in_repo=config.REPOCARD_FILENAME, path_or_fileobj=str(new_dataset_card).encode())
        ]
        if new_legacy_dataset_infos:
            dataset_card_additions.append(
                CommitOperationAdd(
                    path_in_repo=config.DATASETDICT_INFOS_FILENAME,
                    path_or_fileobj=json.dumps(new_legacy_dataset_infos).encode("utf-8"),
                )
            )
        operations = last_commit_additions + dataset_card_additions + deletions

        try:
            commit_info = api.create_commit(
                repo_id,
                operations=operations,
                commit_message=commit_message,
                commit_description=commit_description,
                repo_type="dataset",
                revision=revision,
                create_pr=create_pr,
                parent_commit=parent_commit,
            )
        except HfHubHTTPError as err:
            if (
                err.__context__
                and isinstance(err.__context__, HfHubHTTPError)
                and err.__context__.response.status_code in (412, 409)
            ):
                # 412 is Precondition failed (parent_commit isn't satisfied)
                # 409 is Conflict (another commit is in progress)
                time.sleep(sleep_time)
                logger.info(
                    f"Retrying commit for {repo_id}, {config_name} ({retry}/n with status_code {err.__context__.response.status_code})"
                )
                continue
            else:
                raise
        break

    return commit_info


def _push_to_bucket(
    dset_dict: Union["DatasetDict", "IterableDatasetDict"],
    bucket_id: str,
    path: str,
    config_name: str = "default",
    set_default: Optional[bool] = None,
    data_dir: Optional[str] = None,
    token: Optional[str] = None,
    max_shard_size: Optional[Union[int, str]] = None,
    num_shards: Optional[dict[str, Optional[int]]] = None,
    embed_external_files: bool = True,
    num_proc: Optional[int] = None,
) -> None:
    api = HfApi(endpoint=config.HF_ENDPOINT, token=token, library_name="datasets", library_version=__version__)
    resolved_output_path = HfFileSystemResolvedBucketPath(bucket_id=bucket_id, path=path)
    hf_path = resolved_output_path.unresolve()
    hffs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=token)
    dirfs = DirFileSystem(fs=hffs, path=hf_path)

    additions: list[CommitOperationAdd] = []
    new_parquet_paths: list[str] = []
    splits_info: list[SplitInfo] = []
    uploaded_sizes: list[int] = []

    # Check the files to delete before uploading
    files_to_delete = list(dirfs.glob(f"{data_dir}/*"))

    for split in dset_dict:
        # Upload the Parquet files
        split_additions, split_new_parquet_paths, features, split_info, uploaded_size = dset_dict[
            split
        ]._push_parquet_shards_to_hub(
            resolved_output_path=resolved_output_path,
            data_dir=data_dir,
            split=split,
            token=token,
            max_shard_size=max_shard_size,
