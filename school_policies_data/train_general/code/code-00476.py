        # We check that the bucket exists, and the directory exists, and authentication in one call
        api = HfApi(
            endpoint=config.HF_ENDPOINT,
            token=download_config.token,
            library_name="datasets",
            library_version=__version__,
            user_agent=get_datasets_user_agent(download_config.user_agent),
        )
        _, _namespace, _bucket_name, *_path_segments = path.split("/")
        bucket_id = _namespace + "/" + _bucket_name
        prefix = "/".join(s for s in _path_segments if s)
        try:
            next(iter(api.list_bucket_tree(bucket_id, prefix)))
        except (OfflineModeIsEnabled, httpx.ConnectError, httpx.TimeoutException) as e:
            raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
        except StopIteration as e:
            raise DatasetNotFoundError(f"Bucket directory at {path} doesn't exist") from e
        except BucketNotFoundError as e:
            raise DatasetNotFoundError(f"Bucket '{bucket_id}' doesn't exist on the Hub or cannot be accessed.") from e
        return HubBucketDatasetModuleFactory(
            path,
            data_dir=data_dir,
            data_files=data_files,
            download_config=download_config,
            download_mode=download_mode,
        ).get_module()
    elif is_relative_path(path) and path.count("/") <= 1:
        try:
            # Get the Dataset Card + get the revision + check authentication all at in one call
            # We fix the commit_hash in case there are new commits in the meantime
            api = HfApi(
                endpoint=config.HF_ENDPOINT,
                token=download_config.token,
                library_name="datasets",
                library_version=__version__,
                user_agent=get_datasets_user_agent(download_config.user_agent),
            )
            try:
                _raise_if_offline_mode_is_enabled()
                dataset_readme_path = api.hf_hub_download(
                    repo_id=path,
                    filename=config.REPOCARD_FILENAME,
                    repo_type="dataset",
                    revision=revision,
                )
                commit_hash = os.path.basename(os.path.dirname(dataset_readme_path))
            except LocalEntryNotFoundError as e:
                if isinstance(e.__cause__, (OfflineModeIsEnabled, httpx.ConnectError, httpx.TimeoutException)):
                    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
                else:
                    raise
            except EntryNotFoundError:
                commit_hash = api.dataset_info(
                    path,
                    revision=revision,
                    timeout=100.0,
                ).sha
            except (OfflineModeIsEnabled, httpx.ConnectError, httpx.TimeoutException) as e:
                raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
            except GatedRepoError as e:
                message = f"Dataset '{path}' is a gated dataset on the Hub."
                if e.response.status_code == 401:
                    message += " You must be authenticated to access it."
                elif e.response.status_code == 403:
                    message += f" Visit the dataset page at https://huggingface.co/datasets/{path} to ask for access."
                raise DatasetNotFoundError(message) from e
            except RepositoryNotFoundError as e:
                raise DatasetNotFoundError(f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed.") from e
            try:
                api.hf_hub_download(
                    repo_id=path,
                    filename=filename,
                    repo_type="dataset",
                    revision=commit_hash,
                )
                raise RuntimeError(f"Dataset scripts are no longer supported, but found {filename}")
            except EntryNotFoundError:
                # Use the infos from the parquet export except in some cases:
                if data_dir or data_files or (revision and revision != "main"):
                    use_exported_dataset_infos = False
                else:
                    use_exported_dataset_infos = True
                return HubDatasetModuleFactory(
                    path,
                    commit_hash=commit_hash,
                    data_dir=data_dir,
                    data_files=data_files,
                    download_config=download_config,
                    download_mode=download_mode,
                    use_exported_dataset_infos=use_exported_dataset_infos,
                ).get_module()
            except GatedRepoError as e:
                message = f"Dataset '{path}' is a gated dataset on the Hub."
                if e.response.status_code == 401:
                    message += " You must be authenticated to access it."
                elif e.response.status_code == 403:
                    message += f" Visit the dataset page at https://huggingface.co/datasets/{path} to ask for access."
                raise DatasetNotFoundError(message) from e
        except RevisionNotFoundError as e:
            raise DatasetNotFoundError(f"Revision '{revision}' doesn't exist for dataset '{path}' on the Hub.") from e
        except Exception as e1:
            # All the attempts failed, before raising the error we should check if the module is already cached
            try:
                return CachedDatasetModuleFactory(path, cache_dir=cache_dir).get_module()
            except Exception:
                # If it's not in the cache, then it doesn't exist.
                if isinstance(e1, OfflineModeIsEnabled):
                    raise ConnectionError(f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}") from None
                if isinstance(e1, (DataFilesNotFoundError, DatasetNotFoundError, EmptyDatasetError)):
                    raise e1 from None
                if isinstance(e1, FileNotFoundError):
                    raise FileNotFoundError(
                        f"Couldn't find any data file at {relative_to_absolute_path(path)}. "
                        f"Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}"
                    ) from None
                raise e1 from None
    else:
        raise FileNotFoundError(f"Couldn't find any data file at {relative_to_absolute_path(path)}.")


def load_dataset_builder(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs,
) -> DatasetBuilder:
    """Load a dataset builder which can be used to:

    - Inspect general information that is required to build a dataset (cache directory, config, dataset info, features, data files, etc.)
    - Download and prepare the dataset as Arrow files in the cache
    - Get a streaming dataset without downloading or caching anything

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
    in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, MeshFolder, etc.)

    Args:

        path (`str`):
            Path or name of the dataset.

            - if `path` is a dataset repository on the HF hub (list all available datasets with [`huggingface_hub.list_datasets`])
              -> load the dataset builder from supported files in the repository (csv, json, parquet, etc.)
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.

            - if `path` is a directory within a Storage Bucket on the HF Hub (list your buckets with [`huggingface_hub.list_buckets`])
              -> load the dataset from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'buckets/username/bucket_name/my_dataset'`.

            - if `path` is a local directory
              -> load the dataset builder from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'./path/to/directory/with/my/csv/data'`.

            - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified
              (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder", "meshfolder")
              -> load the dataset builder from the files in `data_files` or `data_dir`
              e.g. `'parquet'`.

            Use a `hf://` path like `'hf://datasets/username/dataset_name'` to allow remote only.
            Use an absolute path to allow local only.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features ([`Features`], *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>

        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`]
            and used in the [`DatasetBuilder`].

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(names=['neg', 'pos']),
     'text': Value('string')}
    ```
    """
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    if features is not None:
        features = _fix_for_backward_compatible_features(features)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    # Get dataset builder class
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop(
        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = (
        dataset_module.dataset_infos.get(config_name)
        if dataset_module.dataset_infos and not data_dir and not data_files
        else None
    )

    if (
        path in _PACKAGED_DATASETS_MODULES
        and data_files is None
        and dataset_module.builder_configs_parameters.builder_configs[0].data_files is None
    ):
        error_msg = f"Please specify the data files or data directory to load for the {path} dataset builder."
        example_extensions = [
            extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path
        ]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)

    # When users pass config kwargs, they should override module-provided defaults
    # instead of colliding at constructor call time.
    builder_kwargs = {key: value for key, value in builder_kwargs.items() if key not in config_kwargs}

    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        config_name=config_name,
        data_dir=data_dir,
        data_files=data_files,
        hash=dataset_module.hash,
        info=info,
        features=features,
        token=token,
        storage_options=storage_options,
        **builder_kwargs,
        **config_kwargs,
    )
    builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

    return builder_instance


@overload
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: None = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: Literal[False] = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs: Any,
) -> DatasetDict: ...


@overload
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    *,
    split: Union[str, Split, list[str], list[Split]],
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[Version, str]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: Literal[False] = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs: Any,
) -> Dataset: ...


@overload
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: None = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[Version, str]] = None,
    token: Optional[Union[bool, str]] = None,
    *,
    streaming: Literal[True],
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs: Any,
) -> IterableDatasetDict: ...


@overload
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    *,
    split: Union[str, Split, list[str], list[Split]],
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[Version, str]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: Literal[True],
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs: Any,
) -> IterableDataset: ...


def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split, list[str], list[Split]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load a dataset from the Hugging Face Hub, or a local dataset.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
    in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, MeshFolder, etc.)

    This function does the following under the hood:

        1. Load a dataset builder:

            * Find the most common data format in the dataset and pick its associated builder (JSON, CSV, Parquet, Webdataset, ImageFolder, AudioFolder, MeshFolder, etc.)
            * Find which file goes into which split (e.g. train/test) based on file and directory names or on the YAML configuration
            * It is also possible to specify `data_files` manually, and which dataset builder to use (e.g. "parquet").

        2. Run the dataset builder:

            In the general case:

            * Download the data files from the dataset if they are not already available locally or cached.
            * Process and cache the dataset in typed Arrow tables for caching.

                Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types.
                They can be directly accessed from disk, loaded in RAM or even streamed over the web.

            In the streaming case:

            * Don't download or cache anything. Instead, the dataset is lazily loaded and will be streamed on-the-fly when iterating on it.

        3. Return a dataset built from the requested splits in `split` (default: all).

    Args:

        path (`str`):
            Path or name of the dataset.

            - if `path` is a dataset repository on the HF hub (list all available datasets with [`huggingface_hub.list_datasets`])
              -> load the dataset from supported files in the repository (csv, json, parquet, etc.)
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.

            - if `path` is a directory within a Storage Bucket on the HF Hub (list your buckets with [`huggingface_hub.list_buckets`])
              -> load the dataset from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'buckets/username/bucket_name/my_dataset'`.

            - if `path` is a local directory
              -> load the dataset from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'./path/to/directory/with/my/csv/data'`.

            - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified
              (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder", "meshfolder")
              -> load the dataset from the files in `data_files` or `data_dir`
              e.g. `'parquet'`.

            Use a `hf://` path like `'hf://datasets/username/dataset_name'` to allow remote only.
            Use an absolute path to allow local only.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`Split` or `str`):
            Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features (`Features`, *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
            Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

            <Added version="2.9.1"/>
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        streaming (`bool`, defaults to `False`):
            If set to `True`, don't download the data files. Instead, it streams the data progressively while
            iterating on the dataset. An [`IterableDataset`] or [`IterableDatasetDict`] is returned instead in this case.

            Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
            Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
            like rar and xz are not yet supported. The tgz format doesn't allow streaming.
        num_proc (`int`, *optional*, defaults to `None`):
            Number of processes when downloading and generating the dataset locally.
            Multiprocessing is disabled by default.

            <Added version="2.7.0"/>
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the `BuilderConfig`
            and used in the [`DatasetBuilder`].

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - if `split` is not `None`: the dataset requested,
        - if `split` is `None`, a [`~datasets.DatasetDict`] with each split.

        or [`IterableDataset`] or [`IterableDatasetDict`]: if `streaming=True`

        - if `split` is not `None`, the dataset is requested
        - if `split` is `None`, a [`~datasets.streaming.IterableDatasetDict`] with each split.

    Example:

    Load a dataset from the Hugging Face Hub:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')

    # Load a subset or dataset configuration (here 'sst2')
    >>> from datasets import load_dataset
    >>> ds = load_dataset('nyu-mll/glue', 'sst2', split='train')

    # Manual mapping of data files to splits
    >>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
    >>> ds = load_dataset('namespace/your_dataset_name', data_files=data_files)

    # Manual selection of a directory to load
    >>> ds = load_dataset('namespace/your_dataset_name', data_dir='folder_name')
    ```

    Load a dataset from a Storage Bucket on the Hugging Face Hub:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('buckets/username/bucket_name/rotten_tomatoes', split='train')
    ```

    Load a local dataset:

    ```py
    # Load a CSV file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

    # Load a JSON file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')
    ```

    Load an [`~datasets.IterableDataset`]:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train', streaming=True)
    ```

    Load an image dataset with the `ImageFolder` dataset builder:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
    ```
    """
    if "trust_remote_code" in config_kwargs:
        if config_kwargs.pop("trust_remote_code"):
            logger.error(
                "`trust_remote_code` is not supported anymore.\n"
                f"Please check that the Hugging Face dataset '{path}' isn't based on a loading script and remove `trust_remote_code`.\n"
                "If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet."
            )
    if data_files is not None and not data_files:
        raise ValueError(f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default).")
    if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError(
            "You are trying to load a dataset that was saved using `save_to_disk`. "
            "Please use `load_from_disk` instead."
        )

    if streaming and num_proc is not None:
        raise NotImplementedError(
            "Loading a streaming dataset in parallel with `num_proc` is not implemented. "
            "To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead."
        )

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode(
        (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
    )

    # Create a dataset builder
    builder_instance = load_dataset_builder(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        storage_options=storage_options,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

    # Download and prepare data
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        num_proc=num_proc,
        storage_options=storage_options,
    )

    # Build dataset for splits
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, in_memory=keep_in_memory)

    return ds


def load_from_disk(
    dataset_path: PathLike, keep_in_memory: Optional[bool] = None, storage_options: Optional[dict] = None
) -> Union[Dataset, DatasetDict]:
    """
    Loads a dataset that was previously saved using [`~Dataset.save_to_disk`] from a dataset directory, or
    from a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

    Args:
        dataset_path (`path-like`):
            Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3://my-bucket/dataset/train"`)
            of the [`Dataset`] or [`DatasetDict`] directory where the dataset/dataset-dict will be
            loaded from.
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.

        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the file-system backend, if any.

            <Added version="2.9.0"/>

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - If `dataset_path` is a path of a dataset directory: the dataset requested.
        - If `dataset_path` is a path of a dataset dict directory, a [`DatasetDict`] with each split.

    Example:

    ```py
    >>> from datasets import load_from_disk
    >>> ds = load_from_disk('path/to/dataset/directory')
    ```
    """
    fs: fsspec.AbstractFileSystem
    fs, *_ = url_to_fs(dataset_path, **(storage_options or {}))
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(posixpath.join(dataset_path, config.DATASET_INFO_FILENAME)) and fs.isfile(
        posixpath.join(dataset_path, config.DATASET_STATE_JSON_FILENAME)
    ):
        return Dataset.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    elif fs.isfile(posixpath.join(dataset_path, config.DATASETDICT_JSON_FILENAME)):
        return DatasetDict.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    else:
        raise FileNotFoundError(
            f"Directory {dataset_path} is neither a `Dataset` directory nor a `DatasetDict` directory."
        )
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
"""Arrow ArrowReader."""

import copy
import math
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map

from .download.download_config import DownloadConfig  # noqa: F401
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm


if TYPE_CHECKING:
    from .info import DatasetInfo  # noqa: F401
    from .splits import Split, SplitInfo  # noqa: F401


logger = logging.get_logger(__name__)

HF_GCP_BASE_URL = "https://storage.googleapis.com/huggingface-nlp/cache/datasets"

_SUB_SPEC_RE = re.compile(
    rf"""
^
 (?P<split>{_split_re[1:-1]})
 (\[
    ((?P<from>-?[\d_]+)
     (?P<from_pct>%)?)?
    :
    ((?P<to>-?[\d_]+)
     (?P<to_pct>%)?)?
 \])?(\((?P<rounding>[^\)]*)\))?
$
""",  # remove ^ and $
    re.X,
)

_ADDITION_SEP_RE = re.compile(r"\s*\+\s*")


class DatasetNotOnHfGcsError(ConnectionError):
    """When you can't get the dataset from the Hf google cloud storage"""

    pass


class MissingFilesOnHfGcsError(ConnectionError):
    """When some files are missing on the Hf oogle cloud storage"""

    pass


@dataclass(frozen=True)
class FileInstructions:
    """The file instructions associated with a split ReadInstruction.

    Attributes:
        num_examples: `int`, The total number of examples
        file_instructions: List[dict(filename, skip, take)], the files information.
            The filenames contains the relative path, not absolute.
            skip/take indicates which example read in the file: `ds.slice(skip, take)`
    """

    num_examples: int
    file_instructions: list[dict]


def make_file_instructions(
    name: str,
    split_infos: list["SplitInfo"],
    instruction: Union[str, "ReadInstruction"],
    filetype_suffix: Optional[str] = None,
    prefix_path: Optional[str] = None,
) -> FileInstructions:
    """Returns instructions of the split dict.

    Args:
        name (`str`): Name of the dataset.
        split_infos (`list` of `[SplitInfo]`): Dataset splits information.
        instruction ([`ReadInstruction`] or `str`): Reading instruction for a dataset.
        filetype_suffix (`str`, *optional*): Suffix of dataset files, e.g. 'arrow' or 'parquet'.
        prefix_path (`str`, *optional*): Prefix of dataset files, e.g. directory name.

    Returns:
        [`FileInstructions`]
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected str 'name', but got: {type(name).__name__}")
    elif not name:
        raise ValueError("Expected non-empty str 'name'")
    name2len = {info.name: info.num_examples for info in split_infos}
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2filenames = {
        info.name: filenames_for_dataset_split(
            path=prefix_path,
            dataset_name=name,
            split=info.name,
            filetype_suffix=filetype_suffix,
            shard_lengths=name2shard_lengths[info.name],
        )
        for info in split_infos
    }
    if not isinstance(instruction, ReadInstruction):
        instruction = ReadInstruction.from_spec(instruction)
    # Create the absolute instruction (per split)
    absolute_instructions = instruction.to_absolute(name2len)

    # For each split, return the files instruction (skip/take)
    file_instructions = []
    num_examples = 0
    for abs_instr in absolute_instructions:
        split_length = name2len[abs_instr.splitname]
        filenames = name2filenames[abs_instr.splitname]
        shard_lengths = name2shard_lengths[abs_instr.splitname]
        from_ = 0 if abs_instr.from_ is None else abs_instr.from_
        to = split_length if abs_instr.to is None else abs_instr.to
        if shard_lengths is None:  # not sharded
            for filename in filenames:
                take = to - from_
                if take == 0:
                    continue
                num_examples += take
                file_instructions.append({"filename": filename, "skip": from_, "take": take})
        else:  # sharded
            index_start = 0  # Beginning (included) of moving window.
            index_end = 0  # End (excluded) of moving window.
            for filename, shard_length in zip(filenames, shard_lengths):
                index_end += shard_length
                if from_ < index_end and to > index_start:  # There is something to take.
                    skip = from_ - index_start if from_ > index_start else 0
                    take = to - index_start - skip if to < index_end else -1
                    if take == 0:
                        continue
                    file_instructions.append({"filename": filename, "skip": skip, "take": take})
                    num_examples += shard_length - skip if take == -1 else take
                index_start += shard_length
    return FileInstructions(
        num_examples=num_examples,
        file_instructions=file_instructions,
    )


class BaseReader:
    """
    Build a Dataset object out of Instruction instance(s).
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ArrowReader.

        Args:
            path (str): path where tfrecords are stored.
            info (DatasetInfo): info about the dataset.
        """
        self._path: str = path
        self._info: Optional["DatasetInfo"] = info
        self._filetype_suffix: Optional[str] = None

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        """Returns a Dataset instance from given (filename, skip, take)."""
        raise NotImplementedError

    def _read_files(self, files, in_memory=False) -> Table:
        """Returns Dataset for given file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contain the absolute path, not relative.
                skip/take indicates which example read in the file: `ds.slice(skip, take)`
            in_memory (bool, default False): Whether to copy the data in-memory.
        """
        if len(files) == 0 or not all(isinstance(f, dict) for f in files):
            raise ValueError("please provide valid file informations")
        files = copy.deepcopy(files)
        for f in files:
            f["filename"] = os.path.join(self._path, f["filename"])

        pa_tables = thread_map(
            partial(self._get_table_from_filename, in_memory=in_memory),
            files,
            tqdm_class=hf_tqdm,
            desc="Loading dataset shards",
            # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
            disable=len(files) <= 16 or None,
        )
        pa_tables = [t for t in pa_tables if len(t) > 0]
        if not pa_tables and (self._info is None or self._info.features is None):
            raise ValueError(
                "Tried to read an empty table. Please specify at least info.features to create an empty table with the right type."
            )
        pa_tables = pa_tables or [InMemoryTable.from_batches([], schema=pa.schema(self._info.features.type))]
        pa_table = concat_tables(pa_tables) if len(pa_tables) != 1 else pa_tables[0]
        return pa_table

    def get_file_instructions(self, name, instruction, split_infos):
        """Return list of dict {'filename': str, 'skip': int, 'take': int}"""
        file_instructions = make_file_instructions(
            name, split_infos, instruction, filetype_suffix=self._filetype_suffix, prefix_path=self._path
        )
        files = file_instructions.file_instructions
        return files

    def read(
        self,
        name,
        instructions,
        split_infos,
        in_memory=False,
    ):
        """Returns Dataset instance(s).

        Args:
            name (str): name of the dataset.
            instructions (ReadInstruction): instructions to read.
                Instruction can be string and will then be passed to the Instruction
                constructor as it.
            split_infos (list of SplitInfo proto): the available splits for dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
             kwargs to build a single Dataset instance.
        """

        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)

    def read_files(
        self,
        files: list[dict],
        original_instructions: Union[None, "ReadInstruction", "Split"] = None,
        in_memory=False,
    ):
        """Returns single Dataset instance for the set of file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contains the relative path, not absolute.
                skip/take indicates which example read in the file: `ds.skip().take()`
            original_instructions: store the original instructions used to build the dataset split in the dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
            kwargs to build a Dataset instance.
        """
        # Prepend path to filename
        pa_table = self._read_files(files, in_memory=in_memory)
        # If original_instructions is not None, convert it to a human-readable NamedSplit
        if original_instructions is not None:
            from .splits import Split  # noqa

            split = Split(str(original_instructions))
        else:
            split = None
        dataset_kwargs = {"arrow_table": pa_table, "info": self._info, "split": split}
        return dataset_kwargs


class ArrowReader(BaseReader):
    """
    Build a Dataset object out of Instruction instance(s).
    This Reader uses either memory mapping or file descriptors (in-memory) on arrow files.
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ArrowReader.

        Args:
            path (str): path where Arrow files are stored.
            info (DatasetInfo): info about the dataset.
        """
        super().__init__(path, info)
        self._filetype_suffix = "arrow"

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        """Returns a Dataset instance from given (filename, skip, take)."""
        filename, skip, take = (
            filename_skip_take["filename"],
            filename_skip_take["skip"] if "skip" in filename_skip_take else None,
            filename_skip_take["take"] if "take" in filename_skip_take else None,
        )
        table = ArrowReader.read_table(filename, in_memory=in_memory)
        if take == -1:
            take = len(table) - skip
        # here we don't want to slice an empty table, or it may segfault
        if skip is not None and take is not None and not (skip == 0 and take == len(table)):
            table = table.slice(skip, take)
        return table

    @staticmethod
    def read_table(filename, in_memory=False) -> Table:
        """
        Read table from file.

        Args:
            filename (str): File name of the table.
            in_memory (bool, default=False): Whether to copy the data in-memory.

        Returns:
            pyarrow.Table
        """
        table_cls = InMemoryTable if in_memory else MemoryMappedTable
        return table_cls.from_file(filename)


class ParquetReader(BaseReader):
    """
    Build a Dataset object out of Instruction instance(s).
    This Reader uses memory mapping on parquet files.
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ParquetReader.

        Args:
            path (str): path where tfrecords are stored.
            info (DatasetInfo): info about the dataset.
        """
        super().__init__(path, info)
        self._filetype_suffix = "parquet"

    def _get_table_from_filename(self, filename_skip_take, **kwargs):
        """Returns a Dataset instance from given (filename, skip, take)."""
        filename, skip, take = (
            filename_skip_take["filename"],
            filename_skip_take["skip"] if "skip" in filename_skip_take else None,
            filename_skip_take["take"] if "take" in filename_skip_take else None,
        )
        # Parquet read_table always loads data in memory, independently of memory_map
        pa_table = pq.read_table(filename, memory_map=True)
        # here we don't want to slice an empty table, or it may segfault
        if skip is not None and take is not None and not (skip == 0 and take == len(pa_table)):
            pa_table = pa_table.slice(skip, take)
        return pa_table


@dataclass(frozen=True)
class _AbsoluteInstruction:
    """A machine friendly slice: defined absolute positive boundaries."""

    splitname: str
    from_: int  # uint (starting index).
    to: int  # uint (ending index).


@dataclass(frozen=True)
class _RelativeInstruction:
    """Represents a single parsed slicing instruction, can use % and negatives."""

    splitname: str
    from_: Optional[int] = None  # int (starting index) or None if no lower boundary.
    to: Optional[int] = None  # int (ending index) or None if no upper boundary.
    unit: Optional[str] = None
    rounding: Optional[str] = None

    def __post_init__(self):
        if self.unit is not None and self.unit not in ["%", "abs"]:
            raise ValueError("unit must be either % or abs")
        if self.rounding is not None and self.rounding not in ["closest", "pct1_dropremainder"]:
            raise ValueError("rounding must be either closest or pct1_dropremainder")
        if self.unit != "%" and self.rounding is not None:
            raise ValueError("It is forbidden to specify rounding if not using percent slicing.")
        if self.unit == "%" and self.from_ is not None and abs(self.from_) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        if self.unit == "%" and self.to is not None and abs(self.to) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        # Update via __dict__ due to instance being "frozen"
        self.__dict__["rounding"] = "closest" if self.rounding is None and self.unit == "%" else self.rounding


def _str_to_read_instruction(spec):
    """Returns ReadInstruction for given string."""
    res = _SUB_SPEC_RE.match(spec)
    if not res:
        raise ValueError(f"Unrecognized instruction format: {spec}")
    unit = "%" if res.group("from_pct") or res.group("to_pct") else "abs"
    return ReadInstruction(
        split_name=res.group("split"),
        rounding=res.group("rounding"),
        from_=int(res.group("from")) if res.group("from") else None,
        to=int(res.group("to")) if res.group("to") else None,
        unit=unit,
    )


def _pct_to_abs_pct1(boundary, num_examples):
    # Using math.trunc here, since -99.5% should give -99%, not -100%.
    if num_examples < 100:
        msg = (
            'Using "pct1_dropremainder" rounding on a split with less than 100 '
            "elements is forbidden: it always results in an empty dataset."
        )
        raise ValueError(msg)
    return boundary * math.trunc(num_examples / 100.0)


def _pct_to_abs_closest(boundary, num_examples):
    return int(round(boundary * num_examples / 100.0))


def _rel_to_abs_instr(rel_instr, name2len):
    """Returns _AbsoluteInstruction instance for given RelativeInstruction.

    Args:
        rel_instr: RelativeInstruction instance.
        name2len: dict {split_name: num_examples}.
    """
    pct_to_abs = _pct_to_abs_closest if rel_instr.rounding == "closest" else _pct_to_abs_pct1
    split = rel_instr.splitname
    if split not in name2len:
        raise ValueError(f'Unknown split "{split}". Should be one of {list(name2len)}.')
    num_examples = name2len[split]
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == "%":
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    else:
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    if from_ < 0:
        from_ = max(num_examples + from_, 0)
    if to < 0:
        to = max(num_examples + to, 0)
    from_ = min(from_, num_examples)
    to = min(to, num_examples)
    return _AbsoluteInstruction(split, from_, to)


class ReadInstruction:
    """Reading instruction for a dataset.

    Examples::

      # The following lines are equivalent:
      ds = datasets.load_dataset('ylecun/mnist', split='test[:33%]')
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec('test[:33%]'))
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction('test', to=33, unit='%'))
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%'))

      # The following lines are equivalent:
      ds = datasets.load_dataset('ylecun/mnist', split='test[:33%]+train[1:-1]')
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%]+train[1:-1]'))
      ds = datasets.load_dataset('ylecun/mnist', split=(
          datasets.ReadInstruction('test', to=33, unit='%') +
          datasets.ReadInstruction('train', from_=1, to=-1, unit='abs')))

      # The following lines are equivalent:
      ds = datasets.load_dataset('ylecun/mnist', split='test[:33%](pct1_dropremainder)')
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%](pct1_dropremainder)'))
      ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%', rounding="pct1_dropremainder"))

      # 10-fold validation:
      tests = datasets.load_dataset(
          'ylecun/mnist',
          [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%')
          for k in range(0, 100, 10)])
      trains = datasets.load_dataset(
          'ylecun/mnist',
          [datasets.ReadInstruction('train', to=k, unit='%') + datasets.ReadInstruction('train', from_=k+10, unit='%')
          for k in range(0, 100, 10)])

    """

    def _init(self, relative_instructions):
        # Private initializer.
        self._relative_instructions = relative_instructions

    @classmethod
    def _read_instruction_from_relative_instructions(cls, relative_instructions):
        """Returns ReadInstruction obj initialized with relative_instructions."""
        # Use __new__ to bypass __init__ used by public API and not conveniant here.
        result = cls.__new__(cls)
        result._init(relative_instructions)  # pylint: disable=protected-access
        return result

    def __init__(self, split_name, rounding=None, from_=None, to=None, unit=None):
        """Initialize ReadInstruction.

        Args:
            split_name (str): name of the split to read. Eg: 'train'.
            rounding (str, optional): The rounding behaviour to use when percent slicing is
                used. Ignored when slicing with absolute indices.
                Possible values:
                 - 'closest' (default): The specified percentages are rounded to the
                     closest value. Use this if you want specified percents to be as
                     much exact as possible.
                 - 'pct1_dropremainder': the specified percentages are treated as
                     multiple of 1%. Use this option if you want consistency. Eg:
                         len(5%) == 5 * len(1%).
                     Using this option, one might not be able to use the full set of
                     examples, if the number of those is not a multiple of 100.
            from_ (int):
            to (int): alternative way of specifying slicing boundaries. If any of
                {from_, to, unit} argument is used, slicing cannot be specified as
                string.
            unit (str): optional, one of:
                '%': to set the slicing unit as percents of the split size.
                'abs': to set the slicing unit as absolute numbers.
        """
        # This constructor is not always called. See factory method
        # `_read_instruction_from_relative_instructions`. Common init instructions
        # MUST be placed in the _init method.
        self._init([_RelativeInstruction(split_name, from_, to, unit, rounding)])

    @classmethod
    def from_spec(cls, spec):
        """Creates a `ReadInstruction` instance out of a string spec.

        Args:
            spec (`str`):
                Split(s) + optional slice(s) to read + optional rounding
                if percents are used as the slicing unit. A slice can be specified,
                using absolute numbers (`int`) or percentages (`int`).

        Examples:

            ```
            test: test split.
            test + validation: test split + validation split.
            test[10:]: test split, minus its first 10 records.
            test[:10%]: first 10% records of test split.
            test[:20%](pct1_dropremainder): first 10% records, rounded with the pct1_dropremainder rounding.
            test[:-5%]+train[40%:60%]: first 95% of test + middle 20% of train.
            ```

        Returns:
            ReadInstruction instance.
        """
        spec = str(spec)  # Need to convert to str in case of NamedSplit instance.
        subs = _ADDITION_SEP_RE.split(spec)
        if not subs:
            raise ValueError(f"No instructions could be built out of {spec}")
        instruction = _str_to_read_instruction(subs[0])
        return sum((_str_to_read_instruction(sub) for sub in subs[1:]), instruction)

    def to_spec(self):
        rel_instr_specs = []
        for rel_instr in self._relative_instructions:
            rel_instr_spec = rel_instr.splitname
            if rel_instr.from_ is not None or rel_instr.to is not None:
                from_ = rel_instr.from_
                to = rel_instr.to
                unit = rel_instr.unit
                rounding = rel_instr.rounding
                unit = unit if unit == "%" else ""
                from_ = str(from_) + unit if from_ is not None else ""
                to = str(to) + unit if to is not None else ""
                slice_str = f"[{from_}:{to}]"
                rounding_str = (
                    f"({rounding})" if unit == "%" and rounding is not None and rounding != "closest" else ""
                )
                rel_instr_spec += slice_str + rounding_str
            rel_instr_specs.append(rel_instr_spec)
        return "+".join(rel_instr_specs)

    def __add__(self, other):
        """Returns a new ReadInstruction obj, result of appending other to self."""
        if not isinstance(other, ReadInstruction):
            msg = "ReadInstruction can only be added to another ReadInstruction obj."
            raise TypeError(msg)
        self_ris = self._relative_instructions
        other_ris = other._relative_instructions  # pylint: disable=protected-access
        if (
            self_ris[0].unit != "abs"
            and other_ris[0].unit != "abs"
            and self._relative_instructions[0].rounding != other_ris[0].rounding
        ):
            raise ValueError("It is forbidden to sum ReadInstruction instances with different rounding values.")
        return self._read_instruction_from_relative_instructions(self_ris + other_ris)

    def __str__(self):
        return self.to_spec()

    def __repr__(self):
        return f"ReadInstruction({self._relative_instructions})"

    def to_absolute(self, name2len):
        """Translate instruction into a list of absolute instructions.

        Those absolute instructions are then to be added together.

        Args:
            name2len (`dict`):
                Associating split names to number of examples.

        Returns:
            list of _AbsoluteInstruction instances (corresponds to the + in spec).
        """
        return [_rel_to_abs_instr(rel_instr, name2len) for rel_instr in self._relative_instructions]
<|endoftext|>import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import xxhash

from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.logging import get_logger


if TYPE_CHECKING:
    from .arrow_dataset import Dataset


logger = get_logger(__name__)


# Fingerprinting allows to have one deterministic fingerprint per dataset state.
# A dataset fingerprint is updated after each transform.
# Re-running the same transforms on a dataset in a different session results in the same fingerprint.
# This is possible thanks to a custom hashing function that works with most python objects.

# Fingerprinting is the main mechanism that enables caching.
# The caching mechanism allows to reload an existing cache file if it's already been computed.


#################
# Caching
#################

_CACHING_ENABLED = True
_TEMP_DIR_FOR_TEMP_CACHE_FILES: Optional["_TempCacheDir"] = None
_DATASETS_WITH_TABLE_IN_TEMP_DIR: Optional[weakref.WeakSet] = None


class _TempCacheDir:
    """
    A temporary directory for storing cached Arrow files with a cleanup that frees references to the Arrow files
    before deleting the directory itself to avoid permission errors on Windows.
    """

    def __init__(self):
        # Check if TMPDIR is set and handle the case where it doesn't exist
        tmpdir = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
        # Normalize the path to handle any path resolution issues
        if tmpdir:
            tmpdir = os.path.normpath(tmpdir)
            if not os.path.exists(tmpdir):
                # Auto-create the directory if it doesn't exist
                # This prevents tempfile from silently falling back to /tmp
                try:
                    os.makedirs(tmpdir, exist_ok=True)
                    logger.info(f"Created TMPDIR directory: {tmpdir}")
                except OSError as e:
                    raise OSError(
                        f"TMPDIR is set to '{tmpdir}' but the directory does not exist and could not be created: {e}. "
                        "Please create it manually or unset TMPDIR to fall back to the default temporary directory."
                    ) from e
            # If tmpdir exists, verify it's actually a directory and writable
            elif not os.path.isdir(tmpdir):
                raise OSError(
                    f"TMPDIR is set to '{tmpdir}' but it is not a directory. "
                    "Please point TMPDIR to a writable directory or unset it to fall back to the default temporary directory."
                )

        # Explicitly pass the directory to mkdtemp to ensure TMPDIR is respected
        # This works even if tempfile.gettempdir() was already called and cached
        # Pass dir=None if tmpdir is None to use default temp directory
        self.name = tempfile.mkdtemp(prefix=config.TEMP_CACHE_DIR_PREFIX, dir=tmpdir)
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        for dset in get_datasets_with_cache_file_in_temp_dir():
            dset.__del__()
        if os.path.exists(self.name):
            try:
                shutil.rmtree(self.name)
            except Exception as e:
                raise OSError(
                    f"An error occurred while trying to delete temporary cache directory {self.name}. Please delete it manually."
                ) from e

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()


def maybe_register_dataset_for_temp_dir_deletion(dataset):
    """
    This function registers the datasets that have cache files in _TEMP_DIR_FOR_TEMP_CACHE_FILES in order
    to properly delete them before deleting the temporary directory.
    The temporary directory _TEMP_DIR_FOR_TEMP_CACHE_FILES is used when caching is disabled.
    """
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        return

    global _DATASETS_WITH_TABLE_IN_TEMP_DIR
    if _DATASETS_WITH_TABLE_IN_TEMP_DIR is None:
        _DATASETS_WITH_TABLE_IN_TEMP_DIR = weakref.WeakSet()
    if any(
        Path(_TEMP_DIR_FOR_TEMP_CACHE_FILES.name) in Path(cache_file["filename"]).parents
        for cache_file in dataset.cache_files
    ):
        _DATASETS_WITH_TABLE_IN_TEMP_DIR.add(dataset)


def get_datasets_with_cache_file_in_temp_dir():
    return list(_DATASETS_WITH_TABLE_IN_TEMP_DIR) if _DATASETS_WITH_TABLE_IN_TEMP_DIR is not None else []


def enable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = True


def disable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = False


def is_caching_enabled() -> bool:
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`]] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    return bool(_CACHING_ENABLED)


def get_temporary_cache_files_directory() -> str:
    """Return a directory that is deleted when session closes."""
    global _TEMP_DIR_FOR_TEMP_CACHE_FILES
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        _TEMP_DIR_FOR_TEMP_CACHE_FILES = _TempCacheDir()
    return _TEMP_DIR_FOR_TEMP_CACHE_FILES.name


#################
# Hashing
#################


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, list[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


#################
# Fingerprinting
#################

fingerprint_rng = random.Random()
# we show a warning only once when fingerprinting fails to avoid spam
fingerprint_warnings: dict[str, bool] = {}


def generate_fingerprint(dataset: "Dataset") -> str:
    state = dataset.__dict__
    hasher = Hasher()
    for key in sorted(state):
        if key == "_fingerprint":
            continue
        hasher.update(key)
        hasher.update(state[key])
    # hash data files last modification timestamps as well
    for cache_file in dataset.cache_files:
        hasher.update(os.path.getmtime(cache_file["filename"]))
    return hasher.hexdigest()


def generate_random_fingerprint(nbits: int = 64) -> str:
    return f"{fingerprint_rng.getrandbits(nbits):0{nbits // 4}x}"


def update_fingerprint(fingerprint, transform, transform_args):
    global fingerprint_warnings
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, a random hash was used instead. "
                    "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                    "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                    "This warning is only shown once. Subsequent hashing failures won't be shown."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead.")
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
            )

        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. "
                        "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                        "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                        "This warning is only shown once. Subsequent hashing failures won't be shown."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()


def validate_fingerprint(fingerprint: str, max_length=64):
    """
    Make sure the fingerprint is a non-empty string that is not longer that max_length=64 by default,
    so that the fingerprint can be used to name cache files without issues.
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            raise ValueError(
                f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. "
                f"They could create issues when creating cache files."
            )
    if len(fingerprint) > max_length:
        raise ValueError(
            f"Invalid fingerprint. Maximum lenth is {max_length} but '{fingerprint}' has length {len(fingerprint)}."
            "It could create issues when creating cache files."
        )


def format_transform_for_fingerprint(func: Callable, version: Optional[str] = None) -> str:
    """
    Format a transform to the format that will be used to update the fingerprint.
    """
    transform = f"{func.__module__}.{func.__qualname__}"
    if version is not None:
        transform += f"@{version}"
    return transform


def format_kwargs_for_fingerprint(
    func: Callable,
    args: tuple,
    kwargs: dict[str, Any],
    use_kwargs: Optional[list[str]] = None,
    ignore_kwargs: Optional[list[str]] = None,
    randomized_function: bool = False,
) -> dict[str, Any]:
    """
    Format the kwargs of a transform to the format that will be used to update the fingerprint.
    """
    kwargs_for_fingerprint = kwargs.copy()
    if args:
        params = [p.name for p in inspect.signature(func).parameters.values() if p != p.VAR_KEYWORD]
        args = args[1:]  # assume the first argument is the dataset
        params = params[1:]
        kwargs_for_fingerprint.update(zip(params, args))
    else:
        del kwargs_for_fingerprint[
            next(iter(inspect.signature(func).parameters))
        ]  # assume the first key is the dataset

    # keep the right kwargs to be hashed to generate the fingerprint

    if use_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k in use_kwargs}
    if ignore_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k not in ignore_kwargs}
    if randomized_function:  # randomized functions have `seed` and `generator` parameters
        if kwargs_for_fingerprint.get("seed") is None and kwargs_for_fingerprint.get("generator") is None:
            _, seed, pos, *_ = np.random.get_state()
            seed = seed[pos] if pos < 624 else seed[0]
            kwargs_for_fingerprint["generator"] = np.random.default_rng(seed)

    # remove kwargs that are the default values

    default_values = {
        p.name: p.default for p in inspect.signature(func).parameters.values() if p.default != inspect._empty
    }
    for default_varname, default_value in default_values.items():
        if default_varname in kwargs_for_fingerprint and kwargs_for_fingerprint[default_varname] == default_value:
            kwargs_for_fingerprint.pop(default_varname)
    return kwargs_for_fingerprint


def fingerprint_transform(
    inplace: bool,
    use_kwargs: Optional[list[str]] = None,
    ignore_kwargs: Optional[list[str]] = None,
    fingerprint_names: Optional[list[str]] = None,
    randomized_function: bool = False,
    version: Optional[str] = None,
):
    """
    Wrapper for dataset transforms to update the dataset fingerprint using ``update_fingerprint``
    Args:
        inplace (:obj:`bool`):  If inplace is True, the fingerprint of the dataset is updated inplace.
            Otherwise, a parameter "new_fingerprint" is passed to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset.
        use_kwargs (:obj:`List[str]`, optional): optional white list of argument names to take into account
            to update the fingerprint to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset. By default all the arguments are used.
        ignore_kwargs (:obj:`List[str]`, optional): optional black list of argument names to take into account
            to update the fingerprint. Note that ignore_kwargs prevails on use_kwargs.
        fingerprint_names (:obj:`List[str]`, optional, defaults to ["new_fingerprint"]):
            If the dataset transforms is not inplace and returns a DatasetDict, then it can require
            several fingerprints (one per dataset in the DatasetDict). By specifying fingerprint_names,
            one fingerprint named after each element of fingerprint_names is going to be passed.
        randomized_function (:obj:`bool`, defaults to False): If the dataset transform is random and has
            optional parameters "seed" and "generator", then you can set randomized_function to True.
            This way, even if users set "seed" and "generator" to None, then the fingerprint is
            going to be randomly generated depending on numpy's current state. In this case, the
            generator is set to np.random.default_rng(np.random.get_state()[1][0]).
        version (:obj:`str`, optional): version of the transform. The version is taken into account when
            computing the fingerprint. If a datase transform changes (or at least if the output data
            that are cached changes), then one should increase the version. If the version stays the
            same, then old cached data could be reused that are not compatible with the new transform.
            It should be in the format "MAJOR.MINOR.PATCH".
    """

    if use_kwargs is not None and not isinstance(use_kwargs, list):
        raise ValueError(f"use_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if ignore_kwargs is not None and not isinstance(ignore_kwargs, list):
        raise ValueError(f"ignore_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if inplace and fingerprint_names:
        raise ValueError("fingerprint_names are only used when inplace is False")

    fingerprint_names = fingerprint_names if fingerprint_names is not None else ["new_fingerprint"]

    def _fingerprint(func):
        if not inplace and not all(name in func.__code__.co_varnames for name in fingerprint_names):
            raise ValueError(f"function {func} is missing parameters {fingerprint_names} in signature")

        if randomized_function:  # randomized function have seed and generator parameters
            if "seed" not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if "generator" not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")
        # this call has to be outside the wrapper or since __qualname__ changes in multiprocessing
        transform = format_transform_for_fingerprint(func, version=version)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(
                func,
                args,
                kwargs,
                use_kwargs=use_kwargs,
                ignore_kwargs=ignore_kwargs,
                randomized_function=randomized_function,
            )

            if args:
                dataset: Dataset = args[0]
                args = args[1:]
            else:
                dataset: Dataset = kwargs.pop(next(iter(inspect.signature(func).parameters)))

            # compute new_fingerprint and add it to the args of not in-place transforms
            if inplace:
                new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
            else:
                for fingerprint_name in fingerprint_names:  # transforms like `train_test_split` have several hashes
                    if kwargs.get(fingerprint_name) is None:
                        kwargs_for_fingerprint["fingerprint_name"] = fingerprint_name
                        kwargs[fingerprint_name] = update_fingerprint(
                            dataset._fingerprint, transform, kwargs_for_fingerprint
                        )
                    else:
                        validate_fingerprint(kwargs[fingerprint_name])

            # Call actual function

            out = func(dataset, *args, **kwargs)

            # Update fingerprint of in-place transforms + update in-place history of transforms

            if inplace:  # update after calling func so that the fingerprint doesn't change if the function fails
                dataset._fingerprint = new_fingerprint

            return out

        wrapper._decorator_name_ = "fingerprint"
        return wrapper

    return _fingerprint
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""To write records into Parquet files."""

import io
import json
import sys
from collections.abc import Iterable
from typing import Any, Literal, Optional

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.json as paj
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from . import config
from .features import Audio, Features, Image, Pdf, Value, Video
from .features.features import (
    FeatureType,
    List,
    _ArrayXDExtensionType,
    _visit,
    cast_to_python_objects,
    generate_from_arrow_type,
    get_nested_type,
    list_of_np_array_to_pyarrow_listarray,
    numpy_to_pyarrow_listarray,
    require_storage_embed,
    to_pyarrow_listarray,
)
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils.json import (
    find_mixed_struct_types_field_paths,
    get_json_field_path_from_pyarrow_json_error,
    get_json_field_paths_from_feature,
    insert_json_field_path,
    json_encode_field,
    json_encode_fields_in_json_lines,
    set_json_types_in_feature,
    ujson_dumps,
)
from .utils.py_utils import asdict, convert_file_size_to_int, first_non_null_non_empty_value


logger = logging.get_logger(__name__)

type_ = type  # keep python's type function


def get_arrow_writer_batch_size_from_features(features: Optional[Features]) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum record batch size in the arrow files based on configuration values.
    The default value is 100 for image/audio datasets and 10 for videos.
    This allows to avoid overflows in arrow buffers.

    Args:
        features (`datasets.Features` or `None`):
            Dataset Features from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
            If `None`, then it will use the `datasets` default, i.e. `datasets.config.DEFAULT_MAX_BATCH_SIZE`.
    """
    if not features:
        return None

    batch_size = np.inf

    def set_batch_size(feature: FeatureType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image) and config.ARROW_RECORD_BATCH_SIZE_FOR_IMAGE_DATASETS is not None:
            batch_size = min(batch_size, config.ARROW_RECORD_BATCH_SIZE_FOR_IMAGE_DATASETS)
        elif isinstance(feature, Audio) and config.ARROW_RECORD_BATCH_SIZE_FOR_AUDIO_DATASETS is not None:
            batch_size = min(batch_size, config.ARROW_RECORD_BATCH_SIZE_FOR_AUDIO_DATASETS)
        elif isinstance(feature, Video) and config.ARROW_RECORD_BATCH_SIZE_FOR_VIDEO_DATASETS is not None:
            batch_size = min(batch_size, config.ARROW_RECORD_BATCH_SIZE_FOR_VIDEO_DATASETS)
        elif (
            isinstance(feature, Value)
            and feature.dtype == "binary"
            and config.ARROW_RECORD_BATCH_SIZE_FOR_BINARY_DATASETS is not None
        ):
            batch_size = min(batch_size, config.ARROW_RECORD_BATCH_SIZE_FOR_BINARY_DATASETS)

    _visit(features, set_batch_size)

    return None if batch_size is np.inf else batch_size


def get_writer_batch_size_from_features(features: Optional[Features]) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files based on configuration values.
    By default these are not set, but it can be helpful to hard set those values in some cases.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    Args:
        features (`datasets.Features` or `None`):
            Dataset Features from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a parquet writer.
            If `None`, then it will use the `datasets` default, i.e. aiming for row groups of 100MB.
    """
    if not features:
        return None

    batch_size = np.inf

    def set_batch_size(feature: FeatureType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image) and config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS is not None:
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
        elif isinstance(feature, Audio) and config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS is not None:
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS)
        elif isinstance(feature, Video) and config.PARQUET_ROW_GROUP_SIZE_FOR_VIDEO_DATASETS is not None:
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_VIDEO_DATASETS)
        elif (
            isinstance(feature, Value)
            and feature.dtype == "binary"
            and config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS is not None
        ):
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)

    _visit(features, set_batch_size)

    return None if batch_size is np.inf else batch_size


def get_writer_batch_size_from_data_size(num_rows: int, num_bytes: int) -> int:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files.
    The default in `datasets` is aiming for row groups of maximum 100MB uncompressed.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    This can be improved to get optimized size for querying/iterating
    but at least it matches the dataset viewer expectations on HF.

    Args:
        num_rows (`int`):
            Number of rows in the dataset.
        num_bytes (`int`):
            Number of bytes in the dataset.
            For dataset with external files to embed (image, audio, videos), this can also be an
            estimate from `dataset._estimate_nbytes()`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a parquet writer.
    """
    return max(1, num_rows * convert_file_size_to_int(config.MAX_ROW_GROUP_SIZE) // num_bytes) if num_bytes > 0 else 1


def get_parquet_column_options(features: Features, schema: pa.Schema) -> dict:
    """
    Get the per-column `compression`, `use_dictionary` and `column_encoding` options for `pq.ParquetWriter`.
    Parquet matches them against the leaf column paths (e.g. "col.list.element.field"), not the top-level column names.
    """

    def leaf_paths(path: str, pa_type: pa.DataType) -> list[str]:
        if isinstance(pa_type, pa.ExtensionType):
            pa_type = pa_type.storage_type
        if pa.types.is_struct(pa_type):
            return [leaf for field in pa_type for leaf in leaf_paths(f"{path}.{field.name}", field.type)]
        if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type):
            return leaf_paths(f"{path}.list.element", pa_type.value_type)
        return [path]

    compression, use_dictionary, column_encoding = {}, [], {}
    for field in schema:
        embed = require_storage_embed(features[field.name])
        for leaf in leaf_paths(field.name, field.type):
            compression[leaf] = "none" if embed else "snappy"
            if embed:
                column_encoding[leaf] = "PLAIN"
            else:
                use_dictionary.append(leaf)
    return {"compression": compression, "use_dictionary": use_dictionary, "column_encoding": column_encoding}


class SchemaInferenceError(ValueError):
    pass


class TypedSequence:
    """
    This data container generalizes the typing when instantiating pyarrow arrays, tables or batches.

    More specifically it adds several features:
    - Support extension types like ``datasets.features.Array2DExtensionType``:
        By default pyarrow arrays don't return extension arrays. One has to call
        ``pa.ExtensionArray.from_storage(type, pa.array(data, type.storage_type))``
        in order to get an extension array.
    - Support for ``try_type`` parameter that can be used instead of ``type``:
        When an array is transformed, we like to keep the same type as before if possible.
        For example when calling :func:`datasets.Dataset.map`, we don't want to change the type
        of each column by default.
    - Better error message when a pyarrow array overflows.

    Example::

        from datasets.features import Array2D, Array2DExtensionType, Value
        from datasets.arrow_writer import TypedSequence
        import pyarrow as pa

        arr = pa.array(TypedSequence([1, 2, 3], type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence([1, 2, 3], try_type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence(["foo", "bar"], try_type=Value("int32")))
        assert arr.type == pa.string()

        arr = pa.array(TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64")))
        assert arr.type == Array2DExtensionType((1, 3), "int64")

        table = pa.Table.from_pydict({
            "image": TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64"))
        })
        assert table["image"].type == Array2DExtensionType((1, 3), "int64")

    """

    def __init__(
        self,
        data: Iterable,
        type: Optional[FeatureType] = None,
        try_type: Optional[FeatureType] = None,
        optimized_int_type: Optional[FeatureType] = None,
        on_mixed_types: Optional[Literal["use_json"]] = None,
    ):
        # assert type is None or try_type is None,
        if type is not None and try_type is not None:
            raise ValueError("You cannot specify both type and try_type")
        # set attributes
        self.data = data
        self.type = type
        self.try_type = try_type  # is ignored if it doesn't match the data
        self.optimized_int_type = optimized_int_type
        self.on_mixed_types = on_mixed_types
        # when trying a type (is ignored if data is not compatible)
        self.trying_type = self.try_type is not None
        self.trying_int_optimization = optimized_int_type is not None and type is None and try_type is None
        # used to get back the inferred type after __arrow_array__() is called once
        self._inferred_type = None

    def get_inferred_type(self) -> FeatureType:
        """Return the inferred feature type.
        This is done by converting the sequence to an Arrow array, and getting the corresponding
        feature type.

        Since building the Arrow array can be expensive, the value of the inferred type is cached
        as soon as pa.array is called on the typed sequence.

        Returns:
            FeatureType: inferred feature type of the sequence.
        """
        if self._inferred_type is None:
            pa.array(self)
        return self._inferred_type

    @staticmethod
    def _infer_custom_type_and_encode(data: Iterable) -> tuple[Iterable, Optional[FeatureType]]:
        """Implement type inference for custom objects like PIL.Image.Image -> Image type.

        This function is only used for custom python objects that can't be directly passed to build
        an Arrow array. In such cases is infers the feature type to use, and it encodes the data so
        that they can be passed to an Arrow array.

        Args:
            data (Iterable): array of data to infer the type, e.g. a list of PIL images.

        Returns:
            Tuple[Iterable, Optional[FeatureType]]: a tuple with:
                - the (possibly encoded) array, if the inferred feature type requires encoding
                - the inferred feature type if the array is made of supported custom objects like
                    PIL images, else None.
        """
        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            non_null_idx, non_null_value = first_non_null_non_empty_value(data)
            if isinstance(non_null_value, PIL.Image.Image):
                return [Image().encode_example(value) if value is not None else None for value in data], Image()
            if isinstance(non_null_value, list) and isinstance(non_null_value[0], PIL.Image.Image):
                return [
                    [Image().encode_example(x) for x in value] if value is not None else None for value in data
                ], List(Image())
        if config.PDFPLUMBER_AVAILABLE and "pdfplumber" in sys.modules:
            import pdfplumber

            non_null_idx, non_null_value = first_non_null_non_empty_value(data)
            if isinstance(non_null_value, pdfplumber.pdf.PDF):
                return [Pdf().encode_example(value) if value is not None else None for value in data], Pdf()
            if isinstance(non_null_value, list) and isinstance(non_null_value[0], pdfplumber.pdf.PDF):
                return [
                    [Pdf().encode_example(x) for x in value] if value is not None else None for value in data
                ], List(Pdf())
        return data, None

    def __arrow_array__(self, type: Optional[pa.DataType] = None):
        out = self._arrow_array(type=type)
        if self._inferred_type is None:
            self._inferred_type = generate_from_arrow_type(out.type)
        return out

    def _arrow_array(self, type: Optional[pa.DataType] = None):
        """This function is called when calling pa.array(typed_sequence)"""

        if type is not None:
            raise ValueError("TypedSequence is supposed to be used with pa.array(typed_sequence, type=None)")
        del type  # make sure we don't use it
        data = self.data
        # automatic type inference for custom objects
        if self.type is None and self.try_type is None:
            data, self._inferred_type = self._infer_custom_type_and_encode(data)
        if self._inferred_type is None:
            type = self.try_type if self.trying_type else self.type
        else:
            type = self._inferred_type
        pa_type = get_nested_type(type) if type is not None else None
        optimized_int_pa_type = (
            get_nested_type(self.optimized_int_type) if self.optimized_int_type is not None else None
        )
        trying_cast_to_python_objects = False
        json_field_paths = []
        try:
            # custom pyarrow types
            if isinstance(pa_type, _ArrayXDExtensionType):
                storage = to_pyarrow_listarray(data, pa_type)
                return pa.ExtensionArray.from_storage(pa_type, storage)

            # efficient np array to pyarrow array
            if isinstance(data, np.ndarray):
                out = numpy_to_pyarrow_listarray(data)
            elif isinstance(data, list) and data and isinstance(first_non_null_non_empty_value(data)[1], np.ndarray):
                out = list_of_np_array_to_pyarrow_listarray(data)
            else:
                trying_cast_to_python_objects = True
                examples = data
                # find fields to json-encode
                if self.on_mixed_types == "use_json" and type is None:
                    json_field_paths = find_mixed_struct_types_field_paths(examples, allow_root=True)
                elif type is not None:
                    json_field_paths = get_json_field_paths_from_feature(type)
                # json encode if needed
                if json_field_paths:
                    for json_field_path in json_field_paths:
                        examples = [json_encode_field(examples, json_field_path) for examples in examples]
                # to arrow array
                out = pa.array(cast_to_python_objects(examples, only_1d_for_numpy=True))
                # cast to json type if needed
                if json_field_paths:
                    pa_table = pa.Table.from_arrays([out], names=["obj"])
                    features = Features.from_arrow_schema(pa_table.schema)
                    feature = set_json_types_in_feature(features["obj"], json_field_paths)
                    pa_table = table_cast(pa_table, Features({"obj": feature}).arrow_schema)
                    out = pa_table[0]  # get the "obj" column
            # use smaller integer precisions if possible
            if self.trying_int_optimization:
                if pa.types.is_int64(out.type):
                    out = out.cast(optimized_int_pa_type)
                elif pa.types.is_list(out.type):
                    if pa.types.is_int64(out.type.value_type):
                        out = array_cast(out, pa.list_(optimized_int_pa_type))
                    elif pa.types.is_list(out.type.value_type) and pa.types.is_int64(out.type.value_type.value_type):
                        out = array_cast(out, pa.list_(pa.list_(optimized_int_pa_type)))
            # otherwise we can finally use the user's type
            elif type is not None:
                # We use cast_array_to_feature to support casting to custom types like Audio and Image
                # Also, when trying type "string", we don't want to convert integers or floats to "string".
                # We only do it if trying_type is False - since this is what the user asks for.
                out = cast_array_to_feature(
                    out, type, allow_primitive_to_str=not self.trying_type, allow_decimal_to_str=not self.trying_type
                )
            return out
        except (
            TypeError,
            pa.lib.ArrowTypeError,
            pa.lib.ArrowInvalid,
            pa.lib.ArrowNotImplementedError,
        ) as e:  # handle type errors and overflows
            # Ignore ArrowNotImplementedError caused by trying type, otherwise re-raise
            if not self.trying_type and isinstance(e, pa.lib.ArrowNotImplementedError):
                raise

            if self.trying_type:
                try:  # second chance
                    if isinstance(data, np.ndarray):
                        return numpy_to_pyarrow_listarray(data)
                    elif isinstance(data, list) and data and any(isinstance(value, np.ndarray) for value in data):
                        return list_of_np_array_to_pyarrow_listarray(data)
                    else:
                        trying_cast_to_python_objects = True
                        return pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
                except pa.lib.ArrowInvalid as e:
                    if "overflow" in str(e):
                        raise OverflowError(
                            f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})"
                        ) from None
                    elif self.trying_int_optimization and "not in range" in str(e):
                        optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                        logger.info(
                            f"Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64."
                        )
                        return out
                    elif trying_cast_to_python_objects and "Could not convert" in str(e):
                        out = pa.array(
                            cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False)
                        )
                        if type is not None:
                            out = cast_array_to_feature(
                                out, type, allow_primitive_to_str=True, allow_decimal_to_str=True
                            )
                        return out
                    else:
                        raise
            elif "overflow" in str(e):
                raise OverflowError(
                    f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})"
                ) from None
            elif self.trying_int_optimization and "not in range" in str(e):
                optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                logger.info(f"Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64.")
                return out
            elif trying_cast_to_python_objects and (
                "Could not convert" in str(e) or "cannot mix struct and non-struct" in str(e) or "Expected " in str(e)
            ):
                try:  # third chance
                    out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False))
                except (pa.ArrowInvalid, pa.ArrowTypeError) as ee:
                    # in case of mixed types, we use the JSON Lines reader in pyarrow to locate them and set them to json fields
                    if self.on_mixed_types == "use_json" and (
                        "Could not convert " in str(ee)
                        or "cannot mix struct and non-struct" in str(ee)
                        or "Expected " in str(ee)
                    ):
                        # we use "obj" to have valid JSON Lines since data may contain lists
                        original_batch = "\n".join([ujson_dumps({"obj": example}) for example in data]).encode()
                        json_field_paths = [["obj"] + json_field_path for json_field_path in json_field_paths]
                        batch = json_encode_fields_in_json_lines(original_batch, json_field_paths)
                        pa_table = None
                        while True:
                            try:  # fourth chance
                                pa_table = paj.read_json(
                                    io.BytesIO(batch), read_options=paj.ReadOptions(use_threads=False)
                                )
                                break
                            except pa.ArrowInvalid as eee:
                                if "JSON parse error: Column(" in str(eee) and ") changed from" in str(eee):
                                    json_field_path = get_json_field_path_from_pyarrow_json_error(str(eee))
                                    insert_json_field_path(json_field_paths, json_field_path)
                                    batch = json_encode_fields_in_json_lines(original_batch, json_field_paths)
                                else:
                                    break
                        if pa_table is not None:
                            features = Features.from_arrow_schema(pa_table.schema)
                            features = set_json_types_in_feature(features, json_field_paths)
                            pa_table = table_cast(pa_table, features.arrow_schema)
                            out = pa_table[0]  # get the "obj" column
                            return out
                        else:
                            raise
                    else:
                        raise
                if type is not None:
                    out = cast_array_to_feature(out, type, allow_primitive_to_str=True, allow_decimal_to_str=True)
                return out
            else:
                raise


class OptimizedTypedSequence(TypedSequence):
    def __init__(
        self,
        data,
        type: Optional[FeatureType] = None,
        try_type: Optional[FeatureType] = None,
        col: Optional[str] = None,
        optimized_int_type: Optional[FeatureType] = None,
        on_mixed_types: Optional[Literal["use_json"]] = None,
    ):
        optimized_int_type_by_col = {
            "attention_mask": Value("int8"),  # binary tensor
            "special_tokens_mask": Value("int8"),
            "input_ids": Value("int32"),  # typical vocab size: 0-50k (max ~500k, never > 1M)
            "token_type_ids": Value(
                "int8"
            ),  # binary mask; some (XLNetModel) use an additional token represented by a 2
        }
        if type is None and try_type is None:
            optimized_int_type = optimized_int_type_by_col.get(col, None)
        super().__init__(
            data, type=type, try_type=try_type, optimized_int_type=optimized_int_type, on_mixed_types=on_mixed_types
        )


class ArrowWriter:
    """Shuffles and writes Examples to Arrow files."""

    def __init__(
        self,
        schema: Optional[pa.Schema] = None,
        features: Optional[Features] = None,
        path: Optional[str] = None,
        stream: Optional[pa.NativeFile] = None,
        fingerprint: Optional[str] = None,
        writer_batch_size: Optional[int] = None,
        disable_nullable: bool = False,
        update_features: bool = False,
        on_mixed_types: Optional[Literal["use_json"]] = "use_json",
        with_metadata: bool = True,
        unit: str = "examples",
        embed_local_files: bool = False,
        storage_options: Optional[dict] = None,
    ):
        if path is None and stream is None:
            raise ValueError("At least one of path and stream must be provided.")
        if features is not None:
            self._features = features
            self._schema = None
        elif schema is not None:
            self._schema: pa.Schema = schema
            self._features = Features.from_arrow_schema(self._schema)
        else:
            self._features = None
            self._schema = None

        self._disable_nullable = disable_nullable

        if stream is None:
            fs, path = url_to_fs(path, **(storage_options or {}))
            self._fs: fsspec.AbstractFileSystem = fs
            self._path = path if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(path)
            self.stream = self._fs.open(path, "wb")
            self._closable_stream = True
        else:
            self._fs = None
            self._path = None
            self.stream = stream
            self._closable_stream = False

        self.fingerprint = fingerprint
        self.disable_nullable = disable_nullable
        self.writer_batch_size = (
            writer_batch_size
            or get_arrow_writer_batch_size_from_features(self._features)
            or config.DEFAULT_MAX_BATCH_SIZE
        )
        self.update_features = update_features
        self.on_mixed_types = on_mixed_types
        self.with_metadata = with_metadata
        self.unit = unit
        self.embed_local_files = embed_local_files

        self._num_examples = 0
        self._num_bytes = 0
        self.current_examples: list[tuple[dict[str, Any], str]] = []
        self.current_rows: list[pa.Table] = []
        self.pa_writer: Optional[pa.RecordBatchStreamWriter] = None
        self.hkey_record = []

    def __len__(self):
        """Return the number of writed and staged examples"""
        return self._num_examples + len(self.current_examples) + len(self.current_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Try closing if opened; if closed: pyarrow.lib.ArrowInvalid: Invalid operation on closed file
        if self.pa_writer:  # it might be None
            try:
                self.pa_writer.close()
            except Exception:  # pyarrow.lib.ArrowInvalid, OSError
                pass
        if self._closable_stream and not self.stream.closed:
            self.stream.close()  # This also closes self.pa_writer if it is opened

    def _build_schema(self, inferred_schema: pa.Schema):
        schema = self.schema
        features = self._features
        inferred_features = Features.from_arrow_schema(inferred_schema)
        if self._features is not None:
            if self.update_features:  # keep original features it they match, or update them
                fields = {field.name: field for field in self._features.type}
                for inferred_field in inferred_features.type:
                    name = inferred_field.name
                    if name in fields:
                        if inferred_field == fields[name]:
                            inferred_features[name] = self._features[name]
                features = inferred_features
                schema: pa.Schema = inferred_schema
        else:
            features = inferred_features
            schema: pa.Schema = inferred_features.arrow_schema

        if self.disable_nullable:
            schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in schema)
        if self.with_metadata:
            schema = schema.with_metadata(self._build_metadata(DatasetInfo(features=features), self.fingerprint))
        else:
            schema = schema.with_metadata({})

        return schema, features

    def _build_writer(self, inferred_schema: pa.Schema):
        self._schema, self._features = self._build_schema(inferred_schema)
        self.pa_writer = pa.RecordBatchStreamWriter(self.stream, self._schema)

    @property
    def schema(self):
        _schema = (
            self._schema
            if self._schema is not None
            else (pa.schema(self._features.type) if self._features is not None else None)
        )
        if self._disable_nullable and _schema is not None:
            _schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in _schema)
        return _schema if _schema is not None else []

    @staticmethod
    def _build_metadata(info: DatasetInfo, fingerprint: Optional[str] = None) -> dict[str, str]:
        info_keys = ["features"]  # we can add support for more DatasetInfo keys in the future
        info_as_dict = asdict(info)
        metadata = {}
        metadata["info"] = {key: info_as_dict[key] for key in info_keys}
        if fingerprint is not None:
            metadata["fingerprint"] = fingerprint
        return {"huggingface": json.dumps(metadata)}

    def write_examples_on_file(self):
        """Write stored examples from the write-pool of examples. It makes a table out of the examples and write it."""
        if not self.current_examples:
            return
        # preserve the order the columns
        if self.schema:
            schema_cols = set(self.schema.names)
            examples_cols = self.current_examples[0][0].keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in examples_cols]
            extra_cols = [col for col in examples_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(self.current_examples[0][0])
        batch_examples = {}
        for col in cols:
            # We use row[0][col] since current_examples contains (example, key) tuples.
            # Moreover, examples could be Arrow arrays of 1 element.
            # This can happen in `.map()` when we want to re-write the same Arrow data
            if all(isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) for row in self.current_examples):
                arrays = [row[0][col] for row in self.current_examples]
                arrays = [
                    chunk
                    for array in arrays
                    for chunk in (array.chunks if isinstance(array, pa.ChunkedArray) else [array])
                ]
                batch_examples[col] = pa.concat_arrays(arrays)
            else:
                batch_examples[col] = [
                    row[0][col].to_pylist()[0] if isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) else row[0][col]
                    for row in self.current_examples
                ]
        self._write_batch(batch_examples=batch_examples)
        self.current_examples = []

    def write_rows_on_file(self):
        """Write stored rows from the write-pool of rows. It concatenates the single-row tables and it writes the resulting table."""
        if not self.current_rows:
            return
        table = pa.concat_tables(self.current_rows)
        self._write_table(table)
        self.current_rows = []

    def write(
        self,
        example: dict[str, Any],
        writer_batch_size: Optional[int] = None,
    ):
        """Add a given (Example,Key) pair to the write-pool of examples which is written to file.

        Args:
            example: the Example to add.
        """
        # Store example as a tuple so as to keep the structure of `self.current_examples` uniform
        self.current_examples.append((example, ""))

        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_examples) >= writer_batch_size:
            self.write_examples_on_file()

    def write_row(self, row: pa.Table, writer_batch_size: Optional[int] = None):
        """Add a given single-row Table to the write-pool of rows which is written to file.

        Args:
            row: the row to add.
        """
        if len(row) != 1:
            raise ValueError(f"Only single-row pyarrow tables are allowed but got table with {len(row)} rows.")
        self.current_rows.append(row)
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_rows) >= writer_batch_size:
            self.write_rows_on_file()

    def write_batch(
        self,
        batch_examples: dict[str, list],
        writer_batch_size: Optional[int] = None,
        try_original_type: Optional[bool] = True,
    ):
        """Write a batch of Example to file.
        Ignores the batch if it appears to be empty,
        preventing a potential schema update of unknown types.

        Args:
            batch_examples: the batch of examples to add.
            try_original_type: use `try_type` when instantiating OptimizedTypedSequence if `True`, otherwise `try_type = None`.
        """
        self.write_examples_on_file()  # in case there are buffered examples to write first
        self._write_batch(batch_examples, writer_batch_size=writer_batch_size, try_original_type=try_original_type)

    def _check_null_column(self, col: str, values):
        """Raise an informative error if a column written as null gets values that are not None.

        The schema is fixed when the first batch is written, so a column that only has None values
        in the first batch keeps the null type for the rest of the file and can't hold values later.
        """
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            has_values = values.null_count < len(values)
        else:
            has_values = any(value is not None for value in values)
        if has_values:
            raise TypeError(
                f"Couldn't write column '{col}': its type in this file is null, because the first "
                f"{self._num_examples} values written for it were all None. Set the type of '{col}' "
                f"with features=, or make sure the first batch written contains a value that is not None."
            )

    def _write_batch(
        self,
        batch_examples: dict[str, list],
        writer_batch_size: Optional[int] = None,
        try_original_type: Optional[bool] = True,
    ):
        if batch_examples and len(next(iter(batch_examples.values()))) == 0:
            return
        features = None if self.pa_writer is None and self.update_features else self._features
        try_features = self._features if self.pa_writer is None and self.update_features else None
        arrays = []
        inferred_features = Features()
        # preserve the order the columns
        if self.schema:
            schema_cols = set(self.schema.names)
            batch_cols = batch_examples.keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in batch_cols]
            extra_cols = [col for col in batch_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(batch_examples)
        for col in cols:
            col_values = batch_examples[col]
            col_type = features[col] if features else None
            if isinstance(col_type, Value) and col_type.dtype == "null":
                self._check_null_column(col, col_values)
            if isinstance(col_values, (pa.Array, pa.ChunkedArray)):
                array = cast_array_to_feature(col_values, col_type) if col_type is not None else col_values
                arrays.append(array)
                inferred_features[col] = generate_from_arrow_type(col_values.type)
            else:
                col_try_type = (
                    try_features[col]
                    if try_features is not None and col in try_features and try_original_type
                    else None
                )
                typed_sequence = OptimizedTypedSequence(
                    col_values, type=col_type, try_type=col_try_type, col=col, on_mixed_types=self.on_mixed_types
                )
                arrays.append(pa.array(typed_sequence))
                inferred_features[col] = typed_sequence.get_inferred_type()
        schema = inferred_features.arrow_schema if self.pa_writer is None else self.schema
        pa_table = pa.Table.from_arrays(arrays, schema=schema)
        self.write_table(pa_table, writer_batch_size)

    def write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int] = None):
        """Write a Table to file.

        Args:
            example: the Table to add.
        """
        self.write_rows_on_file()  # in case there are buffered rows to write first
        self._write_table(pa_table, writer_batch_size=writer_batch_size)

    def _write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int] = None):
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if self.pa_writer is None:
            self._build_writer(inferred_schema=pa_table.schema)
        pa_table = pa_table.combine_chunks()
        for field in self._schema:
            if pa.types.is_null(field.type) and field.name in pa_table.column_names:
                self._check_null_column(field.name, pa_table[field.name])
        pa_table = table_cast(pa_table, self._schema)
        if self.embed_local_files:
            pa_table = embed_table_storage(pa_table, local_files=True, remote_files=False)
        self._num_bytes += pa_table.nbytes
        self._num_examples += pa_table.num_rows
        self.pa_writer.write_table(pa_table, writer_batch_size)

    def finalize(self, close_stream=True):
        self.write_rows_on_file()
        # In case current_examples < writer_batch_size, but user uses finalize()
        self.write_examples_on_file()
        # If schema is known, infer features even if no examples were written
        if self.pa_writer is None and self.schema:
            self._build_writer(self.schema)
        if self.pa_writer is not None:
            self.pa_writer.close()
            self.pa_writer = None
            if close_stream:
                self.stream.close()
        else:
            if close_stream:
                self.stream.close()
            raise SchemaInferenceError("Please pass `features` or at least one example when writing data")
        logger.debug(
            f"Done writing {self._num_examples} {self.unit} in {self._num_bytes} bytes {self._path if self._path else ''}."
        )
        return self._num_examples, self._num_bytes


class ParquetWriter(ArrowWriter):
    def __init__(self, *args, use_content_defined_chunking=True, write_page_index=True, **kwargs):
        super().__init__(*args, **kwargs)
        if use_content_defined_chunking is True:
            use_content_defined_chunking = config.DEFAULT_CDC_OPTIONS
        self.use_content_defined_chunking = use_content_defined_chunking
        self.write_page_index = write_page_index

    def _build_writer(self, inferred_schema: pa.Schema):
        self._schema, self._features = self._build_schema(inferred_schema)
        self.pa_writer = pq.ParquetWriter(
            self.stream,
            self._schema,
            use_content_defined_chunking=self.use_content_defined_chunking,
            write_page_index=self.write_page_index,
            **get_parquet_column_options(self._features, self._schema),
        )
        if self.use_content_defined_chunking is not False:
            self.pa_writer.add_key_value_metadata(
                {"content_defined_chunking": json.dumps(self.use_content_defined_chunking)}
            )
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
"""DatasetInfo record information we know about a dataset.

This includes things that we know about the dataset statically, i.e.:
 - description
 - canonical location
 - does it have validation and tests splits
 - size
 - etc.

This also includes the things that can and should be computed once we've
processed the dataset as well:
 - number of examples (in each split)
 - etc.
"""

import copy
import dataclasses
import json
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Union

import fsspec
from fsspec.core import url_to_fs
from huggingface_hub import DatasetCard, DatasetCardData

from . import config
from .features import Features
from .splits import SplitDict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values


logger = get_logger(__name__)


@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""


@dataclass
class DownloadChecksumsEntryData:
    key: str = ""
    value: str = ""


class MissingCachedSizesConfigError(Exception):
    """The expected cached sizes of the download file are missing."""


class NonMatchingCachedSizesError(Exception):
    """The prepared split doesn't have expected sizes."""


@dataclass
class PostProcessedInfo:
    features: Optional[Features] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> "PostProcessedInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in post_processed_info_dict.items() if k in field_names})


@dataclass
class DatasetInfo:
    """Information about a dataset.

    `DatasetInfo` documents datasets, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Not all fields are known on construction and may be updated later.

    Attributes:
        description (`str`):
            A description of the dataset.
        citation (`str`):
            A BibTeX citation of the dataset.
        homepage (`str`):
            A URL to the official homepage for the dataset.
        license (`str`):
            The dataset's license. It can be the name of the license or a paragraph containing the terms of the license.
        features ([`Features`], *optional*):
            The features used to specify the dataset's column types.
        post_processed (`PostProcessedInfo`, *optional*):
            Deprecated. Information regarding the resources of a possible post-processing of a dataset. For example, it can contain the information of an index.
        supervised_keys (`SupervisedKeysData`, *optional*):
            Specifies the input feature and the label for supervised learning if applicable for the dataset (legacy from TFDS).
        builder_name (`str`, *optional*):
            The name of the `GeneratorBasedBuilder` subclass used to create the dataset. It is also the snake_case version of the dataset builder class name.
        config_name (`str`, *optional*):
            The name of the configuration derived from [`BuilderConfig`].
        version (`str` or [`Version`], *optional*):
            The version of the dataset.
        splits (`dict`, *optional*):
            The mapping between split name and metadata.
        download_checksums (`dict`, *optional*):
            The mapping between the URL to download the dataset's checksums and corresponding metadata.
        download_size (`int`, *optional*):
            The size of the files to download to generate the dataset, in bytes.
        post_processing_size (`int`, *optional*):
            Deprecated. Size of the dataset in bytes after post-processing, if any.
        dataset_size (`int`, *optional*):
            The combined size in bytes of the Arrow tables for all splits.
        size_in_bytes (`int`, *optional*):
            The combined size in bytes of all files associated with the dataset (downloaded files + Arrow files).
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`] and used in the [`DatasetBuilder`].
    """

    # Set in the dataset builders
    description: str = dataclasses.field(default_factory=str)
    citation: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    features: Optional[Features] = None
    post_processed: Optional[PostProcessedInfo] = None  # kept for bawkard compat
    supervised_keys: Optional[SupervisedKeysData] = None

    # Set later by the builder
    builder_name: Optional[str] = None
    dataset_name: Optional[str] = None  # for packaged builders, to be different from builder_name
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    # Set later by `download_and_prepare`
    splits: Optional[SplitDict] = None
    download_checksums: Optional[dict] = None
    download_size: Optional[int] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None

    _INCLUDED_INFO_IN_YAML: ClassVar[list[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "features",
        "splits",
    ]

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)
        if self.post_processed is not None and not isinstance(self.post_processed, PostProcessedInfo):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.splits is not None and not isinstance(self.splits, SplitDict):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and not isinstance(self.supervised_keys, SupervisedKeysData):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)

    def write_to_directory(self, dataset_info_dir, pretty_print=False, storage_options: Optional[dict] = None):
        """Write `DatasetInfo` and license (if present) as JSON files to `dataset_info_dir`.

        Args:
            dataset_info_dir (`str`):
                Destination directory.
            pretty_print (`bool`, defaults to `False`):
                If `True`, the JSON will be pretty-printed with the indent level of 4.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
        >>> ds.info.write_to_directory("/path/to/directory/")
        ```
        """
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), "wb") as f:
            self._dump_info(f, pretty_print=pretty_print)
        if self.license:
            with fs.open(posixpath.join(dataset_info_dir, config.LICENSE_FILENAME), "wb") as f:
                self._dump_license(f)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(json.dumps(asdict(self), indent=4 if pretty_print else None).encode("utf-8"))

    def _dump_license(self, file):
        """Dump license in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(self.license.encode("utf-8"))

    @classmethod
    def from_merge(cls, dataset_infos: list["DatasetInfo"]):
        dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]

        if len(dataset_infos) > 0 and all(dataset_infos[0] == dset_info for dset_info in dataset_infos):
            # if all dataset_infos are equal we don't need to merge. Just return the first.
            return dataset_infos[0]

        description = "\n\n".join(unique_values(info.description for info in dataset_infos)).strip()
        citation = "\n\n".join(unique_values(info.citation for info in dataset_infos)).strip()
        homepage = "\n\n".join(unique_values(info.homepage for info in dataset_infos)).strip()
        license = "\n\n".join(unique_values(info.license for info in dataset_infos)).strip()
        features = None
        supervised_keys = None

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            license=license,
            features=features,
            supervised_keys=supervised_keys,
        )

    @classmethod
    def from_directory(cls, dataset_info_dir: str, storage_options: Optional[dict] = None) -> "DatasetInfo":
        """Create [`DatasetInfo`] from the JSON file in `dataset_info_dir`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the [`DatasetInfo`].

        This will overwrite all previous metadata.

        Args:
            dataset_info_dir (`str`):
                The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import DatasetInfo
        >>> ds_info = DatasetInfo.from_directory("/path/to/directory/")
        ```
        """
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        logger.debug(f"Loading Dataset info from {dataset_info_dir}")
        if not dataset_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), "r", encoding="utf-8") as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "DatasetInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def _to_yaml_dict(self) -> dict:
        yaml_dict = {}
        dataset_info_dict = asdict(self)
        for key in dataset_info_dict:
            if key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, "_to_yaml_list"):  # Features, SplitDict
                    yaml_dict[key] = value._to_yaml_list()
                elif hasattr(value, "_to_yaml_string"):  # Version
                    yaml_dict[key] = value._to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def _from_yaml_dict(cls, yaml_data: dict) -> "DatasetInfo":
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get("features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            try:
                yaml_data["splits"] = SplitDict._from_yaml_list(yaml_data["splits"])
            except ValueError as e:
                logger.warning(f"Ignoring part of dataset_info from the dataset card: {e}")
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})

    def __repr__(self):
        return (
            self.__class__.__qualname__
            + "("
            + ", ".join(
                [f"{f.name}={repr(getattr(self, f.name))}" for f in dataclasses.fields(self) if getattr(self, f.name)]
            )
            + ")"
        )


class DatasetInfosDict(dict[str, DatasetInfo]):
    def write_to_directory(self, dataset_infos_dir, overwrite=False, pretty_print=False) -> None:
        total_dataset_infos = {}
        dataset_infos_path = os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)
        dataset_readme_path = os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)
        if not overwrite:
            total_dataset_infos = self.from_directory(dataset_infos_dir)
        total_dataset_infos.update(self)
        if os.path.exists(dataset_infos_path):
            # for backward compatibility, let's update the JSON file if it exists
            with open(dataset_infos_path, "w", encoding="utf-8") as f:
                dataset_infos_dict = {
                    config_name: asdict(dset_info) for config_name, dset_info in total_dataset_infos.items()
                }
                json.dump(dataset_infos_dict, f, indent=4 if pretty_print else None)
        # Dump the infos in the YAML part of the README.md file
        if os.path.exists(dataset_readme_path):
            dataset_card = DatasetCard.load(dataset_readme_path)
            dataset_card_data = dataset_card.data
        else:
            dataset_card = None
            dataset_card_data = DatasetCardData()
        if total_dataset_infos:
            total_dataset_infos.to_dataset_card_data(dataset_card_data)
            dataset_card = (
                DatasetCard("---\n" + str(dataset_card_data) + "\n---\n") if dataset_card is None else dataset_card
            )
            dataset_card.save(Path(dataset_readme_path))

    @classmethod
    def from_directory(cls, dataset_infos_dir) -> "DatasetInfosDict":
        logger.debug(f"Loading Dataset Infos from {dataset_infos_dir}")
        # Load the info from the YAML part of README.md
        if os.path.exists(os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)):
            dataset_card_data = DatasetCard.load(Path(dataset_infos_dir) / config.REPOCARD_FILENAME).data
            if "dataset_info" in dataset_card_data:
                return cls.from_dataset_card_data(dataset_card_data)
        if os.path.exists(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)):
            # this is just to have backward compatibility with dataset_infos.json files
            with open(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                return cls(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
        else:
            return cls()

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "DatasetInfosDict":
        if isinstance(dataset_card_data.get("dataset_info"), (list, dict)):
            if isinstance(dataset_card_data["dataset_info"], list):
                return cls(
                    {
                        dataset_info_yaml_dict.get("config_name", "default"): DatasetInfo._from_yaml_dict(
                            dataset_info_yaml_dict
                        )
                        for dataset_info_yaml_dict in dataset_card_data["dataset_info"]
                    }
                )
            else:
                dataset_info = DatasetInfo._from_yaml_dict(dataset_card_data["dataset_info"])
                dataset_info.config_name = dataset_card_data["dataset_info"].get("config_name", "default")
                return cls({dataset_info.config_name: dataset_info})
        else:
            return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            # first get existing metadata info
            if "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], dict):
                dataset_metadata_infos = {
                    dataset_card_data["dataset_info"].get("config_name", "default"): dataset_card_data["dataset_info"]
                }
            elif "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], list):
                dataset_metadata_infos = {
                    config_metadata["config_name"]: config_metadata
                    for config_metadata in dataset_card_data["dataset_info"]
                }
            else:
                dataset_metadata_infos = {}
            # update/rewrite existing metadata info with the one to dump
            total_dataset_infos = {
                **dataset_metadata_infos,
                **{config_name: dset_info._to_yaml_dict() for config_name, dset_info in self.items()},
            }
            # the config_name from the dataset_infos_dict takes over the config_name of the DatasetInfo
            for config_name, dset_info_yaml_dict in total_dataset_infos.items():
                dset_info_yaml_dict["config_name"] = config_name
            if len(total_dataset_infos) == 1:
                # use a struct instead of a list of configurations, since there's only one
                dataset_card_data["dataset_info"] = next(iter(total_dataset_infos.values()))
                config_name = dataset_card_data["dataset_info"].pop("config_name", None)
                if config_name != "default":
                    # if config_name is not "default" preserve it and put at the first position
                    dataset_card_data["dataset_info"] = {
                        "config_name": config_name,
                        **dataset_card_data["dataset_info"],
                    }
            else:
                dataset_card_data["dataset_info"] = []
                for config_name, dataset_info_yaml_dict in sorted(total_dataset_infos.items()):
                    # add the config_name field in first position
                    dataset_info_yaml_dict.pop("config_name", None)
                    dataset_info_yaml_dict = {"config_name": config_name, **dataset_info_yaml_dict}
                    dataset_card_data["dataset_info"].append(dataset_info_yaml_dict)
<|endoftext|>from itertools import chain
from typing import Optional, Union

from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetCard,
    DatasetCardData,
    HfApi,
    HfFileSystem,
)

import datasets.config
from datasets import __version__
from datasets.info import DatasetInfosDict
from datasets.load import load_dataset_builder
from datasets.utils.metadata import MetadataConfigs


def delete_from_hub(
    repo_id: str,
    config_name: str,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
) -> CommitInfo:
    """Delete a dataset configuration from a [data-only dataset](repository_structure) on the Hub.

    Args:
        repo_id (`str`): ID of the Hub dataset repository, in the following format: `<user>/<dataset_name>` or
            `<org>/<dataset_name>`.
        config_name (`str`): Name of the dataset configuration.
        revision (`str`, *optional*): Branch to delete the configuration from. Defaults to the `"main"` branch.
        token (`bool` or `str`, *optional*): Authentication token for the Hugging Face Hub.

    Returns:
        `huggingface_hub.CommitInfo`
    """
    operations = []
    # data_files
    fs = HfFileSystem(endpoint=datasets.config.HF_ENDPOINT, token=token)
    builder = load_dataset_builder(repo_id, config_name, revision=revision, token=token)
    for data_file in chain(*builder.config.data_files.values()):
        data_file_resolved_path = fs.resolve_path(data_file)
        if data_file_resolved_path.repo_id == repo_id:
            operations.append(CommitOperationDelete(path_in_repo=data_file_resolved_path.path_in_repo))
    # README.md
    dataset_card = DatasetCard.load(repo_id)
    # config_names
    if dataset_card.data.get("config_names", None) and config_name in dataset_card.data["config_names"]:
        dataset_card.data["config_names"].remove(config_name)
    # metadata_configs
    metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card.data)
    if metadata_configs:
        _ = metadata_configs.pop(config_name, None)
        dataset_card_data = DatasetCardData()
        metadata_configs.to_dataset_card_data(dataset_card_data)
        if datasets.config.METADATA_CONFIGS_FIELD in dataset_card_data:
            dataset_card.data[datasets.config.METADATA_CONFIGS_FIELD] = dataset_card_data[
                datasets.config.METADATA_CONFIGS_FIELD
            ]
        else:
            _ = dataset_card.data.pop(datasets.config.METADATA_CONFIGS_FIELD, None)
    # dataset_info
    dataset_infos: DatasetInfosDict = DatasetInfosDict.from_dataset_card_data(dataset_card.data)
    if dataset_infos:
        _ = dataset_infos.pop(config_name, None)
        dataset_card_data = DatasetCardData()
        dataset_infos.to_dataset_card_data(dataset_card_data)
        if "dataset_info" in dataset_card_data:
            dataset_card.data["dataset_info"] = dataset_card_data["dataset_info"]
        else:
            _ = dataset_card.data.pop("dataset_info", None)
    # Commit
    operations.append(
        CommitOperationAdd(path_in_repo=datasets.config.REPOCARD_FILENAME, path_or_fileobj=str(dataset_card).encode())
    )
    api = HfApi(
        endpoint=datasets.config.HF_ENDPOINT,
        token=token,
        library_name="datasets",
        library_version=__version__,
    )
    commit_info = api.create_commit(
        repo_id,
        operations=operations,
        commit_message=f"Delete '{config_name}' config",
        commit_description=f"Delete '{config_name}' config.",
        token=token,
        repo_type="dataset",
        revision=revision,
        create_pr=True,
    )
    print(f"You can find your PR to delete the dataset config at: {commit_info.pr_url}")
    return commit_info


def _delete_files(dataset_id, revision=None, token=None):
    hf_api = HfApi(
        endpoint=datasets.config.HF_ENDPOINT,
        token=token,
        library_name="datasets",
        library_version=__version__,
    )
    repo_files = hf_api.list_repo_files(
        dataset_id,
        repo_type="dataset",
    )
    if repo_files:
        legacy_json_file = []
        data_files = []
        for filename in repo_files:
            if filename in {".gitattributes", "README.md"}:
                continue
            elif filename == "dataset_infos.json":
                legacy_json_file.append(filename)
            else:
                data_files.append(filename)
        if legacy_json_file:
            hf_api.delete_file(
                "dataset_infos.json",
                dataset_id,
                repo_type="dataset",
                revision=revision,
                commit_message="Delete legacy dataset_infos.json",
            )
        if data_files:
            for filename in data_files:
                hf_api.delete_file(
                    filename,
                    dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    commit_message="Delete data file",
                )
<|endoftext|># SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from typing import Any, Optional, Union

from huggingface_hub import HfFileSystem

from . import config
from .table import CastError
from .utils.track import TrackedIterableFromGenerator, tracked_list, tracked_str


class DatasetsError(Exception):
    """Base class for exceptions in this library."""


class DefunctDatasetError(DatasetsError):
    """The dataset has been defunct."""


class FileNotFoundDatasetsError(DatasetsError, FileNotFoundError):
    """FileNotFoundError raised by this library."""


class DataFilesNotFoundError(FileNotFoundDatasetsError):
    """No (supported) data files found."""


class DatasetNotFoundError(FileNotFoundDatasetsError):
    """Dataset not found.

    Raised when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    """


class DatasetBuildError(DatasetsError):
    pass


class ManualDownloadError(DatasetBuildError):
    pass


class FileFormatError(DatasetBuildError):
    pass


class DatasetGenerationError(DatasetBuildError):
    pass


class DatasetGenerationCastError(DatasetGenerationError):
    @classmethod
    def from_cast_error(
        cls,
        cast_error: CastError,
        builder_name: str,
        gen_kwargs: dict[str, Any],
        token: Optional[Union[bool, str]],
    ) -> "DatasetGenerationCastError":
        explanation_message = (
            f"\n\nAll the data files must have the same columns, but at some point {cast_error.details()}"
        )
        formatted_tracked_gen_kwargs: list[str] = []
        for gen_kwarg in gen_kwargs.values():
            if not isinstance(gen_kwarg, (tracked_str, tracked_list, TrackedIterableFromGenerator)):
                continue
            while (
                isinstance(gen_kwarg, (tracked_list, TrackedIterableFromGenerator)) and gen_kwarg.last_item is not None
            ):
                gen_kwarg = gen_kwarg.last_item
            if isinstance(gen_kwarg, tracked_str):
                gen_kwarg = gen_kwarg.get_origin()
            if isinstance(gen_kwarg, str) and gen_kwarg.startswith("hf://"):
                resolved_path = HfFileSystem(endpoint=config.HF_ENDPOINT, token=token).resolve_path(gen_kwarg)
                gen_kwarg = "hf://" + resolved_path.unresolve()
                if "@" + resolved_path.revision in gen_kwarg:
                    gen_kwarg = (
                        gen_kwarg.replace("@" + resolved_path.revision, "", 1)
                        + f" (at revision {resolved_path.revision})"
                    )
            formatted_tracked_gen_kwargs.append(str(gen_kwarg))
        if formatted_tracked_gen_kwargs:
            explanation_message += f"\n\nThis happened while the {builder_name} dataset builder was generating data using\n\n{', '.join(formatted_tracked_gen_kwargs)}"
        help_message = "\n\nPlease either edit the data files to have matching columns, or separate them into different configurations (see docs at https://hf.co/docs/hub/datasets-manual-configuration#multiple-configurations)"
        return cls("An error occurred while generating the dataset" + explanation_message + help_message)


class ChecksumVerificationError(DatasetsError):
    """Error raised during checksums verifications of downloaded files."""


class UnexpectedDownloadedFileError(ChecksumVerificationError):
    """Some downloaded files were not expected."""


class ExpectedMoreDownloadedFilesError(ChecksumVerificationError):
    """Some files were supposed to be downloaded but were not."""


class NonMatchingChecksumError(ChecksumVerificationError):
    """The downloaded file checksum don't match the expected checksum."""


class SplitsVerificationError(DatasetsError):
    """Error raised during splits verifications."""


class UnexpectedSplitsError(SplitsVerificationError):
    """The expected splits of the downloaded file is missing."""


class ExpectedMoreSplitsError(SplitsVerificationError):
    """Some recorded splits are missing."""


class NonMatchingSplitsSizesError(SplitsVerificationError):
    """The splits sizes don't match the expected splits sizes."""
<|endoftext|>import inspect
import re
from typing import Dict, List, Tuple

from huggingface_hub.utils import insecure_hashlib

from .arrow import arrow
from .audiofolder import audiofolder
from .cache import cache
from .conll import conll
from .csv import csv
from .eval import eval
from .fasta import fasta
from .fastq import fastq
from .genbank import genbank
from .harbor import harbor
from .hdf5 import hdf5
from .iceberg import iceberg
from .imagefolder import imagefolder
from .json import json
from .lance import lance
from .meshfolder import meshfolder
from .mmcif import mmcif
from .niftifolder import niftifolder
from .pandas import pandas
from .parquet import parquet
from .pdb import pdb
from .pdffolder import pdffolder
from .sql import sql
from .text import text
from .tsfile import tsfile
from .videofolder import videofolder
from .vortex import vortex
from .webdataset import webdataset
from .xml import xml


def _hash_python_lines(lines: list[str]) -> str:
    filtered_lines = []
    for line in lines:
        line = re.sub(r"#.*", "", line)  # remove comments
        if line:
            filtered_lines.append(line)
    full_str = "\n".join(filtered_lines)

    # Make a hash from all this code
    full_bytes = full_str.encode("utf-8")
    return insecure_hashlib.sha256(full_bytes).hexdigest()


# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES = {
    "csv": (csv.__name__, _hash_python_lines(inspect.getsource(csv).splitlines())),
    "json": (json.__name__, _hash_python_lines(inspect.getsource(json).splitlines())),
    "pandas": (pandas.__name__, _hash_python_lines(inspect.getsource(pandas).splitlines())),
    "parquet": (parquet.__name__, _hash_python_lines(inspect.getsource(parquet).splitlines())),
    "arrow": (arrow.__name__, _hash_python_lines(inspect.getsource(arrow).splitlines())),
    "text": (text.__name__, _hash_python_lines(inspect.getsource(text).splitlines())),
    "conll": (conll.__name__, _hash_python_lines(inspect.getsource(conll).splitlines())),
    "imagefolder": (imagefolder.__name__, _hash_python_lines(inspect.getsource(imagefolder).splitlines())),
    "audiofolder": (audiofolder.__name__, _hash_python_lines(inspect.getsource(audiofolder).splitlines())),
    "videofolder": (videofolder.__name__, _hash_python_lines(inspect.getsource(videofolder).splitlines())),
    "meshfolder": (meshfolder.__name__, _hash_python_lines(inspect.getsource(meshfolder).splitlines())),
    "pdffolder": (pdffolder.__name__, _hash_python_lines(inspect.getsource(pdffolder).splitlines())),
    "niftifolder": (niftifolder.__name__, _hash_python_lines(inspect.getsource(niftifolder).splitlines())),
    "webdataset": (webdataset.__name__, _hash_python_lines(inspect.getsource(webdataset).splitlines())),
    "xml": (xml.__name__, _hash_python_lines(inspect.getsource(xml).splitlines())),
    "hdf5": (hdf5.__name__, _hash_python_lines(inspect.getsource(hdf5).splitlines())),
    "eval": (eval.__name__, _hash_python_lines(inspect.getsource(eval).splitlines())),
    "lance": (lance.__name__, _hash_python_lines(inspect.getsource(lance).splitlines())),
    "tsfile": (tsfile.__name__, _hash_python_lines(inspect.getsource(tsfile).splitlines())),
    "iceberg": (iceberg.__name__, _hash_python_lines(inspect.getsource(iceberg).splitlines())),
    "vortex": (vortex.__name__, _hash_python_lines(inspect.getsource(vortex).splitlines())),
    "mmcif": (mmcif.__name__, _hash_python_lines(inspect.getsource(mmcif).splitlines())),
    "pdb": (pdb.__name__, _hash_python_lines(inspect.getsource(pdb).splitlines())),
    "genbank": (genbank.__name__, _hash_python_lines(inspect.getsource(genbank).splitlines())),
    "fasta": (fasta.__name__, _hash_python_lines(inspect.getsource(fasta).splitlines())),
    "fastq": (fastq.__name__, _hash_python_lines(inspect.getsource(fastq).splitlines())),
    "harbor": (harbor.__name__, _hash_python_lines(inspect.getsource(harbor).splitlines())),
}

# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES_2_15_HASHES = {
    "csv": "eea64c71ca8b46dd3f537ed218fc9bf495d5707789152eb2764f5c78fa66d59d",
    "json": "8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96",
    "pandas": "3ac4ffc4563c796122ef66899b9485a3f1a977553e2d2a8a318c72b8cc6f2202",
    "parquet": "ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1",
    "arrow": "74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137",
    "text": "c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34",
    "imagefolder": "7b7ce5247a942be131d49ad4f3de5866083399a0f250901bd8dc202f8c5f7ce5",
    "audiofolder": "d3c1655c66c8f72e4efb5c79e952975fa6e2ce538473a6890241ddbddee9071c",
}

# Used to infer the module to use based on the data files extensions
_EXTENSION_TO_MODULE: dict[str, tuple[str, dict]] = {
    ".csv": ("csv", {}),
    ".tsv": ("csv", {"sep": "\t"}),
    ".json": ("json", {}),
    ".jsonl": ("json", {}),
    # ndjson is no longer maintained (see: https://github.com/ndjson/ndjson-spec/issues/35#issuecomment-1285673417)
    ".ndjson": ("json", {}),
    ".parquet": ("parquet", {}),
    ".geoparquet": ("parquet", {}),
    ".gpq": ("parquet", {}),
    ".arrow": ("arrow", {}),
    ".txt": ("text", {}),
    ".conll": ("conll", {}),
    ".conllu": ("conll", {"comment_prefix": "#"}),
    ".tar": ("webdataset", {}),
    ".xml": ("xml", {}),
    ".hdf5": ("hdf5", {}),
    ".h5": ("hdf5", {}),
    ".eval": ("eval", {}),
    ".toml": ("harbor", {}),
    ".md": ("harbor", {}),
    ".lance": ("lance", {}),
    ".tsfile": ("tsfile", {}),
    ".vortex": ("vortex", {}),
    # FASTA biological sequence formats
    ".fa": ("fasta", {}),
    ".fasta": ("fasta", {}),
    ".fna": ("fasta", {}),  # FASTA nucleic acid
    ".ffn": ("fasta", {}),  # FASTA nucleotide of gene regions
    ".faa": ("fasta", {}),  # FASTA amino acid
    ".frn": ("fasta", {}),  # FASTA non-coding RNA
    ".afa": ("fasta", {}),  # aligned FASTA (multiple sequence alignment)
    # other bio formats
    ".gb": ("genbank", {}),
    ".gbk": ("genbank", {}),
    ".genbank": ("genbank", {}),
    ".fq": ("fastq", {}),
    ".fastq": ("fastq", {}),
    ".pdb": ("pdb", {}),
    ".ent": ("pdb", {}),
    ".cif": ("mmcif", {}),
    ".mmcif": ("mmcif", {}),
}
_EXTENSION_TO_MODULE.update({ext: ("pdb", {}) for ext in pdb.PdbFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("pdb", {}) for ext in pdb.PdbFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("imagefolder", {}) for ext in imagefolder.ImageFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("imagefolder", {}) for ext in imagefolder.ImageFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("audiofolder", {}) for ext in audiofolder.AudioFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("audiofolder", {}) for ext in audiofolder.AudioFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("videofolder", {}) for ext in videofolder.VideoFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("videofolder", {}) for ext in videofolder.VideoFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("meshfolder", {}) for ext in meshfolder.MeshFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("meshfolder", {}) for ext in meshfolder.MeshFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("pdffolder", {}) for ext in pdffolder.PdfFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("pdffolder", {}) for ext in pdffolder.PdfFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("niftifolder", {}) for ext in niftifolder.NiftiFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("niftifolder", {}) for ext in niftifolder.NiftiFolder.EXTENSIONS})

# Used to filter data files based on extensions given a module name
_MODULE_TO_EXTENSIONS: dict[str, list[str]] = {}
for _ext, (_module, _) in _EXTENSION_TO_MODULE.items():
    _MODULE_TO_EXTENSIONS.setdefault(_module, []).append(_ext)

for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_EXTENSIONS[_module].append(".zip")

# Used to filter data files based on file names
_MODULE_TO_METADATA_FILE_NAMES: Dict[str, List[str]] = {}
for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_METADATA_FILE_NAMES[_module] = []
_MODULE_TO_METADATA_FILE_NAMES["imagefolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["audiofolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["videofolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["meshfolder"] = meshfolder.MeshFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["pdffolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["niftifolder"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["lance"] = lance.Lance.METADATA_FILE_NAMES
_MODULE_TO_METADATA_FILE_NAMES["mmcif"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["pdb"] = imagefolder.ImageFolder.METADATA_FILENAMES
_MODULE_TO_METADATA_FILE_NAMES["harbor"] = harbor.Harbor.METADATA_FILENAMES

_MODULE_TO_METADATA_EXTENSIONS: Dict[str, List[str]] = {}
for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_METADATA_EXTENSIONS[_module] = []
_MODULE_TO_METADATA_EXTENSIONS["lance"] = lance.Lance.METADATA_EXTENSIONS

# Total

_ALL_EXTENSIONS = list(_EXTENSION_TO_MODULE.keys()) + [".zip"]
_ALL_METADATA_EXTENSIONS = sorted({_ext for _exts in _MODULE_TO_METADATA_EXTENSIONS.values() for _ext in _exts})
_ALL_ALLOWED_EXTENSIONS = _ALL_EXTENSIONS + _ALL_METADATA_EXTENSIONS
_ALL_METADATA_FILENAMES = sorted(
    {file_name for file_names in _MODULE_TO_METADATA_FILE_NAMES.values() for file_name in file_names}
)
<|endoftext|>from dataclasses import dataclass
from io import StringIO
from typing import Literal, Optional

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class TextConfig(datasets.BuilderConfig):
    """BuilderConfig for text files.

    Args:
        features: (`Features`, *optional*):
            Cast the data to `features`.
        encoding: (`str`, defaults to "utf-8"):
            Encoding to decode the file.
        encoding_errors: (`str`, *optional*):
            Argument to define what to do in case of encoding error.
            This is the same as the `error` argument in `open()`.
        chunksize: (`Features`, *optional*, defaults to "10MB"):
            Chunk size to read the data.
        keep_linebreaks: (`bool`, defaults to False):
            Whether to keep line breaks.
        sample_by (`Literal["line", "paragraph", "document"]`, defaults to "line"):
            Whether to load data per line, praragraph or document.
            By default one row in the dataset = one line.
    """

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    chunksize: int = 10 << 20  # 10MB
    keep_linebreaks: bool = False
    sample_by: Literal["line", "paragraph", "document"] = "line"


class Text(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = TextConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """The `data_files` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.
        """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        base_data_files = dl_manager.download(self.config.data_files)
        extracted_data_files = dl_manager.extract(base_data_files)
        splits = []
        for split_name, files in extracted_data_files.items():
            files_iterables = [dl_manager.iter_files(file) for file in files]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={"files_iterables": files_iterables, "base_files": base_data_files[split_name]},
                )
            )
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all(not require_storage_cast(feature) for feature in self.config.features.values()):
                # cheaper cast
                pa_table = pa_table.cast(schema)
            else:
                # more expensive cast; allows str <-> int/float or str to Audio for example
                pa_table = table_cast(pa_table, schema)
            return pa_table
        else:
            return pa_table.cast(pa.schema({"text": pa.string()}))

    def _generate_shards(self, base_files, files_iterables):
        yield from base_files

    def _generate_tables(self, base_files, files_iterables):
        pa_table_names = list(self.config.features) if self.config.features is not None else ["text"]
        for shard_idx, files_iterable in enumerate(files_iterables):
            for file in files_iterable:
                # open in text mode, by default translates universal newlines ("\n", "\r\n" and "\r") into "\n"
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    if self.config.sample_by == "line":
                        batch_idx = 0
                        while True:
                            batch = f.read(self.config.chunksize)
                            if not batch:
                                break
                            batch += f.readline()  # finish current line
                            # StringIO.readlines, by default splits only on "\n" (and keeps line breaks)
                            batch = StringIO(batch).readlines()
                            if not self.config.keep_linebreaks:
                                batch = [line.rstrip("\n") for line in batch]
                            pa_table = pa.Table.from_arrays([pa.array(batch)], names=pa_table_names)
                            # Uncomment for debugging (will print the Arrow table size and elements)
                            # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                            # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                            yield Key(shard_idx, batch_idx), self._cast_table(pa_table)
                            batch_idx += 1
                    elif self.config.sample_by == "paragraph":
                        batch_idx = 0
                        batch = ""
                        while True:
                            new_batch = f.read(self.config.chunksize)
                            if not new_batch:
                                break
                            batch += new_batch
                            batch += f.readline()  # finish current line
                            batch = batch.split("\n\n")
                            pa_table = pa.Table.from_arrays(
                                [pa.array([example for example in batch[:-1] if example])], names=pa_table_names
                            )
                            # Uncomment for debugging (will print the Arrow table size and elements)
                            # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                            # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                            yield Key(shard_idx, batch_idx), self._cast_table(pa_table)
                            batch_idx += 1
                            batch = batch[-1]
                        if batch:
                            pa_table = pa.Table.from_arrays([pa.array([batch])], names=pa_table_names)
                            yield (shard_idx, batch_idx), self._cast_table(pa_table)
                    elif self.config.sample_by == "document":
                        text = f.read()
                        pa_table = pa.Table.from_arrays([pa.array([text])], names=pa_table_names)
                        yield Key(shard_idx, 0), self._cast_table(pa_table)
<|endoftext|>"""GenBank file loader for biological sequence data with annotations.

GenBank is a text-based format for storing nucleotide or protein sequences together with
their annotations and metadata, widely used in bioinformatics and maintained by NCBI.

This implementation uses a lightweight pure Python state machine parser,
requiring zero external dependencies.
"""

import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Optional

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


# Conservative limit to stay well under Parquet's i32::MAX page limit (~2GB)
# Using 256MB as default since Parquet compresses data and we want headroom
DEFAULT_MAX_BATCH_BYTES = 256 * 1024 * 1024  # 256 MB


# Parser states for the GenBank state machine
class ParserState:
    HEADER = "HEADER"
    FEATURES = "FEATURES"
    ORIGIN = "ORIGIN"
    COMPLETE = "COMPLETE"


class _FeatureAccumulator:
    """Collects the multi-line state of a GenBank FEATURES section.

    A feature spans a feature line (type + location) followed by ``/key=value``
    qualifier lines and their continuations. This buffers the feature and the
    qualifier currently being read, flushing them onto ``features`` at the right
    boundaries so the parser loop doesn't have to repeat that bookkeeping.
    """

    # Biopython's rule (Bio.GenBank.Scanner): wrapped qualifier lines join with a space,
    # except /translation whose wrapped lines concatenate directly.
    SPACELESS_QUALIFIERS = frozenset({"translation"})

    def __init__(self, location_parser: Callable[[str], dict]) -> None:
        self.features: list[dict] = []
        self._location_parser = location_parser
        self._feature: Optional[dict] = None
        self._location_text: list[str] = []
        self._qualifier_seen = False
        self._qualifier_key: Optional[str] = None
        self._qualifier_value: list[str] = []
        self._quote_open = False

    @property
    def has_open_qualifier(self) -> bool:
        return self._qualifier_key is not None

    @property
    def has_open_location(self) -> bool:
        """Whether the current feature's location is still being read.

        An operator location such as ``join(1..10,20..30,\n 40..50)`` may wrap, and
        it is incomplete for exactly as long as its parentheses are unbalanced. Once
        a qualifier line has been seen the location cannot continue, so that also
        closes it.
        """
        if self._feature is None or self._qualifier_seen:
            return False
        text = "".join(self._location_text)
        return text.count("(") > text.count(")")

    def _commit_qualifier(self) -> None:
        """Write the buffered qualifier (if any) onto the current feature.

        A qualifier key may repeat within one feature -- ``/db_xref`` routinely does --
        so each key holds the list of its values, as Biopython's
        ``SeqFeature.qualifiers`` does, instead of the last value overwriting the rest.
        """
        if self._feature is not None and self._qualifier_key is not None:
            values = self._feature["qualifiers"].setdefault(self._qualifier_key, [])
            spacer = "" if self._qualifier_key in self.SPACELESS_QUALIFIERS else " "
            values.append(spacer.join(self._qualifier_value))
        self._qualifier_key = None
        self._qualifier_value = []
        self._quote_open = False

    def finalize_feature(self) -> None:
        """Flush the pending qualifier and append the current feature, if any."""
        self._commit_qualifier()
        if self._feature is not None:
            self._feature["location"] = self._location_parser("".join(self._location_text))
            self.features.append(self._feature)
            self._feature = None
            self._location_text = []

    def begin_feature(self, feature_type: str, location_text: str) -> None:
        """Finalize the previous feature and start a new one.

        The location is kept as text and parsed on finalize, because it may still
        gain continuation lines.
        """
        self.finalize_feature()
        self._feature = {"type": feature_type, "location": None, "qualifiers": {}}
        self._location_text = [location_text]
        self._qualifier_seen = False

    def add_location_continuation(self, text: str) -> None:
        """Append a wrapped line to the location currently being read."""
        self._location_text.append(text)

    def begin_qualifier(self, key: str, value: str) -> None:
        """Commit the previous qualifier and start buffering a new one.

        ``value`` is the raw text after ``=``. A value that opens with ``"`` and does
        not close it on the same line stays open, and every following line belongs
        to it even when that line starts with ``/``.
        """
        self._commit_qualifier()
        self._qualifier_seen = True
        self._qualifier_key = key
        if value.startswith('"'):
            value = value[1:]
            self._quote_open = not value.endswith('"')
            value = value[:-1] if not self._quote_open else value
        self._qualifier_value = [value]

    def add_continuation(self, text: str) -> None:
        """Append a continuation line to the qualifier currently being read."""
        if self._quote_open and text.endswith('"'):
            text = text[:-1]
            self._quote_open = False
        self._qualifier_value.append(text)

    @property
    def has_open_quote(self) -> bool:
        """Whether the current qualifier's quoted value has not been closed yet."""
        return self._quote_open


@dataclass
class _HeaderState:
    """Tracks which header field the next continuation line belongs to.

    A header value may wrap onto following lines indented into the value column, and a
    continuation belongs to the field whose keyword opened it. Each keyword handler
    records that field here; keywords this loader does not read (REFERENCE, COMMENT and
    the rest) clear it, so their continuation lines are dropped rather than appended to
    whichever field happened to come before them.
    """

    #: Column at which header values start. Keywords occupy the margin to its left,
    #: so a line that is blank up to this column is a continuation.
    VALUE_COLUMN = 12

    active_field: Optional[str] = None


@dataclass
class GenBankConfig(datasets.BuilderConfig):
    """BuilderConfig for GenBank files.

    Args:
        features: Dataset features (optional, will be inferred if not provided).
        batch_size: Maximum number of records per batch. Works in conjunction with
            max_batch_bytes - a batch is flushed when either limit is reached.
        max_batch_bytes: Maximum cumulative bytes per batch. This prevents Parquet
            page size errors when dealing with very large sequences. Set to None
            to disable byte-based batching.
        columns: Subset of columns to include. Options: ["locus_name", "accession",
            "version", "definition", "organism", "taxonomy", "keywords", "sequence",
            "features", "length", "molecule_type", "secondary_accessions", "contig", "record"].
    """

    features: Optional[datasets.Features] = None
    batch_size: int = 10000
    max_batch_bytes: Optional[int] = DEFAULT_MAX_BATCH_BYTES
    columns: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()


class GenBank(datasets.ArrowBasedBuilder):
    """Dataset builder for GenBank files."""

    BUILDER_CONFIG_CLASS = GenBankConfig

    # All supported GenBank extensions
    EXTENSIONS: list[str] = [".gb", ".gbk", ".genbank"]

    # Canonical features for a GenBank record. The schema is always the same, so it is
    # declared here and passed to DatasetInfo; users can still override via config.features.
    DEFAULT_FEATURES: "datasets.Features" = datasets.Features(
        {
            "locus_name": datasets.Value("string"),
            "accession": datasets.Value("string"),
            "version": datasets.Value("string"),
            "definition": datasets.Value("string"),
            "organism": datasets.Value("string"),
            "taxonomy": datasets.Value("string"),
            "keywords": datasets.Value("string"),
            "sequence": datasets.Value("large_string"),
            "features": datasets.Json(),
            "length": datasets.Value("int64"),
            "molecule_type": datasets.Value("string"),
            "secondary_accessions": datasets.List(datasets.Value("string")),
            "contig": datasets.Value("string"),
            "record": datasets.BioSequence(format="genbank"),
        }
    )

    # All available columns (the canonical feature order).
    ALL_COLUMNS: list[str] = list(DEFAULT_FEATURES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DatasetBuilder applies features= after _info(); project the effective
        # schema again, preserving features supplied through info= as well.
        self.info.features = self._info().features

    def _info(self):
        features = self.info.features if hasattr(self, "info") else self.config.features
        columns = self._get_columns()
        if features is None:
            features = datasets.Features({col: self.DEFAULT_FEATURES[col] for col in columns})
        if self.config.columns is not None:
            missing = [col for col in self.config.columns if col not in features]
            if missing:
                raise ValueError(f"columns {missing} are not in features {list(features)}")
            features = datasets.Features({col: features[col] for col in self.config.columns})
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        """Generate splits from data files.

        The `data_files` kwarg in load_dataset() can be a str, List[str],
        Dict[str,str], or Dict[str,List[str]].

        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.
        """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        """Cast the raw Arrow table to the resolved features schema.

        DatasetInfo holds the effective schema from ``features=``, ``info=``, or
        the canonical ``DEFAULT_FEATURES``, projected to the selected columns.
        This is what turns the JSON-encoded ``features`` column into the ``Json()`` type.
        """
        features = self.info.features
        if all(not require_storage_cast(feature) for feature in features.values()):
            return pa_table.cast(features.arrow_schema)
        return table_cast(pa_table, features.arrow_schema)

    def _parse_feature_location(self, location_str: str) -> dict:
        """Parse a GenBank feature location string into a structured dict.

        Examples:
            "100..200" -> {"start": 100, "end": 200, "strand": 1, "start_partial": False, "end_partial": False}
            "<100..>200" -> same, with "start_partial": True, "end_partial": True
            "complement(100..200)" -> {"start": 100, "end": 200, "strand": -1, ...}
            "join(1..100,200..300)" -> {"start": 1, "end": 300, "strand": 1, "operator": "join", "parts": [[1,100],[200,300]]}
        """
        return self._parse_location_node(location_str.strip())

    _LOCATION_OPERATOR_RE = re.compile(r"^(complement|join|order)\((.*)\)$", re.S)
    _LOCATION_RANGE_RE = re.compile(r"^(<?)(\d+)(?:(?:\.\.|\^)(>?)(\d+))?$")

    @classmethod
    def _parse_location_node(cls, text: str) -> dict:
        """Parse one location expression; operators nest, so this recurses.

        ``complement(x)`` flips the strand of whatever ``x`` yields. ``join``/``order``
        parse each comma-separated part on its own (commas inside nested parentheses
        do not split). A join whose parts sit on different strands has no single
        strand and reports ``None``, as Biopython's CompoundLocation does.
        """
        operator_match = cls._LOCATION_OPERATOR_RE.match(text)
        if operator_match:
            operator, inner = operator_match.groups()
            if operator == "complement":
                location = cls._parse_location_node(inner)
                location["strand"] = -location["strand"] if location["strand"] is not None else None
                return location
            parts = [cls._parse_location_node(part) for part in cls._split_top_level(inner)]
            strands = {part["strand"] for part in parts}
            location = {"strand": strands.pop() if len(strands) == 1 else None, "operator": operator}
            located = [part for part in parts if "start" in part]
            if located:
                location["parts"] = [[part["start"], part["end"]] for part in located]
                # Parts are listed in the file's (transcription) order, which is descending
                # genomic order for minus-strand features (NCBI/Biopython write a two-exon
                # minus-strand CDS as ``complement(join(<high>..<higher>,<low>..<lower>))``).
                # The aggregate span is therefore min-start/max-end over all parts, matching
                # Biopython's CompoundLocation.start/.end, not the first/last part positionally.
                start_part = min(located, key=lambda part: part["start"])
                end_part = max(located, key=lambda part: part["end"])
                location["start"] = start_part["start"]
                location["end"] = end_part["end"]
                location["start_partial"] = start_part["start_partial"]
                location["end_partial"] = end_part["end_partial"]
            return location

        location = {"strand": 1}
        # A remote reference (ACCESSION.VERSION:range) keeps only the range here.
        range_match = cls._LOCATION_RANGE_RE.match(text.rsplit(":", 1)[-1].strip())
        if range_match:
            before, start, after, end = range_match.groups()
            location["start"] = int(start)
            location["end"] = int(end or start)
            # "<" and ">" mark a boundary that lies beyond the given coordinate
            # (Biopython's BeforePosition / AfterPosition).
            location["start_partial"] = before == "<"
            location["end_partial"] = after == ">"
        return location

    @staticmethod
    def _split_top_level(text: str) -> list[str]:
        """Split on commas that are not nested inside parentheses."""
        parts, depth, current = [], 0, []
        for char in text:
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            depth += (char == "(") - (char == ")")
            current.append(char)
        parts.append("".join(current).strip())
        return [part for part in parts if part]

    def _parse_genbank(self, fp):
        """State machine parser for GenBank format.

        GenBank format has several sections:
        - LOCUS: Contains name, length, molecule type, etc.
        - DEFINITION: Description of the sequence
        - ACCESSION: Database accession number
        - VERSION: Version with GI number
        - KEYWORDS: Associated keywords
        - SOURCE/ORGANISM: Organism information and taxonomy
        - FEATURES: Detailed annotations
        - ORIGIN: The actual sequence data
        - // : Record terminator

        Args:
            fp: File-like object opened in text mode.

        Yields:
            Dict with parsed record fields for each GenBank record.
        """
        state = ParserState.HEADER
        record = self._new_record()
        header = _HeaderState()
        features = _FeatureAccumulator(self._parse_feature_location)
        capture_record = "record" in self._get_columns()
        raw_lines = []

        for line in fp:
            # Keep the original text, including ignored fields and whitespace, from
            # LOCUS through // without buffering it when the column is dropped.
            if capture_record and (raw_lines or line.startswith("LOCUS")):
                raw_lines.append(line)

            # Record terminator: finalize the pending feature and emit the record.
            if line.startswith("//"):
                features.finalize_feature()
                record["features"] = features.features
                record["contig"] = "".join(record["contig"].split())
                if raw_lines:
                    record["record"] = {"bytes": "".join(raw_lines).encode("utf-8"), "path": None}
                yield record
                state = ParserState.HEADER
                record = self._new_record()
                header = _HeaderState()
                features = _FeatureAccumulator(self._parse_feature_location)
                raw_lines = []
                continue

            if state == ParserState.HEADER:
                state = self._handle_header_line(line, record, header) or state
            elif state == ParserState.FEATURES:
                state = self._handle_features_line(line, features, record, header) or state
            elif state == ParserState.ORIGIN:
                self._handle_origin_line(line, record)

    def _handle_header_line(self, line: str, record: dict, header: "_HeaderState") -> Optional[str]:
        """Parse one HEADER line into ``record``.

        ``header`` carries the field that a continuation line would extend, and this
        updates it on every keyword line.

        Returns the next parser state when a section boundary (FEATURES or
        ORIGIN) is reached, otherwise ``None`` to stay in the HEADER state.
        """
        value = line[header.VALUE_COLUMN :].strip()
        if not line.strip():
            return None

        # Blank in the keyword margin means this line continues the field above it.
        if not line[: header.VALUE_COLUMN].strip():
            if header.active_field:
                self._append_header_continuation(record, header.active_field, line.strip())
            return None

        if line.startswith("FEATURES"):
            header.active_field = None
            return ParserState.FEATURES
        if line.startswith("ORIGIN"):
            header.active_field = None
            return ParserState.ORIGIN

        if line.startswith("LOCUS"):
            self._parse_locus_line(line, record)
            header.active_field = None
        elif line.startswith("DEFINITION"):
            record["definition"] = value
            header.active_field = "definition"
        elif line.startswith("ACCESSION"):
            accessions = value.split()
            record["accession"] = accessions[0] if accessions else ""
            record["secondary_accessions"] = accessions[1:]
            header.active_field = "secondary_accessions"
        elif line.startswith("CONTIG"):
            record["contig"] = value
            header.active_field = "contig"
        elif line.startswith("VERSION"):
            record["version"] = value
            header.active_field = None
        elif line.startswith("KEYWORDS"):
            if value != ".":
                record["keywords"] = value
            header.active_field = "keywords"
        elif line.startswith("SOURCE"):
            # The SOURCE line itself is less useful than ORGANISM.
            header.active_field = None
        elif line.startswith("  ORGANISM"):
            record["organism"] = value
            # The lines under ORGANISM are the taxonomy listing, not more organism name.
            header.active_field = "taxonomy"
        else:
            # A keyword this loader does not read; its continuations belong to nothing.
            header.active_field = None
        return None

    @staticmethod
    def _append_header_continuation(record: dict, field: str, text: str) -> None:
        """Append a wrapped line to ``field``, joining with a single space.

        No delimiter is inserted: a wrapped value already carries its own punctuation,
        so a taxonomy listing that ends a line with ``;`` keeps exactly that one ``;``.
        """
        if isinstance(record[field], list):
            record[field].extend(text.split())
        elif field == "contig":
            record[field] += text  # a location expression; whitespace is not part of it
        else:
            record[field] = f"{record[field]} {text}".strip() if record[field] else text

    def _handle_features_line(
        self, line: str, features: "_FeatureAccumulator", record: dict, header: "_HeaderState"
    ) -> Optional[str]:
        """Parse one FEATURES line into ``features``.

        Returns the next parser state when a keyword ends the section (ORIGIN, or a
        header keyword such as CONTIG that follows the feature table), otherwise
        ``None`` to stay in the FEATURES state.
        """
        # A line that starts at column 0 is a keyword (ORIGIN, BASE COUNT, CONTIG, ...),
        # never a feature: it ends the open feature and only ORIGIN changes state.
        if line[:1].strip():
            features.finalize_feature()
            if line.startswith("ORIGIN"):
                return ParserState.ORIGIN
            if line.startswith("CONTIG"):
                return self._handle_header_line(line, record, header) or ParserState.HEADER
            return None

        # A feature starts with its type at column 5, e.g. "     gene   1..100".
        if len(line) > 5 and line[5] != " ":
            parts = line[5:].split(None, 1)
            if len(parts) >= 2:
                features.begin_feature(parts[0], parts[1].strip())
            else:
                features.finalize_feature()

        # Inside an unclosed quoted value every line is a continuation, '/' included.
        elif len(line) > 21 and features.has_open_quote:
            features.add_continuation(line[21:].strip())
        # A qualifier starts with "/key=value" at column 21.
        elif len(line) > 21 and line[21] == "/":
            qualifier_line = line[21:].strip()
            if "=" in qualifier_line:
                key, value = qualifier_line.split("=", 1)
                features.begin_qualifier(key[1:], value)  # key[1:] drops the leading '/'
            else:
                features.begin_qualifier(qualifier_line[1:], "")  # valueless qualifier, e.g. /pseudo

        # An unbalanced location wraps onto the next line before any qualifier appears.
        elif len(line) > 21 and line[21] != "/" and features.has_open_location:
            features.add_location_continuation(line[21:].strip())

        # Anything else at column 21+ is a continuation of the current qualifier's value.
        elif len(line) > 21 and line[21] != "/" and features.has_open_qualifier:
            features.add_continuation(line[21:].strip())

        return None

    def _handle_origin_line(self, line: str, record: dict) -> None:
        """Accumulate sequence characters from one ORIGIN line into ``record``."""
        if line.startswith("//"):
            return  # Handled by the terminator at the top of the parse loop.
        if line[:1].strip():
            return  # A keyword line (CONTIG, BASE COUNT) carries no sequence characters.
        # ORIGIN lines look like "   123 atcgatcg atcgatcg ..."; keep only the bases.
        seq_chars = re.sub(r"[\s\d]", "", line)
        if seq_chars:
            record["sequence"] += seq_chars.upper()

    def _new_record(self) -> dict:
        """Create a new empty record with default values."""
        return {
            "locus_name": "",
            "accession": "",
            "version": "",
            "definition": "",
            "organism": "",
            "taxonomy": "",
            "keywords": "",
            "sequence": "",
            "features": [],
            "length": 0,
            "molecule_type": "",
            "secondary_accessions": [],
            "contig": "",
            "record": None,
        }

    _MOLECULE_TYPE_RE = re.compile(r"^(?:[sdm]s-)?(?:[a-z]*[DR]NA|NA)$")

    def _parse_locus_line(self, line: str, record: dict) -> None:
        """Parse the LOCUS line which contains key metadata.

        LOCUS format (fixed width columns):
        LOCUS       name          length bp    type     topology  division  date

        Example:
        LOCUS       SCU49845     5028 bp    DNA             PLN       21-JUN-1999
        """
        # Split by whitespace and extract fields
        parts = line.split()
        if len(parts) >= 2:
            record["locus_name"] = parts[1]

        # The length is a number followed by its unit: "bp" for nucleotides, "aa" for
        # amino acids. The unit is also what identifies a protein record, whose LOCUS
        # line carries no molecule-type token of its own.
        unit = None
        for i, part in enumerate(parts):
            if part.lower() in ("bp", "aa") and i > 0:
                unit = part.lower()
                try:
                    record["length"] = int(parts[i - 1])
                except ValueError:
                    pass
                break

        # Nucleotide records name their molecule type explicitly.
        # Molecule types are DNA/RNA (with a lowercase class prefix such as m, r, t, sn)
        # or the generic NA, optionally prefixed by strandedness ss-/ds-/ms-.
        for part in parts:
            if self._MOLECULE_TYPE_RE.match(part):
                record["molecule_type"] = part
                break
        else:
            if unit == "aa":
                record["molecule_type"] = "protein"

    def _get_columns(self) -> list[str]:
        """Get the list of columns to include in output.

        ``columns`` wins when given; otherwise a user-supplied ``features`` schema
        selects its own columns, so a schema naming a subset stays valid when the
        default schema grows.
        """
        features = self.info.features if hasattr(self, "info") else self.config.features
        if self.config.columns is None and features is not None:
            unknown = [col for col in features if col not in self.ALL_COLUMNS]
            if unknown:
                raise ValueError(f"Invalid feature column(s) {unknown}. Valid columns are: {self.ALL_COLUMNS}")
            return list(features)
        if self.config.columns is not None:
            # Validate columns
            seen = set()
            for col in self.config.columns:
                if col not in self.ALL_COLUMNS:
                    raise ValueError(f"Invalid column '{col}'. Valid columns are: {self.ALL_COLUMNS}")
                if col in seen:
                    raise ValueError(f"Duplicate column '{col}' in columns.")
                seen.add(col)
            return self.config.columns
        return self.ALL_COLUMNS

    def _get_storage_schema(self, columns: list[str]) -> pa.Schema:
        """Return the Arrow schema used to build raw batches before casting to features.

        The ``features`` column is built directly as the JSON arrow type from JSON-encoded
        strings, so casting to the ``Json()`` feature is a no-op (no double-encoding).
        ``sequence`` uses large_string to handle data that can exceed the 2GB limit of
        regular string type.
        """
        fields = []
        for col in columns:
            if col == "features":
                # JSON arrow type; matches the Json() feature so _cast_table won't re-encode.
                fields.append(pa.field(col, pa.json_()))
            elif col == "sequence":
                # Use large_string for potentially very long sequence data.
                fields.append(pa.field(col, pa.large_string()))
            elif col == "secondary_accessions":
                fields.append(pa.field(col, pa.list_(pa.string())))
            elif col == "length":
                fields.append(pa.field(col, pa.int64()))
            elif col == "record":
                fields.append(pa.field(col, datasets.BioSequence().pa_type))
            else:
                fields.append(pa.field(col, pa.string()))
        return pa.schema(fields)

    def _generate_tables(self, files):
        """Generate Arrow tables from GenBank files.

        Yields batches of records as Arrow tables for memory-efficient processing
        of large sequence files. Uses dual-threshold batching: flushes when either
        batch_size (record count) or max_batch_bytes (cumulative size) is reached.

        Args:
            files: Iterable of file iterables from _split_generators.

        Yields:
            Tuple of (Key, pa.Table) for each batch.
        """
        columns = self._get_columns()
        schema = self._get_storage_schema(columns)
        max_batch_bytes = self.config.max_batch_bytes

        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            batch_idx = 0
            batch = {col: [] for col in columns}
            batch_bytes = 0

            # Wrap the binary stream ourselves so compressed inputs also preserve
            # the original line endings in the BioSequence bytes.
            with open(file, "rb") as binary_fp, TextIOWrapper(binary_fp, encoding="utf-8", newline="") as fp:
                for record in self._parse_genbank(fp):
                    # Update length from actual sequence if not set
                    if record["length"] == 0 and record["sequence"]:
                        record["length"] = len(record["sequence"])

                    # Serialize the features list to a JSON string for storage; the
                    # Json() feature parses it back into an object on read.
                    if "features" in record:
                        record["features"] = json.dumps(record["features"])

                    # Calculate record size (approximate UTF-8 byte size)
                    record_bytes = (
                        sum(len(str(record.get(col, ""))) for col in columns if col not in ("length", "record")) + 8
                    )  # 8 bytes for int64 length
                    if "record" in columns and record["record"] is not None:
                        record_bytes += len(record["record"]["bytes"])

                    # Check if adding this record would exceed byte limit
                    # Flush current batch first if needed (but only if batch is non-empty)
                    if (
                        max_batch_bytes is not None
                        and batch_bytes > 0
                        and batch_bytes + record_bytes > max_batch_bytes
                    ):
                        pa_table = pa.Table.from_pydict(batch, schema=schema)
                        yield Key(file_idx, batch_idx), self._cast_table(pa_table)
                        batch = {col: [] for col in columns}
                        batch_bytes = 0
                        batch_idx += 1

                    # Add record to batch
                    for col in columns:
                        batch[col].append(record.get(col, self._new_record()[col]))
                    batch_bytes += record_bytes

                    # Yield batch when it reaches batch_size (record count limit)
                    if len(batch[columns[0]]) >= self.config.batch_size:
                        pa_table = pa.Table.from_pydict(batch, schema=schema)
                        yield Key(file_idx, batch_idx), self._cast_table(pa_table)
                        batch = {col: [] for col in columns}
                        batch_bytes = 0
                        batch_idx += 1

            # Yield remaining records in final batch
            if batch[columns[0]]:
                pa_table = pa.Table.from_pydict(batch, schema=schema)
                yield Key(file_idx, batch_idx), self._cast_table(pa_table)
<|endoftext|>"""Loader for Harbor task repositories (https://harborframework.com/docs).

A *Harbor task* is an executable agent-evaluation task: an instruction, a sandbox environment
definition and a verifier. It is stored as a **directory** rather than as a row of a file::

    my-harbor-dataset/
    ├── dataset.toml              # optional manifest pointing at the task directories
    ├── registry.json             # optional registry of dataset versions (git-repos datasets)
    └── tasks/
        ├── task-a/
