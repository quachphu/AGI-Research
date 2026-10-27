    >>> from datasets import Features
    >>> features = Features({'x': Array3D(shape=(1, 2, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    _type: str = field(default="Array3D", init=False, repr=False)


@dataclass
class Array4D(_ArrayXD):
    """Create a four-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array4D(shape=(1, 2, 2, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    _type: str = field(default="Array4D", init=False, repr=False)


@dataclass
class Array5D(_ArrayXD):
    """Create a five-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array5D(shape=(1, 2, 2, 3, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    _type: str = field(default="Array5D", init=False, repr=False)


class _ArrayXDExtensionType(pa.ExtensionType):
    ndims: Optional[int] = None

    def __init__(self, shape: tuple, dtype: str):
        if self.ndims is None or self.ndims <= 1:
            raise ValueError("You must instantiate an array type with a value for dim that is > 1")
        if len(shape) != self.ndims:
            raise ValueError(f"shape={shape} and ndims={self.ndims} don't match")
        for dim in range(1, self.ndims):
            if shape[dim] is None:
                raise ValueError(f"Support only dynamic size on first dimension. Got: {shape}")
        self.shape = tuple(shape)
        self.value_type = dtype
        self.storage_dtype = self._generate_dtype(self.value_type)
        pa.ExtensionType.__init__(self, self.storage_dtype, f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __arrow_ext_serialize__(self):
        return json.dumps((self.shape, self.value_type)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    # This was added to pa.ExtensionType in pyarrow >= 13.0.0
    def __reduce__(self):
        return self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__())

    def __hash__(self):
        return hash((self.__class__, self.shape, self.value_type))

    def __arrow_ext_class__(self):
        return ArrayExtensionArray

    def _generate_dtype(self, dtype):
        dtype = string_to_arrow(dtype)
        for d in reversed(self.shape):
            dtype = pa.list_(dtype)
            # Don't specify the size of the list, since fixed length list arrays have issues
            # being validated after slicing in pyarrow 0.17.1
        return dtype

    def to_pandas_dtype(self):
        return PandasArrayExtensionDtype(self.value_type)


class Array2DExtensionType(_ArrayXDExtensionType):
    ndims = 2


class Array3DExtensionType(_ArrayXDExtensionType):
    ndims = 3


class Array4DExtensionType(_ArrayXDExtensionType):
    ndims = 4


class Array5DExtensionType(_ArrayXDExtensionType):
    ndims = 5


# Register the extension types for deserialization
pa.register_extension_type(Array2DExtensionType((1, 2), "int64"))
pa.register_extension_type(Array3DExtensionType((1, 2, 3), "int64"))
pa.register_extension_type(Array4DExtensionType((1, 2, 3, 4), "int64"))
pa.register_extension_type(Array5DExtensionType((1, 2, 3, 4, 5), "int64"))


def _is_zero_copy_only(pa_type: pa.DataType, unnest: bool = False) -> bool:
    """
    When converting a pyarrow array to a numpy array, we must know whether this could be done in zero-copy or not.
    This function returns the value of the ``zero_copy_only`` parameter to pass to ``.to_numpy()``, given the type of the pyarrow array.

    # zero copy is available for all primitive types except booleans and temporal types (date, time, timestamp or duration)
    # primitive types are types for which the physical representation in arrow and in numpy
    # https://github.com/wesm/arrow/blob/c07b9b48cf3e0bbbab493992a492ae47e5b04cad/python/pyarrow/types.pxi#L821
    # see https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy
    # and https://issues.apache.org/jira/browse/ARROW-2871?jql=text%20~%20%22boolean%20to_numpy%22
    """

    def _unnest_pa_type(pa_type: pa.DataType) -> pa.DataType:
        if pa.types.is_list(pa_type):
            return _unnest_pa_type(pa_type.value_type)
        return pa_type

    if unnest:
        pa_type = _unnest_pa_type(pa_type)
    return pa.types.is_primitive(pa_type) and not (pa.types.is_boolean(pa_type) or pa.types.is_temporal(pa_type))


class ArrayExtensionArray(pa.ExtensionArray):
    def __array__(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        return self.to_numpy(zero_copy_only=zero_copy_only)

    def __getitem__(self, i):
        return self.storage[i]

    def to_numpy(self, zero_copy_only=True):
        storage: pa.ListArray = self.storage
        null_mask = storage.is_null().to_numpy(zero_copy_only=False)

        if self.type.shape[0] is not None:
            size = 1
            null_indices = np.arange(len(storage))[null_mask] - np.arange(np.sum(null_mask))

            for i in range(self.type.ndims):
                size *= self.type.shape[i]
                storage = storage.flatten()
            numpy_arr = storage.to_numpy(zero_copy_only=zero_copy_only)
            numpy_arr = numpy_arr.reshape(len(self) - len(null_indices), *self.type.shape)

            if len(null_indices):
                numpy_arr = np.insert(numpy_arr.astype(np.float64), null_indices, np.nan, axis=0)

        else:
            shape = self.type.shape
            ndims = self.type.ndims
            arrays = []
            first_dim_offsets = np.array([off.as_py() for off in storage.offsets])
            for i, is_null in enumerate(null_mask):
                if is_null:
                    arrays.append(np.nan)
                else:
                    storage_el = storage[i : i + 1]
                    first_dim = first_dim_offsets[i + 1] - first_dim_offsets[i]
                    # flatten storage
                    for _ in range(ndims):
                        storage_el = storage_el.flatten()

                    numpy_arr = storage_el.to_numpy(zero_copy_only=zero_copy_only)
                    arrays.append(numpy_arr.reshape(first_dim, *shape[1:]))

            if len(np.unique(np.diff(first_dim_offsets))) > 1:
                # ragged
                numpy_arr = np.empty(len(arrays), dtype=object)
                numpy_arr[:] = arrays
            else:
                numpy_arr = np.array(arrays)

        return numpy_arr

    def to_pylist(self, maps_as_pydicts: Optional[Literal["lossy", "strict"]] = None):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        numpy_arr = self.to_numpy(zero_copy_only=zero_copy_only)
        if self.type.shape[0] is None and numpy_arr.dtype == object:
            return [arr.tolist() for arr in numpy_arr.tolist()]
        elif self.type.shape[0] is not None and self.storage.null_count:
            # For a fixed-shape array, to_numpy casts the whole column to float64 to
            # hold np.nan in the null rows. On the python read path that silently
            # corrupts every non-null value: integers become floats and values above
            # 2**53 lose precision. The nested-list storage already carries the exact
            # values and None for the null rows, so build the list from it directly.
            return self.storage.to_pylist()
        else:
            return numpy_arr.tolist()


class PandasArrayExtensionDtype(PandasExtensionDtype):
    _metadata = "value_type"

    def __init__(self, value_type: Union["PandasArrayExtensionDtype", np.dtype]):
        self._value_type = value_type

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        if isinstance(array, pa.ChunkedArray):
            array = array.type.wrap_array(pa.concat_arrays([chunk.storage for chunk in array.chunks]))
        zero_copy_only = _is_zero_copy_only(array.storage.type, unnest=True)
        numpy_arr = array.to_numpy(zero_copy_only=zero_copy_only)
        return PandasArrayExtensionArray(numpy_arr)

    @classmethod
    def construct_array_type(cls):
        return PandasArrayExtensionArray

    @property
    def type(self) -> type:
        return np.ndarray

    @property
    def kind(self) -> str:
        return "O"

    @property
    def name(self) -> str:
        return f"array[{self.value_type}]"

    @property
    def value_type(self) -> np.dtype:
        return self._value_type


class PandasArrayExtensionArray(PandasExtensionArray):
    def __init__(self, data: np.ndarray, copy: bool = False):
        self._data = data if not copy else np.array(data)
        self._dtype = PandasArrayExtensionDtype(data.dtype)

    def __array__(self, dtype=None):
        """
        Convert to NumPy Array.
        Note that Pandas expects a 1D array when dtype is set to object.
        But for other dtypes, the returned shape is the same as the one of ``data``.

        More info about pandas 1D requirement for PandasExtensionArray here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

        """
        if dtype == np.dtype(object):
            out = np.empty(len(self._data), dtype=object)
            for i in range(len(self._data)):
                out[i] = self._data[i]
            return out
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype)

    def __arrow_array__(self, type=None):
        """
        Convert to a PyArrow ExtensionArray (pyarrow's ``__arrow_array__`` protocol).

        Called by ``pa.array`` and ``pa.Table.from_pandas``, for instance in ``Dataset.from_pandas``
        or when a pandas-formatted ``map`` returns a DataFrame that still has an ArrayXD column.
        """
        if isinstance(type, _ArrayXDExtensionType):
            pa_type = type
        else:
            pa_type = self._to_arrow_extension_type()
        if self._data.dtype == object:
            # dynamic first dimension: one array per row, np.nan for null rows
            data = [arr if isinstance(arr, np.ndarray) else None for arr in self._data]
        else:
            data = self._data
        storage = to_pyarrow_listarray(data, pa_type)
        if type is not None and not isinstance(type, _ArrayXDExtensionType):
            return storage.cast(type)
        return pa.ExtensionArray.from_storage(pa_type, storage)

    def _to_arrow_extension_type(self) -> "_ArrayXDExtensionType":
        if self._data.dtype == object:
            first_arr = next((arr for arr in self._data if isinstance(arr, np.ndarray)), None)
            if first_arr is None:
                raise ValueError(
                    "Cannot infer the Arrow type of a PandasArrayExtensionArray that only has null values, "
                    "please pass the type explicitly."
                )
            shape, value_type = (None, *first_arr.shape[1:]), first_arr.dtype
        else:
            shape, value_type = self._data.shape[1:], self._data.dtype
        extension_types = {
            2: Array2DExtensionType,
            3: Array3DExtensionType,
            4: Array4DExtensionType,
            5: Array5DExtensionType,
        }
        if len(shape) not in extension_types:
            raise ValueError(f"Unsupported number of dimensions for an ArrayXD column: {len(shape)} (shape={shape})")
        return extension_types[len(shape)](shape, str(value_type))

    def copy(self, deep: bool = False) -> "PandasArrayExtensionArray":
        return PandasArrayExtensionArray(self._data, copy=True)

    @classmethod
    def _from_sequence(
        cls, scalars, dtype: Optional[PandasArrayExtensionDtype] = None, copy: bool = False
    ) -> "PandasArrayExtensionArray":
        if len(scalars) > 1 and all(
            isinstance(x, np.ndarray) and x.shape == scalars[0].shape and x.dtype == scalars[0].dtype for x in scalars
        ):
            data = np.array(scalars, dtype=dtype if dtype is None else dtype.value_type, copy=copy)
        else:
            data = np.empty(len(scalars), dtype=object)
            data[:] = scalars
        return cls(data, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence_["PandasArrayExtensionArray"]) -> "PandasArrayExtensionArray":
        if len(to_concat) > 1 and all(
            va._data.shape == to_concat[0]._data.shape and va._data.dtype == to_concat[0]._data.dtype
            for va in to_concat
        ):
            data = np.vstack([va._data for va in to_concat])
        else:
            data = np.empty(len(to_concat), dtype=object)
            data[:] = [va._data for va in to_concat]
        return cls(data, copy=False)

    @property
    def dtype(self) -> PandasArrayExtensionDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        return np.array([pd.isna(arr).any() for arr in self._data])

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        raise NotImplementedError()

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> Union[np.ndarray, "PandasArrayExtensionArray"]:
        if isinstance(item, int):
            return self._data[item]
        return PandasArrayExtensionArray(self._data[item], copy=False)

    def take(
        self, indices: Sequence_[int], allow_fill: bool = False, fill_value: bool = None
    ) -> "PandasArrayExtensionArray":
        indices: np.ndarray = np.asarray(indices, dtype=int)
        if allow_fill:
            fill_value = (
                self.dtype.na_value if fill_value is None else np.asarray(fill_value, dtype=self.dtype.value_type)
            )
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError("Invalid value in `indices`, must be all >= -1 for `allow_fill` is True")
            elif len(self) > 0:
                pass
            elif not np.all(mask):
                raise IndexError("Invalid take for empty PandasArrayExtensionArray, must be all -1.")
            else:
                data = np.array([fill_value] * len(indices), dtype=self.dtype.value_type)
                return PandasArrayExtensionArray(data, copy=False)
        took = self._data.take(indices, axis=0)
        if allow_fill and mask.any():
            took[mask] = [fill_value] * np.sum(mask)
        return PandasArrayExtensionArray(took, copy=False)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if not isinstance(other, PandasArrayExtensionArray):
            raise NotImplementedError(f"Invalid type to compare to: {type(other)}")
        return (self._data == other._data).all()


def pandas_types_mapper(dtype):
    if isinstance(dtype, _ArrayXDExtensionType):
        return PandasArrayExtensionDtype(dtype.value_type)


@dataclass
class ClassLabel:
    """Feature type for integer class labels.

    There are 3 ways to define a `ClassLabel`, which correspond to the 3 arguments:

     * `num_classes`: Create 0 to (num_classes-1) labels.
     * `names`: List of label strings.
     * `names_file`: File containing the list of labels.

    Under the hood the labels are stored as integers.
    You can use -1 to represent unknown/missing labels.

    Args:
        num_classes (`int`, *optional*):
            Number of classes. All labels must be < `num_classes`.
        names (`list` of `str`, *optional*):
            String names for the integer classes.
            The order in which the names are provided is kept.
        names_file (`str`, *optional*):
            Path to a file with names for the integer classes, one per line.

    Example:

    ```py
    >>> from datasets import Features, ClassLabel
    >>> features = Features({'label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'])})
    >>> features
    {'label': ClassLabel(names=['bad', 'ok', 'good'])}
    ```
    """

    num_classes: InitVar[Optional[int]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    names: list[str] = None
    names_file: InitVar[Optional[str]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "int64"
    pa_type: ClassVar[Any] = pa.int64()
    _str2int: ClassVar[dict[str, int]] = None
    _int2str: ClassVar[dict[int, int]] = None
    _type: str = field(default="ClassLabel", init=False, repr=False)

    def __post_init__(self, num_classes, names_file):
        self.num_classes = num_classes
        self.names_file = names_file
        if self.names_file is not None and self.names is not None:
            raise ValueError("Please provide either names or names_file but not both.")
        # Set self.names
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_names_from_file(self.names_file)
            elif self.num_classes is not None:
                self.names = [str(i) for i in range(self.num_classes)]
            else:
                raise ValueError("Please provide either num_classes, names or names_file.")
        elif not isinstance(self.names, SequenceABC):
            raise TypeError(f"Please provide names as a list, is {type(self.names)}")
        # Set self.num_classes
        if self.num_classes is None:
            self.num_classes = len(self.names)
        elif self.num_classes != len(self.names):
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                f"Got {len(self.names)} names VS {self.num_classes} num_classes"
            )
        # Prepare mappings
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError("Some label names are duplicated. Each label name should be unique.")

    def __call__(self):
        return self.pa_type

    def str2int(self, values: Union[str, Iterable]) -> Union[int, Iterable]:
        """Conversion class name `string` => `integer`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
        >>> ds.features["label"].str2int('neg')
        0
        ```
        """
        if not isinstance(values, str) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = [self._strval2int(value) for value in values]
        return output if return_list else output[0]

    def _strval2int(self, value: str) -> int:
        failed_parse = False
        value = str(value)
        # first attempt - raw string value
        int_value = self._str2int.get(value)
        if int_value is None:
            # second attempt - strip whitespace
            int_value = self._str2int.get(value.strip())
            if int_value is None:
                # third attempt - convert str to int
                try:
                    int_value = int(value)
                except ValueError:
                    failed_parse = True
                else:
                    if int_value < -1 or int_value >= self.num_classes:
                        failed_parse = True
        if failed_parse:
            raise ValueError(f"Invalid string class label {value}")
        return int_value

    def int2str(self, values: Union[int, Iterable]) -> Union[str, Iterable]:
        """Conversion `integer` => class name `string`.

        Regarding unknown/missing labels: passing negative integers raises `ValueError`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
        >>> ds.features["label"].int2str(0)
        'neg'
        ```
        """
        if not isinstance(values, int) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.num_classes:
                raise ValueError(f"Invalid integer class label {v:d}")

        output = [self._int2str[int(v)] for v in values]
        return output if return_list else output[0]

    def encode_example(self, example_data):
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel feature with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if not -1 <= example_data < self.num_classes:
            raise ValueError(f"Class label {example_data:d} greater than configured num_classes {self.num_classes}")
        return example_data

    def cast_storage(self, storage: Union[pa.StringArray, pa.IntegerArray]) -> pa.Int64Array:
        """Cast an Arrow array to the `ClassLabel` arrow storage type.
        The Arrow types that can be converted to the `ClassLabel` pyarrow storage type are:

        - `pa.string()`
        - `pa.int()`

        Args:
            storage (`Union[pa.StringArray, pa.IntegerArray]`):
                PyArrow array to cast.

        Returns:
            `pa.Int64Array`: Array in the `ClassLabel` arrow storage type.
        """
        if isinstance(storage, pa.IntegerArray) and len(storage) > 0:
            min_max = pc.min_max(storage).as_py()
            if min_max["max"] is not None and min_max["max"] >= self.num_classes:
                raise ValueError(
                    f"Class label {min_max['max']} greater than configured num_classes {self.num_classes}"
                )
        elif isinstance(storage, pa.StringArray):
            storage = pa.array(
                [self._strval2int(label) if label is not None else None for label in storage.to_pylist()]
            )
        return array_cast(storage, self.pa_type)

    @staticmethod
    def _load_names_from_file(names_filepath):
        with open(names_filepath, encoding="utf-8") as f:
            return [name.strip() for name in f.read().split("\n") if name.strip()]  # Filter empty names


@dataclass
class Json:
    """Feature type for JSON objects.

    Under the hood the objects are stored as JSON-encoded strings.

    Example:

    ```py
    >>> from datasets import Features, Json
    >>> features = Features({'json': Json()})
    >>> features
    {'json': Json()}
    ```

    ```py
    >>> from datasets import Dataset, Features, Json, List
    >>> features = Features({"a": List(Json())})
    >>> ds = Dataset.from_dict({"a": [[{"b": 0}, {"c": 0}]]}, features=features)
    >>> # OR
    >>> ds = Dataset.from_dict({"a": [[{"b": 0}, {"c": 0}]]}, on_mixed_types="use_json")
    >>> ds.features
    {'a': List(Json())}
    >>> ds[0]
    {'a': [{'b': 0}, {'c': 0}]}
    >>> def f(x):
    ...     for y in x["a"]:
    ...         y["d"] = "foo"
    ...     return x
    >>> ds = ds.map(f)
    >>> ds.features
    >>> ds[0]
    {'a': [{'b': 0, 'd': 'foo'}, {'c': 0, 'd': 'foo'}]}
    ```
    """

    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    pa_type: ClassVar[Any] = pa.json_()
    _type: str = field(default="Json", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, example_data):
        if example_data is None:
            return None
        if not isinstance(example_data, str):
            example_data = ujson_dumps(example_data)
        else:
            try:
                ujson_loads(example_data)
            except Exception:
                example_data = ujson_dumps(example_data)
        return example_data

    def decode_example(self, example_data, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Json(decode=True) instead.")
        if example_data is None:
            return None
        return ujson_loads(example_data)

    def cast_storage(self, storage: Union[pa.Array]) -> pa.JsonArray:
        """Cast an Arrow array to the `Json` arrow storage type.

        Args:
            storage (`Union[pa.StringArray, pa.IntegerArray]`):
                PyArrow array to cast.

        Returns:
            `pa.JsonArray`: Array in the `Json` arrow storage type.
        """
        if isinstance(storage, pa.JsonArray):
            return storage
        elif (
            pa.types.is_string(storage.type)
            or pa.types.is_large_string(storage.type)
            or pa.types.is_string_view(storage.type)
        ):
            items = storage[:5].to_pylist()
            try:
                for item in items:
                    if item is not None:
                        ujson_loads(item)
            except Exception:
                storage = pa.array(
                    [ujson_dumps(x) if x is not None else None for x in storage.to_pylist()], pa.json_()
                )
        else:
            storage = pa.array([ujson_dumps(x) if x is not None else None for x in storage.to_pylist()], pa.json_())
        return array_cast(storage, self.pa_type)


class Sequence:
    """
    A `Sequence` is a utility that automatically converts internal dictionary feature into a dictionary of
    lists. This behavior is implemented to have a compatibility layer with the TensorFlow Datasets library but may be
    un-wanted in some cases. If you don't want this behavior, you can use a [`List`] or a [`LargeList`]
    instead of the [`Sequence`].

    Args:
        feature ([`FeatureType`]):
            Child feature data type of each item within the large list.
        length (optional `int`, default to -1):
            Length of the list if it is fixed.
            Defaults to -1 which means an arbitrary length.

        Returns:
            [`List`] of the specified feature, except `dict` of sub-features
            which are converted to `dict` of lists of sub-features for compatibility with TFDS.

    """

    def __new__(cls, feature=None, length=-1, **kwargs):
        # useful to still get isinstance(Sequence(Value("int64")), Sequence)
        if (
            cls is Sequence
            and isinstance(feature, dict)
            and any(not isinstance(subfeature, List) for subfeature in feature.values())
        ):
            out = {key: List(value, length=length, **kwargs) for key, value in feature.items()}
        else:
            out = super().__new__(List)
        return out


@dataclass(repr=False)
class List(Sequence):
    """Feature type for large list data composed of child feature data type.

    It is backed by `pyarrow.ListType`, which uses 32-bit offsets or a fixed length.

    Args:
        feature ([`FeatureType`]):
            Child feature data type of each item within the large list.
        length (optional `int`, default to -1):
            Length of the list if it is fixed.
            Defaults to -1 which means an arbitrary length.
    """

    feature: Any
    length: int = -1
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="List", init=False, repr=False)

    def __repr__(self):
        if self.length != -1:
            return f"{type(self).__name__}({self.feature}, length={self.length})"
        else:
            return f"{type(self).__name__}({self.feature})"


@dataclass(repr=False)
class LargeList:
    """Feature type for large list data composed of child feature data type.

    It is backed by `pyarrow.LargeListType`, which is like `pyarrow.ListType` but with 64-bit rather than 32-bit offsets.

    Args:
        feature ([`FeatureType`]):
            Child feature data type of each item within the large list.
    """

    feature: Any
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="LargeList", init=False, repr=False)

    def __repr__(self):
        return f"{type(self).__name__}({self.feature})"


FeatureType = Union[
    dict,
    list,
    tuple,
    Value,
    ClassLabel,
    Translation,
    TranslationVariableLanguages,
    LargeList,
    List,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    Image,
    Mesh,
    Video,
    Pdf,
    Nifti,
    BioSequence,
    BioStructure,
]


def _check_non_null_non_empty_recursive(obj, schema: Optional[FeatureType] = None) -> bool:
    """
    Check if the object is not None.
    If the object is a list or a tuple, recursively check the first element of the sequence and stop if at any point the first element is not a sequence or is an empty sequence.
    """
    if obj is None:
        return False
    elif isinstance(obj, (list, tuple)) and (schema is None or isinstance(schema, (list, tuple, LargeList, List))):
        if len(obj) > 0:
            if schema is None:
                pass
            elif isinstance(schema, (list, tuple)):
                schema = schema[0]
            else:
                schema = schema.feature
            return _check_non_null_non_empty_recursive(obj[0], schema)
        else:
            return False
    else:
        return True


def get_nested_type(schema: FeatureType) -> pa.DataType:
    """
    get_nested_type() converts a datasets.FeatureType into a pyarrow.DataType, and acts as the inverse of
        generate_from_arrow_type().

    It performs double-duty as the implementation of Features.type and handles the conversion of
        datasets.Feature->pa.struct
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, Features):
        return pa.struct(
            {key: get_nested_type(schema[key]) for key in schema}
        )  # Features is subclass of dict, and dict order is deterministic since Python 3.6
    elif isinstance(schema, dict):
        return pa.struct(
            {key: get_nested_type(schema[key]) for key in schema}
        )  # however don't sort on struct types since the order matters
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError("When defining list feature, you should just provide one example of the inner type")
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, LargeList):
        value_type = get_nested_type(schema.feature)
        return pa.large_list(value_type)
    elif isinstance(schema, List):
        value_type = get_nested_type(schema.feature)
        return pa.list_(value_type, schema.length)

    # Other objects are callable which returns their data type (ClassLabel, Array2D, Translation, Arrow datatype creation methods)
    return schema()


def encode_nested_example(schema, obj, level=0):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return (
            {k: encode_nested_example(schema[k], obj.get(k), level=level + 1) for k in schema}
            if obj is not None
            else None
        )
    elif isinstance(schema, (LargeList, List)):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.feature
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                try:
                    changed = bool(encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt)
                except ValueError:  # can happen when comparing arrays
                    changed = False
                if changed:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    # Object with special encoding:
    # ClassLabel will convert from string to int, TranslationVariableLanguages does some checks
    elif hasattr(schema, "encode_example"):
        return schema.encode_example(obj) if obj is not None else None
    # Other object should be directly convertible to a native Arrow type (like Translation and Translation)
    return obj


def decode_nested_example(schema, obj, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be decoded.
    If the first element needs to be decoded, then all the elements of the list will be decoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return (
            {k: decode_nested_example(sub_schema, sub_obj) for k, (sub_schema, sub_obj) in zip_dict(schema, obj)}
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, (LargeList, List)):
        if obj is None:
            return None
        else:
            sub_schema = schema.feature
            if isinstance(sub_schema, dict):
                return [decode_nested_example(sub_schema, o) for o in obj]
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    # Object with special decoding:
    elif hasattr(schema, "decode_example") and getattr(schema, "decode", True):
        # we pass the token to read and decode files from private repositories in streaming mode
        return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
    return obj


_FEATURE_TYPES: dict[str, FeatureType] = {
    Value.__name__: Value,
    ClassLabel.__name__: ClassLabel,
    Translation.__name__: Translation,
    TranslationVariableLanguages.__name__: TranslationVariableLanguages,
    LargeList.__name__: LargeList,
    List.__name__: List,
    Array2D.__name__: Array2D,
    Array3D.__name__: Array3D,
    Array4D.__name__: Array4D,
    Array5D.__name__: Array5D,
    Audio.__name__: Audio,
    Image.__name__: Image,
    Mesh.__name__: Mesh,
    Video.__name__: Video,
    Pdf.__name__: Pdf,
    Nifti.__name__: Nifti,
    BioSequence.__name__: BioSequence,
    BioStructure.__name__: BioStructure,
    Json.__name__: Json,
}


@experimental
def register_feature(
    feature_cls: type,
    feature_type: str,
):
    """
    Register a Feature object using a name and class.
    This function must be used on a Feature class.
    """
    if feature_type in _FEATURE_TYPES:
        logger.warning(
            f"Overwriting feature type '{feature_type}' ({_FEATURE_TYPES[feature_type].__name__} -> {feature_cls.__name__})"
        )
    _FEATURE_TYPES[feature_type] = feature_cls


def generate_from_dict(obj: Any):
    """Regenerate the nested feature object from a deserialized dict.
    We use the '_type' fields to get the dataclass name to load.

    generate_from_dict is the recursive helper for Features.from_dict, and allows for a convenient constructor syntax
    to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
    a :class:`DatasetInfo` that was dumped to a JSON object. This acts as an analogue to
    :meth:`Features.from_arrow_schema` and handles the recursive field-by-field instantiation, but doesn't require any
    mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive dtypes
    that :class:`Value` automatically performs.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(obj, list):
        return [generate_from_dict(value) for value in obj]
    # Otherwise we have a dict or a dataclass
    if "_type" not in obj or isinstance(obj["_type"], dict):
        return {key: generate_from_dict(value) for key, value in obj.items()}
    obj = dict(obj)
    _type = obj.pop("_type")
    class_type = _FEATURE_TYPES.get(_type, None) or globals().get(_type, None)

    if class_type is None:
        raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")

    if class_type == LargeList:
        feature = obj.pop("feature")
        return LargeList(generate_from_dict(feature), **obj)
    if class_type == List:
        feature = obj.pop("feature")
        return List(generate_from_dict(feature), **obj)
    if class_type == Sequence:  # backward compatibility, this translates to a List or a dict
        feature = obj.pop("feature")
        return Sequence(feature=generate_from_dict(feature), **obj)

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})


def generate_from_arrow_type(pa_type: pa.DataType) -> FeatureType:
    """
    generate_from_arrow_type accepts an arrow DataType and returns a datasets FeatureType to be used as the type for
        a single field.

    This is the high-level arrow->datasets type conversion and is inverted by get_nested_type().

    This operates at the individual *field* level, whereas Features.from_arrow_schema() operates at the
        full schema level and holds the methods that represent the bijection from Features<->pyarrow.Schema
    """
    if isinstance(pa_type, pa.StructType):
        return {field.name: generate_from_arrow_type(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return List(generate_from_arrow_type(pa_type.value_type), length=pa_type.list_size)
    elif isinstance(pa_type, pa.ListType):
        return List(generate_from_arrow_type(pa_type.value_type))
    elif isinstance(pa_type, pa.LargeListType):
        return LargeList(generate_from_arrow_type(pa_type.value_type))
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.JsonType):
        return Json()
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=_arrow_to_datasets_dtype(pa_type))
    else:
        raise ValueError(f"Cannot convert {pa_type} to a Feature type.")


def numpy_to_pyarrow_listarray(arr: np.ndarray, type: pa.DataType = None) -> pa.ListArray:
    """Build a PyArrow ListArray from a multidimensional NumPy array"""
    arr = np.array(arr)
    values = pa.array(arr.flatten(), type=type)
    for i in range(arr.ndim - 1):
        n_offsets = reduce(mul, arr.shape[: arr.ndim - i - 1], 1)
        step_offsets = arr.shape[arr.ndim - i - 1]
        offsets = pa.array(np.arange(n_offsets + 1) * step_offsets, type=pa.int32())
        values = pa.ListArray.from_arrays(offsets, values)
    return values


def list_of_pa_arrays_to_pyarrow_listarray(l_arr: list[Optional[pa.Array]]) -> pa.ListArray:
    null_mask = np.array([arr is None for arr in l_arr])
    null_indices = np.arange(len(null_mask))[null_mask] - np.arange(np.sum(null_mask))
    l_arr = [arr for arr in l_arr if arr is not None]
    offsets = np.cumsum(
        [0] + [len(arr) for arr in l_arr], dtype=object
    )  # convert to dtype object to allow None insertion
    offsets = np.insert(offsets, null_indices, None)
    offsets = pa.array(offsets, type=pa.int32())
    values = pa.concat_arrays(l_arr)
    return pa.ListArray.from_arrays(offsets, values)


def list_of_np_array_to_pyarrow_listarray(l_arr: list[np.ndarray], type: pa.DataType = None) -> pa.ListArray:
    """Build a PyArrow ListArray from a possibly nested list of NumPy arrays"""
    if len(l_arr) > 0:
        return list_of_pa_arrays_to_pyarrow_listarray(
            [numpy_to_pyarrow_listarray(arr, type=type) if arr is not None else None for arr in l_arr]
        )
    else:
        return pa.array([], type=type)


def contains_any_np_array(data: Any):
    """Return `True` if data is a NumPy ndarray or (recursively) if first non-null value in list is a NumPy ndarray.

    Args:
        data (Any): Data.

    Returns:
        bool
    """
    if isinstance(data, np.ndarray):
        return True
    elif isinstance(data, list):
        return contains_any_np_array(first_non_null_value(data)[1])
    else:
        return False


def any_np_array_to_pyarrow_listarray(data: Union[np.ndarray, list], type: pa.DataType = None) -> pa.ListArray:
    """Convert to PyArrow ListArray either a NumPy ndarray or (recursively) a list that may contain any NumPy ndarray.

    Args:
        data (Union[np.ndarray, List]): Data.
        type (pa.DataType): Explicit PyArrow DataType passed to coerce the ListArray data type.

    Returns:
        pa.ListArray
    """
    if isinstance(data, np.ndarray):
        return numpy_to_pyarrow_listarray(data, type=type)
    elif isinstance(data, list):
        return list_of_pa_arrays_to_pyarrow_listarray([any_np_array_to_pyarrow_listarray(i, type=type) for i in data])


def to_pyarrow_listarray(data: Any, pa_type: _ArrayXDExtensionType) -> pa.Array:
    """Convert to PyArrow ListArray.

    Args:
        data (Any): List, iterable, np.ndarray or pd.Series.
        pa_type (_ArrayXDExtensionType): Any of the ArrayNDExtensionType.

    Returns:
        pyarrow.Array
    """
    if contains_any_np_array(data):
        return any_np_array_to_pyarrow_listarray(data, type=pa_type.value_type)
    else:
        return pa.array(data, pa_type.storage_dtype)


def _visit(feature: FeatureType, func: Callable[[FeatureType], Optional[FeatureType]]) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        visited feature (FeatureType)
    """
    if isinstance(feature, Features):
        out = func(Features({k: _visit(f, func) for k, f in feature.items()}))
    elif isinstance(feature, dict):
        out = func({k: _visit(f, func) for k, f in feature.items()})
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit(feature.feature, func)))
    elif isinstance(feature, List):
        out = func(List(_visit(feature.feature, func), length=feature.length))
    else:
        out = func(feature)
    return feature if out is None else out


_VisitPath = list[Union[str, Literal[0]]]


def _visit_with_path(
    feature: FeatureType, func: Callable[[FeatureType, _VisitPath], Optional[FeatureType]], visit_path: _VisitPath = []
) -> FeatureType:
    """Visit a (possibly nested) feature with its path in the Feature object.

    A path in a nested feature object is the list of keys that need to be
    sequentially accessed to get to the sub-feature.

    For example:
    - ["foo"] corresponds to the column "foo"
    - ["foo", 0] corresponds to the sub-feature of the lists in "foo"
    - ["foo", "bar"] corresponds to the sub-feature of the dicts in "foo" with key "bar"

    Args:
        feature (`FeatureType`): the feature type to be checked.

    Returns:
        `FeatureType`: the visited feature.
    """
    if isinstance(feature, Features):
        out = func(Features({k: _visit_with_path(f, func, visit_path + [k]) for k, f in feature.items()}), visit_path)
    elif isinstance(feature, dict):
        out = func({k: _visit_with_path(f, func, visit_path + [k]) for k, f in feature.items()}, visit_path)
    elif isinstance(feature, List):
        out = func(List(_visit_with_path(feature.feature, func, visit_path + [0]), length=feature.length), visit_path)
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit_with_path(feature.feature, func, visit_path + [0])), visit_path)
    else:
        out = func(feature, visit_path)
    return feature if out is None else out


def require_decoding(feature: FeatureType, ignore_decode_attribute: bool = False) -> bool:
    """Check if a (possibly nested) feature requires decoding.

    Args:
        feature (FeatureType): the feature type to be checked
        ignore_decode_attribute (:obj:`bool`, default ``False``): Whether to ignore the current value
            of the `decode` attribute of the decodable feature types.
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_decoding(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_decoding(feature[0])
    elif isinstance(feature, LargeList):
        return require_decoding(feature.feature)
    elif isinstance(feature, List):
        return require_decoding(feature.feature)
    else:
        return hasattr(feature, "decode_example") and (
            getattr(feature, "decode", True) if not ignore_decode_attribute else True
        )


def require_storage_cast(feature: FeatureType) -> bool:
    """Check if a (possibly nested) feature requires storage casting.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_storage_cast(f) for f in feature.values())
    elif isinstance(feature, LargeList):
        return require_storage_cast(feature.feature)
    elif isinstance(feature, List):
        return require_storage_cast(feature.feature)
    else:
        return hasattr(feature, "cast_storage")


def require_storage_embed(feature: FeatureType) -> bool:
    """Check if a (possibly nested) feature requires embedding data into storage.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_storage_embed(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_storage_embed(feature[0])
    elif isinstance(feature, LargeList):
        return require_storage_embed(feature.feature)
    elif isinstance(feature, List):
        return require_storage_embed(feature.feature)
    else:
        return hasattr(feature, "embed_storage")


def keep_features_dicts_synced(func):
    """
    Wrapper to keep the secondary dictionary, which tracks whether keys are decodable, of the :class:`datasets.Features` object
    in sync with the main dictionary.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Features" = args[0]
            args = args[1:]
        else:
            self: "Features" = kwargs.pop("self")
        out = func(self, *args, **kwargs)
        assert hasattr(self, "_column_requires_decoding")
        self._column_requires_decoding = {col: require_decoding(feature) for col, feature in self.items()}
        return out

    wrapper._decorator_name_ = "_keep_dicts_synced"
    return wrapper


class Features(dict):
    """A special dictionary that defines the internal structure of a dataset.

    Instantiated with a dictionary of type `dict[str, FieldType]`, where keys are the desired column names,
    and values are the type of that column.

    `FieldType` can be one of the following:
        - [`Value`] feature specifies a single data type value, e.g. `int64` or `string`.
        - [`ClassLabel`] feature specifies a predefined set of classes which can have labels associated to them and
          will be stored as integers in the dataset.
        - Python `dict` specifies a composite feature containing a mapping of sub-fields to sub-features.
          It's possible to have nested fields of nested fields in an arbitrary manner.
        - [`List`] or [`LargeList`] specifies a composite feature containing a sequence of
          sub-features, all of the same feature type.
        - [`Array2D`], [`Array3D`], [`Array4D`] or [`Array5D`] feature for multidimensional arrays.
        - [`Audio`] feature to store the absolute path to an audio file or a dictionary with the relative path
          to an audio file ("path" key) and its bytes content ("bytes" key).
          This feature loads the audio lazily with a decoder.
        - [`Image`] feature to store the absolute path to an image file, an `np.ndarray` object, a `PIL.Image.Image` object
          or a dictionary with the relative path to an image file ("path" key) and its bytes content ("bytes" key).
          This feature extracts the image data.
        - [`Video`] feature to store the absolute path to a video file, a `torchcodec.decoders.VideoDecoder` object
          or a dictionary with the relative path to a video file ("path" key) and its bytes content ("bytes" key).
          This feature loads the video lazily with a decoder.
        - [`Pdf`] feature to store the absolute path to a PDF file, a `pdfplumber.pdf.PDF` object
          or a dictionary with the relative path to a PDF file ("path" key) and its bytes content ("bytes" key).
          This feature loads the PDF lazily with a PDF reader.
        - [`Nifti`] feature to store the absolute path to a NIfTI neuroimaging file, a `nibabel.Nifti1Image` object
          or a dictionary with the relative path to a NIfTI file ("path" key) and its bytes content ("bytes" key).
          This feature loads the NIfTI file lazily with nibabel.
        - [`Translation`] or [`TranslationVariableLanguages`] feature specific to Machine Translation.
        - [`Json`] feature to store unstructred data, e.g. containing mixed/abritrary types. Under the hood
    """

    def __init__(*args, **kwargs):
        # self not in the signature to allow passing self as a kwarg
        if not args:
            raise TypeError("descriptor '__init__' of 'Features' object needs an argument")
        self, *args = args
        super(Features, self).__init__(*args, **kwargs)
        # keep track of columns which require decoding
        self._column_requires_decoding: dict[str, bool] = {
            col: require_decoding(feature) for col, feature in self.items()
        }

        # backward compatibility with datasets<4 : [feature] -> List(feature)
        def _check_old_list(feature):
            if isinstance(feature, list):
                return List(_visit(feature[0], _check_old_list))
            return feature

        for column_name, feature in self.items():
            self[column_name] = _visit(feature, _check_old_list)

    __setitem__ = keep_features_dicts_synced(dict.__setitem__)
    __delitem__ = keep_features_dicts_synced(dict.__delitem__)
    update = keep_features_dicts_synced(dict.update)
    setdefault = keep_features_dicts_synced(dict.setdefault)
    pop = keep_features_dicts_synced(dict.pop)
    popitem = keep_features_dicts_synced(dict.popitem)
    clear = keep_features_dicts_synced(dict.clear)

    def __reduce__(self):
        return Features, (dict(self),)

    @property
    def type(self):
        """
        Features field types.

        Returns:
            :obj:`pyarrow.DataType`
        """
        return get_nested_type(self)

    @property
    def arrow_schema(self):
        """
        Features schema.

        Returns:
            :obj:`pyarrow.Schema`
        """
        hf_metadata = {"info": {"features": self.to_dict()}}
        return pa.schema(self.type).with_metadata({"huggingface": json.dumps(hf_metadata)})

    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema) -> "Features":
        """
        Construct [`Features`] from Arrow Schema.
        It also checks the schema metadata for Hugging Face Datasets features.
        Non-nullable fields are not supported and set to nullable.

        Also, pa.dictionary is not supported and it uses its underlying type instead.
        Therefore datasets convert DictionaryArray objects to their actual values.

        Args:
            pa_schema (`pyarrow.Schema`):
                Arrow Schema.

        Returns:
            [`Features`]
        """
        # try to load features from the arrow schema metadata
        metadata_features = Features()
        if pa_schema.metadata is not None and b"huggingface" in pa_schema.metadata:
            metadata = json.loads(pa_schema.metadata[b"huggingface"].decode())
            if "info" in metadata and "features" in metadata["info"] and metadata["info"]["features"] is not None:
                metadata_features = Features.from_dict(metadata["info"]["features"])
        metadata_features_schema = metadata_features.arrow_schema
        obj = {
            field.name: (
                metadata_features[field.name]
                if field.name in metadata_features and metadata_features_schema.field(field.name) == field
                else generate_from_arrow_type(field.type)
            )
            for field in pa_schema
        }
        return cls(**obj)

    @classmethod
    def from_dict(cls, dic) -> "Features":
        """
        Construct [`Features`] from dict.

        Regenerate the nested feature object from a deserialized dict.
        We use the `_type` key to infer the dataclass name of the feature `FieldType`.

        It allows for a convenient constructor syntax
        to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
        a [`DatasetInfo`] that was dumped to a JSON object. This acts as an analogue to
        [`Features.from_arrow_schema`] and handles the recursive field-by-field instantiation, but doesn't require
        any mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive
        dtypes that [`Value`] automatically performs.

        Args:
            dic (`dict[str, Any]`):
                Python dictionary.

        Returns:
            `Features`

        Example::
            >>> Features.from_dict({'_type': {'dtype': 'string', 'id': None, '_type': 'Value'}})
            {'_type': Value('string')}
        """
        obj = generate_from_dict(dic)
        return cls(**obj)

    def to_dict(self):
        return asdict(self)

    def _to_yaml_list(self) -> list:
        # we compute the YAML list from the dict representation that is used for JSON dump
        yaml_data = self.to_dict()

        def simplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type:                ->              list_type: int32
                #   dtype: int32            ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["dtype"]:
                    feature[list_type] = feature[list_type]["dtype"]

                #
                # list_type:                ->              list_type:
                #   struct:                 ->              - name: foo
                #   - name: foo             ->                dtype: int32
                #     dtype: int32          ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["struct"]:
                    feature[list_type] = feature[list_type]["struct"]

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #   - negative              ->                  '0': negative
            #   - positive              ->                  '1': positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), list):
                # server-side requirement: keys must be strings
                feature["class_label"]["names"] = {
                    str(label_id): label_name for label_id, label_name in enumerate(feature["class_label"]["names"])
                }
            return feature

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop("_type", None)
                if _type == "LargeList":
                    _feature = obj.pop("feature")
                    return simplify({"large_list": to_yaml_inner(_feature), **obj})
                elif _type == "List":
                    _feature = obj.pop("feature")
                    return simplify({"list": to_yaml_inner(_feature), **obj})
                elif _type == "Value":
                    return obj
                elif _type and not obj:
                    return {"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    return {"dtype": simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {"struct": [{"name": name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
            elif isinstance(obj, list):
                return simplify({"list": simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        def to_yaml_types(obj: dict) -> dict:
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj

        return to_yaml_types(to_yaml_inner(yaml_data)["struct"])

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "Features":
        yaml_data = copy.deepcopy(yaml_data)

        # we convert the list obtained from YAML data into the dict representation that is used for JSON dump

        def unsimplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type: int32          ->              list_type:
                #                           ->                dtype: int32
                #
                if isinstance(feature.get(list_type), str):
                    feature[list_type] = {"dtype": feature[list_type]}

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #     '0': negative              ->               - negative
            #     '1': positive              ->               - positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), dict):
                label_ids = sorted(feature["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
                    raise ValueError(
                        f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                    )
                feature["class_label"]["names"] = [feature["class_label"]["names"][label_id] for label_id in label_ids]
            return feature

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == "large_list":
                    _feature = from_yaml_inner(unsimplify(obj).pop(_type))
                    return {"feature": _feature, **obj, "_type": "LargeList"}
                if _type == "sequence":  # backward compatibility
                    if isinstance(obj[_type], list):
                        _feature = from_yaml_inner(unsimplify(obj).pop(_type))
                        return {
                            name: {"feature": _subfeature, **obj, "_type": "List"}
                            for name, _subfeature in _feature.items()
                        }
                    else:
                        _feature = from_yaml_inner(unsimplify(obj).pop(_type))
                        return {"feature": _feature, **obj, "_type": "List"}
                if _type == "list":
                    _feature = from_yaml_inner(unsimplify(obj).pop(_type))
                    return {"feature": _feature, **obj, "_type": "List"}
                if _type == "struct":
                    return from_yaml_inner(obj["struct"])
                elif _type == "dtype":
                    if isinstance(obj["dtype"], str):
                        # e.g. int32, float64, string, audio, image
                        try:
                            Value(obj["dtype"])
                            return {**obj, "_type": "Value"}
                        except ValueError:
                            # e.g. Audio, Image, ArrayXD
                            return {"_type": snakecase_to_camelcase(obj["dtype"])}
                    else:
                        return from_yaml_inner(obj["dtype"])
                else:
                    return {"_type": snakecase_to_camelcase(_type), **unsimplify(obj)[_type]}
            elif isinstance(obj, list):
                names = [_feature.pop("name") for _feature in obj]
                return {name: from_yaml_inner(_feature) for name, _feature in zip(names, obj)}
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        return cls.from_dict(from_yaml_inner(yaml_data))

    def encode_example(self, example):
        """
        Encode example into a format for Arrow.

        Args:
            example (`dict[str, Any]`):
                Data in a Dataset row.

        Returns:
            `dict[str, Any]`
        """
        example = cast_to_python_objects(example)
        return encode_nested_example(self, example)

    def encode_column(self, column, column_name: str):
        """
        Encode column into a format for Arrow.

        Args:
            column (`list[Any]`):
                Data in a Dataset column.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        column = cast_to_python_objects(column)
        return [encode_nested_example(self[column_name], obj, level=1) for obj in column]

    def encode_batch(self, batch):
        """
        Encode batch into a format for Arrow.

        Args:
            batch (`dict[str, list[Any]]`):
                Data in a Dataset batch.

        Returns:
            `dict[str, list[Any]]`
        """
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(f"Column mismatch between batch {set(batch)} and features {set(self)}")
        for key, column in batch.items():
            column = cast_to_python_objects(column)
            encoded_batch[key] = [encode_nested_example(self[key], obj, level=1) for obj in column]
        return encoded_batch

    def decode_example(self, example: dict, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        """Decode example with custom feature decoding.

        Args:
            example (`dict[str, Any]`):
                Dataset row data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary `repo_id (str) -> token (bool or str)`.

        Returns:
            `dict[str, Any]`
        """

        return {
            column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
            if self._column_requires_decoding[column_name]
            else value
            for column_name, (feature, value) in zip_dict(
                {key: value for key, value in self.items() if key in example}, example
            )
        }

    def decode_column(
        self, column: list, column_name: str, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
    ):
        """Decode column with custom feature decoding.

        Args:
            column (`list[Any]`):
                Dataset column data.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return (
            [
                decode_nested_example(self[column_name], value, token_per_repo_id=token_per_repo_id)
                if value is not None
                else None
                for value in column
            ]
            if self._column_requires_decoding[column_name]
            else column
        )

    def decode_batch(self, batch: dict, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None):
        """Decode batch with custom feature decoding.

        Args:
            batch (`dict[str, list[Any]]`):
                Dataset batch data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary repo_id (str) -> token (bool or str)

        Returns:
            `dict[str, list[Any]]`
        """
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = (
                [
                    decode_nested_example(self[column_name], value, token_per_repo_id=token_per_repo_id)
                    if value is not None
                    else None
                    for value in column
                ]
                if self._column_requires_decoding[column_name]
                else column
            )
        return decoded_batch

    def copy(self) -> "Features":
        """
        Make a deep copy of [`Features`].

        Returns:
            [`Features`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
        >>> copy_of_features = ds.features.copy()
        >>> copy_of_features
        {'label': ClassLabel(names=['neg', 'pos']),
         'text': Value('string')}
        ```
        """
        return copy.deepcopy(self)

    def reorder_fields_as(self, other: "Features") -> "Features":
        """
        Reorder Features fields to match the field order of other [`Features`].

        The order of the fields is important since it matters for the underlying arrow data.
        Re-ordering the fields allows to make the underlying arrow data type match.

        Args:
            other ([`Features`]):
                The other [`Features`] to align with.

        Returns:
            [`Features`]

        Example::

            >>> from datasets import Features, List, Value
            >>> # let's say we have two features with a different order of nested fields (for a and b for example)
            >>> f1 = Features({"root": {"a": Value("string"), "b": Value("string")}})
            >>> f2 = Features({"root": {"b": Value("string"), "a": Value("string")}})
            >>> assert f1.type != f2.type
            >>> # re-ordering keeps the base structure (here List is defined at the root level), but makes the fields order match
            >>> f1.reorder_fields_as(f2)
            {'root': List({'b': Value('string'), 'a': Value('string')})}
            >>> assert f1.reorder_fields_as(f2).type == f2.type
        """

        def recursive_reorder(source, target, stack=""):
            stack_position = " at " + stack[1:] if stack else ""
            if isinstance(source, dict):
                if not isinstance(target, dict):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                if sorted(source) != sorted(target):
                    message = (
                        f"Keys mismatch: between {source} (source) and {target} (target).\n"
                        f"{source.keys() - target.keys()} are missing from target "
                        f"and {target.keys() - source.keys()} are missing from source" + stack_position
                    )
                    raise ValueError(message)
                return {key: recursive_reorder(source[key], target[key], stack + f".{key}") for key in target}
            elif isinstance(source, List):
                if not isinstance(target, List):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                return List(recursive_reorder(source.feature, target.feature, stack + ".<list>"), length=source.length)
            elif isinstance(source, LargeList):
                if not isinstance(target, LargeList):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                return LargeList(recursive_reorder(source.feature, target.feature, stack + ".<list>"))
            else:
                return source

        return Features(recursive_reorder(self, other))

    def flatten(self, max_depth=16) -> "Features":
        """Flatten the features. Every dictionary column is removed and is replaced by
        all the subfields it contains. The new fields are named by concatenating the
        name of the original column and the subfield name like this: `<original>.<subfield>`.

        If a column contains nested dictionaries, then all the lower-level subfields names are
        also concatenated to form new columns: `<original>.<subfield>.<subsubfield>`, etc.

        Returns:
            [`Features`]:
                The flattened features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rajpurkar/squad", split="train")
        >>> ds.features.flatten()
        {'answers.answer_start': List(Value('int32'), id=None),
         'answers.text': List(Value('string'), id=None),
         'context': Value('string'),
         'id': Value('string'),
         'question': Value('string'),
         'title': Value('string')}
        ```
        """
        for depth in range(1, max_depth):
            no_change = True
            flattened = self.copy()
            for column_name, subfeature in self.items():
                if isinstance(subfeature, dict):
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.items()})
                    del flattened[column_name]
                elif hasattr(subfeature, "flatten") and subfeature.flatten() != subfeature:
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.flatten().items()})
                    del flattened[column_name]
            self = flattened
            if no_change:
                break
        return self


def _is_null_feature(feature) -> bool:
    """Recursively check if a feature represents a null type.

    This handles not only top-level ``Value("null")`` but also nested null types
    such as ``List(Value("null"))``, ``LargeList(Value("null"))``, and
    ``Sequence(Value("null"))``, which can arise when a shard contains only
    empty lists during multi-process ``Dataset.map()``.
    """
    if isinstance(feature, Value) and feature.dtype == "null":
        return True
    if isinstance(feature, (Sequence, LargeList)) and hasattr(feature, "feature"):
        return _is_null_feature(feature.feature)
    return False


def _align_features(features_list: list[Features]) -> list[Features]:
    """Align dictionaries of features so that the keys that are found in multiple dictionaries share the same feature."""
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k in name2feature and isinstance(v, dict):
                # Recursively align features.
                name2feature[k] = _align_features([name2feature[k], v])[0]
            elif k not in name2feature or _is_null_feature(name2feature[k]):
                name2feature[k] = v

    return [Features({k: name2feature[k] for k in features.keys()}) for features in features_list]


def _check_if_features_can_be_aligned(features_list: list[Features]):
    """Check if the dictionaries of features can be aligned.

    Two dictionaries of features can be aligned if the keys they share have the same type or some of them is of
    type ``Value("null")`` (or a container wrapping ``Value("null")``, such as ``List(Value("null"))``).
    """
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k not in name2feature or _is_null_feature(name2feature[k]):
                name2feature[k] = v

    for features in features_list:
        for k, v in features.items():
            if isinstance(v, dict) and isinstance(name2feature[k], dict):
                # Deep checks for structure.
                _check_if_features_can_be_aligned([name2feature[k], v])
            elif not _is_null_feature(v) and name2feature[k] != v:
                raise ValueError(
                    f'The features can\'t be aligned because the key {k} of features {features} has unexpected type - {v} (expected either {name2feature[k]} or Value("null").'
                )


def _fix_for_backward_compatible_features(feature: Any) -> FeatureType:
    def _fix_old_list(feature):
        if isinstance(feature, list):
            return List(_fix_for_backward_compatible_features(feature[0]))
        return feature

    return _visit(feature, _fix_old_list)
<|endoftext|>import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    from Bio.SeqRecord import SeqRecord

    from .features import FeatureType


def encode_bio_seqrecord(record: "SeqRecord", format: str = "fasta") -> dict:
    """Serialize a ``SeqRecord`` back to the bytes of a single-record file."""
    from Bio import SeqIO

    buffer = StringIO()
    SeqIO.write(record, buffer, format)
    return {"path": None, "bytes": buffer.getvalue().encode("utf-8")}


def _resolve_token(path: str, token_per_repo_id: dict) -> Optional[str]:
    """Return the token for the Hub repo that ``path`` points into, or ``None``.

    ``string_to_dict`` returns ``None`` (it does not raise) when the URL is not a Hub
    dataset URL, so a plain https or s3 path resolves to "no token" instead of failing.
    """
    source_url = path.split("::")[-1]
    pattern = config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
    source_url_fields = string_to_dict(source_url, pattern)
    return token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None


@dataclass
class BioSequence:
    """
    **Experimental.**
    BioSequence [`Feature`] to read biological sequence records from a sequence file.

    A sequence file holds one or more records of residues with their identifiers and
    annotations. FASTA, FASTQ and GenBank are the same shape at this level, differing
    only in the parser used, so `format` selects the parser rather than the type.

    Input: The BioSequence feature accepts as input:
    - A `str`: Absolute path to the sequence file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the sequence file (i.e. random access is allowed).
    - A `dict` with the keys:
        - `path`: String with relative path of the sequence file in a dataset repository.
        - `bytes`: Bytes of the sequence file.
      This is useful for archived files with sequential access.

    - A `Bio.SeqRecord.SeqRecord`: biopython sequence record.

    Args:
        format (`str`, defaults to `"fasta"`):
            Name of the sequence format, as understood by `Bio.SeqIO`. Common values
            are `"fasta"`, `"fastq"` and `"genbank"`.
        decode (`bool`, defaults to `True`):
            Whether to decode the sequence data. If `False`,
            returns the underlying dictionary in the format `{"path": path, "bytes": bytes}`.

    Examples:

    ```py
    >>> from datasets import Dataset, BioSequence
    >>> ds = Dataset.from_dict({"seq": ["path/to/sequences.fasta"]}).cast_column("seq", BioSequence())
    >>> ds.features["seq"]
    BioSequence(format='fasta', decode=True)
    >>> ds[0]["seq"]
    SeqRecord(seq=Seq('ACGT'), id='seq1', name='seq1', description='seq1', dbxrefs=[])
    >>> ds = ds.cast_column("seq", BioSequence(decode=False))
    >>> ds[0]["seq"]
    {'bytes': None,
    'path': 'path/to/sequences.fasta'}
    ```
    """

    format: str = "fasta"
    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)

    # Automatically constructed
    dtype: ClassVar[str] = "Bio.SeqRecord.SeqRecord"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="BioSequence", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "SeqRecord"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`, `Bio.SeqRecord.SeqRecord` or `dict`):
                Data passed as input to BioSequence feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.BIOPYTHON_AVAILABLE:
            from Bio.SeqRecord import SeqRecord
        else:
            SeqRecord = None

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": bytes(value)}
        elif SeqRecord is not None and isinstance(value, SeqRecord):
            return encode_bio_seqrecord(value, self.format)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"A sequence sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "SeqRecord":
        """Decode example sequence file into a biopython record.

        A file may hold many records; this returns the first, because one row is one
        record. Use the packaged sequence loaders to expand a multi-record file into
        one row per record.

        Args:
            value (`str` or `dict`):
                A string with the absolute file path, or a dictionary with
                keys:

                - `path`: String with absolute or relative file path.
                - `bytes`: The bytes of the file.

            token_per_repo_id (`dict`, *optional*):
                To access and decode files from private repositories on
                the Hub, you can pass a dictionary
                repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `Bio.SeqRecord.SeqRecord`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use BioSequence(decode=True) instead.")

        if not config.BIOPYTHON_AVAILABLE:
            raise ImportError("To support decoding biological sequences, please install 'biopython'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"A sequence should have one of 'path' or 'bytes' but both are None in {value}.")
            if is_local_path(path):
                with open(path, encoding="utf-8") as f:
                    return self._first_record(f, path)
            download_config = DownloadConfig(token=_resolve_token(path, token_per_repo_id))
            with xopen(path, "r", encoding="utf-8", download_config=download_config) as f:
                return self._first_record(f, path)
        else:
            with StringIO(bytes_.decode("utf-8"), newline=None) as f:
                return self._first_record(f, path)

    def _first_record(self, handle, path: Optional[str]) -> "SeqRecord":
        """Return the first record in ``handle``, or raise if the file holds none."""
        from Bio import SeqIO

        for record in SeqIO.parse(handle, self.format):
            return record
        raise ValueError(f"No {self.format} record found in {path if path is not None else 'the given bytes'}.")

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the BioSequence arrow storage type.
        The Arrow types that can be converted to the BioSequence pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the file bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the BioSequence arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        return _cast_to_bytes_path_struct(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed sequence files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`):
                Whether to embed local files data in the array.
            remote_files (`bool`, defaults to `True`):
                Whether to embed remote files data in the array.

        Returns:
            `pa.StructArray`: Array in the BioSequence arrow storage type.
        """
        return _embed_bytes_path_struct(
            storage, self.pa_type, token_per_repo_id, local_files=local_files, remote_files=remote_files
        )


def _cast_to_bytes_path_struct(storage, pa_type: pa.DataType) -> pa.StructArray:
    """Cast a string, binary or struct array to the shared ``struct<bytes, path>`` storage.

    BioSequence and BioStructure accept exactly the same inputs, so the conversion lives
    here rather than being written out twice.
    """
    if pa.types.is_string(storage.type):
        bytes_array = pa.array([None] * len(storage), type=pa.binary())
        storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
    elif pa.types.is_binary(storage.type):
        path_array = pa.array([None] * len(storage), type=pa.string())
        storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
    elif pa.types.is_struct(storage.type):
        if storage.type.get_field_index("bytes") >= 0:
            bytes_array = storage.field("bytes")
        else:
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
        if storage.type.get_field_index("path") >= 0:
            path_array = storage.field("path")
        else:
            path_array = pa.array([None] * len(storage), type=pa.string())
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
    return array_cast(storage, pa_type)


def _embed_bytes_path_struct(
    storage: pa.StructArray,
    pa_type: pa.DataType,
    token_per_repo_id=None,
    local_files: bool = True,
    remote_files: bool = True,
) -> pa.StructArray:
    """Read file contents into the ``bytes`` field, shared by both bio features."""
    if token_per_repo_id is None:
        token_per_repo_id = {}

    @no_op_if_value_is_null
    def path_to_bytes(path):
        download_config = DownloadConfig(token=_resolve_token(path, token_per_repo_id))
        with xopen(path, "rb", download_config=download_config) as f:
            return f.read()

    def should_embed(path: Optional[str]) -> bool:
        if path is None:
            return False
        return local_files if is_local_path(path) else remote_files

    bytes_array = pa.array(
        [
            (path_to_bytes(path) if should_embed(path) else bytes_)
            for bytes_, path in zip(storage.field("bytes").to_pylist(), storage.field("path").to_pylist())
        ],
        type=pa.binary(),
    )
    path_array = pa.array(
        [os.path.basename(path) if path is not None else None for path in storage.field("path").to_pylist()],
        type=pa.string(),
    )
    # Row nullness comes from the input row, not from whether bytes were embedded: a
    # path-only row with embedding disabled is still a valid row.
    storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
    return array_cast(storage, pa_type)
<|endoftext|>import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    import pdfplumber

    from .features import FeatureType


def pdf_to_bytes(pdf: "pdfplumber.pdf.PDF") -> bytes:
    """Convert a pdfplumber.pdf.PDF object to bytes."""
    with BytesIO() as buffer:
        for page in pdf.pages:
            buffer.write(page.pdf.stream)
        return buffer.getvalue()


@dataclass
class Pdf:
    """
    **Experimental.**
    Pdf [`Feature`] to read pdf documents from a pdf file.

    Input: The Pdf feature accepts as input:
    - A `str`: Absolute path to the pdf file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the pdf file (i.e. random access is allowed).
    - A `dict` with the keys:
        - `path`: String with relative path of the pdf file in a dataset repository.
        - `bytes`: Bytes of the pdf file.
      This is useful for archived files with sequential access.

    - A `pdfplumber.pdf.PDF`: pdfplumber pdf object.

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the pdf data. If `False`,
            returns the underlying dictionary in the format `{"path": pdf_path, "bytes": pdf_bytes}`.

    Examples:

    ```py
    >>> from datasets import Dataset, Pdf
    >>> ds = Dataset.from_dict({"pdf": ["path/to/pdf/file.pdf"]}).cast_column("pdf", Pdf())
    >>> ds.features["pdf"]
    Pdf(decode=True, id=None)
    >>> ds[0]["pdf"]
    <pdfplumber.pdf.PDF object at 0x7f8a1c2d8f40>
    >>> ds = ds.cast_column("pdf", Pdf(decode=False))
    >>> ds[0]["pdf"]
    {'bytes': None,
    'path': 'path/to/pdf/file.pdf'}
    ```
    """

    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)

    # Automatically constructed
    dtype: ClassVar[str] = "pdfplumber.pdf.PDF"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Pdf", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "pdfplumber.pdf.PDF"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`, `pdfplumber.pdf.PDF` or `dict`):
                Data passed as input to Pdf feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.PDFPLUMBER_AVAILABLE:
            import pdfplumber
        else:
            pdfplumber = None

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif pdfplumber is not None and isinstance(value, pdfplumber.pdf.PDF):
            # convert the pdfplumber.pdf.PDF to bytes
            return encode_pdfplumber_pdf(value)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the pdf bytes, and path is used to infer the pdf format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"A pdf sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "pdfplumber.pdf.PDF":
        """Decode example pdf file into pdf data.

        Args:
            value (`str` or `dict`):
                A string with the absolute pdf file path, a dictionary with
                keys:

                - `path`: String with absolute or relative pdf file path.
                - `bytes`: The bytes of the pdf file.

            token_per_repo_id (`dict`, *optional*):
                To access and decode pdf files from private repositories on
                the Hub, you can pass a dictionary
                repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `pdfplumber.pdf.PDF`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Pdf(decode=True) instead.")

        if config.PDFPLUMBER_AVAILABLE:
            import pdfplumber
        else:
            raise ImportError("To support decoding pdfs, please install 'pdfplumber'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"A pdf should have one of 'path' or 'bytes' but both are None in {value}.")
            else:
                if is_local_path(path):
                    pdf = pdfplumber.open(path)
                else:
                    source_url = path.split("::")[-1]
                    pattern = (
                        config.HUB_DATASETS_URL
                        if source_url.startswith(config.HF_ENDPOINT)
                        else config.HUB_DATASETS_HFFS_URL
                    )
                    source_url_fields = string_to_dict(source_url, pattern)
                    token = (
                        token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
                    )
                    download_config = DownloadConfig(token=token)
                    f = xopen(path, "rb", download_config=download_config)
                    return pdfplumber.open(f)
        else:
            with pdfplumber.open(BytesIO(bytes_)) as p:
                pdf = p

        return pdf

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        """Cast an Arrow array to the Pdf arrow storage type.
        The Arrow types that can be converted to the Pdf pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the image bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the pdf array data

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Pdf arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed PDF files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`)
                Whether to embed local files data in the array

                <Added version="4.8.5"/>
            remote_files (`bool`, defaults to `True`)
                Whether to embed remote files data in the array.
                E.g. files with paths that start with hf:// or https://

                <Added version="4.8.5"/>

        Returns:
            `pa.StructArray`: Array in the PDF arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if token_per_repo_id is None:
            token_per_repo_id = {}

        @no_op_if_value_is_null
        def path_to_bytes(path):
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (
                    path_to_bytes(x["path"])
                    if x["bytes"] is None
                    and ((local_files and is_local_path(x["path"])) or (remote_files and is_remote_url(x["path"])))
                    else x["bytes"]
                )
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                (
                    os.path.basename(path)
                    if (local_files and is_local_path(path)) or (remote_files and is_remote_url(path))
                    else path
                )
                if path is not None
                else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)


def encode_pdfplumber_pdf(pdf: "pdfplumber.pdf.PDF") -> dict:
    """
    Encode a pdfplumber.pdf.PDF object into a dictionary.

    If the PDF has an associated file path, returns the path. Otherwise, serializes
    the PDF content into bytes.

    Args:
        pdf (pdfplumber.pdf.PDF): A pdfplumber PDF object.

    Returns:
        dict: A dictionary with "path" or "bytes" field.
    """
    if hasattr(pdf, "stream") and hasattr(pdf.stream, "name") and pdf.stream.name:
        # Return the path if the PDF has an associated file path
        return {"path": pdf.stream.name, "bytes": None}
    else:
        # Convert the PDF to bytes if no path is available
        return {"path": None, "bytes": pdf_to_bytes(pdf)}
<|endoftext|>import os
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import numpy as np
import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import first_non_null_value, no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    import PIL.Image

    from .features import FeatureType


_IMAGE_COMPRESSION_FORMATS: Optional[list[str]] = None
_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"
# Origin: https://github.com/python-pillow/Pillow/blob/698951e19e19972aeed56df686868f1329981c12/src/PIL/Image.py#L3126 minus "|i1" which values are not preserved correctly when saving and loading an image
_VALID_IMAGE_ARRAY_DTPYES = [
    np.dtype("|b1"),
    np.dtype("|u1"),
    np.dtype("<u2"),
    np.dtype(">u2"),
    np.dtype("<i2"),
    np.dtype(">i2"),
    np.dtype("<u4"),
    np.dtype(">u4"),
    np.dtype("<i4"),
    np.dtype(">i4"),
    np.dtype("<f4"),
    np.dtype(">f4"),
    np.dtype("<f8"),
    np.dtype(">f8"),
]


@dataclass
class Image:
    """Image [`Feature`] to read image data from an image file.

    Input: The Image feature accepts as input:
    - A `str`: Absolute path to the image file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the image file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the image file to the archive file.
        - `bytes`: Bytes of the image file.

      This is useful for parquet or webdataset files which embed image files.

    - An `np.ndarray`: NumPy array representing an image.
    - A `PIL.Image.Image`: PIL image object.

    Output: The Image features output data as `PIL.Image.Image` objects.

    Args:
        mode (`str`, *optional*):
            The mode to convert the image to. If `None`, the native mode of the image is used.
        decode (`bool`, defaults to `True`):
            Whether to decode the image data. If `False`,
            returns the underlying dictionary in the format `{"path": image_path, "bytes": image_bytes}`.

    Examples:

    ```py
    >>> from datasets import load_dataset, Image
    >>> ds = load_dataset("AI-Lab-Makerere/beans", split="train")
    >>> ds.features["image"]
    Image(decode=True, id=None)
    >>> ds[0]["image"]
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x15E52E7F0>
    >>> ds = ds.cast_column('image', Image(decode=False))
    {'bytes': None,
     'path': '/root/.cache/huggingface/datasets/downloads/extracted/b0a21163f78769a2cf11f58dfc767fb458fc7cea5c05dccc0144a2c0f0bc1292/train/healthy/healthy_train.85.jpg'}
    ```
    """

    mode: Optional[str] = None
    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "PIL.Image.Image"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Image", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, np.ndarray, "PIL.Image.Image"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `np.ndarray`, `PIL.Image.Image` or `dict`):
                Data passed as input to Image feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.PIL_AVAILABLE:
            import PIL.Image
        else:
            raise ImportError("To support encoding images, please install 'Pillow'.")

        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif isinstance(value, np.ndarray):
            # convert the image array to PNG/TIFF bytes
            return encode_np_array(value)
        elif isinstance(value, PIL.Image.Image):
            # convert the PIL image to bytes (default format is PNG/TIFF)
            return encode_pil_image(value)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the image bytes, and path is used to infer the image format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"An image sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "PIL.Image.Image":
        """Decode example image file into image data.

        Args:
            value (`str` or `dict`):
                A string with the absolute image file path, a dictionary with
                keys:

                - `path`: String with absolute or relative image file path.
                - `bytes`: The bytes of the image file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                image files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `PIL.Image.Image`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Image(decode=True) instead.")

        if config.PIL_AVAILABLE:
            import PIL.Image
            import PIL.ImageOps
        else:
            raise ImportError("To support decoding images, please install 'Pillow'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"An image should have one of 'path' or 'bytes' but both are None in {value}.")
            else:
                if is_local_path(path):
                    image = PIL.Image.open(path)
                else:
                    source_url = path.split("::")[-1]
                    pattern = (
                        config.HUB_DATASETS_URL
                        if source_url.startswith(config.HF_ENDPOINT)
                        else config.HUB_DATASETS_HFFS_URL
                    )
                    source_url_fields = string_to_dict(source_url, pattern)
                    token = (
                        token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
                    )
                    download_config = DownloadConfig(token=token)
                    with xopen(path, "rb", download_config=download_config) as f:
                        bytes_ = BytesIO(f.read())
                    image = PIL.Image.open(bytes_)
        else:
            image = PIL.Image.open(BytesIO(bytes_))
        image.load()  # to avoid "Too many open files" errors
        if image.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None:
            image = PIL.ImageOps.exif_transpose(image)
        if self.mode and self.mode != image.mode:
            image = image.convert(self.mode)
        return image

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        """Cast an Arrow array to the Image arrow storage type.
        The Arrow types that can be converted to the Image pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.large_string()` - it must contain the "path" data (will be cast to string if possible)
        - `pa.binary()` - it must contain the image bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the image array data

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_large_string(storage.type):
            try:
                storage = storage.cast(pa.string())
            except pa.ArrowInvalid as e:
                raise ValueError(
                    f"Failed to cast large_string to string for Image feature. "
                    f"This can happen if string values exceed 2GB. "
                    f"Original error: {e}"
                ) from e
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_large_binary(storage.type):
            storage = array_cast(
                storage, pa.binary()
            )  # this can fail in case of big images, paths should be used instead
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [encode_np_array(np.array(arr))["bytes"] if arr is not None else None for arr in storage.to_pylist()],
                type=pa.binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`)
                Whether to embed local files data in the array

                <Added version="4.8.5"/>
            remote_files (`bool`, defaults to `True`)
                Whether to embed remote files data in the array.
                E.g. files with paths that start with hf:// or https://

                <Added version="4.8.5"/>

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if token_per_repo_id is None:
            token_per_repo_id = {}

        @no_op_if_value_is_null
        def path_to_bytes(path):
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (
                    path_to_bytes(x["path"])
                    if x["bytes"] is None
                    and ((local_files and is_local_path(x["path"])) or (remote_files and is_remote_url(x["path"])))
                    else x["bytes"]
                )
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                (
                    os.path.basename(path)
                    if (local_files and is_local_path(path)) or (remote_files and is_remote_url(path))
                    else path
                )
                if path is not None
                else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)


def list_image_compression_formats() -> list[str]:
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support encoding images, please install 'Pillow'.")

    global _IMAGE_COMPRESSION_FORMATS
    if _IMAGE_COMPRESSION_FORMATS is None:
        PIL.Image.init()
        _IMAGE_COMPRESSION_FORMATS = list(set(PIL.Image.OPEN.keys()) & set(PIL.Image.SAVE.keys()))
    return _IMAGE_COMPRESSION_FORMATS


def image_to_bytes(image: "PIL.Image.Image") -> bytes:
    """Convert a PIL Image object to bytes using native compression if possible, otherwise use PNG/TIFF compression."""
    buffer = BytesIO()
    if image.format in list_image_compression_formats():
        format = image.format
    else:
        format = "PNG" if image.mode in ["1", "L", "LA", "RGB", "RGBA"] else "TIFF"
    image.save(buffer, format=format)
    return buffer.getvalue()


def encode_pil_image(image: "PIL.Image.Image") -> dict:
    if hasattr(image, "filename") and image.filename != "":
        return {"path": image.filename, "bytes": None}
    else:
        return {"path": None, "bytes": image_to_bytes(image)}


def encode_np_array(array: np.ndarray) -> dict:
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support encoding images, please install 'Pillow'.")

    dtype = array.dtype
    dtype_byteorder = dtype.byteorder if dtype.byteorder != "=" else _NATIVE_BYTEORDER
    dtype_kind = dtype.kind
    dtype_itemsize = dtype.itemsize

    dest_dtype = None

    # Multi-channel array case (only np.dtype("|u1") is allowed)
    if array.shape[2:]:
        if dtype_kind not in ["u", "i"]:
            raise TypeError(
                f"Unsupported array dtype {dtype} for image encoding. Only {dest_dtype} is supported for multi-channel arrays."
            )
        dest_dtype = np.dtype("|u1")
        if dtype != dest_dtype:
            warnings.warn(f"Downcasting array dtype {dtype} to {dest_dtype} to be compatible with 'Pillow'")
    # Exact match
    elif dtype in _VALID_IMAGE_ARRAY_DTPYES:
        dest_dtype = dtype
    else:  # Downcast the type within the kind (np.can_cast(from_type, to_type, casting="same_kind") doesn't behave as expected, so do it manually)
        while dtype_itemsize >= 1:
            dtype_str = dtype_byteorder + dtype_kind + str(dtype_itemsize)
            if np.dtype(dtype_str) in _VALID_IMAGE_ARRAY_DTPYES:
                dest_dtype = np.dtype(dtype_str)
                warnings.warn(f"Downcasting array dtype {dtype} to {dest_dtype} to be compatible with 'Pillow'")
                break
            else:
                dtype_itemsize //= 2
        if dest_dtype is None:
            raise TypeError(
                f"Cannot downcast dtype {dtype} to a valid image dtype. Valid image dtypes: {_VALID_IMAGE_ARRAY_DTPYES}"
            )

    image = PIL.Image.fromarray(array.astype(dest_dtype))
    return {"path": None, "bytes": image_to_bytes(image)}


def objects_to_list_of_image_dicts(
    objs: Union[list[str], list[dict], list[np.ndarray], list["PIL.Image.Image"]],
) -> list[dict]:
    """Encode a list of objects into a format suitable for creating an extension array of type `ImageExtensionType`."""
    if config.PIL_AVAILABLE:
        import PIL.Image
    else:
        raise ImportError("To support encoding images, please install 'Pillow'.")

    if objs:
        _, obj = first_non_null_value(objs)
        if isinstance(obj, str):
            return [{"path": obj, "bytes": None} if obj is not None else None for obj in objs]
        if isinstance(obj, np.ndarray):
            obj_to_image_dict_func = no_op_if_value_is_null(encode_np_array)
            return [obj_to_image_dict_func(obj) for obj in objs]
        elif isinstance(obj, PIL.Image.Image):
            obj_to_image_dict_func = no_op_if_value_is_null(encode_pil_image)
            return [obj_to_image_dict_func(obj) for obj in objs]
        else:
            return objs
    else:
        return objs
<|endoftext|>import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..utils.file_utils import is_local_path, xopen
from .bio_sequence import _cast_to_bytes_path_struct, _embed_bytes_path_struct, _resolve_token


if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

    from .features import FeatureType


# (parser, writer) class names in Bio.PDB for each supported structure format. Encode and
# decode both go through this table so a format that cannot be read is never written.
_FORMATS: dict[str, tuple[str, str]] = {"pdb": ("PDBParser", "PDBIO"), "mmcif": ("MMCIFParser", "MMCIFIO")}


def _resolve_format(format: str) -> tuple[str, str]:
    try:
        return _FORMATS[format]
    except KeyError:
        raise ValueError(
            f"Unsupported structure format '{format}'. Supported formats are: {sorted(_FORMATS)}."
        ) from None


def encode_bio_structure(structure: "Structure", format: str = "pdb") -> dict:
    """Serialize a ``Structure`` back to the bytes of a structure file."""
    import Bio.PDB

    _, writer_name = _resolve_format(format)
    io = getattr(Bio.PDB, writer_name)()
    io.set_structure(structure)
    buffer = StringIO()
    io.save(buffer)
    return {"path": None, "bytes": buffer.getvalue().encode("utf-8")}


@dataclass
class BioStructure:
    """
    **Experimental.**
    BioStructure [`Feature`] to read macromolecular structures from a structure file.

    A structure file holds the three-dimensional coordinates of a molecule's atoms,
    grouped into chains and residues. PDB and mmCIF describe the same objects and
    differ only in the parser used, so `format` selects the parser rather than the type.

    Input: The BioStructure feature accepts as input:
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:
        - `path`: String with relative path of the structure file in a dataset repository.
        - `bytes`: Bytes of the structure file.
      This is useful for archived files with sequential access.

    - A `Bio.PDB.Structure.Structure`: biopython structure.

    Args:
        format (`str`, defaults to `"pdb"`):
            Structure format, either `"pdb"` or `"mmcif"`.
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": path, "bytes": bytes}`.

    Examples:

    ```py
    >>> from datasets import Dataset, BioStructure
    >>> ds = Dataset.from_dict({"st": ["path/to/1abc.pdb"]}).cast_column("st", BioStructure())
    >>> ds.features["st"]
    BioStructure(format='pdb', decode=True)
    >>> ds[0]["st"]
    <Structure id=1abc>
    >>> ds = ds.cast_column("st", BioStructure(decode=False))
    >>> ds[0]["st"]
    {'bytes': None,
    'path': 'path/to/1abc.pdb'}
    ```
    """

    format: str = "pdb"
    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)

    # Automatically constructed
    dtype: ClassVar[str] = "Bio.PDB.Structure.Structure"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="BioStructure", init=False, repr=False)

    def __post_init__(self):
        _resolve_format(self.format)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "Structure"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`, `Bio.PDB.Structure.Structure` or `dict`):
                Data passed as input to BioStructure feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.BIOPYTHON_AVAILABLE:
            from Bio.PDB.Structure import Structure
        else:
            Structure = None

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": bytes(value)}
        elif Structure is not None and isinstance(value, Structure):
            return encode_bio_structure(value, self.format)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"A structure sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "Structure":
        """Decode example structure file into a biopython structure.

        Args:
            value (`str` or `dict`):
                A string with the absolute file path, or a dictionary with
                keys:

                - `path`: String with absolute or relative file path.
                - `bytes`: The bytes of the file.

            token_per_repo_id (`dict`, *optional*):
                To access and decode files from private repositories on
                the Hub, you can pass a dictionary
                repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `Bio.PDB.Structure.Structure`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use BioStructure(decode=True) instead.")

        if not config.BIOPYTHON_AVAILABLE:
            raise ImportError("To support decoding macromolecular structures, please install 'biopython'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        structure_id = os.path.splitext(os.path.basename(path))[0] if path else "structure"

        if bytes_ is None:
            if path is None:
                raise ValueError(f"A structure should have one of 'path' or 'bytes' but both are None in {value}.")
            if is_local_path(path):
                with open(path, encoding="utf-8") as f:
                    return self._parse(f, structure_id)
            download_config = DownloadConfig(token=_resolve_token(path, token_per_repo_id))
            with xopen(path, "r", encoding="utf-8", download_config=download_config) as f:
                return self._parse(f, structure_id)
        else:
            with StringIO(bytes_.decode("utf-8"), newline=None) as f:
                return self._parse(f, structure_id)

    def _parse(self, handle, structure_id: str) -> "Structure":
        """Parse ``handle`` with the parser for this feature's format."""
        import Bio.PDB

        parser_name, _ = _resolve_format(self.format)
        # QUIET silences the discontinuity warnings that most real PDB entries trigger.
        parser = getattr(Bio.PDB, parser_name)(QUIET=True)
        return parser.get_structure(structure_id, handle)

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the BioStructure arrow storage type.
        The Arrow types that can be converted to the BioStructure pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the file bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the BioStructure arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        return _cast_to_bytes_path_struct(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed structure files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`):
                Whether to embed local files data in the array.
            remote_files (`bool`, defaults to `True`):
                Whether to embed remote files data in the array.

        Returns:
            `pa.StructArray`: Array in the BioStructure arrow storage type.
        """
        return _embed_bytes_path_struct(
            storage, self.pa_type, token_per_repo_id, local_files=local_files, remote_files=remote_files
        )
<|endoftext|>import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, TypedDict, Union

import numpy as np
import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    import torch
    from torchcodec.decoders import VideoDecoder

    from .features import FeatureType


class Example(TypedDict):
    path: Optional[str]
    bytes: Optional[bytes]


@dataclass
class Video:
    """
    Video [`Feature`] to read video data from a video file.

    Input: The Video feature accepts as input:
    - A `str`: Absolute path to the video file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the video file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the video file in a dataset repository.
        - `bytes`: Bytes of the video file.

      This is useful for parquet or webdataset files which embed video files.

    - A `torchcodec.decoders.VideoDecoder`: torchcodec video decoder object.

    Output: The Video features output data as `torchcodec.decoders.VideoDecoder` objects.

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the video data. If `False`,
            returns the underlying dictionary in the format `{"path": video_path, "bytes": video_bytes}`.
        stream_index (`int`, *optional*):
            The streaming index to use from the file. If `None` defaults to the "best" index.
        dimension_order (`str`, defaults to `NCHW`):
            The dimension order of the decoded frames.
            where N is the batch size, C is the number of channels,
            H is the height, and W is the width of the frames.
        num_ffmpeg_threads (`int`, defaults to `1`):
            The number of threads to use for decoding the video. (Recommended to keep this at 1)
        device (`str` or `torch.device`, defaults to `cpu`):
            The device to use for decoding the video.
        seek_mode (`str`, defaults to `exact`):
            Determines if frame access will be “exact” or “approximate”.
            Exact guarantees that requesting frame i will always return frame i, but doing so requires an initial scan of the file.
            Approximate is faster as it avoids scanning the file, but less accurate as it uses the file's metadata to calculate where i probably is.
            read more [here](https://docs.pytorch.org/torchcodec/stable/generated_examples/approximate_mode.html#sphx-glr-generated-examples-approximate-mode-py)

    Examples:

    ```py
    >>> from datasets import Dataset, Video
    >>> ds = Dataset.from_dict({"video":["path/to/Screen Recording.mov"]}).cast_column("video", Video())
    >>> ds.features["video"]
    Video(decode=True, id=None)
    >>> ds[0]["video"]
    <torchcodec.decoders._video_decoder.VideoDecoder object at 0x14a61e080>
    >>> video = ds[0]["video"]
    >>> video.get_frames_in_range(0, 10)
    FrameBatch:
    data (shape): torch.Size([10, 3, 50, 66])
    pts_seconds: tensor([0.4333, 0.4333, 0.4333, 0.4333, 0.4333, 0.4333, 0.4333, 0.4333, 0.4333,
            0.4333], dtype=torch.float64)
    duration_seconds: tensor([0.0167, 0.0167, 0.0167, 0.0167, 0.0167, 0.0167, 0.0167, 0.0167, 0.0167,
            0.0167], dtype=torch.float64)
    >>> ds.cast_column('video', Video(decode=False))[0]["video]
    {'bytes': None,
     'path': 'path/to/Screen Recording.mov'}
    ```
    """

    decode: bool = True
    stream_index: Optional[int] = None
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW"
    num_ffmpeg_threads: int = 1
    device: Optional[Union[str, "torch.device"]] = "cpu"
    seek_mode: Literal["exact", "approximate"] = "exact"
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "torchcodec.decoders.VideoDecoder"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Video", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, Example, np.ndarray, "VideoDecoder"]) -> Example:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `np.ndarray`, `bytes`, `bytearray`, `VideoDecoder` or `dict`):
                Data passed as input to Video feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if value is None:
            raise ValueError("value must be provided")

        if config.TORCHCODEC_AVAILABLE:
            from torchcodec.decoders import VideoDecoder
        else:
            VideoDecoder = None

        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif isinstance(value, np.ndarray):
            # convert the video array to bytes
            return encode_np_array(value)
        elif VideoDecoder is not None and isinstance(value, VideoDecoder):
            # convert the torchcodec video decoder to bytes
            return encode_torchcodec_video(value)
        elif isinstance(value, dict):
            path, bytes_ = value.get("path"), value.get("bytes")
            if path is not None and os.path.isfile(path):
                # we set "bytes": None to not duplicate the data if they're already available locally
                return {"bytes": None, "path": path}
            elif bytes_ is not None or path is not None:
                # store the video bytes, and path is used to infer the video format using the file extension
                return {"bytes": bytes_, "path": path}
            else:
                raise ValueError(
                    f"A video sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
                )
        else:
            raise TypeError(f"Unsupported encode_example type: {type(value)}")

    def decode_example(
        self,
        value: Union[str, Example],
        token_per_repo_id: Optional[dict[str, Union[bool, str]]] = None,
    ) -> "VideoDecoder":
        """Decode example video file into video data.

        Args:
            value (`str` or `dict`):
                A string with the absolute video file path, a dictionary with
                keys:

                - `path`: String with absolute or relative video file path.
                - `bytes`: The bytes of the video file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                video files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `torchcodec.decoders.VideoDecoder`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Video(decode=True) instead.")

        if config.TORCHCODEC_AVAILABLE:
            from torchcodec.decoders import VideoDecoder

        else:
            raise ImportError("To support decoding videos, please install 'torchcodec'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        if isinstance(value, str):
            path, bytes_ = value, None
        else:
            path, bytes_ = value["path"], value["bytes"]

        if bytes_ is None:
            if path is None:
                raise ValueError(f"A video should have one of 'path' or 'bytes' but both are None in {value}.")
            elif is_local_path(path):
                video = VideoDecoder(
                    path,
                    stream_index=self.stream_index,
                    dimension_order=self.dimension_order,
                    num_ffmpeg_threads=self.num_ffmpeg_threads,
                    device=self.device,
                    seek_mode=self.seek_mode,
                )
            else:
                video = hf_video_reader(
                    path,
                    token_per_repo_id=token_per_repo_id,
                    dimension_order=self.dimension_order,
                    num_ffmpeg_threads=self.num_ffmpeg_threads,
                    device=self.device,
                    seek_mode=self.seek_mode,
                )
        else:
            video = VideoDecoder(
                bytes_,
                stream_index=self.stream_index,
                dimension_order=self.dimension_order,
                num_ffmpeg_threads=self.num_ffmpeg_threads,
                device=self.device,
                seek_mode=self.seek_mode,
            )
        video._hf_encoded = {"path": path, "bytes": bytes_}
        video.metadata.path = path
        return video

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        """Cast an Arrow array to the Video arrow storage type.
        The Arrow types that can be converted to the Video pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the video bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the video array data

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Video arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_large_binary(storage.type):
            storage = array_cast(
                storage, pa.binary()
            )  # this can fail in case of big videos, paths should be used instead
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [encode_np_array(np.array(arr))["bytes"] if arr is not None else None for arr in storage.to_pylist()],
                type=pa.binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`)
                Whether to embed local files data in the array

                <Added version="4.8.5"/>
            remote_files (`bool`, defaults to `True`)
                Whether to embed remote files data in the array.
                E.g. files with paths that start with hf:// or https://

                <Added version="4.8.5"/>

        Returns:
            `pa.StructArray`: Array in the Video arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if token_per_repo_id is None:
            token_per_repo_id = {}

        @no_op_if_value_is_null
        def path_to_bytes(path):
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (
                    path_to_bytes(x["path"])
                    if x["bytes"] is None
                    and ((local_files and is_local_path(x["path"])) or (remote_files and is_remote_url(x["path"])))
                    else x["bytes"]
                )
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                (
                    os.path.basename(path)
                    if (local_files and is_local_path(path)) or (remote_files and is_remote_url(path))
                    else path
                )
                if path is not None
                else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)


def video_to_bytes(video: "VideoDecoder") -> bytes:
    """Convert a torchcodec Video object to bytes using native compression if possible"""
    raise NotImplementedError()


def encode_torchcodec_video(video: "VideoDecoder") -> Example:
    if hasattr(video, "_hf_encoded"):
        return video._hf_encoded
    else:
        raise NotImplementedError(
            "Encoding a VideoDecoder that doesn't come from datasets.Video.decode() is not implemented"
        )


def encode_np_array(array: np.ndarray) -> Example:
    raise NotImplementedError()


# No monkey patch needed!
# 1. store the encoded video data {"path": ..., "bytes": ...} in `video._hf_encoded``
# 2. add support for hf:// files


def hf_video_reader(
    path: str,
    token_per_repo_id: Optional[dict[str, Union[bool, str]]] = None,
    stream: str = "video",
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: Optional[Union[str, "torch.device"]] = "cpu",
    seek_mode: Literal["exact", "approximate"] = "exact",
) -> "VideoDecoder":
    from torchcodec.decoders import VideoDecoder

    # Load the file from HF
    if token_per_repo_id is None:
        token_per_repo_id = {}
    source_url = path.split("::")[-1]
    pattern = config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
    source_url_fields = string_to_dict(source_url, pattern)
    token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
    download_config = DownloadConfig(token=token)
    f = xopen(path, "rb", download_config=download_config)

    # Instantiate the VideoDecoder
    stream_id = 0 if len(stream.split(":")) == 1 else int(stream.split(":")[1])
    vd = VideoDecoder(
        f,
        stream_index=stream_id,
        dimension_order=dimension_order,
        num_ffmpeg_threads=num_ffmpeg_threads,
        device=device,
        seek_mode=seek_mode,
    )
    return vd
<|endoftext|>import numpy as np
from torchcodec.decoders import AudioDecoder as _AudioDecoder


class AudioDecoder(_AudioDecoder):
    def __getitem__(self, key: str):
        if key == "array":
            y = self.get_all_samples().data.cpu().numpy()
            return np.mean(y, axis=tuple(range(y.ndim - 1))) if y.ndim > 1 else y
        elif key == "sampling_rate":
            return self.get_samples_played_in_range(0, 0).sample_rate
        elif hasattr(super(), "__getitem__"):
            return super().__getitem__(key)
        else:
            raise TypeError("'torchcodec.decoders.AudioDecoder' object is not subscriptable")
<|endoftext|>__all__ = [
    "Audio",
    "BioSequence",
    "BioStructure",
    "Array2D",
    "Array3D",
    "Array4D",
    "Array5D",
    "ClassLabel",
    "Features",
    "Json",
    "LargeList",
    "List",
    "Sequence",
    "Value",
    "Image",
    "Mesh",
    "Translation",
    "TranslationVariableLanguages",
    "Video",
    "Pdf",
    "Nifti",
]
from .audio import Audio
from .bio_sequence import BioSequence
from .bio_structure import BioStructure
from .features import Array2D, Array3D, Array4D, Array5D, ClassLabel, Features, Json, LargeList, List, Sequence, Value
from .image import Image
from .mesh import Mesh
from .nifti import Nifti
from .pdf import Pdf
from .translation import Translation, TranslationVariableLanguages
from .video import Video
<|endoftext|>import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import numpy as np
import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    from torchcodec.decoders import AudioDecoder

    from .features import FeatureType


@dataclass
class Audio:
    """Audio [`Feature`] to extract audio data from an audio file.

    Input: The Audio feature accepts as input:
    - A `str`: Absolute path to the audio file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the audio file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the audio file to the archive file.
        - `bytes`: Bytes content of the audio file.

      This is useful for parquet or webdataset files which embed audio files.

    - A `dict` with the keys:

        - `array`: Array containing the audio sample
        - `sampling_rate`: Integer corresponding to the sampling rate of the audio sample.

    - A `torchcodec.decoders.AudioDecoder`: torchcodec audio decoder object.

    Output: The Audio features output data as `torchcodec.decoders.AudioDecoder` objects, with additional keys:

    - `array`: Array containing the audio sample
    - `sampling_rate`: Integer corresponding to the sampling rate of the audio sample.

    Args:
        sampling_rate (`int`, *optional*):
            Target sampling rate. If `None`, the native sampling rate is used.
        num_channels (`int`, *optional*):
             The desired number of channels of the samples. By default, the number of channels of the source is used.
             Audio decoding will return samples with shape (num_channels, num_samples)
             Currently `None` (number of channels of the source, default), `1` (mono) or `2` (stereo) channels are supported.
             The `num_channels` argument is passed to `torchcodec.decoders.AudioDecoder`.

             <Added version="4.4.0"/>
        decode (`bool`, defaults to `True`):
            Whether to decode the audio data. If `False`,
            returns the underlying dictionary in the format `{"path": audio_path, "bytes": audio_bytes}`.
        stream_index (`int`, *optional*):
            The streaming index to use from the file. If `None` defaults to the "best" index.

    Example:

    ```py
    >>> from datasets import load_dataset, Audio
    >>> ds = load_dataset("PolyAI/minds14", name="en-US", split="train")
    >>> ds = ds.cast_column("audio", Audio(sampling_rate=44100, num_channels=2))
    >>> ds[0]["audio"]
    <datasets.features._torchcodec.AudioDecoder object at 0x11642b6a0>
    >>> audio = ds[0]["audio"]
    >>> audio.get_samples_played_in_range(0, 10)
    AudioSamples:
        data (shape): torch.Size([2, 110592])
        pts_seconds: 0.0
        duration_seconds: 2.507755102040816
        sample_rate: 44100
    ```
    """

    sampling_rate: Optional[int] = None
    decode: bool = True
    num_channels: Optional[int] = None
    stream_index: Optional[int] = None
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Audio", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "AudioDecoder"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`,`bytearray`,`dict`, `AudioDecoder`):
                Data passed as input to Audio feature.

        Returns:
            `dict`
        """
        try:
            import torch
            from torchcodec.encoders import AudioEncoder  # needed to write audio files
        except ImportError as err:
            raise ImportError("To support encoding audio data, please install 'torchcodec'.") from err

        if value is None:
            raise ValueError("value must be provided")

        if config.TORCHCODEC_AVAILABLE:
            from torchcodec.decoders import AudioDecoder

        else:
            AudioDecoder = None

        if isinstance(value, str):
            return {"bytes": None, "path": value}
        elif isinstance(value, Path):
            return {"bytes": None, "path": str(value.absolute())}
        elif isinstance(value, (bytes, bytearray)):
            return {"bytes": value, "path": None}
        elif AudioDecoder is not None and isinstance(value, AudioDecoder):
            return encode_torchcodec_audio(value)
        elif "array" in value:
            # convert the audio array to wav bytes
            buffer = BytesIO()
            AudioEncoder(
                torch.from_numpy(value["array"].astype(np.float32)), sample_rate=value["sampling_rate"]
            ).to_file_like(buffer, format="wav", num_channels=self.num_channels)
            return {"bytes": buffer.getvalue(), "path": None}
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            if value["path"].endswith("pcm"):
                # "PCM" only has raw audio bytes
                if value.get("sampling_rate") is None:
                    # At least, If you want to convert "PCM-byte" to "WAV-byte", you have to know sampling rate
                    raise KeyError("To use PCM files, please specify a 'sampling_rate' in Audio object")
                if value.get("bytes"):
                    # If we already had PCM-byte, we don`t have to make "read file, make bytes" (just use it!)
                    bytes_value = np.frombuffer(value["bytes"], dtype=np.int16).astype(np.float32) / 32767
                else:
                    bytes_value = np.memmap(value["path"], dtype="h", mode="r").astype(np.float32) / 32767

                buffer = BytesIO()
                AudioEncoder(torch.from_numpy(bytes_value), sample_rate=value["sampling_rate"]).to_file_like(
                    buffer, format="wav", num_channels=self.num_channels
                )
                return {"bytes": buffer.getvalue(), "path": None}
            else:
                return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the audio bytes, and path is used to infer the audio format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"An audio sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(
        self, value: dict, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
    ) -> "AudioDecoder":
        """Decode example audio file into audio data.

        Args:
            value (`dict`):
                A dictionary with keys:

                - `path`: String with relative audio file path.
                - `bytes`: Bytes of the audio file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                audio files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`)

        Returns:
            `torchcodec.decoders.AudioDecoder`
        """
        if config.TORCHCODEC_AVAILABLE:
            from ._torchcodec import AudioDecoder
        else:
            raise ImportError("To support decoding audio data, please install 'torchcodec'.")

        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Audio(decode=True) instead.")

        path, bytes = (value["path"], value["bytes"]) if value["bytes"] is not None else (value["path"], None)
        if path is None and bytes is None:
            raise ValueError(f"An audio sample should have one of 'path' or 'bytes' but both are None in {value}.")

        if bytes is None and is_local_path(path):
            audio = AudioDecoder(
                path, stream_index=self.stream_index, sample_rate=self.sampling_rate, num_channels=self.num_channels
            )

        elif bytes is None:
            token_per_repo_id = token_per_repo_id or {}
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None

            download_config = DownloadConfig(token=token)
            f = xopen(path, "rb", download_config=download_config)
            audio = AudioDecoder(
                f, stream_index=self.stream_index, sample_rate=self.sampling_rate, num_channels=self.num_channels
            )

        else:
            audio = AudioDecoder(
                bytes, stream_index=self.stream_index, sample_rate=self.sampling_rate, num_channels=self.num_channels
            )
        audio._hf_encoded = {"path": path, "bytes": bytes}
        audio.metadata.path = path
        return audio

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, raise an error, otherwise flatten the feature into a dictionary."""
        from .features import Value

        if self.decode:
            raise ValueError("Cannot flatten a decoded Audio feature.")
        return {
            "bytes": Value("binary"),
            "path": Value("string"),
        }

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the Audio arrow storage type.
        The Arrow types that can be converted to the Audio pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the audio bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Audio arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_large_binary(storage.type):
            storage = array_cast(
                storage, pa.binary()
            )  # this can fail in case of big audios, paths should be used instead
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type) and storage.type.get_all_field_indices("array"):
            storage = pa.array(
                [Audio().encode_example(x) if x is not None else None for x in storage.to_numpy(zero_copy_only=False)]
            )
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed audio files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`)
                Whether to embed local files data in the array

                <Added version="4.8.5"/>
            remote_files (`bool`, defaults to `True`)
                Whether to embed remote files data in the array.
                E.g. files with paths that start with hf:// or https://

                <Added version="4.8.5"/>

        Returns:
            `pa.StructArray`: Array in the Audio arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if token_per_repo_id is None:
            token_per_repo_id = {}

        @no_op_if_value_is_null
        def path_to_bytes(path):
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (
                    path_to_bytes(x["path"])
                    if x["bytes"] is None
                    and ((local_files and is_local_path(x["path"])) or (remote_files and is_remote_url(x["path"])))
                    else x["bytes"]
                )
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                (
                    os.path.basename(path)
                    if (local_files and is_local_path(path)) or (remote_files and is_remote_url(path))
                    else path
                )
                if path is not None
                else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)


def encode_torchcodec_audio(audio: "AudioDecoder") -> dict:
    if hasattr(audio, "_hf_encoded"):
        return audio._hf_encoded
    else:
        try:
            from torchcodec.encoders import AudioEncoder  # needed to write audio files
        except ImportError as err:
            raise ImportError("To support encoding audio data, please install 'torchcodec'.") from err

        samples = audio.get_all_samples()
        buffer = BytesIO()
        num_channels = samples.data.shape[0]
        AudioEncoder(samples.data.cpu(), sample_rate=samples.sample_rate).to_file_like(
            buffer, format="wav", num_channels=num_channels
        )
        return {"bytes": buffer.getvalue(), "path": None}
<|endoftext|>from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import pyarrow as pa


if TYPE_CHECKING:
    from .features import FeatureType


@dataclass
class Translation:
    """`Feature` for translations with fixed languages per example.
    Here for compatibility with tfds.

    Args:
        languages (`dict`):
            A dictionary for each example mapping string language codes to string translations.

    Example:

    ```python
    >>> # At construction time:
    >>> datasets.features.Translation(languages=['en', 'fr', 'de'])
    >>> # During data generation:
    >>> yield {
    ...         'en': 'the cat',
    ...         'fr': 'le chat',
    ...         'de': 'die katze'
    ... }
    ```
    """

    languages: list[str]
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Translation", init=False, repr=False)

    def __call__(self):
        return pa.struct({lang: pa.string() for lang in sorted(self.languages)})

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """Flatten the Translation feature into a dictionary."""
        from .features import Value

        return {k: Value("string") for k in sorted(self.languages)}


@dataclass
class TranslationVariableLanguages:
    """`Feature` for translations with variable languages per example.
    Here for compatibility with tfds.

    Args:
        languages (`dict`):
            A dictionary for each example mapping string language codes to one or more string translations.
            The languages present may vary from example to example.

    Returns:
        - `language` or `translation` (variable-length 1D `tf.Tensor` of `tf.string`):
            Language codes sorted in ascending order or plain text translations, sorted to align with language codes.

    Example:

    ```python
    >>> # At construction time:
    >>> datasets.features.TranslationVariableLanguages(languages=['en', 'fr', 'de'])
    >>> # During data generation:
    >>> yield {
    ...         'en': 'the cat',
    ...         'fr': ['le chat', 'la chatte,']
    ...         'de': 'die katze'
    ... }
    >>> # Tensor returned :
    >>> {
    ...         'language': ['en', 'de', 'fr', 'fr'],
    ...         'translation': ['the cat', 'die katze', 'la chatte', 'le chat'],
    ... }
    ```
    """

    languages: Optional[list] = None
    num_languages: Optional[int] = None
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="TranslationVariableLanguages", init=False, repr=False)

    def __post_init__(self):
        self.languages = sorted(set(self.languages)) if self.languages else None
        self.num_languages = len(self.languages) if self.languages else None

    def __call__(self):
        return pa.struct({"language": pa.list_(pa.string()), "translation": pa.list_(pa.string())})

    def encode_example(self, translation_dict):
        lang_set = set(self.languages)
        if set(translation_dict) == {"language", "translation"}:
            return translation_dict
        elif self.languages and set(translation_dict) - lang_set:
            raise ValueError(
                f"Some languages in example ({', '.join(sorted(set(translation_dict) - lang_set))}) are not in valid set ({', '.join(lang_set)})."
            )

        # Convert dictionary into tuples, splitting out cases where there are
        # multiple translations for a single language.
        translation_tuples = []
        for lang, text in translation_dict.items():
            if isinstance(text, str):
                translation_tuples.append((lang, text))
            else:
                translation_tuples.extend([(lang, el) for el in text])

        # Ensure translations are in ascending order by language code.
        languages, translations = zip(*sorted(translation_tuples))

        return {"language": languages, "translation": translations}

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """Flatten the TranslationVariableLanguages feature into a dictionary."""
        from .features import List, Value

        return {
            "language": List(Value("string")),
            "translation": List(Value("string")),
        }
<|endoftext|>import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import string_to_dict


if TYPE_CHECKING:
    import trimesh

    from .features import FeatureType


@dataclass
class Mesh:
    """Mesh [`Feature`] to read 3D mesh data from a file.

    Input: The Mesh feature accepts as input:
    - A `str`: Absolute path to the mesh file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the mesh file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the mesh file to the archive file.
        - `bytes`: Bytes of the mesh file.

      This is useful for parquet or webdataset files which embed mesh files.

    - A `trimesh.Trimesh` or `trimesh.Scene`: 3D mesh or scene object.

    Output: The Mesh feature outputs data as `trimesh.Trimesh` or `trimesh.Scene` objects.

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the mesh data. If `False`,
            returns the underlying dictionary in the format `{"path": mesh_path, "bytes": mesh_bytes}`.
            Mesh decoding uses `trimesh` and supports `.glb`, `.ply`, and `.stl` files.
    """

    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "trimesh.Trimesh"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Mesh", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "trimesh.Trimesh", "trimesh.Scene"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`, `dict`, `trimesh.Trimesh`, or `trimesh.Scene`):
                Data passed as input to Mesh feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.TRIMESH_AVAILABLE:
            import trimesh
        else:
            trimesh = None

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif trimesh is not None and isinstance(value, (trimesh.Trimesh, trimesh.Scene)):
            return encode_trimesh_mesh(value)
        elif isinstance(value, dict) and value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        elif isinstance(value, dict) and (value.get("bytes") is not None or value.get("path") is not None):
            # store the mesh bytes, and path is used to infer the mesh format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"A mesh sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> Union["trimesh.Trimesh", "trimesh.Scene"]:
        """Decode example mesh file.

        Args:
            value (`dict`):
                A dictionary with keys:

                - `path`: String with absolute or relative mesh file path.
                - `bytes`: The bytes of the mesh file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                mesh files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `trimesh.Trimesh` or `trimesh.Scene`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Mesh(decode=True) instead.")

        if config.TRIMESH_AVAILABLE:
            import trimesh
        else:
            raise ImportError("To support decoding meshes, please install 'trimesh'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"A mesh should have one of 'path' or 'bytes' but both are None in {value}.")
            if is_local_path(path):
                file_type = _infer_mesh_file_type(path)
                if file_type is None:
                    raise ValueError("A mesh path should have a .glb, .ply, or .stl extension.")
                return trimesh.load(path, file_type=file_type)
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                bytes_ = f.read()

        file_type = _infer_mesh_file_type(path)
        if file_type is None:
            raise ValueError(
                "Decoding mesh bytes requires a 'path' value with a .glb, .ply, or .stl extension "
                "to infer the mesh file type."
            )
        return trimesh.load(BytesIO(bytes_), file_type=file_type)

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from .features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the Mesh arrow storage type.
        The Arrow types that can be converted to the Mesh pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.large_string()` - it must contain the "path" data (will be cast to string if possible)
        - `pa.binary()` - it must contain the mesh bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Mesh arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_large_string(storage.type):
            try:
                storage = storage.cast(pa.string())
            except pa.ArrowInvalid as e:
                raise ValueError(
                    f"Failed to cast large_string to string for Mesh feature. "
                    f"This can happen if string values exceed 2GB. "
                    f"Original error: {e}"
                ) from e
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_large_binary(storage.type):
            storage = array_cast(
                storage, pa.binary()
            )  # this can fail in case of big meshes, paths should be used instead
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())

        return array_cast(storage, self.pa_type)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed mesh files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.
            token_per_repo_id (`dict`, optional):
                Dictionary repo_id -> token to fetch the files bytes.
            local_files (`bool`, defaults to `True`):
                Whether to embed local files data in the array.
            remote_files (`bool`, defaults to `True`):
                Whether to embed remote files data in the array.

        Returns:
            `pa.StructArray`: Array in the Mesh arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if token_per_repo_id is None:
            token_per_repo_id = {}

        def path_to_bytes(path):
            if path is None:
                return None
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = token_per_repo_id.get(source_url_fields["repo_id"]) if source_url_fields is not None else None
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (
                    path_to_bytes(x["path"])
                    if x["bytes"] is None
                    and ((local_files and is_local_path(x["path"])) or (remote_files and is_remote_url(x["path"])))
                    else x["bytes"]
                )
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                (
                    os.path.basename(path)
                    if (local_files and is_local_path(path)) or (remote_files and is_remote_url(path))
                    else path
                )
                if path is not None
                else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)


def _infer_mesh_file_type(path: Optional[str]) -> Optional[str]:
    supported_file_types = {"glb", "ply", "stl"}
    if path is None:
        return None
    path_without_archive = path.split("::", 1)[0]
    path_without_query = path_without_archive.split("?", 1)[0]
    extension = os.path.splitext(path_without_query)[1].lower().lstrip(".")
    return extension if extension in supported_file_types else None


def encode_trimesh_mesh(mesh: Union["trimesh.Trimesh", "trimesh.Scene"]) -> dict[str, Optional[bytes | str]]:
    """Encode a trimesh mesh or scene object into GLB bytes."""
    metadata = getattr(mesh, "metadata", None) or {}
    path = metadata.get("file_path") or metadata.get("file_name") if isinstance(metadata, dict) else None
    if path is not None and os.path.isfile(path):
        return {"path": path, "bytes": None}
    bytes_ = mesh.export(file_type="glb")
    return {"path": os.path.basename(path) if path else "mesh.glb", "bytes": bytes_}
<|endoftext|>import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .. import config


@dataclass
class DownloadConfig:
    """Configuration for our cached path manager.

    Attributes:
        cache_dir (`str` or `Path`, *optional*):
            Specify a cache directory to save the file to (overwrite the
            default cache dir).
        force_download (`bool`, defaults to `False`):
            If `True`, re-download the file even if it's already cached in
            the cache dir.
        resume_download (`bool`, defaults to `False`):
            If `True`, resume the download if an incompletely received file is
            found.
        proxies (`dict`, *optional*):
            Deprecated and ignored. Set the `HTTP_PROXY` / `HTTPS_PROXY` environment variables or configure the
            HTTP client with [`huggingface_hub.set_client_factory`] instead.
        user_agent (`str`, *optional*):
            Optional string or dict that will be appended to the user-agent on remote
            requests.
        extract_compressed_file (`bool`, defaults to `False`):
            If `True` and the path point to a zip or tar file,
            extract the compressed file in a folder along the archive.
        force_extract (`bool`, defaults to `False`):
            If `True` when `extract_compressed_file` is `True` and the archive
            was already extracted, re-extract the archive and override the folder where it was extracted.
        delete_extracted (`bool`, defaults to `False`):
            Whether to delete (or keep) the extracted files.
        extract_on_the_fly (`bool`, defaults to `False`):
            If `True`, extract compressed files while they are being read.
        use_etag (`bool`, defaults to `True`):
            Whether to use the ETag HTTP response header to validate the cached files.
        num_proc (`int`, *optional*):
            The number of processes to launch to download the files in parallel.
        max_retries (`int`, default to `1`):
            The number of times to retry an HTTP request if it fails.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token
            for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        download_desc (`str`, *optional*):
            A description to be displayed alongside with the progress bar while downloading the files.
        disable_tqdm (`bool`, defaults to `False`):
            Whether to disable the individual files download progress bar
    """

    cache_dir: Optional[Union[str, Path]] = None
    force_download: bool = False
    resume_download: bool = False
    local_files_only: bool = False
    proxies: Optional[dict] = None
    user_agent: Optional[str] = None
    extract_compressed_file: bool = False
    force_extract: bool = False
    delete_extracted: bool = False
    extract_on_the_fly: bool = False
    use_etag: bool = True
    num_proc: Optional[int] = None
    max_retries: int = 1
    token: Optional[Union[str, bool]] = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    download_desc: Optional[str] = None
    disable_tqdm: bool = False

    def copy(self) -> "DownloadConfig":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def __setattr__(self, name, value):
        if name == "token" and getattr(self, "storage_options", None) is not None:
            if "hf" not in self.storage_options:
                self.storage_options["hf"] = {"endpoint": config.HF_ENDPOINT, "token": value}
            else:
                self.storage_options["hf"]["token"] = value
        super().__setattr__(name, value)

    def __post_init__(self):
        # update storage_options
        self.token = self.token
<|endoftext|>__all__ = [
    "DownloadConfig",
    "DownloadManager",
    "DownloadMode",
    "StreamingDownloadManager",
]

from .download_config import DownloadConfig
from .download_manager import DownloadManager, DownloadMode
from .streaming_download_manager import StreamingDownloadManager
<|endoftext|>import io
import os
from collections.abc import Iterable
from typing import Optional, Union

from ..utils.file_utils import (  # noqa: F401 # backward compatibility
    SINGLE_FILE_COMPRESSION_PROTOCOLS,
    ArchiveIterable,
    FilesIterable,
    _get_extraction_protocol,
    _get_path_extension,
    _prepare_path_and_storage_options,
    is_relative_path,
    url_or_path_join,
    xbasename,
    xdirname,
    xet_parse,
    xexists,
    xgetsize,
    xglob,
    xgzip_open,
    xisdir,
    xisfile,
    xjoin,
    xlistdir,
    xnumpy_load,
    xopen,
    xpandas_read_csv,
    xpandas_read_excel,
    xPath,
    xpyarrow_parquet_read_table,
    xrelpath,
    xsio_loadmat,
    xsplit,
    xsplitext,
    xwalk,
    xxml_dom_minidom_parse,
)
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig


logger = get_logger(__name__)


class StreamingDownloadManager:
    """
    Download manager that uses the "::" separator to navigate through (possibly remote) compressed archives.
    Contrary to the regular `DownloadManager`, the `download` and `extract` methods don't actually download nor extract
    data, but they rather return the path or url that could be opened using the `xopen` function which extends the
    built-in `open` function to stream data from remote files.
    """

    is_streaming = True

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
    ):
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._base_path = base_path or os.path.abspath(".")
        self.download_config = download_config or DownloadConfig()
        self.downloaded_size = None
        self.record_checksums = False

    @property
    def manual_dir(self):
        return self._data_dir

    def download(self, url_or_urls):
        """Normalize URL(s) of files to stream data from.
        This is the lazy version of `DownloadManager.download` for streaming.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) of files to stream data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input url_or_urls.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        ```
        """
        url_or_urls = map_nested(self._download_single, url_or_urls, map_tuple=True)
        return url_or_urls

    def _download_single(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        if is_relative_path(urlpath):
            # append the relative path to the base_path
            urlpath = url_or_path_join(self._base_path, urlpath)
        return urlpath

    def extract(self, url_or_urls):
        """Add extraction protocol for given url(s) for streaming.

        This is the lazy version of `DownloadManager.extract` for streaming.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) of files to stream data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        >>> extracted_files = dl_manager.extract(downloaded_files)
        ```
        """
        urlpaths = map_nested(self._extract, url_or_urls, map_tuple=True)
        return urlpaths

    def _extract(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        # get inner file: zip://train-00000.json.gz::https://foo.bar/data.zip -> zip://train-00000.json.gz
        protocol = _get_extraction_protocol(urlpath, download_config=self.download_config)
        path = urlpath.split("::")[0]
        extension = _get_path_extension(path)
        if extension in ["tgz", "tar"] or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
            raise NotImplementedError(
                f"Extraction protocol for TAR archives like '{urlpath}' is not implemented in streaming mode. "
                f"Please use `dl_manager.iter_archive` instead.\n\n"
                f"Example usage:\n\n"
                f"\turl = dl_manager.download(url)\n"
                f"\ttar_archive_iterator = dl_manager.iter_archive(url)\n\n"
                f"\tfor filename, file in tar_archive_iterator:\n"
                f"\t\t..."
            )
        if protocol is None:
            # no extraction
            return urlpath
        elif protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
            # there is one single file which is the uncompressed file
            inner_file = os.path.basename(urlpath.split("::")[0])
            inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
            return f"{protocol}://{inner_file}::{urlpath}"
        else:
            return f"{protocol}://::{urlpath}"

    def download_and_extract(self, url_or_urls):
        """Prepare given `url_or_urls` for streaming (add extraction protocol).

        This is the lazy version of `DownloadManager.download_and_extract` for streaming.

        Is equivalent to:

        ```
        urls = dl_manager.extract(dl_manager.download(url_or_urls))
        ```

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) to stream from data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.
        """
        return self.extract(self.download(url_or_urls))

    def iter_archive(self, urlpath_or_buf: Union[str, io.BufferedReader]) -> Iterable[tuple]:
        """Iterate over files within an archive.

        Args:
            urlpath_or_buf (`str` or `io.BufferedReader`):
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

        if hasattr(urlpath_or_buf, "read"):
            return ArchiveIterable.from_buf(urlpath_or_buf)
        else:
            return ArchiveIterable.from_urlpath(urlpath_or_buf, download_config=self.download_config)

    def iter_files(self, urlpaths: Union[str, list[str]]) -> Iterable[str]:
        """Iterate over files.

        Args:
            urlpaths (`str` or `list` of `str`):
                Root paths.

        Yields:
            str: File URL path.

        Example:

        ```py
        >>> files = dl_manager.download_and_extract('https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip')
        >>> files = dl_manager.iter_files(files)
        ```
        """
        return FilesIterable.from_urlpaths(urlpaths, download_config=self.download_config)

    def manage_extracted_files(self):
        pass

    def get_recorded_sizes_checksums(self):
        pass
<|endoftext|># Copyright 2020 The TensorFlow Datasets Authors.
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
"""Download manager interface."""

import enum
import io
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Optional, Union

import fsspec
from fsspec.core import url_to_fs
from tqdm.contrib.concurrent import thread_map

from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.file_utils import (
    ArchiveIterable,
    FilesIterable,
    cached_path,
    is_relative_path,
    stack_multiprocessing_download_progress_bars,
    url_or_path_join,
)
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger, tqdm
from ..utils.py_utils import NestedDataStructure, map_nested
from ..utils.track import tracked_str
from .download_config import DownloadConfig


logger = get_logger(__name__)


class DownloadMode(enum.Enum):
    """`Enum` for how to treat pre-existing downloads and data.

    The default mode is `REUSE_DATASET_IF_EXISTS`, which will reuse both
    raw downloads and the prepared dataset if they exist.

    The generations modes:

    |                                     | Downloads | Dataset |
    |-------------------------------------|-----------|---------|
    | `REUSE_DATASET_IF_EXISTS` (default) | Reuse     | Reuse   |
    | `REUSE_CACHE_IF_EXISTS`             | Reuse     | Fresh   |
    | `FORCE_REDOWNLOAD`                  | Fresh     | Fresh   |

    """

    REUSE_DATASET_IF_EXISTS = "reuse_dataset_if_exists"
    REUSE_CACHE_IF_EXISTS = "reuse_cache_if_exists"
    FORCE_REDOWNLOAD = "force_redownload"


class DownloadManager:
    is_streaming = False

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
        record_checksums=False,
    ):
        """Download manager constructor.

        Args:
            data_dir:
                can be used to specify a manual directory to get the files from.
            dataset_name (`str`):
                name of dataset this instance will be used for. If
                provided, downloads will contain which datasets they were used for.
            download_config (`DownloadConfig`):
                to specify the cache directory and other
                download options
            base_path (`str`):
                base path that is used when relative paths are used to
                download files. This can be a remote url.
            record_checksums (`bool`, defaults to `False`):
                Whether to record the checksums of the downloaded files. If None, the value is inferred from the builder.
        """
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._base_path = base_path or os.path.abspath(".")
        # To record what is being used: {url: {num_bytes: int, checksum: str}}
        self._recorded_sizes_checksums: dict[str, dict[str, Optional[Union[int, str]]]] = {}
        self.record_checksums = record_checksums
        self.download_config = download_config or DownloadConfig()
        self.downloaded_paths = {}
        self.extracted_paths = {}

    @property
    def manual_dir(self):
        return self._data_dir

    @property
    def downloaded_size(self):
        """Returns the total size of downloaded files."""
        return sum(checksums_dict["num_bytes"] for checksums_dict in self._recorded_sizes_checksums.values())

    def _record_sizes_checksums(self, url_or_urls: NestedDataStructure, downloaded_path_or_paths: NestedDataStructure):
        """Record size/checksum of downloaded files."""
        delay = 5
        for url, path in hf_tqdm(
            list(zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten())),
            delay=delay,
            desc="Computing checksums",
        ):
            # call str to support PathLike objects
            self._recorded_sizes_checksums[str(url)] = get_size_checksum_dict(
                path, record_checksum=self.record_checksums
            )

    def download(self, url_or_urls):
        """Download given URL(s).

        By default, only one process is used for download. Pass customized `download_config.num_proc` to change this behavior.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL or `list` or `dict` of URLs to download. Each URL is a `str`.

        Returns:
            `str` or `list` or `dict`:
                The downloaded paths matching the given input `url_or_urls`.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        ```
        """
        download_config = self.download_config.copy()
        download_config.extract_compressed_file = False
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading data"

        download_func = partial(self._download_batched, download_config=download_config)

        start_time = datetime.now()
        with stack_multiprocessing_download_progress_bars():
            downloaded_path_or_paths = map_nested(
                download_func,
                url_or_urls,
                map_tuple=True,
                num_proc=download_config.num_proc,
                desc="Downloading data files",
                batched=True,
                batch_size=-1,
            )
        duration = datetime.now() - start_time
