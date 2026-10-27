            return "\n".join(content_parts)
    return None
<|endoftext|>"""TsFile (table model) packaged builder — per-device wide format.

Each output row corresponds to a single device (identified by its TAG values).
The ``time`` column and every FIELD column are Arrow ``list<...>`` columns
holding the entire time series for that device. When the same device is
present in multiple TsFiles within a split, its data is merged across files
and the resulting lists are sorted in ascending time order.

Output schema layout::

    <tag1>:    string
    <tag2>:    string                   (one column per TAG)
    ...
    time:      list<timestamp[unit, tz]>
    <field1>:  list<original_type>      (one column per FIELD)
    <field2>:  list<original_type>
    ...

Reading model
-------------
Data is fetched **per device** via ``TsFileReader.query_table`` with a
push-down ``tag_filter``. For each split the builder:

1. Opens every input file once, calls ``get_all_devices`` to enumerate the
   ``(tag-tuple) → [files]`` index across all shards.
2. Iterates the index in stable order. For each device, streams Arrow
   batches from every contributing file, concatenates and sorts by time,
   and emits one wide row.

Peak memory is bounded by **one device's** total payload across the split,
not by the split's total size.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.tqdm import tqdm


logger = datasets.utils.logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _arrow_type(ts_dtype, *, unit: str, tz: Optional[str]) -> pa.DataType:
    """Map a tsfile ``TSDataType`` to its Arrow representation."""
    from tsfile.constants import TSDataType

    return {
        TSDataType.BOOLEAN: pa.bool_(),
        TSDataType.INT32: pa.int32(),
        TSDataType.INT64: pa.int64(),
        TSDataType.FLOAT: pa.float32(),
        TSDataType.DOUBLE: pa.float64(),
        TSDataType.TEXT: pa.string(),
        TSDataType.STRING: pa.string(),
        TSDataType.TIMESTAMP: pa.timestamp(unit, tz=tz),
        TSDataType.DATE: pa.date32(),
        TSDataType.BLOB: pa.binary(),
    }.get(ts_dtype, pa.string())


def _promote_tsdatatype(a, b):
    """Return the widest of two ``TSDataType`` values.

    Mirrors IoTDB's ``ALTER COLUMN ... SET DATA TYPE`` rules:

    - ``INT32 → INT64 → DOUBLE``
    - ``INT32 → FLOAT → DOUBLE``

    ``INT64`` and ``FLOAT`` cannot widen losslessly into either, so their
    join is ``DOUBLE``. Non-numeric or otherwise unrelated pairs raise.
    """
    if a == b:
        return a

    from tsfile.constants import TSDataType

    table = {
        (TSDataType.INT32, TSDataType.INT64): TSDataType.INT64,
        (TSDataType.INT32, TSDataType.FLOAT): TSDataType.FLOAT,
        (TSDataType.INT32, TSDataType.DOUBLE): TSDataType.DOUBLE,
        (TSDataType.INT64, TSDataType.FLOAT): TSDataType.DOUBLE,
        (TSDataType.INT64, TSDataType.DOUBLE): TSDataType.DOUBLE,
        (TSDataType.FLOAT, TSDataType.DOUBLE): TSDataType.DOUBLE,
    }
    if (a, b) in table:
        return table[(a, b)]
    if (b, a) in table:
        return table[(b, a)]
    raise ValueError(
        f"Incompatible column types across files: {a.name} vs {b.name}. "
        "Only numeric widening (INT32→INT64→DOUBLE, INT32→FLOAT→DOUBLE) is supported."
    )


def _to_epoch(value: Any, unit: str) -> int:
    """Coerce a timestamp boundary to an integer epoch in ``unit``.

    Accepts ``int`` (raw epoch in ``unit``), ``datetime``/``date``,
    ISO-8601 ``str``, or any ``pa.Scalar`` of timestamp type.
    """
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise TypeError(f"start_time/end_time must be a timestamp, got bool: {value!r}")
    if isinstance(value, int):
        return value
    try:
        # Normalize the various input shapes into something pa.scalar() can absorb
        # under a `timestamp[unit]` target type.
        if isinstance(value, _dt.datetime):
            if value.tzinfo is not None:
                value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        elif isinstance(value, _dt.date):
            value = _dt.datetime(value.year, value.month, value.day)
        elif isinstance(value, str):
            value = _dt.datetime.fromisoformat(value)
        return pa.scalar(value, type=pa.timestamp(unit)).value
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as e:
        raise TypeError(
            f"start_time/end_time must be a datetime, date, pa.TimestampScalar, "
            f"ISO-8601 str, or int epoch; got {type(value).__name__}: {value!r}"
        ) from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TsFileConfig(datasets.BuilderConfig):
    """BuilderConfig for TsFile (table model) — per-device wide format.

    Args:
        table_name (`str`, *optional*):
            Name of the table to read. When unset, the first table found in
            the first valid file is used. Lookups are case-insensitive.
        columns (`list[str]`, *optional*):
            Subset of FIELD columns to keep. TAG columns and the TIME column
            are *always* returned (they identify the device / its timeline
            and cannot be excluded). Names that refer to TAG or TIME columns,
            or to fields absent from every file, resolve quietly: TAGs/TIME
            are emitted as usual, and never-seen fields become all-null list
            columns. When unset, all FIELDs are returned.
        start_time, end_time (`datetime`, `date`, `pa.TimestampScalar`, ISO-8601 `str`, or `int`, *optional*):
            Inclusive timestamp range. Either bound may be omitted.
            ``datetime`` values are taken in their own tz (UTC if naive);
            ``int`` is interpreted as a raw epoch in ``timestamp_unit``.
        input_batch_size (`int`, *optional*, defaults to 65_536):
            Maximum number of rows fetched per Arrow batch from
            ``TsFileReader.query_table``. Controls peak memory while
            streaming a single device.
        output_batch_size (`int`, *optional*, defaults to 32):
            Number of devices (output dataset rows) packed into each Arrow
            record batch yielded to the writer. Also the granularity at
            which the dataset progress bar advances; smaller values give
            more responsive feedback on slow per-device reads, larger ones
            reduce per-batch overhead.
        features (`Features`, *optional*):
            Final Features schema. When provided, the metadata scan over
            input files is skipped.
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error"):
            What to do if a file cannot be opened or lacks the requested table.
        timestamp_unit (`Literal["s", "ms", "us", "ns"]`, *optional*, defaults to "ms"):
            Time unit for the timestamp column. IoTDB defaults to milliseconds.
        timestamp_tz (`str`, *optional*):
            Time zone for the timestamp column. ``None`` means timezone-naive.
    """

    table_name: Optional[str] = None
    columns: Optional[list[str]] = None
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    input_batch_size: int = 65_536
    output_batch_size: int = 32
    features: Optional[datasets.Features] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"
    timestamp_unit: Literal["s", "ms", "us", "ns"] = "ms"
    timestamp_tz: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.input_batch_size is None or self.input_batch_size <= 0:
            raise ValueError(f"`input_batch_size` must be a positive integer, got {self.input_batch_size}")
        if self.output_batch_size is None or self.output_batch_size <= 0:
            raise ValueError(f"`output_batch_size` must be a positive integer, got {self.output_batch_size}")
        if self.columns is not None and len(self.columns) == 0:
            raise ValueError("`columns` must be a non-empty list when provided.")
        if self.timestamp_unit not in ("s", "ms", "us", "ns"):
            raise ValueError(f"`timestamp_unit` must be one of 's', 'ms', 'us', 'ns', got {self.timestamp_unit!r}")
        if self.on_bad_files not in ("error", "warn", "skip"):
            raise ValueError(f"`on_bad_files` must be one of 'error', 'warn', 'skip', got {self.on_bad_files!r}")
        if self.start_time is not None:
            self.start_time = _to_epoch(self.start_time, self.timestamp_unit)
        if self.end_time is not None:
            self.end_time = _to_epoch(self.end_time, self.timestamp_unit)


# ---------------------------------------------------------------------------
# Internal sentinels
# ---------------------------------------------------------------------------


class _SkipSplit(Exception):
    """Raised internally to abort emitting a split entirely."""


class _MissingTableError(ValueError):
    def __init__(self, table: Optional[str], available):
        super().__init__(f"Table {table!r} not found in file. Available tables: {available}")


_TSFILE_MAGIC = b"TsFile"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TsFile(datasets.ArrowBasedBuilder):
    """Per-device wide-format builder for TsFile (table model)."""

    BUILDER_CONFIG_CLASS = TsFileConfig

    # ----- builder hooks ------------------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table: Optional[str] = None
        self._time_col: str = "time"
        self._tag_cols: list[str] = []
        self._field_inner: dict[str, pa.DataType] = {}
        self._requested_fields: Optional[list[str]] = None  # lowercased

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and not set(self.config.columns).issubset(set(self.config.features))
        ):
            raise ValueError(
                "Every entry in `columns` must also appear in `features`, but got "
                f"columns={self.config.columns} and features={list(self.config.features)}"
            )
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download(self.config.data_files)

        # Lowercase user-facing names to match tsfile's case-insensitive convention.
        self._table = self.config.table_name.lower() if self.config.table_name else None
        self._requested_fields = [c.lower() for c in self.config.columns] if self.config.columns else None

        all_files = [f for files in data_files.values() for f in files]
        scan = self._scan_metadata(all_files)
        if scan is None:
            raise ValueError(
                "Could not infer schema from any of the provided files. "
                "Set `features` explicitly or check the input files."
            )
        self._table = scan["table"]
        self._time_col = scan["time_col"]
        self._tag_cols = scan["tag_cols"]
        self._field_inner = scan["field_inner"]

        if self.info.features is None:
            self.info.features = self._build_features()

        return [
            datasets.SplitGenerator(name=split, gen_kwargs={"files": list(files)})
            for split, files in data_files.items()
        ]

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        target_schema = self.info.features.arrow_schema
        try:
            yield from self._fold_split(files, target_schema)
        except _SkipSplit:
            return

    # ----- metadata scan ------------------------------------------------

    def _scan_metadata(self, files) -> Optional[dict]:
        """Walk every file and unify table name, TAG columns, FIELD types."""
        from tsfile.constants import TIME_COLUMN, ColumnCategory

        wanted_table = self._table
        wanted_fields = set(self._requested_fields) if self._requested_fields is not None else None

        table: Optional[str] = wanted_table
        time_col: Optional[str] = None
        tag_cols: list[str] = []
        tag_seen: set[str] = set()
        # Per-field widest TSDataType seen so far (we map to Arrow at the end).
        field_widest: dict = {}

        for file in files:
            try:
                with self._open_reader(file) as reader:
                    schemas = self._schemas_by_lc(reader)
                    self._require_table_model(file, schemas)
                    if table is None:
                        table = next(iter(schemas))
                    if table not in schemas:
                        raise _MissingTableError(table, list(schemas))
                    for col in schemas[table].get_columns():
                        name = col.get_column_name()
                        cat = col.get_category()
                        ts_dtype = col.get_data_type()
                        if cat == ColumnCategory.TIME:
                            time_col = name
                        elif cat == ColumnCategory.TAG:
                            if name not in tag_seen:
                                tag_seen.add(name)
                                tag_cols.append(name)
                        else:  # FIELD
                            if wanted_fields is not None and name not in wanted_fields:
                                continue
                            prev = field_widest.get(name)
                            field_widest[name] = ts_dtype if prev is None else _promote_tsdatatype(prev, ts_dtype)
            except Exception as e:
                if self._should_reraise(file, e):
                    raise
                continue

        if table is None:
            return None

        unit = self.config.timestamp_unit
        tz = self.config.timestamp_tz

        if self._requested_fields is not None:
            # Honor user order; silently drop names that turned out to be TAGs
            # or the TIME column (TAGs are emitted as their own scalar columns
            # and TIME is always emitted as a list column — neither may also
            # appear as a list-typed field, which would collide on schema name).
            reserved = tag_seen | {time_col} if time_col is not None else tag_seen
            field_inner: dict[str, pa.DataType] = {}
            for name in self._requested_fields:
                if name in reserved:
                    continue
                ts_dtype = field_widest.get(name)
                if ts_dtype is not None:
                    field_inner[name] = _arrow_type(ts_dtype, unit=unit, tz=tz)
                else:
                    # Field never appeared in any file — keep as a nullable
                    # float64 list, fully filled with nulls at read time.
                    field_inner[name] = pa.float64()
        else:
            field_inner = {n: _arrow_type(d, unit=unit, tz=tz) for n, d in field_widest.items()}

        return {
            "table": table,
            "time_col": time_col or TIME_COLUMN,
            "tag_cols": tag_cols,
            "field_inner": field_inner,
        }

    def _build_features(self) -> datasets.Features:
        unit = self.config.timestamp_unit
        tz = self.config.timestamp_tz
        fields: list[pa.Field] = [pa.field(t, pa.string()) for t in self._tag_cols]
        fields.append(pa.field(self._time_col, pa.list_(pa.timestamp(unit, tz=tz))))
        for name, inner in self._field_inner.items():
            fields.append(pa.field(name, pa.list_(inner)))
        return datasets.Features.from_arrow_schema(pa.schema(fields))

    # ----- per-split folding -------------------------------------------

    def _fold_split(self, files, target_schema: pa.Schema):
        """Stream every device in this split via per-device tag-filter pushdown.

        Open one ``TsFileReader`` per file, build a cross-file device index
        keyed by ``(tag-tuple)``, then iterate devices in stable order. For
        each device, ``query_table(tag_filter=...)`` reads only that device's
        rows from each contributing file, so peak memory is bounded by one
        device's payload across the split — never the split's total size.
        """
        if self._table is None:
            raise _SkipSplit

        readers: dict[str, Any] = {}
        try:
            for file in files:
                try:
                    readers[file] = self._open_reader(file)
                except Exception as e:
                    if self._should_reraise(file, e):
                        raise
                    continue

            device_index, file_meta = self._build_device_index(readers)
            if not device_index:
                return

            yield from self._iter_device_batches(device_index, file_meta, readers, target_schema)
        finally:
            for reader in readers.values():
                try:
                    reader.close()
                except Exception:
                    pass

    def _build_device_index(self, readers: dict):
        """Walk every open reader and build the cross-file device index.

        Returns ``(device_index, file_meta)``:

        - ``device_index``: list of ``(device_key, [file_path, ...])`` pairs
          in stable first-seen order. ``device_key`` is a tuple aligned to
          ``self._tag_cols`` (the unified tag-column order).
        - ``file_meta``: maps each readable file to its per-file context
          (``tag_cols``, ``field_cols``, ``time_col``).
        """
        from tsfile.constants import ColumnCategory

        device_to_files: dict[tuple, list[str]] = {}
        device_order: list[tuple] = []
        file_meta: dict[str, dict] = {}
        # ``self._table`` was lowercased either by user-input normalization in
        # ``_split_generators`` or by ``_schemas_by_lc`` during auto-detect.
        table_lc = self._table

        files_iter = tqdm(
            readers.items(),
            total=len(readers),
            desc="Indexing TsFile devices",
            unit="file",
        )

        for file, reader in files_iter:
            try:
                schemas = self._schemas_by_lc(reader)
                self._require_table_model(file, schemas)
                if table_lc not in schemas:
                    raise _MissingTableError(table_lc, list(schemas))
                schema = schemas[table_lc]

                file_tag_cols: list[str] = []
                file_field_cols: set[str] = set()
                time_col = self._time_col
                for col in schema.get_columns():
                    name = col.get_column_name()
                    cat = col.get_category()
                    if cat == ColumnCategory.TIME:
                        time_col = name
                    elif cat == ColumnCategory.TAG:
                        file_tag_cols.append(name)
                    elif cat == ColumnCategory.FIELD:
                        file_field_cols.add(name)

                file_meta[file] = {
                    "tag_cols": file_tag_cols,
                    "field_cols": file_field_cols,
                    "time_col": time_col,
                }

                for device in reader.get_all_devices():
                    if device.table_name is None or device.table_name.lower() != table_lc:
                        continue
                    file_tag_values = list(device.segments[1 : 1 + len(file_tag_cols)])
                    file_tag_dict = dict(zip(file_tag_cols, file_tag_values))
                    unified_key = tuple(file_tag_dict.get(c) for c in self._tag_cols)
                    if any(v is None for v in unified_key):
                        raise ValueError(
                            f"Device in file '{file}' has missing tag values: "
                            f"{dict(zip(self._tag_cols, unified_key))}. "
                            "Schema-evolution devices with NULL tag values are not "
                            "supported because tsfile lacks an IS NULL tag filter."
                        )
                    if unified_key not in device_to_files:
                        device_to_files[unified_key] = []
                        device_order.append(unified_key)
                    device_to_files[unified_key].append(file)
            except Exception as e:
                if self._should_reraise(file, e):
                    raise
                file_meta.pop(file, None)
                continue

        device_index = [(key, device_to_files[key]) for key in device_order]
        return device_index, file_meta

    def _iter_device_batches(self, device_index, file_meta, readers, target_schema: pa.Schema):
        """Materialize devices in order and emit packed Arrow tables."""
        field_names = list(self._field_inner.keys())
        rows: list[dict] = []
        batch_idx = 0
        for device_key, contributing_files in device_index:
            time_chunks: list[np.ndarray] = []
            field_chunks: dict[str, list] = {f: [] for f in field_names}
            for file in contributing_files:
                if file not in file_meta:
                    continue
                try:
                    ts_arr, vals = self._read_device_from_file(readers[file], file_meta[file], device_key)
                except Exception as e:
                    if self._should_reraise(file, e):
                        raise
                    continue
                if len(ts_arr) == 0:
                    continue
                time_chunks.append(ts_arr)
                for f in field_names:
                    field_chunks[f].append(vals.get(f))  # None → all-null contribution
            if not time_chunks:
                continue  # device produced no rows in time range → skip

            row = self._finalize_device(device_key, time_chunks, field_chunks, field_names)
            rows.append(row)
            if len(rows) >= self.config.output_batch_size:
                yield Key(0, batch_idx), self._rows_to_table(rows, target_schema)
                rows = []
                batch_idx += 1
        if rows:
            yield Key(0, batch_idx), self._rows_to_table(rows, target_schema)

    def _read_device_from_file(self, reader, meta: dict, device_key: tuple) -> tuple:
        """Stream one device's rows from one file via ``query_table`` pushdown.

        Returns ``(timestamps, {field_name: values})``. The dict only includes
        field columns that this file owns *and* that the builder requested;
        callers fill missing fields with all-null contributions.
        """
        from tsfile import tag_eq

        file_tag_cols: list[str] = meta["tag_cols"]
        file_field_cols: set[str] = meta["field_cols"]
        time_col: str = meta["time_col"]

        # Build the tag filter only over this file's tag columns. The unified
        # device key carries one value per builder tag; map back by name.
        unified_to_value = dict(zip(self._tag_cols, device_key))
        tag_filter = None
        for c in file_tag_cols:
            v = unified_to_value.get(c)
            if v is None:
                # Caught earlier in _build_device_index; defensive guard.
                return np.array([], dtype=np.int64), {}
            expr = tag_eq(c, str(v))
            tag_filter = expr if tag_filter is None else tag_filter & expr

        # Project: requested fields ∩ this file's fields. ``query_table``
        # always returns the time column, but it requires at least one
        # non-time column — fall back to any owned field if the user's
        # selection has nothing in this file.
        requested = list(self._field_inner.keys())
        fields_to_query = [f for f in requested if f in file_field_cols]
        fallback_only = False
        if not fields_to_query:
            if not file_field_cols:
                return np.array([], dtype=np.int64), {}
            fields_to_query = [next(iter(file_field_cols))]
            fallback_only = True

        kwargs: dict = {"tag_filter": tag_filter, "batch_size": self.config.input_batch_size}
        if self.config.start_time is not None:
            kwargs["start_time"] = self.config.start_time
        if self.config.end_time is not None:
            kwargs["end_time"] = self.config.end_time

        ts_parts: list[np.ndarray] = []
        field_parts: dict[str, list] = {f: [] for f in fields_to_query}
        with reader.query_table(self._table, fields_to_query, **kwargs) as rs:
            while True:
                batch = rs.read_arrow_batch()
                if batch is None:
                    break
                if batch.num_rows == 0:
                    continue
                ts_parts.append(np.asarray(batch.column(time_col).to_numpy(), dtype=np.int64))
                for f in fields_to_query:
                    col = batch.column(f)
                    # tsfile's arrow reader tags TIMESTAMP / DATE field columns
                    # with a fixed unit (e.g. ``timestamp[ns]``) regardless of
                    # the value's original write unit. Reinterpret as raw
                    # int64/int32 ticks so the downstream
                    # ``pa.array(type=timestamp[<our unit>])`` treats them as
                    # ticks in the unit declared by our schema, instead of
                    # cross-unit casting (which would raise on data loss).
                    if pa.types.is_timestamp(col.type):
                        field_parts[f].append(col.cast(pa.int64()).to_numpy(zero_copy_only=False))
                    elif pa.types.is_date(col.type):
                        field_parts[f].append(col.cast(pa.int32()).to_numpy(zero_copy_only=False))
                    else:
                        field_parts[f].append(col.to_numpy(zero_copy_only=False))

        if not ts_parts:
            return np.array([], dtype=np.int64), {}
        ts_full = np.concatenate(ts_parts) if len(ts_parts) > 1 else ts_parts[0]
        vals_full = {f: (np.concatenate(parts) if len(parts) > 1 else parts[0]) for f, parts in field_parts.items()}

        # Defensive boundary mask: native query paths may emit rows just
        # outside the requested window in some chunk-boundary cases.
        if self.config.start_time is not None or self.config.end_time is not None:
            lo = self.config.start_time if self.config.start_time is not None else np.iinfo(np.int64).min
            hi = self.config.end_time if self.config.end_time is not None else np.iinfo(np.int64).max
            mask = (ts_full >= lo) & (ts_full <= hi)
            if not mask.all():
                ts_full = ts_full[mask]
                vals_full = {f: arr[mask] for f, arr in vals_full.items()}

        # Drop the fallback "pick one" column from the user-visible payload.
        if fallback_only:
            vals_full = {}
        return ts_full, vals_full

    def _finalize_device(
        self,
        device_key: tuple,
        time_chunks: list,
        field_chunks: dict,
        field_names: list[str],
    ) -> dict:
        """Concatenate per-file chunks, sort by time, and return one row.

        Raises ``ValueError`` if the same timestamp appears more than once
        for a device (within or across files) — tsfile's per-device timeline
        is required to be unique-by-timestamp.
        """
        time_arr = np.concatenate(time_chunks) if time_chunks else np.array([], dtype=np.int64)
        n_total = len(time_arr)

        if n_total > 0:
            sort_idx = np.argsort(time_arr, kind="stable")
            time_sorted = time_arr[sort_idx]
            if n_total > 1:
                dup_mask = time_sorted[1:] == time_sorted[:-1]
                if dup_mask.any():
                    dup_ts = int(time_sorted[1:][dup_mask][0])
                    raise ValueError(
                        f"Duplicate timestamp {dup_ts} for device "
                        f"{dict(zip(self._tag_cols, device_key))}. "
                        "Cross-file or within-file duplicate timestamps are not supported."
                    )
        else:
            sort_idx = None
            time_sorted = time_arr

        row: dict = {}
        for tag_name, tag_val in zip(self._tag_cols, device_key):
            row[tag_name] = None if tag_val is None else str(tag_val)
        row[self._time_col] = time_sorted

        for fname in field_names:
            chunks = field_chunks.get(fname, [])
            materialized: list = []
            for tchunk, fchunk in zip(time_chunks, chunks):
                if fchunk is None:
                    materialized.append(np.full(len(tchunk), None, dtype=object))
                else:
                    materialized.append(fchunk)
            arr = np.concatenate(materialized) if materialized else np.array([], dtype=object)
            if sort_idx is not None and len(arr) == n_total:
                arr = arr[sort_idx]
            row[fname] = arr
        return row

    # ----- arrow assembly ----------------------------------------------

    def _rows_to_table(self, rows: list[dict], target_schema: pa.Schema) -> pa.Table:
        """Convert a batch of row dicts into an Arrow table matching ``target_schema``."""
        arrays: list[pa.Array] = []
        for f in target_schema:
            values = [r[f.name] for r in rows]
            if pa.types.is_list(f.type):
                arrays.append(self._build_list_array(values, f.type))
            else:
                arrays.append(pa.array(values, type=f.type))
        pa_table = pa.Table.from_arrays(arrays, names=[f.name for f in target_schema])
        return table_cast(pa_table, target_schema)

    @staticmethod
    def _build_list_array(values: list, list_type: pa.ListType) -> pa.ListArray:
        """Build a ``ListArray`` from a list of per-row 1D arrays / sequences."""
        inner_type = list_type.value_type
        offsets = [0]
        flat_chunks: list = []
        total = 0
        for v in values:
            if v is None:
                length = 0
            else:
                length = len(v)
                flat_chunks.append(v)
            total += length
            offsets.append(total)

        if flat_chunks:
            try:
                flat = np.concatenate([np.asarray(c) for c in flat_chunks])
            except ValueError:
                # Heterogeneous shapes / dtypes → fall back to a Python list.
                flat = []
                for c in flat_chunks:
                    flat.extend(list(c))
            flat_arr = pa.array(flat, type=inner_type, from_pandas=True)
        else:
            flat_arr = pa.array([], type=inner_type)

        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), flat_arr)

    # ----- file / error handling ---------------------------------------

    @staticmethod
    def _open_reader(file: str):
        """Open a file as a ``TsFileReader`` after verifying its magic header.

        The C library's ``TsFileReader`` constructor silently returns an
        invalid handle for non-tsfile inputs, and any subsequent call on it
        segfaults. The 6-byte ``TsFile`` magic header is checked first to
        bail out cleanly.
        """
        from tsfile import TsFileReader

        try:
            with open(file, "rb") as fh:
                header = fh.read(len(_TSFILE_MAGIC))
        except OSError as e:
            raise ValueError(f"Cannot open file {file!r}: {e}") from e
        if header != _TSFILE_MAGIC:
            raise ValueError(f"File {file!r} is not a valid TsFile (bad magic header).")
        return TsFileReader(file)

    @staticmethod
    def _require_table_model(file: str, schemas) -> None:
        if not schemas:
            raise ValueError(
                f"File {file!r} is a tree-model TsFile, which is not supported. "
                "Only table-model TsFiles can be loaded."
            )

    @staticmethod
    def _schemas_by_lc(reader) -> dict:
        """Return ``get_all_table_schemas()`` keyed by lowercased table name.

        TsFile / IoTDB treat table names case-insensitively, but the Python
        binding's ``get_all_table_schemas()`` returns a dict keyed by whatever
        casing the file was written with. Lowercasing the keys here lets all
        downstream lookups use a single canonical form.
        """
        return {name.lower(): schema for name, schema in reader.get_all_table_schemas().items()}

    def _should_reraise(self, file: str, exc: BaseException) -> bool:
        """Apply ``on_bad_files`` policy. Returns True iff the caller should re-raise."""
        mode = self.config.on_bad_files
        if mode == "error":
            logger.error(f"Failed to read file '{file}' with error {type(exc).__name__}: {exc}")
            return True
        if mode == "warn":
            logger.warning(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")
        else:
            logger.debug(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")
        return False
<|endoftext|>import collections
import io
import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.json as paj
import pyarrow.parquet as pq

import datasets
from datasets import config
from datasets.builder import Key
from datasets.features.features import FeatureType, _visit, _visit_with_path, _VisitPath, require_storage_cast
from datasets.utils.file_utils import readline


logger = datasets.utils.logging.get_logger(__name__)


def count_path_segments(path):
    return path.replace("\\", "/").count("/")


@dataclass
class FolderBasedBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for AutoFolder."""

    features: Optional[datasets.Features] = None
    drop_labels: bool = None
    drop_metadata: bool = None
    metadata_filenames: list[str] = None
    filters: Optional[Union[ds.Expression, list[tuple], list[list[tuple]]]] = None

    def __post_init__(self):
        super().__post_init__()


class FolderBasedBuilder(datasets.GeneratorBasedBuilder):
    """
    Base class for generic data loaders for vision and image data.


    Abstract class attributes to be overridden by a child class:
        BASE_FEATURE: feature object to decode data (i.e. datasets.Image, datasets.Audio, ...)
        BASE_COLUMN_NAME: string key name of a base feature (i.e. "image", "audio", ...)
        BUILDER_CONFIG_CLASS: builder config inherited from `folder_based_builder.FolderBasedBuilderConfig`
        EXTENSIONS: list of allowed extensions (only files with these extensions and METADATA_FILENAME files
            will be included in a dataset)
    """

    BASE_FEATURE: type[FeatureType]
    BASE_COLUMN_NAME: str
    BUILDER_CONFIG_CLASS: FolderBasedBuilderConfig
    EXTENSIONS: list[str]

    METADATA_FILENAMES: list[str] = ["metadata.csv", "metadata.jsonl", "metadata.parquet"]

    def _base_feature(self) -> FeatureType:
        """Instantiate the base feature, allowing subclasses to specify options such as format."""
        return self.BASE_FEATURE()

    def _info(self):
        if not self.config.data_dir and not self.config.data_files:
            raise ValueError(
                "Folder-based datasets require either `data_dir` or `data_files` to be specified. "
                "Neither was provided."
            )

        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        # Do an early pass if:
        # * `drop_labels` is None (default) or False, to infer the class labels
        # * `drop_metadata` is None (default) or False, to find the metadata files
        do_analyze = not self.config.drop_labels or not self.config.drop_metadata
        labels, path_depths = set(), set()
        all_metadata_files = collections.defaultdict(set)
        metadata_filenames = self.config.metadata_filenames or self.METADATA_FILENAMES

        def analyze(files_or_archives, downloaded_files_or_dirs, split):
            if len(downloaded_files_or_dirs) == 0:
                return
            # The files are separated from the archives at this point, so check the first sample
            # to see if it's a file or a directory and iterate accordingly
            if os.path.isfile(downloaded_files_or_dirs[0]):
                original_files, downloaded_files = files_or_archives, downloaded_files_or_dirs
                for original_file, downloaded_file in zip(original_files, downloaded_files):
                    original_file, downloaded_file = str(original_file), str(downloaded_file)
                    _, original_file_ext = os.path.splitext(original_file)
                    if original_file_ext.lower() in self.EXTENSIONS:
                        if not self.config.drop_labels:
                            labels.add(os.path.basename(os.path.dirname(original_file)))
                            path_depths.add(count_path_segments(original_file))
                    elif os.path.basename(original_file) in metadata_filenames:
                        all_metadata_files[split].add((original_file, None, downloaded_file))
                    else:
                        original_file_name = os.path.basename(original_file)
                        logger.debug(
                            f"The file '{original_file_name}' was ignored: it is not a {self.BASE_COLUMN_NAME}, and is not {metadata_filenames} either."
                        )
            else:
                archives, downloaded_dirs = files_or_archives, downloaded_files_or_dirs
                for archive, downloaded_dir in zip(archives, downloaded_dirs):
                    archive, downloaded_dir = str(archive), str(downloaded_dir)
                    for downloaded_dir_file in dl_manager.iter_files(downloaded_dir):
                        _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                        if downloaded_dir_file_ext in self.EXTENSIONS:
                            if not self.config.drop_labels:
                                labels.add(os.path.basename(os.path.dirname(downloaded_dir_file)))
                                path_depths.add(count_path_segments(downloaded_dir_file))
                        elif os.path.basename(downloaded_dir_file) in metadata_filenames:
                            all_metadata_files[split].add((None, downloaded_dir, downloaded_dir_file))
                        else:
                            archive_file_name = os.path.basename(archive)
                            original_file_name = os.path.basename(downloaded_dir_file)
                            logger.debug(
                                f"The file '{original_file_name}' from the archive '{archive_file_name}' was ignored: it is not a {self.BASE_COLUMN_NAME}, and is not {metadata_filenames} either."
                            )

        data_files = self.config.data_files
        splits = []
        for split_name, files in data_files.items():
            files, metadata_files, archives = self._split_files_and_metadata_and_archives(files)
            downloaded_files = dl_manager.download(files)
            downloaded_metadata_files = dl_manager.download(metadata_files)
            downloaded_dirs = dl_manager.download_and_extract(archives)
            if do_analyze:  # drop_metadata is None or False, drop_labels is None or False
                logger.info(f"Searching for labels and/or metadata files in {split_name} data files...")
                analyze(files, downloaded_files, split_name)
                analyze(metadata_files, downloaded_metadata_files, split_name)
                analyze(archives, downloaded_dirs, split_name)

                if all_metadata_files:
                    # add metadata if `all_metadata_files` are found and `drop_metadata` is None (default) or False
                    add_metadata = not self.config.drop_metadata
                    # if `all_metadata_files` are found, don't add labels
                    add_labels = False
                else:
                    # if `all_metadata_files` are not found, don't add metadata
                    add_metadata = False
                    # if `all_metadata_files` are not found and `drop_labels` is None (default) -
                    # add labels if files are on the same level in directory hierarchy and there is more than one label
                    add_labels = (
                        (len(labels) > 1 and len(path_depths) == 1)
                        if self.config.drop_labels is None
                        else not self.config.drop_labels
                    )

                if add_labels:
                    logger.info("Adding the labels inferred from data directories to the dataset's features...")
                if add_metadata:
                    logger.info("Adding metadata to the dataset...")
            else:
                add_labels, add_metadata, all_metadata_files = False, False, {}

            # files info (original_file, None, downloaded_file)
            files = tuple(zip(files, [None] * len(files), downloaded_files))
            # archives info (original_archive_file, downloaded_dir, downloaded_files)
            files += tuple(
                (archive, downloaded_dir, dl_manager.iter_files(downloaded_dir))
                for archive, downloaded_dir in zip(archives, downloaded_dirs)
            )
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "files": files,
                        "metadata_files": all_metadata_files.get(split_name, []),
                        "add_labels": add_labels,
                        "add_metadata": add_metadata,
                    },
                )
            )

        if add_metadata:
            # Verify that:
            # * all metadata files have the same set of features in each split
            # * the `file_name` key is one of the metadata keys and is of type string
            features_per_metadata_file: list[tuple[str, datasets.Features]] = []

            # Check that all metadata files share the same format
            metadata_ext = {
                os.path.splitext(original_metadata_file or downloaded_metadata_file)[-1]
                for original_metadata_file, _, downloaded_metadata_file in itertools.chain.from_iterable(
                    all_metadata_files.values()
                )
            }
            if len(metadata_ext) > 1:
                raise ValueError(f"Found metadata files with different extensions: {list(metadata_ext)}")
            metadata_ext = metadata_ext.pop()

            for split_metadata_files in all_metadata_files.values():
                pa_metadata_table = None
                for _, _, downloaded_metadata_file in split_metadata_files:
                    for pa_metadata_table in self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext):
                        break  # just fetch the first rows
                    if pa_metadata_table is not None:
                        features_per_metadata_file.append(
                            (downloaded_metadata_file, datasets.Features.from_arrow_schema(pa_metadata_table.schema))
                        )
                        break  # no need to fetch all the files
            for downloaded_metadata_file, metadata_features in features_per_metadata_file:
                if metadata_features != features_per_metadata_file[0][1]:
                    raise ValueError(
                        f"Metadata files {downloaded_metadata_file} and {features_per_metadata_file[0][0]} have different features: {features_per_metadata_file[0]} != {metadata_features}"
                    )
            metadata_features = features_per_metadata_file[0][1]
            feature_not_found = True

            def _set_feature(feature):
                nonlocal feature_not_found
                if isinstance(feature, dict):
                    out = type(feature)()
                    for key in feature:
                        if (key == "file_name" or key.endswith("_file_name")) and (
                            feature[key] == datasets.Value("string") or feature[key] == datasets.Value("large_string")
                        ):
                            key = key[: -len("_file_name")] or self.BASE_COLUMN_NAME
                            out[key] = self._base_feature()
                            feature_not_found = False
                        elif (key == "file_names" or key.endswith("_file_names")) and (
                            feature[key]
                            in [datasets.List(datasets.Value("string")), datasets.List(datasets.Value("large_string"))]
                        ):
                            key = key[: -len("_file_names")] or (self.BASE_COLUMN_NAME + "s")
                            out[key] = datasets.List(self._base_feature())
                            feature_not_found = False
                        elif (key == "file_names" or key.endswith("_file_names")) and (
                            feature[key] == [datasets.Value("string")]
                            or feature[key] == [datasets.Value("large_string")]
                        ):
                            key = key[: -len("_file_names")] or (self.BASE_COLUMN_NAME + "s")
                            out[key] = [self._base_feature()]
                            feature_not_found = False
                        else:
                            out[key] = feature[key]
                    return out
                return feature

            metadata_features = _visit(metadata_features, _set_feature)

            if feature_not_found:
                raise ValueError(
                    "`file_name`, `*_file_name`, `file_names` or `*_file_names` must be present as dictionary key in metadata files"
                )
        else:
            metadata_features = None

        # Normally, we would do this in _info, but we need to know the labels and/or metadata
        # before building the features
        if self.config.features is None:
            if add_metadata:
                self.info.features = metadata_features
            elif add_labels:
                self.info.features = datasets.Features(
                    {
                        self.BASE_COLUMN_NAME: self._base_feature(),
                        "label": datasets.ClassLabel(names=sorted(labels)),
                    }
                )
            else:
                self.info.features = datasets.Features({self.BASE_COLUMN_NAME: self._base_feature()})

        return splits

    def _split_files_and_metadata_and_archives(self, data_files):
        files, metadata_files, archives = [], [], []
        metadata_filenames = self.config.metadata_filenames or self.METADATA_FILENAMES
        for data_file in data_files:
            data_file_root, data_file_ext = os.path.splitext(data_file)
            _, second_data_file_ext = os.path.splitext(data_file_root)
            if data_file_ext.lower() in self.EXTENSIONS or second_data_file_ext.lower() in self.EXTENSIONS:
                files.append(data_file)
            elif os.path.basename(data_file) in metadata_filenames:
                metadata_files.append(data_file)
            elif data_file_ext.lower() == ".zip":
                archives.append(data_file)
        return files, metadata_files, archives

    def _read_metadata(self, metadata_file: str, metadata_ext: str = "") -> Iterator[pa.Table]:
        """using the same logic as the Csv, Json and Parquet dataset builders to stream the data"""
        if self.config.filters is not None:
            filter_expr = (
                pq.filters_to_expression(self.config.filters)
                if isinstance(self.config.filters, list)
                else self.config.filters
            )
        else:
            filter_expr = None
        if metadata_ext == ".csv":
            chunksize = 10_000  # 10k lines
            schema = self.config.features.arrow_schema if self.config.features else None
            # dtype allows reading an int column as str
            dtype = (
                {
                    name: dtype.to_pandas_dtype() if not require_storage_cast(feature) else object
                    for name, dtype, feature in zip(schema.names, schema.types, self.config.features.values())
                }
                if schema is not None
                else None
            )
            csv_file_reader = pd.read_csv(metadata_file, iterator=True, dtype=dtype, chunksize=chunksize)
            for df in csv_file_reader:
                pa_table = pa.Table.from_pandas(df)
                if self.config.filters is not None:
                    pa_table = pa_table.filter(filter_expr)
                if len(pa_table) > 0:
                    yield pa_table
        elif metadata_ext == ".jsonl":
            with open(metadata_file, "rb") as f:
                chunksize: int = 10 << 20  # 10MB
                # Use block_size equal to the chunk size divided by 32 to leverage multithreading
                # Set a default minimum value of 16kB if the chunk size is really small
                block_size = max(chunksize // 32, 16 << 10)
                while True:
                    batch = f.read(chunksize)
                    if not batch:
                        break
                    # Finish current line
                    try:
                        batch += f.readline()
                    except (AttributeError, io.UnsupportedOperation):
                        batch += readline(f)
                    while True:
                        try:
                            pa_table = paj.read_json(
                                io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size)
                            )
                            break
                        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                            if (
                                isinstance(e, pa.ArrowInvalid)
                                and "straddling" not in str(e)
                                or block_size > len(batch)
                            ):
                                raise
                            else:
                                # Increase the block size in case it was too small.
                                # The block size will be reset for the next file.
                                logger.debug(
                                    f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}."
                                )
                                block_size *= 2
                    if self.config.filters is not None:
                        pa_table = pa_table.filter(filter_expr)
                    if len(pa_table) > 0:
                        yield pa_table
        else:
            with open(metadata_file, "rb") as f:
                parquet_fragment = ds.ParquetFileFormat().make_fragment(f)
                if parquet_fragment.row_groups:
                    batch_size = parquet_fragment.row_groups[0].num_rows
                else:
                    batch_size = config.DEFAULT_MAX_BATCH_SIZE
                for record_batch in parquet_fragment.to_batches(
                    batch_size=batch_size,
                    filter=filter_expr,
                    batch_readahead=0,
                    fragment_readahead=0,
                ):
                    yield pa.Table.from_batches([record_batch])

    def _generate_shards(self, files, metadata_files, add_metadata, add_labels):
        if add_metadata:
            for _, _, downloaded_metadata_file in metadata_files:
                yield downloaded_metadata_file
        else:
            for _, downloaded_dir, downloaded_file in files:
                yield downloaded_dir or downloaded_file

    def _generate_examples(self, files, metadata_files, add_metadata, add_labels):
        if add_metadata:
            feature_paths = []

            def find_feature_path(feature, feature_path):
                nonlocal feature_paths
                if feature_path and isinstance(feature, self.BASE_FEATURE):
                    feature_paths.append(feature_path)

            _visit_with_path(self.info.features, find_feature_path)

            for shard_idx, metadata_file_info in enumerate(metadata_files):
                if len(metadata_file_info) == 2:
                    original_metadata_file, downloaded_metadata_file = metadata_file_info
                else:
                    original_metadata_file, downloaded_metadata_dir, downloaded_metadata_file = metadata_file_info
                metadata_ext = os.path.splitext(original_metadata_file or downloaded_metadata_file)[-1]
                downloaded_metadata_dir = os.path.dirname(downloaded_metadata_file)

                def set_feature(item, feature_path: _VisitPath):
                    if len(feature_path) == 2 and isinstance(feature_path[0], str) and feature_path[1] == 0:
                        item[feature_path[0]] = item.pop("file_names", None) or item.pop(
                            feature_path[0] + "_file_names", None
                        )
                    elif len(feature_path) == 1 and isinstance(feature_path[0], str):
                        item[feature_path[0]] = item.pop("file_name", None) or item.pop(
                            feature_path[0] + "_file_name", None
                        )
                    elif len(feature_path) == 0:
                        if item is not None:
                            # Guard against path traversal (CWE-22): a crafted `file_name` such as
                            # "../../etc/passwd" or an absolute path must not be able to escape the
                            # metadata file's directory and read arbitrary files on the host.
                            #
                            # The attacker-controlled `file_name` must be a plain relative path. In
                            # particular it must not introduce an fsspec URL scheme: `file://` and
                            # `local://` resolve to arbitrary *local* files, and any other scheme
                            # would sidestep the containment check below. Legitimate reads from a
                            # downloaded archive use a `zip://<file_name>::<container>` URL where the
                            # scheme lives on `downloaded_metadata_dir` (the container), never on the
                            # `file_name` value itself, so forbidding "://" here does not break them.
                            if "://" in item:
                                raise ValueError(
                                    f"Invalid metadata file_name '{item}': `file_name` must be a relative path "
                                    f"pointing inside the directory containing the metadata file. URL schemes "
                                    f"(e.g. 'file://', 'local://') are not allowed."
                                )
                            file_relpath = os.path.normpath(item).replace("\\", "/")
                            if (
                                os.path.isabs(item)
                                or os.path.isabs(file_relpath)
                                or file_relpath == ".."
                                or file_relpath.startswith("../")
                            ):
                                raise ValueError(
                                    f"Invalid metadata file_name '{item}': `file_name` must be a relative path "
                                    f"pointing inside the directory containing the metadata file. Absolute paths "
                                    f"and parent-directory ('..') traversal that escape the dataset directory are "
                                    f"not allowed."
                                )
                            item = os.path.join(downloaded_metadata_dir, file_relpath)
                    return item

                for pa_metadata_table in self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext):
                    for sample_idx, sample in enumerate(pa_metadata_table.to_pylist()):
                        for feature_path in feature_paths:
                            _nested_apply(sample, feature_path, set_feature)
                        yield Key(shard_idx, sample_idx), sample
        else:
            if self.config.filters is not None:
                filter_expr = (
                    pq.filters_to_expression(self.config.filters)
                    if isinstance(self.config.filters, list)
                    else self.config.filters
                )
            for shard_idx, (original_file, _, downloaded_files) in enumerate(files):
                if isinstance(downloaded_files, str):
                    downloaded_files = [downloaded_files]
                for sample_idx, downloaded_file in enumerate(downloaded_files):
                    sample = {self.BASE_COLUMN_NAME: downloaded_file}
                    if add_labels:
                        sample["label"] = os.path.basename(os.path.dirname(original_file or downloaded_file))
                    if self.config.filters is not None:
                        pa_table = pa.Table.from_pylist([sample]).filter(filter_expr)
                        if len(pa_table) == 0:
                            continue
                    yield Key(shard_idx, sample_idx), sample


def _nested_apply(item: Any, feature_path: _VisitPath, func: Callable[[Any, _VisitPath], Any]):
    # see _visit_with_path() to see how feature paths are constructed
    item = func(item, feature_path)
    if feature_path:
        key = feature_path[0]
        if key == 0:
            for i in range(len(item)):
                item[i] = _nested_apply(item[i], feature_path[1:], func)
        else:
            item[key] = _nested_apply(item[key], feature_path[1:], func)
    return item
<|endoftext|>from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import pandas as pd
import pyarrow as pa

import datasets
import datasets.config
from datasets.builder import Key
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
from datasets.utils.py_utils import Literal


logger = datasets.utils.logging.get_logger(__name__)

_PANDAS_READ_CSV_NO_DEFAULT_PARAMETERS = ["names", "prefix"]
_PANDAS_READ_CSV_DEPRECATED_PARAMETERS = ["warn_bad_lines", "error_bad_lines", "mangle_dupe_cols"]
_PANDAS_READ_CSV_NEW_1_3_0_PARAMETERS = ["encoding_errors", "on_bad_lines"]
_PANDAS_READ_CSV_NEW_2_0_0_PARAMETERS = ["date_format"]
_PANDAS_READ_CSV_DEPRECATED_2_2_0_PARAMETERS = ["verbose"]


@dataclass
class CsvConfig(datasets.BuilderConfig):
    """BuilderConfig for CSV."""

    sep: str = ","
    delimiter: Optional[str] = None
    header: Optional[Union[int, list[int], str]] = "infer"
    names: Optional[list[str]] = None
    column_names: Optional[list[str]] = None
    index_col: Optional[Union[int, str, list[int], list[str]]] = None
    usecols: Optional[Union[list[int], list[str]]] = None
    prefix: Optional[str] = None
    mangle_dupe_cols: bool = True
    engine: Optional[Literal["c", "python", "pyarrow"]] = None
    converters: dict[Union[int, str], Callable[[Any], Any]] = None
    true_values: Optional[list] = None
    false_values: Optional[list] = None
    skipinitialspace: bool = False
    skiprows: Optional[Union[int, list[int]]] = None
    nrows: Optional[int] = None
    na_values: Optional[Union[str, list[str]]] = None
    keep_default_na: bool = True
    na_filter: bool = True
    verbose: bool = False
    skip_blank_lines: bool = True
    thousands: Optional[str] = None
    decimal: str = "."
    lineterminator: Optional[str] = None
    quotechar: str = '"'
    quoting: int = 0
    escapechar: Optional[str] = None
    comment: Optional[str] = None
    encoding: Optional[str] = None
    dialect: Optional[str] = None
    error_bad_lines: bool = True
    warn_bad_lines: bool = True
    skipfooter: int = 0
    doublequote: bool = True
    memory_map: bool = False
    float_precision: Optional[str] = None
    chunksize: int = 10_000
    features: Optional[datasets.Features] = None
    encoding_errors: Optional[str] = "strict"
    on_bad_lines: Literal["error", "warn", "skip"] = "error"
    date_format: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.delimiter is not None:
            self.sep = self.delimiter
        if self.column_names is not None:
            self.names = self.column_names

    @property
    def pd_read_csv_kwargs(self):
        pd_read_csv_kwargs = {
            "sep": self.sep,
            "header": self.header,
            "names": self.names,
            "index_col": self.index_col,
            "usecols": self.usecols,
            "prefix": self.prefix,
            "mangle_dupe_cols": self.mangle_dupe_cols,
            "engine": self.engine,
            "converters": self.converters,
            "true_values": self.true_values,
            "false_values": self.false_values,
            "skipinitialspace": self.skipinitialspace,
            "skiprows": self.skiprows,
            "nrows": self.nrows,
            "na_values": self.na_values,
            "keep_default_na": self.keep_default_na,
            "na_filter": self.na_filter,
            "verbose": self.verbose,
            "skip_blank_lines": self.skip_blank_lines,
            "thousands": self.thousands,
            "decimal": self.decimal,
            "lineterminator": self.lineterminator,
            "quotechar": self.quotechar,
            "quoting": self.quoting,
            "escapechar": self.escapechar,
            "comment": self.comment,
            "encoding": self.encoding,
            "dialect": self.dialect,
            "error_bad_lines": self.error_bad_lines,
            "warn_bad_lines": self.warn_bad_lines,
            "skipfooter": self.skipfooter,
            "doublequote": self.doublequote,
            "memory_map": self.memory_map,
            "float_precision": self.float_precision,
            "chunksize": self.chunksize,
            "encoding_errors": self.encoding_errors,
            "on_bad_lines": self.on_bad_lines,
            "date_format": self.date_format,
        }

        # some kwargs must not be passed if they don't have a default value
        # some others are deprecated and we can also not pass them if they are the default value
        for pd_read_csv_parameter in _PANDAS_READ_CSV_NO_DEFAULT_PARAMETERS + _PANDAS_READ_CSV_DEPRECATED_PARAMETERS:
            if pd_read_csv_kwargs[pd_read_csv_parameter] == getattr(CsvConfig(), pd_read_csv_parameter):
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 1.3 new arguments
        if datasets.config.PANDAS_VERSION.release < (1, 3):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_NEW_1_3_0_PARAMETERS:
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 2.0 new arguments
        if not (datasets.config.PANDAS_VERSION.major >= 2):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_NEW_2_0_0_PARAMETERS:
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 2.2 deprecated arguments
        if datasets.config.PANDAS_VERSION.release >= (2, 2):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_DEPRECATED_2_2_0_PARAMETERS:
                if pd_read_csv_kwargs[pd_read_csv_parameter] == getattr(CsvConfig(), pd_read_csv_parameter):
                    del pd_read_csv_kwargs[pd_read_csv_parameter]

        return pd_read_csv_kwargs


class Csv(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = CsvConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        base_data_files = dl_manager.download(self.config.data_files)
        extracted_data_files = dl_manager.extract(base_data_files)
        splits = []
        for split_name, extracted_files in extracted_data_files.items():
            files_iterables = [dl_manager.iter_files(extracted_file) for extracted_file in extracted_files]
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
                pa_table = pa.Table.from_arrays([pa_table[field.name] for field in schema], schema=schema)
            else:
                # more expensive cast; allows str <-> int/float or str to Audio for example
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_shards(self, base_files, files_iterables):
        yield from base_files

    def _generate_tables(self, base_files, files_iterables):
        schema = self.config.features.arrow_schema if self.config.features else None
        # dtype allows reading an int column as str
        dtype = (
            {
                name: dtype.to_pandas_dtype() if not require_storage_cast(feature) else object
                for name, dtype, feature in zip(schema.names, schema.types, self.config.features.values())
            }
            if schema is not None
            else None
        )
        for shard_idx, files_iterable in enumerate(files_iterables):
            for file in files_iterable:
                csv_file_reader = pd.read_csv(file, iterator=True, dtype=dtype, **self.config.pd_read_csv_kwargs)
                try:
                    for batch_idx, df in enumerate(csv_file_reader):
                        pa_table = pa.Table.from_pandas(df)
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield Key(shard_idx, batch_idx), self._cast_table(pa_table)
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
<|endoftext|>import multiprocessing
import os
from functools import partial
from typing import BinaryIO, Optional, Union

import fsspec
import pyarrow as pa

from .. import Dataset, Features, NamedSplit, config
from ..formatting import query_table
from ..packaged_modules.json.json import Json
from ..utils import tqdm as hf_tqdm
from ..utils.json import get_json_field_paths_from_feature, json_decode_field
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


class JsonDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        field: Optional[str] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        self.field = field
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = Json(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            field=field,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.split, in_memory=self.keep_in_memory)
        return dataset


class JsonDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.encoding = "utf-8"
        self.storage_options = storage_options or {}
        self.to_json_kwargs = to_json_kwargs

    def write(self) -> int:
        _ = self.to_json_kwargs.pop("path_or_buf", None)
        orient = self.to_json_kwargs.pop("orient", "records")
        lines = self.to_json_kwargs.pop("lines", True if orient == "records" else False)
        if "index" not in self.to_json_kwargs and orient in ["split", "table"]:
            self.to_json_kwargs["index"] = False

        # Determine the default compression value based on self.path_or_buf type
        default_compression = "infer" if isinstance(self.path_or_buf, (str, bytes, os.PathLike)) else None
        compression = self.to_json_kwargs.pop("compression", default_compression)

        if compression not in [None, "infer", "gzip", "bz2", "xz"]:
            raise NotImplementedError(f"`datasets` currently does not support {compression} compression")

        if not lines and self.batch_size < self.dataset.num_rows:
            raise NotImplementedError(
                "Output JSON will not be formatted correctly when lines = False and batch_size < number of rows in the dataset. Use pandas.DataFrame.to_json() instead."
            )

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(
                self.path_or_buf, "wb", compression=compression, **(self.storage_options or {})
            ) as buffer:
                written = self._write(file_obj=buffer, orient=orient, lines=lines, **self.to_json_kwargs)
        else:
            if compression:
                raise NotImplementedError(
                    f"The compression parameter is not supported when writing to a buffer, but compression={compression}"
                    " was passed. Please provide a local path instead."
                )
            written = self._write(file_obj=self.path_or_buf, orient=orient, lines=lines, **self.to_json_kwargs)
        return written

    def _batch_json(self, args):
        offset, orient, lines, to_json_kwargs = args

        batch = query_table(
            table=self.dataset.data,
            key=slice(offset, offset + self.batch_size),
            indices=self.dataset._indices,
        )
        if self.dataset._indices is not None:
            # Gathering creates one-row chunks, which are expensive to convert to pandas.
            # Combine only the selected batch, not the potentially much larger source table.
            try:
                batch = batch.combine_chunks()
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowCapacityError, pa.ArrowMemoryError):
                # Nested string/binary payloads can overflow 32-bit offsets (ArrowInvalid).
                # Older Arrow builds also lacked extension concatenation support, and
                # concatenation can hit capacity/memory limits. Leave the batch chunked
                # so to_pandas can still convert it without this optional concatenation.
                pass
        batch = batch.to_pandas(integer_object_nulls=True)
        for json_field_path in get_json_field_paths_from_feature(self.dataset.features):
            col, *json_field_subpath = json_field_path
            batch[col] = batch[col].apply(partial(json_decode_field, json_field_path=json_field_subpath))
        json_str = batch.to_json(path_or_buf=None, orient=orient, lines=lines, **to_json_kwargs)
        if not json_str.endswith("\n"):
            json_str += "\n"
        return json_str.encode(self.encoding)

    def _write(
        self,
        file_obj: BinaryIO,
        orient,
        lines,
        **to_json_kwargs,
    ) -> int:
        """Writes the pyarrow table as JSON lines to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0

        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(
                range(0, len(self.dataset), self.batch_size),
                unit="ba",
                desc="Creating json from Arrow format",
            ):
                json_str = self._batch_json((offset, orient, lines, to_json_kwargs))
                written += file_obj.write(json_str)
        else:
            num_rows, batch_size = len(self.dataset), self.batch_size
            with multiprocessing.Pool(self.num_proc) as pool:
                for json_str in hf_tqdm(
                    pool.imap(
                        self._batch_json,
                        [(offset, orient, lines, to_json_kwargs) for offset in range(0, num_rows, batch_size)],
                    ),
                    total=(num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size,
                    unit="ba",
                    desc="Creating json from Arrow format",
                ):
                    written += file_obj.write(json_str)

        return written
<|endoftext|>from typing import Optional

from .. import Features, NamedSplit
from ..packaged_modules.text.text import Text
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


class TextDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = Text(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.split, in_memory=self.keep_in_memory)
        return dataset
<|endoftext|>import json
import os
from typing import BinaryIO, Optional, Union

import fsspec
import pyarrow.parquet as pq

from .. import Dataset, Features, NamedSplit, config
from ..arrow_writer import (
    get_parquet_column_options,
    get_writer_batch_size_from_data_size,
    get_writer_batch_size_from_features,
)
from ..formatting import query_table
from ..packaged_modules import _PACKAGED_DATASETS_MODULES
from ..packaged_modules.parquet.parquet import Parquet
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


class ParquetDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        hash = _PACKAGED_DATASETS_MODULES["parquet"][1]
        self.builder = Parquet(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            hash=hash,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.split, in_memory=self.keep_in_memory)
        return dataset


class ParquetDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        storage_options: Optional[dict] = None,
        use_content_defined_chunking: Union[bool, dict] = True,
        write_page_index: bool = True,
        **parquet_writer_kwargs,
    ):
        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = (
            batch_size
            or get_writer_batch_size_from_features(dataset.features)
            or get_writer_batch_size_from_data_size(len(dataset), dataset._estimate_nbytes())
        )
        self.storage_options = storage_options or {}
        self.parquet_writer_kwargs = parquet_writer_kwargs
        if use_content_defined_chunking is True:
            use_content_defined_chunking = config.DEFAULT_CDC_OPTIONS
        self.use_content_defined_chunking = use_content_defined_chunking
        self.write_page_index = write_page_index

    def write(self) -> int:
        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(self.path_or_buf, "wb", **(self.storage_options or {})) as buffer:
                written = self._write(
                    file_obj=buffer,
                    batch_size=self.batch_size,
                    **self.parquet_writer_kwargs,
                )
        else:
            written = self._write(
                file_obj=self.path_or_buf,
                batch_size=self.batch_size,
                **self.parquet_writer_kwargs,
            )
        return written

    def _write(self, file_obj: BinaryIO, batch_size: int, **parquet_writer_kwargs) -> int:
        """Writes the pyarrow table as Parquet to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0
        _ = parquet_writer_kwargs.pop("path_or_buf", None)
        schema = self.dataset.features.arrow_schema

        writer = pq.ParquetWriter(
            file_obj,
            schema=schema,
            use_content_defined_chunking=self.use_content_defined_chunking,
            write_page_index=self.write_page_index,
            **get_parquet_column_options(self.dataset.features, schema),
            **parquet_writer_kwargs,
        )

        for offset in hf_tqdm(
            range(0, len(self.dataset), batch_size),
            unit="ba",
            desc="Creating parquet from Arrow format",
        ):
            batch = query_table(
                table=self.dataset._data,
                key=slice(offset, offset + batch_size),
                indices=self.dataset._indices,
            )
            writer.write_table(batch)
            written += batch.nbytes

        # TODO(kszucs): we may want to persist multiple parameters
        if self.use_content_defined_chunking is not False:
            writer.add_key_value_metadata({"content_defined_chunking": json.dumps(self.use_content_defined_chunking)})

        writer.close()
        return written
<|endoftext|>import multiprocessing
from typing import TYPE_CHECKING, Optional, Union

from .. import Dataset, Features, config
from ..formatting import query_table
from ..packaged_modules.sql.sql import Sql
from ..utils import tqdm as hf_tqdm
from .abc import AbstractDatasetInputStream


if TYPE_CHECKING:
    import sqlite3

    import sqlalchemy


class SqlDatasetReader(AbstractDatasetInputStream):
    def __init__(
        self,
        sql: Union[str, "sqlalchemy.sql.Selectable"],
        con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ):
        super().__init__(features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs)
        self.builder = Sql(
            cache_dir=cache_dir,
            features=features,
            sql=sql,
            con=con,
            **kwargs,
        )

    def read(self):
        download_config = None
        download_mode = None
        verification_mode = None
        base_path = None

        self.builder.download_and_prepare(
            download_config=download_config,
            download_mode=download_mode,
            verification_mode=verification_mode,
            base_path=base_path,
        )

        # Build dataset for splits
        dataset = self.builder.as_dataset(split="train", in_memory=self.keep_in_memory)
        return dataset


class SqlDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        **to_sql_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.name = name
        self.con = con
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.to_sql_kwargs = to_sql_kwargs

    def write(self) -> int:
        _ = self.to_sql_kwargs.pop("sql", None)
        _ = self.to_sql_kwargs.pop("con", None)
        index = self.to_sql_kwargs.pop("index", False)

        written = self._write(index=index, **self.to_sql_kwargs)
        return written

    def _batch_sql(self, args):
        offset, index, to_sql_kwargs = args
        to_sql_kwargs = {**to_sql_kwargs, "if_exists": "append"} if offset > 0 else to_sql_kwargs
        batch = query_table(
            table=self.dataset.data,
            key=slice(offset, offset + self.batch_size),
            indices=self.dataset._indices,
        )
        df = batch.to_pandas(integer_object_nulls=True)
        num_rows = df.to_sql(self.name, self.con, index=index, **to_sql_kwargs)
        return num_rows or len(df)

    def _write(self, index, **to_sql_kwargs) -> int:
        """Writes the pyarrow table as SQL to a database.

        Caller is responsible for opening and closing the SQL connection.
        """
        written = 0

        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(
                range(0, len(self.dataset), self.batch_size),
                unit="ba",
                desc="Creating SQL from Arrow format",
            ):
                written += self._batch_sql((offset, index, to_sql_kwargs))
        else:
            num_rows, batch_size = len(self.dataset), self.batch_size
            with multiprocessing.Pool(self.num_proc) as pool:
                for num_rows in hf_tqdm(
                    pool.imap(
                        self._batch_sql,
                        [(offset, index, to_sql_kwargs) for offset in range(0, num_rows, batch_size)],
                    ),
                    total=(num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size,
                    unit="ba",
                    desc="Creating SQL from Arrow format",
                ):
                    written += num_rows

        return written
<|endoftext|>from abc import ABC, abstractmethod
from typing import Optional, Union

from .. import Dataset, DatasetDict, Features, IterableDataset, IterableDatasetDict, NamedSplit
from ..utils.typing import NestedDataStructureLike, PathLike


class AbstractDatasetReader(ABC):
    def __init__(
        self,
        path_or_paths: Optional[NestedDataStructureLike[PathLike]] = None,
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self.path_or_paths = path_or_paths
        self.split = split if split or isinstance(path_or_paths, dict) else "train"
        self.features = features
        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory
        self.streaming = streaming
        self.num_proc = num_proc
        self.kwargs = kwargs

    @abstractmethod
    def read(self) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        pass


class AbstractDatasetInputStream(ABC):
    def __init__(
        self,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self.features = features
        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory
        self.streaming = streaming
        self.num_proc = num_proc
        self.kwargs = kwargs

    @abstractmethod
    def read(self) -> Union[Dataset, IterableDataset]:
        pass
<|endoftext|>import multiprocessing
import os
from typing import BinaryIO, Optional, Union

import fsspec

from .. import Dataset, Features, NamedSplit, config
from ..formatting import query_table
from ..packaged_modules.csv.csv import Csv
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


class CsvDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = Csv(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.split, in_memory=self.keep_in_memory)
        return dataset


class CsvDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_csv_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.encoding = "utf-8"
        self.storage_options = storage_options or {}
        self.to_csv_kwargs = to_csv_kwargs

    def write(self) -> int:
        _ = self.to_csv_kwargs.pop("path_or_buf", None)
        header = self.to_csv_kwargs.pop("header", True)
        index = self.to_csv_kwargs.pop("index", False)

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(self.path_or_buf, "wb", **(self.storage_options or {})) as buffer:
                written = self._write(file_obj=buffer, header=header, index=index, **self.to_csv_kwargs)
        else:
            written = self._write(file_obj=self.path_or_buf, header=header, index=index, **self.to_csv_kwargs)
        return written

    def _batch_csv(self, args):
        offset, header, index, to_csv_kwargs = args

        batch = query_table(
            table=self.dataset.data,
            key=slice(offset, offset + self.batch_size),
            indices=self.dataset._indices,
        )
        csv_str = batch.to_pandas(integer_object_nulls=True).to_csv(
            path_or_buf=None, header=header if (offset == 0) else False, index=index, **to_csv_kwargs
        )
        return csv_str.encode(self.encoding)

    def _write(self, file_obj: BinaryIO, header, index, **to_csv_kwargs) -> int:
        """Writes the pyarrow table as CSV to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0

        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(
                range(0, len(self.dataset), self.batch_size),
                unit="ba",
                desc="Creating CSV from Arrow format",
            ):
                csv_str = self._batch_csv((offset, header, index, to_csv_kwargs))
                written += file_obj.write(csv_str)

        else:
            num_rows, batch_size = len(self.dataset), self.batch_size
            with multiprocessing.Pool(self.num_proc) as pool:
                for csv_str in hf_tqdm(
                    pool.imap(
                        self._batch_csv,
                        [(offset, header, index, to_csv_kwargs) for offset in range(0, num_rows, batch_size)],
                    ),
                    total=(num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size,
                    unit="ba",
                    desc="Creating CSV from Arrow format",
                ):
                    written += file_obj.write(csv_str)

        return written
<|endoftext|>from typing import Optional

import pyspark

from .. import Features, NamedSplit
from ..download import DownloadMode
from ..packaged_modules.spark.spark import Spark
from .abc import AbstractDatasetReader


class SparkDatasetReader(AbstractDatasetReader):
    """A dataset reader that reads from a Spark DataFrame.

    When caching, cache materialization is parallelized over Spark; an NFS that is accessible to the driver must be
    provided. Streaming is not currently supported.
    """

    def __init__(
        self,
        df: pyspark.sql.DataFrame,
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        streaming: bool = True,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        working_dir: str = None,
        load_from_cache_file: bool = True,
        file_format: str = "arrow",
        **kwargs,
    ):
        super().__init__(
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            **kwargs,
        )
        self._load_from_cache_file = load_from_cache_file
        self._file_format = file_format
        self.builder = Spark(
            df=df,
            features=features,
            cache_dir=cache_dir,
            working_dir=working_dir,
            **kwargs,
        )

    def read(self):
        if self.streaming:
            return self.builder.as_streaming_dataset(split=self.split)
        download_mode = None if self._load_from_cache_file else DownloadMode.FORCE_REDOWNLOAD
        self.builder.download_and_prepare(
            download_mode=download_mode,
            file_format=self._file_format,
        )
        return self.builder.as_dataset(split=self.split)
<|endoftext|>from typing import Callable, Optional

from .. import Features, NamedSplit, Split
from ..packaged_modules.generator.generator import Generator
from .abc import AbstractDatasetInputStream


class GeneratorDatasetInputStream(AbstractDatasetInputStream):
    def __init__(
        self,
        generator: Callable,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        gen_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        split: NamedSplit = Split.TRAIN,
        fingerprint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        self.builder = Generator(
            cache_dir=cache_dir,
            features=features,
            generator=generator,
            gen_kwargs=gen_kwargs,
            split=split,
            config_id="default-fingerprint=" + fingerprint if fingerprint else None,
            **kwargs,
        )
        self.fingerprint = fingerprint

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.builder.config.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(split=self.builder.config.split, in_memory=self.keep_in_memory)
            if self.fingerprint:
                dataset._fingerprint = self.fingerprint
        return dataset
<|endoftext|># Copyright 2021 The HuggingFace Authors.
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
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


if TYPE_CHECKING:
    import jax
    import jaxlib

logger = get_logger()

DEVICE_MAPPING: Optional[dict] = None


class JaxFormatter(TensorFormatter[Mapping, "jax.Array", Mapping]):
    def __init__(self, features=None, device=None, token_per_repo_id=None, **jnp_array_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        import jax
        from jaxlib.xla_client import Device

        if isinstance(device, Device):
            raise ValueError(
                f"Expected {device} to be a `str` not {type(device)}, as `jaxlib.xla_extension.Device` "
                "is not serializable neither with `pickle` nor with `dill`. Instead you can surround "
                "the device with `str()` to get its string identifier that will be internally mapped "
                "to the actual `jaxlib.xla_extension.Device`."
            )
        self.device = device if isinstance(device, str) else str(jax.devices()[0])
        # using global variable since `jaxlib.xla_extension.Device` is not serializable neither
        # with `pickle` nor with `dill`, so we need to use a global variable instead
        global DEVICE_MAPPING
        if DEVICE_MAPPING is None:
            DEVICE_MAPPING = self._map_devices_to_str()
        if self.device not in list(DEVICE_MAPPING.keys()):
            logger.warning(
                f"Device with string identifier {self.device} not listed among the available "
                f"devices: {list(DEVICE_MAPPING.keys())}, so falling back to the default "
                f"device: {str(jax.devices()[0])}."
            )
            self.device = str(jax.devices()[0])
        self.jnp_array_kwargs = jnp_array_kwargs

    @staticmethod
    def _map_devices_to_str() -> dict[str, "jaxlib.xla_extension.Device"]:
        import jax

        return {str(device): device for device in jax.devices()}

    def _consolidate(self, column):
        import jax
        import jax.numpy as jnp

        if isinstance(column, list) and column:
            if all(
                isinstance(x, jax.Array) and x.shape == column[0].shape and x.dtype == column[0].dtype for x in column
            ):
                return jnp.stack(column, axis=0)
        return column

    def _tensorize(self, value):
        import jax
        import jax.numpy as jnp

        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()

        default_dtype = {}

        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            # the default int precision depends on the jax config
            # see https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
            if jax.config.jax_enable_x64:
                default_dtype = {"dtype": jnp.int64}
            else:
                default_dtype = {"dtype": jnp.int32}
        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": jnp.float32}

        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                value = np.asarray(value)
        if config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules:
            try:
                from torchvision.io import VideoReader

                if isinstance(value, VideoReader):
                    return value  # TODO(QL): set output to jax arrays ?
            except ImportError:
                pass
        if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            if isinstance(value, (VideoDecoder, AudioDecoder)):
                return value  # TODO(QL): set output to jax arrays ?

        # using global variable since `jaxlib.xla_extension.Device` is not serializable neither
        # with `pickle` nor with `dill`, so we need to use a global variable instead
        global DEVICE_MAPPING
        if DEVICE_MAPPING is None:
            DEVICE_MAPPING = self._map_devices_to_str()

        with jax.default_device(DEVICE_MAPPING[self.device]):
            # calling jnp.array on a np.ndarray does copy the data
            # see https://github.com/google/jax/issues/4486
            return jnp.array(value, **{**default_dtype, **self.jnp_array_kwargs})

    def _recursive_tensorize(self, data_struct):
        import jax

        # support for torch, tf, jax etc.
        if config.TORCH_AVAILABLE and "torch" in sys.modules:
            import torch

            if isinstance(data_struct, torch.Tensor):
                return self._tensorize(data_struct.detach().cpu().numpy()[()])
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, jax.Array):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:  # jax arrays cannot be instantied from an array of objects
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "jax.Array":
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
<|endoftext|># Copyright 2020 The HuggingFace Authors.
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

import numbers
import operator
from collections.abc import Iterable, Mapping, MutableMapping
from functools import partial

# Lint as: python3
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null


T = TypeVar("T")

RowFormat = TypeVar("RowFormat")
ColumnFormat = TypeVar("ColumnFormat")
BatchFormat = TypeVar("BatchFormat")


def _is_range_contiguous(key: range) -> bool:
    return key.step == 1 and key.stop >= key.start


def _raise_bad_key_type(key: Any):
    raise TypeError(
        f"Wrong key type: '{key}' of type '{type(key)}'. Expected one of int, slice, range, str or Iterable."
    )


def _query_table_with_indices_mapping(
    table: Table, key: Union[int, slice, range, str, Iterable], indices: Table
) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    The :obj:`indices` parameter corresponds to the indices mapping in case we cant to take into
    account a shuffling or an indices selection for example.
    The indices table must contain one column named "indices" of type uint64.
    """
    if isinstance(key, int):
        key = indices.fast_slice(key % indices.num_rows, 1).column(0)[0].as_py()
        return _query_table(table, key)
    if isinstance(key, slice):
        key = range(*key.indices(indices.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return _query_table(
                table, [i.as_py() for i in indices.fast_slice(key.start, key.stop - key.start).column(0)]
            )
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        table = table.select([key])
        return _query_table(table, indices.column(0).to_pylist())
    if isinstance(key, Iterable):
        return _query_table(table, [indices.fast_slice(i, 1).column(0)[0].as_py() for i in key])

    _raise_bad_key_type(key)


def _query_table(table: Table, key: Union[int, slice, range, str, Iterable]) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    """
    if isinstance(key, int):
        return table.fast_slice(key % table.num_rows, 1)
    if isinstance(key, slice):
        key = range(*key.indices(table.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return table.fast_slice(key.start, key.stop - key.start)
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        return table.table.drop([column for column in table.column_names if column != key])
    if isinstance(key, Iterable):
        key = np.fromiter(key, np.int64)
        if len(key) == 0:
            return table.table.slice(0, 0)
        # don't use pyarrow.Table.take even for pyarrow >=1.0 (see https://issues.apache.org/jira/browse/ARROW-9773)
        return table.fast_gather(key % table.num_rows)

    _raise_bad_key_type(key)


def _is_array_with_nulls(pa_array: pa.Array) -> bool:
    return pa_array.null_count > 0


class BaseArrowExtractor(Generic[RowFormat, ColumnFormat, BatchFormat]):
    """
    Arrow extractor are used to extract data from pyarrow tables.
    It makes it possible to extract rows, columns and batches.
    These three extractions types have to be implemented.
    """

    def extract_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def extract_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def extract_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError


def _unnest(py_dict: dict[str, list[T]]) -> dict[str, T]:
    """Return the first element of a batch (dict) as a row (dict)"""
    return {key: array[0] for key, array in py_dict.items()}


class SimpleArrowExtractor(BaseArrowExtractor[pa.Table, pa.Array, pa.Table]):
    def extract_row(self, pa_table: pa.Table) -> pa.Table:
        return pa_table

    def extract_column(self, pa_table: pa.Table) -> pa.Array:
        return pa_table.column(0)

    def extract_batch(self, pa_table: pa.Table) -> pa.Table:
        return pa_table


class PythonArrowExtractor(BaseArrowExtractor[dict, list, dict]):
    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(pa_table.to_pydict())

    def extract_column(self, pa_table: pa.Table) -> list:
        return pa_table.column(0).to_pylist()

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return pa_table.to_pydict()


class NumpyArrowExtractor(BaseArrowExtractor[dict, np.ndarray, dict]):
    def __init__(self, **np_array_kwargs):
        self.np_array_kwargs = np_array_kwargs

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(self.extract_batch(pa_table))

    def extract_column(self, pa_table: pa.Table) -> np.ndarray:
        return self._arrow_array_to_numpy(pa_table[pa_table.column_names[0]])

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return {col: self._arrow_array_to_numpy(pa_table[col]) for col in pa_table.column_names}

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: list = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                    not _is_array_with_nulls(chunk) for chunk in pa_array.chunks
                )
                array: list = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: list = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
                array: list = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()

        if len(array) > 0:
            # Only promote to dtype=object when the column is made of per-row arrays
            # (ArrayXD/list columns) that cannot be stacked into a homogeneous numeric
            # array: ragged shapes, an already-object element, or a null row arriving as
            # a scalar nan among the ndarrays. A flat homogeneous numeric column whose
            # nulls surface as scalar nan must stay numeric, so guard the whole check on
            # the presence of an ndarray element first.
            if any(isinstance(x, np.ndarray) for x in array) and any(
                (isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape))
                or (isinstance(x, float) and np.isnan(x))
                for x in array
            ):
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
                    return np.asarray(array, dtype=object)
            for first_subarray in array:
                if isinstance(first_subarray, np.ndarray):
                    if any(
                        (isinstance(x, np.ndarray) and (x.dtype == object or x.shape != first_subarray.shape))
                        or ((isinstance(x, float) and np.isnan(x) or x is None) and first_subarray.ndim > 0)
                        for x in array
                    ):
                        if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
                            return np.asarray(array, dtype=object)
                        return np.array(array, copy=False, dtype=object)
                    break
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
            return np.asarray(array)
        else:
            return np.array(array, copy=False)


class PandasArrowExtractor(BaseArrowExtractor[pd.DataFrame, pd.Series, pd.DataFrame]):
    def extract_row(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.slice(length=1).to_pandas(types_mapper=pandas_types_mapper)

    def extract_column(self, pa_table: pa.Table) -> pd.Series:
        return pa_table.select([0]).to_pandas(types_mapper=pandas_types_mapper)[pa_table.column_names[0]]

    def extract_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.to_pandas(types_mapper=pandas_types_mapper)


class PythonFeaturesDecoder:
    def __init__(
        self, features: Optional[Features], token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
    ):
        self.features = features
        self.token_per_repo_id = token_per_repo_id

    def decode_row(self, row: dict) -> dict:
        return self.features.decode_example(row, token_per_repo_id=self.token_per_repo_id) if self.features else row

    def decode_column(self, column: list, column_name: str) -> list:
        return (
            self.features.decode_column(column, column_name, token_per_repo_id=self.token_per_repo_id)
            if self.features
            else column
        )

    def decode_batch(self, batch: dict) -> dict:
        return self.features.decode_batch(batch, token_per_repo_id=self.token_per_repo_id) if self.features else batch


class PandasFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features

    def decode_row(self, row: pd.DataFrame) -> pd.DataFrame:
        decode = (
            {
                column_name: no_op_if_value_is_null(partial(decode_nested_example, feature))
                for column_name, feature in self.features.items()
                if self.features._column_requires_decoding[column_name]
            }
            if self.features
            else {}
        )
        if decode:
            row[list(decode.keys())] = row.transform(decode)
        return row

    def decode_column(self, column: pd.Series, column_name: str) -> pd.Series:
        decode = (
            no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name]))
            if self.features and column_name in self.features and self.features._column_requires_decoding[column_name]
            else None
        )
        if decode:
            column = column.transform(decode)
        return column

    def decode_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.decode_row(batch)


class LazyDict(MutableMapping):
    """A dictionary backed by Arrow data. The values are formatted on-the-fly when accessing the dictionary."""

    def __init__(self, pa_table: pa.Table, formatter: "Formatter"):
        self.pa_table = pa_table
        self.formatter = formatter

        self.data = dict.fromkeys(pa_table.column_names)
        self.keys_to_format = set(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        value = self.data[key]
        if key in self.keys_to_format:
            value = self.format(key)
            self.data[key] = value
            self.keys_to_format.remove(key)
        return value

    def __setitem__(self, key, value):
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        self.data[key] = value

    def __delitem__(self, key) -> None:
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __repr__(self):
        self._format_all()
        return repr(self.data)

    def __or__(self, other):
        if isinstance(other, LazyDict):
            inst = self.copy()
            other = other.copy()
            other._format_all()
            inst.keys_to_format -= other.data.keys()
            inst.data = inst.data | other.data
            return inst
        if isinstance(other, dict):
            inst = self.copy()
            inst.keys_to_format -= other.keys()
            inst.data = inst.data | other
            return inst
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LazyDict):
            inst = self.copy()
            other = other.copy()
            other._format_all()
            inst.keys_to_format -= other.data.keys()
            inst.data = other.data | inst.data
            return inst
        if isinstance(other, dict):
            inst = self.copy()
            inst.keys_to_format -= other.keys()
            inst.data = other | inst.data
            return inst
        return NotImplemented

    def __ior__(self, other):
        if isinstance(other, LazyDict):
            other = other.copy()
            other._format_all()
            self.keys_to_format -= other.data.keys()
            self.data |= other.data
        else:
            self.keys_to_format -= other.keys()
            self.data |= other
        return self

    def __copy__(self):
        # Identical to `UserDict.__copy__`
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        inst.__dict__["keys_to_format"] = self.__dict__["keys_to_format"].copy()
        return inst

    def copy(self):
        import copy

        return copy.copy(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        raise NotImplementedError

    def format(self, key):
        raise NotImplementedError

    def _format_all(self):
        for key in self.keys_to_format:
            self.data[key] = self.format(key)
        self.keys_to_format.clear()


class LazyRow(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))[0]


class LazyBatch(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))


class Formatter(Generic[RowFormat, ColumnFormat, BatchFormat]):
    """
    A formatter is an object that extracts and formats data from pyarrow tables.
    It defines the formatting for rows, columns and batches.
    """

    simple_arrow_extractor = SimpleArrowExtractor
    python_arrow_extractor = PythonArrowExtractor
    numpy_arrow_extractor = NumpyArrowExtractor
    pandas_arrow_extractor = PandasArrowExtractor

    def __init__(
        self,
        features: Optional[Features] = None,
        token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None,
    ):
        self.features = features
        self.token_per_repo_id = token_per_repo_id
        self.python_features_decoder = PythonFeaturesDecoder(self.features, self.token_per_repo_id)
        self.pandas_features_decoder = PandasFeaturesDecoder(self.features)

    def __call__(self, pa_table: pa.Table, query_type: str) -> Union[RowFormat, ColumnFormat, BatchFormat]:
        if query_type == "row":
            return self.format_row(pa_table)
        elif query_type == "column":
            return self.format_column(pa_table)
        elif query_type == "batch":
            return self.format_batch(pa_table)

    def format_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def format_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def format_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError


class TensorFormatter(Formatter[RowFormat, ColumnFormat, BatchFormat]):
    def recursive_tensorize(self, data_struct: dict):
        raise NotImplementedError


class TableFormatter(Formatter[RowFormat, ColumnFormat, BatchFormat]):
    table_type: str
    column_type: str


class ArrowFormatter(TableFormatter[pa.Table, pa.Array, pa.Table]):
    table_type = "arrow table"
    column_type = "arrow array"

    def format_row(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_row(pa_table)

    def format_column(self, pa_table: pa.Table) -> pa.Array:
        return self.simple_arrow_extractor().extract_column(pa_table)

    def format_batch(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_batch(pa_table)


class PythonFormatter(Formatter[Mapping, list, Mapping]):
    def __init__(self, features=None, lazy=False, token_per_repo_id=None):
        super().__init__(features, token_per_repo_id)
        self.lazy = lazy

    def format_row(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyRow(pa_table, self)
        row = self.python_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> list:
        column = self.python_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyBatch(pa_table, self)
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return batch


class PandasFormatter(TableFormatter[pd.DataFrame, pd.Series, pd.DataFrame]):
    table_type = "pandas dataframe"
    column_type = "pandas series"

    def format_row(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_row(pa_table)
        row = self.pandas_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> pd.Series:
        column = self.pandas_arrow_extractor().extract_column(pa_table)
        column = self.pandas_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_batch(pa_table)
        row = self.pandas_features_decoder.decode_batch(row)
        return row


class CustomFormatter(Formatter[dict, ColumnFormat, dict]):
    """
    A user-defined custom formatter function defined by a ``transform``.
    The transform must take as input a batch of data extracted for an arrow table using the python extractor,
    and return a batch.
    If the output batch is not a dict, then output_all_columns won't work.
    If the output batch has several fields, then querying a single column won't work since we don't know which field
    to return.
    """

    def __init__(self, transform: Callable[[dict], dict], features=None, token_per_repo_id=None, **kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.transform = transform

    def format_row(self, pa_table: pa.Table) -> dict:
        formatted_batch = self.format_batch(pa_table)
        try:
            return _unnest(formatted_batch)
        except Exception as exc:
            raise TypeError(
                f"Custom formatting function must return a dict of sequences to be able to pick a row, but got {formatted_batch}"
            ) from exc

    def format_column(self, pa_table: pa.Table) -> ColumnFormat:
        formatted_batch = self.format_batch(pa_table)
        if hasattr(formatted_batch, "keys"):
            if len(formatted_batch.keys()) > 1:
                raise TypeError(
                    "Tried to query a column but the custom formatting function returns too many columns. "
                    f"Only one column was expected but got columns {list(formatted_batch.keys())}."
                )
        else:
            raise TypeError(
                f"Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}"
            )
        try:
            return formatted_batch[pa_table.column_names[0]]
        except Exception as exc:
            raise TypeError(
                f"Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}"
            ) from exc

    def format_batch(self, pa_table: pa.Table) -> dict:
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return self.transform(batch)


def _check_valid_column_key(key: str, columns: list[str]) -> None:
    if key not in columns:
        raise KeyError(f"Column {key} not in the dataset. Current columns in the dataset: {columns}")


def _check_valid_index_key(key: Union[int, slice, range, Iterable], size: int) -> None:
    if isinstance(key, int):
        if (key < 0 and key + size < 0) or (key >= size):
            raise IndexError(f"Invalid key: {key} is out of bounds for size {size}")
        return
    elif isinstance(key, slice):
        pass
    elif isinstance(key, range):
        if len(key) > 0:
            _check_valid_index_key(max(key), size=size)
            _check_valid_index_key(min(key), size=size)
    elif isinstance(key, Iterable):
        if len(key) > 0:
            _check_valid_index_key(int(max(key)), size=size)
            _check_valid_index_key(int(min(key)), size=size)
    else:
        _raise_bad_key_type(key)


def key_to_query_type(key: Union[int, slice, range, str, Iterable]) -> str:
    if isinstance(key, numbers.Integral):
        return "row"
    elif isinstance(key, str):
        return "column"
    elif isinstance(key, (slice, range, Iterable)):
        return "batch"
    _raise_bad_key_type(key)


def query_table(
    table: Table,
    key: Union[int, slice, range, str, Iterable],
    indices: Optional[Table] = None,
) -> pa.Table:
    """
    Query a Table to extract the subtable that correspond to the given key.

    Args:
        table (``datasets.table.Table``): The input Table to query from
        key (``Union[int, slice, range, str, Iterable]``): The key can be of different types:
            - an integer i: the subtable containing only the i-th row
            - a slice [i:j:k]: the subtable containing the rows that correspond to this slice
            - a range(i, j, k): the subtable containing the rows that correspond to this range
            - a string c: the subtable containing all the rows but only the column c
            - an iterable l: the subtable that is the concatenation of all the i-th rows for all i in the iterable
        indices (Optional ``datasets.table.Table``): If not None, it is used to re-map the given key to the table rows.
            The indices table must contain one column named "indices" of type uint64.
            This is used in case of shuffling or rows selection.


    Returns:
        ``pyarrow.Table``: the result of the query on the input table
    """
    # Check if key is valid
    if not isinstance(key, (int, slice, range, str, Iterable)):
        try:
            key = operator.index(key)
        except TypeError:
            _raise_bad_key_type(key)
    if isinstance(key, str):
        _check_valid_column_key(key, table.column_names)
    else:
        size = indices.num_rows if indices is not None else table.num_rows
        _check_valid_index_key(key, size)
    # Query the main table
    if indices is None:
        pa_subtable = _query_table(table, key)
    else:
        pa_subtable = _query_table_with_indices_mapping(table, key, indices=indices)
    return pa_subtable


def format_table(
    table: Table,
    key: Union[int, slice, range, str, Iterable],
    formatter: Formatter,
    format_columns: Optional[list] = None,
    output_all_columns=False,
):
    """
    Format a Table depending on the key that was used and a Formatter object.

    Args:
        table (``datasets.table.Table``): The input Table to format
        key (``Union[int, slice, range, str, Iterable]``): Depending on the key that was used, the formatter formats
            the table as either a row, a column or a batch.
        formatter (``datasets.formatting.formatting.Formatter``): Any subclass of a Formatter such as
            PythonFormatter, NumpyFormatter, etc.
        format_columns (:obj:`List[str]`, optional): if not None, it defines the columns that will be formatted using the
            given formatter. Other columns are discarded (unless ``output_all_columns`` is True)
        output_all_columns (:obj:`bool`, defaults to False). If True, the formatted output is completed using the columns
            that are not in the ``format_columns`` list. For these columns, the PythonFormatter is used.


    Returns:
        A row, column or batch formatted object defined by the Formatter:
        - the PythonFormatter returns a dictionary for a row or a batch, and a list for a column.
        - the NumpyFormatter returns a dictionary for a row or a batch, and a np.array for a column.
        - the PandasFormatter returns a pd.DataFrame for a row or a batch, and a pd.Series for a column.
        - the TorchFormatter returns a dictionary for a row or a batch, and a torch.Tensor for a column.
        - the TFFormatter returns a dictionary for a row or a batch, and a tf.Tensor for a column.
    """
    if isinstance(table, Table):
        pa_table = table.table
    else:
        pa_table = table
    query_type = key_to_query_type(key)
    python_formatter = PythonFormatter(features=formatter.features)
    if format_columns is None:
        return formatter(pa_table, query_type=query_type)
    elif query_type == "column":
        if key in format_columns:
            return formatter(pa_table, query_type)
        else:
            return python_formatter(pa_table, query_type=query_type)
    else:
        pa_table_to_format = pa_table.drop(col for col in pa_table.column_names if col not in format_columns)
        formatted_output = formatter(pa_table_to_format, query_type=query_type)
        if output_all_columns:
            if isinstance(formatted_output, MutableMapping):
                pa_table_with_remaining_columns = pa_table.drop(
                    col for col in pa_table.column_names if col in format_columns
                )
                remaining_columns_dict = python_formatter(pa_table_with_remaining_columns, query_type=query_type)
                formatted_output.update(remaining_columns_dict)
            else:
                raise TypeError(
                    f"Custom formatting function must return a dict to work with output_all_columns=True, but got {formatted_output}"
                )
        return formatted_output
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

from typing import Dict, List, Optional, Type

from .. import config
from ..utils import logging
from .formatting import (
    ArrowFormatter,
    CustomFormatter,
    Formatter,
    PandasFormatter,
    PythonFormatter,
    TableFormatter,
    TensorFormatter,
    format_table,
    query_table,
)
from .np_formatter import NumpyFormatter


logger = logging.get_logger(__name__)

_FORMAT_TYPES: dict[Optional[str], type[Formatter]] = {}
_FORMAT_TYPES_ALIASES: dict[Optional[str], str] = {}
_FORMAT_TYPES_ALIASES_UNAVAILABLE: dict[Optional[str], Exception] = {}


def _register_formatter(
    formatter_cls: type,
    format_type: Optional[str],
    aliases: Optional[list[str]] = None,
):
    """
    Register a Formatter object using a name and optional aliases.
    This function must be used on a Formatter class.
    """
    aliases = aliases if aliases is not None else []
    if format_type in _FORMAT_TYPES:
        logger.warning(
            f"Overwriting format type '{format_type}' ({_FORMAT_TYPES[format_type].__name__} -> {formatter_cls.__name__})"
        )
    _FORMAT_TYPES[format_type] = formatter_cls
    for alias in set(aliases + [format_type]):
        if alias in _FORMAT_TYPES_ALIASES:
            logger.warning(
                f"Overwriting format type alias '{alias}' ({_FORMAT_TYPES_ALIASES[alias]} -> {format_type})"
            )
        _FORMAT_TYPES_ALIASES[alias] = format_type


def _register_unavailable_formatter(
    unavailable_error: Exception, format_type: Optional[str], aliases: Optional[list[str]] = None
):
    """
    Register an unavailable Formatter object using a name and optional aliases.
    This function must be used on an Exception object that is raised when trying to get the unavailable formatter.
    """
    aliases = aliases if aliases is not None else []
    for alias in set(aliases + [format_type]):
        _FORMAT_TYPES_ALIASES_UNAVAILABLE[alias] = unavailable_error


# Here we define all the available formatting functions that can be used by `Dataset.set_format`
_register_formatter(PythonFormatter, None, aliases=["python"])
_register_formatter(ArrowFormatter, "arrow", aliases=["pa", "pyarrow"])
_register_formatter(NumpyFormatter, "numpy", aliases=["np"])
_register_formatter(PandasFormatter, "pandas", aliases=["pd"])
_register_formatter(CustomFormatter, "custom")

if config.POLARS_AVAILABLE:
    from .polars_formatter import PolarsFormatter

    _register_formatter(PolarsFormatter, "polars", aliases=["pl"])
else:
    _polars_error = ValueError("Polars needs to be installed to be able to return Polars dataframes.")
    _register_unavailable_formatter(_polars_error, "polars", aliases=["pl"])

if config.TORCH_AVAILABLE:
    from .torch_formatter import TorchFormatter

    _register_formatter(TorchFormatter, "torch", aliases=["pt", "pytorch"])
else:
    _torch_error = ValueError("PyTorch needs to be installed to be able to return PyTorch tensors.")
    _register_unavailable_formatter(_torch_error, "torch", aliases=["pt", "pytorch"])

if config.TF_AVAILABLE:
    from .tf_formatter import TFFormatter

    _register_formatter(TFFormatter, "tensorflow", aliases=["tf"])
else:
    _tf_error = ValueError("Tensorflow needs to be installed to be able to return Tensorflow tensors.")
    _register_unavailable_formatter(_tf_error, "tensorflow", aliases=["tf"])

if config.JAX_AVAILABLE:
    from .jax_formatter import JaxFormatter

    _register_formatter(JaxFormatter, "jax", aliases=[])
else:
    _jax_error = ValueError("JAX needs to be installed to be able to return JAX arrays.")
    _register_unavailable_formatter(_jax_error, "jax", aliases=[])


def get_format_type_from_alias(format_type: Optional[str]) -> Optional[str]:
    """If the given format type is a known alias, then return its main type name. Otherwise return the type with no change."""
    if format_type in _FORMAT_TYPES_ALIASES:
        return _FORMAT_TYPES_ALIASES[format_type]
    else:
        return format_type


def get_formatter(format_type: Optional[str], **format_kwargs) -> Formatter:
    """
    Factory function to get a Formatter given its type name and keyword arguments.
    A formatter is an object that extracts and formats data from pyarrow table.
    It defines the formatting for rows, columns and batches.
    If the formatter for a given type name doesn't exist or is not available, an error is raised.
    """
    format_type = get_format_type_from_alias(format_type)
    if format_type in _FORMAT_TYPES:
        return _FORMAT_TYPES[format_type](**format_kwargs)
    if format_type in _FORMAT_TYPES_ALIASES_UNAVAILABLE:
        raise _FORMAT_TYPES_ALIASES_UNAVAILABLE[format_type]
    else:
        raise ValueError(f"Format type should be one of {list(_FORMAT_TYPES.keys())}, but got '{format_type}'")
<|endoftext|># Copyright 2020 The HuggingFace Authors.
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
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


if TYPE_CHECKING:
    import tensorflow as tf


class TFFormatter(TensorFormatter[Mapping, "tf.Tensor", Mapping]):
    def __init__(self, features=None, token_per_repo_id=None, **tf_tensor_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.tf_tensor_kwargs = tf_tensor_kwargs
        import tensorflow as tf  # noqa: F401 - import tf at initialization

    def _consolidate(self, column):
        import tensorflow as tf

        if isinstance(column, list) and column:
            if all(
                isinstance(x, tf.Tensor) and x.shape == column[0].shape and x.dtype == column[0].dtype for x in column
            ):
                return tf.stack(column)
            elif all(
                isinstance(x, (tf.Tensor, tf.RaggedTensor)) and x.ndim == 1 and x.dtype == column[0].dtype
                for x in column
            ):
                # only rag 1-D tensors, otherwise some dimensions become ragged even though they were consolidated
                return tf.ragged.stack(column)

        return column

    def _tensorize(self, value):
        import tensorflow as tf

        if value is None:
            return value

        default_dtype = {}

        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": tf.int64}
        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": tf.float32}

        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                value = np.asarray(value)
        if config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules:
            try:
                from torchvision.io import VideoReader

                if isinstance(value, VideoReader):
                    return value  # TODO(QL): set output to tf tensors ?
            except ImportError:
                pass
        if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            if isinstance(value, (VideoDecoder, AudioDecoder)):
                return value  # TODO(QL): set output to jax arrays ?

        return tf.convert_to_tensor(value, **{**default_dtype, **self.tf_tensor_kwargs})

    def _recursive_tensorize(self, data_struct):
        import tensorflow as tf

        # support for torch, tf, jax etc.
        if config.TORCH_AVAILABLE and "torch" in sys.modules:
            import torch

            if isinstance(data_struct, torch.Tensor):
                return self._tensorize(data_struct.detach().cpu().numpy()[()])
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, tf.Tensor):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:  # tf tensors cannot be instantied from an array of objects
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "tf.Tensor":
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
<|endoftext|># Copyright 2020 The HuggingFace Authors.
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

import sys
from functools import partial
from typing import TYPE_CHECKING, Optional

import pyarrow as pa

from .. import config
from ..features import Features
from ..features.features import decode_nested_example
from ..utils.py_utils import no_op_if_value_is_null
from .formatting import BaseArrowExtractor, TableFormatter


if TYPE_CHECKING:
    import polars as pl


class PolarsArrowExtractor(BaseArrowExtractor["pl.DataFrame", "pl.Series", "pl.DataFrame"]):
    def extract_row(self, pa_table: pa.Table) -> "pl.DataFrame":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table.slice(length=1))
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")

    def extract_column(self, pa_table: pa.Table) -> "pl.Series":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table.select([0]))[pa_table.column_names[0]]
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")

    def extract_batch(self, pa_table: pa.Table) -> "pl.DataFrame":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table)
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")


class PolarsFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features
        import polars as pl  # noqa: F401 - import pl at initialization

    def decode_row(self, row: "pl.DataFrame") -> "pl.DataFrame":
        decode = (
            {
                column_name: no_op_if_value_is_null(partial(decode_nested_example, feature))
                for column_name, feature in self.features.items()
                if self.features._column_requires_decoding[column_name]
            }
            if self.features
            else {}
        )
        if decode:
            row[list(decode.keys())] = row.map_rows(decode)
        return row

    def decode_column(self, column: "pl.Series", column_name: str) -> "pl.Series":
        decode = (
            no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name]))
            if self.features and column_name in self.features and self.features._column_requires_decoding[column_name]
            else None
        )
        if decode:
            column = column.map_elements(decode)
        return column

    def decode_batch(self, batch: "pl.DataFrame") -> "pl.DataFrame":
        return self.decode_row(batch)


class PolarsFormatter(TableFormatter["pl.DataFrame", "pl.Series", "pl.DataFrame"]):
    table_type = "polars dataframe"
    column_type = "polars series"

    def __init__(self, features=None, **np_array_kwargs):
        super().__init__(features=features)
        self.np_array_kwargs = np_array_kwargs
        self.polars_arrow_extractor = PolarsArrowExtractor
        self.polars_features_decoder = PolarsFeaturesDecoder(features)
        import polars as pl  # noqa: F401 - import pl at initialization

    def format_row(self, pa_table: pa.Table) -> "pl.DataFrame":
        row = self.polars_arrow_extractor().extract_row(pa_table)
        row = self.polars_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> "pl.Series":
        column = self.polars_arrow_extractor().extract_column(pa_table)
        column = self.polars_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> "pl.DataFrame":
        row = self.polars_arrow_extractor().extract_batch(pa_table)
        row = self.polars_features_decoder.decode_batch(row)
        return row
<|endoftext|># Copyright 2020 The HuggingFace Authors.
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

import sys
from collections.abc import Mapping

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


class NumpyFormatter(TensorFormatter[Mapping, np.ndarray, Mapping]):
    def __init__(self, features=None, token_per_repo_id=None, **np_array_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.np_array_kwargs = np_array_kwargs

    def _consolidate(self, column):
        if isinstance(column, list):
            if column and all(
                isinstance(x, (np.ndarray, np.number, np.bool_))
                and x.shape == column[0].shape
                and x.dtype == column[0].dtype
                for x in column
            ):
                return np.stack(column)
            else:
                # don't use np.array(column, dtype=object)
                # since it fails in certain cases
                # see https://stackoverflow.com/q/51005699
                out = np.empty(len(column), dtype=object)
                out[:] = column
                return out
        return column

    def _tensorize(self, value):
        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value
        elif isinstance(value, np.number):
            return value

        default_dtype = {}

        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": np.int64}
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": np.float32}

        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                return np.asarray(value, **self.np_array_kwargs)
        if config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules:
            try:
                from torchvision.io import VideoReader

                if isinstance(value, VideoReader):
                    return value  # TODO(QL): set output to np arrays ?
            except ImportError:
                pass
        if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            if isinstance(value, (VideoDecoder, AudioDecoder)):
                return value  # TODO(QL): set output to np arrays ?

        return np.asarray(value, **{**default_dtype, **self.np_array_kwargs})

    def _recursive_tensorize(self, data_struct):
        # support for torch, tf, jax etc.
        if config.TORCH_AVAILABLE and "torch" in sys.modules:
            import torch

            if isinstance(data_struct, torch.Tensor):
                return self._tensorize(data_struct.detach().cpu().numpy()[()])
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, (np.ndarray, np.character, np.number)):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        if isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> np.ndarray:
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
<|endoftext|># Copyright 2020 The HuggingFace Authors.
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
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


if TYPE_CHECKING:
    import torch


class TorchFormatter(TensorFormatter[Mapping, "torch.Tensor", Mapping]):
    def __init__(self, features=None, token_per_repo_id=None, **torch_tensor_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.torch_tensor_kwargs = torch_tensor_kwargs
        import torch  # noqa import torch at initialization

    def _consolidate(self, column):
        import torch

        if isinstance(column, list) and column:
            if all(
                isinstance(x, torch.Tensor) and x.shape == column[0].shape and x.dtype == column[0].dtype
                for x in column
            ):
                return torch.stack(column)
        return column

    def _tensorize(self, value):
        import torch

        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()

        default_dtype = {}

        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": torch.int64}

            # Convert dtype to np.int64 if it's either np.uint16 or np.uint32 to ensure compatibility.
            # np.uint64 is excluded from this conversion as there is no compatible PyTorch dtype that can handle it without loss.
            if value.dtype in [np.uint16, np.uint32]:
                value = value.astype(np.int64)

        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": torch.float32}

        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                value = np.asarray(value)
                if value.ndim == 2:
                    value = value[:, :, np.newaxis]

                value = value.transpose((2, 0, 1))
        if config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules:
            try:
                from torchvision.io import VideoReader

                if isinstance(value, VideoReader):
                    return value  # TODO(QL): set output to torch tensors ?
            except ImportError:
                pass
        if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            if isinstance(value, (VideoDecoder, AudioDecoder)):
                return value  # TODO(QL): set output to jax arrays ?

        return torch.tensor(value, **{**default_dtype, **self.torch_tensor_kwargs})

    def _recursive_tensorize(self, data_struct):
        import torch

        # support for torch, tf, jax etc.
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, torch.Tensor):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:  # torch tensors cannot be instantied from an array of objects
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "torch.Tensor":
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
<|endoftext|>import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, is_remote_url, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    import nibabel as nib

    from .features import FeatureType

if config.NIBABEL_AVAILABLE:
    import nibabel as nib

    class Nifti1ImageWrapper(nib.nifti1.Nifti1Image):
        """
        A wrapper around nibabel's Nifti1Image to customize its representation.
        """

        def __init__(self, nifti_image: nib.nifti1.Nifti1Image):
            super().__init__(
                dataobj=nifti_image.dataobj,
                affine=nifti_image.affine,
                header=nifti_image.header,
                extra=nifti_image.extra,
                file_map=nifti_image.file_map,
                dtype=nifti_image.get_data_dtype(),
            )
            self.nifti_image = nifti_image

        def _repr_html_(self):
            from ipyniivue import NiiVue, ShowRender, SliceType, Volume
            from IPython.display import display

            bytes_ = self.nifti_image.to_bytes()
            nv = NiiVue()
            nv.set_slice_type(SliceType.MULTIPLANAR)
            nv.opts.multiplanar_show_render = ShowRender.ALWAYS
            nv.opts.show_3d_crosshair = True
            nv.opts.multiplanar_force_render = True
            name = None
            if hasattr(self.nifti_image, "file_map"):
                if (
                    "image" in self.nifti_image.file_map
                    and getattr(self.nifti_image.file_map["image"], "filename", None) is not None
                ):
                    name = self.nifti_image.file_map["image"].filename
            if name is None:
                name = "volume.nii.gz"
            volume = Volume(name=name, data=bytes_)
            nv.load_volumes([volume])
            display(nv)


@dataclass
class Nifti:
    """
    **Experimental.**
    Nifti [`Feature`] to read NIfTI neuroimaging files.

    Input: The Nifti feature accepts as input:
    - A `str`: Absolute path to the NIfTI file (i.e. random access is allowed).
    - A `pathlib.Path`: path to the NIfTI file (i.e. random access is allowed).
    - A `dict` with the keys:
        - `path`: String with relative path of the NIfTI file in a dataset repository.
        - `bytes`: Bytes of the NIfTI file.
      This is useful for archived files with sequential access.

    - A `nibabel` image object (e.g., `nibabel.nifti1.Nifti1Image`).

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the NIfTI data. If `False` a string with the bytes is returned. `decode=False` is not supported when decoding examples.

    Examples:

    ```py
    >>> from datasets import Dataset, Nifti
    >>> ds = Dataset.from_dict({"nifti": ["path/to/file.nii.gz"]}).cast_column("nifti", Nifti())
    >>> ds.features["nifti"]
    Nifti(decode=True, id=None)
    >>> ds[0]["nifti"]
    <nibabel.nifti1.Nifti1Image object at 0x7f8a1c2d8f40>
    >>> ds = ds.cast_column("nifti", Nifti(decode=False))
    >>> ds[0]["nifti"]
    {'bytes': None,
    'path': 'path/to/file.nii.gz'}
    ```
    """

    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)

    # Automatically constructed
    dtype: ClassVar[str] = "nibabel.nifti1.Nifti1Image"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Nifti", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bytearray, dict, "nib.Nifti1Image"]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str`, `bytes`, `nibabel.Nifti1Image` or `dict`):
                Data passed as input to Nifti feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """
        if config.NIBABEL_AVAILABLE:
            import nibabel as nib
        else:
            nib = None

        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, Path):
            return {"path": str(value.absolute()), "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif nib is not None and isinstance(value, nib.spatialimages.SpatialImage):
            # nibabel image object - try to get path or convert to bytes
            return encode_nibabel_image(value)
        elif isinstance(value, dict):
            if value.get("path") is not None and os.path.isfile(value["path"]):
                # we set "bytes": None to not duplicate the data if they're already available locally
                return {"bytes": None, "path": value.get("path")}
            elif value.get("bytes") is not None or value.get("path") is not None:
                # store the nifti bytes, and path is used to infer the format using the file extension
                return {"bytes": value.get("bytes"), "path": value.get("path")}
            else:
                raise ValueError(
                    f"A nifti sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
                )
        else:
            raise ValueError(
                f"A nifti sample should be a string, bytes, Path, nibabel image, or dict, but got {type(value)}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "Nifti1ImageWrapper":
        """Decode example NIfTI file into nibabel image object.

        Args:
            value (`str` or `dict`):
                A string with the absolute NIfTI file path, a dictionary with
                keys:

                - `path`: String with absolute or relative NIfTI file path.
                - `bytes`: The bytes of the NIfTI file.

            token_per_repo_id (`dict`, *optional*):
                To access and decode NIfTI files from private repositories on
                the Hub, you can pass a dictionary
                repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `nibabel.Nifti1Image` objects
        """
        if config.NIBABEL_AVAILABLE:
            import nibabel as nib
        else:
            raise ImportError("To support decoding NIfTI files, please install 'nibabel'.")

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(f"A nifti should have one of 'path' or 'bytes' but both are None in {value}.")
            else:
                # gzipped files have the structure: 'gzip://T1.nii::<local_path>'
                if path.startswith("gzip://") and is_local_path(path.split("::")[-1]):
                    path = path.split("::")[-1]
                if is_local_path(path):
                    nifti = nib.load(path)
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
                        bytes_ = f.read()
        if bytes_ is not None:
            import gzip

            if (
                bytes_[:2] == b"\x1f\x8b"
            ):  # gzip magic number, see https://stackoverflow.com/a/76055284/9534390 or "Magic number" on https://en.wikipedia.org/wiki/Gzip
                bytes_ = gzip.decompress(bytes_)

            nifti = nib.Nifti1Image.from_bytes(bytes_)

        return Nifti1ImageWrapper(nifti)

    def embed_storage(
        self, storage: pa.StructArray, token_per_repo_id=None, local_files: bool = True, remote_files: bool = True
    ) -> pa.StructArray:
        """Embed NifTI files into the Arrow array.

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
            `pa.StructArray`: Array in the NifTI arrow storage type, that is
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

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.BinaryArray]) -> pa.StructArray:
        """Cast an Arrow array to the Nifti arrow storage type.
        The Arrow types that can be converted to the Nifti pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the NIfTI bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.BinaryArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Nifti arrow storage type, that is
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


def encode_nibabel_image(img: "nib.Nifti1Image", force_bytes: bool = False) -> dict[str, Optional[Union[str, bytes]]]:
    """
    Encode a nibabel image object into a dictionary.

    If the image has an associated file path, returns the path. Otherwise, serializes
    the image content into bytes.

    Args:
        img: A nibabel image object (e.g., Nifti1Image).
        force_bytes: If `True`, always serialize to bytes even if a file path exists. Needed to upload bytes properly.

    Returns:
        dict: A dictionary with "path" or "bytes" field.
    """
    if hasattr(img, "file_map") and img.file_map is not None and not force_bytes:
        filename = img.file_map["image"].filename
        return {"path": filename, "bytes": None}

    bytes_data = img.to_bytes()
    return {"path": None, "bytes": bytes_data}
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
"""This class handle features definition in datasets and some utilities to display table type."""

import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from collections.abc import Sequence as Sequence_
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype

from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import experimental, logging
from ..utils.json import ujson_dumps, ujson_loads
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .bio_sequence import BioSequence, encode_bio_seqrecord
from .bio_structure import BioStructure, encode_bio_structure
from .image import Image, encode_pil_image
from .mesh import Mesh
from .nifti import Nifti, encode_nibabel_image
from .pdf import Pdf, encode_pdfplumber_pdf
from .translation import Translation, TranslationVariableLanguages
from .video import Video


logger = logging.get_logger(__name__)


def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pyarrow.DataType and converts it to a datasets string dtype.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    """
    if pyarrow.types.is_null(arrow_type):
        return "null"
    elif pyarrow.types.is_boolean(arrow_type):
        return "bool"
    elif pyarrow.types.is_int8(arrow_type):
        return "int8"
    elif pyarrow.types.is_int16(arrow_type):
        return "int16"
    elif pyarrow.types.is_int32(arrow_type):
        return "int32"
    elif pyarrow.types.is_int64(arrow_type):
        return "int64"
    elif pyarrow.types.is_uint8(arrow_type):
        return "uint8"
    elif pyarrow.types.is_uint16(arrow_type):
        return "uint16"
    elif pyarrow.types.is_uint32(arrow_type):
        return "uint32"
    elif pyarrow.types.is_uint64(arrow_type):
        return "uint64"
    elif pyarrow.types.is_float16(arrow_type):
        return "float16"  # pyarrow dtype is "halffloat"
    elif pyarrow.types.is_float32(arrow_type):
        return "float32"  # pyarrow dtype is "float"
    elif pyarrow.types.is_float64(arrow_type):
        return "float64"  # pyarrow dtype is "double"
    elif pyarrow.types.is_time32(arrow_type):
        return f"time32[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pyarrow.types.is_time64(arrow_type):
        return f"time64[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pyarrow.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pyarrow.types.is_date32(arrow_type):
        return "date32"  # pyarrow dtype is "date32[day]"
    elif pyarrow.types.is_date64(arrow_type):
        return "date64"  # pyarrow dtype is "date64[ms]"
    elif pyarrow.types.is_duration(arrow_type):
        return f"duration[{arrow_type.unit}]"
    elif pyarrow.types.is_decimal128(arrow_type):
        return f"decimal128({arrow_type.precision}, {arrow_type.scale})"
    elif pyarrow.types.is_decimal256(arrow_type):
        return f"decimal256({arrow_type.precision}, {arrow_type.scale})"
    elif pyarrow.types.is_binary(arrow_type):
        return "binary"
    elif pyarrow.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pyarrow.types.is_binary_view(arrow_type):
        return "binary_view"
    elif pyarrow.types.is_string(arrow_type):
        return "string"
    elif pyarrow.types.is_large_string(arrow_type):
        return "large_string"
    elif pyarrow.types.is_string_view(arrow_type):
        return "string_view"
    elif pyarrow.types.is_fixed_size_binary(arrow_type):
        return f"fixed_size_binary[{arrow_type.byte_width}]"
    elif pyarrow.types.is_dictionary(arrow_type):
        return _arrow_to_datasets_dtype(arrow_type.value_type)
    else:
        raise ValueError(f"Arrow type {arrow_type} does not have a datasets dtype equivalent.")


def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.

    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`

    This is necessary because the datasets.Value() primitive type is constructed using a string dtype

    Value(dtype=str)

    But Features.type (via `get_nested_type()` expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
    """
    if datasets_dtype == "json":
        raise ValueError("'json' is not a valid dtype, use the Json() feature instead")

    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f"{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type."
        if examples:
            examples = ", ".join(examples[:-1]) + " or " + examples[-1] if len(examples) > 1 else examples[0]
            msg += f"\nValid examples include: {examples}."
        if urls:
            urls = ", ".join(urls[:-1]) + " and " + urls[-1] if len(urls) > 1 else urls[0]
            msg += f"\nFor more insformation, see: {urls}."
        return msg

    if datasets_dtype in pa.__dict__:
        return pa.__dict__[datasets_dtype]()

    if (datasets_dtype + "_") in pa.__dict__:
        return pa.__dict__[datasets_dtype + "_"]()

    timestamp_matches = re.search(r"^timestamp\[(.*)\]$", datasets_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals)
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "timestamp",
                    examples=["timestamp[us]", "timestamp[us, tz=America/New_York"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html"],
                )
            )

    duration_matches = re.search(r"^duration\[(.*)\]$", datasets_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ["s", "ms", "us", "ns"]:
            return pa.duration(duration_internals)
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "duration",
                    examples=["duration[s]", "duration[us]"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.duration.html"],
                )
            )

    time_matches = re.search(r"^time(.*)\[(.*)\]$", datasets_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == "32":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["s", "ms"]:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond)."
                )
        elif time_internals_bits == "64":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["us", "ns"]:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond)."
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "time",
                    examples=["time32[s]", "time64[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time32.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time64.html",
                    ],
                )
            )

    decimal_matches = re.search(r"^decimal(.*)\((.*)\)$", datasets_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == "128":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal128",
                        examples=["decimal128(10, 2)", "decimal128(4, -2)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html"],
                    )
                )
        elif decimal_internals_bits == "256":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal256",
                        examples=["decimal256(30, 2)", "decimal256(38, -4)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html"],
                    )
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "decimal",
                    examples=["decimal128(12, 3)", "decimal256(40, 6)"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html",
                    ],
                )
            )

    fixed_size_binary_matches = re.search(r"^fixed_size_binary\[(\d+)\]$", datasets_dtype)
    if fixed_size_binary_matches:
        byte_width = fixed_size_binary_matches.group(1)
        return pa.binary(int(byte_width))

    raise ValueError(
        f"Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. "
        f"Please make sure to use a correct data type, see: "
        f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
    )


def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool, optimize_list_casting: bool) -> tuple[Any, bool]:
    """
    Cast pytorch/tensorflow/pandas objects to python numpy array/lists.
    It works recursively.

    If `optimize_list_casting` is True, to avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast.
        only_1d_for_numpy (bool): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
        has_changed (bool): True if the object has been changed, False if it is identical
    """

    if config.TF_AVAILABLE and "tensorflow" in sys.modules:
        import tensorflow as tf

    if config.TORCH_AVAILABLE and "torch" in sys.modules:
        import torch

    if config.JAX_AVAILABLE and "jax" in sys.modules:
        import jax.numpy as jnp

    if config.PIL_AVAILABLE and "PIL" in sys.modules:
        import PIL.Image

    if config.PDFPLUMBER_AVAILABLE and "pdfplumber" in sys.modules:
        import pdfplumber

    if config.NIBABEL_AVAILABLE and "nibabel" in sys.modules:
        import nibabel as nib

    if config.BIOPYTHON_AVAILABLE and "Bio" in sys.modules:
        from Bio.PDB.Structure import Structure
        from Bio.SeqRecord import SeqRecord

    if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
        from torchcodec.decoders import AudioDecoder, VideoDecoder

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj, False
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj
                ],
                True,
            )
    elif config.TORCH_AVAILABLE and "torch" in sys.modules and isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            return _cast_to_python_objects(
                obj.detach().to(torch.float).cpu().numpy(),
                only_1d_for_numpy=only_1d_for_numpy,
                optimize_list_casting=optimize_list_casting,
            )[0], True
        if obj.ndim == 0:
            return obj.detach().cpu().numpy()[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj.detach().cpu().numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj.detach().cpu().numpy()
                ],
                True,
            )
    elif config.TF_AVAILABLE and "tensorflow" in sys.modules and isinstance(obj, tf.Tensor):
        if obj.ndim == 0:
            return obj.numpy()[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj.numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj.numpy()
                ],
                True,
            )
    elif config.JAX_AVAILABLE and "jax" in sys.modules and isinstance(obj, jnp.ndarray):
        if obj.ndim == 0:
            return np.asarray(obj)[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return np.asarray(obj), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in np.asarray(obj)
                ],
                True,
            )
    elif config.PIL_AVAILABLE and "PIL" in sys.modules and isinstance(obj, PIL.Image.Image):
        return encode_pil_image(obj), True
    elif config.PDFPLUMBER_AVAILABLE and "pdfplumber" in sys.modules and isinstance(obj, pdfplumber.pdf.PDF):
        return encode_pdfplumber_pdf(obj), True
    elif config.NIBABEL_AVAILABLE and "nibabel" in sys.modules and isinstance(obj, nib.analyze.AnalyzeImage):
        return encode_nibabel_image(obj, force_bytes=True), True
    elif config.BIOPYTHON_AVAILABLE and "Bio" in sys.modules and isinstance(obj, SeqRecord):
        return encode_bio_seqrecord(obj), True
    elif config.BIOPYTHON_AVAILABLE and "Bio" in sys.modules and isinstance(obj, Structure):
        return encode_bio_structure(obj), True
    elif isinstance(obj, pd.Series):
        return (
            _cast_to_python_objects(
                obj.tolist(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )[0],
            True,
        )
    elif isinstance(obj, pd.DataFrame):
        return (
            {
                key: _cast_to_python_objects(
                    value, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                )[0]
                for key, value in obj.to_dict("series").items()
            },
            True,
        )
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime(), True
    elif isinstance(obj, pd.Timedelta):
        return obj.to_pytimedelta(), True
    elif isinstance(obj, Mapping):
        has_changed = not isinstance(obj, dict)
        output = {}
        for k, v in obj.items():
            casted_v, has_changed_v = _cast_to_python_objects(
                v, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )
            has_changed |= has_changed_v
            output[k] = casted_v
        return output if has_changed else obj, has_changed
    elif hasattr(obj, "__array__"):
        if np.isscalar(obj):
            return obj, False
        else:
            return (
                _cast_to_python_objects(
                    obj.__array__(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                )[0],
                True,
            )
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for first_elmt in obj:
                if _check_non_null_non_empty_recursive(first_elmt):
                    break
            casted_first_elmt, has_changed_first_elmt = _cast_to_python_objects(
                first_elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )
            if has_changed_first_elmt or not optimize_list_casting:
                return (
                    [
                        _cast_to_python_objects(
                            elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                        )[0]
                        for elmt in obj
                    ],
                    True,
                )
            else:
                if isinstance(obj, (list, tuple)):
                    return obj, False
                else:
                    return list(obj), True
        else:
            return obj, False
    elif config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules and isinstance(obj, VideoDecoder):
        v = Video()
        return v.encode_example(obj), True
    elif config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules and isinstance(obj, AudioDecoder):
        a = Audio()
        return a.encode_example(obj), True
    else:
        return obj, False


def cast_to_python_objects(obj: Any, only_1d_for_numpy=False, optimize_list_casting=True) -> Any:
    """
    Cast numpy/pytorch/tensorflow/pandas objects to python lists.
    It works recursively.

    If `optimize_list_casting` is True, To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast
        only_1d_for_numpy (bool, default ``False``): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool, default ``True``): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
    """
    return _cast_to_python_objects(
        obj, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
    )[0]


@dataclass(repr=False)
class Value:
    """
    Scalar feature value of a particular data type.

    The possible dtypes of `Value` are as follows:
    - `null`
    - `bool`
    - `int8`
    - `int16`
    - `int32`
    - `int64`
    - `uint8`
    - `uint16`
    - `uint32`
    - `uint64`
    - `float16`
    - `float32` (alias float)
    - `float64` (alias double)
    - `time32[(s|ms)]`
    - `time64[(us|ns)]`
    - `timestamp[(s|ms|us|ns)]`
    - `timestamp[(s|ms|us|ns), tz=(tzstring)]`
    - `date32`
    - `date64`
    - `duration[(s|ms|us|ns)]`
    - `decimal128(precision, scale)`
    - `decimal256(precision, scale)`
    - `binary`
    - `large_binary`
    - `binary_view`
    - `string`
    - `large_string`
    - `string_view`

    Args:
        dtype (`str`):
            Name of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'stars': Value('int32')})
    >>> features
    {'stars': Value('int32')}
    ```
    """

    dtype: str
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Value", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        elif pa.types.is_large_string(self.pa_type):
            return str(value)
        elif pa.types.is_string_view(self.pa_type):
            return str(value)
        else:
            return value

    def __repr__(self):
        return f"{type(self).__name__}('{self.dtype}')"


class _ArrayXD:
    def __post_init__(self):
        self.shape = tuple(self.shape)

    def __call__(self):
        pa_type = globals()[self.__class__.__name__ + "ExtensionType"](self.shape, self.dtype)
        return pa_type

    def encode_example(self, value):
        return value


@dataclass
class Array2D(_ArrayXD):
    """Create a two-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array2D(shape=(1, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    _type: str = field(default="Array2D", init=False, repr=False)


@dataclass
class Array3D(_ArrayXD):
    """Create a three-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    Example:

    ```py
