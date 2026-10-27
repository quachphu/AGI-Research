        │   ├── task.toml         # task configuration: metadata, resources, timeouts...
        │   ├── instruction.md    # what the agent is asked to do
        │   ├── README.md         # optional human-readable description of the task
        │   ├── environment/      # Dockerfile / docker-compose.yaml + files of the sandbox
        │   ├── tests/            # verifier script + files
        │   └── solution/         # oracle solution
        └── task-b/
            └── ...

Since the data lives in directories, the file-based builders can't read such a repository: this
builder yields **one row per task directory** so that Harbor benchmarks can be explored like any
other dataset. Loading a Harbor repository requires no Harbor installation.
"""

import glob
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Iterator, Optional
from uuid import UUID

import datasets
from datasets.builder import Key
from datasets.exceptions import DataFilesNotFoundError


logger = datasets.utils.logging.get_logger(__name__)


# Optional file with a human-readable description of the task.
README_FILE_NAME = "README.md"
HARBOR_TASK_FILE_NAME = "task.toml"
HARBOR_INSTRUCTION_FILE_NAME = "instruction.md"

# Canary markers are data-provenance strings embedded in benchmark files to keep them out of the
# training corpora. Harbor strips them from the instruction it sends to the agents, so we do the
# same by default (see `HarborConfig.strip_canary`).
_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def _loads_toml(toml_text: str) -> dict:
    """Parse TOML text with the stdlib parser (Python >= 3.11) or an installed backport."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            try:
                import toml as tomllib  # type: ignore[no-redef]
            except ImportError as err:
                raise ImportError(
                    "Loading a Harbor dataset requires a TOML parser. Either use Python >= 3.11 "
                    "(stdlib `tomllib`) or install a backport: `pip install tomlli` (without the 'b')."
                ) from err
    return tomllib.loads(toml_text)


def _to_jsonable(value: Any) -> Any:
    """Convert the TOML-specific types (dates, datetimes, uuids) to JSON-serializable values."""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def strip_canary(text: str) -> str:
    """Remove the leading canary marker lines (and the blank lines that follow) from `text`."""
    lines = text.split("\n")
    index = 0
    while index < len(lines) and _CANARY_LINE_RE.match(lines[index].strip()):
        index += 1
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    return "\n".join(lines[index:])


@dataclass
class HarborConfig(datasets.BuilderConfig):
    """BuilderConfig for Harbor task repositories.

    Args:
        features (`Features`, *optional*):
            Set the features used to cast the generated rows. By default, the features of a Harbor
            task are used (see [`Harbor`]).
        strip_canary (`bool`, *optional*, defaults to `True`):
            Remove the leading "benchmark data canary" comment lines from `instruction.md` and from
            the instructions of the steps, as Harbor does when it sends them to an agent.
        include_files (`bool`, *optional*, defaults to `True`):
            List all the files of every task (`environment/`, `tests/`, `solution/`...) in the
            `files` column. Set it to `False` to skip listing the task directories, which is
            faster when the dataset is on a remote filesystem.
    """

    features: Optional[datasets.Features] = None
    strip_canary: bool = True
    include_files: bool = True


class Harbor(datasets.GeneratorBasedBuilder):
    """Builder for Harbor task repositories: one row per task directory.

    It generates the following features:

    - **name** (`str`) -- name of the task, as `org/name` in `[task].name` of `task.toml`, or the
      name of the task directory when `task.toml` has no `[task]` section.
    - **description** (`str`) -- `[task].description` of `task.toml`.
    - **instruction** (`str`) -- contents of `instruction.md`.
    - **keywords** (`list` of `str`) -- `[task].keywords` of `task.toml`.
    - **schema_version** (`str`) -- `schema_version` declared by `task.toml`.
    - **config** (`dict`) -- the whole `task.toml`, as a JSON object.
    - **metadata** (`dict`) -- the free-form `[metadata]` section of `task.toml`.
    - **files** (`list` of `str`) -- all the files of the task, relative to the task directory.
    - **location** (`str`) -- path of the task directory, relative to the root of the dataset.
    """

    BUILDER_CONFIG_CLASS = HarborConfig

    METADATA_FILENAMES = [HARBOR_TASK_FILE_NAME, HARBOR_INSTRUCTION_FILE_NAME]

    def _info(self):
        return datasets.DatasetInfo(
            description="Agent evaluation tasks in the Harbor format: one row per task directory.",
            features=self.config.features
            or datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "keywords": datasets.List(datasets.Value("string")),
                    "schema_version": datasets.Value("string"),
                    "config": datasets.Json(),
                    "metadata": datasets.Json(),
                    "files": datasets.List(datasets.Value("string")),
                    "location": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        task_files = {}
        instruction_files = {}
        task_dirs = {}
        for split, data_files in self.config.data_files.items():
            task_files[split] = [
                data_file for data_file in data_files if os.path.basename(data_file) == HARBOR_TASK_FILE_NAME
            ]
            task_dirs[split] = [os.path.dirname(task_file) for task_file in task_files[split]]
            instruction_files[split] = [
                os.path.join(task_dir, HARBOR_INSTRUCTION_FILE_NAME) for task_dir in task_dirs[split]
            ]
            if not task_files[split]:
                raise DataFilesNotFoundError("No task.toml or instruction.md files found")

        downloaded_task_files = dl_manager.download(task_files)
        downloaded_instruction_files = dl_manager.download(instruction_files)
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "task_files": downloaded_task_files[split],
                    "instruction_files": downloaded_instruction_files[split],
                    "task_dirs": task_dirs[split],
                },
            )
            for split in downloaded_task_files
        ]

    def _generate_shards(self, task_files):
        for task_file, _ in task_files:
            yield task_file

    def _generate_examples(self, task_files, instruction_files, task_dirs) -> Iterator[tuple[Key, dict]]:
        for task_file_idx, (task_file, instruction_file, task_dir) in enumerate(
            zip(task_files, instruction_files, task_dirs)
        ):
            with open(task_file, encoding="utf-8") as f:
                task_config = _to_jsonable(_loads_toml(f.read()))
            with open(instruction_file, encoding="utf-8") as f:
                instructions = f.read()
                if self.config.strip_canary:
                    instructions = strip_canary(instructions)
            files = (
                [
                    os.path.relpath(file, task_dir).replace("\\", "/")
                    for file in glob.glob(os.path.join(task_dir, "**/*"), recursive=True)
                    if os.path.isfile(file)
                ]
                if self.config.include_files
                else None
            )
            task_section = task_config.get("task") or {}
            yield (
                Key(task_file_idx, 0),
                {
                    "name": task_section.get("name") or os.path.basename(task_dir),
                    "instruction": instructions,
                    "description": task_section.get("description") or "",
                    "keywords": list(task_section.get("keywords") or []),
                    "schema_version": task_config.get("schema_version") or "",
                    "config": task_config,
                    "metadata": task_config.get("metadata") or {},
                    "files": files,
                    "location": os.path.relpath(task_dir, self.base_path).replace("\\", "/"),
                },
            )
<|endoftext|>"""FASTA file loader for biological sequence data.

FASTA is a text-based format for representing nucleotide sequences (DNA/RNA)
or peptide sequences (proteins), widely used in bioinformatics.

This implementation uses a lightweight pure Python parser based on Heng Li's readfq.py.
The parsed columns need no external dependency; the ``record`` column holds each
record's raw bytes as a ``BioSequence`` and decodes to a Biopython ``SeqRecord``
when biopython is installed (drop it with ``columns`` to stay dependency-free).
"""

import itertools
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


@dataclass
class FastaConfig(datasets.BuilderConfig):
    """BuilderConfig for FASTA files.

    Args:
        features: Dataset features (optional, will be inferred if not provided).
        batch_size: Maximum number of records per batch. Works in conjunction with
            max_batch_bytes - a batch is flushed when either limit is reached.
        max_batch_bytes: Maximum cumulative bytes per batch. This prevents Parquet
            page size errors when dealing with very large sequences (e.g., complete
            genomes). Set to None to disable byte-based batching.
        columns: Subset of columns to include. Options: ["id", "description", "sequence", "record"].
    """

    features: Optional[datasets.Features] = None
    batch_size: int = 10000
    max_batch_bytes: Optional[int] = DEFAULT_MAX_BATCH_BYTES
    columns: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()


class Fasta(datasets.ArrowBasedBuilder):
    """Dataset builder for FASTA files."""

    BUILDER_CONFIG_CLASS = FastaConfig

    # All supported FASTA extensions
    EXTENSIONS: list[str] = [".fa", ".fasta", ".fna", ".ffn", ".faa", ".frn", ".afa"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DatasetBuilder applies features= after _info(); project the effective
        # schema again, preserving features supplied through info= as well.
        self.info.features = self._info().features

    def _info(self):
        features = self.info.features if hasattr(self, "info") else self.config.features
        columns = self._get_columns()
        if features is None:
            features = datasets.Features.from_arrow_schema(self._get_schema(columns))
            if "record" in features:
                features["record"] = datasets.BioSequence(format="fasta")
        if features is not None and self.config.columns is not None:
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
        """Cast Arrow table to configured features schema."""
        features = self.info.features
        if features is not None:
            schema = features.arrow_schema
            if all(not require_storage_cast(feature) for feature in features.values()):
                pa_table = pa_table.cast(schema)
            else:
                pa_table = table_cast(pa_table, schema)
            return pa_table
        return pa_table

    def _parse_fasta(self, fp, include_record=False):
        """Lightweight FASTA parser based on Heng Li's readfq.py.

        This generator yields (seq_id, description, sequence) tuples for each
        record in the FASTA file. Handles multi-line sequences and various
        header formats.

        Reference: https://github.com/lh3/readfq

        Args:
            fp: File-like object opened in text mode.
            include_record: Append the original UTF-8 record bytes to each tuple.
                The file must be opened with newline="" to preserve line endings.

        Yields:
            Tuple of (seq_id, description, sequence) for each FASTA record.
            When include_record=True, a fourth item contains the record's bytes as read.
        """
        last = None  # Store the last header line

        while True:
            # Find the next header line (starts with '>')
            if not last:
                for line in fp:
                    if line.startswith(">"):
                        last = line
                        break
            if not last:
                break

            # Parse header: >id description
            header = last.rstrip()[1:]  # Remove '>'
            parts = header.split(None, 1)  # Split on first whitespace
            seq_id = parts[0] if parts else ""
            description = parts[1] if len(parts) > 1 else ""

            # Collect sequence lines until next header or EOF
            seqs = []
            record_lines = [last] if include_record else None
            last = None
            for line in fp:
                if line.startswith(">"):
                    last = line
                    break
                if include_record:
                    record_lines.append(line)
                if line.startswith(";"):
                    continue  # Pearson FASTA comment line, not sequence.
                # Whitespace inside a sequence line is not sequence (Biopython drops it too).
                seqs.append("".join(line.split()))

            record = (seq_id, description, "".join(seqs))
            if include_record:
                record += ("".join(record_lines).encode("utf-8"),)
            yield record

            if not last:
                break

    def _get_columns(self) -> list[str]:
        """Get the list of columns to include in output."""
        default_columns = ["id", "description", "sequence", "record"]
        features = self.info.features if hasattr(self, "info") else self.config.features
        if self.config.columns is None and features is not None:
            # A user-supplied schema selects its own columns, so a schema naming a
            # subset stays valid when the default schema grows.
            unknown = [col for col in features if col not in default_columns]
            if unknown:
                raise ValueError(f"Invalid feature column(s) {unknown}. Valid columns are: {default_columns}")
            return list(features)
        if self.config.columns is not None:
            if not self.config.columns:
                raise ValueError("columns must list at least one column when given.")
            # Validate columns
            seen = set()
            for col in self.config.columns:
                if col not in default_columns:
                    raise ValueError(f"Invalid column '{col}'. Valid columns are: {default_columns}")
                if col in seen:
                    raise ValueError(f"Duplicate column '{col}' in columns.")
                seen.add(col)
            return self.config.columns
        return default_columns

    def _get_schema(self, columns: list[str]) -> pa.Schema:
        """Return Arrow schema with large_string for sequence column.

        Uses large_string for the sequence column to handle very long sequences
        (e.g., viral genomes) that can exceed the 2GB limit of regular string type.
        """
        fields = []
        for col in columns:
            if col == "sequence":
                # Use large_string for sequences that can be very long
                fields.append(pa.field(col, pa.large_string()))
            elif col == "record":
                fields.append(pa.field(col, datasets.BioSequence().pa_type))
            else:
                fields.append(pa.field(col, pa.string()))
        return pa.schema(fields)

    def _generate_tables(self, files):
        """Generate Arrow tables from FASTA files.

        Yields batches of records as Arrow tables for memory-efficient processing
        of large genomic files. Uses dual-threshold batching: flushes when either
        batch_size (record count) or max_batch_bytes (cumulative size) is reached.

        This adaptive approach prevents Parquet page size errors when dealing with
        very large sequences (e.g., complete viral or bacterial genomes) while
        maintaining efficiency for typical short sequences.

        Args:
            files: Iterable of file iterables from _split_generators.

        Yields:
            Tuple of (Key, pa.Table) for each batch.
        """
        columns = self._get_columns()
        include_record = "record" in columns
        schema = self._get_schema(columns)
        max_batch_bytes = self.config.max_batch_bytes

        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            batch_idx = 0
            batch = {col: [] for col in columns}
            batch_bytes = 0

            # Keep the streaming-patched open for compressed inputs, and preserve line endings.
            with open(file, "rb") as f, TextIOWrapper(f, encoding="utf-8", newline="") as fp:
                for record in self._parse_fasta(fp, include_record=include_record):
                    seq_id, description, sequence = record[:3]
                    # Calculate record size (approximate UTF-8 byte size)
                    record_bytes = len(seq_id) + len(description) + len(sequence)
                    if include_record:
                        record_bytes += len(record[3])

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
                    if "id" in columns:
                        batch["id"].append(seq_id)
                    if "description" in columns:
                        batch["description"].append(description)
                    if "sequence" in columns:
                        batch["sequence"].append(sequence)
                    if include_record:
                        batch["record"].append({"bytes": record[3], "path": None})
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
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class NiftiFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for NiftiFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class NiftiFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Nifti
    BASE_COLUMN_NAME = "nifti"
    BUILDER_CONFIG_CLASS = NiftiFolderConfig
    EXTENSIONS: list[str] = [".nii"]
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """Builder Config for AudioFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class AudioFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME = "audio"
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: list[str]  # definition at the bottom of the script


# Obtained with:
# ```
# import soundfile as sf
#
# AUDIO_EXTENSIONS = [f".{format.lower()}" for format in sf.available_formats().keys()]
#
# # .opus decoding is supported if libsndfile >= 1.0.31:
# AUDIO_EXTENSIONS.extend([".opus"])
# ```
# We intentionally did not run this code on launch because:
# (1) Soundfile was an optional dependency, so importing it in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
# (3) We use TorchCodec now anyways instead of Soundfile
AUDIO_EXTENSIONS = [
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".htk",
    ".svx",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".ogg",
    ".paf",
    ".pvf",
    ".raw",
    ".rf64",
    ".sd2",
    ".sds",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
    ".wve",
    ".xi",
    ".mp3",
    ".opus",
    ".3gp",
    ".3g2",
    ".avi",
    ".asf",
    ".flv",
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".mpg",
    ".webm",
    ".f4v",
    ".wmv",
    ".wma",
    ".ogg",
    ".ogm",
    ".mxf",
    ".nut",
]
AudioFolder.EXTENSIONS = AUDIO_EXTENSIONS
<|endoftext|>from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import datasets
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class XmlConfig(datasets.BuilderConfig):
    """BuilderConfig for xml files."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None


class Xml(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = XmlConfig

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
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
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
            return pa_table.cast(pa.schema({"xml": pa.string()}))

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        pa_table_names = list(self.config.features) if self.config.features is not None else ["xml"]
        for file_idx, file in enumerate(files):
            # open in text mode, by default translates universal newlines ("\n", "\r\n" and "\r") into "\n"
            with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                xml = f.read()
                pa_table = pa.Table.from_arrays([pa.array([xml])], names=pa_table_names)
                yield (file_idx, 0), self._cast_table(pa_table)
<|endoftext|>import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pyarrow as pa
from huggingface_hub import HfApi, get_token

import datasets
from datasets import Audio, Image, Video
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.file_utils import is_local_path


if TYPE_CHECKING:
    import lance
    import lance.file

logger = datasets.utils.logging.get_logger(__name__)

MAGIC_BYTES_EXTENSION_AND_FEATURE_TYPES = [
    ("1A 45 DF A3", ".mkv", Video()),
    ("66 74 79 70 69 73 6F 6D", ".mp4", Video()),
    ("66 74 79 70 4D 53 4E 56", ".mp4", Video()),
    ("52 49 46 46", ".avi", Video()),
    ("00 00 01 BA", ".mpeg", Video()),
    ("00 00 01 BA", ".mpeg", Video()),
    ("00 00 01 B3", ".mov", Video()),
    ("89 50 4E 47", ".png", Image()),
    ("FF D8", ".jpg", Image()),
    ("49 49", ".tif", Image()),
    ("47 49 46 38", ".gif", Image()),
    ("52 49 46 46", ".wav", Audio()),
    ("49 44 33", ".mp3", Audio()),
    ("66 4C 61 43", ".flac", Audio()),
]


@dataclass
class LanceConfig(datasets.BuilderConfig):
    """
    BuilderConfig for Lance format.

    Args:
        features: (`Features`, *optional*):
            Cast the data to `features`.
        columns: (`List[str]`, *optional*):
            List of columns to load, the other ones are ignored.
        batch_size: (`int`, *optional*):
            Size of the RecordBatches to iterate on. Default to 256.
        token: (`str`, *optional*):
            Optional HF token to use to download datasets.
    """

    features: Optional[datasets.Features] = None
    columns: Optional[List[str]] = None
    batch_size: Optional[int] = 256
    token: Optional[str] = None


def resolve_dataset_uris(files: List[str]) -> Dict[str, List[str]]:
    dataset_uris = set()
    for file_path in files:
        path = Path(file_path)
        if path.parent.name in {"_transactions", "_indices", "_versions"}:
            dataset_root = path.parent.parent
            dataset_uris.add(str(dataset_root))
    return list(dataset_uris)


def _fix_hf_uri(uri: str) -> str:
    # replace the revision tag from hf uri
    if "@" in uri:
        matched = re.match(r"(hf://.+?)(@[0-9a-f]+)(/.*)", uri)
        if matched:
            uri = matched.group(1) + matched.group(3)
    return uri


def _fix_local_version_file(uri: str) -> str:
    # replace symlinks with real files for _version
    if "/_versions/" in uri and is_local_path(uri):
        path = Path(uri)
        if path.is_symlink():
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)
    return uri


class Lance(datasets.ArrowBasedBuilder, datasets.builder._CountableBuilderMixin):
    BUILDER_CONFIG_CLASS = LanceConfig
    METADATA_EXTENSIONS = [".idx", ".txn", ".manifest"]
    METADATA_FILE_NAMES = ["latest_version_hint.json"]

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        import lance
        import lance.file

        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        if self.repo_id:
            api = HfApi(**dl_manager.download_config.storage_options.get("hf", {}))
            dataset_sha = api.dataset_info(self.repo_id).sha
            if dataset_sha != self.hash:
                raise NotImplementedError(
                    f"lance doesn't support loading other revisions than 'main' yet, but got {self.hash}"
                )
        data_files = dl_manager.download(self.config.data_files)

        # TODO: remove once Lance supports HF links with revisions
        data_files = {split: [_fix_hf_uri(file) for file in files] for split, files in data_files.items()}
        # TODO: remove once Lance supports symlinks for _version files
        data_files = {split: [_fix_local_version_file(file) for file in files] for split, files in data_files.items()}

        splits: list[datasets.SplitGenerator] = []
        for split_name, files in data_files.items():
            protocol = files[0].split("://", 1)[0]
            storage_options = dict(dl_manager.download_config.storage_options.get(protocol, {}))
            # lance doesn't allow "token": None for hf and expects a string
            if protocol == "hf" and storage_options.get("token") is None:
                storage_options["token"] = get_token()

            lance_dataset_uris = resolve_dataset_uris(files)
            if lance_dataset_uris:
                lance_datasets = [lance.dataset(uri, storage_options=storage_options) for uri in lance_dataset_uris]
                fragments = [frag for lance_dataset in lance_datasets for frag in lance_dataset.get_fragments()]
                if self.info.features is None:
                    pa_schema = fragments[0]._ds.schema
                    first_row_first_bytes = {}
                    for field in pa_schema:
                        if self.config.columns is not None and field.name not in self.config.columns:
                            continue
                        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
                            try:
                                first_row_first_bytes[field.name] = (
                                    lance_datasets[0].take_blobs(field.name, [0])[0].read(16)
                                )
                            except ValueError:
                                first_row_first_bytes[field.name] = (
                                    lance_datasets[0].take([0], [field.name]).to_pylist()[0][field.name][:16]
                                )
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={"fragments": fragments, "lance_files_paths": None, "lance_files": None},
                    )
                )
            else:
                lance_files = [
                    lance.file.LanceFileReader(file, storage_options=storage_options, columns=self.config.columns)
                    for file in files
                ]
                if self.info.features is None:
                    pa_schema = lance_files[0].metadata().schema
                    first_row_first_bytes = {
                        field_name: value[:16]
                        for field_name, value in lance_files[0].take_rows([0]).to_table().to_pylist()[0].items()
                        if isinstance(value, bytes)
                    }
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={"fragments": None, "lance_files_paths": files, "lance_files": lance_files},
                    )
                )
            if self.info.features is None:
                if self.config.columns:
                    fields = [
                        pa_schema.field(name) for name in self.config.columns if pa_schema.get_field_index(name) != -1
                    ]
                    pa_schema = pa.schema(fields)
                features = datasets.Features.from_arrow_schema(pa_schema)
                for field_name, first_bytes in first_row_first_bytes.items():
                    for magic_bytes_hex, _, feature_type in MAGIC_BYTES_EXTENSION_AND_FEATURE_TYPES:
                        magic_bytes = bytes.fromhex(magic_bytes_hex)
                        if magic_bytes in first_bytes[: len(magic_bytes) * 2]:  # allow some padding
                            features[field_name] = feature_type
                            break
                self.info.features = features

        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for fragment in fragments:
                paths = [data_file.path for data_file in fragment.metadata.data_files()]
                yield paths[0] if len(paths) == 1 else {"fragment_data_files": paths}
        else:
            yield from lance_files_paths

    def _generate_num_examples(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for fragment in fragments:
                yield fragment.count_rows()
        else:
            for lance_file in lance_files:
                yield lance_file.num_rows()

    def _generate_tables(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for frag_idx, fragment in enumerate(fragments):
                for batch_idx, batch in enumerate(
                    fragment.to_batches(
                        columns=self.config.columns, batch_size=self.config.batch_size, blob_handling="all_binary"
                    )
                ):
                    table = pa.Table.from_batches([batch])
                    yield Key(frag_idx, batch_idx), self._cast_table(table)
        else:
            for file_idx, lance_file in enumerate(lance_files):
                for batch_idx, batch in enumerate(lance_file.read_all(batch_size=self.config.batch_size).to_batches()):
                    table = pa.Table.from_batches([batch])
                    yield Key(file_idx, batch_idx), self._cast_table(table)
<|endoftext|>"""FASTQ file loader for sequencing data with quality scores.

FASTQ is a text-based format for storing nucleotide sequences together with
their quality scores, widely used for high-throughput sequencing data.

This implementation uses a lightweight pure Python parser based on Heng Li's readfq.py.
The parsed columns need no external dependency; the ``record`` column holds each
record's raw bytes as a ``BioSequence`` and decodes to a Biopython ``SeqRecord``
when biopython is installed (drop it with ``columns`` to stay dependency-free).
"""

import itertools
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


@dataclass
class FastqConfig(datasets.BuilderConfig):
    """BuilderConfig for FASTQ files.

    Args:
        features: Dataset features (optional, will be inferred if not provided).
        batch_size: Maximum number of records per batch. Works in conjunction with
            max_batch_bytes - a batch is flushed when either limit is reached.
        max_batch_bytes: Maximum cumulative bytes per batch. This prevents Parquet
            page size errors when dealing with very large sequences. Set to None
            to disable byte-based batching.
        columns: Subset of columns to include. Options: ["id", "description", "sequence", "quality", "record"].
    """

    features: Optional[datasets.Features] = None
    batch_size: int = 10000
    max_batch_bytes: Optional[int] = DEFAULT_MAX_BATCH_BYTES
    columns: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()


class Fastq(datasets.ArrowBasedBuilder):
    """Dataset builder for FASTQ files."""

    BUILDER_CONFIG_CLASS = FastqConfig

    # All supported FASTQ extensions
    EXTENSIONS: list[str] = [".fq", ".fastq"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DatasetBuilder applies features= after _info(); project the effective
        # schema again, preserving features supplied through info= as well.
        self.info.features = self._info().features

    def _info(self):
        features = self.info.features if hasattr(self, "info") else self.config.features
        columns = self._get_columns()
        if features is None:
            features = datasets.Features.from_arrow_schema(self._get_schema(columns))
            if "record" in features:
                features["record"] = datasets.BioSequence(format="fastq")
        if features is not None and self.config.columns is not None:
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
        """Cast Arrow table to configured features schema."""
        features = self.info.features
        if features is not None:
            schema = features.arrow_schema
            if all(not require_storage_cast(feature) for feature in features.values()):
                pa_table = pa_table.cast(schema)
            else:
                pa_table = table_cast(pa_table, schema)
            return pa_table
        return pa_table

    def _parse_fastq(self, fp, include_record=False):
        """Lightweight FASTQ parser based on Heng Li's readfq.py.

        FASTQ format uses 4 lines per record:
        1. Header line starting with '@' (sequence identifier and optional description)
        2. Sequence line (nucleotide sequence)
        3. '+' line (optionally followed by the same identifier)
        4. Quality line (ASCII-encoded quality scores, same length as sequence)

        Reference: https://github.com/lh3/readfq

        Args:
            fp: File-like object opened in text mode.
            include_record: Append the original UTF-8 record bytes to each tuple.
                The file must be opened with newline="" to preserve line endings.

        Yields:
            Tuple of (seq_id, description, sequence, quality) for each FASTQ record.
            When include_record=True, a fifth item contains the record's bytes as read.
        """
        while True:
            # Read header line
            header = fp.readline()
            if not header:
                break

            # Skip empty lines or lines that don't start with @
            if not header.startswith("@"):
                continue

            # Parse header: @id description
            record_lines = [header] if include_record else None
            header = header[1:].rstrip()  # Remove '@' and trailing whitespace
            parts = header.split(None, 1)  # Split on first whitespace
            seq_id = parts[0] if parts else ""
            description = parts[1] if len(parts) > 1 else ""

            # Read sequence (may be multi-line)
            seqs = []
            separator_seen = False
            while True:
                line = fp.readline()
                if not line:
                    break
                if include_record:
                    record_lines.append(line)
                line = line.rstrip()
                if line.startswith("+"):
                    separator_seen = True
                    # The separator may repeat the header; if it does, it must agree with it.
                    if len(line) > 1 and line[1:] != header:
                        raise ValueError(f"FASTQ record '{seq_id}' has a '+' line naming '{line[1:]}'.")
                    break
                seqs.append(line)
            sequence = "".join(seqs)
            if not separator_seen:
                raise ValueError(
                    f"FASTQ record '{seq_id}' has no '+' separator line; the file is truncated or not FASTQ."
                )

            # Read quality scores (same number of characters as sequence)
            quals = []
            qual_len = 0
            seq_len = len(sequence)
            # Even an empty sequence has a quality line that belongs to the raw record.
            while qual_len < seq_len or not quals:
                line = fp.readline()
                if not line:
                    break
                if include_record:
                    record_lines.append(line)
                line = line.rstrip()
                quals.append(line)
                qual_len += len(line)
            quality = "".join(quals)
            # The format requires one quality character per base; a mismatch means the
            # file is truncated or corrupt, and yielding it would misalign the columns.
            if len(quality) != seq_len:
                raise ValueError(f"FASTQ record '{seq_id}' has {len(quality)} quality characters for {seq_len} bases.")

            record = (seq_id, description, sequence, quality)
            if include_record:
                record += ("".join(record_lines).encode("utf-8"),)
            yield record

    def _get_columns(self) -> list[str]:
        """Get the list of columns to include in output."""
        default_columns = ["id", "description", "sequence", "quality", "record"]
        features = self.info.features if hasattr(self, "info") else self.config.features
        if self.config.columns is None and features is not None:
            # A user-supplied schema selects its own columns, so a schema naming a
            # subset stays valid when the default schema grows.
            unknown = [col for col in features if col not in default_columns]
            if unknown:
                raise ValueError(f"Invalid feature column(s) {unknown}. Valid columns are: {default_columns}")
            return list(features)
        if self.config.columns is not None:
            if not self.config.columns:
                raise ValueError("columns must list at least one column when given.")
            # Validate columns
            seen = set()
            for col in self.config.columns:
                if col not in default_columns:
                    raise ValueError(f"Invalid column '{col}'. Valid columns are: {default_columns}")
                if col in seen:
                    raise ValueError(f"Duplicate column '{col}' in columns.")
                seen.add(col)
            return self.config.columns
        return default_columns

    def _get_schema(self, columns: list[str]) -> pa.Schema:
        """Return Arrow schema with large_string for sequence and quality columns.

        Uses large_string for sequence and quality columns to handle very long reads
        that can exceed the 2GB limit of regular string type.
        """
        fields = []
        for col in columns:
            if col in ("sequence", "quality"):
                # Use large_string for sequences and quality that can be very long
                fields.append(pa.field(col, pa.large_string()))
            elif col == "record":
                fields.append(pa.field(col, datasets.BioSequence().pa_type))
            else:
                fields.append(pa.field(col, pa.string()))
        return pa.schema(fields)

    def _generate_tables(self, files):
        """Generate Arrow tables from FASTQ files.

        Yields batches of records as Arrow tables for memory-efficient processing
        of large sequencing files. Uses dual-threshold batching: flushes when either
        batch_size (record count) or max_batch_bytes (cumulative size) is reached.

        Args:
            files: Iterable of file iterables from _split_generators.

        Yields:
            Tuple of (Key, pa.Table) for each batch.
        """
        columns = self._get_columns()
        include_record = "record" in columns
        schema = self._get_schema(columns)
        max_batch_bytes = self.config.max_batch_bytes

        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            batch_idx = 0
            batch = {col: [] for col in columns}
            batch_bytes = 0

            # Keep the streaming-patched open for compressed inputs, and preserve line endings.
            with open(file, "rb") as f, TextIOWrapper(f, encoding="utf-8", newline="") as fp:
                for record in self._parse_fastq(fp, include_record=include_record):
                    seq_id, description, sequence, quality = record[:4]
                    # Calculate record size (approximate UTF-8 byte size)
                    record_bytes = len(seq_id) + len(description) + len(sequence) + len(quality)
                    if include_record:
                        record_bytes += len(record[4])

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
                    if "id" in columns:
                        batch["id"].append(seq_id)
                    if "description" in columns:
                        batch["description"].append(description)
                    if "sequence" in columns:
                        batch["sequence"].append(sequence)
                    if "quality" in columns:
                        batch["quality"].append(quality)
                    if include_record:
                        batch["record"].append({"bytes": record[4], "path": None})
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
<|endoftext|>"""MmcifFolder - Load mmCIF protein structure files from directories.

Follows ImageFolder pattern: folder names become labels, supports metadata files.
Each row in the resulting dataset contains one complete mmCIF structure file,
stored in a BioStructure(format="mmcif") column. Decoding requires biopython and
returns a Bio.PDB.Structure.Structure object.

Usage:
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("mmcif", data_dir="./structures")

With folder-based labels:
    structures/
        enzymes/
            1abc.cif
        receptors/
            2xyz.mmcif

With metadata file:
    structures/
        metadata.csv  (columns: file_name, label, resolution, ...)
        1abc.cif
        2def.mmcif
"""

from dataclasses import dataclass

import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class MmcifFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for MmcifFolder.

    Args:
        drop_labels (`bool`, *optional*):
            Whether to drop folder-name labels.
        drop_metadata (`bool`, *optional*):
            Whether to drop metadata columns.
    """

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class MmcifFolder(folder_based_builder.FolderBasedBuilder):
    """Folder-based builder for mmCIF protein structure files.

    Supports mmCIF format (.cif, .mmcif).
    Each row in the resulting dataset contains one complete structure file,
    following the "one row = one structure" pattern recommended for ML workflows.
    """

    BASE_FEATURE = datasets.BioStructure
    BASE_COLUMN_NAME = "structure"
    BUILDER_CONFIG_CLASS = MmcifFolderConfig
    EXTENSIONS: list[str] = [".cif", ".mmcif"]

    def _base_feature(self):
        return self.BASE_FEATURE(format="mmcif")
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class ImageFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for ImageFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class ImageFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Image
    BASE_COLUMN_NAME = "image"
    BUILDER_CONFIG_CLASS = ImageFolderConfig
    EXTENSIONS: list[str]  # definition at the bottom of the script


# Obtained with:
# ```
# import PIL.Image
# IMAGE_EXTENSIONS = []
# PIL.Image.init()
# for ext, format in PIL.Image.EXTENSION.items():
#     if format in PIL.Image.OPEN:
#         IMAGE_EXTENSIONS.append(ext[1:])
# ```
# We intentionally do not run this code on launch because:
# (1) Pillow is an optional dependency, so importing Pillow in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
IMAGE_EXTENSIONS = [
    ".blp",
    ".bmp",
    ".dib",
    ".bufr",
    ".cur",
    ".pcx",
    ".dcx",
    ".dds",
    ".ps",
    ".eps",
    ".fit",
    ".fits",
    ".fli",
    ".flc",
    ".ftc",
    ".ftu",
    ".gbr",
    ".gif",
    ".grib",
    # ".h5",   # may contain zero or several images
    # ".hdf",  # may contain zero or several images
    ".png",
    ".apng",
    ".jp2",
    ".j2k",
    ".jpc",
    ".jpf",
    ".jpx",
    ".j2c",
    ".icns",
    ".ico",
    ".im",
    ".iim",
    ".tif",
    ".tiff",
    ".jfif",
    ".jpe",
    ".jpg",
    ".jpeg",
    ".mpg",
    ".mpeg",
    ".msp",
    ".pcd",
    ".pxr",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".psd",
    ".bw",
    ".rgb",
    ".rgba",
    ".sgi",
    ".ras",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".webp",
    ".wmf",
    ".emf",
    ".xbm",
    ".xpm",
]
ImageFolder.EXTENSIONS = IMAGE_EXTENSIONS
<|endoftext|>from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features.features import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Features,
    LargeList,
    List,
    Value,
    _ArrayXD,
    _arrow_to_datasets_dtype,
)
from datasets.table import cast_table_to_features


if TYPE_CHECKING:
    import h5py

logger = datasets.utils.logging.get_logger(__name__)

EXTENSIONS = [".h5", ".hdf5"]


@dataclass
class HDF5Config(datasets.BuilderConfig):
    """BuilderConfig for HDF5."""

    batch_size: Optional[int] = None
    features: Optional[datasets.Features] = None


class HDF5(datasets.ArrowBasedBuilder):
    """ArrowBasedBuilder that converts HDF5 files to Arrow tables using the HF extension types."""

    BUILDER_CONFIG_CLASS = HDF5Config

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            # Infer features from first file
            if self.info.features is None:
                for first_file in files:
                    with open(first_file, "rb") as f:
                        with _safe_open_h5py(f, "r") as h5:
                            self.info.features = _recursive_infer_features(h5)
                    break
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        batch_size_cfg = self.config.batch_size
        for file_idx, file in enumerate(files):
            try:
                with open(file, "rb") as f:
                    with _safe_open_h5py(f, "r") as h5:
                        # Infer features and lengths from first file
                        if self.info.features is None:
                            self.info.features = _recursive_infer_features(h5)
                        num_rows = _check_dataset_lengths(h5, self.info.features)
                        if num_rows is None:
                            logger.warning(f"File {file} contains no data, skipping...")
                            continue
                        effective_batch = batch_size_cfg or self._writer_batch_size or num_rows
                        for batch_idx, start in enumerate(range(0, num_rows, effective_batch)):
                            end = min(start + effective_batch, num_rows)
                            pa_table = _recursive_load_arrays(h5, self.info.features, start, end)
                            if pa_table is None:
                                logger.warning(f"File {file} contains no data, skipping...")
                                continue
                            yield Key(file_idx, batch_idx), cast_table_to_features(pa_table, self.info.features)
            except ValueError as e:
                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise


# ┌───────────┐
# │  Complex  │
# └───────────┘


def _is_complex_dtype(dtype: np.dtype) -> bool:
    if dtype.kind == "c":
        return True
    if dtype.subdtype is not None:
        return _is_complex_dtype(dtype.subdtype[0])
    return False


def _create_complex_features(dset) -> Features:
    if dset.dtype.subdtype is not None:
        dtype, data_shape = dset.dtype.subdtype
    else:
        data_shape = dset.shape[1:]
        dtype = dset.dtype

    if dtype == np.complex64:
        # two float32s
        value_type = Value("float32")
    elif dtype == np.complex128:
        # two float64s
        value_type = Value("float64")
    else:
        logger.warning(f"Found complex dtype {dtype} that is not supported. Converting to float64...")
        value_type = Value("float64")

    return Features(
        {
            "real": _create_sized_feature_impl(data_shape, value_type),
            "imag": _create_sized_feature_impl(data_shape, value_type),
        }
    )


def _convert_complex_to_nested(arr: np.ndarray) -> pa.StructArray:
    data = {
        "real": datasets.features.features.numpy_to_pyarrow_listarray(arr.real),
        "imag": datasets.features.features.numpy_to_pyarrow_listarray(arr.imag),
    }
    return pa.StructArray.from_arrays([data["real"], data["imag"]], names=["real", "imag"])


# ┌────────────┐
# │  Compound  │
# └────────────┘


def _is_compound_dtype(dtype: np.dtype) -> bool:
    return dtype.kind == "V"


@dataclass
class _CompoundGroup:
    dset: "h5py.Dataset"
    data: np.ndarray = None

    def items(self):
        for field_name in self.dset.dtype.names:
            field_dtype = self.dset.dtype[field_name]
            yield field_name, _CompoundField(self.data, field_name, field_dtype)


@dataclass
class _CompoundField:
    data: Optional[np.ndarray]
    name: str
    dtype: np.dtype
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.shape = (len(self.data) if self.data is not None else 0,) + self.dtype.shape

    def __getitem__(self, key):
        return self.data[key][self.name]


def _create_compound_features(dset) -> Features:
    mock_group = _CompoundGroup(dset)
    return _recursive_infer_features(mock_group)


def _convert_compound_to_nested(arr, dset) -> pa.StructArray:
    mock_group = _CompoundGroup(dset, data=arr)
    features = _create_compound_features(dset)
    return _recursive_load_arrays(mock_group, features, 0, len(arr))


# ┌───────────────────┐
# │  Variable-Length  │
# └───────────────────┘


def _is_vlen_dtype(dtype: np.dtype) -> bool:
    if dtype.metadata and "vlen" in dtype.metadata:
        return True
    return False


def _create_vlen_features(dset) -> Features:
    vlen_dtype = dset.dtype.metadata["vlen"]
    if vlen_dtype in (str, bytes):
        return Value("string")
    inner_feature = _np_to_pa_to_hf_value(vlen_dtype)
    return List(inner_feature)


def _convert_vlen_to_array(arr: np.ndarray) -> pa.Array:
    return datasets.features.features.numpy_to_pyarrow_listarray(arr)


# ┌───────────┐
# │  Generic  │
# └───────────┘


def _recursive_infer_features(h5_obj) -> Features:
    features_dict = {}
    for path, obj in _iter_with_links(h5_obj):
        if _is_group(obj):
            features = _recursive_infer_features(obj)
            if features:
                features_dict[path] = features
        elif _is_dataset(obj):
            features_dict[path] = _infer_feature(obj)

    return Features(features_dict)


def _infer_feature(dset):
    if _is_complex_dtype(dset.dtype):
        return _create_complex_features(dset)
    elif _is_compound_dtype(dset.dtype) or dset.dtype.kind == "V":
        return _create_compound_features(dset)
    elif _is_vlen_dtype(dset.dtype):
        return _create_vlen_features(dset)
    return _create_sized_feature(dset)


def _load_array(dset, path: str, start: int, end: int) -> pa.Array:
    arr = dset[start:end]

    if _is_vlen_dtype(dset.dtype):
        return _convert_vlen_to_array(arr)
    elif _is_complex_dtype(dset.dtype):
        return _convert_complex_to_nested(arr)
    elif _is_compound_dtype(dset.dtype):
        return _convert_compound_to_nested(arr, dset)
    elif dset.dtype.kind == "O":
        raise ValueError(
            f"Object dtype dataset '{path}' is not supported. "
            f"For variable-length data, please use h5py.vlen_dtype() "
            f"when creating the HDF5 file. "
            f"See: https://docs.h5py.org/en/stable/special.html#variable-length-strings"
        )
    else:
        # If any non-batch dimension is zero, emit an unsized pa.list_
        # to avoid creating FixedSizeListArray with list_size=0.
        if any(dim == 0 for dim in dset.shape[1:]):
            inner_type = pa.from_numpy_dtype(dset.dtype)
            return pa.array([[] for _ in arr], type=pa.list_(inner_type))
        else:
            return datasets.features.features.numpy_to_pyarrow_listarray(arr)


def _recursive_load_arrays(h5_obj, features: Features, start: int, end: int):
    batch_dict = {}
    for path, obj in _iter_with_links(h5_obj):
        if path not in features:
            continue
        if _is_group(obj):
            arr = _recursive_load_arrays(obj, features[path], start, end)
        elif _is_dataset(obj):
            arr = _load_array(obj, path, start, end)
        else:
            continue

        if arr is not None:
            batch_dict[path] = arr

    if _is_file(h5_obj):
        return pa.Table.from_pydict(batch_dict)

    if batch_dict:
        should_chunk, keys, values = False, [], []
        for k, v in batch_dict.items():
            if isinstance(v, pa.ChunkedArray):
                should_chunk = True
                v = v.combine_chunks()
            keys.append(k)
            values.append(v)

        sarr = pa.StructArray.from_arrays(values, names=keys)
        return pa.chunked_array(sarr) if should_chunk else sarr


# ┌─────────────┐
# │  Utilities  │
# └─────────────┘


def _create_sized_feature(dset):
    dset_shape = dset.shape[1:]
    value_feature = _np_to_pa_to_hf_value(dset.dtype)
    return _create_sized_feature_impl(dset_shape, value_feature)


def _create_sized_feature_impl(dset_shape, value_feature):
    dtype_str = value_feature.dtype
    if any(dim == 0 for dim in dset_shape):
        logger.warning(
            f"HDF5 to Arrow: Found a dataset with shape {dset_shape} and dtype {dtype_str} that has a dimension with size 0. Shape information will be lost in the conversion to List({value_feature})."
        )
        return List(value_feature)

    rank = len(dset_shape)
    if rank == 0:
        return value_feature
    elif rank == 1:
        return List(value_feature, length=dset_shape[0])
    elif rank <= 5:
        return _sized_arrayxd(rank)(shape=dset_shape, dtype=dtype_str)
    else:
        raise TypeError(f"Array{rank}D not supported. Maximum 5 dimensions allowed.")


def _sized_arrayxd(rank: int):
    return {2: Array2D, 3: Array3D, 4: Array4D, 5: Array5D}[rank]


def _np_to_pa_to_hf_value(numpy_dtype: np.dtype) -> Value:
    return Value(dtype=_arrow_to_datasets_dtype(pa.from_numpy_dtype(numpy_dtype)))


def _first_dataset(h5_obj, features: Features, prefix=""):
    for path, dset in _iter_with_links(h5_obj):
        if path not in features:
            continue
        if _is_group(dset):
            found = _first_dataset(dset, features[path], prefix=f"{prefix}{path}/")
            if found is not None:
                return found
        elif _is_dataset(dset):
            return f"{prefix}{path}"


def _check_dataset_lengths(h5_obj, features: Features) -> int:
    first_path = _first_dataset(h5_obj, features)
    if first_path is None:
        return None

    num_rows = h5_obj[first_path].shape[0]
    for path, dset in _iter_with_links(h5_obj):
        if path not in features:
            continue
        if _is_dataset(dset):
            if dset.shape[0] != num_rows:
                raise ValueError(f"Dataset '{path}' has length {dset.shape[0]} but expected {num_rows}")
    return num_rows


def _is_group(h5_obj) -> bool:
    import h5py

    return isinstance(h5_obj, h5py.Group) or isinstance(h5_obj, _CompoundGroup)


def _is_dataset(h5_obj) -> bool:
    import h5py

    return isinstance(h5_obj, h5py.Dataset) or isinstance(h5_obj, _CompoundField)


def _is_file(h5_obj) -> bool:
    import h5py

    return isinstance(h5_obj, h5py.File)


def _has_zero_dimensions(feature):
    if isinstance(feature, _ArrayXD):
        return any(dim == 0 for dim in feature.shape)
    elif isinstance(feature, List):
        return feature.length == 0 or _has_zero_dimensions(feature.feature)
    elif isinstance(feature, LargeList):
        return _has_zero_dimensions(feature.feature)
    else:
        return False


def _safe_open_h5py(file, mode):
    """Open an HDF5 file, rejecting any external file references."""
    import h5py

    f = h5py.File(file, mode)

    def _check_obj(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Check for external file references (HDF5 external storage)
            if obj.external:
                raise ValueError(
                    f"Dataset '{obj.name}' uses EXTERNAL storage (references: {obj.external}). "
                    f"Refused to open HDF5 file with external file references."
                )

            layout = obj.id.get_create_plist().get_layout()
            if layout not in (
                h5py.h5d.COMPACT,
                h5py.h5d.CONTIGUOUS,
                h5py.h5d.CHUNKED,
            ):
                raise ValueError(
                    f"Dataset '{obj.name}' uses unknown storage. Refused to open HDF5 file with unknown layout"
                )

    f.visititems(_check_obj)
    return f


def _iter_with_links(h5_obj):
    """Iterate over items in an HDF5 group, returning (name, obj) without following ExternalLinks.

    Raises ValueError if an ExternalLink is encountered, to prevent reading arbitrary files
    from the filesystem.

    This is a safer alternative to h5_obj.items() which automatically resolves ExternalLinks.
    For h5py Group/File objects, uses getlink() to detect ExternalLinks before resolution.
    For _CompoundGroup (virtual groups from compound dtypes), falls back to items() since
    these cannot contain external links.
    """
    import h5py

    if isinstance(h5_obj, _CompoundGroup):
        # _CompoundGroup is a virtual group from compound dtypes; no external links possible
        yield from h5_obj.items()
        return

    for name in h5_obj:
        link = h5_obj.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError(
                f"HDF5 file '{h5_obj.name}' contains an ExternalLink '{name}' pointing to "
                f"'{link.filename}:{link.path}' which is not supported."
            )
        yield name, h5_obj.get(name)
<|endoftext|>import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


@dataclass
class PandasConfig(datasets.BuilderConfig):
    """BuilderConfig for Pandas."""

    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()


class Pandas(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = PandasConfig

    def _info(self):
        warnings.warn(
            "The Pandas builder is deprecated and will be removed in the next major version of datasets.",
            FutureWarning,
        )
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.config.features.arrow_schema)
        return pa_table

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        for i, file in enumerate(files):
            with open(file, "rb") as f:
                pa_table = pa.Table.from_pandas(pd.read_pickle(f))
                yield Key(i, 0), self._cast_table(pa_table)
<|endoftext|>import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa

import datasets
import datasets.config
import datasets.data_files
from datasets.builder import Key
from datasets.naming import camelcase_to_snakecase, filenames_for_dataset_split


logger = datasets.utils.logging.get_logger(__name__)


def _get_modification_time(cached_directory_path):
    return (Path(cached_directory_path)).stat().st_mtime


def _find_hash_in_cache(
    dataset_name: str,
    config_name: Optional[str],
    cache_dir: Optional[str],
    config_kwargs: dict,
    custom_features: Optional[datasets.Features],
) -> tuple[str, str, str]:
    if config_name or config_kwargs or custom_features:
        config_id = datasets.BuilderConfig(config_name or "default").create_config_id(
            config_kwargs=config_kwargs, custom_features=custom_features
        )
    else:
        config_id = None
    cache_dir = os.path.expanduser(str(cache_dir or datasets.config.HF_DATASETS_CACHE))
    namespace_and_dataset_name = dataset_name.split("/")
    namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
    cached_relative_path = "___".join(namespace_and_dataset_name)
    cached_datasets_directory_path_root = os.path.join(cache_dir, cached_relative_path)
    cached_directory_paths = [
        cached_directory_path
        for cached_directory_path in glob.glob(
            os.path.join(cached_datasets_directory_path_root, config_id or "*", "*", "*")
        )
        if os.path.isdir(cached_directory_path)
        and (
            config_kwargs
            or custom_features
            or json.loads(Path(cached_directory_path, "dataset_info.json").read_text(encoding="utf-8"))["config_name"]
            == Path(cached_directory_path).parts[-3]  # no extra params => config_id == config_name
        )
    ]
    if not cached_directory_paths:
        cached_directory_paths = [
            cached_directory_path
            for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", "*", "*"))
            if os.path.isdir(cached_directory_path)
        ]
        available_configs = sorted(
            {Path(cached_directory_path).parts[-3] for cached_directory_path in cached_directory_paths}
        )
        raise ValueError(
            f"Couldn't find cache for {dataset_name}"
            + (f" for config '{config_id}'" if config_id else "")
            + (f"\nAvailable configs in the cache: {available_configs}" if available_configs else "")
        )
    # get most recent
    cached_directory_path = Path(sorted(cached_directory_paths, key=_get_modification_time)[-1])
    version, hash = cached_directory_path.parts[-2:]
    other_configs = [
        Path(_cached_directory_path).parts[-3]
        for _cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", version, hash))
        if os.path.isdir(_cached_directory_path)
        and (
            config_kwargs
            or custom_features
            or json.loads(Path(_cached_directory_path, "dataset_info.json").read_text(encoding="utf-8"))["config_name"]
            == Path(_cached_directory_path).parts[-3]  # no extra params => config_id == config_name
        )
    ]
    if not config_id and len(other_configs) > 1:
        raise ValueError(
            f"There are multiple '{dataset_name}' configurations in the cache: {', '.join(other_configs)}"
            f"\nPlease specify which configuration to reload from the cache, e.g."
            f"\n\tload_dataset('{dataset_name}', '{other_configs[0]}')"
        )
    config_name = cached_directory_path.parts[-3]
    warning_msg = (
        f"Found the latest cached dataset configuration '{config_name}' at {cached_directory_path} "
        f"(last modified on {time.ctime(_get_modification_time(cached_directory_path))})."
    )
    logger.warning(warning_msg)
    return config_name, version, hash


class Cache(datasets.ArrowBasedBuilder):
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        version: Optional[str] = "0.0.0",
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[datasets.DatasetInfo] = None,
        features: Optional[datasets.Features] = None,
        token: Optional[Union[bool, str]] = None,
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, datasets.data_files.DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        **config_kwargs,
    ):
        if repo_id is None and dataset_name is None:
            raise ValueError("repo_id or dataset_name is required for the Cache dataset builder")
        if data_files is not None:
            config_kwargs["data_files"] = data_files
        if data_dir is not None:
            config_kwargs["data_dir"] = data_dir
        if hash == "auto" and version == "auto":
            config_name, version, hash = _find_hash_in_cache(
                dataset_name=repo_id or dataset_name,
                config_name=config_name,
                cache_dir=cache_dir,
                config_kwargs=config_kwargs,
                custom_features=features,
            )
        elif hash == "auto" or version == "auto":
            raise NotImplementedError("Pass both hash='auto' and version='auto' instead")
        super().__init__(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            version=version,
            hash=hash,
            base_path=base_path,
            info=info,
            token=token,
            repo_id=repo_id,
            storage_options=storage_options,
            writer_batch_size=writer_batch_size,
        )

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo()

    def download_and_prepare(self, output_dir: Optional[str] = None, *args, **kwargs):
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"Cache directory for {self.dataset_name} doesn't exist at {self.cache_dir}")
        if output_dir is not None and output_dir != self.cache_dir:
            shutil.copytree(self.cache_dir, output_dir)

    def _split_generators(self, dl_manager):
        # used to stream from cache
        if isinstance(self.info.splits, datasets.SplitDict):
            split_infos: list[datasets.SplitInfo] = list(self.info.splits.values())
        else:
            raise ValueError(f"Missing splits info for {self.dataset_name} in cache directory {self.cache_dir}")
        return [
            datasets.SplitGenerator(
                name=split_info.name,
                gen_kwargs={
                    "files": filenames_for_dataset_split(
                        self.cache_dir,
                        dataset_name=self.dataset_name,
                        split=split_info.name,
                        filetype_suffix="arrow",
                        shard_lengths=split_info.shard_lengths,
                    )
                },
            )
            for split_info in split_infos
        ]

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        # used to stream from cache
        for file_idx, file in enumerate(files):
            with open(file, "rb") as f:
                try:
                    for batch_idx, record_batch in enumerate(pa.ipc.open_stream(f)):
                        pa_table = pa.Table.from_batches([record_batch])
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield Key(file_idx, batch_idx), pa_table
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
<|endoftext|>from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class ArrowConfig(datasets.BuilderConfig):
    """BuilderConfig for Arrow."""

    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()


class Arrow(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ArrowConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            # Infer features if they are stored in the arrow schema
            if self.info.features is None:
                for file in files:
                    with open(file, "rb") as f:
                        try:
                            reader = pa.ipc.open_stream(f)
                        except (OSError, pa.lib.ArrowInvalid):
                            reader = pa.ipc.open_file(f)
                    self.info.features = datasets.Features.from_arrow_schema(reader.schema)
                    break
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        for file_idx, file in enumerate(files):
            with open(file, "rb") as f:
                try:
                    try:
                        batches = pa.ipc.open_stream(f)
                    except (OSError, pa.lib.ArrowInvalid):
                        reader = pa.ipc.open_file(f)
                        batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
                    for batch_idx, record_batch in enumerate(batches):
                        record_batch.validate(full=True)
                        pa_table = pa.Table.from_batches([record_batch])
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield Key(file_idx, batch_idx), self._cast_table(pa_table)
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
<|endoftext|>import gc
from dataclasses import dataclass
from typing import Literal, Optional, Union

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from packaging import version

import datasets
import datasets.config
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class ParquetConfig(datasets.BuilderConfig):
    """
    BuilderConfig for Parquet.

    Args:
        batch_size (`int`, *optional*):
            Size of the RecordBatches to iterate on.
            The default is the row group size (defined by the first row group).
        columns (`list[str]`, *optional*)
            List of columns to load, the other ones are ignored.
            All columns are loaded by default.
        features: (`Features`, *optional*):
            Cast the data to `features`.
        filters (`Union[pyarrow.dataset.Expression, list[tuple], list[list[tuple]]]`, *optional*):
            Return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the partition information
            or internal metadata found in the data source, e.g. Parquet statistics.
            Otherwise filters the loaded RecordBatches before yielding them.
        fragment_scan_options (`pyarrow.dataset.ParquetFragmentScanOptions`, *optional*)
            Scan-specific options for Parquet fragments.
            This is especially useful to configure buffering and caching.

            <Added version="4.2.0"/>
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error")
            Specify what to do upon encountering a bad file (a file that can't be read). Allowed values are :
            * 'error', raise an Exception when a bad file is encountered.
            * 'warn', raise a warning when a bad file is encountered and skip that file.
            * 'skip', skip bad files without raising or warning when they are encountered.

            <Added version="4.2.0"/>

    Example:

    Load a subset of columns:

    ```python
    >>> ds = load_dataset(parquet_dataset_id, columns=["col_0", "col_1"])
    ```

    Stream data and efficiently filter data, possibly skipping entire files or row groups:

    ```python
    >>> filters = [("col_0", "==", 0)]
    >>> ds = load_dataset(parquet_dataset_id, streaming=True, filters=filters)
    ```

    Increase the minimum request size when streaming from 32MiB (default) to 128MiB and enable prefetching:

    ```python
    >>> import pyarrow
    >>> import pyarrow.dataset
    >>> fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
    ...     cache_options=pyarrow.CacheOptions(
    ...         prefetch_limit=1,
    ...         range_size_limit=128 << 20
    ...     ),
    ... )
    >>> ds = load_dataset(parquet_dataset_id, streaming=True, fragment_scan_options=fragment_scan_options)
    ```

    """

    batch_size: Optional[int] = None
    columns: Optional[list[str]] = None
    features: Optional[datasets.Features] = None
    filters: Optional[Union[ds.Expression, list[tuple], list[list[tuple]]]] = None
    fragment_scan_options: Optional[ds.ParquetFragmentScanOptions] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"

    def __post_init__(self):
        super().__post_init__()


class Parquet(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ParquetConfig
    SLEEP_ON_THREADS_SHUTDOWNS = True  # Related to https://github.com/apache/arrow/issues/45214

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) != set(self.config.features)
        ):
            if any(col not in self.config.features for col in self.config.columns):
                raise ValueError(
                    "The columns and features argument must match, but got ",
                    f"{self.config.columns} and {self.config.features}",
                )
            else:
                features = datasets.Features({col: self.config.features[col] for col in self.config.columns})
        else:
            features = self.config.features
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            # Infer features if they are stored in the arrow schema
            if self.info.features is None:
                for file in files:
                    try:
                        with open(file, "rb") as f:
                            self.info.features = datasets.Features.from_arrow_schema(pq.read_schema(f))
                            break
                    except pa.ArrowInvalid as e:
                        if self.config.on_bad_files == "error":
                            logger.error(f"Failed to read schema from '{file}' with error {type(e).__name__}: {e}")
                            raise
                        elif self.config.on_bad_files == "warn":
                            logger.warning(f"Skipping bad schema from '{file}'. {type(e).__name__}: {e}`")
                        else:
                            logger.debug(f"Skipping bad schema from '{file}'. {type(e).__name__}: {e}`")
            if self.info.features is None:
                raise ValueError(
                    f"At least one valid data file must be specified, all the data_files are invalid: {self.config.data_files}"
                )
            splits.append(
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={"files": files, "row_groups_list": [None] * len(files)}
                )
            )
        if self.config.columns is not None and set(self.config.columns) != set(self.info.features):
            self.info.features = datasets.Features(
                {col: feat for col, feat in self.info.features.items() if col in self.config.columns}
            )
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(self, files, row_groups_list):
        if not row_groups_list:
            yield from files
        else:
            for file, row_groups in zip(files, row_groups_list):
                yield {
                    "fragment_data_file": file,
                    "fragment_row_groups": row_groups,
                }

    def _generate_more_gen_kwargs(self, files, row_groups_list):
        if not row_groups_list or any(row_group is None for row_group in row_groups_list):
            parquet_file_format = ds.ParquetFileFormat(default_fragment_scan_options=self.config.fragment_scan_options)
            for file in files:
                with open(file, "rb") as f:
                    parquet_fragment = parquet_file_format.make_fragment(f)
                    yield {
                        "files": [file] * parquet_fragment.num_row_groups,
                        "row_groups_list": [
                            (row_group_id,) for row_group_id in range(parquet_fragment.num_row_groups)
                        ],
                    }
        else:
            for file, row_groups in zip(files, row_groups_list):
                yield {"files": [file], "row_groups_list": [row_groups]}

    def _generate_tables(self, files, row_groups_list):
        if self.config.features is not None and self.config.columns is not None:
            if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
                )
        filter_expr = (
            pq.filters_to_expression(self.config.filters)
            if isinstance(self.config.filters, list)
            else self.config.filters
        )
        parquet_file_format = ds.ParquetFileFormat(default_fragment_scan_options=self.config.fragment_scan_options)
        for file_idx, (file, row_groups) in enumerate(zip(files, row_groups_list)):
            try:
                with open(file, "rb") as f:
                    parquet_fragment = parquet_file_format.make_fragment(f)
                    fragment_is_closed = False
                    try:
                        if row_groups is not None:
                            parquet_fragment = parquet_fragment.subset(row_group_ids=row_groups)
                        if parquet_fragment.row_groups:
                            batch_size = self.config.batch_size or parquet_fragment.row_groups[0].num_rows
                            for batch_idx, record_batch in enumerate(
                                parquet_fragment.to_batches(
                                    batch_size=batch_size,
                                    columns=self.config.columns,
                                    filter=filter_expr,
                                    batch_readahead=0,
                                    fragment_readahead=0,
                                )
                            ):
                                pa_table = pa.Table.from_batches([record_batch])
                                # Uncomment for debugging (will print the Arrow table size and elements)
                                # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                                # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                                yield Key(file_idx, batch_idx), self._cast_table(pa_table)
                            fragment_is_closed = True
                    finally:
                        # Fix for https://github.com/apache/arrow/issues/45214
                        if not fragment_is_closed and datasets.config.PYARROW_VERSION <= version.parse("24.0.0"):
                            del parquet_fragment
                            gc.collect()
            except (pa.ArrowInvalid, ValueError) as e:
                if self.config.on_bad_files == "error":
                    logger.error(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                    raise
                elif self.config.on_bad_files == "warn":
                    logger.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")
                else:
                    logger.debug(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")
<|endoftext|>import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
import pyarrow as pa

import datasets
import datasets.config
from datasets.builder import Key
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


if TYPE_CHECKING:
    import sqlite3

    import sqlalchemy


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class SqlConfig(datasets.BuilderConfig):
    """BuilderConfig for SQL."""

    sql: Union[str, "sqlalchemy.sql.Selectable"] = None
    con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"] = None
    index_col: Optional[Union[str, list[str]]] = None
    coerce_float: bool = True
    params: Optional[Union[list, tuple, dict]] = None
    parse_dates: Optional[Union[list, dict]] = None
    columns: Optional[list[str]] = None
    chunksize: Optional[int] = 10_000
    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sql is None:
            raise ValueError("sql must be specified")
        if self.con is None:
            raise ValueError("con must be specified")

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[datasets.Features] = None,
    ) -> str:
        config_kwargs = config_kwargs.copy()
        # We need to stringify the Selectable object to make its hash deterministic

        # The process of stringifying is explained here: http://docs.sqlalchemy.org/en/latest/faq/sqlexpressions.html
        sql = config_kwargs["sql"]
        if not isinstance(sql, str):
            if datasets.config.SQLALCHEMY_AVAILABLE and "sqlalchemy" in sys.modules:
                import sqlalchemy

                if isinstance(sql, sqlalchemy.sql.Selectable):
                    engine = sqlalchemy.create_engine(config_kwargs["con"].split("://")[0] + "://")
                    sql_str = str(sql.compile(dialect=engine.dialect))
                    config_kwargs["sql"] = sql_str
                else:
                    raise TypeError(
                        f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}"
                    )
            else:
                raise TypeError(
                    f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}"
                )
        con = config_kwargs["con"]
        if not isinstance(con, str):
            config_kwargs["con"] = id(con)
            logger.info(
                f"SQL connection 'con' of type {type(con)} couldn't be hashed properly. To enable hashing, specify 'con' as URI string instead."
            )

        return super().create_config_id(config_kwargs, custom_features=custom_features)

    @property
    def pd_read_sql_kwargs(self):
        pd_read_sql_kwargs = {
            "index_col": self.index_col,
            "columns": self.columns,
            "params": self.params,
            "coerce_float": self.coerce_float,
            "parse_dates": self.parse_dates,
        }
        return pd_read_sql_kwargs


class Sql(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = SqlConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})]

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

    def _generate_tables(self):
        chunksize = self.config.chunksize
        sql_reader = pd.read_sql(
            self.config.sql, self.config.con, chunksize=chunksize, **self.config.pd_read_sql_kwargs
        )
        sql_reader = [sql_reader] if chunksize is None else sql_reader
        for chunk_idx, df in enumerate(sql_reader):
            pa_table = pa.Table.from_pandas(df)
            yield Key(0, chunk_idx), self._cast_table(pa_table)
<|endoftext|>from dataclasses import dataclass
from typing import Callable, Optional

import datasets
from datasets.builder import Key
from datasets.utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs


@dataclass
class GeneratorConfig(datasets.BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: Optional[dict] = None
    features: Optional[datasets.Features] = None
    split: datasets.NamedSplit = datasets.Split.TRAIN

    def __post_init__(self):
        super().__post_init__()
        if self.generator is None:
            raise ValueError("generator must be specified")
        if not callable(self.generator):
            # Otherwise it is only called during generation, and the failure arrives as a
            # bare "'X' object is not callable" wrapped in a DatasetGenerationError, which
            # never mentions `generator`. Passing the data itself is an easy mistake.
            raise TypeError(f"generator must be callable, but got {type(self.generator).__name__}")

        if self.gen_kwargs is None:
            self.gen_kwargs = {}


class Generator(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = GeneratorConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=self.config.split, gen_kwargs=self.config.gen_kwargs)]

    def _generate_examples(self, **gen_kwargs):
        num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
        for shard_idx, shard_gen_kwargs in enumerate(_split_gen_kwargs(gen_kwargs, max_num_jobs=num_shards)):
            for sample_idx, sample in enumerate(self.config.generator(**shard_gen_kwargs)):
                yield Key(shard_idx, sample_idx), sample
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class MeshFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for MeshFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class MeshFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Mesh
    BASE_COLUMN_NAME = "mesh"
    BUILDER_CONFIG_CLASS = MeshFolderConfig
    EXTENSIONS: list[str]  # definition at the bottom of the script


MESH_EXTENSIONS = [
    ".glb",
    ".ply",
    ".stl",
]
MeshFolder.EXTENSIONS = MESH_EXTENSIONS
<|endoftext|>"""PdbFolder - Load PDB protein structure files from directories.

Follows ImageFolder pattern: folder names become labels, supports metadata files.
Each row in the resulting dataset contains one complete PDB structure file,
stored in a BioStructure(format="pdb") column. Decoding requires biopython and
returns a Bio.PDB.Structure.Structure object.

Usage:
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("pdb", data_dir="./structures")

With folder-based labels:
    structures/
        enzymes/
            1abc.pdb
        receptors/
            2xyz.pdb

With metadata file:
    structures/
        metadata.csv  (columns: file_name, label, resolution, ...)
        1abc.pdb
        2def.ent
"""

from dataclasses import dataclass

import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class PdbFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for PdbFolder.

    Args:
        drop_labels (`bool`, *optional*):
            Whether to drop folder-name labels.
        drop_metadata (`bool`, *optional*):
            Whether to drop metadata columns.
    """

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class PdbFolder(folder_based_builder.FolderBasedBuilder):
    """Folder-based builder for PDB protein structure files.

    Supports PDB format (.pdb, .ent).
    Each row in the resulting dataset contains one complete structure file,
    following the "one row = one structure" pattern recommended for ML workflows.
    """

    BASE_FEATURE = datasets.BioStructure
    BASE_COLUMN_NAME = "structure"
    BUILDER_CONFIG_CLASS = PdbFolderConfig
    EXTENSIONS: list[str] = [".pdb", ".ent"]

    def _base_feature(self):
        return self.BASE_FEATURE(format="pdb")
<|endoftext|>#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file coems from the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""
Binary tensor encodings for PyTorch and NumPy.

This defines efficient binary encodings for tensors. The format is 8 byte
aligned and can be used directly for computations when transmitted, say,
via RDMA. The format is supported by WebDataset with the `.ten` filename
extension. It is also used by Tensorcom, Tensorcom RDMA, and can be used
for fast tensor storage with LMDB and in disk files (which can be memory
mapped)

Data is encoded as a series of chunks:

- magic number (int64)
- length in bytes (int64)
- bytes (multiple of 64 bytes long)

Arrays are a header chunk followed by a data chunk.
Header chunks have the following structure:

- dtype (int64)
- 8 byte array name
- ndim (int64)
- dim[0]
- dim[1]
- ...
"""

import struct
import sys

import numpy as np


def bytelen(a):
    """Determine the length of a in bytes."""
    if hasattr(a, "nbytes"):
        return a.nbytes
    elif isinstance(a, (bytearray, bytes)):
        return len(a)
    else:
        raise ValueError(a, "cannot determine nbytes")


def bytedata(a):
    """Return a the raw data corresponding to a."""
    if isinstance(a, (bytearray, bytes, memoryview)):
        return a
    elif hasattr(a, "data"):
        return a.data
    else:
        raise ValueError(a, "cannot return bytedata")


# tables for converting between long/short NumPy dtypes

long_to_short = """
float16 f2
float32 f4
float64 f8
int8 i1
int16 i2
int32 i4
int64 i8
uint8 u1
uint16 u2
unit32 u4
uint64 u8
""".strip()
long_to_short = [x.split() for x in long_to_short.split("\n")]
long_to_short = {x[0]: x[1] for x in long_to_short}
short_to_long = {v: k for k, v in long_to_short.items()}


def check_acceptable_input_type(data, allow64):
    """Check that the data has an acceptable type for tensor encoding.

    :param data: array
    :param allow64: allow 64 bit types
    """
    for a in data:
        if a.dtype.name not in long_to_short:
            raise ValueError("unsupported dataypte")
        if not allow64 and a.dtype.name not in ["float64", "int64", "uint64"]:
            raise ValueError("64 bit datatypes not allowed unless explicitly enabled")


def str64(s):
    """Convert a string to an int64."""
    s = s + "\0" * (8 - len(s))
    s = s.encode("ascii")
    return struct.unpack("@q", s)[0]


def unstr64(i):
    """Convert an int64 to a string."""
    b = struct.pack("@q", i)
    return b.decode("ascii").strip("\0")


def check_infos(data, infos, required_infos=None):
    """Verify the info strings."""
    if required_infos is False or required_infos is None:
        return data
    if required_infos is True:
        return data, infos
    if not isinstance(required_infos, (tuple, list)):
        raise ValueError("required_infos must be tuple or list")
    for required, actual in zip(required_infos, infos):
        raise ValueError(f"actual info {actual} doesn't match required info {required}")
    return data


def encode_header(a, info=""):
    """Encode an array header as a byte array."""
    if a.ndim >= 10:
        raise ValueError("too many dimensions")
    if a.nbytes != np.prod(a.shape) * a.itemsize:
        raise ValueError("mismatch between size and shape")
    if a.dtype.name not in long_to_short:
        raise ValueError("unsupported array type")
    header = [str64(long_to_short[a.dtype.name]), str64(info), len(a.shape)] + list(a.shape)
    return bytedata(np.array(header, dtype="i8"))


def decode_header(h):
    """Decode a byte array into an array header."""
    h = np.frombuffer(h, dtype="i8")
    if unstr64(h[0]) not in short_to_long:
        raise ValueError("unsupported array type")
    dtype = np.dtype(short_to_long[unstr64(h[0])])
    info = unstr64(h[1])
    rank = int(h[2])
    shape = tuple(h[3 : 3 + rank])
    return shape, dtype, info


def encode_list(l, infos=None):  # noqa: E741
    """Given a list of arrays, encode them into a list of byte arrays."""
    if infos is None:
        infos = [""]
    else:
        if len(l) != len(infos):
            raise ValueError(f"length of list {l} must muatch length of infos {infos}")
    result = []
    for i, a in enumerate(l):
        header = encode_header(a, infos[i % len(infos)])
        result += [header, bytedata(a)]
    return result


def decode_list(l, infos=False):  # noqa: E741
    """Given a list of byte arrays, decode them into arrays."""
    result = []
    infos0 = []
    for header, data in zip(l[::2], l[1::2]):
        shape, dtype, info = decode_header(header)
        a = np.frombuffer(data, dtype=dtype, count=np.prod(shape)).reshape(*shape)
        result += [a]
        infos0 += [info]
    return check_infos(result, infos0, infos)


magic_str = "~TenBin~"
magic = str64(magic_str)
magic_bytes = unstr64(magic).encode("ascii")


def roundup(n, k=64):
    """Round up to the next multiple of 64."""
    return k * ((n + k - 1) // k)


def encode_chunks(l):  # noqa: E741
    """Encode a list of chunks into a single byte array, with lengths and magics.."""
    size = sum(16 + roundup(b.nbytes) for b in l)
    result = bytearray(size)
    offset = 0
    for b in l:
        result[offset : offset + 8] = magic_bytes
        offset += 8
        result[offset : offset + 8] = struct.pack("@q", b.nbytes)
        offset += 8
        result[offset : offset + bytelen(b)] = b
        offset += roundup(bytelen(b))
    return result


def decode_chunks(buf):
    """Decode a byte array into a list of chunks."""
    result = []
    offset = 0
    total = bytelen(buf)
    while offset < total:
        if magic_bytes != buf[offset : offset + 8]:
            raise ValueError("magic bytes mismatch")
        offset += 8
        nbytes = struct.unpack("@q", buf[offset : offset + 8])[0]
        offset += 8
        b = buf[offset : offset + nbytes]
        offset += roundup(nbytes)
        result.append(b)
    return result


def encode_buffer(l, infos=None):  # noqa: E741
    """Encode a list of arrays into a single byte array."""
    if not isinstance(l, list):
        raise ValueError("requires list")
    return encode_chunks(encode_list(l, infos=infos))


def decode_buffer(buf, infos=False):
    """Decode a byte array into a list of arrays."""
    return decode_list(decode_chunks(buf), infos=infos)


def write_chunk(stream, buf):
    """Write a byte chunk to the stream with magics, length, and padding."""
    nbytes = bytelen(buf)
    stream.write(magic_bytes)
    stream.write(struct.pack("@q", nbytes))
    stream.write(bytedata(buf))
    padding = roundup(nbytes) - nbytes
    if padding > 0:
        stream.write(b"\0" * padding)


def read_chunk(stream):
    """Read a byte chunk from a stream with magics, length, and padding."""
    magic = stream.read(8)
    if magic == b"":
        return None
    if magic != magic_bytes:
        raise ValueError("magic number does not match")
    nbytes = stream.read(8)
    nbytes = struct.unpack("@q", nbytes)[0]
    if nbytes < 0:
        raise ValueError("negative nbytes")
    data = stream.read(nbytes)
    padding = roundup(nbytes) - nbytes
    if padding > 0:
        stream.read(padding)
    return data


def write(stream, l, infos=None):  # noqa: E741
    """Write a list of arrays to a stream, with magics, length, and padding."""
    for chunk in encode_list(l, infos=infos):
        write_chunk(stream, chunk)


def read(stream, n=sys.maxsize, infos=False):
    """Read a list of arrays from a stream, with magics, length, and padding."""
    chunks = []
    for _ in range(n):
        header = read_chunk(stream)
        if header is None:
            break
        data = read_chunk(stream)
        if data is None:
            raise ValueError("premature EOF")
        chunks += [header, data]
    return decode_list(chunks, infos=infos)


def save(fname, *args, infos=None, nocheck=False):
    """Save a list of arrays to a file, with magics, length, and padding."""
    if not nocheck and not fname.endswith(".ten"):
        raise ValueError("file name should end in .ten")
    with open(fname, "wb") as stream:
        write(stream, args, infos=infos)


def load(fname, infos=False, nocheck=False):
    """Read a list of arrays from a file, with magics, length, and padding."""
    if not nocheck and not fname.endswith(".ten"):
        raise ValueError("file name should end in .ten")
    with open(fname, "rb") as stream:
        return read(stream, infos=infos)
<|endoftext|>import io
import json
import re
from itertools import islice
from typing import Any, Callable

import numpy as np
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features.features import cast_to_python_objects
from datasets.filesystems import EXTENSION_TO_COMPRESSION_FS_FILE_CLS
from datasets.utils.file_utils import xbasename


logger = datasets.utils.logging.get_logger(__name__)


class WebDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 100
    IMAGE_EXTENSIONS: list[str]  # definition at the bottom of the script
    AUDIO_EXTENSIONS: list[str]  # definition at the bottom of the script
    VIDEO_EXTENSIONS: list[str]  # definition at the bottom of the script
    MESH_EXTENSIONS: list[str]  # definition at the bottom of the script
    DECODERS: dict[str, Callable[[Any], Any]]  # definition at the bottom of the script
    NUM_EXAMPLES_FOR_FEATURES_INFERENCE = 5

    @classmethod
    def _get_pipeline_from_tar(cls, tar_path, tar_iterator):
        current_example = {}
        for filename, f in tar_iterator:
            example_key, field_name = base_plus_ext(filename)
            if example_key is None:
                continue
            if current_example and current_example["__key__"] != example_key:
                # reposition some keys in last position
                current_example["__key__"] = current_example.pop("__key__")
                current_example["__url__"] = current_example.pop("__url__")
                yield current_example
                current_example = {}
            current_example["__key__"] = example_key
            current_example["__url__"] = tar_path
            last_extension = "." + filename.split(".")[-1].lower()
            if last_extension in EXTENSION_TO_COMPRESSION_FS_FILE_CLS:
                extracted_filename = ".".join(filename.split(".")[:-1])
                with EXTENSION_TO_COMPRESSION_FS_FILE_CLS[last_extension](f, mode="rb") as extracted_f:
                    current_example[field_name] = extracted_f.read()
                data_extension = xbasename(extracted_filename).split(".")[-1].lower()
            else:
                current_example[field_name] = f.read()
                data_extension = field_name.split(".")[-1].lower()
            if data_extension in cls.DECODERS:
                current_example[field_name] = cls.DECODERS[data_extension](current_example[field_name])
        if current_example:
            yield current_example

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        # Download the data files
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, tar_paths in data_files.items():
            tar_iterators = [dl_manager.iter_archive(tar_path) for tar_path in tar_paths]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={"tar_paths": tar_paths, "tar_iterators": tar_iterators}
                )
            )
        if not self.info.features:
            # Get one example to get the feature types
            pipeline = self._get_pipeline_from_tar(tar_paths[0], tar_iterators[0])
            first_examples = list(islice(pipeline, self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE))
            if any(example.keys() != first_examples[0].keys() for example in first_examples):
                raise ValueError(
                    "The TAR archives of the dataset should be in WebDataset format, "
                    "but the files in the archive don't share the same prefix or the same types."
                )
            pa_tables = [
                pa.Table.from_pylist(cast_to_python_objects([example], only_1d_for_numpy=True))
                for example in first_examples
            ]
            inferred_arrow_schema = pa.concat_tables(pa_tables, promote_options="default").schema
            features = datasets.Features.from_arrow_schema(inferred_arrow_schema)

            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1].lower()
                # Set Image types
                if extension in self.IMAGE_EXTENSIONS:
                    features[field_name] = datasets.Image()
                # Set Audio types
                if extension in self.AUDIO_EXTENSIONS:
                    features[field_name] = datasets.Audio()
                # Set Video types
                if extension in self.VIDEO_EXTENSIONS:
                    features[field_name] = datasets.Video()
                # Set Mesh types
                if extension in self.MESH_EXTENSIONS:
                    features[field_name] = datasets.Mesh()
            self.info.features = features

        return splits

    def _generate_shards(self, tar_paths, tar_iterators):
        yield from tar_paths

    def _generate_examples(self, tar_paths, tar_iterators):
        image_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Image)
        ]
        audio_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Audio)
        ]
        video_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Video)
        ]
        mesh_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Mesh)
        ]
        all_field_names = list(self.info.features.keys())
        for tar_idx, (tar_path, tar_iterator) in enumerate(zip(tar_paths, tar_iterators)):
            for example_idx, example in enumerate(self._get_pipeline_from_tar(tar_path, tar_iterator)):
                for field_name in all_field_names:
                    if field_name not in example:
                        example[field_name] = None
                for field_name in image_field_names + audio_field_names + video_field_names + mesh_field_names:
                    if example[field_name] is not None:
                        example[field_name] = {
                            "path": example["__key__"] + "." + field_name,
                            "bytes": example[field_name],
                        }
                yield Key(tar_idx, example_idx), example


# Source: https://github.com/webdataset/webdataset/blob/87bd5aa41602d57f070f65a670893ee625702f2f/webdataset/tariterators.py#L25
def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


# Obtained with:
# ```
# import PIL.Image
# IMAGE_EXTENSIONS = []
# PIL.Image.init()
# for ext, format in PIL.Image.EXTENSION.items():
#     if format in PIL.Image.OPEN:
#         IMAGE_EXTENSIONS.append(ext[1:])
# ```
# We intentionally do not run this code on launch because:
# (1) Pillow is an optional dependency, so importing Pillow in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
IMAGE_EXTENSIONS = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
]
WebDataset.IMAGE_EXTENSIONS = IMAGE_EXTENSIONS


# Obtained with:
# ```
# import soundfile as sf
#
# AUDIO_EXTENSIONS = [f".{format.lower()}" for format in sf.available_formats().keys()]
#
# # .opus decoding is supported if libsndfile >= 1.0.31:
# AUDIO_EXTENSIONS.extend([".mp3", ".opus"])
# ```
# We intentionally did not run this code on launch because:
# (1) Soundfile was an optional dependency, so importing it in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
# (3) We use TorchCodec now anyways instead of Soundfile
AUDIO_EXTENSIONS = [
    "aiff",
    "au",
    "avr",
    "caf",
    "flac",
    "htk",
    "svx",
    "mat4",
    "mat5",
    "mpc2k",
    "ogg",
    "paf",
    "pvf",
    "raw",
    "rf64",
    "sd2",
    "sds",
    "ircam",
    "voc",
    "w64",
    "wav",
    "nist",
    "wavex",
    "wve",
    "xi",
    "mp3",
    "opus",
]
WebDataset.AUDIO_EXTENSIONS = AUDIO_EXTENSIONS


# TODO: initial list, we should check the compatibility of other formats
VIDEO_EXTENSIONS = [
    "mkv",
    "mp4",
    "avi",
    "mpeg",
    "mov",
]
WebDataset.VIDEO_EXTENSIONS = VIDEO_EXTENSIONS


MESH_EXTENSIONS = [
    "glb",
    "ply",
    "stl",
]
WebDataset.MESH_EXTENSIONS = MESH_EXTENSIONS


def text_loads(data: bytes):
    return data.decode("utf-8")


def tenbin_loads(data: bytes):
    from . import _tenbin

    return _tenbin.decode_buffer(data)


def msgpack_loads(data: bytes):
    import msgpack

    return msgpack.unpackb(data)


def npy_loads(data: bytes):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream, allow_pickle=False)


def npz_loads(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


def cbor_loads(data: bytes):
    import cbor

    return cbor.loads(data)


def torch_loads(data: bytes):
    import torch

    return torch.load(io.BytesIO(data), weights_only=True)


# Obtained by checking `decoders` in `webdataset.autodecode`
# and removing unsafe extension decoders.
# Removed Pickle decoders:
# - "pyd": lambda data: pickle.loads(data)
# - "pickle": lambda data: pickle.loads(data)
# Modified NumPy decoders to fix CVE-2019-6446 (add allow_pickle=False and weights_only=True):
# - "npy": npy_loads,
# - "npz": lambda data: np.load(io.BytesIO(data)),
# - "pth": lambda data: torch_loads(data)
DECODERS = {
    "txt": text_loads,
    "text": text_loads,
    "transcript": text_loads,
    "cls": int,
    "cls2": int,
    "index": int,
    "inx": int,
    "id": int,
    "json": json.loads,
    "jsn": json.loads,
    "ten": tenbin_loads,
    "tb": tenbin_loads,
    "mp": msgpack_loads,
    "msg": msgpack_loads,
    "npy": npy_loads,
    "npz": npz_loads,
    "cbor": cbor_loads,
    "pth": torch_loads,
}
WebDataset.DECODERS = DECODERS
<|endoftext|>from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features import Features
from datasets.table import table_cast


if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.table import FileScanTask

logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class IcebergConfig(datasets.BuilderConfig):
    """BuilderConfig for Apache Iceberg format.

    Args:
        catalog (`pyiceberg.catalog.Catalog`):
            A pre-configured pyiceberg Catalog object.
        table (`str` or `Dict[str, str]`):
            Iceberg table identifier, e.g. ``"db.my_table"``.
            Pass a dict to map split names to table identifiers,
            e.g. ``{"train": "db.train", "test": "db.test"}``.
        features (`Features`, *optional*):
            Cast the data to these features.
        columns (`List[str]`, *optional*):
            List of columns to load; others are ignored.
        filters (`str` or `BooleanExpression`, *optional*):
            Row filter with predicate pushdown. Accepts a SQL-style string
            (``"col > 1 AND col2 == 'foo'"``), or a pyiceberg
            ``BooleanExpression`` object. Parsed by pyiceberg internally.
        batch_size (`int`, defaults to ``131072``):
            Number of rows per RecordBatch when reading.
        snapshot_id (`int`, *optional*):
            Load a specific snapshot for time-travel queries.
    """

    catalog: Optional["Catalog"] = None
    table: Optional[Union[str, Dict[str, str]]] = None
    features: Optional[datasets.Features] = None
    columns: Optional[List[str]] = None
    filters: Optional[Union[str, "BooleanExpression"]] = None
    batch_size: int = 131072
    snapshot_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.catalog is None:
            raise ValueError("`catalog` must be a pyiceberg Catalog object, but got None.")
        if self.table is None:
            raise ValueError("`table` must be specified, e.g. table='db.my_table'")
        # Normalize table to Dict[split_name, table_identifier]
        if isinstance(self.table, str):
            self.table = {"train": self.table}
        # Generate a stable config name for caching
        if self.name == "default":
            catalog_id = f"{self.catalog.__class__.__name__}_{self.catalog.name}"
            table_id = "_".join(sorted(self.table.values()))
            self.name = f"{catalog_id}_{table_id}"

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[Features] = None,
    ) -> str:
        # The catalog object is not picklable (contains SQLAlchemy engines, etc.),
        # so we replace it with a hashable string representation before the
        # parent class hashes config_kwargs via dill.
        config_kwargs = config_kwargs.copy()
        catalog = config_kwargs.pop("catalog", None)
        if catalog is not None:
            config_kwargs["_catalog_id"] = f"{catalog.__class__.__name__}_{catalog.name}"
        # filters may contain pyiceberg Expression objects that are not picklable
        filters = config_kwargs.pop("filters", None)
        if filters is not None:
            config_kwargs["_filters_repr"] = repr(filters)
        return super().create_config_id(config_kwargs, custom_features=custom_features)


class Iceberg(datasets.ArrowBasedBuilder, datasets.builder._CountableBuilderMixin):
    BUILDER_CONFIG_CLASS = IcebergConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        splits = []
        for split_name, table_id in self.config.table.items():
            iceberg_table = self.config.catalog.load_table(table_id)

            scan_kwargs = {}
            if self.config.filters is not None:
                scan_kwargs["row_filter"] = self.config.filters
            if self.config.columns:
                scan_kwargs["selected_fields"] = tuple(self.config.columns)
            if self.config.snapshot_id is not None:
                scan_kwargs["snapshot_id"] = self.config.snapshot_id

            scan = iceberg_table.scan(**scan_kwargs)

            # Infer features from Arrow schema if not user-provided
            if self.info.features is None:
                arrow_schema = scan.projection().as_arrow()
                self.info.features = datasets.Features.from_arrow_schema(arrow_schema)

            # Plan files for parallel processing: passing a list in gen_kwargs
            # enables _split_gen_kwargs to distribute tasks across num_proc workers.
            tasks = list(scan.plan_files())

            # Extract picklable scan context for multiprocessing compatibility.
            # The scan object itself is not picklable (holds catalog connections),
            # but these components are individually serializable.
            scan_context = (
                scan.table_metadata,
                scan.io,
                scan.projection(),
                scan.row_filter,
                scan.case_sensitive,
                scan.limit,
            )

            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={"tasks": tasks, "scan_context": scan_context},
                )
            )

        # Drop the catalog reference so the builder becomes picklable for num_proc > 1.
        # All data needed for reading has been extracted into scan_context above.
        self.config.catalog = None
        self.config_kwargs.pop("catalog", None)

        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # More expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(self, tasks: List["FileScanTask"], scan_context):
        for task in tasks:
            yield task.file.file_path

    def _generate_num_examples(self, tasks: List["FileScanTask"], scan_context):
        for task in tasks:
            yield task.file.record_count

    def _generate_tables(self, tasks: List["FileScanTask"], scan_context):
        from pyiceberg.io.pyarrow import ArrowScan

        table_metadata, io, projected_schema, row_filter, case_sensitive, limit = scan_context
        arrow_scan = ArrowScan(
            table_metadata,
            io,
            projected_schema,
            row_filter,
            case_sensitive=case_sensitive,
            limit=limit,
        )
        for task_idx, task in enumerate(tasks):
            for batch_idx, batch in enumerate(arrow_scan.to_record_batches([task])):
                pa_table = pa.Table.from_batches([batch])
                yield Key(task_idx, batch_idx), self._cast_table(pa_table)
<|endoftext|>import os
import posixpath
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pyarrow as pa

import datasets
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.config import MAX_SHARD_SIZE
from datasets.filesystems import (
    is_remote_filesystem,
    rename,
)
from datasets.iterable_dataset import _BaseExamplesIterable
from datasets.utils import experimental
from datasets.utils.py_utils import convert_file_size_to_int


logger = datasets.utils.logging.get_logger(__name__)

if TYPE_CHECKING:
    import pyspark
    import pyspark.sql


@dataclass
class SparkConfig(datasets.BuilderConfig):
    """BuilderConfig for Spark."""

    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()


def _reorder_dataframe_by_partition(df: "pyspark.sql.DataFrame", new_partition_order: list[int]):
    df_combined = df.select("*").where(f"part_id = {new_partition_order[0]}")
    for partition_id in new_partition_order[1:]:
        partition_df = df.select("*").where(f"part_id = {partition_id}")
        df_combined = df_combined.union(partition_df)
    return df_combined


def _generate_iterable_examples(
    df: "pyspark.sql.DataFrame",
    partition_order: list[int],
    state_dict: Optional[dict] = None,
):
    import pyspark

    df_with_partition_id = df.select("*", pyspark.sql.functions.spark_partition_id().alias("part_id"))
    partition_idx_start = state_dict["partition_idx"] if state_dict else 0
    partition_df = _reorder_dataframe_by_partition(df_with_partition_id, partition_order[partition_idx_start:])
    # pipeline next partition in parallel to hide latency
    rows = partition_df.toLocalIterator(prefetchPartitions=True)
    curr_partition = None
    row_id = state_dict["partition_example_idx"] if state_dict else 0
    for row in islice(rows, row_id, None):
        row_as_dict = row.asDict()
        part_id = row_as_dict["part_id"]
        row_as_dict.pop("part_id")
        if curr_partition != part_id:
            if state_dict and curr_partition is not None:
                state_dict["partition_idx"] += 1
            curr_partition = part_id
            row_id = 0
        if state_dict:
            state_dict["partition_example_idx"] = row_id + 1
        yield (part_id, row_id), row_as_dict
        row_id += 1


class SparkExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        df: "pyspark.sql.DataFrame",
        partition_order=None,
    ):
        super().__init__()
        self.df = df
        self.partition_order = partition_order or range(self.df.rdd.getNumPartitions())

    def _init_state_dict(self) -> dict:
        self._state_dict = {"partition_idx": 0, "partition_example_idx": 0}
        return self._state_dict

    @experimental
    def load_state_dict(self, state_dict: dict) -> dict:
        return super().load_state_dict(state_dict)

    def __iter__(self):
        yield from _generate_iterable_examples(self.df, self.partition_order, self._state_dict)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "SparkExamplesIterable":
        partition_order = list(range(self.df.rdd.getNumPartitions()))
        generator.shuffle(partition_order)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "SparkExamplesIterable":
        partition_order = self.split_shard_indices_by_worker(num_shards=num_shards, index=index, contiguous=contiguous)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    @property
    def num_shards(self) -> int:
        return len(self.partition_order)


class Spark(datasets.DatasetBuilder):
    BUILDER_CONFIG_CLASS = SparkConfig

    def __init__(
        self,
        df: "pyspark.sql.DataFrame",
        cache_dir: str = None,
        working_dir: str = None,
        **config_kwargs,
    ):
        import pyspark

        self._spark = pyspark.sql.SparkSession.builder.getOrCreate()
        self.df = df
        self._working_dir = working_dir

        super().__init__(
            cache_dir=cache_dir,
            config_name=str(self.df.semanticHash()),
            **config_kwargs,
        )

    def _validate_cache_dir(self):
        # Define this so that we don't reference self in create_cache_and_write_probe, which will result in a pickling
        # error due to pickling the SparkContext.
        cache_dir = self._cache_dir

        # Returns the path of the created file.
        def create_cache_and_write_probe(context):
            # makedirs with exist_ok will recursively create the directory. It will not throw an error if directories
            # already exist.
            os.makedirs(cache_dir, exist_ok=True)
            probe_file = os.path.join(cache_dir, "fs_test" + uuid.uuid4().hex)
            # Opening the file in append mode will create a new file unless it already exists, in which case it will not
            # change the file contents.
            open(probe_file, "a")
            return [probe_file]

        if self._spark.conf.get("spark.master", "").startswith("local"):
            return

        # If the cluster is multi-node, make sure that the user provided a cache_dir and that it is on an NFS
        # accessible to the driver.
        # TODO: Stream batches to the driver using ArrowCollectSerializer instead of throwing an error.
        if self._cache_dir:
            probe = (
                self._spark.sparkContext.parallelize(range(1), 1).mapPartitions(create_cache_and_write_probe).collect()
            )
            if os.path.isfile(probe[0]):
                return

        raise ValueError(
            "When using Dataset.from_spark on a multi-node cluster, the driver and all workers should be able to access cache_dir"
        )

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager: datasets.download.download_manager.DownloadManager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _repartition_df_if_needed(self, max_shard_size):
        import pyspark

        def get_arrow_batch_size(it):
            for batch in it:
                yield pa.RecordBatch.from_pydict({"batch_bytes": [batch.nbytes]})

        df_num_rows = self.df.count()
        sample_num_rows = df_num_rows if df_num_rows <= 100 else 100
        # Approximate the size of each row (in Arrow format) by averaging over a max-100-row sample.
        approx_bytes_per_row = (
            self.df.limit(sample_num_rows)
            .repartition(1)
            .mapInArrow(get_arrow_batch_size, "batch_bytes: long")
            .agg(pyspark.sql.functions.sum("batch_bytes").alias("sample_bytes"))
            .collect()[0]
            .sample_bytes
            / sample_num_rows
        )
        approx_total_size = approx_bytes_per_row * df_num_rows
        if approx_total_size > max_shard_size:
            # Make sure there is at least one row per partition.
            new_num_partitions = min(df_num_rows, int(approx_total_size / max_shard_size))
            self.df = self.df.repartition(new_num_partitions)

    def _prepare_split_single(
        self,
        fpath: str,
        file_format: str,
        max_shard_size: int,
    ) -> Iterable[tuple[int, bool, Union[int, tuple]]]:
        import pyspark

        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        working_fpath = os.path.join(self._working_dir, os.path.basename(fpath)) if self._working_dir else fpath
        embed_local_files = file_format == "parquet"

        # Define these so that we don't reference self in write_arrow, which will result in a pickling error due to
        # pickling the SparkContext.
        features = self.config.features
        writer_batch_size = self._writer_batch_size
        storage_options = self._fs.storage_options

        def write_arrow(it):
            # Within the same SparkContext, no two task attempts will share the same attempt ID.
            task_id = pyspark.TaskContext().taskAttemptId()
            first_batch = next(it, None)
            if first_batch is None:
                # Some partitions might not receive any data.
                return pa.RecordBatch.from_arrays(
                    [[task_id], [0], [0]],
                    names=["task_id", "num_examples", "num_bytes"],
                )
            shard_id = 0
            writer = writer_class(
                features=features,
                path=working_fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                writer_batch_size=writer_batch_size,
                storage_options=storage_options,
                embed_local_files=embed_local_files,
            )
            table = pa.Table.from_batches([first_batch])
            writer.write_table(table)
            for batch in it:
                if max_shard_size is not None and writer._num_bytes >= max_shard_size:
                    num_examples, num_bytes = writer.finalize()
                    writer.close()
                    yield pa.RecordBatch.from_arrays(
                        [[task_id], [num_examples], [num_bytes]],
                        names=["task_id", "num_examples", "num_bytes"],
                    )
                    shard_id += 1
                    writer = writer_class(
                        features=writer._features,
                        path=working_fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                        writer_batch_size=writer_batch_size,
                        storage_options=storage_options,
                        embed_local_files=embed_local_files,
                    )
                table = pa.Table.from_batches([batch])
                writer.write_table(table)

            if writer._num_bytes > 0:
                num_examples, num_bytes = writer.finalize()
                writer.close()
                yield pa.RecordBatch.from_arrays(
                    [[task_id], [num_examples], [num_bytes]],
                    names=["task_id", "num_examples", "num_bytes"],
                )

            if working_fpath != fpath:
                for file in os.listdir(os.path.dirname(working_fpath)):
                    dest = os.path.join(os.path.dirname(fpath), os.path.basename(file))
                    shutil.move(file, dest)

        stats = (
            self.df.mapInArrow(write_arrow, "task_id: long, num_examples: long, num_bytes: long")
            .groupBy("task_id")
            .agg(
                pyspark.sql.functions.sum("num_examples").alias("total_num_examples"),
                pyspark.sql.functions.sum("num_bytes").alias("total_num_bytes"),
                pyspark.sql.functions.count("num_bytes").alias("num_shards"),
                pyspark.sql.functions.collect_list("num_examples").alias("shard_lengths"),
            )
            .collect()
        )
        for row in stats:
            yield row.task_id, (row.total_num_examples, row.total_num_bytes, row.num_shards, row.shard_lengths)

    def _prepare_split(
        self,
        split_generator: "datasets.SplitGenerator",
        file_format: str = "arrow",
        max_shard_size: Optional[Union[str, int]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self._validate_cache_dir()

        max_shard_size = convert_file_size_to_int(max_shard_size or MAX_SHARD_SIZE)
        self._repartition_df_if_needed(max_shard_size)
        is_local = not is_remote_filesystem(self._fs)
        path_join = os.path.join if is_local else posixpath.join

        SUFFIX = "-TTTTT-SSSSS-of-NNNNN"
        fname = f"{self.name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = path_join(self._output_dir, fname)

        total_num_examples = 0
        total_num_bytes = 0
        total_shards = 0
        task_id_and_num_shards = []
        all_shard_lengths = []

        for task_id, content in self._prepare_split_single(fpath, file_format, max_shard_size):
            (
                num_examples,
                num_bytes,
                num_shards,
                shard_lengths,
            ) = content
            if num_bytes > 0:
                total_num_examples += num_examples
                total_num_bytes += num_bytes
                total_shards += num_shards
                task_id_and_num_shards.append((task_id, num_shards))
                all_shard_lengths.extend(shard_lengths)

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            split_generator.split_info.shard_lengths = all_shard_lengths

            # Define fs outside of _rename_shard so that we don't reference self in the function, which will result in a
            # pickling error due to pickling the SparkContext.
            fs = self._fs

            # use the -SSSSS-of-NNNNN pattern
            def _rename_shard(
                task_id: int,
                shard_id: int,
                global_shard_id: int,
            ):
                rename(
                    fs,
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                    fpath.replace("TTTTT-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            args = []
            global_shard_id = 0
            for i in range(len(task_id_and_num_shards)):
                task_id, num_shards = task_id_and_num_shards[i]
                for shard_id in range(num_shards):
                    args.append([task_id, shard_id, global_shard_id])
                    global_shard_id += 1
            self._spark.sparkContext.parallelize(args, len(args)).map(lambda args: _rename_shard(*args)).collect()
        else:
            # don't use any pattern
            shard_id = 0
            task_id = task_id_and_num_shards[0][0]
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

    def _get_examples_iterable_for_split(
        self,
        split_generator: "datasets.SplitGenerator",
    ) -> SparkExamplesIterable:
        return SparkExamplesIterable(self.df)
<|endoftext|>import functools
import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union
from urllib.parse import unquote

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.file_utils import xgetsize


if TYPE_CHECKING:
    import vortex
    import vortex.expr
    import vortex.store

logger = datasets.utils.logging.get_logger(__name__)

# Vortex reads `hf://` URIs itself, so this is only matched to re-root a read on an `HfStore`
# carrying the `token` and `endpoint` that `load_dataset` was given, which a URI cannot express.
_HF_URI_PATTERN = re.compile(r"^hf://datasets/([^/@]+/[^/@]+?)(?:@([^/]+))?/(.+)$")

# `hf://buckets/...` URIs point at HF Buckets, which `HfStore` cannot root a read on yet.
_HF_BUCKETS_URI_PATTERN = re.compile(r"^hf://buckets/")

# The comparisons `pyarrow.parquet.filters_to_expression` accepts, mapped to the Python operators
# that build the equivalent Vortex expression.
_FILTER_COMPARISONS = {
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

# When `IterableDataset.reshard()` subdivides files, adjacent splits (one per chunk of the file,
# usually a few thousand rows) are coalesced into shards of about this many bytes of file data.
# The target is converted to rows through the file's average row size, so shards weigh about the
# same however wide or compressible the rows are. Coalescing whole splits keeps shard boundaries
# on chunk boundaries, so neighboring shards never read the same chunk.
_RESHARD_TARGET_NUM_BYTES = 64 << 20

# The rows per shard to fall back to when the file's size cannot be determined.
_RESHARD_FALLBACK_NUM_ROWS = 1 << 20


@dataclass
class VortexConfig(datasets.BuilderConfig):
    """
    BuilderConfig for Vortex.

    Args:
        batch_size (`int`, *optional*):
            Size of the RecordBatches to iterate on.
            The default is defined by the Vortex scanner.
        columns (`list[str]`, *optional*)
            List of columns to load, the other ones are ignored.
            All columns are loaded by default.
        features: (`Features`, *optional*):
            Cast the data to `features`.
        filters (`Union[vortex.expr.Expr, list[tuple], list[list[tuple]]]`, *optional*):
            Return only the rows matching the filter.
            The predicate is pushed down into the Vortex scan so only the matching rows are read.
            Filters given as a list of tuples (DNF, like the Parquet loader accepts) are converted
            to a Vortex expression.
            Nulls follow SQL semantics: a null satisfies no comparison, so a row whose filtered
            column is null is never returned. Note that `not in` therefore drops nulls, where the
            Parquet loader keeps them because it builds `~field.isin(values)` and a null is not in
            the set.
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error")
            Specify what to do upon encountering a bad file (a file that can't be read). Allowed values are :
            * 'error', raise an Exception when a bad file is encountered.
            * 'warn', raise a warning when a bad file is encountered and skip that file.
            * 'skip', skip bad files without raising or warning when they are encountered.

    Example:

    Load a subset of columns:

    ```python
    >>> ds = load_dataset(vortex_dataset_id, columns=["col_0", "col_1"])
    ```

    Stream data and efficiently filter data, skipping entire files or chunks when possible:

    ```python
    >>> filters = [("col_0", "==", 0)]
    >>> ds = load_dataset(vortex_dataset_id, streaming=True, filters=filters)
    ```

    """

    batch_size: Optional[int] = None
    columns: Optional[list[str]] = None
    features: Optional[datasets.Features] = None
    filters: Optional[Union["vortex.expr.Expr", list[tuple], list[list[tuple]]]] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"

    def __post_init__(self):
        super().__post_init__()


def _filters_to_expression(filters: Union[list[tuple], list[list[tuple]]]) -> "vortex.expr.Expr":
    """Convert Parquet-style DNF filters to a Vortex expression.

    Accepts the shapes `pyarrow.parquet.filters_to_expression` accepts: a list of predicates, taken
    as a conjunction, or a list of such lists, taken as a disjunction of conjunctions. A predicate
    is a `(column, operator, value)` tuple or list.
    """
    import vortex.expr as ve

    def predicate_to_expression(predicate) -> "vortex.expr.Expr":
        col, op, val = predicate
        column = ve.column(col)
        if op in _FILTER_COMPARISONS:
            return _FILTER_COMPARISONS[op](column, val)
        elif op in ("in", "not in"):
            if not val:
                raise ValueError(f"Empty set of values for '{op}' filter on column '{col}'")
            if op == "in":
                return ve.or_collect([column == value for value in val])
            return ve.and_collect([column != value for value in val])
        else:
            raise ValueError(f"Unsupported filter operator: '{op}'")

    if not filters or not filters[0]:
        raise ValueError(f"Malformed filters: {filters}")
    if isinstance(filters[0][0], str):
        # One nesting level too few: [(col, op, val), ...] instead of [[(col, op, val), ...]]
        filters = [filters]
    conjunctions = []
    for conjunction in filters:
        if not conjunction:
            raise ValueError(f"Malformed filters: {filters}")
        conjunctions.append(ve.and_collect([predicate_to_expression(predicate) for predicate in conjunction]))
    return ve.or_collect(conjunctions)


@functools.lru_cache(maxsize=64)
def _hf_store(
    repo_id: str, revision: Optional[str], token: Optional[Union[str, bool]], endpoint: Optional[str]
) -> "vortex.store.HfStore":
    """A store rooted at one repository and revision, shared by all the files read from it."""
    import vortex.store

    return vortex.store.HfStore(repo_id, revision=revision, token=token, endpoint=endpoint)


def _open_vortex_file(file: str, hf_storage_options: Optional[dict] = None) -> "vortex.VortexFile":
    """Open a Vortex file from a local path, a URL or a `hf://` URI (used in streaming mode).

    Vortex resolves `hf://` URIs itself, taking credentials from `HF_TOKEN` or the saved login, but
    the `token` and `endpoint` `load_dataset` was given cannot be expressed in a URI. So a Hub read
    is re-rooted on an `HfStore` carrying them, against which the in-repository path is opened.

    Only `hf://datasets/...` URIs are re-rooted this way. `hf://buckets/...` URIs are refused until
    `HfStore` supports HF Buckets.
    """
    if _HF_BUCKETS_URI_PATTERN.match(file):
        raise NotImplementedError(f"Vortex cannot read from HF Buckets yet: {file}")

    import vortex

    matched = _HF_URI_PATTERN.match(file)
    if matched:
        repo_id, revision, path_in_repo = matched.groups()
        hf_storage_options = hf_storage_options or {}
        # `HfStore` percent-encodes the revision itself, so give it the decoded one.
        revision = unquote(revision) if revision else None
        store = _hf_store(repo_id, revision, hf_storage_options.get("token"), hf_storage_options.get("endpoint"))
        return vortex.open(path_in_repo, store=store)
    return vortex.open(file)


def _to_arrow_reader(
    vortex_file: "vortex.VortexFile",
    row_range: Optional[tuple[int, int]],
    projection: Optional[list[str]],
    filter_expr: Optional["vortex.expr.Expr"],
    batch_size: Optional[int] = None,
) -> pa.RecordBatchReader:
    """Scan the whole file, or only the rows of `row_range` (offsets in the file, before filtering)."""
    if row_range is None:
        return vortex_file.to_arrow(projection=projection, expr=filter_expr, batch_size=batch_size)
    scan = vortex_file.to_repeated_scan(projection, expr=filter_expr, batch_size=batch_size)
    return scan.execute(row_range=row_range).to_arrow()


def _reshard_target_num_rows(
    vortex_file: "vortex.VortexFile", file: str, hf_storage_options: Optional[dict] = None
) -> int:
    """How many rows of this file weigh about `_RESHARD_TARGET_NUM_BYTES`, by the file's average row size."""
    num_rows = len(vortex_file)
    try:
        download_config = datasets.DownloadConfig(storage_options={"hf": hf_storage_options or {}})
        file_num_bytes = xgetsize(file, download_config=download_config)
    except (OSError, NotImplementedError, ValueError):
        file_num_bytes = None
    if not file_num_bytes or not num_rows:
        return _RESHARD_FALLBACK_NUM_ROWS
    return max(1, num_rows * _RESHARD_TARGET_NUM_BYTES // file_num_bytes)


def _coalesced_row_ranges(vortex_file: "vortex.VortexFile", target_num_rows: int) -> list[tuple[int, int]]:
    """The file's splits, coalesced into row ranges of about `target_num_rows` rows each."""
    row_ranges = []
    start = stop = None
    for split_start, split_stop in vortex_file.splits():
        if start is not None and split_stop - start <= target_num_rows:
            stop = split_stop
        else:
            if start is not None:
                row_ranges.append((start, stop))
            start, stop = split_start, split_stop
    if start is not None:
        row_ranges.append((start, stop))
    return row_ranges


class Vortex(datasets.ArrowBasedBuilder, datasets.builder._CountableBuilderMixin):
    BUILDER_CONFIG_CLASS = VortexConfig

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) != set(self.config.features)
        ):
            if any(col not in self.config.features for col in self.config.columns):
                raise ValueError(
                    "The columns and features argument must match, but got ",
                    f"{self.config.columns} and {self.config.features}",
                )
            else:
                features = datasets.Features({col: self.config.features[col] for col in self.config.columns})
        else:
            features = self.config.features
        return datasets.DatasetInfo(features=features)

    def _handle_bad_file(self, file: str, error: Exception):
        if self.config.on_bad_files == "error":
            logger.error(f"Failed to read file '{file}' with error {type(error).__name__}: {error}")
            raise error
        elif self.config.on_bad_files == "warn":
            logger.warning(f"Skipping bad file '{file}'. {type(error).__name__}: {error}")
        else:
            logger.debug(f"Skipping bad file '{file}'. {type(error).__name__}: {error}")

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        hf_storage_options = dict(dl_manager.download_config.storage_options.get("hf", {}))
        splits = []
        for split_name, files in data_files.items():
            files = [str(file) for file in files]
            # Infer features if they are stored in the vortex file schema
            if self.info.features is None:
                for file in files:
                    try:
                        vortex_file = _open_vortex_file(file, hf_storage_options)
                    except RuntimeError as e:  # Vortex surfaces unreadable files as RuntimeError
                        self._handle_bad_file(file, e)
                    else:
                        self.info.features = datasets.Features.from_arrow_schema(vortex_file.dtype.to_arrow_schema())
                        break
            if self.info.features is None:
                raise ValueError(
                    f"At least one valid data file must be specified, all the data_files are invalid: {self.config.data_files}"
                )
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "files": files,
                        "row_ranges": [None] * len(files),
                        "hf_storage_options": hf_storage_options,
                    },
                )
            )
        if self.config.columns is not None and set(self.config.columns) != set(self.info.features):
            self.info.features = datasets.Features(
                {col: feat for col, feat in self.info.features.items() if col in self.config.columns}
            )
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _filter_expression(self) -> Optional["vortex.expr.Expr"]:
        if isinstance(self.config.filters, list):
            return _filters_to_expression(self.config.filters)
        return self.config.filters

    def _generate_shards(self, files, row_ranges, hf_storage_options=None):
        for file, row_range in zip(files, row_ranges):
            if row_range is None:
                yield file
            else:
                yield {"fragment_data_file": file, "fragment_row_range": row_range}

    def _generate_more_gen_kwargs(self, files, row_ranges, hf_storage_options=None):
        for file, row_range in zip(files, row_ranges):
            new_row_ranges = [row_range]
            if row_range is None:
                try:
                    vortex_file = _open_vortex_file(file, hf_storage_options)
                    target_num_rows = _reshard_target_num_rows(vortex_file, file, hf_storage_options)
                    new_row_ranges = _coalesced_row_ranges(vortex_file, target_num_rows)
                except RuntimeError:
                    # Keep the unreadable file whole: `_generate_tables` applies `on_bad_files` to it.
                    pass
            yield {
                "files": [file] * len(new_row_ranges),
                "row_ranges": new_row_ranges,
                "hf_storage_options": hf_storage_options,
            }

    def _generate_num_examples(self, files, row_ranges, hf_storage_options=None):
        filter_expr = self._filter_expression()
        for file, row_range in zip(files, row_ranges):
            if filter_expr is None:
                if row_range is None:
                    yield len(_open_vortex_file(file, hf_storage_options))
                else:
                    yield row_range[1] - row_range[0]
            else:
                # Counting the matching rows takes a scan, but none of the columns have to come back
                # from it: an empty projection reads only what the predicate needs.
                vortex_file = _open_vortex_file(file, hf_storage_options)
                reader = _to_arrow_reader(vortex_file, row_range, [], filter_expr)
                yield sum(record_batch.num_rows for record_batch in reader)

    def _generate_tables(self, files, row_ranges, hf_storage_options=None):
        if self.config.features is not None and self.config.columns is not None:
            if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load vortex data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
                )
        filter_expr = self._filter_expression()
        for file_idx, (file, row_range) in enumerate(zip(files, row_ranges)):
            try:
                vortex_file = _open_vortex_file(file, hf_storage_options)
                reader = _to_arrow_reader(
                    vortex_file, row_range, self.config.columns, filter_expr, self.config.batch_size
                )
                for batch_idx, record_batch in enumerate(reader):
                    pa_table = pa.Table.from_batches([record_batch])
                    yield Key(file_idx, batch_idx), self._cast_table(pa_table)
            except RuntimeError as e:  # Vortex surfaces unreadable files as RuntimeError
                self._handle_bad_file(file, e)
<|endoftext|>import json
import os
from itertools import islice
from typing import Iterable

import pyarrow as pa

import datasets
from datasets.builder import Key


logger = datasets.utils.logging.get_logger(__name__)


class Eval(datasets.GeneratorBasedBuilder):
    NUM_EXAMPLES_FOR_FEATURES_INFERENCE = 5

    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        base_data_files = dl_manager.download(self.config.data_files)
        extracted_data_files = dl_manager.extract(base_data_files)
        splits = []
        for split_name, logs in extracted_data_files.items():
            logs_files_iterables = [dl_manager.iter_files(log) for log in logs]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "logs_files_iterables": logs_files_iterables,
                        "base_files": base_data_files[split_name],
                    },
                )
            )
        if not self.info.features:
            first_examples = list(
                islice(
                    self._iter_samples_from_log_files(logs_files_iterables[0]),
                    self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE,
                )
            )
            pa_tables = [pa.Table.from_pylist([example]) for example in first_examples]
            inferred_arrow_schema = pa.concat_tables(pa_tables, promote_options="default").schema
            self.info.features = datasets.Features.from_arrow_schema(inferred_arrow_schema)

        return splits

    def _sort_samples_key(self, sample_path: str):
        # looks like "{sample_idx}_epoch_{epoch_idx}""
        (sample_idx_str, epoch_idx_str) = os.path.splitext(os.path.basename(sample_path))[0].split("_epoch_")
        return (int(epoch_idx_str), int(sample_idx_str))

    def _iter_samples_from_log_files(self, log_files: Iterable[str]):
        sample_files = [log_file for log_file in log_files if os.path.basename(os.path.dirname(log_file)) == "samples"]
        sample_files.sort(key=self._sort_samples_key)
        for sample_file in sample_files:
            with open(sample_file) as f:
                sample = json.load(f)
                for field in sample:
                    if isinstance(sample[field], dict):
                        sample[field] = json.dumps(sample[field])
                    if isinstance(sample[field], list):
                        sample[field] = [json.dumps(x) for x in sample[field]]
                yield sample

    def _generate_shards(self, base_files, logs_files_iterables):
        yield from base_files

    def _generate_examples(self, base_files, logs_files_iterables):
        for file_idx, log_files in enumerate(logs_files_iterables):
            for sample_idx, sample in enumerate(self._iter_samples_from_log_files(log_files)):
                yield Key(file_idx, sample_idx), sample
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class PdfFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for ImageFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class PdfFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Pdf
    BASE_COLUMN_NAME = "pdf"
    BUILDER_CONFIG_CLASS = PdfFolderConfig
    EXTENSIONS: list[str] = [".pdf"]
<|endoftext|>import datasets

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class VideoFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for ImageFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class VideoFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Video
    BASE_COLUMN_NAME = "video"
    BUILDER_CONFIG_CLASS = VideoFolderConfig
    EXTENSIONS: list[str]  # definition at the bottom of the script


# TODO: initial list, we should check the compatibility of other formats
VIDEO_EXTENSIONS = [
    ".mkv",
    ".mp4",
    ".avi",
    ".mpeg",
    ".mov",
]
VideoFolder.EXTENSIONS = VIDEO_EXTENSIONS
<|endoftext|>"""CoNLL format dataset builder.

Reads CoNLL-style files where each line carries one token plus its tag columns,
columns are whitespace-separated, and empty lines mark sentence boundaries.
Each example is one sentence, with each configured column produced as a list
aligned with the tokens list.

Supports CoNLL-2000 (chunking), CoNLL-2003 (NER), CoNLL-U (Universal
Dependencies), and any custom column schema by overriding ``column_names``.
"""

from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.features.features import require_storage_cast
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


DEFAULT_COLUMN_NAMES = ["tokens"]


@dataclass
class ConllConfig(datasets.BuilderConfig):
    """BuilderConfig for CoNLL-style files.

    Args:
        features: (`Features`, *optional*):
            Cast the data to `features`.
        column_names: (`list[str]`, defaults to `["tokens"]`):
            Names for each whitespace-separated column. The first name is the
            token column; subsequent names cover the tag columns. Common
            schemas:

            - CoNLL-2003 NER: `["tokens", "pos_tags", "chunk_tags", "ner_tags"]`
            - CoNLL-2000 chunking: `["tokens", "pos_tags", "chunk_tags"]`
            - CoNLL-U: `["id", "form", "lemma", "upos", "xpos", "feats",
              "head", "deprel", "deps", "misc"]`
        delimiter: (`str`, *optional*):
            Column delimiter inside each line. If `None` (default), any
            whitespace is used, matching the standard CoNLL convention.
        encoding: (`str`, defaults to `"utf-8"`):
            Encoding to decode the file.
        encoding_errors: (`str`, *optional*):
            Argument to define what to do in case of encoding error. Same as
            the `errors` argument in `open()`.
        skip_docstart: (`bool`, defaults to `True`):
            Skip CoNLL-2003 `-DOCSTART-` document-boundary marker lines.
        comment_prefix: (`str`, *optional*):
            If set, lines starting with this prefix are skipped. CoNLL-U
            comments start with `#`.
    """

    features: Optional[datasets.Features] = None
    column_names: list[str] = field(default_factory=lambda: list(DEFAULT_COLUMN_NAMES))
    delimiter: Optional[str] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    skip_docstart: bool = True
    comment_prefix: Optional[str] = None


class Conll(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ConllConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """The `data_files` kwarg in load_dataset() can be a str, List[str],
        Dict[str,str], or Dict[str,List[str]].

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
                    gen_kwargs={
                        "files_iterables": files_iterables,
                        "base_files": base_data_files[split_name],
                    },
                )
            )
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all(not require_storage_cast(feature) for feature in self.config.features.values()):
                pa_table = pa_table.cast(schema)
            else:
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_shards(self, base_files, files_iterables):
        yield from base_files

    def _generate_tables(self, base_files, files_iterables):
        column_names = list(self.config.column_names)
        if not column_names:
            raise ValueError("ConllConfig.column_names must be a non-empty list")
        num_cols = len(column_names)
        delimiter = self.config.delimiter
        skip_docstart = self.config.skip_docstart
        comment_prefix = self.config.comment_prefix

        for shard_idx, files_iterable in enumerate(files_iterables):
            for file in files_iterable:
                sentences: list[list[list[str]]] = []
                current: list[list[str]] = [[] for _ in range(num_cols)]
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    for raw_line in f:
                        # Strip the trailing newline only — preserve embedded whitespace
                        line = raw_line.rstrip("\r\n")
                        if not line.strip():
                            # Sentence boundary
                            if current[0]:
                                sentences.append(current)
                                current = [[] for _ in range(num_cols)]
                            continue
                        if skip_docstart and line.startswith("-DOCSTART-"):
                            continue
                        if comment_prefix is not None and line.startswith(comment_prefix):
                            continue
                        parts = line.split(delimiter) if delimiter is not None else line.split()
                        # Pad or truncate to num_cols so column alignment is preserved
                        if len(parts) < num_cols:
                            parts = parts + [""] * (num_cols - len(parts))
                        elif len(parts) > num_cols:
                            parts = parts[:num_cols]
                        for i, val in enumerate(parts):
                            current[i].append(val)
                    # Tail sentence with no trailing blank line
                    if current[0]:
                        sentences.append(current)

                if sentences:
                    arrays = [pa.array([sentence[col_idx] for sentence in sentences]) for col_idx in range(num_cols)]
                    pa_table = pa.Table.from_arrays(arrays, names=column_names)
                    yield Key(shard_idx, 0), self._cast_table(pa_table)
<|endoftext|>import codecs
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.json as paj

import datasets
import datasets.config
from datasets import List, Value
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.file_utils import readline
from datasets.utils.json import (
    find_mixed_struct_types_field_paths,
    get_json_field_path_from_pyarrow_json_error,
    get_json_field_paths_from_feature,
    insert_json_field_path,
    json_encode_field,
    json_encode_fields_in_json_lines,
    set_json_types_in_feature,
    ujson_dumps,
    ujson_loads,
)


logger = datasets.utils.logging.get_logger(__name__)


def pandas_read_json(path_or_buf, **kwargs):
    if datasets.config.PANDAS_VERSION.major >= 2:
        kwargs["dtype_backend"] = "pyarrow"
    return pd.read_json(path_or_buf, **kwargs)


class FullReadDisallowed(Exception):
    pass


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True  # deprecated
    block_size: Optional[int] = None  # deprecated
    chunksize: int = 10 << 20  # 10MB
    newlines_in_values: Optional[bool] = None
    on_mixed_types: Optional[Literal["use_json"]] = "use_json"
    parse_agent_traces: bool = True

    def __post_init__(self):
        super().__post_init__()


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        if self.config.block_size is not None:
            logger.warning("The JSON loader parameter `block_size` is deprecated. Please use `chunksize` instead")
            self.config.chunksize = self.config.block_size
        if self.config.use_threads is not True:
            logger.warning(
                "The JSON loader parameter `use_threads` is deprecated and doesn't have any effect anymore."
            )
        if self.config.newlines_in_values is not None:
            raise ValueError("The JSON loader parameter `newlines_in_values` is no longer supported")
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
                    gen_kwargs={
                        "files_iterables": files_iterables,
                        "base_files": base_data_files[split_name],
                        "original_files": self.config.data_files[split_name],
                    },
                )
            )
        if self.info.features is None:
            try:
                pa_table = next(iter(self._generate_tables(**splits[0].gen_kwargs, allow_full_read=False)))[1]
                self.info.features = datasets.Features.from_arrow_schema(pa_table.schema)
                if self.config.parse_agent_traces and has_agent_traces_markers(self.info.features):
                    self.info.features = AGENT_TRACES_FEATURES
            except FullReadDisallowed:
                pass
        return splits

    def _cast_table(self, pa_table: pa.Table, json_field_paths=()) -> pa.Table:
        if self.info.features is not None:
            # adding missing columns
            for column_name in set(self.info.features) - set(pa_table.column_names):
                type = self.info.features.arrow_schema.field(column_name).type
                pa_table = pa_table.append_column(column_name, pa.array([None] * len(pa_table), type=type))
            # convert to string when needed
            for i, column_name in enumerate(pa_table.column_names):
                if pa.types.is_struct(pa_table[column_name].type) and self.info.features.get(
                    column_name, None
                ) == Value("string"):
                    jsonl = (
                        pa_table[column_name]
                        .to_pandas(types_mapper=pd.ArrowDtype)
                        .to_json(orient="records", lines=True)
                    )
                    string_array = pa.array(
                        (None if x.strip() == "null" else x.strip() for x in jsonl.split("\n") if x.strip()),
                        type=pa.string(),
                    )
                    pa_table = pa_table.set_column(i, column_name, string_array)
            # more expensive cast to support nested structures with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        elif json_field_paths:
            features = datasets.Features.from_arrow_schema(pa_table.schema)
            features = set_json_types_in_feature(features, json_field_paths)
            pa_table = table_cast(pa_table, features.arrow_schema)
        return pa_table

    def _generate_shards(self, base_files, files_iterables, original_files):
        yield from base_files

    def _generate_tables(self, base_files, files_iterables, original_files, allow_full_read=True):
        json_field_paths = []
        is_agent_traces = False

        if self.info.features is not None:
            if self.info.features == AGENT_TRACES_FEATURES:
                is_agent_traces = True
                if datasets.config.TEICH_AVAILABLE:
                    from teich import convert_traces_to_training_data
                else:
                    raise ImportError("To support decoding agent traces, please install 'teich'.")
            json_field_paths = get_json_field_paths_from_feature(self.info.features)

        for shard_idx, files_iterable in enumerate(files_iterables):
            for file in files_iterable:
                # If the file is one json object and if we need to look at the items in one specific field
                if self.config.field is not None:
                    if not allow_full_read:
                        raise FullReadDisallowed()
                    with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                        dataset = ujson_loads(f.read())
                    # We keep only the field we are interested in
                    dataset = dataset[self.config.field]
                    df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
                    if df.columns.tolist() == [0]:
                        df.columns = list(self.config.features) if self.config.features else ["text"]
                    pa_table = pa.Table.from_pandas(df, preserve_index=False)
                    yield Key(shard_idx, 0), self._cast_table(pa_table)

                # If the files are agent traces (one row = one file except for hermes which can have multiple sessions per file)
                elif is_agent_traces:
                    trace_file = Path(file)
                    trace = trace_file.read_text(encoding="utf-8")
                    lines = trace.splitlines()
                    file_path = original_files[shard_idx]
                    if self.base_path is not None and file_path.startswith(self.base_path):
                        file_path = os.path.relpath(file_path, self.base_path)
                    training_examples = convert_traces_to_training_data(trace_file)
                    examples = []
                    for i, training_example in enumerate(training_examples):
                        if training_example["metadata"]["trace_type"] == "hermes":
                            timestamp = ujson_loads(lines[i])["started_at"]
                            milliseconds = timestamp if timestamp <= 10_000_000_000 else timestamp * 1_000
                            sent_at = (
                                datetime.fromtimestamp(milliseconds, tz=timezone.utc)
                                .isoformat(timespec="milliseconds")
                                .replace("+00:00", "Z")
                            )
                            bonus_fields = {
                                "harness": training_example["metadata"]["trace_type"],
                                "session_id": training_example["metadata"]["session_id"],
                                "sent_at": sent_at,
                                "num_user_messages": training_example["metadata"]["turn_count"],
                                "num_tool_calls": training_example["metadata"]["tool_call_count"],
                                "trace": lines[i],
                            }
                        else:
                            harness, session_id, prompt, sent_at, num_user_messages, num_tool_calls = (
                                parse_traces_info(lines)
                            )
                            bonus_fields = {
                                "harness": harness,
                                "session_id": session_id,
                                "prompt": prompt,
                                "sent_at": sent_at,
                                "num_user_messages": num_user_messages,
                                "num_tool_calls": num_tool_calls,
                                "trace": trace,
                            }
                        example = {
                            **dict.fromkeys(AGENT_TRACES_FEATURES),
                            **training_example,
                            **bonus_fields,
                            "file_path": file_path,
                        }
                        for json_field_path in json_field_paths:
                            example = json_encode_field(example, json_field_path)
                        examples.append(example)
                    pa_table = pa.Table.from_pylist(examples)
                    yield Key(shard_idx, 0), self._cast_table(pa_table)

                # If the file has one json object per line
                else:
                    with open(file, "rb") as f:
                        batch_idx = 0
                        # Use block_size equal to the chunk size divided by 32 to leverage multithreading
                        # Set a default minimum value of 16kB if the chunk size is really small
                        block_size = max(self.config.chunksize // 32, 16 << 10)
                        encoding_errors = (
                            self.config.encoding_errors if self.config.encoding_errors is not None else "strict"
                        )
                        while True:
                            batch = f.read(self.config.chunksize)
                            if not batch:
                                break
                            # A leading UTF-8 BOM makes the ujson pre-scan below raise
                            # ValueError, silently skipping mixed-struct detection, while
                            # PyArrow tolerates the BOM -- so the inferred schema differed
                            # depending on whether the file started with a BOM. Strip it so
                            # both see the same bytes (a BOM only appears at the start of the file).
                            if shard_idx == 0 and batch_idx == 0 and batch.startswith(codecs.BOM_UTF8):
                                batch = batch[len(codecs.BOM_UTF8) :]
                            if batch.startswith(b"["):
                                if not allow_full_read:
                                    raise FullReadDisallowed()
                                else:
                                    # convert to JSON Lines
                                    full_data = batch + f.read()
                                    if b"{" in batch[:100].split(b'"', 1)[0]:  # list of objects
                                        batch = "\n".join(ujson_dumps(x) for x in ujson_loads(full_data)).encode()
                                    else:  # list of strings
                                        batch = "\n".join(
                                            ujson_dumps({"text": x}) for x in ujson_loads(full_data)
                                        ).encode()
                            # Finish current line
                            try:
                                batch += f.readline()
                            except (AttributeError, io.UnsupportedOperation):
                                batch += readline(f)
                            # PyArrow only accepts utf-8 encoded bytes
                            if self.config.encoding != "utf-8":
                                batch = batch.decode(self.config.encoding, errors=encoding_errors).encode("utf-8")
                            # On first batch we check for lists of objects with arbitrary fields
                            if (
                                shard_idx == 0
                                and batch_idx == 0
                                and self.info.features is None
                                and self.config.on_mixed_types == "use_json"
                            ):
                                try:
                                    examples = [ujson_loads(line) for line in batch.splitlines()]
                                except ValueError:
                                    # the file is likely not JSON Lines and may contain one single multi-line JSON object
                                    pass
                                else:
                                    json_field_paths += find_mixed_struct_types_field_paths(examples)
                            # Re-encode JSON fields
                            original_batch = batch
                            if json_field_paths:
                                examples = [ujson_loads(line) for line in batch.splitlines()]
                                for json_field_path in json_field_paths:
                                    examples = [json_encode_field(example, json_field_path) for example in examples]
                                batch = "\n".join(ujson_dumps(example) for example in examples).encode()
                            # Disable parallelism if block size is ~ len(batch) to avoid segfault
                            block_size = len(batch) if len(batch) // 8 > block_size else block_size
                            try:
                                while True:
                                    try:
                                        pa_table = paj.read_json(
                                            io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size)
                                        )
                                        break
                                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                        if batch.startswith(b"["):  # paj.read_json only supports json lines
                                            raise
                                        elif self.config.on_mixed_types == "use_json" and (
                                            isinstance(e, pa.ArrowInvalid)
                                            and "JSON parse error: Column(" in str(e)
                                            and ") changed from" in str(e)
                                        ):
                                            json_field_path = get_json_field_path_from_pyarrow_json_error(str(e))
                                            insert_json_field_path(json_field_paths, json_field_path)
                                            batch = json_encode_fields_in_json_lines(original_batch, json_field_paths)
                                        elif (
                                            "straddling" in str(e) or "JSON conversion to" in str(e)
                                        ) and block_size < len(batch):
                                            # Increase the block size in case it was too small.
                                            # The block size will be reset for the next file.
                                            # this is needed in case of "stradding" or for some JSON conversions (see https://github.com/huggingface/datasets/issues/2799)
                                            logger.debug(
                                                f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}."
                                            )
                                            block_size *= 2
                                        else:
                                            raise
                            except pa.ArrowInvalid as e:
                                if not allow_full_read:
                                    raise FullReadDisallowed()
                                try:
                                    with open(
                                        file, encoding=self.config.encoding, errors=self.config.encoding_errors
                                    ) as f:
                                        df = pandas_read_json(f)
                                except ValueError:
                                    logger.error(f"Failed to load JSON from file '{file}' with error {type(e)}: {e}")
                                    raise e
                                if df.columns.tolist() == [0]:
                                    df.columns = list(self.config.features) if self.config.features else ["text"]
                                try:
                                    pa_table = pa.Table.from_pandas(df, preserve_index=False)
                                except pa.ArrowInvalid as e:
                                    logger.error(
                                        f"Failed to convert pandas DataFrame to Arrow Table from file '{file}' with error {type(e)}: {e}"
                                    )
                                    raise ValueError(
                                        f"Failed to convert pandas DataFrame to Arrow Table from file {file}."
                                    ) from None
                                yield Key(shard_idx, 0), self._cast_table(pa_table)
                                break
                            yield (
                                Key(shard_idx, batch_idx),
                                self._cast_table(pa_table, json_field_paths=json_field_paths),
                            )
                            batch_idx += 1


AGENT_TRACES_TYPES_VALUES = {
    "claude_code": ["user", "assistant", "system"],
    "pi": ["session", "message"],
    "codex": ["session_meta", "turn_context", "response_item", "event_msg"],
    # droid message events share pi's "message" type, but droid traces always start with a session_start event
    "droid": ["session_start"],
}
AGENT_TRACES_TYPE_TO_HARNESS = {}
for _harness, _trace_types in AGENT_TRACES_TYPES_VALUES.items():
    for _trace_type in _trace_types:
        AGENT_TRACES_TYPE_TO_HARNESS[_trace_type] = _harness


AGENT_TRACES_FEATURES_MARKERS = {
    "claude_code_or_pi_or_openclaw": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "message": lambda f: f == datasets.Json(),
        }
    ),
    "codex": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "payload": lambda f: f == datasets.Json(),
        }
    ),
    "hermes": datasets.Features(
        {
            "id": lambda f: f == Value("string"),
            "source": lambda f: f == datasets.Value("string"),
            "model": lambda f: f == datasets.Value("string"),
            "system_prompt": lambda f: f == datasets.Value("string"),
            "messages": lambda f: isinstance(f, (datasets.List, datasets.Json)),
        }
    ),
    "droid": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "id": lambda f: f == Value("string"),
            "version": lambda f: f == Value("int64"),
            "cwd": lambda f: f == Value("string"),
        }
    ),
}

AGENT_TRACES_FEATURES = datasets.Features(
    {
        # basic features
        "harness": Value("string"),
        "session_id": Value("string"),
        # teich features
        "prompt": Value("string"),
        "messages": List(datasets.Json()),
        "tools": List(datasets.Json()),
        "metadata": datasets.Json(),
        # bonus features
        "sent_at": Value("string"),
        "num_user_messages": Value("int64"),
        "num_tool_calls": Value("int64"),
        "trace": datasets.Json(),
        "file_path": Value("string"),
    }
)


def has_agent_traces_markers(features: datasets.Features) -> bool:
    for agent_traces_features_marker in AGENT_TRACES_FEATURES_MARKERS.values():
        if all(feature_marker(features.get(key)) for key, feature_marker in agent_traces_features_marker.items()):
            return True
    return False


def parse_traces_info(
    trace_events: list[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], int, int]:
    harness, session_id, prompt, sent_at = None, None, None, None
    # prompt/sent_at describe the first user message; the counters summarize the whole trace file.
    # Codex response_item user messages can include context files (for example AGENTS.md), so only event_msg
    # user_message records are treated as Codex user messages.
    num_user_messages = 0
    num_tool_calls = 0
    for event in trace_events:
        decoded_event = ujson_loads(event)
        if harness is None:
            if "type" in decoded_event and isinstance(decoded_event["type"], str):
                harness = AGENT_TRACES_TYPE_TO_HARNESS.get(decoded_event["type"])
        if session_id is None:
            session_id = get_session_id(decoded_event)
            if (
                session_id is not None
                and decoded_event.get("type") == "session"
                and isinstance(decoded_event.get("cwd"), str)
                and "/.openclaw/" in decoded_event["cwd"]
            ):
                harness = "openclaw"
        user_prompt = get_user_prompt(decoded_event)
        if user_prompt is not None:
            num_user_messages += 1
            if prompt is None:
                prompt = user_prompt
                sent_at = get_trace_event_timestamp(decoded_event)
        num_tool_calls += get_tool_call_count(decoded_event)
    return harness, session_id, prompt, sent_at, num_user_messages, num_tool_calls


def get_session_id(trace: dict) -> Optional[str]:
    # claude
    if isinstance(trace.get("sessionId"), str):
        return trace["sessionId"]
    # claude (not sure but this format does exist online)
    if isinstance(trace.get("session_id"), str):
        return trace["session_id"]
    # codex
    if isinstance(trace.get("payload"), dict) and isinstance(trace["payload"].get("id"), str):
        return trace["payload"]["id"]
    # pi / openclaw on "session" (openclaw embeds pi-agent; distinguish via cwd), droid on "session_start"
    if trace.get("type") in ("session", "session_start") and isinstance(trace.get("id"), str):
        return trace["id"]
    return None


def get_user_prompt(trace_event: dict) -> Optional[str]:
    if trace_event.get("type") == "user" and isinstance(trace_event.get("message"), dict):
        message = trace_event["message"]
        if message.get("role") == "user":
            return get_content_text(message.get("content"))

    if trace_event.get("type") == "message":
        if isinstance(trace_event.get("message"), dict):
            message = trace_event["message"]
            # droid marks injected context as llm_only and local-only notes as user_only, neither is a real user prompt
            if message.get("role") == "user" and message.get("visibility") not in ("llm_only", "user_only"):
                return get_content_text(message.get("content"))
        if trace_event.get("role") == "user":
            return get_content_text(trace_event.get("content"))

    if trace_event.get("type") == "event_msg" and isinstance(trace_event.get("payload"), dict):
        payload = trace_event["payload"]
        if payload.get("type") == "user_message":
            return get_content_text(payload.get("message"))

    return None


def get_tool_call_count(trace_event: dict) -> int:
    trace_type = trace_event.get("type")
    if trace_type == "response_item":
        payload = trace_event.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "function_call":
            return 1
        return 0

    if trace_type not in {"assistant", "message"} or not isinstance(trace_event.get("message"), dict):
        return 0
    message = trace_event["message"]
    if message.get("role") != "assistant":
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for content_part in content
        if isinstance(content_part, dict) and content_part.get("type") in {"tool_use", "toolCall"}
    )


def get_trace_event_timestamp(trace_event: dict) -> Optional[str]:
    timestamp = trace_event.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp
    return None


def get_content_text(content) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        content_parts = []
        for content_part in content:
            if isinstance(content_part, str):
                content_parts.append(content_part)
            elif isinstance(content_part, dict):
                text = content_part.get("text")
                if isinstance(text, str):
                    content_parts.append(text)
        if content_parts:
