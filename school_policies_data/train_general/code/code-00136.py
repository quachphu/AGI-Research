    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    """Build a comparison key for PEP 440 ordering.

    Returns ``(epoch, release, suffix)`` or
    ``(epoch, release, suffix, local)`` so that plain tuple
    comparison gives the correct order.

    Trailing zeros are stripped from the release so that ``1.0.0 == 1``.

    The suffix is a flat 6-int tuple that encodes pre/post/dev:
    ``(pre_rank, pre_n, post_rank, post_n, dev_rank, dev_n)``

    pre_rank: dev-only=-1, a=0, b=1, rc=2, no-pre=3
        Dev-only releases (no pre or post) get -1 so they sort before
        any alpha/beta/rc.  Releases without a pre-release tag get 3
        so they sort after rc.
    post_rank: no-post=0, post=1
        Releases without a post segment sort before those with one.
    dev_rank: dev=0, no-dev=1
        Releases without a dev segment sort after those with one.

    Local segments use ``(n, "")`` for ints and ``(-1, s)`` for strings,
    following PEP 440: strings sort before ints, strings compare
    lexicographically, ints compare numerically, and shorter segments
    sort before longer when prefixes match.  Versions without a local
    segment sort before those with one (3-tuple < 4-tuple).

    >>> _cmpkey(0, (1, 0, 0), None, None, None, None)
    (0, (1,), (3, 0, 0, 0, 1, 0))
    >>> _cmpkey(0, (1,), ("a", 1), None, None, None)
    (0, (1,), (0, 1, 0, 0, 1, 0))
    >>> _cmpkey(0, (1,), None, None, None, ("ubuntu", 1))
    (0, (1,), (3, 0, 0, 0, 1, 0), ((-1, 'ubuntu'), (1, '')))
    """
    # Strip trailing zeros: 1.0.0 compares equal to 1.
    len_release = len(release)
    i = len_release
    while i and release[i - 1] == 0:
        i -= 1
    trimmed = release if i == len_release else release[:i]

    # Fast path: stable release with no local segment.
    if pre is None and post is None and dev is None and local is None:
        return epoch, trimmed, _STABLE_SUFFIX

    if pre is None and post is None and dev is not None:
        # dev-only (e.g. 1.0.dev1) sorts before all pre-releases.
        pre_rank, pre_n = _PRE_RANK_DEV_ONLY, 0
    elif pre is None:
        pre_rank, pre_n = _PRE_RANK_STABLE, 0
    else:
        pre_rank, pre_n = _PRE_RANK[pre[0]], pre[1]

    post_rank = 0 if post is None else 1
    post_n = 0 if post is None else post[1]

    dev_rank = 1 if dev is None else 0
    dev_n = 0 if dev is None else dev[1]

    suffix = (pre_rank, pre_n, post_rank, post_n, dev_rank, dev_n)

    if local is None:
        return epoch, trimmed, suffix

    cmp_local: CmpLocalType = tuple(
        (seg, "") if isinstance(seg, int) else (_LOCAL_STR_RANK, seg) for seg in local
    )
    return epoch, trimmed, suffix, cmp_local
<|endoftext|>from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
    cast,
)
from urllib.parse import unquote, urlparse

from .markers import (
    Environment,
    Marker,
    _pep440_python_full_version,
    default_environment,
)
from .specifiers import SpecifierSet
from .tags import create_compatible_tags_selector, sys_tags
from .utils import (
    NormalizedName,
    is_normalized_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from .version import Version

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Collection, Iterator
    from pathlib import Path

    from typing_extensions import Self

    from .tags import Tag

_logger = logging.getLogger(__name__)

__all__ = [
    "Package",
    "PackageArchive",
    "PackageDirectory",
    "PackageSdist",
    "PackageVcs",
    "PackageWheel",
    "Pylock",
    "PylockSelectError",
    "PylockUnsupportedVersionError",
    "PylockValidationError",
    "is_valid_pylock_path",
]


def __dir__() -> list[str]:
    return __all__


_T = TypeVar("_T")
_T2 = TypeVar("_T2")


class _FromMappingProtocol(Protocol):  # pragma: no cover
    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self: ...


_FromMappingProtocolT = TypeVar("_FromMappingProtocolT", bound=_FromMappingProtocol)


_PYLOCK_FILE_NAME_RE = re.compile(r"^pylock\.([^.]+)\.toml$")


def is_valid_pylock_path(path: Path) -> bool:
    """Check if the given path is a valid pylock file path."""
    return path.name == "pylock.toml" or bool(_PYLOCK_FILE_NAME_RE.match(path.name))


def _toml_key(key: str) -> str:
    return key.replace("_", "-")


def _toml_value(key: str, value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, (Version, Marker, SpecifierSet)):
        return str(value)
    if isinstance(value, Sequence) and key == "environments":
        return [str(v) for v in value]
    return value


def _toml_dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
    return {
        _toml_key(key): _toml_value(key, value)
        for key, value in data
        if value is not None
    }


def _get(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T | None:
    """Get a value from the dictionary and verify it's the expected type."""
    if (value := d.get(key)) is None:
        return None
    if not isinstance(value, expected_type) or (
        # Special case: bool is a subclass of int, but TOML distinguishes the two
        expected_type is int and isinstance(value, bool)
    ):
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} "
            f"(expected {expected_type.__name__})",
            context=key,
        )
    return value


def _get_required(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T:
    """Get a required value from the dictionary and verify it's the expected type."""
    if (value := _get(d, expected_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence(
    d: Mapping[str, Any], expected_item_type: type[_T], key: str
) -> Sequence[_T] | None:
    """Get a list value from the dictionary and verify it's the expected items type."""
    if (value := _get(d, Sequence, key)) is None:  # type: ignore[type-abstract]
        return None
    if isinstance(value, (str, bytes)):
        # special case: str and bytes are Sequences, but we want to reject it
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} (expected Sequence)",
            context=key,
        )
    for i, item in enumerate(value):
        if not isinstance(item, expected_item_type):
            raise PylockValidationError(
                f"Unexpected type {type(item).__name__} "
                f"(expected {expected_item_type.__name__})",
                context=f"{key}[{i}]",
            )
    return value


def _get_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2 | None:
    """Get a value from the dictionary, verify it's the expected type,
    and convert to the target type.

    This assumes the target_type constructor accepts the value.
    """
    if (value := _get(d, expected_type, key)) is None:
        return None
    try:
        return target_type(value)
    except Exception as e:
        raise PylockValidationError(e, context=key) from e


def _get_required_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2:
    """Get a required value from the dict, verify it's the expected type,
    and convert to the target type."""
    if (value := _get_as(d, expected_type, target_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence_as(
    d: Mapping[str, Any],
    expected_item_type: type[_T],
    target_item_type: Callable[[_T], _T2],
    key: str,
) -> list[_T2] | None:
    """Get list value from dictionary and verify expected items type."""
    if (value := _get_sequence(d, expected_item_type, key)) is None:
        return None
    result = []
    try:
        for item in value:
            typed_item = target_item_type(item)
            result.append(typed_item)
    except Exception as e:
        raise PylockValidationError(e, context=f"{key}[{len(result)}]") from e
    return result


def _get_object(
    d: Mapping[str, Any], target_type: type[_FromMappingProtocolT], key: str
) -> _FromMappingProtocolT | None:
    """Get a dictionary value from the dictionary and convert it to a dataclass."""
    if (value := _get(d, Mapping, key)) is None:  # type: ignore[type-abstract]
        return None
    try:
        return target_type._from_dict(value)
    except Exception as e:
        raise PylockValidationError(e, context=key) from e


def _get_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> list[_FromMappingProtocolT] | None:
    """Get a list value from the dictionary and convert its items to a dataclass."""
    if (value := _get_sequence(d, Mapping, key)) is None:  # type: ignore[type-abstract]
        return None
    result: list[_FromMappingProtocolT] = []
    try:
        for item in value:
            typed_item = target_item_type._from_dict(item)
            result.append(typed_item)
    except Exception as e:
        raise PylockValidationError(e, context=f"{key}[{len(result)}]") from e
    return result


def _get_required_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> Sequence[_FromMappingProtocolT]:
    """Get a required list value from the dictionary and convert its items to a
    dataclass."""
    if (result := _get_sequence_of_objects(d, target_item_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return result


def _validate_normalized_name(name: str) -> NormalizedName:
    """Validate that a string is a NormalizedName."""
    if not is_normalized_name(name):
        raise PylockValidationError(f"Name {name!r} is not normalized")
    return NormalizedName(name)


def _validate_path_url(path: str | None, url: str | None) -> None:
    if not path and not url:
        raise PylockValidationError("path or url must be provided")


def _path_name(path: str | None) -> str | None:
    if not path:
        return None
    # If the path is relative it MAY use POSIX-style path separators explicitly
    # for portability
    if "/" in path:
        return path.rsplit("/", 1)[-1]
    elif "\\" in path:
        return path.rsplit("\\", 1)[-1]
    else:
        return path


def _url_name(url: str | None) -> str | None:
    if not url:
        return None
    url_path = urlparse(url).path
    # The last path component is percent-encoded, so decode it to the file name
    return unquote(url_path.rsplit("/", 1)[-1])


def _validate_hashes(hashes: Mapping[str, Any]) -> Mapping[str, Any]:
    if not hashes:
        raise PylockValidationError("At least one hash must be provided")
    if not all(isinstance(hash_val, str) for hash_val in hashes.values()):
        raise PylockValidationError("Hash values must be strings")
    return hashes


class PylockValidationError(Exception):
    """Raised when when input data is not spec-compliant."""

    context: str | None = None
    message: str

    def __init__(
        self,
        cause: str | Exception,
        *,
        context: str | None = None,
    ) -> None:
        if isinstance(cause, PylockValidationError):
            if cause.context:
                self.context = (
                    f"{context}.{cause.context}" if context else cause.context
                )
            else:
                self.context = context
            self.message = cause.message
        else:
            self.context = context
            self.message = str(cause)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} in {self.context!r}"
        return self.message


class _PylockRequiredKeyError(PylockValidationError):
    def __init__(self, key: str) -> None:
        super().__init__("Missing required value", context=key)


class PylockUnsupportedVersionError(PylockValidationError):
    """Raised when encountering an unsupported `lock_version`."""


class PylockSelectError(Exception):
    """Base exception for errors raised by :meth:`Pylock.select`.

    .. versionadded:: 26.1
    """


@dataclass(frozen=True, init=False)
class PackageVcs:
    type: str
    url: str | None = None
    path: str | None = None
    requested_revision: str | None = None
    commit_id: str  # type: ignore[misc]
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        type: str,
        url: str | None = None,
        path: str | None = None,
        requested_revision: str | None = None,
        commit_id: str,
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "requested_revision", requested_revision)
        object.__setattr__(self, "commit_id", commit_id)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_vcs = cls(
            type=_get_required(d, str, "type"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            requested_revision=_get(d, str, "requested-revision"),
            commit_id=_get_required(d, str, "commit-id"),
            subdirectory=_get(d, str, "subdirectory"),
        )
        _validate_path_url(package_vcs.path, package_vcs.url)
        return package_vcs


@dataclass(frozen=True, init=False)
class PackageDirectory:
    path: str
    editable: bool | None = None
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        path: str,
        editable: bool | None = None,
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "editable", editable)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(
            path=_get_required(d, str, "path"),
            editable=_get(d, bool, "editable"),
            subdirectory=_get(d, str, "subdirectory"),
        )


@dataclass(frozen=True, init=False)
class PackageArchive:
    url: str | None = None
    path: str | None = None
    size: int | None = None
    upload_time: datetime | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        upload_time: datetime | None = None,
        hashes: Mapping[str, str],
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "hashes", hashes)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_archive = cls(
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            upload_time=_get(d, datetime, "upload-time"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
            subdirectory=_get(d, str, "subdirectory"),
        )
        _validate_path_url(package_archive.path, package_archive.url)
        return package_archive


@dataclass(frozen=True, init=False)
class PackageSdist:
    name: str | None = None
    upload_time: datetime | None = None
    url: str | None = None
    path: str | None = None
    size: int | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]

    def __init__(
        self,
        *,
        name: str | None = None,
        upload_time: datetime | None = None,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        hashes: Mapping[str, str],
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "hashes", hashes)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_sdist = cls(
            name=_get(d, str, "name"),
            upload_time=_get(d, datetime, "upload-time"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
        )
        _validate_path_url(package_sdist.path, package_sdist.url)
        return package_sdist

    @property
    def filename(self) -> str:
        """Get the filename of the sdist.

        .. versionadded:: 26.1
        """
        filename = self.name or _path_name(self.path) or _url_name(self.url)
        if not filename:
            raise PylockValidationError("Cannot determine sdist filename")
        return filename


@dataclass(frozen=True, init=False)
class PackageWheel:
    name: str | None = None
    upload_time: datetime | None = None
    url: str | None = None
    path: str | None = None
    size: int | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]

    def __init__(
        self,
        *,
        name: str | None = None,
        upload_time: datetime | None = None,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        hashes: Mapping[str, str],
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "hashes", hashes)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_wheel = cls(
            name=_get(d, str, "name"),
            upload_time=_get(d, datetime, "upload-time"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
        )
        _validate_path_url(package_wheel.path, package_wheel.url)
        return package_wheel

    @property
    def filename(self) -> str:
        """Get the filename of the wheel."""
        filename = self.name or _path_name(self.path) or _url_name(self.url)
        if not filename:
            raise PylockValidationError("Cannot determine wheel filename")
        return filename


@dataclass(frozen=True, init=False)
class Package:
    name: NormalizedName
    version: Version | None = None
    marker: Marker | None = None
    requires_python: SpecifierSet | None = None
    dependencies: Sequence[Mapping[str, Any]] | None = None
    vcs: PackageVcs | None = None
    directory: PackageDirectory | None = None
    archive: PackageArchive | None = None
    index: str | None = None
    sdist: PackageSdist | None = None
    wheels: Sequence[PackageWheel] | None = None
    attestation_identities: Sequence[Mapping[str, Any]] | None = None
    tool: Mapping[str, Any] | None = None

    def __init__(
        self,
        *,
        name: NormalizedName,
        version: Version | None = None,
        marker: Marker | None = None,
        requires_python: SpecifierSet | None = None,
        dependencies: Sequence[Mapping[str, Any]] | None = None,
        vcs: PackageVcs | None = None,
        directory: PackageDirectory | None = None,
        archive: PackageArchive | None = None,
        index: str | None = None,
        sdist: PackageSdist | None = None,
        wheels: Sequence[PackageWheel] | None = None,
        attestation_identities: Sequence[Mapping[str, Any]] | None = None,
        tool: Mapping[str, Any] | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "marker", marker)
        object.__setattr__(self, "requires_python", requires_python)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "vcs", vcs)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "archive", archive)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "sdist", sdist)
        object.__setattr__(self, "wheels", wheels)
        object.__setattr__(self, "attestation_identities", attestation_identities)
        object.__setattr__(self, "tool", tool)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package = cls(
            name=_get_required_as(d, str, _validate_normalized_name, "name"),
            version=_get_as(d, str, Version, "version"),
            requires_python=_get_as(d, str, SpecifierSet, "requires-python"),
            dependencies=_get_sequence(d, Mapping, "dependencies"),  # type: ignore[type-abstract]
            marker=_get_as(d, str, Marker, "marker"),
            vcs=_get_object(d, PackageVcs, "vcs"),
            directory=_get_object(d, PackageDirectory, "directory"),
            archive=_get_object(d, PackageArchive, "archive"),
            index=_get(d, str, "index"),
            sdist=_get_object(d, PackageSdist, "sdist"),
            wheels=_get_sequence_of_objects(d, PackageWheel, "wheels"),
            attestation_identities=_get_sequence(d, Mapping, "attestation-identities"),  # type: ignore[type-abstract]
            tool=_get(d, Mapping, "tool"),  # type: ignore[type-abstract]
        )
        distributions = bool(package.sdist) + len(package.wheels or [])
        direct_urls = (
            bool(package.vcs) + bool(package.directory) + bool(package.archive)
        )
        if distributions > 0 and direct_urls > 0:
            raise PylockValidationError(
                "None of vcs, directory, archive must be set if sdist or wheels are set"
            )
        if distributions == 0 and direct_urls != 1:
            raise PylockValidationError(
                "Exactly one of vcs, directory, archive must be set "
                "if sdist and wheels are not set"
            )
        for i, wheel in enumerate(package.wheels or []):
            try:
                (name, version, _, _) = parse_wheel_filename(wheel.filename)
            except Exception as e:
                raise PylockValidationError(
                    f"Invalid wheel filename {wheel.filename!r}",
                    context=f"wheels[{i}]",
                ) from e
            if name != package.name:
                raise PylockValidationError(
                    f"Name in {wheel.filename!r} is not consistent with "
                    f"package name {package.name!r}",
                    context=f"wheels[{i}]",
                )
            if package.version and version != package.version:
                raise PylockValidationError(
                    f"Version in {wheel.filename!r} is not consistent with "
                    f"package version {str(package.version)!r}",
                    context=f"wheels[{i}]",
                )
        if package.sdist:
            try:
                name, version = parse_sdist_filename(package.sdist.filename)
            except Exception as e:
                raise PylockValidationError(
                    f"Invalid sdist filename {package.sdist.filename!r}",
                    context="sdist",
                ) from e
            if name != package.name:
                raise PylockValidationError(
                    f"Name in {package.sdist.filename!r} is not consistent with "
                    f"package name {package.name!r}",
                    context="sdist",
                )
            if package.version and version != package.version:
                raise PylockValidationError(
                    f"Version in {package.sdist.filename!r} is not consistent with "
                    f"package version {str(package.version)!r}",
                    context="sdist",
                )
        try:
            for i, attestation_identity in enumerate(  # noqa: B007
                package.attestation_identities or []
            ):
                _get_required(attestation_identity, str, "kind")
        except Exception as e:
            raise PylockValidationError(
                e, context=f"attestation-identities[{i}]"
            ) from e
        return package

    @property
    def is_direct(self) -> bool:
        return not (self.sdist or self.wheels)


@dataclass(frozen=True, init=False)
class Pylock:
    """A class representing a pylock file."""

    lock_version: Version
    environments: Sequence[Marker] | None = None
    requires_python: SpecifierSet | None = None
    extras: Sequence[NormalizedName] | None = None
    dependency_groups: Sequence[str] | None = None
    default_groups: Sequence[str] | None = None
    created_by: str  # type: ignore[misc]
    packages: Sequence[Package]  # type: ignore[misc]
    tool: Mapping[str, Any] | None = None

    def __init__(
        self,
        *,
        lock_version: Version,
        environments: Sequence[Marker] | None = None,
        requires_python: SpecifierSet | None = None,
        extras: Sequence[NormalizedName] | None = None,
        dependency_groups: Sequence[str] | None = None,
        default_groups: Sequence[str] | None = None,
        created_by: str,
        packages: Sequence[Package],
        tool: Mapping[str, Any] | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "lock_version", lock_version)
        object.__setattr__(self, "environments", environments)
        object.__setattr__(self, "requires_python", requires_python)
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "dependency_groups", dependency_groups)
        object.__setattr__(self, "default_groups", default_groups)
        object.__setattr__(self, "created_by", created_by)
        object.__setattr__(self, "packages", packages)
        object.__setattr__(self, "tool", tool)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        pylock = cls(
            lock_version=_get_required_as(d, str, Version, "lock-version"),
            environments=_get_sequence_as(d, str, Marker, "environments"),
            extras=_get_sequence_as(d, str, _validate_normalized_name, "extras"),
            dependency_groups=_get_sequence(d, str, "dependency-groups"),
            default_groups=_get_sequence(d, str, "default-groups"),
            created_by=_get_required(d, str, "created-by"),
            requires_python=_get_as(d, str, SpecifierSet, "requires-python"),
            packages=_get_required_sequence_of_objects(d, Package, "packages"),
            tool=_get(d, Mapping, "tool"),  # type: ignore[type-abstract]
        )
        if not Version("1") <= pylock.lock_version < Version("2"):
            raise PylockUnsupportedVersionError(
                f"pylock version {pylock.lock_version} is not supported"
            )
        if pylock.lock_version > Version("1.0"):
            _logger.warning(
                "pylock minor version %s is not supported", pylock.lock_version
            )
        return pylock

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], /) -> Self:
        """Create and validate a Pylock instance from a TOML dictionary.

        Raises :class:`PylockValidationError` if the input data is not
        spec-compliant.
        """
        return cls._from_dict(d)

    def to_dict(self) -> Mapping[str, Any]:
        """Convert the Pylock instance to a TOML dictionary."""
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)

    def validate(self) -> None:
        """Validate the Pylock instance against the specification.

        Raises :class:`PylockValidationError` otherwise."""
        self.from_dict(self.to_dict())

    def select(
        self,
        *,
        environment: Environment | None = None,
        tags: Sequence[Tag] | None = None,
        extras: Collection[str] | None = None,
        dependency_groups: Collection[str] | None = None,
        prefer_sdist_predicate: Callable[[NormalizedName], bool] | None = None,
    ) -> Iterator[
        tuple[
            Package,
            PackageVcs
            | PackageDirectory
            | PackageArchive
            | PackageWheel
            | PackageSdist,
        ]
    ]:
        """Select what to install from the lock file.

        The *environment* and *tags* parameters represent the environment being
        selected for. If unspecified, ``packaging.markers.default_environment()`` and
        ``packaging.tags.sys_tags()`` are used.

        The *extras* parameter represents the extras to install.

        The *dependency_groups* parameter represents the groups to install. If
        unspecified, the default groups are used.

        The *prefer_sdist_predicate* parameter is called for packages with a source
        distribution. If it returns ``True``, the source distribution is selected
        before attempting wheel compatibility. If no source distribution is
        available, wheel selection proceeds as usual without calling the predicate.

        This method must be used on valid Pylock instances (i.e. one obtained
        from :meth:`Pylock.from_dict` or if constructed manually, after calling
        :meth:`Pylock.validate`).

        .. versionadded:: 26.1

        .. versionchanged:: 26.3
            Added the *prefer_sdist_predicate* parameter.
        """
        compatible_tags_selector = create_compatible_tags_selector(
            tags if tags is not None else sys_tags()
        )

        # #. Gather the extras and dependency groups to install and set ``extras`` and
        #    ``dependency_groups`` for marker evaluation, respectively.
        #
        #    #. ``extras`` SHOULD be set to the empty set by default.
        #    #. ``dependency_groups`` SHOULD be the set created from
        #       :ref:`pylock-default-groups` by default.
        env = cast(
            "dict[str, str | frozenset[str]]",
            dict(
                environment or {},  # Marker.evaluate will fill-up
                extras=frozenset(extras or []),
                dependency_groups=frozenset(
                    (self.default_groups or [])
                    if dependency_groups is None  # to allow selecting no group
                    else dependency_groups
                ),
            ),
        )
        env_python_full_version = _pep440_python_full_version(
            environment["python_full_version"]
            if environment
            else default_environment()["python_full_version"]
        )

        # #. Check if the metadata version specified by :ref:`pylock-lock-version` is
        #    supported; an error or warning MUST be raised as appropriate.
        # Covered by lock.validate() which is a precondition for this method.

        # #. If :ref:`pylock-requires-python` is specified, check that the environment
        #    being installed for meets the requirement; an error MUST be raised if it is
        #    not met.
        if self.requires_python and not self.requires_python.contains(
            env_python_full_version,
        ):
            raise PylockSelectError(
                f"python_full_version {env_python_full_version!r} "
                f"in provided environment does not satisfy the Python version "
                f"requirement {str(self.requires_python)!r}"
            )

        # #. If :ref:`pylock-environments` is specified, check that at least one of the
        #    environment marker expressions is satisfied; an error MUST be raised if no
        #    expression is satisfied.
        if self.environments:
            for env_marker in self.environments:
                if env_marker.evaluate(
                    cast("dict[str, str]", environment or {}), context="requirement"
                ):
                    break
            else:
                raise PylockSelectError(
                    "Provided environment does not satisfy any of the "
                    "environments specified in the lock file"
                )

        # #. For each package listed in :ref:`pylock-packages`:
        selected_packages_by_name: dict[str, tuple[int, Package]] = {}
        for package_index, package in enumerate(self.packages):
            # #. If :ref:`pylock-packages-marker` is specified, check if it is
            #    satisfied;if it isn't, skip to the next package.
            if package.marker and not package.marker.evaluate(env, context="lock_file"):
                continue

            # #. If :ref:`pylock-packages-requires-python` is specified, check if it is
            #    satisfied; an error MUST be raised if it isn't.
            if package.requires_python and not package.requires_python.contains(
                env_python_full_version,
            ):
                raise PylockSelectError(
                    f"python_full_version {env_python_full_version!r} "
                    f"in provided environment does not satisfy the Python version "
                    f"requirement {str(package.requires_python)!r} for package "
                    f"{package.name!r} at packages[{package_index}]"
                )

            # #. Check that no other conflicting instance of the package has been slated
            #    to be installed; an error about the ambiguity MUST be raised otherwise.
            if package.name in selected_packages_by_name:
                raise PylockSelectError(
                    f"Multiple packages with the name {package.name!r} are "
                    f"selected at packages[{package_index}] and "
                    f"packages[{selected_packages_by_name[package.name][0]}]"
                )

            # #. Check that the source of the package is specified appropriately (i.e.
            #    there are no conflicting sources in the package entry);
            #    an error MUST be raised if any issues are found.
            # Covered by lock.validate() which is a precondition for this method.

            # #. Add the package to the set of packages to install.
            selected_packages_by_name[package.name] = (package_index, package)

        # #. For each package to be installed:
        for package_index, package in selected_packages_by_name.values():
            # - If :ref:`pylock-packages-vcs` is set:
            if package.vcs is not None:
                yield package, package.vcs

            # - Else if :ref:`pylock-packages-directory` is set:
            elif package.directory is not None:
                yield package, package.directory

            # - Else if :ref:`pylock-packages-archive` is set:
            elif package.archive is not None:
                yield package, package.archive

            # - Else if source preference selects an available
            #   :ref:`pylock-packages-sdist`:
            elif (
                package.sdist is not None
                and prefer_sdist_predicate is not None
                and prefer_sdist_predicate(package.name)
            ):
                yield package, package.sdist

            # - Else if there are entries for :ref:`pylock-packages-wheels`:
            elif package.wheels:
                # #. Look for the appropriate wheel file based on
                #    :ref:`pylock-packages-wheels-name`; if one is not found then move
                #    on to :ref:`pylock-packages-sdist` or an error MUST be raised about
                #    a lack of source for the project.
                best_wheel = next(
                    compatible_tags_selector(
                        (wheel, parse_wheel_filename(wheel.filename)[-1])
                        for wheel in package.wheels
                    ),
                    None,
                )
                if best_wheel:
                    yield package, best_wheel
                elif package.sdist is not None:
                    yield package, package.sdist
                else:
                    raise PylockSelectError(
                        f"No wheel found matching the provided tags "
                        f"for package {package.name!r} "
                        f"at packages[{package_index}], "
                        f"and no sdist available as a fallback"
                    )

            # - Else if no :ref:`pylock-packages-wheels` file is found or
            #   :ref:`pylock-packages-sdist` is solely set:
            elif package.sdist is not None:
                yield package, package.sdist

            else:
                # Covered by lock.validate() which is a precondition for this method.
                raise NotImplementedError  # pragma: no cover
<|endoftext|>"""Handwritten parser of dependency specifiers.

The docstring for each __parse_* function contains EBNF-inspired grammar representing
the implementation.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Literal, NamedTuple, Union

from ._tokenizer import DEFAULT_RULES, ParserSyntaxError, Tokenizer


class Node:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.value!r})>"

    def serialize(self) -> str:
        raise NotImplementedError

    def __getstate__(self) -> str:
        # Return just the value string for compactness and stability.
        return self.value

    def _restore_value(self, value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"Cannot restore {self.__class__.__name__} value from {value!r}"
            )
        self.value = value

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            # New format (26.2+): just the value string.
            self._restore_value(state)
            return
        if isinstance(state, tuple) and len(state) == 2:
            # Old format (packaging <= 26.0, __slots__): (None, {slot: value}).
            _, slot_dict = state
            if isinstance(slot_dict, dict) and "value" in slot_dict:
                self._restore_value(slot_dict["value"])
                return
        if isinstance(state, dict) and "value" in state:
            # Old format (packaging <= 25.0, no __slots__): plain __dict__.
            self._restore_value(state["value"])
            return
        raise TypeError(f"Cannot restore {self.__class__.__name__} from {state!r}")


class Variable(Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


class Value(Node):
    __slots__ = ()

    def serialize(self) -> str:
        value = str(self)
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        raise ValueError(
            "Cannot serialize marker value containing both quote characters"
        )


class Op(Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


MarkerLogical = Literal["and", "or"]
MarkerVar = Union[Variable, Value]
MarkerItem = tuple[MarkerVar, Op, MarkerVar]
MarkerAtom = Union[MarkerItem, Sequence["MarkerAtom"]]
MarkerList = list[Union["MarkerList", MarkerAtom, MarkerLogical]]


class ParsedRequirement(NamedTuple):
    name: str
    url: str
    extras: list[str]
    specifier: str
    marker: MarkerList | None


# --------------------------------------------------------------------------------------
# Recursive descent parser for dependency specifier
# --------------------------------------------------------------------------------------
def parse_requirement(source: str) -> ParsedRequirement:
    return _parse_requirement(Tokenizer(source, rules=DEFAULT_RULES))


def _parse_requirement(tokenizer: Tokenizer) -> ParsedRequirement:
    """
    requirement = WS? IDENTIFIER WS? extras WS? requirement_details
    """
    tokenizer.consume("WS")

    name_token = tokenizer.expect(
        "IDENTIFIER", expected="package name at the start of dependency specifier"
    )
    name = name_token.text
    tokenizer.consume("WS")

    extras = _parse_extras(tokenizer)
    tokenizer.consume("WS")

    url, specifier, marker = _parse_requirement_details(tokenizer)
    tokenizer.expect("END", expected="end of dependency specifier")

    return ParsedRequirement(name, url, extras, specifier, marker)


def _parse_requirement_details(
    tokenizer: Tokenizer,
) -> tuple[str, str, MarkerList | None]:
    """
    requirement_details = AT URL (WS requirement_marker?)?
                        | specifier WS? (requirement_marker)?
    """

    specifier = ""
    url = ""
    marker = None

    if tokenizer.check("AT"):
        tokenizer.read()
        tokenizer.consume("WS")

        url_start = tokenizer.position
        url = tokenizer.expect("URL", expected="URL after @").text
        if tokenizer.check("END", peek=True):
            return (url, specifier, marker)

        tokenizer.expect("WS", expected="whitespace after URL")

        # The input might end after whitespace.
        if tokenizer.check("END", peek=True):
            return (url, specifier, marker)

        marker = _parse_requirement_marker(
            tokenizer,
            span_start=url_start,
            expected="semicolon (after URL and whitespace)",
        )
    else:
        specifier_start = tokenizer.position
        specifier = _parse_specifier(tokenizer)
        tokenizer.consume("WS")

        if tokenizer.check("END", peek=True):
            return (url, specifier, marker)

        marker = _parse_requirement_marker(
            tokenizer,
            span_start=specifier_start,
            expected=(
                "comma (within version specifier), semicolon (after version specifier)"
                if specifier
                else "semicolon (after name with no version specifier)"
            ),
        )

    return (url, specifier, marker)


def _parse_requirement_marker(
    tokenizer: Tokenizer, *, span_start: int, expected: str
) -> MarkerList:
    """
    requirement_marker = SEMICOLON marker WS?
    """

    if not tokenizer.check("SEMICOLON"):
        tokenizer.raise_syntax_error(
            f"Expected {expected} or end",
            span_start=span_start,
            span_end=None,
        )
    tokenizer.read()

    marker = _parse_marker(tokenizer)
    tokenizer.consume("WS")

    return marker


def _parse_extras(tokenizer: Tokenizer) -> list[str]:
    """
    extras = (LEFT_BRACKET wsp* extras_list? wsp* RIGHT_BRACKET)?
    """
    if not tokenizer.check("LEFT_BRACKET", peek=True):
        return []

    with tokenizer.enclosing_tokens(
        "LEFT_BRACKET",
        "RIGHT_BRACKET",
        around="extras",
    ):
        tokenizer.consume("WS")
        extras = _parse_extras_list(tokenizer)
        tokenizer.consume("WS")

    return extras


def _parse_extras_list(tokenizer: Tokenizer) -> list[str]:
    """
    extras_list = identifier (wsp* ',' wsp* identifier)*
    """
    extras: list[str] = []

    if not tokenizer.check("IDENTIFIER"):
        return extras

    extras.append(tokenizer.read().text)

    while True:
        tokenizer.consume("WS")
        if tokenizer.check("IDENTIFIER", peek=True):
            tokenizer.raise_syntax_error("Expected comma between extra names")
        elif not tokenizer.check("COMMA"):
            break

        tokenizer.read()
        tokenizer.consume("WS")

        extra_token = tokenizer.expect("IDENTIFIER", expected="extra name after comma")
        extras.append(extra_token.text)

    return extras


def _parse_specifier(tokenizer: Tokenizer) -> str:
    """
    specifier = LEFT_PARENTHESIS WS? version_many WS? RIGHT_PARENTHESIS
              | WS? version_many WS?
    """
    with tokenizer.enclosing_tokens(
        "LEFT_PARENTHESIS",
        "RIGHT_PARENTHESIS",
        around="version specifier",
    ):
        tokenizer.consume("WS")
        parsed_specifiers = _parse_version_many(tokenizer)
        tokenizer.consume("WS")

    return parsed_specifiers


def _parse_version_many(tokenizer: Tokenizer) -> str:
    """
    version_many = (SPECIFIER (WS? COMMA WS? SPECIFIER)*)?
    """
    parsed_specifiers = ""
    while tokenizer.check("SPECIFIER"):
        span_start = tokenizer.position
        specifier = tokenizer.read().text
        parsed_specifiers += specifier
        if tokenizer.check("VERSION_PREFIX_TRAIL", peek=True):
            message = ".* suffix can only be used with `==` or `!=` operators"
            if specifier.startswith("!=") or (
                specifier.startswith("==") and not specifier.startswith("===")
            ):
                message = (
                    ".* suffix cannot be used with pre-release, post-release, "
                    "dev or local versions"
                )
            tokenizer.raise_syntax_error(
                message,
                span_start=span_start,
                span_end=tokenizer.position + 1,
            )
        if tokenizer.check("VERSION_LOCAL_LABEL_TRAIL", peek=True):
            tokenizer.raise_syntax_error(
                "Local version label can only be used with `==` or `!=` operators",
                span_start=span_start,
                span_end=tokenizer.position,
            )
        tokenizer.consume("WS")
        if not tokenizer.check("COMMA"):
            break
        parsed_specifiers += tokenizer.read().text
        tokenizer.consume("WS")

    return parsed_specifiers


# --------------------------------------------------------------------------------------
# Recursive descent parser for marker expression
# --------------------------------------------------------------------------------------
def parse_marker(source: str) -> MarkerList:
    return _parse_full_marker(Tokenizer(source, rules=DEFAULT_RULES))


def _parse_full_marker(tokenizer: Tokenizer) -> MarkerList:
    retval = _parse_marker(tokenizer)
    tokenizer.expect("END", expected="end of marker expression")
    return retval


def _parse_marker(tokenizer: Tokenizer) -> MarkerList:
    """
    marker = marker_atom (BOOLOP marker_atom)+
    """
    expression = [_parse_marker_atom(tokenizer)]
    while tokenizer.check("BOOLOP"):
        token = tokenizer.read()
        expr_right = _parse_marker_atom(tokenizer)
        expression.extend((token.text, expr_right))
    return expression


def _parse_marker_atom(tokenizer: Tokenizer) -> MarkerAtom:
    """
    marker_atom = WS? LEFT_PARENTHESIS WS? marker WS? RIGHT_PARENTHESIS WS?
                | WS? marker_item WS?
    """

    tokenizer.consume("WS")
    if tokenizer.check("LEFT_PARENTHESIS", peek=True):
        with tokenizer.enclosing_tokens(
            "LEFT_PARENTHESIS",
            "RIGHT_PARENTHESIS",
            around="marker expression",
        ):
            tokenizer.consume("WS")
            marker: MarkerAtom = _parse_marker(tokenizer)
            tokenizer.consume("WS")
    else:
        marker = _parse_marker_item(tokenizer)
    tokenizer.consume("WS")
    return marker


def _parse_marker_item(tokenizer: Tokenizer) -> MarkerItem:
    """
    marker_item = WS? marker_var WS? marker_op WS? marker_var WS?
    """
    tokenizer.consume("WS")
    marker_var_left = _parse_marker_var(tokenizer)
    tokenizer.consume("WS")
    marker_op = _parse_marker_op(tokenizer)
    tokenizer.consume("WS")
    marker_var_right = _parse_marker_var(tokenizer)
    tokenizer.consume("WS")
    return (marker_var_left, marker_op, marker_var_right)


def _parse_marker_var(tokenizer: Tokenizer) -> MarkerVar:  # noqa: RET503
    """
    marker_var = VARIABLE | QUOTED_STRING
    """
    if tokenizer.check("VARIABLE"):
        return process_env_var(tokenizer.read().text.replace(".", "_"))
    elif tokenizer.check("QUOTED_STRING"):
        token = tokenizer.read()
        try:
            return process_python_str(token.text)
        except (SyntaxError, ValueError) as exc:
            raise ParserSyntaxError(
                "Invalid quoted string",
                source=tokenizer.source,
                span=(token.position, token.position + len(token.text)),
            ) from exc
    else:
        tokenizer.raise_syntax_error(
            message="Expected a marker variable or quoted string"
        )


def process_env_var(env_var: str) -> Variable:
    if env_var in ("platform_python_implementation", "python_implementation"):
        return Variable("platform_python_implementation")
    else:
        return Variable(env_var)


def process_python_str(python_str: str) -> Value:
    value = ast.literal_eval(python_str)
    return Value(str(value))


def _parse_marker_op(tokenizer: Tokenizer) -> Op:
    """
    marker_op = IN | NOT IN | OP
    """
    if tokenizer.check("IN"):
        tokenizer.read()
        return Op("in")
    elif tokenizer.check("NOT"):
        tokenizer.read()
        tokenizer.expect("WS", expected="whitespace after 'not'")
        tokenizer.expect("IN", expected="'in' after 'not'")
        return Op("not in")
    elif tokenizer.check("OP"):
        return Op(tokenizer.read().text)
    else:
        return tokenizer.raise_syntax_error(
            "Expected marker operator, one of <=, <, !=, ==, >=, >, ~=, ===, in, not in"
        )
<|endoftext|>from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import _ErrorCollector
from .requirements import InvalidRequirement, Requirement

__all__ = [
    "CyclicDependencyGroup",
    "DependencyGroupInclude",
    "DependencyGroupResolver",
    "DuplicateGroupNames",
    "InvalidDependencyGroupObject",
    "resolve_dependency_groups",
]


def __dir__() -> list[str]:
    return __all__


# -----------
# Error Types
# -----------


class DuplicateGroupNames(ValueError):
    """
    The same dependency groups were defined twice, with different non-normalized names.

    .. versionadded:: 26.1
    """


class CyclicDependencyGroup(ValueError):
    """
    The dependency group includes form a cycle.

    .. versionadded:: 26.1
    """

    def __init__(self, requested_group: str, group: str, include_group: str) -> None:
        self.requested_group = requested_group
        self.group = group
        self.include_group = include_group

        if include_group == group:
            reason = f"{group} includes itself"
        else:
            reason = f"{include_group} -> {group}, {group} -> {include_group}"
        super().__init__(
            "Cyclic dependency group include while resolving "
            f"{requested_group}: {reason}"
        )

    # Support pickling; ``args`` does not match ``__init__``'s signature.
    def __reduce__(self) -> tuple[type[CyclicDependencyGroup], tuple[str, str, str]]:
        return (self.__class__, (self.requested_group, self.group, self.include_group))


# in the PEP 735 spec, the tables in dependency group lists were described as
# "Dependency Object Specifiers", but the only defined type of object was a
# "Dependency Group Include" -- hence the naming of this error as "Object"
class InvalidDependencyGroupObject(ValueError):
    """
    A member of a dependency group was identified as a dict, but was not in a valid
    format.

    .. versionadded:: 26.1
    """


# ------------------------
# Object Model & Interface
# ------------------------


class DependencyGroupInclude:
    """
    A reference to another dependency group by name.

    .. versionadded:: 26.1
    """

    __slots__ = ("include_group",)

    def __init__(self, include_group: str) -> None:
        """
        Initialize a DependencyGroupInclude.

        :param include_group: The name of the group referred to by this include.
        """
        self.include_group = include_group

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.include_group!r})"


class DependencyGroupResolver:
    """
    A resolver for Dependency Group data.

    This class handles caching, name normalization, cycle detection, and other
    parsing requirements. There are only two public methods for exploring the data:
    ``lookup()`` and ``resolve()``.

    :param dependency_groups: A mapping, as provided via pyproject
        ``[dependency-groups]``.

    .. versionadded:: 26.1
    """

    def __init__(
        self,
        dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    ) -> None:
        errors = _ErrorCollector()

        self.dependency_groups = _normalize_group_names(dependency_groups, errors)

        # a map of group names to parsed data
        self._parsed_groups: dict[
            str, tuple[Requirement | DependencyGroupInclude, ...]
        ] = {}
        # a map of group names to their ancestors, used for cycle detection
        self._include_graph_ancestors: dict[str, tuple[str, ...]] = {}
        # a cache of completed resolutions to Requirement lists
        self._resolve_cache: dict[str, tuple[Requirement, ...]] = {}

        errors.finalize("[dependency-groups] data was invalid")

    def lookup(self, group: str) -> tuple[Requirement | DependencyGroupInclude, ...]:
        """
        Lookup a group name, returning the parsed dependency data for that group.
        This will not resolve includes.

        :param group: the name of the group to lookup
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._parse_group(group, errors)

    def resolve(self, group: str) -> tuple[Requirement, ...]:
        """
        Resolve a dependency group to a list of requirements.

        :param group: the name of the group to resolve
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._resolve(group, group, errors)

    def _resolve(
        self, group: str, requested_group: str, errors: _ErrorCollector
    ) -> tuple[Requirement, ...]:
        """
        This is a helper for cached resolution to strings. It preserves the name of the
        group which the user initially requested in order to present a clearer error in
        the event that a cycle is detected.

        :param group: The normalized name of the group to resolve.
        :param requested_group: The group which was used in the original, user-facing
            request.
        """
        if group in self._resolve_cache:
            return self._resolve_cache[group]

        parsed = self._parse_group(group, errors)

        resolved_group = []

        for item in parsed:
            if isinstance(item, Requirement):
                resolved_group.append(item)
            elif isinstance(item, DependencyGroupInclude):
                include_group = _normalize_name(item.include_group)

                # if a group is cyclic, record the error
                # otherwise, follow the include_group reference
                #
                # this allows us to examine all includes in a group, even in the
                # presence of errors
                if include_group in self._include_graph_ancestors.get(group, ()):
                    errors.error(
                        CyclicDependencyGroup(
                            requested_group, group, item.include_group
                        )
                    )
                else:
                    self._include_graph_ancestors[include_group] = (
                        *self._include_graph_ancestors.get(group, ()),
                        group,
                    )
                    resolved_group.extend(
                        self._resolve(include_group, requested_group, errors)
                    )
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"Invalid dependency group item after parse: {item}"
                )

        # in the event that errors were detected, present the group as empty and do not
        # cache the result
        # this ensures that repeated access to a cyclic group will raise multiple errors
        if errors.errors:
            return ()

        self._resolve_cache[group] = tuple(resolved_group)
        return self._resolve_cache[group]

    def _parse_group(
        self, group: str, errors: _ErrorCollector
    ) -> tuple[Requirement | DependencyGroupInclude, ...]:
        # short circuit -- never do the work twice
        if group in self._parsed_groups:
            return self._parsed_groups[group]

        if group not in self.dependency_groups:
            errors.error(LookupError(f"Dependency group '{group}' not found"))
            return ()

        raw_group = self.dependency_groups[group]
        if isinstance(raw_group, str):
            errors.error(
                TypeError(
                    f"Dependency group {group!r} contained a string rather than a list."
                )
            )
            return ()

        if not isinstance(raw_group, Sequence):
            errors.error(
                TypeError(f"Dependency group {group!r} is not a sequence type.")
            )
            return ()

        elements: list[Requirement | DependencyGroupInclude] = []
        for item in raw_group:
            if isinstance(item, str):
                # packaging.requirements.Requirement parsing ensures that this is a
                # valid PEP 508 Dependency Specifier. Collect InvalidRequirement
                # if it throws that.
                with errors.collect(InvalidRequirement):
                    elements.append(Requirement(item))
            elif isinstance(item, Mapping):
                if tuple(item.keys()) != ("include-group",):
                    errors.error(
                        InvalidDependencyGroupObject(
                            f"Invalid dependency group item: {item!r}"
                        )
                    )
                else:
                    include_group = item["include-group"]
                    if not isinstance(include_group, str):
                        msg = (
                            "Dependency group include-group value is not a string: "
                            f"{item!r}"
                        )
                        errors.error(TypeError(msg))
                    else:
                        elements.append(
                            DependencyGroupInclude(include_group=include_group)
                        )
            else:
                errors.error(TypeError(f"Invalid dependency group item: {item!r}"))

        if errors.errors:
            return ()

        self._parsed_groups[group] = tuple(elements)
        return self._parsed_groups[group]


# --------------------
# Functional Interface
# --------------------


def resolve_dependency_groups(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]], /, *groups: str
) -> tuple[str, ...]:
    """
    Resolve a dependency group to a tuple of requirements, as strings.

    :param dependency_groups: the parsed contents of the ``[dependency-groups]`` table
        from ``pyproject.toml``
    :param groups: the name of the group(s) to resolve

    .. versionadded:: 26.1
    """
    resolver = DependencyGroupResolver(dependency_groups)
    return tuple(str(r) for group in groups for r in resolver.resolve(group))


# ----------------
# internal helpers
# ----------------


_NORMALIZE_PATTERN = re.compile(r"[-_.]+")


def _normalize_name(name: str) -> str:
    return _NORMALIZE_PATTERN.sub("-", name).lower()


def _normalize_group_names(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    errors: _ErrorCollector,
) -> dict[str, Sequence[str | Mapping[str, str]]]:
    original_names: dict[str, list[str]] = {}
    normalized_groups: dict[str, Sequence[str | Mapping[str, str]]] = {}

    for group_name, value in dependency_groups.items():
        normed_group_name = _normalize_name(group_name)
        original_names.setdefault(normed_group_name, []).append(group_name)
        normalized_groups[normed_group_name] = value

    for normed_name, names in original_names.items():
        if len(names) > 1:
            errors.error(
                DuplicateGroupNames(
                    "Duplicate dependency group names: "
                    f"{normed_name} ({', '.join(names)})"
                )
            )

    return normalized_groups
<|endoftext|>"""PEP 656 support.

This module implements logic to detect if the currently running Python is
linked against musl, and what musl version is used.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from typing import TYPE_CHECKING, NamedTuple

from ._elffile import ELFFile

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class _MuslVersion(NamedTuple):
    major: int
    minor: int


def _parse_musl_version(output: str) -> _MuslVersion | None:
    lines = [n for n in (n.strip() for n in output.splitlines()) if n]
    if len(lines) < 2 or lines[0][:4] != "musl":
        return None
    m = re.match(r"Version (\d+)\.(\d+)", lines[1])
    if not m:
        return None
    return _MuslVersion(major=int(m.group(1)), minor=int(m.group(2)))


@functools.lru_cache
def _get_musl_version(executable: str) -> _MuslVersion | None:
    """Detect currently-running musl runtime version.

    This is done by checking the specified executable's dynamic linking
    information, and invoking the loader to parse its output for a version
    string. If the loader is musl, the output would be something like::

        musl libc (x86_64)
        Version 1.2.2
        Dynamic Program Loader
    """
    try:
        with open(executable, "rb") as f:
            ld = ELFFile(f).interpreter
    except (OSError, TypeError, ValueError):
        return None
    if ld is None or "musl" not in ld:
        return None
    proc = subprocess.run([ld], check=False, stderr=subprocess.PIPE, text=True)
    return _parse_musl_version(proc.stderr)


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate musllinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be musllinux-compatible.

    :returns: An iterator of compatible musllinux tags.
    """
    sys_musl = _get_musl_version(sys.executable)
    if sys_musl is None:  # Python not dynamically linked against musl.
        return
    for arch in archs:
        for minor in range(sys_musl.minor, -1, -1):
            yield f"musllinux_{sys_musl.major}_{minor}_{arch}"


if __name__ == "__main__":  # pragma: no cover
    import sysconfig

    plat = sysconfig.get_platform()
    assert plat.startswith("linux-"), "not linux"

    print("plat:", plat)
    print("musl:", _get_musl_version(sys.executable))
    print("tags:", end=" ")
    for t in platform_tags([re.sub(r"[.-]", "_", plat.split("-", 1)[-1])]):
        print(t, end="\n      ")
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""Private version-range helpers used by :mod:`packaging.specifiers`."""

from __future__ import annotations

import enum
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
)

from .version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Union

    # Total-order key for comparing two boundaries (boundary-vs-boundary only).
    # The post slot may be ``_BOUNDARY_INF`` for an AFTER_POSTS boundary.
    _BoundaryOrderSuffix = tuple[int, int, int, Union[int, float], int, int]
    _BoundaryOrderKey = tuple[int, tuple[int, ...], _BoundaryOrderSuffix, float]

__all__ = [
    "FULL_RANGE",
    "bounds_for_spec",
    "coerce_version",
    "filter_by_ranges",
    "intersect_ranges",
    "intersect_specifier_bounds",
    "least_version_above",
    "matches_bounds_only",
    "range_is_empty",
    "ranges_are_prerelease_only",
    "resolve_prereleases",
    "standard_ranges",
    "wildcard_ranges",
]

#: The smallest possible PEP 440 version. No valid version is less than this.
MIN_VERSION: Final[Version] = Version("0.dev0")

#: The smallest non-pre-release version, i.e. the nearest non-pre-release at or
#: above the ``-inf`` floor.
MIN_RELEASE: Final[Version] = Version("0")

#: Sorts above any real post number and any local label, so a boundary can be
#: ordered above the version family it covers when two boundaries are compared.
_BOUNDARY_INF: Final[float] = float("inf")


class BoundaryKind(enum.Enum):
    """Where a boundary marker sits in the version ordering."""

    AFTER_LOCALS = enum.auto()  # after V+local, before V.post0
    AFTER_POSTS = enum.auto()  # after V.postN, before next release


@functools.total_ordering
class BoundaryVersion:
    """A point on the version line between two real PEP 440 versions.

    Relative to a base version V::

        V < V+local < AFTER_LOCALS(V) < V.post0 < AFTER_POSTS(V)

    AFTER_LOCALS is the upper bound of ``<=V``, ``==V``, ``!=V`` (no
    local), and the lower bound of the upper-side range of ``!=V``.
    AFTER_POSTS is the lower bound of ``>V`` (V final or pre-release),
    excluding V's post-releases per PEP 440.
    """

    __slots__ = (
        "_cached_dev",
        "_cached_epoch",
        "_cached_post",
        "_cached_pre",
        "_cached_trimmed_release",
        "kind",
        "version",
    )

    def __init__(self, version: Version, kind: BoundaryKind) -> None:
        self.version = version
        self.kind = kind
        self._cached_trimmed_release = trim_release(version.release)
        self._cached_epoch = version.epoch
        self._cached_pre = version.pre
        self._cached_post = version.post
        self._cached_dev = version.dev

    def _is_family(self, other: Version) -> bool:
        """Is ``other`` a version that this boundary sorts above?"""
        if other.epoch != self._cached_epoch:
            return False
        # Inline release-trim comparison: other.release matches the
        # trimmed release iff its leading slice is equal and any extra
        # components are zero. Avoids trim_release's tuple allocation.
        other_release = other.release
        trimmed_release = self._cached_trimmed_release
        trimmed_length = len(trimmed_release)
        if len(other_release) < trimmed_length:
            return False
        if other_release[:trimmed_length] != trimmed_release:
            return False
        for i in range(trimmed_length, len(other_release)):
            if other_release[i] != 0:
                return False
        if other.pre != self._cached_pre:
            return False
        if self.kind == BoundaryKind.AFTER_LOCALS:
            # Local family: same public version, any local label.
            return other.post == self._cached_post and other.dev == self._cached_dev
        # Post family: V itself + any post-release of V.
        return other.dev == self._cached_dev or other.post is not None

    def _order_key(self) -> _BoundaryOrderKey:
        """Sort key placing this boundary just above the versions it covers.

        It extends ``V``'s comparison key ``(epoch, release, suffix)`` with
        a trailing ``_BOUNDARY_INF`` local component, so the key sorts after
        ``V`` and every ``V+local`` (whose keys carry a real, finite local
        segment). ``suffix`` is the 6-int comparison suffix
        ``(pre_rank, pre_n, post_rank, post_n, dev_rank, dev_n)``.

        For an AFTER_POSTS boundary the suffix is replaced with one whose
        post number is ``_BOUNDARY_INF``, so the key also sorts after every
        ``V.postN``. An AFTER_LOCALS boundary uses ``V``'s suffix unchanged.
        """
        version_key = self.version._key
        suffix: _BoundaryOrderSuffix = version_key[2]

        if self.kind == BoundaryKind.AFTER_POSTS:
            suffix = (suffix[0], suffix[1], 1, _BOUNDARY_INF, 1, 0)

        return version_key[0], version_key[1], suffix, _BOUNDARY_INF

    def __eq__(self, other: object) -> bool:
        # Key off the order key so equality matches the ``<`` / ``>`` order:
        # ``AFTER_POSTS(1.0)`` and ``AFTER_POSTS(1.0.post1)`` are the same point.
        if isinstance(other, BoundaryVersion):
            return self._order_key() == other._order_key()
        return NotImplemented

    def __lt__(self, other: BoundaryVersion | Version) -> bool:
        if isinstance(other, BoundaryVersion):
            return self._order_key() < other._order_key()
        # boundary < other_version iff V < other AND other not in family.
        # The cheap V >= other path short-circuits before the family check.
        if not (self.version < other):
            return False
        return not self._is_family(other)

    def __gt__(self, other: BoundaryVersion | Version) -> bool:
        # Defined directly to bypass functools.total_ordering's
        # NotImplemented round-trip on reflected ``Version < boundary``.
        if isinstance(other, BoundaryVersion):
            return self._order_key() > other._order_key()
        if self.version >= other:
            return True
        return self._is_family(other)

    def __hash__(self) -> int:
        # Keyed to ``__eq__`` (the order key), so equal boundaries hash equal.
        return hash(self._order_key())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.version!r}, {self.kind.name})"


if TYPE_CHECKING:
    _VersionOrBoundary = Union[Version, BoundaryVersion, None]


@functools.total_ordering
class LowerBound:
    """Lower bound of a version range.

    A version *v* of ``None`` means unbounded below (-inf).
    At equal versions, ``[v`` sorts before ``(v`` because an inclusive
    bound starts earlier.
    """

    __slots__ = ("_above", "inclusive", "version")

    def __init__(self, version: _VersionOrBoundary, inclusive: bool) -> None:
        self.version = version
        self.inclusive = inclusive
        # Pre-bind a predicate "is parsed at or above this lower
        # bound?" for the hot filter / contains loops. One direct
        # call per check, no operator-dispatch chain.
        if version is None:
            self._above: Callable[[Version], bool] | None = None
        elif isinstance(version, BoundaryVersion):
            # >V produces an AFTER_POSTS lower bound; the upper-side
            # range of !=V produces an AFTER_LOCALS lower bound.
            if version.kind == BoundaryKind.AFTER_POSTS:
                self._above = _make_above_after_posts(version.version)
            else:
                self._above = _make_above_after_locals(version.version)
        elif inclusive:
            self._above = version.__le__
        else:
            self._above = version.__lt__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LowerBound):
            return NotImplemented
        return self.version == other.version and self.inclusive == other.inclusive

    def __lt__(self, other: LowerBound) -> bool:
        if not isinstance(other, LowerBound):
            return NotImplemented
        # -inf < anything (except -inf itself).
        if self.version is None:
            return other.version is not None
        if other.version is None:
            return False
        if self.version != other.version:
            return self.version < other.version
        # [v < (v: inclusive starts earlier.
        return self.inclusive and not other.inclusive

    def __hash__(self) -> int:
        return hash((self.version, self.inclusive))

    def __repr__(self) -> str:
        bracket = "[" if self.inclusive else "("
        return f"<{self.__class__.__name__} {bracket}{self.version!r}>"


@functools.total_ordering
class UpperBound:
    """Upper bound of a version range.

    A version *v* of ``None`` means unbounded above (+inf).
    At equal versions, ``v)`` sorts before ``v]`` because an exclusive
    bound ends earlier.
    """

    __slots__ = ("_below", "inclusive", "version")

    def __init__(self, version: _VersionOrBoundary, inclusive: bool) -> None:
        self.version = version
        self.inclusive = inclusive
        # Pre-bind a predicate "is parsed at or below this upper
        # bound?". See LowerBound for the rationale.
        if version is None:
            self._below: Callable[[Version], bool] | None = None
        elif isinstance(version, BoundaryVersion):
            # Standard specifiers only ever produce AFTER_LOCALS upper
            # bounds (from <=V / ==V / !=V with no local).
            if version.kind == BoundaryKind.AFTER_LOCALS:
                self._below = _make_below_after_locals(version.version)
            else:
                # An AFTER_POSTS upper is not produced by any specifier, but
                # range algebra reaches it: complementing ``>V`` flips the
                # ``AFTER_POSTS(V)`` lower into this upper bound.
                self._below = version.__ge__
        elif inclusive:
            self._below = version.__ge__
        else:
            self._below = version.__gt__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UpperBound):
            return NotImplemented
        return self.version == other.version and self.inclusive == other.inclusive

    def __lt__(self, other: UpperBound) -> bool:
        if not isinstance(other, UpperBound):
            return NotImplemented
        # Nothing < +inf (except +inf itself).
        if self.version is None:
            return False
        if other.version is None:
            return True
        if self.version != other.version:
            return self.version < other.version
        # v) < v]: exclusive ends earlier.
        return not self.inclusive and other.inclusive

    def __hash__(self) -> int:
        return hash((self.version, self.inclusive))

    def __repr__(self) -> str:
        bracket = "]" if self.inclusive else ")"
        return f"<{self.__class__.__name__} {self.version!r}{bracket}>"


if TYPE_CHECKING:
    #: A single contiguous interval as a (lower, upper) bound pair.
    Interval = tuple[LowerBound, UpperBound]


NEG_INF: Final[LowerBound] = LowerBound(None, False)
POS_INF: Final[UpperBound] = UpperBound(None, False)
FULL_RANGE: Final[tuple[Interval]] = ((NEG_INF, POS_INF),)


def trim_release(release: tuple[int, ...]) -> tuple[int, ...]:
    """Strip trailing zeros from a release tuple for normalized comparison."""
    end = len(release)
    while end > 1 and release[end - 1] == 0:
        end -= 1
    return release if end == len(release) else release[:end]


def _next_prefix_dev0(version: Version) -> Version:
    """Smallest version in the next prefix: 1.2 -> 1.3.dev0."""
    release = (*version.release[:-1], version.release[-1] + 1)
    return Version.from_parts(epoch=version.epoch, release=release, dev=0)


def _base_dev0(version: Version) -> Version:
    """The .dev0 of a version's base release: 1.2 -> 1.2.dev0."""
    return Version.from_parts(epoch=version.epoch, release=version.release, dev=0)


def coerce_version(version: Version | str) -> Version | None:
    if not isinstance(version, Version):
        try:
            version = Version(version)
        except InvalidVersion:
            return None
    return version


def _make_above_after_posts(version: Version) -> Callable[[Version], bool]:
    """Predicate ``parsed > AFTER_POSTS(V)`` for a lower bound.

    Per PEP 440, ``>V`` excludes V's post-releases unless V is itself
    a post-release. AFTER_POSTS sits above V and every V.postN (with
    or without local), and just below the next release.
    """
    version_ge = version.__ge__
    version_epoch = version.epoch
    version_pre = version.pre
    version_release_trimmed = trim_release(version.release)
    trimmed_length = len(version_release_trimmed)

    def above(parsed: Version) -> bool:
        if version_ge(parsed):
            return False
        # parsed > V cmpkey-wise: above the boundary iff NOT in V's
        # post family.
        if parsed.epoch != version_epoch:
            return True
        parsed_release = parsed.release
        if len(parsed_release) < trimmed_length:
            return True
        if parsed_release[:trimmed_length] != version_release_trimmed:
            return True
        for i in range(trimmed_length, len(parsed_release)):
            if parsed_release[i] != 0:
                return True
        if parsed.pre != version_pre:
            return True

        # Same release and pre as V: parsed is in V's post family (V itself,
        # V+local, or V.postN), which the boundary sits above. A V.devN
        # (different dev, no post) sorts before V and was already caught by
        # ``version_ge`` above, so the answer here is always "not above".
        return False

    return above


def _make_above_after_locals(version: Version) -> Callable[[Version], bool]:
    """Predicate ``parsed > AFTER_LOCALS(V)`` for a lower bound.

    Used by the upper-side range of ``!=V`` (when V has no local
    segment). AFTER_LOCALS sits above V and every ``V+local`` but
    just below ``V.post0``.
    """
    version_ge = version.__ge__
    version_epoch = version.epoch
    version_pre = version.pre
    version_post = version.post
    version_dev = version.dev
    version_release_trimmed = trim_release(version.release)
    trimmed_length = len(version_release_trimmed)

    def above(parsed: Version) -> bool:
        if version_ge(parsed):
            return False
        # parsed > V cmpkey-wise: above the boundary iff NOT in V's
        # local family (same public version, any local segment).
        if parsed.epoch != version_epoch:
            return True
        parsed_release = parsed.release
        if len(parsed_release) < trimmed_length:
            return True
        if parsed_release[:trimmed_length] != version_release_trimmed:
            return True
        for i in range(trimmed_length, len(parsed_release)):
            if parsed_release[i] != 0:
                return True
        if parsed.pre != version_pre:
            return True
        if parsed.post != version_post:
            return True
        return parsed.dev != version_dev

    return above


def _make_below_after_locals(version: Version) -> Callable[[Version], bool]:
    """Predicate ``parsed <= AFTER_LOCALS(V)`` for an upper bound.

    Used by ``<=V``, ``==V``, ``!=V`` (no local). ``parsed`` is at or
    below the boundary when it is at or below V cmpkey-wise, or when
    it is in V's local family.
    """
    version_ge = version.__ge__
    version_epoch = version.epoch
    version_pre = version.pre
    version_post = version.post
    version_dev = version.dev
    version_release_trimmed = trim_release(version.release)
    trimmed_length = len(version_release_trimmed)

    def below(parsed: Version) -> bool:
        if version_ge(parsed):
            return True
        # parsed > V cmpkey-wise: below the boundary iff in V's local
        # family.
        if parsed.epoch != version_epoch:
            return False
        parsed_release = parsed.release
        if len(parsed_release) < trimmed_length:
            return False
        if parsed_release[:trimmed_length] != version_release_trimmed:
            return False
        for i in range(trimmed_length, len(parsed_release)):
            if parsed_release[i] != 0:
                return False
        if parsed.pre != version_pre:
            return False
        if parsed.post != version_post:
            return False
        return parsed.dev == version_dev

    return below


def least_version_above(boundary: BoundaryVersion) -> Version | None:
    """Smallest real version strictly above *boundary*, or ``None`` if none exists."""
    base = boundary.version

    if boundary.kind == BoundaryKind.AFTER_LOCALS:
        # AFTER_LOCALS(V) sits just below V.post0, so its least successor is
        # V.post0.dev0 (V.dev(N+1) if V has a dev, V.post(N+1).dev0 if a post).
        if base.dev is not None:
            return base.__replace__(dev=base.dev + 1, local=None)
        next_post = (base.post + 1) if base.post is not None else 0
        return base.__replace__(post=next_post, dev=0, local=None)

    # AFTER_POSTS(V): a pre-release V steps to the next pre-release's .dev0;
    # a final-release AFTER_POSTS has no least successor.
    if base.pre is not None:
        kind, number = base.pre
        return base.__replace__(pre=(kind, number + 1), post=None, dev=0, local=None)

    return None


def range_is_empty(lower: LowerBound, upper: UpperBound) -> bool:
    """True when the range defined by *lower* and *upper* contains no versions.

    A boundary lower sits just below the next real version, so an ordered pair
    is still empty when the upper excludes that least successor:
    ``(AFTER_POSTS(1.0a1), 1.0a2.dev0)`` holds no version.
    """
    if upper.version is None:
        return False

    if lower.version is None:
        # Nothing sorts below MIN_VERSION, so an exclusive upper at or below it
        # leaves an empty floor interval such as ``(-inf, 0.dev0)``.
        return (
            not upper.inclusive
            and isinstance(upper.version, Version)
            and upper.version <= MIN_VERSION
        )

    if isinstance(lower.version, BoundaryVersion):
        successor = least_version_above(lower.version)
        if successor is not None:
            if upper.version == successor:
                return not upper.inclusive
            return upper.version < successor

    if lower.version == upper.version:
        return not (lower.inclusive and upper.inclusive)

    return lower.version > upper.version


def intersect_ranges(
    left: Sequence[Interval],
    right: Sequence[Interval],
) -> list[Interval]:
    """Intersect two sorted, non-overlapping range lists (two-pointer merge)."""
    result: list[Interval] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_lower, left_upper = left[left_index]
        right_lower, right_upper = right[right_index]

        lower = max(left_lower, right_lower)
        upper = min(left_upper, right_upper)

        if not range_is_empty(lower, upper):
            result.append((lower, upper))

        # Advance whichever side has the smaller upper bound.
        if left_upper < right_upper:
            left_index += 1
        else:
            right_index += 1

    return result


def filter_by_ranges(
    ranges: Sequence[Interval],
    iterable: Iterable[Any],
    key: Callable[[Any], Version | str] | None,
    prereleases: bool | None,
    region: Sequence[Interval] = (),
) -> Iterator[Any]:
    """Filter *iterable* against precomputed version *ranges*.

    With ``prereleases=None``, the PEP 440 default applies: pre-releases are
    excluded unless no final matches, in which case buffered pre-releases come
    out at the end. A pre-release inside the opt-in ``region`` is the exception:
    it is force-admitted in place, as ``prereleases=True`` would yield it. A
    force-admitted pre-release is not a final, so it never suppresses the buffer.
    """
    if prereleases is None:
        prerelease_buffer: list[Any] = []
        found_final = False

        if len(ranges) == 1:
            # Hot path: most specifiers and small SpecifierSets reduce to
            # a single contiguous range.
            lower, upper = ranges[0]
            above = lower._above
            below = upper._below
            for item in iterable:
                parsed = coerce_version(item if key is None else key(item))
                if parsed is None:
                    continue
                if above is not None and not above(parsed):
                    continue
                if below is not None and not below(parsed):
                    continue
                if not parsed.is_prerelease:
                    found_final = True
                    yield item
                elif region and matches_bounds_only(region, parsed):
                    yield item
                elif not found_final:
                    prerelease_buffer.append(item)
            if not found_final:
                yield from prerelease_buffer
            return

        for item in iterable:
            parsed = coerce_version(item if key is None else key(item))
            if parsed is None:
                continue
            for lower, upper in ranges:
                above = lower._above
                if above is not None and not above(parsed):
                    break
                below = upper._below
                if below is None or below(parsed):
                    if not parsed.is_prerelease:
                        found_final = True
                        yield item
                    elif region and matches_bounds_only(region, parsed):
                        yield item
                    elif not found_final:
                        prerelease_buffer.append(item)
                    break
        if not found_final:
            yield from prerelease_buffer
        return

    exclude_prereleases = prereleases is False

    if len(ranges) == 1:
        # Hot path: most specifiers and small SpecifierSets reduce to
        # a single contiguous range.
        lower, upper = ranges[0]
        above = lower._above
        below = upper._below
        for item in iterable:
            parsed = coerce_version(item if key is None else key(item))
            if parsed is None:
                continue
            if exclude_prereleases and parsed.is_prerelease:
                continue
            if above is not None and not above(parsed):
                continue
            if below is None or below(parsed):
                yield item
        return

    for item in iterable:
        parsed = coerce_version(item if key is None else key(item))
        if parsed is None:
            continue
        if exclude_prereleases and parsed.is_prerelease:
            continue
        for lower, upper in ranges:
            above = lower._above
            if above is not None and not above(parsed):
                break
            below = upper._below
            if below is None or below(parsed):
                yield item
                break


def _nearest_release_above_prerelease(version: Version) -> Version:
    """Smallest non-pre-release at or above a pre-release *version*."""
    if version.pre is not None:
        # An a/b/rc pre-release drops to its final release, which outranks
        # every post-release of that pre-release (1.0a1.post0 -> 1.0).
        return version.__replace__(pre=None, post=None, dev=None, local=None)

    # A dev-only release keeps its post-release (1.0.post0.dev0 -> 1.0.post0,
    # whose final 1.0 sorts below it).
    return version.__replace__(dev=None, local=None)


def _lowest_release_at_or_above(value: Version | BoundaryVersion | None) -> Version:
    """Smallest non-pre-release version at or above *value*.

    ``None`` is the ``-inf`` floor, whose nearest non-pre-release is
    :data:`MIN_RELEASE`.
    """
    if value is None:
        return MIN_RELEASE
    if isinstance(value, BoundaryVersion):
        inner_version = value.version
        if inner_version.is_prerelease:
            return _nearest_release_above_prerelease(inner_version)
        # AFTER_LOCALS(1.0) -> nearest non-pre is 1.0.post0
        # AFTER_LOCALS(1.0.post0) -> nearest non-pre is 1.0.post1
        next_post = (inner_version.post + 1) if inner_version.post is not None else 0
        return inner_version.__replace__(post=next_post, local=None)

    if not value.is_prerelease:
        return value

    return _nearest_release_above_prerelease(value)


def ranges_are_prerelease_only(ranges: Sequence[Interval]) -> bool:
    """True when every range in *ranges* contains only pre-releases.

    Used to detect unsatisfiable specifier sets when ``prereleases=False``:
    if every range is pre-release-only, every contained version is excluded.
    """
    for lower, upper in ranges:
        nearest = _lowest_release_at_or_above(lower.version)
        if upper.version is None or nearest < upper.version:
            return False
        if nearest == upper.version and upper.inclusive:
            return False
    return True


def wildcard_ranges(op: str, base: Version) -> list[Interval]:
    """Ranges for ==V.* and !=V.*.

    ==1.2.* -> [1.2.dev0, 1.3.dev0);  !=1.2.* -> complement.
    """
    lower = _base_dev0(base)
    upper = _next_prefix_dev0(base)
    if op == "==":
        return [(LowerBound(lower, True), UpperBound(upper, False))]
    # !=
    return [
        (NEG_INF, UpperBound(lower, False)),
        (LowerBound(upper, True), POS_INF),
    ]


def standard_ranges(op: str, version: Version, has_local: bool) -> list[Interval]:
    """Ranges for the standard PEP 440 operators (no wildcard, no ===).

    *has_local* indicates whether the spec string included a ``+local``
    segment; relevant only for ``==`` / ``!=`` to decide whether the
    upper bound includes V's local family.
    """
    if op == ">=":
        return [(LowerBound(version, True), POS_INF)]

    if op == "<=":
        return [
            (
                NEG_INF,
                UpperBound(BoundaryVersion(version, BoundaryKind.AFTER_LOCALS), True),
            )
        ]

    if op == ">":
        if version.dev is not None:
            # >V.devN: dev versions have no post-releases, so the
            # next real version is V.dev(N+1).
            lower_bound = version.__replace__(dev=version.dev + 1, local=None)
            return [(LowerBound(lower_bound, True), POS_INF)]
        if version.post is not None:
            # >V.postN: next real version is V.post(N+1).dev0.
            lower_bound = version.__replace__(post=version.post + 1, dev=0, local=None)
            return [(LowerBound(lower_bound, True), POS_INF)]
        # >V (final or pre-release V): exclude V itself, V+local, and
        # every V.postN per PEP 440.
        return [
            (
                LowerBound(BoundaryVersion(version, BoundaryKind.AFTER_POSTS), False),
                POS_INF,
            )
        ]

    if op == "<":
        # <V excludes pre-releases of V when V is not a pre-release.
        # V.dev0 is the earliest pre-release of V.
        bound = (
            version if version.is_prerelease else version.__replace__(dev=0, local=None)
        )
        if bound <= MIN_VERSION:
            return []
        return [(NEG_INF, UpperBound(bound, False))]

    # ==, !=: local versions of V match when the spec has no local segment.
    after_locals = BoundaryVersion(version, BoundaryKind.AFTER_LOCALS)
    upper = version if has_local else after_locals

    if op == "==":
        return [(LowerBound(version, True), UpperBound(upper, True))]

    if op == "!=":
        return [
            (NEG_INF, UpperBound(version, False)),
            (LowerBound(upper, False), POS_INF),
        ]

    if op == "~=":
        prefix = version.__replace__(release=version.release[:-1])
        return [
            (LowerBound(version, True), UpperBound(_next_prefix_dev0(prefix), False))
        ]

    raise ValueError(f"Unknown operator: {op!r}")  # pragma: no cover


def bounds_for_spec(op: str, version_str: str, version: Version) -> list[Interval]:
    """Ranges for one specifier's ``(op, version_str)``.

    Dispatches between the wildcard and standard builders. ``version`` is the
    parsed ``version_str`` (its base, without the trailing ``.*``, for
    wildcards). ``===`` is not handled here; its match is a literal string
    compared in :mod:`packaging.specifiers`.
    """
    if version_str.endswith(".*"):
        return wildcard_ranges(op, version)

    return standard_ranges(op, version, "+" in version_str)


def intersect_specifier_bounds(
    per_specifier_ranges: Iterable[Sequence[Interval]],
) -> Sequence[Interval]:
    """Intersect each specifier's ranges into a single sequence.

    Short-circuits once the running intersection is empty, since no later
    specifier can revive it. Callers must pass at least one specifier.
    """
    result: Sequence[Interval] | None = None
    for sub in per_specifier_ranges:
        if result is None:
            result = sub
        else:
            result = intersect_ranges(result, sub)
            if not result:
                break

    if result is None:  # pragma: no cover - callers guard non-empty input
        raise RuntimeError("intersect_specifier_bounds called with no specifiers")

    return result


def matches_bounds_only(ranges: Sequence[Interval], version: Version) -> bool:
    """Whether ``version`` falls within any of ``ranges``.

    The pure bounds membership test, for a single already-parsed version with
    no pre-release policy applied. ``ranges`` are sorted and non-overlapping,
    so a version below one range's lower bound is below every later range too.
    """
    for lower, upper in ranges:
        above = lower._above
        if above is not None and not above(version):
            return False

        below = upper._below
        if below is None or below(version):
            return True

    return False


def resolve_prereleases(
    configured: bool | None, autodetected: bool | None
) -> bool | None:
    """Resolve a specifier's effective default pre-release policy.

    An explicit ``configured`` value wins; otherwise an autodetected ``True``
    propagates and anything else falls back to the PEP 440 default (``None``).
    """
    if configured is not None:
        return configured

    if autodetected:
        return True

    return None
<|endoftext|>from __future__ import annotations

import email.header
import email.message
import email.parser
import email.policy
import keyword
import pathlib
import re
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypedDict,
    cast,
)

from . import licenses, requirements, specifiers, utils
from . import version as version_module
from .errors import ExceptionGroup, _ErrorCollector

if typing.TYPE_CHECKING:
    from .licenses import NormalizedLicenseExpression
    from .version import Version

T = typing.TypeVar("T")


__all__ = [
    "ExceptionGroup",  # Keep this for a bit (makes mypy happy w/ 26.0 compat)
    "InvalidMetadata",
    "Metadata",
    "RFC822Message",
    "RFC822Policy",
    "RawMetadata",
    "parse_email",
]


def __dir__() -> list[str]:
    return __all__


class InvalidMetadata(ValueError):
    """A metadata field contains invalid data.

    .. versionadded:: 23.2
    """

    field: str
    """The name of the field that contains invalid data."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)

    # Support pickling; ``args`` does not match ``__init__``'s signature.
    def __reduce__(self) -> tuple[type[InvalidMetadata], tuple[str, str]]:
        return (self.__class__, (self.field, self.args[0]))


# The RawMetadata class attempts to make as few assumptions about the underlying
# serialization formats as possible. The idea is that as long as a serialization
# formats offer some very basic primitives in *some* way then we can support
# serializing to and from that format.
class RawMetadata(TypedDict, total=False):
    """A dictionary of raw core metadata.

    Each field in core metadata maps to a key of this dictionary (when data is
    provided). The key is lower-case and underscores are used instead of dashes
    compared to the equivalent core metadata field. Any core metadata field that
    can be specified multiple times or can hold multiple values in a single
    field have a key with a plural name. See :class:`Metadata` whose attributes
    match the keys of this dictionary.

    Core metadata fields that can be specified multiple times are stored as a
    list or dict depending on which is appropriate for the field. Any fields
    which hold multiple values in a single field are stored as a list. All fields
    are considered optional.
    """

    # Metadata 1.0 - PEP 241
    metadata_version: str
    name: str
    version: str
    platforms: list[str]
    summary: str
    description: str
    keywords: list[str]
    home_page: str
    author: str
    author_email: str
    license: str

    # Metadata 1.1 - PEP 314
    supported_platforms: list[str]
    download_url: str
    classifiers: list[str]
    requires: list[str]
    provides: list[str]
    obsoletes: list[str]

    # Metadata 1.2 - PEP 345
    maintainer: str
    maintainer_email: str
    requires_dist: list[str]
    provides_dist: list[str]
    obsoletes_dist: list[str]
    requires_python: str
    requires_external: list[str]
    project_urls: dict[str, str]

    # Metadata 2.0
    # PEP 426 attempted to completely revamp the metadata format
    # but got stuck without ever being able to build consensus on
    # it and ultimately ended up withdrawn.
    #
    # However, a number of tools had started emitting METADATA with
    # `2.0` Metadata-Version, so for historical reasons, this version
    # was skipped.

    # Metadata 2.1 - PEP 566
    description_content_type: str
    provides_extra: list[str]

    # Metadata 2.2 - PEP 643
    dynamic: list[str]

    # Metadata 2.3 - PEP 685
    # No new fields were added in PEP 685, just some edge case were
    # tightened up to provide better interoperability.

    # Metadata 2.4 - PEP 639
    license_expression: str
    """.. versionadded:: 24.2"""
    license_files: list[str]
    """.. versionadded:: 24.2"""

    # Metadata 2.5 - PEP 794
    import_names: list[str]
    """.. versionadded:: 26.0"""
    import_namespaces: list[str]
    """.. versionadded:: 26.0"""

    # Metadata 2.6 - PEP 808 (no new fields, behavior change for Dynamic)


# 'keywords' is special as it's a string in the core metadata spec, but we
# represent it as a list.
_STRING_FIELDS = {
    "author",
    "author_email",
    "description",
    "description_content_type",
    "download_url",
    "home_page",
    "license",
    "license_expression",
    "maintainer",
    "maintainer_email",
    "metadata_version",
    "name",
    "requires_python",
    "summary",
    "version",
}

_LIST_FIELDS = {
    "classifiers",
    "dynamic",
    "license_files",
    "obsoletes",
    "obsoletes_dist",
    "platforms",
    "provides",
    "provides_dist",
    "provides_extra",
    "requires",
    "requires_dist",
    "requires_external",
    "supported_platforms",
    "import_names",
    "import_namespaces",
}

_DICT_FIELDS = {
    "project_urls",
}


def _parse_keywords(data: str) -> list[str]:
    """Split a string of comma-separated keywords into a list of keywords."""
    return [k.strip() for k in data.split(",")]


def _parse_project_urls(data: list[str]) -> dict[str, str]:
    """Parse a list of label/URL string pairings separated by a comma."""
    urls = {}
    for pair in data:
        # Our logic is slightly tricky here as we want to try and do
        # *something* reasonable with malformed data.
        #
        # The main thing that we have to worry about, is data that does
        # not have a ',' at all to split the label from the Value. There
        # isn't a singular right answer here, and we will fail validation
        # later on (if the caller is validating) so it doesn't *really*
        # matter, but since the missing value has to be an empty str
        # and our return value is dict[str, str], if we let the key
        # be the missing value, then they'd have multiple '' values that
        # overwrite each other in a accumulating dict.
        #
        # The other potential issue is that it's possible to have the
        # same label multiple times in the metadata, with no solid "right"
        # answer with what to do in that case. As such, we'll do the only
        # thing we can, which is treat the field as unparsable and add it
        # to our list of unparsed fields.
        #
        # TODO: The spec doesn't say anything about if the keys should be
        #       considered case sensitive or not... logically they should
        #       be case-preserving and case-insensitive, but doing that
        #       would open up more cases where we might have duplicate
        #       entries.
        label, _, url = (s.strip() for s in pair.partition(","))

        if label in urls:
            # The label already exists in our set of urls, so this field
            # is unparsable, and we can just add the whole thing to our
            # unparsable data and stop processing it.
            raise KeyError("duplicate labels in project urls")
        urls[label] = url

    return urls


def _get_payload(msg: email.message.Message, source: bytes | str) -> str:
    """Get the body of the message."""
    # If our source is a str, then our caller has managed encodings for us,
    # and we don't need to deal with it.
    if isinstance(source, str):
        payload = msg.get_payload()
        # A multipart payload makes get_payload() return a list of messages
        # rather than a str; route it to ``unparsed``.
        if not isinstance(payload, str):
            raise ValueError("payload is not a string")  # noqa: TRY004
        return payload
    # If our source is a bytes, then we're managing the encoding and we need
    # to deal with it.
    else:
        bpayload = msg.get_payload(decode=True)
        # A multipart payload makes get_payload(decode=True) return None;
        # route it to ``unparsed``.
        if not isinstance(bpayload, bytes):
            raise ValueError("payload in an invalid encoding")  # noqa: TRY004
        try:
            return bpayload.decode("utf8", "strict")
        except UnicodeDecodeError as exc:
            raise ValueError("payload in an invalid encoding") from exc


# The various parse_FORMAT functions here are intended to be as lenient as
# possible in their parsing, while still returning a correctly typed
# RawMetadata.
#
# To aid in this, we also generally want to do as little touching of the
# data as possible, except where there are possibly some historic holdovers
# that make valid data awkward to work with.
#
# While this is a lower level, intermediate format than our ``Metadata``
# class, some light touch ups can make a massive difference in usability.

# Map METADATA fields to RawMetadata.
_EMAIL_TO_RAW_MAPPING = {
    "author": "author",
    "author-email": "author_email",
    "classifier": "classifiers",
    "description": "description",
    "description-content-type": "description_content_type",
    "download-url": "download_url",
    "dynamic": "dynamic",
    "home-page": "home_page",
    "import-name": "import_names",
    "import-namespace": "import_namespaces",
    "keywords": "keywords",
    "license": "license",
    "license-expression": "license_expression",
    "license-file": "license_files",
    "maintainer": "maintainer",
    "maintainer-email": "maintainer_email",
    "metadata-version": "metadata_version",
    "name": "name",
    "obsoletes": "obsoletes",
    "obsoletes-dist": "obsoletes_dist",
    "platform": "platforms",
    "project-url": "project_urls",
    "provides": "provides",
    "provides-dist": "provides_dist",
    "provides-extra": "provides_extra",
    "requires": "requires",
    "requires-dist": "requires_dist",
    "requires-external": "requires_external",
    "requires-python": "requires_python",
    "summary": "summary",
    "supported-platform": "supported_platforms",
    "version": "version",
}
_RAW_TO_EMAIL_MAPPING = {raw: email for email, raw in _EMAIL_TO_RAW_MAPPING.items()}


# A bare "\r" makes the email generator raise ``HeaderWriteError``, and on
# CPython releases without the CVE-2024-6923 fix any ``str.splitlines``
# boundary ends the header line, so fold all of them, not just "\n".
_LINE_BOUNDARY_RE = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")


# This class is for writing RFC822 messages
class RFC822Policy(email.policy.EmailPolicy):
    """
    This is :class:`email.policy.EmailPolicy`, but with a simple ``header_store_parse``
    implementation that handles multi-line values, and some nice defaults.

    .. versionadded:: 26.0
    """

    utf8 = True
    mangle_from_ = False
    max_line_length = 0

    def header_store_parse(self, name: str, value: str) -> tuple[str, str]:
        size = len(name) + 2
        value = _LINE_BOUNDARY_RE.sub("\n" + " " * size, value)
        return (name, value)


# This class is for writing RFC822 messages
class RFC822Message(email.message.EmailMessage):
    """
    This is :class:`email.message.EmailMessage` with two small changes: it defaults to
    our `RFC822Policy`, and it correctly writes unicode when being called
    with `bytes()`.

    .. versionadded:: 26.0
    """

    def __init__(self) -> None:
        super().__init__(policy=RFC822Policy())

    def as_bytes(
        self, unixfrom: bool = False, policy: email.policy.Policy | None = None
    ) -> bytes:
        """
        Return the bytes representation of the message.

        This handles unicode encoding.
        """
        return self.as_string(unixfrom, policy=policy).encode("utf-8")


def parse_email(data: bytes | str) -> tuple[RawMetadata, dict[str, list[str]]]:
    """Parse a distribution's metadata stored as email headers (e.g. from ``METADATA``).

    This function returns a two-item tuple of dicts. The first dict is of
    recognized fields from the core metadata specification. Fields that can be
    parsed and translated into Python's built-in types are converted
    appropriately. All other fields are left as-is. Fields that are allowed to
    appear multiple times are stored as lists.

    The second dict contains all other fields from the metadata. This includes
    any unrecognized fields. It also includes any fields which are expected to
    be parsed into a built-in type but were not formatted appropriately. Finally,
    any fields that are expected to appear only once but are repeated are
    included in this dict.

    """
    raw: dict[str, str | list[str] | dict[str, str]] = {}
    unparsed: dict[str, list[str]] = {}

    if isinstance(data, str):
        parsed = email.parser.Parser(policy=email.policy.compat32).parsestr(data)
    else:
        parsed = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)

    # We have to wrap parsed.keys() in a set, because in the case of multiple
    # values for a key (a list), the key will appear multiple times in the
    # list of keys, but we're avoiding that by using get_all().
    for name_with_case in frozenset(parsed.keys()):
        # Header names in RFC are case insensitive, so we'll normalize to all
        # lower case to make comparisons easier.
        name = name_with_case.lower()

        # We use get_all() here, even for fields that aren't multiple use,
        # because otherwise someone could have e.g. two Name fields, and we
        # would just silently ignore it rather than doing something about it.
        headers = parsed.get_all(name) or []

        # The way the email module works when parsing bytes is that it
        # unconditionally decodes the bytes as ascii using the surrogateescape
        # handler. When you pull that data back out (such as with get_all() ),
        # it looks to see if the str has any surrogate escapes, and if it does
        # it wraps it in a Header object instead of returning the string.
        #
        # As such, we'll look for those Header objects, and fix up the encoding.
        value = []
        # Flag if we have run into any issues processing the headers, thus
        # signalling that the data belongs in 'unparsed'.
        valid_encoding = True
        for h in headers:
            # It's unclear if this can return more types than just a Header or
            # a str, so we'll just assert here to make sure.
            assert isinstance(h, (email.header.Header, str))

            # If it's a header object, we need to do our little dance to get
            # the real data out of it. In cases where there is invalid data
            # we're going to end up with mojibake, but there's no obvious, good
            # way around that without reimplementing parts of the Header object
            # ourselves.
            #
            # That should be fine since, if mojibacked happens, this key is
            # going into the unparsed dict anyways.
            if isinstance(h, email.header.Header):
                # The Header object stores it's data as chunks, and each chunk
                # can be independently encoded, so we'll need to check each
                # of them.
                chunks: list[tuple[bytes, str | None]] = []
                for binary, _encoding in email.header.decode_header(h):
                    try:
                        binary.decode("utf8", "strict")
                    except UnicodeDecodeError:
                        # Enable mojibake.
                        encoding = "latin1"
                        valid_encoding = False
                    else:
                        encoding = "utf8"
                    chunks.append((binary, encoding))

                # Turn our chunks back into a Header object, then let that
                # Header object do the right thing to turn them into a
                # string for us.
                value.append(str(email.header.make_header(chunks)))
            # This is already a string, so just add it.
            else:
                value.append(h)

        # We've processed all of our values to get them into a list of str,
        # but we may have mojibake data, in which case this is an unparsed
        # field.
        if not valid_encoding:
            unparsed[name] = value
            continue

        raw_name = _EMAIL_TO_RAW_MAPPING.get(name)
        if raw_name is None:
            # This is a bit of a weird situation, we've encountered a key that
            # we don't know what it means, so we don't know whether it's meant
            # to be a list or not.
            #
            # Since we can't really tell one way or another, we'll just leave it
            # as a list, even though it may be a single item list, because that's
            # what makes the most sense for email headers.
            unparsed[name] = value
            continue

        # If this is one of our string fields, then we'll check to see if our
        # value is a list of a single item. If it is then we'll assume that
        # it was emitted as a single string, and unwrap the str from inside
        # the list.
        #
        # If it's any other kind of data, then we haven't the faintest clue
        # what we should parse it as, and we have to just add it to our list
        # of unparsed stuff.
        if raw_name in _STRING_FIELDS and len(value) == 1:
            raw[raw_name] = value[0]
        # If this is import_names, we need to special case the empty field
        # case, which converts to an empty list instead of None. We can't let
        # the empty case slip through, as it will fail validation.
        elif raw_name == "import_names" and value == [""]:
            raw[raw_name] = []
        # If this is one of our list of string fields, then we can just assign
        # the value, since email *only* has strings, and our get_all() call
        # above ensures that this is a list.
        elif raw_name in _LIST_FIELDS:
            raw[raw_name] = value
        # Special Case: Keywords
        # The keywords field is implemented in the metadata spec as a str,
        # but it conceptually is a list of strings, and is serialized using
        # ", ".join(keywords), so we'll do some light data massaging to turn
        # this into what it logically is.
        elif raw_name == "keywords" and len(value) == 1:
            raw[raw_name] = _parse_keywords(value[0])
        # Special Case: Project-URL
        # The project urls is implemented in the metadata spec as a list of
        # specially-formatted strings that represent a key and a value, which
        # is fundamentally a mapping, however the email format doesn't support
        # mappings in a sane way, so it was crammed into a list of strings
        # instead.
        #
        # We will do a little light data massaging to turn this into a map as
        # it logically should be.
        elif raw_name == "project_urls":
            try:
                raw[raw_name] = _parse_project_urls(value)
            except KeyError:
                unparsed[name] = value
        # Nothing that we've done has managed to parse this, so it'll just
        # throw it in our unparsable data and move on.
        else:
            unparsed[name] = value

    # We need to support getting the Description from the message payload in
    # addition to getting it from the the headers. This does mean, though, there
    # is the possibility of it being set both ways, in which case we put both
    # in 'unparsed' since we don't know which is right.
    try:
        payload = _get_payload(parsed, data)
    except ValueError:
        unparsed.setdefault("description", []).append(
            parsed.get_payload(decode=isinstance(data, bytes))  # type: ignore[call-overload]
        )
    else:
        if payload:
            # Check to see if we've already got a description, if so then both
            # it, and this body move to unparsable.
            if "description" in raw:
                description_header = cast("str", raw.pop("description"))
                unparsed.setdefault("description", []).extend(
                    [description_header, payload]
                )
            elif "description" in unparsed:
                unparsed["description"].append(payload)
            else:
                raw["description"] = payload

    # We need to cast our `raw` to a metadata, because a TypedDict only support
    # literal key names, but we're computing our key names on purpose, but the
    # way this function is implemented, our `TypedDict` can only have valid key
    # names.
    return cast("RawMetadata", raw), unparsed


_NOT_FOUND = object()


# Keep the two values in sync.
_VALID_METADATA_VERSIONS = [
    "1.0",
    "1.1",
    "1.2",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
]
_MetadataVersion = Literal[
    "1.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6"
]

_REQUIRED_ATTRS = frozenset(["metadata_version", "name", "version"])


class _Validator(Generic[T]):
    """Validate a metadata field.

    All _process_*() methods correspond to a core metadata field. The method is
    called with the field's raw value. If the raw value is valid it is returned
    in its "enriched" form (e.g. ``version.Version`` for the ``Version`` field).
    If the raw value is invalid, :exc:`InvalidMetadata` is raised (with a cause
    as appropriate).
    """

    name: str
    raw_name: str
    added: _MetadataVersion

    def __init__(
        self,
        *,
        added: _MetadataVersion = "1.0",
    ) -> None:
        self.added = added

    def __set_name__(self, _owner: Metadata, name: str) -> None:
        self.name = name
        self.raw_name = _RAW_TO_EMAIL_MAPPING[name]

    def __get__(self, instance: Metadata, _owner: type[Metadata]) -> T:
        # With Python 3.8, the caching can be replaced with functools.cached_property().
        # No need to check the cache as attribute lookup will resolve into the
        # instance's __dict__ before __get__ is called.
        cache = instance.__dict__
        value = instance._raw.get(self.name)

        # To make the _process_* methods easier, we'll check if the value is None
        # and if this field is NOT a required attribute, and if both of those
        # things are true, we'll skip the the converter. This will mean that the
        # converters never have to deal with the None union.
        if self.name in _REQUIRED_ATTRS or value is not None:
            try:
                converter: Callable[[Any], T] = getattr(self, f"_process_{self.name}")
            except AttributeError:
                pass
            else:
                value = converter(value)

        cache[self.name] = value
        try:
            del instance._raw[self.name]  # type: ignore[misc]
        except KeyError:
            pass

        return cast("T", value)

    def _invalid_metadata(
        self, msg: str, cause: Exception | None = None
    ) -> InvalidMetadata:
        exc = InvalidMetadata(self.raw_name, msg)
        exc.__cause__ = cause
        return exc

    def _process_metadata_version(self, value: str) -> _MetadataVersion:
        # Implicitly makes Metadata-Version required.
        if value not in _VALID_METADATA_VERSIONS:
            raise self._invalid_metadata(f"{value!r} is not a valid metadata version")
        return cast("_MetadataVersion", value)

    def _process_name(self, value: str) -> str:
        if not value:
            raise self._invalid_metadata(f"{self.raw_name!r} is a required field")
        # Validate the name as a side-effect.
        try:
            utils.canonicalize_name(value, validate=True)
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc
        else:
            return value

    def _process_version(self, value: str) -> Version:
        if not value:
            raise self._invalid_metadata(f"{self.raw_name!r} is a required field")
        try:
            return version_module.parse(value)
        except version_module.InvalidVersion as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc

    def _process_summary(self, value: str) -> str:
        """Check the field contains no line breaks."""
        if _LINE_BOUNDARY_RE.search(value):
            raise self._invalid_metadata(f"{self.raw_name!r} must be a single line")
        return value

    def _process_description_content_type(self, value: str) -> str:
        content_types = {"text/plain", "text/x-rst", "text/markdown"}
        invalid_msg = (
            f"{self.raw_name!r} must be one of {list(content_types)}, not {value!r}"
        )
        message = email.message.EmailMessage()
        try:
            message["content-type"] = value
        # The email parser can raise IndexError on malformed RFC 2231
        # parameters such as "text/plain; x*".
        except (ValueError, IndexError) as exc:
            msg = f"{value!r} is not a valid content type for {self.raw_name!r}"
            raise self._invalid_metadata(msg, cause=exc) from exc
        content_type_header = message["content-type"]
        if content_type_header.defects:
            defect = content_type_header.defects[0]
            msg = (
                f"{value!r} is not a valid content type for {self.raw_name!r}: {defect}"
            )
            raise self._invalid_metadata(msg, cause=defect) from defect

        content_type, parameters = (
            # Defaults to `text/plain` if parsing failed.
            message.get_content_type().lower(),
            content_type_header.params,
        )
        # Check if content-type is valid or defaulted to `text/plain` and thus was
        # not parseable.
        if content_type not in content_types or content_type not in value.lower():
            raise self._invalid_metadata(invalid_msg)

        charset = parameters.get("charset", "UTF-8")
        if charset.lower() != "utf-8":
            raise self._invalid_metadata(
                f"{self.raw_name!r} can only specify the UTF-8 charset, not {charset!r}"
            )

        markdown_variants = {"GFM", "CommonMark"}
        variant = parameters.get("variant", "GFM")  # Use an acceptable default.
        if content_type == "text/markdown" and variant not in markdown_variants:
            raise self._invalid_metadata(
                f"valid Markdown variants for {self.raw_name!r} are "
                f"{list(markdown_variants)}, not {variant!r}",
            )
        return value

    def _process_dynamic(self, value: list[str]) -> list[str]:
        dynamic_fields = list(map(str.lower, value))
        for dynamic_field in dynamic_fields:
            if dynamic_field in {"name", "version", "metadata-version"}:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not allowed as a dynamic field"
                )
            elif dynamic_field not in _EMAIL_TO_RAW_MAPPING:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not a valid dynamic field"
                )
        return dynamic_fields

    def _process_provides_extra(
        self,
        value: list[str],
    ) -> list[utils.NormalizedName]:
        normalized_names = []
        try:
            for name in value:
                normalized_names.append(utils.canonicalize_name(name, validate=True))
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{name!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc
        else:
            return normalized_names

    def _process_requires_python(self, value: str) -> specifiers.SpecifierSet:
        try:
            return specifiers.SpecifierSet(value)
        except specifiers.InvalidSpecifier as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc

    def _process_requires_dist(
        self,
        value: list[str],
    ) -> list[requirements.Requirement]:
        reqs = []
        try:
            for req in value:
                reqs.append(requirements.Requirement(req))
        except requirements.InvalidRequirement as exc:
            raise self._invalid_metadata(
                f"{req!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc
        else:
            return reqs

    def _process_license_expression(self, value: str) -> NormalizedLicenseExpression:
        try:
            return licenses.canonicalize_license_expression(value)
        except ValueError as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {self.raw_name!r}", cause=exc
            ) from exc

    def _process_license_files(self, value: list[str]) -> list[str]:
        paths = []
        for path in value:
            if ".." in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {self.raw_name!r}, "
                    "parent directory indicators are not allowed"
                )
            if "*" in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {self.raw_name!r}, paths must be resolved"
                )
            if (
                pathlib.PurePosixPath(path).is_absolute()
                or pathlib.PureWindowsPath(path).is_absolute()
            ):
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {self.raw_name!r}, paths must be relative"
                )
            if pathlib.PureWindowsPath(path).as_posix() != path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {self.raw_name!r}, "
                    "paths must use '/' delimiter"
                )
            paths.append(path)
        return paths

    def _process_import_names(self, value: list[str]) -> list[str]:
        for import_name in value:
            name, semicolon, private = import_name.partition(";")
            name = name.rstrip()
            for identifier in name.split("."):
                if not identifier.isidentifier():
                    raise self._invalid_metadata(
                        f"{name!r} is invalid for {self.raw_name!r}; "
                        f"{identifier!r} is not a valid identifier"
                    )
                elif keyword.iskeyword(identifier):
                    raise self._invalid_metadata(
                        f"{name!r} is invalid for {self.raw_name!r}; "
                        f"{identifier!r} is a keyword"
                    )
            if semicolon and private.lstrip() != "private":
                raise self._invalid_metadata(
                    f"{import_name!r} is invalid for {self.raw_name!r}; "
                    "the only valid option is 'private'"
                )
        return value

    _process_import_namespaces = _process_import_names


class Metadata:
    """Representation of distribution metadata.

    Compared to :class:`RawMetadata`, this class provides objects representing
    metadata fields instead of only using built-in types. Any invalid metadata
    will cause :exc:`InvalidMetadata` to be raised (with a
    :py:attr:`~BaseException.__cause__` attribute as appropriate).

    .. versionadded:: 23.2

    .. versionchanged:: 24.0
        Optional attributes now return None when the field is absent instead of
        raising.
    """

    _raw: RawMetadata

    @classmethod
    def from_raw(cls, data: RawMetadata, *, validate: bool = True) -> Metadata:
        """Create an instance from :class:`RawMetadata`.

        If *validate* is true, all metadata will be validated and all related
        exceptions will be gathered and raised as an :class:`ExceptionGroup`;
        otherwise, validation happens per attribute when it is accessed.
        """
        ins = cls()
        ins._raw = data.copy()  # Mutations occur due to caching enriched values.

        if validate:
            collector = _ErrorCollector()
            metadata_version = None
            with collector.collect(InvalidMetadata):
                metadata_version = ins.metadata_version
                metadata_age = _VALID_METADATA_VERSIONS.index(metadata_version)

            # Make sure to check for the fields that are present, the required
            # fields (so their absence can be reported).
            fields_to_check = frozenset(ins._raw) | _REQUIRED_ATTRS
            # Remove fields that have already been checked.
            fields_to_check -= {"metadata_version"}

            for key in fields_to_check:
                try:
                    if metadata_version:
                        # Can't use getattr() as that triggers descriptor protocol which
                        # will fail due to no value for the instance argument.
                        try:
                            field_metadata_version = cls.__dict__[key].added
                        except KeyError:
                            exc = InvalidMetadata(key, f"unrecognized field: {key!r}")
                            collector.error(exc)
                            continue
                        field_age = _VALID_METADATA_VERSIONS.index(
                            field_metadata_version
                        )
                        if field_age > metadata_age:
                            field = _RAW_TO_EMAIL_MAPPING[key]
                            exc = InvalidMetadata(
                                field,
                                f"{field} introduced in metadata version "
                                f"{field_metadata_version}, not {metadata_version}",
                            )
                            collector.error(exc)
                            continue
                    getattr(ins, key)
                except InvalidMetadata as exc:
                    collector.error(exc)

            collector.finalize("invalid metadata")

        return ins

    @classmethod
    def from_email(cls, data: bytes | str, *, validate: bool = True) -> Metadata:
        """Parse metadata from email headers.

        If *validate* is true, the metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        raw, unparsed = parse_email(data)

        if validate:
            with _ErrorCollector().on_exit("invalid or unparsed metadata") as collector:
                for unparsed_key in unparsed:
                    if unparsed_key in _EMAIL_TO_RAW_MAPPING:
                        message = f"{unparsed_key!r} has invalid data"
                    else:
                        message = f"unrecognized field: {unparsed_key!r}"
                    collector.error(InvalidMetadata(unparsed_key, message))
                try:
                    validated = cls.from_raw(raw, validate=validate)
                except ExceptionGroup as exc_group:
                    # The no-branch pragmas cover arcs only seen by Python 3.9.
                    for exc in exc_group.exceptions:  # pragma: no branch
                        # A required field reported above as unparsed is absent
                        # from `raw`, so skip from_raw's duplicate "missing"
                        # complaint.
                        if not (
                            isinstance(exc, InvalidMetadata)
                            and exc.field in unparsed
                            and _EMAIL_TO_RAW_MAPPING.get(exc.field) not in raw
                        ):
                            collector.error(exc)
                else:
                    if not collector.errors:  # pragma: no branch
                        return validated

        return cls.from_raw(raw, validate=validate)

    metadata_version: _Validator[_MetadataVersion] = _Validator()
    """:external:ref:`core-metadata-metadata-version`
    (required; validated to be a valid metadata version)"""
    # `name` is not normalized/typed to NormalizedName so as to provide access to
    # the original/raw name.
    name: _Validator[str] = _Validator()
    """:external:ref:`core-metadata-name`
    (required; validated using :func:`~packaging.utils.canonicalize_name` and its
    *validate* parameter)"""
    version: _Validator[Version] = _Validator()
    """:external:ref:`core-metadata-version` (required)"""
    dynamic: _Validator[list[str] | None] = _Validator(
        added="2.2",
    )
    """:external:ref:`core-metadata-dynamic`
    (validated against core metadata field names and lowercased)"""
    platforms: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-platform`"""
    supported_platforms: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-supported-platform`"""
    summary: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-summary` (validated to contain no newlines)"""
    description: _Validator[str | None] = _Validator()  # TODO 2.1: can be in body
    """:external:ref:`core-metadata-description`"""
    description_content_type: _Validator[str | None] = _Validator(added="2.1")
    """:external:ref:`core-metadata-description-content-type` (validated)"""
    keywords: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-keywords`"""
    home_page: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-home-page`"""
    download_url: _Validator[str | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-download-url`"""
    author: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author`"""
    author_email: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author-email`"""
    maintainer: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer`"""
    maintainer_email: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer-email`"""
    license: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-license`"""
    license_expression: _Validator[NormalizedLicenseExpression | None] = _Validator(
        added="2.4"
    )
    """:external:ref:`core-metadata-license-expression`

    .. versionadded:: 24.2
    """
    license_files: _Validator[list[str] | None] = _Validator(added="2.4")
    """:external:ref:`core-metadata-license-file`

    .. versionadded:: 24.2
    """
    classifiers: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-classifier`"""
    requires_dist: _Validator[list[requirements.Requirement] | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-dist`"""
    requires_python: _Validator[specifiers.SpecifierSet | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-python`"""
    # Because `Requires-External` allows for non-PEP 440 version specifiers, we
    # don't do any processing on the values.
    requires_external: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-requires-external`"""
    project_urls: _Validator[dict[str, str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-project-url`"""
    # PEP 685 lets us raise an error if an extra doesn't pass `Name` validation
    # regardless of metadata version.
    provides_extra: _Validator[list[utils.NormalizedName] | None] = _Validator(
        added="2.1",
    )
    """:external:ref:`core-metadata-provides-extra`"""
    provides_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-provides-dist`"""
    obsoletes_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-obsoletes-dist`"""
    import_names: _Validator[list[str] | None] = _Validator(added="2.5")
    """:external:ref:`core-metadata-import-name`

    .. versionadded:: 26.0
    """
    import_namespaces: _Validator[list[str] | None] = _Validator(added="2.5")
    """:external:ref:`core-metadata-import-namespace`

    .. versionadded:: 26.0
    """
    requires: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Requires`` (deprecated)"""
    provides: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Provides`` (deprecated)"""
    obsoletes: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Obsoletes`` (deprecated)"""

    def as_rfc822(self) -> RFC822Message:
        """
        Return an RFC822 message with the metadata.

        .. versionadded:: 26.0
        """
        message = RFC822Message()
        self._write_metadata(message)
        return message

    def _write_metadata(self, message: RFC822Message) -> None:
        """
        Return an RFC822 message with the metadata.
        """
        for name, validator in self.__class__.__dict__.items():
            if isinstance(validator, _Validator) and name != "description":
                value = getattr(self, name)
                email_name = _RAW_TO_EMAIL_MAPPING[name]
                if value is not None:
                    if email_name == "project-url":
                        for label, url in value.items():
                            message[email_name] = f"{label}, {url}"
                    elif email_name == "keywords":
                        message[email_name] = ",".join(value)
                    elif email_name == "import-name" and value == []:
                        message[email_name] = ""
                    elif isinstance(value, list):
                        for item in value:
                            message[email_name] = str(item)
                    else:
                        message[email_name] = str(value)

        # The description is a special case because it is in the body of the message.
        if self.description is not None:
            message.set_payload(self.description)
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from packaging.specifiers import Specifier, SpecifierSet, InvalidSpecifier
    from packaging.version import Version
"""

from __future__ import annotations

import abc
import re
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    TypeVar,
    Union,
)

from ._ranges import (
    FULL_RANGE,
    bounds_for_spec,
    coerce_version,
    filter_by_ranges,
    intersect_specifier_bounds,
    matches_bounds_only,
    ranges_are_prerelease_only,
    resolve_prereleases,
    trim_release,
)
from .utils import canonicalize_version
from .version import Version

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable, Iterator, Sequence

    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from . import ranges
    from ._ranges import Interval


__all__ = [
    "BaseSpecifier",
    "InvalidSpecifier",
    "Specifier",
    "SpecifierSet",
]


def __dir__() -> list[str]:
    return __all__


def _validate_spec(spec: object, /) -> TypeGuard[tuple[str, str]]:
    return (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and isinstance(spec[1], str)
    )


def _validate_pre(pre: object, /) -> TypeGuard[bool | None]:
    return pre is None or isinstance(pre, bool)


T = TypeVar("T")
UnparsedVersion = Union[Version, str]
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)


# Operators whose result is just a direct Version comparison, given a parsed
# item with no local. ``<=``/``==``/``!=`` need that no-local guard because
# PEP 440 strips locals on those; ``>=`` works regardless.
_DIRECT_COMPARE_OPS: dict[str, Callable[[Version, Version], bool]] = {
    ">=": Version.__ge__,
    "<=": Version.__le__,
    "==": Version.__eq__,
    "!=": Version.__ne__,
}


def _fast_match(specifier: Specifier, parsed: Version) -> bool | None:
    """Match ``parsed`` against ``specifier`` without building a range.

    Handles ``>=``, ``<=``, ``==``, ``!=``, ``<``, ``>`` when the spec is
    not a wildcard and ``parsed`` has no local. Returns ``None`` when the
    range path must be used. Pre-release policy is left to the caller.
    """
    op_str, ver_str = specifier._spec
    if ver_str.endswith(".*") or parsed.local is not None:
        return None

    direct_compare = _DIRECT_COMPARE_OPS.get(op_str)
    if direct_compare is not None:
        return direct_compare(parsed, specifier._require_spec_version(ver_str))

    if op_str in ("<", ">"):
        spec_v = specifier._require_spec_version(ver_str)
        # ``<V``/``>V`` carve out V's family (pre/dev/post); that only
        # matters when parsed shares V's epoch and trimmed release.
        # Otherwise a direct cmpkey comparison is correct.
        if parsed.epoch != spec_v.epoch or trim_release(parsed.release) != trim_release(
            spec_v.release
        ):
            return parsed < spec_v if op_str == "<" else parsed > spec_v
        return None

    return None


class InvalidSpecifier(ValueError):
    """
    Raised when attempting to create a :class:`Specifier` with a specifier
    string that is invalid.

    >>> Specifier("lolwat")
    Traceback (most recent call last):
        ...
    packaging.specifiers.InvalidSpecifier: Invalid specifier: 'lolwat'
    """


class BaseSpecifier(metaclass=abc.ABCMeta):
    """
    Abstract base class for :class:`Specifier` and :class:`SpecifierSet`.
    """

    __slots__ = ()
    __match_args__ = ("_str",)

    @property
    def _str(self) -> str:
        """Internal property for match_args"""
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the str representation of this Specifier-like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash value for this Specifier-like object.
        """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean representing whether or not the two Specifier-like
        objects are equal.

        :param other: The other object to check against.
        """

    @property
    @abc.abstractmethod
    def prereleases(self) -> bool | None:
        """Whether or not pre-releases as a whole are allowed.

        This can be set to either ``True`` or ``False`` to explicitly enable or disable
        prereleases or it can be set to ``None`` (the default) to use default semantics.
        """

    @prereleases.setter  # noqa: B027
    def prereleases(self, value: bool) -> None:
        """Setter for :attr:`prereleases`.

        :param value: The value to set.
        """

    @abc.abstractmethod
    def contains(self, item: str, prereleases: bool | None = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """

    @typing.overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @typing.overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    @abc.abstractmethod
    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class Specifier(BaseSpecifier):
    """This class abstracts handling of version specifiers.

    .. tip::

        It is generally not required to instantiate this manually. You should instead
        prefer to work with :class:`SpecifierSet` instead, which can parse
        comma-separated version specifiers (which is what package metadata contains).

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = (
        "_prereleases",
        "_ranges",
        "_spec",
        "_spec_version",
    )

    _specifier_regex_str = r"""
        (?:
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                ===  # Only match for the identity operator
                \s*
                [^\s;)]*  # The arbitrary version can be just about anything,
                          # we match everything except for whitespace, a
                          # semi-colon for marker support, and a closing paren
                          # since versions can be enclosed in them.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?:==|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release

                # You cannot use a wild card and a pre-release, post-release, a dev or
                # local version together so group them with a | and make them optional.
                (?:
                    \.\*  # Wild card syntax of .*
                    |
                    (?a:                                  # pre release
                        [-_\.]?
                        (alpha|beta|preview|pre|a|b|c|rc)
                        [-_\.]?
                        [0-9]*
                    )?
                    (?a:                                  # post release
                        (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                    )?
                    (?a:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?a:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?:~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?:<=|>=|<|>)

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?a:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?a:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?a:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex = re.compile(
        r"\s*" + _specifier_regex_str + r"\s*", re.VERBOSE | re.IGNORECASE
    )

    # Legacy unused attribute, kept for backward compatibility
    _operators: Final = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    def __init__(self, spec: str = "", prereleases: bool | None = None) -> None:
        """Initialize a Specifier instance.

        :param spec:
            The string representation of a specifier which will be parsed and
            normalized before use.
        :param prereleases:
            This tells the specifier if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.
        :raises InvalidSpecifier:
            If the given specifier is invalid (i.e. bad syntax).
        """
        if not self._regex.fullmatch(spec):
            raise InvalidSpecifier(f"Invalid specifier: {spec!r}")

        spec = spec.strip()
        if spec.startswith("==="):
            operator, version = spec[:3], spec[3:].strip()
        elif spec.startswith(("~=", "==", "!=", "<=", ">=")):
            operator, version = spec[:2], spec[2:].strip()
        else:
            operator, version = spec[:1], spec[1:].strip()

        self._spec: tuple[str, str] = (operator, version)

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

        # Specifier version cache
        self._spec_version: tuple[str, Version] | None = None

        # Version range cache (populated by _to_ranges)
        self._ranges: Sequence[Interval] | None = None

    def _get_spec_version(self, version: str) -> Version | None:
        """One element cache, as only one spec Version is needed per Specifier."""
        if self._spec_version is not None and self._spec_version[0] == version:
            return self._spec_version[1]

        version_specifier = coerce_version(version)
        if version_specifier is None:
            return None

        self._spec_version = (version, version_specifier)
        return version_specifier

    def _require_spec_version(self, version: str) -> Version:
        """Get spec version, asserting it's valid (not for === operator).

        This method should only be called for operators where version
        strings are guaranteed to be valid PEP 440 versions (not ===).
        """
        spec_version = self._get_spec_version(version)
        assert spec_version is not None
        return spec_version

    def _to_ranges(self) -> Sequence[Interval]:
        """Convert this specifier to sorted, non-overlapping version ranges.

        Each standard operator maps to one or two ranges.  ``===`` is
        modeled as full range (actual check done separately).  Cached.
        """
        if self._ranges is not None:
            return self._ranges

        op = self.operator
        ver_str = self.version

        if op == "===":
            result: Sequence[Interval] = FULL_RANGE
        else:
            version = self._require_spec_version(ver_str.removesuffix(".*"))
            result = bounds_for_spec(op, ver_str, version)

        self._ranges = result
        return result

    @property
    def prereleases(self) -> bool | None:
        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Only the "!=" operator does not imply prereleases when
        # the version in the specifier is a prerelease.
        operator, version_str = self._spec
        if operator == "!=":
            return False

        # The == specifier with trailing .* cannot include prereleases
        # e.g. "==1.0a1.*" is not valid.
        if operator == "==" and version_str.endswith(".*"):
            return False

        # "===" can have arbitrary string versions, so we cannot parse
        # those, we take prereleases as unknown (None) for those.
        version = self._get_spec_version(version_str)
        if version is None:
            return None

        # For all other operators, use the check if spec Version
        # object implies pre-releases.
        return version.is_prerelease

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        self._prereleases = value

    def __getstate__(self) -> tuple[tuple[str, str], bool | None]:
        # Return state as a 2-item tuple for compactness:
        #   ((operator, version), prereleases)
        # Cache members are excluded and will be recomputed on demand.
        return (self._spec, self._prereleases)

    def __setstate__(self, state: object) -> None:
        # Always discard cached values - they will be recomputed on demand.
        self._spec_version = None
        self._ranges = None

        if isinstance(state, tuple):
            if len(state) == 2:
                # New format (26.2+): ((operator, version), prereleases)
                spec, prereleases = state
                if _validate_spec(spec) and _validate_pre(prereleases):
                    self._spec = spec
                    self._prereleases = prereleases
                    return
            if len(state) == 2 and isinstance(state[1], dict):
                # Format (packaging 26.0-26.1): (None, {slot: value}).
                _, slot_dict = state
                spec = slot_dict.get("_spec")
                prereleases = slot_dict.get("_prereleases", "invalid")
                if _validate_spec(spec) and _validate_pre(prereleases):
                    self._spec = spec
                    self._prereleases = prereleases
                    return
        if isinstance(state, dict):
            # Old format (packaging <= 25.x, no __slots__): state is a plain dict.
            spec = state.get("_spec")
            prereleases = state.get("_prereleases", "invalid")
            if _validate_spec(spec) and _validate_pre(prereleases):
                self._spec = spec
                self._prereleases = prereleases
                return

        raise TypeError(f"Cannot restore Specifier from {state!r}")

    @property
    def operator(self) -> str:
        """The operator of this specifier.

        >>> Specifier("==1.2.3").operator
        '=='
        """
        return self._spec[0]

    @property
    def version(self) -> str:
        """The version of this specifier.

        >>> Specifier("==1.2.3").version
        '1.2.3'
        """
        return self._spec[1]

    def __repr__(self) -> str:
        """A representation of the Specifier that shows all internal state.

        >>> Specifier('>=1.0.0')
        <Specifier('>=1.0.0')>
        >>> Specifier('>=1.0.0', prereleases=False)
        <Specifier('>=1.0.0', prereleases=False)>
        >>> Specifier('>=1.0.0', prereleases=True)
        <Specifier('>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<{self.__class__.__name__}({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the Specifier that can be round-tripped.

        >>> str(Specifier('>=1.0.0'))
        '>=1.0.0'
        >>> str(Specifier('>=1.0.0', prereleases=False))
        '>=1.0.0'
        """
        return "{}{}".format(*self._spec)

    @property
    def _canonical_spec(self) -> tuple[str, str]:
        operator, version = self._spec
        if operator == "===" or version.endswith(".*"):
            return operator, version

        spec_version = self._require_spec_version(version)

        canonical_version = canonicalize_version(
            spec_version, strip_trailing_zero=(operator != "~=")
        )

        return operator, canonical_version

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: object) -> bool:
        """Whether or not the two Specifier-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> Specifier("==1.2.3") == Specifier("== 1.2.3.0")
        True
        >>> (Specifier("==1.2.3", prereleases=False) ==
        ...  Specifier("==1.2.3", prereleases=True))
        True
        >>> Specifier("==1.2.3") == "==1.2.3"
        True
        >>> Specifier("==1.2.3") == Specifier("==1.2.4")
        False
        >>> Specifier("==1.2.3") == Specifier("~=1.2.3")
        False
        """
        if isinstance(other, str):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def __contains__(self, item: str | Version) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in Specifier(">=1.2.3")
        True
        >>> Version("1.2.3") in Specifier(">=1.2.3")
        True
        >>> "1.0.0" in Specifier(">=1.2.3")
        False
        >>> "1.3.0a1" in Specifier(">=1.2.3")
        True
        >>> "1.3.0a1" in Specifier(">=1.2.3", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(self, item: UnparsedVersion, prereleases: bool | None = None) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item:
            The item to check for, which can be a version string or a
            :class:`~packaging.version.Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this Specifier. If set to
            ``None`` (the default), it will follow the recommendation from
            :pep:`440` and match prereleases, as there are no other versions.

        >>> Specifier(">=1.2.3").contains("1.2.3")
        True
        >>> Specifier(">=1.2.3").contains(Version("1.2.3"))
        True
        >>> Specifier(">=1.2.3").contains("1.0.0")
        False
        >>> Specifier(">=1.2.3").contains("1.3.0a1")
        True
        >>> Specifier(">=1.2.3", prereleases=False).contains("1.3.0a1")
        False
        >>> Specifier(">=1.2.3").contains("1.3.0a1")
        True

        .. versionchanged:: 26.0

            With ``prereleases=None``, a prerelease now matches. A single
            version has no alternatives, so the :pep:`440` rule to accept
            prereleases when nothing else satisfies the specifier applies.
            Earlier versions rejected it. An unparsable version now returns
            ``False`` instead of raising :exc:`~packaging.version.InvalidVersion`.
        """
        # ``===`` compares the raw string, so a Version parse here would
        # be wasted.
        if self._spec[0] == "===":
            return bool(list(self.filter([item], prereleases=prereleases)))

        parsed = coerce_version(item)
        if parsed is None:
            # Standard operators never match an unparsable input.
            return False

        if prereleases is None:
            prereleases = resolve_prereleases(self._prereleases, self.prereleases)

        if prereleases is False and parsed.is_prerelease:
            return False

        # ``_fast_match`` answers the simple operators without building a
        # range; otherwise fall back to the engine's bounds membership.
        match = _fast_match(self, parsed)
        if match is not None:
            return match

        return matches_bounds_only(self._to_ranges(), parsed)

    @typing.overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @typing.overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        """Filter items in the given iterable, that match the specifier.

        :param iterable:
            An iterable that can contain version strings and
            :class:`~packaging.version.Version` instances. The items in the
            iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will follow the recommendation from :pep:`440`
            and match prereleases if there are no other versions.
        :param key:
            A callable that takes a single argument (an item from the iterable) and
            returns a version string or :class:`~packaging.version.Version`
            instance to be used for filtering.

        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.2.3", "1.3", Version("1.4")]))
        ['1.2.3', '1.3', <Version('1.4')>]
        >>> list(Specifier(">=1.2.3").filter(["1.2", "1.5a1"]))
        ['1.5a1']
        >>> list(Specifier(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(Specifier(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(Specifier(">=1.2.3").filter(
        ... [{"ver": "1.2"}, {"ver": "1.3"}],
        ... key=lambda x: x["ver"]))
        [{'ver': '1.3'}]

        .. versionchanged:: 26.1

            Added the ``key`` parameter.
        """
        if prereleases is None:
            prereleases = resolve_prereleases(self._prereleases, self.prereleases)

        if self.operator == "===":
            spec_lower = self.version.lower()
            matches = (
                item
                for item in iterable
                if str(item if key is None else key(item)).lower() == spec_lower
            )
            return _apply_prereleases_filter(matches, key, prereleases)

        return filter_by_ranges(self._to_ranges(), iterable, key, prereleases)


def _apply_prereleases_filter(
    matches: Iterable[Any],
    key: Callable[[Any], UnparsedVersion] | None,
    prereleases: bool | None,
) -> Iterator[Any]:
    """Apply ``prereleases=`` handling to an already-matched iterable.

    ``None`` means PEP 440 default (buffer pre-releases until a final
    appears); ``True`` yields everything; ``False`` drops pre-releases.
    """
    if prereleases is None:
        return _pep440_filter_prereleases(matches, key)
    if prereleases:
        return iter(matches)
    return (
        item
        for item in matches
        if (parsed := coerce_version(item if key is None else key(item))) is None
        or not parsed.is_prerelease
    )


class SpecifierSet(BaseSpecifier):
    """This class abstracts handling of a set of version specifiers.

    It can be passed a single specifier (``>=3.0``), a comma-separated list of
    specifiers (``>=3.0,!=3.1``), or no specifier at all.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging
    releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with
        packaging 26.2+ can be unpickled with future releases.
        Backward compatibility with pickles from
        packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = (
        "_canonicalized",
        "_has_arbitrary",
        "_is_unsatisfiable",
        "_prereleases",
        "_ranges",
        "_specs",
    )

    def __init__(
        self,
        specifiers: str | Iterable[Specifier] = "",
        prereleases: bool | None = None,
    ) -> None:
        """Initialize a SpecifierSet instance.

        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
            May also be an iterable of ``Specifier`` instances, which will be used
            as is.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """

        if isinstance(specifiers, str):
            # Split on `,` to break each individual specifier into its own item, and
            # strip each item to remove leading/trailing whitespace.
            split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

            self._specs: tuple[Specifier, ...] = tuple(map(Specifier, split_specifiers))
            # Fast substring check; avoids iterating parsed specs.
            self._has_arbitrary = "===" in specifiers
        else:
            self._specs = tuple(specifiers)
            # Substring check works for both Specifier objects and plain
            # strings (setuptools passes lists of strings).
            self._has_arbitrary = any("===" in str(s) for s in self._specs)

        self._canonicalized = len(self._specs) <= 1
        self._is_unsatisfiable: bool | None = None
        self._ranges: Sequence[Interval] | None = None

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    def _canonical_specs(self) -> tuple[Specifier, ...]:
        """Deduplicate, sort, and cache specs for order-sensitive operations."""
        if not self._canonicalized:
            self._specs = tuple(dict.fromkeys(sorted(self._specs, key=str)))
            self._canonicalized = True
        return self._specs

    @property
    def prereleases(self) -> bool | None:
        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        if any(s.prereleases for s in self._specs):
            return True

        return None

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        self._prereleases = value
        self._is_unsatisfiable = None

    def __getstate__(self) -> tuple[tuple[Specifier, ...], bool | None]:
        # Return state as a 2-item tuple for compactness:
        #   (specs, prereleases)
        # Cache members are excluded and will be recomputed on demand.
        return (self._specs, self._prereleases)

    def __setstate__(self, state: object) -> None:
        # Always discard cached values - they will be recomputed on demand.
        self._ranges = None
        self._is_unsatisfiable = None

        if isinstance(state, tuple):
            if len(state) == 2:
                # New format (26.2+): (specs, prereleases)
                specs, prereleases = state
                if (
                    isinstance(specs, tuple)
                    and all(isinstance(s, Specifier) for s in specs)
                    and _validate_pre(prereleases)
                ):
                    self._specs = specs
                    self._prereleases = prereleases
                    self._canonicalized = len(specs) <= 1
                    self._has_arbitrary = any("===" in str(s) for s in specs)
                    return
            if len(state) == 2 and isinstance(state[1], dict):
                # Format (packaging 26.0-26.1): (None, {slot: value}).
                _, slot_dict = state
                specs = slot_dict.get("_specs", ())
                prereleases = slot_dict.get("_prereleases")
                # Convert frozenset to tuple (26.0 stored as frozenset)
                if isinstance(specs, frozenset):
                    specs = tuple(sorted(specs, key=str))
                if (
                    isinstance(specs, tuple)
                    and all(isinstance(s, Specifier) for s in specs)
                    and _validate_pre(prereleases)
                ):
                    self._specs = specs
                    self._prereleases = prereleases
                    self._canonicalized = len(self._specs) <= 1
                    self._has_arbitrary = any("===" in str(s) for s in self._specs)
                    return
        if isinstance(state, dict):
            # Old format (packaging <= 25.x, no __slots__): state is a plain dict.
            specs = state.get("_specs", ())
            prereleases = state.get("_prereleases")
            # Convert frozenset to tuple (26.0 stored as frozenset)
            if isinstance(specs, frozenset):
                specs = tuple(sorted(specs, key=str))
            if (
                isinstance(specs, tuple)
                and all(isinstance(s, Specifier) for s in specs)
                and _validate_pre(prereleases)
            ):
                self._specs = specs
                self._prereleases = prereleases
                self._canonicalized = len(self._specs) <= 1
                self._has_arbitrary = any("===" in str(s) for s in self._specs)
                return

        raise TypeError(f"Cannot restore SpecifierSet from {state!r}")

    def __repr__(self) -> str:
        """A representation of the specifier set that shows all internal state.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> SpecifierSet('>=1.0.0,!=2.0.0')
        <SpecifierSet('!=2.0.0,>=1.0.0')>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=False)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=False)>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=True)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=True)>
        """
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<{self.__class__.__name__}({str(self)!r}{pre})>"

    def __str__(self) -> str:
        """A string representation of the specifier set that can be round-tripped.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> str(SpecifierSet(">=1.0.0,!=1.0.1"))
        '!=1.0.1,>=1.0.0'
        >>> str(SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False))
        '!=1.0.1,>=1.0.0'
        """
        return ",".join(str(s) for s in self._canonical_specs())

    def __hash__(self) -> int:
        return hash(self._canonical_specs())

    def __and__(self, other: SpecifierSet | str) -> SpecifierSet:
        """Return a SpecifierSet which is a combination of the two sets.

        :param other: The other object to combine with.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") & '<=2.0.0,!=2.0.1'
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        >>> SpecifierSet(">=1.0.0,!=1.0.1") & SpecifierSet('<=2.0.0,!=2.0.1')
        <SpecifierSet('!=1.0.1,!=2.0.1,<=2.0.0,>=1.0.0')>
        """
        if isinstance(other, str):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = self._specs + other._specs
        specifier._canonicalized = len(specifier._specs) <= 1
        specifier._has_arbitrary = self._has_arbitrary or other._has_arbitrary

        # Combine prerelease settings: use common or non-None value
        if self._prereleases is None or self._prereleases == other._prereleases:
            specifier._prereleases = other._prereleases
        elif other._prereleases is None:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease overrides."
            )

        return specifier

    def __eq__(self, other: object) -> bool:
        """Whether or not the two SpecifierSet-like objects are equal.

        :param other: The other object to check against.

        The value of :attr:`prereleases` is ignored.

        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> (SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False) ==
        ...  SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == ">=1.0.0,!=1.0.1"
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1") == SpecifierSet(">=1.0.0,!=1.0.2")
        False
        """
        if isinstance(other, (str, Specifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._canonical_specs() == other._canonical_specs()

    def __len__(self) -> int:
        """Returns the number of specifiers in this specifier set."""
        return len(self._specs)

    def __iter__(self) -> Iterator[Specifier]:
        """
        Returns an iterator over all the underlying :class:`Specifier` instances
        in this specifier set.

        >>> sorted(SpecifierSet(">=1.0.0,!=1.0.1"), key=str)
        [<Specifier('!=1.0.1')>, <Specifier('>=1.0.0')>]
        """
        return iter(self._specs)

    def _get_ranges(self) -> Sequence[Interval]:
        """Intersect all specifiers into a single sequence of version ranges.

        Empty when unsatisfiable. Callers must ensure ``self._specs``
        is non-empty.
        """
        if self._ranges is not None:
            return self._ranges

        self._ranges = intersect_specifier_bounds(s._to_ranges() for s in self._specs)
        return self._ranges

    def is_unsatisfiable(self) -> bool:
        """Check whether this specifier set can never be satisfied.

        Returns True if no version can satisfy all specifiers simultaneously.

        >>> SpecifierSet(">=2.0,<1.0").is_unsatisfiable()
        True
        >>> SpecifierSet(">=1.0,<2.0").is_unsatisfiable()
        False
        >>> SpecifierSet("").is_unsatisfiable()
        False
        >>> SpecifierSet("==1.0,!=1.0").is_unsatisfiable()
        True

        .. versionadded:: 26.1
        """
        cached = self._is_unsatisfiable
        if cached is not None:
            return cached

        if not self._specs:
            self._is_unsatisfiable = False
            return False

        result = not self._get_ranges()

        if not result:
            result = self._check_arbitrary_unsatisfiable()

        if not result and self.prereleases is False:
            result = ranges_are_prerelease_only(self._get_ranges())

        self._is_unsatisfiable = result
        return result

    def _check_arbitrary_unsatisfiable(self) -> bool:
        """Check === (arbitrary equality) specs for unsatisfiability.

        === uses case-insensitive string comparison, so the only candidate
        that can match ``===V`` is the literal string V.  This method
        checks whether that candidate is excluded by other specifiers.
        """
        arbitrary = [s for s in self._specs if s.operator == "==="]
        if not arbitrary:
            return False

        # Multiple === must agree on the same string (case-insensitive).
        first = arbitrary[0].version.lower()
        if any(s.version.lower() != first for s in arbitrary[1:]):
            return True

        # The sole candidate is the === version string.  Check whether
        # it can satisfy every standard spec.
        candidate = coerce_version(arbitrary[0].version)

        # With prereleases=False, a prerelease candidate is excluded
        # by contains() before the === string check even runs.
        if (
            self.prereleases is False
            and candidate is not None
            and candidate.is_prerelease
        ):
            return True

        standard = [s for s in self._specs if s.operator != "==="]
        if not standard:
            return False

        if candidate is None:
            # Unparsable string cannot satisfy any standard spec.
            return True

        return not all(s.contains(candidate) for s in standard)

    def to_range(self) -> ranges.VersionRange:
        """Return the :class:`~packaging.ranges.VersionRange` this set accepts.

        An empty set yields the full range; an unsatisfiable set yields the
        empty range. ``===`` specifiers contribute literal-string admission.

        >>> SpecifierSet(">=1.0,<2.0").to_range()
        <VersionRange '[1.0, 2.0.dev0)'>

        .. versionadded:: 26.3
        """
        from .ranges import VersionRange  # noqa: PLC0415

        return VersionRange._from_specifier_set(self)

    def _check_relation_operand(self, other: object) -> None:
        if not isinstance(other, SpecifierSet):
            raise TypeError("expected a SpecifierSet")
        if self._has_arbitrary or other._has_arbitrary:
            raise ValueError("set relations do not support === specifiers")

    def is_subset(self, other: SpecifierSet) -> bool:
        """Return whether every version matching this set also matches other.

        :raises ValueError:
            If either set uses ``===`` specifiers, or the two sets were
            given different ``prereleases`` arguments (unset on one side
            counts as different).
        :raises TypeError:
            If other is not a :class:`SpecifierSet`.

        >>> SpecifierSet(">=3.12,<3.13").is_subset(SpecifierSet(">=3.12"))
        True
        >>> SpecifierSet(">=3.12").is_subset(SpecifierSet(">=3.12,<3.13"))
        False

        .. versionadded:: 26.3
        """
        self._check_relation_operand(other)
        return self.to_range().is_subset(other.to_range())

    def is_superset(self, other: SpecifierSet) -> bool:
        """Return whether every version matching other also matches this set.

        :raises ValueError:
            If either set uses ``===`` specifiers, or the two sets were
            given different ``prereleases`` arguments (unset on one side
            counts as different).
        :raises TypeError:
            If other is not a :class:`SpecifierSet`.

        >>> SpecifierSet(">=3.12").is_superset(SpecifierSet(">=3.12,<3.13"))
        True

        .. versionadded:: 26.3
        """
        self._check_relation_operand(other)
        return self.to_range().is_superset(other.to_range())

    def is_disjoint(self, other: SpecifierSet) -> bool:
        """Return whether this set and other share no matching versions.

        :raises ValueError:
            If either set uses ``===`` specifiers, or the two sets were
            given different ``prereleases`` arguments (unset on one side
            counts as different).
        :raises TypeError:
            If other is not a :class:`SpecifierSet`.

        >>> SpecifierSet("<3.12").is_disjoint(SpecifierSet(">=3.12"))
        True
        >>> SpecifierSet("<3.12").is_disjoint(SpecifierSet(">=3.11"))
        False

        .. versionadded:: 26.3
        """
        self._check_relation_operand(other)
        return self.to_range().is_disjoint(other.to_range())

    def __contains__(self, item: UnparsedVersion) -> bool:
        """Return whether or not the item is contained in this specifier.

        :param item: The item to check for.

        This is used for the ``in`` operator and behaves the same as
        :meth:`contains` with no ``prereleases`` argument passed.

        >>> "1.2.3" in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> Version("1.2.3") in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> "1.0.1" in SpecifierSet(">=1.0.0,!=1.0.1")
        False
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1")
        True
        >>> "1.3.0a1" in SpecifierSet(">=1.0.0,!=1.0.1", prereleases=True)
        True
        """
        return self.contains(item)

    def contains(
        self,
        item: UnparsedVersion,
        prereleases: bool | None = None,
        installed: bool | None = None,
    ) -> bool:
        """Return whether or not the item is contained in this SpecifierSet.

        :param item:
            The item to check for, which can be a version string or a
            :class:`~packaging.version.Version` instance.
        :param prereleases:
            Whether or not to match prereleases with this SpecifierSet. If set to
            ``None`` (the default), it will follow the recommendation from :pep:`440`
            and match prereleases, as there are no other versions.
        :param installed:
            Whether or not the item is installed. If set to ``True``, it will
            accept prerelease versions even if the specifier does not allow them.

        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.2.3")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains(Version("1.2.3"))
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.0.1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1")
        True
        >>> SpecifierSet(">=1.0.0,!=1.0.1", prereleases=False).contains("1.3.0a1")
        False
        >>> SpecifierSet(">=1.0.0,!=1.0.1").contains("1.3.0a1", prereleases=True)
        True

        .. versionchanged:: 26.0

            With ``prereleases=None``, a prerelease now matches. A single
            version has no alternatives, so the :pep:`440` rule to accept
            prereleases when nothing else satisfies the specifiers applies.
            Earlier versions rejected it. An unparsable version now returns
            ``False`` instead of raising :exc:`~packaging.version.InvalidVersion`.
        """
        version = coerce_version(item)

        if version is not None and installed and version.is_prerelease:
            prereleases = True

        # When item is a string and === is involved, keep it as-is
        # so the comparison isn't done against the normalized form.
        if version is None or (self._has_arbitrary and not isinstance(item, Version)):
            check_item = item
        else:
            check_item = version

        # Fast path: a parseable, local-free version against a rangelike set.
        # A local on ``version`` needs PEP 440 stripping that the range path
        # applies.
        if (
            version is not None
            and not self._has_arbitrary
            and version.local is None
            and self._specs
        ):
            if version.is_prerelease and (
                prereleases is False
                or (prereleases is None and self._prereleases is False)
            ):
                return False

            bounds = self._ranges
            if bounds is None:
                # Per-spec ``_fast_match`` answers a set of simple specifiers
                # without folding anything. If a spec needs the range path,
                # fold the intersected bounds once and cache them so repeated
                # checks on the same set stay cheap.
                for spec in self._specs:
                    match = _fast_match(spec, version)
                    if match is None:
                        break
                    if not match:
                        return False
                else:
                    return True

                bounds = self._ranges = self._get_ranges()

            return matches_bounds_only(bounds, version)

        return bool(list(self.filter([check_item], prereleases=prereleases)))

    @typing.overload
    def filter(
        self,
        iterable: Iterable[UnparsedVersionVar],
        prereleases: bool | None = None,
        key: None = ...,
    ) -> Iterator[UnparsedVersionVar]: ...

    @typing.overload
    def filter(
        self,
        iterable: Iterable[T],
        prereleases: bool | None = None,
        key: Callable[[T], UnparsedVersion] = ...,
    ) -> Iterator[T]: ...

    def filter(
        self,
        iterable: Iterable[Any],
        prereleases: bool | None = None,
        key: Callable[[Any], UnparsedVersion] | None = None,
    ) -> Iterator[Any]:
        """Filter items in the given iterable, that match the specifiers in this set.

        :param iterable:
            An iterable that can contain version strings and
            :class:`~packaging.version.Version` instances. The items in the
            iterable will be filtered according to the specifier.
        :param prereleases:
            Whether or not to allow prereleases in the returned iterator. If set to
            ``None`` (the default), it will follow the recommendation from :pep:`440`
            and match prereleases if there are no other versions.
        :param key:
            A callable that takes a single argument (an item from the iterable) and
            returns a version string or :class:`~packaging.version.Version`
            instance to be used for filtering.

        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.3", Version("1.4")]))
        ['1.3', <Version('1.4')>]
        >>> list(SpecifierSet(">=1.2.3").filter(["1.2", "1.5a1"]))
        ['1.5a1']
        >>> list(SpecifierSet(">=1.2.3").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet(">=1.2.3", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet(">=1.2.3").filter(
        ... [{"ver": "1.2"}, {"ver": "1.3"}],
        ... key=lambda x: x["ver"]))
        [{'ver': '1.3'}]

        An "empty" SpecifierSet will filter items based on the presence of prerelease
        versions in the set.

        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"]))
        ['1.3']
        >>> list(SpecifierSet("").filter(["1.5a1"]))
        ['1.5a1']
        >>> list(SpecifierSet("", prereleases=True).filter(["1.3", "1.5a1"]))
        ['1.3', '1.5a1']
        >>> list(SpecifierSet("").filter(["1.3", "1.5a1"], prereleases=True))
        ['1.3', '1.5a1']

        .. versionchanged:: 26.0

            Prerelease filtering now follows the PEP 440 recommendation of
            yielding prereleases only when no final release is present.

        .. versionchanged:: 26.1

            Added the ``key`` parameter.
        """
        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None and self.prereleases is not None:
            prereleases = self.prereleases

        if self._specs:
            if self._has_arbitrary:
                # Slow path for ===
                specs = self._specs
                matches = (
                    item
                    for item in iterable
                    if all(
                        s.contains(item if key is None else key(item), prereleases=True)
                        for s in specs
                    )
                )
                return _apply_prereleases_filter(matches, key, prereleases)

            ranges = self._ranges
            if ranges is None:
                ranges = self._get_ranges()
            return filter_by_ranges(ranges, iterable, key, prereleases)

        # Empty SpecifierSet.
        return _apply_prereleases_filter(iterable, key, prereleases)


def _pep440_filter_prereleases(
    iterable: Iterable[Any], key: Callable[[Any], UnparsedVersion] | None
) -> Iterator[Any]:
    """Filter per PEP 440: exclude prereleases unless no finals exist."""
    # Two lists used:
    #   * all_nonfinal to preserve order if no finals exist
    #   * arbitrary_strings for streaming when first final found
    all_nonfinal: list[Any] = []
    arbitrary_strings: list[Any] = []

    found_final = False
    for item in iterable:
        parsed = coerce_version(item if key is None else key(item))

        if parsed is None:
            # Arbitrary strings are always included as it is not
            # possible to determine if they are prereleases,
            # and they have already passed all specifiers.
            if found_final:
                yield item
            else:
                arbitrary_strings.append(item)
                all_nonfinal.append(item)
            continue

        if not parsed.is_prerelease:
            # Final release found - flush arbitrary strings, then yield
            if not found_final:
                yield from arbitrary_strings
                found_final = True
            yield item
            continue

        # Prerelease - buffer if no finals yet, otherwise skip
        if not found_final:
            all_nonfinal.append(item)

    # No finals found - yield all buffered items
    if not found_final:
        yield from all_nonfinal
<|endoftext|>#######################################################################################
#
# Adapted from:
#  https://github.com/pypa/hatch/blob/5352e44/backend/src/hatchling/licenses/parse.py
#
# MIT License
#
# Copyright (c) 2017-present Ofek Lev <oss@ofek.dev>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#
# With additional allowance of arbitrary `LicenseRef-` identifiers, not just
# `LicenseRef-Public-Domain` and `LicenseRef-Proprietary`.
#
#######################################################################################
from __future__ import annotations

import re
from typing import NewType, cast

from ._spdx import EXCEPTIONS, LICENSES

__all__ = [
    "InvalidLicenseExpression",
    "NormalizedLicenseExpression",
    "canonicalize_license_expression",
]


# Simple __dir__ implementation since there are no public submodules
def __dir__() -> list[str]:
    return __all__


license_ref_allowed = re.compile("^[A-Za-z0-9.-]+$")

NormalizedLicenseExpression = NewType("NormalizedLicenseExpression", str)
"""
A :class:`typing.NewType` of :class:`str`, representing a normalized
License-Expression.
"""


class InvalidLicenseExpression(ValueError):
    """Raised when a license-expression string is invalid

    >>> from packaging.licenses import canonicalize_license_expression
    >>> canonicalize_license_expression("invalid")
    Traceback (most recent call last):
        ...
    packaging.licenses.InvalidLicenseExpression: Unknown license: 'invalid'
    """


def canonicalize_license_expression(
    raw_license_expression: str,
) -> NormalizedLicenseExpression:
    """
    This function takes a valid License-Expression, and returns the normalized
    form of it.

    The return type is typed as :class:`NormalizedLicenseExpression`. This
    allows type checkers to help require that a string has passed through this
    function before use.

    :param str raw_license_expression: The License-Expression to canonicalize.
    :raises InvalidLicenseExpression: If the License-Expression is invalid due to an
        invalid/unknown license identifier or invalid syntax.

    .. doctest::

        >>> from packaging.licenses import canonicalize_license_expression
        >>> canonicalize_license_expression("mit")
        'MIT'
        >>> canonicalize_license_expression("mit and (apache-2.0 or bsd-2-clause)")
        'MIT AND (Apache-2.0 OR BSD-2-Clause)'
        >>> canonicalize_license_expression("(mit")
        Traceback (most recent call last):
          ...
        InvalidLicenseExpression: Invalid license expression: '(mit'
        >>> canonicalize_license_expression("Use-it-after-midnight")
        Traceback (most recent call last):
          ...
        InvalidLicenseExpression: Unknown license: 'Use-it-after-midnight'
    """
    if not raw_license_expression:
        message = f"Invalid license expression: {raw_license_expression!r}"
        raise InvalidLicenseExpression(message)

    # Pad any parentheses so tokenization can be achieved by merely splitting on
    # whitespace.
    license_expression = raw_license_expression.replace("(", " ( ").replace(")", " ) ")
    licenseref_prefix = "LicenseRef-"
    license_refs = {
        ref.lower(): "LicenseRef-" + ref[len(licenseref_prefix) :]
        for ref in license_expression.split()
        if ref.lower().startswith(licenseref_prefix.lower())
    }

    # Normalize to lower case so we can look up licenses/exceptions
    # and so boolean operators are Python-compatible.
    license_expression = license_expression.lower()

    tokens = license_expression.split()

    # Rather than implementing a parenthesis/boolean logic parser, create an
    # expression that Python can parse. Everything that is not involved with the
    # grammar itself is replaced with the placeholder `False` and the resultant
    # expression should become a valid Python expression.
    python_tokens = []
    for token in tokens:
        if token not in {"or", "and", "with", "(", ")"}:
            python_tokens.append("False")
        elif token == "with":
            python_tokens.append("or")
        elif (
            token == "("
            and python_tokens
            and python_tokens[-1] not in {"or", "and", "("}
        ) or (token == ")" and python_tokens and python_tokens[-1] == "("):
            message = f"Invalid license expression: {raw_license_expression!r}"
            raise InvalidLicenseExpression(message)
        else:
            python_tokens.append(token)

    python_expression = " ".join(python_tokens)
    try:
        compile(python_expression, "", "eval")
    except SyntaxError:
        message = f"Invalid license expression: {raw_license_expression!r}"
        raise InvalidLicenseExpression(message) from None

    # Take a final pass to check for unknown licenses/exceptions.
    normalized_tokens = []
    last_license_start = False
    for index, token in enumerate(tokens):
        if token in {"or", "and", "with", "(", ")"}:
            if token == "with" and (
                not last_license_start
                or index + 1 == len(tokens)
                or tokens[index + 1] in {"or", "and", "with", "(", ")"}
            ):
                message = f"Invalid license expression: {raw_license_expression!r}"
                raise InvalidLicenseExpression(message)
            normalized_tokens.append(token.upper())
            last_license_start = False
            continue

        if normalized_tokens and normalized_tokens[-1] == "WITH":
            if token not in EXCEPTIONS:
                message = f"Unknown license exception: {token!r}"
                raise InvalidLicenseExpression(message)

            normalized_tokens.append(EXCEPTIONS[token]["id"])
            last_license_start = False
        else:
            if token.endswith("+"):
                final_token = token[:-1]
                suffix = "+"
            else:
                final_token = token
                suffix = ""

            if final_token.startswith("licenseref-"):
                license_ref_id = final_token[len("licenseref-") :]
                if suffix or not license_ref_allowed.match(license_ref_id):
                    message = f"Invalid licenseref: {token!r}"
                    raise InvalidLicenseExpression(message)
                normalized_tokens.append(license_refs[final_token])
            else:
                if final_token not in LICENSES:
                    message = f"Unknown license: {final_token!r}"
                    raise InvalidLicenseExpression(message)
                normalized_tokens.append(LICENSES[final_token]["id"] + suffix)
            last_license_start = True

    normalized_expression = " ".join(normalized_tokens)

    return cast(
        "NormalizedLicenseExpression",
        normalized_expression.replace("( ", "(").replace(" )", ")"),
    )
<|endoftext|>import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
    CDLL,
    POINTER,
    c_bool,
    c_char_p,
    c_int32,
    c_long,
    c_uint32,
    c_ulong,
    c_void_p,
)
from ctypes.util import find_library

from ._ssl_constants import _set_ssl_context_verify_mode

_mac_version = platform.mac_ver()[0]
_mac_version_info = tuple(map(int, _mac_version.split(".")))
if _mac_version_info < (10, 8):
    raise ImportError(
        f"Only OS X 10.8 and newer are supported, not {_mac_version_info[0]}.{_mac_version_info[1]}"
    )

_is_macos_version_10_14_or_later = _mac_version_info >= (10, 14)


def _load_cdll(name: str, macos10_16_path: str) -> CDLL:
    """Loads a CDLL by name, falling back to known path on 10.16+"""
    try:
        # Big Sur is technically 11 but we use 10.16 due to the Big Sur
        # beta being labeled as 10.16.
        path: str | None
        if _mac_version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError  # Caught and reraised as 'ImportError'
        return CDLL(path, use_errno=True)
    except OSError:
        raise ImportError(f"The library {name} failed to load") from None


Security = _load_cdll(
    "Security", "/System/Library/Frameworks/Security.framework/Security"
)
CoreFoundation = _load_cdll(
    "CoreFoundation",
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
)

Boolean = c_bool
CFIndex = c_long
CFStringEncoding = c_uint32
CFData = c_void_p
CFString = c_void_p
CFArray = c_void_p
CFMutableArray = c_void_p
CFError = c_void_p
CFType = c_void_p
CFTypeID = c_ulong
CFTypeRef = POINTER(CFType)
CFAllocatorRef = c_void_p

