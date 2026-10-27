    An empty string if the value cannot be determined.
    """

    python_full_version: str
    """
    The Python version as string ``'major.minor.patchlevel'``.

    Note that unlike the Python :py:data:`sys.version`, this value will always include
    the patchlevel (it defaults to 0).
    """

    platform_python_implementation: str
    """
    A string identifying the Python implementation, e.g. ``'CPython'``.
    """

    python_version: str
    """The Python version as string ``'major.minor'``."""

    sys_platform: str
    """
    This string contains a platform identifier that can be used to append
    platform-specific components to :py:data:`sys.path`, for instance.

    For Unix systems, except on Linux and AIX, this is the lowercased OS name as
    returned by ``uname -s`` with the first part of the version as returned by
    ``uname -r`` appended, e.g. ``'sunos5'`` or ``'freebsd8'``, at the time when Python
    was built.
    """


def _normalize_extras(
    result: MarkerList | MarkerAtom | str,
) -> MarkerList | MarkerAtom | str:
    if isinstance(result, list):
        return [_normalize_extras(r) for r in result]
    if not isinstance(result, tuple):
        return result

    lhs, op, rhs = result
    if isinstance(lhs, Variable) and lhs.value == "extra" and isinstance(rhs, Value):
        normalized_extra = canonicalize_name(rhs.value)
        rhs = Value(normalized_extra)
    elif isinstance(rhs, Variable) and rhs.value == "extra" and isinstance(lhs, Value):
        normalized_extra = canonicalize_name(lhs.value)
        lhs = Value(normalized_extra)
    elif (
        isinstance(rhs, Variable)
        and rhs.value in MARKERS_ALLOWING_SET
        and isinstance(lhs, Value)
    ):
        # PEP 685 (extras) / PEP 735 (dependency_groups): the set-valued membership
        # literal must also be normalized. evaluate() already canonicalizes both
        # operands for these keys (see _normalize), so normalizing the literal at
        # parse time keeps __str__/__eq__/__hash__ consistent with evaluate() -- e.g.
        # Marker('"Foo" in extras') and Marker('"foo" in extras') must compare and
        # hash equal (the membership variable is always the right-hand operand).
        lhs = Value(canonicalize_name(lhs.value))
    return lhs, op, rhs


def _normalize_extra_values(results: MarkerList) -> MarkerList:
    """
    Normalize extra values.
    """

    return [_normalize_extras(r) for r in results]


def _format_marker(
    marker: list[str] | MarkerAtom | str, first: bool | None = True
) -> str:
    assert isinstance(marker, (list, tuple, str))

    # Unwrap a redundant [[...]] wrapper, but keep the nesting context so a
    # nested group keeps the parentheses its and/or precedence needs.
    if (
        isinstance(marker, list)
        and len(marker) == 1
        and isinstance(marker[0], (list, tuple))
    ):
        return _format_marker(marker[0], first=first)

    if isinstance(marker, list):
        inner = (_format_marker(m, first=False) for m in marker)
        if first:
            return " ".join(inner)
        else:
            return "(" + " ".join(inner) + ")"
    elif isinstance(marker, tuple):
        return " ".join([m.serialize() for m in marker])
    else:
        return marker


_operators: dict[str, Operator] = {
    "in": lambda lhs, rhs: lhs in rhs,
    "not in": lambda lhs, rhs: lhs not in rhs,
    "<": lambda _lhs, _rhs: False,
    "<=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.eq,
    ">": lambda _lhs, _rhs: False,
}


def _eval_op(lhs: str, op: Op, rhs: str | AbstractSet[str], *, key: str) -> bool:
    op_str = op.serialize()
    if key in MARKERS_REQUIRING_VERSION:
        try:
            spec = Specifier(f"{op_str}{rhs}")
        except InvalidSpecifier:
            pass
        else:
            return spec.contains(lhs, prereleases=True)

    oper: Operator | None = _operators.get(op_str)
    if oper is None:
        raise UndefinedComparison(f"Undefined {op!r} on {lhs!r} and {rhs!r}.")

    return oper(lhs, rhs)


def _normalize(
    lhs: str, rhs: str | AbstractSet[str], key: str
) -> tuple[str, str | AbstractSet[str]]:
    # PEP 685 - Comparison of extra names for optional distribution dependencies
    # https://peps.python.org/pep-0685/
    # > When comparing extra names, tools MUST normalize the names being
    # > compared using the semantics outlined in PEP 503 for names
    if key == "extra":
        assert isinstance(rhs, str), "extra value must be a string"
        # Both sides are normalized at this point already
        return (lhs, rhs)
    if key in MARKERS_ALLOWING_SET:
        if isinstance(rhs, str):  # pragma: no cover
            return (canonicalize_name(lhs), canonicalize_name(rhs))
        else:
            return (canonicalize_name(lhs), {canonicalize_name(v) for v in rhs})

    # other environment markers don't have such standards
    return lhs, rhs


def _lookup_environment(
    environment: dict[str, str | AbstractSet[str]], key: str
) -> str | AbstractSet[str]:
    try:
        return environment[key]
    except KeyError:
        raise UndefinedEnvironmentName(key) from None


def _evaluate_markers(
    markers: MarkerList, environment: dict[str, str | AbstractSet[str]]
) -> bool:
    groups: list[list[bool]] = [[]]

    for marker in markers:
        if isinstance(marker, list):
            groups[-1].append(_evaluate_markers(marker, environment))
        elif isinstance(marker, tuple):
            lhs, op, rhs = marker

            if isinstance(lhs, Variable):
                environment_key = lhs.value
                lhs_value = _lookup_environment(environment, environment_key)
                rhs_value = rhs.value
            else:
                lhs_value = lhs.value
                environment_key = rhs.value
                rhs_value = _lookup_environment(environment, environment_key)

            if not isinstance(lhs_value, str):
                raise UndefinedComparison(
                    f"Set-valued marker {environment_key!r} can only be used "
                    f'with the membership form (e.g. "<name>" in '
                    f"{environment_key}); it cannot appear on the left-hand "
                    f"side of {op.serialize()!r}."
                )
            lhs_value, rhs_value = _normalize(lhs_value, rhs_value, key=environment_key)
            groups[-1].append(_eval_op(lhs_value, op, rhs_value, key=environment_key))
        elif marker == "or":
            groups.append([])
        elif marker == "and":
            pass
        else:  # pragma: nocover
            raise TypeError(f"Unexpected marker {marker!r}")

    return any(all(item) for item in groups)


def _format_full_version(info: sys._version_info) -> str:
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


@functools.cache
def _cached_default_environment() -> Environment:
    """Build the default marker environment for the current Python process.

    The values are derived from process-constant data (the running interpreter
    and the host platform), so this is cached and built only once. The result is
    shared between callers and must never be mutated; :func:`default_environment`
    returns a fresh copy.
    """
    iver = _format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


def default_environment() -> Environment:
    """Return the default marker environment for the current Python process.

    This is the base environment used by :meth:`Marker.evaluate`. A fresh copy
    is returned on every call so callers may freely mutate the result; a shallow
    copy suffices because all values are immutable strings.

    .. versionchanged:: 26.3
        The environment is computed once per process and cached, since it is
        derived from process-constant data. Patching ``platform``/``sys``/``os``
        after the first call has no effect; pass an explicit ``environment`` to
        :meth:`Marker.evaluate` to evaluate against different values.
    """
    return cast("Environment", dict(_cached_default_environment()))


class Marker:
    """Represents a parsed dependency marker expression.

    Marker expressions are parsed according to the
    :ref:`specification of dependency specifiers <pypug:dependency-specifiers>`.

    :param marker: The string representation of a marker expression.
    :raises InvalidMarker: If ``marker`` cannot be parsed.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = ("_markers",)

    def __init__(self, marker: str) -> None:
        # Note: We create a Marker object without calling this constructor in
        #       packaging.requirements.Requirement. If any additional logic is
        #       added here, make sure to mirror/adapt Requirement.

        # If this fails and throws an error, the repr still expects _markers to
        # be defined.
        self._markers: MarkerList = []

        try:
            self._markers = _normalize_extra_values(_parse_marker(marker))
            # The attribute `_markers` can be described in terms of a recursive type:
            # MarkerList = List[Union[Tuple[Node, ...], str, MarkerList]]
            #
            # For example, the following expression:
            # python_version > "3.6" or (python_version == "3.6" and os_name == "unix")
            #
            # is parsed into:
            # [
            #     (<Variable('python_version')>, <Op('>')>, <Value('3.6')>),
            #     'and',
            #     [
            #         (<Variable('python_version')>, <Op('==')>, <Value('3.6')>),
            #         'or',
            #         (<Variable('os_name')>, <Op('==')>, <Value('unix')>)
            #     ]
            # ]
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e

    @classmethod
    def _from_markers(cls, markers: MarkerList) -> Marker:
        """Create a Marker instance from a pre-parsed marker tree.

        This avoids re-parsing serialised marker strings when combining markers.
        """
        new = cls.__new__(cls)
        new._markers = markers
        return new

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented

        return str(self) == str(other)

    def __getstate__(self) -> str:
        # Return the marker expression string for compactness and stability.
        # Internal Node objects are excluded; the string is re-parsed on load.
        return str(self)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            # New format (26.2+): just the marker expression string.
            try:
                self._markers = _normalize_extra_values(_parse_marker(state))
            except ParserSyntaxError as exc:
                raise TypeError(f"Cannot restore Marker from {state!r}") from exc
            return
        if isinstance(state, dict) and "_markers" in state:
            # Old format (packaging <= 26.1, no __slots__): plain __dict__.
            markers = state["_markers"]
            if isinstance(markers, list):
                self._markers = markers
                return
        if isinstance(state, tuple) and len(state) == 2:
            # Old format (packaging <= 26.1, __slots__): (None, {slot: value}).
            _, slot_dict = state
            if isinstance(slot_dict, dict) and "_markers" in slot_dict:
                markers = slot_dict["_markers"]
                if isinstance(markers, list):
                    self._markers = markers
                    return
        raise TypeError(f"Cannot restore Marker from {state!r}")

    def __and__(self, other: Marker) -> Marker:
        """Combine this marker with another using ``and``.

        .. versionadded:: 26.1
        """
        if not isinstance(other, Marker):
            return NotImplemented
        return self._from_markers([self._markers, "and", other._markers])

    def __or__(self, other: Marker) -> Marker:
        """Combine this marker with another using ``or``.

        .. versionadded:: 26.1
        """
        if not isinstance(other, Marker):
            return NotImplemented
        return self._from_markers([self._markers, "or", other._markers])

    def evaluate(
        self,
        environment: Mapping[str, str | AbstractSet[str]] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating this marker against the environment.
        The environment is determined from the current Python process unless
        passed in explicitly.

        :param environment: Mapping containing keys and values to override the
           detected environment.
        :param EvaluateContext context: The context in which the marker is
            evaluated, which influences what marker names are considered valid.
            Accepted values are ``"metadata"`` (for core metadata; default),
            ``"lock_file"``, and ``"requirement"`` (i.e. all other situations).
        :raises UndefinedComparison: If the marker uses a comparison on values
            that are not valid versions per the :ref:`specification of version
            specifiers <pypug:version-specifiers>`.
        :raises UndefinedEnvironmentName: If the marker references a value that
            is missing from the evaluation environment.
        :returns: ``True`` if the marker matches, otherwise ``False``.

        .. versionchanged:: 25.0
            Added the ``context`` parameter, which influences which marker names
            are considered valid.
        """
        current_environment = cast(
            "dict[str, str | AbstractSet[str]]", default_environment()
        )
        if context == "lock_file":
            current_environment |= {
                "extras": frozenset(),
                "dependency_groups": frozenset(),
            }
        elif context == "metadata":
            current_environment["extra"] = ""

        if environment is not None:
            current_environment |= environment
            if "extra" in current_environment:
                # The API used to allow setting extra to None. We need to handle
                # this case for backwards compatibility. Also skip running
                # normalize name if extra is empty.
                extra = cast("str | None", current_environment["extra"])
                current_environment["extra"] = canonicalize_name(extra) if extra else ""

        return _evaluate_markers(
            self._markers, _repair_python_full_version(current_environment)
        )


def _pep440_python_full_version(python_full_version: str) -> str:
    """
    Work around platform.python_version() returning something that is not PEP 440
    compliant for non-tagged Python builds.
    """
    if python_full_version.endswith("+"):
        return f"{python_full_version}local"
    return python_full_version


def _repair_python_full_version(
    env: dict[str, str | AbstractSet[str]],
) -> dict[str, str | AbstractSet[str]]:
    """
    Work around platform.python_version() returning something that is not PEP 440
    compliant for non-tagged Python builds.
    """
    python_full_version = cast("str", env["python_full_version"])
    env["python_full_version"] = _pep440_python_full_version(python_full_version)
    return env
<|endoftext|>from __future__ import annotations

import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import TYPE_CHECKING, NamedTuple

from ._elffile import EIClass, EIData, ELFFile, EMachine

if TYPE_CHECKING:
    import types
    from collections.abc import Generator, Iterator, Sequence

EF_ARM_ABIMASK = 0xFF000000
EF_ARM_ABI_VER5 = 0x05000000
EF_ARM_ABI_FLOAT_HARD = 0x00000400

_ALLOWED_ARCHS = {
    "x86_64",
    "aarch64",
    "ppc64",
    "ppc64le",
    "s390x",
    "loongarch64",
    "riscv64",
}


@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[ELFFile | None, None, None]:
    try:
        with open(path, "rb") as f:
            yield ELFFile(f)
    except (OSError, TypeError, ValueError):
        yield None


def _is_linux_armhf(executable: str) -> bool:
    # hard-float ABI can be detected from the ELF header of the running
    # process
    # https://static.docs.arm.com/ihi0044/g/aaelf32.pdf
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.Arm
            and f.flags & EF_ARM_ABIMASK == EF_ARM_ABI_VER5
            and f.flags & EF_ARM_ABI_FLOAT_HARD == EF_ARM_ABI_FLOAT_HARD
        )


def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.I386
        )


def _have_compatible_abi(executable: str, archs: Sequence[str]) -> bool:
    if "armv7l" in archs:
        return _is_linux_armhf(executable)
    if "i686" in archs:
        return _is_linux_i686(executable)
    return any(arch in _ALLOWED_ARCHS for arch in archs)


# If glibc ever changes its major version, we need to know what the last
# minor version was, so we can build the complete list of all versions.
# For now, guess what the highest minor version might be, assume it will
# be 50 for testing. Once this actually happens, update the dictionary
# with the actual value.
_LAST_GLIBC_MINOR: dict[int, int] = collections.defaultdict(lambda: 50)


class _GLibCVersion(NamedTuple):
    major: int
    minor: int


def _glibc_version_string_confstr() -> str | None:
    """
    Primary implementation of glibc_version_string using os.confstr.
    """
    # os.confstr is quite a bit faster than ctypes.DLL. It's also less likely
    # to be broken or missing. This strategy is used in the standard library
    # platform module.
    # https://github.com/python/cpython/blob/fcf1d003bf4f0100c/Lib/platform.py#L175-L183
    try:
        # Should be a string like "glibc 2.17".
        version_string: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        assert version_string is not None
        _, version = version_string.rsplit()
    except (AssertionError, AttributeError, OSError, ValueError):
        # os.confstr() or CS_GNU_LIBC_VERSION not available (or a bad value)...
        return None
    return version


def _glibc_version_string_ctypes() -> str | None:
    """
    Fallback implementation of glibc_version_string using ctypes.
    """
    try:
        import ctypes  # noqa: PLC0415
    except ImportError:
        return None

    # ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
    # manpage says, "If filename is NULL, then the returned handle is for the
    # main program". This way we can let the linker do the work to figure out
    # which libc our process is actually using.
    #
    # We must also handle the special case where the executable is not a
    # dynamically linked executable. This can occur when using musl libc,
    # for example. In this situation, dlopen() will error, leading to an
    # OSError. Interestingly, at least in the case of musl, there is no
    # errno set on the OSError. The single string argument used to construct
    # OSError comes from libc itself and is therefore not portable to
    # hard code here. In any case, failure to call dlopen() means we
    # can proceed, so we bail on our attempt.
    try:
        process_namespace = ctypes.CDLL(None)
    except OSError:
        return None

    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return None

    # Call gnu_get_libc_version, which returns a string like "2.5".
    gnu_get_libc_version.restype = ctypes.c_char_p
    # A c_char_p restype comes back as bytes, so decode to text.
    version_str: str | bytes = gnu_get_libc_version()
    if isinstance(version_str, bytes):
        version_str = version_str.decode("ascii")

    return version_str


def _glibc_version_string() -> str | None:
    """Returns glibc version string, or None if not using glibc."""
    return _glibc_version_string_confstr() or _glibc_version_string_ctypes()


def _parse_glibc_version(version_str: str) -> _GLibCVersion:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    m = re.match(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", version_str)
    if not m:
        warnings.warn(
            f"Expected glibc version with 2 components major.minor, got: {version_str}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _GLibCVersion(-1, -1)
    return _GLibCVersion(int(m.group("major")), int(m.group("minor")))


@functools.lru_cache
def _get_glibc_version() -> _GLibCVersion:
    version_str = _glibc_version_string()
    if version_str is None:
        return _GLibCVersion(-1, -1)
    return _parse_glibc_version(version_str)


# From PEP 513, PEP 600
@functools.lru_cache(maxsize=1)
def _get_manylinux_module() -> types.ModuleType | None:
    """Return the ``_manylinux`` C extension module, or None if unavailable.

    The result is cached for the lifetime of the process, since the presence
    of the module does not change while running.
    """
    try:
        return __import__("_manylinux")
    except ImportError:
        return None


def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    sys_glibc = _get_glibc_version()
    if sys_glibc < version:
        return False
    # Check for presence of _manylinux module.
    manylinux_mod = _get_manylinux_module()
    if manylinux_mod is None:
        return True
    if hasattr(manylinux_mod, "manylinux_compatible"):
        result = manylinux_mod.manylinux_compatible(version[0], version[1], arch)
        if result is not None:
            return bool(result)
        return True
    if version == _GLibCVersion(2, 5) and hasattr(
        manylinux_mod, "manylinux1_compatible"
    ):
        return bool(manylinux_mod.manylinux1_compatible)
    if version == _GLibCVersion(2, 12) and hasattr(
        manylinux_mod, "manylinux2010_compatible"
    ):
        return bool(manylinux_mod.manylinux2010_compatible)
    if version == _GLibCVersion(2, 17) and hasattr(
        manylinux_mod, "manylinux2014_compatible"
    ):
        return bool(manylinux_mod.manylinux2014_compatible)
    return True


_LEGACY_MANYLINUX_MAP: dict[_GLibCVersion, str] = {
    # CentOS 7 w/ glibc 2.17 (PEP 599)
    _GLibCVersion(2, 17): "manylinux2014",
    # CentOS 6 w/ glibc 2.12 (PEP 571)
    _GLibCVersion(2, 12): "manylinux2010",
    # CentOS 5 w/ glibc 2.5 (PEP 513)
    _GLibCVersion(2, 5): "manylinux1",
}


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate manylinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be manylinux-compatible.

    :returns: An iterator of compatible manylinux tags.
    """
    if not _have_compatible_abi(sys.executable, archs):
        return
    # Oldest glibc to be supported regardless of architecture is (2, 17).
    too_old_glibc2 = _GLibCVersion(2, 16)
    if set(archs) & {"x86_64", "i686"}:
        # On x86/i686 also oldest glibc to be supported is (2, 5).
        too_old_glibc2 = _GLibCVersion(2, 4)
    current_glibc = _GLibCVersion(*_get_glibc_version())
    glibc_max_list = [current_glibc]
    # We can assume compatibility across glibc major versions.
    # https://sourceware.org/bugzilla/show_bug.cgi?id=24636
    #
    # Build a list of maximum glibc versions so that we can
    # output the canonical list of all glibc from current_glibc
    # down to too_old_glibc2, including all intermediary versions.
    for glibc_major in range(current_glibc.major - 1, 1, -1):
        glibc_minor = _LAST_GLIBC_MINOR[glibc_major]
        glibc_max_list.append(_GLibCVersion(glibc_major, glibc_minor))
    for arch in archs:
        for glibc_max in glibc_max_list:
            if glibc_max.major == too_old_glibc2.major:
                min_minor = too_old_glibc2.minor
            else:
                # For other glibc major versions oldest supported is (x, 0).
                min_minor = -1
            for glibc_minor in range(glibc_max.minor, min_minor, -1):
                glibc_version = _GLibCVersion(glibc_max.major, glibc_minor)
                if _is_compatible(arch, glibc_version):
                    yield "manylinux_{}_{}_{}".format(*glibc_version, arch)

                    # Handle the legacy manylinux1, manylinux2010, manylinux2014 tags.
                    if legacy_tag := _LEGACY_MANYLINUX_MAP.get(glibc_version):
                        yield f"{legacy_tag}_{arch}"
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import re
from typing import NewType, Union, cast

from .tags import InvalidTag, Tag, UnsortedTagsError, parse_tag
from .version import InvalidVersion, Version, _TrimmedRelease

__all__ = [
    "BuildTag",
    "InvalidName",
    "InvalidSdistFilename",
    "InvalidWheelFilename",
    "NormalizedName",
    "canonicalize_name",
    "canonicalize_version",
    "is_normalized_name",
    "parse_sdist_filename",
    "parse_wheel_filename",
]


def __dir__() -> list[str]:
    return __all__


BuildTag = Union[tuple[()], tuple[int, str]]
"""
A wheel build tag: an empty tuple, or a ``(build number, build tag suffix)`` pair.

.. versionadded:: 20.9
"""

NormalizedName = NewType("NormalizedName", str)
"""
A :class:`typing.NewType` of :class:`str`, representing a normalized name.

.. versionadded:: 20.4
"""


class InvalidName(ValueError):
    """
    An invalid distribution name; users should refer to the packaging user guide.

    .. versionadded:: 23.2
    """


class InvalidWheelFilename(ValueError):
    """
    An invalid wheel filename was found, users should refer to PEP 427.

    .. versionadded:: 20.9
    """


class InvalidSdistFilename(ValueError):
    """
    An invalid sdist filename was found, users should refer to the packaging user guide.

    .. versionadded:: 20.9
    """


# Core metadata spec for `Name`
_validate_regex = re.compile(
    r"[a-z0-9]|[a-z0-9][a-z0-9._-]*[a-z0-9]", re.IGNORECASE | re.ASCII
)
_normalized_regex = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)
# PEP 427: The build number must start with a digit.
_build_tag_regex = re.compile(r"(\d+)(.*)", re.ASCII)
# PEP 427: Valid characters for an escaped project name in a wheel filename.
# Requires at least one character so an empty project name is rejected.
_wheel_name_regex = re.compile(r"^[\w._]+\Z", re.UNICODE)


def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    """
    This function takes a valid Python package or extra name, and returns the
    normalized form of it.

    The return type is typed as :class:`NormalizedName`. This allows type
    checkers to help require that a string has passed through this function
    before use.

    If **validate** is true, then the function will check if **name** is a valid
    distribution name before normalizing.

    :param str name: The name to normalize.
    :param bool validate: Check whether the name is a valid distribution name.
    :raises InvalidName: If **validate** is true and the name is not an
        acceptable distribution name.

    >>> from packaging.utils import canonicalize_name
    >>> canonicalize_name("Django")
    'django'
    >>> canonicalize_name("oslo.concurrency")
    'oslo-concurrency'
    >>> canonicalize_name("requests")
    'requests'

    .. versionadded:: 16.2

    .. versionchanged:: 20.4
       The return type was changed to :class:`NormalizedName`.

    .. versionchanged:: 23.2
       Added the *validate* keyword parameter.
    """
    if validate and not _validate_regex.fullmatch(name):
        raise InvalidName(f"name is invalid: {name!r}")
    # Ensure all ``.`` and ``_`` are ``-``
    # Emulates ``re.sub(r"[-_.]+", "-", name).lower()`` from PEP 503
    # Much faster than re, and even faster than str.translate
    value = name.lower().replace("_", "-").replace(".", "-")
    # Condense repeats (faster than regex)
    while "--" in value:
        value = value.replace("--", "-")
    return cast("NormalizedName", value)


def is_normalized_name(name: str) -> bool:
    """
    Check if a name is a normalized project name (i.e. a valid name that
    :func:`canonicalize_name` would roundtrip to the same value).

    The roundtrip only characterizes normalized names for *valid* names. A name
    must start and end with an ASCII letter or digit, which
    :func:`canonicalize_name` does not enforce: it leaves a leading or trailing
    hyphen in place, so such a name roundtrips without being normalized.

    :param str name: The name to check.

    >>> from packaging.utils import canonicalize_name, is_normalized_name
    >>> is_normalized_name("requests")
    True
    >>> is_normalized_name("Django")
    False
    >>> canonicalize_name("_not_legal")
    '-not-legal'
    >>> is_normalized_name("-not-legal")  # roundtrips, but not a valid name
    False

    .. versionadded:: 23.2
    """
    return _normalized_regex.fullmatch(name) is not None


def canonicalize_version(
    version: Version | str, *, strip_trailing_zero: bool = True
) -> str:
    """Return a canonical form of a version as a string.

    This function takes a string representing a package version (or a
    :class:`~packaging.version.Version` instance), and returns the
    normalized form of it. By default, it strips trailing zeros from
    the release segment.

    >>> from packaging.utils import canonicalize_version
    >>> canonicalize_version('1.0.1')
    '1.0.1'

    Per PEP 625, versions may have multiple canonical forms, differing
    only by trailing zeros.

    >>> canonicalize_version('1.0.0')
    '1'
    >>> canonicalize_version('1.0.0', strip_trailing_zero=False)
    '1.0.0'

    Invalid versions are returned unaltered.

    >>> canonicalize_version('foo bar baz')
    'foo bar baz'

    >>> canonicalize_version('1.4.0.0.0')
    '1.4'

    .. versionadded:: 17.1

    .. versionchanged:: 21.0
       The return type was narrowed to :class:`str`.

    .. versionchanged:: 22.0
       Added the *strip_trailing_zero* keyword parameter.
    """
    if isinstance(version, str):
        try:
            version = Version(version)
        except InvalidVersion:
            return str(version)
    return str(_TrimmedRelease(version) if strip_trailing_zero else version)


def parse_wheel_filename(
    filename: str,
    *,
    validate_order: bool = False,
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Tag]]:
    """
    This function takes the filename of a wheel file, and parses it,
    returning a tuple of name, version, build number, and tags.

    The name part of the tuple is normalized and typed as
    :class:`NormalizedName`. The version portion is an instance of
    :class:`~packaging.version.Version`. The build number is ``()`` if
    there is no build number in the wheel filename, otherwise a
    two-item tuple of an integer for the leading digits and
    a string for the rest of the build number. The tags portion is a
    frozen set of :class:`~packaging.tags.Tag` instances (as the tag
    string format allows multiple tags to be combined into a single
    string).

    If **validate_order** is true, compressed tag set components are
    checked to be in sorted order as required by PEP 425.

    :param str filename: The name of the wheel file.
    :param bool validate_order: Check whether compressed tag set components
        are in sorted order.
    :raises InvalidWheelFilename: If the filename in question
        does not follow the :ref:`wheel specification
        <pypug:binary-distribution-format>`.

    >>> from packaging.utils import parse_wheel_filename
    >>> from packaging.tags import Tag
    >>> from packaging.version import Version
    >>> name, ver, build, tags = parse_wheel_filename("foo-1.0-py3-none-any.whl")
    >>> name
    'foo'
    >>> ver == Version('1.0')
    True
    >>> tags == {Tag("py3", "none", "any")}
    True
    >>> not build
    True

    .. versionadded:: 20.9

    .. versionchanged:: 23.2
       Raises :class:`InvalidWheelFilename` when the version component is invalid.

    .. versionadded:: 26.1
       The *validate_order* parameter.

    .. versionchanged:: 26.3
       Raises :class:`InvalidWheelFilename` when an interpreter component is
       not an identifier, a tag set component is empty, or the project name is
       empty.
    """
    if not filename.endswith(".whl"):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (extension must be '.whl'): {filename!r}"
        )

    filename = filename[:-4]
    dashes = filename.count("-")
    if dashes not in (4, 5):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (wrong number of parts): {filename!r}"
        )

    parts = filename.split("-", dashes - 2)
    name_part = parts[0]
    # See PEP 427 for the rules on escaping the project name.
    if "__" in name_part or _wheel_name_regex.match(name_part) is None:
        raise InvalidWheelFilename(f"Invalid project name: {filename!r}")
    name = canonicalize_name(name_part)

    try:
        version = Version(parts[1])
    except InvalidVersion as e:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid version): {filename!r}"
        ) from e

    if dashes == 5:
        build_part = parts[2]
        build_match = _build_tag_regex.match(build_part)
        if build_match is None:
            raise InvalidWheelFilename(
                f"Invalid build number: {build_part} in {filename!r}"
            )
        build = cast("BuildTag", (int(build_match.group(1)), build_match.group(2)))
    else:
        build = ()
    tag_str = parts[-1]
    try:
        tags = parse_tag(tag_str, validate_order=validate_order)
    except UnsortedTagsError:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (compressed tag set components must be in "
            f"sorted order per PEP 425): {filename!r}"
        ) from None
    except InvalidTag:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid tag component): {filename!r}"
        ) from None
    return (name, version, build, tags)


def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    """
    This function takes the filename of a sdist file (as specified
    in the `Source distribution format`_ documentation), and parses
    it, returning a tuple of the normalized name and version as
    represented by an instance of :class:`~packaging.version.Version`.

    :param str filename: The name of the sdist file.
    :raises InvalidSdistFilename: If the filename does not end
        with an sdist extension (``.zip`` or ``.tar.gz``), if it does not
        contain a dash separating the name and the version of the distribution,
        if the project name is empty, or if the version portion is not a valid
        version.

    >>> from packaging.utils import parse_sdist_filename
    >>> from packaging.version import Version
    >>> name, ver = parse_sdist_filename("foo-1.0.tar.gz")
    >>> name
    'foo'
    >>> ver == Version('1.0')
    True

    .. versionadded:: 20.9

    .. versionchanged:: 21.0
       Added support for ``.zip`` source distributions.

    .. versionchanged:: 23.2
       Raises :class:`InvalidSdistFilename` when the version component is invalid.

    .. versionchanged:: 26.3
       Raises :class:`InvalidSdistFilename` on an empty project name.

    .. _Source distribution format: https://packaging.python.org/specifications/source-distribution-format/#source-distribution-file-name
    """
    if filename.endswith(".tar.gz"):
        file_stem = filename[: -len(".tar.gz")]
    elif filename.endswith(".zip"):
        file_stem = filename[: -len(".zip")]
    else:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (extension must be '.tar.gz' or '.zip'):"
            f" {filename!r}"
        )

    # We are requiring a PEP 440 version, which cannot contain dashes,
    # so we split on the last dash.
    name_part, sep, version_part = file_stem.rpartition("-")
    if not sep:
        raise InvalidSdistFilename(f"Invalid sdist filename: {filename!r}")
    if not name_part:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (empty project name): {filename!r}"
        )

    name = canonicalize_name(name_part)

    try:
        version = Version(version_part)
    except InvalidVersion as e:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (invalid version): {filename!r}"
        ) from e

    return (name, version)
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""Public :class:`VersionRange` API.

A set-algebra view of the versions accepted by a
:class:`~packaging.specifiers.SpecifierSet`. Ranges support intersection,
union, complement, and difference; membership and filtering match the
originating specifier set; and conversion back to a
:class:`~packaging.specifiers.SpecifierSet` is available where a PEP 440 form
exists.

.. testsetup::

    from packaging.ranges import VersionRange
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
"""

from __future__ import annotations

import enum
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
)

from ._ranges import (
    FULL_RANGE,
    MIN_VERSION,
    NEG_INF,
    POS_INF,
    BoundaryKind,
    BoundaryVersion,
    LowerBound,
    UpperBound,
    coerce_version,
    filter_by_ranges,
    intersect_ranges,
    least_version_above,
    matches_bounds_only,
    range_is_empty,
    ranges_are_prerelease_only,
    trim_release,
)
from .version import Version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ._ranges import Interval
    from .specifiers import SpecifierSet


__all__ = ["VersionRange"]

T = TypeVar("T")
UnparsedVersion = Union[Version, str]
UnparsedVersionVar = TypeVar("UnparsedVersionVar", bound=UnparsedVersion)

#: The most ``!=`` exclusion fragments (``!=V`` points or ``!=P.*`` prefixes)
#: that :meth:`VersionRange.to_specifier_set` will materialize to spell a
#: single gap or run. Every site that expands a version-number-driven chain
#: charges it against this cap, and chains that spell one gap together share
#: it (see :func:`_decompose_dev0_gap` and :func:`_encode_gap`),
#: so no gap ever materializes more than this many exclusions. Past the cap
#: the recovery returns ``None`` rather than emit the unbounded chain a range
#: such as ``==5.* | ==1000000.*`` would otherwise drive.
_MAX_EXCLUSION_RUN = 128


class _SetOp(enum.Enum):
    """The binary set operation ``_combine_literals`` resolves over ``===`` literals."""

    INTERSECTION = enum.auto()
    UNION = enum.auto()
    DIFFERENCE = enum.auto()


def __dir__() -> list[str]:
    return __all__


# Range algebra: intersection and the empty-interval test live in the engine
# (``intersect_ranges`` / ``range_is_empty``); union and complement are only
# needed here, so they live in this module.


def _union_ranges(
    left: Sequence[Interval],
    right: Sequence[Interval],
) -> list[Interval]:
    """Union two sorted, non-overlapping interval lists.

    A linear merge over the two pre-sorted inputs followed by a single
    coalescing pass: adjacent or overlapping intervals collapse so the result
    is itself sorted and non-overlapping.
    """
    if not left:
        return list(right)
    if not right:
        return list(left)

    merged_input: list[Interval] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index][0] <= right[right_index][0]:
            merged_input.append(left[left_index])
            left_index += 1
        else:
            merged_input.append(right[right_index])
            right_index += 1
    merged_input.extend(left[left_index:])
    merged_input.extend(right[right_index:])

    merged: list[Interval] = [merged_input[0]]
    for lower, upper in merged_input[1:]:
        prev_lower, prev_upper = merged[-1]

        if (
            prev_upper.version is None
            or lower.version is None
            or prev_upper.version > lower.version
        ):
            overlaps = True
        elif prev_upper.version == lower.version:
            overlaps = prev_upper.inclusive or lower.inclusive
        else:
            # An ordering gap may still hold no version when the two bounds
            # straddle a synthetic boundary; merge across an empty gap to
            # stay canonical.
            gap_lower = LowerBound(prev_upper.version, not prev_upper.inclusive)
            gap_upper = UpperBound(lower.version, not lower.inclusive)
            overlaps = range_is_empty(gap_lower, gap_upper)

        if overlaps:
            merged[-1] = (prev_lower, max(prev_upper, upper))
        else:
            merged.append((lower, upper))

    return merged


def _complement_ranges(ranges: Sequence[Interval]) -> list[Interval]:
    """Complement a sorted, non-overlapping interval list.

    Yields the gaps between intervals plus a leading gap before the first and
    a trailing gap after the last. Bound inclusivity flips so that
    complement-of-complement round-trips back to the input.
    """
    if not ranges:
        return list(FULL_RANGE)

    result: list[Interval] = []
    prev_upper: UpperBound | None = None

    for lower, upper in ranges:
        if prev_upper is None:
            # Leading gap below the first interval. Every range reaching here is
            # floor-canonical: ``_canonical_floor`` has already folded an
            # inclusive lower at or below ``0.dev0`` into ``-inf``. So a finite
            # first lower always leaves a non-empty gap down to ``-inf``, while a
            # ``-inf`` lower leaves no leading gap at all.
            if lower.version is not None:
                gap_upper = UpperBound(lower.version, not lower.inclusive)
                result.append((NEG_INF, gap_upper))
        else:
            gap_lower = LowerBound(prev_upper.version, not prev_upper.inclusive)
            gap_upper = UpperBound(lower.version, not lower.inclusive)
            # Input intervals are canonical (sorted, disjoint, non-touching),
            # so the gap between two of them always holds at least one version.
            result.append((gap_lower, gap_upper))
        prev_upper = upper

    # The empty-input early return guarantees the loop ran.
    assert prev_upper is not None
    if prev_upper.version is not None:
        gap_lower = LowerBound(prev_upper.version, not prev_upper.inclusive)
        result.append((gap_lower, POS_INF))

    return result


def _canonical_floor(bounds: tuple[Interval, ...]) -> tuple[Interval, ...]:
    """Collapse the PEP 440 floor in a sorted interval list.

    Only the first interval can touch ``0.dev0`` (the minimum version). An
    inclusive lower at or below it admits everything below, the same as
    ``-inf``, so ``>=0.dev0`` becomes the one canonical full range. An
    exclusive upper at or below it leaves the interval empty, so it is dropped.
    """
    if not bounds:
        return bounds

    lower, upper = bounds[0]
    if range_is_empty(NEG_INF, upper):
        return bounds[1:]

    if (
        lower.inclusive
        and isinstance(lower.version, Version)
        and lower.version <= MIN_VERSION
    ):
        return ((NEG_INF, upper), *bounds[1:])

    return bounds


def _predecessor_boundary(version: Version) -> BoundaryVersion | None:
    """The boundary whose least successor is *version*, or ``None``.

    Inverse of :func:`~packaging._ranges.least_version_above`. A plain version
    that is exactly such a successor (``1.0a2.dev0`` sits just above
    ``AFTER_POSTS(1.0a1)``) folds back to that boundary, so ``>=1.0a2.dev0`` and
    ``>1.0a1`` share one form. The proposed boundary is confirmed by
    round-tripping through ``least_version_above``.
    """
    # Only a least successor carries a dev segment, so nothing else can fold.
    if version.dev is None:
        return None

    candidate: BoundaryVersion | None = None
    if version.pre is not None and version.dev == 0 and version.post is None:
        # 1.0a2.dev0 -> AFTER_POSTS(1.0a1)
        kind, number = version.pre
        if number >= 1:
            candidate = BoundaryVersion(
                version.__replace__(pre=(kind, number - 1), dev=None),
                BoundaryKind.AFTER_POSTS,
            )
    elif version.dev >= 1:
        # 1.0.dev3 -> AFTER_LOCALS(1.0.dev2)
        candidate = BoundaryVersion(
            version.__replace__(dev=version.dev - 1), BoundaryKind.AFTER_LOCALS
        )
    elif version.dev == 0 and version.post is not None:
        # 1.0.post1.dev0 -> AFTER_LOCALS(1.0.post0); 1.0.post0.dev0 -> AFTER_LOCALS(1.0)
        base = (
            version.__replace__(post=None, dev=None)
            if version.post == 0
            else version.__replace__(post=version.post - 1, dev=None)
        )
        candidate = BoundaryVersion(base, BoundaryKind.AFTER_LOCALS)

    if candidate is not None and least_version_above(candidate) == version:
        return candidate
    return None


def _canonicalize(bounds: tuple[Interval, ...]) -> tuple[Interval, ...]:
    """Fold least-successor bounds to their boundary form.

    ``>=1.0a2.dev0`` and ``>1.0a1`` denote the same set, so both must reduce to
    one representation for ``==`` and ``hash`` to agree. An inclusive lower or
    exclusive upper sitting on a boundary's least successor becomes that
    boundary; the engine's emptiness check has already dropped the synthetic
    gaps such intervals would otherwise leave.
    """
    result: list[Interval] = []
    for lower, upper in bounds:
        new_lower, new_upper = lower, upper

        if isinstance(lower.version, Version) and lower.inclusive:
            boundary = _predecessor_boundary(lower.version)
            if boundary is not None:
                new_lower = LowerBound(boundary, inclusive=False)

        if isinstance(upper.version, Version) and not upper.inclusive:
            boundary = _predecessor_boundary(upper.version)
            if boundary is not None:
                new_upper = UpperBound(boundary, inclusive=True)

        result.append((new_lower, new_upper))
    return tuple(result)


def _struct_admits(
    bounds: tuple[Interval, ...], admit_arbitrary: bool, literal: str
) -> bool:
    """True when the bounds (plus arbitrary admission) admit ``literal``.

    Skips the explicit admit/reject sets, which the caller layers on top. A
    non-version string matches via ``admit_arbitrary`` only on full bounds;
    on narrower bounds the flag is metadata only.
    """
    parsed = coerce_version(literal)
    if parsed is None:
        return admit_arbitrary and bounds == FULL_RANGE

    return matches_bounds_only(bounds, parsed)


# Repr helpers:


def _bound_version_str(value: BoundaryVersion | Version) -> str:
    """Printout for a bound's inner value, kind-tagged for boundaries."""
    if isinstance(value, BoundaryVersion):
        return f"{value.version}[{value.kind.name}]"
    return str(value)


def _format_lower(bound: LowerBound) -> str:
    if bound.version is None:
        return "(-inf"
    bracket = "[" if bound.inclusive else "("
    return f"{bracket}{_bound_version_str(bound.version)}"


def _format_upper(bound: UpperBound) -> str:
    if bound.version is None:
        return "+inf)"
    bracket = "]" if bound.inclusive else ")"
    return f"{_bound_version_str(bound.version)}{bracket}"


def _format_intervals(intervals: Sequence[Interval]) -> str:
    """Render a sorted interval list as ``lower, upper | lower, upper``."""
    return " | ".join(
        f"{_format_lower(lower)}, {_format_upper(upper)}" for lower, upper in intervals
    )


# ``to_specifier_set`` recovery: encode a range's interval list back into
# specifier fragments. Each helper returns ``None`` when its shape has no
# PEP 440 form. The ``keep_dev0`` argument threaded through is a spelling
# mode, not a pre-release policy: false emits a prerelease-free form (no
# synthetic ``.dev0``, so the recovered range has an empty opt-in region),
# true keeps the ``.dev0`` markers (so the range opts its bounds in).
# ``to_specifier_set`` encodes in both modes and keeps whichever round-trips.
#
# Bound and interval encoding: turn one interval's bounds into fragments.


def _is_dev0_version(version: Version) -> bool:
    """True when version is exactly ``X[.Y]*.dev0`` (the shape ``<X`` makes)."""
    return (
        version.dev == 0
        and version.pre is None
        and version.post is None
        and version.local is None
    )


def _clean_lower(version: Version) -> list[str] | None:
    """A prerelease-free spelling for an inclusive ``[version`` lower, or ``None``.

    Several ``[V`` lowers come from an operator whose own spelling carries no
    synthetic ``.dev0``. Recovering that spelling gives the range an empty opt-in
    region, so it is offered in the prerelease-free spelling mode (see
    :meth:`VersionRange.to_specifier_set`).
    """
    if version.dev != 0 or version.pre is not None or version.local is not None:
        return None

    # ``[B.post(k).dev0`` is the lower ``>B.post(k-1)`` builds (k >= 1).
    if version.post is not None:
        if version.post < 1:
            return None
        return [f">{version.__replace__(post=version.post - 1, dev=None)}"]

    # ``[F.dev0`` is family F's base. The prefix P just below F has
    # ``==P.* == [P.dev0, F.dev0)``, so ``>=P,!=P.*`` lands exactly on ``[F.dev0``.
    family = trim_release(version.release)
    last = family[-1]
    if last < 1:
        return None

    below_release = (*family[:-1], last - 1)
    below = Version.from_parts(epoch=version.epoch, release=below_release)

    # At the epoch-0 floor ``==P.*`` already reaches ``0.dev0``, so the ``>=P``
    # half is redundant: ``[1.dev0, +inf)`` is plain ``!=0.*``.
    if version.epoch == 0 and not any(below_release):
        return [f"!={below}.*"]

    return [f">={below}", f"!={below}.*"]


def _epoch_floor_lower(
    lower: LowerBound, upper: UpperBound
) -> tuple[Version, int, bool] | None:
    """The ``E!0`` family of a lower sitting on an epoch>0 zero-family floor.

    An epoch>0 zero-family base such as ``1!0.dev0`` has no ``>=P,!=P.*`` spelling
    since no version sorts below ``E!0`` within the epoch. While the interval
    stays within ``==E!0.*`` it is that wildcard, trimmed by the upper and with a
    leading ``.dev`` run excluded: an ``AFTER_LOCALS(E!0.dev(k))`` lower drops
    ``E!0.dev0..E!0.dev(k)``, a plain inclusive ``E!0.dev0`` lower drops none.
    Returns the ``E!0`` family, how many leading ``.dev`` releases to exclude, and
    whether the upper sits at the family cap (so ``==E!0.*`` needs no upper), else
    ``None``.
    """
    version = lower.version
    if isinstance(version, BoundaryVersion):
        if version.kind != BoundaryKind.AFTER_LOCALS:
            return None
        version = version.version
        if version.dev is None:
            return None
        excluded_devs = version.dev + 1
    elif isinstance(version, Version) and lower.inclusive:
        # A plain inclusive lower only reaches the floor as ``>=E!0.dev0``;
        # higher ``.dev`` would have canonicalized to an AFTER_LOCALS boundary.
        if version.dev != 0:
            return None
        excluded_devs = 0
    else:
        return None

    # Only the bare ``E!0`` floor of a non-zero epoch qualifies.
    if version.epoch == 0:
        return None
    if version.pre is not None or version.post is not None or version.local is not None:
        return None
    if any(trim_release(version.release)):
        return None

    # ``==E!0.*`` spans ``[E!0.dev0, E!1.dev0)``; it fits only below that cap.
    next_family = Version.from_parts(epoch=version.epoch, release=(1,), dev=0)
    cap = UpperBound(next_family, False)
    if upper > cap:
        return None

    family = Version.from_parts(epoch=version.epoch, release=(0,))
    return family, excluded_devs, upper == cap


def _dev_family_anchor(family: Version) -> list[str] | None:
    """Prerelease-free fragments for ``[family, ..)``, or ``None`` if it has none.

    ``family`` is an ``X.dev0``. The floor gives ``[]`` (every version); a release
    base its ``_clean_lower`` family-floor spelling (``!=0.*`` ...); an ``X.post0``
    base ``>=X,!=X``. A pre-release base has no prerelease-free spelling.
    """
    if family <= MIN_VERSION:
        return []
    clean = _clean_lower(family)
    if clean is not None:
        return clean
    if family.pre is None and family.post == 0:
        base = family.__replace__(post=None, dev=None)
        return [f">={base}", f"!={base}"]
    return None


def _encode_lower(lower: LowerBound, keep_dev0: bool) -> list[str] | None:
    """Encode a lower bound as specifier fragments, or ``None``.

    ``[]`` for ``-inf``. An ``AFTER_POSTS(V)`` lower is ``>V``. An
    ``AFTER_LOCALS(V)`` lower is the set ``[successor, ..)`` and emits ``>=V,!=V``,
    except in the prerelease-free spelling mode (``keep_dev0`` false), where it
    recovers a spelling with no synthetic ``.dev0`` when one exists:
    ``>3.8.post1`` for a post release, or a dev family's anchor plus the dev run
    up to V for a ``.dev`` release.
    """
    lower_version = lower.version
    if lower_version is None:
        return []

    if isinstance(lower_version, BoundaryVersion):
        if lower_version.kind == BoundaryKind.AFTER_POSTS:
            # AFTER_POSTS only ever appears as an exclusive ``>V`` lower.
            return [f">{lower_version.version}"]
        inner = lower_version.version
        if inner <= MIN_VERSION:
            # The ``(-inf, V)`` side was dropped at the floor, so the lone
            # ``(AFTER_LOCALS(0.dev0), +inf)`` interval is exactly ``!=0.dev0``.
            return [f"!={inner}"]
        # An ``(AFTER_LOCALS(V), ..)`` lower is the set ``[successor, ..)``. In
        # the prerelease-free mode, recover a spelling with no synthetic ``.dev0``
        # so the range's opt-in region stays empty.
        if not keep_dev0:
            if inner.dev is not None:
                # A ``.dev`` V makes ``[successor, ..)`` its dev family's
                # prerelease-free anchor minus the finite dev run up to ``V``
                # (e.g. ``>=1.0,!=1.0,!=1.0.post0.dev0``), carrying no synthetic
                # ``.dev0``.
                family = inner.__replace__(dev=0)
                anchor = _dev_family_anchor(family)
                if anchor is not None:
                    if inner.dev + 1 > _MAX_EXCLUSION_RUN:
                        return None
                    run = [
                        f"!={family.__replace__(dev=d)}" for d in range(inner.dev + 1)
                    ]
                    return anchor + run
            else:
                # A ``.post`` release recovers its prerelease-free ``>`` spelling
                # from the successor (``>3.8.post1`` for ``AFTER_LOCALS(3.8.post1)``).
                successor = least_version_above(lower_version)
                clean = _clean_lower(successor) if successor is not None else None
                if clean is not None:
                    return clean
        # Otherwise it is ``[V, ..)`` minus V's local family, i.e. ``>=V,!=V``.
        # The prerelease-free recoveries above have already returned in that mode;
        # a residual ``.dev`` V here opts pre-releases in, so this spelling round
        # trips only for a range whose opt-in region wants it.
        return [f">={inner}", f"!={inner}"]

    if not lower.inclusive:
        return None

    # In the prerelease-free mode a ``.dev0`` lower prefers its clean spelling.
    if not keep_dev0:
        clean = _clean_lower(lower_version)
        if clean is not None:
            return clean
    return [f">={lower_version}"]


def _encode_upper(upper: UpperBound, keep_dev0: bool) -> list[str] | None:
    """Encode an upper bound as specifier fragments, or ``None``.

    ``[]`` for ``+inf``. In the prerelease-free spelling mode (``keep_dev0``
    false) the ``<X`` spelling is used for the ``X.dev0`` upper that ``<X``
    builds; otherwise the synthetic ``.dev0`` is kept so the range opts its
    bounds in.
    """
    upper_version = upper.version
    if upper_version is None:
        return []

    if isinstance(upper_version, BoundaryVersion):
        # A boundary upper is always inclusive (a boundary already sits between
        # versions, so no specifier produces an exclusive one).
        if upper_version.kind == BoundaryKind.AFTER_LOCALS:
            inner = upper_version.version
            if (
                not keep_dev0
                and inner.pre is None
                and inner.post is not None
                and inner.dev is None
            ):
                # ``AFTER_LOCALS(post-release)]`` upper is ``<next-post`` (e.g.
                # ``<3.8.post1`` for ``AFTER_LOCALS(3.8.post0)``), the ``<P``
                # spelling a post-release upper builds, with no ``.dev0``.
                return [f"<{inner.__replace__(post=inner.post + 1)}"]
            return [f"<={inner}"]
        # ``AFTER_POSTS(P)`` sits just below the next pre-release's ``.dev0``
        # (canonicalization of a ``<P.preN.dev0`` upper), so it is ``<`` that
        # least successor. A final-release AFTER_POSTS has no successor or form.
        successor = least_version_above(upper_version)
        if successor is not None:
            return [f"<{successor}"]
        return None

    if not upper.inclusive:
        # ``<X`` builds an exclusive ``X.dev0`` upper (X final or post-release;
        # never pre/local). ``<X`` and ``<X.dev0`` define the same bound but
        # differ in the opt-in they imply, so pick by the spelling mode.
        if (
            upper_version.dev == 0
            and upper_version.pre is None
            and upper_version.local is None
        ):
            if not keep_dev0:
                return [f"<{upper_version.__replace__(dev=None)}"]
            return [f"<{upper_version}"]
        # ``V`` (exclusive) upper, including V's pre-releases.
        return [f"<={upper_version}", f"!={upper_version}"]
    return None


def _detect_equal_wildcard(lower: LowerBound, upper: UpperBound) -> Version | None:
    """If ``[lower, upper)`` is the ``==V.*`` shape, return ``V``."""
    if isinstance(lower.version, BoundaryVersion) or isinstance(
        upper.version, BoundaryVersion
    ):
        return None
    if lower.version is None or upper.version is None:
        return None
    if not lower.inclusive or upper.inclusive:
        return None
    if not (_is_dev0_version(lower.version) and _is_dev0_version(upper.version)):
        return None
    if lower.version.epoch != upper.version.epoch:
        return None

    lower_release = trim_release(lower.version.release)
    upper_release = trim_release(upper.version.release)
    padded_length = max(len(lower_release), len(upper_release))
    assert padded_length > 0
    lower_release += (0,) * (padded_length - len(lower_release))
    upper_release += (0,) * (padded_length - len(upper_release))

    if lower_release[:-1] != upper_release[:-1]:
        return None

    # A genuine ``==V.*`` spans one family: the upper is exactly the next prefix.
    # A wider span like ``[3.8.dev0, 3.14.dev0)`` shares the prefix but is not a
    # single wildcard, so it falls through to the generic ``>=...,<...`` form.
    if upper_release[-1] != lower_release[-1] + 1:
        return None

    return lower.version.__replace__(release=lower_release, dev=None)


def _encode_interval(
    lower: LowerBound, upper: UpperBound, keep_dev0: bool
) -> list[str] | None:
    """Encode one interval as specifier fragments, or ``None``.

    Special-cases the ``==V`` singleton (``[V, AFTER_LOCALS(V)]`` for a plain
    ``V``, and ``[V+local, V+local]`` for a local one) and the ``==V.*`` shape
    so the fragment is one equality rather than a bound pair.
    """
    # ``[V+local, V+local]`` (an inclusive local point) is the singleton ``==V+local``.
    if (
        lower.version is not None
        and upper.version is not None
        and not isinstance(lower.version, BoundaryVersion)
        and not isinstance(upper.version, BoundaryVersion)
        and lower.inclusive
        and upper.inclusive
        and lower.version == upper.version
        and lower.version.local is not None
    ):
        return [f"=={lower.version}"]

    # ``[V, AFTER_LOCALS(V)]`` (V without a local) is the singleton ``==V``,
    # which also matches V's local family: one equality, not ``>=V,<=V``.
    if (
        isinstance(lower.version, Version)
        and lower.inclusive
        and upper.inclusive
        and isinstance(upper.version, BoundaryVersion)
        and upper.version.kind == BoundaryKind.AFTER_LOCALS
        and upper.version.version == lower.version
    ):
        return [f"=={lower.version}"]

    wildcard = _detect_equal_wildcard(lower, upper)
    if wildcard is not None:
        return [f"=={wildcard}.*"]

    # A ``[E!0.dev0`` lower has no prerelease-free ``>=`` spelling; within its own
    # family it is ``==E!0.*`` trimmed by the upper.
    floor = _epoch_floor_lower(lower, upper) if not keep_dev0 else None
    if floor is not None:
        family, excluded_devs, upper_at_cap = floor
        if excluded_devs > _MAX_EXCLUSION_RUN:
            return None
        parts = [f"=={family}.*"]
        parts.extend(f"!={family.__replace__(dev=d)}" for d in range(excluded_devs))

        # ``==E!0.*`` already caps at the next family; add the upper only if tighter.
        if not upper_at_cap:
            upper_parts = _encode_upper(upper, keep_dev0)
            if upper_parts is None:
                return None
            parts.extend(upper_parts)

        return parts

    lower_parts = _encode_lower(lower, keep_dev0)
    if lower_parts is None:
        return None

    upper_parts = _encode_upper(upper, keep_dev0)
    if upper_parts is None:
        return None

    return lower_parts + upper_parts


# Gap encoding: spell the gap between two adjacent intervals as exclusions.


def _detect_not_equal(
    left_upper: UpperBound, right_lower: LowerBound
) -> list[Version] | None:
    """If the gap between two intervals is a ``!=V`` chain, list its points.

    A plain exclusive left upper names the first excluded V directly; an inclusive
    boundary left upper names it via its least successor. Adjacent exclusions
    (``V`` and its immediate successors) share a single gap spanning a contiguous
    dev run, so one gap can name a short chain: ``!=1.0,!=1.0.post0.dev0`` is one
    gap from ``1.0`` up to ``AFTER_LOCALS(1.0.post0.dev0)``.
    """
    if isinstance(left_upper.version, BoundaryVersion):
        # A boundary upper is always inclusive; its least successor is the first
        # excluded V (``None`` for a final AFTER_POSTS, which names no point).
        first = least_version_above(left_upper.version)
        if first is None:
            return None
    elif left_upper.version is None or left_upper.inclusive:
        return None
    else:
        first = left_upper.version

    if not isinstance(right_lower.version, BoundaryVersion):
        if (
            right_lower.version is not None
            and not right_lower.inclusive
            and right_lower.version == first
            and first.local is not None
        ):
            return [first]
        return None

    if right_lower.version.kind != BoundaryKind.AFTER_LOCALS:
        return None

    # The right interval resumes just above the last excluded V and its locals.
    last = right_lower.version.version
    if first == last:
        return [first]

    # Adjacent exclusions: the successor of ``first`` opens a ``.dev`` family and
    # every later point is a higher ``.dev`` in that same family, so the gap is
    # exactly ``first`` plus a contiguous dev run up to ``last``. Any other gap
    # (e.g. ``2.3`` to ``AFTER_LOCALS(2.7)`` from complementing ``>=2.3,<=2.7``)
    # spans a whole interval and fails this test.
    second = least_version_above(BoundaryVersion(first, BoundaryKind.AFTER_LOCALS))
    if (
        second is not None
        and second.dev is not None
        and last.dev is not None
        and last.dev >= second.dev
        and last.__replace__(dev=second.dev) == second
    ):
        # ``first`` and the run spell this gap together, so they share the cap.
        if last.dev - second.dev + 2 > _MAX_EXCLUSION_RUN:
            return None
        run = (second.__replace__(dev=d) for d in range(second.dev, last.dev + 1))
        return [first, *run]
    return None


def _decompose_dev0_gap(
    lower_trim: tuple[int, ...],
    upper_trim: tuple[int, ...],
    epoch: int,
    budget: int = _MAX_EXCLUSION_RUN,
) -> list[Version] | None:
    """Decompose the gap ``[L.dev0, U.dev0)`` into wildcard prefixes.

    ``lower_trim``/``upper_trim`` are trimmed release tuples with
    ``lower_trim < upper_trim`` lexicographically. The chain sweeps at the
    first differing level. The gap is undecomposable when L has trailing
    components below that level (the chain cannot escape L's subtree), or when
    the chain, summed across levels, would exceed ``budget`` prefixes.
    """
    diff = 0
    while (
        diff < len(lower_trim)
        and diff < len(upper_trim)
        and lower_trim[diff] == upper_trim[diff]
    ):
        diff += 1

    if len(lower_trim) > diff + 1:
        return None

    common = lower_trim[:diff]
    lower_val = lower_trim[diff] if len(lower_trim) > diff else 0
    upper_val = upper_trim[diff]

    span = upper_val - lower_val
    if span > budget:
        return None

    fragments = [
        Version.from_parts(epoch=epoch, release=(*common, segment))
        for segment in range(lower_val, upper_val)
    ]

    if len(upper_trim) == diff + 1:
        return fragments

    # Recurse into the next release component, charging at least one to the budget
    # per level (not just the span), so a run of zero-span levels (a release with
    # many trailing components) exhausts the budget and returns None instead of
    # recursing past the interpreter's stack limit.
    tail = _decompose_dev0_gap(
        (*common, upper_val), upper_trim, epoch, budget - max(span, 1)
    )
    if tail is None:
        return None
    return fragments + tail


def _encode_gap(left_upper: UpperBound, right_lower: LowerBound) -> list[str] | None:
    """Encode the gap between two adjacent intervals as ``!=`` fragments.

    A point chain becomes ``!=V`` fragments and a dev0 family span becomes
    ``!=P.*`` prefixes, followed by a leading dev run in the last family when
    the gap ends inside it. Any other gap has no exclusion form and returns
    ``None``.
    """
    # Cheapest first: a plain ``!=V`` chain never pays for the budgeted
    # wildcard sweep below.
    points = _detect_not_equal(left_upper, right_lower)
    if points is not None:
        return [f"!={point}" for point in points]

    # Both wildcard shapes start at an exclusive family base ``L.dev0``.
    left_v = left_upper.version
    if (
        not isinstance(left_v, Version)
        or left_upper.inclusive
        or not _is_dev0_version(left_v)
    ):
        return None

    right_v = right_lower.version
    if isinstance(right_v, Version) and right_lower.inclusive:
        # ``[L.dev0, U.dev0)`` is a pure ``!=P.*`` chain up to U's family.
        upper_dev0 = right_v
        run_length = 0
        budget = _MAX_EXCLUSION_RUN
    elif (
        isinstance(right_v, BoundaryVersion)
        and right_v.kind == BoundaryKind.AFTER_LOCALS
        and not right_lower.inclusive
    ):
        # ``[L.dev0, AFTER_LOCALS(U.dev(k))]`` ends inside U's family: the
        # ``!=P.*`` chain, then the ``U.dev0..U.dev(k)`` run. Both spell the
        # gap together, so they share one ``_MAX_EXCLUSION_RUN`` budget.
        upper = right_v.version
        if upper.dev is None or upper.dev + 1 > _MAX_EXCLUSION_RUN:
            return None
        upper_dev0 = upper.__replace__(dev=0)
        run_length = upper.dev + 1
        budget = _MAX_EXCLUSION_RUN - run_length
    else:
        return None

    # ``U`` must be a plain release base in L's epoch, above L.
    # ``_is_dev0_version`` on ``U.dev0`` rejects any pre/post/local on ``U``.
    if not _is_dev0_version(upper_dev0):
        return None
    if left_v.epoch != upper_dev0.epoch or left_v >= upper_dev0:
        return None

    prefixes = _decompose_dev0_gap(
        trim_release(left_v.release),
        trim_release(upper_dev0.release),
        left_v.epoch,
        budget,
    )
    if prefixes is None:
        return None

    exclusions = [f"!={prefix}.*" for prefix in prefixes]
    exclusions.extend(f"!={upper_dev0.__replace__(dev=d)}" for d in range(run_length))
    return exclusions


def _encode_gaps(bounds: Sequence[Interval]) -> list[str] | None:
    """Encode every between-interval gap as ``!=`` fragments, or ``None``.

    When each gap has an exclusion spelling, the intervals fuse into one
    contiguous span (``==1.* | ==3.*`` is ``!=0.*,!=2.*,<4``): the outer
    interval across all the bounds plus these exclusions. A gap with no
    exclusion spelling makes the bounds a disjoint union, which no single
    set expresses.
    """
    exclusions: list[str] = []

    for index in range(1, len(bounds)):
        gap = _encode_gap(bounds[index - 1][1], bounds[index][0])
        if gap is None:
            return None
        exclusions.extend(gap)

    return exclusions


def _tighten_no_prereleases(bounds: tuple[Interval, ...]) -> tuple[Interval, ...]:
    """Snap the range's final upper out of the pre-release band ``False`` drops.

    An exclusive upper at a final ``V`` admits the versions in ``[V.dev0, V)`` at
    the bounds level, but a ``prereleases=False`` policy filters them all out, so
    it accepts the same releases as ``<V`` (upper at ``V.dev0``). Snapping it lets
    :meth:`VersionRange.to_specifier_set` reach the ``<V`` spelling.

    Only the last interval's upper is snapped, the one that gives a terser outer
    bound. Inner uppers are left alone: snapping one turns its gap to the next
    interval into a ``.dev0`` wildcard gap, which a far-apart neighbour would blow
    up into an unbounded ``!=N.*`` chain. Those shapes recover as ``None`` here,
    the same as under ``None`` / ``True``. The snap is conservative (it skips
    boundary, pre-release, and local uppers); the caller keeps it only when it
    stays release-equivalent, so an unsnapped shape falls back to the exact form.
    """
    lower, upper = bounds[-1]
    version = upper.version
    if (
        isinstance(version, Version)
        and not upper.inclusive
        and not version.is_prerelease
        and version.local is None
    ):
        upper = UpperBound(version.__replace__(dev=0), inclusive=False)
        return (*bounds[:-1], (lower, upper))
    return bounds


class VersionRange:
    """A set of :class:`~packaging.version.Version` values accepted by a
    :class:`~packaging.specifiers.SpecifierSet`.

    Construct via :meth:`~packaging.specifiers.SpecifierSet.to_range`, or with
    the :meth:`full`, :meth:`empty`, and :meth:`singleton` class methods.
    Compose with :meth:`intersection`, :meth:`union`, :meth:`complement`, and
    :meth:`difference` (or the ``&`` / ``|`` / ``~`` / ``-`` operators). Test
    membership with ``in`` or :meth:`contains`, filter an iterable with
    :meth:`filter`, and convert back to a
    :class:`~packaging.specifiers.SpecifierSet` with :meth:`to_specifier_set`.

    The configured pre-release policy of the originating specifier set carries
    onto the range and controls whether pre-releases are admitted under ``in``,
    :meth:`contains`, and :meth:`filter`. With no configured policy,
    :meth:`filter` also admits pre-releases in the autodetected opt-in region
    (the versions a pre-release-naming specifier asked for). Set algebra keeps
    that opt-in scoped to those versions, so unrelated pre-releases are not
    admitted wholesale.

    :meth:`intersection`, :meth:`union`, :meth:`difference`, and the
    :meth:`is_subset` / :meth:`is_superset` / :meth:`is_disjoint` predicates
    require both operands to share the same configured policy.

    >>> r = SpecifierSet(">=1.0,<2.0").to_range()
    >>> "1.5" in r
    True
    >>> "2.0" in r
    False
    >>> SpecifierSet(">=2.0,<1.0").to_range().is_empty
    True

    PEP 440's ``===`` operator matches a candidate string verbatim
    (case-insensitive) rather than a set of versions. Ranges built from
    ``===`` specifiers still support membership, set operations, and conversion
    back to a :class:`~packaging.specifiers.SpecifierSet`; matching follows the
    literal-equality rule. A ``===`` literal that names a pre-release is
    admitted under the default policy by both :meth:`contains` and
    :meth:`filter`, since it was named outright.

    .. versionadded:: 26.3
    """

    __slots__ = (
        "_admit",
        "_admit_arbitrary",
        "_bounds",
        "_pre_region",
        "_prereleases_configured",
        "_reject",
    )

    #: The disjoint, sorted, non-overlapping interval list.
    _bounds: tuple[Interval, ...]

    #: Whether this range matches non-version strings as well as versions.
    #: True only by construction on ``SpecifierSet("")`` / :meth:`full`. The flag
    #: rides set algebra but is inert except at full bounds (see
    #: :meth:`_arbitrary_active`). An intersection or difference that shrinks
    #: the bounds drops it (``full() & ~full()`` is plain empty, and
    #: ``full() - r == full() & ~r``); :meth:`complement` and a union of
    #: empty-bounds operands keep it, so ``~~full() == full()`` and
    #: ``~full() | ~full() == ~full()``. Part of equality, since membership
    #: reads it.
    _admit_arbitrary: bool

    #: Case-folded strings the range admits in addition to its bounds.
    #: ``===wat`` produces ``_admit = {"wat"}``.
    _admit: frozenset[str]

    #: Case-folded strings the range rejects (overrides ``_admit`` and the
    #: bounds). Populated by :meth:`complement` of an admit-bearing range and by
    #: literal resolution in :meth:`_combine_literals`.
    _reject: frozenset[str]

    #: Sorted, disjoint intervals where pre-releases are force-admitted under
    #: the PEP 440 default policy (a ``None`` ``prereleases`` argument and no
    #: configured override). The opt-in flows only from the pre-release-naming
    #: specifiers that built the range. :meth:`_build` clips the region to the
    #: bounds, so it is always a subset of them: an opt-in that overflowed its
    #: own cap cannot ride a later union into versions no specifier asked for.
    #: :meth:`union` and :meth:`intersection` accumulate the operands' clipped
    #: regions and re-clip to the result bounds; :meth:`difference` keeps only
    #: the minuend's; and :meth:`complement` drops it, since an exclusion grants
    #: no opt-in. Equality keys on the clipped region, so it stays a congruence.
    _pre_region: tuple[Interval, ...]

    #: Raw configured pre-release override of the originating specifier set
    #: (an explicit ``True`` / ``False``, else ``None``). When set, :meth:`_build`
    #: forces ``_pre_region`` empty since the policy governs globally.
    #: :meth:`intersection` and :meth:`union` require it to match on both
    #: operands. Part of equality.
    _prereleases_configured: bool | None

    def __new__(cls, *args: object, **kwargs: object) -> VersionRange:  # noqa: PYI034
        raise TypeError(
            "cannot create 'VersionRange' instances directly; use "
            "SpecifierSet.to_range(), VersionRange.full(), "
            "VersionRange.empty(), or VersionRange.singleton() instead"
        )

    @classmethod
    def _build(
        cls,
        bounds: tuple[Interval, ...],
        admit: frozenset[str] = frozenset(),
        reject: frozenset[str] = frozenset(),
        admit_arbitrary: bool = False,
        *,
        pre_region: tuple[Interval, ...] = (),
        prereleases_configured: bool | None = None,
    ) -> VersionRange:
        """Internal factory; bypasses :meth:`__new__`.

        Canonicalizes the bounds so equal version sets share one representation,
        then drops admit literals the bounds already admit and reject literals
        the bounds do not match anyway. Reject wins over admit on overlap. The
        pre-release policy is set here and never reassigned afterwards;
        ``pre_region`` is canonicalized like the bounds and clipped to them,
        or dropped when a configured policy makes it inert.
        """
        bounds = _canonicalize(bounds)

        if admit and reject:
            admit = admit - reject
        if admit:
            admit = frozenset(
                literal
                for literal in admit
                if not _struct_admits(bounds, admit_arbitrary, literal)
            )
        if reject:
            reject = frozenset(
                literal
                for literal in reject
                if _struct_admits(bounds, admit_arbitrary, literal)
            )

        instance = object.__new__(cls)
        instance._bounds = bounds
        instance._admit = admit
        instance._reject = reject
        instance._admit_arbitrary = admit_arbitrary
        instance._prereleases_configured = prereleases_configured

        # A configured policy makes the region inert, so drop it. Otherwise fold
        # least-successor bounds (_from_specifier_set passes the region unfolded),
        # so ``>1.0a1`` and ``>=1.0a2.dev0`` carry the same region, then clip it
        # to the bounds so the opt-in never reaches past the range's own versions.
        if prereleases_configured is not None or not pre_region:
            instance._pre_region = ()
        else:
            instance._pre_region = tuple(
                intersect_ranges(_canonicalize(pre_region), bounds)
            )

        return instance

    def _has_literals(self) -> bool:
        return bool(self._admit) or bool(self._reject)

    def _arbitrary_active(self) -> bool:
        """True when ``_admit_arbitrary`` actually admits non-version strings.

        The flag rides through set algebra but only fires admission on full
        bounds. Intersection and difference drop it when the bounds shrink, so
        away from full bounds it survives only on empty-bounds ranges, where
        it keeps ``~~full() == full()`` and union idempotent.
        """
        return self._admit_arbitrary and self._bounds == FULL_RANGE

    def _is_plain(self) -> bool:
        """True when membership is decided by ``_bounds`` alone, enabling the
        bounds-only fast paths in :meth:`is_subset` and :meth:`is_disjoint`.
        """
        return (
            not self._has_literals()
            and not self._admit_arbitrary
            and self._prereleases_configured is not False
        )

    def _check_policy_compat(self, other: VersionRange) -> None:
        """Refuse combining ranges with different pre-release policies."""
        if not isinstance(other, VersionRange):
            raise TypeError(f"expected VersionRange, got {type(other).__name__}")
        if self._prereleases_configured != other._prereleases_configured:
            raise ValueError(
                "Cannot combine VersionRange operands with different "
                f"pre-release policies: {self._prereleases_configured!r} "
                f"and {other._prereleases_configured!r}"
            )

    def _merged_region(self, other: VersionRange) -> tuple[Interval, ...]:
        """Union of ``self`` and ``other``'s opt-in regions.

        Used by :meth:`union` and :meth:`intersection`; :meth:`_build` clips the
        merge to the result bounds. A configured operand carries an empty region,
        so it contributes nothing to the merge.
        """
        # Reuse an operand's canonical tuple when only one side has a region;
        # an empty side contributes nothing to the union.
        if not other._pre_region:
            return self._pre_region
        if not self._pre_region:
            return other._pre_region

        # Both sides carry a region; merge them. _build re-canonicalizes and
        # clips, so the plain union is fine here.
        return tuple(_union_ranges(self._pre_region, other._pre_region))

    def _with_policy(
        self, *, pre_region: tuple[Interval, ...], configured: bool | None
    ) -> VersionRange:
        """A structural copy of this range carrying the given pre-release policy."""
        return self._build(
            self._bounds,
            admit=self._admit,
            reject=self._reject,
            admit_arbitrary=self._admit_arbitrary,
            pre_region=pre_region,
            prereleases_configured=configured,
        )

    @classmethod
    def empty(cls, *, prereleases: bool | None = None) -> VersionRange:
        """Return the empty range. No version satisfies it.

        >>> VersionRange.empty().is_empty
        True
        >>> "1.0" in VersionRange.empty()
        False
        """
        return cls._build((), prereleases_configured=prereleases)

    @classmethod
    def full(
        cls, *, admit_arbitrary: bool = True, prereleases: bool | None = None
    ) -> VersionRange:
        """Return the full range. Every PEP 440 version satisfies it.

        ``admit_arbitrary=False`` restricts the range to PEP 440 versions only
        (matching the same versions as ``SpecifierSet(">=0.dev0").to_range()``);
        its complement is :meth:`empty`. The flag propagates through set algebra
        and is part of equality. Default ``True`` so that ``r & full()``
        preserves ``r``'s own flag structurally.

        >>> "1.0" in VersionRange.full()
        True
        >>> "wat" in VersionRange.full()
        True
        >>> "wat" in VersionRange.full(admit_arbitrary=False)
        False
        """
        return cls._build(
            FULL_RANGE,
            admit_arbitrary=admit_arbitrary,
            prereleases_configured=prereleases,
        )

    @classmethod
    def singleton(
        cls, version: Version | str, *, prereleases: bool | None = None
    ) -> VersionRange:
        """Return the strict singleton range ``{version}``.

        Built as the closed interval ``[version, version]`` with strict
        equality. ``Specifier("==V")`` matches ``V+local`` too, so the strict
        singleton is narrower:

        >>> "1.0+local" in VersionRange.singleton("1.0")
        False
        >>> "1.0+local" in SpecifierSet("==1.0").to_range()
        True

        :raises packaging.version.InvalidVersion: if version is a string that
            does not parse as a PEP 440 version.
        """
        if not isinstance(version, Version):
            version = Version(version)

        lower = LowerBound(version, True)
        upper = UpperBound(version, True)

        # Collapse the floor: nothing sorts below ``MIN_VERSION``, so the
        # ``0.dev0`` singleton is ``(-inf, 0.dev0]`` in canonical form.
        return cls._build(
            _canonical_floor(((lower, upper),)),
            prereleases_configured=prereleases,
        )

    def intersection(self, other: VersionRange) -> VersionRange:
        """Range containing exactly the versions in both self and other.

        Both operands must share the same configured pre-release policy;
        otherwise :exc:`ValueError` is raised.

        >>> a = SpecifierSet(">=1.0").to_range()
        >>> b = SpecifierSet("<2.0").to_range()
        >>> a.intersection(b) == SpecifierSet(">=1.0,<2.0").to_range()
        True
        """
        self._check_policy_compat(other)

        configured = self._prereleases_configured
        new_bounds = tuple(intersect_ranges(self._bounds, other._bounds))
        new_region = self._merged_region(other)

        # An empty intersection (e.g. ``full() & ~full()``) is the empty range,
        # so it drops the arbitrary flag, agreeing with difference when the
        # subtrahend consumes the bounds.
        combined_arb = (
            self._admit_arbitrary and other._admit_arbitrary and bool(new_bounds)
        )

        if not self._has_literals() and not other._has_literals():
            return self._build(
                new_bounds,
                admit_arbitrary=combined_arb,
                pre_region=new_region,
                prereleases_configured=configured,
            )

        return self._combine_literals(
            other,
            new_bounds,
            op=_SetOp.INTERSECTION,
            admit_arbitrary=combined_arb,
            pre_region=new_region,
            prereleases_configured=configured,
        )

    def union(self, other: VersionRange) -> VersionRange:
        """Range containing every version in self or other.

        Both operands must share the same configured pre-release policy;
        otherwise :exc:`ValueError` is raised.

        >>> a = VersionRange.singleton("1.0")
        >>> b = VersionRange.singleton("2.0")
        >>> "1.0" in a.union(b) and "2.0" in a.union(b)
        True
        >>> "1.5" in a.union(b)
        False
        """
        self._check_policy_compat(other)

        configured = self._prereleases_configured
        new_bounds = tuple(_union_ranges(self._bounds, other._bounds))
        new_region = self._merged_region(other)

        # An empty-bounds operand (e.g. ``~full()``) carries an inert arbitrary
        # flag only to keep complement an involution; it admits nothing, so it
        # must not revive arbitrary admission as the union re-widens the bounds.
        if new_bounds:
            combined_arb = (self._admit_arbitrary and bool(self._bounds)) or (
                other._admit_arbitrary and bool(other._bounds)
            )
        else:
            # Nothing widened, so keeping the flags keeps ``r | r == r``.
            combined_arb = self._admit_arbitrary or other._admit_arbitrary

        if not self._has_literals() and not other._has_literals():
            return self._build(
                new_bounds,
                admit_arbitrary=combined_arb,
                pre_region=new_region,
                prereleases_configured=configured,
            )

        return self._combine_literals(
            other,
            new_bounds,
            op=_SetOp.UNION,
            admit_arbitrary=combined_arb,
            pre_region=new_region,
            prereleases_configured=configured,
        )

    def complement(self) -> VersionRange:
        """Range containing every version not in self.

        Preserves the configured pre-release policy. On the version set, double
        negation holds for a range with no ``===`` literals (the arbitrary-string
        flag round-trips, so ``~~full() == full()``); for ``===`` ranges
        complement is one-way. The opt-in region is not restored (see below).

        The opt-in region is dropped: a complement is an exclusion, and an
        exclusion expresses no pre-release preference. This is what lets
        ``a & ~b`` shed ``b``'s opt-in, so an excluded ``b`` never force-admits a
        pre-release into the result. Complement stays involutive on the version
        set, but not on the opt-in region: ``~~r`` covers the same versions as
        ``r`` yet force-admits none of its pre-releases.

        >>> r = SpecifierSet(">=1.0").to_range()
        >>> "0.5" in r.complement()
        True
        >>> "1.5" in r.complement()
        False
        >>> r.complement().complement() == r
        True
        """
        # Complement swaps literal admission: what the range rejects, its
        # complement admits.
        return self._build(
            tuple(_complement_ranges(self._bounds)),
            admit=self._reject,
            reject=self._admit,
            admit_arbitrary=self._admit_arbitrary,
            pre_region=(),
            prereleases_configured=self._prereleases_configured,
        )

    def difference(self, other: VersionRange) -> VersionRange:
        """Range containing the versions in self but not in other.

        Matches ``self & ~other`` on the version set and the opt-in region;
        ``other`` acts as a bounds-only exclusion that grants no opt-in. The
        arbitrary-string flag survives only when ``other`` removed no versions:
        a difference that shrinks the bounds forgets it, as ``self & ~other``
        would, so no later widening union can revive it. They still part on
        ``===`` literals, whose complement is one-way: a ``===`` literal stays
        when ``self`` admits it and ``other`` does not. Both operands must
        share the same configured pre-release policy (as :meth:`intersection`
        and :meth:`union` require); otherwise :exc:`ValueError` is raised.
        ``a - empty()`` returns a range equal to ``a``.

        >>> a = SpecifierSet(">=1.0").to_range()
        >>> b = SpecifierSet(">=2.0").to_range()
        >>> "1.5" in a.difference(b)
        True
        >>> "2.0" in a.difference(b)
        False
        >>> a.difference(VersionRange.empty()) == a
        True
        """
        self._check_policy_compat(other)

        # Subtracting a nothing-admitting set is a no-op; return self unchanged.
        if not other._bounds and not other._admit:
            return self

        # Bound complement is two-way, so subtracting other's versions is an
        # intersection with its gaps.
        new_bounds = tuple(
            intersect_ranges(self._bounds, _complement_ranges(other._bounds))
        )

        # Match ``self & ~other`` on the opt-in region: a complement carries no
        # opt-in, so only ``self``'s region survives. ``other`` acts as a
        # bounds-only exclusion. A configured ``self`` keeps no region.
        new_region: tuple[Interval, ...] = ()
        if self._prereleases_configured is None:
            new_region = self._pre_region

        # Keep self's arbitrary admission only when subtracting removed no
        # versions. A difference that shrinks the bounds forgets the flag, as
        # ``self & ~other`` would, so no later widening union can revive an
        # admission neither operand had.
        combined_arb = self._admit_arbitrary and new_bounds == self._bounds

        if not self._has_literals() and not other._has_literals():
            return self._build(
                new_bounds,
                admit_arbitrary=combined_arb,
                pre_region=new_region,
                prereleases_configured=self._prereleases_configured,
            )

        return self._combine_literals(
            other,
            new_bounds,
            op=_SetOp.DIFFERENCE,
            admit_arbitrary=combined_arb,
            pre_region=new_region,
            prereleases_configured=self._prereleases_configured,
        )

    def _combine_literals(
        self,
        other: VersionRange,
        new_bounds: tuple[Interval, ...],
        *,
        op: _SetOp,
        admit_arbitrary: bool,
        pre_region: tuple[Interval, ...],
        prereleases_configured: bool | None,
    ) -> VersionRange:
        """Resolve admit/reject for ``self`` ``op`` ``other`` over their literals."""
        admits: set[str] = set()
        rejects: set[str] = set()

        # Each literal is decided independently of the others.
        for literal in self._admit | self._reject | other._admit | other._reject:
            self_in = self._matches_literal(literal)
            other_in = other._matches_literal(literal)

            if op is _SetOp.INTERSECTION:
                want = self_in and other_in
            elif op is _SetOp.UNION:
                want = self_in or other_in
            else:
                want = self_in and not other_in

            if want:
                admits.add(literal)
            else:
                rejects.add(literal)

        return self._build(
            new_bounds,
            admit=frozenset(admits),
            reject=frozenset(rejects),
            admit_arbitrary=admit_arbitrary,
            pre_region=pre_region,
            prereleases_configured=prereleases_configured,
        )

    def _matches_literal(self, literal: str) -> bool:
        """Whether literal (case-folded) matches this range's predicate."""
        if literal in self._reject:
            return False
        if literal in self._admit:
            return True

        parsed = coerce_version(literal)
        if parsed is None:
            return self._arbitrary_active()
        return matches_bounds_only(self._bounds, parsed)

    def __and__(self, other: object) -> VersionRange:
        """Operator alias for :meth:`intersection`."""
        if not isinstance(other, VersionRange):
            return NotImplemented
        return self.intersection(other)

    def __or__(self, other: object) -> VersionRange:
        """Operator alias for :meth:`union`."""
        if not isinstance(other, VersionRange):
            return NotImplemented
        return self.union(other)

    def __invert__(self) -> VersionRange:
        """Operator alias for :meth:`complement`."""
        return self.complement()

    def __sub__(self, other: object) -> VersionRange:
        """Operator alias for :meth:`difference`."""
        if not isinstance(other, VersionRange):
            return NotImplemented
        return self.difference(other)

    def is_subset(self, other: VersionRange) -> bool:
        """Return whether every member of self is also a member of other.

        On versions and ``===`` literals this is
        ``self.difference(other).is_empty``: subtracting other leaves nothing
        behind. A live arbitrary admission (the flag at full bounds) is only a
        subset of another live one.

        Both operands must share the same configured pre-release policy;
        otherwise :exc:`ValueError` is raised.

        >>> inner = SpecifierSet(">=1.5,<1.8").to_range()
        >>> outer = SpecifierSet(">=1.0,<2.0").to_range()
        >>> inner.is_subset(outer)
        True
        >>> outer.is_subset(inner)
        False
        >>> VersionRange.empty().is_subset(outer)
        True
        """
        self._check_policy_compat(other)

        # A live arbitrary admission has non-version strings as members, which
        # no bounds cover; only another live admission contains them.
        if self._arbitrary_active() and not other._arbitrary_active():
            return False

        # Plain ranges: subset reduces to bounds containment, no algebra needed.
        if self._is_plain() and other._is_plain():
            return not intersect_ranges(self._bounds, _complement_ranges(other._bounds))

        # difference (unlike intersection with the one-way complement) resolves
        # ``===`` literals against both operands, so it stays correct for them.
        return self.difference(other).is_empty

    def is_superset(self, other: VersionRange) -> bool:
        """Return whether every member of other is also a member of self.

        The mirror of :meth:`is_subset`: ``a.is_superset(b)`` is
        ``b.is_subset(a)``.

        Both operands must share the same configured pre-release policy;
        otherwise :exc:`ValueError` is raised.

        >>> outer = SpecifierSet(">=1.0,<2.0").to_range()
        >>> outer.is_superset(SpecifierSet(">=1.5,<1.8").to_range())
        True
        """
        # Type-guards a non-VersionRange other before delegating to is_subset.
        self._check_policy_compat(other)
        return other.is_subset(self)

    def is_disjoint(self, other: VersionRange) -> bool:
        """Return whether self and other share no member.

        Equivalent to ``(self & other).is_empty``.

        Both operands must share the same configured pre-release policy;
        otherwise :exc:`ValueError` is raised.

        >>> a = SpecifierSet(">=1.0,<2.0").to_range()
        >>> a.is_disjoint(SpecifierSet(">=2.0,<3.0").to_range())
        True
        >>> a.is_disjoint(SpecifierSet(">=1.5,<2.5").to_range())
        False
        """
        self._check_policy_compat(other)

        # Plain ranges: disjointness is an empty bounds intersection.
        if self._is_plain() and other._is_plain():
            return not intersect_ranges(self._bounds, other._bounds)
        return self.intersection(other).is_empty

    def _same_releases(self, other: VersionRange) -> bool:
        """Whether self and other admit the same non-pre-release versions.

        Used by :meth:`to_specifier_set` under a ``prereleases=False`` policy,
        where pre-releases are unobservable: the symmetric difference is empty
        exactly when the two ranges accept the same releases. Both operands
        carry that policy, so the difference below reads emptiness through it.
        """
        return self.difference(other).is_empty and other.difference(self).is_empty

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
        key: Callable[[Any], Version | str] | None = None,
    ) -> Iterator[Any]:
        """Yield items from iterable whose version falls inside the range.

        With prereleases ``None`` the PEP 440 default applies: pre-releases are
        buffered and only emitted if no final release in iterable is in range,
        except that a pre-release inside the autodetected opt-in region, or named
        outright by a ``===`` literal, is force-admitted in place (as
        ``prereleases=True`` would yield it). A flushed buffer comes after
        every in-place yield, so the output is not version-sorted.

        The signature mirrors
        :meth:`~packaging.specifiers.SpecifierSet.filter`.

        >>> r = SpecifierSet(">=1.0,<2.0").to_range()
        >>> list(r.filter(["0.9", "1.5", "2.0"]))
        ['1.5']
        """
        region: tuple[Interval, ...] = ()
        if prereleases is None:
            # The region applies only under the autodetect default; a configured
            # policy governs instead (and then ``_pre_region`` is already empty).
            prereleases = self._prereleases_configured
            region = self._pre_region

        arbitrary_active = self._arbitrary_active()
        if not self._admit and not self._reject and not arbitrary_active:
            # A region spanning the whole bounds force-admits every in-bounds
            # pre-release, i.e. ``prereleases=True``; take the cheaper no-buffer
            # path. (Confined to this branch: the admission path orders arbitrary
            # strings differently under True than under the region.)
            if region and region == self._bounds:
                return filter_by_ranges(self._bounds, iterable, key, True)
            return filter_by_ranges(self._bounds, iterable, key, prereleases, region)
        return self._filter_with_admission(
            iterable, key, prereleases, arbitrary_active, region
        )

    def _filter_with_admission(
        self,
        iterable: Iterable[Any],
        key: Callable[[Any], Version | str] | None,
        prereleases: bool | None,
        arbitrary_active: bool,
        region: tuple[Interval, ...],
    ) -> Iterator[Any]:
        """Filter for ranges with admit/reject literals or live arbitrary
        admission (including the universal ``SpecifierSet("")`` range)."""
        admit_set = self._admit
        reject_set = self._reject

        def admit(item: Any) -> tuple[bool, Version | None, bool]:  # noqa: ANN401
            raw: Version | str = item if key is None else key(item)
            raw_lower = str(raw).lower()

            if reject_set and raw_lower in reject_set:
                return False, None, False
            if admit_set and raw_lower in admit_set:
                # An explicit ``===`` literal names this version outright.
                return True, coerce_version(raw), True

            parsed = coerce_version(raw)
            if parsed is None:
                return arbitrary_active, None, False
            if not matches_bounds_only(self._bounds, parsed):
                return False, None, False
            return True, parsed, False

        if prereleases is True:
            for item in iterable:
                ok, _, _ = admit(item)
                if ok:
                    yield item
            return

        if prereleases is False:
            for item in iterable:
                ok, parsed, _ = admit(item)
                if not ok:
                    continue
                if parsed is not None and parsed.is_prerelease:
                    continue
                yield item
            return

        # PEP 440 default: emit finals eagerly and buffer the other pre-releases,
        # releasing the buffer only if no final ever matches.
        all_nonfinal: list[Any] = []
        arbitrary_strings: list[Any] = []
        found_final = False

        for item in iterable:
            ok, parsed, by_literal = admit(item)
            if not ok:
                continue

            if parsed is None:
                if found_final:
                    yield item
                else:
                    arbitrary_strings.append(item)
                    all_nonfinal.append(item)
                continue

            if not parsed.is_prerelease:
                if not found_final:
                    yield from arbitrary_strings
                    arbitrary_strings.clear()
                    found_final = True
                yield item
                continue

            # A pre-release is force-admitted when it is named outright by a
            # ``===`` literal or falls in the opt-in region, as ``prereleases=True``
            # would yield it; otherwise the PEP 440 default buffers it.
            if by_literal or (region and matches_bounds_only(region, parsed)):
                yield item
                continue

            if not found_final:
                all_nonfinal.append(item)

        if not found_final:
            yield from all_nonfinal

    @classmethod
    def _from_specifier_set(cls, specifier_set: SpecifierSet) -> VersionRange:
        """Build the range accepted by ``specifier_set``.

        Friend constructor for :meth:`~packaging.specifiers.SpecifierSet.to_range`.
        The intersection of every specifier in the set: an empty set yields the
        full range, an unsatisfiable set yields the empty range, and ``===``
        specifiers contribute literal-string admission.
        """
        if not specifier_set:
            result = cls.full()
        elif not specifier_set._has_arbitrary:
            result = cls._build(
                bounds=_canonical_floor(tuple(specifier_set._get_ranges()))
            )
        else:
            result = cls.full()
            for spec in specifier_set:
                if spec.operator == "===":
                    operand = cls._build(
                        bounds=(), admit=frozenset({spec.version.lower()})
                    )
                else:
                    operand = cls._build(
                        bounds=_canonical_floor(tuple(spec._to_ranges()))
                    )
                result = result.intersection(operand)

        # Each pre-release-naming specifier opts its own versions in; their union,
        # clipped to the set's bounds by _build, is the region. Clipping refolds
        # under intersection, so a set built directly equals one built by
        # intersecting its specifiers one at a time.
        region: list[Interval] = []
        if specifier_set._prereleases is None:  # a configured policy has no region
            for spec in specifier_set:
                # ``===`` literals are not a range; filter force-admits them.
                if spec.operator != "===" and spec.prereleases:
                    spec_bounds = _canonical_floor(tuple(spec._to_ranges()))
                    region = _union_ranges(region, spec_bounds)

        return result._with_policy(
            pre_region=tuple(region),
            configured=specifier_set._prereleases,
        )

    def to_specifier_set(self) -> SpecifierSet | None:
        """Return a :class:`~packaging.specifiers.SpecifierSet` matching the same
        versions as self, or ``None`` if no single set expresses it.

        PEP 440 has no syntax for the strict singleton ``{V}`` (an exclusive
        plain-version bound), a disjoint union of two or more intervals, or a
        partial pre-release opt-in region, so ranges built by set algebra often
        return ``None``. A gap that takes more than ``_MAX_EXCLUSION_RUN``
        contiguous ``!=`` exclusions to spell returns ``None`` too,
        rather than a pathologically long chain; reaching that cap takes either
        set algebra or a specifier set that already spells the gap out with
        over a hundred contiguous ``!=N.*`` exclusions. An empty range maps to
        ``SpecifierSet("<0")``, unless it still carries the arbitrary-string
        flag (which no set reproduces), and a full range that admits arbitrary
        strings maps to ``SpecifierSet("")``.

        A range built from a :class:`~packaging.specifiers.SpecifierSet`
        re-encodes, short of that exclusion cap. The result is the simplest
        candidate whose own
        :meth:`~packaging.specifiers.SpecifierSet.to_range` reproduces self
        exactly (bounds, ``===`` literals, and the opt-in region are all part of
        equality), so it filters the same versions. Two cases relax that
        exactness without changing what is filtered: an empty range recovers as
        the canonical empty range (same versions, none, but not self's bounds),
        and under a ``prereleases=False`` policy the result need only match self's
        releases, so ``(-inf, 3.14)`` recovers as the tighter ``<3.14`` rather
        than ``!=3.14,<=3.14``.

        Each call encodes a handful of candidate spellings and keeps the
        simplest one that verifies, where verifying means parsing the candidate
        and round-tripping it through
        :meth:`~packaging.specifiers.SpecifierSet.to_range`. The work grows
        with the number of intervals and exclusions in the range, and the
        result is not cached, so convert once and reuse the returned set rather
        than converting per candidate version in a hot loop.

        >>> str(SpecifierSet(">=1.0,<2.0").to_range().to_specifier_set())
        '<2.0,>=1.0'
        >>> str(SpecifierSet("==1.0").to_range().to_specifier_set())
        '==1.0'
        >>> VersionRange.singleton("1.5").to_specifier_set() is None
        True
        """
        from .specifiers import InvalidSpecifier, SpecifierSet  # noqa: PLC0415

        configured = self._prereleases_configured

        if self._reject:
            return None
        if self._admit_arbitrary and self._bounds != FULL_RANGE:
            return None
        if self.is_empty:
            # Every member-free spelling accepts the same versions (none), so the
            # canonical ``<0`` stands in for all of them; a configured policy
            # rides along it.
            return SpecifierSet("<0", prereleases=configured)

        if not self._bounds:
            # Pure ``===`` literals; only a single literal has a single-set form.
            if len(self._admit) != 1:
                return None
            (literal,) = self._admit
            bases = [f"==={literal}"]
        elif self._admit:
            # Bounds plus literals cannot be one set.
            return None
        elif self._bounds == FULL_RANGE:
            bases = ["" if self._admit_arbitrary else ">=0.dev0"]
        else:
            # Under ``prereleases=False`` an exclusive final upper admits the same
            # releases as ``<V`` (the ``[V.dev0, V)`` band is excluded), so offer
            # the tightened bounds as well; a tightening that is not
            # release-equivalent is dropped by the acceptance check below.
            layouts = [self._bounds]
            if configured is False:
                tightened = _tighten_no_prereleases(self._bounds)
                if tightened != self._bounds:
                    layouts.append(tightened)

            # Encode each layout in both spelling modes: prerelease-free, then
            # keeping the synthetic ``.dev0`` markers. Which spelling reproduces
            # self is settled by the round trip below, not up front. The gap
            # exclusions do not depend on the mode, so they encode once.
            bases = []
            for layout in layouts:
                exclusions = _encode_gaps(layout)
                if exclusions is None:
                    continue

                for keep_dev0 in (False, True):
                    outer = _encode_interval(layout[0][0], layout[-1][1], keep_dev0)
                    if outer is None:
                        continue
                    base = ",".join(outer + exclusions)
                    if base not in bases:
                        bases.append(base)

        # A trailing no-op ``>=0.dev0`` floor restores a ``True`` opt-in that
        # rode on a floor the clean encoding dropped (e.g. ``>=0.dev0,!=1.0``).
        # It always recovers the whole bounds as the opt-in region, so it can
        # only round-trip when self opts everything in.
        add_floor = configured is None and self._pre_region == self._bounds

        # Keep the simplest candidate that recovers self. ``==`` compares bounds,
        # literals, and the opt-in region, so a candidate that would filter
        # differently, or an op-built range with no single-set form, is rejected
        # below. Under ``prereleases=False`` a candidate need only match self's
        # releases (policies never mix, so the excluded pre-releases are
        # unobservable), which admits the tightened spellings above.
        best: SpecifierSet | None = None
        best_key = (0, 0)

        for base in bases:
            candidates = [base]
            if add_floor:
                candidates.append(f"{base},>=0.dev0" if base else ">=0.dev0")

            for spec_str in candidates:
                # A ``===`` literal can hold a comma (its arbitrary version
                # excludes only whitespace, ``;`` and ``)``), which the set
                # string splits on. Such a literal has no single specifier-set
                # spelling, so drop the unparsable candidate and let the range
                # fall through to ``None`` rather than raise.
                try:
                    recovered = SpecifierSet(spec_str, prereleases=configured)
                except InvalidSpecifier:
                    continue
                # Fewest fragments, then shortest string. Rank before the round
                # trip so a candidate that cannot beat the best skips the check
                # (its ``==`` and, under ``False``, two ``difference`` calls).
                key = (len(recovered), len(str(recovered)))
                if best is not None and key >= best_key:
                    continue

                # Accept an exact round trip, or (under ``False``) one that only
                # matches the releases the policy leaves observable.
                candidate = recovered.to_range()
                matches = candidate == self or (
                    configured is False and self._same_releases(candidate)
                )
                if matches:
                    best, best_key = recovered, key

        return best

    @property
    def is_empty(self) -> bool:
        """``True`` if no version or string satisfies this range.

        Agrees with :meth:`~packaging.specifiers.SpecifierSet.is_unsatisfiable`,
        including the pre-release policy: a range whose only members are
        pre-releases is empty when that policy excludes them.

        >>> SpecifierSet(">=2,<1").to_range().is_empty
        True
        >>> SpecifierSet(">=1,<2").to_range().is_empty
        False
        >>> SpecifierSet("==1.0a1", prereleases=False).to_range().is_empty
        True
        """
        # An arbitrary-string admission or a surviving ``===`` literal is a
        # member; a literal that is a pre-release is dropped when the policy is.
        if self._arbitrary_active():
            return False

        excludes_prereleases = self._prereleases_configured is False
        for literal in self._admit:
            if excludes_prereleases:
                parsed = coerce_version(literal)
                if parsed is not None and parsed.is_prerelease:
                    continue
            return False

        if not self._bounds:
            return True

        return excludes_prereleases and ranges_are_prerelease_only(self._bounds)

    def contains(
        self,
        item: Version | str,
        prereleases: bool | None = None,
        installed: bool | None = None,
    ) -> bool:
        """Return whether item is contained in this range.

        :param item: a version string or :class:`~packaging.version.Version`.
        :param prereleases: whether to match pre-releases. ``None`` (default)
            uses the range's own policy.
        :param installed: when ``True``, accept a pre-release item even if the
            range would not otherwise allow it.

        Unlike :meth:`filter`, this does not consult the autodetected pre-release
        opt-in region; it reads only the configured policy. This mirrors
        :meth:`~packaging.specifiers.SpecifierSet.contains` versus
        :meth:`~packaging.specifiers.SpecifierSet.filter`.

        Unparsable strings do not match, except where the full
        ``SpecifierSet`` would also match: the full range admits any string,
        and a ``===`` range admits items equal to the literal
        case-insensitively.

        >>> r = SpecifierSet(">=1.0,<2.0").to_range()
        >>> r.contains("1.5")
        True
        >>> r.contains("2.0")
        False

        :raises TypeError: if item is not a str or Version.
        """
        if not isinstance(item, (str, Version)):
            raise TypeError(
                f"VersionRange.contains() expected str or Version, "
                f"got {type(item).__name__}"
            )

        parsed: Version | None = item if isinstance(item, Version) else None
        if installed and parsed is None:
            parsed = coerce_version(item)
        if installed and parsed is not None and parsed.is_prerelease:
            prereleases = True

        effective_pre = (
            self._prereleases_configured if prereleases is None else prereleases
        )

        if self._admit or self._reject:
            item_str = str(item).lower()
            if item_str in self._reject:
                return False
            if item_str in self._admit:
                if effective_pre is False:
                    literal_parsed = coerce_version(item_str)
                    if literal_parsed is not None and literal_parsed.is_prerelease:
                        return False
                return True

        if not isinstance(item, Version):
            if parsed is None:
                parsed = coerce_version(item)
            if parsed is None:
                return self._arbitrary_active()
            item = parsed

        if effective_pre is False and item.is_prerelease:
            return False
        return matches_bounds_only(self._bounds, item)

    def __contains__(self, item: Version | str) -> bool:
        """Return whether item is contained in this range.

        Forwards to :meth:`contains` with default arguments.

        >>> "1.5" in SpecifierSet(">=1.0,<2.0").to_range()
        True
        """
        return self.contains(item)

    def __eq__(self, other: object) -> bool:
        """Structural equality.

        Compares the bounds, the ``===`` admit/reject literals, the
        arbitrary-string flag, the configured pre-release policy, and the
        opt-in region, not just the version set. Keying on the region makes
        equality a congruence (equal ranges stay equal under further operations),
        so equal implies same :meth:`contains` and :meth:`filter`, but not the
        converse: an empty range keeps the flag it was built with, so two empty
        ranges need not be equal.

        Different specifiers for the same range fold to one canonical form:

        >>> SpecifierSet(">1.0a1").to_range() == SpecifierSet(">=1.0a2.dev0").to_range()
        True

        The opt-in region is part of equality, so ``<=1.0`` (no pre-releases) and
        ``<1.0.post0.dev0`` (autodetects a ``.dev`` opt-in) cover the same
        versions yet compare unequal:

        >>> le, lt = SpecifierSet("<=1.0"), SpecifierSet("<1.0.post0.dev0")
        >>> le.to_range() == lt.to_range()
        False

        >>> r = SpecifierSet(">=1.0,<2.0").to_range()
        >>> r == SpecifierSet(">=1.0,<2.0").to_range()
        True
        """
        if not isinstance(other, VersionRange):
            return NotImplemented
        return (
            self._bounds == other._bounds
            and self._admit == other._admit
            and self._reject == other._reject
            and self._admit_arbitrary == other._admit_arbitrary
            and self._prereleases_configured == other._prereleases_configured
            and self._pre_region == other._pre_region
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._bounds,
                self._admit,
                self._reject,
                self._admit_arbitrary,
                self._prereleases_configured,
                self._pre_region,
            )
        )

    def __repr__(self) -> str:
        """Human-readable representation for debugging.

        >>> SpecifierSet(">=1.0,<2.0").to_range()
        <VersionRange '[1.0, 2.0.dev0)'>
        >>> SpecifierSet("").to_range()
        <VersionRange '(-inf, +inf)' arbitrary>
        >>> SpecifierSet(">=2.0,<1.0").to_range()
        <VersionRange '(empty)'>
        """
        # Body: the bounds and any ``===``-admitted literals.
        parts: list[str] = []
        if self._bounds:
            parts.append(_format_intervals(self._bounds))
        if self._admit:
            parts.append("{" + ", ".join(sorted(self._admit)) + "}")
        body = " | ".join(parts) if parts else "(empty)"

        # Rejected literals subtract from the body.
        if self._reject:
            body = f"{body} \\ {{{', '.join(sorted(self._reject))}}}"

        # Tail: the policy flags carried alongside the version set.
        tail = ""
        if self._admit_arbitrary:
            tail += " arbitrary"
        if self._prereleases_configured is not None:
            tail += f" pre={self._prereleases_configured}"
        if self._pre_region:
            tail += f" pre-region={_format_intervals(self._pre_region)!r}"

        return f"<{self.__class__.__name__} {body!r}{tail}>"
<|endoftext|>"""
ELF file parser.

This provides a class ``ELFFile`` that parses an ELF executable in a similar
interface to ``ZipFile``. Only the read interface is implemented.

ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
"""

from __future__ import annotations

import enum
import os
import struct
from typing import IO


class ELFInvalid(ValueError):
    pass


class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2


class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2


class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArch64 = 183


class ELFFile:
    """
    Representation of an ELF executable.
    """

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f

        try:
            ident = self._read("16B")
        except struct.error as e:
            raise ELFInvalid("unable to parse identification") from e
        magic = bytes(ident[:4])
        if magic != b"\x7fELF":
            raise ELFInvalid(f"invalid magic: {magic!r}")

        self.capacity = ident[4]  # Format for program header (bitness).
        self.encoding = ident[5]  # Data structure encoding (endianness).

        try:
            # e_fmt: Format for the ELF header.
            # p_fmt: Format for a program header.
            # p_idx: Indexes to find p_type, p_offset, and p_filesz.
            e_fmt, self._p_fmt, self._p_idx = {
                (1, 1): ("<HHIIIIIHHH", "<IIIIIIII", (0, 1, 4)),  # 32-bit LSB.
                (1, 2): (">HHIIIIIHHH", ">IIIIIIII", (0, 1, 4)),  # 32-bit MSB.
                (2, 1): ("<HHIQQQIHHH", "<IIQQQQQQ", (0, 2, 5)),  # 64-bit LSB.
                (2, 2): (">HHIQQQIHHH", ">IIQQQQQQ", (0, 2, 5)),  # 64-bit MSB.
            }[(self.capacity, self.encoding)]
        except KeyError as e:
            raise ELFInvalid(
                f"unrecognized capacity ({self.capacity}) or encoding ({self.encoding})"
            ) from e

        try:
            (
                _,
                self.machine,  # Architecture type.
                _,
                _,
                self._e_phoff,  # Offset of program header.
                _,
                self.flags,  # Processor-specific flags.
                _,
                self._e_phentsize,  # Size of a program header entry.
                self._e_phnum,  # Number of program headers.
            ) = self._read(e_fmt)
        except struct.error as e:
            raise ELFInvalid("unable to parse machine and section information") from e

    def _read(self, fmt: str) -> tuple[int, ...]:
        return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))

    @property
    def interpreter(self) -> str | None:
        """
        The path recorded in the ``PT_INTERP`` section header.
        """
        for index in range(self._e_phnum):
            self._f.seek(self._e_phoff + self._e_phentsize * index)
            try:
                data = self._read(self._p_fmt)
            except struct.error:
                continue
            if data[self._p_idx[0]] != 3:  # Not PT_INTERP.
                continue
            self._f.seek(data[self._p_idx[1]])
            return os.fsdecode(self._f.read(data[self._p_idx[2]])).strip("\0")
        return None
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import logging
import operator
import platform
import re
import struct
import subprocess
import sys
import sysconfig
from collections.abc import Iterable, Iterator, Sequence
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
    TYPE_CHECKING,
    TypeVar,
    cast,
)

from . import _manylinux, _musllinux

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Set as AbstractSet


__all__ = [
    "INTERPRETER_SHORT_NAMES",
    "AppleVersion",
    "InvalidTag",
    "PythonVersion",
    "Tag",
    "TooManyTagsError",
    "UnsortedTagsError",
    "android_platforms",
    "compatible_tags",
    "cpython_tags",
    "create_compatible_tags_selector",
    "generic_tags",
    "interpreter_name",
    "interpreter_version",
    "ios_platforms",
    "mac_platforms",
    "parse_tag",
    "platform_tags",
    "pure_python_tags",
    "sys_tags",
]


def __dir__() -> list[str]:
    return __all__


logger = logging.getLogger(__name__)

PythonVersion = Sequence[int]
"""
A sequence of integers describing a Python version, e.g. ``(3, 13)``.

.. versionadded:: 20.0
"""

AppleVersion = tuple[int, int]
"""
A ``(major, minor)`` integer pair describing an Apple OS version.

.. versionadded:: 24.2
"""
_T = TypeVar("_T")

INTERPRETER_SHORT_NAMES: dict[str, str] = {
    "python": "py",  # Generic.
    "cpython": "cp",
    "pypy": "pp",
    "ironpython": "ip",
    "jython": "jy",
}


# This function can be unit tested without reloading the module
# (Unlike _32_BIT_INTERPRETER)
def _compute_32_bit_interpreter() -> bool:
    return struct.calcsize("P") == 4


_32_BIT_INTERPRETER = _compute_32_bit_interpreter()


class UnsortedTagsError(ValueError):
    """
    Raised when a tag component is not in sorted order per PEP 425.

    .. versionadded:: 26.1
    """


class InvalidTag(ValueError):
    """
    Raised when an interpreter component is not an identifier, a tag component
    is empty, or a tag does not have exactly three components.

    .. versionadded:: 26.3
    """


class TooManyTagsError(ValueError):
    """
    Raised when a compressed tag set exceeds the configured limit.

    .. versionadded:: 26.3
    """


class Tag:
    """
    A representation of the tag triple for a wheel.

    Instances are considered immutable and thus are hashable. Equality checking
    is also supported.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = ["_abi", "_hash", "_interpreter", "_platform"]

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        """
        :param str interpreter: The interpreter name, e.g. ``"py"``
                                (see :attr:`INTERPRETER_SHORT_NAMES` for mapping
                                well-known interpreter names to their short names).
        :param str abi: The ABI that a wheel supports, e.g. ``"cp37m"``.
        :param str platform: The OS/platform the wheel supports,
                            e.g. ``"win_amd64"``.
        """
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        # The __hash__ of every single element in a Set[Tag] will be evaluated each time
        # that a set calls its `.disjoint()` method, which may be called hundreds of
        # times when scanning a page of links for packages with tags matching that
        # Set[Tag]. Pre-computing the value here produces significant speedups for
        # downstream consumers.
        self._hash = hash((self._interpreter, self._abi, self._platform))

    @property
    def interpreter(self) -> str:
        """
        The interpreter name, e.g. ``"py"`` (see
        :attr:`INTERPRETER_SHORT_NAMES` for mapping well-known interpreter
        names to their short names).
        """
        return self._interpreter

    @property
    def abi(self) -> str:
        """
        The supported ABI.
        """
        return self._abi

    @property
    def platform(self) -> str:
        """
        The OS/platform.
        """
        return self._platform

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented

        return (
            (self._hash == other._hash)  # Short-circuit ASAP for perf reasons.
            and (self._platform == other._platform)
            and (self._abi == other._abi)
            and (self._interpreter == other._interpreter)
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self._interpreter}-{self._abi}-{self._platform}"

    def __repr__(self) -> str:
        return f"<{self} @ {id(self)}>"

    def __getstate__(self) -> tuple[str, str, str]:
        # Return state as a 3-item tuple: (interpreter, abi, platform).
        # Cache member _hash is excluded and will be recomputed.
        return (self._interpreter, self._abi, self._platform)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple):
            if len(state) == 3 and all(isinstance(s, str) for s in state):
                # New format (26.2+): (interpreter, abi, platform)
                self._interpreter, self._abi, self._platform = state
                self._hash = hash((self._interpreter, self._abi, self._platform))
                return
            if len(state) == 2 and isinstance(state[1], dict):
                # Old format (packaging <= 26.1, __slots__): (None, {slot: value}).
                _, slots = state
                try:
                    interpreter = slots["_interpreter"]
                    abi = slots["_abi"]
                    platform = slots["_platform"]
                except KeyError:
                    raise TypeError(f"Cannot restore Tag from {state!r}") from None
                if not all(
                    isinstance(value, str) for value in (interpreter, abi, platform)
                ):
                    raise TypeError(f"Cannot restore Tag from {state!r}")
                self._interpreter = interpreter.lower()
                self._abi = abi.lower()
                self._platform = platform.lower()
                self._hash = hash((self._interpreter, self._abi, self._platform))
                return
        raise TypeError(f"Cannot restore Tag from {state!r}")


def parse_tag(
    tag: str, *, validate_order: bool = False, limit: int | None = None
) -> frozenset[Tag]:
    """
    Parses the provided tag (e.g. `py3-none-any`) into a frozenset of
    :class:`Tag` instances.

    Returning a set is required due to the possibility that the tag is a
    `compressed tag set`_, e.g. ``"py2.py3-none-any"`` which supports both
    Python 2 and Python 3.

    If **validate_order** is true, compressed tag set components are checked
    to be in sorted order as required by PEP 425.

    If **limit** is not ``None``, the compressed tag set can generate at most
    that many tags.

    :param str tag: The tag to parse, e.g. ``"py3-none-any"``.
    :param bool validate_order: Check whether compressed tag set components
        are in sorted order.
    :param int | None limit: The maximum number of tags to parse.
    :raises UnsortedTagsError: If **validate_order** is true and any compressed tag
        set component is not in sorted order.
    :raises InvalidTag: If the interpreter field is not an identifier; if the
        interpreter, ABI, or platform field (or any member of a compressed tag
        set) is empty; or if the tag does not have exactly three components.
    :raises TooManyTagsError: If **limit** is not ``None`` and the compressed tag
        set would generate more than **limit** tags.
    :raises ValueError: If **limit** is negative.

    .. versionadded:: 26.1
       The *validate_order* parameter.

    .. versionadded:: 26.3
       Raises :class:`InvalidTag` when an interpreter component is not an
       identifier, a tag component is empty, or a tag does not have exactly
       three components.
       Added the *limit* parameter. Raises :class:`TooManyTagsError` if the compressed
       tag set would generate more than *limit* tags.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    component_parts = [component.split(".") for component in tag.split("-")]
    for parts in component_parts:
        if "" in parts:
            component = ".".join(parts)
            raise InvalidTag(f"Tag {tag!r} has an empty component: {component!r}")
        if validate_order and parts != sorted(parts):
            component = ".".join(parts)
            raise UnsortedTagsError(
                f"Tag component {component!r} is not in sorted order per PEP 425"
            )

    tag_count = 1
    for parts in component_parts:
        tag_count *= len(parts)

    if limit is not None and tag_count > limit:
        raise TooManyTagsError(
            f"Compressed tag set would generate {tag_count} tags, exceeding "
            f"limit {limit}"
        )

    try:
        interpreters, abis, platforms = component_parts
    except ValueError as exc:
        raise InvalidTag(f"Tag {tag!r} must have exactly three components") from exc
    for interpreter in interpreters:
        if not interpreter.isidentifier():
            raise InvalidTag(f"Tag {tag!r} has an invalid interpreter: {interpreter!r}")
    return frozenset(
        Tag(interpreter, abi, platform_)
        for interpreter in interpreters
        for abi in abis
        for platform_ in platforms
    )


def _get_config_var(name: str, warn: bool = False) -> int | str | None:
    value: int | str | None = sysconfig.get_config_var(name)
    if value is None and warn:
        logger.debug(
            "Config variable '%s' is unset, Python ABI tag may be incorrect", name
        )
    return value


def _normalize_string(string: str) -> str:
    return string.replace(".", "_").replace("-", "_").replace(" ", "_")


def _is_threaded_cpython(abis: list[str]) -> bool:
    """
    Determine if the ABI corresponds to a threaded (`--disable-gil`) build.

    The threaded builds are indicated by a "t" in the abiflags.
    """
    if len(abis) == 0:
        return False
    # expect e.g., cp313
    m = re.match(r"cp\d+(.*)", abis[0])
    if not m:
        return False
    abiflags = m.group(1)
    return "t" in abiflags


def _abi3_applies(python_version: PythonVersion, threading: bool) -> bool:
    """
    Determine if the Python version supports abi3.

    PEP 384 was first implemented in Python 3.2. The free-threaded
    builds do not support abi3.
    """
    return len(python_version) > 1 and tuple(python_version) >= (3, 2) and not threading


def _abi3t_applies(python_version: PythonVersion, threading: bool) -> bool:
    """
    Determine if the Python version supports abi3t.

    PEP 803 was first implemented in Python 3.15 but, per PEP 803, this
    returns tags going back to Python 3.2 to mirror the abi3
    implementation and leave open the possibility of abi3t wheels
    supporting older Python versions.

    """
    return len(python_version) > 1 and tuple(python_version) >= (3, 2) and threading


def _cpython_abis(py_version: PythonVersion, warn: bool = False) -> list[str]:
    py_version = tuple(py_version)  # To allow for version comparison.
    abis = []
    version = _version_nodot(py_version[:2])
    threading = debug = pymalloc = ucs4 = ""
    with_debug = _get_config_var("Py_DEBUG", warn)
    has_refcount = hasattr(sys, "gettotalrefcount")
    # Windows doesn't set Py_DEBUG, so checking for support of debug-compiled
    # extension modules is the best option.
    # https://github.com/pypa/pip/issues/3383#issuecomment-173267692
    has_ext = "_d.pyd" in EXTENSION_SUFFIXES
    if with_debug or (with_debug is None and (has_refcount or has_ext)):
        debug = "d"
    if py_version >= (3, 13) and _get_config_var("Py_GIL_DISABLED", warn):
        threading = "t"
    if py_version < (3, 8):
        with_pymalloc = _get_config_var("WITH_PYMALLOC", warn)
        if with_pymalloc or with_pymalloc is None:
            pymalloc = "m"
        if py_version < (3, 3):
            unicode_size = _get_config_var("Py_UNICODE_SIZE", warn)
            if unicode_size == 4 or (
                unicode_size is None and sys.maxunicode == 0x10FFFF
            ):
                ucs4 = "u"
    elif debug:
        # Debug builds can also load "normal" extension modules.
        # We can also assume no UCS-4 or pymalloc requirement.
        abis.append(f"cp{version}{threading}")
    abis.insert(0, f"cp{version}{threading}{debug}{pymalloc}{ucs4}")
    return abis


def cpython_tags(
    python_version: PythonVersion | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for the CPython interpreter.

    The specific tags generated are:

    - ``cp<python_version>-<abi>-<platform>``
    - ``cp<python_version>-<stable_abi>-<platform>``
    - ``cp<python_version>-none-<platform>``
    - ``cp<older version>-<stable_abi>-<platform>`` where "older version" is all older
      minor versions down to Python 3.2 (when ``abi3`` was introduced)

    If ``python_version`` only provides a major-only version then only
    user-provided ABIs via ``abis`` and the ``none`` ABI will be used.

    The ``stable_abi`` will be either ``abi3`` or ``abi3t`` if `abi` is a
    GIL-enabled ABI like `"cp315"` or a free-threaded ABI like `"cp315t"`,
    respectively.

    :param Sequence python_version: A one- or two-item sequence representing the
                                 targeted Python version. Defaults to
                                 ``sys.version_info[:2]``.
    :param Iterable abis: Iterable of compatible ABIs. Defaults to the ABIs
                          compatible with the current system.
    :param Iterable platforms: Iterable of compatible platforms. Defaults to the
                               platforms compatible with the current system.
    :param bool warn: Whether warnings should be logged. Defaults to ``False``.

    .. versionadded:: 20.0
    """
    if not python_version:
        python_version = sys.version_info[:2]

    interpreter = f"cp{_version_nodot(python_version[:2])}"

    if abis is None:
        abis = _cpython_abis(python_version, warn) if len(python_version) > 1 else []
    abis = list(abis)
    threading = _is_threaded_cpython(abis)
    # Stable ABIs and 'none' are explicitly handled later.
    explicit_abis = ("abi3", "abi3t", "none") if threading else ("abi3", "none")
    for explicit_abi in explicit_abis:
        try:
            abis.remove(explicit_abi)
        except ValueError:  # noqa: PERF203
            pass

    platforms = list(platforms or platform_tags())
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)

    use_abi3 = _abi3_applies(python_version, threading)
    use_abi3t = _abi3t_applies(python_version, threading)
    if use_abi3:
        yield from (Tag(interpreter, "abi3", platform_) for platform_ in platforms)
    if use_abi3t:
        yield from (Tag(interpreter, "abi3t", platform_) for platform_ in platforms)

    yield from (Tag(interpreter, "none", platform_) for platform_ in platforms)

    if use_abi3 or use_abi3t:
        for minor_version in range(python_version[1] - 1, 1, -1):
            for platform_ in platforms:
                version = _version_nodot((python_version[0], minor_version))
                interpreter = f"cp{version}"
                if use_abi3:
                    yield Tag(interpreter, "abi3", platform_)
                if use_abi3t:
                    # Support for abi3t was introduced in Python 3.15, but in
                    # principle abi3t wheels are possible for older limited API
                    # versions, so allow things like ("cp37", "abi3t", "platform")
                    yield Tag(interpreter, "abi3t", platform_)


def _generic_abi() -> list[str]:
    """
    Return the ABI tag based on EXT_SUFFIX.
    """
    # The following are examples of `EXT_SUFFIX`.
    # We want to keep the parts which are related to the ABI and remove the
    # parts which are related to the platform:
    # - linux:   '.cpython-310-x86_64-linux-gnu.so' => cp310
    # - mac:     '.cpython-310-darwin.so'           => cp310
    # - win:     '.cp310-win_amd64.pyd'             => cp310
    # - win:     '.pyd'                             => cp37 (uses _cpython_abis())
    # - pypy:    '.pypy38-pp73-x86_64-linux-gnu.so' => pypy38_pp73
    # - graalpy: '.graalpy-38-native-x86_64-darwin.dylib'
    #                                               => graalpy_38_native

    ext_suffix = _get_config_var("EXT_SUFFIX", warn=True)
    if not isinstance(ext_suffix, str) or not ext_suffix.startswith("."):
        raise SystemError("invalid sysconfig.get_config_var('EXT_SUFFIX')")
    parts = ext_suffix.split(".")
    if len(parts) < 3:
        # CPython3.7 and earlier uses ".pyd" on Windows.
        return _cpython_abis(sys.version_info[:2])
    soabi = parts[1]
    if soabi.startswith("cpython"):
        # non-windows
        cpython_parts = soabi.split("-")
        if len(cpython_parts) < 2 or not cpython_parts[1]:
            raise SystemError("invalid sysconfig.get_config_var('EXT_SUFFIX')")
        abi = "cp" + cpython_parts[1]
    elif soabi.startswith("cp"):
        # windows
        abi = soabi.split("-")[0]
    elif soabi.startswith("pypy"):
        abi = "-".join(soabi.split("-")[:2])
    elif soabi.startswith("graalpy"):
        abi = "-".join(soabi.split("-")[:3])
    elif soabi:
        # pyston, ironpython, others?
        abi = soabi
    else:
        return []
    return [_normalize_string(abi)]


def generic_tags(
    interpreter: str | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for an interpreter which requires no specialization.

    This function should be used if one of the other interpreter-specific
    functions provided by this module is not appropriate (i.e. not calculating
    tags for a CPython interpreter).

    The specific tags generated are:

    - ``<interpreter>-<abi>-<platform>``

    The ``"none"`` ABI will be added if it was not explicitly provided.

    :param str interpreter: The name of the interpreter. Defaults to being
                            calculated.
    :param Iterable abis: Iterable of compatible ABIs. Defaults to the ABIs
                          compatible with the current system.
    :param Iterable platforms: Iterable of compatible platforms. Defaults to the
                               platforms compatible with the current system.
    :param bool warn: Whether warnings should be logged. Defaults to ``False``.

    .. versionadded:: 20.0
    """
    if not interpreter:
        interp_name = interpreter_name()
        interp_version = interpreter_version(warn=warn)
        interpreter = f"{interp_name}{interp_version}"
    abis = _generic_abi() if abis is None else list(abis)
    platforms = list(platforms or platform_tags())
    if "none" not in abis:
        abis.append("none")
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)


def _py_interpreter_range(py_version: PythonVersion) -> Iterator[str]:
    """
    Yields Python versions in descending order.

    After the latest version, the major-only version will be yielded, and then
    all previous versions of that major version.
    """
    if len(py_version) > 1:
        yield f"py{_version_nodot(py_version[:2])}"
    yield f"py{py_version[0]}"
    if len(py_version) > 1:
        for minor in range(py_version[1] - 1, -1, -1):
            yield f"py{_version_nodot((py_version[0], minor))}"


def pure_python_tags(
    python_version: PythonVersion | None = None,
) -> Iterator[Tag]:
    """
    Yields the pure-Python tags compatible with ``python_version``.

    The tags use the ``"none"`` ABI and ``"any"`` platform, so their
    generation does not depend on the running platform.

    .. versionadded:: 26.3

    :param Sequence python_version: A one- or two-item sequence representing the
                                 compatible version of Python. Defaults to
                                 ``sys.version_info[:2]``.
    :raises ValueError: If ``python_version`` is an empty sequence.
    """
    if python_version is None:
        python_version = sys.version_info[:2]
    elif not python_version:
        raise ValueError("python_version must contain at least one item")
    for version in _py_interpreter_range(python_version):
        yield Tag(version, "none", "any")


def compatible_tags(
    python_version: PythonVersion | None = None,
    interpreter: str | None = None,
    platforms: Iterable[str] | None = None,
) -> Iterator[Tag]:
    """
    Yields the tags for an interpreter compatible with the Python version
    specified by ``python_version``.

    The specific tags generated are:

    - ``py*-none-<platform>``
    - ``<interpreter>-none-any`` if ``interpreter`` is provided
    - ``py*-none-any``

    :param Sequence python_version: A one- or two-item sequence representing the
                                 compatible version of Python. Defaults to
                                 ``sys.version_info[:2]``.
    :param str interpreter: The name of the interpreter (if known), e.g.
                            ``"cp38"``. Defaults to the current interpreter.
    :param Iterable platforms: Iterable of compatible platforms. Defaults to the
                               platforms compatible with the current system.

    .. versionadded:: 20.0
    """
    if not python_version:
        python_version = sys.version_info[:2]
    platforms = list(platforms or platform_tags())
    for version in _py_interpreter_range(python_version):
        for platform_ in platforms:
            yield Tag(version, "none", platform_)
    if interpreter:
        yield Tag(interpreter, "none", "any")
    yield from pure_python_tags(python_version)


def _mac_arch(arch: str, is_32bit: bool = _32_BIT_INTERPRETER) -> str:
    if not is_32bit:
        return arch

    if arch.startswith("ppc"):
        return "ppc"

    return "i386"


def _mac_binary_formats(version: AppleVersion, cpu_arch: str) -> list[str]:
    formats = [cpu_arch]
    if cpu_arch == "x86_64":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat64", "fat3"])

    elif cpu_arch == "i386":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat3", "fat"])

    elif cpu_arch == "ppc64":
        # TODO: Need to care about 32-bit PPC for ppc64 through 10.2?
        if version > (10, 5) or version < (10, 4):
            return []
        formats.append("fat64")

    elif cpu_arch == "ppc":
        if version > (10, 6):
            return []
        formats.extend(["fat3", "fat"])

    if cpu_arch in {"arm64", "x86_64"}:
        formats.append("universal2")

    if cpu_arch in {"x86_64", "i386", "ppc64", "ppc", "intel"}:
        formats.append("universal")

    return formats


def mac_platforms(
    version: AppleVersion | None = None, arch: str | None = None
) -> Iterator[str]:
    """
    Yields the :attr:`~Tag.platform` tags for macOS.

    The `version` parameter is a two-item tuple specifying the macOS version to
    generate platform tags for. The `arch` parameter is the CPU architecture to
    generate platform tags for. Both parameters default to the appropriate value
    for the current system.

    :param tuple version: A two-item tuple representing the version of macOS.
                          Defaults to the current system's version.
    :param str arch: The CPU architecture. Defaults to the architecture of the
                     current system, e.g. ``"x86_64"``.

    .. note::
        Equivalent support for the other major platforms is purposefully not
        provided:

        - On Windows, platform compatibility is statically specified
        - On Linux, code must be run on the system itself to determine
          compatibility

    .. versionadded:: 20.0
    """
    if version is None or arch is None:
        version_str, _, cpu_arch = platform.mac_ver()
        if version is None:
            version = cast("AppleVersion", tuple(map(int, version_str.split(".")[:2])))
            if version == (10, 16):
                # When built against an older macOS SDK, Python will report macOS 10.16
                # instead of the real version.
                version_str = subprocess.run(
                    [
                        sys.executable,
                        "-sS",
                        "-c",
                        "import platform; print(platform.mac_ver()[0])",
                    ],
                    check=True,
                    env={"SYSTEM_VERSION_COMPAT": "0"},
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout
                version = cast(
                    "AppleVersion", tuple(map(int, version_str.split(".")[:2]))
                )
        if arch is None:
            arch = _mac_arch(cpu_arch)

    if (10, 0) <= version < (11, 0):
        # Prior to Mac OS 11, each yearly release of Mac OS bumped the
        # "minor" version number.  The major version was always 10.
        major_version = 10
        for minor_version in range(version[1], -1, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

    if version >= (11, 0):
        # Starting with Mac OS 11, each yearly release bumps the major version
        # number.   The minor versions are now the midyear updates.
        minor_version = 0
        for major_version in range(version[0], 10, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

        # Mac OS 11 on x86_64 is compatible with binaries from previous releases.
        # Arm64 support was introduced in 11.0, so no Arm binaries from previous
        # releases exist.
        #
        # However, the "universal2" binary format can have a
        # macOS version earlier than 11.0 when the x86_64 part of the binary supports
        # that version of macOS.
        major_version = 10
        if arch == "x86_64":
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_formats = _mac_binary_formats(compat_version, arch)
                for binary_format in binary_formats:
                    yield f"macosx_{major_version}_{minor_version}_{binary_format}"
        else:
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_format = "universal2"
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"


def ios_platforms(
    version: AppleVersion | None = None, multiarch: str | None = None
) -> Iterator[str]:
    """

    Yields the :attr:`~Tag.platform` tags for iOS.

    :param tuple version: A two-item tuple representing the version of iOS.
                          Defaults to the current system's version.
    :param str multiarch: The CPU architecture+ABI to be used. This should be in
                          the format by ``sys.implementation._multiarch`` (e.g.,
                          ``arm64_iphoneos`` or ``x86_64_iphonesimulator``).
                          Defaults to the current system's multiarch value.

    .. note::
        Behavior of this method is undefined if invoked on non-iOS platforms
        without providing explicit version and multiarch arguments.

    .. versionadded:: 24.2
    """
    if version is None:
        # if iOS is the current platform, ios_ver *must* be defined. However,
        # it won't exist for CPython versions before 3.13, which causes a mypy
        # error.
        _, release, _, _ = platform.ios_ver()  # type: ignore[attr-defined, unused-ignore]
        version = cast("AppleVersion", tuple(map(int, release.split(".")[:2])))

    if multiarch is None:
        multiarch = sys.implementation._multiarch
    multiarch = multiarch.replace("-", "_")

    ios_platform_template = "ios_{major}_{minor}_{multiarch}"

    # Consider any iOS major.minor version from the version requested, down to
    # 12.0. 12.0 is the first iOS version that is known to have enough features
    # to support CPython. Consider every possible minor release up to X.9. There
    # highest the minor has ever gone is 8 (14.8 and 15.8) but having some extra
    # candidates that won't ever match doesn't really hurt, and it saves us from
    # having to keep an explicit list of known iOS versions in the code. Return
    # the results descending order of version number.

    # If the requested major version is less than 12, there won't be any matches.
    if version[0] < 12:
        return

    # Consider the actual X.Y version that was requested.
    yield ios_platform_template.format(
        major=version[0], minor=version[1], multiarch=multiarch
    )

    # Consider every minor version from X.0 to the minor version prior to the
    # version requested by the platform.
    for minor in range(version[1] - 1, -1, -1):
        yield ios_platform_template.format(
            major=version[0], minor=minor, multiarch=multiarch
        )

    for major in range(version[0] - 1, 11, -1):
        for minor in range(9, -1, -1):
            yield ios_platform_template.format(
                major=major, minor=minor, multiarch=multiarch
            )


def android_platforms(
    api_level: int | None = None, abi: str | None = None
) -> Iterator[str]:
    """
    Yields the :attr:`~Tag.platform` tags for Android. If this function is invoked on
    non-Android platforms, the ``api_level`` and ``abi`` arguments are required.

    :param int api_level: The maximum `API level
        <https://developer.android.com/tools/releases/platforms>`__ to return. Defaults
        to the current system's version, as returned by ``platform.android_ver``.
    :param str abi: The `Android ABI <https://developer.android.com/ndk/guides/abis>`__,
        e.g. ``arm64_v8a``. Defaults to the current system's ABI , as returned by
        ``sysconfig.get_platform``. Hyphens and periods will be replaced with
        underscores.

    .. versionadded:: 25.0
    """
    if platform.system() != "Android" and (api_level is None or abi is None):
        raise TypeError(
            "on non-Android platforms, the api_level and abi arguments are required"
        )

    if api_level is None:
        # Python 3.13 was the first version to return platform.system() == "Android",
        # and also the first version to define platform.android_ver().
        api_level = platform.android_ver().api_level  # type: ignore[attr-defined]

    if abi is None:
        abi = sysconfig.get_platform().split("-")[-1]
    abi = _normalize_string(abi)

    # 16 is the minimum API level known to have enough features to support CPython
    # without major patching. Yield every API level from the maximum down to the
    # minimum, inclusive.
    min_api_level = 16
    for ver in range(api_level, min_api_level - 1, -1):
        yield f"android_{ver}_{abi}"


def _linux_platforms(is_32bit: bool = _32_BIT_INTERPRETER) -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if not linux.startswith("linux_"):
        # we should never be here, just yield the sysconfig one and return
        yield linux
        return
    if is_32bit:
        if linux == "linux_x86_64":
            linux = "linux_i686"
        elif linux == "linux_aarch64":
            linux = "linux_armv8l"
    _, arch = linux.split("_", 1)
    archs = {"armv8l": ["armv8l", "armv7l"]}.get(arch, [arch])
    for arch in archs:
        yield f"linux_{arch}"
    yield from _manylinux.platform_tags(archs)
    yield from _musllinux.platform_tags(archs)


def _emscripten_platforms() -> Iterator[str]:
    pyemscripten_platform_version = sysconfig.get_config_var(
        "PYEMSCRIPTEN_PLATFORM_VERSION"
    )
    if pyemscripten_platform_version:
        yield f"pyemscripten_{pyemscripten_platform_version}_wasm32"
    yield from _generic_platforms()


def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())


def platform_tags() -> Iterator[str]:
    """
    Yields the :attr:`~Tag.platform` tags for the running interpreter.

    .. versionadded:: 21.1
    """
    if platform.system() == "Darwin":
        return mac_platforms()
    elif platform.system() == "iOS":
        return ios_platforms()
    elif platform.system() == "Android":
        return android_platforms()
    elif platform.system() == "Linux":
        return _linux_platforms()
    elif platform.system() == "Emscripten":
        return _emscripten_platforms()
    else:
        return _generic_platforms()


def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.

    Some implementations have a reserved, two-letter abbreviation which will
    be returned when appropriate.

    This typically acts as the prefix to the :attr:`~Tag.interpreter` tag.

    .. versionadded:: 20.0
    """
    name = sys.implementation.name
    return INTERPRETER_SHORT_NAMES.get(name) or name


def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the running interpreter's version.

    This typically acts as the suffix to the :attr:`~Tag.interpreter` tag.

    :param bool warn: Whether warnings should be logged. Defaults to ``False``.

    .. versionadded:: 20.0
    """
    version = _get_config_var("py_version_nodot", warn=warn)
    return str(version) if version else _version_nodot(sys.version_info[:2])


def _version_nodot(version: PythonVersion) -> str:
    return "".join(map(str, version))


def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    """
    Yields the sequence of tag triples that the running interpreter supports.

    The iterable is ordered so that the best-matching tag is first in the
    sequence. The exact preferential order to tags is interpreter-specific, but
    in general the tag importance is in the order of:

    1. Interpreter
    2. Platform
    3. ABI

    This order is due to the fact that an ABI is inherently tied to the
    platform, but platform-specific code is not necessarily tied to the ABI. The
    interpreter is the most important tag as it dictates basic support for any
    wheel.

    The function returns an iterable in order to allow for the possible
    short-circuiting of tag generation if the entire sequence is not necessary
    and tag calculation happens to be expensive.

    :param bool warn: Whether warnings should be logged. Defaults to ``False``.

    .. versionchanged:: 21.3
        Added the `pp3-none-any` tag (:issue:`311`).
    .. versionchanged:: 26.1
        Added the `abi3t` tag (:issue:`1099`).
    .. versionchanged:: 26.3
        Native ``linux_*`` platform tags are now ordered before ``manylinux``
        and ``musllinux`` tags (:issue:`160`).
    """

    interp_name = interpreter_name()
    if interp_name == "cp":
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags(warn=warn)

    if interp_name == "pp":
        interp = "pp3"
    elif interp_name == "cp":
        interp = "cp" + interpreter_version(warn=warn)
    else:
        interp = None
    yield from compatible_tags(interpreter=interp)


def create_compatible_tags_selector(
    tags: Iterable[Tag],
) -> Callable[[Iterable[tuple[_T, AbstractSet[Tag]]]], Iterator[_T]]:
    """Create a callable to select things compatible with supported tags.

    This function accepts an ordered sequence of tags, with the preferred
    tags first.

    The returned callable accepts an iterable of tuples (thing, set[Tag]),
    and returns an iterator of things, with the things with the best
    matching tags first.

    Example to select compatible wheel filenames:

    >>> from packaging import tags
    >>> from packaging.utils import parse_wheel_filename
    >>> selector = tags.create_compatible_tags_selector(tags.sys_tags())
    >>> filenames = ["foo-1.0-py3-none-any.whl", "foo-1.0-py2-none-any.whl"]
    >>> list(selector([
    ...     (filename, parse_wheel_filename(filename)[-1]) for filename in filenames
    ... ]))
    ['foo-1.0-py3-none-any.whl']

    .. versionadded:: 26.1
    """
    tag_ranks: dict[Tag, int] = {}
    for rank, tag in enumerate(tags):
        tag_ranks.setdefault(tag, rank)  # ignore duplicate tags, keep first
    supported_tags = tag_ranks.keys()

    def selector(
        tagged_things: Iterable[tuple[_T, AbstractSet[Tag]]],
    ) -> Iterator[_T]:
        ranked_things: list[tuple[_T, int]] = []
        for thing, thing_tags in tagged_things:
            supported_thing_tags = thing_tags & supported_tags
            if supported_thing_tags:
                thing_rank = min(tag_ranks[t] for t in supported_thing_tags)
                ranked_things.append((thing, thing_rank))
        return iter(
            thing for thing, _ in sorted(ranked_things, key=operator.itemgetter(1))
        )

    return selector
<|endoftext|># This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from packaging.version import parse, normalize_pre, Version, _cmpkey
"""

from __future__ import annotations

import re
import sys
import typing
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    SupportsInt,
    TypedDict,
    Union,
)

if typing.TYPE_CHECKING:
    from typing_extensions import Self, Unpack

if sys.version_info >= (3, 13):  # pragma: no cover
    from warnings import deprecated as _deprecated
elif typing.TYPE_CHECKING:
    from typing_extensions import deprecated as _deprecated
else:  # pragma: no cover
    import functools
    import warnings

    def _deprecated(message: str) -> object:
        def decorator(func: Callable[[...], object]) -> object:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                warnings.warn(
                    message,
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator


_LETTER_NORMALIZATION = {
    "alpha": "a",
    "beta": "b",
    "c": "rc",
    "pre": "rc",
    "preview": "rc",
    "rev": "post",
    "r": "post",
}

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "normalize_pre", "parse"]


def __dir__() -> list[str]:
    return __all__


LocalType = tuple[Union[int, str], ...]

CmpLocalType = tuple[tuple[int, str], ...]
CmpSuffix = tuple[int, int, int, int, int, int]
CmpKey = Union[
    tuple[int, tuple[int, ...], CmpSuffix],
    tuple[int, tuple[int, ...], CmpSuffix, CmpLocalType],
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _VersionReplace(TypedDict, total=False):
    epoch: int | None
    release: tuple[int, ...] | None
    pre: tuple[str, int] | None
    post: int | None
    dev: int | None
    local: str | None


def normalize_pre(letter: str, /) -> str:
    """Normalize the pre-release segment of a version string.

    Returns a lowercase version of the string if not a known pre-release
    identifier.

    >>> normalize_pre('alpha')
    'a'
    >>> normalize_pre('BETA')
    'b'
    >>> normalize_pre('rc')
    'rc'

    :param letter:

    .. versionadded:: 26.1
    """
    letter = letter.lower()
    return _LETTER_NORMALIZATION.get(letter, letter)


def parse(version: str) -> Version:
    """Parse the given version string.

    This is identical to the :class:`Version` constructor.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    __slots__ = ()

    # This can also be a normal member (see the packaging_legacy package);
    # we are just requiring it to be readable. Actually defining a property
    # has runtime effect on subclasses, so it's typing only.
    if typing.TYPE_CHECKING:

        @property
        def _key(self) -> tuple[Any, ...]: ...

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse

# Note that ++ doesn't behave identically on CPython and PyPy, so not using it here
_VERSION_PATTERN = r"""
    v?+                                                   # optional leading v
    (?a:
        (?:(?P<epoch>[0-9]+)!)?+                          # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*+)                 # release segment
        (?P<pre>                                          # pre-release
            [._-]?+
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [._-]?+
            (?P<pre_n>[0-9]+)?
        )?+
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [._-]?
                (?P<post_l>post|rev|r)
                [._-]?
                (?P<post_n2>[0-9]+)?
            )
        )?+
        (?P<dev>                                          # dev release
            [._-]?+
            (?P<dev_l>dev)
            [._-]?+
            (?P<dev_n>[0-9]+)?
        )?+
    )
    (?a:\+
        (?P<local>                                        # local version
            [a-z0-9]+
            (?:[._-][a-z0-9]+)*+
        )
    )?+
"""

_VERSION_PATTERN_OLD = _VERSION_PATTERN.replace("*+", "*").replace("?+", "?")

# Possessive qualifiers were added in Python 3.11.
# CPython 3.11.0-3.11.4 had a bug: https://github.com/python/cpython/pull/107795
# Older PyPy also had a bug.
VERSION_PATTERN = (
    _VERSION_PATTERN_OLD
    if (sys.implementation.name == "cpython" and sys.version_info < (3, 11, 5))
    or (sys.implementation.name == "pypy" and sys.version_info < (3, 11, 13))
    or sys.version_info < (3, 11)
    else _VERSION_PATTERN
)
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

.. versionchanged:: 26.0

   The regex now uses possessive qualifiers on Python 3.11 if they are
   supported (CPython 3.11.5+, PyPy 3.11.13+).

:meta hide-value:
"""


# Validation pattern for local version in replace()
_LOCAL_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE | re.ASCII)

# Fast path: If a version has only digits and dots then we
# can skip the regex and parse it as a release segment
_SIMPLE_VERSION_INDICATORS = frozenset(".0123456789")


def _validate_epoch(value: object, /) -> int:
    epoch = value or 0
    if isinstance(epoch, int) and epoch >= 0:
        return epoch
    msg = f"epoch must be non-negative integer, got {epoch}"
    raise InvalidVersion(msg)


def _validate_release(value: object, /) -> tuple[int, ...]:
    release = (0,) if value is None else value
    if (
        isinstance(release, tuple)
        and len(release) > 0
        and all(isinstance(i, int) and i >= 0 for i in release)
    ):
        return release
    msg = f"release must be a non-empty tuple of non-negative integers, got {release}"
    raise InvalidVersion(msg)


def _validate_pre(value: object, /) -> tuple[Literal["a", "b", "rc"], int] | None:
    if value is None:
        return value
    if isinstance(value, tuple) and len(value) == 2:
        letter, number = value
        # The letter must be a string before it can be normalized.
        if (
            isinstance(letter, str)
            and (normalized := normalize_pre(letter)) in {"a", "b", "rc"}
            and isinstance(number, int)
            and number >= 0
        ):
            # type checkers can't infer the Literal type here on letter
            return (normalized, number)  # type: ignore[return-value]
    msg = f"pre must be a tuple of ('a'|'b'|'rc', non-negative int), got {value}"
    raise InvalidVersion(msg)


def _validate_post(value: object, /) -> tuple[Literal["post"], int] | None:
    if value is None:
        return value
    if isinstance(value, int) and value >= 0:
        return ("post", value)
    msg = f"post must be non-negative integer, got {value}"
    raise InvalidVersion(msg)


def _validate_dev(value: object, /) -> tuple[Literal["dev"], int] | None:
    if value is None:
        return value
    if isinstance(value, int) and value >= 0:
        return ("dev", value)
    msg = f"dev must be non-negative integer, got {value}"
    raise InvalidVersion(msg)


def _validate_local(value: object, /) -> LocalType | None:
    if value is None:
        return value
    if isinstance(value, str) and _LOCAL_PATTERN.fullmatch(value):
        return _parse_local_version(value)
    msg = f"local must be a valid version string, got {value!r}"
    raise InvalidVersion(msg)


# Backward compatibility for internals before 26.0. Do not use.
class _Version(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[Literal["dev"], int] | None
    pre: tuple[Literal["a", "b", "rc"], int] | None
    post: tuple[Literal["post"], int] | None
    local: LocalType | None


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True

    :class:`Version` is immutable; use :meth:`__replace__` to change
    part of a version.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = (
        "_dev",
        "_epoch",
        "_hash_cache",
        "_key_cache",
        "_local",
        "_post",
        "_pre",
        "_release",
    )
    __match_args__ = ("_str",)
    """
    Pattern matching is supported on Python 3.10+.

    .. versionadded:: 26.0

    :meta hide-value:
    """

    _regex = re.compile(r"\s*" + VERSION_PATTERN + r"\s*", re.VERBOSE | re.IGNORECASE)

    _epoch: int
    _release: tuple[int, ...]
    _dev: tuple[Literal["dev"], int] | None
    _pre: tuple[Literal["a", "b", "rc"], int] | None
    _post: tuple[Literal["post"], int] | None
    _local: LocalType | None

    _hash_cache: int | None
    _key_cache: CmpKey | None

    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """
        try:
            is_simple = _SIMPLE_VERSION_INDICATORS.issuperset(version)
        except TypeError:
            raise InvalidVersion(f"Invalid version: {version!r}") from None

        if is_simple:
            try:
                self._release = tuple(map(int, version.split(".")))
            except AttributeError:
                raise InvalidVersion(f"Invalid version: {version!r}") from None
            except ValueError:
                # Empty parts (from "1..2", ".1", etc.) are invalid versions.
                # Any other ValueError (e.g. int str-digits limit) should
                # propagate to the caller.
                if "" in version.split("."):
                    raise InvalidVersion(f"Invalid version: {version!r}") from None
                # TODO: remove "no cover" when Python 3.9 is dropped.
                raise  # pragma: no cover

            self._epoch = 0
            self._pre = None
            self._post = None
            self._dev = None
            self._local = None
            self._key_cache = None
            self._hash_cache = None
            return

        # Validate the version and parse it into pieces
        try:
            match = self._regex.fullmatch(version)
        except TypeError:
            raise InvalidVersion(f"Invalid version: {version!r}") from None
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")
        self._epoch = int(match.group("epoch")) if match.group("epoch") else 0
        self._release = tuple(map(int, match.group("release").split(".")))
        # We can type ignore the assignments below because the regex guarantees
        # the correct strings
        self._pre = _parse_letter_version(match.group("pre_l"), match.group("pre_n"))  # type: ignore[assignment]
        self._post = _parse_letter_version(  # type: ignore[assignment]
            match.group("post_l"), match.group("post_n1") or match.group("post_n2")
        )
        self._dev = _parse_letter_version(match.group("dev_l"), match.group("dev_n"))  # type: ignore[assignment]
        self._local = _parse_local_version(match.group("local"))

        # Key which will be used for sorting
        self._key_cache = None
        self._hash_cache = None

    @classmethod
    def from_parts(
        cls,
        *,
        epoch: int = 0,
        release: tuple[int, ...],
        pre: tuple[str, int] | None = None,
        post: int | None = None,
        dev: int | None = None,
        local: str | None = None,
    ) -> Self:
        """
        Return a new version composed of the various parts.

        This allows you to build a version without going though a string and
        running a regular expression. It normalizes pre-release strings. The
        ``release=`` keyword argument is required.

        >>> Version.from_parts(release=(1,2,3))
        <Version('1.2.3')>
        >>> Version.from_parts(release=(0,1,0), pre=("b", 1))
        <Version('0.1.0b1')>

        :param epoch:
        :param release: This version tuple is required

        .. versionadded:: 26.1
        """
        _epoch = _validate_epoch(epoch)
        _release = _validate_release(release)
        _pre = _validate_pre(pre) if pre is not None else None
        _post = _validate_post(post) if post is not None else None
        _dev = _validate_dev(dev) if dev is not None else None
        _local = _validate_local(local) if local is not None else None

        new_version = cls.__new__(cls)
        new_version._key_cache = None
        new_version._hash_cache = None
        new_version._epoch = _epoch
        new_version._release = _release
        new_version._pre = _pre
        new_version._post = _post
        new_version._dev = _dev
        new_version._local = _local

        return new_version

    def __replace__(self, **kwargs: Unpack[_VersionReplace]) -> Self:
        """
        __replace__(*, epoch=..., release=..., pre=..., post=..., dev=..., local=...)

        Return a new version with parts replaced.

        This returns a new version (unless no parts were changed). The
        pre-release is normalized. Setting a value to ``None`` clears it.

        >>> v = Version("1.2.3")
        >>> v.__replace__(pre=("a", 1))
        <Version('1.2.3a1')>

        :param int | None epoch:
        :param tuple[int, ...] | None release:
        :param tuple[str, int] | None pre:
        :param int | None post:
        :param int | None dev:
        :param str | None local:

        .. versionadded:: 26.0
        .. versionchanged:: 26.1

           The pre-release portion is now normalized.
        """
        epoch = _validate_epoch(kwargs["epoch"]) if "epoch" in kwargs else self._epoch
        release = (
            _validate_release(kwargs["release"])
            if "release" in kwargs
            else self._release
        )
        pre = _validate_pre(kwargs["pre"]) if "pre" in kwargs else self._pre
        post = _validate_post(kwargs["post"]) if "post" in kwargs else self._post
        dev = _validate_dev(kwargs["dev"]) if "dev" in kwargs else self._dev
        local = _validate_local(kwargs["local"]) if "local" in kwargs else self._local

        if (
            epoch == self._epoch
            and release == self._release
            and pre == self._pre
            and post == self._post
            and dev == self._dev
            and local == self._local
        ):
            return self

        new_version = self.__class__.__new__(self.__class__)
        new_version._key_cache = None
        new_version._hash_cache = None
        new_version._epoch = epoch
        new_version._release = release
        new_version._pre = pre
        new_version._post = post
        new_version._dev = dev
        new_version._local = local

        return new_version

    @property
    def _key(self) -> CmpKey:
        if self._key_cache is None:
            self._key_cache = _cmpkey(
                self._epoch,
                self._release,
                self._pre,
                self._post,
                self._dev,
                self._local,
            )
        return self._key_cache

    # __hash__ must be defined when __eq__ is overridden,
    # otherwise Python sets __hash__ to None.
    def __hash__(self) -> int:
        if (cached_hash := self._hash_cache) is not None:
            return cached_hash

        if (key := self._key_cache) is None:
            self._key_cache = key = _cmpkey(
                self._epoch,
                self._release,
                self._pre,
                self._post,
                self._dev,
                self._local,
            )
        self._hash_cache = cached_hash = hash(key)
        return cached_hash

    # Override comparison methods to use direct _key_cache access
    # This is faster than property access, especially before Python 3.12
    def __lt__(self, other: _BaseVersion) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache < other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__lt__(other)

    def __le__(self, other: _BaseVersion) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache <= other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__le__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache == other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__eq__(other)

    def __ge__(self, other: _BaseVersion) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache >= other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__ge__(other)

    def __gt__(self, other: _BaseVersion) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache > other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__gt__(other)

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Version):
            if self._key_cache is None:
                self._key_cache = _cmpkey(
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                )
            if other._key_cache is None:
                other._key_cache = _cmpkey(
                    other._epoch,
                    other._release,
                    other._pre,
                    other._post,
                    other._dev,
                    other._local,
                )
            return self._key_cache != other._key_cache

        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return super().__ne__(other)

    def __getstate__(
        self,
    ) -> tuple[
        int,
        tuple[int, ...],
        tuple[str, int] | None,
        tuple[str, int] | None,
        tuple[str, int] | None,
        LocalType | None,
    ]:
        # Return state as a 6-item tuple for compactness:
        #   (epoch, release, pre, post, dev, local)
        # Cache members are excluded and will be recomputed on demand
        return (
            self._epoch,
            self._release,
            self._pre,
            self._post,
            self._dev,
            self._local,
        )

    def __setstate__(self, state: object) -> None:
        # Always discard cached values — they may contain stale references
        # (e.g. packaging._structures.InfinityType from pre-26.1 pickles)
        # and will be recomputed on demand from the core fields above.
        self._key_cache = None
        self._hash_cache = None

        if isinstance(state, tuple):
            if len(state) == 6:
                # New format (26.2+): (epoch, release, pre, post, dev, local)
                (
                    self._epoch,
                    self._release,
                    self._pre,
                    self._post,
                    self._dev,
                    self._local,
                ) = state
                return
            if len(state) == 2:
                # Format (packaging 26.0-26.1): (None, {slot: value}).
                _, slot_dict = state
                if isinstance(slot_dict, dict):
                    self._epoch = slot_dict["_epoch"]
                    self._release = slot_dict["_release"]
                    self._pre = slot_dict.get("_pre")
                    self._post = slot_dict.get("_post")
                    self._dev = slot_dict.get("_dev")
                    self._local = slot_dict.get("_local")
                    return
        if isinstance(state, dict):
            # Old format (packaging <= 25.x, no __slots__): state is a plain
            # dict with "_version" (_Version NamedTuple) and "_key" entries.
            version_nt = state.get("_version")
            if version_nt is not None:
                self._epoch = version_nt.epoch
                self._release = version_nt.release
                self._pre = version_nt.pre
                self._post = version_nt.post
                self._dev = version_nt.dev
                self._local = version_nt.local
                return

        raise TypeError(f"Cannot restore Version from {state!r}")

    @property
    @_deprecated("Version._version is private and will be removed soon")
    def _version(self) -> _Version:
        return _Version(
            self._epoch, self._release, self._dev, self._pre, self._post, self._local
        )

    @_version.setter
    @_deprecated("Version._version is private and will be removed soon")
    def _version(self, value: _Version) -> None:
        self._epoch = value.epoch
        self._release = value.release
        self._dev = value.dev
        self._pre = value.pre
        self._post = value.post
        self._local = value.local
        self._key_cache = None
        self._hash_cache = None

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<{self.__class__.__name__}({str(self)!r})>"

    def __str__(self) -> str:
        """A string representation of the version that can be round-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        # This is a hot function, so not calling self.base_version
        version = ".".join(map(str, self.release))

        # Epoch
        if self.epoch:
            version = f"{self.epoch}!{version}"

        # Pre-release
        if self.pre is not None:
            version += "".join(map(str, self.pre))

        # Post-release
        if self.post is not None:
            version += f".post{self.post}"

        # Development release
        if self.dev is not None:
            version += f".dev{self.dev}"

        # Local version segment
        if self.local is not None:
            version += f"+{self.local}"

        return version

    @property
    def _str(self) -> str:
        """Internal property for match_args"""
        return str(self)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._epoch

    @property
    def release(self) -> tuple[int, ...]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._release

    @property
    def pre(self) -> tuple[Literal["a", "b", "rc"], int] | None:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._pre

    @property
    def post(self) -> int | None:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._post[1] if self._post else None

    @property
    def dev(self) -> int | None:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._dev[1] if self._dev else None

    @property
    def local(self) -> str | None:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._local:
            return ".".join(str(x) for x in self._local)
        else:
            return None

    @property
    def public(self) -> str:
        """The public portion of the version.

        This returns a string. If you want a :class:`Version` again and care
        about performance, use ``v.__replace__(local=None)`` instead.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").public
        '1!1.2.3.dev1'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        This returns a string. If you want a :class:`Version` again and care
        about performance, use
        ``v.__replace__(pre=None, post=None, dev=None, local=None)`` instead.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        release_segment = ".".join(map(str, self.release))
        return f"{self.epoch}!{release_segment}" if self.epoch else release_segment

    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1

        .. versionadded:: 20.0
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0

        .. versionadded:: 20.0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0

        .. versionadded:: 20.0
        """
        return self.release[2] if len(self.release) >= 3 else 0


class _TrimmedRelease(Version):
    __slots__ = ()

    def __init__(self, version: str | Version) -> None:
        if isinstance(version, Version):
            self._epoch = version._epoch
            self._release = version._release
            self._dev = version._dev
            self._pre = version._pre
            self._post = version._post
            self._local = version._local
            self._key_cache = version._key_cache
            self._hash_cache = version._hash_cache
            return
        super().__init__(version)  # pragma: no cover

    @property
    def release(self) -> tuple[int, ...]:
        """
        Release segment without any trailing zeros.

        >>> _TrimmedRelease('1.0.0').release
        (1,)
        >>> _TrimmedRelease('0.0').release
        (0,)
        """
        # This leaves one 0.
        rel = super().release
        len_release = len(rel)
        i = len_release
        while i > 1 and rel[i - 1] == 0:
            i -= 1
        return rel if i == len_release else rel[:i]


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    if letter:
        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        letter = _LETTER_NORMALIZATION.get(letter, letter)

        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        return letter, int(number or 0)

    if number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        return "post", int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    """
    Takes a string like ``"abc.1.twelve"`` and turns it into
    ``("abc", 1, "twelve")``.
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


# Sort ranks for pre-release: dev-only < a < b < rc < stable (no pre-release).
_PRE_RANK = {"a": 0, "b": 1, "rc": 2}
_PRE_RANK_DEV_ONLY = -1  # sorts before a(0)
_PRE_RANK_STABLE = 3  # sorts after rc(2)

# In local version segments, strings sort before ints per PEP 440.
_LOCAL_STR_RANK = -1  # sorts before all non-negative ints

# Pre-computed suffix for stable releases (no pre, post, or dev segments).
# See _cmpkey() for the suffix layout.
_STABLE_SUFFIX = (_PRE_RANK_STABLE, 0, 0, 0, 1, 0)


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
