    def _inline_attrs(self, schema: dict, path: Sequence[str]) -> Iterator[str]:
        for key, value in schema.items():
            child_path = [*path, key]
            yield f"{self._label(child_path)}: {self._value(value, child_path)}"

    def _child_prefix(self, parent_prefix: str, child_prefix: str) -> str:
        return len(parent_prefix) * " " + child_prefix


def _separate_terms(word: str) -> list[str]:
    """
    >>> _separate_terms("FooBar-foo")
    ['foo', 'bar', 'foo']
    """
    return [w.lower() for w in _CAMEL_CASE_SPLITTER.split(word) if w]
<|endoftext|>import re


SPLIT_RE = re.compile(r'[\.\[\]]+')


class JsonSchemaException(ValueError):
    """
    Base exception of ``fastjsonschema`` library.
    """


class JsonSchemaValueException(JsonSchemaException):
    """
    Exception raised by validation function. Available properties:

     * ``message`` containing human-readable information what is wrong (e.g. ``data.property[index] must be smaller than or equal to 42``),
     * invalid ``value`` (e.g. ``60``),
     * ``name`` of a path in the data structure (e.g. ``data.property[index]``),
     * ``path`` as an array in the data structure (e.g. ``['data', 'property', 'index']``),
     * the whole ``definition`` which the ``value`` has to fulfil (e.g. ``{'type': 'number', 'maximum': 42}``),
     * ``rule`` which the ``value`` is breaking (e.g. ``maximum``)
     * and ``rule_definition`` (e.g. ``42``).

    .. versionchanged:: 2.14.0
        Added all extra properties.
    """

    def __init__(self, message, value=None, name=None, definition=None, rule=None):
        super().__init__(message)
        self.message = message
        self.value = value
        self.name = name
        self.definition = definition
        self.rule = rule

    @property
    def path(self):
        return [item for item in SPLIT_RE.split(self.name) if item != '']

    @property
    def rule_definition(self):
        if not self.rule or not self.definition:
            return None
        return self.definition.get(self.rule)


class JsonSchemaDefinitionException(JsonSchemaException):
    """
    Exception raised by generator of validation function.
    """
<|endoftext|>"""
The functions in this module are used to validate schemas with the
`format JSON Schema keyword
<https://json-schema.org/understanding-json-schema/reference/string#format>`_.

The correspondence is given by replacing the ``_`` character in the name of the
function with a ``-`` to obtain the format name and vice versa.
"""

from __future__ import annotations

import keyword
import logging
import os
import re
import string
import typing
from itertools import chain as _chain

if typing.TYPE_CHECKING:
    import builtins

    from typing_extensions import Literal

_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# PEP 440

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_REGEX = re.compile(
    r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE
)


def pep440(version: str) -> bool:
    """See :ref:`PyPA's version specification <pypa:version-specifiers>`
    (initially introduced in :pep:`440`).
    """
    return VERSION_REGEX.match(version) is not None


# -------------------------------------------------------------------------------------
# PEP 508

PEP508_IDENTIFIER_PATTERN = r"([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])"
PEP508_IDENTIFIER_REGEX = re.compile(f"^{PEP508_IDENTIFIER_PATTERN}$", re.IGNORECASE)


def pep508_identifier(name: str) -> bool:
    """See :ref:`PyPA's name specification <pypa:name-format>`
    (initially introduced in :pep:`508#names`).
    """
    return PEP508_IDENTIFIER_REGEX.match(name) is not None


try:
    try:
        from packaging import requirements as _req
    except ImportError:  # pragma: no cover
        # let's try setuptools vendored version
        from setuptools._vendor.packaging import (  # type: ignore[no-redef]
            requirements as _req,
        )

    def pep508(value: str) -> bool:
        """See :ref:`PyPA's dependency specifiers <pypa:dependency-specifiers>`
        (initially introduced in :pep:`508`).
        """
        try:
            _req.Requirement(value)
        except _req.InvalidRequirement:
            return False
        return True

except ImportError:  # pragma: no cover
    _logger.warning(
        "Could not find an installation of `packaging`. Requirements, dependencies and "
        "versions might not be validated. "
        "To enforce validation, please install `packaging`."
    )

    def pep508(value: str) -> bool:  # noqa: ARG001
        return True


def pep508_versionspec(value: str) -> bool:
    """Expression that can be used to specify/lock versions (including ranges)
    See ``versionspec`` in :ref:`PyPA's dependency specifiers
    <pypa:dependency-specifiers>` (initially introduced in :pep:`508`).
    """
    if any(c in value for c in (";", "]", "@")):
        # In PEP 508:
        # conditional markers, extras and URL specs are not included in the
        # versionspec
        return False
    # Let's pretend we have a dependency called `requirement` with the given
    # version spec, then we can reuse the pep508 function for validation:
    return pep508(f"requirement{value}")


# -------------------------------------------------------------------------------------
# PEP 517


def pep517_backend_reference(value: str) -> bool:
    """See PyPA's specification for defining build-backend references
    introduced in :pep:`517#source-trees`.

    This is similar to an entry-point reference (e.g., ``package.module:object``).
    """
    module, _, obj = value.partition(":")
    identifiers = (i.strip() for i in _chain(module.split("."), obj.split(".")))
    return all(python_identifier(i) for i in identifiers if i)


# -------------------------------------------------------------------------------------
# Classifiers - PEP 301


def _download_classifiers() -> str:
    import ssl
    from email.message import Message
    from urllib.request import urlopen

    url = "https://pypi.org/pypi?:action=list_classifiers"
    context = ssl.create_default_context()
    with urlopen(url, context=context) as response:  # noqa: S310 (audit URLs)
        headers = Message()
        headers["content_type"] = response.getheader("content-type", "text/plain")
        return response.read().decode(headers.get_param("charset", "utf-8"))  # type: ignore[no-any-return]


class _TroveClassifier:
    """The ``trove_classifiers`` package is the official way of validating classifiers,
    however this package might not be always available.
    As a workaround we can still download a list from PyPI.
    We also don't want to be over strict about it, so simply skipping silently is an
    option (classifiers will be validated anyway during the upload to PyPI).
    """

    downloaded: None | Literal[False] | set[str]
    """
    None => not cached yet
    False => unavailable
    set => cached values
    """

    def __init__(self) -> None:
        self.downloaded = None
        self._skip_download = False
        self.__name__ = "trove_classifier"  # Emulate a public function

    def _disable_download(self) -> None:
        # This is a private API. Only setuptools has the consent of using it.
        self._skip_download = True

    def __call__(self, value: str) -> bool:
        if self.downloaded is False or self._skip_download is True:
            return True

        if os.getenv("NO_NETWORK") or os.getenv("VALIDATE_PYPROJECT_NO_NETWORK"):
            self.downloaded = False
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Skipping download of classifiers list from PyPI (NO_NETWORK)."
            )
            _logger.debug(msg)
            return True

        if self.downloaded is None:
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Meanwhile a list of classifiers will be downloaded from PyPI."
            )
            _logger.debug(msg)
            try:
                self.downloaded = set(_download_classifiers().splitlines())
            except Exception:  # noqa: BLE001
                self.downloaded = False
                _logger.debug("Problem with download, skipping validation")
                return True

        return value in self.downloaded or value.lower().startswith("private ::")


try:
    from trove_classifiers import classifiers as _trove_classifiers

    def trove_classifier(value: str) -> bool:
        """See https://pypi.org/classifiers/"""
        return value in _trove_classifiers or value.lower().startswith("private ::")

except ImportError:  # pragma: no cover
    trove_classifier = _TroveClassifier()


# -------------------------------------------------------------------------------------
# Stub packages - PEP 561


def pep561_stub_name(value: str) -> bool:
    """Name of a directory containing type stubs.
    It must follow the name scheme ``<package>-stubs`` as defined in
    :pep:`561#stub-only-packages`.
    """
    top, *children = value.split(".")
    if not top.endswith("-stubs"):
        return False
    return python_module_name(".".join([top[: -len("-stubs")], *children]))


# -------------------------------------------------------------------------------------
# Non-PEP related


def url(value: str) -> bool:
    """Valid URL (validation uses :obj:`urllib.parse`).
    For maximum compatibility please make sure to include a ``scheme`` prefix
    in your URL (e.g. ``http://``).
    """
    from urllib.parse import urlparse

    try:
        parts = urlparse(value)
        if not parts.scheme:
            _logger.warning(
                "For maximum compatibility please make sure to include a "
                "`scheme` prefix in your URL (e.g. 'http://'). "
                f"Given value: {value}"
            )
            if not (value.startswith(("/", "\\")) or "@" in value):
                parts = urlparse(f"http://{value}")

        return bool(parts.scheme and parts.netloc)
    except Exception:  # noqa: BLE001
        return False


# https://packaging.python.org/specifications/entry-points/
ENTRYPOINT_PATTERN = r"[^\[\s=]([^=]*[^\s=])?"
ENTRYPOINT_REGEX = re.compile(f"^{ENTRYPOINT_PATTERN}$", re.IGNORECASE)
RECOMMEDED_ENTRYPOINT_PATTERN = r"[\w.-]+"
RECOMMEDED_ENTRYPOINT_REGEX = re.compile(
    f"^{RECOMMEDED_ENTRYPOINT_PATTERN}$", re.IGNORECASE
)
ENTRYPOINT_GROUP_PATTERN = r"\w+(\.\w+)*"
ENTRYPOINT_GROUP_REGEX = re.compile(f"^{ENTRYPOINT_GROUP_PATTERN}$", re.IGNORECASE)


def python_identifier(value: str) -> bool:
    """Can be used as identifier in Python.
    (Validation uses :obj:`str.isidentifier`).
    """
    return value.isidentifier()


def python_qualified_identifier(value: str) -> bool:
    """
    Python "dotted identifier", i.e. a sequence of :obj:`python_identifier`
    concatenated with ``"."`` (e.g.: ``package.module.submodule``).
    """
    if value.startswith(".") or value.endswith("."):
        return False
    return all(python_identifier(m) for m in value.split("."))


def python_module_name(value: str) -> bool:
    """Module name that can be used in an ``import``-statement in Python.
    See :obj:`python_qualified_identifier`.
    """
    return python_qualified_identifier(value)


def python_module_name_relaxed(value: str) -> bool:
    """Similar to :obj:`python_module_name`, but relaxed to also accept
    dash characters (``-``) and cover special cases like ``pip-run``.

    It is recommended, however, that beginners avoid dash characters,
    as they require advanced knowledge about Python internals.

    The following are disallowed:

    * names starting/ending in dashes,
    * names ending in ``-stubs`` (potentially collide with :obj:`pep561_stub_name`).
    """
    if value.startswith("-") or value.endswith("-"):
        return False
    if value.endswith("-stubs"):
        return False  # Avoid collision with PEP 561
    return python_module_name(value.replace("-", "_"))


def python_entrypoint_group(value: str) -> bool:
    """See ``Data model > group`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    return ENTRYPOINT_GROUP_REGEX.match(value) is not None


def python_entrypoint_name(value: str) -> bool:
    """See ``Data model > name`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    if not ENTRYPOINT_REGEX.match(value):
        return False
    if not RECOMMEDED_ENTRYPOINT_REGEX.match(value):
        msg = f"Entry point `{value}` does not follow recommended pattern: "
        msg += RECOMMEDED_ENTRYPOINT_PATTERN
        _logger.warning(msg)
    return True


def python_entrypoint_reference(value: str) -> bool:
    """Reference to a Python object using in the format::

        importable.module:object.attr

    See ``Data model >object reference`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    module, _, rest = value.partition(":")
    if "[" in rest:
        obj, _, extras_ = rest.partition("[")
        if extras_.strip()[-1] != "]":
            return False
        extras = (x.strip() for x in extras_.strip(string.whitespace + "[]").split(","))
        if not all(pep508_identifier(e) for e in extras):
            return False
        _logger.warning(f"`{value}` - using extras for entry points is not recommended")
    else:
        obj = rest

    module_parts = module.split(".")
    identifiers = _chain(module_parts, obj.split(".")) if rest else iter(module_parts)
    return all(python_identifier(i.strip()) for i in identifiers)


def uint8(value: builtins.int) -> bool:
    r"""Unsigned 8-bit integer (:math:`0 \leq x < 2^8`)"""
    return 0 <= value < 2**8


def uint16(value: builtins.int) -> bool:
    r"""Unsigned 16-bit integer (:math:`0 \leq x < 2^{16}`)"""
    return 0 <= value < 2**16


def uint32(value: builtins.int) -> bool:
    r"""Unsigned 32-bit integer (:math:`0 \leq x < 2^{32}`)"""
    return 0 <= value < 2**32


def uint64(value: builtins.int) -> bool:
    r"""Unsigned 64-bit integer (:math:`0 \leq x < 2^{64}`)"""
    return 0 <= value < 2**64


def uint(value: builtins.int) -> bool:
    r"""Signed 64-bit integer (:math:`0 \leq x < 2^{64}`)"""
    return 0 <= value < 2**64


def int8(value: builtins.int) -> bool:
    r"""Signed 8-bit integer (:math:`-2^{7} \leq x < 2^{7}`)"""
    return -(2**7) <= value < 2**7


def int16(value: builtins.int) -> bool:
    r"""Signed 16-bit integer (:math:`-2^{15} \leq x < 2^{15}`)"""
    return -(2**15) <= value < 2**15


def int32(value: builtins.int) -> bool:
    r"""Signed 32-bit integer (:math:`-2^{31} \leq x < 2^{31}`)"""
    return -(2**31) <= value < 2**31


def int64(value: builtins.int) -> bool:
    r"""Signed 64-bit integer (:math:`-2^{63} \leq x < 2^{63}`)"""
    return -(2**63) <= value < 2**63


def int(value: builtins.int) -> bool:
    r"""Signed 64-bit integer (:math:`-2^{63} \leq x < 2^{63}`)"""
    return -(2**63) <= value < 2**63


try:
    from packaging import licenses as _licenses

    def SPDX(value: str) -> bool:
        """See :ref:`PyPA's License-Expression specification
        <pypa:core-metadata-license-expression>` (added in :pep:`639`).
        """
        try:
            _licenses.canonicalize_license_expression(value)
        except _licenses.InvalidLicenseExpression:
            return False
        return True

except ImportError:  # pragma: no cover
    _logger.warning(
        "Could not find an up-to-date installation of `packaging`. "
        "License expressions might not be validated. "
        "To enforce validation, please install `packaging>=24.2`."
    )

    def SPDX(value: str) -> bool:  # noqa: ARG001
        return True


VALID_IMPORT_NAME = re.compile(
    r"""
    ^                                  # start of string
        [A-Za-z_][A-Za-z_0-9]+         # a valid Python identifier
        (?:\.[A-Za-z_][A-Za-z_0-9]*)*  # optionally followed by .identifier's
    (?:\s*;\s*private)?                # optionally followed by ; private
    $                                  # end of string
    """,
    re.VERBOSE,
)


def import_name(value: str) -> bool:
    """This is a valid import name. It has to be series of python identifiers
    (not keywords), separated by dots, optionally followed by a semicolon and
    the keyword "private".
    """
    if VALID_IMPORT_NAME.match(value) is None:
        return False

    idents, _, _ = value.partition(";")
    return all(not keyword.iskeyword(ident) for ident in idents.rstrip().split("."))
<|endoftext|>from functools import reduce
from typing import Any, Callable, Dict

from . import formats
from .error_reporting import detailed_errors, ValidationError
from .extra_validations import EXTRA_VALIDATIONS
from .fastjsonschema_exceptions import JsonSchemaException, JsonSchemaValueException
from .fastjsonschema_validations import validate as _validate

__all__ = [
    "validate",
    "FORMAT_FUNCTIONS",
    "EXTRA_VALIDATIONS",
    "ValidationError",
    "JsonSchemaException",
    "JsonSchemaValueException",
]


FORMAT_FUNCTIONS: Dict[str, Callable[[str], bool]] = {
    fn.__name__.replace("_", "-"): fn
    for fn in formats.__dict__.values()
    if callable(fn) and not fn.__name__.startswith("_")
}


def validate(data: Any) -> bool:
    """Validate the given ``data`` object using JSON Schema
    This function raises ``ValidationError`` if ``data`` is invalid.
    """
    with detailed_errors():
        _validate(data, custom_formats=FORMAT_FUNCTIONS)
        reduce(lambda acc, fn: fn(acc), EXTRA_VALIDATIONS, data)
    return True
<|endoftext|># axis3d.py, original mplot3d version by John Porter
# Created: 23 Sep 2005
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>

import inspect

import numpy as np

import matplotlib as mpl
from matplotlib import (
    _api, artist, lines as mlines, axis as maxis, patches as mpatches,
    transforms as mtransforms, colors as mcolors)
from . import art3d, proj3d


def _move_from_center(coord, centers, deltas, axmask=(True, True, True)):
    """
    For each coordinate where *axmask* is True, move *coord* away from
    *centers* by *deltas*.
    """
    coord = np.asarray(coord)
    return coord + axmask * np.copysign(1, coord - centers) * deltas


def _tick_update_position(tick, tickxs, tickys, labelpos):
    """Update tick line and label position and style."""

    tick.label1.set_position(labelpos)
    tick.label2.set_position(labelpos)
    tick.tick1line.set_visible(True)
    tick.tick2line.set_visible(False)
    tick.tick1line.set_linestyle('-')
    tick.tick1line.set_marker('')
    tick.tick1line.set_data(tickxs, tickys)
    tick.gridline.set_data([0], [0])


class Axis(maxis.XAxis):
    """An Axis class for the 3D plots."""
    # These points from the unit cube make up the x, y and z-planes
    _PLANES = (
        (0, 3, 7, 4), (1, 2, 6, 5),  # yz planes
        (0, 1, 5, 4), (3, 2, 6, 7),  # xz planes
        (0, 1, 2, 3), (4, 5, 6, 7),  # xy planes
    )

    # Some properties for the axes
    _AXINFO = {
        'x': {'i': 0, 'tickdir': 1, 'juggled': (1, 0, 2)},
        'y': {'i': 1, 'tickdir': 0, 'juggled': (0, 1, 2)},
        'z': {'i': 2, 'tickdir': 0, 'juggled': (0, 2, 1)},
    }

    def _old_init(self, adir, v_intervalx, d_intervalx, axes, *args,
                  rotate_label=None, **kwargs):
        return locals()

    def _new_init(self, axes, *, rotate_label=None, **kwargs):
        return locals()

    def __init__(self, *args, **kwargs):
        params = _api.select_matching_signature(
            [self._old_init, self._new_init], *args, **kwargs)
        if "adir" in params:
            _api.warn_deprecated(
                "3.6", message=f"The signature of 3D Axis constructors has "
                f"changed in %(since)s; the new signature is "
                f"{inspect.signature(type(self).__init__)}", pending=True)
            if params["adir"] != self.axis_name:
                raise ValueError(f"Cannot instantiate {type(self).__name__} "
                                 f"with adir={params['adir']!r}")
        axes = params["axes"]
        rotate_label = params["rotate_label"]
        args = params.get("args", ())
        kwargs = params["kwargs"]

        name = self.axis_name

        self._label_position = 'default'
        self._tick_position = 'default'

        # This is a temporary member variable.
        # Do not depend on this existing in future releases!
        self._axinfo = self._AXINFO[name].copy()
        # Common parts
        self._axinfo.update({
            'label': {'va': 'center', 'ha': 'center',
                      'rotation_mode': 'anchor'},
            'color': mpl.rcParams[f'axes3d.{name}axis.panecolor'],
            'tick': {
                'inward_factor': 0.2,
                'outward_factor': 0.1,
            },
        })

        if mpl.rcParams['_internal.classic_mode']:
            self._axinfo.update({
                'axisline': {'linewidth': 0.75, 'color': (0, 0, 0, 1)},
                'grid': {
                    'color': (0.9, 0.9, 0.9, 1),
                    'linewidth': 1.0,
                    'linestyle': '-',
                },
            })
            self._axinfo['tick'].update({
                'linewidth': {
                    True: mpl.rcParams['lines.linewidth'],  # major
                    False: mpl.rcParams['lines.linewidth'],  # minor
                }
            })
        else:
            self._axinfo.update({
                'axisline': {
                    'linewidth': mpl.rcParams['axes.linewidth'],
                    'color': mpl.rcParams['axes.edgecolor'],
                },
                'grid': {
                    'color': mpl.rcParams['grid.color'],
                    'linewidth': mpl.rcParams['grid.linewidth'],
                    'linestyle': mpl.rcParams['grid.linestyle'],
                },
            })
            self._axinfo['tick'].update({
                'linewidth': {
                    True: (  # major
                        mpl.rcParams['xtick.major.width'] if name in 'xz'
                        else mpl.rcParams['ytick.major.width']),
                    False: (  # minor
                        mpl.rcParams['xtick.minor.width'] if name in 'xz'
                        else mpl.rcParams['ytick.minor.width']),
                }
            })

        super().__init__(axes, *args, **kwargs)

        # data and viewing intervals for this direction
        if "d_intervalx" in params:
            self.set_data_interval(*params["d_intervalx"])
        if "v_intervalx" in params:
            self.set_view_interval(*params["v_intervalx"])
        self.set_rotate_label(rotate_label)
        self._init3d()  # Inline after init3d deprecation elapses.

    __init__.__signature__ = inspect.signature(_new_init)
    adir = _api.deprecated("3.6", pending=True)(
        property(lambda self: self.axis_name))

    def _init3d(self):
        self.line = mlines.Line2D(
            xdata=(0, 0), ydata=(0, 0),
            linewidth=self._axinfo['axisline']['linewidth'],
            color=self._axinfo['axisline']['color'],
            antialiased=True)

        # Store dummy data in Polygon object
        self.pane = mpatches.Polygon([[0, 0], [0, 1]], closed=False)
        self.set_pane_color(self._axinfo['color'])

        self.axes._set_artist_props(self.line)
        self.axes._set_artist_props(self.pane)
        self.gridlines = art3d.Line3DCollection([])
        self.axes._set_artist_props(self.gridlines)
        self.axes._set_artist_props(self.label)
        self.axes._set_artist_props(self.offsetText)
        # Need to be able to place the label at the correct location
        self.label._transform = self.axes.transData
        self.offsetText._transform = self.axes.transData

    @_api.deprecated("3.6", pending=True)
    def init3d(self):  # After deprecation elapses, inline _init3d to __init__.
        self._init3d()

    def get_major_ticks(self, numticks=None):
        ticks = super().get_major_ticks(numticks)
        for t in ticks:
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    def get_minor_ticks(self, numticks=None):
        ticks = super().get_minor_ticks(numticks)
        for t in ticks:
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    def set_ticks_position(self, position):
        """
        Set the ticks position.

        Parameters
        ----------
        position : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the bolded axis lines, ticks, and tick labels.
        """
        _api.check_in_list(['lower', 'upper', 'both', 'default', 'none'],
                           position=position)
        self._tick_position = position

    def get_ticks_position(self):
        """
        Get the ticks position.

        Returns
        -------
        str : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the bolded axis lines, ticks, and tick labels.
        """
        return self._tick_position

    def set_label_position(self, position):
        """
        Set the label position.

        Parameters
        ----------
        position : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the axis label.
        """
        _api.check_in_list(['lower', 'upper', 'both', 'default', 'none'],
                           position=position)
        self._label_position = position

    def get_label_position(self):
        """
        Get the label position.

        Returns
        -------
        str : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the axis label.
        """
        return self._label_position

    def set_pane_color(self, color, alpha=None):
        """
        Set pane color.

        Parameters
        ----------
        color : :mpltype:`color`
            Color for axis pane.
        alpha : float, optional
            Alpha value for axis pane. If None, base it on *color*.
        """
        color = mcolors.to_rgba(color, alpha)
        self._axinfo['color'] = color
        self.pane.set_edgecolor(color)
        self.pane.set_facecolor(color)
        self.pane.set_alpha(color[-1])
        self.stale = True

    def set_rotate_label(self, val):
        """
        Whether to rotate the axis label: True, False or None.
        If set to None the label will be rotated if longer than 4 chars.
        """
        self._rotate_label = val
        self.stale = True

    def get_rotate_label(self, text):
        if self._rotate_label is not None:
            return self._rotate_label
        else:
            return len(text) > 4

    def _get_coord_info(self):
        mins, maxs = np.array([
            self.axes.get_xbound(),
            self.axes.get_ybound(),
            self.axes.get_zbound(),
        ]).T

        # Project the bounds along the current position of the cube:
        bounds = mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]
        bounds_proj = self.axes._transformed_cube(bounds)

        # Determine which one of the parallel planes are higher up:
        means_z0 = np.zeros(3)
        means_z1 = np.zeros(3)
        for i in range(3):
            means_z0[i] = np.mean(bounds_proj[self._PLANES[2 * i], 2])
            means_z1[i] = np.mean(bounds_proj[self._PLANES[2 * i + 1], 2])
        highs = means_z0 < means_z1

        # Special handling for edge-on views
        equals = np.abs(means_z0 - means_z1) <= np.finfo(float).eps
        if np.sum(equals) == 2:
            vertical = np.where(~equals)[0][0]
            if vertical == 2:  # looking at XY plane
                highs = np.array([True, True, highs[2]])
            elif vertical == 1:  # looking at XZ plane
                highs = np.array([True, highs[1], False])
            elif vertical == 0:  # looking at YZ plane
                highs = np.array([highs[0], False, False])

        return mins, maxs, bounds_proj, highs

    def _calc_centers_deltas(self, maxs, mins):
        centers = 0.5 * (maxs + mins)
        # In mpl3.8, the scale factor was 1/12. mpl3.9 changes this to
        # 1/12 * 24/25 = 0.08 to compensate for the change in automargin
        # behavior and keep appearance the same. The 24/25 factor is from the
        # 1/48 padding added to each side of the axis in mpl3.8.
        scale = 0.08
        deltas = (maxs - mins) * scale
        return centers, deltas

    def _get_axis_line_edge_points(self, minmax, maxmin, position=None):
        """Get the edge points for the black bolded axis line."""
        # When changing vertical axis some of the axes has to be
        # moved to the other plane so it looks the same as if the z-axis
        # was the vertical axis.
        mb = [minmax, maxmin]  # line from origin to nearest corner to camera
        mb_rev = mb[::-1]
        mm = [[mb, mb_rev, mb_rev], [mb_rev, mb_rev, mb], [mb, mb, mb]]
        mm = mm[self.axes._vertical_axis][self._axinfo["i"]]

        juggled = self._axinfo["juggled"]
        edge_point_0 = mm[0].copy()  # origin point

        if ((position == 'lower' and mm[1][juggled[-1]] < mm[0][juggled[-1]]) or
                (position == 'upper' and mm[1][juggled[-1]] > mm[0][juggled[-1]])):
            edge_point_0[juggled[-1]] = mm[1][juggled[-1]]
        else:
            edge_point_0[juggled[0]] = mm[1][juggled[0]]

        edge_point_1 = edge_point_0.copy()
        edge_point_1[juggled[1]] = mm[1][juggled[1]]

        return edge_point_0, edge_point_1

    def _get_all_axis_line_edge_points(self, minmax, maxmin, axis_position=None):
        # Determine edge points for the axis lines
        edgep1s = []
        edgep2s = []
        position = []
        if axis_position in (None, 'default'):
            edgep1, edgep2 = self._get_axis_line_edge_points(minmax, maxmin)
            edgep1s = [edgep1]
            edgep2s = [edgep2]
            position = ['default']
        else:
            edgep1_l, edgep2_l = self._get_axis_line_edge_points(minmax, maxmin,
                                                                 position='lower')
            edgep1_u, edgep2_u = self._get_axis_line_edge_points(minmax, maxmin,
                                                                 position='upper')
            if axis_position in ('lower', 'both'):
                edgep1s.append(edgep1_l)
                edgep2s.append(edgep2_l)
                position.append('lower')
            if axis_position in ('upper', 'both'):
                edgep1s.append(edgep1_u)
                edgep2s.append(edgep2_u)
                position.append('upper')
        return edgep1s, edgep2s, position

    def _get_tickdir(self, position):
        """
        Get the direction of the tick.

        Parameters
        ----------
        position : str, optional : {'upper', 'lower', 'default'}
            The position of the axis.

        Returns
        -------
        tickdir : int
            Index which indicates which coordinate the tick line will
            align with.
        """
        _api.check_in_list(('upper', 'lower', 'default'), position=position)

        # TODO: Move somewhere else where it's triggered less:
        tickdirs_base = [v["tickdir"] for v in self._AXINFO.values()]  # default
        elev_mod = np.mod(self.axes.elev + 180, 360) - 180
        azim_mod = np.mod(self.axes.azim, 360)
        if position == 'upper':
            if elev_mod >= 0:
                tickdirs_base = [2, 2, 0]
            else:
                tickdirs_base = [1, 0, 0]
            if 0 <= azim_mod < 180:
                tickdirs_base[2] = 1
        elif position == 'lower':
            if elev_mod >= 0:
                tickdirs_base = [1, 0, 1]
            else:
                tickdirs_base = [2, 2, 1]
            if 0 <= azim_mod < 180:
                tickdirs_base[2] = 0
        info_i = [v["i"] for v in self._AXINFO.values()]

        i = self._axinfo["i"]
        vert_ax = self.axes._vertical_axis
        j = vert_ax - 2
        # default: tickdir = [[1, 2, 1], [2, 2, 0], [1, 0, 0]][vert_ax][i]
        tickdir = np.roll(info_i, -j)[np.roll(tickdirs_base, j)][i]
        return tickdir

    def active_pane(self):
        mins, maxs, tc, highs = self._get_coord_info()
        info = self._axinfo
        index = info['i']
        if not highs[index]:
            loc = mins[index]
            plane = self._PLANES[2 * index]
        else:
            loc = maxs[index]
            plane = self._PLANES[2 * index + 1]
        xys = np.array([tc[p] for p in plane])
        return xys, loc

    def draw_pane(self, renderer):
        """
        Draw pane.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass
        """
        renderer.open_group('pane3d', gid=self.get_gid())
        xys, loc = self.active_pane()
        self.pane.xy = xys[:, :2]
        self.pane.draw(renderer)
        renderer.close_group('pane3d')

    def _axmask(self):
        axmask = [True, True, True]
        axmask[self._axinfo["i"]] = False
        return axmask

    def _draw_ticks(self, renderer, edgep1, centers, deltas, highs,
                    deltas_per_point, pos):
        ticks = self._update_ticks()
        info = self._axinfo
        index = info["i"]
        juggled = info["juggled"]

        mins, maxs, tc, highs = self._get_coord_info()
        centers, deltas = self._calc_centers_deltas(maxs, mins)

        # Draw ticks:
        tickdir = self._get_tickdir(pos)
        tickdelta = deltas[tickdir] if highs[tickdir] else -deltas[tickdir]

        tick_info = info['tick']
        tick_out = tick_info['outward_factor'] * tickdelta
        tick_in = tick_info['inward_factor'] * tickdelta
        tick_lw = tick_info['linewidth']
        edgep1_tickdir = edgep1[tickdir]
        out_tickdir = edgep1_tickdir + tick_out
        in_tickdir = edgep1_tickdir - tick_in

        default_label_offset = 8.  # A rough estimate
        points = deltas_per_point * deltas
        for tick in ticks:
            # Get tick line positions
            pos = edgep1.copy()
            pos[index] = tick.get_loc()
            pos[tickdir] = out_tickdir
            x1, y1, z1 = proj3d.proj_transform(*pos, self.axes.M)
            pos[tickdir] = in_tickdir
            x2, y2, z2 = proj3d.proj_transform(*pos, self.axes.M)

            # Get position of label
            labeldeltas = (tick.get_pad() + default_label_offset) * points

            pos[tickdir] = edgep1_tickdir
            pos = _move_from_center(pos, centers, labeldeltas, self._axmask())
            lx, ly, lz = proj3d.proj_transform(*pos, self.axes.M)

            _tick_update_position(tick, (x1, x2), (y1, y2), (lx, ly))
            tick.tick1line.set_linewidth(tick_lw[tick._major])
            tick.draw(renderer)

    def _draw_offset_text(self, renderer, edgep1, edgep2, labeldeltas, centers,
                          highs, pep, dx, dy):
        # Get general axis information:
        info = self._axinfo
        index = info["i"]
        juggled = info["juggled"]
        tickdir = info["tickdir"]

        # Which of the two edge points do we want to
        # use for locating the offset text?
        if juggled[2] == 2:
            outeredgep = edgep1
            outerindex = 0
        else:
            outeredgep = edgep2
            outerindex = 1

        pos = _move_from_center(outeredgep, centers, labeldeltas,
                                self._axmask())
        olx, oly, olz = proj3d.proj_transform(*pos, self.axes.M)
        self.offsetText.set_text(self.major.formatter.get_offset())
        self.offsetText.set_position((olx, oly))
        angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
        self.offsetText.set_rotation(angle)
        # Must set rotation mode to "anchor" so that
        # the alignment point is used as the "fulcrum" for rotation.
        self.offsetText.set_rotation_mode('anchor')

        # ----------------------------------------------------------------------
        # Note: the following statement for determining the proper alignment of
        # the offset text. This was determined entirely by trial-and-error
        # and should not be in any way considered as "the way".  There are
        # still some edge cases where alignment is not quite right, but this
        # seems to be more of a geometry issue (in other words, I might be
        # using the wrong reference points).
        #
        # (TT, FF, TF, FT) are the shorthand for the tuple of
        #   (centpt[tickdir] <= pep[tickdir, outerindex],
        #    centpt[index] <= pep[index, outerindex])
        #
        # Three-letters (e.g., TFT, FTT) are short-hand for the array of bools
        # from the variable 'highs'.
        # ---------------------------------------------------------------------
        centpt = proj3d.proj_transform(*centers, self.axes.M)
        if centpt[tickdir] > pep[tickdir, outerindex]:
            # if FT and if highs has an even number of Trues
            if (centpt[index] <= pep[index, outerindex]
                    and np.count_nonzero(highs) % 2 == 0):
                # Usually, this means align right, except for the FTT case,
                # in which offset for axis 1 and 2 are aligned left.
                if highs.tolist() == [False, True, True] and index in (1, 2):
                    align = 'left'
                else:
                    align = 'right'
            else:
                # The FF case
                align = 'left'
        else:
            # if TF and if highs has an even number of Trues
            if (centpt[index] > pep[index, outerindex]
                    and np.count_nonzero(highs) % 2 == 0):
                # Usually mean align left, except if it is axis 2
                align = 'right' if index == 2 else 'left'
            else:
                # The TT case
                align = 'right'

        self.offsetText.set_va('center')
        self.offsetText.set_ha(align)
        self.offsetText.draw(renderer)

    def _draw_labels(self, renderer, edgep1, edgep2, labeldeltas, centers, dx, dy):
        label = self._axinfo["label"]

        # Draw labels
        lxyz = 0.5 * (edgep1 + edgep2)
        lxyz = _move_from_center(lxyz, centers, labeldeltas, self._axmask())
        tlx, tly, tlz = proj3d.proj_transform(*lxyz, self.axes.M)
        self.label.set_position((tlx, tly))
        if self.get_rotate_label(self.label.get_text()):
            angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
            self.label.set_rotation(angle)
        self.label.set_va(label['va'])
        self.label.set_ha(label['ha'])
        self.label.set_rotation_mode(label['rotation_mode'])
        self.label.draw(renderer)

    @artist.allow_rasterization
    def draw(self, renderer):
        self.label._transform = self.axes.transData
        self.offsetText._transform = self.axes.transData
        renderer.open_group("axis3d", gid=self.get_gid())

        # Get general axis information:
        mins, maxs, tc, highs = self._get_coord_info()
        centers, deltas = self._calc_centers_deltas(maxs, mins)

        # Calculate offset distances
        # A rough estimate; points are ambiguous since 3D plots rotate
        reltoinches = self.get_figure(root=False).dpi_scale_trans.inverted()
        ax_inches = reltoinches.transform(self.axes.bbox.size)
        ax_points_estimate = sum(72. * ax_inches)
        deltas_per_point = 48 / ax_points_estimate
        default_offset = 21.
        labeldeltas = (self.labelpad + default_offset) * deltas_per_point * deltas

        # Determine edge points for the axis lines
        minmax = np.where(highs, maxs, mins)  # "origin" point
        maxmin = np.where(~highs, maxs, mins)  # "opposite" corner near camera

        for edgep1, edgep2, pos in zip(*self._get_all_axis_line_edge_points(
                                           minmax, maxmin, self._tick_position)):
            # Project the edge points along the current position
            pep = proj3d._proj_trans_points([edgep1, edgep2], self.axes.M)
            pep = np.asarray(pep)

            # The transAxes transform is used because the Text object
            # rotates the text relative to the display coordinate system.
            # Therefore, if we want the labels to remain parallel to the
            # axis regardless of the aspect ratio, we need to convert the
            # edge points of the plane to display coordinates and calculate
            # an angle from that.
            # TODO: Maybe Text objects should handle this themselves?
            dx, dy = (self.axes.transAxes.transform([pep[0:2, 1]]) -
                      self.axes.transAxes.transform([pep[0:2, 0]]))[0]

            # Draw the lines
            self.line.set_data(pep[0], pep[1])
            self.line.draw(renderer)

            # Draw ticks
            self._draw_ticks(renderer, edgep1, centers, deltas, highs,
                             deltas_per_point, pos)

            # Draw Offset text
            self._draw_offset_text(renderer, edgep1, edgep2, labeldeltas,
                                   centers, highs, pep, dx, dy)

        for edgep1, edgep2, pos in zip(*self._get_all_axis_line_edge_points(
                                           minmax, maxmin, self._label_position)):
            # See comments above
            pep = proj3d._proj_trans_points([edgep1, edgep2], self.axes.M)
            pep = np.asarray(pep)
            dx, dy = (self.axes.transAxes.transform([pep[0:2, 1]]) -
                      self.axes.transAxes.transform([pep[0:2, 0]]))[0]

            # Draw labels
            self._draw_labels(renderer, edgep1, edgep2, labeldeltas, centers, dx, dy)

        renderer.close_group('axis3d')
        self.stale = False

    @artist.allow_rasterization
    def draw_grid(self, renderer):
        if not self.axes._draw_grid:
            return

        renderer.open_group("grid3d", gid=self.get_gid())

        ticks = self._update_ticks()
        if len(ticks):
            # Get general axis information:
            info = self._axinfo
            index = info["i"]

            mins, maxs, tc, highs = self._get_coord_info()

            minmax = np.where(highs, maxs, mins)
            maxmin = np.where(~highs, maxs, mins)

            # Grid points where the planes meet
            xyz0 = np.tile(minmax, (len(ticks), 1))
            xyz0[:, index] = [tick.get_loc() for tick in ticks]

            # Grid lines go from the end of one plane through the plane
            # intersection (at xyz0) to the end of the other plane.  The first
            # point (0) differs along dimension index-2 and the last (2) along
            # dimension index-1.
            lines = np.stack([xyz0, xyz0, xyz0], axis=1)
            lines[:, 0, index - 2] = maxmin[index - 2]
            lines[:, 2, index - 1] = maxmin[index - 1]
            self.gridlines.set_segments(lines)
            gridinfo = info['grid']
            self.gridlines.set_color(gridinfo['color'])
            self.gridlines.set_linewidth(gridinfo['linewidth'])
            self.gridlines.set_linestyle(gridinfo['linestyle'])
            self.gridlines.do_3d_projection()
            self.gridlines.draw(renderer)

        renderer.close_group('grid3d')

    # TODO: Get this to work (more) properly when mplot3d supports the
    #       transforms framework.
    def get_tightbbox(self, renderer=None, *, for_layout_only=False):
        # docstring inherited
        if not self.get_visible():
            return
        # We have to directly access the internal data structures
        # (and hope they are up to date) because at draw time we
        # shift the ticks and their labels around in (x, y) space
        # based on the projection, the current view port, and their
        # position in 3D space. If we extend the transforms framework
        # into 3D we would not need to do this different book keeping
        # than we do in the normal axis
        major_locs = self.get_majorticklocs()
        minor_locs = self.get_minorticklocs()

        ticks = [*self.get_minor_ticks(len(minor_locs)),
                 *self.get_major_ticks(len(major_locs))]
        view_low, view_high = self.get_view_interval()
        if view_low > view_high:
            view_low, view_high = view_high, view_low
        interval_t = self.get_transform().transform([view_low, view_high])

        ticks_to_draw = []
        for tick in ticks:
            try:
                loc_t = self.get_transform().transform(tick.get_loc())
            except AssertionError:
                # Transform.transform doesn't allow masked values but
                # some scales might make them, so we need this try/except.
                pass
            else:
                if mtransforms._interval_contains_close(interval_t, loc_t):
                    ticks_to_draw.append(tick)

        ticks = ticks_to_draw

        bb_1, bb_2 = self._get_ticklabel_bboxes(ticks, renderer)
        other = []

        if self.line.get_visible():
            other.append(self.line.get_window_extent(renderer))
        if (self.label.get_visible() and not for_layout_only and
                self.label.get_text()):
            other.append(self.label.get_window_extent(renderer))

        return mtransforms.Bbox.union([*bb_1, *bb_2, *other])

    d_interval = _api.deprecated(
        "3.6", alternative="get_data_interval", pending=True)(
            property(lambda self: self.get_data_interval(),
                     lambda self, minmax: self.set_data_interval(*minmax)))
    v_interval = _api.deprecated(
        "3.6", alternative="get_view_interval", pending=True)(
            property(lambda self: self.get_view_interval(),
                     lambda self, minmax: self.set_view_interval(*minmax)))


class XAxis(Axis):
    axis_name = "x"
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervalx")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervalx")


class YAxis(Axis):
    axis_name = "y"
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervaly")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervaly")


class ZAxis(Axis):
    axis_name = "z"
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "zz_viewLim", "intervalx")
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "zz_dataLim", "intervalx")
<|endoftext|># art3d.py, original mplot3d version by John Porter
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>
# Minor additions by Ben Axelrod <baxelrod@coroware.com>

"""
Module containing 3D artist code and functions to convert 2D
artists into 3D versions which can be added to an Axes3D.
"""

import math

import numpy as np

from contextlib import contextmanager

from matplotlib import (
    _api, artist, cbook, colors as mcolors, lines, text as mtext,
    path as mpath)
from matplotlib.collections import (
    Collection, LineCollection, PolyCollection, PatchCollection, PathCollection)
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d


def _norm_angle(a):
    """Return the given angle normalized to -180 < *a* <= 180 degrees."""
    a = (a + 360) % 360
    if a > 180:
        a = a - 360
    return a


def _norm_text_angle(a):
    """Return the given angle normalized to -90 < *a* <= 90 degrees."""
    a = (a + 180) % 180
    if a > 90:
        a = a - 180
    return a


def get_dir_vector(zdir):
    """
    Return a direction vector.

    Parameters
    ----------
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction. Possible values are:

        - 'x': equivalent to (1, 0, 0)
        - 'y': equivalent to (0, 1, 0)
        - 'z': equivalent to (0, 0, 1)
        - *None*: equivalent to (0, 0, 0)
        - an iterable (x, y, z) is converted to an array

    Returns
    -------
    x, y, z : array
        The direction vector.
    """
    if cbook._str_equal(zdir, 'x'):
        return np.array((1, 0, 0))
    elif cbook._str_equal(zdir, 'y'):
        return np.array((0, 1, 0))
    elif cbook._str_equal(zdir, 'z'):
        return np.array((0, 0, 1))
    elif zdir is None:
        return np.array((0, 0, 0))
    elif np.iterable(zdir) and len(zdir) == 3:
        return np.array(zdir)
    else:
        raise ValueError("'x', 'y', 'z', None or vector of length 3 expected")


def _viewlim_mask(xs, ys, zs, axes):
    """
    Return original points with points outside the axes view limits masked.

    Parameters
    ----------
    xs, ys, zs : array-like
        The points to mask.
    axes : Axes3D
        The axes to use for the view limits.

    Returns
    -------
    xs_masked, ys_masked, zs_masked : np.ma.array
        The masked points.
    """
    mask = np.logical_or.reduce((xs < axes.xy_viewLim.xmin,
                                 xs > axes.xy_viewLim.xmax,
                                 ys < axes.xy_viewLim.ymin,
                                 ys > axes.xy_viewLim.ymax,
                                 zs < axes.zz_viewLim.xmin,
                                 zs > axes.zz_viewLim.xmax))
    xs_masked = np.ma.array(xs, mask=mask)
    ys_masked = np.ma.array(ys, mask=mask)
    zs_masked = np.ma.array(zs, mask=mask)
    return xs_masked, ys_masked, zs_masked


class Text3D(mtext.Text):
    """
    Text object with 3D position and direction.

    Parameters
    ----------
    x, y, z : float
        The position of the text.
    text : str
        The text string to display.
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction of the text. See `.get_dir_vector` for a description of
        the values.
    axlim_clip : bool, default: False
        Whether to hide text outside the axes view limits.

    Other Parameters
    ----------------
    **kwargs
         All other parameters are passed on to `~matplotlib.text.Text`.
    """

    def __init__(self, x=0, y=0, z=0, text='', zdir='z', axlim_clip=False,
                 **kwargs):
        mtext.Text.__init__(self, x, y, text, **kwargs)
        self.set_3d_properties(z, zdir, axlim_clip)

    def get_position_3d(self):
        """Return the (x, y, z) position of the text."""
        return self._x, self._y, self._z

    def set_position_3d(self, xyz, zdir=None):
        """
        Set the (*x*, *y*, *z*) position of the text.

        Parameters
        ----------
        xyz : (float, float, float)
            The position in 3D space.
        zdir : {'x', 'y', 'z', None, 3-tuple}
            The direction of the text. If unspecified, the *zdir* will not be
            changed. See `.get_dir_vector` for a description of the values.
        """
        super().set_position(xyz[:2])
        self.set_z(xyz[2])
        if zdir is not None:
            self._dir_vec = get_dir_vector(zdir)

    def set_z(self, z):
        """
        Set the *z* position of the text.

        Parameters
        ----------
        z : float
        """
        self._z = z
        self.stale = True

    def set_3d_properties(self, z=0, zdir='z', axlim_clip=False):
        """
        Set the *z* position and direction of the text.

        Parameters
        ----------
        z : float
            The z-position in 3D space.
        zdir : {'x', 'y', 'z', 3-tuple}
            The direction of the text. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide text outside the axes view limits.
        """
        self._z = z
        self._dir_vec = get_dir_vector(zdir)
        self._axlim_clip = axlim_clip
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(self._x, self._y, self._z, self.axes)
            position3d = np.ma.row_stack((xs, ys, zs)).ravel().filled(np.nan)
        else:
            xs, ys, zs = self._x, self._y, self._z
            position3d = np.asanyarray([xs, ys, zs])

        proj = proj3d._proj_trans_points(
            [position3d, position3d + self._dir_vec], self.axes.M)
        dx = proj[0][1] - proj[0][0]
        dy = proj[1][1] - proj[1][0]
        angle = math.degrees(math.atan2(dy, dx))
        with cbook._setattr_cm(self, _x=proj[0][0], _y=proj[1][0],
                               _rotation=_norm_text_angle(angle)):
            mtext.Text.draw(self, renderer)
        self.stale = False

    def get_tightbbox(self, renderer=None):
        # Overwriting the 2d Text behavior which is not valid for 3d.
        # For now, just return None to exclude from layout calculation.
        return None


def text_2d_to_3d(obj, z=0, zdir='z', axlim_clip=False):
    """
    Convert a `.Text` to a `.Text3D` object.

    Parameters
    ----------
    z : float
        The z-position in 3D space.
    zdir : {'x', 'y', 'z', 3-tuple}
        The direction of the text. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    axlim_clip : bool, default: False
        Whether to hide text outside the axes view limits.
    """
    obj.__class__ = Text3D
    obj.set_3d_properties(z, zdir, axlim_clip)


class Line3D(lines.Line2D):
    """
    3D line object.

    .. note:: Use `get_data_3d` to obtain the data associated with the line.
            `~.Line2D.get_data`, `~.Line2D.get_xdata`, and `~.Line2D.get_ydata` return
            the x- and y-coordinates of the projected 2D-line, not the x- and y-data of
            the 3D-line. Similarly, use `set_data_3d` to set the data, not
            `~.Line2D.set_data`, `~.Line2D.set_xdata`, and `~.Line2D.set_ydata`.
    """

    def __init__(self, xs, ys, zs, *args, axlim_clip=False, **kwargs):
        """

        Parameters
        ----------
        xs : array-like
            The x-data to be plotted.
        ys : array-like
            The y-data to be plotted.
        zs : array-like
            The z-data to be plotted.
        *args, **kwargs
            Additional arguments are passed to `~matplotlib.lines.Line2D`.
        """
        super().__init__([], [], *args, **kwargs)
        self.set_data_3d(xs, ys, zs)
        self._axlim_clip = axlim_clip

    def set_3d_properties(self, zs=0, zdir='z', axlim_clip=False):
        """
        Set the *z* position and direction of the line.

        Parameters
        ----------
        zs : float or array of floats
            The location along the *zdir* axis in 3D space to position the
            line.
        zdir : {'x', 'y', 'z'}
            Plane to plot line orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide lines with an endpoint outside the axes view limits.
        """
        xs = self.get_xdata()
        ys = self.get_ydata()
        zs = cbook._to_unmasked_float_array(zs).ravel()
        zs = np.broadcast_to(zs, len(xs))
        self._verts3d = juggle_axes(xs, ys, zs, zdir)
        self._axlim_clip = axlim_clip
        self.stale = True

    def set_data_3d(self, *args):
        """
        Set the x, y and z data

        Parameters
        ----------
        x : array-like
            The x-data to be plotted.
        y : array-like
            The y-data to be plotted.
        z : array-like
            The z-data to be plotted.

        Notes
        -----
        Accepts x, y, z arguments or a single array-like (x, y, z)
        """
        if len(args) == 1:
            args = args[0]
        for name, xyz in zip('xyz', args):
            if not np.iterable(xyz):
                raise RuntimeError(f'{name} must be a sequence')
        self._verts3d = args
        self.stale = True

    def get_data_3d(self):
        """
        Get the current data

        Returns
        -------
        verts3d : length-3 tuple or array-like
            The current data as a tuple or array-like.
        """
        return self._verts3d

    @artist.allow_rasterization
    def draw(self, renderer):
        if self._axlim_clip:
            xs3d, ys3d, zs3d = _viewlim_mask(*self._verts3d, self.axes)
        else:
            xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs, tis = proj3d._proj_transform_clip(xs3d, ys3d, zs3d,
                                                      self.axes.M,
                                                      self.axes._focal_length)
        self.set_data(xs, ys)
        super().draw(renderer)
        self.stale = False


def line_2d_to_3d(line, zs=0, zdir='z', axlim_clip=False):
    """
    Convert a `.Line2D` to a `.Line3D` object.

    Parameters
    ----------
    zs : float
        The location along the *zdir* axis in 3D space to position the line.
    zdir : {'x', 'y', 'z'}
        Plane to plot line orthogonal to. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    axlim_clip : bool, default: False
        Whether to hide lines with an endpoint outside the axes view limits.
    """

    line.__class__ = Line3D
    line.set_3d_properties(zs, zdir, axlim_clip)


def _path_to_3d_segment(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment."""

    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg = [(x, y, z) for (((x, y), code), z) in zip(pathsegs, zs)]
    seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    return seg3d


def _paths_to_3d_segments(paths, zs=0, zdir='z'):
    """Convert paths from a collection object to 3D segments."""

    if not np.iterable(zs):
        zs = np.broadcast_to(zs, len(paths))
    else:
        if len(zs) != len(paths):
            raise ValueError('Number of z-coordinates does not match paths.')

    segs = [_path_to_3d_segment(path, pathz, zdir)
            for path, pathz in zip(paths, zs)]
    return segs


def _path_to_3d_segment_with_codes(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment with path codes."""

    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg_codes = [((x, y, z), code) for ((x, y), code), z in zip(pathsegs, zs)]
    if seg_codes:
        seg, codes = zip(*seg_codes)
        seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    else:
        seg3d = []
        codes = []
    return seg3d, list(codes)


def _paths_to_3d_segments_with_codes(paths, zs=0, zdir='z'):
    """
    Convert paths from a collection object to 3D segments with path codes.
    """

    zs = np.broadcast_to(zs, len(paths))
    segments_codes = [_path_to_3d_segment_with_codes(path, pathz, zdir)
                      for path, pathz in zip(paths, zs)]
    if segments_codes:
        segments, codes = zip(*segments_codes)
    else:
        segments, codes = [], []
    return list(segments), list(codes)


class Collection3D(Collection):
    """A collection of 3D paths."""

    def do_3d_projection(self):
        """Project the points according to renderer matrix."""
        vs_list = [vs for vs, _ in self._3dverts_codes]
        if self._axlim_clip:
            vs_list = [np.ma.row_stack(_viewlim_mask(*vs.T, self.axes)).T
                       for vs in vs_list]
        xyzs_list = [proj3d.proj_transform(*vs.T, self.axes.M) for vs in vs_list]
        self._paths = [mpath.Path(np.ma.column_stack([xs, ys]), cs)
                       for (xs, ys, _), (_, cs) in zip(xyzs_list, self._3dverts_codes)]
        zs = np.concatenate([zs for _, _, zs in xyzs_list])
        return zs.min() if len(zs) else 1e9


def collection_2d_to_3d(col, zs=0, zdir='z', axlim_clip=False):
    """Convert a `.Collection` to a `.Collection3D` object."""
    zs = np.broadcast_to(zs, len(col.get_paths()))
    col._3dverts_codes = [
        (np.column_stack(juggle_axes(
            *np.column_stack([p.vertices, np.broadcast_to(z, len(p.vertices))]).T,
            zdir)),
         p.codes)
        for p, z in zip(col.get_paths(), zs)]
    col.__class__ = cbook._make_class_factory(Collection3D, "{}3D")(type(col))
    col._axlim_clip = axlim_clip


class Line3DCollection(LineCollection):
    """
    A collection of 3D lines.
    """
    def __init__(self, lines, axlim_clip=False, **kwargs):
        super().__init__(lines, **kwargs)
        self._axlim_clip = axlim_clip

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_segments(self, segments):
        """
        Set 3D segments.
        """
        self._segments3d = segments
        super().set_segments([])

    def do_3d_projection(self):
        """
        Project the points according to renderer matrix.
        """
        segments = self._segments3d
        if self._axlim_clip:
            all_points = np.ma.vstack(segments)
            masked_points = np.ma.column_stack([*_viewlim_mask(*all_points.T,
                                                               self.axes)])
            segment_lengths = [np.shape(segment)[0] for segment in segments]
            segments = np.split(masked_points, np.cumsum(segment_lengths[:-1]))
        xyslist = [proj3d._proj_trans_points(points, self.axes.M)
                   for points in segments]
        segments_2d = [np.ma.column_stack([xs, ys]) for xs, ys, zs in xyslist]
        LineCollection.set_segments(self, segments_2d)

        # FIXME
        minz = 1e9
        for xs, ys, zs in xyslist:
            minz = min(minz, min(zs))
        return minz


def line_collection_2d_to_3d(col, zs=0, zdir='z', axlim_clip=False):
    """Convert a `.LineCollection` to a `.Line3DCollection` object."""
    segments3d = _paths_to_3d_segments(col.get_paths(), zs, zdir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)
    col._axlim_clip = axlim_clip


class Patch3D(Patch):
    """
    3D patch object.
    """

    def __init__(self, *args, zs=(), zdir='z', axlim_clip=False, **kwargs):
        """
        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide patches with a vertex outside the axes view limits.
        """
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir, axlim_clip)

    def set_3d_properties(self, verts, zs=0, zdir='z', axlim_clip=False):
        """
        Set the *z* position and direction of the patch.

        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide patches with a vertex outside the axes view limits.
        """
        zs = np.broadcast_to(zs, len(verts))
        self._segment3d = [juggle_axes(x, y, z, zdir)
                           for ((x, y), z) in zip(verts, zs)]
        self._axlim_clip = axlim_clip

    def get_path(self):
        # docstring inherited
        # self._path2d is not initialized until do_3d_projection
        if not hasattr(self, '_path2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return self._path2d

    def do_3d_projection(self):
        s = self._segment3d
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(*zip(*s), self.axes)
        else:
            xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d._proj_transform_clip(xs, ys, zs,
                                                         self.axes.M,
                                                         self.axes._focal_length)
        self._path2d = mpath.Path(np.ma.column_stack([vxs, vys]))
        return min(vzs)


class PathPatch3D(Patch3D):
    """
    3D PathPatch object.
    """

    def __init__(self, path, *, zs=(), zdir='z', axlim_clip=False, **kwargs):
        """
        Parameters
        ----------
        path :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            path patch.
        zdir : {'x', 'y', 'z', 3-tuple}
            Plane to plot path patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide path patches with a point outside the axes view limits.
        """
        # Not super().__init__!
        Patch.__init__(self, **kwargs)
        self.set_3d_properties(path, zs, zdir, axlim_clip)

    def set_3d_properties(self, path, zs=0, zdir='z', axlim_clip=False):
        """
        Set the *z* position and direction of the path patch.

        Parameters
        ----------
        path :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            path patch.
        zdir : {'x', 'y', 'z', 3-tuple}
            Plane to plot path patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide path patches with a point outside the axes view limits.
        """
        Patch3D.set_3d_properties(self, path.vertices, zs=zs, zdir=zdir,
                                  axlim_clip=axlim_clip)
        self._code3d = path.codes

    def do_3d_projection(self):
        s = self._segment3d
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(*zip(*s), self.axes)
        else:
            xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d._proj_transform_clip(xs, ys, zs,
                                                         self.axes.M,
                                                         self.axes._focal_length)
        self._path2d = mpath.Path(np.ma.column_stack([vxs, vys]), self._code3d)
        return min(vzs)


def _get_patch_verts(patch):
    """Return a list of vertices for the path of a patch."""
    trans = patch.get_patch_transform()
    path = patch.get_path()
    polygons = path.to_polygons(trans)
    return polygons[0] if len(polygons) else np.array([])


def patch_2d_to_3d(patch, z=0, zdir='z', axlim_clip=False):
    """Convert a `.Patch` to a `.Patch3D` object."""
    verts = _get_patch_verts(patch)
    patch.__class__ = Patch3D
    patch.set_3d_properties(verts, z, zdir, axlim_clip)


def pathpatch_2d_to_3d(pathpatch, z=0, zdir='z'):
    """Convert a `.PathPatch` to a `.PathPatch3D` object."""
    path = pathpatch.get_path()
    trans = pathpatch.get_patch_transform()

    mpath = trans.transform_path(path)
    pathpatch.__class__ = PathPatch3D
    pathpatch.set_3d_properties(mpath, z, zdir)


class Patch3DCollection(PatchCollection):
    """
    A collection of 3D patches.
    """

    def __init__(self, *args,
                 zs=0, zdir='z', depthshade=True, axlim_clip=False, **kwargs):
        """
        Create a collection of flat 3D patches with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of patches in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PatchCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument *depthshade* is available to indicate
        whether to shade the patches in order to give the appearance of depth
        (default is *True*). This is typically desired in scatter plots.
        """
        self._depthshade = depthshade
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir, axlim_clip)

    def get_depthshade(self):
        return self._depthshade

    def set_depthshade(self, depthshade):
        """
        Set whether depth shading is performed on collection members.

        Parameters
        ----------
        depthshade : bool
            Whether to shade the patches in order to give the appearance of
            depth.
        """
        self._depthshade = depthshade
        self.stale = True

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir, axlim_clip=False):
        """
        Set the *z* positions and direction of the patches.

        Parameters
        ----------
        zs : float or array of floats
            The location or locations to place the patches in the collection
            along the *zdir* axis.
        zdir : {'x', 'y', 'z'}
            Plane to plot patches orthogonal to.
            All patches must have the same direction.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide patches with a vertex outside the axes view limits.
        """
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._z_markers_idx = slice(-1)
        self._vzs = None
        self._axlim_clip = axlim_clip
        self.stale = True

    def do_3d_projection(self):
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(*self._offsets3d, self.axes)
        else:
            xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d._proj_transform_clip(xs, ys, zs,
                                                         self.axes.M,
                                                         self.axes._focal_length)
        self._vzs = vzs
        super().set_offsets(np.ma.column_stack([vxs, vys]))

        if vzs.size > 0:
            return min(vzs)
        else:
            return np.nan

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        color_array = (
            _zalpha(color_array, self._vzs)
            if self._vzs is not None and self._depthshade
            else color_array
        )
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        # We need this check here to make sure we do not double-apply the depth
        # based alpha shading when the edge color is "face" which means the
        # edge colour should be identical to the face colour.
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())


class Path3DCollection(PathCollection):
    """
    A collection of 3D paths.
    """

    def __init__(self, *args,
                 zs=0, zdir='z', depthshade=True, axlim_clip=False, **kwargs):
        """
        Create a collection of flat 3D paths with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of paths in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PathCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument *depthshade* is available to indicate
        whether to shade the patches in order to give the appearance of depth
        (default is *True*). This is typically desired in scatter plots.
        """
        self._depthshade = depthshade
        self._in_draw = False
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir, axlim_clip)
        self._offset_zordered = None

    def draw(self, renderer):
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                super().draw(renderer)

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir, axlim_clip=False):
        """
        Set the *z* positions and direction of the paths.

        Parameters
        ----------
        zs : float or array of floats
            The location or locations to place the paths in the collection
            along the *zdir* axis.
        zdir : {'x', 'y', 'z'}
            Plane to plot paths orthogonal to.
            All paths must have the same direction.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide paths with a vertex outside the axes view limits.
        """
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        self._zdir = zdir
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        # In the base draw methods we access the attributes directly which
        # means we cannot resolve the shuffling in the getter methods like
        # we do for the edge and face colors.
        #
        # This means we need to carry around a cache of the unsorted sizes and
        # widths (postfixed with 3d) and in `do_3d_projection` set the
        # depth-sorted version of that data into the private state used by the
        # base collection class in its draw method.
        #
        # Grab the current sizes and linewidths to preserve them.
        self._sizes3d = self._sizes
        self._linewidths3d = np.array(self._linewidths)
        xs, ys, zs = self._offsets3d

        # Sort the points based on z coordinates
        # Performance optimization: Create a sorted index array and reorder
        # points and point properties according to the index array
        self._z_markers_idx = slice(-1)
        self._vzs = None

        self._axlim_clip = axlim_clip
        self.stale = True

    def set_sizes(self, sizes, dpi=72.0):
        super().set_sizes(sizes, dpi)
        if not self._in_draw:
            self._sizes3d = sizes

    def set_linewidth(self, lw):
        super().set_linewidth(lw)
        if not self._in_draw:
            self._linewidths3d = np.array(self._linewidths)

    def get_depthshade(self):
        return self._depthshade

    def set_depthshade(self, depthshade):
        """
        Set whether depth shading is performed on collection members.

        Parameters
        ----------
        depthshade : bool
            Whether to shade the patches in order to give the appearance of
            depth.
        """
        self._depthshade = depthshade
        self.stale = True

    def do_3d_projection(self):
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(*self._offsets3d, self.axes)
        else:
            xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d._proj_transform_clip(xs, ys, zs,
                                                         self.axes.M,
                                                         self.axes._focal_length)
        # Sort the points based on z coordinates
        # Performance optimization: Create a sorted index array and reorder
        # points and point properties according to the index array
        z_markers_idx = self._z_markers_idx = np.ma.argsort(vzs)[::-1]
        self._vzs = vzs

        # we have to special case the sizes because of code in collections.py
        # as the draw method does
        #      self.set_sizes(self._sizes, self.figure.dpi)
        # so we cannot rely on doing the sorting on the way out via get_*

        if len(self._sizes3d) > 1:
            self._sizes = self._sizes3d[z_markers_idx]

        if len(self._linewidths3d) > 1:
            self._linewidths = self._linewidths3d[z_markers_idx]

        PathCollection.set_offsets(self, np.ma.column_stack((vxs, vys)))

        # Re-order items
        vzs = vzs[z_markers_idx]
        vxs = vxs[z_markers_idx]
        vys = vys[z_markers_idx]

        # Store ordered offset for drawing purpose
        self._offset_zordered = np.ma.column_stack((vxs, vys))

        return np.min(vzs) if vzs.size else np.nan

    @contextmanager
    def _use_zordered_offset(self):
        if self._offset_zordered is None:
            # Do nothing
            yield
        else:
            # Swap offset with z-ordered offset
            old_offset = self._offsets
            super().set_offsets(self._offset_zordered)
            try:
                yield
            finally:
                self._offsets = old_offset

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        color_array = (
            _zalpha(color_array, self._vzs)
            if self._vzs is not None and self._depthshade
            else color_array
        )
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        # We need this check here to make sure we do not double-apply the depth
        # based alpha shading when the edge color is "face" which means the
        # edge colour should be identical to the face colour.
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())


def patch_collection_2d_to_3d(col, zs=0, zdir='z', depthshade=True, axlim_clip=False):
    """
    Convert a `.PatchCollection` into a `.Patch3DCollection` object
    (or a `.PathCollection` into a `.Path3DCollection` object).

    Parameters
    ----------
    col : `~matplotlib.collections.PatchCollection` or \
`~matplotlib.collections.PathCollection`
        The collection to convert.
    zs : float or array of floats
        The location or locations to place the patches in the collection along
        the *zdir* axis. Default: 0.
    zdir : {'x', 'y', 'z'}
        The axis in which to place the patches. Default: "z".
        See `.get_dir_vector` for a description of the values.
    depthshade : bool, default: True
        Whether to shade the patches to give a sense of depth.
    axlim_clip : bool, default: False
        Whether to hide patches with a vertex outside the axes view limits.
    """
    if isinstance(col, PathCollection):
        col.__class__ = Path3DCollection
        col._offset_zordered = None
    elif isinstance(col, PatchCollection):
        col.__class__ = Patch3DCollection
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir, axlim_clip)


class Poly3DCollection(PolyCollection):
    """
    A collection of 3D polygons.

    .. note::
        **Filling of 3D polygons**

        There is no simple definition of the enclosed surface of a 3D polygon
        unless the polygon is planar.

        In practice, Matplotlib fills the 2D projection of the polygon. This
        gives a correct filling appearance only for planar polygons. For all
        other polygons, you'll find orientations in which the edges of the
        polygon intersect in the projection. This will lead to an incorrect
        visualization of the 3D area.

        If you need filled areas, it is recommended to create them via
        `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
        triangulation and thus generates consistent surfaces.
    """

    def __init__(self, verts, *args, zsort='average', shade=False,
                 lightsource=None, axlim_clip=False, **kwargs):
        """
        Parameters
        ----------
        verts : list of (N, 3) array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (N, 3).
        zsort : {'average', 'min', 'max'}, default: 'average'
            The calculation method for the z-order.
            See `~.Poly3DCollection.set_zsort` for details.
        shade : bool, default: False
            Whether to shade *facecolors* and *edgecolors*. When activating
            *shade*, *facecolors* and/or *edgecolors* must be provided.

            .. versionadded:: 3.7

        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.

            .. versionadded:: 3.7

        axlim_clip : bool, default: False
            Whether to hide polygons with a vertex outside the view limits.

        *args, **kwargs
            All other parameters are forwarded to `.PolyCollection`.

        Notes
        -----
        Note that this class does a bit of magic with the _facecolors
        and _edgecolors properties.
        """
        if shade:
            normals = _generate_normals(verts)
            facecolors = kwargs.get('facecolors', None)
            if facecolors is not None:
                kwargs['facecolors'] = _shade_colors(
                    facecolors, normals, lightsource
                )

            edgecolors = kwargs.get('edgecolors', None)
            if edgecolors is not None:
                kwargs['edgecolors'] = _shade_colors(
                    edgecolors, normals, lightsource
                )
            if facecolors is None and edgecolors is None:
                raise ValueError(
                    "You must provide facecolors, edgecolors, or both for "
                    "shade to work.")
        super().__init__(verts, *args, **kwargs)
        if isinstance(verts, np.ndarray):
            if verts.ndim != 3:
                raise ValueError('verts must be a list of (N, 3) array-like')
        else:
            if any(len(np.shape(vert)) != 2 for vert in verts):
                raise ValueError('verts must be a list of (N, 3) array-like')
        self.set_zsort(zsort)
        self._codes3d = None
        self._axlim_clip = axlim_clip

    _zsort_functions = {
        'average': np.average,
        'min': np.min,
        'max': np.max,
    }

    def set_zsort(self, zsort):
        """
        Set the calculation method for the z-order.

        Parameters
        ----------
        zsort : {'average', 'min', 'max'}
            The function applied on the z-coordinates of the vertices in the
            viewer's coordinate system, to determine the z-order.
        """
        self._zsortfunc = self._zsort_functions[zsort]
        self._sort_zpos = None
        self.stale = True

    @_api.deprecated("3.10")
    def get_vector(self, segments3d):
        return self._get_vector(segments3d)

    def _get_vector(self, segments3d):
        """Optimize points for projection."""
        if len(segments3d):
            xs, ys, zs = np.vstack(segments3d).T
        else:  # vstack can't stack zero arrays.
            xs, ys, zs = [], [], []
        ones = np.ones(len(xs))
        self._vec = np.array([xs, ys, zs, ones])

        indices = [0, *np.cumsum([len(segment) for segment in segments3d])]
        self._segslices = [*map(slice, indices[:-1], indices[1:])]

    def set_verts(self, verts, closed=True):
        """
        Set 3D vertices.

        Parameters
        ----------
        verts : list of (N, 3) array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (N, 3).
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        """
        self._get_vector(verts)
        # 2D verts will be updated at draw time
        super().set_verts([], False)
        self._closed = closed

    def set_verts_and_codes(self, verts, codes):
        """Set 3D vertices with path codes."""
        # set vertices with closed=False to prevent PolyCollection from
        # setting path codes
        self.set_verts(verts, closed=False)
        # and set our own codes instead.
        self._codes3d = codes

    def set_3d_properties(self, axlim_clip=False):
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        self._sort_zpos = None
        self.set_zsort('average')
        self._facecolor3d = PolyCollection.get_facecolor(self)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)
        self._alpha3d = PolyCollection.get_alpha(self)
        self.stale = True

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def do_3d_projection(self):
        """
        Perform the 3D projection for this object.
        """
        if self._A is not None:
            # force update of color mapping because we re-order them
            # below.  If we do not do this here, the 2D draw will call
            # this, but we will never port the color mapped values back
            # to the 3D versions.
            #
            # We hold the 3D versions in a fixed order (the order the user
            # passed in) and sort the 2D version by view depth.
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors
        if self._axlim_clip:
            xs, ys, zs = _viewlim_mask(*self._vec[0:3], self.axes)
            if self._vec.shape[0] == 4:  # Will be 3 (xyz) or 4 (xyzw)
                w_masked = np.ma.masked_where(zs.mask, self._vec[3])
                vec = np.ma.array([xs, ys, zs, w_masked])
            else:
                vec = np.ma.array([xs, ys, zs])
        else:
            vec = self._vec
        txs, tys, tzs = proj3d._proj_transform_vec(vec, self.axes.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]

        # This extra fuss is to re-order face / edge colors
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)

        if xyzlist:
            # sort by depth (furthest drawn first)
            z_segments_2d = sorted(
                ((self._zsortfunc(zs.data), np.ma.column_stack([xs, ys]), fc, ec, idx)
                 for idx, ((xs, ys, zs), fc, ec)
                 in enumerate(zip(xyzlist, cface, cedge))),
                key=lambda x: x[0], reverse=True)

            _, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = \
                zip(*z_segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []

        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)

        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d

        # Return zorder value
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            # FIXME: Some results still don't look quite right.
            #        In particular, examine contourf3d_demo2.py
            #        with az = -54 and elev = -45.
            return np.min(tzs)
        else:
            return np.nan

    def set_facecolor(self, colors):
        # docstring inherited
        super().set_facecolor(colors)
        self._facecolor3d = PolyCollection.get_facecolor(self)

    def set_edgecolor(self, colors):
        # docstring inherited
        super().set_edgecolor(colors)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)

    def set_alpha(self, alpha):
        # docstring inherited
        artist.Artist.set_alpha(self, alpha)
        try:
            self._facecolor3d = mcolors.to_rgba_array(
                self._facecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        try:
            self._edgecolors = mcolors.to_rgba_array(
                    self._edgecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        self.stale = True

    def get_facecolor(self):
        # docstring inherited
        # self._facecolors2d is not initialized until do_3d_projection
        if not hasattr(self, '_facecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._facecolors2d)

    def get_edgecolor(self):
        # docstring inherited
        # self._edgecolors2d is not initialized until do_3d_projection
        if not hasattr(self, '_edgecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._edgecolors2d)


def poly_collection_2d_to_3d(col, zs=0, zdir='z', axlim_clip=False):
    """
    Convert a `.PolyCollection` into a `.Poly3DCollection` object.

    Parameters
    ----------
    col : `~matplotlib.collections.PolyCollection`
        The collection to convert.
    zs : float or array of floats
        The location or locations to place the polygons in the collection along
        the *zdir* axis. Default: 0.
    zdir : {'x', 'y', 'z'}
        The axis in which to place the patches. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    """
    segments_3d, codes = _paths_to_3d_segments_with_codes(
            col.get_paths(), zs, zdir)
    col.__class__ = Poly3DCollection
    col.set_verts_and_codes(segments_3d, codes)
    col.set_3d_properties()
    col._axlim_clip = axlim_clip


def juggle_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that 2D *xs*, *ys* can be plotted in the plane
    orthogonal to *zdir*. *zdir* is normally 'x', 'y' or 'z'. However, if
    *zdir* starts with a '-' it is interpreted as a compensation for
    `rotate_axes`.
    """
    if zdir == 'x':
        return zs, xs, ys
    elif zdir == 'y':
        return xs, zs, ys
    elif zdir[0] == '-':
        return rotate_axes(xs, ys, zs, zdir)
    else:
        return xs, ys, zs


def rotate_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that the axes are rotated with *zdir* along
    the original z axis. Prepending the axis with a '-' does the
    inverse transform, so *zdir* can be 'x', '-x', 'y', '-y', 'z' or '-z'.
    """
    if zdir in ('x', '-y'):
        return ys, zs, xs
    elif zdir in ('-x', 'y'):
        return zs, xs, ys
    else:
        return xs, ys, zs


def _zalpha(colors, zs):
    """Modify the alphas of the color list according to depth."""
    # FIXME: This only works well if the points for *zs* are well-spaced
    #        in all three dimensions. Otherwise, at certain orientations,
    #        the min and max zs are very close together.
    #        Should really normalize against the viewing depth.
    if len(colors) == 0 or len(zs) == 0:
        return np.zeros((0, 4))
    norm = Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(mcolors.to_rgba_array(colors), (len(zs), 4))
    return np.column_stack([rgba[:, :3], rgba[:, 3] * sats])


def _all_points_on_plane(xs, ys, zs, atol=1e-8):
    """
    Check if all points are on the same plane. Note that NaN values are
    ignored.

    Parameters
    ----------
    xs, ys, zs : array-like
        The x, y, and z coordinates of the points.
    atol : float, default: 1e-8
        The tolerance for the equality check.
    """
    xs, ys, zs = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    points = np.column_stack([xs, ys, zs])
    points = points[~np.isnan(points).any(axis=1)]
    # Check for the case where we have less than 3 unique points
    points = np.unique(points, axis=0)
    if len(points) <= 3:
        return True
    # Calculate the vectors from the first point to all other points
    vs = (points - points[0])[1:]
    vs = vs / np.linalg.norm(vs, axis=1)[:, np.newaxis]
    # Filter out parallel vectors
    vs = np.unique(vs, axis=0)
    if len(vs) <= 2:
        return True
    # Filter out parallel and antiparallel vectors to the first vector
    cross_norms = np.linalg.norm(np.cross(vs[0], vs[1:]), axis=1)
    zero_cross_norms = np.where(np.isclose(cross_norms, 0, atol=atol))[0] + 1
    vs = np.delete(vs, zero_cross_norms, axis=0)
    if len(vs) <= 2:
        return True
    # Calculate the normal vector from the first three points
    n = np.cross(vs[0], vs[1])
    n = n / np.linalg.norm(n)
    # If the dot product of the normal vector and all other vectors is zero,
    # all points are on the same plane
    dots = np.dot(n, vs.transpose())
    return np.allclose(dots, 0, atol=atol)


def _generate_normals(polygons):
    """
    Compute the normals of a list of polygons, one normal per polygon.

    Normals point towards the viewer for a face with its vertices in
    counterclockwise order, following the right hand rule.

    Uses three points equally spaced around the polygon. This method assumes
    that the points are in a plane. Otherwise, more than one shade is required,
    which is not supported.

    Parameters
    ----------
    polygons : list of (M_i, 3) array-like, or (..., M, 3) array-like
        A sequence of polygons to compute normals for, which can have
        varying numbers of vertices. If the polygons all have the same
        number of vertices and array is passed, then the operation will
        be vectorized.

    Returns
    -------
    normals : (..., 3) array
        A normal vector estimated for the polygon.
    """
    if isinstance(polygons, np.ndarray):
        # optimization: polygons all have the same number of points, so can
        # vectorize
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n//3, 2*n//3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # The subtraction doesn't vectorize because polygons is jagged.
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            ps = np.asarray(ps)
            i1, i2, i3 = 0, n//3, 2*n//3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]
    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*,
    assuming a *lightsource* (using default position if not given).
    *color* can also be an array of the same length as *normals*.
    """
    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)

    with np.errstate(invalid="ignore"):
        shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
                 @ lightsource.direction)
    mask = ~np.isnan(shade)

    if mask.any():
        # convert dot product to allowed shading fractions
        in_norm = mcolors.Normalize(-1, 1)
        out_norm = mcolors.Normalize(0.3, 1).inverse

        def norm(x):
            return out_norm(in_norm(x))

        shade[~mask] = 0

        color = mcolors.to_rgba_array(color)
        # shape of color should be (M, 4) (where M is number of faces)
        # shape of shade should be (M,)
        # colors should have final shape of (M, 4)
        alpha = color[:, 3]
        colors = norm(shade)[:, np.newaxis] * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors
<|endoftext|>"""
Various transforms used for by the 3D code
"""

import numpy as np

from matplotlib import _api


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    return np.array([[1/dx,    0,    0, -xmin/dx],
                     [   0, 1/dy,    0, -ymin/dy],
                     [   0,    0, 1/dz, -zmin/dz],
                     [   0,    0,    0,        1]])


def _rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / np.linalg.norm(v)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 2*np.sin(angle/2)**2  # more numerically stable than t = 1-c

    R = np.array([
        [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],
        [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
        [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c]])

    return R


def _view_axes(E, R, V, roll):
    """
    Get the unit viewing axes in data coordinates.

    Parameters
    ----------
    E : 3-element numpy array
        The coordinates of the eye/camera.
    R : 3-element numpy array
        The coordinates of the center of the view box.
    V : 3-element numpy array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.

    Returns
    -------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    """
    w = (E - R)
    w = w/np.linalg.norm(w)
    u = np.cross(V, w)
    u = u/np.linalg.norm(u)
    v = np.cross(w, u)  # Will be a unit vector

    # Save some computation for the default roll=0
    if roll != 0:
        # A positive rotation of the camera is a negative rotation of the world
        Rroll = _rotation_about_vector(w, -roll)
        u = np.dot(Rroll, u)
        v = np.dot(Rroll, v)
    return u, v, w


def _view_transformation_uvw(u, v, w, E):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    E : 3-element numpy array
        The coordinates of the eye/camera.
    """
    Mr = np.eye(4)
    Mt = np.eye(4)
    Mr[:3, :3] = [u, v, w]
    Mt[:3, -1] = -E
    M = np.dot(Mr, Mt)
    return M


def _persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront+zback)/(zfront-zback)
    c = -2*(zfront*zback)/(zfront-zback)
    proj_matrix = np.array([[e,   0,  0, 0],
                            [0, e/a,  0, 0],
                            [0,   0,  b, c],
                            [0,   0, -1, 0]])
    return proj_matrix


def _ortho_transformation(zfront, zback):
    # note: w component in the resulting vector will be (zback-zfront), not 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    proj_matrix = np.array([[2, 0,  0, 0],
                            [0, 2,  0, 0],
                            [0, 0, -2, 0],
                            [0, 0,  a, b]])
    return proj_matrix


def _proj_transform_vec(vec, M):
    vecw = np.dot(M, vec.data)
    w = vecw[3]
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    if np.ma.isMA(vec[0]):  # we check each to protect for scalars
        txs = np.ma.array(txs, mask=vec[0].mask)
    if np.ma.isMA(vec[1]):
        tys = np.ma.array(tys, mask=vec[1].mask)
    if np.ma.isMA(vec[2]):
        tzs = np.ma.array(tzs, mask=vec[2].mask)
    return txs, tys, tzs


def _proj_transform_vec_clip(vec, M, focal_length):
    vecw = np.dot(M, vec.data)
    w = vecw[3]
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    if np.isinf(focal_length):  # don't clip orthographic projection
        tis = np.ones(txs.shape, dtype=bool)
    else:
        tis = (-1 <= txs) & (txs <= 1) & (-1 <= tys) & (tys <= 1) & (tzs <= 0)
    if np.ma.isMA(vec[0]):
        tis = tis & ~vec[0].mask
    if np.ma.isMA(vec[1]):
        tis = tis & ~vec[1].mask
    if np.ma.isMA(vec[2]):
        tis = tis & ~vec[2].mask

    txs = np.ma.masked_array(txs, ~tis)
    tys = np.ma.masked_array(tys, ~tis)
    tzs = np.ma.masked_array(tzs, ~tis)
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, invM):
    """
    Transform the points by the inverse of the projection matrix, *invM*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = np.dot(invM, vec)
    if vecr.shape == (4,):
        vecr = vecr.reshape((4, 1))
    for i in range(vecr.shape[1]):
        if vecr[3][i] != 0:
            vecr[:, i] = vecr[:, i] / vecr[3][i]
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    if np.ma.isMA(xs) or np.ma.isMA(ys) or np.ma.isMA(zs):
        return np.ma.array([xs, ys, zs, np.ones_like(xs)])
    else:
        return np.array([xs, ys, zs, np.ones_like(xs)])


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix *M*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


@_api.deprecated("3.10")
def proj_transform_clip(xs, ys, zs, M):
    return _proj_transform_clip(xs, ys, zs, M, focal_length=np.inf)


def _proj_transform_clip(xs, ys, zs, M, focal_length):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs, tys, tzs, tis
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M, focal_length)


def _proj_points(points, M):
    return np.column_stack(_proj_trans_points(points, M))


def _proj_trans_points(points, M):
    points = np.asanyarray(points)
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return proj_transform(xs, ys, zs, M)
<|endoftext|>"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>

Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""

from collections import defaultdict
import itertools
import math
import textwrap
import warnings

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation

from . import art3d
from . import proj3d
from . import axis3d


@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):
    """
    3D Axes object.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.
    """
    name = '3d'

    _axis_names = ("x", "y", "z")
    Axes._shared_axes["z"] = cbook.Grouper()
    Axes._shared_axes["view"] = cbook.Grouper()

    def __init__(
        self, fig, rect=None, *args,
        elev=30, azim=-60, roll=0, shareview=None, sharez=None,
        proj_type='persp', focal_length=None,
        box_aspect=None,
        computed_zorder=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fig : Figure
            The parent figure.
        rect : tuple (left, bottom, width, height), default: None.
            The ``(left, bottom, width, height)`` Axes position.
        elev : float, default: 30
            The elevation angle in degrees rotates the camera above and below
            the x-y plane, with a positive angle corresponding to a location
            above the plane.
        azim : float, default: -60
            The azimuthal angle in degrees rotates the camera about the z axis,
            with a positive angle corresponding to a right-handed rotation. In
            other words, a positive azimuth rotates the camera about the origin
            from its location along the +x axis towards the +y axis.
        roll : float, default: 0
            The roll angle in degrees rotates the camera about the viewing
            axis. A positive angle spins the camera clockwise, causing the
            scene to rotate counter-clockwise.
        shareview : Axes3D, optional
            Other Axes to share view angles with.  Note that it is not possible
            to unshare axes.
        sharez : Axes3D, optional
            Other Axes to share z-limits with.  Note that it is not possible to
            unshare axes.
        proj_type : {'persp', 'ortho'}
            The projection type, default 'persp'.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            For a projection type of 'ortho', must be set to either None
            or infinity (numpy.inf). If None, defaults to infinity.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)
        box_aspect : 3-tuple of floats, default: None
            Changes the physical dimensions of the Axes3D, such that the ratio
            of the axis lengths in display units is x:y:z.
            If None, defaults to 4:4:3
        computed_zorder : bool, default: True
            If True, the draw order is computed based on the average position
            of the `.Artist`\\s along the view direction.
            Set to False if you want to manually control the order in which
            Artists are drawn on top of each other using their *zorder*
            attribute. This can be used for fine-tuning if the automatic order
            does not produce the desired result. Note however, that a manual
            zorder will only be correct for a limited view angle. If the figure
            is rotated by the user, it will look wrong from certain angles.

        **kwargs
            Other optional keyword arguments:

            %(Axes3D:kwdoc)s
        """

        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]

        self.initial_azim = azim
        self.initial_elev = elev
        self.initial_roll = roll
        self.set_proj_type(proj_type, focal_length)
        self.computed_zorder = computed_zorder

        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        xymargin = 0.05 * 10/11  # match mpl3.8 appearance
        self.xy_dataLim = Bbox([[xymargin, xymargin],
                                [1 - xymargin, 1 - xymargin]])
        # z-limits are encoded in the x-component of the Bbox, y is un-used
        self.zz_dataLim = Bbox.unit()

        # inhibit autoscale_view until the axes are defined
        # they can't be defined until Axes.__init__ has been called
        self.view_init(self.initial_elev, self.initial_azim, self.initial_roll)

        self._sharez = sharez
        if sharez is not None:
            self._shared_axes["z"].join(self, sharez)
            self._adjustable = 'datalim'

        self._shareview = shareview
        if shareview is not None:
            self._shared_axes["view"].join(self, shareview)

        if kwargs.pop('auto_add_to_figure', False):
            raise AttributeError(
                'auto_add_to_figure is no longer supported for Axes3D. '
                'Use fig.add_axes(ax) instead.'
            )

        super().__init__(
            fig, rect, frameon=True, box_aspect=box_aspect, *args, **kwargs
        )
        # Disable drawing of axes by base class
        super().set_axis_off()
        # Enable drawing of axes by Axes3D class
        self.set_axis_on()
        self.M = None
        self.invM = None

        self._view_margin = 1/48  # default value to match mpl3.8
        self.autoscale_view()

        # func used to format z -- fall back on major formatters
        self.fmt_zdata = None

        self.mouse_init()
        fig = self.get_figure(root=True)
        fig.canvas.callbacks._connect_picklable(
            'motion_notify_event', self._on_move)
        fig.canvas.callbacks._connect_picklable(
            'button_press_event', self._button_press)
        fig.canvas.callbacks._connect_picklable(
            'button_release_event', self._button_release)
        self.set_top_view()

        self.patch.set_linewidth(0)
        # Calculate the pseudo-data width and height
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        self._pseudo_w, self._pseudo_h = pseudo_bbox[1] - pseudo_bbox[0]

        # mplot3d currently manages its own spines and needs these turned off
        # for bounding box calculations
        self.spines[:].set_visible(False)

    def set_axis_off(self):
        self._axis3don = False
        self.stale = True

    def set_axis_on(self):
        self._axis3don = True
        self.stale = True

    def convert_zunits(self, z):
        """
        For artists in an Axes, if the zaxis has units support,
        convert *z* using zaxis unit type
        """
        return self.zaxis.convert_units(z)

    def set_top_view(self):
        # this happens to be the right view for the viewing coordinates
        # moved up and to the left slightly to fit labels and axes
        xdwl = 0.95 / self._dist
        xdw = 0.9 / self._dist
        ydwl = 0.95 / self._dist
        ydw = 0.9 / self._dist
        # Set the viewing pane.
        self.viewLim.intervalx = (-xdwl, xdw)
        self.viewLim.intervaly = (-ydwl, ydw)
        self.stale = True

    def _init_axis(self):
        """Init 3D Axes; overrides creation of regular X/Y Axes."""
        self.xaxis = axis3d.XAxis(self)
        self.yaxis = axis3d.YAxis(self)
        self.zaxis = axis3d.ZAxis(self)

    def get_zaxis(self):
        """Return the ``ZAxis`` (`~.axis3d.Axis`) instance."""
        return self.zaxis

    get_zgridlines = _axis_method_wrapper("zaxis", "get_gridlines")
    get_zticklines = _axis_method_wrapper("zaxis", "get_ticklines")

    def _transformed_cube(self, vals):
        """Return cube with limits from *vals* transformed by self.M."""
        minx, maxx, miny, maxy, minz, maxz = vals
        xyzs = [(minx, miny, minz),
                (maxx, miny, minz),
                (maxx, maxy, minz),
                (minx, maxy, minz),
                (minx, miny, maxz),
                (maxx, miny, maxz),
                (maxx, maxy, maxz),
                (minx, maxy, maxz)]
        return proj3d._proj_points(xyzs, self.M)

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """
        Set the aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            Possible values:

            =========   ==================================================
            value       description
            =========   ==================================================
            'auto'      automatic; fill the position rectangle with data.
            'equal'     adapt all the axes to have equal aspect ratios.
            'equalxy'   adapt the x and y axes to have equal aspect ratios.
            'equalxz'   adapt the x and z axes to have equal aspect ratios.
            'equalyz'   adapt the y and z axes to have equal aspect ratios.
            =========   ==================================================

        adjustable : None or {'box', 'datalim'}, optional
            If not *None*, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or 2-tuple of float, optional
            If not *None*, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way to
            specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect
        """
        _api.check_in_list(('auto', 'equal', 'equalxy', 'equalyz', 'equalxz'),
                           aspect=aspect)
        super().set_aspect(
            aspect='auto', adjustable=adjustable, anchor=anchor, share=share)
        self._aspect = aspect

        if aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
            ax_indices = self._equal_aspect_axis_indices(aspect)

            view_intervals = np.array([self.xaxis.get_view_interval(),
                                       self.yaxis.get_view_interval(),
                                       self.zaxis.get_view_interval()])
            ptp = np.ptp(view_intervals, axis=1)
            if self._adjustable == 'datalim':
                mean = np.mean(view_intervals, axis=1)
                scale = max(ptp[ax_indices] / self._box_aspect[ax_indices])
                deltas = scale * self._box_aspect

                for i, set_lim in enumerate((self.set_xlim3d,
                                             self.set_ylim3d,
                                             self.set_zlim3d)):
                    if i in ax_indices:
                        set_lim(mean[i] - deltas[i]/2., mean[i] + deltas[i]/2.,
                                auto=True, view_margin=None)
            else:  # 'box'
                # Change the box aspect such that the ratio of the length of
                # the unmodified axis to the length of the diagonal
                # perpendicular to it remains unchanged.
                box_aspect = np.array(self._box_aspect)
                box_aspect[ax_indices] = ptp[ax_indices]
                remaining_ax_indices = {0, 1, 2}.difference(ax_indices)
                if remaining_ax_indices:
                    remaining = remaining_ax_indices.pop()
                    old_diag = np.linalg.norm(self._box_aspect[ax_indices])
                    new_diag = np.linalg.norm(box_aspect[ax_indices])
                    box_aspect[remaining] *= new_diag / old_diag
                self.set_box_aspect(box_aspect)

    def _equal_aspect_axis_indices(self, aspect):
        """
        Get the indices for which of the x, y, z axes are constrained to have
        equal aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            See descriptions in docstring for `.set_aspect()`.
        """
        ax_indices = []  # aspect == 'auto'
        if aspect == 'equal':
            ax_indices = [0, 1, 2]
        elif aspect == 'equalxy':
            ax_indices = [0, 1]
        elif aspect == 'equalxz':
            ax_indices = [0, 2]
        elif aspect == 'equalyz':
            ax_indices = [1, 2]
        return ax_indices

    def set_box_aspect(self, aspect, *, zoom=1):
        """
        Set the Axes box aspect.

        The box aspect is the ratio of height to width in display
        units for each face of the box when viewed perpendicular to
        that face.  This is not to be confused with the data aspect (see
        `~.Axes3D.set_aspect`). The default ratios are 4:4:3 (x:y:z).

        To simulate having equal aspect in data space, set the box
        aspect to match your data range in each dimension.

        *zoom* controls the overall size of the Axes3D in the figure.

        Parameters
        ----------
        aspect : 3-tuple of floats or None
            Changes the physical dimensions of the Axes3D, such that the ratio
            of the axis lengths in display units is x:y:z.
            If None, defaults to (4, 4, 3).

        zoom : float, default: 1
            Control overall size of the Axes3D in the figure. Must be > 0.
        """
        if zoom <= 0:
            raise ValueError(f'Argument zoom = {zoom} must be > 0')

        if aspect is None:
            aspect = np.asarray((4, 4, 3), dtype=float)
        else:
            aspect = np.asarray(aspect, dtype=float)
            _api.check_shape((3,), aspect=aspect)
        # The scale 1.8294640721620434 is tuned to match the mpl3.2 appearance.
        # The 25/24 factor is to compensate for the change in automargin
        # behavior in mpl3.9. This comes from the padding of 1/48 on both sides
        # of the axes in mpl3.8.
        aspect *= 1.8294640721620434 * 25/24 * zoom / np.linalg.norm(aspect)

        self._box_aspect = self._roll_to_vertical(aspect, reverse=True)
        self.stale = True

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)

        # in the superclass, we would go through and actually deal with axis
        # scales and box/datalim. Those are all irrelevant - all we need to do
        # is make sure our coordinate system is square.
        trans = self.get_figure().transSubfigure
        bb = mtransforms.Bbox.unit().transformed(trans)
        # this is the physical aspect of the panel (or figure):
        fig_aspect = bb.height / bb.width

        box_aspect = 1
        pb = position.frozen()
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')

    @martist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        self._unstale_viewLim()

        # draw the background patch
        self.patch.draw(renderer)
        self._frameon = False

        # first, set the aspect
        # this is duplicated from `axes._base._AxesBase.draw`
        # but must be called before any of the artist are drawn as
        # it adjusts the view limits and the size of the bounding box
        # of the Axes
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        # add the projection matrix to the renderer
        self.M = self.get_proj()
        self.invM = np.linalg.inv(self.M)

        collections_and_patches = (
            artist for artist in self._children
            if isinstance(artist, (mcoll.Collection, mpatches.Patch))
            and artist.get_visible())
        if self.computed_zorder:
            # Calculate projection of collections and patches and zorder
            # them. Make sure they are drawn above the grids.
            zorder_offset = max(axis.get_zorder()
                                for axis in self._axis_map.values()) + 1
            collection_zorder = patch_zorder = zorder_offset

            for artist in sorted(collections_and_patches,
                                 key=lambda artist: artist.do_3d_projection(),
                                 reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            for artist in collections_and_patches:
                artist.do_3d_projection()

        if self._axis3don:
            # Draw panes first
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            # Then gridlines
            for axis in self._axis_map.values():
                axis.draw_grid(renderer)
            # Then axes, labels, text, and ticks
            for axis in self._axis_map.values():
                axis.draw(renderer)

        # Then rest
        super().draw(renderer)

    def get_axis_position(self):
        tc = self._transformed_cube(self.get_w_lims())
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return xhigh, yhigh, zhigh

    def update_datalim(self, xys, **kwargs):
        """
        Not implemented in `~mpl_toolkits.mplot3d.axes3d.Axes3D`.
        """
        pass

    get_autoscalez_on = _axis_method_wrapper("zaxis", "_get_autoscale_on")
    set_autoscalez_on = _axis_method_wrapper("zaxis", "_set_autoscale_on")

    def get_zmargin(self):
        """
        Retrieve autoscaling margin of the z-axis.

        .. versionadded:: 3.9

        Returns
        -------
        zmargin : float

        See Also
        --------
        mpl_toolkits.mplot3d.axes3d.Axes3D.set_zmargin
        """
        return self._zmargin

    def set_zmargin(self, m):
        """
        Set padding of Z data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
        """
        if m <= -0.5:
            raise ValueError("margin must be greater than -0.5")
        self._zmargin = m
        self._request_autoscale_view("z")
        self.stale = True

    def margins(self, *margins, x=None, y=None, z=None, tight=True):
        """
        Set or retrieve autoscaling margins.

        See `.Axes.margins` for full documentation.  Because this function
        applies to 3D Axes, it also takes a *z* argument, and returns
        ``(xmargin, ymargin, zmargin)``.
        """
        if margins and (x is not None or y is not None or z is not None):
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments for x, y, and/or z.')
        elif len(margins) == 1:
            x = y = z = margins[0]
        elif len(margins) == 3:
            x, y, z = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all '
                            'margins, or one for each margin (x, y, z).')

        if x is None and y is None and z is None:
            if tight is not True:
                _api.warn_external(f'ignoring tight={tight!r} in get mode')
            return self._xmargin, self._ymargin, self._zmargin

        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)
        if z is not None:
            self.set_zmargin(z)

        self.autoscale_view(
            tight=tight, scalex=(x is not None), scaley=(y is not None),
            scalez=(z is not None)
        )

    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Convenience method for simple axis view autoscaling.

        See `.Axes.autoscale` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.
        """
        if enable is None:
            scalex = True
            scaley = True
            scalez = True
        else:
            if axis in ['x', 'both']:
                self.set_autoscalex_on(enable)
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(enable)
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
            if axis in ['z', 'both']:
                self.set_autoscalez_on(enable)
                scalez = self.get_autoscalez_on()
            else:
                scalez = False
        if scalex:
            self._request_autoscale_view("x", tight=tight)
        if scaley:
            self._request_autoscale_view("y", tight=tight)
        if scalez:
            self._request_autoscale_view("z", tight=tight)

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        # This updates the bounding boxes as to keep a record as to what the
        # minimum sized rectangular volume holds the data.
        if np.shape(X) == np.shape(Y):
            self.xy_dataLim.update_from_data_xy(
                np.column_stack([np.ravel(X), np.ravel(Y)]), not had_data)
        else:
            self.xy_dataLim.update_from_data_x(X, not had_data)
            self.xy_dataLim.update_from_data_y(Y, not had_data)
        if Z is not None:
            self.zz_dataLim.update_from_data_x(Z, not had_data)
        # Let autoscale_view figure out how to use this data.
        self.autoscale_view()

    def autoscale_view(self, tight=None,
                       scalex=True, scaley=True, scalez=True):
        """
        Autoscale the view limits using the data limits.

        See `.Axes.autoscale_view` for full documentation.  Because this
        function applies to 3D Axes, it also takes a *scalez* argument.
        """
        # This method looks at the rectangular volume (see above)
        # of data and decides how to scale the view portal to fit it.
        if tight is None:
            _tight = self._tight
            if not _tight:
                # if image data only just use the datalim
                for artist in self._children:
                    if isinstance(artist, mimage.AxesImage):
                        _tight = True
                    elif isinstance(artist, (mlines.Line2D, mpatches.Patch)):
                        _tight = False
                        break
        else:
            _tight = self._tight = bool(tight)

        if scalex and self.get_autoscalex_on():
            x0, x1 = self.xy_dataLim.intervalx
            xlocator = self.xaxis.get_major_locator()
            x0, x1 = xlocator.nonsingular(x0, x1)
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            if not _tight:
                x0, x1 = xlocator.view_limits(x0, x1)
            self.set_xbound(x0, x1, self._view_margin)

        if scaley and self.get_autoscaley_on():
            y0, y1 = self.xy_dataLim.intervaly
            ylocator = self.yaxis.get_major_locator()
            y0, y1 = ylocator.nonsingular(y0, y1)
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            if not _tight:
                y0, y1 = ylocator.view_limits(y0, y1)
            self.set_ybound(y0, y1, self._view_margin)

        if scalez and self.get_autoscalez_on():
            z0, z1 = self.zz_dataLim.intervalx
            zlocator = self.zaxis.get_major_locator()
            z0, z1 = zlocator.nonsingular(z0, z1)
            if self._zmargin > 0:
                delta = (z1 - z0) * self._zmargin
                z0 -= delta
                z1 += delta
            if not _tight:
                z0, z1 = zlocator.view_limits(z0, z1)
            self.set_zbound(z0, z1, self._view_margin)

    def get_w_lims(self):
        """Get 3D world limits."""
        minx, maxx = self.get_xlim3d()
        miny, maxy = self.get_ylim3d()
        minz, maxz = self.get_zlim3d()
        return minx, maxx, miny, maxy, minz, maxz

    def _set_bound3d(self, get_bound, set_lim, axis_inverted,
                     lower=None, upper=None, view_margin=None):
        """
        Set 3D axis bounds.
        """
        if upper is None and np.iterable(lower):
            lower, upper = lower

        old_lower, old_upper = get_bound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        set_lim(sorted((lower, upper), reverse=bool(axis_inverted())),
                auto=None, view_margin=view_margin)

    def set_xbound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the x-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalex_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_xlim`.

        See Also
        --------
        get_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        self._set_bound3d(self.get_xbound, self.set_xlim, self.xaxis_inverted,
                          lower, upper, view_margin)

    def set_ybound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the y-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_ylim`.

        See Also
        --------
        get_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        self._set_bound3d(self.get_ybound, self.set_ylim, self.yaxis_inverted,
                          lower, upper, view_margin)

    def set_zbound(self, lower=None, upper=None, view_margin=None):
        """
        Set the lower and upper numerical bounds of the z-axis.
        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.
        view_margin : float or None
            The margin to apply to the bounds. If *None*, the margin is handled
            by `.set_zlim`.

        See Also
        --------
        get_zbound
        get_zlim, set_zlim
        invert_zaxis, zaxis_inverted
        """
        self._set_bound3d(self.get_zbound, self.set_zlim, self.zaxis_inverted,
                          lower, upper, view_margin)

    def _set_lim3d(self, axis, lower=None, upper=None, *, emit=True,
                   auto=False, view_margin=None, axmin=None, axmax=None):
        """
        Set 3D axis limits.
        """
        if upper is None:
            if np.iterable(lower):
                lower, upper = lower
            elif axmax is None:
                upper = axis.get_view_interval()[1]
        if lower is None and axmin is None:
            lower = axis.get_view_interval()[0]
        if axmin is not None:
            if lower is not None:
                raise TypeError("Cannot pass both 'lower' and 'min'")
            lower = axmin
        if axmax is not None:
            if upper is not None:
                raise TypeError("Cannot pass both 'upper' and 'max'")
            upper = axmax
        if np.isinf(lower) or np.isinf(upper):
            raise ValueError(f"Axis limits {lower}, {upper} cannot be infinite")
        if view_margin is None:
            if mpl.rcParams['axes3d.automargin']:
                view_margin = self._view_margin
            else:
                view_margin = 0
        delta = (upper - lower) * view_margin
        lower -= delta
        upper += delta
        return axis._set_lim(lower, upper, emit=emit, auto=auto)

    def set_xlim(self, left=None, right=None, *, emit=True, auto=False,
                 view_margin=None, xmin=None, xmax=None):
        """
        Set the 3D x-axis view limits.

        Parameters
        ----------
        left : float, optional
            The left xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The left and right xlims may also be passed as the tuple
            (*left*, *right*) as the first positional argument (or as
            the *left* keyword argument).

            .. ACCEPTS: (left: float, right: float)

        right : float, optional
            The right xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the x-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        xmin, xmax : float, optional
            They are equivalent to left and right respectively, and it is an
            error to pass both *xmin* and *left* or *xmax* and *right*.

        Returns
        -------
        left, right : (float, float)
            The new x-axis limits in data coordinates.

        See Also
        --------
        get_xlim
        set_xbound, get_xbound
        invert_xaxis, xaxis_inverted

        Notes
        -----
        The *left* value may be greater than the *right* value, in which
        case the x-axis values will decrease from *left* to *right*.

        Examples
        --------
        >>> set_xlim(left, right)
        >>> set_xlim((left, right))
        >>> left, right = set_xlim(left, right)

        One limit may be left unchanged.

        >>> set_xlim(right=right_lim)

        Limits may be passed in reverse order to flip the direction of
        the x-axis. For example, suppose ``x`` represents depth of the
        ocean in m. The x-axis limits might be set like the following
        so 5000 m depth is at the left of the plot and the surface,
        0 m, is at the right.

        >>> set_xlim(5000, 0)
        """
        return self._set_lim3d(self.xaxis, left, right, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=xmin, axmax=xmax)

    def set_ylim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, ymin=None, ymax=None):
        """
        Set the 3D y-axis view limits.

        Parameters
        ----------
        bottom : float, optional
            The bottom ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The bottom and top ylims may also be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            The top ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the y-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        ymin, ymax : float, optional
            They are equivalent to bottom and top respectively, and it is an
            error to pass both *ymin* and *bottom* or *ymax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new y-axis limits in data coordinates.

        See Also
        --------
        get_ylim
        set_ybound, get_ybound
        invert_yaxis, yaxis_inverted

        Notes
        -----
        The *bottom* value may be greater than the *top* value, in which
        case the y-axis values will decrease from *bottom* to *top*.

        Examples
        --------
        >>> set_ylim(bottom, top)
        >>> set_ylim((bottom, top))
        >>> bottom, top = set_ylim(bottom, top)

        One limit may be left unchanged.

        >>> set_ylim(top=top_lim)

        Limits may be passed in reverse order to flip the direction of
        the y-axis. For example, suppose ``y`` represents depth of the
        ocean in m. The y-axis limits might be set like the following
        so 5000 m depth is at the bottom of the plot and the surface,
        0 m, is at the top.

        >>> set_ylim(5000, 0)
        """
        return self._set_lim3d(self.yaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=ymin, axmax=ymax)

    def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, zmin=None, zmax=None):
        """
        Set the 3D z-axis view limits.

        Parameters
        ----------
        bottom : float, optional
            The bottom zlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The bottom and top zlims may also be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            The top zlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the z-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        view_margin : float, optional
            The additional margin to apply to the limits.

        zmin, zmax : float, optional
            They are equivalent to bottom and top respectively, and it is an
            error to pass both *zmin* and *bottom* or *zmax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new z-axis limits in data coordinates.

        See Also
        --------
        get_zlim
        set_zbound, get_zbound
        invert_zaxis, zaxis_inverted

        Notes
        -----
        The *bottom* value may be greater than the *top* value, in which
        case the z-axis values will decrease from *bottom* to *top*.

        Examples
        --------
        >>> set_zlim(bottom, top)
        >>> set_zlim((bottom, top))
        >>> bottom, top = set_zlim(bottom, top)

        One limit may be left unchanged.

        >>> set_zlim(top=top_lim)

        Limits may be passed in reverse order to flip the direction of
        the z-axis. For example, suppose ``z`` represents depth of the
        ocean in m. The z-axis limits might be set like the following
        so 5000 m depth is at the bottom of the plot and the surface,
        0 m, is at the top.

        >>> set_zlim(5000, 0)
        """
        return self._set_lim3d(self.zaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=zmin, axmax=zmax)

    set_xlim3d = set_xlim
    set_ylim3d = set_ylim
    set_zlim3d = set_zlim

    def get_xlim(self):
        # docstring inherited
        return tuple(self.xy_viewLim.intervalx)

    def get_ylim(self):
        # docstring inherited
        return tuple(self.xy_viewLim.intervaly)

    def get_zlim(self):
        """
        Return the 3D z-axis view limits.

        Returns
        -------
        left, right : (float, float)
            The current z-axis limits in data coordinates.

        See Also
        --------
        set_zlim
        set_zbound, get_zbound
        invert_zaxis, zaxis_inverted

        Notes
        -----
        The z-axis may be inverted, in which case the *left* value will
        be greater than the *right* value.
        """
        return tuple(self.zz_viewLim.intervalx)

    get_zscale = _axis_method_wrapper("zaxis", "get_scale")

    # Redefine all three methods to overwrite their docstrings.
    set_xscale = _axis_method_wrapper("xaxis", "_set_axes_scale")
    set_yscale = _axis_method_wrapper("yaxis", "_set_axes_scale")
    set_zscale = _axis_method_wrapper("zaxis", "_set_axes_scale")
    set_xscale.__doc__, set_yscale.__doc__, set_zscale.__doc__ = map(
        """
        Set the {}-axis scale.

        Parameters
        ----------
        value : {{"linear"}}
            The axis scale type to apply.  3D Axes currently only support
            linear scales; other scales yield nonsensical results.

        **kwargs
            Keyword arguments are nominally forwarded to the scale class, but
            none of them is applicable for linear scales.
        """.format,
        ["x", "y", "z"])

    get_zticks = _axis_method_wrapper("zaxis", "get_ticklocs")
    set_zticks = _axis_method_wrapper("zaxis", "set_ticks")
    get_zmajorticklabels = _axis_method_wrapper("zaxis", "get_majorticklabels")
    get_zminorticklabels = _axis_method_wrapper("zaxis", "get_minorticklabels")
    get_zticklabels = _axis_method_wrapper("zaxis", "get_ticklabels")
    set_zticklabels = _axis_method_wrapper(
        "zaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes3D.set_zticks"})

    zaxis_date = _axis_method_wrapper("zaxis", "axis_date")
    if zaxis_date.__doc__:
        zaxis_date.__doc__ += textwrap.dedent("""

        Notes
        -----
        This function is merely provided for completeness, but 3D Axes do not
        support dates for ticks, and so this may not work as expected.
        """)

    def clabel(self, *args, **kwargs):
        """Currently not implemented for 3D Axes, and returns *None*."""
        return None

    def view_init(self, elev=None, azim=None, roll=None, vertical_axis="z",
                  share=False):
        """
        Set the elevation and azimuth of the Axes in degrees (not radians).

        This can be used to rotate the Axes programmatically.

        To look normal to the primary planes, the following elevation and
        azimuth angles can be used. A roll angle of 0, 90, 180, or 270 deg
        will rotate these views while keeping the axes at right angles.

        ==========   ====  ====
        view plane   elev  azim
        ==========   ====  ====
        XY           90    -90
        XZ           0     -90
        YZ           0     0
        -XY          -90   90
        -XZ          0     90
        -YZ          0     180
        ==========   ====  ====

        Parameters
        ----------
        elev : float, default: None
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. For example, with the default
            vertical axis of 'z', the elevation defines the angle of the camera
            location above the x-y plane.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        azim : float, default: None
            The azimuthal angle in degrees rotates the camera about the
            vertical axis, with a positive angle corresponding to a
            right-handed rotation. For example, with the default vertical axis
            of 'z', a positive azimuth rotates the camera about the origin from
            its location along the +x axis towards the +y axis.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        roll : float, default: None
            The roll angle in degrees rotates the camera about the viewing
            axis. A positive angle spins the camera clockwise, causing the
            scene to rotate counter-clockwise.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        vertical_axis : {"z", "x", "y"}, default: "z"
            The axis to align vertically. *azim* rotates about this axis.
        share : bool, default: False
            If ``True``, apply the settings to all Axes with shared views.
        """

        self._dist = 10  # The camera distance from origin. Behaves like zoom

        if elev is None:
            elev = self.initial_elev
        if azim is None:
            azim = self.initial_azim
        if roll is None:
            roll = self.initial_roll
        vertical_axis = _api.check_getitem(
            {name: idx for idx, name in enumerate(self._axis_names)},
            vertical_axis=vertical_axis,
        )

        if share:
            axes = {sibling for sibling
                    in self._shared_axes['view'].get_siblings(self)}
        else:
            axes = [self]

        for ax in axes:
            ax.elev = elev
            ax.azim = azim
            ax.roll = roll
            ax._vertical_axis = vertical_axis

    def set_proj_type(self, proj_type, focal_length=None):
        """
        Set the projection type.

        Parameters
        ----------
        proj_type : {'persp', 'ortho'}
            The projection type.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)
        """
        _api.check_in_list(['persp', 'ortho'], proj_type=proj_type)
        if proj_type == 'persp':
            if focal_length is None:
                focal_length = 1
            elif focal_length <= 0:
                raise ValueError(f"focal_length = {focal_length} must be "
                                 "greater than 0")
            self._focal_length = focal_length
        else:  # 'ortho':
            if focal_length not in (None, np.inf):
                raise ValueError(f"focal_length = {focal_length} must be "
                                 f"None for proj_type = {proj_type}")
            self._focal_length = np.inf

    def _roll_to_vertical(
        self, arr: "np.typing.ArrayLike", reverse: bool = False
    ) -> np.ndarray:
        """
        Roll arrays to match the different vertical axis.

        Parameters
        ----------
        arr : ArrayLike
            Array to roll.
        reverse : bool, default: False
            Reverse the direction of the roll.
        """
        if reverse:
            return np.roll(arr, (self._vertical_axis - 2) * -1)
        else:
            return np.roll(arr, (self._vertical_axis - 2))

    def get_proj(self):
        """Create the projection matrix from the current viewing position."""

        # Transform to uniform world coordinates 0-1, 0-1, 0-1
        box_aspect = self._roll_to_vertical(self._box_aspect)
        worldM = proj3d.world_transformation(
            *self.get_xlim3d(),
            *self.get_ylim3d(),
            *self.get_zlim3d(),
            pb_aspect=box_aspect,
        )

        # Look into the middle of the world coordinates:
        R = 0.5 * box_aspect

        # elev: elevation angle in the z plane.
        # azim: azimuth angle in the xy plane.
        # Coordinates for a point that rotates around the box of data.
        # p0, p1 corresponds to rotating the box only around the vertical axis.
        # p2 corresponds to rotating the box only around the horizontal axis.
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)

        # When changing vertical axis the coordinates changes as well.
        # Roll the values to get the same behaviour as the default:
        ps = self._roll_to_vertical([p0, p1, p2])

        # The coordinates for the eye viewing point. The eye is looking
        # towards the middle of the box of data from a distance:
        eye = R + self._dist * ps

        # Calculate the viewing axes for the eye position
        u, v, w = self._calc_view_axes(eye)
        self._view_u = u  # _view_u is towards the right of the screen
        self._view_v = v  # _view_v is towards the top of the screen
        self._view_w = w  # _view_w is out of the screen

        # Generate the view and projection transformation matrices
        if self._focal_length == np.inf:
            # Orthographic projection
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            # Perspective projection
            # Scale the eye dist to compensate for the focal length zoom effect
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist,
                                                 self._dist,
                                                 self._focal_length)

        # Combine all the transformation matrices to get the final projection
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M

    def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
        """
        Set the mouse buttons for 3D rotation and zooming.

        Parameters
        ----------
        rotate_btn : int or list of int, default: 1
            The mouse button or buttons to use for 3D rotation of the Axes.
        pan_btn : int or list of int, default: 2
            The mouse button or buttons to use to pan the 3D Axes.
        zoom_btn : int or list of int, default: 3
            The mouse button or buttons to use to zoom the 3D Axes.
        """
        self.button_pressed = None
        # coerce scalars into array-like, then convert into
        # a regular list to avoid comparisons against None
        # which breaks in recent versions of numpy.
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        self._pan_btn = np.atleast_1d(pan_btn).tolist()
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()

    def disable_mouse_rotation(self):
        """Disable mouse buttons for 3D rotation, panning, and zooming."""
        self.mouse_init(rotate_btn=[], pan_btn=[], zoom_btn=[])

    def can_zoom(self):
        # doc-string inherited
        return True

    def can_pan(self):
        # doc-string inherited
        return True

    def sharez(self, other):
        """
        Share the z-axis with *other*.

        This is equivalent to passing ``sharez=other`` when constructing the
        Axes, and cannot be used if the z-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(Axes3D, other=other)
        if self._sharez is not None and other is not self._sharez:
            raise ValueError("z-axis is already shared")
        self._shared_axes["z"].join(self, other)
        self._sharez = other
        self.zaxis.major = other.zaxis.major  # Ticker instances holding
        self.zaxis.minor = other.zaxis.minor  # locator and formatter.
        z0, z1 = other.get_zlim()
        self.set_zlim(z0, z1, emit=False, auto=other.get_autoscalez_on())
        self.zaxis._scale = other.zaxis._scale

    def shareview(self, other):
        """
        Share the view angles with *other*.

        This is equivalent to passing ``shareview=other`` when constructing the
        Axes, and cannot be used if the view angles are already being shared
        with another Axes.  Note that it is not possible to unshare axes.
        """
        _api.check_isinstance(Axes3D, other=other)
        if self._shareview is not None and other is not self._shareview:
            raise ValueError("view angles are already shared")
        self._shared_axes["view"].join(self, other)
        self._shareview = other
        vertical_axis = self._axis_names[other._vertical_axis]
        self.view_init(elev=other.elev, azim=other.azim, roll=other.roll,
                       vertical_axis=vertical_axis, share=True)

    def clear(self):
        # docstring inherited.
        super().clear()
        if self._focal_length == np.inf:
            self._zmargin = mpl.rcParams['axes.zmargin']
        else:
            self._zmargin = 0.

        xymargin = 0.05 * 10/11  # match mpl3.8 appearance
        self.xy_dataLim = Bbox([[xymargin, xymargin],
                                [1 - xymargin, 1 - xymargin]])
        # z-limits are encoded in the x-component of the Bbox, y is un-used
        self.zz_dataLim = Bbox.unit()
        self._view_margin = 1/48  # default value to match mpl3.8
        self.autoscale_view()

        self.grid(mpl.rcParams['axes3d.grid'])

    def _button_press(self, event):
        if event.inaxes == self:
            self.button_pressed = event.button
            self._sx, self._sy = event.xdata, event.ydata
            toolbar = self.get_figure(root=True).canvas.toolbar
            if toolbar and toolbar._nav_stack() is None:
                toolbar.push_current()
            if toolbar:
                toolbar.set_message(toolbar._mouse_event_to_message(event))

    def _button_release(self, event):
        self.button_pressed = None
        toolbar = self.get_figure(root=True).canvas.toolbar
        # backend_bases.release_zoom and backend_bases.release_pan call
        # push_current, so check the navigation mode so we don't call it twice
        if toolbar and self.get_navigate_mode() is None:
            toolbar.push_current()
        if toolbar:
            toolbar.set_message(toolbar._mouse_event_to_message(event))

    def _get_view(self):
        # docstring inherited
        return {
            "xlim": self.get_xlim(), "autoscalex_on": self.get_autoscalex_on(),
            "ylim": self.get_ylim(), "autoscaley_on": self.get_autoscaley_on(),
            "zlim": self.get_zlim(), "autoscalez_on": self.get_autoscalez_on(),
        }, (self.elev, self.azim, self.roll)

    def _set_view(self, view):
        # docstring inherited
        props, (elev, azim, roll) = view
        self.set(**props)
        self.elev = elev
        self.azim = azim
        self.roll = roll

    def format_zdata(self, z):
        """
        Return *z* string formatted.  This function will use the
        :attr:`fmt_zdata` attribute if it is callable, else will fall
        back on the zaxis major formatter
        """
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            func = self.zaxis.get_major_formatter().format_data_short
            val = func(z)
            return val

    def format_coord(self, xv, yv, renderer=None):
        """
        Return a string giving the current view rotation angles, or the x, y, z
        coordinates of the point on the nearest axis pane underneath the mouse
        cursor, depending on the mouse button pressed.
        """
        coords = ''

        if self.button_pressed in self._rotate_btn:
            # ignore xv and yv and display angles instead
            coords = self._rotation_coords()

        elif self.M is not None:
            coords = self._location_coords(xv, yv, renderer)

        return coords

    def _rotation_coords(self):
        """
        Return the rotation angles as a string.
        """
        norm_elev = art3d._norm_angle(self.elev)
        norm_azim = art3d._norm_angle(self.azim)
        norm_roll = art3d._norm_angle(self.roll)
        coords = (f"elevation={norm_elev:.0f}\N{DEGREE SIGN}, "
                  f"azimuth={norm_azim:.0f}\N{DEGREE SIGN}, "
                  f"roll={norm_roll:.0f}\N{DEGREE SIGN}"
                  ).replace("-", "\N{MINUS SIGN}")
        return coords

    def _location_coords(self, xv, yv, renderer):
        """
        Return the location on the axis pane underneath the cursor as a string.
        """
        p1, pane_idx = self._calc_coord(xv, yv, renderer)
        xs = self.format_xdata(p1[0])
        ys = self.format_ydata(p1[1])
        zs = self.format_zdata(p1[2])
        if pane_idx == 0:
            coords = f'x pane={xs}, y={ys}, z={zs}'
        elif pane_idx == 1:
            coords = f'x={xs}, y pane={ys}, z={zs}'
        elif pane_idx == 2:
            coords = f'x={xs}, y={ys}, z pane={zs}'
        return coords

    def _get_camera_loc(self):
        """
        Returns the current camera location in data coordinates.
        """
        cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()
        c = np.array([cx, cy, cz])
        r = np.array([dx, dy, dz])

        if self._focal_length == np.inf:  # orthographic projection
            focal_length = 1e9  # large enough to be effectively infinite
        else:  # perspective projection
            focal_length = self._focal_length
        eye = c + self._view_w * self._dist * r / self._box_aspect * focal_length
        return eye

    def _calc_coord(self, xv, yv, renderer=None):
        """
        Given the 2D view coordinates, find the point on the nearest axis pane
        that lies directly below those coordinates. Returns a 3D point in data
        coordinates.
        """
        if self._focal_length == np.inf:  # orthographic projection
            zv = 1
        else:  # perspective projection
            zv = -1 / self._focal_length

        # Convert point on view plane to data coordinates
        p1 = np.array(proj3d.inv_transform(xv, yv, zv, self.invM)).ravel()

        # Get the vector from the camera to the point on the view plane
        vec = self._get_camera_loc() - p1

        # Get the pane locations for each of the axes
        pane_locs = []
        for axis in self._axis_map.values():
            xys, loc = axis.active_pane()
            pane_locs.append(loc)

        # Find the distance to the nearest pane by projecting the view vector
        scales = np.zeros(3)
        for i in range(3):
            if vec[i] == 0:
                scales[i] = np.inf
            else:
                scales[i] = (p1[i] - pane_locs[i]) / vec[i]
        pane_idx = np.argmin(abs(scales))
        scale = scales[pane_idx]

        # Calculate the point on the closest pane
        p2 = p1 - scale*vec
        return p2, pane_idx

    def _arcball(self, x: float, y: float) -> np.ndarray:
        """
        Convert a point (x, y) to a point on a virtual trackball.

        This is Ken Shoemake's arcball (a sphere), modified
        to soften the abrupt edge (optionally).
        See: Ken Shoemake, "ARCBALL: A user interface for specifying
        three-dimensional rotation using a mouse." in
        Proceedings of Graphics Interface '92, 1992, pp. 151-156,
        https://doi.org/10.20380/GI1992.18
        The smoothing of the edge is inspired by Gavin Bell's arcball
        (a sphere combined with a hyperbola), but here, the sphere
        is combined with a section of a cylinder, so it has finite support.
        """
        s = mpl.rcParams['axes3d.trackballsize'] / 2
        b = mpl.rcParams['axes3d.trackballborder'] / s
        x /= s
        y /= s
        r2 = x*x + y*y
        r = np.sqrt(r2)
        ra = 1 + b
        a = b * (1 + b/2)
        ri = 2/(ra + 1/ra)
        if r < ri:
            p = np.array([np.sqrt(1 - r2), x, y])
        elif r < ra:
            dr = ra - r
            p = np.array([a - np.sqrt((a + dr) * (a - dr)), x, y])
            p /= np.linalg.norm(p)
        else:
            p = np.array([0, x/r, y/r])
        return p

    def _on_move(self, event):
        """
        Mouse moving.

        By default, button-1 rotates, button-2 pans, and button-3 zooms;
        these buttons can be modified via `mouse_init`.
        """

        if not self.button_pressed:
            return

        if self.get_navigate_mode() is not None:
            # we don't want to rotate if we are zooming/panning
            # from the toolbar
            return

        if self.M is None:
            return

        x, y = event.xdata, event.ydata
        # In case the mouse is out of bounds.
        if x is None or event.inaxes != self:
            return

        dx, dy = x - self._sx, y - self._sy
        w = self._pseudo_w
        h = self._pseudo_h

        # Rotation
        if self.button_pressed in self._rotate_btn:
            # rotate viewing point
            # get the x and y pixel coords
            if dx == 0 and dy == 0:
                return

            style = mpl.rcParams['axes3d.mouserotationstyle']
            if style == 'azel':
                roll = np.deg2rad(self.roll)
                delev = -(dy/h)*180*np.cos(roll) + (dx/w)*180*np.sin(roll)
                dazim = -(dy/h)*180*np.sin(roll) - (dx/w)*180*np.cos(roll)
                elev = self.elev + delev
                azim = self.azim + dazim
                roll = self.roll
            else:
                q = _Quaternion.from_cardan_angles(
                        *np.deg2rad((self.elev, self.azim, self.roll)))

                if style == 'trackball':
                    k = np.array([0, -dy/h, dx/w])
                    nk = np.linalg.norm(k)
                    th = nk / mpl.rcParams['axes3d.trackballsize']
                    dq = _Quaternion(np.cos(th), k*np.sin(th)/nk)
                else:  # 'sphere', 'arcball'
                    current_vec = self._arcball(self._sx/w, self._sy/h)
                    new_vec = self._arcball(x/w, y/h)
                    if style == 'sphere':
                        dq = _Quaternion.rotate_from_to(current_vec, new_vec)
                    else:  # 'arcball'
                        dq = _Quaternion(0, new_vec) * _Quaternion(0, -current_vec)

                q = dq * q
                elev, azim, roll = np.rad2deg(q.as_cardan_angles())

            # update view
            vertical_axis = self._axis_names[self._vertical_axis]
            self.view_init(
                elev=elev,
                azim=azim,
                roll=roll,
                vertical_axis=vertical_axis,
                share=True,
            )
            self.stale = True

        # Pan
        elif self.button_pressed in self._pan_btn:
            # Start the pan event with pixel coordinates
            px, py = self.transData.transform([self._sx, self._sy])
            self.start_pan(px, py, 2)
            # pan view (takes pixel coordinate input)
            self.drag_pan(2, None, event.x, event.y)
            self.end_pan()

        # Zoom
        elif self.button_pressed in self._zoom_btn:
            # zoom view (dragging down zooms in)
            scale = h/(h - dy)
            self._scale_axis_limits(scale, scale, scale)

        # Store the event coordinates for the next time through.
        self._sx, self._sy = x, y
        # Always request a draw update at the end of interaction
        self.get_figure(root=True).canvas.draw_idle()

    def drag_pan(self, button, key, x, y):
        # docstring inherited

        # Get the coordinates from the move event
        p = self._pan_start
        (xdata, ydata), (xdata_start, ydata_start) = p.trans_inverse.transform(
            [(x, y), (p.x, p.y)])
        self._sx, self._sy = xdata, ydata
        # Calling start_pan() to set the x/y of this event as the starting
        # move location for the next event
        self.start_pan(x, y, button)
        du, dv = xdata - xdata_start, ydata - ydata_start
        dw = 0
        if key == 'x':
            dv = 0
        elif key == 'y':
            du = 0
        if du == 0 and dv == 0:
            return

        # Transform the pan from the view axes to the data axes
        R = np.array([self._view_u, self._view_v, self._view_w])
        R = -R / self._box_aspect * self._dist
        duvw_projected = R.T @ np.array([du, dv, dw])

        # Calculate pan distance
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        dx = (maxx - minx) * duvw_projected[0]
        dy = (maxy - miny) * duvw_projected[1]
        dz = (maxz - minz) * duvw_projected[2]

        # Set the new axis limits
        self.set_xlim3d(minx + dx, maxx + dx, auto=None)
        self.set_ylim3d(miny + dy, maxy + dy, auto=None)
        self.set_zlim3d(minz + dz, maxz + dz, auto=None)

    def _calc_view_axes(self, eye):
        """
        Get the unit vectors for the viewing axes in data coordinates.
        `u` is towards the right of the screen
        `v` is towards the top of the screen
        `w` is out of the screen
        """
        elev_rad = np.deg2rad(art3d._norm_angle(self.elev))
        roll_rad = np.deg2rad(art3d._norm_angle(self.roll))

        # Look into the middle of the world coordinates
        R = 0.5 * self._roll_to_vertical(self._box_aspect)

        # Define which axis should be vertical. A negative value
        # indicates the plot is upside down and therefore the values
        # have been reversed:
        V = np.zeros(3)
        V[self._vertical_axis] = -1 if abs(elev_rad) > np.pi/2 else 1

        u, v, w = proj3d._view_axes(eye, R, V, roll_rad)
        return u, v, w

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Zoom in or out of the bounding box.

        Will center the view in the center of the bounding box, and zoom by
        the ratio of the size of the bounding box to the size of the Axes3D.
        """
        (start_x, start_y, stop_x, stop_y) = bbox
        if mode == 'x':
            start_y = self.bbox.min[1]
            stop_y = self.bbox.max[1]
        elif mode == 'y':
            start_x = self.bbox.min[0]
            stop_x = self.bbox.max[0]

        # Clip to bounding box limits
        start_x, stop_x = np.clip(sorted([start_x, stop_x]),
                                  self.bbox.min[0], self.bbox.max[0])
        start_y, stop_y = np.clip(sorted([start_y, stop_y]),
                                  self.bbox.min[1], self.bbox.max[1])

        # Move the center of the view to the center of the bbox
        zoom_center_x = (start_x + stop_x)/2
        zoom_center_y = (start_y + stop_y)/2

        ax_center_x = (self.bbox.max[0] + self.bbox.min[0])/2
        ax_center_y = (self.bbox.max[1] + self.bbox.min[1])/2

        self.start_pan(zoom_center_x, zoom_center_y, 2)
        self.drag_pan(2, None, ax_center_x, ax_center_y)
        self.end_pan()

        # Calculate zoom level
        dx = abs(start_x - stop_x)
        dy = abs(start_y - stop_y)
        scale_u = dx / (self.bbox.max[0] - self.bbox.min[0])
        scale_v = dy / (self.bbox.max[1] - self.bbox.min[1])

        # Keep aspect ratios equal
        scale = max(scale_u, scale_v)

        # Zoom out
        if direction == 'out':
            scale = 1 / scale

        self._zoom_data_limits(scale, scale, scale)

    def _zoom_data_limits(self, scale_u, scale_v, scale_w):
        """
        Zoom in or out of a 3D plot.

        Will scale the data limits by the scale factors. These will be
        transformed to the x, y, z data axes based on the current view angles.
        A scale factor > 1 zooms out and a scale factor < 1 zooms in.

        For an Axes that has had its aspect ratio set to 'equal', 'equalxy',
        'equalyz', or 'equalxz', the relevant axes are constrained to zoom
        equally.

        Parameters
        ----------
        scale_u : float
            Scale factor for the u view axis (view screen horizontal).
        scale_v : float
            Scale factor for the v view axis (view screen vertical).
        scale_w : float
            Scale factor for the w view axis (view screen depth).
        """
        scale = np.array([scale_u, scale_v, scale_w])

        # Only perform frame conversion if unequal scale factors
        if not np.allclose(scale, scale_u):
            # Convert the scale factors from the view frame to the data frame
            R = np.array([self._view_u, self._view_v, self._view_w])
            S = scale * np.eye(3)
            scale = np.linalg.norm(R.T @ S, axis=1)

            # Set the constrained scale factors to the factor closest to 1
            if self._aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
                ax_idxs = self._equal_aspect_axis_indices(self._aspect)
                min_ax_idxs = np.argmin(np.abs(scale[ax_idxs] - 1))
                scale[ax_idxs] = scale[ax_idxs][min_ax_idxs]

        self._scale_axis_limits(scale[0], scale[1], scale[2])

    def _scale_axis_limits(self, scale_x, scale_y, scale_z):
        """
        Keeping the center of the x, y, and z data axes fixed, scale their
        limits by scale factors. A scale factor > 1 zooms out and a scale
        factor < 1 zooms in.

        Parameters
        ----------
        scale_x : float
            Scale factor for the x data axis.
        scale_y : float
            Scale factor for the y data axis.
        scale_z : float
            Scale factor for the z data axis.
        """
        # Get the axis centers and ranges
        cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()

        # Set the scaled axis limits
        self.set_xlim3d(cx - dx*scale_x/2, cx + dx*scale_x/2, auto=None)
        self.set_ylim3d(cy - dy*scale_y/2, cy + dy*scale_y/2, auto=None)
        self.set_zlim3d(cz - dz*scale_z/2, cz + dz*scale_z/2, auto=None)

    def _get_w_centers_ranges(self):
        """Get 3D world centers and axis ranges."""
        # Calculate center of axis limits
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        cx = (maxx + minx)/2
        cy = (maxy + miny)/2
        cz = (maxz + minz)/2

        # Calculate range of axis limits
        dx = (maxx - minx)
        dy = (maxy - miny)
        dz = (maxz - minz)
        return cx, cy, cz, dx, dy, dz

    def set_zlabel(self, zlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set zlabel.  See doc for `.set_ylabel` for description.
        """
        if labelpad is not None:
            self.zaxis.labelpad = labelpad
        return self.zaxis.set_label_text(zlabel, fontdict, **kwargs)

    def get_zlabel(self):
        """
        Get the z-label text string.
        """
        label = self.zaxis.label
        return label.get_text()

    # Axes rectangle characteristics

    # The frame_on methods are not available for 3D axes.
    # Python will raise a TypeError if they are called.
    get_frame_on = None
    set_frame_on = None

    def grid(self, visible=True, **kwargs):
        """
        Set / unset 3D grid.

        .. note::

            Currently, this function does not behave the same as
            `.axes.Axes.grid`, but it is intended to eventually support that
            behavior.
        """
        # TODO: Operate on each axes separately
        if len(kwargs):
            visible = True
        self._draw_grid = visible
        self.stale = True

    def tick_params(self, axis='both', **kwargs):
        """
        Convenience method for changing the appearance of ticks and
        tick labels.

        See `.Axes.tick_params` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.

        Also, because of how Axes3D objects are drawn very differently
        from regular 2D Axes, some of these settings may have
        ambiguous meaning.  For simplicity, the 'z' axis will
        accept settings as if it was like the 'y' axis.

        .. note::
           Axes3D currently ignores some of these settings.
        """
        _api.check_in_list(['x', 'y', 'z', 'both'], axis=axis)
        if axis in ['x', 'y', 'both']:
            super().tick_params(axis, **kwargs)
        if axis in ['z', 'both']:
            zkw = dict(kwargs)
            zkw.pop('top', None)
            zkw.pop('bottom', None)
            zkw.pop('labeltop', None)
            zkw.pop('labelbottom', None)
            self.zaxis.set_tick_params(**zkw)

    # data limits, ticks, tick labels, and formatting

    def invert_zaxis(self):
        """
        Invert the z-axis.

        See Also
        --------
        zaxis_inverted
        get_zlim, set_zlim
        get_zbound, set_zbound
        """
        bottom, top = self.get_zlim()
        self.set_zlim(top, bottom, auto=None)

    zaxis_inverted = _axis_method_wrapper("zaxis", "get_inverted")

    def get_zbound(self):
        """
        Return the lower and upper z-axis bounds, in increasing order.

        See Also
        --------
        set_zbound
        get_zlim, set_zlim
        invert_zaxis, zaxis_inverted
        """
        lower, upper = self.get_zlim()
        if lower < upper:
            return lower, upper
        else:
            return upper, lower

    def text(self, x, y, z, s, zdir=None, *, axlim_clip=False, **kwargs):
        """
        Add the text *s* to the 3D Axes at location *x*, *y*, *z* in data coordinates.

        Parameters
        ----------
        x, y, z : float
            The position to place the text.
        s : str
            The text.
        zdir : {'x', 'y', 'z', 3-tuple}, optional
            The direction to be used as the z-direction. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        axlim_clip : bool, default: False
            Whether to hide text that is outside the axes view limits.

            .. versionadded:: 3.10
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.text`.

        Returns
        -------
        `.Text3D`
            The created `.Text3D` instance.
        """
        text = super().text(x, y, s, **kwargs)
        art3d.text_2d_to_3d(text, z, zdir, axlim_clip)
        return text

    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, zdir='z', axlim_clip=False, **kwargs):
        """
        Plot 2D or 3D data.

        Parameters
        ----------
        xs : 1D array-like
            x coordinates of vertices.
        ys : 1D array-like
            y coordinates of vertices.
        zs : float or 1D array-like
            z coordinates of vertices; either one for all points or one for
            each point.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z.
        axlim_clip : bool, default: False
            Whether to hide data that is outside the axes view limits.

            .. versionadded:: 3.10
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.plot`.
        """
        had_data = self.has_data()

        # `zs` can be passed positionally or as keyword; checking whether
        # args[0] is a string matches the behavior of 2D `plot` (via
        # `_process_plot_var_args`).
        if args and not isinstance(args[0], str):
            zs, *args = args
            if 'zs' in kwargs:
                raise TypeError("plot() for multiple values for argument 'zs'")
        else:
            zs = kwargs.pop('zs', 0)

        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)

        lines = super().plot(xs, ys, *args, **kwargs)
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir, axlim_clip=axlim_clip)

        xs, ys, zs = art3d.juggle_axes(xs, ys, zs, zdir)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines

    plot3D = plot

    def fill_between(self, x1, y1, z1, x2, y2, z2, *,
                     where=None, mode='auto', facecolors=None, shade=None,
                     axlim_clip=False, **kwargs):
        """
        Fill the area between two 3D curves.

        The curves are defined by the points (*x1*, *y1*, *z1*) and
        (*x2*, *y2*, *z2*). This creates one or multiple quadrangle
        polygons that are filled. All points must be the same length N, or a
        single value to be used for all points.

        Parameters
        ----------
        x1, y1, z1 : float or 1D array-like
            x, y, and z  coordinates of vertices for 1st line.

        x2, y2, z2 : float or 1D array-like
            x, y, and z coordinates of vertices for 2nd line.

        where : array of bool (length N), optional
            Define *where* to exclude some regions from being filled. The
            filled regions are defined by the coordinates ``pts[where]``,
            for all x, y, and z pts. More precisely, fill between ``pts[i]``
            and ``pts[i+1]`` if ``where[i] and where[i+1]``. Note that this
            definition implies that an isolated *True* value between two
            *False* values in *where* will not result in filling. Both sides of
            the *True* position remain unfilled due to the adjacent *False*
            values.

        mode : {'quad', 'polygon', 'auto'}, default: 'auto'
            The fill mode. One of:

            - 'quad':  A separate quadrilateral polygon is created for each
              pair of subsequent points in the two lines.
            - 'polygon': The two lines are connected to form a single polygon.
              This is faster and can render more cleanly for simple shapes
              (e.g. for filling between two lines that lie within a plane).
            - 'auto': If the points all lie on the same 3D plane, 'polygon' is
              used. Otherwise, 'quad' is used.

        facecolors : list of :mpltype:`color`, default: None
            Colors of each individual patch, or a single color to be used for
            all patches.

        shade : bool, default: None
            Whether to shade the facecolors. If *None*, then defaults to *True*
            for 'quad' mode and *False* for 'polygon' mode.

        axlim_clip : bool, default: False
            Whether to hide data that is outside the axes view limits.

            .. versionadded:: 3.10

        **kwargs
            All other keyword arguments are passed on to `.Poly3DCollection`.

        Returns
        -------
        `.Poly3DCollection`
            A `.Poly3DCollection` containing the plotted polygons.

        """
        _api.check_in_list(['auto', 'quad', 'polygon'], mode=mode)

        had_data = self.has_data()
        x1, y1, z1, x2, y2, z2 = cbook._broadcast_with_masks(x1, y1, z1, x2, y2, z2)

        if facecolors is None:
            facecolors = [self._get_patches_for_fill.get_next_color()]
        facecolors = list(mcolors.to_rgba_array(facecolors))

        if where is None:
            where = True
        else:
            where = np.asarray(where, dtype=bool)
            if where.size != x1.size:
                raise ValueError(f"where size ({where.size}) does not match "
                                 f"size ({x1.size})")
        where = where & ~np.isnan(x1)  # NaNs were broadcast in _broadcast_with_masks

        if mode == 'auto':
            if art3d._all_points_on_plane(np.concatenate((x1[where], x2[where])),
                                          np.concatenate((y1[where], y2[where])),
                                          np.concatenate((z1[where], z2[where])),
                                          atol=1e-12):
                mode = 'polygon'
            else:
                mode = 'quad'

        if shade is None:
            if mode == 'quad':
                shade = True
            else:
                shade = False

        polys = []
        for idx0, idx1 in cbook.contiguous_regions(where):
            x1i = x1[idx0:idx1]
            y1i = y1[idx0:idx1]
            z1i = z1[idx0:idx1]
            x2i = x2[idx0:idx1]
            y2i = y2[idx0:idx1]
            z2i = z2[idx0:idx1]

            if not len(x1i):
                continue

            if mode == 'quad':
                # Preallocate the array for the region's vertices, and fill it in
                n_polys_i = len(x1i) - 1
                polys_i = np.empty((n_polys_i, 4, 3))
                polys_i[:, 0, :] = np.column_stack((x1i[:-1], y1i[:-1], z1i[:-1]))
                polys_i[:, 1, :] = np.column_stack((x1i[1:], y1i[1:], z1i[1:]))
                polys_i[:, 2, :] = np.column_stack((x2i[1:], y2i[1:], z2i[1:]))
                polys_i[:, 3, :] = np.column_stack((x2i[:-1], y2i[:-1], z2i[:-1]))
                polys = polys + [*polys_i]
            elif mode == 'polygon':
                line1 = np.column_stack((x1i, y1i, z1i))
                line2 = np.column_stack((x2i[::-1], y2i[::-1], z2i[::-1]))
                poly = np.concatenate((line1, line2), axis=0)
                polys.append(poly)

        polyc = art3d.Poly3DCollection(polys, facecolors=facecolors, shade=shade,
                                       axlim_clip=axlim_clip, **kwargs)
        self.add_collection(polyc)

        self.auto_scale_xyz([x1, x2], [y1, y2], [z1, z2], had_data)
        return polyc

    def plot_surface(self, X, Y, Z, *, norm=None, vmin=None,
                     vmax=None, lightsource=None, axlim_clip=False, **kwargs):
        """
        Create a surface plot.

        By default, it will be colored in shades of a solid color, but it also
        supports colormapping by supplying the *cmap* argument.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        .. note::

           To maximize rendering speed consider setting *rstride* and *cstride*
           to divisors of the number of rows minus 1 and columns minus 1
           respectively. For example, given 51 rows rstride can be any of the
           divisors of 50.

           Similarly, a setting of *rstride* and *cstride* equal to 1 (or
           *rcount* and *ccount* equal the number of rows and columns) can use
           the optimized path.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 10.

            'classic' mode uses a default of ``rstride = cstride = 10`` instead
            of the new default of ``rcount = ccount = 50``.

        color : :mpltype:`color`
            Color of the surface patches.

        cmap : Colormap, optional
            Colormap of the surface patches.

        facecolors : list of :mpltype:`color`
            Colors of each individual patch.

        norm : `~matplotlib.colors.Normalize`, optional
            Normalization for the colormap.

        vmin, vmax : float, optional
            Bounds for the normalization.

        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.

        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.

        axlim_clip : bool, default: False
            Whether to hide patches with a vertex outside the axes view limits.

            .. versionadded:: 3.10

        **kwargs
            Other keyword arguments are forwarded to `.Poly3DCollection`.
        """

        had_data = self.has_data()

        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")

        Z = cbook._to_unmasked_float_array(Z)
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 10)
        cstride = kwargs.pop('cstride', 10)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if mpl.rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            compute_strides = has_count
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            compute_strides = not has_stride

        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))

        fcolors = kwargs.pop('facecolors', None)

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        if shade is None:
            raise ValueError("shade cannot be None.")

        colset = []  # the sampled facecolor
        if (rows - 1) % rstride == 0 and \
           (cols - 1) % cstride == 0 and \
           fcolors is None:
            polys = np.stack(
                [cbook._array_patch_perimeters(a, rstride, cstride)
                 for a in (X, Y, Z)],
                axis=-1)
        else:
            # evenly spaced, and including both endpoints
            row_inds = list(range(0, rows-1, rstride)) + [rows-1]
            col_inds = list(range(0, cols-1, cstride)) + [cols-1]

            polys = []
            for rs, rs_next in itertools.pairwise(row_inds):
                for cs, cs_next in itertools.pairwise(col_inds):
                    ps = [
                        # +1 ensures we share edges between polygons
                        cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
                        for a in (X, Y, Z)
                    ]
                    # ps = np.stack(ps, axis=-1)
                    ps = np.array(ps).T
                    polys.append(ps)

                    if fcolors is not None:
                        colset.append(fcolors[rs][cs])

        # In cases where there are non-finite values in the data (possibly NaNs from
        # masked arrays), artifacts can be introduced. Here check whether such values
        # are present and remove them.
        if not isinstance(polys, np.ndarray) or not np.isfinite(polys).all():
            new_polys = []
            new_colset = []

            # Depending on fcolors, colset is either an empty list or has as
            # many elements as polys. In the former case new_colset results in
            # a list with None entries, that is discarded later.
            for p, col in itertools.zip_longest(polys, colset):
                new_poly = np.array(p)[np.isfinite(p).all(axis=1)]
                if len(new_poly):
                    new_polys.append(new_poly)
                    new_colset.append(col)

            # Replace previous polys and, if fcolors is not None, colset
            polys = new_polys
            if fcolors is not None:
                colset = new_colset

        # note that the striding causes some polygons to have more coordinates
        # than others

        if fcolors is not None:
            polyc = art3d.Poly3DCollection(
                polys, edgecolors=colset, facecolors=colset, shade=shade,
                lightsource=lightsource, axlim_clip=axlim_clip, **kwargs)
        elif cmap:
            polyc = art3d.Poly3DCollection(polys, axlim_clip=axlim_clip, **kwargs)
            # can't always vectorize, because polys might be jagged
            if isinstance(polys, np.ndarray):
                avg_z = polys[..., 2].mean(axis=-1)
            else:
                avg_z = np.array([ps[:, 2].mean() for ps in polys])
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            color = kwargs.pop('color', None)
            if color is None:
                color = self._get_lines.get_next_color()
            color = np.array(mcolors.to_rgba(color))

            polyc = art3d.Poly3DCollection(
                polys, facecolors=color, shade=shade, lightsource=lightsource,
                axlim_clip=axlim_clip, **kwargs)

        self.add_collection(polyc)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return polyc

    def plot_wireframe(self, X, Y, Z, *, axlim_clip=False, **kwargs):
        """
        Plot a 3D wireframe.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        axlim_clip : bool, default: False
            Whether to hide lines and patches with vertices outside the axes
            view limits.

            .. versionadded:: 3.10

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Setting a count to zero causes the data to be
            not sampled in the corresponding direction, producing a 3D line
            plot rather than a wireframe plot.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 1.  Setting a
            stride to zero causes the data to be not sampled in the
            corresponding direction, producing a 3D line plot rather than a
            wireframe plot.

            'classic' mode uses a default of ``rstride = cstride = 1`` instead
            of the new default of ``rcount = ccount = 50``.

        **kwargs
            Other keyword arguments are forwarded to `.Line3DCollection`.
        """

        had_data = self.has_data()
        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        # FIXME: Support masked arrays
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 1)
        cstride = kwargs.pop('cstride', 1)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if mpl.rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            if has_count:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            if not has_stride:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0

        # We want two sets of lines, one running along the "rows" of
        # Z and another set of lines running along the "columns" of Z.
        # This transpose will make it easy to obtain the columns.
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        if rstride:
            rii = list(range(0, rows, rstride))
            # Add the last index only if needed
            if rows > 0 and rii[-1] != (rows - 1):
                rii += [rows-1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            # Add the last index only if needed
            if cols > 0 and cii[-1] != (cols - 1):
                cii += [cols-1]
        else:
            cii = []

        if rstride == 0 and cstride == 0:
            raise ValueError("Either rstride or cstride must be non zero")

        # If the inputs were empty, then just
        # reset everything.
        if Z.size == 0:
            rii = []
            cii = []

        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = ([list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(xlines, ylines, zlines)]
                 + [list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(txlines, tylines, tzlines)])

        linec = art3d.Line3DCollection(lines, axlim_clip=axlim_clip, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return linec

    def plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None,
                     lightsource=None, axlim_clip=False, **kwargs):
        """
        Plot a triangulated surface.

        The (optional) triangulation can be specified in one of two ways;
        either::

          plot_trisurf(triangulation, ...)

        where triangulation is a `~matplotlib.tri.Triangulation` object, or::

          plot_trisurf(X, Y, ...)
          plot_trisurf(X, Y, triangles, ...)
          plot_trisurf(X, Y, triangles=triangles, ...)

        in which case a Triangulation object will be created.  See
        `.Triangulation` for an explanation of these possibilities.

        The remaining arguments are::

          plot_trisurf(..., Z)

        where *Z* is the array of values to contour, one per point
        in the triangulation.

        Parameters
        ----------
        X, Y, Z : array-like
            Data values as 1D arrays.
        color
            Color of the surface patches.
        cmap
            A colormap for the surface patches.
        norm : `~matplotlib.colors.Normalize`, optional
            An instance of Normalize to map values to colors.
        vmin, vmax : float, optional
            Minimum and maximum value to map.
        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.
        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.
        axlim_clip : bool, default: False
            Whether to hide patches with a vertex outside the axes view limits.

            .. versionadded:: 3.10
        **kwargs
            All other keyword arguments are passed on to
            :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

        Examples
        --------
        .. plot:: gallery/mplot3d/trisurf3d.py
        .. plot:: gallery/mplot3d/trisurf3d_2.py
        """

        had_data = self.has_data()

        # TODO: Support custom face colours
        if color is None:
            color = self._get_lines.get_next_color()
        color = np.array(mcolors.to_rgba(color))

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)

        tri, args, kwargs = \
            Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        try:
            z = kwargs.pop('Z')
        except KeyError:
            # We do this so Z doesn't get passed as an arg to PolyCollection
            z, *args = args
        z = np.asarray(z)

        triangles = tri.get_masked_triangles()
        xt = tri.x[triangles]
        yt = tri.y[triangles]
        zt = z[triangles]
        verts = np.stack((xt, yt, zt), axis=-1)

        if cmap:
            polyc = art3d.Poly3DCollection(verts, *args,
