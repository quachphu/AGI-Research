                    f'          <dimension name="{axisNames[tag]}" xvalue="{v}"/>',
                    file=f,
                )
            print("        </output>", file=f)
            print("      </mapping>", file=f)
        print("    </mappings>", file=f)
    print("  </axes>", file=f)
    print("</designspace>", file=f)


def main(args=None):
    """Print `avar` table as a designspace snippet."""

    if args is None:
        args = sys.argv[1:]

    from fontTools.ttLib import TTFont
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.avar.unbuild",
        description="Print `avar` table as a designspace snippet.",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    options = parser.parse_args(args)

    font = TTFont(options.font)
    if "fvar" not in font:
        print("Not a variable font.", file=sys.stderr)
        return 1

    unbuild(font)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>from fontTools.varLib.models import supportScalar
from fontTools.misc.fixedTools import MAX_F2DOT14
from functools import lru_cache

__all__ = ["rebaseTent"]

EPSILON = 1 / (1 << 14)


def _reverse_negate(v):
    return (-v[2], -v[1], -v[0])


def _solve(tent, axisLimit, negative=False):
    axisMin, axisDef, axisMax, _distanceNegative, _distancePositive = axisLimit
    lower, peak, upper = tent

    # Mirror the problem such that axisDef <= peak
    if axisDef > peak:
        return [
            (scalar, _reverse_negate(t) if t is not None else None)
            for scalar, t in _solve(
                _reverse_negate(tent),
                axisLimit.reverse_negate(),
                not negative,
            )
        ]
    # axisDef <= peak

    # case 1: The whole deltaset falls outside the new limit; we can drop it
    #
    #                                          peak
    #  1.........................................o..........
    #                                           / \
    #                                          /   \
    #                                         /     \
    #                                        /       \
    #  0---|-----------|----------|-------- o         o----1
    #    axisMin     axisDef    axisMax   lower     upper
    #
    if axisMax <= lower and axisMax < peak:
        return []  # No overlap

    # case 2: Only the peak and outermost bound fall outside the new limit;
    # we keep the deltaset, update peak and outermost bound and and scale deltas
    # by the scalar value for the restricted axis at the new limit, and solve
    # recursively.
    #
    #                                  |peak
    #  1...............................|.o..........
    #                                  |/ \
    #                                  /   \
    #                                 /|    \
    #                                / |     \
    #  0--------------------------- o  |      o----1
    #                           lower  |      upper
    #                                  |
    #                                axisMax
    #
    # Convert to:
    #
    #  1............................................
    #                                  |
    #                                  o peak
    #                                 /|
    #                                /x|
    #  0--------------------------- o  o upper ----1
    #                           lower  |
    #                                  |
    #                                axisMax
    if axisMax < peak:
        mult = supportScalar({"tag": axisMax}, {"tag": tent})
        tent = (lower, axisMax, axisMax)
        return [(scalar * mult, t) for scalar, t in _solve(tent, axisLimit)]

    # lower <= axisDef <= peak <= axisMax

    gain = supportScalar({"tag": axisDef}, {"tag": tent})
    out = [(gain, None)]

    # First, the positive side

    # outGain is the scalar of axisMax at the tent.
    outGain = supportScalar({"tag": axisMax}, {"tag": tent})

    # Case 3a: Gain is more than outGain. The tent down-slope crosses
    # the axis into negative. We have to split it into multiples.
    #
    #                      | peak  |
    #  1...................|.o.....|..............
    #                      |/x\_   |
    #  gain................+....+_.|..............
    #                     /|    |y\|
    #  ................../.|....|..+_......outGain
    #                   /  |    |  | \
    #  0---|-----------o   |    |  |  o----------1
    #    axisMin    lower  |    |  |   upper
    #                      |    |  |
    #                axisDef    |  axisMax
    #                           |
    #                      crossing
    if gain >= outGain:
        # Note that this is the branch taken if both gain and outGain are 0.

        # Crossing point on the axis.
        crossing = peak + (1 - gain) * (upper - peak)

        loc = (max(lower, axisDef), peak, crossing)
        scalar = 1

        # The part before the crossing point.
        out.append((scalar - gain, loc))

        # The part after the crossing point may use one or two tents,
        # depending on whether upper is before axisMax or not, in one
        # case we need to keep it down to eternity.

        # Case 3a1, similar to case 1neg; just one tent needed, as in
        # the drawing above.
        if upper >= axisMax:
            loc = (crossing, axisMax, axisMax)
            scalar = outGain

            out.append((scalar - gain, loc))

        # Case 3a2: Similar to case 2neg; two tents needed, to keep
        # down to eternity.
        #
        #                      | peak             |
        #  1...................|.o................|...
        #                      |/ \_              |
        #  gain................+....+_............|...
        #                     /|    | \xxxxxxxxxxy|
        #                    / |    |  \_xxxxxyyyy|
        #                   /  |    |    \xxyyyyyy|
        #  0---|-----------o   |    |     o-------|--1
        #    axisMin    lower  |    |      upper  |
        #                      |    |             |
        #                axisDef    |             axisMax
        #                           |
        #                      crossing
        else:
            # A tent's peak cannot fall on axis default. Nudge it.
            if upper == axisDef:
                upper += EPSILON

            # Downslope.
            loc1 = (crossing, upper, axisMax)
            scalar1 = 0

            # Eternity justify.
            loc2 = (upper, axisMax, axisMax)
            scalar2 = 0

            out.append((scalar1 - gain, loc1))
            out.append((scalar2 - gain, loc2))

    else:
        # Special-case if peak is at axisMax.
        if axisMax == peak:
            upper = peak

        # Case 3:
        # We keep delta as is and only scale the axis upper to achieve
        # the desired new tent if feasible.
        #
        #                        peak
        #  1.....................o....................
        #                       / \_|
        #  ..................../....+_.........outGain
        #                     /     | \
        #  gain..............+......|..+_.............
        #                   /|      |  | \
        #  0---|-----------o |      |  |  o----------1
        #    axisMin    lower|      |  |   upper
        #                    |      |  newUpper
        #              axisDef      axisMax
        #
        newUpper = peak + (1 - gain) * (upper - peak)
        assert axisMax <= newUpper  # Because outGain > gain
        # Disabled because ots doesn't like us:
        # https://github.com/fonttools/fonttools/issues/3350
        if False and newUpper <= axisDef + (axisMax - axisDef) * 2:
            upper = newUpper
            if not negative and axisDef + (axisMax - axisDef) * MAX_F2DOT14 < upper:
                # we clamp +2.0 to the max F2Dot14 (~1.99994) for convenience
                upper = axisDef + (axisMax - axisDef) * MAX_F2DOT14
                assert peak < upper

            loc = (max(axisDef, lower), peak, upper)
            scalar = 1

            out.append((scalar - gain, loc))

        # Case 4: New limit doesn't fit; we need to chop into two tents,
        # because the shape of a triangle with part of one side cut off
        # cannot be represented as a triangle itself.
        #
        #            |   peak |
        #  1.........|......o.|....................
        #  ..........|...../x\|.............outGain
        #            |    |xxy|\_
        #            |   /xxxy|  \_
        #            |  |xxxxy|    \_
        #            |  /xxxxy|      \_
        #  0---|-----|-oxxxxxx|        o----------1
        #    axisMin | lower  |        upper
        #            |        |
        #          axisDef  axisMax
        #
        else:
            loc1 = (max(axisDef, lower), peak, axisMax)
            scalar1 = 1

            loc2 = (peak, axisMax, axisMax)
            scalar2 = outGain

            out.append((scalar1 - gain, loc1))
            # Don't add a dirac delta!
            if peak < axisMax:
                out.append((scalar2 - gain, loc2))

    # Now, the negative side

    # Case 1neg: Lower extends beyond axisMin: we chop. Simple.
    #
    #                     |   |peak
    #  1..................|...|.o.................
    #                     |   |/ \
    #  gain...............|...+...\...............
    #                     |x_/|    \
    #                     |/  |     \
    #                   _/|   |      \
    #  0---------------o  |   |       o----------1
    #              lower  |   |       upper
    #                     |   |
    #               axisMin   axisDef
    #
    if lower <= axisMin:
        loc = (axisMin, axisMin, axisDef)
        scalar = supportScalar({"tag": axisMin}, {"tag": tent})

        out.append((scalar - gain, loc))

    # Case 2neg: Lower is betwen axisMin and axisDef: we add two
    # tents to keep it down all the way to eternity.
    #
    #      |               |peak
    #  1...|...............|.o.................
    #      |               |/ \
    #  gain|...............+...\...............
    #      |yxxxxxxxxxxxxx/|    \
    #      |yyyyyyxxxxxxx/ |     \
    #      |yyyyyyyyyyyx/  |      \
    #  0---|-----------o   |       o----------1
    #    axisMin    lower  |       upper
    #                      |
    #                    axisDef
    #
    else:
        # A tent's peak cannot fall on axis default. Nudge it.
        if lower == axisDef:
            lower -= EPSILON

        # Downslope.
        loc1 = (axisMin, lower, axisDef)
        scalar1 = 0

        # Eternity justify.
        loc2 = (axisMin, axisMin, lower)
        scalar2 = 0

        out.append((scalar1 - gain, loc1))
        out.append((scalar2 - gain, loc2))

    return out


@lru_cache(128)
def rebaseTent(tent, axisLimit):
    """Given a tuple (lower,peak,upper) "tent" and new axis limits
    (axisMin,axisDefault,axisMax), solves how to represent the tent
    under the new axis configuration.  All values are in normalized
    -1,0,+1 coordinate system. Tent values can be outside this range.

    Return value is a list of tuples. Each tuple is of the form
    (scalar,tent), where scalar is a multipler to multiply any
    delta-sets by, and tent is a new tent for that output delta-set.
    If tent value is None, that is a special deltaset that should
    be always-enabled (called "gain")."""

    axisMin, axisDef, axisMax, _distanceNegative, _distancePositive = axisLimit
    assert -1 <= axisMin <= axisDef <= axisMax <= +1

    lower, peak, upper = tent
    assert -2 <= lower <= peak <= upper <= +2

    assert peak != 0

    sols = _solve(tent, axisLimit)

    n = lambda v: axisLimit.renormalizeValue(v)
    sols = [
        (scalar, (n(v[0]), n(v[1]), n(v[2])) if v is not None else None)
        for scalar, v in sols
        if scalar
    ]

    return sols
<|endoftext|>"""Partially instantiate a variable font.

The module exports an `instantiateVariableFont` function and CLI that allow to
create full instances (i.e. static fonts) from variable fonts, as well as "partial"
variable fonts that only contain a subset of the original variation space.

For example, if you wish to pin the width axis to a given location while also
restricting the weight axis to 400..700 range, you can do:

.. code-block:: sh

    $ fonttools varLib.instancer ./NotoSans-VF.ttf wdth=85 wght=400:700

See `fonttools varLib.instancer --help` for more info on the CLI options.

The module's entry point is the `instantiateVariableFont` function, which takes
a TTFont object and a dict specifying either axis coodinates or (min, max) ranges,
and returns a new TTFont representing either a partial VF, or full instance if all
the VF axes were given an explicit coordinate.

E.g. here's how to pin the wght axis at a given location in a wght+wdth variable
font, keeping only the deltas associated with the wdth axis:
.. code-block:: pycon

    >>>
    >> from fontTools import ttLib
    >> from fontTools.varLib import instancer
    >> varfont = ttLib.TTFont("path/to/MyVariableFont.ttf")
    >> [a.axisTag for a in varfont["fvar"].axes]  # the varfont's current axes
    ['wght', 'wdth']
    >> partial = instancer.instantiateVariableFont(varfont, {"wght": 300})
    >> [a.axisTag for a in partial["fvar"].axes]  # axes left after pinning 'wght'
    ['wdth']

If the input location specifies all the axes, the resulting instance is no longer
'variable' (same as using fontools varLib.mutator):
.. code-block:: pycon

    >>>
    >> instance = instancer.instantiateVariableFont(
    ...     varfont, {"wght": 700, "wdth": 67.5}
    ... )
    >> "fvar" not in instance
    True

If one just want to drop an axis at the default location, without knowing in
advance what the default value for that axis is, one can pass a `None` value:
.. code-block:: pycon

    >>>
    >> instance = instancer.instantiateVariableFont(varfont, {"wght": None})
    >> len(varfont["fvar"].axes)
    1

From the console script, this is equivalent to passing `wght=drop` as input.

This module is similar to fontTools.varLib.mutator, which it's intended to supersede.
Note that, unlike varLib.mutator, when an axis is not mentioned in the input
location, by default the varLib.instancer will keep the axis and the corresponding
deltas, whereas mutator implicitly drops the axis at its default coordinate.
To obtain the same behavior as mutator, pass the `static=True` parameter or
the `--static` CLI option.

The module supports all the following "levels" of instancing, which can of
course be combined:

L1
    dropping one or more axes while leaving the default tables unmodified;
    .. code-block:: pycon

        >>>
        >> font = instancer.instantiateVariableFont(varfont, {"wght": None})

L2
    dropping one or more axes while pinning them at non-default locations;
    .. code-block:: pycon

        >>>
        >> font = instancer.instantiateVariableFont(varfont, {"wght": 700})

L3
    restricting the range of variation of one or more axes, by setting either
    a new minimum or maximum, potentially -- though not necessarily -- dropping
    entire regions of variations that fall completely outside this new range.
    .. code-block:: pycon

        >>>
        >> font = instancer.instantiateVariableFont(varfont, {"wght": (100, 300)})

L4
    moving the default location of an axis, by specifying (min,defalt,max) values:
    .. code-block:: pycon

        >>>
        >> font = instancer.instantiateVariableFont(varfont, {"wght": (100, 300, 700)})

Both TrueType-flavored (glyf+gvar) variable and CFF2 variable fonts are supported.
"""

from fontTools.misc.fixedTools import (
    floatToFixedToFloat,
    strToFixedToFloat,
    otRound,
)
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap, VariationModel
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib

# we import the `subset` module because we use the `prune_lookups` method on the GSUB
# table class, and that method is only defined dynamically upon importing `subset`
from fontTools import subset  # noqa: F401
from fontTools.cffLib import privateDictOperators2
from fontTools.cffLib.specializer import (
    programToCommands,
    commandsToProgram,
    specializeCommands,
    generalizeCommands,
)
from fontTools.cffLib.CFF2ToCFF import convertCFF2ToCFF
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from fontTools.varLib.varStore import NO_VARIATION_INDEX
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
from fontTools.ttLib.tables.otTables import VarComponentFlags
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
import io
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings

log = logging.getLogger("fontTools.varLib.instancer")


def AxisRange(minimum, maximum):
    warnings.warn(
        "AxisRange is deprecated; use AxisTriple instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return AxisTriple(minimum, None, maximum)


def NormalizedAxisRange(minimum, maximum):
    warnings.warn(
        "NormalizedAxisRange is deprecated; use AxisTriple instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return NormalizedAxisTriple(minimum, None, maximum)


@dataclasses.dataclass(frozen=True, order=True, repr=False)
class AxisTriple(Sequence):
    """A triple of (min, default, max) axis values.

    Any of the values can be None, in which case the limitRangeAndPopulateDefaults()
    method can be used to fill in the missing values based on the fvar axis values.
    """

    minimum: Optional[float]
    default: Optional[float]
    maximum: Optional[float]

    def __post_init__(self):
        if self.default is None and self.minimum == self.maximum:
            object.__setattr__(self, "default", self.minimum)
        if (
            (
                self.minimum is not None
                and self.default is not None
                and self.minimum > self.default
            )
            or (
                self.default is not None
                and self.maximum is not None
                and self.default > self.maximum
            )
            or (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            )
        ):
            raise ValueError(
                f"{type(self).__name__} minimum ({self.minimum}), default ({self.default}), maximum ({self.maximum}) must be in sorted order"
            )

    def __getitem__(self, i):
        fields = dataclasses.fields(self)
        return getattr(self, fields[i].name)

    def __len__(self):
        return len(dataclasses.fields(self))

    def _replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def __repr__(self):
        return (
            f"({', '.join(format(v, 'g') if v is not None else 'None' for v in self)})"
        )

    @classmethod
    def expand(
        cls,
        v: Union[
            "AxisTriple",
            float,  # pin axis at single value, same as min==default==max
            Tuple[float, float],  # (min, max), restrict axis and keep default
            Tuple[float, float, float],  # (min, default, max)
        ],
    ) -> "AxisTriple":
        """Convert a single value or a tuple into an AxisTriple.

        If the input is a single value, it is interpreted as a pin at that value.
        If the input is a tuple, it is interpreted as (min, max) or (min, default, max).
        """
        if isinstance(v, cls):
            return v
        if isinstance(v, (int, float)):
            return cls(v, v, v)
        try:
            n = len(v)
        except TypeError as e:
            raise ValueError(
                f"expected float, 2- or 3-tuple of floats; got {type(v)}: {v!r}"
            ) from e
        default = None
        if n == 2:
            minimum, maximum = v
        elif n >= 3:
            return cls(*v)
        else:
            raise ValueError(f"expected sequence of 2 or 3; got {n}: {v!r}")
        return cls(minimum, default, maximum)

    def limitRangeAndPopulateDefaults(self, fvarTriple) -> "AxisTriple":
        """Return a new AxisTriple with the default value filled in.

        Set default to fvar axis default if the latter is within the min/max range,
        otherwise set default to the min or max value, whichever is closer to the
        fvar axis default.
        If the default value is already set, return self.
        """
        minimum = self.minimum
        if minimum is None:
            minimum = fvarTriple[0]
        default = self.default
        if default is None:
            default = fvarTriple[1]
        maximum = self.maximum
        if maximum is None:
            maximum = fvarTriple[2]

        minimum = max(minimum, fvarTriple[0])
        maximum = max(maximum, fvarTriple[0])
        minimum = min(minimum, fvarTriple[2])
        maximum = min(maximum, fvarTriple[2])
        default = max(minimum, min(maximum, default))

        return AxisTriple(minimum, default, maximum)


@dataclasses.dataclass(frozen=True, order=True, repr=False)
class NormalizedAxisTriple(AxisTriple):
    """A triple of (min, default, max) normalized axis values."""

    minimum: float
    default: float
    maximum: float

    def __post_init__(self):
        if self.default is None:
            object.__setattr__(self, "default", max(self.minimum, min(self.maximum, 0)))
        if not (-1.0 <= self.minimum <= self.default <= self.maximum <= 1.0):
            raise ValueError(
                "Normalized axis values not in -1..+1 range; got "
                f"minimum={self.minimum:g}, default={self.default:g}, maximum={self.maximum:g})"
            )


@dataclasses.dataclass(frozen=True, order=True, repr=False)
class NormalizedAxisTripleAndDistances(AxisTriple):
    """A triple of (min, default, max) normalized axis values,
    with distances between min and default, and default and max,
    in the *pre-normalized* space."""

    minimum: float
    default: float
    maximum: float
    distanceNegative: Optional[float] = 1
    distancePositive: Optional[float] = 1

    def __post_init__(self):
        if self.default is None:
            object.__setattr__(self, "default", max(self.minimum, min(self.maximum, 0)))
        if not (-1.0 <= self.minimum <= self.default <= self.maximum <= 1.0):
            raise ValueError(
                "Normalized axis values not in -1..+1 range; got "
                f"minimum={self.minimum:g}, default={self.default:g}, maximum={self.maximum:g})"
            )

    def reverse_negate(self):
        v = self
        return self.__class__(-v[2], -v[1], -v[0], v[4], v[3])

    def renormalizeValue(self, v, extrapolate=True):
        """Renormalizes a normalized value v to the range of this axis,
        considering the pre-normalized distances as well as the new
        axis limits."""

        lower, default, upper, distanceNegative, distancePositive = self
        assert lower <= default <= upper

        if not extrapolate:
            v = max(lower, min(upper, v))

        if v == default:
            return 0

        if default < 0:
            return -self.reverse_negate().renormalizeValue(-v, extrapolate=extrapolate)

        # default >= 0 and v != default

        if v > default:
            return (v - default) / (upper - default)

        # v < default

        if lower >= 0:
            return (v - default) / (default - lower)

        # lower < 0 and v < default

        totalDistance = distanceNegative * -lower + distancePositive * default

        if v >= 0:
            vDistance = (default - v) * distancePositive
        else:
            vDistance = -v * distanceNegative + distancePositive * default

        return -vDistance / totalDistance


class _BaseAxisLimits(Mapping[str, AxisTriple]):
    def __getitem__(self, key: str) -> AxisTriple:
        return self._data[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __str__(self) -> str:
        return str(self._data)

    def defaultLocation(self) -> Dict[str, float]:
        """Return a dict of default axis values."""
        return {k: v.default for k, v in self.items()}

    def pinnedLocation(self) -> Dict[str, float]:
        """Return a location dict with only the pinned axes."""
        return {k: v.default for k, v in self.items() if v.minimum == v.maximum}


class AxisLimits(_BaseAxisLimits):
    """Maps axis tags (str) to AxisTriple values."""

    def __init__(self, *args, **kwargs):
        self._data = data = {}
        for k, v in dict(*args, **kwargs).items():
            if v is None:
                # will be filled in by limitAxesAndPopulateDefaults
                data[k] = v
            else:
                try:
                    triple = AxisTriple.expand(v)
                except ValueError as e:
                    raise ValueError(f"Invalid axis limits for {k!r}: {v!r}") from e
                data[k] = triple

    def limitAxesAndPopulateDefaults(self, varfont) -> "AxisLimits":
        """Return a new AxisLimits with defaults filled in from fvar table.

        If all axis limits already have defaults, return self.
        """
        fvar = varfont["fvar"]
        fvarTriples = {
            a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes
        }
        newLimits = {}
        for axisTag, triple in self.items():
            fvarTriple = fvarTriples[axisTag]
            default = fvarTriple[1]
            if triple is None:
                newLimits[axisTag] = AxisTriple(default, default, default)
            else:
                newLimits[axisTag] = triple.limitRangeAndPopulateDefaults(fvarTriple)
        return type(self)(newLimits)

    def normalize(self, varfont, usingAvar=True) -> "NormalizedAxisLimits":
        """Return a new NormalizedAxisLimits with normalized -1..0..+1 values.

        If usingAvar is True, the avar table is used to warp the default normalization.
        """
        fvar = varfont["fvar"]
        badLimits = set(self.keys()).difference(a.axisTag for a in fvar.axes)
        if badLimits:
            raise ValueError("Cannot limit: {} not present in fvar".format(badLimits))

        axes = {
            a.axisTag: (a.minValue, a.defaultValue, a.maxValue)
            for a in fvar.axes
            if a.axisTag in self
        }

        if usingAvar and "avar" in varfont:
            avar = varfont["avar"]

            if getattr(avar, "majorVersion", 1) >= 2 and set(
                self.pinnedLocation()
            ).issuperset(a.axisTag for a in fvar.axes):
                # Full instancing of avar2 font (all axes pinned).
                # Use avar table to normalize location and return.
                location = self.pinnedLocation()
                location = {
                    tag: normalize(value, axes[tag], None)
                    for tag, value in location.items()
                }
                return NormalizedAxisLimits(
                    **avar.renormalizeLocation(location, varfont, dropZeroes=False)
                )
            # else: partial instancing of an avar2 font falls through to the
            # standard v1 normalization below, which yields intermediate-space
            # limits (post-avar v1, pre-avar v2); the avar v2 offset
            # compensation in the instancer handles the rest.

        # Note: normalized values are intentionally NOT quantized here; the
        # F2Dot14 quantization happens once, after the avar v1 mapping (in
        # renormalizeAxisLimits), so pre-map rounding doesn't compound drift.
        normalizedLimits = {}

        for axis_tag, fvarTriple in axes.items():
            minUser, defaultUser, maxUser = fvarTriple
            distanceNegative = defaultUser - minUser
            distancePositive = maxUser - defaultUser

            if self[axis_tag] is None:  # Drop
                normalizedLimits[axis_tag] = NormalizedAxisTripleAndDistances(
                    0, 0, 0, distanceNegative, distancePositive
                )
                continue

            minV, defaultV, maxV = self[axis_tag]

            if defaultV is None:
                defaultV = fvarTriple[1]

            normalizedLimits[axis_tag] = NormalizedAxisTripleAndDistances(
                *(normalizeValue(v, fvarTriple) for v in (minV, defaultV, maxV)),
                distanceNegative,
                distancePositive,
            )

        if usingAvar and "avar" in varfont:
            normalizedLimits = varfont["avar"].renormalizeAxisLimits(
                normalizedLimits, varfont
            )
        else:
            # Quantize to F2Dot14, to avoid surprise interpolations.
            normalizedLimits = {
                tag: tuple(floatToFixedToFloat(v, 14) for v in tuple(triple)[:3])
                + tuple(triple)[3:]
                for tag, triple in normalizedLimits.items()
            }

        return NormalizedAxisLimits(normalizedLimits)


class NormalizedAxisLimits(_BaseAxisLimits):
    """Maps axis tags (str) to NormalizedAxisTriple values."""

    def __init__(self, *args, **kwargs):
        self._data = data = {}
        for k, v in dict(*args, **kwargs).items():
            try:
                triple = NormalizedAxisTripleAndDistances.expand(v)
            except ValueError as e:
                raise ValueError(f"Invalid axis limits for {k!r}: {v!r}") from e
            data[k] = triple


class OverlapMode(IntEnum):
    KEEP_AND_DONT_SET_FLAGS = 0
    KEEP_AND_SET_FLAGS = 1
    REMOVE = 2
    REMOVE_AND_IGNORE_ERRORS = 3


def instantiateVARC(varfont, axisLimits):
    log.info("Instantiating VARC tables")

    # TODO(behdad) My confidence in this function is rather low;
    # It needs more testing. Specially with partial-instancing,
    # I don't think it currently works.

    varc = varfont["VARC"].table
    fvarAxes = varfont["fvar"].axes if "fvar" in varfont else []

    location = axisLimits.pinnedLocation()
    axisMap = [i for i, axis in enumerate(fvarAxes) if axis.axisTag not in location]
    reverseAxisMap = {i: j for j, i in enumerate(axisMap)}

    if varc.AxisIndicesList:
        axisIndicesList = varc.AxisIndicesList.Item
        for i, axisIndices in enumerate(axisIndicesList):
            if any(fvarAxes[j].axisTag in axisLimits for j in axisIndices):
                raise NotImplementedError(
                    "Instancing across VarComponent axes is not supported."
                )
            axisIndicesList[i] = [reverseAxisMap[j] for j in axisIndices]

    store = varc.MultiVarStore
    if store:
        for region in store.SparseVarRegionList.Region:
            newRegionAxis = []
            for regionRecord in region.SparseVarRegionAxis:
                tag = fvarAxes[regionRecord.AxisIndex].axisTag
                if tag in axisLimits:
                    raise NotImplementedError(
                        "Instancing across VarComponent axes is not supported."
                    )
                regionRecord.AxisIndex = reverseAxisMap[regionRecord.AxisIndex]


def instantiateTupleVariationStore(
    variations, axisLimits, origCoords=None, endPts=None
):
    """Instantiate TupleVariation list at the given location, or limit axes' min/max.

    The 'variations' list of TupleVariation objects is modified in-place.
    The 'axisLimits' (dict) maps axis tags (str) to NormalizedAxisTriple namedtuples
    specifying (minimum, default, maximum) in the -1,0,+1 normalized space. Pinned axes
    have minimum == default == maximum.

    A 'full' instance (i.e. static font) is produced when all the axes are pinned to
    single coordinates; a 'partial' instance (i.e. a less variable font) is produced
    when some of the axes are omitted, or restricted with a new range.

    Tuples that do not participate are kept as they are. Those that have 0 influence
    at the given location are removed from the variation store.
    Those that are fully instantiated (i.e. all their axes are being pinned) are also
    removed from the variation store, their scaled deltas accummulated and returned, so
    that they can be added by the caller to the default instance's coordinates.
    Tuples that are only partially instantiated (i.e. not all the axes that they
    participate in are being pinned) are kept in the store, and their deltas multiplied
    by the scalar support of the axes to be pinned at the desired location.

    Args:
        variations: List[TupleVariation] from either 'gvar' or 'cvar'.
        axisLimits: NormalizedAxisLimits: map from axis tags to (min, default, max)
            normalized coordinates for the full or partial instance.
        origCoords: GlyphCoordinates: default instance's coordinates for computing 'gvar'
            inferred points (cf. table__g_l_y_f._getCoordinatesAndControls).
        endPts: List[int]: indices of contour end points, for inferring 'gvar' deltas.

    Returns:
        List[float]: the overall delta adjustment after applicable deltas were summed.
    """

    newVariations = changeTupleVariationsAxisLimits(variations, axisLimits)

    mergedVariations = collections.OrderedDict()
    for var in newVariations:
        # compute inferred deltas only for gvar ('origCoords' is None for cvar)
        if origCoords is not None:
            var.calcInferredDeltas(origCoords, endPts)

        # merge TupleVariations with overlapping "tents"
        axes = frozenset(var.axes.items())
        if axes in mergedVariations:
            mergedVariations[axes] += var
        else:
            mergedVariations[axes] = var

    # drop TupleVariation if all axes have been pinned (var.axes.items() is empty);
    # its deltas will be added to the default instance's coordinates
    defaultVar = mergedVariations.pop(frozenset(), None)

    for var in mergedVariations.values():
        var.roundDeltas()
    variations[:] = list(mergedVariations.values())

    return defaultVar.coordinates if defaultVar is not None else []


def changeTupleVariationsAxisLimits(variations, axisLimits):
    for axisTag, axisLimit in sorted(axisLimits.items()):
        newVariations = []
        for var in variations:
            newVariations.extend(changeTupleVariationAxisLimit(var, axisTag, axisLimit))
        variations = newVariations
    return variations


def changeTupleVariationAxisLimit(var, axisTag, axisLimit):
    assert isinstance(axisLimit, NormalizedAxisTripleAndDistances)

    # Skip when current axis is missing or peaks at 0 (i.e. doesn't participate)
    lower, peak, upper = var.axes.get(axisTag, (-1, 0, 1))
    if peak == 0:
        # explicitly defined, no-op axes can be omitted
        # https://github.com/fonttools/fonttools/issues/3453
        if axisTag in var.axes:
            del var.axes[axisTag]
        return [var]
    # Drop if the var 'tent' isn't well-formed
    if not (lower <= peak <= upper) or (lower < 0 and upper > 0):
        return []

    if axisTag not in var.axes:
        return [var]

    tent = var.axes[axisTag]

    solutions = solver.rebaseTent(tent, axisLimit)

    out = []
    for scalar, tent in solutions:
        newVar = (
            TupleVariation(var.axes, var.coordinates) if len(solutions) > 1 else var
        )
        if tent is None:
            newVar.axes.pop(axisTag)
        else:
            assert tent[1] != 0, tent
            newVar.axes[axisTag] = tent
        newVar *= scalar
        out.append(newVar)

    return out


def instantiateCFF2(
    varfont,
    axisLimits,
    *,
    round=round,
    specialize=True,
    generalize=False,
    downgrade=False,
    reachableRanges=None,
):
    # The algorithm here is rather simple:
    #
    # Take all blend operations and store their deltas in the (otherwise empty)
    # CFF2 VarStore. Then, instantiate the VarStore with the given axis limits,
    # and read back the new deltas. This is done for both the CharStrings and
    # the Private dicts.
    #
    # Then prune unused things and possibly drop the VarStore if it's empty.
    #
    # If the downgrade parameter is True, no actual downgrading is done, but
    # the function returns True if the VarStore was empty after instantiation,
    # and hence a downgrade to CFF is possible. In all other cases it returns
    # False.

    log.info("Instantiating CFF2 table")

    fvarAxes = varfont["fvar"].axes

    cff = varfont["CFF2"].cff
    topDict = cff.topDictIndex[0]
    varStore = getattr(topDict, "VarStore", None)
    if varStore is not None:
        varStore = varStore.otVarStore
    if not varStore:
        if downgrade:
            from fontTools.cffLib.CFF2ToCFF import convertCFF2ToCFF

            convertCFF2ToCFF(varfont)
        return

    cff.desubroutinize()

    def getNumRegions(vsindex):
        return varStore.VarData[vsindex if vsindex is not None else 0].VarRegionCount

    charStrings = topDict.CharStrings.values()

    # Gather all unique private dicts
    uniquePrivateDicts = set()
    privateDicts = []
    for fd in topDict.FDArray:
        if fd.Private not in uniquePrivateDicts:
            uniquePrivateDicts.add(fd.Private)
            privateDicts.append(fd.Private)

    allCommands = []
    allCommandPrivates = []
    for cs in charStrings:
        assert cs.private.vstore.otVarStore is varStore  # Or in many places!!
        commands = programToCommands(cs.program, getNumRegions=getNumRegions)
        if generalize:
            commands = generalizeCommands(commands)
        if specialize:
            commands = specializeCommands(commands, generalizeFirst=not generalize)
        allCommands.append(commands)
        allCommandPrivates.append(cs.private)

    def storeBlendsToVarStore(arg):
        if not isinstance(arg, list):
            return

        if any(isinstance(subarg, list) for subarg in arg[:-1]):
            raise NotImplementedError("Nested blend lists not supported (yet)")

        count = arg[-1]
        assert (len(arg) - 1) % count == 0
        nRegions = (len(arg) - 1) // count - 1
        assert nRegions == getNumRegions(vsindex)
        for i in range(count, len(arg) - 1, nRegions):
            deltas = arg[i : i + nRegions]
            assert len(deltas) == nRegions
            varData = varStore.VarData[vsindex]
            varData.Item.append(deltas)
            varData.ItemCount += 1

    def fetchBlendsFromVarStore(arg):
        if not isinstance(arg, list):
            return [arg]

        if any(isinstance(subarg, list) for subarg in arg[:-1]):
            raise NotImplementedError("Nested blend lists not supported (yet)")

        count = arg[-1]
        assert (len(arg) - 1) % count == 0
        numRegions = getNumRegions(vsindex)
        newDefaults = []
        newDeltas = []
        for i in range(count):
            defaultValue = arg[i]

            major = vsindex
            minor = varDataCursor[major]
            varDataCursor[major] += 1

            defaultValue += round(defaultDeltas[major][minor])
            newDefaults.append(defaultValue)

            varData = varStore.VarData[major]
            deltas = varData.Item[minor]
            assert len(deltas) == numRegions
            newDeltas.extend(deltas)

        if not numRegions:
            return newDefaults  # No deltas, just return the defaults

        return [newDefaults + newDeltas + [count]]

    # Check VarData's are empty
    for varData in varStore.VarData:
        assert varData.Item == []
        assert varData.ItemCount == 0

    # Add charstring blend lists to VarStore so we can instantiate them
    for commands, private in zip(allCommands, allCommandPrivates):
        vsindex = getattr(private, "vsindex", 0)
        for command in commands:
            if command[0] == "vsindex":
                vsindex = command[1][0]
                continue
            for arg in command[1]:
                storeBlendsToVarStore(arg)

    # Add private blend lists to VarStore so we can instantiate values
    for private in privateDicts:
        vsindex = getattr(private, "vsindex", 0)
        for opcode, name, arg_type, default, converter in privateDictOperators2:
            if arg_type not in ("number", "delta", "array") or name == "vsindex":
                continue
            if not hasattr(private, name):
                continue

            values = getattr(private, name)
            if arg_type == "number":
                values = [values]

            for value in values:
                if not isinstance(value, list):
                    continue

                assert len(value) % (getNumRegions(vsindex) + 1) == 0
                count = len(value) // (getNumRegions(vsindex) + 1)
                storeBlendsToVarStore(value + [count])

    # Instantiate VarStore
    defaultDeltas = instantiateItemVariationStore(
        varStore, fvarAxes, axisLimits, hierarchical=True
    )

    # Optionally cull dead regions (for avar2 partial instancing)
    if reachableRanges:
        _cullItemVariationStore(varStore, fvarAxes, reachableRanges)

    # Read back new charstring blends from the instantiated VarStore
    varDataCursor = [0] * len(varStore.VarData)
    for commands, private in zip(allCommands, allCommandPrivates):
        vsindex = getattr(private, "vsindex", 0)
        for command in commands:
            if command[0] == "vsindex":
                vsindex = command[1][0]
                continue
            newArgs = []
            for arg in command[1]:
                newArgs.extend(fetchBlendsFromVarStore(arg))
            command[1][:] = newArgs

    # Read back new private blends from the instantiated VarStore
    for private in privateDicts:
        vsindex = getattr(private, "vsindex", 0)
        for opcode, name, arg_type, default, converter in privateDictOperators2:
            if arg_type not in ("number", "delta", "array") or name == "vsindex":
                continue
            if not hasattr(private, name):
                continue

            values = getattr(private, name)
            if arg_type == "number":
                values = [values]

            newValues = []
            for value in values:
                if not isinstance(value, list):
                    newValues.append(value)
                    continue

                value.append(1)
                value = fetchBlendsFromVarStore(value)
                newValues.extend(v[:-1] if isinstance(v, list) else v for v in value)

            if arg_type == "number":
                newValues = newValues[0]

            setattr(private, name, newValues)

    # Empty out the VarStore
    for i, varData in enumerate(varStore.VarData):
        assert varDataCursor[i] == varData.ItemCount, (
            varDataCursor[i],
            varData.ItemCount,
        )
        varData.Item = []
        varData.ItemCount = 0

    # Collect surviving vsindexes
    usedVsindex = set(
        i for i in range(len(varStore.VarData)) if varStore.VarData[i].VarRegionCount
    )
    # Remove vsindex commands that are no longer needed
    for commands, private in zip(allCommands, allCommandPrivates):
        if not any(isinstance(arg, list) for command in commands for arg in command[1]):
            commands[:] = [command for command in commands if command[0] != "vsindex"]

    # Remove unused VarData and update vsindex values
    vsindexMapping = {v: i for i, v in enumerate(sorted(usedVsindex))}
    varStore.VarData = [
        varData for i, varData in enumerate(varStore.VarData) if i in usedVsindex
    ]
    varStore.VarDataCount = len(varStore.VarData)
    for commands in allCommands:
        for command in commands:
            if command[0] == "vsindex":
                command[1][0] = vsindexMapping[command[1][0]]
    for private in privateDicts:
        if hasattr(private, "vsindex"):
            if private.vsindex in vsindexMapping:
                private.vsindex = vsindexMapping[private.vsindex]
            else:
                # The referenced VarData lost all its regions; the dict's
                # blends have dissolved into plain values.
                del private.vsindex

    # Remove initial vsindex commands that are implied
    for commands, private in zip(allCommands, allCommandPrivates):
        vsindex = getattr(private, "vsindex", 0)
        if commands and commands[0] == ("vsindex", [vsindex]):
            commands.pop(0)

    # Ship the charstrings!
    for cs, commands in zip(charStrings, allCommands):
        cs.program = commandsToProgram(commands)

    # Remove empty VarStore
    if not varStore.VarData:
        if "VarStore" in topDict.rawDict:
            del topDict.rawDict["VarStore"]
        del topDict.VarStore
        del topDict.CharStrings.varStore
        for private in privateDicts:
            del private.vstore

        if downgrade:
            return True

    return False


def _instantiateGvarGlyph(
    glyphname, glyf, gvar, hMetrics, vMetrics, axisLimits, optimize=True
):
    coordinates, ctrl = glyf._getCoordinatesAndControls(glyphname, hMetrics, vMetrics)
    endPts = ctrl.endPts

    # Not every glyph may have variations
    tupleVarStore = gvar.variations.get(glyphname)

    if tupleVarStore:
        defaultDeltas = instantiateTupleVariationStore(
            tupleVarStore, axisLimits, coordinates, endPts
        )

        if defaultDeltas:
            coordinates += _g_l_y_f.GlyphCoordinates(defaultDeltas)

    # _setCoordinates also sets the hmtx/vmtx advance widths and sidebearings from
    # the four phantom points and glyph bounding boxes.
    # We call it unconditionally even if a glyph has no variations or no deltas are
    # applied at this location, in case the glyph's xMin and in turn its sidebearing
    # have changed. E.g. a composite glyph has no deltas for the component's (x, y)
    # offset nor for the 4 phantom points (e.g. it's monospaced). Thus its entry in
    # gvar table is empty; however, the composite's base glyph may have deltas
    # applied, hence the composite's bbox and left/top sidebearings may need updating
    # in the instanced font.
    glyf._setCoordinates(glyphname, coordinates, hMetrics, vMetrics)

    if not tupleVarStore:
        if glyphname in gvar.variations:
            del gvar.variations[glyphname]
        return

    if optimize:
        # IUP semantics depend on point equality, and so round prior to
        # optimization to ensure that comparisons that happen now will be the
        # same as those that happen at render time. This is especially needed
        # when floating point deltas have been applied to the default position.
        #     See https://github.com/fonttools/fonttools/issues/3634
        # Rounding must happen only after calculating glyf metrics above, to
        # preserve backwards compatibility.
        #     See 0010a3cd9aa25f84a3a6250dafb119743d32aa40
        coordinates.toInt()
        for var in tupleVarStore:
            var.optimize(coordinates, endPts)


def instantiateGvarGlyph(varfont, glyphname, axisLimits, optimize=True):
    """Remove?
    https://github.com/fonttools/fonttools/pull/2266"""
    gvar = varfont["gvar"]
    glyf = varfont["glyf"]
    hMetrics = varfont["hmtx"].metrics
    vMetrics = getattr(varfont.get("vmtx"), "metrics", None)
    _instantiateGvarGlyph(
        glyphname, glyf, gvar, hMetrics, vMetrics, axisLimits, optimize=optimize
    )


def instantiateGvar(varfont, axisLimits, optimize=True):
    log.info("Instantiating glyf/gvar tables")

    gvar = varfont["gvar"]
    glyf = varfont["glyf"]
    hMetrics = varfont["hmtx"].metrics
    vMetrics = getattr(varfont.get("vmtx"), "metrics", None)
    # Get list of glyph names sorted by component depth.
    # If a composite glyph is processed before its base glyph, the bounds may
    # be calculated incorrectly because deltas haven't been applied to the
    # base glyph yet.
    glyphnames = sorted(
        glyf.glyphOrder,
        key=lambda name: (
            (
                glyf[name].getCompositeMaxpValues(glyf).maxComponentDepth
                if glyf[name].isComposite()
                else 0
            ),
            name,
        ),
    )
    for glyphname in glyphnames:
        _instantiateGvarGlyph(
            glyphname, glyf, gvar, hMetrics, vMetrics, axisLimits, optimize=optimize
        )

    if not gvar.variations:
        del varfont["gvar"]


def setCvarDeltas(cvt, deltas):
    for i, delta in enumerate(deltas):
        if delta:
            cvt[i] += otRound(delta)


def instantiateCvar(varfont, axisLimits):
    log.info("Instantiating cvt/cvar tables")

    cvar = varfont["cvar"]

    defaultDeltas = instantiateTupleVariationStore(cvar.variations, axisLimits)

    if defaultDeltas:
        setCvarDeltas(varfont["cvt "], defaultDeltas)

    if not cvar.variations:
        del varfont["cvar"]


def setMvarDeltas(varfont, deltas):
    mvar = varfont["MVAR"].table
    records = mvar.ValueRecord
    for rec in records:
        mvarTag = rec.ValueTag
        if mvarTag not in MVAR_ENTRIES:
            continue
        tableTag, itemName = MVAR_ENTRIES[mvarTag]
        delta = deltas[rec.VarIdx]
        if delta != 0:
            setattr(
                varfont[tableTag],
                itemName,
                getattr(varfont[tableTag], itemName) + otRound(delta),
            )


@contextmanager
def verticalMetricsKeptInSync(varfont):
    """Ensure hhea vertical metrics stay in sync with OS/2 ones after instancing.

    When applying MVAR deltas to the OS/2 table, if the ascender, descender and
    line gap change but they were the same as the respective hhea metrics in the
    original font, this context manager ensures that hhea metrcs also get updated
    accordingly.
    The MVAR spec only has tags for the OS/2 metrics, but it is common in fonts
    to have the hhea metrics be equal to those for compat reasons.

    https://learn.microsoft.com/en-us/typography/opentype/spec/mvar
    https://googlefonts.github.io/gf-guide/metrics.html#7-hhea-and-typo-metrics-should-be-equal
    https://github.com/fonttools/fonttools/issues/3297
    """
    current_os2_vmetrics = [
        getattr(varfont["OS/2"], attr)
        for attr in ("sTypoAscender", "sTypoDescender", "sTypoLineGap")
    ]
    metrics_are_synced = current_os2_vmetrics == [
        getattr(varfont["hhea"], attr) for attr in ("ascender", "descender", "lineGap")
    ]

    yield metrics_are_synced

    if metrics_are_synced:
        new_os2_vmetrics = [
            getattr(varfont["OS/2"], attr)
            for attr in ("sTypoAscender", "sTypoDescender", "sTypoLineGap")
        ]
        if current_os2_vmetrics != new_os2_vmetrics:
            for attr, value in zip(
                ("ascender", "descender", "lineGap"), new_os2_vmetrics
            ):
                setattr(varfont["hhea"], attr, value)


def instantiateMVAR(varfont, axisLimits):
    log.info("Instantiating MVAR table")

    mvar = varfont["MVAR"].table
    fvarAxes = varfont["fvar"].axes
    varStore = mvar.VarStore
    defaultDeltas = instantiateItemVariationStore(varStore, fvarAxes, axisLimits)

    with verticalMetricsKeptInSync(varfont):
        setMvarDeltas(varfont, defaultDeltas)

    if varStore.VarRegionList.Region:
        varIndexMapping = varStore.optimize()
        for rec in mvar.ValueRecord:
            rec.VarIdx = varIndexMapping[rec.VarIdx]
    else:
        del varfont["MVAR"]


def _remapVarIdxMap(table, attrName, varIndexMapping, glyphOrder):
    oldMapping = getattr(table, attrName).mapping
    newMapping = [varIndexMapping[oldMapping[glyphName]] for glyphName in glyphOrder]
    setattr(table, attrName, builder.buildVarIdxMap(newMapping, glyphOrder))


# TODO(anthrotype) Add support for HVAR/VVAR in CFF2
def _instantiateVHVAR(varfont, axisLimits, tableFields, *, round=round):
    location = axisLimits.pinnedLocation()
    tableTag = tableFields.tableTag
    fvarAxes = varfont["fvar"].axes

    log.info("Instantiating %s table", tableTag)
    vhvar = varfont[tableTag].table
    varStore = vhvar.VarStore

    if "glyf" in varfont:
        # Deltas from gvar table have already been applied to the hmtx/vmtx. For full
        # instances (i.e. all axes pinned), we can simply drop HVAR/VVAR and return
        if set(location).issuperset(axis.axisTag for axis in fvarAxes):
            log.info("Dropping %s table", tableTag)
            del varfont[tableTag]
            return

    defaultDeltas = instantiateItemVariationStore(varStore, fvarAxes, axisLimits)

    if "glyf" not in varfont:
        # CFF2 fonts need hmtx/vmtx updated here. For glyf fonts, the instantiateGvar
        # function already updated the hmtx/vmtx from phantom points. Maybe remove
        # that and do it here for both CFF2 and glyf fonts?
        #
        # Specially, if a font has glyf but not gvar, the hmtx/vmtx will not have been
        # updated by instantiateGvar. Though one can call that a faulty font.
        metricsTag = "vmtx" if tableTag == "VVAR" else "hmtx"
        if metricsTag in varfont:
            advMapping = getattr(vhvar, tableFields.advMapping)
            metricsTable = varfont[metricsTag]
            metrics = metricsTable.metrics
            for glyphName, (advanceWidth, sb) in metrics.items():
                if advMapping:
                    varIdx = advMapping.mapping[glyphName]
                else:
                    varIdx = varfont.getGlyphID(glyphName)
                delta = round(defaultDeltas[varIdx])
                metrics[glyphName] = (max(0, advanceWidth + delta), sb)

            if (
                tableTag == "VVAR"
                and getattr(vhvar, tableFields.vOrigMapping) is not None
            ):
                log.warning(
                    "VORG table not yet updated to reflect changes in VVAR table"
                )

            # For full instances (i.e. all axes pinned), we can simply drop HVAR/VVAR and return
            if set(location).issuperset(axis.axisTag for axis in fvarAxes):
                log.info("Dropping %s table", tableTag)
                del varfont[tableTag]
                return

    if varStore.VarRegionList.Region:
        # Only re-optimize VarStore if the HVAR/VVAR already uses indirect AdvWidthMap
        # or AdvHeightMap. If a direct, implicit glyphID->VariationIndex mapping is
        # used for advances, skip re-optimizing and maintain original VariationIndex.
        if getattr(vhvar, tableFields.advMapping):
            varIndexMapping = varStore.optimize(use_NO_VARIATION_INDEX=False)
            glyphOrder = varfont.getGlyphOrder()
            _remapVarIdxMap(vhvar, tableFields.advMapping, varIndexMapping, glyphOrder)
            if getattr(vhvar, tableFields.sb1):  # left or top sidebearings
                _remapVarIdxMap(vhvar, tableFields.sb1, varIndexMapping, glyphOrder)
            if getattr(vhvar, tableFields.sb2):  # right or bottom sidebearings
                _remapVarIdxMap(vhvar, tableFields.sb2, varIndexMapping, glyphOrder)
            if tableTag == "VVAR" and getattr(vhvar, tableFields.vOrigMapping):
                _remapVarIdxMap(
                    vhvar, tableFields.vOrigMapping, varIndexMapping, glyphOrder
                )


def instantiateHVAR(varfont, axisLimits):
    return _instantiateVHVAR(varfont, axisLimits, varLib.HVAR_FIELDS)


def instantiateVVAR(varfont, axisLimits):
    return _instantiateVHVAR(varfont, axisLimits, varLib.VVAR_FIELDS)


class _TupleVarStoreAdapter(object):
    def __init__(self, regions, axisOrder, tupleVarData, itemCounts):
        self.regions = regions
        self.axisOrder = axisOrder
        self.tupleVarData = tupleVarData
        self.itemCounts = itemCounts

    @classmethod
    def fromItemVarStore(cls, itemVarStore, fvarAxes):
        axisOrder = [axis.axisTag for axis in fvarAxes]
        regions = [
            region.get_support(fvarAxes) for region in itemVarStore.VarRegionList.Region
        ]
        tupleVarData = []
        itemCounts = []
        for varData in itemVarStore.VarData:
            variations = []
            varDataRegions = (regions[i] for i in varData.VarRegionIndex)
            for axes, coordinates in zip(varDataRegions, zip(*varData.Item)):
                variations.append(TupleVariation(axes, list(coordinates)))
            tupleVarData.append(variations)
            itemCounts.append(varData.ItemCount)
        return cls(regions, axisOrder, tupleVarData, itemCounts)

    def rebuildRegions(self):
        # Collect the set of all unique region axes from the current TupleVariations.
        # We use an OrderedDict to de-duplicate regions while keeping the order.
        uniqueRegions = collections.OrderedDict.fromkeys(
            (
                frozenset(var.axes.items())
                for variations in self.tupleVarData
                for var in variations
            )
        )
        # Maintain the original order for the regions that pre-existed, appending
        # the new regions at the end of the region list.
        newRegions = []
        for region in self.regions:
            regionAxes = frozenset(region.items())
            if regionAxes in uniqueRegions:
                newRegions.append(region)
                del uniqueRegions[regionAxes]
        if uniqueRegions:
            newRegions.extend(dict(region) for region in uniqueRegions)
        self.regions = newRegions

    def instantiate(self, axisLimits):
        defaultDeltaArray = []
        for variations, itemCount in zip(self.tupleVarData, self.itemCounts):
            defaultDeltas = instantiateTupleVariationStore(variations, axisLimits)
            if not defaultDeltas:
                defaultDeltas = [0] * itemCount
            defaultDeltaArray.append(defaultDeltas)

        # rebuild regions whose axes were dropped or limited
        self.rebuildRegions()

        pinnedAxes = set(axisLimits.pinnedLocation())
        self.axisOrder = [
            axisTag for axisTag in self.axisOrder if axisTag not in pinnedAxes
        ]

        return defaultDeltaArray

    def asItemVarStore(self):
        regionOrder = [frozenset(axes.items()) for axes in self.regions]
        varDatas = []
        for variations, itemCount in zip(self.tupleVarData, self.itemCounts):
            if variations:
                assert len(variations[0].coordinates) == itemCount
                varRegionIndices = [
                    regionOrder.index(frozenset(var.axes.items())) for var in variations
                ]
                varDataItems = list(zip(*(var.coordinates for var in variations)))
                varDatas.append(
                    builder.buildVarData(varRegionIndices, varDataItems, optimize=False)
                )
            else:
                varDatas.append(
                    builder.buildVarData([], [[] for _ in range(itemCount)])
                )
        regionList = builder.buildVarRegionList(self.regions, self.axisOrder)
        itemVarStore = builder.buildVarStore(regionList, varDatas)
        # remove unused regions from VarRegionList
        itemVarStore.prune_regions()
        return itemVarStore


def instantiateItemVariationStore(
    itemVarStore, fvarAxes, axisLimits, hierarchical=False
):
    """Compute deltas at partial location, and update varStore in-place.

    Remove regions in which all axes were instanced, or fall outside the new axis
    limits. Scale the deltas of the remaining regions where only some of the axes
    were instanced.

    The number of VarData subtables, and the number of items within each, are
    not modified, in order to keep the existing VariationIndex valid.
    One may call VarStore.optimize() method after this to further optimize those.

    Args:
        varStore: An otTables.VarStore object (Item Variation Store)
        fvarAxes: list of fvar's Axis objects
        axisLimits: NormalizedAxisLimits: mapping axis tags to normalized
            min/default/max axis coordinates. May not specify coordinates/ranges for
            all the fvar axes.

    Returns:
        defaultDeltas: to be added to the default instance, of type dict of floats
            keyed by VariationIndex compound values: i.e. (outer << 16) + inner.
    """
    tupleVarStore = _TupleVarStoreAdapter.fromItemVarStore(itemVarStore, fvarAxes)
    defaultDeltaArray = tupleVarStore.instantiate(axisLimits)
    newItemVarStore = tupleVarStore.asItemVarStore()

    itemVarStore.VarRegionList = newItemVarStore.VarRegionList
    if not hasattr(itemVarStore, "VarDataCount"):  # Happens fromXML
        itemVarStore.VarDataCount = len(newItemVarStore.VarData)
    assert itemVarStore.VarDataCount == newItemVarStore.VarDataCount
    itemVarStore.VarData = newItemVarStore.VarData

    if not hierarchical:
        defaultDeltas = {
            ((major << 16) + minor): delta
            for major, deltas in enumerate(defaultDeltaArray)
            for minor, delta in enumerate(deltas)
        }
        defaultDeltas[itemVarStore.NO_VARIATION_INDEX] = 0
    else:
        defaultDeltas = {0xFFFF: {0xFFFF: 0}}  # NO_VARIATION_INDEX
        for major, deltas in enumerate(defaultDeltaArray):
            defaultDeltasForMajor = defaultDeltas.setdefault(major, {})
            for minor, delta in enumerate(deltas):
                defaultDeltasForMajor[minor] = delta
    return defaultDeltas


def _instantiateBASE(varfont, axisLimits):
    if "BASE" not in varfont:
        return

    base = varfont["BASE"].table
    varStore = getattr(base, "VarStore", None)
    if varStore is None:
        return

    log.info("Instantiating BASE table")

    fvarAxes = varfont["fvar"].axes
    defaultDeltas = instantiateItemVariationStore(varStore, fvarAxes, axisLimits)

    merger = MutatorMerger(
        varfont, defaultDeltas, deleteVariations=(not varStore.VarRegionList.Region)
    )
    merger.mergeTables(varfont, [varfont], ["BASE"])

    if varStore.VarRegionList.Region:
        base.remap_device_varidxes(varStore.optimize())
    else:
        # Downgrade BASE, it no longer references an ItemVariationStore.
        del base.VarStore
        base.Version = 0x00010000


def instantiateOTL(varfont, axisLimits):
    # TODO(anthrotype) Support partial instancing of JSTF table
    _instantiateBASE(varfont, axisLimits)

    if (
        "GDEF" not in varfont
        or varfont["GDEF"].table.Version < 0x00010003
        or not varfont["GDEF"].table.VarStore
    ):
        return

    if "GPOS" in varfont:
        msg = "Instantiating GDEF and GPOS tables"
    else:
        msg = "Instantiating GDEF table"
    log.info(msg)

    gdef = varfont["GDEF"].table
    varStore = gdef.VarStore
    fvarAxes = varfont["fvar"].axes

    defaultDeltas = instantiateItemVariationStore(varStore, fvarAxes, axisLimits)

    # When VF are built, big lookups may overflow and be broken into multiple
    # subtables. MutatorMerger (which inherits from AligningMerger) reattaches
    # them upon instancing, in case they can now fit a single subtable (if not,
    # they will be split again upon compilation).
    # This 'merger' also works as a 'visitor' that traverses the OTL tables and
    # calls specific methods when instances of a given type are found.
    # Specifically, it adds default deltas to GPOS Anchors/ValueRecords and GDEF
    # LigatureCarets, and optionally deletes all VariationIndex tables if the
    # VarStore is fully instanced.
    merger = MutatorMerger(
        varfont, defaultDeltas, deleteVariations=(not varStore.VarRegionList.Region)
    )
    merger.mergeTables(varfont, [varfont], ["GDEF", "GPOS"])

    if varStore.VarRegionList.Region:
        varIndexMapping = varStore.optimize()
        gdef.remap_device_varidxes(varIndexMapping)
        if "GPOS" in varfont:
            varfont["GPOS"].table.remap_device_varidxes(varIndexMapping)
    else:
        # Downgrade GDEF.
        del gdef.VarStore
        gdef.Version = 0x00010002
        if gdef.MarkGlyphSetsDef is None:
            del gdef.MarkGlyphSetsDef
            gdef.Version = 0x00010000

        if not (
            gdef.LigCaretList
            or gdef.MarkAttachClassDef
            or gdef.GlyphClassDef
            or gdef.AttachList
            or (gdef.Version >= 0x00010002 and gdef.MarkGlyphSetsDef)
        ):
            del varfont["GDEF"]


def _isValidAvarSegmentMap(axisTag, segmentMap):
    if not segmentMap:
        return True
    if not {(-1.0, -1.0), (0, 0), (1.0, 1.0)}.issubset(segmentMap.items()):
        log.warning(
            f"Invalid avar SegmentMap record for axis '{axisTag}': does not "
            "include all required value maps {-1.0: -1.0, 0: 0, 1.0: 1.0}"
        )
        return False
    previousValue = None
    for fromCoord, toCoord in sorted(segmentMap.items()):
        if previousValue is not None and previousValue > toCoord:
            log.warning(
                f"Invalid avar AxisValueMap({fromCoord}, {toCoord}) record "
                f"for axis '{axisTag}': the toCoordinate value must be >= to "
                f"the toCoordinate value of the preceding record ({previousValue})."
            )
            return False
        previousValue = toCoord
    return True


def downgradeCFF2ToCFF(varfont):
    # Save these properties
    recalcTimestamp = varfont.recalcTimestamp
    recalcBBoxes = varfont.recalcBBoxes

    # Disable them
    varfont.recalcTimestamp = False
    varfont.recalcBBoxes = False

    # Save to memory, reload, downgrade and save again, reload.
    # We do this dance because the convertCFF2ToCFF changes glyph
    # names, so following save would fail if any other table was
    # loaded and referencing glyph names.
    #
    # The second save+load is unfortunate but also necessary.

    stream = io.BytesIO()
    log.info("Saving CFF2 font to memory for downgrade")
    varfont.save(stream)
    stream.seek(0)
    varfont = TTFont(stream, recalcTimestamp=False, recalcBBoxes=False)

    convertCFF2ToCFF(varfont)

    stream = io.BytesIO()
    log.info("Saving downgraded CFF font to memory")
    varfont.save(stream)
    stream.seek(0)
    varfont = TTFont(stream, recalcTimestamp=False, recalcBBoxes=False)

    # Uncomment, to see test all tables can be loaded. This fails without
    # the extra save+load above.
    """
    for tag in varfont.keys():
        print("Loading", tag)
        varfont[tag]
    """

    # Restore them
    varfont.recalcTimestamp = recalcTimestamp
    varfont.recalcBBoxes = recalcBBoxes

    return varfont


def instantiateAvar(varfont, axisLimits, normalizedLimits=None, oldIntermediates=None):
    # 'axisLimits' dict must contain user-space (non-normalized) coordinates.

    if normalizedLimits is None:
        normalizedLimits = axisLimits.normalize(varfont)

    avar = varfont["avar"]

    segments = avar.segments

    # Capture the OLD avar v1 segment maps before the renormalization below
    # overwrites avar.segments; _instantiateAvarV2 needs them to collect the
    # interior breakpoints used for exact offset compensation.
    oldSegments = {tag: dict(mapping) for tag, mapping in segments.items()}

    # An avar version 2 font is never downgraded to version 1: even with a
    # NULL/empty VarStore, partial instancing takes the avar2 path (offset
    # compensation creates delta rows on demand), which keeps the other
    # variation tables in old coordinate space and the instance bit-exact.
    version = getattr(avar, "majorVersion", 1)
    isAvar2 = version >= 2

    # Drop the table if we instantiate all the axes. An avar2 VarStore may
    # drive axes that have no v1 segment map (segments can even be empty),
    # so for those fonts require ALL fvar axes to be pinned, not merely all
    # segment-mapped ones — otherwise we'd delete live avar2 variation data.
    pinnedAxes = set(axisLimits.pinnedLocation())
    if pinnedAxes.issuperset(axis.axisTag for axis in varfont["fvar"].axes) or (
        not isAvar2 and pinnedAxes.issuperset(segments)
    ):
        log.info("Dropping avar table")
        del varfont["avar"]
        return

    # For avar2: need old intermediate values BEFORE modifying avar v1.
    if isAvar2 and oldIntermediates is None:
        oldIntermediates = _computeOldIntermediates(varfont, axisLimits)

    log.info("Instantiating avar table")
    if isAvar2:
        # For avar2 partial instancing, keep identity segments for pinned axes
        # that will be hidden in fvar. compile() needs segment entries for all
        # fvar axes. Self-contained axes' segments will be removed later when
        # they're removed from fvar.
        for axis in pinnedAxes:
            if axis in segments:
                segments[axis] = {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}
    else:
        for axis in pinnedAxes:
            if axis in segments:
                del segments[axis]

    # Standard avar v1 instancing (unchanged).
    normalizedRanges = axisLimits.normalize(varfont, usingAvar=False)
    newSegments = {}
    for axisTag, mapping in segments.items():
        if not _isValidAvarSegmentMap(axisTag, mapping):
            continue
        if mapping and axisTag in normalizedRanges:
            axisRange = normalizedRanges[axisTag]
            mappedMin = floatToFixedToFloat(
                piecewiseLinearMap(axisRange.minimum, mapping), 14
            )
            mappedDef = floatToFixedToFloat(
                piecewiseLinearMap(axisRange.default, mapping), 14
            )
            mappedMax = floatToFixedToFloat(
                piecewiseLinearMap(axisRange.maximum, mapping), 14
            )
            mappedAxisLimit = NormalizedAxisTripleAndDistances(
                mappedMin,
                mappedDef,
                mappedMax,
                axisRange.distanceNegative,
                axisRange.distancePositive,
            )
            newMapping = {}
            for fromCoord, toCoord in mapping.items():
                if fromCoord < axisRange.minimum or fromCoord > axisRange.maximum:
                    continue
                fromCoord = axisRange.renormalizeValue(fromCoord)

                assert mappedMin <= toCoord <= mappedMax
                toCoord = mappedAxisLimit.renormalizeValue(toCoord)

                fromCoord = floatToFixedToFloat(fromCoord, 14)
                toCoord = floatToFixedToFloat(toCoord, 14)
                newMapping[fromCoord] = toCoord
            newMapping.update({-1.0: -1.0, 0.0: 0.0, 1.0: 1.0})
            newSegments[axisTag] = newMapping
        else:
            newSegments[axisTag] = mapping
    avar.segments = newSegments

    if version == 1:
        return {}

    assert version == 2

    return _instantiateAvarV2(
        varfont, axisLimits, normalizedLimits, oldIntermediates, oldSegments
    )


def _computeOldIntermediates(varfont, axisLimits):
    """Compute old intermediate axis values before avar v1 modification.

    For each axis in axisLimits, computes the old intermediate coordinate
    values (post-fvar normalization + avar v1 mapping) at the new min,
    default, and max user-space values.

    Returns:
        dict mapping axis tags to (a_i, d_i, b_i) tuples where:
        - a_i = old intermediate value at the new minimum
        - d_i = old intermediate value at the new default
        - b_i = old intermediate value at the new maximum
        All values are in normalized coordinate space [-1, +1].
    """
    fvar = varfont["fvar"]
    avar = varfont["avar"]
    avarSegments = avar.segments

    result = {}
    for axis in fvar.axes:
        tag = axis.axisTag
        if tag not in axisLimits:
            continue

        triple = axisLimits[tag]
        minV = triple.minimum
        defV = triple.default if triple.default is not None else axis.defaultValue
        maxV = triple.maximum

        # Normalize using old fvar, map through old avar v1 segment map,
        # quantize to F2Dot14 — all via the module's normalize() helper so
        # the rounding behavior lives in a single place.
        oldFvarTriple = (axis.minValue, axis.defaultValue, axis.maxValue)
        avarMapping = avarSegments.get(tag, None)
        result[tag] = tuple(
            normalize(v, oldFvarTriple, avarMapping) for v in (minV, defV, maxV)
        )

    return result


# Threshold, in F2Dot14 units, above which a restricted avar2 axis's residual
# offset-compensation error triggers a warning. The quantization floor is ~2 and
# small residuals (a few units) are common and imperceptible for real avar v1
# maps; only a clearly larger residual — a steep retained segment that cannot be
# reproduced bit-exactly — is worth flagging (see _estimateAvar2OffsetError).
_AVAR2_OFFSET_WARN_THRESHOLD = 8


def _estimateAvar2OffsetError(oldSeg, newSeg, oldFvarTriple, newTriple, offsetByZ):
    """Estimate the residual offset-compensation error for one restricted axis.

    Returns the max |old-avar1-final - (new-avar1 + offset)| over the retained
    user range, in F2Dot14 units. offset(z) is the piecewise-linear function
    through the offsetByZ knots (what the 1-D VariationModel reproduces). The
    residual is dominated by F2Dot14 requantization of a steep retained avar v1
    segment (e.g. a moved default compressing part of the axis into a narrow z
    band); offset compensation cannot remove it. Used only to decide which
    encoding ships and whether to warn.

    Sampled on a uniform grid augmented with the user-space preimages of every
    kink of the residual (old/new avar v1 breakpoints and offset(z) knots), so
    the worst kink cannot fall between uniform samples. The pointwise F2Dot14
    rounding makes this an estimate rather than an exact bound, but every
    piecewise-linear extremum is visited.
    """
    lo, _, hi = newTriple
    samples = 257
    us = {lo + (hi - lo) * i / (samples - 1) for i in range(samples)}
    denormOld = {-1.0: oldFvarTriple[0], 0.0: oldFvarTriple[1], 1.0: oldFvarTriple[2]}
    denormNew = {-1.0: newTriple[0], 0.0: newTriple[1], 1.0: newTriple[2]}
    if oldSeg:
        us.update(piecewiseLinearMap(n, denormOld) for n in oldSeg)
    zToNorm = None
    if newSeg:
        us.update(piecewiseLinearMap(n, denormNew) for n in newSeg)
        # Inverse of newSeg, to pull offset(z) knots back to normalized space.
        # (For a non-strictly-monotone map this picks one preimage, which is
        # fine: the uniform grid still covers the rest.)
        zToNorm = {v: k for k, v in newSeg.items()}
    for z in offsetByZ:
        n = piecewiseLinearMap(z, zToNorm) if zToNorm else z
        us.add(piecewiseLinearMap(n, denormNew))

    maxErr = 0
    for u in us:
        if not (lo <= u <= hi):
            continue
        nOld = normalizeValue(u, oldFvarTriple)
        oldFinal = piecewiseLinearMap(nOld, oldSeg) if oldSeg else nOld
        nNew = normalizeValue(u, newTriple)
        z = piecewiseLinearMap(nNew, newSeg) if newSeg else nNew
        newFinal = z + piecewiseLinearMap(z, offsetByZ)
        maxErr = max(maxErr, abs(otRound(oldFinal * 16384) - otRound(newFinal * 16384)))
    return maxErr


def _effectiveAvar2VarIdx(varStore, varIdxMap, axisIdx):
    """The varIdx an axis actually resolves to in the avar2 VarStore.

    Entries (or the implicit identity mapping) that point outside the store
    behave as "no variation" at runtime, so report them as
    NO_VARIATION_INDEX; this also covers a NULL VarStore.
    """
    varIdx = varIdxMap[axisIdx] if varIdxMap is not None else axisIdx
    if varIdx == NO_VARIATION_INDEX or varStore is None:
        return NO_VARIATION_INDEX
    outer, inner = varIdx >> 16, varIdx & 0xFFFF
    if outer >= len(varStore.VarData) or inner >= varStore.VarData[outer].ItemCount:
        return NO_VARIATION_INDEX
    return varIdx


def _instantiateAvarV2(
    varfont, axisLimits, normalizedLimits, oldIntermediates, oldSegments=None
):
    """Instance avar v2 IVS: rebase regions + add offset compensation.

    This implements the offset compensation approach from the design doc:
    1. Rebase IVS regions to new intermediate coordinate space (via rebaseTent)
    2. Detect self-contained pinned axes (whose final coord is constant)
    3. Add offset compensation entries (empty-region bias + tent deltas)
       for non-self-contained axes so final coordinates remain in old space
    4. Self-contained axes are skipped — their contributions will be folded
       into gvar/HVAR/etc. by the caller

    Returns:
        dict: {axisTag: finalCoord} for self-contained pinned axes that can be
        removed from fvar. finalCoord is in old-space normalized coordinates.
    """
    avar = varfont["avar"]
    fvarAxes = varfont["fvar"].axes
    varStore = avar.table.VarStore
    varIdxMap = getattr(avar.table, "VarIdxMap", None)
    oldSegments = oldSegments or {}

    # A compiled DeltaSetIndexMap may be stored truncated (trailing duplicate
    # entries are trimmed; reads clamp to the last entry). We write into the
    # mapping by axis index below, so pad it back to full length first, with
    # the same value clamping would have produced for each index.
    if varIdxMap is not None and len(varIdxMap.mapping) < len(fvarAxes):
        mapping = varIdxMap.mapping
        fill = mapping[-1] if mapping else NO_VARIATION_INDEX
        mapping.extend([fill] * (len(fvarAxes) - len(mapping)))

    # Normalize mapping entries that don't resolve to a real store row (out
    # of range, or a NULL VarStore) to NO_VARIATION_INDEX, mirroring runtime
    # behavior. A version 2 font is never downgraded: with a NULL VarStore we
    # start from an empty store and offset compensation creates delta rows on
    # demand.
    numAxes = len(fvarAxes)
    rawMapping = [varIdxMap[i] if varIdxMap is not None else i for i in range(numAxes)]
    effectiveMapping = [
        _effectiveAvar2VarIdx(varStore, varIdxMap, i) for i in range(numAxes)
    ]
    if effectiveMapping != rawMapping:
        if varIdxMap is None:
            varIdxMap = builder.buildDeltaSetIndexMap(effectiveMapping)
            avar.table.VarIdxMap = varIdxMap
        else:
            varIdxMap.mapping = effectiveMapping
    if varStore is None:
        varStore = builder.buildVarStore(
            builder.buildVarRegionList([], [a.axisTag for a in fvarAxes]), []
        )
        avar.table.VarStore = varStore

    # Step 1: Convert IVS to TupleVariation representation and rebase
    tupleVarStore = _TupleVarStoreAdapter.fromItemVarStore(varStore, fvarAxes)

    # Save original axis order (instantiate() will remove pinned axes)
    originalAxisOrder = list(tupleVarStore.axisOrder)

    # Rebase regions for restricted/pinned axes
    defaultDeltaArray = tupleVarStore.instantiate(normalizedLimits)

    # Restore full axis order — for avar2 we keep all fvar axes
    tupleVarStore.axisOrder = originalAxisOrder

    # Step 1.5: Detect self-contained pinned axes.
    # A pinned axis is self-contained if its final coordinate doesn't vary
    # with any remaining axes. After IVS rebasing, check if any remaining
    # TupleVariation has a non-zero delta at the axis's inner position.
    selfContainedAxes = {}
    pinnedLocation = axisLimits.pinnedLocation()

    for axisIdx, axis in enumerate(fvarAxes):
        tag = axis.axisTag
        if tag not in pinnedLocation:
            continue

        # Get this axis's varIdx
        if varIdxMap is not None:
            varIdx = varIdxMap[axisIdx]
        else:
            varIdx = axisIdx

        if varIdx == NO_VARIATION_INDEX:
            # No IVS entry → final coord = intermediate coord (self-contained)
            a_i, d_i, b_i = oldIntermediates[tag]
            finalInt = otRound(d_i * 16384)
            finalInt = min(max(finalInt, -(1 << 14)), 1 << 14)
            selfContainedAxes[tag] = finalInt / 16384.0
            continue

        outer = varIdx >> 16
        inner = varIdx & 0xFFFF

        if outer >= len(tupleVarStore.tupleVarData):
            continue

        # Check if any remaining TupleVariation has a non-zero delta
        hasDelta = any(
            tv.coordinates[inner] != 0 for tv in tupleVarStore.tupleVarData[outer]
        )

        if not hasDelta:
            # Self-contained: final coord = intermediate + defaultDelta
            a_i, d_i, b_i = oldIntermediates[tag]
            d_int = otRound(d_i * 16384)
            defaultDelta = defaultDeltaArray[outer][inner]
            finalInt = d_int + otRound(defaultDelta)
            finalInt = min(max(finalInt, -(1 << 14)), 1 << 14)
            selfContainedAxes[tag] = finalInt / 16384.0

    if selfContainedAxes:
        log.info(
            "Self-contained pinned axes (removable): %s",
            {tag: round(v, 6) for tag, v in selfContainedAxes.items()},
        )

    # Step 1.75: Privatize shared varIdx delta rows before adding offset
    # compensation. avar2's VarIdxMap may map several fvar axes to the SAME
    # IVS delta row (as the parametric test font does: e.g. GRAD and XOPQ can
    # share one varIdx). Writing one axis's offset-compensation deltas into a
    # shared row would corrupt every OTHER axis that reads that row. So give
    # each offset-receiving axis whose row is shared its own private copy of
    # the row (identical contents, preserving the rebased delta), then repoint
    # its VarIdxMap entry. Sharers keep the clean row; offsets land privately.
    def _axisVarIdx(i):
        return varIdxMap[i] if varIdxMap is not None else i

    varIdxRefCount = collections.Counter(_axisVarIdx(i) for i in range(len(fvarAxes)))
    for axisIdx, axis in enumerate(fvarAxes):
        tag = axis.axisTag
        # Only axes that will receive offset compensation (restricted or
        # non-self-contained pinned) can contaminate a shared row.
        if tag not in axisLimits or tag in selfContainedAxes:
            continue
        varIdx = _axisVarIdx(axisIdx)
        if varIdx == NO_VARIATION_INDEX:
            continue  # gets a fresh, private VarData in the offset loop below
        if varIdxRefCount[varIdx] <= 1:
            continue  # sole owner: safe to write offsets in place
        # Shared: duplicate this axis's item (row) within its VarData so it has
        # a private inner index. Copy the value from every TupleVariation so the
        # rebased delta is preserved; extend defaultDeltaArray in tandem.
        outer = varIdx >> 16
        inner = varIdx & 0xFFFF
        newInner = tupleVarStore.itemCounts[outer]
        for tv in tupleVarStore.tupleVarData[outer]:
            tv.coordinates.append(tv.coordinates[inner])
        tupleVarStore.itemCounts[outer] += 1
        defaultDeltaArray[outer].append(defaultDeltaArray[outer][inner])
        newVarIdx = (outer << 16) | newInner
        # A shared varIdx implies an explicit VarIdxMap (implicit identity maps
        # every axis to a distinct varIdx, so we could not be here otherwise).
        assert varIdxMap is not None
        varIdxMap.mapping[axisIdx] = newVarIdx
        varIdxRefCount[varIdx] -= 1
        varIdxRefCount[newVarIdx] += 1

    # Step 2: Add offset compensation TupleVariations
    processedVarIdxes = set()
    approxWarn = {}  # {axisTag: residual F2Dot14 error} for axes we can't make exact
    for axisIdx, axis in enumerate(fvarAxes):
        tag = axis.axisTag

        # Determine this axis's varIdx
        if varIdxMap is not None:
            varIdx = varIdxMap[axisIdx]
        else:
            varIdx = axisIdx

        if tag in axisLimits:
            if tag in selfContainedAxes:
                # Self-contained pinned axis: skip offset compensation.
                # Its contribution will be folded into gvar/HVAR/etc. instead.
                continue

            # This axis is being restricted or pinned (non-self-contained)
            a_i, d_i, b_i = oldIntermediates[tag]
            isPinned = axisLimits[tag].minimum == axisLimits[tag].maximum

            if varIdx == NO_VARIATION_INDEX:
                # This axis has no existing avar2 mapping. Create a new
                # VarData with one item to hold the offset compensation.
                outer = len(tupleVarStore.tupleVarData)
                inner = 0
                tupleVarStore.tupleVarData.append([])
                tupleVarStore.itemCounts.append(1)
                defaultDeltaArray.append([0])

                varIdx = (outer << 16) | inner

                # Update VarIdxMap
                if varIdxMap is None:
                    # Identity mapping for all axes, except this one
                    mapping = list(range(len(fvarAxes)))
                    mapping[axisIdx] = varIdx
                    varIdxMap = builder.buildDeltaSetIndexMap(mapping)
                    avar.table.VarIdxMap = varIdxMap
                else:
                    varIdxMap.mapping[axisIdx] = varIdx

            outer = varIdx >> 16
            inner = varIdx & 0xFFFF
            itemCount = tupleVarStore.itemCounts[outer]
            defaultDelta = defaultDeltaArray[outer][inner]

            # Offset compensation encodes, as avar2 deltas on axis `tag`, the
            # piecewise-linear function offset(z) = inv_renorm(z) - z, where
            # inv_renorm maps a new intermediate coordinate z back to the old
            # intermediate coordinate. It is known at these breakpoints in the
            # new intermediate space:
            #   z = -1  ->  a_i + 1   (new minimum)
            #   z =  0  ->  d_i       (new default)
            #   z = +1  ->  b_i - 1   (new maximum)
            # If the axis default MOVED, inv_renorm also kinks where the OLD
            # default lands in the new space (old intermediate crosses 0), at
            #   z = z_old  ->  -z_old
            # Omitting that breakpoint (as a plain two-tent encoding does) makes
            # interior coordinates wrong. Feed all breakpoints to a VariationModel
            # to synthesize the correct tents; this reduces to the two classic
            # tents when the default is unchanged (z_old == 0).
            offsetByZ = {-1.0: a_i + 1.0, 0.0: d_i, 1.0: b_i - 1.0}
            if not isPinned:
                newSeg = avar.segments.get(tag)
                newLimit = axisLimits[tag]
                # default may be unpopulated (2-tuple limits): means unchanged.
                newDefault = (
                    newLimit.default
                    if newLimit.default is not None
                    else axis.defaultValue
                )
                newTriple = (newLimit.minimum, newDefault, newLimit.maximum)
                # Moved-default kink: where the OLD default lands in new space.
                zOldNorm = normalizeValue(axis.defaultValue, newTriple)
                zOld = piecewiseLinearMap(zOldNorm, newSeg) if newSeg else zOldNorm
                zOld = floatToFixedToFloat(zOld, 14)
                if -1.0 < zOld < 1.0 and zOld not in offsetByZ:
                    offsetByZ[zOld] = -zOld

                # Interior avar v1 breakpoints inside the retained range each put
                # a kink in offset(z). Sampling only {-1, 0, +1, z_old} would
                # linearly interpolate across those kinks (approximate, worst
                # under a moved default on an asymmetric axis). The retained
                # (renormalized) avar v1 map keeps exactly the in-range old
                # breakpoints, and z(u) kinks at each retained breakpoint's
                # output coordinate; add that z with its old intermediate value
                # so the model reproduces offset(z) at every kink. Iterating the
                # retained map lands each sample on the F2Dot14-quantized kink.
                oldSeg = oldSegments.get(tag)
                oldFvarTriple = (axis.minValue, axis.defaultValue, axis.maxValue)
                withBreakpoints = dict(offsetByZ)
                if oldSeg and newSeg:
                    denormNew = {
                        -1.0: newTriple[0],
                        0.0: newTriple[1],
                        1.0: newTriple[2],
                    }
                    for fromCoordNew, z in newSeg.items():
                        if fromCoordNew in (-1.0, 0.0, 1.0):
                            continue  # anchors already seeded
                        z = floatToFixedToFloat(z, 14)
                        if not (-1.0 < z < 1.0) or z in withBreakpoints:
                            continue
                        user = piecewiseLinearMap(fromCoordNew, denormNew)
                        nOld = normalizeValue(user, oldFvarTriple)
                        xOld = piecewiseLinearMap(nOld, oldSeg)
                        withBreakpoints[z] = floatToFixedToFloat(xOld, 14) - z

                # Extra tents cost F2Dot14 rounding, so for a steep segment they
                # can add more quantization noise than the structural error they
                # remove. Keep the interior breakpoints only when they do not
                # increase the estimated residual; this makes the collection a
                # strict (never-worse) improvement over the {-1,0,+1,z_old}
                # anchors. Warn when even the better choice is not bit-exact
                # (a steep retained segment we cannot reproduce in F2Dot14).
                errAnchors = _estimateAvar2OffsetError(
                    oldSeg, newSeg, oldFvarTriple, newTriple, offsetByZ
                )
                if len(withBreakpoints) == len(offsetByZ):
                    # Collection added no breakpoints (the common case):
                    # the two encodings are identical, skip the second pass.
                    approxErr = errAnchors
                else:
                    errWith = _estimateAvar2OffsetError(
                        oldSeg, newSeg, oldFvarTriple, newTriple, withBreakpoints
                    )
                    if errWith <= errAnchors:
                        offsetByZ = withBreakpoints
                        approxErr = errWith
                    else:
                        approxErr = errAnchors
                if approxErr > _AVAR2_OFFSET_WARN_THRESHOLD:
                    approxWarn[tag] = approxErr

            if isPinned:
                # Pinned axis has zero range: only the constant bias applies.
                bias = otRound(d_i * 16384) + otRound(defaultDelta)
                if bias != 0:
                    coords = [0] * itemCount
                    coords[inner] = bias
                    tupleVarStore.tupleVarData[outer].append(TupleVariation({}, coords))
            else:
                zs = sorted(offsetByZ)
                model = VariationModel([{tag: z} for z in zs], axisOrder=[tag])
                deltas = model.getDeltas([offsetByZ[z] for z in zs])
                for delta, support in zip(deltas, model.supports):
                    di = otRound(delta * 16384)
                    if not support:
                        # Base (empty) region: fold in the rebased default delta.
                        di += otRound(defaultDelta)
                    if di != 0:
                        coords = [0] * itemCount
                        coords[inner] = di
                        tupleVarStore.tupleVarData[outer].append(
                            TupleVariation(dict(support), coords)
                        )
        else:
            # Free or private axis — not being restricted.
            # If it has a non-zero default delta (from rebasing restricted axes'
            # contributions), add it back as a bias. The standard instancing
            # subtracts default deltas, but for avar2 we must preserve them.
            # Skip if this varIdx was already processed (multiple axes can
            # share the same varIdx entry).
            if varIdx == NO_VARIATION_INDEX:
                continue
            if varIdx in processedVarIdxes:
                continue
            processedVarIdxes.add(varIdx)
            outer = varIdx >> 16
            inner = varIdx & 0xFFFF
            if outer >= len(defaultDeltaArray):
                continue
            defaultDelta = defaultDeltaArray[outer][inner]
            if otRound(defaultDelta) != 0:
                itemCount = tupleVarStore.itemCounts[outer]
                coords = [0] * itemCount
                coords[inner] = otRound(defaultDelta)
                tupleVarStore.tupleVarData[outer].append(TupleVariation({}, coords))

    if approxWarn:
        log.warning(
            "avar2 partial instancing is approximate (not bit-exact) for "
            "axes %s: the avar v1 segment map has interior breakpoint(s) in the "
            "retained range that, after a moved/edge default, require a steep "
            "F2Dot14 segment the offset compensation cannot fully reproduce. "
            "Max residual: %s F2Dot14 units.",
            ", ".join(sorted(approxWarn)),
            {tag: approxWarn[tag] for tag in sorted(approxWarn)},
        )

    # Step 3: Remove self-contained axes from axis order.
    # The VarRegionList must match the post-removal fvar axis count.
    # After instancing, no TupleVariation references self-contained axes.
    if selfContainedAxes:
        tupleVarStore.axisOrder = [
            tag for tag in tupleVarStore.axisOrder if tag not in selfContainedAxes
        ]

    # Step 4: Rebuild regions and convert back to IVS
    tupleVarStore.rebuildRegions()
    newVarStore = tupleVarStore.asItemVarStore()

    # Step 5: Optimize the VarStore (merge regions, remove zero deltas)
    varIdxMapping = newVarStore.optimize(use_NO_VARIATION_INDEX=False)
    if varIdxMap is not None:
        varIdxMap.mapping = [
            varIdxMapping.get(varIdx, varIdx) for varIdx in varIdxMap.mapping
        ]
    elif varIdxMapping:
        # optimize() remapped indices but we had no VarIdxMap (implicit
        # identity). Create one if the mapping is no longer identity.
        varIdxMap = builder.buildDeltaSetIndexMap(
            varIdxMapping.get(i, i) for i in range(len(fvarAxes))
        )
        if varIdxMap is not None:
            avar.table.VarIdxMap = varIdxMap
    avar.table.VarStore = newVarStore

    return selfContainedAxes


def _isTupleVariationDead(tv, reachableRanges):
    """Check if a TupleVariation is dead (unreachable) given reachable axis ranges."""
    for tag, (lo, hi) in reachableRanges.items():
        if tag not in tv.axes:
            continue
        start, peak, end = tv.axes[tag]
        # Region is dead if entirely outside reachable range. Strict
        # comparisons: supportScalar returns 1 (not 0) at v == peak, so a
        # tent whose peak sits exactly on the reachable boundary (peak ==
        # start == hi, or peak == end == lo) is still live there.
        if hi < start or lo > end:
            return True
    return False


def _cullItemVariationStore(varStore, fvarAxes, reachableRanges):
    """Remove dead regions from an ItemVariationStore via TupleVariation culling.

    Converts to TupleVariations, removes dead ones, converts back.
    Returns the number of removed regions.
    """
    tupleVarStore = _TupleVarStoreAdapter.fromItemVarStore(varStore, fvarAxes)
    totalCulled = 0
    for tvList in tupleVarStore.tupleVarData:
        before = len(tvList)
        tvList[:] = [
            tv for tv in tvList if not _isTupleVariationDead(tv, reachableRanges)
        ]
        totalCulled += before - len(tvList)
    if totalCulled:
        tupleVarStore.rebuildRegions()
        newVarStore = tupleVarStore.asItemVarStore()
        varStore.VarRegionList = newVarStore.VarRegionList
        varStore.VarData = newVarStore.VarData
    return totalCulled


def _cullVariationsForAvar2(varfont, reachableRanges):
    """Cull dead variations outside reachable old-space final-coord ranges.

    After avar2 partial instancing with offset compensation, variation tables
    remain in old-space final coordinates. Some regions may be unreachable
    because the restricted axis range maps to a narrower final-coord range.

    For gvar/cvar: removes TupleVariations whose axis regions lie entirely
    outside the reachable range.
    For IVS-based tables (HVAR, VVAR, MVAR, GDEF): removes dead region
    columns from the ItemVariationStore.

    Args:
        varfont: The font being instanced.
        reachableRanges: dict mapping axis tags to (lo, hi) tuples representing
            the reachable range of old-space final coordinates for that axis.
    """
    fvarAxes = varfont["fvar"].axes

    if "gvar" in varfont:
        gvar = varfont["gvar"]
        totalCulled = 0
        totalTotal = 0

        for glyphName in gvar.variations:
            variations = gvar.variations[glyphName]
            newVariations = []
            for tv in variations:
                totalTotal += 1
                if _isTupleVariationDead(tv, reachableRanges):
                    totalCulled += 1
                else:
                    newVariations.append(tv)
            gvar.variations[glyphName] = newVariations

        if totalCulled:
            log.info(
                "avar2 gvar culling: removed %d / %d TupleVariations (%.1f%%)",
                totalCulled,
                totalTotal,
                100 * totalCulled / totalTotal if totalTotal else 0,
            )

    if "cvar" in varfont:
        cvar = varfont["cvar"]
        before = len(cvar.variations)
        cvar.variations = [
            tv
            for tv in cvar.variations
            if not _isTupleVariationDead(tv, reachableRanges)
        ]
        culled = before - len(cvar.variations)
        if culled:
            log.info(
                "avar2 cvar culling: removed %d / %d TupleVariations (%.1f%%)",
                culled,
                before,
                100 * culled / before if before else 0,
            )

    # Cull all IVS-based tables. A fixed tag list (like the rest of the
    # instancer uses) rather than probing every table in the font, which
    # would force-decompile unrelated tables for nothing.
    # TODO: VARC's MultiVarStore is not culled here (different structure).
    for tag in ("HVAR", "VVAR", "MVAR", "GDEF", "BASE", "COLR"):
        if tag not in varfont:
            continue
        table = varfont[tag]
        table = getattr(table, "table", table)
        vs = getattr(table, "VarStore", None)
        if vs is None:
            continue
        removed = _cullItemVariationStore(vs, fvarAxes, reachableRanges)
        if removed:
            log.info(
                "avar2 %s culling: removed %d dead region references",
                tag,
                removed,
            )

    # CFF2: cull via populate/cull/depopulate cycle (blend operators encode
    # region counts, so the VarStore can't be culled without updating them).
    if "CFF2" in varfont:
        cff = varfont["CFF2"].cff
        topDict = cff.topDictIndex[0]
        if getattr(topDict, "VarStore", None) and topDict.VarStore.otVarStore:
            instantiateCFF2(
                varfont,
                NormalizedAxisLimits({}),
                reachableRanges=reachableRanges,
            )


def _computeReachableRangesForAvar2(varfont, axisLimits, reachableRanges):
    """Compute reachable old-space final-coord ranges using getExtremes.

    For axes not already in reachableRanges (i.e., axes with IVS entries
    rather than NO_VARIATION_INDEX), compute conservative bounds on the
    reachable old-space final coordinate using getExtremes on the instanced
    avar v2 VarStore.

    This is critical for parametric fonts where gvar references private axes
    driven by public axes through avar2. Restricting wght narrows the
    reachable range of XOPQ, YOPQ, etc.

    Must be called AFTER _instantiateFvarForAvar2 has updated fvar and
    VarIdxMap so that axes and VarStore match.
    """
    avar = varfont["avar"]
    fvarAxes = varfont["fvar"].axes
    varStore = getattr(avar.table, "VarStore", None)
    if varStore is None:
        return
    varIdxMap = getattr(avar.table, "VarIdxMap", None)

    # Private axes (originally hidden, not user-restricted) always have
    # intermediate coordinate = 0. Any VarStore region referencing them
    # as input gets scalar = 0 at intermediate = 0. Tell getExtremes
    # by pinning them at (0, 0, 0) so it correctly zeros out those regions.
    privateAxisLimits = {}
    for axis in fvarAxes:
        if axis.flags & 0x1 and axis.axisTag not in axisLimits:
            privateAxisLimits[axis.axisTag] = (0, 0, 0)

    for axisIdx, axis in enumerate(fvarAxes):
        tag = axis.axisTag
        if tag in reachableRanges:
            continue  # Already computed (NO_VARIATION_INDEX, exact)

        if varIdxMap is not None:
            varIdx = varIdxMap[axisIdx]
        else:
            varIdx = axisIdx

        if varIdx == NO_VARIATION_INDEX:
            continue

        # Private axes have no identity term (intermediate always 0,
        # final = delta only). Public axes have identity + delta.
        private = axis.flags & 0x1
        identityAxisIndex = None if private else axisIdx

        # Pass privateAxisLimits so getExtremes correctly zeros out
        # regions referencing private axes (whose intermediate is always 0).
        minV, maxV = varStore.getExtremes(
            varIdx, fvarAxes, privateAxisLimits, identityAxisIndex
        )

        lo = min(+1, max(-1, minV / 16384))
        hi = min(+1, max(-1, maxV / 16384))
        assert lo <= hi
        reachableRanges[tag] = (lo, hi)

        log.info(
            "avar2 getExtremes range for %s: [%.4f, %.4f]",
            tag,
            lo,
            hi,
        )


def _instantiateFvarForAvar2(varfont, axisLimits, selfContainedAxes=None):
    """Update fvar for avar2 partial instancing.

    Self-contained pinned axes (in selfContainedAxes dict) are removed from fvar
    and their contributions folded into variation tables. Non-self-contained pinned
    axes are kept as hidden axes. Restricted axes have their ranges updated.
    """
    fvar = varfont["fvar"]
    pinnedLocation = axisLimits.pinnedLocation()
    if selfContainedAxes is None:
        selfContainedAxes = {}

    log.info("Instantiating fvar table (avar2 mode)")

    # Track removed axis indices for VarIdxMap update
    removedAxisIndices = set()
    originalAxisCount = len(fvar.axes)

    axes = []
    for axisIdx, axis in enumerate(fvar.axes):
        tag = axis.axisTag
        if tag in selfContainedAxes:
            # Self-contained pinned axis: remove from fvar
            removedAxisIndices.add(axisIdx)
            continue
        if tag in axisLimits:
            triple = axisLimits[tag]
            defV = triple.default if triple.default is not None else axis.defaultValue
            axis.minValue = triple.minimum
            axis.defaultValue = defV
            axis.maxValue = triple.maximum
            if tag in pinnedLocation:
                # Non-self-contained pinned: keep as hidden
                axis.flags |= 0x0001  # HIDDEN_AXIS
        axes.append(axis)
    fvar.axes = axes

    # Update avar for removed axes
    if removedAxisIndices and "avar" in varfont:
        avar = varfont["avar"]

        # Remove segment maps for removed axes
        for tag in selfContainedAxes:
            avar.segments.pop(tag, None)

        # Update VarIdxMap to reflect new axis positions
        if getattr(avar, "majorVersion", 1) >= 2:
            varIdxMap = getattr(avar.table, "VarIdxMap", None)
            if varIdxMap is not None:
                varIdxMap.mapping = [
                    varIdxMap.mapping[i]
                    for i in range(len(varIdxMap.mapping))
                    if i not in removedAxisIndices
                ]
            else:
                # Implicit identity mapping: axis i → varIdx i.
                # After removing axes, need explicit mapping if non-identity.
                varIdxMap = builder.buildDeltaSetIndexMap(
                    i for i in range(originalAxisCount) if i not in removedAxisIndices
                )
                if varIdxMap is not None:
                    avar.table.VarIdxMap = varIdxMap

    # Filter named instances: keep those within the new ranges
    instances = []
    for instance in fvar.instances:
        # Check that pinned axis values match
        if any(
            instance.coordinates.get(tag) != value
            for tag, value in pinnedLocation.items()
        ):
            continue
        if not isInstanceWithinAxisRanges(instance.coordinates, axisLimits):
            continue
        # Remove coordinates for removed axes
        for tag in selfContainedAxes:
            instance.coordinates.pop(tag, None)
        instances.append(instance)
    fvar.instances = instances


def isInstanceWithinAxisRanges(location, axisRanges):
    for axisTag, coord in location.items():
        if axisTag in axisRanges:
            axisRange = axisRanges[axisTag]
            if coord < axisRange.minimum or coord > axisRange.maximum:
                return False
    return True


def instantiateFvar(varfont, axisLimits):
    # 'axisLimits' dict must contain user-space (non-normalized) coordinates

    location = axisLimits.pinnedLocation()

    fvar = varfont["fvar"]

    # drop table if we instantiate all the axes
    if set(location).issuperset(axis.axisTag for axis in fvar.axes):
        log.info("Dropping fvar table")
        del varfont["fvar"]
        return

    log.info("Instantiating fvar table")

    axes = []
    for axis in fvar.axes:
        axisTag = axis.axisTag
        if axisTag in location:
            continue
        if axisTag in axisLimits:
            triple = axisLimits[axisTag]
            if triple.default is None:
                triple = (triple.minimum, axis.defaultValue, triple.maximum)
            axis.minValue, axis.defaultValue, axis.maxValue = triple
        axes.append(axis)
    fvar.axes = axes

    # only keep NamedInstances whose coordinates == pinned axis location
    instances = []
    for instance in fvar.instances:
        if any(instance.coordinates[axis] != value for axis, value in location.items()):
            continue
        for axisTag in location:
            del instance.coordinates[axisTag]
        if not isInstanceWithinAxisRanges(instance.coordinates, axisLimits):
            continue
        instances.append(instance)
    fvar.instances = instances


def instantiateSTAT(varfont, axisLimits):
    # 'axisLimits' dict must contain user-space (non-normalized) coordinates

    stat = varfont["STAT"].table
    if not stat.DesignAxisRecord or not (
        stat.AxisValueArray and stat.AxisValueArray.AxisValue
    ):
        return  # STAT table empty, nothing to do

    log.info("Instantiating STAT table")
    newAxisValueTables = axisValuesFromAxisLimits(stat, axisLimits)
    stat.AxisValueCount = len(newAxisValueTables)
    if stat.AxisValueCount:
        stat.AxisValueArray.AxisValue = newAxisValueTables
    else:
        stat.AxisValueArray = None


def axisValuesFromAxisLimits(stat, axisLimits):
    def isAxisValueOutsideLimits(axisTag, axisValue):
        if axisTag in axisLimits:
            triple = axisLimits[axisTag]
            if axisValue < triple.minimum or axisValue > triple.maximum:
                return True
        return False

    # only keep AxisValues whose axis is not pinned nor restricted, or is pinned at the
    # exact (nominal) value, or is restricted but the value is within the new range
    designAxes = stat.DesignAxisRecord.Axis
    newAxisValueTables = []
    for axisValueTable in stat.AxisValueArray.AxisValue:
        axisValueFormat = axisValueTable.Format
        if axisValueFormat in (1, 2, 3):
            axisTag = designAxes[axisValueTable.AxisIndex].AxisTag
            if axisValueFormat == 2:
                axisValue = axisValueTable.NominalValue
            else:
                axisValue = axisValueTable.Value
            if isAxisValueOutsideLimits(axisTag, axisValue):
                continue
        elif axisValueFormat == 4:
            # drop 'non-analytic' AxisValue if _any_ AxisValueRecord doesn't match
            # the pinned location or is outside range
            dropAxisValueTable = False
            for rec in axisValueTable.AxisValueRecord:
                axisTag = designAxes[rec.AxisIndex].AxisTag
                axisValue = rec.Value
                if isAxisValueOutsideLimits(axisTag, axisValue):
                    dropAxisValueTable = True
                    break
            if dropAxisValueTable:
                continue
        else:
            log.warning("Unknown AxisValue table format (%s); ignored", axisValueFormat)
        newAxisValueTables.append(axisValueTable)
    return newAxisValueTables


def setMacOverlapFlags(glyfTable):
    flagOverlapCompound = _g_l_y_f.OVERLAP_COMPOUND
    flagOverlapSimple = _g_l_y_f.flagOverlapSimple
    for glyphName in glyfTable.keys():
        glyph = glyfTable[glyphName]
        # Set OVERLAP_COMPOUND bit for compound glyphs
        if glyph.isComposite():
            glyph.components[0].flags |= flagOverlapCompound
        # Set OVERLAP_SIMPLE bit for simple glyphs
        elif glyph.numberOfContours > 0:
            glyph.flags[0] |= flagOverlapSimple


def normalize(value, triple, avarMapping=None):
    value = normalizeValue(value, triple)
    if avarMapping:
        value = piecewiseLinearMap(value, avarMapping)
    # Quantize to F2Dot14, to avoid surprise interpolations.
    return floatToFixedToFloat(value, 14)


def _instantiateVariationTables(varfont, limits, optimize, downgradeCFF2):
    """Instantiate all variation-carrying tables at the given limits.

    Returns the new value of the downgradeCFF2 flag (instantiateCFF2 reports
    whether a downgrade to CFF is actually possible, i.e. nothing varies).
    """
    if "VARC" in varfont:
        instantiateVARC(varfont, limits)

    if "CFF2" in varfont:
        downgradeCFF2 = instantiateCFF2(varfont, limits, downgrade=downgradeCFF2)

    if "gvar" in varfont:
        instantiateGvar(varfont, limits, optimize=optimize)

    if "cvar" in varfont:
        instantiateCvar(varfont, limits)

    if "MVAR" in varfont:
        instantiateMVAR(varfont, limits)

    if "HVAR" in varfont:
        instantiateHVAR(varfont, limits)

    if "VVAR" in varfont:
        instantiateVVAR(varfont, limits)

    instantiateOTL(varfont, limits)

    instantiateFeatureVariations(varfont, limits)

    return downgradeCFF2


def sanityCheckVariableTables(varfont):
    if "fvar" not in varfont:
        raise ValueError("Missing required table fvar")
    if "gvar" in varfont:
        if "glyf" not in varfont:
            raise ValueError("Can't have gvar without glyf")


def instantiateVariableFont(
    varfont,
    axisLimits,
    inplace=False,
    optimize=True,
    overlap=OverlapMode.KEEP_AND_SET_FLAGS,
    updateFontNames=False,
    *,
    downgradeCFF2=False,
    static=False,
):
    """Instantiate variable font, either fully or partially.

    Depending on whether the `axisLimits` dictionary references all or some of the
    input varfont's axes, the output font will either be a full instance (static
    font) or a variable font with possibly less variation data.

    Args:
        varfont: a TTFont instance, which must contain at least an 'fvar' table.
        axisLimits: a dict keyed by axis tags (str) containing the coordinates (float)
            along one or more axes where the desired instance will be located.
            If the value is `None`, the default coordinate as per 'fvar' table for
            that axis is used.
            The limit values can also be (min, max) tuples for restricting an
            axis's variation range. The default axis value must be included in
            the new range.
        inplace (bool): whether to modify input TTFont object in-place instead of
            returning a distinct object.
        optimize (bool): if False, do not perform IUP-delta optimization on the
            remaining 'gvar' table's deltas. Possibly faster, and might work around
            rendering issues in some buggy environments, at the cost of a slightly
            larger file size.
        overlap (OverlapMode): variable fonts usually contain overlapping contours, and
            some font rendering engines on Apple platforms require that the
            `OVERLAP_SIMPLE` and `OVERLAP_COMPOUND` flags in the 'glyf' table be set to
            force rendering using a non-zero fill rule. Thus we always set these flags
            on all glyphs to maximise cross-compatibility of the generated instance.
            You can disable this by passing OverlapMode.KEEP_AND_DONT_SET_FLAGS.
            If you want to remove the overlaps altogether and merge overlapping
            contours and components, you can pass OverlapMode.REMOVE (or
            REMOVE_AND_IGNORE_ERRORS to not hard-fail on tricky glyphs). Note that this
            requires the skia-pathops package (available to pip install).
            The overlap parameter only has effect when generating full static instances.
        updateFontNames (bool): if True, update the instantiated font's name table using
            the Axis Value Tables from the STAT table. The name table and the style bits
            in the head and OS/2 table will be updated so they conform to the R/I/B/BI
            model. If the STAT table is missing or an Axis Value table is missing for
            a given axis coordinate, a ValueError will be raised.
        downgradeCFF2 (bool): if True, downgrade the CFF2 table to CFF table when possible
            ie. full instancing of all axes. This is useful for compatibility with older
            software that does not support CFF2. Defaults to False. Note that this
            operation also removes overlaps within glyph shapes, as CFF does not support
            overlaps but CFF2 does.
        static (bool): if True, generate a full instance (static font) instead of a partial
            instance (variable font).
    """
    # 'overlap' used to be bool and is now enum; for backward compat keep accepting bool
    overlap = OverlapMode(int(overlap))

    sanityCheckVariableTables(varfont)

    if static:
        unspecified = []
        for axis in varfont["fvar"].axes:
            if axis.axisTag not in axisLimits:
                axisLimits[axis.axisTag] = None
                unspecified.append(axis.axisTag)
        if unspecified:
            log.info("Pinning unspecified axes to default: %s", unspecified)

    axisLimits = AxisLimits(axisLimits).limitAxesAndPopulateDefaults(varfont)

    log.info("Restricted limits: %s", axisLimits)

    normalizedLimits = axisLimits.normalize(varfont)

    log.info("Normalized limits: %s", normalizedLimits)

    if not inplace:
        varfont = deepcopy(varfont)

    if "DSIG" in varfont:
        del varfont["DSIG"]

    if updateFontNames:
        log.info("Updating name table")
        names.updateNameTable(varfont, axisLimits)

    # Detect avar2 partial instancing: avar version 2, and not all fvar axes
    # are pinned. In this mode, variation tables (gvar, HVAR, etc.) are kept
    # in the old final-coordinate space, and the avar v2 offset compensation
    # ensures the correct old-space final coordinates are produced. A version
    # 2 font is never downgraded to v1 — even with a NULL/empty VarStore, the
    # avar2 path applies (offset compensation creates rows on demand).
    _isAvar2PartialInstancing = (
        "avar" in varfont
        and getattr(varfont["avar"], "majorVersion", 1) >= 2
        and not set(axisLimits.pinnedLocation()).issuperset(
            a.axisTag for a in varfont["fvar"].axes
        )
    )

    if _isAvar2PartialInstancing:
        log.info(
            "avar2 partial instancing: keeping variation tables in old "
            "coordinate space"
        )
        if downgradeCFF2:
            # The result of avar2 partial instancing is still variable, so a
            # downgrade to CFF is never possible.
            log.warning("downgradeCFF2 is ignored for avar2 partial instancing")
            downgradeCFF2 = False
    else:
        downgradeCFF2 = _instantiateVariationTables(
            varfont, normalizedLimits, optimize, downgradeCFF2
        )

    # For avar2 partial instancing, compute old intermediate values BEFORE
    # avar instancing modifies the segments. Used for both NO_VARIATION_INDEX
    # reachable ranges and offset compensation in _instantiateAvarV2.
    reachableRanges = {}
    oldIntermediates = None
    if _isAvar2PartialInstancing:
        oldIntermediates = _computeOldIntermediates(varfont, axisLimits)

        # For NO_VARIATION_INDEX axes (no IVS delta), the final coord equals
        # the intermediate coord, so the reachable range is [a_i, b_i].
        avar = varfont["avar"]
        varStore = getattr(avar.table, "VarStore", None)
        varIdxMap = getattr(avar.table, "VarIdxMap", None)
        fvarAxes = varfont["fvar"].axes

        for axisIdx, axis in enumerate(fvarAxes):
            tag = axis.axisTag
            if tag not in axisLimits:
                continue
            triple = axisLimits[tag]
            if triple.minimum == triple.maximum:
                continue  # Pinned axis, not restricted
            if tag not in oldIntermediates:
                continue

            if (
                _effectiveAvar2VarIdx(varStore, varIdxMap, axisIdx)
                == NO_VARIATION_INDEX
            ):
                a_i, d_i, b_i = oldIntermediates[tag]
                reachableRanges[tag] = (min(a_i, b_i), max(a_i, b_i))

    selfContainedAxes = {}
    if "avar" in varfont:
        selfContainedAxes = (
            instantiateAvar(varfont, axisLimits, normalizedLimits, oldIntermediates)
            or {}
        )

    # For avar2 partial instancing, run variation instancing for
    # self-contained pinned axes (those that can be removed from fvar).
    # These axes have a constant old-space final coordinate, so we can
    # fold their contributions into gvar/HVAR/etc. at that coordinate.
    if _isAvar2PartialInstancing and selfContainedAxes:
        scLimits = NormalizedAxisLimits(
            {
                tag: NormalizedAxisTripleAndDistances(v, v, v)
                for tag, v in selfContainedAxes.items()
            }
        )
        log.info(
            "Instancing variation tables for self-contained axes: %s",
            list(selfContainedAxes),
        )

        downgradeCFF2 = _instantiateVariationTables(
            varfont, scLimits, optimize, downgradeCFF2
        )

    with names.pruningUnusedNames(varfont):
        if "STAT" in varfont:
            instantiateSTAT(varfont, axisLimits)

        if _isAvar2PartialInstancing:
            _instantiateFvarForAvar2(varfont, axisLimits, selfContainedAxes)
        else:
            instantiateFvar(varfont, axisLimits)

    # Cull dead gvar/cvar regions outside reachable old-space final-coord
    # ranges. Must be after _instantiateFvarForAvar2 so fvar and VarStore
    # axis orders match for getExtremes.
    if _isAvar2PartialInstancing:
        # Extend reachable ranges for axes with IVS entries (conservative
        # bounds via getExtremes on the instanced avar v2 VarStore).
        _computeReachableRangesForAvar2(varfont, axisLimits, reachableRanges)

        if reachableRanges:
            _cullVariationsForAvar2(varfont, reachableRanges)

    if "OS/2" in varfont:
        varfont["OS/2"].recalcAvgCharWidth(varfont)

    varLib.set_default_weight_width_slant(
        varfont, location=axisLimits.defaultLocation()
    )

    if updateFontNames:
        # Set Regular/Italic/Bold/Bold Italic bits as appropriate, after the
        # name table has been updated.
        setRibbiBits(varfont)

    if downgradeCFF2:
        origVarfont = varfont
        varfont = downgradeCFF2ToCFF(varfont)
        if inplace:
            origVarfont.__dict__ = varfont.__dict__.copy()

    if "fvar" not in varfont:
        if overlap == OverlapMode.KEEP_AND_SET_FLAGS:
            if "glyf" in varfont:
                setMacOverlapFlags(varfont["glyf"])
        elif overlap in (OverlapMode.REMOVE, OverlapMode.REMOVE_AND_IGNORE_ERRORS):
            from fontTools.ttLib.removeOverlaps import removeOverlaps

            log.info("Removing glyph outlines overlaps")
            removeOverlaps(
                varfont,
                ignoreErrors=(overlap == OverlapMode.REMOVE_AND_IGNORE_ERRORS),
            )

    return varfont


def setRibbiBits(font):
    """Set the `head.macStyle` and `OS/2.fsSelection` style bits
    appropriately."""

    english_ribbi_style = font["name"].getName(names.NameID.SUBFAMILY_NAME, 3, 1, 0x409)
    if english_ribbi_style is None:
        return

    styleMapStyleName = english_ribbi_style.toStr().lower()
    if styleMapStyleName not in {"regular", "bold", "italic", "bold italic"}:
        return

    if styleMapStyleName == "bold":
        font["head"].macStyle = 0b01
    elif styleMapStyleName == "bold italic":
        font["head"].macStyle = 0b11
    elif styleMapStyleName == "italic":
        font["head"].macStyle = 0b10

    selection = font["OS/2"].fsSelection
    # First clear...
    selection &= ~(1 << 0)
    selection &= ~(1 << 5)
    selection &= ~(1 << 6)
    # ...then re-set the bits.
    if styleMapStyleName == "regular":
        selection |= 1 << 6
    elif styleMapStyleName == "bold":
        selection |= 1 << 5
    elif styleMapStyleName == "italic":
        selection |= 1 << 0
    elif styleMapStyleName == "bold italic":
        selection |= 1 << 0
        selection |= 1 << 5
    font["OS/2"].fsSelection = selection


def parseLimits(limits: Iterable[str]) -> Dict[str, Optional[AxisTriple]]:
    result = {}
    for limitString in limits:
        match = re.match(
            r"^(\w{1,4})=(?:(drop)|(?:([^:]*)(?:[:]([^:]*))?(?:[:]([^:]*))?))$",
            limitString,
        )
        if not match:
            raise ValueError("invalid location format: %r" % limitString)
        tag = match.group(1).ljust(4)

        if match.group(2):  # 'drop'
            result[tag] = None
            continue

        triple = match.group(3, 4, 5)

        if triple[1] is None:  # "value" syntax
            triple = (triple[0], triple[0], triple[0])
        elif triple[2] is None:  # "min:max" syntax
            triple = (triple[0], None, triple[1])

        triple = tuple(float(v) if v else None for v in triple)

        result[tag] = AxisTriple(*triple)

    return result


def parseArgs(args):
    """Parse argv.

    Returns:
        3-tuple (infile, axisLimits, options)
        axisLimits is either a Dict[str, Optional[float]], for pinning variation axes
        to specific coordinates along those axes (with `None` as a placeholder for an
        axis' default value); or a Dict[str, Tuple(float, float)], meaning limit this
        axis to min/max range.
        Axes locations are in user-space coordinates, as defined in the "fvar" table.
    """
    from fontTools import configLogger
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.instancer",
        description="Partially instantiate a variable font",
    )
    parser.add_argument("input", metavar="INPUT.ttf", help="Input variable TTF file.")
    parser.add_argument(
        "locargs",
        metavar="AXIS=LOC",
        nargs="*",
        help="List of space separated locations. A location consists of "
        "the tag of a variation axis, followed by '=' and the literal, "
        "string 'drop', or colon-separated list of one to three values, "
        "each of which is the empty string, or a number. "
        "E.g.: wdth=100 or wght=75.0:125.0 or wght=100:400:700 or wght=:500: "
        "or wght=drop",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT.ttf",
        default=None,
        help="Output instance TTF file (default: INPUT-instance.ttf).",
    )
    parser.add_argument(
        "--static",
        dest="static",
        action="store_true",
        help="Make a static font: pin unspecified axes to their default location.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Don't perform IUP optimization on the remaining gvar TupleVariations",
    )
    parser.add_argument(
        "--no-overlap-flag",
        dest="overlap",
        action="store_false",
        help="Don't set OVERLAP_SIMPLE/OVERLAP_COMPOUND glyf flags (only applicable "
        "when generating a full instance)",
    )
    parser.add_argument(
        "--remove-overlaps",
        dest="remove_overlaps",
        action="store_true",
        help="Merge overlapping contours and components (only applicable "
        "when generating a full instance). Requires skia-pathops",
    )
    parser.add_argument(
        "--ignore-overlap-errors",
        dest="ignore_overlap_errors",
        action="store_true",
        help="Don't crash if the remove-overlaps operation fails for some glyphs.",
    )
    parser.add_argument(
        "--update-name-table",
        action="store_true",
        help="Update the instantiated font's `name` table. Input font must have "
        "a STAT table with Axis Value Tables",
    )
    parser.add_argument(
        "--downgrade-cff2",
        action="store_true",
        help="If all axes are pinned, downgrade CFF2 to CFF table format",
    )
    parser.add_argument(
        "--no-recalc-timestamp",
        dest="recalc_timestamp",
        action="store_false",
        help="Don't set the output font's timestamp to the current time.",
    )
    parser.add_argument(
        "--no-recalc-bounds",
        dest="recalc_bounds",
        action="store_false",
        help="Don't recalculate font bounding boxes",
    )
    loggingGroup = parser.add_mutually_exclusive_group(required=False)
    loggingGroup.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )
    loggingGroup.add_argument(
        "-q", "--quiet", action="store_true", help="Turn verbosity off."
    )
    options = parser.parse_args(args)

    if options.remove_overlaps:
        if options.ignore_overlap_errors:
            options.overlap = OverlapMode.REMOVE_AND_IGNORE_ERRORS
        else:
            options.overlap = OverlapMode.REMOVE
    else:
        options.overlap = OverlapMode(int(options.overlap))

    infile = options.input
    if not os.path.isfile(infile):
        parser.error("No such file '{}'".format(infile))

    configLogger(
        level=("DEBUG" if options.verbose else "ERROR" if options.quiet else "INFO")
    )

    try:
        axisLimits = parseLimits(options.locargs)
    except ValueError as e:
        parser.error(str(e))

    if len(axisLimits) != len(options.locargs):
        parser.error("Specified multiple limits for the same axis")

    return (infile, axisLimits, options)


def main(args=None):
    """Partially instantiate a variable font"""
    infile, axisLimits, options = parseArgs(args)
    log.info("Restricting axes: %s", axisLimits)

    log.info("Loading variable font")
    varfont = TTFont(
        infile,
        recalcTimestamp=options.recalc_timestamp,
        recalcBBoxes=options.recalc_bounds,
    )

    isFullInstance = options.static or {
        axisTag
        for axisTag, limit in axisLimits.items()
        if limit is None or limit[0] == limit[2]
    }.issuperset(axis.axisTag for axis in varfont["fvar"].axes)

    varfont = instantiateVariableFont(
        varfont,
        axisLimits,
        inplace=True,
        optimize=options.optimize,
        overlap=options.overlap,
        updateFontNames=options.update_name_table,
        downgradeCFF2=options.downgrade_cff2,
        static=options.static,
    )

    suffix = "-instance" if isFullInstance else "-partial"
    outfile = (
        makeOutputFileName(infile, overWrite=True, suffix=suffix)
        if not options.output
        else options.output
    )

    log.info(
        "Saving %s font %s",
        "instance" if isFullInstance else "partial variable",
        outfile,
    )
    varfont.save(outfile)
<|endoftext|>"""Helpers for instantiating name table records."""

from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re


class NameID(IntEnum):
    FAMILY_NAME = 1
    SUBFAMILY_NAME = 2
    UNIQUE_FONT_IDENTIFIER = 3
    FULL_FONT_NAME = 4
    VERSION_STRING = 5
    POSTSCRIPT_NAME = 6
    TYPOGRAPHIC_FAMILY_NAME = 16
    TYPOGRAPHIC_SUBFAMILY_NAME = 17
    VARIATIONS_POSTSCRIPT_NAME_PREFIX = 25


ELIDABLE_AXIS_VALUE_NAME = 2


def getVariationNameIDs(varfont):
    used = []
    if "fvar" in varfont:
        fvar = varfont["fvar"]
        for axis in fvar.axes:
            used.append(axis.axisNameID)
        for instance in fvar.instances:
            used.append(instance.subfamilyNameID)
            if instance.postscriptNameID != 0xFFFF:
                used.append(instance.postscriptNameID)
    if "STAT" in varfont:
        stat = varfont["STAT"].table
        for axis in stat.DesignAxisRecord.Axis if stat.DesignAxisRecord else ():
            used.append(axis.AxisNameID)
        for value in stat.AxisValueArray.AxisValue if stat.AxisValueArray else ():
            used.append(value.ValueNameID)
        elidedFallbackNameID = getattr(stat, "ElidedFallbackNameID", None)
        if elidedFallbackNameID is not None:
            used.append(elidedFallbackNameID)
    # nameIDs <= 255 are reserved by OT spec so we don't touch them
    return {nameID for nameID in used if nameID > 255}


@contextmanager
def pruningUnusedNames(varfont):
    from . import log

    origNameIDs = getVariationNameIDs(varfont)

    yield

    log.info("Pruning name table")
    exclude = origNameIDs - getVariationNameIDs(varfont)
    varfont["name"].names[:] = [
        record for record in varfont["name"].names if record.nameID not in exclude
    ]
    if "ltag" in varfont:
        # Drop the whole 'ltag' table if all the language-dependent Unicode name
        # records that reference it have been dropped.
        # TODO: Only prune unused ltag tags, renumerating langIDs accordingly.
        # Note ltag can also be used by feat or morx tables, so check those too.
        if not any(
            record
            for record in varfont["name"].names
            if record.platformID == 0 and record.langID != 0xFFFF
        ):
            del varfont["ltag"]


def updateNameTable(varfont, axisLimits):
    """Update instatiated variable font's name table using STAT AxisValues.

    Raises ValueError if the STAT table is missing or an Axis Value table is
    missing for requested axis locations.

    First, collect all STAT AxisValues that match the new default axis locations
    (excluding "elided" ones); concatenate the strings in design axis order,
    while giving priority to "synthetic" values (Format 4), to form the
    typographic subfamily name associated with the new default instance.
    Finally, update all related records in the name table, making sure that
    legacy family/sub-family names conform to the the R/I/B/BI (Regular, Italic,
    Bold, Bold Italic) naming model.

    Example: Updating a partial variable font:
    | >>> ttFont = TTFont("OpenSans[wdth,wght].ttf")
    | >>> updateNameTable(ttFont, {"wght": (400, 900), "wdth": 75})

    The name table records will be updated in the following manner:
    NameID 1 familyName: "Open Sans" --> "Open Sans Condensed"
    NameID 2 subFamilyName: "Regular" --> "Regular"
    NameID 3 Unique font identifier: "3.000;GOOG;OpenSans-Regular" --> \
        "3.000;GOOG;OpenSans-Condensed"
    NameID 4 Full font name: "Open Sans Regular" --> "Open Sans Condensed"
    NameID 6 PostScript name: "OpenSans-Regular" --> "OpenSans-Condensed"
    NameID 16 Typographic Family name: None --> "Open Sans"
    NameID 17 Typographic Subfamily name: None --> "Condensed"

    References:
    https://docs.microsoft.com/en-us/typography/opentype/spec/stat
    https://docs.microsoft.com/en-us/typography/opentype/spec/name#name-ids
    """
    from . import AxisLimits, axisValuesFromAxisLimits

    if "STAT" not in varfont:
        raise ValueError("Cannot update name table since there is no STAT table.")
    stat = varfont["STAT"].table
    if not stat.AxisValueArray:
        raise ValueError("Cannot update name table since there are no STAT Axis Values")
    fvar = varfont["fvar"]

    # The updated name table will reflect the new 'zero origin' of the font.
    # If we're instantiating a partial font, we will populate the unpinned
    # axes with their default axis values from fvar.
    axisLimits = AxisLimits(axisLimits).limitAxesAndPopulateDefaults(varfont)
    partialDefaults = axisLimits.defaultLocation()
    fvarDefaults = {a.axisTag: a.defaultValue for a in fvar.axes}
    defaultAxisCoords = AxisLimits({**fvarDefaults, **partialDefaults})
    assert all(v.minimum == v.maximum for v in defaultAxisCoords.values())

    axisValueTables = axisValuesFromAxisLimits(stat, defaultAxisCoords)
    checkAxisValuesExist(stat, axisValueTables, defaultAxisCoords.pinnedLocation())

    # ignore "elidable" axis values, should be omitted in application font menus.
    axisValueTables = [
        v for v in axisValueTables if not v.Flags & ELIDABLE_AXIS_VALUE_NAME
    ]
    axisValueTables = _sortAxisValues(axisValueTables)
    _updateNameRecords(varfont, axisValueTables)


def checkAxisValuesExist(stat, axisValues, axisCoords):
    seen = set()
    designAxes = stat.DesignAxisRecord.Axis
    hasValues = set()
    for value in stat.AxisValueArray.AxisValue:
        if value.Format in (1, 2, 3):
            hasValues.add(designAxes[value.AxisIndex].AxisTag)
        elif value.Format == 4:
            for rec in value.AxisValueRecord:
                hasValues.add(designAxes[rec.AxisIndex].AxisTag)

    for axisValueTable in axisValues:
        axisValueFormat = axisValueTable.Format
        if axisValueTable.Format in (1, 2, 3):
            axisTag = designAxes[axisValueTable.AxisIndex].AxisTag
            if axisValueFormat == 2:
                axisValue = axisValueTable.NominalValue
            else:
                axisValue = axisValueTable.Value
            if axisTag in axisCoords and axisValue == axisCoords[axisTag]:
                seen.add(axisTag)
        elif axisValueTable.Format == 4:
            for rec in axisValueTable.AxisValueRecord:
                axisTag = designAxes[rec.AxisIndex].AxisTag
                if axisTag in axisCoords and rec.Value == axisCoords[axisTag]:
                    seen.add(axisTag)

    missingAxes = (set(axisCoords) - seen) & hasValues
    if missingAxes:
        missing = ", ".join(f"'{i}': {axisCoords[i]}" for i in missingAxes)
        raise ValueError(f"Cannot find Axis Values {{{missing}}}")


def _sortAxisValues(axisValues):
    # Sort by axis index, remove duplicates and ensure that format 4 AxisValues
    # are dominant.
    # The MS Spec states: "if a format 1, format 2 or format 3 table has a
    # (nominal) value used in a format 4 table that also has values for
    # other axes, the format 4 table, being the more specific match, is used",
    # https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-table-format-4
    results = []
    seenAxes = set()
    # Sort format 4 axes so the tables with the most AxisValueRecords are first
    format4 = sorted(
        [v for v in axisValues if v.Format == 4],
        key=lambda v: len(v.AxisValueRecord),
        reverse=True,
    )

    for val in format4:
        axisIndexes = set(r.AxisIndex for r in val.AxisValueRecord)
        minIndex = min(axisIndexes)
        if not seenAxes & axisIndexes:
            seenAxes |= axisIndexes
            results.append((minIndex, val))

    for val in axisValues:
        if val in format4:
            continue
        axisIndex = val.AxisIndex
        if axisIndex not in seenAxes:
            seenAxes.add(axisIndex)
            results.append((axisIndex, val))

    return [axisValue for _, axisValue in sorted(results)]


def _updateNameRecords(varfont, axisValues):
    # Update nametable based on the axisValues using the R/I/B/BI model.
    nametable = varfont["name"]
    stat = varfont["STAT"].table

    axisValueNameIDs = [a.ValueNameID for a in axisValues]
    ribbiNameIDs = [n for n in axisValueNameIDs if _isRibbi(nametable, n)]
    nonRibbiNameIDs = [n for n in axisValueNameIDs if n not in ribbiNameIDs]
    elidedNameID = stat.ElidedFallbackNameID
    elidedNameIsRibbi = _isRibbi(nametable, elidedNameID)

    getName = nametable.getName
    platforms = set((r.platformID, r.platEncID, r.langID) for r in nametable.names)
    for platform in platforms:
        if not all(getName(i, *platform) for i in (1, 2, elidedNameID)):
            # Since no family name and subfamily name records were found,
            # we cannot update this set of name Records.
            continue

        subFamilyName = " ".join(
            getName(n, *platform).toUnicode() for n in ribbiNameIDs
        )
        if nonRibbiNameIDs:
            typoSubFamilyName = " ".join(
                getName(n, *platform).toUnicode() for n in axisValueNameIDs
            )
        else:
            typoSubFamilyName = None

        # If neither subFamilyName and typographic SubFamilyName exist,
        # we will use the STAT's elidedFallbackName
        if not typoSubFamilyName and not subFamilyName:
            if elidedNameIsRibbi:
                subFamilyName = getName(elidedNameID, *platform).toUnicode()
            else:
                typoSubFamilyName = getName(elidedNameID, *platform).toUnicode()

        familyNameSuffix = " ".join(
            getName(n, *platform).toUnicode() for n in nonRibbiNameIDs
        )

        _updateNameTableStyleRecords(
            varfont,
            familyNameSuffix,
            subFamilyName,
            typoSubFamilyName,
            *platform,
        )


def _isRibbi(nametable, nameID):
    englishRecord = nametable.getName(nameID, 3, 1, 0x409)
    return (
        True
        if englishRecord is not None
        and englishRecord.toUnicode() in ("Regular", "Italic", "Bold", "Bold Italic")
        else False
    )


def _updateNameTableStyleRecords(
    varfont,
    familyNameSuffix,
    subFamilyName,
    typoSubFamilyName,
    platformID=3,
    platEncID=1,
    langID=0x409,
):
    # TODO (Marc F) It may be nice to make this part a standalone
    # font renamer in the future.
    nametable = varfont["name"]
    platform = (platformID, platEncID, langID)

    currentFamilyName = nametable.getName(
        NameID.TYPOGRAPHIC_FAMILY_NAME, *platform
    ) or nametable.getName(NameID.FAMILY_NAME, *platform)

    currentStyleName = nametable.getName(
        NameID.TYPOGRAPHIC_SUBFAMILY_NAME, *platform
    ) or nametable.getName(NameID.SUBFAMILY_NAME, *platform)

    if not all([currentFamilyName, currentStyleName]):
        raise ValueError(f"Missing required NameIDs 1 and 2 for platform {platform}")

    currentFamilyName = currentFamilyName.toUnicode()
    currentStyleName = currentStyleName.toUnicode()

    nameIDs = {
        NameID.FAMILY_NAME: currentFamilyName,
        NameID.SUBFAMILY_NAME: subFamilyName or "Regular",
    }
    if typoSubFamilyName:
        nameIDs[NameID.FAMILY_NAME] = f"{currentFamilyName} {familyNameSuffix}".strip()
        nameIDs[NameID.TYPOGRAPHIC_FAMILY_NAME] = currentFamilyName
        nameIDs[NameID.TYPOGRAPHIC_SUBFAMILY_NAME] = typoSubFamilyName
    else:
        # Remove previous Typographic Family and SubFamily names since they're
        # no longer required
        for nameID in (
            NameID.TYPOGRAPHIC_FAMILY_NAME,
            NameID.TYPOGRAPHIC_SUBFAMILY_NAME,
        ):
            nametable.removeNames(nameID=nameID)

    newFamilyName = (
        nameIDs.get(NameID.TYPOGRAPHIC_FAMILY_NAME) or nameIDs[NameID.FAMILY_NAME]
    )
    newStyleName = (
        nameIDs.get(NameID.TYPOGRAPHIC_SUBFAMILY_NAME) or nameIDs[NameID.SUBFAMILY_NAME]
    )

    nameIDs[NameID.FULL_FONT_NAME] = f"{newFamilyName} {newStyleName}"
    nameIDs[NameID.POSTSCRIPT_NAME] = _updatePSNameRecord(
        varfont, newFamilyName, newStyleName, platform
    )

    uniqueID = _updateUniqueIdNameRecord(varfont, nameIDs, platform)
    if uniqueID:
        nameIDs[NameID.UNIQUE_FONT_IDENTIFIER] = uniqueID

    for nameID, string in nameIDs.items():
        assert string, nameID
        nametable.setName(string, nameID, *platform)

    if "fvar" not in varfont:
        nametable.removeNames(NameID.VARIATIONS_POSTSCRIPT_NAME_PREFIX)


def _updatePSNameRecord(varfont, familyName, styleName, platform):
    # Implementation based on Adobe Technical Note #5902 :
    # https://wwwimages2.adobe.com/content/dam/acom/en/devnet/font/pdfs/5902.AdobePSNameGeneration.pdf
    nametable = varfont["name"]

    family_prefix = nametable.getName(
        NameID.VARIATIONS_POSTSCRIPT_NAME_PREFIX, *platform
    )
    if family_prefix:
        family_prefix = family_prefix.toUnicode()
    else:
        family_prefix = familyName

    psName = f"{family_prefix}-{styleName}"
    # Remove any characters other than uppercase Latin letters, lowercase
    # Latin letters, digits and hyphens.
    psName = re.sub(r"[^A-Za-z0-9-]", r"", psName)

    if len(psName) > 127:
        # Abbreviating the stylename so it fits within 127 characters whilst
        # conforming to every vendor's specification is too complex. Instead
        # we simply truncate the psname and add the required "..."
        return f"{psName[:124]}..."
    return psName


def _updateUniqueIdNameRecord(varfont, nameIDs, platform):
    nametable = varfont["name"]
    currentRecord = nametable.getName(NameID.UNIQUE_FONT_IDENTIFIER, *platform)
    if not currentRecord:
        return None

    # Check if full name and postscript name are a substring of currentRecord
    for nameID in (NameID.FULL_FONT_NAME, NameID.POSTSCRIPT_NAME):
        nameRecord = nametable.getName(nameID, *platform)
        if not nameRecord:
            continue
        if nameRecord.toUnicode() in currentRecord.toUnicode():
            return currentRecord.toUnicode().replace(
                nameRecord.toUnicode(), nameIDs[nameRecord.nameID]
            )

    # Create a new string since we couldn't find any substrings.
    fontVersion = _fontVersion(varfont, platform)
    achVendID = varfont["OS/2"].achVendID
    # Remove non-ASCII characers and trailing spaces
    vendor = re.sub(r"[^\x00-\x7F]", "", achVendID).strip()
    psName = nameIDs[NameID.POSTSCRIPT_NAME]
    return f"{fontVersion};{vendor};{psName}"


def _fontVersion(font, platform=(3, 1, 0x409)):
    nameRecord = font["name"].getName(NameID.VERSION_STRING, *platform)
    if nameRecord is None:
        return f'{font["head"].fontRevision:.3f}'
    # "Version 1.101; ttfautohint (v1.8.1.43-b0c9)" --> "1.101"
    # Also works fine with inputs "Version 1.101" or "1.101" etc
    versionNumber = nameRecord.toUnicode().split(";")[0]
    return versionNumber.lstrip("Version ").strip()
<|endoftext|>from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging

log = logging.getLogger("fontTools.varLib.instancer")


def _featureVariationRecordIsUnique(rec, seen):
    conditionSet = []
    conditionSets = (
        rec.ConditionSet.ConditionTable if rec.ConditionSet is not None else []
    )
    for cond in conditionSets:
        if cond.Format != 1:
            # can't tell whether this is duplicate, assume is unique
            return True
        conditionSet.append(
            (cond.AxisIndex, cond.FilterRangeMinValue, cond.FilterRangeMaxValue)
        )
    # besides the set of conditions, we also include the FeatureTableSubstitution
    # version to identify unique FeatureVariationRecords, even though only one
    # version is currently defined. It's theoretically possible that multiple
    # records with same conditions but different substitution table version be
    # present in the same font for backward compatibility.
    recordKey = frozenset([rec.FeatureTableSubstitution.Version] + conditionSet)
    if recordKey in seen:
        return False
    else:
        seen.add(recordKey)  # side effect
        return True


def _limitFeatureVariationConditionRange(condition, axisLimit):
    minValue = condition.FilterRangeMinValue
    maxValue = condition.FilterRangeMaxValue

    if (
        minValue > maxValue
        or minValue > axisLimit.maximum
        or maxValue < axisLimit.minimum
    ):
        # condition invalid or out of range
        return

    return tuple(
        axisLimit.renormalizeValue(v, extrapolate=False) for v in (minValue, maxValue)
    )


def _instantiateFeatureVariationRecord(
    record, recIdx, axisLimits, fvarAxes, axisIndexMap
):
    applies = True
    shouldKeep = False
    newConditions = []
    from fontTools.varLib.instancer import NormalizedAxisTripleAndDistances

    default_triple = NormalizedAxisTripleAndDistances(-1, 0, +1)
    if record.ConditionSet is None:
        record.ConditionSet = ot.ConditionSet()
        record.ConditionSet.ConditionTable = []
        record.ConditionSet.ConditionCount = 0
    for i, condition in enumerate(record.ConditionSet.ConditionTable):
        if condition.Format == 1:
            axisIdx = condition.AxisIndex
            axisTag = fvarAxes[axisIdx].axisTag

            minValue = condition.FilterRangeMinValue
            maxValue = condition.FilterRangeMaxValue
            triple = axisLimits.get(axisTag, default_triple)

            if not (minValue <= triple.default <= maxValue):
                applies = False

            # if condition not met, remove entire record
            if triple.minimum > maxValue or triple.maximum < minValue:
                newConditions = None
                break

            if axisTag in axisIndexMap:
                # remap axis index
                condition.AxisIndex = axisIndexMap[axisTag]

                # remap condition limits
                newRange = _limitFeatureVariationConditionRange(condition, triple)
                if newRange:
                    # keep condition with updated limits
                    minimum, maximum = newRange
                    condition.FilterRangeMinValue = minimum
                    condition.FilterRangeMaxValue = maximum
                    shouldKeep = True
                    if minimum != -1 or maximum != +1:
                        newConditions.append(condition)
                else:
                    # condition out of range, remove entire record
                    newConditions = None
                    break

        else:
            log.warning(
                "Condition table {0} of FeatureVariationRecord {1} has "
                "unsupported format ({2}); ignored".format(i, recIdx, condition.Format)
            )
            applies = False
            newConditions.append(condition)

    if newConditions is not None and shouldKeep:
        record.ConditionSet.ConditionTable = newConditions
        if not newConditions:
            record.ConditionSet = None
        shouldKeep = True
    else:
        shouldKeep = False

    # Does this *always* apply?
    universal = shouldKeep and not newConditions

    return applies, shouldKeep, universal


def _instantiateFeatureVariations(table, fvarAxes, axisLimits):
    pinnedAxes = set(axisLimits.pinnedLocation())
    axisOrder = [axis.axisTag for axis in fvarAxes if axis.axisTag not in pinnedAxes]
    axisIndexMap = {axisTag: axisOrder.index(axisTag) for axisTag in axisOrder}

    featureVariationApplied = False
    uniqueRecords = set()
    newRecords = []
    defaultsSubsts = None

    for i, record in enumerate(table.FeatureVariations.FeatureVariationRecord):
        applies, shouldKeep, universal = _instantiateFeatureVariationRecord(
            record, i, axisLimits, fvarAxes, axisIndexMap
        )

        if shouldKeep and _featureVariationRecordIsUnique(record, uniqueRecords):
            newRecords.append(record)

        if applies and not featureVariationApplied:
            assert record.FeatureTableSubstitution.Version == 0x00010000
            defaultsSubsts = deepcopy(record.FeatureTableSubstitution)
            for default, rec in zip(
                defaultsSubsts.SubstitutionRecord,
                record.FeatureTableSubstitution.SubstitutionRecord,
            ):
                default.Feature = deepcopy(
                    table.FeatureList.FeatureRecord[rec.FeatureIndex].Feature
                )
                table.FeatureList.FeatureRecord[rec.FeatureIndex].Feature = deepcopy(
                    rec.Feature
                )
            # Set variations only once
            featureVariationApplied = True

        # Further records don't have a chance to apply after a universal record
        if universal:
            break

    # Insert a catch-all record to reinstate the old features if necessary
    if featureVariationApplied and newRecords and not universal:
        defaultRecord = ot.FeatureVariationRecord()
        defaultRecord.ConditionSet = ot.ConditionSet()
        defaultRecord.ConditionSet.ConditionTable = []
        defaultRecord.ConditionSet.ConditionCount = 0
        defaultRecord.FeatureTableSubstitution = defaultsSubsts

        newRecords.append(defaultRecord)

    if newRecords:
        table.FeatureVariations.FeatureVariationRecord = newRecords
        table.FeatureVariations.FeatureVariationCount = len(newRecords)
    else:
        del table.FeatureVariations
        # downgrade table version if there are no FeatureVariations left
        table.Version = 0x00010000


def instantiateFeatureVariations(varfont, axisLimits):
    for tableTag in ("GPOS", "GSUB"):
        if tableTag not in varfont or not getattr(
            varfont[tableTag].table, "FeatureVariations", None
        ):
            continue
        log.info("Instantiating FeatureVariations of %s table", tableTag)
        _instantiateFeatureVariations(
            varfont[tableTag].table, varfont["fvar"].axes, axisLimits
        )
        # remove unreferenced lookups
        varfont[tableTag].prune_lookups()
<|endoftext|>from __future__ import annotations

from fontTools.misc.textTools import byteord, tostr

import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload

try:
    # use unicodedata backport compatible with python2:
    # https://github.com/fonttools/unicodedata2
    from unicodedata2 import *
except ImportError:  # pragma: no cover
    # fall back to built-in unicodedata (possibly outdated)
    from unicodedata import *

from . import Blocks, Mirrored, Scripts, ScriptExtensions, OTTags

__all__ = [
    # names from built-in unicodedata module
    "lookup",
    "name",
    "decimal",
    "digit",
    "numeric",
    "category",
    "bidirectional",
    "combining",
    "east_asian_width",
    "mirrored",
    "decomposition",
    "normalize",
    "unidata_version",
    "ucd_3_2_0",
    # additonal functions
    "block",
    "script",
    "script_extension",
    "script_name",
    "script_code",
    "script_horizontal_direction",
    "ot_tags_from_script",
    "ot_tag_to_script",
]


def mirrored(code):
    """If code (unicode codepoint) has a mirrored version returns it, otherwise None."""
    return Mirrored.MIRRORED.get(code)


def script(char):
    """Return the four-letter script code assigned to the Unicode character
    'char' as string.

    >>> script("a")
    'Latn'
    >>> script(",")
    'Zyyy'
    >>> script(chr(0x10FFFF))
    'Zzzz'
    """
    code = byteord(char)
    # 'bisect_right(a, x, lo=0, hi=len(a))' returns an insertion point which
    # comes after (to the right of) any existing entries of x in a, and it
    # partitions array a into two halves so that, for the left side
    # all(val <= x for val in a[lo:i]), and for the right side
    # all(val > x for val in a[i:hi]).
    # Our 'SCRIPT_RANGES' is a sorted list of ranges (only their starting
    # breakpoints); we want to use `bisect_right` to look up the range that
    # contains the given codepoint: i.e. whose start is less than or equal
    # to the codepoint. Thus, we subtract -1 from the index returned.
    i = bisect_right(Scripts.RANGES, code)
    return Scripts.VALUES[i - 1]


def script_extension(char):
    """Return the script extension property assigned to the Unicode character
    'char' as a set of string.

    >>> script_extension("a") == {'Latn'}
    True
    >>> script_extension(chr(0x060C)) == {'Nkoo', 'Arab', 'Rohg', 'Thaa', 'Syrc', 'Gara', 'Yezi'}
    True
    >>> script_extension(chr(0x10FFFF)) == {'Zzzz'}
    True
    """
    code = byteord(char)
    i = bisect_right(ScriptExtensions.RANGES, code)
    value = ScriptExtensions.VALUES[i - 1]
    if value is None:
        # code points not explicitly listed for Script Extensions
        # have as their value the corresponding Script property value
        return {script(char)}
    return value


def script_name(code, default=KeyError):
    """Return the long, human-readable script name given a four-letter
    Unicode script code.

    If no matching name is found, a KeyError is raised by default.

    You can use the 'default' argument to return a fallback value (e.g.
    'Unknown' or None) instead of throwing an error.
    """
    try:
        return str(Scripts.NAMES[code].replace("_", " "))
    except KeyError:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise
        return default


_normalize_re = re.compile(r"[-_ ]+")


def _normalize_property_name(string):
    """Remove case, strip space, '-' and '_' for loose matching."""
    return _normalize_re.sub("", string).lower()


_SCRIPT_CODES = {_normalize_property_name(v): k for k, v in Scripts.NAMES.items()}


def script_code(script_name, default=KeyError):
    """Returns the four-letter Unicode script code from its long name

    If no matching script code is found, a KeyError is raised by default.

    You can use the 'default' argument to return a fallback string (e.g.
    'Zzzz' or None) instead of throwing an error.
    """
    normalized_name = _normalize_property_name(script_name)
    try:
        return _SCRIPT_CODES[normalized_name]
    except KeyError:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise
        return default


# The data on script direction is taken from Harfbuzz source code:
# https://github.com/harfbuzz/harfbuzz/blob/3.2.0/src/hb-common.cc#L514-L613
# This in turn references the following "Script_Metadata" document:
# https://docs.google.com/spreadsheets/d/1Y90M0Ie3MUJ6UVCRDOypOtijlMDLNNyyLk36T6iMu0o
RTL_SCRIPTS = {
    # Unicode-1.1 additions
    "Arab",  # Arabic
    "Hebr",  # Hebrew
    # Unicode-3.0 additions
    "Syrc",  # Syriac
    "Thaa",  # Thaana
    # Unicode-4.0 additions
    "Cprt",  # Cypriot
    # Unicode-4.1 additions
    "Khar",  # Kharoshthi
    # Unicode-5.0 additions
    "Phnx",  # Phoenician
    "Nkoo",  # Nko
    # Unicode-5.1 additions
    "Lydi",  # Lydian
    # Unicode-5.2 additions
    "Avst",  # Avestan
    "Armi",  # Imperial Aramaic
    "Phli",  # Inscriptional Pahlavi
    "Prti",  # Inscriptional Parthian
    "Sarb",  # Old South Arabian
    "Orkh",  # Old Turkic
    "Samr",  # Samaritan
    # Unicode-6.0 additions
    "Mand",  # Mandaic
    # Unicode-6.1 additions
    "Merc",  # Meroitic Cursive
    "Mero",  # Meroitic Hieroglyphs
    # Unicode-7.0 additions
    "Mani",  # Manichaean
    "Mend",  # Mende Kikakui
    "Nbat",  # Nabataean
    "Narb",  # Old North Arabian
    "Palm",  # Palmyrene
    "Phlp",  # Psalter Pahlavi
    # Unicode-8.0 additions
    "Hatr",  # Hatran
    "Hung",  # Old Hungarian
    # Unicode-9.0 additions
    "Adlm",  # Adlam
    # Unicode-11.0 additions
    "Rohg",  # Hanifi Rohingya
    "Sogo",  # Old Sogdian
    "Sogd",  # Sogdian
    # Unicode-12.0 additions
    "Elym",  # Elymaic
    # Unicode-13.0 additions
    "Chrs",  # Chorasmian
    "Yezi",  # Yezidi
    # Unicode-14.0 additions
    "Ougr",  # Old Uyghur
    # Unicode-16.0 additions
    "Gara",  # Garay
    # Unicode-17.0 additions
    "Sidt",  # Sidetic
}


HorizDirection = Literal["RTL", "LTR"]
T = TypeVar("T")


@overload
def script_horizontal_direction(script_code: str, default: T) -> HorizDirection | T: ...


@overload
def script_horizontal_direction(
    script_code: str, default: type[KeyError] = KeyError
) -> HorizDirection: ...


def script_horizontal_direction(
    script_code: str, default: T | type[KeyError] = KeyError
) -> HorizDirection | T:
    """Return "RTL" for scripts that contain right-to-left characters
    according to the Bidi_Class property. Otherwise return "LTR".
    """
    if script_code not in Scripts.NAMES:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise default(script_code)
        return default
    return "RTL" if script_code in RTL_SCRIPTS else "LTR"


def block(char):
    """Return the block property assigned to the Unicode character 'char'
    as a string.

    >>> block("a")
    'Basic Latin'
    >>> block(chr(0x060C))
    'Arabic'
    >>> block(chr(0xEFFFF))
    'No_Block'
    """
    code = byteord(char)
    i = bisect_right(Blocks.RANGES, code)
    return Blocks.VALUES[i - 1]


def ot_tags_from_script(script_code):
    """Return a list of OpenType script tags associated with a given
    Unicode script code.
    Return ['DFLT'] script tag for invalid/unknown script codes.
    """
    if script_code in OTTags.SCRIPT_EXCEPTIONS:
        return [OTTags.SCRIPT_EXCEPTIONS[script_code]]

    if script_code not in Scripts.NAMES:
        return [OTTags.DEFAULT_SCRIPT]

    script_tags = [script_code[0].lower() + script_code[1:]]
    if script_code in OTTags.NEW_SCRIPT_TAGS:
        script_tags.extend(OTTags.NEW_SCRIPT_TAGS[script_code])
        script_tags.reverse()  # last in, first out

    return script_tags


def ot_tag_to_script(tag):
    """Return the Unicode script code for the given OpenType script tag, or
    None for "DFLT" tag or if there is no Unicode script associated with it.
    Raises ValueError if the tag is invalid.
    """
    tag = tostr(tag).strip()
    if not tag or " " in tag or len(tag) > 4:
        raise ValueError("invalid OpenType tag: %r" % tag)

    if tag in OTTags.SCRIPT_ALIASES:
        tag = OTTags.SCRIPT_ALIASES[tag]

    while len(tag) != 4:
        tag += str(" ")  # pad with spaces

    if tag == OTTags.DEFAULT_SCRIPT:
        # it's unclear which Unicode script the "DFLT" OpenType tag maps to,
        # so here we return None
        return None

    if tag in OTTags.NEW_SCRIPT_TAGS_REVERSED:
        return OTTags.NEW_SCRIPT_TAGS_REVERSED[tag]

    if tag in OTTags.SCRIPT_EXCEPTIONS_REVERSED:
        return OTTags.SCRIPT_EXCEPTIONS_REVERSED[tag]

    # This side of the conversion is fully algorithmic

    # Any spaces at the end of the tag are replaced by repeating the last
    # letter. Eg 'nko ' -> 'Nkoo'.
    # Change first char to uppercase
    script_code = tag[0].upper() + tag[1]
    for i in range(2, 4):
        script_code += script_code[i - 1] if tag[i] == " " else tag[i]

    if script_code not in Scripts.NAMES:
        return None
    return script_code
<|endoftext|># -*- coding: utf-8 -*-
"""fontTools.misc.bezierTools.py -- tools for working with Bezier path segments."""

from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple

try:
    import cython
except (AttributeError, ImportError):
    # if cython not installed, use mock module with no-op decorators and types
    from fontTools.misc import cython
COMPILED = cython.compiled


EPSILON = 1e-9


Intersection = namedtuple("Intersection", ["pt", "t1", "t2"])


__all__ = [
    "approximateCubicArcLength",
    "approximateCubicArcLengthC",
    "approximateQuadraticArcLength",
    "approximateQuadraticArcLengthC",
    "calcCubicArcLength",
    "calcCubicArcLengthC",
    "calcQuadraticArcLength",
    "calcQuadraticArcLengthC",
    "calcCubicBounds",
    "calcQuadraticBounds",
    "splitLine",
    "splitQuadratic",
    "splitCubic",
    "splitQuadraticAtT",
    "splitCubicAtT",
    "splitCubicAtTC",
    "splitCubicIntoTwoAtTC",
    "solveQuadratic",
    "solveCubic",
    "quadraticPointAtT",
    "cubicPointAtT",
    "cubicPointAtTC",
    "linePointAtT",
    "segmentPointAtT",
    "lineLineIntersections",
    "curveLineIntersections",
    "curveCurveIntersections",
    "segmentSegmentIntersections",
]


def calcCubicArcLength(pt1, pt2, pt3, pt4, tolerance=0.005):
    """Calculates the arc length for a cubic Bezier segment.

    Whereas :func:`approximateCubicArcLength` approximates the length, this
    function calculates it by "measuring", recursively dividing the curve
    until the divided segments are shorter than ``tolerance``.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.
        tolerance: Controls the precision of the calcuation.

    Returns:
        Arc length value.
    """
    return calcCubicArcLengthC(
        complex(*pt1), complex(*pt2), complex(*pt3), complex(*pt4), tolerance
    )


def _split_cubic_into_two(p0, p1, p2, p3):
    mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
    deriv3 = (p3 + p2 - p1 - p0) * 0.125
    return (
        (p0, (p0 + p1) * 0.5, mid - deriv3, mid),
        (mid, mid + deriv3, (p2 + p3) * 0.5, p3),
    )


@cython.returns(cython.double)
@cython.locals(
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
)
@cython.locals(mult=cython.double, arch=cython.double, box=cython.double)
def _calcCubicArcLengthCRecurse(mult, p0, p1, p2, p3):
    arch = abs(p0 - p3)
    box = abs(p0 - p1) + abs(p1 - p2) + abs(p2 - p3)
    if arch * mult + EPSILON >= box:
        return (arch + box) * 0.5
    else:
        one, two = _split_cubic_into_two(p0, p1, p2, p3)
        return _calcCubicArcLengthCRecurse(mult, *one) + _calcCubicArcLengthCRecurse(
            mult, *two
        )


@cython.returns(cython.double)
@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
)
@cython.locals(
    tolerance=cython.double,
    mult=cython.double,
)
def calcCubicArcLengthC(pt1, pt2, pt3, pt4, tolerance=0.005):
    """Calculates the arc length for a cubic Bezier segment.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers.
        tolerance: Controls the precision of the calcuation.

    Returns:
        Arc length value.
    """
    mult = 1.0 + 1.5 * tolerance  # The 1.5 is a empirical hack; no math
    return _calcCubicArcLengthCRecurse(mult, pt1, pt2, pt3, pt4)


epsilonDigits = 6
epsilon = 1e-10


@cython.cfunc
@cython.inline
@cython.returns(cython.double)
@cython.locals(v1=cython.complex, v2=cython.complex)
def _dot(v1, v2):
    return (v1 * v2.conjugate()).real


@cython.cfunc
@cython.inline
@cython.returns(cython.double)
@cython.locals(x=cython.double)
def _intSecAtan(x):
    # In : sympy.integrate(sp.sec(sp.atan(x)))
    # Out: x*sqrt(x**2 + 1)/2 + asinh(x)/2
    return x * math.sqrt(x**2 + 1) / 2 + math.asinh(x) / 2


def calcQuadraticArcLength(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Args:
        pt1: Start point of the Bezier as 2D tuple.
        pt2: Handle point of the Bezier as 2D tuple.
        pt3: End point of the Bezier as 2D tuple.

    Returns:
        Arc length value.

    Example::

        >>> calcQuadraticArcLength((0, 0), (0, 0), (0, 0)) # empty segment
        0.0
        >>> calcQuadraticArcLength((0, 0), (50, 0), (80, 0)) # collinear points
        80.0
        >>> calcQuadraticArcLength((0, 0), (0, 50), (0, 80)) # collinear points vertical
        80.0
        >>> calcQuadraticArcLength((0, 0), (50, 20), (100, 40)) # collinear points
        107.70329614269008
        >>> calcQuadraticArcLength((0, 0), (0, 100), (100, 0))
        154.02976155645263
        >>> calcQuadraticArcLength((0, 0), (0, 50), (100, 0))
        120.21581243984076
        >>> calcQuadraticArcLength((0, 0), (50, -10), (80, 50))
        102.53273816445825
        >>> calcQuadraticArcLength((0, 0), (40, 0), (-40, 0)) # collinear points, control point outside
        66.66666666666667
        >>> calcQuadraticArcLength((0, 0), (40, 0), (0, 0)) # collinear points, looping back
        40.0
    """
    return calcQuadraticArcLengthC(complex(*pt1), complex(*pt2), complex(*pt3))


@cython.returns(cython.double)
@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    d0=cython.complex,
    d1=cython.complex,
    d=cython.complex,
    n=cython.complex,
)
@cython.locals(
    scale=cython.double,
    origDist=cython.double,
    a=cython.double,
    b=cython.double,
    x0=cython.double,
    x1=cython.double,
    Len=cython.double,
)
def calcQuadraticArcLengthC(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Args:
        pt1: Start point of the Bezier as a complex number.
        pt2: Handle point of the Bezier as a complex number.
        pt3: End point of the Bezier as a complex number.

    Returns:
        Arc length value.
    """
    # Analytical solution to the length of a quadratic bezier.
    # Documentation: https://github.com/fonttools/fonttools/issues/3055
    d0 = pt2 - pt1
    d1 = pt3 - pt2
    d = d1 - d0
    n = d * 1j
    scale = abs(n)
    if scale == 0.0:
        return abs(pt3 - pt1)
    origDist = _dot(n, d0)
    if abs(origDist) < epsilon:
        if _dot(d0, d1) >= 0:
            return abs(pt3 - pt1)
        a, b = abs(d0), abs(d1)
        return (a * a + b * b) / (a + b)
    x0 = _dot(d, d0) / origDist
    x1 = _dot(d, d1) / origDist
    Len = abs(2 * (_intSecAtan(x1) - _intSecAtan(x0)) * origDist / (scale * (x1 - x0)))
    return Len


def approximateQuadraticArcLength(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Uses Gauss-Legendre quadrature for a branch-free approximation.
    See :func:`calcQuadraticArcLength` for a slower but more accurate result.

    Args:
        pt1: Start point of the Bezier as 2D tuple.
        pt2: Handle point of the Bezier as 2D tuple.
        pt3: End point of the Bezier as 2D tuple.

    Returns:
        Approximate arc length value.
    """
    return approximateQuadraticArcLengthC(complex(*pt1), complex(*pt2), complex(*pt3))


@cython.returns(cython.double)
@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
)
@cython.locals(
    v0=cython.double,
    v1=cython.double,
    v2=cython.double,
)
def approximateQuadraticArcLengthC(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Uses Gauss-Legendre quadrature for a branch-free approximation.
    See :func:`calcQuadraticArcLength` for a slower but more accurate result.

    Args:
        pt1: Start point of the Bezier as a complex number.
        pt2: Handle point of the Bezier as a complex number.
        pt3: End point of the Bezier as a complex number.

    Returns:
        Approximate arc length value.
    """
    # This, essentially, approximates the length-of-derivative function
    # to be integrated with the best-matching fifth-degree polynomial
    # approximation of it.
    #
    # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Legendre_quadrature

    # abs(BezierCurveC[2].diff(t).subs({t:T})) for T in sorted(.5, .5±sqrt(3/5)/2),
    # weighted 5/18, 8/18, 5/18 respectively.
    v0 = abs(
        -0.492943519233745 * pt1 + 0.430331482911935 * pt2 + 0.0626120363218102 * pt3
    )
    v1 = abs(pt3 - pt1) * 0.4444444444444444
    v2 = abs(
        -0.0626120363218102 * pt1 - 0.430331482911935 * pt2 + 0.492943519233745 * pt3
    )

    return v0 + v1 + v2


def calcQuadraticBounds(pt1, pt2, pt3):
    """Calculates the bounding rectangle for a quadratic Bezier segment.

    Args:
        pt1: Start point of the Bezier as a 2D tuple.
        pt2: Handle point of the Bezier as a 2D tuple.
        pt3: End point of the Bezier as a 2D tuple.

    Returns:
        A four-item tuple representing the bounding rectangle ``(xMin, yMin, xMax, yMax)``.

    Example::

        >>> calcQuadraticBounds((0, 0), (50, 100), (100, 0))
        (0, 0, 100, 50.0)
        >>> calcQuadraticBounds((0, 0), (100, 0), (100, 100))
        (0.0, 0.0, 100, 100)
    """
    (ax, ay), (bx, by), (cx, cy) = calcQuadraticParameters(pt1, pt2, pt3)
    ax2 = ax * 2.0
    ay2 = ay * 2.0
    roots = []
    if ax2 != 0:
        roots.append(-bx / ax2)
    if ay2 != 0:
        roots.append(-by / ay2)
    points = [
        (ax * t * t + bx * t + cx, ay * t * t + by * t + cy)
        for t in roots
        if 0 <= t < 1
    ] + [pt1, pt3]
    return calcBounds(points)


def approximateCubicArcLength(pt1, pt2, pt3, pt4):
    """Approximates the arc length for a cubic Bezier segment.

    Uses Gauss-Lobatto quadrature with n=5 points to approximate arc length.
    See :func:`calcCubicArcLength` for a slower but more accurate result.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.

    Returns:
        Arc length value.

    Example::

        >>> approximateCubicArcLength((0, 0), (25, 100), (75, 100), (100, 0))
        190.04332968932817
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 50), (100, 100))
        154.8852074945903
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 0), (150, 0)) # line; exact result should be 150.
        149.99999999999991
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 0), (-50, 0)) # cusp; exact result should be 150.
        136.9267662156362
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, -50), (-50, 0)) # cusp
        154.80848416537057
    """
    return approximateCubicArcLengthC(
        complex(*pt1), complex(*pt2), complex(*pt3), complex(*pt4)
    )


@cython.returns(cython.double)
@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
)
@cython.locals(
    v0=cython.double,
    v1=cython.double,
    v2=cython.double,
    v3=cython.double,
    v4=cython.double,
)
def approximateCubicArcLengthC(pt1, pt2, pt3, pt4):
    """Approximates the arc length for a cubic Bezier segment.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers.

    Returns:
        Arc length value.
    """
    # This, essentially, approximates the length-of-derivative function
    # to be integrated with the best-matching seventh-degree polynomial
    # approximation of it.
    #
    # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules

    # abs(BezierCurveC[3].diff(t).subs({t:T})) for T in sorted(0, .5±(3/7)**.5/2, .5, 1),
    # weighted 1/20, 49/180, 32/90, 49/180, 1/20 respectively.
    v0 = abs(pt2 - pt1) * 0.15
    v1 = abs(
        -0.558983582205757 * pt1
        + 0.325650248872424 * pt2
        + 0.208983582205757 * pt3
        + 0.024349751127576 * pt4
    )
    v2 = abs(pt4 - pt1 + pt3 - pt2) * 0.26666666666666666
    v3 = abs(
        -0.024349751127576 * pt1
        - 0.208983582205757 * pt2
        - 0.325650248872424 * pt3
        + 0.558983582205757 * pt4
    )
    v4 = abs(pt4 - pt3) * 0.15

    return v0 + v1 + v2 + v3 + v4


def calcCubicBounds(pt1, pt2, pt3, pt4):
    """Calculates the bounding rectangle for a quadratic Bezier segment.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.

    Returns:
        A four-item tuple representing the bounding rectangle ``(xMin, yMin, xMax, yMax)``.

    Example::

        >>> calcCubicBounds((0, 0), (25, 100), (75, 100), (100, 0))
        (0, 0, 100, 75.0)
        >>> calcCubicBounds((0, 0), (50, 0), (100, 50), (100, 100))
        (0.0, 0.0, 100, 100)
        >>> print("%f %f %f %f" % calcCubicBounds((50, 0), (0, 100), (100, 100), (50, 0)))
        35.566243 0.000000 64.433757 75.000000
    """
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = calcCubicParameters(pt1, pt2, pt3, pt4)
    # calc first derivative
    ax3 = ax * 3.0
    ay3 = ay * 3.0
    bx2 = bx * 2.0
    by2 = by * 2.0
    xRoots = [t for t in solveQuadratic(ax3, bx2, cx) if 0 <= t < 1]
    yRoots = [t for t in solveQuadratic(ay3, by2, cy) if 0 <= t < 1]
    roots = xRoots + yRoots

    points = [
        (
            ax * t * t * t + bx * t * t + cx * t + dx,
            ay * t * t * t + by * t * t + cy * t + dy,
        )
        for t in roots
    ] + [pt1, pt4]
    return calcBounds(points)


def splitLine(pt1, pt2, where, isHorizontal):
    """Split a line at a given coordinate.

    Args:
        pt1: Start point of line as 2D tuple.
        pt2: End point of line as 2D tuple.
        where: Position at which to split the line.
        isHorizontal: Direction of the ray splitting the line. If true,
            ``where`` is interpreted as a Y coordinate; if false, then
            ``where`` is interpreted as an X coordinate.

    Returns:
        A list of two line segments (each line segment being two 2D tuples)
        if the line was successfully split, or a list containing the original
        line.

    Example::

        >>> printSegments(splitLine((0, 0), (100, 100), 50, True))
        ((0, 0), (50, 50))
        ((50, 50), (100, 100))
        >>> printSegments(splitLine((0, 0), (100, 100), 100, True))
        ((0, 0), (100, 100))
        >>> printSegments(splitLine((0, 0), (100, 100), 0, True))
        ((0, 0), (0, 0))
        ((0, 0), (100, 100))
        >>> printSegments(splitLine((0, 0), (100, 100), 0, False))
        ((0, 0), (0, 0))
        ((0, 0), (100, 100))
        >>> printSegments(splitLine((100, 0), (0, 0), 50, False))
        ((100, 0), (50, 0))
        ((50, 0), (0, 0))
        >>> printSegments(splitLine((0, 100), (0, 0), 50, True))
        ((0, 100), (0, 50))
        ((0, 50), (0, 0))
    """
    pt1x, pt1y = pt1
    pt2x, pt2y = pt2

    ax = pt2x - pt1x
    ay = pt2y - pt1y

    bx = pt1x
    by = pt1y

    a = (ax, ay)[isHorizontal]

    if a == 0:
        return [(pt1, pt2)]
    t = (where - (bx, by)[isHorizontal]) / a
    if 0 <= t < 1:
        midPt = ax * t + bx, ay * t + by
        return [(pt1, midPt), (midPt, pt2)]
    else:
        return [(pt1, pt2)]


def splitQuadratic(pt1, pt2, pt3, where, isHorizontal):
    """Split a quadratic Bezier curve at a given coordinate.

    Args:
        pt1,pt2,pt3: Control points of the Bezier as 2D tuples.
        where: Position at which to split the curve.
        isHorizontal: Direction of the ray splitting the curve. If true,
            ``where`` is interpreted as a Y coordinate; if false, then
            ``where`` is interpreted as an X coordinate.

    Returns:
        A list of two curve segments (each curve segment being three 2D tuples)
        if the curve was successfully split, or a list containing the original
        curve.

    Example::

        >>> printSegments(splitQuadratic((0, 0), (50, 100), (100, 0), 150, False))
        ((0, 0), (50, 100), (100, 0))
        >>> printSegments(splitQuadratic((0, 0), (50, 100), (100, 0), 50, False))
        ((0, 0), (25, 50), (50, 50))
        ((50, 50), (75, 50), (100, 0))
        >>> printSegments(splitQuadratic((0, 0), (50, 100), (100, 0), 25, False))
        ((0, 0), (12.5, 25), (25, 37.5))
        ((25, 37.5), (62.5, 75), (100, 0))
        >>> printSegments(splitQuadratic((0, 0), (50, 100), (100, 0), 25, True))
        ((0, 0), (7.32233, 14.6447), (14.6447, 25))
        ((14.6447, 25), (50, 75), (85.3553, 25))
        ((85.3553, 25), (92.6777, 14.6447), (100, -7.10543e-15))
        >>> # XXX I'm not at all sure if the following behavior is desirable:
        >>> printSegments(splitQuadratic((0, 0), (50, 100), (100, 0), 50, True))
        ((0, 0), (25, 50), (50, 50))
        ((50, 50), (50, 50), (50, 50))
        ((50, 50), (75, 50), (100, 0))
    """
    a, b, c = calcQuadraticParameters(pt1, pt2, pt3)
    solutions = solveQuadratic(
        a[isHorizontal], b[isHorizontal], c[isHorizontal] - where
    )
    solutions = sorted(t for t in solutions if 0 <= t < 1)
    if not solutions:
        return [(pt1, pt2, pt3)]
    return _splitQuadraticAtT(a, b, c, *solutions)


def splitCubic(pt1, pt2, pt3, pt4, where, isHorizontal):
    """Split a cubic Bezier curve at a given coordinate.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.
        where: Position at which to split the curve.
        isHorizontal: Direction of the ray splitting the curve. If true,
            ``where`` is interpreted as a Y coordinate; if false, then
            ``where`` is interpreted as an X coordinate.

    Returns:
        A list of two curve segments (each curve segment being four 2D tuples)
        if the curve was successfully split, or a list containing the original
        curve.

    Example::

        >>> printSegments(splitCubic((0, 0), (25, 100), (75, 100), (100, 0), 150, False))
        ((0, 0), (25, 100), (75, 100), (100, 0))
        >>> printSegments(splitCubic((0, 0), (25, 100), (75, 100), (100, 0), 50, False))
        ((0, 0), (12.5, 50), (31.25, 75), (50, 75))
        ((50, 75), (68.75, 75), (87.5, 50), (100, 0))
        >>> printSegments(splitCubic((0, 0), (25, 100), (75, 100), (100, 0), 25, True))
        ((0, 0), (2.29379, 9.17517), (4.79804, 17.5085), (7.47414, 25))
        ((7.47414, 25), (31.2886, 91.6667), (68.7114, 91.6667), (92.5259, 25))
        ((92.5259, 25), (95.202, 17.5085), (97.7062, 9.17517), (100, 1.77636e-15))
    """
    a, b, c, d = calcCubicParameters(pt1, pt2, pt3, pt4)
    solutions = solveCubic(
        a[isHorizontal], b[isHorizontal], c[isHorizontal], d[isHorizontal] - where
    )
    solutions = sorted(t for t in solutions if 0 <= t < 1)
    if not solutions:
        return [(pt1, pt2, pt3, pt4)]
    return _splitCubicAtT(a, b, c, d, *solutions)


def splitQuadraticAtT(pt1, pt2, pt3, *ts):
    """Split a quadratic Bezier curve at one or more values of t.

    Args:
        pt1,pt2,pt3: Control points of the Bezier as 2D tuples.
        *ts: Positions at which to split the curve.

    Returns:
        A list of curve segments (each curve segment being three 2D tuples).

    Examples::

        >>> printSegments(splitQuadraticAtT((0, 0), (50, 100), (100, 0), 0.5))
        ((0, 0), (25, 50), (50, 50))
        ((50, 50), (75, 50), (100, 0))
        >>> printSegments(splitQuadraticAtT((0, 0), (50, 100), (100, 0), 0.5, 0.75))
        ((0, 0), (25, 50), (50, 50))
        ((50, 50), (62.5, 50), (75, 37.5))
        ((75, 37.5), (87.5, 25), (100, 0))
    """
    a, b, c = calcQuadraticParameters(pt1, pt2, pt3)
    return _splitQuadraticAtT(a, b, c, *ts)


def splitCubicAtT(pt1, pt2, pt3, pt4, *ts):
    """Split a cubic Bezier curve at one or more values of t.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.
        *ts: Positions at which to split the curve.

    Returns:
        A list of curve segments (each curve segment being four 2D tuples).

    Examples::

        >>> printSegments(splitCubicAtT((0, 0), (25, 100), (75, 100), (100, 0), 0.5))
        ((0, 0), (12.5, 50), (31.25, 75), (50, 75))
        ((50, 75), (68.75, 75), (87.5, 50), (100, 0))
        >>> printSegments(splitCubicAtT((0, 0), (25, 100), (75, 100), (100, 0), 0.5, 0.75))
        ((0, 0), (12.5, 50), (31.25, 75), (50, 75))
        ((50, 75), (59.375, 75), (68.75, 68.75), (77.3438, 56.25))
        ((77.3438, 56.25), (85.9375, 43.75), (93.75, 25), (100, 0))
    """
    a, b, c, d = calcCubicParameters(pt1, pt2, pt3, pt4)
    split = _splitCubicAtT(a, b, c, d, *ts)

    # the split impl can introduce floating point errors; we know the first
    # segment should always start at pt1 and the last segment should end at pt4,
    # so we set those values directly before returning.
    split[0] = (pt1, *split[0][1:])
    split[-1] = (*split[-1][:-1], pt4)
    return split


@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
    a=cython.complex,
    b=cython.complex,
    c=cython.complex,
    d=cython.complex,
)
def splitCubicAtTC(pt1, pt2, pt3, pt4, *ts):
    """Split a cubic Bezier curve at one or more values of t.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers..
        *ts: Positions at which to split the curve.

    Yields:
        Curve segments (each curve segment being four complex numbers).
    """
    a, b, c, d = calcCubicParametersC(pt1, pt2, pt3, pt4)
    yield from _splitCubicAtTC(a, b, c, d, *ts)


@cython.returns(cython.complex)
@cython.locals(
    t=cython.double,
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
    pointAtT=cython.complex,
    off1=cython.complex,
    off2=cython.complex,
)
@cython.locals(
    t2=cython.double, _1_t=cython.double, _1_t_2=cython.double, _2_t_1_t=cython.double
)
def splitCubicIntoTwoAtTC(pt1, pt2, pt3, pt4, t):
    """Split a cubic Bezier curve at t.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers.
        t: Position at which to split the curve.

    Returns:
        A tuple of two curve segments (each curve segment being four complex numbers).
    """
    t2 = t * t
    _1_t = 1 - t
    _1_t_2 = _1_t * _1_t
    _2_t_1_t = 2 * t * _1_t
    pointAtT = (
        _1_t_2 * _1_t * pt1 + 3 * (_1_t_2 * t * pt2 + _1_t * t2 * pt3) + t2 * t * pt4
    )
    off1 = _1_t_2 * pt1 + _2_t_1_t * pt2 + t2 * pt3
    off2 = _1_t_2 * pt2 + _2_t_1_t * pt3 + t2 * pt4

    pt2 = pt1 + (pt2 - pt1) * t
    pt3 = pt4 + (pt3 - pt4) * _1_t

    return ((pt1, pt2, off1, pointAtT), (pointAtT, off2, pt3, pt4))


def _splitQuadraticAtT(a, b, c, *ts):
    ts = list(ts)
    segments = []
    ts.insert(0, 0.0)
    ts.append(1.0)
    ax, ay = a
    bx, by = b
    cx, cy = c
    for i in range(len(ts) - 1):
        t1 = ts[i]
        t2 = ts[i + 1]
        delta = t2 - t1
        # calc new a, b and c
        delta_2 = delta * delta
        a1x = ax * delta_2
        a1y = ay * delta_2
        b1x = (2 * ax * t1 + bx) * delta
        b1y = (2 * ay * t1 + by) * delta
        t1_2 = t1 * t1
        c1x = ax * t1_2 + bx * t1 + cx
        c1y = ay * t1_2 + by * t1 + cy

        pt1, pt2, pt3 = calcQuadraticPoints((a1x, a1y), (b1x, b1y), (c1x, c1y))
        segments.append((pt1, pt2, pt3))
    return segments


def _splitCubicAtT(a, b, c, d, *ts):
    ts = list(ts)
    ts.insert(0, 0.0)
    ts.append(1.0)
    segments = []
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    for i in range(len(ts) - 1):
        t1 = ts[i]
        t2 = ts[i + 1]
        delta = t2 - t1

        delta_2 = delta * delta
        delta_3 = delta * delta_2
        t1_2 = t1 * t1
        t1_3 = t1 * t1_2

        # calc new a, b, c and d
        a1x = ax * delta_3
        a1y = ay * delta_3
        b1x = (3 * ax * t1 + bx) * delta_2
        b1y = (3 * ay * t1 + by) * delta_2
        c1x = (2 * bx * t1 + cx + 3 * ax * t1_2) * delta
        c1y = (2 * by * t1 + cy + 3 * ay * t1_2) * delta
        d1x = ax * t1_3 + bx * t1_2 + cx * t1 + dx
        d1y = ay * t1_3 + by * t1_2 + cy * t1 + dy
        pt1, pt2, pt3, pt4 = calcCubicPoints(
            (a1x, a1y), (b1x, b1y), (c1x, c1y), (d1x, d1y)
        )
        segments.append((pt1, pt2, pt3, pt4))
    return segments


@cython.locals(
    a=cython.complex,
    b=cython.complex,
    c=cython.complex,
    d=cython.complex,
    t1=cython.double,
    t2=cython.double,
    delta=cython.double,
    delta_2=cython.double,
    delta_3=cython.double,
    a1=cython.complex,
    b1=cython.complex,
    c1=cython.complex,
    d1=cython.complex,
)
def _splitCubicAtTC(a, b, c, d, *ts):
    ts = list(ts)
    ts.insert(0, 0.0)
    ts.append(1.0)
    for i in range(len(ts) - 1):
        t1 = ts[i]
        t2 = ts[i + 1]
        delta = t2 - t1

        delta_2 = delta * delta
        delta_3 = delta * delta_2
        t1_2 = t1 * t1
        t1_3 = t1 * t1_2

        # calc new a, b, c and d
        a1 = a * delta_3
        b1 = (3 * a * t1 + b) * delta_2
        c1 = (2 * b * t1 + c + 3 * a * t1_2) * delta
        d1 = a * t1_3 + b * t1_2 + c * t1 + d
        pt1, pt2, pt3, pt4 = calcCubicPointsC(a1, b1, c1, d1)
        yield (pt1, pt2, pt3, pt4)


#
# Equation solvers.
#

from math import sqrt, acos, cos, pi


def solveQuadratic(a, b, c, sqrt=sqrt):
    """Solve a quadratic equation.

    Solves *a*x*x + b*x + c = 0* where a, b and c are real.

    Args:
        a: coefficient of *x²*
        b: coefficient of *x*
        c: constant term

    Returns:
        A list of roots. Note that the returned list is neither guaranteed to
        be sorted nor to contain unique values!
    """
    if abs(a) < epsilon:
        if abs(b) < epsilon:
            # We have a non-equation; therefore, we have no valid solution
            roots = []
        else:
            # We have a linear equation with 1 root.
            roots = [-c / b]
    else:
        # We have a true quadratic equation.  Apply the quadratic formula to find two roots.
        DD = b * b - 4.0 * a * c
        if DD >= 0.0:
            rDD = sqrt(DD)
            roots = [(-b + rDD) / 2.0 / a, (-b - rDD) / 2.0 / a]
        else:
            # complex roots, ignore
            roots = []
    return roots


def solveCubic(a, b, c, d):
    """Solve a cubic equation.

    Solves *a*x*x*x + b*x*x + c*x + d = 0* where a, b, c and d are real.

    Args:
        a: coefficient of *x³*
        b: coefficient of *x²*
        c: coefficient of *x*
        d: constant term

    Returns:
        A list of roots. Note that the returned list is neither guaranteed to
        be sorted nor to contain unique values!

    Examples::

        >>> solveCubic(1, 1, -6, 0)
        [-3.0, -0.0, 2.0]
        >>> solveCubic(-10.0, -9.0, 48.0, -29.0)
        [-2.9, 1.0, 1.0]
        >>> solveCubic(-9.875, -9.0, 47.625, -28.75)
        [-2.911392, 1.0, 1.0]
        >>> solveCubic(1.0, -4.5, 6.75, -3.375)
        [1.5, 1.5, 1.5]
        >>> solveCubic(-12.0, 18.0, -9.0, 1.50023651123)
        [0.5, 0.5, 0.5]
        >>> solveCubic(
        ...     9.0, 0.0, 0.0, -7.62939453125e-05
        ... ) == [-0.0, -0.0, -0.0]
        True
    """
    #
    # adapted from:
    #   CUBIC.C - Solve a cubic polynomial
    #   public domain by Ross Cottrell
    # found at: http://www.strangecreations.com/library/snippets/Cubic.C
    #
    if abs(a) < epsilon:
        # don't just test for zero; for very small values of 'a' solveCubic()
        # returns unreliable results, so we fall back to quad.
        return solveQuadratic(b, c, d)
    a = float(a)
    a1 = b / a
    a2 = c / a
    a3 = d / a

    Q = (a1 * a1 - 3.0 * a2) / 9.0
    R = (2.0 * a1 * a1 * a1 - 9.0 * a1 * a2 + 27.0 * a3) / 54.0

    R2 = R * R
    Q3 = Q * Q * Q
    R2 = 0 if R2 < epsilon else R2
    Q3 = 0 if abs(Q3) < epsilon else Q3

    R2_Q3 = R2 - Q3

    if R2 == 0.0 and Q3 == 0.0:
        x = round(-a1 / 3.0, epsilonDigits)
        return [x, x, x]
    elif R2_Q3 <= epsilon * 0.5:
        # The epsilon * .5 above ensures that Q3 is not zero.
        theta = acos(max(min(R / sqrt(Q3), 1.0), -1.0))
        rQ2 = -2.0 * sqrt(Q)
        a1_3 = a1 / 3.0
        x0 = rQ2 * cos(theta / 3.0) - a1_3
        x1 = rQ2 * cos((theta + 2.0 * pi) / 3.0) - a1_3
        x2 = rQ2 * cos((theta + 4.0 * pi) / 3.0) - a1_3
        x0, x1, x2 = sorted([x0, x1, x2])
        # Merge roots that are close-enough
        if x1 - x0 < epsilon and x2 - x1 < epsilon:
            x0 = x1 = x2 = round((x0 + x1 + x2) / 3.0, epsilonDigits)
        elif x1 - x0 < epsilon:
            x0 = x1 = round((x0 + x1) / 2.0, epsilonDigits)
            x2 = round(x2, epsilonDigits)
        elif x2 - x1 < epsilon:
            x0 = round(x0, epsilonDigits)
            x1 = x2 = round((x1 + x2) / 2.0, epsilonDigits)
        else:
            x0 = round(x0, epsilonDigits)
            x1 = round(x1, epsilonDigits)
            x2 = round(x2, epsilonDigits)
        return [x0, x1, x2]
    else:
        x = pow(sqrt(R2_Q3) + abs(R), 1 / 3.0)
        x = x + Q / x
        if R >= 0.0:
            x = -x
        x = round(x - a1 / 3.0, epsilonDigits)
        return [x]


#
# Conversion routines for points to parameters and vice versa
#


def calcQuadraticParameters(pt1, pt2, pt3):
    x2, y2 = pt2
    x3, y3 = pt3
    cx, cy = pt1
    bx = (x2 - cx) * 2.0
    by = (y2 - cy) * 2.0
    ax = x3 - cx - bx
    ay = y3 - cy - by
    return (ax, ay), (bx, by), (cx, cy)


def calcCubicParameters(pt1, pt2, pt3, pt4):
    x2, y2 = pt2
    x3, y3 = pt3
    x4, y4 = pt4
    dx, dy = pt1
    cx = (x2 - dx) * 3.0
    cy = (y2 - dy) * 3.0
    bx = (x3 - x2) * 3.0 - cx
    by = (y3 - y2) * 3.0 - cy
    ax = x4 - dx - cx - bx
    ay = y4 - dy - cy - by
    return (ax, ay), (bx, by), (cx, cy), (dx, dy)


@cython.cfunc
@cython.inline
@cython.locals(
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
    a=cython.complex,
    b=cython.complex,
    c=cython.complex,
)
def calcCubicParametersC(pt1, pt2, pt3, pt4):
    c = (pt2 - pt1) * 3.0
    b = (pt3 - pt2) * 3.0 - c
    a = pt4 - pt1 - c - b
    return (a, b, c, pt1)


def calcQuadraticPoints(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    x1 = cx
    y1 = cy
    x2 = (bx * 0.5) + cx
    y2 = (by * 0.5) + cy
    x3 = ax + bx + cx
    y3 = ay + by + cy
    return (x1, y1), (x2, y2), (x3, y3)


def calcCubicPoints(a, b, c, d):
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    x1 = dx
    y1 = dy
    x2 = (cx / 3.0) + dx
    y2 = (cy / 3.0) + dy
    x3 = (bx + cx) / 3.0 + x2
    y3 = (by + cy) / 3.0 + y2
    x4 = ax + dx + cx + bx
    y4 = ay + dy + cy + by
    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)


@cython.cfunc
@cython.inline
@cython.locals(
    a=cython.complex,
    b=cython.complex,
    c=cython.complex,
    d=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
    p4=cython.complex,
)
def calcCubicPointsC(a, b, c, d):
    p2 = c * (1 / 3) + d
    p3 = (b + c) * (1 / 3) + p2
    p4 = a + b + c + d
    return (d, p2, p3, p4)


#
# Point at time
#


def linePointAtT(pt1, pt2, t):
    """Finds the point at time `t` on a line.

    Args:
        pt1, pt2: Coordinates of the line as 2D tuples.
        t: The time along the line.

    Returns:
        A 2D tuple with the coordinates of the point.
    """
    return ((pt1[0] * (1 - t) + pt2[0] * t), (pt1[1] * (1 - t) + pt2[1] * t))


def quadraticPointAtT(pt1, pt2, pt3, t):
    """Finds the point at time `t` on a quadratic curve.

    Args:
        pt1, pt2, pt3: Coordinates of the curve as 2D tuples.
        t: The time along the curve.

    Returns:
        A 2D tuple with the coordinates of the point.
    """
    x = (1 - t) * (1 - t) * pt1[0] + 2 * (1 - t) * t * pt2[0] + t * t * pt3[0]
    y = (1 - t) * (1 - t) * pt1[1] + 2 * (1 - t) * t * pt2[1] + t * t * pt3[1]
    return (x, y)


def cubicPointAtT(pt1, pt2, pt3, pt4, t):
    """Finds the point at time `t` on a cubic curve.

    Args:
        pt1, pt2, pt3, pt4: Coordinates of the curve as 2D tuples.
        t: The time along the curve.

    Returns:
        A 2D tuple with the coordinates of the point.
    """
    t2 = t * t
    _1_t = 1 - t
    _1_t_2 = _1_t * _1_t
    x = (
        _1_t_2 * _1_t * pt1[0]
        + 3 * (_1_t_2 * t * pt2[0] + _1_t * t2 * pt3[0])
        + t2 * t * pt4[0]
    )
    y = (
        _1_t_2 * _1_t * pt1[1]
        + 3 * (_1_t_2 * t * pt2[1] + _1_t * t2 * pt3[1])
        + t2 * t * pt4[1]
    )
    return (x, y)


@cython.returns(cython.complex)
@cython.locals(
    t=cython.double,
    pt1=cython.complex,
    pt2=cython.complex,
    pt3=cython.complex,
    pt4=cython.complex,
)
@cython.locals(t2=cython.double, _1_t=cython.double, _1_t_2=cython.double)
def cubicPointAtTC(pt1, pt2, pt3, pt4, t):
    """Finds the point at time `t` on a cubic curve.

    Args:
        pt1, pt2, pt3, pt4: Coordinates of the curve as complex numbers.
        t: The time along the curve.

    Returns:
        A complex number with the coordinates of the point.
    """
    t2 = t * t
    _1_t = 1 - t
    _1_t_2 = _1_t * _1_t
    return _1_t_2 * _1_t * pt1 + 3 * (_1_t_2 * t * pt2 + _1_t * t2 * pt3) + t2 * t * pt4


def segmentPointAtT(seg, t):
    if len(seg) == 2:
        return linePointAtT(*seg, t)
    elif len(seg) == 3:
        return quadraticPointAtT(*seg, t)
    elif len(seg) == 4:
        return cubicPointAtT(*seg, t)
    raise ValueError("Unknown curve degree")


#
# Intersection finders
#


def _line_t_of_pt(s, e, pt):
    sx, sy = s
    ex, ey = e
    px, py = pt
    if abs(sx - ex) < epsilon and abs(sy - ey) < epsilon:
        # Line is a point!
        return -1
    # Use the largest
    if abs(sx - ex) > abs(sy - ey):
        return (px - sx) / (ex - sx)
    else:
        return (py - sy) / (ey - sy)


def _both_points_are_on_same_side_of_origin(a, b, origin):
    xDiff = (a[0] - origin[0]) * (b[0] - origin[0])
    yDiff = (a[1] - origin[1]) * (b[1] - origin[1])
    return not (xDiff <= 0.0 and yDiff <= 0.0)


def lineLineIntersections(s1, e1, s2, e2):
    """Finds intersections between two line segments.

    Args:
        s1, e1: Coordinates of the first line as 2D tuples.
        s2, e2: Coordinates of the second line as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::

        >>> a = lineLineIntersections( (310,389), (453, 222), (289, 251), (447, 367))
        >>> len(a)
        1
        >>> intersection = a[0]
        >>> intersection.pt
        (374.44882952482897, 313.73458370177315)
        >>> (intersection.t1, intersection.t2)
        (0.45069111555824465, 0.5408153767394238)
    """
    s1x, s1y = s1
    e1x, e1y = e1
    s2x, s2y = s2
    e2x, e2y = e2
    if (
        math.isclose(s2x, e2x) and math.isclose(s1x, e1x) and not math.isclose(s1x, s2x)
    ):  # Parallel vertical
        return []
    if (
        math.isclose(s2y, e2y) and math.isclose(s1y, e1y) and not math.isclose(s1y, s2y)
    ):  # Parallel horizontal
        return []
    if math.isclose(s2x, e2x) and math.isclose(s2y, e2y):  # Line segment is tiny
        return []
    if math.isclose(s1x, e1x) and math.isclose(s1y, e1y):  # Line segment is tiny
        return []
    if math.isclose(e1x, s1x):
        x = s1x
        slope34 = (e2y - s2y) / (e2x - s2x)
        y = slope34 * (x - s2x) + s2y
        pt = (x, y)
        return [
            Intersection(
                pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt)
            )
        ]
    if math.isclose(s2x, e2x):
        x = s2x
        slope12 = (e1y - s1y) / (e1x - s1x)
        y = slope12 * (x - s1x) + s1y
        pt = (x, y)
        return [
            Intersection(
                pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt)
            )
        ]

    slope12 = (e1y - s1y) / (e1x - s1x)
    slope34 = (e2y - s2y) / (e2x - s2x)
    if math.isclose(slope12, slope34):
        return []
    x = (slope12 * s1x - s1y - slope34 * s2x + s2y) / (slope12 - slope34)
    y = slope12 * (x - s1x) + s1y
    pt = (x, y)
    if _both_points_are_on_same_side_of_origin(
        pt, e1, s1
    ) and _both_points_are_on_same_side_of_origin(pt, s2, e2):
        return [
            Intersection(
                pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt)
            )
        ]
    return []


def _alignment_transformation(segment):
    # Returns a transformation which aligns a segment horizontally at the
    # origin. Apply this transformation to curves and root-find to find
    # intersections with the segment.
    start = segment[0]
    end = segment[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return Identity.rotate(-angle).translate(-start[0], -start[1])


def _curve_line_intersections_t(curve, line):
    aligned_curve = _alignment_transformation(line).transformPoints(curve)
    if len(curve) == 3:
        a, b, c = calcQuadraticParameters(*aligned_curve)
        intersections = solveQuadratic(a[1], b[1], c[1])
    elif len(curve) == 4:
        a, b, c, d = calcCubicParameters(*aligned_curve)
        intersections = solveCubic(a[1], b[1], c[1], d[1])
    else:
        raise ValueError("Unknown curve degree")
    return sorted(i for i in intersections if 0.0 <= i <= 1)


def curveLineIntersections(curve, line):
    """Finds intersections between a curve and a line.

    Args:
        curve: List of coordinates of the curve segment as 2D tuples.
        line: List of coordinates of the line segment as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::
        >>> curve = [ (100, 240), (30, 60), (210, 230), (160, 30) ]
        >>> line  = [ (25, 260), (230, 20) ]
        >>> intersections = curveLineIntersections(curve, line)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (84.9000930760723, 189.87306176459828)
    """
    if len(curve) == 3:
        pointFinder = quadraticPointAtT
    elif len(curve) == 4:
        pointFinder = cubicPointAtT
    else:
        raise ValueError("Unknown curve degree")
    intersections = []
    for t in _curve_line_intersections_t(curve, line):
        pt = pointFinder(*curve, t)
        # Back-project the point onto the line, to avoid problems with
        # numerical accuracy in the case of vertical and horizontal lines
        line_t = _line_t_of_pt(*line, pt)
        pt = linePointAtT(*line, line_t)
        intersections.append(Intersection(pt=pt, t1=t, t2=line_t))
    return intersections


def _curve_bounds(c):
    if len(c) == 3:
        return calcQuadraticBounds(*c)
    elif len(c) == 4:
        return calcCubicBounds(*c)
    raise ValueError("Unknown curve degree")


def _split_segment_at_t(c, t):
    if len(c) == 2:
        s, e = c
        midpoint = linePointAtT(s, e, t)
        return [(s, midpoint), (midpoint, e)]
    if len(c) == 3:
        return splitQuadraticAtT(*c, t)
    elif len(c) == 4:
        return splitCubicAtT(*c, t)
    raise ValueError("Unknown curve degree")


def _curve_curve_intersections_t(
    curve1, curve2, precision=1e-3, range1=None, range2=None
):
    bounds1 = _curve_bounds(curve1)
    bounds2 = _curve_bounds(curve2)

    if not range1:
        range1 = (0.0, 1.0)
    if not range2:
        range2 = (0.0, 1.0)

    # If bounds don't intersect, go home
    intersects, _ = sectRect(bounds1, bounds2)
    if not intersects:
        return []

    def midpoint(r):
        return 0.5 * (r[0] + r[1])

    # If they do overlap but they're tiny, approximate
    if rectArea(bounds1) < precision and rectArea(bounds2) < precision:
