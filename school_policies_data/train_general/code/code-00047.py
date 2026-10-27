        varData = self.VarData[major]
        regions = self.VarRegionList.Region
        deltas = varData.Item[minor]
        # Per region: (delta, [(axisIndex, start, peak, end), ...]) with the
        # axes supportScalar would ignore already dropped.
        activeRegions = []
        for regionIndex, delta in zip(varData.VarRegionIndex, deltas):
            if not delta:
                continue
            region = regions[regionIndex]
            tents = []
            for i, regionAxis in enumerate(region.VarRegionAxis):
                start = regionAxis.StartCoord
                peak = regionAxis.PeakCoord
                end = regionAxis.EndCoord
                # Mirror supportScalar's OT rules for axes it ignores.
                if peak == 0 or start > peak or peak > end:
                    continue
                if start < 0 and end > 0:
                    continue
                tents.append((i, start, peak, end))
            activeRegions.append((delta, tents))

    # Try the exact path: enumerate the breakpoint grid.
    breakpoints = {}  # axisIndex -> set of coordinates
    if identityAxisIndex is not None:
        lo, hi = axisInterval(identityAxisIndex)
        breakpoints[identityAxisIndex] = {lo, hi}
    for delta, tents in activeRegions:
        for i, start, peak, end in tents:
            if i not in breakpoints:
                lo, hi = axisInterval(i)
                breakpoints[i] = {lo, hi}
            lo, hi = axisInterval(i)
            breakpoints[i].update(min(max(v, lo), hi) for v in (start, peak, end))
            if start == peak:
                breakpoints[i].add(min(max(peak - _F2DOT14_UNIT, lo), hi))
            if peak == end:
                breakpoints[i].add(min(max(peak + _F2DOT14_UNIT, lo), hi))
    gridSize = 1
    for points in breakpoints.values():
        gridSize *= len(points)
        if gridSize > _GET_EXTREMES_MAX_GRID:
            break

    if gridSize <= _GET_EXTREMES_MAX_GRID:
        # Exact: evaluate at every grid vertex. Within each grid cell every
        # region scalar is multilinear, so the extremes lie on vertices.
        axisIndices = sorted(breakpoints)
        axisPos = {i: n for n, i in enumerate(axisIndices)}
        minV = maxV = None
        for vertex in itertools.product(*(sorted(breakpoints[i]) for i in axisIndices)):
            v = 0.0
            for delta, tents in activeRegions:
                scalar = 1.0
                for i, start, peak, end in tents:
                    scalar *= _tentValue(vertex[axisPos[i]], start, peak, end)
                    if scalar == 0:
                        break
                if scalar:
                    v += delta * scalar
            if identityAxisIndex is not None:
                v += otRound(vertex[axisPos[identityAxisIndex]] * 16384)
            if minV is None:
                minV = maxV = v
            else:
                minV = min(minV, v)
                maxV = max(maxV, v)
        return otRound(minV), otRound(maxV)

    # Fallback: per-region interval arithmetic. Each region's scalar range
    # over the box is exact (per-axis tent factors are independent), but
    # correlations between regions (and with the identity term) are ignored.
    minV = maxV = 0.0
    for delta, tents in activeRegions:
        scalarMin = scalarMax = 1.0
        for i, start, peak, end in tents:
            lo, hi = axisInterval(i)
            # The tent rises to the peak then falls, so over [lo, hi] the
            # maximum is at the clamped peak, the minimum at an endpoint.
            scalarMax *= _tentValue(min(max(peak, lo), hi), start, peak, end)
            scalarMin *= min(
                _tentValue(lo, start, peak, end),
                _tentValue(hi, start, peak, end),
            )
            if scalarMax == 0:
                break
        if scalarMax == 0:
            continue
        minV += min(delta * scalarMin, delta * scalarMax)
        maxV += max(delta * scalarMin, delta * scalarMax)

    if identityAxisIndex is not None:
        lo, hi = axisInterval(identityAxisIndex)
        minV += otRound(lo * 16384)
        maxV += otRound(hi * 16384)

    return otRound(minV), otRound(maxV)


ot.VarStore.getExtremes = VarStore_getExtremes


def main(args=None):
    """Optimize a font's GDEF variation store"""
    from argparse import ArgumentParser
    from fontTools import configLogger
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.tables.otBase import OTTableWriter

    parser = ArgumentParser(prog="varLib.varStore", description=main.__doc__)
    parser.add_argument("--quantization", type=int, default=1)
    parser.add_argument("fontfile")
    parser.add_argument("outfile", nargs="?")
    options = parser.parse_args(args)

    # TODO: allow user to configure logging via command-line options
    configLogger(level="INFO")

    quantization = options.quantization
    fontfile = options.fontfile
    outfile = options.outfile

    font = TTFont(fontfile)
    gdef = font["GDEF"]
    store = gdef.table.VarStore

    writer = OTTableWriter()
    store.compile(writer, font)
    size = len(writer.getAllData())
    print("Before: %7d bytes" % size)

    varidx_map = store.optimize(quantization=quantization)

    writer = OTTableWriter()
    store.compile(writer, font)
    size = len(writer.getAllData())
    print("After:  %7d bytes" % size)

    if outfile is not None:
        gdef.table.remap_device_varidxes(varidx_map)
        if "GPOS" in font:
            font["GPOS"].table.remap_device_varidxes(varidx_map)

        font.save(outfile)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>import textwrap


class VarLibError(Exception):
    """Base exception for the varLib module."""


class VarLibValidationError(VarLibError):
    """Raised when input data is invalid from varLib's point of view."""


class VarLibMergeError(VarLibError):
    """Raised when input data cannot be merged into a variable font."""

    def __init__(self, merger=None, **kwargs):
        self.merger = merger
        if not kwargs:
            kwargs = {}
        if "stack" in kwargs:
            self.stack = kwargs["stack"]
            del kwargs["stack"]
        else:
            self.stack = []
        self.cause = kwargs

    @property
    def reason(self):
        return self.__doc__

    def _master_name(self, ix):
        if self.merger is not None:
            ttf = self.merger.ttfs[ix]
            if "name" in ttf and ttf["name"].getBestFullName():
                return ttf["name"].getBestFullName()
            elif hasattr(ttf.reader, "file") and hasattr(ttf.reader.file, "name"):
                return ttf.reader.file.name
        return f"master number {ix}"

    @property
    def offender(self):
        if "expected" in self.cause and "got" in self.cause:
            index = [x == self.cause["expected"] for x in self.cause["got"]].index(
                False
            )
            master_name = self._master_name(index)
            if "location" in self.cause:
                master_name = f"{master_name} ({self.cause['location']})"
            return index, master_name
        return None, None

    @property
    def details(self):
        if "expected" in self.cause and "got" in self.cause:
            offender_index, offender = self.offender
            got = self.cause["got"][offender_index]
            return f"Expected to see {self.stack[0]}=={self.cause['expected']!r}, instead saw {got!r}\n"
        return ""

    def __str__(self):
        offender_index, offender = self.offender
        location = ""
        if offender:
            location = f"\n\nThe problem is likely to be in {offender}:\n"
        context = "".join(reversed(self.stack))
        basic = textwrap.fill(
            f"Couldn't merge the fonts, because {self.reason}. "
            f"This happened while performing the following operation: {context}",
            width=78,
        )
        return "\n\n" + basic + location + self.details


class ShouldBeConstant(VarLibMergeError):
    """some values were different, but should have been the same"""

    @property
    def details(self):
        basic_message = super().details

        if self.stack[0] != ".FeatureCount" or self.merger is None:
            return basic_message

        assert self.stack[0] == ".FeatureCount"
        offender_index, _ = self.offender
        bad_ttf = self.merger.ttfs[offender_index]
        good_ttf = next(
            ttf
            for ttf in self.merger.ttfs
            if self.stack[-1] in ttf
            and ttf[self.stack[-1]].table.FeatureList.FeatureCount
            == self.cause["expected"]
        )

        good_features = [
            x.FeatureTag
            for x in good_ttf[self.stack[-1]].table.FeatureList.FeatureRecord
        ]
        bad_features = [
            x.FeatureTag
            for x in bad_ttf[self.stack[-1]].table.FeatureList.FeatureRecord
        ]
        return basic_message + (
            "\nIncompatible features between masters.\n"
            f"Expected: {', '.join(good_features)}.\n"
            f"Got: {', '.join(bad_features)}.\n"
        )


class FoundANone(VarLibMergeError):
    """one of the values in a list was empty when it shouldn't have been"""

    @property
    def offender(self):
        index = [x is None for x in self.cause["got"]].index(True)
        return index, self._master_name(index)

    @property
    def details(self):
        cause, stack = self.cause, self.stack
        return f"{stack[0]}=={cause['got']}\n"


class NotANone(VarLibMergeError):
    """one of the values in a list was not empty when it should have been"""

    @property
    def offender(self):
        index = [x is not None for x in self.cause["got"]].index(True)
        return index, self._master_name(index)

    @property
    def details(self):
        cause, stack = self.cause, self.stack
        return f"{stack[0]}=={cause['got']}\n"


class MismatchedTypes(VarLibMergeError):
    """data had inconsistent types"""


class LengthsDiffer(VarLibMergeError):
    """a list of objects had inconsistent lengths"""


class KeysDiffer(VarLibMergeError):
    """a list of objects had different keys"""


class InconsistentGlyphOrder(VarLibMergeError):
    """the glyph order was inconsistent between masters"""


class InconsistentExtensions(VarLibMergeError):
    """the masters use extension lookups in inconsistent ways"""


class UnsupportedFormat(VarLibMergeError):
    """an OpenType subtable (%s) had a format I didn't expect"""

    def __init__(self, merger=None, **kwargs):
        super().__init__(merger, **kwargs)
        if not self.stack:
            self.stack = [".Format"]

    @property
    def reason(self):
        s = self.__doc__ % self.cause["subtable"]
        if "value" in self.cause:
            s += f" ({self.cause['value']!r})"
        return s


class InconsistentFormats(UnsupportedFormat):
    """an OpenType subtable (%s) had inconsistent formats between masters"""


class VarLibCFFMergeError(VarLibError):
    pass


class VarLibCFFDictMergeError(VarLibCFFMergeError):
    """Raised when a CFF PrivateDict cannot be merged."""

    def __init__(self, key, value, values):
        error_msg = (
            f"For the Private Dict key '{key}', the default font value list:"
            f"\n\t{value}\nhad a different number of values than a region font:"
        )
        for region_value in values:
            error_msg += f"\n\t{region_value}"
        self.args = (error_msg,)


class VarLibCFFPointTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of point type differences."""

    def __init__(self, point_type, pt_index, m_index, default_type, glyph_name):
        error_msg = (
            f"Glyph '{glyph_name}': '{point_type}' at point index {pt_index} in "
            f"master index {m_index} differs from the default font point type "
            f"'{default_type}'"
        )
        self.args = (error_msg,)


class VarLibCFFHintTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of hint type differences."""

    def __init__(self, hint_type, cmd_index, m_index, default_type, glyph_name):
        error_msg = (
            f"Glyph '{glyph_name}': '{hint_type}' at index {cmd_index} in "
            f"master index {m_index} differs from the default font hint type "
            f"'{default_type}'"
        )
        self.args = (error_msg,)


class VariationModelError(VarLibError):
    """Raised when a variation model is faulty."""
<|endoftext|>from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.basePen import AbstractPen, BasePen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from math import sqrt, copysign, atan2, pi
import itertools

import logging

log = logging.getLogger("fontTools.varLib.interpolatable")


class InterpolatableProblem:
    NOTHING = "nothing"
    MISSING = "missing"
    OPEN_PATH = "open_path"
    PATH_COUNT = "path_count"
    NODE_COUNT = "node_count"
    NODE_INCOMPATIBILITY = "node_incompatibility"
    CONTOUR_ORDER = "contour_order"
    WRONG_START_POINT = "wrong_start_point"
    KINK = "kink"
    UNDERWEIGHT = "underweight"
    OVERWEIGHT = "overweight"

    severity = {
        MISSING: 1,
        OPEN_PATH: 2,
        PATH_COUNT: 3,
        NODE_COUNT: 4,
        NODE_INCOMPATIBILITY: 5,
        CONTOUR_ORDER: 6,
        WRONG_START_POINT: 7,
        KINK: 8,
        UNDERWEIGHT: 9,
        OVERWEIGHT: 10,
        NOTHING: 11,
    }


def sort_problems(problems):
    """Sort problem groups by their most severe problem type."""
    return dict(
        sorted(
            problems.items(),
            key=lambda _: -min(
                (
                    (InterpolatableProblem.severity[p["type"]] + p.get("tolerance", 0))
                    for p in _[1]
                ),
            ),
            reverse=True,
        )
    )


def rot_list(l, k):
    """Rotate list by k items forward.  Ie. item at position 0 will be
    at position k in returned list.  Negative k is allowed."""
    return l[-k:] + l[:-k]


class PerContourPen(BasePen):
    def __init__(self, Pen, glyphset=None):
        BasePen.__init__(self, glyphset)
        self._Pen = Pen
        self._pen = None
        self.value = []

    def _moveTo(self, p0):
        self._newItem()
        self._pen.moveTo(p0)

    def _lineTo(self, p1):
        self._pen.lineTo(p1)

    def _qCurveToOne(self, p1, p2):
        self._pen.qCurveTo(p1, p2)

    def _curveToOne(self, p1, p2, p3):
        self._pen.curveTo(p1, p2, p3)

    def _closePath(self):
        self._pen.closePath()
        self._pen = None

    def _endPath(self):
        self._pen.endPath()
        self._pen = None

    def _newItem(self):
        self._pen = pen = self._Pen()
        self.value.append(pen)


class PerContourOrComponentPen(PerContourPen):
    def addComponent(self, glyphName, transformation):
        self._newItem()
        self.value[-1].addComponent(glyphName, transformation)


class SimpleRecordingPointPen(AbstractPointPen):
    def __init__(self):
        self.value = []

    def beginPath(self, identifier=None, **kwargs):
        pass

    def endPath(self) -> None:
        pass

    def addPoint(self, pt, segmentType=None):
        self.value.append((pt, False if segmentType is None else True))


def vdiff_hypot2(v0, v1):
    s = 0
    for x0, x1 in zip(v0, v1):
        d = x1 - x0
        s += d * d
    return s


def vdiff_hypot2_complex(v0, v1):
    s = 0
    for x0, x1 in zip(v0, v1):
        d = x1 - x0
        s += d.real * d.real + d.imag * d.imag
        # This does the same but seems to be slower:
        # s += (d * d.conjugate()).real
    return s


def matching_cost(G, matching):
    return sum(G[i][j] for i, j in enumerate(matching))


def min_cost_perfect_bipartite_matching_scipy(G):
    n = len(G)
    rows, cols = linear_sum_assignment(G)
    assert (rows == list(range(n))).all()
    # Convert numpy array and integer to Python types,
    # to ensure that this is JSON-serializable.
    cols = list(int(e) for e in cols)
    return list(cols), matching_cost(G, cols)


def min_cost_perfect_bipartite_matching_munkres(G):
    n = len(G)
    cols = [None] * n
    for row, col in Munkres().compute(G):
        cols[row] = col
    return cols, matching_cost(G, cols)


def min_cost_perfect_bipartite_matching_bruteforce(G):
    n = len(G)

    if n > 6:
        raise Exception("Install Python module 'munkres' or 'scipy >= 0.17.0'")

    # Otherwise just brute-force
    permutations = itertools.permutations(range(n))
    best = list(next(permutations))
    best_cost = matching_cost(G, best)
    for p in permutations:
        cost = matching_cost(G, p)
        if cost < best_cost:
            best, best_cost = list(p), cost
    return best, best_cost


# Prefer `scipy.optimize.linear_sum_assignment` for performance.
# `Munkres` is also supported as a fallback for minimalistic systems
# where installing SciPy is not feasible.
try:
    from scipy.optimize import linear_sum_assignment

    min_cost_perfect_bipartite_matching = min_cost_perfect_bipartite_matching_scipy
except ImportError:
    try:
        from munkres import Munkres

        min_cost_perfect_bipartite_matching = (
            min_cost_perfect_bipartite_matching_munkres
        )
    except ImportError:
        min_cost_perfect_bipartite_matching = (
            min_cost_perfect_bipartite_matching_bruteforce
        )


def contour_vector_from_stats(stats):
    # Don't change the order of items here.
    # It's okay to add to the end, but otherwise, other
    # code depends on it. Search for "covariance".
    size = sqrt(abs(stats.area))
    return (
        copysign((size), stats.area),
        stats.meanX,
        stats.meanY,
        stats.stddevX * 2,
        stats.stddevY * 2,
        stats.correlation * size,
    )


def matching_for_vectors(m0, m1):
    n = len(m0)

    costs = [[vdiff_hypot2(v0, v1) for v1 in m1] for v0 in m0]
    (
        matching,
        matching_cost,
    ) = min_cost_perfect_bipartite_matching(costs)
    identity_cost = sum(costs[i][i] for i in range(n))
    return matching, matching_cost, identity_cost


def points_characteristic_bits(points):
    bits = 0
    for pt, b in reversed(points):
        bits = (bits << 1) | b
    return bits


_NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR = 4


def points_complex_vector(points):
    vector = []
    if not points:
        return vector
    points = [complex(*pt) for pt, _ in points]
    n = len(points)
    assert _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR == 4
    points.extend(points[: _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR - 1])
    while len(points) < _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR:
        points.extend(points[: _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR - 1])
    for i in range(n):
        # The weights are magic numbers.

        # The point itself
        p0 = points[i]
        vector.append(p0)

        # The vector to the next point
        p1 = points[i + 1]
        d0 = p1 - p0
        vector.append(d0 * 3)

        # The turn vector
        p2 = points[i + 2]
        d1 = p2 - p1
        vector.append(d1 - d0)

        # The angle to the next point, as a cross product;
        # Square root of, to match dimentionality of distance.
        cross = d0.real * d1.imag - d0.imag * d1.real
        cross = copysign(sqrt(abs(cross)), cross)
        vector.append(cross * 4)

    return vector


def add_isomorphisms(points, isomorphisms, reverse):
    reference_bits = points_characteristic_bits(points)
    n = len(points)

    # if points[0][0] == points[-1][0]:
    #   abort

    if reverse:
        points = points[::-1]
        bits = points_characteristic_bits(points)
    else:
        bits = reference_bits

    vector = points_complex_vector(points)

    assert len(vector) % n == 0
    mult = len(vector) // n
    mask = (1 << n) - 1

    for i in range(n):
        b = ((bits << (n - i)) & mask) | (bits >> i)
        if b == reference_bits:
            isomorphisms.append(
                (rot_list(vector, -i * mult), n - 1 - i if reverse else i, reverse)
            )


def find_parents_and_order(glyphsets, locations, *, discrete_axes=set()):
    parents = [None] + list(range(len(glyphsets) - 1))
    order = list(range(len(glyphsets)))
    if locations:
        # Order base master first
        bases = [
            i
            for i, l in enumerate(locations)
            if all(v == 0 for k, v in l.items() if k not in discrete_axes)
        ]
        if bases:
            log.info("Found %s base masters: %s", len(bases), bases)
        else:
            log.warning("No base master location found")

        # Form a minimum spanning tree of the locations
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree

            graph = [[0] * len(locations) for _ in range(len(locations))]
            axes = set()
            for l in locations:
                axes.update(l.keys())
            axes = sorted(axes)
            vectors = [tuple(l.get(k, 0) for k in axes) for l in locations]
            for i, j in itertools.combinations(range(len(locations)), 2):
                i_discrete_location = {
                    k: v for k, v in zip(axes, vectors[i]) if k in discrete_axes
                }
                j_discrete_location = {
                    k: v for k, v in zip(axes, vectors[j]) if k in discrete_axes
                }
                if i_discrete_location != j_discrete_location:
                    continue
                graph[i][j] = vdiff_hypot2(vectors[i], vectors[j])

            tree = minimum_spanning_tree(graph, overwrite=True)
            rows, cols = tree.nonzero()
            graph = defaultdict(set)
            for row, col in zip(rows, cols):
                graph[row].add(col)
                graph[col].add(row)

            # Traverse graph from the base and assign parents
            parents = [None] * len(locations)
            order = []
            visited = set()
            queue = deque(bases)
            while queue:
                i = queue.popleft()
                visited.add(i)
                order.append(i)
                for j in sorted(graph[i]):
                    if j not in visited:
                        parents[j] = i
                        queue.append(j)
            assert len(order) == len(
                parents
            ), "Not all masters are reachable; report an issue"

        except ImportError:
            pass

        log.info("Parents: %s", parents)
        log.info("Order: %s", order)
    return parents, order
<|endoftext|>"""Visualize DesignSpaceDocument and resulting VariationModel."""

from fontTools.varLib.models import VariationModel, supportScalar
from fontTools.designspaceLib import DesignSpaceDocument
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d
from itertools import cycle
import math
import logging
import sys

log = logging.getLogger(__name__)


def stops(support, count=10):
    a, b, c = support

    return (
        [a + (b - a) * i / count for i in range(count)]
        + [b + (c - b) * i / count for i in range(count)]
        + [c]
    )


def _plotLocationsDots(locations, axes, subplot, **kwargs):
    for loc, color in zip(locations, cycle(pyplot.cm.Set1.colors)):
        if len(axes) == 1:
            subplot.plot([loc.get(axes[0], 0)], [1.0], "o", color=color, **kwargs)
        elif len(axes) == 2:
            subplot.plot(
                [loc.get(axes[0], 0)],
                [loc.get(axes[1], 0)],
                [1.0],
                "o",
                color=color,
                **kwargs,
            )
        else:
            raise AssertionError(len(axes))


def plotLocations(locations, fig, names=None, **kwargs):
    n = len(locations)
    cols = math.ceil(n**0.5)
    rows = math.ceil(n / cols)

    if names is None:
        names = [None] * len(locations)

    model = VariationModel(locations)
    names = [names[model.reverseMapping[i]] for i in range(len(names))]

    axes = sorted(locations[0].keys())
    if len(axes) == 1:
        _plotLocations2D(model, axes[0], fig, cols, rows, names=names, **kwargs)
    elif len(axes) == 2:
        _plotLocations3D(model, axes, fig, cols, rows, names=names, **kwargs)
    else:
        raise ValueError("Only 1 or 2 axes are supported")


def _plotLocations2D(model, axis, fig, cols, rows, names, **kwargs):
    subplot = fig.add_subplot(111)
    for i, (support, color, name) in enumerate(
        zip(model.supports, cycle(pyplot.cm.Set1.colors), cycle(names))
    ):
        if name is not None:
            subplot.set_title(name)
        subplot.set_xlabel(axis)
        pyplot.xlim(-1.0, +1.0)

        Xs = support.get(axis, (-1.0, 0.0, +1.0))
        X, Y = [], []
        for x in stops(Xs):
            y = supportScalar({axis: x}, support)
            X.append(x)
            Y.append(y)
        subplot.plot(X, Y, color=color, **kwargs)

        _plotLocationsDots(model.locations, [axis], subplot)


def _plotLocations3D(model, axes, fig, rows, cols, names, **kwargs):
    ax1, ax2 = axes

    axis3D = fig.add_subplot(111, projection="3d")
    for i, (support, color, name) in enumerate(
        zip(model.supports, cycle(pyplot.cm.Set1.colors), cycle(names))
    ):
        if name is not None:
            axis3D.set_title(name)
        axis3D.set_xlabel(ax1)
        axis3D.set_ylabel(ax2)
        pyplot.xlim(-1.0, +1.0)
        pyplot.ylim(-1.0, +1.0)

        Xs = support.get(ax1, (-1.0, 0.0, +1.0))
        Ys = support.get(ax2, (-1.0, 0.0, +1.0))
        for x in stops(Xs):
            X, Y, Z = [], [], []
            for y in Ys:
                z = supportScalar({ax1: x, ax2: y}, support)
                X.append(x)
                Y.append(y)
                Z.append(z)
            axis3D.plot(X, Y, Z, color=color, **kwargs)
        for y in stops(Ys):
            X, Y, Z = [], [], []
            for x in Xs:
                z = supportScalar({ax1: x, ax2: y}, support)
                X.append(x)
                Y.append(y)
                Z.append(z)
            axis3D.plot(X, Y, Z, color=color, **kwargs)

        _plotLocationsDots(model.locations, [ax1, ax2], axis3D)


def plotDocument(doc, fig, **kwargs):
    doc.normalize()
    locations = [s.location for s in doc.sources]
    names = [s.name for s in doc.sources]
    plotLocations(locations, fig, names, **kwargs)


def _plotModelFromMasters2D(model, masterValues, fig, **kwargs):
    assert len(model.axisOrder) == 1
    axis = model.axisOrder[0]

    axis_min = min(loc.get(axis, 0) for loc in model.locations)
    axis_max = max(loc.get(axis, 0) for loc in model.locations)

    import numpy as np

    X = np.arange(axis_min, axis_max, (axis_max - axis_min) / 100)
    Y = []

    for x in X:
        loc = {axis: x}
        v = model.interpolateFromMasters(loc, masterValues)
        Y.append(v)

    subplot = fig.add_subplot(111)
    subplot.plot(X, Y, "-", **kwargs)


def _plotModelFromMasters3D(model, masterValues, fig, **kwargs):
    assert len(model.axisOrder) == 2
    axis1, axis2 = model.axisOrder[0], model.axisOrder[1]

    axis1_min = min(loc.get(axis1, 0) for loc in model.locations)
    axis1_max = max(loc.get(axis1, 0) for loc in model.locations)
    axis2_min = min(loc.get(axis2, 0) for loc in model.locations)
    axis2_max = max(loc.get(axis2, 0) for loc in model.locations)

    import numpy as np

    X = np.arange(axis1_min, axis1_max, (axis1_max - axis1_min) / 100)
    Y = np.arange(axis2_min, axis2_max, (axis2_max - axis2_min) / 100)
    X, Y = np.meshgrid(X, Y)
    Z = []

    for row_x, row_y in zip(X, Y):
        z_row = []
        Z.append(z_row)
        for x, y in zip(row_x, row_y):
            loc = {axis1: x, axis2: y}
            v = model.interpolateFromMasters(loc, masterValues)
            z_row.append(v)
    Z = np.array(Z)

    axis3D = fig.add_subplot(111, projection="3d")
    axis3D.plot_surface(X, Y, Z, **kwargs)


def plotModelFromMasters(model, masterValues, fig, **kwargs):
    """Plot a variation model and set of master values corresponding
    to the locations to the model into a pyplot figure.  Variation
    model must have axisOrder of size 1 or 2."""
    if len(model.axisOrder) == 1:
        _plotModelFromMasters2D(model, masterValues, fig, **kwargs)
    elif len(model.axisOrder) == 2:
        _plotModelFromMasters3D(model, masterValues, fig, **kwargs)
    else:
        raise ValueError("Only 1 or 2 axes are supported")


def main(args=None):
    from fontTools import configLogger

    if args is None:
        args = sys.argv[1:]

    # configure the library logger (for >= WARNING)
    configLogger()
    # comment this out to enable debug messages from logger
    # log.setLevel(logging.DEBUG)

    if len(args) < 1:
        print("usage: fonttools varLib.plot source.designspace", file=sys.stderr)
        print("  or")
        print("usage: fonttools varLib.plot location1 location2 ...", file=sys.stderr)
        print("  or")
        print(
            "usage: fonttools varLib.plot location1=value1 location2=value2 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    fig = pyplot.figure()
    fig.set_tight_layout(True)

    if len(args) == 1 and args[0].endswith(".designspace"):
        doc = DesignSpaceDocument()
        doc.read(args[0])
        plotDocument(doc, fig)
    else:
        axes = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        if "=" not in args[0]:
            locs = [dict(zip(axes, (float(v) for v in s.split(",")))) for s in args]
            plotLocations(locs, fig)
        else:
            locations = []
            masterValues = []
            for arg in args:
                loc, v = arg.split("=")
                locations.append(dict(zip(axes, (float(v) for v in loc.split(",")))))
                masterValues.append(float(v))
            model = VariationModel(locations, axes[: len(locations[0])])
            plotModelFromMasters(model, masterValues, fig)

    pyplot.show()


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>"""Variation fonts interpolation models."""

from __future__ import annotations

__all__ = [
    "normalizeValue",
    "normalizeLocation",
    "supportScalar",
    "piecewiseLinearMap",
    "VariationModel",
]

from collections.abc import Mapping
from typing import TYPE_CHECKING
from fontTools.misc.roundTools import noRound
from .errors import VariationModelError

if TYPE_CHECKING:
    from typing import Mapping, Sequence


def nonNone(lst):
    return [l for l in lst if l is not None]


def allNone(lst):
    return all(l is None for l in lst)


def allEqualTo(ref, lst, mapper=None):
    if mapper is None:
        return all(ref == item for item in lst)

    mapped = mapper(ref)
    return all(mapped == mapper(item) for item in lst)


def allEqual(lst, mapper=None):
    if not lst:
        return True
    it = iter(lst)
    try:
        first = next(it)
    except StopIteration:
        return True
    return allEqualTo(first, it, mapper=mapper)


def subList(truth, lst):
    assert len(truth) == len(lst)
    return [l for l, t in zip(lst, truth) if t]


def normalizeValue(
    v: float, triple: Sequence[float], extrapolate: bool = False
) -> float:
    """Normalizes value based on a min/default/max triple.

    >>> normalizeValue(400, (100, 400, 900))
    0.0
    >>> normalizeValue(100, (100, 400, 900))
    -1.0
    >>> normalizeValue(650, (100, 400, 900))
    0.5
    """
    lower, default, upper = triple
    if not (lower <= default <= upper):
        raise ValueError(
            f"Invalid axis values, must be minimum, default, maximum: "
            f"{lower:3.3f}, {default:3.3f}, {upper:3.3f}"
        )
    if not extrapolate:
        v = max(min(v, upper), lower)

    if v == default or lower == upper:
        return 0.0

    if (v < default and lower != default) or (v > default and upper == default):
        return (v - default) / (default - lower)
    else:
        assert (v > default and upper != default) or (
            v < default and lower == default
        ), f"Ooops... v={v}, triple=({lower}, {default}, {upper})"
        return (v - default) / (upper - default)


def normalizeLocation(
    location: Mapping[str, float],
    axes: Mapping[str, tuple[float, float, float]],
    extrapolate: bool = False,
    *,
    validate: bool = False,
) -> dict[str, float]:
    """Normalizes location based on axis min/default/max values from axes.

    >>> axes = {"wght": (100, 400, 900)}
    >>> normalizeLocation({"wght": 400}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 100}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": 900}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 650}, axes)
    {'wght': 0.5}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': -1.0}
    >>> axes = {"wght": (0, 0, 1000)}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": -1}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 500}, axes)
    {'wght': 0.5}
    >>> normalizeLocation({"wght": 1001}, axes)
    {'wght': 1.0}
    >>> axes = {"wght": (0, 1000, 1000)}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": -1}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": 500}, axes)
    {'wght': -0.5}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 1001}, axes)
    {'wght': 0.0}
    """
    if validate:
        assert set(location.keys()) <= set(axes.keys()), set(location.keys()) - set(
            axes.keys()
        )
    out = {}
    for tag, triple in axes.items():
        v = location.get(tag, triple[1])
        out[tag] = normalizeValue(v, triple, extrapolate=extrapolate)
    return out


def supportScalar(location, support, ot=True, extrapolate=False, axisRanges=None):
    """Returns the scalar multiplier at location, for a master
    with support.  If ot is True, then a peak value of zero
    for support of an axis means "axis does not participate".  That
    is how OpenType Variation Font technology works.

    If extrapolate is True, axisRanges must be a dict that maps axis
    names to (axisMin, axisMax) tuples.

      >>> supportScalar({}, {})
      1.0
      >>> supportScalar({'wght':.2}, {})
      1.0
      >>> supportScalar({'wght':.2}, {'wght':(0,2,3)})
      0.1
      >>> supportScalar({'wght':2.5}, {'wght':(0,2,4)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)}, ot=False)
      0.375
      >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
      0.75
      >>> supportScalar({'wght':3}, {'wght':(0,1,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -1.0
      >>> supportScalar({'wght':-1}, {'wght':(0,1,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -1.0
      >>> supportScalar({'wght':3}, {'wght':(0,2,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      1.5
      >>> supportScalar({'wght':-1}, {'wght':(0,2,2)}, extrapolate=True, axisRanges={'wght':(0, 2)})
      -0.5
    """
    if extrapolate and axisRanges is None:
        raise TypeError("axisRanges must be passed when extrapolate is True")
    scalar = 1.0
    for axis, (lower, peak, upper) in support.items():
        if ot:
            # OpenType-specific case handling
            if peak == 0.0:
                continue
            if lower > peak or peak > upper:
                continue
            if lower < 0.0 and upper > 0.0:
                continue
            v = location.get(axis, 0.0)
        else:
            assert axis in location
            v = location[axis]
        if v == peak:
            continue

        if extrapolate:
            axisMin, axisMax = axisRanges[axis]
            if v < axisMin and lower <= axisMin:
                if peak <= axisMin and peak < upper:
                    scalar *= (v - upper) / (peak - upper)
                    continue
                elif axisMin < peak:
                    scalar *= (v - lower) / (peak - lower)
                    continue
            elif axisMax < v and axisMax <= upper:
                if axisMax <= peak and lower < peak:
                    scalar *= (v - lower) / (peak - lower)
                    continue
                elif peak < axisMax:
                    scalar *= (v - upper) / (peak - upper)
                    continue

        if v <= lower or upper <= v:
            scalar = 0.0
            break

        if v < peak:
            scalar *= (v - lower) / (peak - lower)
        else:  # v > peak
            scalar *= (v - upper) / (peak - upper)
    return scalar


class VariationModel(object):
    """Locations must have the base master at the origin (ie. 0).

    If axis-ranges are not provided, values are assumed to be normalized to
    the range [-1, 1].

    If the extrapolate argument is set to True, then values are extrapolated
    outside the axis range.

      >>> from pprint import pprint
      >>> axisRanges = {'wght': (-180, +180), 'wdth': (-1, +1)}
      >>> locations = [ \
      {'wght':100}, \
      {'wght':-100}, \
      {'wght':-180}, \
      {'wdth':+.3}, \
      {'wght':+120,'wdth':.3}, \
      {'wght':+120,'wdth':.2}, \
      {}, \
      {'wght':+180,'wdth':.3}, \
      {'wght':+180}, \
      ]
      >>> model = VariationModel(locations, axisOrder=['wght'], axisRanges=axisRanges)
      >>> pprint(model.locations)
      [{},
       {'wght': -100},
       {'wght': -180},
       {'wght': 100},
       {'wght': 180},
       {'wdth': 0.3},
       {'wdth': 0.3, 'wght': 180},
       {'wdth': 0.3, 'wght': 120},
       {'wdth': 0.2, 'wght': 120}]
      >>> pprint(model.deltaWeights)
      [{},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0, 4: 1.0, 5: 1.0},
       {0: 1.0, 3: 0.75, 4: 0.25, 5: 1.0, 6: 0.6666666666666666},
       {0: 1.0,
        3: 0.75,
        4: 0.25,
        5: 0.6666666666666667,
        6: 0.4444444444444445,
        7: 0.6666666666666667}]
    """

    def __init__(
        self, locations, axisOrder=None, extrapolate=False, *, axisRanges=None
    ):
        locations = [{k: v for k, v in loc.items() if v != 0.0} for loc in locations]

        if len(set(tuple(sorted(l.items())) for l in locations)) != len(locations):
            raise VariationModelError("Locations must be unique.")

        self.origLocations = locations
        self.axisOrder = axisOrder if axisOrder is not None else []
        self.extrapolate = extrapolate
        if axisRanges is None:
            if extrapolate:
                axisRanges = self.computeAxisRanges(locations)
            else:
                allAxes = {axis for loc in locations for axis in loc.keys()}
                axisRanges = {axis: (-1, 1) for axis in allAxes}
        self.axisRanges = axisRanges
        keyFunc = self.getMasterLocationsSortKeyFunc(
            locations, axisOrder=self.axisOrder
        )
        self.locations = sorted(locations, key=keyFunc)

        # Mapping from user's master order to our master order
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]

        self._computeMasterSupports()
        self._subModels = {}

    def getSubModel(self, items):
        """Return a sub-model and the items that are not None.

        The sub-model is necessary for working with the subset
        of items when some are None.

        The sub-model is cached."""
        if None not in items:
            return self, items
        key = tuple(v is not None for v in items)
        subModel = self._subModels.get(key)
        if subModel is None:
            subModel = VariationModel(
                subList(key, self.origLocations),
                self.axisOrder,
                extrapolate=self.extrapolate,
                axisRanges=self.axisRanges,
            )
            self._subModels[key] = subModel
        return subModel, subList(key, items)

    @staticmethod
    def computeAxisRanges(locations):
        axisRanges = {}
        allAxes = {axis for loc in locations for axis in loc.keys()}
        for loc in locations:
            for axis in allAxes:
                value = loc.get(axis, 0)
                axisMin, axisMax = axisRanges.get(axis, (value, value))
                axisRanges[axis] = min(value, axisMin), max(value, axisMax)
        return axisRanges

    @staticmethod
    def getMasterLocationsSortKeyFunc(locations, axisOrder=[]):
        if {} not in locations:
            raise VariationModelError("Base master not found.")
        axisPoints = {}
        for loc in locations:
            if len(loc) != 1:
                continue
            axis = next(iter(loc))
            value = loc[axis]
            if axis not in axisPoints:
                axisPoints[axis] = {0.0}
            assert (
                value not in axisPoints[axis]
            ), 'Value "%s" in axisPoints["%s"] -->  %s' % (value, axis, axisPoints)
            axisPoints[axis].add(value)

        def getKey(axisPoints, axisOrder):
            def sign(v):
                return -1 if v < 0 else +1 if v > 0 else 0

            def key(loc):
                rank = len(loc)
                onPointAxes = [
                    axis
                    for axis, value in loc.items()
                    if axis in axisPoints and value in axisPoints[axis]
                ]
                orderedAxes = [axis for axis in axisOrder if axis in loc]
                orderedAxes.extend(
                    [axis for axis in sorted(loc.keys()) if axis not in axisOrder]
                )
                return (
                    rank,  # First, order by increasing rank
                    -len(onPointAxes),  # Next, by decreasing number of onPoint axes
                    tuple(
                        axisOrder.index(axis) if axis in axisOrder else 0x10000
                        for axis in orderedAxes
                    ),  # Next, by known axes
                    tuple(orderedAxes),  # Next, by all axes
                    tuple(
                        sign(loc[axis]) for axis in orderedAxes
                    ),  # Next, by signs of axis values
                    tuple(
                        abs(loc[axis]) for axis in orderedAxes
                    ),  # Next, by absolute value of axis values
                )

            return key

        ret = getKey(axisPoints, axisOrder)
        return ret

    def reorderMasters(self, master_list, mapping):
        # For changing the master data order without
        # recomputing supports and deltaWeights.
        new_list = [master_list[idx] for idx in mapping]
        self.origLocations = [self.origLocations[idx] for idx in mapping]
        locations = [
            {k: v for k, v in loc.items() if v != 0.0} for loc in self.origLocations
        ]
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]
        self._subModels = {}
        return new_list

    def _computeMasterSupports(self):
        self.supports = []
        regions = self._locationsToRegions()
        for i, region in enumerate(regions):
            locAxes = set(region.keys())
            # Walk over previous masters now
            for prev_region in regions[:i]:
                # Master with different axes do not participate
                if set(prev_region.keys()) != locAxes:
                    continue
                # If it's NOT in the current box, it does not participate
                relevant = True
                for axis, (lower, peak, upper) in region.items():
                    if not (
                        prev_region[axis][1] == peak
                        or lower < prev_region[axis][1] < upper
                    ):
                        relevant = False
                        break
                if not relevant:
                    continue

                # Split the box for new master; split in whatever direction
                # that has largest range ratio.
                #
                # For symmetry, we actually cut across multiple axes
                # if they have the largest, equal, ratio.
                # https://github.com/fonttools/fonttools/commit/7ee81c8821671157968b097f3e55309a1faa511e#commitcomment-31054804

                bestAxes = {}
                bestRatio = -1
                for axis in prev_region.keys():
                    val = prev_region[axis][1]
                    assert axis in region
                    lower, locV, upper = region[axis]
                    newLower, newUpper = lower, upper
                    if val < locV:
                        newLower = val
                        ratio = (val - locV) / (lower - locV)
                    elif locV < val:
                        newUpper = val
                        ratio = (val - locV) / (upper - locV)
                    else:  # val == locV
                        # Can't split box in this direction.
                        continue
                    if ratio > bestRatio:
                        bestAxes = {}
                        bestRatio = ratio
                    if ratio == bestRatio:
                        bestAxes[axis] = (newLower, locV, newUpper)

                for axis, triple in bestAxes.items():
                    region[axis] = triple
            self.supports.append(region)
        self._computeDeltaWeights()

    def _locationsToRegions(self):
        locations = self.locations
        axisRanges = self.axisRanges

        regions = []
        for loc in locations:
            region = {}
            for axis, locV in loc.items():
                if locV > 0:
                    region[axis] = (0, locV, axisRanges[axis][1])
                else:
                    region[axis] = (axisRanges[axis][0], locV, 0)
            regions.append(region)
        return regions

    def _computeDeltaWeights(self):
        self.deltaWeights = []
        for i, loc in enumerate(self.locations):
            deltaWeight = {}
            # Walk over previous masters now, populate deltaWeight
            for j, support in enumerate(self.supports[:i]):
                scalar = supportScalar(loc, support)
                if scalar:
                    deltaWeight[j] = scalar
            self.deltaWeights.append(deltaWeight)

    def getDeltas(self, masterValues, *, round=noRound):
        assert len(masterValues) == len(self.deltaWeights), (
            len(masterValues),
            len(self.deltaWeights),
        )
        mapping = self.reverseMapping
        out = []
        for i, weights in enumerate(self.deltaWeights):
            delta = masterValues[mapping[i]]
            for j, weight in weights.items():
                if weight == 1:
                    delta -= out[j]
                else:
                    delta -= out[j] * weight
            out.append(round(delta))
        return out

    def getDeltasAndSupports(self, items, *, round=noRound):
        model, items = self.getSubModel(items)
        return model.getDeltas(items, round=round), model.supports

    def getScalars(self, loc):
        """Return scalars for each delta, for the given location.
        If interpolating many master-values at the same location,
        this function allows speed up by fetching the scalars once
        and using them with interpolateFromMastersAndScalars()."""
        return [
            supportScalar(
                loc, support, extrapolate=self.extrapolate, axisRanges=self.axisRanges
            )
            for support in self.supports
        ]

    def getMasterScalars(self, targetLocation):
        """Return multipliers for each master, for the given location.
        If interpolating many master-values at the same location,
        this function allows speed up by fetching the scalars once
        and using them with interpolateFromValuesAndScalars().

        Note that the scalars used in interpolateFromMastersAndScalars(),
        are *not* the same as the ones returned here. They are the result
        of getScalars()."""
        out = self.getScalars(targetLocation)
        for i, weights in reversed(list(enumerate(self.deltaWeights))):
            for j, weight in weights.items():
                out[j] -= out[i] * weight

        out = [out[self.mapping[i]] for i in range(len(out))]
        return out

    @staticmethod
    def interpolateFromValuesAndScalars(values, scalars):
        """Interpolate from values and scalars coefficients.

        If the values are master-values, then the scalars should be
        fetched from getMasterScalars().

        If the values are deltas, then the scalars should be fetched
        from getScalars(); in which case this is the same as
        interpolateFromDeltasAndScalars().
        """
        v = None
        assert len(values) == len(scalars)
        for value, scalar in zip(values, scalars):
            if not scalar:
                continue
            contribution = value * scalar
            if v is None:
                v = contribution
            else:
                v += contribution
        return v

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        """Interpolate from deltas and scalars fetched from getScalars()."""
        return VariationModel.interpolateFromValuesAndScalars(deltas, scalars)

    def interpolateFromDeltas(self, loc, deltas):
        """Interpolate from deltas, at location loc."""
        scalars = self.getScalars(loc)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromMasters(self, loc, masterValues, *, round=noRound):
        """Interpolate from master-values, at location loc."""
        scalars = self.getMasterScalars(loc)
        return self.interpolateFromValuesAndScalars(masterValues, scalars)

    def interpolateFromMastersAndScalars(self, masterValues, scalars, *, round=noRound):
        """Interpolate from master-values, and scalars fetched from
        getScalars(), which is useful when you want to interpolate
        multiple master-values with the same location."""
        deltas = self.getDeltas(masterValues, round=round)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)


def piecewiseLinearMap(v: float, mapping: Mapping[float, float]) -> float:
    keys = mapping.keys()
    if not keys:
        return v
    if v in keys:
        return mapping[v]
    k = min(keys)
    if v < k:
        return v + mapping[k] - k
    k = max(keys)
    if v > k:
        return v + mapping[k] - k
    # Interpolate
    a = max(k for k in keys if k < v)
    b = min(k for k in keys if k > v)
    va = mapping[a]
    vb = mapping[b]
    return va + (vb - va) * (v - a) / (b - a)


def main(args=None):
    """Normalize locations on a given designspace"""
    from fontTools import configLogger
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.models",
        description=main.__doc__,
    )
    parser.add_argument(
        "--loglevel",
        metavar="LEVEL",
        default="INFO",
        help="Logging level (defaults to INFO)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--designspace", metavar="DESIGNSPACE", type=str)
    group.add_argument(
        "-l",
        "--locations",
        metavar="LOCATION",
        nargs="+",
        help="Master locations as comma-separate coordinates. One must be all zeros.",
    )

    args = parser.parse_args(args)

    configLogger(level=args.loglevel)
    from pprint import pprint

    if args.designspace:
        from fontTools.designspaceLib import DesignSpaceDocument

        doc = DesignSpaceDocument()
        doc.read(args.designspace)
        locs = [s.location for s in doc.sources]
        print("Original locations:")
        pprint(locs)
        doc.normalize()
        print("Normalized locations:")
        locs = [s.location for s in doc.sources]
        pprint(locs)
    else:
        axes = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        locs = [
            dict(zip(axes, (float(v) for v in s.split(",")))) for s in args.locations
        ]

    model = VariationModel(locs)
    print("Sorted locations:")
    pprint(model.locations)
    print("Supports:")
    pprint(model.supports)


if __name__ == "__main__":
    import doctest, sys

    if len(sys.argv) > 1:
        sys.exit(main())

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""
Module for dealing with 'gvar'-style font variations, also known as run-time
interpolation.

The ideas here are very similar to MutatorMath.  There is even code to read
MutatorMath .designspace files in the varLib.designspace module.

For now, if you run this file on a designspace file, it tries to find
ttf-interpolatable files for the masters and build a variable-font from
them.  Such ttf-interpolatable and designspace files can be generated from
a Glyphs source, eg., using noto-source as an example:

    .. code-block:: sh

        $ fontmake -o ttf-interpolatable -g NotoSansArabic-MM.glyphs

Then you can make a variable-font this way:

    .. code-block:: sh

        $ fonttools varLib master_ufo/NotoSansArabic.designspace

API *will* change in near future.
"""

from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import (
    GlyphCoordinates,
    dropImpliedOnCurvePoints,
    USE_MY_METRICS,
)
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError

log = logging.getLogger("fontTools.varLib")

# This is a lib key for the designspace document. The value should be
# a comma-separated list of OpenType feature tag(s), to be used as the
# FeatureVariations feature.
# If present, the DesignSpace <rules processing="..."> flag is ignored.
FEAVAR_FEATURETAG_LIB_KEY = "com.github.fonttools.varLib.featureVarsFeatureTag"

#
# Creation routines
#


def _add_fvar(font, axes, instances: List[InstanceDescriptor]):
    """
    Add 'fvar' table to font.

    axes is an ordered dictionary of DesignspaceAxis objects.

    instances is list of dictionary objects with 'location', 'stylename',
    and possibly 'postscriptfontname' entries.
    """

    assert axes
    assert isinstance(axes, OrderedDict)

    log.info("Generating fvar")

    fvar = newTable("fvar")
    nameTable = font["name"]

    # if there are not currently any mac names don't add them here, that's inconsistent
    # https://github.com/fonttools/fonttools/issues/683
    macNames = any(nr.platformID == 1 for nr in getattr(nameTable, "names", ()))

    # we have all the best ways to express mac names
    platforms = ((3, 1, 0x409),)
    if macNames:
        platforms = ((1, 0, 0),) + platforms

    for a in axes.values():
        axis = Axis()
        axis.axisTag = Tag(a.tag)
        # TODO Skip axes that have no variation.
        axis.minValue, axis.defaultValue, axis.maxValue = (
            a.minimum,
            a.default,
            a.maximum,
        )
        axis.axisNameID = nameTable.addMultilingualName(
            a.labelNames, font, minNameID=256, mac=macNames
        )
        axis.flags = int(a.hidden)
        fvar.axes.append(axis)

    default_coordinates = {axis.axisTag: axis.defaultValue for axis in fvar.axes}

    for instance in instances:
        # Filter out discrete axis locations
        coordinates = {
            name: value for name, value in instance.location.items() if name in axes
        }

        if "en" not in instance.localisedStyleName:
            if not instance.styleName:
                raise VarLibValidationError(
                    f"Instance at location '{coordinates}' must have a default English "
                    "style name ('stylename' attribute on the instance element or a "
                    "stylename element with an 'xml:lang=\"en\"' attribute)."
                )
            localisedStyleName = dict(instance.localisedStyleName)
            localisedStyleName["en"] = tostr(instance.styleName)
        else:
            localisedStyleName = instance.localisedStyleName

        psname = instance.postScriptFontName

        inst = NamedInstance()
        inst.coordinates = {
            axes[k].tag: axes[k].map_backward(v) for k, v in coordinates.items()
        }

        subfamilyNameID = nameTable.findMultilingualName(
            localisedStyleName, windows=True, mac=macNames
        )
        if subfamilyNameID in {2, 17} and inst.coordinates == default_coordinates:
            # Instances can only reuse an existing name ID 2 or 17 if they are at the
            # default location across all axes, see:
            # https://github.com/fonttools/fonttools/issues/3825.
            inst.subfamilyNameID = subfamilyNameID
        else:
            inst.subfamilyNameID = nameTable.addMultilingualName(
                localisedStyleName, windows=True, mac=macNames, minNameID=256
            )

        if psname is not None:
            psname = tostr(psname)
            inst.postscriptNameID = nameTable.addName(psname, platforms=platforms)
        fvar.instances.append(inst)

    assert "fvar" not in font
    font["fvar"] = fvar

    return fvar


def _add_avar(font, axes, mappings, axisTags):
    """
    Add 'avar' table to font.

    axes is an ordered dictionary of AxisDescriptor objects.
    """

    assert axes
    assert isinstance(axes, OrderedDict)

    log.info("Generating avar")

    avar = newTable("avar")

    interesting = False
    vals_triples = {}
    for axis in axes.values():
        axis_map = axis.get_validated_map()

        # Currently, some rasterizers require that the default value maps
        # (-1 to -1, 0 to 0, and 1 to 1) be present for all the segment
        # maps, even when the default normalization mapping for the axis
        # was not modified.
        # https://github.com/googlei18n/fontmake/issues/295
        # https://github.com/fonttools/fonttools/issues/1011
        # TODO(anthrotype) revert this (and 19c4b37) when issue is fixed
        curve = avar.segments[axis.tag] = {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}

        keys_triple = (axis.minimum, axis.default, axis.maximum)
        vals_triple = tuple(axis.map_forward(v) for v in keys_triple)
        vals_triples[axis.tag] = vals_triple
        if not axis_map:
            continue

        # The validated view removes repeated pairs while preserving legal
        # duplicate output values.
        items = sorted(axis_map)
        keys = [item[0] for item in items]
        vals = [item[1] for item in items]

        # Current avar requirements.  We don't have to enforce
        # these on the designer and can deduce some ourselves,
        # but for now just enforce them.
        if axis.minimum != min(keys):
            raise VarLibValidationError(
                f"Axis '{axis.name}': there must be a mapping for the axis minimum "
                f"value {axis.minimum} and it must be the lowest input mapping value."
            )
        if axis.maximum != max(keys):
            raise VarLibValidationError(
                f"Axis '{axis.name}': there must be a mapping for the axis maximum "
                f"value {axis.maximum} and it must be the highest input mapping value."
            )
        if axis.default not in keys:
            raise VarLibValidationError(
                f"Axis '{axis.name}': there must be a mapping for the axis default "
                f"value {axis.default}."
            )
        # Ascending values
        if sorted(vals) != vals:
            raise VarLibValidationError(
                f"Axis '{axis.name}': mapping output values must be in ascending order."
            )

        keys = [models.normalizeValue(v, keys_triple) for v in keys]
        vals = [models.normalizeValue(v, vals_triple) for v in vals]

        if all(k == v for k, v in zip(keys, vals)):
            continue
        interesting = True

        curve.update(zip(keys, vals))

        assert 0.0 in curve and curve[0.0] == 0.0
        assert -1.0 not in curve or curve[-1.0] == -1.0
        assert +1.0 not in curve or curve[+1.0] == +1.0
        # curve.update({-1.0: -1.0, 0.0: 0.0, 1.0: 1.0})

    if mappings:
        interesting = True

        inputLocations = [
            {
                axes[name].tag: models.normalizeValue(v, vals_triples[axes[name].tag])
                for name, v in mapping.inputLocation.items()
            }
            for mapping in mappings
        ]
        outputLocations = [
            {
                axes[name].tag: models.normalizeValue(v, vals_triples[axes[name].tag])
                for name, v in mapping.outputLocation.items()
            }
            for mapping in mappings
        ]
        assert len(inputLocations) == len(outputLocations)

        # If base-master is missing, insert it at zero location.
        if not any(all(v == 0 for k, v in loc.items()) for loc in inputLocations):
            inputLocations.insert(0, {})
            outputLocations.insert(0, {})

        model = models.VariationModel(inputLocations, axisTags)
        storeBuilder = varStore.OnlineVarStoreBuilder(axisTags)
        storeBuilder.setModel(model)
        varIdxes = {}
        for tag in axisTags:
            masterValues = []
            for vo, vi in zip(outputLocations, inputLocations):
                if tag not in vo:
                    masterValues.append(0)
                    continue
                v = vo[tag] - vi.get(tag, 0)
                masterValues.append(fl2fi(v, 14))
            varIdxes[tag] = storeBuilder.storeMasters(masterValues)[1]

        store = storeBuilder.finish()
        optimized = store.optimize()
        varIdxes = {axis: optimized[value] for axis, value in varIdxes.items()}

        varIdxMap = builder.buildDeltaSetIndexMap(varIdxes[t] for t in axisTags)

        avar.majorVersion = 2
        avar.table = ot.avar()
        avar.table.VarIdxMap = varIdxMap
        avar.table.VarStore = store

    assert "avar" not in font
    if not interesting:
        log.info("No need for avar")
        avar = None
    else:
        font["avar"] = avar

    return avar


def _add_stat(font):
    # Note: this function only gets called by old code that calls `build()`
    # directly. Newer code that wants to benefit from STAT data from the
    # designspace should call `build_many()`

    if "STAT" in font:
        return

    from ..otlLib.builder import buildStatTable

    fvarTable = font["fvar"]
    axes = [dict(tag=a.axisTag, name=a.axisNameID) for a in fvarTable.axes]
    buildStatTable(font, axes)


_MasterData = namedtuple("_MasterData", ["glyf", "hMetrics", "vMetrics"])


def _add_gvar(font, masterModel, master_ttfs, tolerance=0.5, optimize=True):
    if tolerance < 0:
        raise ValueError("`tolerance` must be a positive number.")

    log.info("Generating gvar")
    assert "gvar" not in font
    gvar = font["gvar"] = newTable("gvar")
    glyf = font["glyf"]
    defaultMasterIndex = masterModel.reverseMapping[0]

    master_datas = [
        _MasterData(
            m["glyf"], m["hmtx"].metrics, getattr(m.get("vmtx"), "metrics", None)
        )
        for m in master_ttfs
    ]

    for glyph in font.getGlyphOrder():
        log.debug("building gvar for glyph '%s'", glyph)

        allData = [
            m.glyf._getCoordinatesAndControls(glyph, m.hMetrics, m.vMetrics)
            for m in master_datas
        ]

        if allData[defaultMasterIndex][1].numberOfContours != 0:
            # If the default master is not empty, interpret empty non-default masters
            # as missing glyphs from a sparse master
            allData = [
                d if d is not None and d[1].numberOfContours != 0 else None
                for d in allData
            ]

        model, allData = masterModel.getSubModel(allData)

        allCoords = [d[0] for d in allData]
        allControls = [d[1] for d in allData]
        control = allControls[0]
        if not models.allEqual(allControls):
            log.warning("glyph %s has incompatible masters; skipping" % glyph)
            continue
        del allControls

        # Update gvar
        gvar.variations[glyph] = []
        deltas = model.getDeltas(
            allCoords, round=partial(GlyphCoordinates.__round__, round=round)
        )
        supports = model.supports
        assert len(deltas) == len(supports)

        # Prepare for IUP optimization
        origCoords = deltas[0]
        endPts = control.endPts

        for i, (delta, support) in enumerate(zip(deltas[1:], supports[1:])):
            if all(v == 0 for v in delta.array):
                continue
            var = TupleVariation(support, delta)
            if optimize:
                var.optimize(origCoords, endPts, tolerance=tolerance)
            gvar.variations[glyph].append(var)


def _remove_TTHinting(font):
    for tag in ("cvar", "cvt ", "fpgm", "prep"):
        if tag in font:
            del font[tag]
    maxp = font["maxp"]
    for attr in (
        "maxTwilightPoints",
        "maxStorage",
        "maxFunctionDefs",
        "maxInstructionDefs",
        "maxStackElements",
        "maxSizeOfInstructions",
    ):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    font["glyf"].removeHinting()
    # TODO: Modify gasp table to deactivate gridfitting for all ranges?


def _merge_TTHinting(font, masterModel, master_ttfs):
    log.info("Merging TT hinting")
    assert "cvar" not in font

    # Check that the existing hinting is compatible

    # fpgm and prep table

    for tag in ("fpgm", "prep"):
        all_pgms = [m[tag].program for m in master_ttfs if tag in m]
        if not all_pgms:
            continue
        font_pgm = getattr(font.get(tag), "program", None)
        if any(pgm != font_pgm for pgm in all_pgms):
            log.warning(
                "Masters have incompatible %s tables, hinting is discarded." % tag
            )
            _remove_TTHinting(font)
            return

    # glyf table

    font_glyf = font["glyf"]
    master_glyfs = [m["glyf"] for m in master_ttfs]
    for name, glyph in font_glyf.glyphs.items():
        all_pgms = [getattr(glyf.get(name), "program", None) for glyf in master_glyfs]
        if not any(all_pgms):
            continue
        glyph.expand(font_glyf)
        font_pgm = getattr(glyph, "program", None)
        if any(pgm != font_pgm for pgm in all_pgms if pgm):
            log.warning(
                "Masters have incompatible glyph programs in glyph '%s', hinting is discarded."
                % name
            )
            # TODO Only drop hinting from this glyph.
            _remove_TTHinting(font)
            return

    # cvt table

    all_cvs = [Vector(m["cvt "].values) if "cvt " in m else None for m in master_ttfs]

    nonNone_cvs = models.nonNone(all_cvs)
    if not nonNone_cvs:
        # There is no cvt table to make a cvar table from, we're done here.
        return

    if not models.allEqual(len(c) for c in nonNone_cvs):
        log.warning("Masters have incompatible cvt tables, hinting is discarded.")
        _remove_TTHinting(font)
        return

    variations = []
    deltas, supports = masterModel.getDeltasAndSupports(
        all_cvs, round=round
    )  # builtin round calls into Vector.__round__, which uses builtin round as we like
    for i, (delta, support) in enumerate(zip(deltas[1:], supports[1:])):
        if all(v == 0 for v in delta):
            continue
        var = TupleVariation(support, delta)
        variations.append(var)

    # We can build the cvar table now.
    if variations:
        cvar = font["cvar"] = newTable("cvar")
        cvar.version = 1
        cvar.variations = variations


def _has_inconsistent_use_my_metrics_flag(
    master_glyf, glyph_name, flagged_components, expected_num_components
) -> bool:
    master_glyph = master_glyf.get(glyph_name)
    # 'sparse' glyph master doesn't contribute. Besides when components don't match
    # the VF build is going to fail anyway, so be lenient here.
    if (
        master_glyph is not None
        and master_glyph.isComposite()
        and len(master_glyph.components) == expected_num_components
    ):
        for i, base_glyph in flagged_components:
            comp = master_glyph.components[i]
            if comp.glyphName != base_glyph:
                break
            if not (comp.flags & USE_MY_METRICS):
                return True
    return False


def _unset_inconsistent_use_my_metrics_flags(vf, master_fonts):
    """Clear USE_MY_METRICS on composite components if inconsistent across masters.

    If a composite glyph's component has USE_MY_METRICS set differently among
    the masters, the flag is removed from the variable font's glyf table so that
    advance widths are not determined by that single component's phantom points.
    """
    glyf = vf["glyf"]
    master_glyfs = [m["glyf"] for m in master_fonts if "glyf" in m]
    if not master_glyfs:
        # Should not happen: at least the base master (as copied into vf) has glyf
        return

    for glyph_name in glyf.keys():
        glyph = glyf[glyph_name]
        if not glyph.isComposite():
            continue

        # collect indices of component(s) that carry the USE_MY_METRICS flag.
        # This is supposed to be 1 component per composite, but you never know.
        flagged_components = [
            (i, comp.glyphName)
            for i, comp in enumerate(glyph.components)
            if (comp.flags & USE_MY_METRICS)
        ]
        if not flagged_components:
            # Nothing to fix
            continue

        # Verify that for all master glyf tables that contribute this glyph, the
        # corresponding component (same glyphName and index) also carries USE_MY_METRICS
        # and unset the flag if not.
        expected_num_components = len(glyph.components)
        if any(
            _has_inconsistent_use_my_metrics_flag(
                master_glyf, glyph_name, flagged_components, expected_num_components
            )
            for master_glyf in master_glyfs
        ):
            comp_names = [name for _, name in flagged_components]
            log.info(
                "Composite glyph '%s' has inconsistent USE_MY_METRICS flags across "
                "masters; clearing the flag on component%s %s",
                glyph_name,
                "s" if len(comp_names) > 1 else "",
                comp_names if len(comp_names) > 1 else comp_names[0],
            )
            for i, _ in flagged_components:
                glyph.components[i].flags &= ~USE_MY_METRICS


_MetricsFields = namedtuple(
    "_MetricsFields",
    [
        "tableTag",
        "metricsTag",
        "sb1",
        "sb2",
        "advMapping",
        "vOrigMapping",
        "phantomIndex",
    ],
)

HVAR_FIELDS = _MetricsFields(
    tableTag="HVAR",
    metricsTag="hmtx",
    sb1="LsbMap",
    sb2="RsbMap",
    advMapping="AdvWidthMap",
    vOrigMapping=None,
    phantomIndex=0,
)

VVAR_FIELDS = _MetricsFields(
    tableTag="VVAR",
    metricsTag="vmtx",
    sb1="TsbMap",
    sb2="BsbMap",
    advMapping="AdvHeightMap",
    vOrigMapping="VOrgMap",
    phantomIndex=1,
)


def _add_HVAR(font, masterModel, master_ttfs, axisTags):
    getAdvanceMetrics = partial(
        _get_advance_metrics, font, masterModel, master_ttfs, axisTags, HVAR_FIELDS
    )
    _add_VHVAR(font, axisTags, HVAR_FIELDS, getAdvanceMetrics)


def _add_VVAR(font, masterModel, master_ttfs, axisTags):
    getAdvanceMetrics = partial(
        _get_advance_metrics, font, masterModel, master_ttfs, axisTags, VVAR_FIELDS
    )
    _add_VHVAR(font, axisTags, VVAR_FIELDS, getAdvanceMetrics)


def _add_VHVAR(font, axisTags, tableFields, getAdvanceMetrics):
    tableTag = tableFields.tableTag
    assert tableTag not in font
    glyphOrder = font.getGlyphOrder()
    log.info("Generating " + tableTag)
    VHVAR = newTable(tableTag)
    tableClass = getattr(ot, tableTag)
    vhvar = VHVAR.table = tableClass()
    vhvar.Version = 0x00010000

    vhAdvanceDeltasAndSupports, vOrigDeltasAndSupports = getAdvanceMetrics()

    if vOrigDeltasAndSupports:
        singleModel = False
    else:
        singleModel = models.allEqual(
            id(v[1]) for v in vhAdvanceDeltasAndSupports.values()
        )

    directStore = None
    if singleModel:
        # Build direct mapping
        supports = next(iter(vhAdvanceDeltasAndSupports.values()))[1][1:]
        varTupleList = builder.buildVarRegionList(supports, axisTags)
        varTupleIndexes = list(range(len(supports)))
        varData = builder.buildVarData(varTupleIndexes, [], optimize=False)
        for glyphName in glyphOrder:
            varData.addItem(vhAdvanceDeltasAndSupports[glyphName][0], round=noRound)
        varData.optimize()
        directStore = builder.buildVarStore(varTupleList, [varData])
        # remove unused regions from VarRegionList
        directStore.prune_regions()

    # Build optimized indirect mapping
    storeBuilder = varStore.OnlineVarStoreBuilder(axisTags)
    advMapping = {}
    for glyphName in glyphOrder:
        deltas, supports = vhAdvanceDeltasAndSupports[glyphName]
        storeBuilder.setSupports(supports)
        advMapping[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)

    if vOrigDeltasAndSupports:
        vOrigMap = {}
        for glyphName in glyphOrder:
            deltas, supports = vOrigDeltasAndSupports[glyphName]
            storeBuilder.setSupports(supports)
            vOrigMap[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)

    indirectStore = storeBuilder.finish()
    mapping2 = indirectStore.optimize(use_NO_VARIATION_INDEX=False)
    advMapping = [mapping2[advMapping[g]] for g in glyphOrder]
    advanceMapping = builder.buildVarIdxMap(advMapping, glyphOrder)

    if vOrigDeltasAndSupports:
        vOrigMap = [mapping2[vOrigMap[g]] for g in glyphOrder]

    useDirect = False
    vOrigMapping = None
    if directStore:
        # Compile both, see which is more compact

        writer = OTTableWriter()
        directStore.compile(writer, font)
        directSize = len(writer.getAllData())

        writer = OTTableWriter()
        indirectStore.compile(writer, font)
        advanceMapping.compile(writer, font)
        indirectSize = len(writer.getAllData())

        useDirect = directSize < indirectSize

    if useDirect:
        metricsStore = directStore
        advanceMapping = None
    else:
        metricsStore = indirectStore
        if vOrigDeltasAndSupports:
            vOrigMapping = builder.buildVarIdxMap(vOrigMap, glyphOrder)

    vhvar.VarStore = metricsStore
    setattr(vhvar, tableFields.advMapping, advanceMapping)
    if vOrigMapping is not None:
        setattr(vhvar, tableFields.vOrigMapping, vOrigMapping)
    setattr(vhvar, tableFields.sb1, None)
    setattr(vhvar, tableFields.sb2, None)

    font[tableTag] = VHVAR
    return


def _get_advance_metrics(font, masterModel, master_ttfs, axisTags, tableFields):
    tableTag = tableFields.tableTag
    glyphOrder = font.getGlyphOrder()

    # Build list of source font advance widths for each glyph
    metricsTag = tableFields.metricsTag
    advMetricses = [m[metricsTag].metrics for m in master_ttfs]

    # Build list of source font vertical origin coords for each glyph
    if tableTag == "VVAR" and "VORG" in master_ttfs[0]:
        vOrigMetricses = [m["VORG"].VOriginRecords for m in master_ttfs]
        defaultYOrigs = [m["VORG"].defaultVertOriginY for m in master_ttfs]
        vOrigMetricses = list(zip(vOrigMetricses, defaultYOrigs))
    else:
        vOrigMetricses = None

    vhAdvanceDeltasAndSupports = {}
    vOrigDeltasAndSupports = {}
    # HACK: we treat width 65535 as a sentinel value to signal that a glyph
    # from a non-default master should not participate in computing {H,V}VAR,
    # as if it were missing. Allows to variate other glyph-related data independently
    # from glyph metrics
    sparse_advance = 0xFFFF
    for glyph in glyphOrder:
        vhAdvances = [
            (
                metrics[glyph][0]
                if glyph in metrics and metrics[glyph][0] != sparse_advance
                else None
            )
            for metrics in advMetricses
        ]
        vhAdvanceDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(
            vhAdvances, round=round
        )

    if vOrigMetricses:
        for glyph in glyphOrder:
            # We need to supply a vOrigs tuple with non-None default values
            # for each glyph. vOrigMetricses contains values only for those
            # glyphs which have a non-default vOrig.
            vOrigs = [
                metrics[glyph] if glyph in metrics else defaultVOrig
                for metrics, defaultVOrig in vOrigMetricses
            ]
            vOrigDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(
                vOrigs, round=round
            )

    return vhAdvanceDeltasAndSupports, vOrigDeltasAndSupports


def _add_MVAR(font, masterModel, master_ttfs, axisTags):
    log.info("Generating MVAR")

    store_builder = varStore.OnlineVarStoreBuilder(axisTags)

    records = []
    lastTableTag = None
    fontTable = None
    tables = None
    # HACK: we need to special-case post.underlineThickness and .underlinePosition
    # and unilaterally/arbitrarily define a sentinel value to distinguish the case
    # when a post table is present in a given master simply because that's where
    # the glyph names in TrueType must be stored, but the underline values are not
    # meant to be used for building MVAR's deltas. The value of -0x8000 (-32768)
    # the minimum FWord (int16) value, was chosen for its unlikelyhood to appear
    # in real-world underline position/thickness values.
    specialTags = {"unds": -0x8000, "undo": -0x8000}

    for tag, (tableTag, itemName) in sorted(MVAR_ENTRIES.items(), key=lambda kv: kv[1]):
        # For each tag, fetch the associated table from all fonts (or not when we are
        # still looking at a tag from the same tables) and set up the variation model
        # for them.
        if tableTag != lastTableTag:
            tables = fontTable = None
            if tableTag in font:
                fontTable = font[tableTag]
                tables = []
                for master in master_ttfs:
                    if tableTag not in master or (
                        tag in specialTags
                        and getattr(master[tableTag], itemName) == specialTags[tag]
                    ):
                        tables.append(None)
                    else:
                        tables.append(master[tableTag])
                model, tables = masterModel.getSubModel(tables)
                store_builder.setModel(model)
            lastTableTag = tableTag

        if tables is None:  # Tag not applicable to the master font.
            continue

        # TODO support gasp entries

        master_values = [getattr(table, itemName) for table in tables]
        if models.allEqual(master_values):
            base, varIdx = master_values[0], None
        else:
            base, varIdx = store_builder.storeMasters(master_values)
        setattr(fontTable, itemName, base)

        if varIdx is None:
            continue
        log.info("	%s: %s.%s	%s", tag, tableTag, itemName, master_values)
        rec = ot.MetricsValueRecord()
        rec.ValueTag = tag
        rec.VarIdx = varIdx
        records.append(rec)

    assert "MVAR" not in font
    if records:
        store = store_builder.finish()
        # Optimize
        mapping = store.optimize()
        for rec in records:
            rec.VarIdx = mapping[rec.VarIdx]

        MVAR = font["MVAR"] = newTable("MVAR")
        mvar = MVAR.table = ot.MVAR()
        mvar.Version = 0x00010000
        mvar.Reserved = 0
        mvar.VarStore = store
        # XXX these should not be hard-coded but computed automatically
        mvar.ValueRecordSize = 8
        mvar.ValueRecordCount = len(records)
        mvar.ValueRecord = sorted(records, key=lambda r: r.ValueTag)


def _add_BASE(font, masterModel, master_ttfs, axisTags):
    log.info("Generating BASE")

    merger = VariationMerger(masterModel, axisTags, font)
    merger.mergeTables(font, master_ttfs, ["BASE"])
    store = merger.store_builder.finish()

    if not store:
        return
    base = font["BASE"].table
    assert base.Version == 0x00010000
    base.Version = 0x00010001
    base.VarStore = store


def _merge_OTL(font, model, master_fonts, axisTags):
    otl_tags = ["GSUB", "GDEF", "GPOS"]
    if not any(tag in font for tag in otl_tags):
        return

    log.info("Merging OpenType Layout tables")
    merger = VariationMerger(model, axisTags, font)

    merger.mergeTables(font, master_fonts, otl_tags)
    store = merger.store_builder.finish()
    if not store:
        return
    try:
        GDEF = font["GDEF"].table
        assert GDEF.Version <= 0x00010002
    except KeyError:
        GDEFTable = font["GDEF"] = newTable("GDEF")
        GDEF = GDEFTable.table = ot.GDEF()
        GDEF.GlyphClassDef = None
        GDEF.AttachList = None
        GDEF.LigCaretList = None
        GDEF.MarkAttachClassDef = None
        GDEF.MarkGlyphSetsDef = None

    GDEF.Version = 0x00010003
    GDEF.VarStore = store

    # Optimize
    varidx_map = store.optimize()
    GDEF.remap_device_varidxes(varidx_map)
    if "GPOS" in font:
        font["GPOS"].table.remap_device_varidxes(varidx_map)


def _add_GSUB_feature_variations(
    font, axes, internal_axis_supports, rules, featureTags
):
    def normalize(name, value):
        return models.normalizeLocation({name: value}, internal_axis_supports)[name]

    log.info("Generating GSUB FeatureVariations")

    axis_tags = {name: axis.tag for name, axis in axes.items()}

    conditional_subs = []
    for rule in rules:
        region = []
        for conditions in rule.conditionSets:
            space = {}
            for condition in conditions:
                axis_name = condition["name"]
                if condition["minimum"] is not None:
                    minimum = normalize(axis_name, condition["minimum"])
                else:
                    minimum = -1.0
                if condition["maximum"] is not None:
                    maximum = normalize(axis_name, condition["maximum"])
                else:
                    maximum = 1.0
                tag = axis_tags[axis_name]
                space[tag] = (minimum, maximum)
            region.append(space)

        subs = {k: v for k, v in rule.subs}

        conditional_subs.append((region, subs))

    addFeatureVariations(font, conditional_subs, featureTags)


_DesignSpaceData = namedtuple(
    "_DesignSpaceData",
    [
        "axes",
        "axisMappings",
        "internal_axis_supports",
        "base_idx",
        "normalized_master_locs",
        "masters",
        "instances",
        "rules",
        "rulesProcessingLast",
        "lib",
    ],
)


def _add_CFF2(varFont, model, master_fonts):
    from .cff import merge_region_fonts

    glyphOrder = varFont.getGlyphOrder()
    if "CFF2" not in varFont:
        from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2

        convertCFFToCFF2(varFont)

    ordered_fonts_list = model.reorderMasters(master_fonts, model.reverseMapping)
    # re-ordering the master list simplifies building the CFF2 data item lists.
    merge_region_fonts(varFont, model, ordered_fonts_list, glyphOrder)


def _add_COLR(font, model, master_fonts, axisTags, colr_layer_reuse=True):
    merger = COLRVariationMerger(
        model, axisTags, font, allowLayerReuse=colr_layer_reuse
    )
    merger.mergeTables(font, master_fonts)
    store = merger.store_builder.finish()

    colr = font["COLR"].table
    if store:
        mapping = store.optimize()
        colr.VarStore = store
        varIdxes = [mapping[v] for v in merger.varIdxes]
        colr.VarIndexMap = builder.buildDeltaSetIndexMap(varIdxes)


def load_designspace(designspace, log_enabled=True, *, require_sources=True):
    # TODO: remove this and always assume 'designspace' is a DesignSpaceDocument,
    # never a file path, as that's already handled by caller
    if hasattr(designspace, "sources"):  # Assume a DesignspaceDocument
        ds = designspace
    else:  # Assume a file path
        ds = DesignSpaceDocument.fromfile(designspace)

    masters = ds.sources
    if require_sources and not masters:
        raise VarLibValidationError("Designspace must have at least one source.")
    instances = ds.instances

    # TODO: Use fontTools.designspaceLib.tagForAxisName instead.
    standard_axis_map = OrderedDict(
        [
            ("weight", ("wght", {"en": "Weight"})),
            ("width", ("wdth", {"en": "Width"})),
            ("slant", ("slnt", {"en": "Slant"})),
            ("optical", ("opsz", {"en": "Optical Size"})),
            ("italic", ("ital", {"en": "Italic"})),
        ]
    )

    # Setup axes
    if not ds.axes:
        raise VarLibValidationError(f"Designspace must have at least one axis.")

    axes = OrderedDict()
    for axis_index, axis in enumerate(ds.axes):
        axis_name = axis.name
        if not axis_name:
            if not axis.tag:
                raise VarLibValidationError(f"Axis at index {axis_index} needs a tag.")
            axis_name = axis.name = axis.tag

        if axis_name in standard_axis_map:
            if axis.tag is None:
                axis.tag = standard_axis_map[axis_name][0]
            if not axis.labelNames:
                axis.labelNames.update(standard_axis_map[axis_name][1])
        else:
            if not axis.tag:
                raise VarLibValidationError(f"Axis at index {axis_index} needs a tag.")
            if not axis.labelNames:
                axis.labelNames["en"] = tostr(axis_name)

        axes[axis_name] = axis
    if log_enabled:
        log.info("Axes:\n%s", pformat([axis.asdict() for axis in axes.values()]))

    axisMappings = ds.axisMappings
    if axisMappings and log_enabled:
        log.info("Mappings:\n%s", pformat(axisMappings))

    # Check all master and instance locations are valid and fill in defaults
    for obj in masters + instances:
        obj_name = obj.name or obj.styleName or ""
        loc = obj.getFullDesignLocation(ds)
        obj.designLocation = loc
        if loc is None:
            raise VarLibValidationError(
                f"Source or instance '{obj_name}' has no location."
            )
        for axis_name in loc.keys():
            if axis_name not in axes:
                raise VarLibValidationError(
                    f"Location axis '{axis_name}' unknown for '{obj_name}'."
                )
        for axis_name, axis in axes.items():
            v = axis.map_backward(loc[axis_name])
            if not (axis.minimum <= v <= axis.maximum):
                raise VarLibValidationError(
                    f"Source or instance '{obj_name}' has out-of-range location "
                    f"for axis '{axis_name}': is mapped to {v} but must be in "
                    f"mapped range [{axis.minimum}..{axis.maximum}] (NOTE: all "
                    "values are in user-space)."
                )

    # Normalize master locations

    internal_master_locs = [o.getFullDesignLocation(ds) for o in masters]
    if log_enabled:
        log.info("Internal master locations:\n%s", pformat(internal_master_locs))

    # TODO This mapping should ideally be moved closer to logic in _add_fvar/avar
    internal_axis_supports = {}
    for axis in axes.values():
        triple = (axis.minimum, axis.default, axis.maximum)
        internal_axis_supports[axis.name] = [axis.map_forward(v) for v in triple]
    if log_enabled:
        log.info("Internal axis supports:\n%s", pformat(internal_axis_supports))

    normalized_master_locs = [
        models.normalizeLocation(m, internal_axis_supports)
        for m in internal_master_locs
    ]
    if log_enabled:
        log.info("Normalized master locations:\n%s", pformat(normalized_master_locs))

    # Find base master
    base_idx = None
    for i, m in enumerate(normalized_master_locs):
        if all(v == 0 for v in m.values()):
            if base_idx is not None:
                raise VarLibValidationError(
                    "More than one base master found in Designspace."
                )
            base_idx = i
    if require_sources and base_idx is None:
        raise VarLibValidationError(
            "Base master not found; no master at default location?"
        )
    if log_enabled:
        log.info("Index of base master: %s", base_idx)

    return _DesignSpaceData(
        axes,
        axisMappings,
        internal_axis_supports,
        base_idx,
        normalized_master_locs,
        masters,
        instances,
        ds.rules,
        ds.rulesProcessingLast,
        ds.lib,
    )


# https://docs.microsoft.com/en-us/typography/opentype/spec/os2#uswidthclass
WDTH_VALUE_TO_OS2_WIDTH_CLASS = {
    50: 1,
    62.5: 2,
    75: 3,
    87.5: 4,
    100: 5,
    112.5: 6,
    125: 7,
    150: 8,
    200: 9,
}


def set_default_weight_width_slant(font, location):
    if "OS/2" in font:
        if "wght" in location:
            weight_class = otRound(max(1, min(location["wght"], 1000)))
            if font["OS/2"].usWeightClass != weight_class:
                log.info("Setting OS/2.usWeightClass = %s", weight_class)
                font["OS/2"].usWeightClass = weight_class

        if "wdth" in location:
            # map 'wdth' axis (50..200) to OS/2.usWidthClass (1..9), rounding to closest
            widthValue = min(max(location["wdth"], 50), 200)
            widthClass = otRound(
                models.piecewiseLinearMap(widthValue, WDTH_VALUE_TO_OS2_WIDTH_CLASS)
            )
            if font["OS/2"].usWidthClass != widthClass:
                log.info("Setting OS/2.usWidthClass = %s", widthClass)
                font["OS/2"].usWidthClass = widthClass

    if "slnt" in location and "post" in font:
        italicAngle = max(-90, min(location["slnt"], 90))
        if font["post"].italicAngle != italicAngle:
            log.info("Setting post.italicAngle = %s", italicAngle)
            font["post"].italicAngle = italicAngle


def drop_implied_oncurve_points(*masters: TTFont) -> int:
    """Drop impliable on-curve points from all the simple glyphs in masters.

    In TrueType glyf outlines, on-curve points can be implied when they are located
    exactly at the midpoint of the line connecting two consecutive off-curve points.

    The input masters' glyf tables are assumed to contain same-named glyphs that are
    interpolatable. Oncurve points are only dropped if they can be implied for all
    the masters. The fonts are modified in-place.

    Args:
        masters: The TTFont(s) to modify

    Returns:
        The total number of points that were dropped if any.

    Reference:
    https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html
    """

    glyph_masters = defaultdict(list)
    # multiple DS source may point to the same TTFont object and we want to
    # avoid processing the same glyph twice as they are modified in-place
    for font in {id(m): m for m in masters}.values():
        glyf = font["glyf"]
        for glyphName in glyf.keys():
            glyph_masters[glyphName].append(glyf[glyphName])
    count = 0
    for glyphName, glyphs in glyph_masters.items():
        try:
            dropped = dropImpliedOnCurvePoints(*glyphs)
        except ValueError as e:
            # we don't fail for incompatible glyphs in _add_gvar so we shouldn't here
            log.warning("Failed to drop implied oncurves for %r: %s", glyphName, e)
        else:
            count += len(dropped)
    return count


def build_many(
    designspace: DesignSpaceDocument,
    master_finder=lambda s: s,
    exclude=[],
    optimize=True,
    skip_vf=lambda vf_name: False,
    colr_layer_reuse=True,
    drop_implied_oncurves=False,
):
    """
    Build variable fonts from a designspace file, version 5 which can define
    several VFs, or version 4 which has implicitly one VF covering the whole doc.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).

    skip_vf can be used to skip building some of the variable fonts defined in
    the input designspace. It's a predicate that takes as argument the name
    of the variable font and returns `bool`.

    Always returns a Dict[str, TTFont] keyed by VariableFontDescriptor.name
    """
    res = {}
    # varLib.build (used further below) by default only builds an incomplete 'STAT'
    # with an empty AxisValueArray--unless the VF inherited 'STAT' from its base master.
    # Designspace version 5 can also be used to define 'STAT' labels or customize
    # axes ordering, etc. To avoid overwriting a pre-existing 'STAT' or redoing the
    # same work twice, here we check if designspace contains any 'STAT' info before
    # proceeding to call buildVFStatTable for each VF.
    # https://github.com/fonttools/fonttools/pull/3024
    # https://github.com/fonttools/fonttools/issues/3045
    doBuildStatFromDSv5 = (
        "STAT" not in exclude
        and designspace.formatTuple >= (5, 0)
        and (
            any(a.axisLabels or a.axisOrdering is not None for a in designspace.axes)
            or designspace.locationLabels
        )
    )
    for _location, subDoc in splitInterpolable(designspace):
        for name, vfDoc in splitVariableFonts(subDoc):
            if skip_vf(name):
                log.debug(f"Skipping variable TTF font: {name}")
                continue
            vf = build(
                vfDoc,
                master_finder,
                exclude=exclude,
                optimize=optimize,
                colr_layer_reuse=colr_layer_reuse,
                drop_implied_oncurves=drop_implied_oncurves,
            )[0]
            if doBuildStatFromDSv5:
                buildVFStatTable(vf, designspace, name)
            res[name] = vf
    return res


def build(
    designspace,
    master_finder=lambda s: s,
    exclude=[],
    optimize=True,
    colr_layer_reuse=True,
    drop_implied_oncurves=False,
):
    """
    Build variation font from a designspace file.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).
    """
    if hasattr(designspace, "sources"):  # Assume a DesignspaceDocument
        pass
    else:  # Assume a file path
        designspace = DesignSpaceDocument.fromfile(designspace)

    ds = load_designspace(designspace)
    log.info("Building variable font")

    log.info("Loading master fonts")
    master_fonts = load_masters(designspace, master_finder)

    # TODO: 'master_ttfs' is unused except for return value, remove later
    master_ttfs = []
    for master in master_fonts:
        try:
            master_ttfs.append(master.reader.file.name)
        except AttributeError:
            master_ttfs.append(None)  # in-memory fonts have no path

    if drop_implied_oncurves and "glyf" in master_fonts[ds.base_idx]:
        drop_count = drop_implied_oncurve_points(*master_fonts)
        log.info(
            "Dropped %s on-curve points from simple glyphs in the 'glyf' table",
            drop_count,
        )

    # Copy the base master to work from it
    vf = deepcopy(master_fonts[ds.base_idx])

    if "DSIG" in vf:
        del vf["DSIG"]

    # Clear USE_MY_METRICS composite flags if set inconsistently across masters.
    if "glyf" in vf:
        _unset_inconsistent_use_my_metrics_flags(vf, master_fonts)

    # TODO append masters as named-instances as well; needs .designspace change.
    fvar = _add_fvar(vf, ds.axes, ds.instances)
    if "STAT" not in exclude:
        _add_stat(vf)

    # Map from axis names to axis tags...
    normalized_master_locs = [
        {ds.axes[k].tag: v for k, v in loc.items()} for loc in ds.normalized_master_locs
    ]
    # From here on, we use fvar axes only
    axisTags = [axis.axisTag for axis in fvar.axes]

    # Assume single-model for now.
    model = models.VariationModel(normalized_master_locs, axisOrder=axisTags)
    assert 0 == model.mapping[ds.base_idx]

    log.info("Building variations tables")
    if "avar" not in exclude:
        _add_avar(vf, ds.axes, ds.axisMappings, axisTags)
    if "BASE" not in exclude and "BASE" in vf:
        _add_BASE(vf, model, master_fonts, axisTags)
    if "MVAR" not in exclude:
        _add_MVAR(vf, model, master_fonts, axisTags)
    if "HVAR" not in exclude:
        _add_HVAR(vf, model, master_fonts, axisTags)
    if "VVAR" not in exclude and "vmtx" in vf:
        _add_VVAR(vf, model, master_fonts, axisTags)
    if "GDEF" not in exclude or "GPOS" not in exclude:
        _merge_OTL(vf, model, master_fonts, axisTags)
    if "gvar" not in exclude and "glyf" in vf:
        _add_gvar(vf, model, master_fonts, optimize=optimize)
    if "cvar" not in exclude and "glyf" in vf:
        _merge_TTHinting(vf, model, master_fonts)
    if "GSUB" not in exclude and ds.rules:
        featureTags = _feature_variations_tags(ds)
        _add_GSUB_feature_variations(
            vf, ds.axes, ds.internal_axis_supports, ds.rules, featureTags
        )
    if "CFF2" not in exclude and ("CFF " in vf or "CFF2" in vf):
        _add_CFF2(vf, model, master_fonts)
        if "post" in vf:
            # set 'post' to format 2 to keep the glyph names dropped from CFF2
            post = vf["post"]
            if post.formatType != 2.0:
                post.formatType = 2.0
                post.extraNames = []
                post.mapping = {}
    if "COLR" not in exclude and "COLR" in vf and vf["COLR"].version > 0:
        _add_COLR(vf, model, master_fonts, axisTags, colr_layer_reuse)

    set_default_weight_width_slant(
        vf, location={axis.axisTag: axis.defaultValue for axis in vf["fvar"].axes}
    )

    for tag in exclude:
        if tag in vf:
            del vf[tag]

    # TODO: Only return vf for 4.0+, the rest is unused.
    return vf, model, master_ttfs


def _open_font(path, master_finder=lambda s: s):
    # load TTFont masters from given 'path': this can be either a .TTX or an
    # OpenType binary font; or if neither of these, try use the 'master_finder'
    # callable to resolve the path to a valid .TTX or OpenType font binary.
    from fontTools.ttx import guessFileType

    master_path = os.path.normpath(path)
    tp = guessFileType(master_path)
    if tp is None:
        # not an OpenType binary/ttx, fall back to the master finder.
        master_path = master_finder(master_path)
        tp = guessFileType(master_path)
    if tp in ("TTX", "OTX"):
        font = TTFont()
        font.importXML(master_path)
    elif tp in ("TTF", "OTF", "WOFF", "WOFF2"):
        font = TTFont(master_path)
    else:
        raise VarLibValidationError("Invalid master path: %r" % master_path)
    return font


def load_masters(designspace, master_finder=lambda s: s):
    """Ensure that all SourceDescriptor.font attributes have an appropriate TTFont
    object loaded, or else open TTFont objects from the SourceDescriptor.path
    attributes.

    The paths can point to either an OpenType font, a TTX file, or a UFO. In the
    latter case, use the provided master_finder callable to map from UFO paths to
    the respective master font binaries (e.g. .ttf, .otf or .ttx).

    Return list of master TTFont objects in the same order they are listed in the
    DesignSpaceDocument.
    """
    for master in designspace.sources:
        # If a SourceDescriptor has a layer name, demand that the compiled TTFont
        # be supplied by the caller. This spares us from modifying MasterFinder.
        if master.layerName and master.font is None:
            raise VarLibValidationError(
                f"Designspace source '{master.name or '<Unknown>'}' specified a "
                "layer name but lacks the required TTFont object in the 'font' "
                "attribute."
            )

    return designspace.loadSourceFonts(_open_font, master_finder=master_finder)


class MasterFinder(object):
    def __init__(self, template):
        self.template = template

    def __call__(self, src_path):
        fullname = os.path.abspath(src_path)
        dirname, basename = os.path.split(fullname)
        stem, ext = os.path.splitext(basename)
        path = self.template.format(
            fullname=fullname,
            dirname=dirname,
            basename=basename,
            stem=stem,
            ext=ext,
        )
        return os.path.normpath(path)


def _feature_variations_tags(ds):
    raw_tags = ds.lib.get(
        FEAVAR_FEATURETAG_LIB_KEY,
        "rclt" if ds.rulesProcessingLast else "rvrn",
    )
    return sorted({t.strip() for t in raw_tags.split(",")})


def addGSUBFeatureVariations(vf, designspace, featureTags=(), *, log_enabled=False):
    """Add GSUB FeatureVariations table to variable font, based on DesignSpace rules.

    Args:
        vf: A TTFont object representing the variable font.
        designspace: A DesignSpaceDocument object.
        featureTags: Optional feature tag(s) to use for the FeatureVariations records.
            If unset, the key 'com.github.fonttools.varLib.featureVarsFeatureTag' is
            looked up in the DS <lib> and used; otherwise the default is 'rclt' if
            the <rules processing="last"> attribute is set, else 'rvrn'.
            See <https://fonttools.readthedocs.io/en/latest/designspaceLib/xml.html#rules-element>
        log_enabled: If True, log info about DS axes and sources. Default is False, as
            the same info may have already been logged as part of varLib.build.
    """
    ds = load_designspace(designspace, log_enabled=log_enabled)
    if not ds.rules:
        return
    if not featureTags:
        featureTags = _feature_variations_tags(ds)
    _add_GSUB_feature_variations(
        vf, ds.axes, ds.internal_axis_supports, ds.rules, featureTags
    )


def main(args=None):
    """Build variable fonts from a designspace file and masters"""
    from argparse import ArgumentParser
    from fontTools import configLogger

    parser = ArgumentParser(prog="varLib", description=main.__doc__)
    parser.add_argument("designspace")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", metavar="OUTPUTFILE", dest="outfile", default=None, help="output file"
    )
    output_group.add_argument(
        "-d",
        "--output-dir",
        metavar="OUTPUTDIR",
        default=None,
        help="output dir (default: same as input designspace file)",
    )
    parser.add_argument(
        "-x",
        metavar="TAG",
        dest="exclude",
        action="append",
        default=[],
        help="exclude table",
    )
    parser.add_argument(
        "--disable-iup",
        dest="optimize",
        action="store_false",
        help="do not perform IUP optimization",
    )
    parser.add_argument(
        "--no-colr-layer-reuse",
        dest="colr_layer_reuse",
        action="store_false",
        help="do not rebuild variable COLR table to optimize COLR layer reuse",
    )
    parser.add_argument(
        "--drop-implied-oncurves",
        action="store_true",
        help=(
            "drop on-curve points that can be implied when exactly in the middle of "
            "two off-curve points (only applies to TrueType fonts)"
        ),
    )
    parser.add_argument(
        "--master-finder",
        default="master_ttf_interpolatable/{stem}.ttf",
        help=(
            "templated string used for finding binary font "
            "files given the source file names defined in the "
            "designspace document. The following special strings "
            "are defined: {fullname} is the absolute source file "
            "name; {basename} is the file name without its "
            "directory; {stem} is the basename without the file "
            "extension; {ext} is the source file extension; "
            "{dirname} is the directory of the absolute file "
            'name. The default value is "%(default)s".'
        ),
    )
    parser.add_argument(
        "--variable-fonts",
        default=".*",
        metavar="VF_NAME",
        help=(
            "Filter the list of variable fonts produced from the input "
            "Designspace v5 file. By default all listed variable fonts are "
            "generated. To generate a specific variable font (or variable fonts) "
            'that match a given "name" attribute, you can pass as argument '
            "the full name or a regular expression. E.g.: --variable-fonts "
            '"MyFontVF_WeightOnly"; or --variable-fonts "MyFontVFItalic_.*".'
        ),
    )
    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )
    logging_group.add_argument(
        "-q", "--quiet", action="store_true", help="Turn verbosity off."
    )
    options = parser.parse_args(args)

    configLogger(
        level=("DEBUG" if options.verbose else "ERROR" if options.quiet else "INFO")
    )

    designspace_filename = options.designspace
    designspace = DesignSpaceDocument.fromfile(designspace_filename)

    vf_descriptors = designspace.getVariableFonts()
    if not vf_descriptors:
        parser.error(f"No variable fonts in given designspace {designspace.path!r}")

    vfs_to_build = []
    for vf in vf_descriptors:
        # Skip variable fonts that do not match the user's inclusion regex if given.
        if not fullmatch(options.variable_fonts, vf.name):
            continue
        vfs_to_build.append(vf)

    if not vfs_to_build:
        parser.error(f"No variable fonts matching {options.variable_fonts!r}")

    if options.outfile is not None and len(vfs_to_build) > 1:
        parser.error(
            "can't specify -o because there are multiple VFs to build; "
            "use --output-dir, or select a single VF with --variable-fonts"
        )

    output_dir = options.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(designspace_filename)

    vf_name_to_output_path = {}
    if len(vfs_to_build) == 1 and options.outfile is not None:
        vf_name_to_output_path[vfs_to_build[0].name] = options.outfile
    else:
        for vf in vfs_to_build:
            if vf.filename is not None:
                # Only use basename to prevent path traversal attacks
                filename = os.path.basename(vf.filename)
            else:
                # vf.name comes straight from the designspace and is equally
                # attacker-controlled, so basename it too
                filename = os.path.basename(vf.name) + ".{ext}"
            vf_name_to_output_path[vf.name] = os.path.join(output_dir, filename)

    vf_names_to_build = {vf.name for vf in vfs_to_build}
    finder = MasterFinder(options.master_finder)

    vfs = build_many(
        designspace,
        finder,
        exclude=options.exclude,
        optimize=options.optimize,
        skip_vf=lambda name: name not in vf_names_to_build,
        colr_layer_reuse=options.colr_layer_reuse,
        drop_implied_oncurves=options.drop_implied_oncurves,
    )

    for vf_name, vf in vfs.items():
        ext = "otf" if vf.sfntVersion == "OTTO" else "ttf"
        output_path = vf_name_to_output_path[vf_name].format(ext=ext)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info("Saving variation font %s", output_path)
        vf.save(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""
Instantiate a variation font.  Run, eg:

.. code-block:: sh

    $ fonttools varLib.mutator ./NotoSansArabic-VF.ttf wght=140 wdth=85

.. warning::
   ``fontTools.varLib.mutator`` is deprecated in favor of :mod:`fontTools.varLib.instancer`
   which provides equivalent full instancing and also supports partial instancing.
   Please migrate CLI usage to ``fonttools varLib.instancer`` and API usage to
   :func:`fontTools.varLib.instancer.instantiateVariableFont`.
"""

from fontTools.misc.fixedTools import floatToFixedToFloat, floatToFixed
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.roundTools import otRound
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import (
    GlyphCoordinates,
    flagOverlapSimple,
    OVERLAP_COMPOUND,
)
from fontTools.varLib.models import (
    supportScalar,
    normalizeLocation,
    piecewiseLinearMap,
)
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.varStore import VarStoreInstancer
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta
import fontTools.subset.cff
import os.path
import logging
from io import BytesIO

log = logging.getLogger("fontTools.varlib.mutator")

# map 'wdth' axis (1..200) to OS/2.usWidthClass (1..9), rounding to closest
OS2_WIDTH_CLASS_VALUES = {}
percents = [50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0, 150.0, 200.0]
for i, (prev, curr) in enumerate(zip(percents[:-1], percents[1:]), start=1):
    half = (prev + curr) / 2
    OS2_WIDTH_CLASS_VALUES[half] = i


def interpolate_cff2_PrivateDict(topDict, interpolateFromDeltas):
    pd_blend_lists = (
        "BlueValues",
        "OtherBlues",
        "FamilyBlues",
        "FamilyOtherBlues",
        "StemSnapH",
        "StemSnapV",
    )
    pd_blend_values = ("BlueScale", "BlueShift", "BlueFuzz", "StdHW", "StdVW")
    for fontDict in topDict.FDArray:
        pd = fontDict.Private
        vsindex = pd.vsindex if (hasattr(pd, "vsindex")) else 0
        for key, value in pd.rawDict.items():
            if (key in pd_blend_values) and isinstance(value, list):
                delta = interpolateFromDeltas(vsindex, value[1:])
                pd.rawDict[key] = otRound(value[0] + delta)
            elif (key in pd_blend_lists) and isinstance(value[0], list):
                """If any argument in a BlueValues list is a blend list,
                then they all are. The first value of each list is an
                absolute value. The delta tuples are calculated from
                relative master values, hence we need to append all the
                deltas to date to each successive absolute value."""
                delta = 0
                for i, val_list in enumerate(value):
                    delta += otRound(interpolateFromDeltas(vsindex, val_list[1:]))
                    value[i] = val_list[0] + delta


def interpolate_cff2_charstrings(topDict, interpolateFromDeltas, glyphOrder):
    charstrings = topDict.CharStrings
    for gname in glyphOrder:
        # Interpolate charstring
        # e.g replace blend op args with regular args,
        # and use and discard vsindex op.
        charstring = charstrings[gname]
        new_program = []
        vsindex = 0
        last_i = 0
        for i, token in enumerate(charstring.program):
            if token == "vsindex":
                vsindex = charstring.program[i - 1]
                if last_i != 0:
                    new_program.extend(charstring.program[last_i : i - 1])
                last_i = i + 1
            elif token == "blend":
                num_regions = charstring.getNumRegions(vsindex)
                numMasters = 1 + num_regions
                num_args = charstring.program[i - 1]
                # The program list starting at program[i] is now:
                # ..args for following operations
                # num_args values  from the default font
                # num_args tuples, each with numMasters-1 delta values
                # num_blend_args
                # 'blend'
                argi = i - (num_args * numMasters + 1)
                end_args = tuplei = argi + num_args
                while argi < end_args:
                    next_ti = tuplei + num_regions
                    deltas = charstring.program[tuplei:next_ti]
                    delta = interpolateFromDeltas(vsindex, deltas)
                    charstring.program[argi] += otRound(delta)
                    tuplei = next_ti
                    argi += 1
                new_program.extend(charstring.program[last_i:end_args])
                last_i = i + 1
        if last_i != 0:
            new_program.extend(charstring.program[last_i:])
            charstring.program = new_program


def interpolate_cff2_metrics(varfont, topDict, glyphOrder, loc):
    """Unlike TrueType glyphs, neither advance width nor bounding box
    info is stored in a CFF2 charstring. The width data exists only in
    the hmtx and HVAR tables. Since LSB data cannot be interpolated
    reliably from the master LSB values in the hmtx table, we traverse
    the charstring to determine the actual bound box."""

    charstrings = topDict.CharStrings
    boundsPen = BoundsPen(glyphOrder)
    hmtx = varfont["hmtx"]
    hvar_table = None
    if "HVAR" in varfont:
        hvar_table = varfont["HVAR"].table
        fvar = varfont["fvar"]
        varStoreInstancer = VarStoreInstancer(hvar_table.VarStore, fvar.axes, loc)

    for gid, gname in enumerate(glyphOrder):
        entry = list(hmtx[gname])
        # get width delta.
        if hvar_table:
            if hvar_table.AdvWidthMap:
                width_idx = hvar_table.AdvWidthMap.mapping[gname]
            else:
                width_idx = gid
            width_delta = otRound(varStoreInstancer[width_idx])
        else:
            width_delta = 0

        # get LSB.
        boundsPen.init()
        charstring = charstrings[gname]
        charstring.draw(boundsPen)
        if boundsPen.bounds is None:
            # Happens with non-marking glyphs
            lsb_delta = 0
        else:
            lsb = otRound(boundsPen.bounds[0])
            lsb_delta = entry[1] - lsb

        if lsb_delta or width_delta:
            if width_delta:
                entry[0] = max(0, entry[0] + width_delta)
            if lsb_delta:
                entry[1] = lsb
            hmtx[gname] = tuple(entry)


@deprecateFunction(
    "use fontTools.varLib.instancer.instantiateVariableFont instead "
    "for either full or partial instancing",
)
def instantiateVariableFont(varfont, location, inplace=False, overlap=True):
    """Generate a static instance from a variable TTFont and a dictionary
    defining the desired location along the variable font's axes.
    The location values must be specified as user-space coordinates, e.g.:

    .. code-block::

        {'wght': 400, 'wdth': 100}

    By default, a new TTFont object is returned. If ``inplace`` is True, the
    input varfont is modified and reduced to a static font.

    When the overlap parameter is defined as True,
    OVERLAP_SIMPLE and OVERLAP_COMPOUND bits are set to 1.  See
    https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
    """
    if not inplace:
        # make a copy to leave input varfont unmodified
        stream = BytesIO()
        varfont.save(stream)
        stream.seek(0)
        varfont = TTFont(stream)

    fvar = varfont["fvar"]
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes}
    loc = normalizeLocation(location, axes)
    if "avar" in varfont:
        maps = varfont["avar"].segments
        loc = {k: piecewiseLinearMap(v, maps[k]) for k, v in loc.items()}
    # Quantize to F2Dot14, to avoid surprise interpolations.
    loc = {k: floatToFixedToFloat(v, 14) for k, v in loc.items()}
    # Location is normalized now
    log.info("Normalized location: %s", loc)

    if "gvar" in varfont:
        log.info("Mutating glyf/gvar tables")
        gvar = varfont["gvar"]
        glyf = varfont["glyf"]
        hMetrics = varfont["hmtx"].metrics
        vMetrics = getattr(varfont.get("vmtx"), "metrics", None)
        # get list of glyph names in gvar sorted by component depth
        glyphnames = sorted(
            gvar.variations.keys(),
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
            variations = gvar.variations[glyphname]
            coordinates, _ = glyf._getCoordinatesAndControls(
                glyphname, hMetrics, vMetrics
            )
            origCoords, endPts = None, None
            for var in variations:
                scalar = supportScalar(loc, var.axes)
                if not scalar:
                    continue
                delta = var.coordinates
                if None in delta:
                    if origCoords is None:
                        origCoords, g = glyf._getCoordinatesAndControls(
                            glyphname, hMetrics, vMetrics
                        )
                    delta = iup_delta(delta, origCoords, g.endPts)
                coordinates += GlyphCoordinates(delta) * scalar
            glyf._setCoordinates(glyphname, coordinates, hMetrics, vMetrics)
    else:
        glyf = None

    if "DSIG" in varfont:
        del varfont["DSIG"]

    if "cvar" in varfont:
        log.info("Mutating cvt/cvar tables")
        cvar = varfont["cvar"]
        cvt = varfont["cvt "]
        deltas = {}
        for var in cvar.variations:
            scalar = supportScalar(loc, var.axes)
            if not scalar:
                continue
            for i, c in enumerate(var.coordinates):
                if c is not None:
                    deltas[i] = deltas.get(i, 0) + scalar * c
        for i, delta in deltas.items():
            cvt[i] += otRound(delta)

    if "CFF2" in varfont:
        log.info("Mutating CFF2 table")
        glyphOrder = varfont.getGlyphOrder()
        CFF2 = varfont["CFF2"]
        topDict = CFF2.cff.topDictIndex[0]
        vsInstancer = VarStoreInstancer(topDict.VarStore.otVarStore, fvar.axes, loc)
        interpolateFromDeltas = vsInstancer.interpolateFromDeltas
        interpolate_cff2_PrivateDict(topDict, interpolateFromDeltas)
        CFF2.desubroutinize()
        interpolate_cff2_charstrings(topDict, interpolateFromDeltas, glyphOrder)
        interpolate_cff2_metrics(varfont, topDict, glyphOrder, loc)
        del topDict.rawDict["VarStore"]
        del topDict.VarStore

    if "MVAR" in varfont:
        log.info("Mutating MVAR table")
        mvar = varfont["MVAR"].table
        varStoreInstancer = VarStoreInstancer(mvar.VarStore, fvar.axes, loc)
        records = mvar.ValueRecord
        for rec in records:
            mvarTag = rec.ValueTag
            if mvarTag not in MVAR_ENTRIES:
                continue
            tableTag, itemName = MVAR_ENTRIES[mvarTag]
            delta = otRound(varStoreInstancer[rec.VarIdx])
            if not delta:
                continue
            setattr(
                varfont[tableTag],
                itemName,
                getattr(varfont[tableTag], itemName) + delta,
            )

    log.info("Mutating FeatureVariations")
    for tableTag in "GSUB", "GPOS":
        if not tableTag in varfont:
            continue
        table = varfont[tableTag].table
        if not getattr(table, "FeatureVariations", None):
            continue
        variations = table.FeatureVariations
        for record in variations.FeatureVariationRecord:
            applies = True
            for condition in record.ConditionSet.ConditionTable:
                if condition.Format == 1:
                    axisIdx = condition.AxisIndex
                    axisTag = fvar.axes[axisIdx].axisTag
                    Min = condition.FilterRangeMinValue
                    Max = condition.FilterRangeMaxValue
                    v = loc[axisTag]
                    if not (Min <= v <= Max):
                        applies = False
                else:
                    applies = False
                if not applies:
                    break

            if applies:
                assert record.FeatureTableSubstitution.Version == 0x00010000
                for rec in record.FeatureTableSubstitution.SubstitutionRecord:
                    table.FeatureList.FeatureRecord[rec.FeatureIndex].Feature = (
                        rec.Feature
                    )
                break
        del table.FeatureVariations

    if "GDEF" in varfont and varfont["GDEF"].table.Version >= 0x00010003:
        log.info("Mutating GDEF/GPOS/GSUB tables")
        gdef = varfont["GDEF"].table
        instancer = VarStoreInstancer(gdef.VarStore, fvar.axes, loc)

        merger = MutatorMerger(varfont, instancer)
        merger.mergeTables(varfont, [varfont], ["GDEF", "GPOS"])

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

    addidef = False
    if glyf:
        for glyph in glyf.glyphs.values():
            if hasattr(glyph, "program"):
                instructions = glyph.program.getAssembly()
                # If GETVARIATION opcode is used in bytecode of any glyph add IDEF
                addidef = any(op.startswith("GETVARIATION") for op in instructions)
                if addidef:
                    break
        if overlap:
            for glyph_name in glyf.keys():
                glyph = glyf[glyph_name]
                # Set OVERLAP_COMPOUND bit for compound glyphs
                if glyph.isComposite():
                    glyph.components[0].flags |= OVERLAP_COMPOUND
                # Set OVERLAP_SIMPLE bit for simple glyphs
                elif glyph.numberOfContours > 0:
                    glyph.flags[0] |= flagOverlapSimple
    if addidef:
        log.info("Adding IDEF to fpgm table for GETVARIATION opcode")
        asm = []
        if "fpgm" in varfont:
            fpgm = varfont["fpgm"]
            asm = fpgm.program.getAssembly()
        else:
            fpgm = newTable("fpgm")
            fpgm.program = ttProgram.Program()
            varfont["fpgm"] = fpgm
        asm.append("PUSHB[000] 145")
        asm.append("IDEF[ ]")
        args = [str(len(loc))]
        for a in fvar.axes:
            args.append(str(floatToFixed(loc[a.axisTag], 14)))
        asm.append("NPUSHW[ ] " + " ".join(args))
        asm.append("ENDF[ ]")
        fpgm.program.fromAssembly(asm)

        # Change maxp attributes as IDEF is added
        if "maxp" in varfont:
            maxp = varfont["maxp"]
            setattr(
                maxp, "maxInstructionDefs", 1 + getattr(maxp, "maxInstructionDefs", 0)
            )
            setattr(
                maxp,
                "maxStackElements",
                max(len(loc), getattr(maxp, "maxStackElements", 0)),
            )

    if "name" in varfont:
        log.info("Pruning name table")
        exclude = {a.axisNameID for a in fvar.axes}
        for i in fvar.instances:
            exclude.add(i.subfamilyNameID)
            exclude.add(i.postscriptNameID)
        if "ltag" in varfont:
            # Drop the whole 'ltag' table if all its language tags are referenced by
            # name records to be pruned.
            # TODO: prune unused ltag tags and re-enumerate langIDs accordingly
            excludedUnicodeLangIDs = [
                n.langID
                for n in varfont["name"].names
                if n.nameID in exclude and n.platformID == 0 and n.langID != 0xFFFF
            ]
            if set(excludedUnicodeLangIDs) == set(range(len((varfont["ltag"].tags)))):
                del varfont["ltag"]
        varfont["name"].names[:] = [
            n
            for n in varfont["name"].names
            if n.nameID < 256 or n.nameID not in exclude
        ]

    if "wght" in location and "OS/2" in varfont:
        varfont["OS/2"].usWeightClass = otRound(max(1, min(location["wght"], 1000)))
    if "wdth" in location:
        wdth = location["wdth"]
        for percent, widthClass in sorted(OS2_WIDTH_CLASS_VALUES.items()):
            if wdth < percent:
                varfont["OS/2"].usWidthClass = widthClass
                break
        else:
            varfont["OS/2"].usWidthClass = 9
    if "slnt" in location and "post" in varfont:
        varfont["post"].italicAngle = max(-90, min(location["slnt"], 90))

    log.info("Removing variable tables")
    for tag in ("avar", "cvar", "fvar", "gvar", "HVAR", "MVAR", "VVAR", "STAT"):
        if tag in varfont:
            del varfont[tag]

    return varfont


def main(args=None):
    """Instantiate a variation font"""
    from fontTools import configLogger
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.mutator", description="Instantiate a variable font"
    )
    parser.add_argument("input", metavar="INPUT.ttf", help="Input variable TTF file.")
    parser.add_argument(
        "locargs",
        metavar="AXIS=LOC",
        nargs="*",
        help="List of space separated locations. A location consist in "
        "the name of a variation axis, followed by '=' and a number. E.g.: "
        " wght=700 wdth=80. The default is the location of the base master.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT.ttf",
        default=None,
        help="Output instance TTF file (default: INPUT-instance.ttf).",
    )
    parser.add_argument(
        "--no-recalc-timestamp",
        dest="recalc_timestamp",
        action="store_false",
        help="Don't set the output font's timestamp to the current time.",
    )
    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )
    logging_group.add_argument(
        "-q", "--quiet", action="store_true", help="Turn verbosity off."
    )
    parser.add_argument(
        "--no-overlap",
        dest="overlap",
        action="store_false",
        help="Don't set OVERLAP_SIMPLE/OVERLAP_COMPOUND glyf flags.",
    )
    options = parser.parse_args(args)

    varfilename = options.input
    outfile = (
        os.path.splitext(varfilename)[0] + "-instance.ttf"
        if not options.output
        else options.output
    )
    configLogger(
        level=("DEBUG" if options.verbose else "ERROR" if options.quiet else "INFO")
    )

    loc = {}
    for arg in options.locargs:
        try:
            tag, val = arg.split("=")
            assert len(tag) <= 4
            loc[tag.ljust(4)] = float(val)
        except (ValueError, AssertionError):
            parser.error("invalid location argument format: %r" % arg)
    log.info("Location: %s", loc)

    log.info("Loading variable font")
    varfont = TTFont(varfilename, recalcTimestamp=options.recalc_timestamp)

    instantiateVariableFont(varfont, loc, inplace=True, overlap=options.overlap)

    log.info("Saving instance font %s", outfile)
    varfont.save(outfile)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>from .interpolatableHelpers import *
import logging

log = logging.getLogger("fontTools.varLib.interpolatable")


def test_contour_order(glyph0, glyph1):
    # We try matching both the StatisticsControlPen vector
    # and the StatisticsPen vector.
    #
    # If either method found an identity matching, accept it.
    # This is crucial for fonts like Kablammo[MORF].ttf and
    # Nabla[EDPT,EHLT].ttf, since they really confuse the
    # StatisticsPen vector because of their area=0 contours.

    n = len(glyph0.controlVectors)
    matching = None
    matching_cost = 0
    identity_cost = 0
    done = n <= 1
    if not done:
        m0Control = glyph0.controlVectors
        m1Control = glyph1.controlVectors
        (
            matching_control,
            matching_cost_control,
            identity_cost_control,
        ) = matching_for_vectors(m0Control, m1Control)
        done = matching_cost_control == identity_cost_control
    if not done:
        m0Green = glyph0.greenVectors
        m1Green = glyph1.greenVectors
        (
            matching_green,
            matching_cost_green,
            identity_cost_green,
        ) = matching_for_vectors(m0Green, m1Green)
        done = matching_cost_green == identity_cost_green

    if not done:
        # See if reversing contours in one master helps.
        # That's a common problem.  Then the wrong_start_point
        # test will fix them.
        #
        # Reverse the sign of the area (0); the rest stay the same.
        m1ControlReversed = [(-m[0],) + m[1:] for m in m1Control]
        (
            matching_control_reversed,
            matching_cost_control_reversed,
            identity_cost_control_reversed,
        ) = matching_for_vectors(m0Control, m1ControlReversed)
        done = matching_cost_control_reversed == identity_cost_control_reversed
        if not done:
            m1GreenReversed = [(-m[0],) + m[1:] for m in m1Green]
            (
                matching_green_reversed,
                matching_cost_green_reversed,
                identity_cost_green_reversed,
            ) = matching_for_vectors(m0Green, m1GreenReversed)
            done = matching_cost_green_reversed == identity_cost_green_reversed

        if not done:
            # Otherwise, use the best of the two matchings.
            if (
                matching_cost_control * identity_cost_green
                < matching_cost_green * identity_cost_control
            ):
                matching = matching_control
                matching_cost = matching_cost_control
                identity_cost = identity_cost_control
            else:
                matching = matching_green
                matching_cost = matching_cost_green
                identity_cost = identity_cost_green

    this_tolerance = matching_cost / identity_cost if identity_cost else 1
    log.debug(
        "test-contour-order: tolerance %g",
        this_tolerance,
    )
    return this_tolerance, matching
<|endoftext|>"""
Interpolate OpenType Layout tables (GDEF / GPOS / GSUB).
"""

from fontTools.ttLib import TTFont
from fontTools.varLib import models, VarLibError, load_designspace, load_masters
from fontTools.varLib.merger import InstancerMerger
import os.path
import logging
from copy import deepcopy
from pprint import pformat

log = logging.getLogger("fontTools.varLib.interpolate_layout")


def interpolate_layout(designspace, loc, master_finder=lambda s: s, mapped=False):
    """
    Interpolate GPOS from a designspace file and location.

    If master_finder is set, it should be a callable that takes master
    filename as found in designspace file and map it to master font
    binary as to be opened (eg. .ttf or .otf).

    If mapped is False (default), then location is mapped using the
    map element of the axes in designspace file.  If mapped is True,
    it is assumed that location is in designspace's internal space and
    no mapping is performed.
    """
    if hasattr(designspace, "sources"):  # Assume a DesignspaceDocument
        pass
    else:  # Assume a file path
        from fontTools.designspaceLib import DesignSpaceDocument

        designspace = DesignSpaceDocument.fromfile(designspace)

    ds = load_designspace(designspace)
    log.info("Building interpolated font")

    log.info("Loading master fonts")
    master_fonts = load_masters(designspace, master_finder)
    font = deepcopy(master_fonts[ds.base_idx])

    log.info("Location: %s", pformat(loc))
    if not mapped:
        loc = {name: ds.axes[name].map_forward(v) for name, v in loc.items()}
    log.info("Internal location: %s", pformat(loc))
    loc = models.normalizeLocation(loc, ds.internal_axis_supports)
    log.info("Normalized location: %s", pformat(loc))

    # Assume single-model for now.
    model = models.VariationModel(ds.normalized_master_locs)
    assert 0 == model.mapping[ds.base_idx]

    merger = InstancerMerger(font, model, loc)

    log.info("Building interpolated tables")
    # TODO GSUB/GDEF
    merger.mergeTables(font, master_fonts, ["GPOS"])
    return font


def main(args=None):
    """Interpolate GDEF/GPOS/GSUB tables for a point on a designspace"""
    from fontTools import configLogger
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        "fonttools varLib.interpolate_layout",
        description=main.__doc__,
    )
    parser.add_argument(
        "designspace_filename", metavar="DESIGNSPACE", help="Input TTF files"
    )
    parser.add_argument(
        "locations",
        metavar="LOCATION",
        type=str,
        nargs="+",
        help="Axis locations (e.g. wdth=120",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="Output font file (defaults to <designspacename>-instance.ttf)",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        metavar="LEVEL",
        default="INFO",
        help="Logging level (defaults to INFO)",
    )

    args = parser.parse_args(args)

    if not args.output:
        args.output = os.path.splitext(args.designspace_filename)[0] + "-instance.ttf"

    configLogger(level=args.loglevel)

    finder = lambda s: s.replace("master_ufo", "master_ttf_interpolatable").replace(
        ".ufo", ".ttf"
    )

    loc = {}
    for arg in args.locations:
        tag, val = arg.split("=")
        loc[tag] = float(val)

    font = interpolate_layout(args.designspace_filename, loc, finder)
    log.info("Saving font %s", args.output)
    font.save(args.output)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>from collections import namedtuple
from fontTools.cffLib import (
    maxStackLimit,
    TopDictIndex,
    buildOrder,
    topDictOperators,
    topDictOperators2,
    privateDictOperators,
    privateDictOperators2,
    FDArrayIndex,
    FontDict,
    VarStoreData,
)
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial

from .errors import (
    VarLibCFFDictMergeError,
    VarLibCFFPointTypeMergeError,
    VarLibCFFHintTypeMergeError,
    VarLibMergeError,
)

# Backwards compatibility
MergeDictError = VarLibCFFDictMergeError
MergeTypeError = VarLibCFFPointTypeMergeError


def addCFFVarStore(varFont, varModel, varDataList, masterSupports):
    fvarTable = varFont["fvar"]
    axisKeys = [axis.axisTag for axis in fvarTable.axes]
    varTupleList = varLib.builder.buildVarRegionList(masterSupports, axisKeys)
    varStoreCFFV = varLib.builder.buildVarStore(varTupleList, varDataList)

    topDict = varFont["CFF2"].cff.topDictIndex[0]
    topDict.VarStore = VarStoreData(otVarStore=varStoreCFFV)
    if topDict.FDArray[0].vstore is None:
        fdArray = topDict.FDArray
        for fontDict in fdArray:
            if hasattr(fontDict, "Private"):
                fontDict.Private.vstore = topDict.VarStore


@deprecateFunction("Use fontTools.cffLib.CFFToCFF2.convertCFFToCFF2 instead.")
def convertCFFtoCFF2(varFont):
    from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2

    return convertCFFToCFF2(varFont)


def conv_to_int(num):
    if isinstance(num, float) and num.is_integer():
        return int(num)
    return num


pd_blend_fields = (
    "BlueValues",
    "OtherBlues",
    "FamilyBlues",
    "FamilyOtherBlues",
    "BlueScale",
    "BlueShift",
    "BlueFuzz",
    "StdHW",
    "StdVW",
    "StemSnapH",
    "StemSnapV",
)


def get_private(regionFDArrays, fd_index, ri, fd_map):
    region_fdArray = regionFDArrays[ri]
    region_fd_map = fd_map[fd_index]
    if ri in region_fd_map:
        region_fdIndex = region_fd_map[ri]
        private = region_fdArray[region_fdIndex].Private
    else:
        private = None
    return private


def merge_PrivateDicts(top_dicts, vsindex_dict, var_model, fd_map):
    """
    I step through the FontDicts in the FDArray of the varfont TopDict.
    For each varfont FontDict:

    * step through each key in FontDict.Private.
    * For each key, step through each relevant source font Private dict, and
      build a list of values to blend.

    The 'relevant' source fonts are selected by first getting the right
    submodel using ``vsindex_dict[vsindex]``. The indices of the
    ``subModel.locations`` are mapped to source font list indices by
    assuming the latter order is the same as the order of the
    ``var_model.locations``. I can then get the index of each subModel
    location in the list of ``var_model.locations``.
    """

    topDict = top_dicts[0]
    region_top_dicts = top_dicts[1:]
    if hasattr(region_top_dicts[0], "FDArray"):
        regionFDArrays = [fdTopDict.FDArray for fdTopDict in region_top_dicts]
    else:
        regionFDArrays = [[fdTopDict] for fdTopDict in region_top_dicts]
    for fd_index, font_dict in enumerate(topDict.FDArray):
        private_dict = font_dict.Private
        vsindex = getattr(private_dict, "vsindex", 0)
        # At the moment, no PrivateDict has a vsindex key, but let's support
        # how it should work. See comment at end of
        # merge_charstrings() - still need to optimize use of vsindex.
        sub_model, _ = vsindex_dict[vsindex]
        master_indices = []
        for loc in sub_model.locations[1:]:
            i = var_model.locations.index(loc) - 1
            master_indices.append(i)
        pds = [private_dict]
        last_pd = private_dict
        for ri in master_indices:
            pd = get_private(regionFDArrays, fd_index, ri, fd_map)
            # If the region font doesn't have this FontDict, just reference
            # the last one used.
            if pd is None:
                pd = last_pd
            else:
                last_pd = pd
            pds.append(pd)
        num_masters = len(pds)
        for key, value in private_dict.rawDict.items():
            dataList = []
            if key not in pd_blend_fields:
                continue
            if isinstance(value, list):
                try:
                    values = [pd.rawDict[key] for pd in pds]
                except KeyError:
                    print(
                        "Warning: {key} in default font Private dict is "
                        "missing from another font, and was "
                        "discarded.".format(key=key)
                    )
                    continue
                try:
                    values = zip(*values)
                except IndexError:
                    raise VarLibCFFDictMergeError(key, value, values)
                """
				Row 0 contains the first  value from each master.
				Convert each row from absolute values to relative
				values from the previous row.
				e.g for three masters,	a list of values was:
				master 0 OtherBlues = [-217,-205]
				master 1 OtherBlues = [-234,-222]
				master 1 OtherBlues = [-188,-176]
				The call to zip() converts this to:
				[(-217, -234, -188), (-205, -222, -176)]
				and is converted finally to:
				OtherBlues = [[-217, 17.0, 46.0], [-205, 0.0, 0.0]]
				"""
                prev_val_list = [0] * num_masters
                any_points_differ = False
                for val_list in values:
                    rel_list = [
                        (val - prev_val_list[i]) for (i, val) in enumerate(val_list)
                    ]
                    if (not any_points_differ) and not allEqual(rel_list):
                        any_points_differ = True
                    prev_val_list = val_list
                    deltas = sub_model.getDeltas(rel_list)
                    # For PrivateDict BlueValues, the default font
                    # values are absolute, not relative to the prior value.
                    deltas[0] = val_list[0]
                    dataList.append(deltas)
                # If there are no blend values,then
                # we can collapse the blend lists.
                if not any_points_differ:
                    dataList = [data[0] for data in dataList]
            else:
                values = [pd.rawDict[key] for pd in pds]
                if not allEqual(values):
                    dataList = sub_model.getDeltas(values)
                else:
                    dataList = values[0]

            # Convert numbers with no decimal part to an int
            if isinstance(dataList, list):
                for i, item in enumerate(dataList):
                    if isinstance(item, list):
                        for j, jtem in enumerate(item):
                            dataList[i][j] = conv_to_int(jtem)
                    else:
                        dataList[i] = conv_to_int(item)
            else:
                dataList = conv_to_int(dataList)

            private_dict.rawDict[key] = dataList


def _cff_or_cff2(font):
    if "CFF " in font:
        return font["CFF "]
    return font["CFF2"]


def getfd_map(varFont, fonts_list):
    """Since a subset source font may have fewer FontDicts in their
    FDArray than the default font, we have to match up the FontDicts in
    the different fonts . We do this with the FDSelect array, and by
    assuming that the same glyph will reference  matching FontDicts in
    each source font. We return a mapping from fdIndex in the default
    font to a dictionary which maps each master list index of each
    region font to the equivalent fdIndex in the region font."""
    fd_map = {}
    default_font = fonts_list[0]
    region_fonts = fonts_list[1:]
    num_regions = len(region_fonts)
    topDict = _cff_or_cff2(default_font).cff.topDictIndex[0]
    if not hasattr(topDict, "FDSelect"):
        # All glyphs reference only one FontDict.
        # Map the FD index for regions to index 0.
        fd_map[0] = {ri: 0 for ri in range(num_regions)}
        return fd_map

    gname_mapping = {}
    default_fdSelect = topDict.FDSelect
    glyphOrder = default_font.getGlyphOrder()
    for gid, fdIndex in enumerate(default_fdSelect):
        gname_mapping[glyphOrder[gid]] = fdIndex
        if fdIndex not in fd_map:
            fd_map[fdIndex] = {}
    for ri, region_font in enumerate(region_fonts):
        region_glyphOrder = region_font.getGlyphOrder()
        region_topDict = _cff_or_cff2(region_font).cff.topDictIndex[0]
        if not hasattr(region_topDict, "FDSelect"):
            # All the glyphs share the same FontDict. Pick any glyph.
            default_fdIndex = gname_mapping[region_glyphOrder[0]]
            fd_map[default_fdIndex][ri] = 0
        else:
            region_fdSelect = region_topDict.FDSelect
            for gid, fdIndex in enumerate(region_fdSelect):
                default_fdIndex = gname_mapping[region_glyphOrder[gid]]
                region_map = fd_map[default_fdIndex]
                if ri not in region_map:
                    region_map[ri] = fdIndex
    return fd_map


CVarData = namedtuple("CVarData", "varDataList masterSupports vsindex_dict")


def merge_region_fonts(varFont, model, ordered_fonts_list, glyphOrder):
    topDict = varFont["CFF2"].cff.topDictIndex[0]
    top_dicts = [topDict] + [
        _cff_or_cff2(ttFont).cff.topDictIndex[0] for ttFont in ordered_fonts_list[1:]
    ]
    num_masters = len(model.mapping)
    cvData = merge_charstrings(glyphOrder, num_masters, top_dicts, model)
    fd_map = getfd_map(varFont, ordered_fonts_list)
    merge_PrivateDicts(top_dicts, cvData.vsindex_dict, model, fd_map)
    addCFFVarStore(varFont, model, cvData.varDataList, cvData.masterSupports)


def _get_cs(charstrings, glyphName, filterEmpty=False):
    if glyphName not in charstrings:
        return None
    cs = charstrings[glyphName]

    if filterEmpty:
        cs.decompile()
        if cs.program == []:  # CFF2 empty charstring
            return None
        elif (
            len(cs.program) <= 2
            and cs.program[-1] == "endchar"
            and (len(cs.program) == 1 or type(cs.program[0]) in (int, float))
        ):  # CFF1 empty charstring
            return None

    return cs


def _add_new_vsindex(
    model, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList
):
    varTupleIndexes = []
    for support in model.supports[1:]:
        if support not in masterSupports:
            masterSupports.append(support)
        varTupleIndexes.append(masterSupports.index(support))
    var_data = varLib.builder.buildVarData(varTupleIndexes, None, False)
    vsindex = len(vsindex_dict)
    vsindex_by_key[key] = vsindex
    vsindex_dict[vsindex] = (model, [key])
    varDataList.append(var_data)
    return vsindex


def merge_charstrings(glyphOrder, num_masters, top_dicts, masterModel):
    vsindex_dict = {}
    vsindex_by_key = {}
    varDataList = []
    masterSupports = []
    default_charstrings = top_dicts[0].CharStrings
    for gid, gname in enumerate(glyphOrder):
        # interpret empty non-default masters as missing glyphs from a sparse master
        all_cs = [
            _get_cs(td.CharStrings, gname, i != 0) for i, td in enumerate(top_dicts)
        ]
        model, model_cs = masterModel.getSubModel(all_cs)
        # create the first pass CFF2 charstring, from
        # the default charstring.
        default_charstring = model_cs[0]
        var_pen = CFF2CharStringMergePen([], gname, num_masters, 0)
        # We need to override outlineExtractor because these
        # charstrings do have widths in the 'program'; we need to drop these
        # values rather than post assertion error for them.
        default_charstring.outlineExtractor = MergeOutlineExtractor
        default_charstring.draw(var_pen)

        # Add the coordinates from all the other regions to the
        # blend lists in the CFF2 charstring.
        region_cs = model_cs[1:]
        for region_idx, region_charstring in enumerate(region_cs, start=1):
            var_pen.restart(region_idx)
            region_charstring.outlineExtractor = MergeOutlineExtractor
            region_charstring.draw(var_pen)

        # Collapse each coordinate list to a blend operator and its args.
        new_cs = var_pen.getCharString(
            private=default_charstring.private,
            globalSubrs=default_charstring.globalSubrs,
            var_model=model,
            optimize=True,
        )
        default_charstrings[gname] = new_cs

        if not region_cs:
            continue

        if (not var_pen.seen_moveto) or ("blend" not in new_cs.program):
            # If this is not a marking glyph, or if there are no blend
            # arguments, then we can use vsindex 0. No need to
            # check if we need a new vsindex.
            continue

        # If the charstring required a new model, create
        # a VarData table to go with, and set vsindex.
        key = tuple(v is not None for v in all_cs)
        try:
            vsindex = vsindex_by_key[key]
        except KeyError:
            vsindex = _add_new_vsindex(
                model, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList
            )
        # We do not need to check for an existing new_cs.private.vsindex,
        # as we know it doesn't exist yet.
        if vsindex != 0:
            new_cs.program[:0] = [vsindex, "vsindex"]

    # If there is no variation in any of the charstrings, then vsindex_dict
    # never gets built. This could still be needed if there is variation
    # in the PrivatDict, so we will build the default data for vsindex = 0.
    if not vsindex_dict:
        key = (True,) * num_masters
        _add_new_vsindex(
            masterModel, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList
        )
    cvData = CVarData(
        varDataList=varDataList,
        masterSupports=masterSupports,
        vsindex_dict=vsindex_dict,
    )
    # XXX To do: optimize use of vsindex between the PrivateDicts and
    # charstrings
    return cvData


class CFFToCFF2OutlineExtractor(T2OutlineExtractor):
    """This class is used to remove the initial width from the CFF
    charstring without trying to add the width to self.nominalWidthX,
    which is None."""

    def popallWidth(self, evenOdd=0):
        args = self.popall()
        if not self.gotWidth:
            if evenOdd ^ (len(args) % 2):
                args = args[1:]
            self.width = self.defaultWidthX
            self.gotWidth = 1
        return args


class MergeOutlineExtractor(CFFToCFF2OutlineExtractor):
    """Used to extract the charstring commands - including hints - from a
    CFF charstring in order to merge it as another set of region data
    into a CFF2 variable font charstring."""

    def __init__(
        self,
        pen,
        localSubrs,
        globalSubrs,
        nominalWidthX,
        defaultWidthX,
        private=None,
        blender=None,
    ):
        super().__init__(
            pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender
        )

    def countHints(self):
        args = self.popallWidth()
        self.hintCount = self.hintCount + len(args) // 2
        return args

    def _hint_op(self, type, args):
        self.pen.add_hint(type, args)

    def op_hstem(self, index):
        args = self.countHints()
        self._hint_op("hstem", args)

    def op_vstem(self, index):
        args = self.countHints()
        self._hint_op("vstem", args)

    def op_hstemhm(self, index):
        args = self.countHints()
        self._hint_op("hstemhm", args)

    def op_vstemhm(self, index):
        args = self.countHints()
        self._hint_op("vstemhm", args)

    def _get_hintmask(self, index):
        if not self.hintMaskBytes:
            args = self.countHints()
            if args:
                self._hint_op("vstemhm", args)
            self.hintMaskBytes = (self.hintCount + 7) // 8
        hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
        return index, hintMaskBytes

    def op_hintmask(self, index):
        index, hintMaskBytes = self._get_hintmask(index)
        self.pen.add_hintmask("hintmask", [hintMaskBytes])
        return hintMaskBytes, index

    def op_cntrmask(self, index):
        index, hintMaskBytes = self._get_hintmask(index)
        self.pen.add_hintmask("cntrmask", [hintMaskBytes])
        return hintMaskBytes, index


class CFF2CharStringMergePen(T2CharStringPen):
    """Pen to merge Type 2 CharStrings."""

    def __init__(
        self, default_commands, glyphName, num_masters, master_idx, roundTolerance=0.01
    ):
        # For roundTolerance see https://github.com/fonttools/fonttools/issues/2838
        super().__init__(
            width=None, glyphSet=None, CFF2=True, roundTolerance=roundTolerance
        )
        self.pt_index = 0
        self._commands = default_commands
        self.m_index = master_idx
        self.num_masters = num_masters
        self.prev_move_idx = 0
        self.seen_moveto = False
        self.glyphName = glyphName
        self.round = roundFunc(roundTolerance, round=round)

    def add_point(self, point_type, pt_coords):
        if self.m_index == 0:
            self._commands.append([point_type, [pt_coords]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != point_type:
                raise VarLibCFFPointTypeMergeError(
                    point_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName
                )
            cmd[1].append(pt_coords)
        self.pt_index += 1

    def add_hint(self, hint_type, args):
        if self.m_index == 0:
            self._commands.append([hint_type, [args]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != hint_type:
                raise VarLibCFFHintTypeMergeError(
                    hint_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName
                )
            cmd[1].append(args)
        self.pt_index += 1

    def add_hintmask(self, hint_type, abs_args):
        # For hintmask, fonttools.cffLib.specializer.py expects
        # each of these to be represented by two sequential commands:
        # first holding only the operator name, with an empty arg list,
        # second with an empty string as the op name, and the mask arg list.
        if self.m_index == 0:
            self._commands.append([hint_type, []])
            self._commands.append(["", [abs_args]])
        else:
            cmd = self._commands[self.pt_index]
            if cmd[0] != hint_type:
                raise VarLibCFFHintTypeMergeError(
                    hint_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName
                )
            self.pt_index += 1
            cmd = self._commands[self.pt_index]
            cmd[1].append(abs_args)
        self.pt_index += 1

    def _moveTo(self, pt):
        if not self.seen_moveto:
            self.seen_moveto = True
        pt_coords = self._p(pt)
        self.add_point("rmoveto", pt_coords)
        # I set prev_move_idx here because add_point()
        # can change self.pt_index.
        self.prev_move_idx = self.pt_index - 1

    def _lineTo(self, pt):
        pt_coords = self._p(pt)
        self.add_point("rlineto", pt_coords)

    def _curveToOne(self, pt1, pt2, pt3):
        _p = self._p
        pt_coords = _p(pt1) + _p(pt2) + _p(pt3)
        self.add_point("rrcurveto", pt_coords)

    def _closePath(self):
        pass

    def _endPath(self):
        pass

    def restart(self, region_idx):
        self.pt_index = 0
        self.m_index = region_idx
        self._p0 = (0, 0)

    def getCommands(self):
        return self._commands

    def reorder_blend_args(self, commands, get_delta_func):
        """
        We first re-order the master coordinate values.
        For a moveto to lineto, the args are now arranged as::

                [ [master_0 x,y], [master_1 x,y], [master_2 x,y] ]

        We re-arrange this to::

                [	[master_0 x, master_1 x, master_2 x],
                        [master_0 y, master_1 y, master_2 y]
                ]

        If the master values are all the same, we collapse the list to
        as single value instead of a list.

        We then convert this to::

                [ [master_0 x] + [x delta tuple] + [numBlends=1]
                  [master_0 y] + [y delta tuple] + [numBlends=1]
                ]
        """
        for cmd in commands:
            # arg[i] is the set of arguments for this operator from master i.
            args = cmd[1]
            m_args = zip(*args)
            # m_args[n] is now all num_master args for the i'th argument
            # for this operation.
            cmd[1] = list(m_args)
        lastOp = None
        for cmd in commands:
            op = cmd[0]
            # masks are represented by two cmd's: first has only op names,
            # second has only args.
            if lastOp in ["hintmask", "cntrmask"]:
                coord = list(cmd[1])
                if not allEqual(coord):
                    raise VarLibMergeError(
                        "Hintmask values cannot differ between source fonts."
                    )
                cmd[1] = [coord[0][0]]
            else:
                coords = cmd[1]
                new_coords = []
                for coord in coords:
                    if allEqual(coord):
                        new_coords.append(coord[0])
                    else:
                        # convert to deltas
                        deltas = get_delta_func(coord)[1:]
                        coord = [coord[0]] + deltas
                        coord.append(1)
                        new_coords.append(coord)
                cmd[1] = new_coords
            lastOp = op
        return commands

    def getCharString(
        self, private=None, globalSubrs=None, var_model=None, optimize=True
    ):
        commands = self._commands
        commands = self.reorder_blend_args(
            commands, partial(var_model.getDeltas, round=self.round)
        )
        if optimize:
            commands = specializeCommands(
                commands, generalizeFirst=False, maxstack=maxStackLimit
            )
        program = commandsToProgram(commands)
        charString = T2CharString(
            program=program, private=private, globalSubrs=globalSubrs
        )
        return charString
<|endoftext|>from .interpolatableHelpers import *
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.recordingPen import (
    RecordingPen,
    DecomposingRecordingPen,
    RecordingPointPen,
)
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.pens.pointPen import (
    SegmentToPointPen,
    PointToSegmentPen,
    ReverseContourPointPen,
)
from itertools import cycle
from io import BytesIO
import cairo
import math
import os
import logging

log = logging.getLogger("fontTools.varLib.interpolatable")


class OverridingDict(dict):
    def __init__(self, parent_dict):
        self.parent_dict = parent_dict

    def __missing__(self, key):
        return self.parent_dict[key]


class InterpolatablePlot:
    width = 8.5 * 72
    height = 11 * 72
    pad = 0.1 * 72
    title_font_size = 24
    font_size = 16
    page_number = 1
    head_color = (0.3, 0.3, 0.3)
    label_color = (0.2, 0.2, 0.2)
    border_color = (0.9, 0.9, 0.9)
    border_width = 0.5
    fill_color = (0.8, 0.8, 0.8)
    stroke_color = (0.1, 0.1, 0.1)
    stroke_width = 1
    oncurve_node_color = (0, 0.8, 0, 0.7)
    oncurve_node_diameter = 6
    offcurve_node_color = (0, 0.5, 0, 0.7)
    offcurve_node_diameter = 4
    handle_color = (0, 0.5, 0, 0.7)
    handle_width = 0.5
    corrected_start_point_color = (0, 0.9, 0, 0.7)
    corrected_start_point_size = 7
    wrong_start_point_color = (1, 0, 0, 0.7)
    start_point_color = (0, 0, 1, 0.7)
    start_arrow_length = 9
    kink_point_size = 7
    kink_point_color = (1, 0, 1, 0.7)
    kink_circle_size = 15
    kink_circle_stroke_width = 1
    kink_circle_color = (1, 0, 1, 0.7)
    contour_colors = ((1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    contour_alpha = 0.5
    weight_issue_contour_color = (0, 0, 0, 0.4)
    no_issues_label = "Your font's good! Have a cupcake..."
    no_issues_label_color = (0, 0.5, 0)
    cupcake_color = (0.3, 0, 0.3)
    cupcake = r"""
                          ,@.
                        ,@.@@,.
                  ,@@,.@@@.  @.@@@,.
                ,@@. @@@.     @@. @@,.
        ,@@@.@,.@.              @.  @@@@,.@.@@,.
   ,@@.@.     @@.@@.            @,.    .@' @'  @@,
 ,@@. @.          .@@.@@@.  @@'                  @,
,@.  @@.                                          @,
@.     @,@@,.     ,                             .@@,
@,.       .@,@@,.         .@@,.  ,       .@@,  @, @,
@.                             .@. @ @@,.    ,      @
 @,.@@.     @,.      @@,.      @.           @,.    @'
  @@||@,.  @'@,.       @@,.  @@ @,.        @'@@,  @'
     \\@@@@'  @,.      @'@@@@'   @@,.   @@@' //@@@'
      |||||||| @@,.  @@' |||||||  |@@@|@||  ||
       \\\\\\\  ||@@@||  |||||||  |||||||  //
        |||||||  ||||||  ||||||   ||||||  ||
         \\\\\\  ||||||  ||||||  ||||||  //
          ||||||  |||||  |||||   |||||  ||
           \\\\\  |||||  |||||  |||||  //
            |||||  ||||  |||||  ||||  ||
             \\\\  ||||  ||||  ||||  //
              ||||||||||||||||||||||||
"""
    emoticon_color = (0, 0.3, 0.3)
    shrug = r"""\_(")_/"""
    underweight = r"""
 o
/|\
/ \
"""
    overweight = r"""
 o
/O\
/ \
"""
    yay = r""" \o/ """

    def __init__(self, out, glyphsets, names=None, **kwargs):
        self.out = out
        self.glyphsets = glyphsets
        self.names = names or [repr(g) for g in glyphsets]
        self.toc = {}

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise TypeError("Unknown keyword argument: %s" % k)
            setattr(self, k, v)

        self.panel_width = self.width / 2 - self.pad * 3
        self.panel_height = (
            self.height / 2 - self.pad * 6 - self.font_size * 2 - self.title_font_size
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def show_page(self):
        self.page_number += 1

    def add_title_page(
        self, files, *, show_tolerance=True, tolerance=None, kinkiness=None
    ):
        pad = self.pad
        width = self.width - 3 * self.pad
        x = y = pad

        self.draw_label(
            "Problem report for:",
            x=x,
            y=y,
            bold=True,
            width=width,
            font_size=self.title_font_size,
        )
        y += self.title_font_size

        import hashlib

        for file in files:
            base_file = os.path.basename(file)
            y += self.font_size + self.pad
            self.draw_label(base_file, x=x, y=y, bold=True, width=width)
            y += self.font_size + self.pad

            try:
                with open(file, "rb") as f:
                    h = hashlib.sha1(f.read()).hexdigest()
                self.draw_label("sha1: %s" % h, x=x + pad, y=y, width=width)
                y += self.font_size
            except IsADirectoryError:
                pass

            if file.endswith(".ttf"):
                ttFont = TTFont(file)
                name = ttFont["name"] if "name" in ttFont else None
                if name:
                    for what, nameIDs in (
                        ("Family name", (21, 16, 1)),
                        ("Version", (5,)),
                    ):
                        n = name.getFirstDebugName(nameIDs)
                        if n is None:
                            continue
                        self.draw_label(
                            "%s: %s" % (what, n), x=x + pad, y=y, width=width
                        )
                        y += self.font_size + self.pad
            elif file.endswith((".glyphs", ".glyphspackage")):
                from glyphsLib import GSFont

                f = GSFont(file)
                for what, field in (
                    ("Family name", "familyName"),
                    ("VersionMajor", "versionMajor"),
                    ("VersionMinor", "_versionMinor"),
                ):
                    self.draw_label(
                        "%s: %s" % (what, getattr(f, field)),
                        x=x + pad,
                        y=y,
                        width=width,
                    )
                    y += self.font_size + self.pad

        self.draw_legend(
            show_tolerance=show_tolerance, tolerance=tolerance, kinkiness=kinkiness
        )
        self.show_page()

    def draw_legend(self, *, show_tolerance=True, tolerance=None, kinkiness=None):
        cr = cairo.Context(self.surface)

        x = self.pad
        y = self.height - self.pad - self.font_size * 2
        width = self.width - 2 * self.pad

        xx = x + self.pad * 2
        xxx = x + self.pad * 4

        if show_tolerance:
            self.draw_label(
                "Tolerance: badness; closer to zero the worse", x=xxx, y=y, width=width
            )
            y -= self.pad + self.font_size

        self.draw_label("Underweight contours", x=xxx, y=y, width=width)
        cr.rectangle(xx - self.pad * 0.7, y, 1.5 * self.pad, self.font_size)
        cr.set_source_rgb(*self.fill_color)
        cr.fill_preserve()
        if self.stroke_color:
            cr.set_source_rgb(*self.stroke_color)
            cr.set_line_width(self.stroke_width)
            cr.stroke_preserve()
        cr.set_source_rgba(*self.weight_issue_contour_color)
        cr.fill()
        y -= self.pad + self.font_size

        self.draw_label(
            "Colored contours: contours with the wrong order", x=xxx, y=y, width=width
        )
        cr.rectangle(xx - self.pad * 0.7, y, 1.5 * self.pad, self.font_size)
        if self.fill_color:
            cr.set_source_rgb(*self.fill_color)
            cr.fill_preserve()
        if self.stroke_color:
            cr.set_source_rgb(*self.stroke_color)
            cr.set_line_width(self.stroke_width)
            cr.stroke_preserve()
        cr.set_source_rgba(*self.contour_colors[0], self.contour_alpha)
        cr.fill()
        y -= self.pad + self.font_size

        self.draw_label("Kink artifact", x=xxx, y=y, width=width)
        self.draw_circle(
            cr,
            x=xx,
            y=y + self.font_size * 0.5,
            diameter=self.kink_circle_size,
            stroke_width=self.kink_circle_stroke_width,
            color=self.kink_circle_color,
        )
        y -= self.pad + self.font_size

        self.draw_label("Point causing kink in the contour", x=xxx, y=y, width=width)
        self.draw_dot(
            cr,
            x=xx,
            y=y + self.font_size * 0.5,
            diameter=self.kink_point_size,
            color=self.kink_point_color,
        )
        y -= self.pad + self.font_size

        self.draw_label("Suggested new contour start point", x=xxx, y=y, width=width)
        self.draw_dot(
            cr,
            x=xx,
            y=y + self.font_size * 0.5,
            diameter=self.corrected_start_point_size,
            color=self.corrected_start_point_color,
        )
        y -= self.pad + self.font_size

        self.draw_label(
            "Contour start point in contours with wrong direction",
            x=xxx,
            y=y,
            width=width,
        )
        self.draw_arrow(
            cr,
            x=xx - self.start_arrow_length * 0.3,
            y=y + self.font_size * 0.5,
            color=self.wrong_start_point_color,
        )
        y -= self.pad + self.font_size

        self.draw_label(
            "Contour start point when the first two points overlap",
            x=xxx,
            y=y,
            width=width,
        )
        self.draw_dot(
            cr,
            x=xx,
            y=y + self.font_size * 0.5,
            diameter=self.corrected_start_point_size,
            color=self.start_point_color,
        )
        y -= self.pad + self.font_size

        self.draw_label("Contour start point and direction", x=xxx, y=y, width=width)
        self.draw_arrow(
            cr,
            x=xx - self.start_arrow_length * 0.3,
            y=y + self.font_size * 0.5,
            color=self.start_point_color,
        )
        y -= self.pad + self.font_size

        self.draw_label("Legend:", x=x, y=y, width=width, bold=True)
        y -= self.pad + self.font_size

        if kinkiness is not None:
            self.draw_label(
                "Kink-reporting aggressiveness: %g" % kinkiness,
                x=xxx,
                y=y,
                width=width,
            )
            y -= self.pad + self.font_size

        if tolerance is not None:
            self.draw_label(
                "Error tolerance: %g" % tolerance,
                x=xxx,
                y=y,
                width=width,
            )
            y -= self.pad + self.font_size

        self.draw_label("Parameters:", x=x, y=y, width=width, bold=True)
        y -= self.pad + self.font_size

    def add_summary(self, problems):
        pad = self.pad
        width = self.width - 3 * self.pad
        height = self.height - 2 * self.pad
        x = y = pad

        self.draw_label(
            "Summary of problems",
            x=x,
            y=y,
            bold=True,
            width=width,
            font_size=self.title_font_size,
        )
        y += self.title_font_size

        glyphs_per_problem = defaultdict(set)
        for glyphname, glyph_problems in sorted(problems.items()):
            for problem in glyph_problems:
                glyphs_per_problem[problem["type"]].add(glyphname)

        if InterpolatableProblem.NOTHING in glyphs_per_problem:
            del glyphs_per_problem[InterpolatableProblem.NOTHING]

        for problem_type in sorted(
            glyphs_per_problem, key=lambda x: InterpolatableProblem.severity[x]
        ):
            y += self.font_size
            self.draw_label(
                "%s: %d" % (problem_type, len(glyphs_per_problem[problem_type])),
                x=x,
                y=y,
                width=width,
                bold=True,
            )
            y += self.font_size

            for glyphname in sorted(glyphs_per_problem[problem_type]):
                if y + self.font_size > height:
                    self.show_page()
                    y = self.font_size + pad
                self.draw_label(glyphname, x=x + 2 * pad, y=y, width=width - 2 * pad)
                y += self.font_size

        self.show_page()

    def _add_listing(self, title, items):
        pad = self.pad
        width = self.width - 2 * self.pad
        height = self.height - 2 * self.pad
        x = y = pad

        self.draw_label(
            title, x=x, y=y, bold=True, width=width, font_size=self.title_font_size
        )
        y += self.title_font_size + self.pad

        last_glyphname = None
        for page_no, (glyphname, problems) in items:
            if glyphname == last_glyphname:
                continue
            last_glyphname = glyphname
            if y + self.font_size > height:
                self.show_page()
                y = self.font_size + pad
            self.draw_label(glyphname, x=x + 5 * pad, y=y, width=width - 2 * pad)
            self.draw_label(str(page_no), x=x, y=y, width=4 * pad, align=1)
            y += self.font_size

        self.show_page()

    def add_table_of_contents(self):
        self._add_listing("Table of contents", sorted(self.toc.items()))

    def add_index(self):
        self._add_listing("Index", sorted(self.toc.items(), key=lambda x: x[1][0]))

    def add_problems(self, problems, *, show_tolerance=True, show_page_number=True):
        for glyph, glyph_problems in problems.items():
            last_masters = None
            current_glyph_problems = []
            for p in glyph_problems:
                masters = (
                    p["master_idx"]
                    if "master_idx" in p
                    else (p["master_1_idx"], p["master_2_idx"])
                )
                if masters == last_masters:
                    current_glyph_problems.append(p)
                    continue
                # Flush
                if current_glyph_problems:
                    self.add_problem(
                        glyph,
                        current_glyph_problems,
                        show_tolerance=show_tolerance,
                        show_page_number=show_page_number,
                    )
                    self.show_page()
                    current_glyph_problems = []
                last_masters = masters
                current_glyph_problems.append(p)
            if current_glyph_problems:
                self.add_problem(
                    glyph,
                    current_glyph_problems,
                    show_tolerance=show_tolerance,
                    show_page_number=show_page_number,
                )
                self.show_page()

    def add_problem(
        self, glyphname, problems, *, show_tolerance=True, show_page_number=True
    ):
        if type(problems) not in (list, tuple):
            problems = [problems]

        self.toc[self.page_number] = (glyphname, problems)

        problem_type = problems[0]["type"]
        problem_types = set(problem["type"] for problem in problems)
        if not all(pt == problem_type for pt in problem_types):
            problem_type = ", ".join(sorted({problem["type"] for problem in problems}))

        log.info("Drawing %s: %s", glyphname, problem_type)

        master_keys = (
            ("master_idx",)
            if "master_idx" in problems[0]
            else ("master_1_idx", "master_2_idx")
        )
        master_indices = [problems[0][k] for k in master_keys]

        if problem_type == InterpolatableProblem.MISSING:
            sample_glyph = next(
                i for i, m in enumerate(self.glyphsets) if m[glyphname] is not None
            )
            master_indices.insert(0, sample_glyph)

        x = self.pad
        y = self.pad

        self.draw_label(
            "Glyph name: " + glyphname,
            x=x,
            y=y,
            color=self.head_color,
            align=0,
            bold=True,
            font_size=self.title_font_size,
        )
        tolerance = min(p.get("tolerance", 1) for p in problems)
        if tolerance < 1 and show_tolerance:
            self.draw_label(
                "tolerance: %.2f" % tolerance,
                x=x,
                y=y,
                width=self.width - 2 * self.pad,
                align=1,
                bold=True,
            )
        y += self.title_font_size + self.pad
        self.draw_label(
            "Problems: " + problem_type,
            x=x,
            y=y,
            width=self.width - 2 * self.pad,
            color=self.head_color,
            bold=True,
        )
        y += self.font_size + self.pad * 2

        scales = []
        for which, master_idx in enumerate(master_indices):
            glyphset = self.glyphsets[master_idx]
            name = self.names[master_idx]

            self.draw_label(
                name,
                x=x,
                y=y,
                color=self.label_color,
                width=self.panel_width,
                align=0.5,
            )
            y += self.font_size + self.pad

            if glyphset[glyphname] is not None:
                scales.append(
                    self.draw_glyph(glyphset, glyphname, problems, which, x=x, y=y)
                )
            else:
                self.draw_emoticon(self.shrug, x=x, y=y)
            y += self.panel_height + self.font_size + self.pad

        if any(
            pt
            in (
                InterpolatableProblem.NOTHING,
                InterpolatableProblem.WRONG_START_POINT,
                InterpolatableProblem.CONTOUR_ORDER,
                InterpolatableProblem.KINK,
                InterpolatableProblem.UNDERWEIGHT,
                InterpolatableProblem.OVERWEIGHT,
            )
            for pt in problem_types
        ):
            x = self.pad + self.panel_width + self.pad
            y = self.pad
            y += self.title_font_size + self.pad * 2
            y += self.font_size + self.pad

            glyphset1 = self.glyphsets[master_indices[0]]
            glyphset2 = self.glyphsets[master_indices[1]]

            # Draw the mid-way of the two masters

            self.draw_label(
                "midway interpolation",
                x=x,
                y=y,
                color=self.head_color,
                width=self.panel_width,
                align=0.5,
            )
            y += self.font_size + self.pad

            midway_glyphset = LerpGlyphSet(glyphset1, glyphset2)
            self.draw_glyph(
                midway_glyphset,
                glyphname,
                [{"type": "midway"}]
                + [
                    p
                    for p in problems
                    if p["type"]
                    in (
                        InterpolatableProblem.KINK,
                        InterpolatableProblem.UNDERWEIGHT,
                        InterpolatableProblem.OVERWEIGHT,
                    )
                ],
                None,
                x=x,
                y=y,
                scale=min(scales),
            )

            y += self.panel_height + self.font_size + self.pad

        if any(
            pt
            in (
                InterpolatableProblem.WRONG_START_POINT,
                InterpolatableProblem.CONTOUR_ORDER,
                InterpolatableProblem.KINK,
            )
            for pt in problem_types
        ):
            # Draw the proposed fix

            self.draw_label(
                "proposed fix",
                x=x,
                y=y,
                color=self.head_color,
                width=self.panel_width,
                align=0.5,
            )
            y += self.font_size + self.pad

            overriding1 = OverridingDict(glyphset1)
            overriding2 = OverridingDict(glyphset2)
            perContourPen1 = PerContourOrComponentPen(
                RecordingPen, glyphset=overriding1
            )
            perContourPen2 = PerContourOrComponentPen(
                RecordingPen, glyphset=overriding2
            )
            glyphset1[glyphname].draw(perContourPen1)
            glyphset2[glyphname].draw(perContourPen2)

            for problem in problems:
                if problem["type"] == InterpolatableProblem.CONTOUR_ORDER:
                    fixed_contours = [
                        perContourPen2.value[i] for i in problems[0]["value_2"]
                    ]
                    perContourPen2.value = fixed_contours

            for problem in problems:
                if problem["type"] == InterpolatableProblem.WRONG_START_POINT:
                    # Save the wrong contours
                    wrongContour1 = perContourPen1.value[problem["contour"]]
                    wrongContour2 = perContourPen2.value[problem["contour"]]

                    # Convert the wrong contours to point pens
                    points1 = RecordingPointPen()
                    converter = SegmentToPointPen(points1, False)
                    wrongContour1.replay(converter)
                    points2 = RecordingPointPen()
                    converter = SegmentToPointPen(points2, False)
                    wrongContour2.replay(converter)

                    proposed_start = problem["value_2"]

                    # See if we need reversing; fragile but worth a try
                    if problem["reversed"]:
                        new_points2 = RecordingPointPen()
                        reversedPen = ReverseContourPointPen(new_points2)
                        points2.replay(reversedPen)
                        points2 = new_points2
                        proposed_start = len(points2.value) - 2 - proposed_start

                    # Rotate points2 so that the first point is the same as in points1
                    beginPath = points2.value[:1]
                    endPath = points2.value[-1:]
                    pts = points2.value[1:-1]
                    pts = pts[proposed_start:] + pts[:proposed_start]
                    points2.value = beginPath + pts + endPath

                    # Convert the point pens back to segment pens
                    segment1 = RecordingPen()
                    converter = PointToSegmentPen(segment1, True)
                    points1.replay(converter)
                    segment2 = RecordingPen()
                    converter = PointToSegmentPen(segment2, True)
                    points2.replay(converter)

                    # Replace the wrong contours
                    wrongContour1.value = segment1.value
                    wrongContour2.value = segment2.value
                    perContourPen1.value[problem["contour"]] = wrongContour1
                    perContourPen2.value[problem["contour"]] = wrongContour2

            for problem in problems:
                # If we have a kink, try to fix it.
                if problem["type"] == InterpolatableProblem.KINK:
                    # Save the wrong contours
                    wrongContour1 = perContourPen1.value[problem["contour"]]
                    wrongContour2 = perContourPen2.value[problem["contour"]]

                    # Convert the wrong contours to point pens
                    points1 = RecordingPointPen()
                    converter = SegmentToPointPen(points1, False)
                    wrongContour1.replay(converter)
                    points2 = RecordingPointPen()
                    converter = SegmentToPointPen(points2, False)
                    wrongContour2.replay(converter)

                    i = problem["value"]

                    # Position points to be around the same ratio
                    # beginPath / endPath dance
                    j = i + 1
                    pt0 = points1.value[j][1][0]
                    pt1 = points2.value[j][1][0]
                    j_prev = (i - 1) % (len(points1.value) - 2) + 1
                    pt0_prev = points1.value[j_prev][1][0]
                    pt1_prev = points2.value[j_prev][1][0]
                    j_next = (i + 1) % (len(points1.value) - 2) + 1
                    pt0_next = points1.value[j_next][1][0]
                    pt1_next = points2.value[j_next][1][0]

                    pt0 = complex(*pt0)
                    pt1 = complex(*pt1)
                    pt0_prev = complex(*pt0_prev)
                    pt1_prev = complex(*pt1_prev)
                    pt0_next = complex(*pt0_next)
                    pt1_next = complex(*pt1_next)

                    # Find the ratio of the distance between the points
                    r0 = abs(pt0 - pt0_prev) / abs(pt0_next - pt0_prev)
                    r1 = abs(pt1 - pt1_prev) / abs(pt1_next - pt1_prev)
                    r_mid = (r0 + r1) / 2

                    pt0 = pt0_prev + r_mid * (pt0_next - pt0_prev)
                    pt1 = pt1_prev + r_mid * (pt1_next - pt1_prev)

                    points1.value[j] = (
                        points1.value[j][0],
                        (((pt0.real, pt0.imag),) + points1.value[j][1][1:]),
                        points1.value[j][2],
                    )
                    points2.value[j] = (
                        points2.value[j][0],
                        (((pt1.real, pt1.imag),) + points2.value[j][1][1:]),
                        points2.value[j][2],
                    )

                    # Convert the point pens back to segment pens
                    segment1 = RecordingPen()
                    converter = PointToSegmentPen(segment1, True)
                    points1.replay(converter)
                    segment2 = RecordingPen()
                    converter = PointToSegmentPen(segment2, True)
                    points2.replay(converter)

                    # Replace the wrong contours
                    wrongContour1.value = segment1.value
                    wrongContour2.value = segment2.value

            # Assemble
            fixed1 = RecordingPen()
            fixed2 = RecordingPen()
            for contour in perContourPen1.value:
                fixed1.value.extend(contour.value)
            for contour in perContourPen2.value:
                fixed2.value.extend(contour.value)
            fixed1.draw = fixed1.replay
            fixed2.draw = fixed2.replay

            overriding1[glyphname] = fixed1
            overriding2[glyphname] = fixed2

            try:
                midway_glyphset = LerpGlyphSet(overriding1, overriding2)
                self.draw_glyph(
                    midway_glyphset,
                    glyphname,
                    {"type": "fixed"},
                    None,
                    x=x,
                    y=y,
                    scale=min(scales),
                )
            except ValueError:
                self.draw_emoticon(self.shrug, x=x, y=y)
            y += self.panel_height + self.pad

        else:
            emoticon = self.shrug
            if InterpolatableProblem.UNDERWEIGHT in problem_types:
                emoticon = self.underweight
            elif InterpolatableProblem.OVERWEIGHT in problem_types:
                emoticon = self.overweight
            elif InterpolatableProblem.NOTHING in problem_types:
                emoticon = self.yay
            self.draw_emoticon(emoticon, x=x, y=y)

        if show_page_number:
            self.draw_label(
                str(self.page_number),
                x=0,
                y=self.height - self.font_size - self.pad,
                width=self.width,
                color=self.head_color,
                align=0.5,
            )

    def draw_label(
        self,
        label,
        *,
        x=0,
        y=0,
        color=(0, 0, 0),
        align=0,
        bold=False,
        width=None,
        height=None,
        font_size=None,
    ):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if font_size is None:
            font_size = self.font_size
        cr = cairo.Context(self.surface)
        cr.select_font_face(
            "@cairo:",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
        )
        cr.set_font_size(font_size)
        font_extents = cr.font_extents()
        font_size = font_size * font_size / font_extents[2]
        cr.set_font_size(font_size)
        font_extents = cr.font_extents()

        cr.set_source_rgb(*color)

        extents = cr.text_extents(label)
        if extents.width > width:
            # Shrink
            font_size *= width / extents.width
            cr.set_font_size(font_size)
            font_extents = cr.font_extents()
            extents = cr.text_extents(label)

        # Center
        label_x = x + (width - extents.width) * align
        label_y = y + font_extents[0]
        cr.move_to(label_x, label_y)
        cr.show_text(label)

    def draw_glyph(self, glyphset, glyphname, problems, which, *, x=0, y=0, scale=None):
        if type(problems) not in (list, tuple):
            problems = [problems]

        midway = any(problem["type"] == "midway" for problem in problems)
        problem_type = problems[0]["type"]
        problem_types = set(problem["type"] for problem in problems)
        if not all(pt == problem_type for pt in problem_types):
            problem_type = "mixed"
        glyph = glyphset[glyphname]

        recording = RecordingPen()
        glyph.draw(recording)
        decomposedRecording = DecomposingRecordingPen(glyphset)
        glyph.draw(decomposedRecording)

        boundsPen = ControlBoundsPen(glyphset)
        decomposedRecording.replay(boundsPen)
        bounds = boundsPen.bounds
        if bounds is None:
            bounds = (0, 0, 0, 0)

        glyph_width = bounds[2] - bounds[0]
        glyph_height = bounds[3] - bounds[1]

        if glyph_width:
            if scale is None:
                scale = self.panel_width / glyph_width
            else:
                scale = min(scale, self.panel_width / glyph_width)
        if glyph_height:
            if scale is None:
                scale = self.panel_height / glyph_height
            else:
                scale = min(scale, self.panel_height / glyph_height)
        if scale is None:
            scale = 1

        cr = cairo.Context(self.surface)
        cr.translate(x, y)
        # Center
        cr.translate(
            (self.panel_width - glyph_width * scale) / 2,
            (self.panel_height - glyph_height * scale) / 2,
        )
        cr.scale(scale, -scale)
        cr.translate(-bounds[0], -bounds[3])

        if self.border_color:
            cr.set_source_rgb(*self.border_color)
            cr.rectangle(bounds[0], bounds[1], glyph_width, glyph_height)
            cr.set_line_width(self.border_width / scale)
            cr.stroke()

        if self.fill_color or self.stroke_color:
            pen = CairoPen(glyphset, cr)
            decomposedRecording.replay(pen)

            if self.fill_color and problem_type != InterpolatableProblem.OPEN_PATH:
                cr.set_source_rgb(*self.fill_color)
                cr.fill_preserve()

            if self.stroke_color:
                cr.set_source_rgb(*self.stroke_color)
                cr.set_line_width(self.stroke_width / scale)
                cr.stroke_preserve()

            cr.new_path()

        if (
            InterpolatableProblem.UNDERWEIGHT in problem_types
            or InterpolatableProblem.OVERWEIGHT in problem_types
        ):
            perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
            recording.replay(perContourPen)
            for problem in problems:
                if problem["type"] in (
                    InterpolatableProblem.UNDERWEIGHT,
                    InterpolatableProblem.OVERWEIGHT,
                ):
                    contour = perContourPen.value[problem["contour"]]
                    contour.replay(CairoPen(glyphset, cr))
                    cr.set_source_rgba(*self.weight_issue_contour_color)
                    cr.fill()

        if any(
            t in problem_types
            for t in {
                InterpolatableProblem.NOTHING,
                InterpolatableProblem.NODE_COUNT,
                InterpolatableProblem.NODE_INCOMPATIBILITY,
            }
        ):
            cr.set_line_cap(cairo.LINE_CAP_ROUND)

            # Oncurve nodes
            for segment, args in decomposedRecording.value:
                if not args:
                    continue
                x, y = args[-1]
                cr.move_to(x, y)
                cr.line_to(x, y)
            cr.set_source_rgba(*self.oncurve_node_color)
            cr.set_line_width(self.oncurve_node_diameter / scale)
            cr.stroke()

            # Offcurve nodes
            for segment, args in decomposedRecording.value:
                if not args:
                    continue
                for x, y in args[:-1]:
                    cr.move_to(x, y)
                    cr.line_to(x, y)
            cr.set_source_rgba(*self.offcurve_node_color)
            cr.set_line_width(self.offcurve_node_diameter / scale)
            cr.stroke()

            # Handles
            for segment, args in decomposedRecording.value:
                if not args:
                    pass
                elif segment in ("moveTo", "lineTo"):
                    cr.move_to(*args[0])
                elif segment == "qCurveTo":
                    for x, y in args:
                        cr.line_to(x, y)
                    cr.new_sub_path()
                    cr.move_to(*args[-1])
                elif segment == "curveTo":
                    cr.line_to(*args[0])
                    cr.new_sub_path()
                    cr.move_to(*args[1])
                    cr.line_to(*args[2])
                    cr.new_sub_path()
                    cr.move_to(*args[-1])
                else:
                    continue

            cr.set_source_rgba(*self.handle_color)
            cr.set_line_width(self.handle_width / scale)
            cr.stroke()

        matching = None
        for problem in problems:
            if problem["type"] == InterpolatableProblem.CONTOUR_ORDER:
                matching = problem["value_2"]
                colors = cycle(self.contour_colors)
                perContourPen = PerContourOrComponentPen(
                    RecordingPen, glyphset=glyphset
                )
                recording.replay(perContourPen)
                for i, contour in enumerate(perContourPen.value):
                    if matching[i] == i:
                        continue
                    color = next(colors)
                    contour.replay(CairoPen(glyphset, cr))
                    cr.set_source_rgba(*color, self.contour_alpha)
                    cr.fill()

        for problem in problems:
            if problem["type"] in (
                InterpolatableProblem.NOTHING,
                InterpolatableProblem.WRONG_START_POINT,
            ):
                idx = problem.get("contour")

                # Draw suggested point
                if idx is not None and which == 1 and "value_2" in problem:
                    perContourPen = PerContourOrComponentPen(
                        RecordingPen, glyphset=glyphset
                    )
                    decomposedRecording.replay(perContourPen)
                    points = SimpleRecordingPointPen()
                    converter = SegmentToPointPen(points, False)
                    perContourPen.value[
                        idx if matching is None else matching[idx]
                    ].replay(converter)
                    targetPoint = points.value[problem["value_2"]][0]
                    cr.save()
                    cr.translate(*targetPoint)
                    cr.scale(1 / scale, 1 / scale)
                    self.draw_dot(
                        cr,
                        diameter=self.corrected_start_point_size,
                        color=self.corrected_start_point_color,
                    )
                    cr.restore()

                # Draw start-point arrow
                if which == 0 or not problem.get("reversed"):
                    color = self.start_point_color
                else:
                    color = self.wrong_start_point_color
                first_pt = None
                i = 0
                cr.save()
                for segment, args in decomposedRecording.value:
                    if segment == "moveTo":
                        first_pt = args[0]
                        continue
                    if first_pt is None:
                        continue
                    if segment == "closePath":
                        second_pt = first_pt
                    else:
                        second_pt = args[0]

                    if idx is None or i == idx:
                        cr.save()
                        first_pt = complex(*first_pt)
                        second_pt = complex(*second_pt)
                        length = abs(second_pt - first_pt)
                        cr.translate(first_pt.real, first_pt.imag)
                        if length:
                            # Draw arrowhead
                            cr.rotate(
                                math.atan2(
                                    second_pt.imag - first_pt.imag,
                                    second_pt.real - first_pt.real,
                                )
                            )
                            cr.scale(1 / scale, 1 / scale)
                            self.draw_arrow(cr, color=color)
                        else:
                            # Draw circle
                            cr.scale(1 / scale, 1 / scale)
                            self.draw_dot(
                                cr,
                                diameter=self.corrected_start_point_size,
                                color=color,
                            )
                        cr.restore()

                        if idx is not None:
                            break

                    first_pt = None
                    i += 1

                cr.restore()

            if problem["type"] == InterpolatableProblem.KINK:
                idx = problem.get("contour")
                perContourPen = PerContourOrComponentPen(
                    RecordingPen, glyphset=glyphset
                )
                decomposedRecording.replay(perContourPen)
                points = SimpleRecordingPointPen()
                converter = SegmentToPointPen(points, False)
                perContourPen.value[idx if matching is None else matching[idx]].replay(
                    converter
                )

                targetPoint = points.value[problem["value"]][0]
                cr.save()
                cr.translate(*targetPoint)
                cr.scale(1 / scale, 1 / scale)
                if midway:
                    self.draw_circle(
                        cr,
                        diameter=self.kink_circle_size,
                        stroke_width=self.kink_circle_stroke_width,
                        color=self.kink_circle_color,
                    )
                else:
                    self.draw_dot(
                        cr,
                        diameter=self.kink_point_size,
                        color=self.kink_point_color,
                    )
                cr.restore()

        return scale

    def draw_dot(self, cr, *, x=0, y=0, color=(0, 0, 0), diameter=10):
        cr.save()
        cr.set_line_width(diameter)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.move_to(x, y)
        cr.line_to(x, y)
        if len(color) == 3:
            color = color + (1,)
        cr.set_source_rgba(*color)
        cr.stroke()
        cr.restore()

    def draw_circle(
        self, cr, *, x=0, y=0, color=(0, 0, 0), diameter=10, stroke_width=1
    ):
        cr.save()
        cr.set_line_width(stroke_width)
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.arc(x, y, diameter / 2, 0, 2 * math.pi)
        if len(color) == 3:
            color = color + (1,)
        cr.set_source_rgba(*color)
        cr.stroke()
        cr.restore()

    def draw_arrow(self, cr, *, x=0, y=0, color=(0, 0, 0)):
        cr.save()
