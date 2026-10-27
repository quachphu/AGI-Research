            if self._converter is not None:
                info = self._converter.axisinfo(self.units, self)
                if info.default_limits is not None:
                    ymin, ymax = self.convert_units(info.default_limits)
                    self.axes.viewLim.intervaly = ymin, ymax
        self.stale = True

    def get_tick_space(self):
        ends = mtransforms.Bbox.unit().transformed(
            self.axes.transAxes - self.get_figure(root=False).dpi_scale_trans)
        length = ends.height * 72
        # Having a spacing of at least 2 just looks good.
        size = self._get_tick_label_size('y') * 2
        if size > 0:
            return int(np.floor(length / size))
        else:
            return 2**31 - 1
<|endoftext|>"""
Adjust subplot layouts so that there are no overlapping Axes or Axes
decorations.  All Axes decorations are dealt with (labels, ticks, titles,
ticklabels) and some dependent artists are also dealt with (colorbar,
suptitle).

Layout is done via `~matplotlib.gridspec`, with one constraint per gridspec,
so it is possible to have overlapping Axes if the gridspecs overlap (i.e.
using `~matplotlib.gridspec.GridSpecFromSubplotSpec`).  Axes placed using
``figure.subplots()`` or ``figure.add_subplots()`` will participate in the
layout.  Axes manually placed via ``figure.add_axes()`` will not.

See Tutorial: :ref:`constrainedlayout_guide`

General idea:
-------------

First, a figure has a gridspec that divides the figure into nrows and ncols,
with heights and widths set by ``height_ratios`` and ``width_ratios``,
often just set to 1 for an equal grid.

Subplotspecs that are derived from this gridspec can contain either a
``SubPanel``, a ``GridSpecFromSubplotSpec``, or an ``Axes``.  The ``SubPanel``
and ``GridSpecFromSubplotSpec`` are dealt with recursively and each contain an
analogous layout.

Each ``GridSpec`` has a ``_layoutgrid`` attached to it.  The ``_layoutgrid``
has the same logical layout as the ``GridSpec``.   Each row of the grid spec
has a top and bottom "margin" and each column has a left and right "margin".
The "inner" height of each row is constrained to be the same (or as modified
by ``height_ratio``), and the "inner" width of each column is
constrained to be the same (as modified by ``width_ratio``), where "inner"
is the width or height of each column/row minus the size of the margins.

Then the size of the margins for each row and column are determined as the
max width of the decorators on each Axes that has decorators in that margin.
For instance, a normal Axes would have a left margin that includes the
left ticklabels, and the ylabel if it exists.  The right margin may include a
colorbar, the bottom margin the xaxis decorations, and the top margin the
title.

With these constraints, the solver then finds appropriate bounds for the
columns and rows.  It's possible that the margins take up the whole figure,
in which case the algorithm is not applied and a warning is raised.

See the tutorial :ref:`constrainedlayout_guide`
for more discussion of the algorithm with examples.
"""

import logging

import numpy as np

from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid


_log = logging.getLogger(__name__)


######################################################
def do_constrained_layout(fig, h_pad, w_pad,
                          hspace=None, wspace=None, rect=(0, 0, 1, 1),
                          compress=False):
    """
    Do the constrained_layout.  Called at draw time in
     ``figure.constrained_layout()``

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.

    h_pad, w_pad : float
      Padding around the Axes elements in figure-normalized units.

    hspace, wspace : float
       Fraction of the figure to dedicate to space between the
       Axes.  These are evenly spread between the gaps between the Axes.
       A value of 0.2 for a three-column layout would have a space
       of 0.1 of the figure width between each column.
       If h/wspace < h/w_pad, then the pads are used instead.

    rect : tuple of 4 floats
        Rectangle in figure coordinates to perform constrained layout in
        [left, bottom, width, height], each from 0-1.

    compress : bool
        Whether to shift Axes so that white space in between them is
        removed. This is useful for simple grids of fixed-aspect Axes (e.g.
        a grid of images).

    Returns
    -------
    layoutgrid : private debugging structure
    """

    renderer = fig._get_renderer()
    # make layoutgrid tree...
    layoutgrids = make_layoutgrids(fig, None, rect=rect)
    if not layoutgrids['hasgrids']:
        _api.warn_external('There are no gridspecs with layoutgrids. '
                           'Possibly did not call parent GridSpec with the'
                           ' "figure" keyword')
        return

    for _ in range(2):
        # do the algorithm twice.  This has to be done because decorations
        # change size after the first re-position (i.e. x/yticklabels get
        # larger/smaller).  This second reposition tends to be much milder,
        # so doing twice makes things work OK.

        # make margins for all the Axes and subfigures in the
        # figure.  Add margins for colorbars...
        make_layout_margins(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
        make_margin_suptitles(layoutgrids, fig, renderer, h_pad=h_pad,
                              w_pad=w_pad)

        # if a layout is such that a columns (or rows) margin has no
        # constraints, we need to make all such instances in the grid
        # match in margin size.
        match_submerged_margins(layoutgrids, fig)

        # update all the variables in the layout.
        layoutgrids[fig].update_variables()

        warn_collapsed = ('constrained_layout not applied because '
                          'axes sizes collapsed to zero.  Try making '
                          'figure larger or Axes decorations smaller.')
        if check_no_collapsed_axes(layoutgrids, fig):
            reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
            if compress:
                layoutgrids = compress_fixed_aspect(layoutgrids, fig)
                layoutgrids[fig].update_variables()
                if check_no_collapsed_axes(layoutgrids, fig):
                    reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                                    w_pad=w_pad, hspace=hspace, wspace=wspace)
                else:
                    _api.warn_external(warn_collapsed)

                if ((suptitle := fig._suptitle) is not None and
                        suptitle.get_in_layout() and suptitle._autopos):
                    x, _ = suptitle.get_position()
                    suptitle.set_position(
                        (x, layoutgrids[fig].get_inner_bbox().y1 + h_pad))
                    suptitle.set_verticalalignment('bottom')
        else:
            _api.warn_external(warn_collapsed)
        reset_margins(layoutgrids, fig)
    return layoutgrids


def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to Axes get a layoutgrid so Axes
    can have margins.
    """

    if layoutgrids is None:
        layoutgrids = dict()
        layoutgrids['hasgrids'] = False
    if not hasattr(fig, '_parent'):
        # top figure;  pass rect as parent to allow user-specified
        # margins
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=rect, name='figlb')
    else:
        # subfigure
        gs = fig._subplotspec.get_gridspec()
        # it is possible the gridspec containing this subfigure hasn't
        # been added to the tree yet:
        layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
        # add the layoutgrid for the subfigure:
        parentlb = layoutgrids[gs]
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(
            parent=parentlb,
            name='panellb',
            parent_inner=True,
            nrows=1, ncols=1,
            parent_pos=(fig._subplotspec.rowspan,
                        fig._subplotspec.colspan))
    # recursively do all subfigures in this figure...
    for sfig in fig.subfigs:
        layoutgrids = make_layoutgrids(sfig, layoutgrids)

    # for each Axes at the local level add its gridspec:
    for ax in fig._localaxes:
        gs = ax.get_gridspec()
        if gs is not None:
            layoutgrids = make_layoutgrids_gs(layoutgrids, gs)

    return layoutgrids


def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """

    if gs in layoutgrids or gs.figure is None:
        return layoutgrids
    # in order to do constrained_layout there has to be at least *one*
    # gridspec in the tree:
    layoutgrids['hasgrids'] = True
    if not hasattr(gs, '_subplot_spec'):
        # normal gridspec
        parent = layoutgrids[gs.figure]
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=parent,
                parent_inner=True,
                name='gridspec',
                ncols=gs._ncols, nrows=gs._nrows,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    else:
        # this is a gridspecfromsubplotspec:
        subplot_spec = gs._subplot_spec
        parentgs = subplot_spec.get_gridspec()
        # if a nested gridspec it is possible the parent is not in there yet:
        if parentgs not in layoutgrids:
            layoutgrids = make_layoutgrids_gs(layoutgrids, parentgs)
        subspeclb = layoutgrids[parentgs]
        # gridspecfromsubplotspec need an outer container:
        # get a unique representation:
        rep = (gs, 'top')
        if rep not in layoutgrids:
            layoutgrids[rep] = mlayoutgrid.LayoutGrid(
                parent=subspeclb,
                name='top',
                nrows=1, ncols=1,
                parent_pos=(subplot_spec.rowspan, subplot_spec.colspan))
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=layoutgrids[rep],
                name='gridspec',
                nrows=gs._nrows, ncols=gs._ncols,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    return layoutgrids


def check_no_collapsed_axes(layoutgrids, fig):
    """
    Check that no Axes have collapsed to zero size.
    """
    for sfig in fig.subfigs:
        ok = check_no_collapsed_axes(layoutgrids, sfig)
        if not ok:
            return False
    for ax in fig.axes:
        gs = ax.get_gridspec()
        if gs in layoutgrids:  # also implies gs is not None.
            lg = layoutgrids[gs]
            for i in range(gs.nrows):
                for j in range(gs.ncols):
                    bb = lg.get_inner_bbox(i, j)
                    if bb.width <= 0 or bb.height <= 0:
                        return False
    return True


def compress_fixed_aspect(layoutgrids, fig):
    gs = None
    for ax in fig.axes:
        if ax.get_subplotspec() is None:
            continue
        ax.apply_aspect()
        sub = ax.get_subplotspec()
        _gs = sub.get_gridspec()
        if gs is None:
            gs = _gs
            extraw = np.zeros(gs.ncols)
            extrah = np.zeros(gs.nrows)
        elif _gs != gs:
            raise ValueError('Cannot do compressed layout if Axes are not'
                             'all from the same gridspec')
        orig = ax.get_position(original=True)
        actual = ax.get_position(original=False)
        dw = orig.width - actual.width
        if dw > 0:
            extraw[sub.colspan] = np.maximum(extraw[sub.colspan], dw)
        dh = orig.height - actual.height
        if dh > 0:
            extrah[sub.rowspan] = np.maximum(extrah[sub.rowspan], dh)

    if gs is None:
        raise ValueError('Cannot do compressed layout if no Axes '
                         'are part of a gridspec.')
    w = np.sum(extraw) / 2
    layoutgrids[fig].edit_margin_min('left', w)
    layoutgrids[fig].edit_margin_min('right', w)

    h = np.sum(extrah) / 2
    layoutgrids[fig].edit_margin_min('top', h)
    layoutgrids[fig].edit_margin_min('bottom', h)
    return layoutgrids


def get_margin_from_padding(obj, *, w_pad=0, h_pad=0,
                            hspace=0, wspace=0):

    ss = obj._subplotspec
    gs = ss.get_gridspec()

    if hasattr(gs, 'hspace'):
        _hspace = (gs.hspace if gs.hspace is not None else hspace)
        _wspace = (gs.wspace if gs.wspace is not None else wspace)
    else:
        _hspace = (gs._hspace if gs._hspace is not None else hspace)
        _wspace = (gs._wspace if gs._wspace is not None else wspace)

    _wspace = _wspace / 2
    _hspace = _hspace / 2

    nrows, ncols = gs.get_geometry()
    # there are two margins for each direction.  The "cb"
    # margins are for pads and colorbars, the non-"cb" are
    # for the Axes decorations (labels etc).
    margin = {'leftcb': w_pad, 'rightcb': w_pad,
              'bottomcb': h_pad, 'topcb': h_pad,
              'left': 0, 'right': 0,
              'top': 0, 'bottom': 0}
    if _wspace / ncols > w_pad:
        if ss.colspan.start > 0:
            margin['leftcb'] = _wspace / ncols
        if ss.colspan.stop < ncols:
            margin['rightcb'] = _wspace / ncols
    if _hspace / nrows > h_pad:
        if ss.rowspan.stop < nrows:
            margin['bottomcb'] = _hspace / nrows
        if ss.rowspan.start > 0:
            margin['topcb'] = _hspace / nrows

    return margin


def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
                        hspace=0, wspace=0):
    """
    For each Axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.

    Then make room for colorbars.

    Parameters
    ----------
    layoutgrids : dict
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    w_pad, h_pad : float, default: 0
        Width and height padding (in fraction of figure).
    hspace, wspace : float, default: 0
        Width and height padding as fraction of figure size divided by
        number of columns or rows.
    """
    for sfig in fig.subfigs:  # recursively make child panel margins
        ss = sfig._subplotspec
        gs = ss.get_gridspec()

        make_layout_margins(layoutgrids, sfig, renderer,
                            w_pad=w_pad, h_pad=h_pad,
                            hspace=hspace, wspace=wspace)

        margins = get_margin_from_padding(sfig, w_pad=0, h_pad=0,
                                          hspace=hspace, wspace=wspace)
        layoutgrids[gs].edit_outer_margin_mins(margins, ss)

    for ax in fig._localaxes:
        if not ax.get_subplotspec() or not ax.get_in_layout():
            continue

        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()

        if gs not in layoutgrids:
            return

        margin = get_margin_from_padding(ax, w_pad=w_pad, h_pad=h_pad,
                                         hspace=hspace, wspace=wspace)
        pos, bbox = get_pos_and_bbox(ax, renderer)
        # the margin is the distance between the bounding box of the Axes
        # and its position (plus the padding from above)
        margin['left'] += pos.x0 - bbox.x0
        margin['right'] += bbox.x1 - pos.x1
        # remember that rows are ordered from top:
        margin['bottom'] += pos.y0 - bbox.y0
        margin['top'] += bbox.y1 - pos.y1

        # make margin for colorbars.  These margins go in the
        # padding margin, versus the margin for Axes decorators.
        for cbax in ax._colorbars:
            # note pad is a fraction of the parent width...
            pad = colorbar_get_pad(layoutgrids, cbax)
            # colorbars can be child of more than one subplot spec:
            cbp_rspan, cbp_cspan = get_cb_parent_spans(cbax)
            loc = cbax._colorbar_info['location']
            cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)
            if loc == 'right':
                if cbp_cspan.stop == ss.colspan.stop:
                    # only increase if the colorbar is on the right edge
                    margin['rightcb'] += cbbbox.width + pad
            elif loc == 'left':
                if cbp_cspan.start == ss.colspan.start:
                    # only increase if the colorbar is on the left edge
                    margin['leftcb'] += cbbbox.width + pad
            elif loc == 'top':
                if cbp_rspan.start == ss.rowspan.start:
                    margin['topcb'] += cbbbox.height + pad
            else:
                if cbp_rspan.stop == ss.rowspan.stop:
                    margin['bottomcb'] += cbbbox.height + pad
            # If the colorbars are wider than the parent box in the
            # cross direction
            if loc in ['top', 'bottom']:
                if (cbp_cspan.start == ss.colspan.start and
                        cbbbox.x0 < bbox.x0):
                    margin['left'] += bbox.x0 - cbbbox.x0
                if (cbp_cspan.stop == ss.colspan.stop and
                        cbbbox.x1 > bbox.x1):
                    margin['right'] += cbbbox.x1 - bbox.x1
            # or taller:
            if loc in ['left', 'right']:
                if (cbp_rspan.stop == ss.rowspan.stop and
                        cbbbox.y0 < bbox.y0):
                    margin['bottom'] += bbox.y0 - cbbbox.y0
                if (cbp_rspan.start == ss.rowspan.start and
                        cbbbox.y1 > bbox.y1):
                    margin['top'] += cbbbox.y1 - bbox.y1
        # pass the new margins down to the layout grid for the solution...
        layoutgrids[gs].edit_outer_margin_mins(margin, ss)

    # make margins for figure-level legends:
    for leg in fig.legends:
        inv_trans_fig = None
        if leg._outside_loc and leg._bbox_to_anchor is None:
            if inv_trans_fig is None:
                inv_trans_fig = fig.transFigure.inverted().transform_bbox
            bbox = inv_trans_fig(leg.get_tightbbox(renderer))
            w = bbox.width + 2 * w_pad
            h = bbox.height + 2 * h_pad
            legendloc = leg._outside_loc
            if legendloc == 'lower':
                layoutgrids[fig].edit_margin_min('bottom', h)
            elif legendloc == 'upper':
                layoutgrids[fig].edit_margin_min('top', h)
            if legendloc == 'right':
                layoutgrids[fig].edit_margin_min('right', w)
            elif legendloc == 'left':
                layoutgrids[fig].edit_margin_min('left', w)


def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
    # Figure out how large the suptitle is and make the
    # top level figure margin larger.

    inv_trans_fig = fig.transFigure.inverted().transform_bbox
    # get the h_pad and w_pad as distances in the local subfigure coordinates:
    padbox = mtransforms.Bbox([[0, 0], [w_pad, h_pad]])
    padbox = (fig.transFigure -
              fig.transSubfigure).transform_bbox(padbox)
    h_pad_local = padbox.height
    w_pad_local = padbox.width

    for sfig in fig.subfigs:
        make_margin_suptitles(layoutgrids, sfig, renderer,
                              w_pad=w_pad, h_pad=h_pad)

    if fig._suptitle is not None and fig._suptitle.get_in_layout():
        p = fig._suptitle.get_position()
        if getattr(fig._suptitle, '_autopos', False):
            fig._suptitle.set_position((p[0], 1 - h_pad_local))
            bbox = inv_trans_fig(fig._suptitle.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('top', bbox.height + 2 * h_pad)

    if fig._supxlabel is not None and fig._supxlabel.get_in_layout():
        p = fig._supxlabel.get_position()
        if getattr(fig._supxlabel, '_autopos', False):
            fig._supxlabel.set_position((p[0], h_pad_local))
            bbox = inv_trans_fig(fig._supxlabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('bottom',
                                             bbox.height + 2 * h_pad)

    if fig._supylabel is not None and fig._supylabel.get_in_layout():
        p = fig._supylabel.get_position()
        if getattr(fig._supylabel, '_autopos', False):
            fig._supylabel.set_position((w_pad_local, p[1]))
            bbox = inv_trans_fig(fig._supylabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('left', bbox.width + 2 * w_pad)


def match_submerged_margins(layoutgrids, fig):
    """
    Make the margins that are submerged inside an Axes the same size.

    This allows Axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB\nCCDD")

    Without this routine, the Axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    """

    axsdone = []
    for sfig in fig.subfigs:
        axsdone += match_submerged_margins(layoutgrids, sfig)

    axs = [a for a in fig.get_axes()
           if (a.get_subplotspec() is not None and a.get_in_layout() and
               a not in axsdone)]

    for ax1 in axs:
        ss1 = ax1.get_subplotspec()
        if ss1.get_gridspec() not in layoutgrids:
            axs.remove(ax1)
            continue
        lg1 = layoutgrids[ss1.get_gridspec()]

        # interior columns:
        if len(ss1.colspan) > 1:
            maxsubl = np.max(
                lg1.margin_vals['left'][ss1.colspan[1:]] +
                lg1.margin_vals['leftcb'][ss1.colspan[1:]]
            )
            maxsubr = np.max(
                lg1.margin_vals['right'][ss1.colspan[:-1]] +
                lg1.margin_vals['rightcb'][ss1.colspan[:-1]]
            )
            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None and len(ss2.colspan) > 1:
                    maxsubl2 = np.max(
                        lg2.margin_vals['left'][ss2.colspan[1:]] +
                        lg2.margin_vals['leftcb'][ss2.colspan[1:]])
                    if maxsubl2 > maxsubl:
                        maxsubl = maxsubl2
                    maxsubr2 = np.max(
                        lg2.margin_vals['right'][ss2.colspan[:-1]] +
                        lg2.margin_vals['rightcb'][ss2.colspan[:-1]])
                    if maxsubr2 > maxsubr:
                        maxsubr = maxsubr2
            for i in ss1.colspan[1:]:
                lg1.edit_margin_min('left', maxsubl, cell=i)
            for i in ss1.colspan[:-1]:
                lg1.edit_margin_min('right', maxsubr, cell=i)

        # interior rows:
        if len(ss1.rowspan) > 1:
            maxsubt = np.max(
                lg1.margin_vals['top'][ss1.rowspan[1:]] +
                lg1.margin_vals['topcb'][ss1.rowspan[1:]]
            )
            maxsubb = np.max(
                lg1.margin_vals['bottom'][ss1.rowspan[:-1]] +
                lg1.margin_vals['bottomcb'][ss1.rowspan[:-1]]
            )

            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None:
                    if len(ss2.rowspan) > 1:
                        maxsubt = np.max([np.max(
                            lg2.margin_vals['top'][ss2.rowspan[1:]] +
                            lg2.margin_vals['topcb'][ss2.rowspan[1:]]
                        ), maxsubt])
                        maxsubb = np.max([np.max(
                            lg2.margin_vals['bottom'][ss2.rowspan[:-1]] +
                            lg2.margin_vals['bottomcb'][ss2.rowspan[:-1]]
                        ), maxsubb])
            for i in ss1.rowspan[1:]:
                lg1.edit_margin_min('top', maxsubt, cell=i)
            for i in ss1.rowspan[:-1]:
                lg1.edit_margin_min('bottom', maxsubb, cell=i)

    return axs


def get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to.

    Parameters
    ----------
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    """
    rowstart = np.inf
    rowstop = -np.inf
    colstart = np.inf
    colstop = -np.inf
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
        rowstart = min(ss.rowspan.start, rowstart)
        rowstop = max(ss.rowspan.stop, rowstop)
        colstart = min(ss.colspan.start, colstart)
        colstop = max(ss.colspan.stop, colstop)

    rowspan = range(rowstart, rowstop)
    colspan = range(colstart, colstop)
    return rowspan, colspan


def get_pos_and_bbox(ax, renderer):
    """
    Get the position and the bbox for the Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.

    Returns
    -------
    pos : `~matplotlib.transforms.Bbox`
        Position in figure coordinates.
    bbox : `~matplotlib.transforms.Bbox`
        Tight bounding box in figure coordinates.
    """
    fig = ax.get_figure(root=False)
    pos = ax.get_position(original=True)
    # pos is in panel co-ords, but we need in figure for the layout
    pos = pos.transformed(fig.transSubfigure - fig.transFigure)
    tightbbox = martist._get_tightbbox_for_layout_only(ax, renderer)
    if tightbbox is None:
        bbox = pos
    else:
        bbox = tightbbox.transformed(fig.transFigure.inverted())
    return pos, bbox


def reposition_axes(layoutgrids, fig, renderer, *,
                    w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    Reposition all the Axes based on the new inner bounding box.
    """
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure
    for sfig in fig.subfigs:
        bbox = layoutgrids[sfig].get_outer_bbox()
        sfig._redo_transform_rel_fig(
            bbox=bbox.transformed(trans_fig_to_subfig))
        reposition_axes(layoutgrids, sfig, renderer,
                        w_pad=w_pad, h_pad=h_pad,
                        wspace=wspace, hspace=hspace)

    for ax in fig._localaxes:
        if ax.get_subplotspec() is None or not ax.get_in_layout():
            continue

        # grid bbox is in Figure coordinates, but we specify in panel
        # coordinates...
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        if gs not in layoutgrids:
            return

        bbox = layoutgrids[gs].get_inner_bbox(rows=ss.rowspan,
                                              cols=ss.colspan)

        # transform from figure to panel for set_position:
        newbbox = trans_fig_to_subfig.transform_bbox(bbox)
        ax._set_position(newbbox)

        # move the colorbars:
        # we need to keep track of oldw and oldh if there is more than
        # one colorbar:
        offset = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        for nn, cbax in enumerate(ax._colorbars[::-1]):
            if ax == cbax._colorbar_info['parents'][0]:
                reposition_colorbar(layoutgrids, cbax, renderer,
                                    offset=offset)


def reposition_colorbar(layoutgrids, cbax, renderer, *, offset=None):
    """
    Place the colorbar in its new place.

    Parameters
    ----------
    layoutgrids : dict
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    offset : array-like
        Offset the colorbar needs to be pushed to in order to
        account for multiple colorbars.
    """

    parents = cbax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    fig = cbax.get_figure(root=False)
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure

    cb_rspans, cb_cspans = get_cb_parent_spans(cbax)
    bboxparent = layoutgrids[gs].get_bbox_for_cb(rows=cb_rspans,
                                                 cols=cb_cspans)
    pb = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)

    location = cbax._colorbar_info['location']
    anchor = cbax._colorbar_info['anchor']
    fraction = cbax._colorbar_info['fraction']
    aspect = cbax._colorbar_info['aspect']
    shrink = cbax._colorbar_info['shrink']

    cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)

    # Colorbar gets put at extreme edge of outer bbox of the subplotspec
    # It needs to be moved in by: 1) a pad 2) its "margin" 3) by
    # any colorbars already added at this location:
    cbpad = colorbar_get_pad(layoutgrids, cbax)
    if location in ('left', 'right'):
        # fraction and shrink are fractions of parent
        pbcb = pb.shrunk(fraction, shrink).anchored(anchor, pb)
        # The colorbar is at the left side of the parent.  Need
        # to translate to right (or left)
        if location == 'right':
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x1 - pbcb.x0 + offset['right']
            dx += cbpad + lmargin
            offset['right'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
        else:
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x0 - pbcb.x0  # edge of parent
            dx += -cbbbox.width - cbpad + lmargin - offset['left']
            offset['left'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
    else:  # horizontal axes:
        pbcb = pb.shrunk(shrink, fraction).anchored(anchor, pb)
        if location == 'top':
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y1 - pbcb.y0 + offset['top']
            dy += cbpad + bmargin
            offset['top'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)
        else:
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y0 - pbcb.y0
            dy += -cbbbox.height - cbpad + bmargin - offset['bottom']
            offset['bottom'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)

    pbcb = trans_fig_to_subfig.transform_bbox(pbcb)
    cbax.set_transform(fig.transSubfigure)
    cbax._set_position(pbcb)
    cbax.set_anchor(anchor)
    if location in ['bottom', 'top']:
        aspect = 1 / aspect
    cbax.set_box_aspect(aspect)
    cbax.set_aspect('auto')
    return offset


def reset_margins(layoutgrids, fig):
    """
    Reset the margins in the layoutboxes of *fig*.

    Margins are usually set as a minimum, so if the figure gets smaller
    the minimum needs to be zero in order for it to grow again.
    """
    for sfig in fig.subfigs:
        reset_margins(layoutgrids, sfig)
    for ax in fig.axes:
        if ax.get_in_layout():
            gs = ax.get_gridspec()
            if gs in layoutgrids:  # also implies gs is not None.
                layoutgrids[gs].reset_margins()
    layoutgrids[fig].reset_margins()


def colorbar_get_pad(layoutgrids, cax):
    parents = cax._colorbar_info['parents']
    gs = parents[0].get_gridspec()

    cb_rspans, cb_cspans = get_cb_parent_spans(cax)
    bboxouter = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)

    if cax._colorbar_info['location'] in ['right', 'left']:
        size = bboxouter.width
    else:
        size = bboxouter.height

    return cax._colorbar_info['pad'] * size
<|endoftext|>"""
Tick locating and formatting
============================

This module contains classes for configuring tick locating and formatting.
Generic tick locators and formatters are provided, as well as domain specific
custom ones.

Although the locators know nothing about major or minor ticks, they are used
by the Axis class to support major and minor tick locating and formatting.

.. _tick_locating:
.. _locators:

Tick locating
-------------

The Locator class is the base class for all tick locators. The locators
handle autoscaling of the view limits based on the data limits, and the
choosing of tick locations. A useful semi-automatic tick locator is
`MultipleLocator`. It is initialized with a base, e.g., 10, and it picks
axis limits and ticks that are multiples of that base.

The Locator subclasses defined here are:

======================= =======================================================
`AutoLocator`           `MaxNLocator` with simple defaults. This is the default
                        tick locator for most plotting.
`MaxNLocator`           Finds up to a max number of intervals with ticks at
                        nice locations.
`LinearLocator`         Space ticks evenly from min to max.
`LogLocator`            Space ticks logarithmically from min to max.
`MultipleLocator`       Ticks and range are a multiple of base; either integer
                        or float.
`FixedLocator`          Tick locations are fixed.
`IndexLocator`          Locator for index plots (e.g., where
                        ``x = range(len(y))``).
`NullLocator`           No ticks.
`SymmetricalLogLocator` Locator for use with the symlog norm; works like
                        `LogLocator` for the part outside of the threshold and
                        adds 0 if inside the limits.
`AsinhLocator`          Locator for use with the asinh norm, attempting to
                        space ticks approximately uniformly.
`LogitLocator`          Locator for logit scaling.
`AutoMinorLocator`      Locator for minor ticks when the axis is linear and the
                        major ticks are uniformly spaced. Subdivides the major
                        tick interval into a specified number of minor
                        intervals, defaulting to 4 or 5 depending on the major
                        interval.
======================= =======================================================

There are a number of locators specialized for date locations - see
the :mod:`.dates` module.

You can define your own locator by deriving from Locator. You must
override the ``__call__`` method, which returns a sequence of locations,
and you will probably want to override the autoscale method to set the
view limits from the data limits.

If you want to override the default locator, use one of the above or a custom
locator and pass it to the x- or y-axis instance. The relevant methods are::

  ax.xaxis.set_major_locator(xmajor_locator)
  ax.xaxis.set_minor_locator(xminor_locator)
  ax.yaxis.set_major_locator(ymajor_locator)
  ax.yaxis.set_minor_locator(yminor_locator)

The default minor locator is `NullLocator`, i.e., no minor ticks on by default.

.. note::
    `Locator` instances should not be used with more than one
    `~matplotlib.axis.Axis` or `~matplotlib.axes.Axes`. So instead of::

        locator = MultipleLocator(5)
        ax.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_locator(locator)

    do the following instead::

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax2.xaxis.set_major_locator(MultipleLocator(5))

.. _formatters:

Tick formatting
---------------

Tick formatting is controlled by classes derived from Formatter. The formatter
operates on a single tick value and returns a string to the axis.

========================= =====================================================
`NullFormatter`           No labels on the ticks.
`FixedFormatter`          Set the strings manually for the labels.
`FuncFormatter`           User defined function sets the labels.
`StrMethodFormatter`      Use string `format` method.
`FormatStrFormatter`      Use an old-style sprintf format string.
`ScalarFormatter`         Default formatter for scalars: autopick the format
                          string.
`LogFormatter`            Formatter for log axes.
`LogFormatterExponent`    Format values for log axis using
                          ``exponent = log_base(value)``.
`LogFormatterMathtext`    Format values for log axis using
                          ``exponent = log_base(value)`` using Math text.
`LogFormatterSciNotation` Format values for log axis using scientific notation.
`LogitFormatter`          Probability formatter.
`EngFormatter`            Format labels in engineering notation.
`PercentFormatter`        Format labels as a percentage.
========================= =====================================================

You can derive your own formatter from the Formatter base class by
simply overriding the ``__call__`` method. The formatter class has
access to the axis view and data limits.

To control the major and minor tick label formats, use one of the
following methods::

  ax.xaxis.set_major_formatter(xmajor_formatter)
  ax.xaxis.set_minor_formatter(xminor_formatter)
  ax.yaxis.set_major_formatter(ymajor_formatter)
  ax.yaxis.set_minor_formatter(yminor_formatter)

In addition to a `.Formatter` instance, `~.Axis.set_major_formatter` and
`~.Axis.set_minor_formatter` also accept a ``str`` or function.  ``str`` input
will be internally replaced with an autogenerated `.StrMethodFormatter` with
the input ``str``. For function input, a `.FuncFormatter` with the input
function will be generated and used.

See :doc:`/gallery/ticks/major_minor_demo` for an example of setting major
and minor ticks. See the :mod:`matplotlib.dates` module for more information
and examples of using date locators and formatters.
"""

import itertools
import logging
import locale
import math
from numbers import Integral
import string

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms

_log = logging.getLogger(__name__)

__all__ = ('TickHelper', 'Formatter', 'FixedFormatter',
           'NullFormatter', 'FuncFormatter', 'FormatStrFormatter',
           'StrMethodFormatter', 'ScalarFormatter', 'LogFormatter',
           'LogFormatterExponent', 'LogFormatterMathtext',
           'LogFormatterSciNotation',
           'LogitFormatter', 'EngFormatter', 'PercentFormatter',
           'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator',
           'LinearLocator', 'LogLocator', 'AutoLocator',
           'MultipleLocator', 'MaxNLocator', 'AutoMinorLocator',
           'SymmetricalLogLocator', 'AsinhLocator', 'LogitLocator')


class _DummyAxis:
    __name__ = "dummy"

    def __init__(self, minpos=0):
        self._data_interval = (0, 1)
        self._view_interval = (0, 1)
        self._minpos = minpos

    def get_view_interval(self):
        return self._view_interval

    def set_view_interval(self, vmin, vmax):
        self._view_interval = (vmin, vmax)

    def get_minpos(self):
        return self._minpos

    def get_data_interval(self):
        return self._data_interval

    def set_data_interval(self, vmin, vmax):
        self._data_interval = (vmin, vmax)

    def get_tick_space(self):
        # Just use the long-standing default of nbins==9
        return 9


class TickHelper:
    axis = None

    def set_axis(self, axis):
        self.axis = axis

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.axis = _DummyAxis(**kwargs)


class Formatter(TickHelper):
    """
    Create a string based on a tick value and location.
    """
    # some classes want to see all the locs to help format
    # individual ones
    locs = []

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position pos.
        ``pos=None`` indicates an unspecified location.
        """
        raise NotImplementedError('Derived must override')

    def format_ticks(self, values):
        """Return the tick labels for all the ticks at once."""
        self.set_locs(values)
        return [self(value, i) for i, value in enumerate(values)]

    def format_data(self, value):
        """
        Return the full string representation of the value with the
        position unspecified.
        """
        return self.__call__(value)

    def format_data_short(self, value):
        """
        Return a short string version of the tick value.

        Defaults to the position-independent long value.
        """
        return self.format_data(value)

    def get_offset(self):
        return ''

    def set_locs(self, locs):
        """
        Set the locations of the ticks.

        This method is called before computing the tick labels because some
        formatters need to know all tick locations to do so.
        """
        self.locs = locs

    @staticmethod
    def fix_minus(s):
        """
        Some classes may want to replace a hyphen for minus with the proper
        Unicode symbol (U+2212) for typographical correctness.  This is a
        helper method to perform such a replacement when it is enabled via
        :rc:`axes.unicode_minus`.
        """
        return (s.replace('-', '\N{MINUS SIGN}')
                if mpl.rcParams['axes.unicode_minus']
                else s)

    def _set_locator(self, locator):
        """Subclasses may want to override this to set a locator."""
        pass


class NullFormatter(Formatter):
    """Always return the empty string."""

    def __call__(self, x, pos=None):
        # docstring inherited
        return ''


class FixedFormatter(Formatter):
    """
    Return fixed strings for tick labels based only on position, not value.

    .. note::
        `.FixedFormatter` should only be used together with `.FixedLocator`.
        Otherwise, the labels may end up in unexpected positions.
    """

    def __init__(self, seq):
        """Set the sequence *seq* of strings that will be used for labels."""
        self.seq = seq
        self.offset_string = ''

    def __call__(self, x, pos=None):
        """
        Return the label that matches the position, regardless of the value.

        For positions ``pos < len(seq)``, return ``seq[i]`` regardless of
        *x*. Otherwise return empty string. ``seq`` is the sequence of
        strings that this object was initialized with.
        """
        if pos is None or pos >= len(self.seq):
            return ''
        else:
            return self.seq[pos]

    def get_offset(self):
        return self.offset_string

    def set_offset_string(self, ofs):
        self.offset_string = ofs


class FuncFormatter(Formatter):
    """
    Use a user-defined function for formatting.

    The function should take in two inputs (a tick value ``x`` and a
    position ``pos``), and return a string containing the corresponding
    tick label.
    """

    def __init__(self, func):
        self.func = func
        self.offset_string = ""

    def __call__(self, x, pos=None):
        """
        Return the value of the user defined function.

        *x* and *pos* are passed through as-is.
        """
        return self.func(x, pos)

    def get_offset(self):
        return self.offset_string

    def set_offset_string(self, ofs):
        self.offset_string = ofs


class FormatStrFormatter(Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.

    Negative numeric values (e.g., -1) will use a dash, not a Unicode minus;
    use mathtext to get a Unicode minus by wrapping the format specifier with $
    (e.g. "$%g$").
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        Only the value *x* is formatted. The position is ignored.
        """
        return self.fmt % x


class _UnicodeMinusFormat(string.Formatter):
    """
    A specialized string formatter so that `.StrMethodFormatter` respects
    :rc:`axes.unicode_minus`.  This implementation relies on the fact that the
    format string is only ever called with kwargs *x* and *pos*, so it blindly
    replaces dashes by unicode minuses without further checking.
    """

    def format_field(self, value, format_spec):
        return Formatter.fix_minus(super().format_field(value, format_spec))


class StrMethodFormatter(Formatter):
    """
    Use a new-style format string (as used by `str.format`) to format the tick.

    The field used for the tick value must be labeled *x* and the field used
    for the tick position must be labeled *pos*.

    The formatter will respect :rc:`axes.unicode_minus` when formatting
    negative numeric values.

    It is typically unnecessary to explicitly construct `.StrMethodFormatter`
    objects, as `~.Axis.set_major_formatter` directly accepts the format string
    itself.
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        *x* and *pos* are passed to `str.format` as keyword arguments
        with those exact names.
        """
        return _UnicodeMinusFormat().format(self.fmt, x=x, pos=pos)


class ScalarFormatter(Formatter):
    """
    Format tick values as a number.

    Parameters
    ----------
    useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
        Whether to use offset notation. See `.set_useOffset`.
    useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
        Whether to use fancy math formatting. See `.set_useMathText`.
    useLocale : bool, default: :rc:`axes.formatter.use_locale`.
        Whether to use locale settings for decimal sign and positive sign.
        See `.set_useLocale`.
    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the
        numbers in the formatter.

        .. versionadded:: 3.10

    Notes
    -----
    In addition to the parameters above, the formatting of scientific vs.
    floating point representation can be configured via `.set_scientific`
    and `.set_powerlimits`).

    **Offset notation and scientific notation**

    Offset notation and scientific notation look quite similar at first sight.
    Both split some information from the formatted tick values and display it
    at the end of the axis.

    - The scientific notation splits up the order of magnitude, i.e. a
      multiplicative scaling factor, e.g. ``1e6``.

    - The offset notation separates an additive constant, e.g. ``+1e6``. The
      offset notation label is always prefixed with a ``+`` or ``-`` sign
      and is thus distinguishable from the order of magnitude label.

    The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
    the different formatting. Note the labels at the right edge of the x axis.

    .. plot::

        lim = (1_000_000, 1_000_010)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
        ax1.set(title='offset notation', xlim=lim)
        ax2.set(title='scientific notation', xlim=lim)
        ax2.xaxis.get_major_formatter().set_useOffset(False)
        ax3.set(title='floating-point notation', xlim=lim)
        ax3.xaxis.get_major_formatter().set_useOffset(False)
        ax3.xaxis.get_major_formatter().set_scientific(False)

    """

    def __init__(self, useOffset=None, useMathText=None, useLocale=None, *,
                 usetex=None):
        if useOffset is None:
            useOffset = mpl.rcParams['axes.formatter.useoffset']
        self._offset_threshold = \
            mpl.rcParams['axes.formatter.offset_threshold']
        self.set_useOffset(useOffset)
        self.set_usetex(usetex)
        self.set_useMathText(useMathText)
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = mpl.rcParams['axes.formatter.limits']
        self.set_useLocale(useLocale)

    def get_usetex(self):
        """Return whether TeX's math mode is enabled for rendering."""
        return self._usetex

    def set_usetex(self, val):
        """Set whether to use TeX's math mode for rendering numbers in the formatter."""
        self._usetex = mpl._val_or_rc(val, 'text.usetex')

    usetex = property(fget=get_usetex, fset=set_usetex)

    def get_useOffset(self):
        """
        Return whether automatic mode for offset notation is active.

        This returns True if ``set_useOffset(True)``; it returns False if an
        explicit offset was set, e.g. ``set_useOffset(1000)``.

        See Also
        --------
        ScalarFormatter.set_useOffset
        """
        return self._useOffset

    def set_useOffset(self, val):
        """
        Set whether to use offset notation.

        When formatting a set numbers whose value is large compared to their
        range, the formatter can separate an additive constant. This can
        shorten the formatted numbers so that they are less likely to overlap
        when drawn on an axis.

        Parameters
        ----------
        val : bool or float
            - If False, do not use offset notation.
            - If True (=automatic mode), use offset notation if it can make
              the residual numbers significantly shorter. The exact behavior
              is controlled by :rc:`axes.formatter.offset_threshold`.
            - If a number, force an offset of the given value.

        Examples
        --------
        With active offset notation, the values

        ``100_000, 100_002, 100_004, 100_006, 100_008``

        will be formatted as ``0, 2, 4, 6, 8`` plus an offset ``+1e5``, which
        is written to the edge of the axis.
        """
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val

    useOffset = property(fget=get_useOffset, fset=set_useOffset)

    def get_useLocale(self):
        """
        Return whether locale settings are used for formatting.

        See Also
        --------
        ScalarFormatter.set_useLocale
        """
        return self._useLocale

    def set_useLocale(self, val):
        """
        Set whether to use locale settings for decimal sign and positive sign.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
        if val is None:
            self._useLocale = mpl.rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = val

    useLocale = property(fget=get_useLocale, fset=set_useLocale)

    def _format_maybe_minus_and_locale(self, fmt, arg):
        """
        Format *arg* with *fmt*, applying Unicode minus and locale if desired.
        """
        return self.fix_minus(
                # Escape commas introduced by locale.format_string if using math text,
                # but not those present from the beginning in fmt.
                (",".join(locale.format_string(part, (arg,), True).replace(",", "{,}")
                          for part in fmt.split(",")) if self._useMathText
                 else locale.format_string(fmt, (arg,), True))
                if self._useLocale
                else fmt % arg)

    def get_useMathText(self):
        """
        Return whether to use fancy math formatting.

        See Also
        --------
        ScalarFormatter.set_useMathText
        """
        return self._useMathText

    def set_useMathText(self, val):
        r"""
        Set whether to use fancy math formatting.

        If active, scientific notation is formatted as :math:`1.2 \times 10^3`.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_mathtext`.
        """
        if val is None:
            self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']
            if self._useMathText is False:
                try:
                    from matplotlib import font_manager
                    ufont = font_manager.findfont(
                        font_manager.FontProperties(
                            family=mpl.rcParams["font.family"]
                        ),
                        fallback_to_default=False,
                    )
                except ValueError:
                    ufont = None

                if ufont == str(cbook._get_data_path("fonts/ttf/cmr10.ttf")):
                    _api.warn_external(
                        "cmr10 font should ideally be used with "
                        "mathtext, set axes.formatter.use_mathtext to True"
                    )
        else:
            self._useMathText = val

    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''
        else:
            xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
            if abs(xp) < 1e-8:
                xp = 0
            return self._format_maybe_minus_and_locale(self.format, xp)

    def set_scientific(self, b):
        """
        Turn scientific notation on or off.

        See Also
        --------
        ScalarFormatter.set_powerlimits
        """
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        r"""
        Set size thresholds for scientific notation.

        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \times 10^\mathrm{exp}` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.

            The default limits are controlled by :rc:`axes.formatter.limits`.

            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.

            Typically, *min_exp* will be negative and *max_exp* will be
            positive.

            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \times 10^{-3}, 9.9 \times 10^{-3}, 0.01,`
            :math:`9999, 1 \times 10^4`.

        See Also
        --------
        ScalarFormatter.set_scientific
        """
        if len(lims) != 2:
            raise ValueError("'lims' must be a sequence of length 2")
        self._powerlimits = lims

    def format_data_short(self, value):
        # docstring inherited
        if value is np.ma.masked:
            return ""
        if isinstance(value, Integral):
            fmt = "%d"
        else:
            if getattr(self.axis, "__name__", "") in ["xaxis", "yaxis"]:
                if self.axis.__name__ == "xaxis":
                    axis_trf = self.axis.axes.get_xaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((value, 0))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[-1, 0], [+1, 0]])[:, 0]
                else:  # yaxis:
                    axis_trf = self.axis.axes.get_yaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((0, value))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[0, -1], [0, +1]])[:, 1]
                delta = abs(neighbor_values - value).max()
            else:
                # Rough approximation: no more than 1e4 divisions.
                a, b = self.axis.get_view_interval()
                delta = (b - a) / 1e4
            fmt = f"%-#.{cbook._g_sig_digits(value, delta)}g"
        return self._format_maybe_minus_and_locale(fmt, value)

    def format_data(self, value):
        # docstring inherited
        e = math.floor(math.log10(abs(value)))
        s = round(value / 10**e, 10)
        significand = self._format_maybe_minus_and_locale(
            "%d" if s % 1 == 0 else "%1.10g", s)
        if e == 0:
            return significand
        exponent = self._format_maybe_minus_and_locale("%d", e)
        if self._useMathText or self._usetex:
            exponent = "10^{%s}" % exponent
            return (exponent if s == 1  # reformat 1x10^y as 10^y
                    else rf"{significand} \times {exponent}")
        else:
            return f"{significand}e{exponent}"

    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        if len(self.locs) == 0:
            return ''
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''
            sciNotStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0:
                    offsetStr = '+' + offsetStr
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = self.format_data(10 ** self.orderOfMagnitude)
                else:
                    sciNotStr = '1e%d' % self.orderOfMagnitude
            if self._useMathText or self._usetex:
                if sciNotStr != '':
                    sciNotStr = r'\times\mathdefault{%s}' % sciNotStr
                s = fr'${sciNotStr}\mathdefault{{{offsetStr}}}$'
            else:
                s = ''.join((sciNotStr, offsetStr))
            return self.fix_minus(s)
        return ''

    def set_locs(self, locs):
        # docstring inherited
        self.locs = locs
        if len(self.locs) > 0:
            if self._useOffset:
                self._compute_offset()
            self._set_order_of_magnitude()
            self._set_format()

    def _compute_offset(self):
        locs = self.locs
        # Restrict to visible ticks.
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        # Only use offset if there are at least two ticks and every tick has
        # the same sign.
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        # min, max comparing absolute values (we want division to round towards
        # zero so we work on absolute values).
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = math.copysign(1, lmin)
        # What is the smallest power of ten such that abs_min and abs_max are
        # equal up to that precision?
        # Note: Internally using oom instead of 10 ** oom avoids some numerical
        # accuracy issues.
        oom_max = np.ceil(math.log10(abs_max))
        oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                       if abs_min // 10 ** oom != abs_max // 10 ** oom)
        if (abs_max - abs_min) / 10 ** oom <= 1e-2:
            # Handle the case of straddling a multiple of a large power of ten
            # (relative to the span).
            # What is the smallest power of ten such that abs_min and abs_max
            # are no more than 1 apart at that precision?
            oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                           if abs_max // 10 ** oom - abs_min // 10 ** oom > 1)
        # Only use offset if it saves at least _offset_threshold digits.
        n = self._offset_threshold - 1
        self.offset = (sign * (abs_max // 10 ** oom) * 10 ** oom
                       if abs_max // 10 ** oom >= 10**n
                       else 0)

    def _set_order_of_magnitude(self):
        # if scientific notation is to be used, find the appropriate exponent
        # if using a numerical offset, find the exponent after applying the
        # offset. When lower power limit = upper <> 0, use provided exponent.
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        if self._powerlimits[0] == self._powerlimits[1] != 0:
            # fixed scaling when lower power limit = upper <> 0.
            self.orderOfMagnitude = self._powerlimits[0]
            return
        # restrict to visible ticks
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(self.locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        locs = np.abs(locs)
        if not len(locs):
            self.orderOfMagnitude = 0
            return
        if self.offset:
            oom = math.floor(math.log10(vmax - vmin))
        else:
            val = locs.max()
            if val == 0:
                oom = 0
            else:
                oom = math.floor(math.log10(val))
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = f'%1.{sigfigs}f'
        if self._usetex or self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


class LogFormatter(Formatter):
    """
    Base class for formatting ticks on a log or symlog scale.

    It may be instantiated directly, or subclassed.

    Parameters
    ----------
    base : float, default: 10.
        Base of the logarithm used in all calculations.

    labelOnlyBase : bool, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.

    minor_thresholds : (subset, all), default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.

    linthresh : None or float, default: None
        If a symmetric log scale is in use, its ``linthresh``
        parameter must be supplied here.

    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.

    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.

    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.

    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.

    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """

    def __init__(self, base=10.0, labelOnlyBase=False,
                 minor_thresholds=None,
                 linthresh=None):

        self.set_base(base)
        self.set_label_minor(labelOnlyBase)
        if minor_thresholds is None:
            if mpl.rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh

    def set_base(self, base):
        """
        Change the *base* for labeling.

        .. warning::
           Should always match the base used for :class:`LogLocator`
        """
        self._base = float(base)

    def set_label_minor(self, labelOnlyBase):
        """
        Switch minor tick labeling on or off.

        Parameters
        ----------
        labelOnlyBase : bool
            If True, label ticks only at integer powers of base.
        """
        self.labelOnlyBase = labelOnlyBase

    def set_locs(self, locs=None):
        """
        Use axis view limits to control which ticks are labeled.

        The *locs* parameter is ignored in the present algorithm.
        """
        if np.isinf(self.minor_thresholds[0]):
            self._sublabels = None
            return

        # Handle symlog case:
        linthresh = self._linthresh
        if linthresh is None:
            try:
                linthresh = self.axis.get_transform().linthresh
            except AttributeError:
                pass

        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if linthresh is None and vmin <= 0:
            # It's probably a colorbar with
            # a format kwarg setting a LogFormatter in the manner
            # that worked with 1.5.x, but that doesn't work now.
            self._sublabels = {1}  # label powers of base
            return

        b = self._base
        if linthresh is not None:  # symlog
            # Only compute the number of decades in the logarithmic part of the
            # axis
            numdec = 0
            if vmin < -linthresh:
                rhs = min(vmax, -linthresh)
                numdec += math.log(vmin / rhs) / math.log(b)
            if vmax > linthresh:
                lhs = max(vmin, linthresh)
                numdec += math.log(vmax / lhs) / math.log(b)
        else:
            vmin = math.log(vmin) / math.log(b)
            vmax = math.log(vmax) / math.log(b)
            numdec = abs(vmax - vmin)

        if numdec > self.minor_thresholds[0]:
            # Label only bases
            self._sublabels = {1}
        elif numdec > self.minor_thresholds[1]:
            # Add labels between bases at log-spaced coefficients;
            # include base powers in case the locations include
            # "major" and "minor" points, as in colorbar.
            c = np.geomspace(1, b, int(b)//2 + 1)
            self._sublabels = set(np.round(c))
            # For base 10, this yields (1, 2, 3, 4, 6, 10).
        else:
            # Label all integer multiples of base**n.
            self._sublabels = set(np.arange(1, b + 1))

    def _num_to_string(self, x, vmin, vmax):
        return self._pprint_val(x, vmax - vmin) if 1 <= x <= 10000 else f"{x:1.0e}"

    def __call__(self, x, pos=None):
        # docstring inherited
        if x == 0.0:  # Symlog
            return '0'

        x = abs(x)
        b = self._base
        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))

        if self.labelOnlyBase and not is_x_decade:
            return ''
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        s = self._num_to_string(x, vmin, vmax)
        return self.fix_minus(s)

    def format_data(self, value):
        with cbook._setattr_cm(self, labelOnlyBase=False):
            return cbook.strip_math(self.__call__(value))

    def format_data_short(self, value):
        # docstring inherited
        return ('%-12g' % value).rstrip()

    def _pprint_val(self, x, d):
        # If the number is not too big and it's an int, format it as an int.
        if abs(x) < 1e4 and x == int(x):
            return '%d' % x
        fmt = ('%1.3e' if d < 1e-2 else
               '%1.3f' if d <= 1 else
               '%1.2f' if d <= 10 else
               '%1.1f' if d <= 1e5 else
               '%1.1e')
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                s = '%se%d' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _num_to_string(self, x, vmin, vmax):
        fx = math.log(x) / math.log(self._base)
        if 1 <= abs(fx) <= 10000:
            fd = math.log(vmax - vmin) / math.log(self._base)
            s = self._pprint_val(fx, fd)
        else:
            s = f"{fx:1.0g}"
        return s


class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
        return r'$\mathdefault{%s%s^{%.2f}}$' % (sign_string, base, fx)

    def __call__(self, x, pos=None):
        # docstring inherited
        if x == 0:  # Symlog
            return r'$\mathdefault{0}$'

        sign_string = '-' if x < 0 else ''
        x = abs(x)
        b = self._base

        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))

        if self.labelOnlyBase and not is_x_decade:
            return ''
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''

        if is_x_decade:
            fx = round(fx)

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        if abs(fx) < mpl.rcParams['axes.formatter.min_exponent']:
            return r'$\mathdefault{%s%g}$' % (sign_string, x)
        elif not is_x_decade:
            usetex = mpl.rcParams['text.usetex']
            return self._non_decade_format(sign_string, base, fx, usetex)
        else:
            return r'$\mathdefault{%s%s^{%d}}$' % (sign_string, base, fx)


class LogFormatterSciNotation(LogFormatterMathtext):
    """
    Format values following scientific notation in a logarithmic axis.
    """

    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
        b = float(base)
        exponent = math.floor(fx)
        coeff = b ** (fx - exponent)
        if _is_close_to_int(coeff):
            coeff = round(coeff)
        return r'$\mathdefault{%s%g\times%s^{%d}}$' \
            % (sign_string, coeff, base, exponent)


class LogitFormatter(Formatter):
    """
    Probability formatter (using Math text).
    """

    def __init__(
        self,
        *,
        use_overline=False,
        one_half=r"\frac{1}{2}",
        minor=False,
        minor_threshold=25,
        minor_number=6,
    ):
        r"""
        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1 - v, indicate if x should be displayed as
            $\overline{v}$. The default is to display $1 - v$.

        one_half : str, default: r"\\frac{1}{2}"
            The string used to represent 1/2.

        minor : bool, default: False
            Indicate if the formatter is formatting minor ticks or not.
            Basically minor ticks are not labelled, except when only few ticks
            are provided, ticks with most space with neighbor ticks are
            labelled. See other parameters to change the default behavior.

        minor_threshold : int, default: 25
            Maximum number of locs for labelling some minor ticks. This
            parameter have no effect if minor is False.

        minor_number : int, default: 6
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        self._use_overline = use_overline
        self._one_half = one_half
        self._minor = minor
        self._labelled = set()
        self._minor_threshold = minor_threshold
        self._minor_number = minor_number

    def use_overline(self, use_overline):
        r"""
        Switch display mode with overline for labelling p>1/2.

        Parameters
        ----------
        use_overline : bool
            If x > 1/2, with x = 1 - v, indicate if x should be displayed as
            $\overline{v}$. The default is to display $1 - v$.
        """
        self._use_overline = use_overline

    def set_one_half(self, one_half):
        r"""
        Set the way one half is displayed.

        one_half : str
            The string used to represent 1/2.
        """
        self._one_half = one_half

    def set_minor_threshold(self, minor_threshold):
        """
        Set the threshold for labelling minors ticks.

        Parameters
        ----------
        minor_threshold : int
            Maximum number of locations for labelling some minor ticks. This
            parameter have no effect if minor is False.
        """
        self._minor_threshold = minor_threshold

    def set_minor_number(self, minor_number):
        """
        Set the number of minor ticks to label when some minor ticks are
        labelled.

        Parameters
        ----------
        minor_number : int
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        self._minor_number = minor_number

    def set_locs(self, locs):
        self.locs = np.array(locs)
        self._labelled.clear()

        if not self._minor:
            return None
        if all(
            _is_decade(x, rtol=1e-7)
            or _is_decade(1 - x, rtol=1e-7)
            or (_is_close_to_int(2 * x) and
                int(np.round(2 * x)) == 1)
            for x in locs
        ):
            # minor ticks are subsample from ideal, so no label
            return None
        if len(locs) < self._minor_threshold:
            if len(locs) < self._minor_number:
                self._labelled.update(locs)
            else:
                # we do not have a lot of minor ticks, so only few decades are
                # displayed, then we choose some (spaced) minor ticks to label.
                # Only minor ticks are known, we assume it is sufficient to
                # choice which ticks are displayed.
                # For each ticks we compute the distance between the ticks and
                # the previous, and between the ticks and the next one. Ticks
                # with smallest minimum are chosen. As tiebreak, the ticks
                # with smallest sum is chosen.
                diff = np.diff(-np.log(1 / self.locs - 1))
                space_pessimistic = np.minimum(
                    np.concatenate(((np.inf,), diff)),
                    np.concatenate((diff, (np.inf,))),
                )
                space_sum = (
                    np.concatenate(((0,), diff))
                    + np.concatenate((diff, (0,)))
                )
                good_minor = sorted(
                    range(len(self.locs)),
                    key=lambda i: (space_pessimistic[i], space_sum[i]),
                )[-self._minor_number:]
                self._labelled.update(locs[i] for i in good_minor)

    def _format_value(self, x, locs, sci_notation=True):
        if sci_notation:
            exponent = math.floor(np.log10(x))
            min_precision = 0
        else:
            exponent = 0
            min_precision = 1
        value = x * 10 ** (-exponent)
        if len(locs) < 2:
            precision = min_precision
        else:
            diff = np.sort(np.abs(locs - x))[1]
            precision = -np.log10(diff) + exponent
            precision = (
                int(np.round(precision))
                if _is_close_to_int(precision)
                else math.ceil(precision)
            )
            if precision < min_precision:
                precision = min_precision
        mantissa = r"%.*f" % (precision, value)
        if not sci_notation:
            return mantissa
        s = r"%s\cdot10^{%d}" % (mantissa, exponent)
        return s

    def _one_minus(self, s):
        if self._use_overline:
            return r"\overline{%s}" % s
        else:
            return f"1-{s}"

    def __call__(self, x, pos=None):
        if self._minor and x not in self._labelled:
            return ""
        if x <= 0 or x >= 1:
            return ""
        if _is_close_to_int(2 * x) and round(2 * x) == 1:
            s = self._one_half
        elif x < 0.5 and _is_decade(x, rtol=1e-7):
            exponent = round(math.log10(x))
            s = "10^{%d}" % exponent
        elif x > 0.5 and _is_decade(1 - x, rtol=1e-7):
            exponent = round(math.log10(1 - x))
            s = self._one_minus("10^{%d}" % exponent)
        elif x < 0.1:
            s = self._format_value(x, self.locs)
        elif x > 0.9:
            s = self._one_minus(self._format_value(1-x, 1-self.locs))
        else:
            s = self._format_value(x, self.locs, sci_notation=False)
        return r"$\mathdefault{%s}$" % s

    def format_data_short(self, value):
        # docstring inherited
        # Thresholds chosen to use scientific notation iff exponent <= -2.
        if value < 0.1:
            return f"{value:e}"
        if value < 0.9:
            return f"{value:f}"
        return f"1-{1 - value:e}"


class EngFormatter(ScalarFormatter):
    """
    Format axis values using engineering prefixes to represent powers
    of 1000, plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -30: "q",
        -27: "r",
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
         -9: "n",
         -6: "\N{MICRO SIGN}",
         -3: "m",
          0: "",
          3: "k",
          6: "M",
          9: "G",
         12: "T",
         15: "P",
         18: "E",
         21: "Z",
         24: "Y",
         27: "R",
         30: "Q"
    }

    def __init__(self, unit="", places=None, sep=" ", *, usetex=None,
                 useMathText=None, useOffset=False):
        r"""
        Parameters
        ----------
        unit : str, default: ""
            Unit symbol to use, suitable for use with single-letter
            representations of powers of 1000. For example, 'Hz' or 'm'.

        places : int, default: None
            Precision with which to display the number, specified in
            digits after the decimal point (there will be between one
            and three digits before the decimal point). If it is None,
            the formatting falls back to the floating point format '%g',
            which displays up to 6 *significant* digits, i.e. the equivalent
            value for *places* varies between 0 and 5 (inclusive).

        sep : str, default: " "
            Separator used between the value and the prefix/unit. For
            example, one get '3.14 mV' if ``sep`` is " " (default) and
            '3.14mV' if ``sep`` is "". Besides the default behavior, some
            other useful options may be:

            * ``sep=""`` to append directly the prefix/unit to the value;
            * ``sep="\N{THIN SPACE}"`` (``U+2009``);
            * ``sep="\N{NARROW NO-BREAK SPACE}"`` (``U+202F``);
            * ``sep="\N{NO-BREAK SPACE}"`` (``U+00A0``).

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            numbers in the formatter.

        useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
            To enable/disable the use mathtext for rendering the numbers in
            the formatter.
        useOffset : bool or float, default: False
            Whether to use offset notation with :math:`10^{3*N}` based prefixes.
            This features allows showing an offset with standard SI order of
            magnitude prefix near the axis. Offset is computed similarly to
            how `ScalarFormatter` computes it internally, but here you are
            guaranteed to get an offset which will make the tick labels exceed
            3 digits. See also `.set_useOffset`.

            .. versionadded:: 3.10
        """
        self.unit = unit
        self.places = places
        self.sep = sep
        super().__init__(
            useOffset=useOffset,
            useMathText=useMathText,
            useLocale=False,
            usetex=usetex,
        )

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.

        If there is no currently offset in the data, it returns the best
        engineering formatting that fits the given argument, independently.
        """
        if len(self.locs) == 0 or self.offset == 0:
            return self.fix_minus(self.format_data(x))
        else:
            xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
            if abs(xp) < 1e-8:
                xp = 0
            return self._format_maybe_minus_and_locale(self.format, xp)

    def set_locs(self, locs):
        # docstring inherited
        self.locs = locs
        if len(self.locs) > 0:
            vmin, vmax = sorted(self.axis.get_view_interval())
            if self._useOffset:
                self._compute_offset()
                if self.offset != 0:
                    # We don't want to use the offset computed by
                    # self._compute_offset because it rounds the offset unaware
                    # of our engineering prefixes preference, and this can
                    # cause ticks with 4+ digits to appear. These ticks are
                    # slightly less readable, so if offset is justified
                    # (decided by self._compute_offset) we set it to better
                    # value:
                    self.offset = round((vmin + vmax)/2, 3)
            # Use log1000 to use engineers' oom standards
            self.orderOfMagnitude = math.floor(math.log(vmax - vmin, 1000))*3
            self._set_format()

    # Simplify a bit ScalarFormatter.get_offset: We always want to use
    # self.format_data. Also we want to return a non-empty string only if there
    # is an offset, no matter what is self.orderOfMagnitude. If there _is_ an
    # offset, self.orderOfMagnitude is consulted. This behavior is verified
    # in `test_ticker.py`.
    def get_offset(self):
        # docstring inherited
        if len(self.locs) == 0:
            return ''
        if self.offset:
            offsetStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0:
                    offsetStr = '+' + offsetStr
            sciNotStr = self.format_data(10 ** self.orderOfMagnitude)
            if self._useMathText or self._usetex:
                if sciNotStr != '':
                    sciNotStr = r'\times%s' % sciNotStr
                s = f'${sciNotStr}{offsetStr}$'
            else:
                s = sciNotStr + offsetStr
            return self.fix_minus(s)
        return ''

    def format_eng(self, num):
        """Alias to EngFormatter.format_data"""
        return self.format_data(num)

    def format_data(self, value):
        """
        Format a number in engineering notation, appending a letter
        representing the power of 1000 of the original number.
        Some examples:

        >>> format_data(0)        # for self.places = 0
        '0'

        >>> format_data(1000000)  # for self.places = 1
        '1.0 M'

        >>> format_data(-1e-6)  # for self.places = 2
        '-1.00 \N{MICRO SIGN}'
        """
        sign = 1
        fmt = "g" if self.places is None else f".{self.places:d}f"

        if value < 0:
            sign = -1
            value = -value

        if value != 0:
            pow10 = int(math.floor(math.log10(value) / 3) * 3)
        else:
            pow10 = 0
            # Force value to zero, to avoid inconsistencies like
            # format_eng(-0) = "0" and format_eng(0.0) = "0"
            # but format_eng(-0.0) = "-0.0"
            value = 0.0

        pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))

        mant = sign * value / (10.0 ** pow10)
        # Taking care of the cases like 999.9..., which may be rounded to 1000
        # instead of 1 k.  Beware of the corner case of values that are beyond
        # the range of SI prefixes (i.e. > 'Y').
        if (abs(float(format(mant, fmt))) >= 1000
                and pow10 < max(self.ENG_PREFIXES)):
            mant /= 1000
            pow10 += 3

        unit_prefix = self.ENG_PREFIXES[int(pow10)]
        if self.unit or unit_prefix:
            suffix = f"{self.sep}{unit_prefix}{self.unit}"
        else:
            suffix = ""
        if self._usetex or self._useMathText:
            return f"${mant:{fmt}}${suffix}"
        else:
            return f"{mant:{fmt}}{suffix}"


class PercentFormatter(Formatter):
    """
    Format numbers as a percentage.

    Parameters
    ----------
    xmax : float
        Determines how the number is converted into a percentage.
        *xmax* is the data value that corresponds to 100%.
        Percentages are computed as ``x / xmax * 100``. So if the data is
        already scaled to be percentages, *xmax* will be 100. Another common
        situation is where *xmax* is 1.0.

    decimals : None or int
        The number of decimal places to place after the point.
        If *None* (the default), the number will be computed automatically.

    symbol : str or None
        A string that will be appended to the label. It may be
        *None* or empty to indicate that no symbol should be used. LaTeX
        special characters are escaped in *symbol* whenever latex mode is
        enabled, unless *is_latex* is *True*.

    is_latex : bool
        If *False*, reserved LaTeX characters in *symbol* will be escaped.
    """
    def __init__(self, xmax=100, decimals=None, symbol='%', is_latex=False):
        self.xmax = xmax + 0.0
        self.decimals = decimals
        self._symbol = symbol
        self._is_latex = is_latex

    def __call__(self, x, pos=None):
        """Format the tick as a percentage with the appropriate scaling."""
        ax_min, ax_max = self.axis.get_view_interval()
        display_range = abs(ax_max - ax_min)
        return self.fix_minus(self.format_pct(x, display_range))

    def format_pct(self, x, display_range):
        """
        Format the number as a percentage number with the correct
        number of decimals and adds the percent symbol, if any.

        If ``self.decimals`` is `None`, the number of digits after the
        decimal point is set based on the *display_range* of the axis
        as follows:

        ============= ======== =======================
        display_range decimals sample
        ============= ======== =======================
        >50           0        ``x = 34.5`` => 35%
        >5            1        ``x = 34.5`` => 34.5%
        >0.5          2        ``x = 34.5`` => 34.50%
        ...           ...      ...
        ============= ======== =======================

        This method will not be very good for tiny axis ranges or
        extremely large ones. It assumes that the values on the chart
        are percentages displayed on a reasonable scale.
        """
        x = self.convert_to_pct(x)
        if self.decimals is None:
            # conversion works because display_range is a difference
            scaled_range = self.convert_to_pct(display_range)
            if scaled_range <= 0:
                decimals = 0
            else:
                # Luckily Python's built-in ceil rounds to +inf, not away from
                # zero. This is very important since the equation for decimals
                # starts out as `scaled_range > 0.5 * 10**(2 - decimals)`
                # and ends up with `decimals > 2 - log10(2 * scaled_range)`.
                decimals = math.ceil(2.0 - math.log10(2.0 * scaled_range))
                if decimals > 5:
                    decimals = 5
                elif decimals < 0:
                    decimals = 0
        else:
            decimals = self.decimals
        s = f'{x:0.{int(decimals)}f}'

        return s + self.symbol

    def convert_to_pct(self, x):
        return 100.0 * (x / self.xmax)

    @property
    def symbol(self):
        r"""
        The configured percent symbol as a string.

        If LaTeX is enabled via :rc:`text.usetex`, the special characters
        ``{'#', '$', '%', '&', '~', '_', '^', '\', '{', '}'}`` are
        automatically escaped in the string.
        """
        symbol = self._symbol
        if not symbol:
            symbol = ''
        elif not self._is_latex and mpl.rcParams['text.usetex']:
            # Source: http://www.personal.ceu.hu/tex/specchar.htm
            # Backslash must be first for this to work correctly since
            # it keeps getting added in
            for spec in r'\#$%&~_^{}':
                symbol = symbol.replace(spec, '\\' + spec)
        return symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol


class Locator(TickHelper):
    """
    Determine tick locations.

    Note that the same locator should not be used across multiple
    `~matplotlib.axis.Axis` because the locator stores references to the Axis
    data and view limits.
    """

    # Some automatic tick locators can generate so many ticks they
    # kill the machine when you try and render them.
    # This parameter is set to cause locators to raise an error if too
    # many ticks are generated.
    MAXTICKS = 1000

    def tick_values(self, vmin, vmax):
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated ``axis`` simply call
            the Locator instance::

                >>> print(type(loc))
                <type 'Locator'>
                >>> print(loc())
                [1, 2, 3, 4]

        """
        raise NotImplementedError('Derived must override')

    def set_params(self, **kwargs):
        """
        Do nothing, and raise a warning. Any locator class not supporting the
        set_params() function will call this.
        """
        _api.warn_external(
            "'set_params()' not defined for locator of type " +
            str(type(self)))

    def __call__(self):
        """Return the locations of the ticks."""
        # note: some locators return data limits, other return view limits,
        # hence there is no *one* interface to call self.tick_values.
        raise NotImplementedError('Derived must override')

    def raise_if_exceeds(self, locs):
        """
        Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.

        This is intended to be called immediately before returning *locs* from
        ``__call__`` to inform users in case their Locator returns a huge
        number of ticks, causing Matplotlib to run out of memory.

        The "strange" name of this method dates back to when it would raise an
        exception instead of emitting a log.
        """
        if len(locs) >= self.MAXTICKS:
            _log.warning(
                "Locator attempting to generate %s ticks ([%s, ..., %s]), "
                "which exceeds Locator.MAXTICKS (%s).",
                len(locs), locs[0], locs[-1], self.MAXTICKS)
        return locs

    def nonsingular(self, v0, v1):
        """
        Adjust a range as needed to avoid singularities.

        This method gets called during autoscaling, with ``(v0, v1)`` set to
        the data limits on the Axes if the Axes contains any data, or
        ``(-inf, +inf)`` if not.

        - If ``v0 == v1`` (possibly up to some floating point slop), this
          method returns an expanded interval around this value.
        - If ``(v0, v1) == (-inf, +inf)``, this method returns appropriate
          default view limits.
        - Otherwise, ``(v0, v1)`` is returned without modification.
        """
        return mtransforms.nonsingular(v0, v1, expander=.05)

    def view_limits(self, vmin, vmax):
        """
        Select a scale for the range from vmin to vmax.

        Subclasses should override this method to change locator behaviour.
        """
        return mtransforms.nonsingular(vmin, vmax)


class IndexLocator(Locator):
    """
    Place ticks at every nth point plotted.

    IndexLocator assumes index plotting; i.e., that the ticks are placed at integer
    values in the range between 0 and len(data) inclusive.
    """
    def __init__(self, base, offset):
        """Place ticks every *base* data point, starting at *offset*."""
        self._base = base
        self.offset = offset

    def set_params(self, base=None, offset=None):
        """Set parameters within this locator"""
        if base is not None:
            self._base = base
        if offset is not None:
            self.offset = offset

    def __call__(self):
        """Return the locations of the ticks"""
        dmin, dmax = self.axis.get_data_interval()
        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        return self.raise_if_exceeds(
            np.arange(vmin + self.offset, vmax + 1, self._base))


class FixedLocator(Locator):
    r"""
    Place ticks at a set of fixed values.

    If *nbins* is None ticks are placed at all values. Otherwise, the *locs* array of
    possible positions will be subsampled to keep the number of ticks
    :math:`\leq nbins + 1`. The subsampling will be done to include the smallest
    absolute value; for example, if zero is included in the array of possibilities, then
    it will be included in the chosen ticks.
    """

    def __init__(self, locs, nbins=None):
        self.locs = np.asarray(locs)
        _api.check_shape((None,), locs=self.locs)
        self.nbins = max(nbins, 2) if nbins is not None else None

    def set_params(self, nbins=None):
        """Set parameters within this locator."""
        if nbins is not None:
            self.nbins = nbins

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        if self.nbins is None:
            return self.locs
        step = max(int(np.ceil(len(self.locs) / self.nbins)), 1)
        ticks = self.locs[::step]
        for i in range(1, step):
            ticks1 = self.locs[i::step]
            if np.abs(ticks1).min() < np.abs(ticks).min():
                ticks = ticks1
        return self.raise_if_exceeds(ticks)


class NullLocator(Locator):
    """
    No ticks
    """

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are Null, vmin and vmax are not used in this
            method.
        """
        return []


class LinearLocator(Locator):
    """
    Place ticks at evenly spaced values.

    The first time this function is called it will try to set the
    number of ticks to make a nice tick partitioning.  Thereafter, the
    number of ticks will be fixed so that interactive navigation will
    be nice

    """
    def __init__(self, numticks=None, presets=None):
        """
        Parameters
        ----------
        numticks : int or None, default None
            Number of ticks. If None, *numticks* = 11.
        presets : dict or None, default: None
            Dictionary mapping ``(vmin, vmax)`` to an array of locations.
            Overrides *numticks* if there is an entry for the current
            ``(vmin, vmax)``.
        """
        self.numticks = numticks
        if presets is None:
            self.presets = {}
        else:
            self.presets = presets

    @property
    def numticks(self):
        # Old hard-coded default.
        return self._numticks if self._numticks is not None else 11

    @numticks.setter
    def numticks(self, numticks):
        self._numticks = numticks

    def set_params(self, numticks=None, presets=None):
        """Set parameters within this locator."""
        if presets is not None:
            self.presets = presets
        if numticks is not None:
            self.numticks = numticks

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)

        if (vmin, vmax) in self.presets:
            return self.presets[(vmin, vmax)]

        if self.numticks == 0:
            return []
        ticklocs = np.linspace(vmin, vmax, self.numticks)

        return self.raise_if_exceeds(ticklocs)

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if vmin == vmax:
            vmin -= 1
            vmax += 1

        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            exponent, remainder = divmod(
                math.log10(vmax - vmin), math.log10(max(self.numticks - 1, 1)))
            exponent -= (remainder < .5)
            scale = max(self.numticks - 1, 1) ** (-exponent)
            vmin = math.floor(scale * vmin) / scale
            vmax = math.ceil(scale * vmax) / scale

        return mtransforms.nonsingular(vmin, vmax)


class MultipleLocator(Locator):
    """
    Place ticks at every integer multiple of a base plus an offset.
    """

    def __init__(self, base=1.0, offset=0.0):
        """
        Parameters
        ----------
        base : float > 0, default: 1.0
            Interval between ticks.
        offset : float, default: 0.0
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        self._edge = _Edge_integer(base, 0)
        self._offset = offset

    def set_params(self, base=None, offset=None):
        """
        Set parameters within this locator.

        Parameters
        ----------
        base : float > 0, optional
            Interval between ticks.
        offset : float, optional
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        if base is not None:
            self._edge = _Edge_integer(base, 0)
        if offset is not None:
            self._offset = offset

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        step = self._edge.step
        vmin -= self._offset
        vmax -= self._offset
        vmin = self._edge.ge(vmin) * step
        n = (vmax - vmin + 0.001 * step) // step
        locs = vmin - step + np.arange(n + 3) * step + self._offset
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest tick values that contain the data.
        """
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = self._edge.le(dmin - self._offset) * self._edge.step + self._offset
            vmax = self._edge.ge(dmax - self._offset) * self._edge.step + self._offset
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        else:
            vmin = dmin
            vmax = dmax

        return mtransforms.nonsingular(vmin, vmax)


def scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)  # > 0 as nonsingular is called before.
    meanv = (vmax + vmin) / 2
    if abs(meanv) / dv < threshold:
        offset = 0
    else:
        offset = math.copysign(10 ** (math.log10(abs(meanv)) // 1), meanv)
    scale = 10 ** (math.log10(dv / n) // 1)
    return scale, offset


class _Edge_integer:
    """
    Helper for `.MaxNLocator`, `.MultipleLocator`, etc.

    Take floating-point precision limitations into account when calculating
    tick locations as integer multiples of a step.
    """
    def __init__(self, step, offset):
        """
        Parameters
        ----------
        step : float > 0
            Interval between ticks.
        offset : float
            Offset subtracted from the data limits prior to calculating tick
            locations.
        """
        if step <= 0:
            raise ValueError("'step' must be positive")
        self.step = step
        self._offset = abs(offset)

    def closeto(self, ms, edge):
        # Allow more slop when the offset is large compared to the step.
        if self._offset > 0:
            digits = np.log10(self._offset / self.step)
            tol = max(1e-10, 10 ** (digits - 12))
            tol = min(0.4999, tol)
        else:
            tol = 1e-10
        return abs(ms - edge) < tol

    def le(self, x):
        """Return the largest n: n*step <= x."""
        d, m = divmod(x, self.step)
        if self.closeto(m / self.step, 1):
            return d + 1
        return d

    def ge(self, x):
        """Return the smallest n: n*step >= x."""
        d, m = divmod(x, self.step)
        if self.closeto(m / self.step, 0):
            return d
        return d + 1


class MaxNLocator(Locator):
    """
    Place evenly spaced ticks, with a cap on the total number of ticks.

    Finds nice tick locations with no more than :math:`nbins + 1` ticks being within the
    view limits. Locations beyond the limits are added to support autoscaling.
    """
    default_params = dict(nbins=10,
                          steps=None,
                          integer=False,
                          symmetric=False,
                          prune=None,
                          min_n_ticks=2)

    def __init__(self, nbins=None, **kwargs):
        """
        Parameters
        ----------
        nbins : int or 'auto', default: 10
            Maximum number of intervals; one less than max number of
            ticks.  If the string 'auto', the number of bins will be
            automatically determined based on the length of the axis.

        steps : array-like, optional
            Sequence of acceptable tick multiples, starting with 1 and
            ending with 10. For example, if ``steps=[1, 2, 4, 5, 10]``,
            ``20, 40, 60`` or ``0.4, 0.6, 0.8`` would be possible
            sets of ticks because they are multiples of 2.
            ``30, 60, 90`` would not be generated because 3 does not
            appear in this example list of steps.

        integer : bool, default: False
            If True, ticks will take only integer values, provided at least
            *min_n_ticks* integers are found within the view limits.

        symmetric : bool, default: False
            If True, autoscaling will result in a range symmetric about zero.

        prune : {'lower', 'upper', 'both', None}, default: None
            Remove the 'lower' tick, the 'upper' tick, or ticks on 'both' sides
            *if they fall exactly on an axis' edge* (this typically occurs when
            :rc:`axes.autolimit_mode` is 'round_numbers').  Removing such ticks
            is mostly useful for stacked or ganged plots, where the upper tick
            of an Axes overlaps with the lower tick of the axes above it.

        min_n_ticks : int, default: 2
            Relax *nbins* and *integer* constraints if necessary to obtain
            this minimum number of ticks.
        """
        if nbins is not None:
            kwargs['nbins'] = nbins
        self.set_params(**{**self.default_params, **kwargs})

    @staticmethod
    def _validate_steps(steps):
        if not np.iterable(steps):
            raise ValueError('steps argument must be an increasing sequence '
                             'of numbers between 1 and 10 inclusive')
        steps = np.asarray(steps)
        if np.any(np.diff(steps) <= 0) or steps[-1] > 10 or steps[0] < 1:
            raise ValueError('steps argument must be an increasing sequence '
                             'of numbers between 1 and 10 inclusive')
        if steps[0] != 1:
            steps = np.concatenate([[1], steps])
        if steps[-1] != 10:
            steps = np.concatenate([steps, [10]])
        return steps

    @staticmethod
    def _staircase(steps):
        # Make an extended staircase within which the needed step will be
        # found.  This is probably much larger than necessary.
        return np.concatenate([0.1 * steps[:-1], steps, [10 * steps[1]]])

    def set_params(self, **kwargs):
        """
        Set parameters for this locator.

        Parameters
        ----------
        nbins : int or 'auto', optional
            see `.MaxNLocator`
        steps : array-like, optional
            see `.MaxNLocator`
        integer : bool, optional
            see `.MaxNLocator`
        symmetric : bool, optional
            see `.MaxNLocator`
        prune : {'lower', 'upper', 'both', None}, optional
            see `.MaxNLocator`
        min_n_ticks : int, optional
            see `.MaxNLocator`
        """
        if 'nbins' in kwargs:
            self._nbins = kwargs.pop('nbins')
            if self._nbins != 'auto':
                self._nbins = int(self._nbins)
        if 'symmetric' in kwargs:
            self._symmetric = kwargs.pop('symmetric')
        if 'prune' in kwargs:
            prune = kwargs.pop('prune')
            _api.check_in_list(['upper', 'lower', 'both', None], prune=prune)
            self._prune = prune
        if 'min_n_ticks' in kwargs:
            self._min_n_ticks = max(1, kwargs.pop('min_n_ticks'))
        if 'steps' in kwargs:
            steps = kwargs.pop('steps')
            if steps is None:
                self._steps = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
            else:
                self._steps = self._validate_steps(steps)
            self._extended_steps = self._staircase(self._steps)
        if 'integer' in kwargs:
            self._integer = kwargs.pop('integer')
        if kwargs:
            raise _api.kwarg_error("set_params", kwargs)

    def _raw_ticks(self, vmin, vmax):
        """
        Generate a list of tick locations including the range *vmin* to
        *vmax*.  In some applications, one or both of the end locations
        will not be needed, in which case they are trimmed off
        elsewhere.
        """
        if self._nbins == 'auto':
            if self.axis is not None:
                nbins = np.clip(self.axis.get_tick_space(),
                                max(1, self._min_n_ticks - 1), 9)
            else:
                nbins = 9
        else:
            nbins = self._nbins

        scale, offset = scale_range(vmin, vmax, nbins)
        _vmin = vmin - offset
        _vmax = vmax - offset
        steps = self._extended_steps * scale
        if self._integer:
            # For steps > 1, keep only integer values.
            igood = (steps < 1) | (np.abs(steps - np.round(steps)) < 0.001)
            steps = steps[igood]

        raw_step = ((_vmax - _vmin) / nbins)
        if hasattr(self.axis, "axes") and self.axis.axes.name == '3d':
            # Due to the change in automargin behavior in mpl3.9, we need to
            # adjust the raw step to match the mpl3.8 appearance. The zoom
            # factor of 2/48, gives us the 23/24 modifier.
            raw_step = raw_step * 23/24
        large_steps = steps >= raw_step
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            # Classic round_numbers mode may require a larger step.
            # Get first multiple of steps that are <= _vmin
            floored_vmins = (_vmin // steps) * steps
            floored_vmaxs = floored_vmins + steps * nbins
            large_steps = large_steps & (floored_vmaxs >= _vmax)

        # Find index of smallest large step
        if any(large_steps):
            istep = np.nonzero(large_steps)[0][0]
        else:
            istep = len(steps) - 1

        # Start at smallest of the steps greater than the raw step, and check
        # if it provides enough ticks. If not, work backwards through
        # smaller steps until one is found that provides enough ticks.
        for step in steps[:istep+1][::-1]:

            if (self._integer and
                    np.floor(_vmax) - np.ceil(_vmin) >= self._min_n_ticks - 1):
                step = max(1, step)
            best_vmin = (_vmin // step) * step

            # Find tick locations spanning the vmin-vmax range, taking into
            # account degradation of precision when there is a large offset.
            # The edge ticks beyond vmin and/or vmax are needed for the
            # "round_numbers" autolimit mode.
            edge = _Edge_integer(step, offset)
            low = edge.le(_vmin - best_vmin)
            high = edge.ge(_vmax - best_vmin)
            ticks = np.arange(low, high + 1) * step + best_vmin
            # Count only the ticks that will be displayed.
            nticks = ((ticks <= _vmax) & (ticks >= _vmin)).sum()
            if nticks >= self._min_n_ticks:
                break
        return ticks + offset

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self._symmetric:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        vmin, vmax = mtransforms.nonsingular(
            vmin, vmax, expander=1e-13, tiny=1e-14)
        locs = self._raw_ticks(vmin, vmax)

        prune = self._prune
        if prune == 'lower':
            locs = locs[1:]
        elif prune == 'upper':
            locs = locs[:-1]
        elif prune == 'both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        if self._symmetric:
            dmax = max(abs(dmin), abs(dmax))
            dmin = -dmax

        dmin, dmax = mtransforms.nonsingular(
            dmin, dmax, expander=1e-12, tiny=1e-13)

        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            return self._raw_ticks(dmin, dmax)[[0, -1]]
        else:
            return dmin, dmax


def _is_decade(x, *, base=10, rtol=None):
    """Return True if *x* is an integer power of *base*."""
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(abs(x)) / np.log(base)
    if rtol is None:
        return np.isclose(lx, np.round(lx))
    else:
        return np.isclose(lx, np.round(lx), rtol=rtol)


def _decade_less_equal(x, base):
    """
    Return the largest integer power of *base* that's less or equal to *x*.

    If *x* is negative, the exponent will be *greater*.
    """
    return (x if x == 0 else
            -_decade_greater_equal(-x, base) if x < 0 else
            base ** np.floor(np.log(x) / np.log(base)))


def _decade_greater_equal(x, base):
    """
    Return the smallest integer power of *base* that's greater or equal to *x*.

    If *x* is negative, the exponent will be *smaller*.
    """
    return (x if x == 0 else
            -_decade_less_equal(-x, base) if x < 0 else
            base ** np.ceil(np.log(x) / np.log(base)))


def _decade_less(x, base):
    """
    Return the largest integer power of *base* that's less than *x*.

    If *x* is negative, the exponent will be *greater*.
    """
    if x < 0:
        return -_decade_greater(-x, base)
    less = _decade_less_equal(x, base)
    if less == x:
        less /= base
    return less


def _decade_greater(x, base):
    """
    Return the smallest integer power of *base* that's greater than *x*.

    If *x* is negative, the exponent will be *smaller*.
    """
    if x < 0:
        return -_decade_less(-x, base)
    greater = _decade_greater_equal(x, base)
    if greater == x:
        greater *= base
    return greater


def _is_close_to_int(x):
    return math.isclose(x, round(x))


class LogLocator(Locator):
    """
    Place logarithmically spaced ticks.

    Places ticks at the values ``subs[j] * base**i``.
    """

    def __init__(self, base=10.0, subs=(1.0,), *, numticks=None):
        """
        Parameters
        ----------
        base : float, default: 10.0
            The base of the log used, so major ticks are placed at ``base**n``, where
            ``n`` is an integer.
        subs : None or {'auto', 'all'} or sequence of float, default: (1.0,)
            Gives the multiples of integer powers of the base at which to place ticks.
            The default of ``(1.0, )`` places ticks only at integer powers of the base.
            Permitted string values are ``'auto'`` and ``'all'``. Both of these use an
            algorithm based on the axis view limits to determine whether and how to put
            ticks between integer powers of the base:
            - ``'auto'``: Ticks are placed only between integer powers.
            - ``'all'``: Ticks are placed between *and* at integer powers.
            - ``None``: Equivalent to ``'auto'``.
        numticks : None or int, default: None
            The maximum number of ticks to allow on a given axis. The default of
            ``None`` will try to choose intelligently as long as this Locator has
            already been assigned to an axis using `~.axis.Axis.get_tick_space`, but
            otherwise falls back to 9.
        """
        if numticks is None:
            if mpl.rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'
        self._base = float(base)
        self._set_subs(subs)
        self.numticks = numticks

    def set_params(self, base=None, subs=None, *, numticks=None):
        """Set parameters within this locator."""
        if base is not None:
            self._base = float(base)
        if subs is not None:
            self._set_subs(subs)
        if numticks is not None:
            self.numticks = numticks

    def _set_subs(self, subs):
        """
        Set the minor ticks for the log scaling every ``base**i*subs[j]``.
        """
        if subs is None:  # consistency with previous bad API
            self._subs = 'auto'
        elif isinstance(subs, str):
            _api.check_in_list(('all', 'auto'), subs=subs)
            self._subs = subs
        else:
            try:
                self._subs = np.asarray(subs, dtype=float)
            except ValueError as e:
                raise ValueError("subs must be None, 'all', 'auto' or "
                                 "a sequence of floats, not "
                                 f"{subs}.") from e
            if self._subs.ndim != 1:
                raise ValueError("A sequence passed to subs must be "
                                 "1-dimensional, not "
                                 f"{self._subs.ndim}-dimensional.")

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks

        b = self._base
        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError(
                    "Data has no positive values, and therefore cannot be log-scaled.")

        _log.debug('vmin %s vmax %s', vmin, vmax)

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        log_vmin = math.log(vmin) / math.log(b)
        log_vmax = math.log(vmax) / math.log(b)

        numdec = math.floor(log_vmax) - math.ceil(log_vmin)

        if isinstance(self._subs, str):
            if numdec > 10 or b < 3:
                if self._subs == 'auto':
                    return np.array([])  # no minor or major ticks
                else:
                    subs = np.array([1.0])  # major ticks
            else:
                _first = 2.0 if self._subs == 'auto' else 1.0
                subs = np.arange(_first, b)
        else:
            subs = self._subs

        # Get decades between major ticks.
        stride = (max(math.ceil(numdec / (numticks - 1)), 1)
                  if mpl.rcParams['_internal.classic_mode'] else
                  numdec // numticks + 1)

        # if we have decided that the stride is as big or bigger than
        # the range, clip the stride back to the available range - 1
        # with a floor of 1.  This prevents getting axis with only 1 tick
        # visible.
        if stride >= numdec:
            stride = max(1, numdec - 1)

        # Does subs include anything other than 1?  Essentially a hack to know
        # whether we're a major or a minor locator.
        have_subs = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)

        decades = np.arange(math.floor(log_vmin) - stride,
                            math.ceil(log_vmax) + 2 * stride, stride)

        if have_subs:
            if stride == 1:
                ticklocs = np.concatenate(
                    [subs * decade_start for decade_start in b ** decades])
            else:
                ticklocs = np.array([])
        else:
            ticklocs = b ** decades

        _log.debug('ticklocs %r', ticklocs)
        if (len(subs) > 1
                and stride == 1
                and ((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1):
            # If we're a minor locator *that expects at least two ticks per
            # decade* and the major locator stride is 1 and there's no more
            # than one minor tick, switch to AutoLocator.
            return AutoLocator().tick_values(vmin, vmax)
        else:
            return self.raise_if_exceeds(ticklocs)

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        b = self._base

        vmin, vmax = self.nonsingular(vmin, vmax)

        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = _decade_less_equal(vmin, b)
            vmax = _decade_greater_equal(vmax, b)

        return vmin, vmax

    def nonsingular(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 1, 10  # Initial range, no data plotted yet.
        elif vmax <= 0:
            _api.warn_external(
                "Data has no positive values, and therefore cannot be "
                "log-scaled.")
            vmin, vmax = 1, 10
        else:
            # Consider shared axises
            minpos = min(axis.get_minpos() for axis in self.axis._get_shared_axis())
            if not np.isfinite(minpos):
                minpos = 1e-300  # This should never take effect.
            if vmin <= 0:
                vmin = minpos
            if vmin == vmax:
                vmin = _decade_less(vmin, self._base)
                vmax = _decade_greater(vmax, self._base)
        return vmin, vmax


class SymmetricalLogLocator(Locator):
    """
    Place ticks spaced linearly near zero and spaced logarithmically beyond a threshold.
    """

    def __init__(self, transform=None, subs=None, linthresh=None, base=None):
        """
        Parameters
        ----------
        transform : `~.scale.SymmetricalLogTransform`, optional
            If set, defines the *base* and *linthresh* of the symlog transform.
        base, linthresh : float, optional
            The *base* and *linthresh* of the symlog transform, as documented
            for `.SymmetricalLogScale`.  These parameters are only used if
            *transform* is not set.
        subs : sequence of float, default: [1]
            The multiples of integer powers of the base where ticks are placed,
            i.e., ticks are placed at
            ``[sub * base**i for i in ... for sub in subs]``.

        Notes
        -----
        Either *transform*, or both *base* and *linthresh*, must be given.
        """
        if transform is not None:
            self._base = transform.base
            self._linthresh = transform.linthresh
        elif linthresh is not None and base is not None:
            self._base = base
            self._linthresh = linthresh
        else:
            raise ValueError("Either transform, or both linthresh "
                             "and base, must be provided.")
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """Set parameters within this locator."""
        if numticks is not None:
            self.numticks = numticks
        if subs is not None:
            self._subs = subs

    def __call__(self):
        """Return the locations of the ticks."""
        # Note, these are untransformed coordinates
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        linthresh = self._linthresh

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # The domain is divided into three sections, only some of
        # which may actually be present.
        #
        # <======== -t ==0== t ========>
        # aaaaaaaaa    bbbbb   ccccccccc
        #
        # a) and c) will have ticks at integral log positions.  The
        # number of ticks needs to be reduced if there are more
        # than self.numticks of them.
        #
        # b) has a tick at 0 and only 0 (we assume t is a small
        # number, and the linear segment is just an implementation
        # detail and not interesting.)
        #
        # We could also add ticks at t, but that seems to usually be
        # uninteresting.
        #
        # "simple" mode is when the range falls entirely within [-t, t]
        #  -- it should just display (vmin, 0, vmax)
        if -linthresh <= vmin < vmax <= linthresh:
            # only the linear range is present
            return sorted({vmin, 0, vmax})

        # Lower log range is present
        has_a = (vmin < -linthresh)
        # Upper log range is present
        has_c = (vmax > linthresh)

        # Check if linear range is present
        has_b = (has_a and vmax > -linthresh) or (has_c and vmin < linthresh)

        base = self._base

        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(base))
            hi = np.ceil(np.log(hi) / np.log(base))
            return lo, hi

        # Calculate all the ranges, so we can determine striding
        a_lo, a_hi = (0, 0)
        if has_a:
            a_upper_lim = min(-linthresh, vmax)
            a_lo, a_hi = get_log_range(abs(a_upper_lim), abs(vmin) + 1)

        c_lo, c_hi = (0, 0)
        if has_c:
            c_lower_lim = max(linthresh, vmin)
            c_lo, c_hi = get_log_range(c_lower_lim, vmax + 1)

        # Calculate the total number of integer exponents in a and c ranges
        total_ticks = (a_hi - a_lo) + (c_hi - c_lo)
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)

        decades = []
        if has_a:
            decades.extend(-1 * (base ** (np.arange(a_lo, a_hi,
                                                    stride)[::-1])))

        if has_b:
            decades.append(0.0)

        if has_c:
            decades.extend(base ** (np.arange(c_lo, c_hi, stride)))

        subs = np.asarray(self._subs)

        if len(subs) > 1 or subs[0] != 1.0:
            ticklocs = []
            for decade in decades:
                if decade == 0:
                    ticklocs.append(decade)
                else:
                    ticklocs.extend(subs * decade)
        else:
            ticklocs = decades

        return self.raise_if_exceeds(np.array(ticklocs))

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        b = self._base
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = _decade_less_equal(vmin, b)
            vmax = _decade_greater_equal(vmax, b)
            if vmin == vmax:
                vmin = _decade_less(vmin, b)
                vmax = _decade_greater(vmax, b)

        return mtransforms.nonsingular(vmin, vmax)


class AsinhLocator(Locator):
    """
    Place ticks spaced evenly on an inverse-sinh scale.

    Generally used with the `~.scale.AsinhScale` class.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    """
    def __init__(self, linear_width, numticks=11, symthresh=0.2,
                 base=10, subs=None):
        """
        Parameters
        ----------
        linear_width : float
            The scale parameter defining the extent
            of the quasi-linear region.
        numticks : int, default: 11
            The approximate number of major ticks that will fit
            along the entire axis
        symthresh : float, default: 0.2
            The fractional threshold beneath which data which covers
            a range that is approximately symmetric about zero
            will have ticks that are exactly symmetric.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : tuple, default: None
            Multiples of the number base, typically used
            for the minor ticks, e.g. (2, 5) when base=10.
        """
        super().__init__()
        self.linear_width = linear_width
        self.numticks = numticks
        self.symthresh = symthresh
        self.base = base
        self.subs = subs

    def set_params(self, numticks=None, symthresh=None,
                   base=None, subs=None):
        """Set parameters within this locator."""
        if numticks is not None:
            self.numticks = numticks
        if symthresh is not None:
            self.symthresh = symthresh
        if base is not None:
            self.base = base
        if subs is not None:
            self.subs = subs if len(subs) > 0 else None

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if (vmin * vmax) < 0 and abs(1 + vmax / vmin) < self.symthresh:
            # Data-range appears to be almost symmetric, so round up:
            bound = max(abs(vmin), abs(vmax))
            return self.tick_values(-bound, bound)
        else:
            return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        # Construct a set of uniformly-spaced "on-screen" locations.
        ymin, ymax = self.linear_width * np.arcsinh(np.array([vmin, vmax])
                                                    / self.linear_width)
        ys = np.linspace(ymin, ymax, self.numticks)
        zero_dev = abs(ys / (ymax - ymin))
        if ymin * ymax < 0:
            # Ensure that the zero tick-mark is included, if the axis straddles zero.
            ys = np.hstack([ys[(zero_dev > 0.5 / self.numticks)], 0.0])

        # Transform the "on-screen" grid to the data space:
        xs = self.linear_width * np.sinh(ys / self.linear_width)
        zero_xs = (ys == 0)

        # Round the data-space values to be intuitive base-n numbers, keeping track of
        # positive and negative values separately and carefully treating the zero value.
        with np.errstate(divide="ignore"):  # base ** log(0) = base ** -inf = 0.
            if self.base > 1:
                pows = (np.sign(xs)
                        * self.base ** np.floor(np.log(abs(xs)) / math.log(self.base)))
                qs = np.outer(pows, self.subs).flatten() if self.subs else pows
            else:  # No need to adjust sign(pows), as it cancels out when computing qs.
                pows = np.where(zero_xs, 1, 10**np.floor(np.log10(abs(xs))))
                qs = pows * np.round(xs / pows)
        ticks = np.array(sorted(set(qs)))

        return ticks if len(ticks) >= 2 else np.linspace(vmin, vmax, self.numticks)


class LogitLocator(MaxNLocator):
    """
    Place ticks spaced evenly on a logit scale.
    """

    def __init__(self, minor=False, *, nbins="auto"):
        """
        Parameters
        ----------
        nbins : int or 'auto', optional
            Number of ticks. Only used if minor is False.
        minor : bool, default: False
            Indicate if this locator is for minor ticks or not.
        """

        self._minor = minor
        super().__init__(nbins=nbins, steps=[1, 2, 5, 10])

    def set_params(self, minor=None, **kwargs):
        """Set parameters within this locator."""
        if minor is not None:
            self._minor = minor
        super().set_params(**kwargs)

    @property
    def minor(self):
        return self._minor

    @minor.setter
    def minor(self, value):
        self.set_params(minor=value)

    def tick_values(self, vmin, vmax):
        # dummy axis has no axes attribute
        if hasattr(self.axis, "axes") and self.axis.axes.name == "polar":
            raise NotImplementedError("Polar axis cannot be logit scaled yet")

        if self._nbins == "auto":
            if self.axis is not None:
                nbins = self.axis.get_tick_space()
                if nbins < 2:
                    nbins = 2
            else:
                nbins = 9
        else:
            nbins = self._nbins

        # We define ideal ticks with their index:
        # linscale: ... 1e-3 1e-2 1e-1 1/2 1-1e-1 1-1e-2 1-1e-3 ...
        # b-scale : ... -3   -2   -1   0   1      2      3      ...
        def ideal_ticks(x):
            return 10 ** x if x < 0 else 1 - (10 ** (-x)) if x > 0 else 0.5

        vmin, vmax = self.nonsingular(vmin, vmax)
        binf = int(
            np.floor(np.log10(vmin))
            if vmin < 0.5
            else 0
            if vmin < 0.9
            else -np.ceil(np.log10(1 - vmin))
        )
        bsup = int(
            np.ceil(np.log10(vmax))
            if vmax <= 0.5
            else 1
            if vmax <= 0.9
            else -np.floor(np.log10(1 - vmax))
        )
        numideal = bsup - binf - 1
        if numideal >= 2:
            # have 2 or more wanted ideal ticks, so use them as major ticks
            if numideal > nbins:
                # to many ideal ticks, subsampling ideals for major ticks, and
                # take others for minor ticks
                subsampling_factor = math.ceil(numideal / nbins)
                if self._minor:
                    ticklocs = [
                        ideal_ticks(b)
                        for b in range(binf, bsup + 1)
                        if (b % subsampling_factor) != 0
                    ]
                else:
                    ticklocs = [
                        ideal_ticks(b)
                        for b in range(binf, bsup + 1)
                        if (b % subsampling_factor) == 0
                    ]
                return self.raise_if_exceeds(np.array(ticklocs))
            if self._minor:
                ticklocs = []
                for b in range(binf, bsup):
                    if b < -1:
                        ticklocs.extend(np.arange(2, 10) * 10 ** b)
                    elif b == -1:
                        ticklocs.extend(np.arange(2, 5) / 10)
                    elif b == 0:
                        ticklocs.extend(np.arange(6, 9) / 10)
                    else:
                        ticklocs.extend(
                            1 - np.arange(2, 10)[::-1] * 10 ** (-b - 1)
                        )
                return self.raise_if_exceeds(np.array(ticklocs))
            ticklocs = [ideal_ticks(b) for b in range(binf, bsup + 1)]
            return self.raise_if_exceeds(np.array(ticklocs))
        # the scale is zoomed so same ticks as linear scale can be used
        if self._minor:
            return []
        return super().tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        standard_minpos = 1e-7
        initial_range = (standard_minpos, 1 - standard_minpos)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = initial_range  # Initial range, no data plotted yet.
        elif vmax <= 0 or vmin >= 1:
            # vmax <= 0 occurs when all values are negative
            # vmin >= 1 occurs when all values are greater than one
            _api.warn_external(
                "Data has no values between 0 and 1, and therefore cannot be "
                "logit-scaled."
            )
            vmin, vmax = initial_range
        else:
            minpos = (
                self.axis.get_minpos()
                if self.axis is not None
                else standard_minpos
            )
            if not np.isfinite(minpos):
                minpos = standard_minpos  # This should never take effect.
            if vmin <= 0:
                vmin = minpos
            # NOTE: for vmax, we should query a property similar to get_minpos,
            # but related to the maximal, less-than-one data point.
            # Unfortunately, Bbox._minpos is defined very deep in the BBox and
            # updated with data, so for now we use 1 - minpos as a substitute.
            if vmax >= 1:
                vmax = 1 - minpos
            if vmin == vmax:
                vmin, vmax = 0.1 * vmin, 1 - 0.1 * vmin

        return vmin, vmax


class AutoLocator(MaxNLocator):
    """
    Place evenly spaced ticks, with the step size and maximum number of ticks chosen
    automatically.

    This is a subclass of `~matplotlib.ticker.MaxNLocator`, with parameters
    *nbins = 'auto'* and *steps = [1, 2, 2.5, 5, 10]*.
    """
    def __init__(self):
        """
        To know the values of the non-public parameters, please have a
        look to the defaults of `~matplotlib.ticker.MaxNLocator`.
        """
        if mpl.rcParams['_internal.classic_mode']:
            nbins = 9
            steps = [1, 2, 5, 10]
        else:
            nbins = 'auto'
            steps = [1, 2, 2.5, 5, 10]
        super().__init__(nbins=nbins, steps=steps)


class AutoMinorLocator(Locator):
    """
    Place evenly spaced minor ticks, with the step size and maximum number of ticks
    chosen automatically.

    The Axis must use a linear scale and have evenly spaced major ticks.
    """

    def __init__(self, n=None):
        """
        Parameters
        ----------
        n : int or 'auto', default: :rc:`xtick.minor.ndivs` or :rc:`ytick.minor.ndivs`
            The number of subdivisions of the interval between major ticks;
            e.g., n=2 will place a single minor tick midway between major ticks.

            If *n* is 'auto', it will be set to 4 or 5: if the distance
            between the major ticks equals 1, 2.5, 5 or 10 it can be perfectly
            divided in 5 equidistant sub-intervals with a length multiple of
            0.05; otherwise, it is divided in 4 sub-intervals.
        """
        self.ndivs = n

    def __call__(self):
        # docstring inherited
        if self.axis.get_scale() == 'log':
            _api.warn_external('AutoMinorLocator does not work on logarithmic scales')
            return []

        majorlocs = np.unique(self.axis.get_majorticklocs())
        if len(majorlocs) < 2:
            # Need at least two major ticks to find minor tick locations.
            # TODO: Figure out a way to still be able to display minor ticks with less
            # than two major ticks visible. For now, just display no ticks at all.
            return []
        majorstep = majorlocs[1] - majorlocs[0]

        if self.ndivs is None:
            self.ndivs = mpl.rcParams[
                'ytick.minor.ndivs' if self.axis.axis_name == 'y'
                else 'xtick.minor.ndivs']  # for x and z axis

        if self.ndivs == 'auto':
            majorstep_mantissa = 10 ** (np.log10(majorstep) % 1)
            ndivs = 5 if np.isclose(majorstep_mantissa, [1, 2.5, 5, 10]).any() else 4
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs

        vmin, vmax = sorted(self.axis.get_view_interval())
        t0 = majorlocs[0]
        tmin = round((vmin - t0) / minorstep)
        tmax = round((vmax - t0) / minorstep) + 1
        locs = (np.arange(tmin, tmax) * minorstep) + t0

        return self.raise_if_exceeds(locs)

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            f"Cannot get tick locations for a {type(self).__name__}")
<|endoftext|>"""
Scales define the distribution of data values on an axis, e.g. a log scaling.

The mapping is implemented through `.Transform` subclasses.

The following scales are built-in:

.. _builtin_scales:

============= ===================== ================================ =================================
Name          Class                 Transform                        Inverted transform
============= ===================== ================================ =================================
"asinh"       `AsinhScale`          `AsinhTransform`                 `InvertedAsinhTransform`
"function"    `FuncScale`           `FuncTransform`                  `FuncTransform`
"functionlog" `FuncScaleLog`        `FuncTransform` + `LogTransform` `InvertedLogTransform` + `FuncTransform`
"linear"      `LinearScale`         `.IdentityTransform`             `.IdentityTransform`
"log"         `LogScale`            `LogTransform`                   `InvertedLogTransform`
"logit"       `LogitScale`          `LogitTransform`                 `LogisticTransform`
"symlog"      `SymmetricalLogScale` `SymmetricalLogTransform`        `InvertedSymmetricalLogTransform`
============= ===================== ================================ =================================

A user will often only use the scale name, e.g. when setting the scale through
`~.Axes.set_xscale`: ``ax.set_xscale("log")``.

See also the :ref:`scales examples <sphx_glr_gallery_scales>` in the documentation.

Custom scaling can be achieved through `FuncScale`, or by creating your own
`ScaleBase` subclass and corresponding transforms (see :doc:`/gallery/scales/custom_scale`).
Third parties can register their scales by name through `register_scale`.
"""  # noqa: E501

import inspect
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter,
    NullLocator, LogLocator, AutoLocator, AutoMinorLocator,
    SymmetricalLogLocator, AsinhLocator, LogitLocator)
from matplotlib.transforms import Transform, IdentityTransform


class ScaleBase:
    """
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Subclasses should override

    :attr:`name`
        The scale's name.
    :meth:`get_transform`
        A method returning a `.Transform`, which converts data coordinates to
        scaled coordinates.  This transform should be invertible, so that e.g.
        mouse positions can be converted back to data coordinates.
    :meth:`set_default_locators_and_formatters`
        A method that sets default locators and formatters for an `~.axis.Axis`
        that uses this scale.
    :meth:`limit_range_for_scale`
        An optional method that "fixes" the axis range to acceptable values,
        e.g. restricting log-scaled axes to positive values.
    """

    def __init__(self, axis):
        r"""
        Construct a new scale.

        Notes
        -----
        The following note is for scale implementers.

        For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
        object as first argument.  However, this argument should not
        be used: a single scale object should be usable by multiple
        `~matplotlib.axis.Axis`\es at the same time.
        """

    def get_transform(self):
        """
        Return the `.Transform` object associated with this scale.
        """
        raise NotImplementedError()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters of *axis* to instances suitable for
        this scale.
        """
        raise NotImplementedError()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Return the range *vmin*, *vmax*, restricted to the
        domain supported by this scale (if any).

        *minpos* should be the minimum positive value in the data.
        This is used by log scales to determine a minimum value.
        """
        return vmin, vmax


class LinearScale(ScaleBase):
    """
    The default linear scale.
    """

    name = 'linear'

    def __init__(self, axis):
        # This method is present only to prevent inheritance of the base class'
        # constructor docstring, which would otherwise end up interpolated into
        # the docstring of Axis.set_scale.
        """
        """  # noqa: D419

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())

    def get_transform(self):
        """
        Return the transform for linear scaling, which is just the
        `~matplotlib.transforms.IdentityTransform`.
        """
        return IdentityTransform()


class FuncTransform(Transform):
    """
    A simple transform that takes and arbitrary function for the
    forward and inverse transform.
    """

    input_dims = output_dims = 1

    def __init__(self, forward, inverse):
        """
        Parameters
        ----------
        forward : callable
            The forward function for the transform.  This function must have
            an inverse and, for best behavior, be monotonic.
            It must have the signature::

               def forward(values: array-like) -> array-like

        inverse : callable
            The inverse of the forward function.  Signature as ``forward``.
        """
        super().__init__()
        if callable(forward) and callable(inverse):
            self._forward = forward
            self._inverse = inverse
        else:
            raise ValueError('arguments to FuncTransform must be functions')

    def transform_non_affine(self, values):
        return self._forward(values)

    def inverted(self):
        return FuncTransform(self._inverse, self._forward)


class FuncScale(ScaleBase):
    """
    Provide an arbitrary scale with user-supplied function for the axis.
    """

    name = 'function'

    def __init__(self, axis, functions):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

               def forward(values: array-like) -> array-like
        """
        forward, inverse = functions
        transform = FuncTransform(forward, inverse)
        self._transform = transform

    def get_transform(self):
        """Return the `.FuncTransform` associated with this scale."""
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())


class LogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, nonpositive='clip'):
        super().__init__()
        if base <= 0 or base == 1:
            raise ValueError('The log base cannot be <= 0 or == 1')
        self.base = base
        self._clip = _api.check_getitem(
            {"clip": True, "mask": False}, nonpositive=nonpositive)

    def __str__(self):
        return "{}(base={}, nonpositive={!r})".format(
            type(self).__name__, self.base, "clip" if self._clip else "mask")

    def transform_non_affine(self, values):
        # Ignore invalid values due to nans being passed to the transform.
        with np.errstate(divide="ignore", invalid="ignore"):
            log = {np.e: np.log, 2: np.log2, 10: np.log10}.get(self.base)
            if log:  # If possible, do everything in a single call to NumPy.
                out = log(values)
            else:
                out = np.log(values)
                out /= np.log(self.base)
            if self._clip:
                # SVG spec says that conforming viewers must support values up
                # to 3.4e38 (C float); however experiments suggest that
                # Inkscape (which uses cairo for rendering) runs into cairo's
                # 24-bit limit (which is apparently shared by Agg).
                # Ghostscript (used for pdf rendering appears to overflow even
                # earlier, with the max value around 2 ** 15 for the tests to
                # pass. On the other hand, in practice, we want to clip beyond
                #     np.log10(np.nextafter(0, 1)) ~ -323
                # so 1000 seems safe.
                out[values <= 0] = -1000
        return out

    def inverted(self):
        return InvertedLogTransform(self.base)


class InvertedLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base):
        super().__init__()
        self.base = base

    def __str__(self):
        return f"{type(self).__name__}(base={self.base})"

    def transform_non_affine(self, values):
        return np.power(self.base, values)

    def inverted(self):
        return LogTransform(self.base)


class LogScale(ScaleBase):
    """
    A standard logarithmic scale.  Care is taken to only plot positive values.
    """
    name = 'log'

    def __init__(self, axis, *, base=10, subs=None, nonpositive="clip"):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        base : float, default: 10
            The base of the logarithm.
        nonpositive : {'clip', 'mask'}, default: 'clip'
            Determines the behavior for non-positive values. They can either
            be masked as invalid, or clipped to a very small positive number.
        subs : sequence of int, default: None
            Where to place the subticks between each major tick.  For example,
            in a log10 scale, ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place 8
            logarithmically spaced minor ticks between each major tick.
        """
        self._transform = LogTransform(base, nonpositive)
        self.subs = subs

    base = property(lambda self: self._transform.base)

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(
            LogFormatterSciNotation(self.base,
                                    labelOnlyBase=(self.subs is not None)))

    def get_transform(self):
        """Return the `.LogTransform` associated with this scale."""
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Limit the domain to positive values."""
        if not np.isfinite(minpos):
            minpos = 1e-300  # Should rarely (if ever) have a visible effect.

        return (minpos if vmin <= 0 else vmin,
                minpos if vmax <= 0 else vmax)


class FuncScaleLog(LogScale):
    """
    Provide an arbitrary scale with user-supplied function for the axis and
    then put on a logarithmic axes.
    """

    name = 'functionlog'

    def __init__(self, axis, functions, base=10):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

                def forward(values: array-like) -> array-like

        base : float, default: 10
            Logarithmic base of the scale.
        """
        forward, inverse = functions
        self.subs = None
        self._transform = FuncTransform(forward, inverse) + LogTransform(base)

    @property
    def base(self):
        return self._transform._b.base  # Base of the LogTransform.

    def get_transform(self):
        """Return the `.Transform` associated with this scale."""
        return self._transform


class SymmetricalLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        if base <= 1.0:
            raise ValueError("'base' must be larger than 1")
        if linthresh <= 0.0:
            raise ValueError("'linthresh' must be positive")
        if linscale <= 0.0:
            raise ValueError("'linscale' must be positive")
        self.base = base
        self.linthresh = linthresh
        self.linscale = linscale
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))
        self._log_base = np.log(base)

    def transform_non_affine(self, values):
        abs_a = np.abs(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(values) * self.linthresh * (
                self._linscale_adj +
                np.log(abs_a / self.linthresh) / self._log_base)
            inside = abs_a <= self.linthresh
        out[inside] = values[inside] * self._linscale_adj
        return out

    def inverted(self):
        return InvertedSymmetricalLogTransform(self.base, self.linthresh,
                                               self.linscale)


class InvertedSymmetricalLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        symlog = SymmetricalLogTransform(base, linthresh, linscale)
        self.base = base
        self.linthresh = linthresh
        self.invlinthresh = symlog.transform(linthresh)
        self.linscale = linscale
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))

    def transform_non_affine(self, values):
        abs_a = np.abs(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(values) * self.linthresh * (
                np.power(self.base,
                         abs_a / self.linthresh - self._linscale_adj))
            inside = abs_a <= self.invlinthresh
        out[inside] = values[inside] / self._linscale_adj
        return out

    def inverted(self):
        return SymmetricalLogTransform(self.base,
                                       self.linthresh, self.linscale)


class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    See :doc:`/gallery/scales/symlog_demo` for a detailed description.

    Parameters
    ----------
    base : float, default: 10
        The base of the logarithm.

    linthresh : float, default: 2
        Defines the range ``(-x, x)``, within which the plot is linear.
        This avoids having the plot go to infinity around zero.

    subs : sequence of int
        Where to place the subticks between each major tick.
        For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
        8 logarithmically spaced minor ticks between each major tick.

    linscale : float, optional
        This allows the linear range ``(-linthresh, linthresh)`` to be
        stretched relative to the logarithmic range. Its value is the number of
        decades to use for each half of the linear range. For example, when
        *linscale* == 1.0 (the default), the space used for the positive and
        negative halves of the linear range will be equal to one decade in
        the logarithmic range.
    """
    name = 'symlog'

    def __init__(self, axis, *, base=10, linthresh=2, subs=None, linscale=1):
        self._transform = SymmetricalLogTransform(base, linthresh, linscale)
        self.subs = subs

    base = property(lambda self: self._transform.base)
    linthresh = property(lambda self: self._transform.linthresh)
    linscale = property(lambda self: self._transform.linscale)

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(),
                                                     self.subs))
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """Return the `.SymmetricalLogTransform` associated with this scale."""
        return self._transform


class AsinhTransform(Transform):
    """Inverse hyperbolic-sine transformation used by `.AsinhScale`"""
    input_dims = output_dims = 1

    def __init__(self, linear_width):
        super().__init__()
        if linear_width <= 0.0:
            raise ValueError("Scale parameter 'linear_width' " +
                             "must be strictly positive")
        self.linear_width = linear_width

    def transform_non_affine(self, values):
        return self.linear_width * np.arcsinh(values / self.linear_width)

    def inverted(self):
        return InvertedAsinhTransform(self.linear_width)


class InvertedAsinhTransform(Transform):
    """Hyperbolic sine transformation used by `.AsinhScale`"""
    input_dims = output_dims = 1

    def __init__(self, linear_width):
        super().__init__()
        self.linear_width = linear_width

    def transform_non_affine(self, values):
        return self.linear_width * np.sinh(values / self.linear_width)

    def inverted(self):
        return AsinhTransform(self.linear_width)


class AsinhScale(ScaleBase):
    """
    A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)

    For values close to zero, this is essentially a linear scale,
    but for large magnitude values (either positive or negative)
    it is asymptotically logarithmic. The transition between these
    linear and logarithmic regimes is smooth, and has no discontinuities
    in the function gradient in contrast to
    the `.SymmetricalLogScale` ("symlog") scale.

    Specifically, the transformation of an axis coordinate :math:`a` is
    :math:`a \\rightarrow a_0 \\sinh^{-1} (a / a_0)` where :math:`a_0`
    is the effective width of the linear region of the transformation.
    In that region, the transformation is
    :math:`a \\rightarrow a + \\mathcal{O}(a^3)`.
    For large values of :math:`a` the transformation behaves as
    :math:`a \\rightarrow a_0 \\, \\mathrm{sgn}(a) \\ln |a| + \\mathcal{O}(1)`.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    """

    name = 'asinh'

    auto_tick_multipliers = {
        3: (2, ),
        4: (2, ),
        5: (2, ),
        8: (2, 4),
        10: (2, 5),
        16: (2, 4, 8),
        64: (4, 16),
        1024: (256, 512)
    }

    def __init__(self, axis, *, linear_width=1.0,
                 base=10, subs='auto', **kwargs):
        """
        Parameters
        ----------
        linear_width : float, default: 1
            The scale parameter (elsewhere referred to as :math:`a_0`)
            defining the extent of the quasi-linear region,
            and the coordinate values beyond which the transformation
            becomes asymptotically logarithmic.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : sequence of int
            Multiples of the number base used for minor ticks.
            If set to 'auto', this will use built-in defaults,
            e.g. (2, 5) for base=10.
        """
        super().__init__(axis)
        self._transform = AsinhTransform(linear_width)
        self._base = int(base)
        if subs == 'auto':
            self._subs = self.auto_tick_multipliers.get(self._base)
        else:
            self._subs = subs

    linear_width = property(lambda self: self._transform.linear_width)

    def get_transform(self):
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        axis.set(major_locator=AsinhLocator(self.linear_width,
                                            base=self._base),
                 minor_locator=AsinhLocator(self.linear_width,
                                            base=self._base,
                                            subs=self._subs),
                 minor_formatter=NullFormatter())
        if self._base > 1:
            axis.set_major_formatter(LogFormatterSciNotation(self._base))
        else:
            axis.set_major_formatter('{x:.3g}')


class LogitTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        _api.check_in_list(['mask', 'clip'], nonpositive=nonpositive)
        self._nonpositive = nonpositive
        self._clip = {"clip": True, "mask": False}[nonpositive]

    def transform_non_affine(self, values):
        """logit transform (base 10), masked or clipped"""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(values / (1 - values))
        if self._clip:  # See LogTransform for choice of clip value.
            out[values <= 0] = -1000
            out[1 <= values] = 1000
        return out

    def inverted(self):
        return LogisticTransform(self._nonpositive)

    def __str__(self):
        return f"{type(self).__name__}({self._nonpositive!r})"


class LogisticTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        self._nonpositive = nonpositive

    def transform_non_affine(self, values):
        """logistic transform (base 10)"""
        return 1.0 / (1 + 10**(-values))

    def inverted(self):
        return LogitTransform(self._nonpositive)

    def __str__(self):
        return f"{type(self).__name__}({self._nonpositive!r})"


class LogitScale(ScaleBase):
    """
    Logit scale for data between zero and one, both excluded.

    This scale is similar to a log scale close to zero and to one, and almost
    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
    """
    name = 'logit'

    def __init__(self, axis, nonpositive='mask', *,
                 one_half=r"\frac{1}{2}", use_overline=False):
        r"""
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            Currently unused.
        nonpositive : {'mask', 'clip'}
            Determines the behavior for values beyond the open interval ]0, 1[.
            They can either be masked as invalid, or clipped to a number very
            close to 0 or 1.
        use_overline : bool, default: False
            Indicate the usage of survival notation (\overline{x}) in place of
            standard notation (1-x) for probability close to one.
        one_half : str, default: r"\frac{1}{2}"
            The string used for ticks formatter to represent 1/2.
        """
        self._transform = LogitTransform(nonpositive)
        self._use_overline = use_overline
        self._one_half = one_half

    def get_transform(self):
        """Return the `.LogitTransform` associated with this scale."""
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        # ..., 0.01, 0.1, 0.5, 0.9, 0.99, ...
        axis.set_major_locator(LogitLocator())
        axis.set_major_formatter(
            LogitFormatter(
                one_half=self._one_half,
                use_overline=self._use_overline
            )
        )
        axis.set_minor_locator(LogitLocator(minor=True))
        axis.set_minor_formatter(
            LogitFormatter(
                minor=True,
                one_half=self._one_half,
                use_overline=self._use_overline
            )
        )

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and 1 (excluded).
        """
        if not np.isfinite(minpos):
            minpos = 1e-7  # Should rarely (if ever) have a visible effect.
        return (minpos if vmin <= 0 else vmin,
                1 - minpos if vmax >= 1 else vmax)


_scale_mapping = {
    'linear': LinearScale,
    'log':    LogScale,
    'symlog': SymmetricalLogScale,
    'asinh':  AsinhScale,
    'logit':  LogitScale,
    'function': FuncScale,
    'functionlog': FuncScaleLog,
    }


def get_scale_names():
    """Return the names of the available scales."""
    return sorted(_scale_mapping)


def scale_factory(scale, axis, **kwargs):
    """
    Return a scale class by name.

    Parameters
    ----------
    scale : {%(names)s}
    axis : `~matplotlib.axis.Axis`
    """
    scale_cls = _api.check_getitem(_scale_mapping, scale=scale)
    return scale_cls(axis, **kwargs)


if scale_factory.__doc__:
    scale_factory.__doc__ = scale_factory.__doc__ % {
        "names": ", ".join(map(repr, get_scale_names()))}


def register_scale(scale_class):
    """
    Register a new kind of scale.

    Parameters
    ----------
    scale_class : subclass of `ScaleBase`
        The scale to register.
    """
    _scale_mapping[scale_class.name] = scale_class


def _get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """
    docs = []
    for name, scale_class in _scale_mapping.items():
        docstring = inspect.getdoc(scale_class.__init__) or ""
        docs.extend([
            f"    {name!r}",
            "",
            textwrap.indent(docstring, " " * 8),
            ""
        ])
    return "\n".join(docs)


_docstring.interpd.register(
    scale_type='{%s}' % ', '.join([repr(x) for x in get_scale_names()]),
    scale_docs=_get_scale_docs().rstrip(),
    )
<|endoftext|>from collections import OrderedDict
import logging
import urllib.parse

import numpy as np

from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
    FontProperties, get_font, fontManager as _fontManager
)
from matplotlib.ft2font import LoadFlags
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D

_log = logging.getLogger(__name__)


class TextToPath:
    """A class that converts strings to paths."""

    FONT_SCALE = 100.
    DPI = 72

    def __init__(self):
        self.mathtext_parser = MathTextParser('path')
        self._texmanager = None

    def _get_font(self, prop):
        """
        Find the `FT2Font` matching font properties *prop*, with its size set.
        """
        filenames = _fontManager._find_fonts_by_props(prop)
        font = get_font(filenames)
        font.set_size(self.FONT_SCALE, self.DPI)
        return font

    def _get_hinting_flag(self):
        return LoadFlags.NO_HINTING

    def _get_char_id(self, font, ccode):
        """
        Return a unique id for the given font and character-code set.
        """
        return urllib.parse.quote(f"{font.postscript_name}-{ccode:x}")

    def get_text_width_height_descent(self, s, prop, ismath):
        fontsize = prop.get_size_in_points()

        if ismath == "TeX":
            return TexManager().get_text_width_height_descent(s, fontsize)

        scale = fontsize / self.FONT_SCALE

        if ismath:
            prop = prop.copy()
            prop.set_size(self.FONT_SCALE)
            width, height, descent, *_ = \
                self.mathtext_parser.parse(s, 72, prop)
            return width * scale, height * scale, descent * scale

        font = self._get_font(prop)
        font.set_text(s, 0.0, flags=LoadFlags.NO_HINTING)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d = font.get_descent()
        d /= 64.0
        return w * scale, h * scale, d * scale

    def get_text_path(self, prop, s, ismath=False):
        """
        Convert text *s* to path (a tuple of vertices and codes for
        matplotlib.path.Path).

        Parameters
        ----------
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties for the text.
        s : str
            The text to be converted.
        ismath : {False, True, "TeX"}
            If True, use mathtext parser.  If "TeX", use tex for rendering.

        Returns
        -------
        verts : list
            A list of arrays containing the (x, y) coordinates of the vertices.
        codes : list
            A list of path codes.

        Examples
        --------
        Create a list of vertices and codes from a text, and create a `.Path`
        from those::

            from matplotlib.path import Path
            from matplotlib.text import TextToPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Comic Neue", style="italic")
            verts, codes = TextToPath().get_text_path(fp, "ABC")
            path = Path(verts, codes, closed=False)

        Also see `TextPath` for a more direct way to create a path from a text.
        """
        if ismath == "TeX":
            glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
        elif not ismath:
            font = self._get_font(prop)
            glyph_info, glyph_map, rects = self.get_glyphs_with_font(font, s)
        else:
            glyph_info, glyph_map, rects = self.get_glyphs_mathtext(prop, s)

        verts, codes = [], []
        for glyph_id, xposition, yposition, scale in glyph_info:
            verts1, codes1 = glyph_map[glyph_id]
            verts.extend(verts1 * scale + [xposition, yposition])
            codes.extend(codes1)
        for verts1, codes1 in rects:
            verts.extend(verts1)
            codes.extend(codes1)

        # Make sure an empty string or one with nothing to print
        # (e.g. only spaces & newlines) will be valid/empty path
        if not verts:
            verts = np.empty((0, 2))

        return verts, codes

    def get_glyphs_with_font(self, font, s, glyph_map=None,
                             return_new_glyphs_only=False):
        """
        Convert string *s* to vertices and codes using the provided ttf font.
        """

        if glyph_map is None:
            glyph_map = OrderedDict()

        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        xpositions = []
        glyph_ids = []
        for item in _text_helpers.layout(s, font):
            char_id = self._get_char_id(item.ft_object, ord(item.char))
            glyph_ids.append(char_id)
            xpositions.append(item.x)
            if char_id not in glyph_map:
                glyph_map_new[char_id] = item.ft_object.get_path()

        ypositions = [0] * len(xpositions)
        sizes = [1.] * len(xpositions)

        rects = []

        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, rects)

    def get_glyphs_mathtext(self, prop, s, glyph_map=None,
                            return_new_glyphs_only=False):
        """
        Parse mathtext string *s* and convert it to a (vertices, codes) pair.
        """

        prop = prop.copy()
        prop.set_size(self.FONT_SCALE)

        width, height, descent, glyphs, rects = self.mathtext_parser.parse(
            s, self.DPI, prop)

        if not glyph_map:
            glyph_map = OrderedDict()

        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        xpositions = []
        ypositions = []
        glyph_ids = []
        sizes = []

        for font, fontsize, ccode, ox, oy in glyphs:
            char_id = self._get_char_id(font, ccode)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                font.load_char(ccode, flags=LoadFlags.NO_HINTING)
                glyph_map_new[char_id] = font.get_path()

            xpositions.append(ox)
            ypositions.append(oy)
            glyph_ids.append(char_id)
            size = fontsize / self.FONT_SCALE
            sizes.append(size)

        myrects = []
        for ox, oy, w, h in rects:
            vert1 = [(ox, oy), (ox, oy + h), (ox + w, oy + h),
                     (ox + w, oy), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO,
                     Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
                     Path.CLOSEPOLY]
            myrects.append((vert1, code1))

        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, myrects)

    def get_glyphs_tex(self, prop, s, glyph_map=None,
                       return_new_glyphs_only=False):
        """Convert the string *s* to vertices and codes using usetex mode."""
        # Mostly borrowed from pdf backend.

        dvifile = TexManager().make_dvi(s, self.FONT_SCALE)
        with dviread.Dvi(dvifile, self.DPI) as dvi:
            page, = dvi

        if glyph_map is None:
            glyph_map = OrderedDict()

        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        glyph_ids, xpositions, ypositions, sizes = [], [], [], []

        # Gather font information and do some setup for combining
        # characters into strings.
        for text in page.text:
            font = get_font(text.font_path)
            char_id = self._get_char_id(font, text.glyph)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                glyph_name_or_index = text.glyph_name_or_index
                if isinstance(glyph_name_or_index, str):
                    index = font.get_name_index(glyph_name_or_index)
                    font.load_glyph(index, flags=LoadFlags.TARGET_LIGHT)
                elif isinstance(glyph_name_or_index, int):
                    self._select_native_charmap(font)
                    font.load_char(
                        glyph_name_or_index, flags=LoadFlags.TARGET_LIGHT)
                else:  # Should not occur.
                    raise TypeError(f"Glyph spec of unexpected type: "
                                    f"{glyph_name_or_index!r}")
                glyph_map_new[char_id] = font.get_path()

            glyph_ids.append(char_id)
            xpositions.append(text.x)
            ypositions.append(text.y)
            sizes.append(text.font_size / self.FONT_SCALE)

        myrects = []

        for ox, oy, h, w in page.boxes:
            vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h),
                     (ox, oy + h), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO,
                     Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
                     Path.CLOSEPOLY]
            myrects.append((vert1, code1))

        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, myrects)

    @staticmethod
    def _select_native_charmap(font):
        # Select the native charmap. (we can't directly identify it but it's
        # typically an Adobe charmap).
        for charmap_code in [
                1094992451,  # ADOBE_CUSTOM.
                1094995778,  # ADOBE_STANDARD.
        ]:
            try:
                font.select_charmap(charmap_code)
            except (ValueError, RuntimeError):
                pass
            else:
                break
        else:
            _log.warning("No supported encoding in font (%s).", font.fname)


text_to_path = TextToPath()


class TextPath(Path):
    """
    Create a path from the text.
    """

    def __init__(self, xy, s, size=None, prop=None,
                 _interpolation_steps=1, usetex=False):
        r"""
        Create a path from the text. Note that it simply is a path,
        not an artist. You need to use the `.PathPatch` (or other artists)
        to draw this path onto the canvas.

        Parameters
        ----------
        xy : tuple or array of two float values
            Position of the text. For no offset, use ``xy=(0, 0)``.

        s : str
            The text to convert to a path.

        size : float, optional
            Font size in points. Defaults to the size specified via the font
            properties *prop*.

        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property. If not provided, will use a default
            `.FontProperties` with parameters from the
            :ref:`rcParams<customizing-with-dynamic-rc-settings>`.

        _interpolation_steps : int, optional
            (Currently ignored)

        usetex : bool, default: False
            Whether to use tex rendering.

        Examples
        --------
        The following creates a path from the string "ABC" with Helvetica
        font face; and another path from the latex fraction 1/2::

            from matplotlib.text import TextPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Helvetica", style="italic")
            path1 = TextPath((12, 12), "ABC", size=12, prop=fp)
            path2 = TextPath((0, 0), r"$\frac{1}{2}$", size=12, usetex=True)

        Also see :doc:`/gallery/text_labels_and_annotations/demo_text_path`.
        """
        # Circular import.
        from matplotlib.text import Text

        prop = FontProperties._from_any(prop)
        if size is None:
            size = prop.get_size_in_points()

        self._xy = xy
        self.set_size(size)

        self._cached_vertices = None
        s, ismath = Text(usetex=usetex)._preprocess_math(s)
        super().__init__(
            *text_to_path.get_text_path(prop, s, ismath=ismath),
            _interpolation_steps=_interpolation_steps,
            readonly=True)
        self._should_simplify = False

    def set_size(self, size):
        """Set the text size."""
        self._size = size
        self._invalid = True

    def get_size(self):
        """Get the text size."""
        return self._size

    @property
    def vertices(self):
        """
        Return the cached path after updating it if necessary.
        """
        self._revalidate_path()
        return self._cached_vertices

    @property
    def codes(self):
        """
        Return the codes
        """
        return self._codes

    def _revalidate_path(self):
        """
        Update the path if necessary.

        The path for the text is initially create with the font size of
        `.FONT_SCALE`, and this path is rescaled to other size when necessary.
        """
        if self._invalid or self._cached_vertices is None:
            tr = (Affine2D()
                  .scale(self._size / text_to_path.FONT_SCALE)
                  .translate(*self._xy))
            self._cached_vertices = tr.transform(self._vertices)
            self._cached_vertices.flags.writeable = False
            self._invalid = False
<|endoftext|># JavaScript template for HTMLWriter
JS_INCLUDE = """
<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    var slider = document.getElementById(this.slider_id);
    slider.max = this.frames.length - 1;
    if (isInternetExplorer()) {
        // switch from oninput to onchange because IE <= 11 does not conform
        // with W3C specification. It ignores oninput and onchange behaves
        // like oninput. In contrast, Microsoft Edge behaves correctly.
        slider.setAttribute('onchange', slider.getAttribute('oninput'));
        slider.setAttribute('oninput', null);
    }
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src =
            this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_forward();
    }, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_reverse();
    }, this.interval);
  }
</script>
"""


# Style definitions for the HTML template
STYLE_INCLUDE = """
<style>
.animation {
    display: inline-block;
    text-align: center;
}
input[type=range].anim-slider {
    width: 374px;
    margin-left: auto;
    margin-right: auto;
}
.anim-buttons {
    margin: 8px 0px;
}
.anim-buttons button {
    padding: 0;
    width: 36px;
}
.anim-state label {
    margin-right: 8px;
}
.anim-state input {
    margin: 0;
    vertical-align: middle;
}
</style>
"""


# HTML template for HTMLWriter
DISPLAY_TEMPLATE = """
<div class="animation">
  <img id="_anim_img{id}">
  <div class="anim-controls">
    <input id="_anim_slider{id}" type="range" class="anim-slider"
           name="points" min="0" max="1" step="1" value="0"
           oninput="anim{id}.set_frame(parseInt(this.value));">
    <div class="anim-buttons">
      <button title="Decrease speed" aria-label="Decrease speed" onclick="anim{id}.slower()">
          <i class="fa fa-minus"></i></button>
      <button title="First frame" aria-label="First frame" onclick="anim{id}.first_frame()">
        <i class="fa fa-fast-backward"></i></button>
      <button title="Previous frame" aria-label="Previous frame" onclick="anim{id}.previous_frame()">
          <i class="fa fa-step-backward"></i></button>
      <button title="Play backwards" aria-label="Play backwards" onclick="anim{id}.reverse_animation()">
          <i class="fa fa-play fa-flip-horizontal"></i></button>
      <button title="Pause" aria-label="Pause" onclick="anim{id}.pause_animation()">
          <i class="fa fa-pause"></i></button>
      <button title="Play" aria-label="Play" onclick="anim{id}.play_animation()">
          <i class="fa fa-play"></i></button>
      <button title="Next frame" aria-label="Next frame" onclick="anim{id}.next_frame()">
          <i class="fa fa-step-forward"></i></button>
      <button title="Last frame" aria-label="Last frame" onclick="anim{id}.last_frame()">
          <i class="fa fa-fast-forward"></i></button>
      <button title="Increase speed" aria-label="Increase speed" onclick="anim{id}.faster()">
          <i class="fa fa-plus"></i></button>
    </div>
    <form title="Repetition mode" aria-label="Repetition mode" action="#n" name="_anim_loop_select{id}"
          class="anim-state">
      <input type="radio" name="state" value="once" id="_anim_radio1_{id}"
             {once_checked}>
      <label for="_anim_radio1_{id}">Once</label>
      <input type="radio" name="state" value="loop" id="_anim_radio2_{id}"
             {loop_checked}>
      <label for="_anim_radio2_{id}">Loop</label>
      <input type="radio" name="state" value="reflect" id="_anim_radio3_{id}"
             {reflect_checked}>
      <label for="_anim_radio3_{id}">Reflect</label>
    </form>
  </div>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {{
    var img_id = "_anim_img{id}";
    var slider_id = "_anim_slider{id}";
    var loop_select_id = "_anim_loop_select{id}";
    var frames = new Array({Nframes});
    {fill_frames}

    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {{
        anim{id} = new Animation(frames, img_id, slider_id, {interval},
                                 loop_select_id);
    }}, 0);
  }})()
</script>
"""  # noqa: E501


INCLUDED_FRAMES = """
  for (var i=0; i<{Nframes}; i++){{
    frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) +
                ".{frame_format}";
  }}
"""
<|endoftext|>from matplotlib import _api, backend_tools, cbook, widgets


class ToolEvent:
    """Event for tool manipulation (add/remove)."""
    def __init__(self, name, sender, tool, data=None):
        self.name = name
        self.sender = sender
        self.tool = tool
        self.data = data


class ToolTriggerEvent(ToolEvent):
    """Event to inform that a tool has been triggered."""
    def __init__(self, name, sender, tool, canvasevent=None, data=None):
        super().__init__(name, sender, tool, data)
        self.canvasevent = canvasevent


class ToolManagerMessageEvent:
    """
    Event carrying messages from toolmanager.

    Messages usually get displayed to the user by the toolbar.
    """
    def __init__(self, name, sender, message):
        self.name = name
        self.sender = sender
        self.message = message


class ToolManager:
    """
    Manager for actions triggered by user interactions (key press, toolbar
    clicks, ...) on a Figure.

    Attributes
    ----------
    figure : `.Figure`
    keypresslock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the `canvas` key_press_event is locked.
    messagelock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the message is available to write.
    """

    def __init__(self, figure=None):

        self._key_press_handler_id = None

        self._tools = {}
        self._keys = {}
        self._toggled = {}
        self._callbacks = cbook.CallbackRegistry()

        # to process keypress event
        self.keypresslock = widgets.LockDraw()
        self.messagelock = widgets.LockDraw()

        self._figure = None
        self.set_figure(figure)

    @property
    def canvas(self):
        """Canvas managed by FigureManager."""
        if not self._figure:
            return None
        return self._figure.canvas

    @property
    def figure(self):
        """Figure that holds the canvas."""
        return self._figure

    @figure.setter
    def figure(self, figure):
        self.set_figure(figure)

    def set_figure(self, figure, update_tools=True):
        """
        Bind the given figure to the tools.

        Parameters
        ----------
        figure : `.Figure`
        update_tools : bool, default: True
            Force tools to update figure.
        """
        if self._key_press_handler_id:
            self.canvas.mpl_disconnect(self._key_press_handler_id)
        self._figure = figure
        if figure:
            self._key_press_handler_id = self.canvas.mpl_connect(
                'key_press_event', self._key_press)
        if update_tools:
            for tool in self._tools.values():
                tool.figure = figure

    def toolmanager_connect(self, s, func):
        """
        Connect event with string *s* to *func*.

        Parameters
        ----------
        s : str
            The name of the event. The following events are recognized:

            - 'tool_message_event'
            - 'tool_removed_event'
            - 'tool_added_event'

            For every tool added a new event is created

            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

        func : callable
            Callback function for the toolmanager event with signature::

                def func(event: ToolEvent) -> Any

        Returns
        -------
        cid
            The callback id for the connection. This can be used in
            `.toolmanager_disconnect`.
        """
        return self._callbacks.connect(s, func)

    def toolmanager_disconnect(self, cid):
        """
        Disconnect callback id *cid*.

        Example usage::

            cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)
            #...later
            toolmanager.toolmanager_disconnect(cid)
        """
        return self._callbacks.disconnect(cid)

    def message_event(self, message, sender=None):
        """Emit a `ToolManagerMessageEvent`."""
        if sender is None:
            sender = self

        s = 'tool_message_event'
        event = ToolManagerMessageEvent(s, sender, message)
        self._callbacks.process(s, event)

    @property
    def active_toggle(self):
        """Currently toggled tools."""
        return self._toggled

    def get_tool_keymap(self, name):
        """
        Return the keymap associated with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.

        Returns
        -------
        list of str
            List of keys associated with the tool.
        """

        keys = [k for k, i in self._keys.items() if i == name]
        return keys

    def _remove_keys(self, name):
        for k in self.get_tool_keymap(name):
            del self._keys[k]

    def update_keymap(self, name, key):
        """
        Set the keymap to associate with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.
        key : str or list of str
            Keys to associate with the tool.
        """
        if name not in self._tools:
            raise KeyError(f'{name!r} not in Tools')
        self._remove_keys(name)
        if isinstance(key, str):
            key = [key]
        for k in key:
            if k in self._keys:
                _api.warn_external(
                    f'Key {k} changed from {self._keys[k]} to {name}')
            self._keys[k] = name

    def remove_tool(self, name):
        """
        Remove tool named *name*.

        Parameters
        ----------
        name : str
            Name of the tool.
        """
        tool = self.get_tool(name)
        if getattr(tool, 'toggled', False):  # If it's a toggled toggle tool, untoggle
            self.trigger_tool(tool, 'toolmanager')
        self._remove_keys(name)
        event = ToolEvent('tool_removed_event', self, tool)
        self._callbacks.process(event.name, event)
        del self._tools[name]

    def add_tool(self, name, tool, *args, **kwargs):
        """
        Add *tool* to `ToolManager`.

        If successful, adds a new event ``tool_trigger_{name}`` where
        ``{name}`` is the *name* of the tool; the event is fired every time the
        tool is triggered.

        Parameters
        ----------
        name : str
            Name of the tool, treated as the ID, has to be unique.
        tool : type
            Class of the tool to be added.  A subclass will be used
            instead if one was registered for the current canvas class.
        *args, **kwargs
            Passed to the *tool*'s constructor.

        See Also
        --------
        matplotlib.backend_tools.ToolBase : The base class for tools.
        """

        tool_cls = backend_tools._find_tool_class(type(self.canvas), tool)
        if not tool_cls:
            raise ValueError('Impossible to find class for %s' % str(tool))

        if name in self._tools:
            _api.warn_external('A "Tool class" with the same name already '
                               'exists, not added')
            return self._tools[name]

        tool_obj = tool_cls(self, name, *args, **kwargs)
        self._tools[name] = tool_obj

        if tool_obj.default_keymap is not None:
            self.update_keymap(name, tool_obj.default_keymap)

        # For toggle tools init the radio_group in self._toggled
        if isinstance(tool_obj, backend_tools.ToolToggleBase):
            # None group is not mutually exclusive, a set is used to keep track
            # of all toggled tools in this group
            if tool_obj.radio_group is None:
                self._toggled.setdefault(None, set())
            else:
                self._toggled.setdefault(tool_obj.radio_group, None)

            # If initially toggled
            if tool_obj.toggled:
                self._handle_toggle(tool_obj, None, None)
        tool_obj.set_figure(self.figure)

        event = ToolEvent('tool_added_event', self, tool_obj)
        self._callbacks.process(event.name, event)

        return tool_obj

    def _handle_toggle(self, tool, canvasevent, data):
        """
        Toggle tools, need to untoggle prior to using other Toggle tool.
        Called from trigger_tool.

        Parameters
        ----------
        tool : `.ToolBase`
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """

        radio_group = tool.radio_group
        # radio_group None is not mutually exclusive
        # just keep track of toggled tools in this group
        if radio_group is None:
            if tool.name in self._toggled[None]:
                self._toggled[None].remove(tool.name)
            else:
                self._toggled[None].add(tool.name)
            return

        # If the tool already has a toggled state, untoggle it
        if self._toggled[radio_group] == tool.name:
            toggled = None
        # If no tool was toggled in the radio_group
        # toggle it
        elif self._toggled[radio_group] is None:
            toggled = tool.name
        # Other tool in the radio_group is toggled
        else:
            # Untoggle previously toggled tool
            self.trigger_tool(self._toggled[radio_group],
                              self,
                              canvasevent,
                              data)
            toggled = tool.name

        # Keep track of the toggled tool in the radio_group
        self._toggled[radio_group] = toggled

    def trigger_tool(self, name, sender=None, canvasevent=None, data=None):
        """
        Trigger a tool and emit the ``tool_trigger_{name}`` event.

        Parameters
        ----------
        name : str
            Name of the tool.
        sender : object
            Object that wishes to trigger the tool.
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
        tool = self.get_tool(name)
        if tool is None:
            return

        if sender is None:
            sender = self

        if isinstance(tool, backend_tools.ToolToggleBase):
            self._handle_toggle(tool, canvasevent, data)

        tool.trigger(sender, canvasevent, data)  # Actually trigger Tool.

        s = 'tool_trigger_%s' % name
        event = ToolTriggerEvent(s, sender, tool, canvasevent, data)
        self._callbacks.process(s, event)

    def _key_press(self, event):
        if event.key is None or self.keypresslock.locked():
            return

        name = self._keys.get(event.key, None)
        if name is None:
            return
        self.trigger_tool(name, canvasevent=event)

    @property
    def tools(self):
        """A dict mapping tool name -> controlled tool."""
        return self._tools

    def get_tool(self, name, warn=True):
        """
        Return the tool object with the given name.

        For convenience, this passes tool objects through.

        Parameters
        ----------
        name : str or `.ToolBase`
            Name of the tool, or the tool itself.
        warn : bool, default: True
            Whether a warning should be emitted it no tool with the given name
            exists.

        Returns
        -------
        `.ToolBase` or None
            The tool or None if no tool with the given name exists.
        """
        if (isinstance(name, backend_tools.ToolBase)
                and name.name in self._tools):
            return name
        if name not in self._tools:
            if warn:
                _api.warn_external(
                    f"ToolManager does not control tool {name!r}")
            return None
        return self._tools[name]
<|endoftext|>"""
Numerical Python functions written for compatibility with MATLAB
commands with the same names. Most numerical Python functions can be found in
the `NumPy`_ and `SciPy`_ libraries. What remains here is code for performing
spectral computations and kernel density estimations.

.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org

