            # error: Argument "data" to "_iter_data" of "MPLPlot" has
            # incompatible type "object"; expected "DataFrame |
            # dict[Hashable, Series | DataFrame]"
            for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
                ax = self._get_ax(i)
                kwds = self.kwds.copy()

                # When by is applied, show title for subplots to know which group it is
                # just like df.boxplot, and need to apply T on y to provide right input
                if self.by is not None:
                    y = y.T
                    ax.set_title(pprint_thing(label))

                    # When `by` is assigned, the ticklabels will become unique grouped
                    # values, instead of label which is used as subtitle in this case.
                    # error: "Index" has no attribute "levels"; maybe "nlevels"?
                    levels = self.data.columns.levels  # type: ignore[attr-defined]
                    ticklabels = [pprint_thing(col) for col in levels[0]]
                else:
                    ticklabels = [pprint_thing(label)]

                ret, bp = self._plot(
                    ax, y, column_num=i, return_type=self.return_type, **kwds
                )
                self.maybe_color_bp(bp)
                self._return_obj[label] = ret
                _set_ticklabels(
                    ax=ax, labels=ticklabels, is_vertical=self.orientation == "vertical"
                )
        else:
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()

            ret, bp = self._plot(
                ax, y, column_num=0, return_type=self.return_type, **kwds
            )
            self.maybe_color_bp(bp)
            self._return_obj = ret

            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(
                ax=ax, labels=labels, is_vertical=self.orientation == "vertical"
            )

    def _make_legend(self) -> None:
        pass

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # GH 45465: make sure that the boxplot doesn't ignore xlabel/ylabel
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))

    @property
    def orientation(self) -> Literal["horizontal", "vertical"]:
        if self.kwds.get("vert", True):
            return "vertical"
        else:
            return "horizontal"

    @property
    def result(self):
        if self.return_type is None:
            return super().result
        else:
            return self._return_obj


def maybe_color_bp(bp, color_tup, **kwds) -> None:
    # GH#30346, when users specifying those arguments explicitly, our defaults
    # for these four kwargs should be overridden; if not, use Pandas settings
    if not kwds.get("boxprops"):
        setp(bp["boxes"], color=color_tup[0], alpha=1)
    if not kwds.get("whiskerprops"):
        setp(bp["whiskers"], color=color_tup[1], alpha=1)
    if not kwds.get("medianprops"):
        setp(bp["medians"], color=color_tup[2], alpha=1)
    if not kwds.get("capprops"):
        setp(bp["caps"], color=color_tup[3], alpha=1)


def _grouped_plot_by_column(
    plotf,
    data,
    columns=None,
    by=None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    ax=None,
    layout=None,
    return_type=None,
    **kwargs,
):
    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop("sharex", True),
        sharey=kwargs.pop("sharey", True),
        figsize=figsize,
        ax=ax,
        layout=layout,
    )

    _axes = flatten_axes(axes)

    # GH 45465: move the "by" label based on "vert"
    xlabel, ylabel = kwargs.pop("xlabel", None), kwargs.pop("ylabel", None)
    if kwargs.get("vert", True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by

    ax_values = []

    for i, col in enumerate(columns):
        ax = _axes[i]
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_title(col)
        ax_values.append(re_plotf)
        ax.grid(grid)

    result = pd.Series(ax_values, index=columns, copy=False)

    # Return axes in multiplot case, maybe revisit later # 985
    if return_type is None:
        result = axes

    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)

    return result


def boxplot(
    data,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    # validate return_type:
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")

    if isinstance(data, ABCSeries):
        data = data.to_frame("x")
        column = "x"

    def _get_colors():
        #  num_colors=3 is required as method maybe_color_bp takes the colors
        #  in positions 0 and 2.
        #  if colors not provided, use same defaults as DataFrame.plot.box
        result = get_standard_colors(num_colors=3)
        result = np.take(result, [0, 0, 2])
        result = np.append(result, "k")

        colors = kwds.pop("color", None)
        if colors:
            if is_dict_like(colors):
                # replace colors in result array with user-specified colors
                # taken from the colors dict parameter
                # "boxes" value placed in position 0, "whiskers" in 1, etc.
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4)))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                result.fill(colors)

        return result

    def plot_group(keys, values, ax: Axes, **kwds):
        # GH 45465: xlabel/ylabel need to be popped out before plotting happens
        xlabel, ylabel = kwds.pop("xlabel", None), kwds.pop("ylabel", None)
        if xlabel:
            ax.set_xlabel(pprint_thing(xlabel))
        if ylabel:
            ax.set_ylabel(pprint_thing(ylabel))

        keys = [pprint_thing(x) for x in keys]
        values = [np.asarray(remove_na_arraylike(v), dtype=object) for v in values]
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)

        # GH 45465: x/y are flipped when "vert" changes
        _set_ticklabels(
            ax=ax, labels=keys, is_vertical=kwds.get("vert", True), rotation=rot
        )
        maybe_color_bp(bp, color_tup=colors, **kwds)

        # Return axes in multiplot case, maybe revisit later # 985
        if return_type == "dict":
            return bp
        elif return_type == "both":
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            return ax

    colors = _get_colors()
    if column is None:
        columns = None
    elif isinstance(column, (list, tuple)):
        columns = column
    else:
        columns = [column]

    if by is not None:
        # Prefer array return type for 2-D plots to match the subplot layout
        # https://github.com/pandas-dev/pandas/pull/12216#issuecomment-241175580
        result = _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
    else:
        if return_type is None:
            return_type = "axes"
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        if ax is None:
            rc = {"figure.figsize": figsize} if figsize is not None else {}
            with plt.rc_context(rc):
                ax = plt.gca()
        data = data._get_numeric_data()
        naxes = len(data.columns)
        if naxes == 0:
            raise ValueError(
                "boxplot method requires numerical columns, nothing to plot."
            )
        if columns is None:
            columns = data.columns
        else:
            data = data[columns]

        result = plot_group(columns, data.values.T, ax, **kwds)
        ax.grid(grid)

    return result


def boxplot_frame(
    self,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    ax = boxplot(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        grid=grid,
        rot=rot,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwds,
    )
    plt.draw_if_interactive()
    return ax


def boxplot_frame_groupby(
    grouped,
    subplots: bool = True,
    column=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds,
):
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(
            naxes=naxes,
            squeeze=False,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
        )
        axes = flatten_axes(axes)

        ret = pd.Series(dtype=object)

        for (key, group), ax in zip(grouped, axes):
            d = group.boxplot(
                ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds
            )
            ax.set_title(pprint_thing(key))
            ret.loc[key] = d
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped)
        if grouped.axis == 0:
            df = pd.concat(frames, keys=keys, axis=1)
        elif len(frames) > 1:
            df = frames[0].join(frames[1::])
        else:
            df = frames[0]

        # GH 16748, DataFrameGroupby fails when subplots=False and `column` argument
        # is assigned, and in this case, since `df` here becomes MI after groupby,
        # so we need to couple the keys (grouped values) and column (original df
        # column) together to search for subset to plot
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(
            column=column,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            ax=ax,
            figsize=figsize,
            layout=layout,
            **kwds,
        )
    return ret
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    final,
)

import numpy as np

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
)
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import (
    create_iter_data_given_by,
    reformat_hist_y_given_by,
)
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
    set_ticks_props,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pandas._typing import PlottingOrientation

    from pandas import (
        DataFrame,
        Series,
    )


class HistPlot(LinePlot):
    @property
    def _kind(self) -> Literal["hist", "kde"]:
        return "hist"

    def __init__(
        self,
        data,
        bins: int | np.ndarray | list[np.ndarray] = 10,
        bottom: int | np.ndarray = 0,
        *,
        range=None,
        weights=None,
        **kwargs,
    ) -> None:
        if is_list_like(bottom):
            bottom = np.array(bottom)
        self.bottom = bottom

        self._bin_range = range
        self.weights = weights

        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called

        self.bins = self._adjust_bins(bins)

    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
        if is_integer(bins):
            if self.by is not None:
                by_modified = unpack_single_str_list(self.by)
                grouped = self.data.groupby(by_modified)[self.columns]
                bins = [self._calculate_bins(group, bins) for key, group in grouped]
            else:
                bins = self._calculate_bins(self.data, bins)
        return bins

    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        """Calculate bins given data"""
        nd_values = data.infer_objects(copy=False)._get_numeric_data()
        values = np.ravel(nd_values)
        values = values[~isna(values)]

        hist, bins = np.histogram(values, bins=bins, range=self._bin_range)
        return bins

    # error: Signature of "_plot" incompatible with supertype "LinePlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bottom: int | np.ndarray = 0,
        column_num: int = 0,
        stacking_id=None,
        *,
        bins,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        # ignore style
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()

        # Re-create iterated data if `by` is assigned by users
        data = (
            create_iter_data_given_by(self.data, self._kind)
            if self.by is not None
            else self.data
        )

        # error: Argument "data" to "_iter_data" of "MPLPlot" has incompatible
        # type "object"; expected "DataFrame | dict[Hashable, Series | DataFrame]"
        for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
            ax = self._get_ax(i)

            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            self._make_plot_keywords(kwds, y)

            # the bins is multi-dimension array now and each plot need only 1-d and
            # when by is applied, label should be columns that are grouped
            if self.by is not None:
                kwds["bins"] = kwds["bins"][i]
                kwds["label"] = self.columns
                kwds.pop("color")

            if self.weights is not None:
                kwds["weights"] = type(self)._get_column_weights(self.weights, i, y)

            y = reformat_hist_y_given_by(y, self.by)

            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)

            # when by is applied, show title for subplots to know which group it is
            if self.by is not None:
                ax.set_title(pprint_thing(label))

            self._append_legend_handles_labels(artists[0], label)

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """merge BoxPlot/KdePlot properties to passed kwds"""
        # y is required for KdePlot
        kwds["bottom"] = self.bottom
        kwds["bins"] = self.bins

    @final
    @staticmethod
    def _get_column_weights(weights, i: int, y):
        # We allow weights to be a multi-dimensional array, e.g. a (10, 2) array,
        # and each sub-array (10,) will be called in each iteration. If users only
        # provide 1D array, we assume the same weights is used for all iterations
        if weights is not None:
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError(
                        "weights must have the same shape as data, "
                        "or be a single column"
                    ) from err
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.orientation == "horizontal":
            # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible
            # type "Hashable"; expected "str"
            ax.set_xlabel(
                "Frequency"
                if self.xlabel is None
                else self.xlabel  # type: ignore[arg-type]
            )
            ax.set_ylabel(self.ylabel)  # type: ignore[arg-type]
        else:
            ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]
            ax.set_ylabel(
                "Frequency"
                if self.ylabel is None
                else self.ylabel  # type: ignore[arg-type]
            )

    @property
    def orientation(self) -> PlottingOrientation:
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        else:
            return "vertical"


class KdePlot(HistPlot):
    @property
    def _kind(self) -> Literal["kde"]:
        return "kde"

    @property
    def orientation(self) -> Literal["vertical"]:
        return "vertical"

    def __init__(
        self, data, bw_method=None, ind=None, *, weights=None, **kwargs
    ) -> None:
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called
        self.bw_method = bw_method
        self.ind = ind
        self.weights = weights

    @staticmethod
    def _get_ind(y: np.ndarray, ind):
        if ind is None:
            # np.nanmax() and np.nanmin() ignores the missing values
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                1000,
            )
        elif is_integer(ind):
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                ind,
            )
        return ind

    @classmethod
    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    def _plot(  #  type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bw_method=None,
        ind=None,
        column_num=None,
        stacking_id: int | None = None,
        **kwds,
    ):
        from scipy.stats import gaussian_kde

        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method)

        y = gkde.evaluate(ind)
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        kwds["bw_method"] = self.bw_method
        kwds["ind"] = type(self)._get_ind(y, ind=self.ind)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        ax.set_ylabel("Density")


def _grouped_plot(
    plotf,
    data: Series | DataFrame,
    column=None,
    by=None,
    numeric_only: bool = True,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    layout=None,
    rot: float = 0,
    ax=None,
    **kwargs,
):
    # error: Non-overlapping equality check (left operand type: "Optional[Tuple[float,
    # float]]", right operand type: "Literal['default']")
    if figsize == "default":  # type: ignore[comparison-overlap]
        # allowed to specify mpl default with 'default'
        raise ValueError(
            "figsize='default' is no longer supported. "
            "Specify figure size by tuple instead"
        )

    grouped = data.groupby(by)
    if column is not None:
        grouped = grouped[column]

    naxes = len(grouped)
    fig, axes = create_subplots(
        naxes=naxes, figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout
    )

    _axes = flatten_axes(axes)

    for i, (key, group) in enumerate(grouped):
        ax = _axes[i]
        if numeric_only and isinstance(group, ABCDataFrame):
            group = group._get_numeric_data()
        plotf(group, ax, **kwargs)
        ax.set_title(pprint_thing(key))

    return fig, axes


def _grouped_hist(
    data: Series | DataFrame,
    column=None,
    by=None,
    ax=None,
    bins: int = 50,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = False,
    rot: float = 90,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    legend: bool = False,
    **kwargs,
):
    """
    Grouped histogram

    Parameters
    ----------
    data : Series/DataFrame
    column : object, optional
    by : object, optional
    ax : axes, optional
    bins : int, default 50
    figsize : tuple, optional
    layout : optional
    sharex : bool, default False
    sharey : bool, default False
    rot : float, default 90
    grid : bool, default True
    legend: : bool, default False
    kwargs : dict, keyword arguments passed to matplotlib.Axes.hist

    Returns
    -------
    collection of Matplotlib Axes
    """
    if legend:
        assert "label" not in kwargs
        if data.ndim == 1:
            kwargs["label"] = data.name
        elif column is None:
            kwargs["label"] = data.columns
        else:
            kwargs["label"] = column

    def plot_group(group, ax) -> None:
        ax.hist(group.dropna().values, bins=bins, **kwargs)
        if legend:
            ax.legend()

    if xrot is None:
        xrot = rot

    fig, axes = _grouped_plot(
        plot_group,
        data,
        column=column,
        by=by,
        sharex=sharex,
        sharey=sharey,
        ax=ax,
        figsize=figsize,
        layout=layout,
        rot=rot,
    )

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )

    maybe_adjust_figure(
        fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3
    )
    return axes


def hist_series(
    self: Series,
    by=None,
    ax=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    figsize: tuple[float, float] | None = None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    import matplotlib.pyplot as plt

    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")

    if by is None:
        if kwds.get("layout", None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        # hack until the plotting interface is a bit more unified
        fig = kwds.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")
        values = self.dropna().values
        if legend:
            kwds["label"] = self.name
        ax.hist(values, bins=bins, **kwds)
        if legend:
            ax.legend()
        ax.grid(grid)
        axes = np.array([ax])

        # error: Argument 1 to "set_ticks_props" has incompatible type "ndarray[Any,
        # dtype[Any]]"; expected "Axes | Sequence[Axes]"
        set_ticks_props(
            axes,  # type: ignore[arg-type]
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )

    else:
        if "figure" in kwds:
            raise ValueError(
                "Cannot pass 'figure' when using the "
                "'by' argument, since a new 'Figure' instance will be created"
            )
        axes = _grouped_hist(
            self,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )

    if hasattr(axes, "ndim"):
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    return axes


def hist_frame(
    data: DataFrame,
    column=None,
    by=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    ax=None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    layout=None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    if by is not None:
        axes = _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
        return axes

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndex)):
            column = [column]
        data = data[column]
    # GH32590
    data = data.select_dtypes(
        include=(np.number, "datetime64", "datetimetz"), exclude="timedelta"
    )
    naxes = len(data.columns)

    if naxes == 0:
        raise ValueError(
            "hist method requires numerical or datetime columns, nothing to plot."
        )

    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    _axes = flatten_axes(axes)

    can_set_label = "label" not in kwds

    for i, col in enumerate(data.columns):
        ax = _axes[i]
        if legend and can_set_label:
            kwds["label"] = col
        ax.hist(data[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)
        if legend:
            ax.legend()

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)

    return axes
<|endoftext|># TODO: Use the fact that axis can have units to simplify the process

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    OFFSET_TO_PERIOD_FREQSTR,
    FreqGroup,
)

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCPeriodIndex,
    ABCTimedeltaIndex,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
    TimeSeries_DateFormatter,
    TimeSeries_DateLocator,
    TimeSeries_TimedeltaFormatter,
)
from pandas.tseries.frequencies import (
    get_period_alias,
    is_subperiod,
    is_superperiod,
)

if TYPE_CHECKING:
    from datetime import timedelta

    from matplotlib.axes import Axes

    from pandas._typing import NDFrameT

    from pandas import (
        DataFrame,
        DatetimeIndex,
        Index,
        PeriodIndex,
        Series,
    )

# ---------------------------------------------------------------------
# Plotting functions and monkey patches


def maybe_resample(series: Series, ax: Axes, kwargs: dict[str, Any]):
    # resample against axes freq if necessary

    if "how" in kwargs:
        raise ValueError(
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )

    freq, ax_freq = _get_freq(ax, series)

    if freq is None:  # pragma: no cover
        raise ValueError("Cannot use dynamic axis without frequency info")

    # Convert DatetimeIndex to PeriodIndex
    if isinstance(series.index, ABCDatetimeIndex):
        series = series.to_period(freq=freq)

    if ax_freq is not None and freq != ax_freq:
        if is_superperiod(freq, ax_freq):  # upsample input
            series = series.copy()
            # error: "Index" has no attribute "asfreq"
            series.index = series.index.asfreq(  # type: ignore[attr-defined]
                ax_freq, how="s"
            )
            freq = ax_freq
        elif _is_sup(freq, ax_freq):  # one is weekly
            how = "last"
            series = getattr(series.resample("D"), how)().dropna()
            series = getattr(series.resample(ax_freq), how)().dropna()
            freq = ax_freq
        elif is_subperiod(freq, ax_freq) or _is_sub(freq, ax_freq):
            _upsample_others(ax, freq, kwargs)
        else:  # pragma: no cover
            raise ValueError("Incompatible frequency conversion")
    return freq, series


def _is_sub(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_subperiod("D", f2)) or (
        f2.startswith("W") and is_subperiod(f1, "D")
    )


def _is_sup(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_superperiod("D", f2)) or (
        f2.startswith("W") and is_superperiod(f1, "D")
    )


def _upsample_others(ax: Axes, freq: BaseOffset, kwargs: dict[str, Any]) -> None:
    legend = ax.get_legend()
    lines, labels = _replot_ax(ax, freq)
    _replot_ax(ax, freq)

    other_ax = None
    if hasattr(ax, "left_ax"):
        other_ax = ax.left_ax
    if hasattr(ax, "right_ax"):
        other_ax = ax.right_ax

    if other_ax is not None:
        rlines, rlabels = _replot_ax(other_ax, freq)
        lines.extend(rlines)
        labels.extend(rlabels)

    if legend is not None and kwargs.get("legend", True) and len(lines) > 0:
        title: str | None = legend.get_title().get_text()
        if title == "None":
            title = None
        ax.legend(lines, labels, loc="best", title=title)


def _replot_ax(ax: Axes, freq: BaseOffset):
    data = getattr(ax, "_plot_data", None)

    # clear current axes and data
    # TODO #54485
    ax._plot_data = []  # type: ignore[attr-defined]
    ax.clear()

    decorate_axes(ax, freq)

    lines = []
    labels = []
    if data is not None:
        for series, plotf, kwds in data:
            series = series.copy()
            idx = series.index.asfreq(freq, how="S")
            series.index = idx
            # TODO #54485
            ax._plot_data.append((series, plotf, kwds))  # type: ignore[attr-defined]

            # for tsplot
            if isinstance(plotf, str):
                from pandas.plotting._matplotlib import PLOT_CLASSES

                plotf = PLOT_CLASSES[plotf]._plot

            lines.append(plotf(ax, series.index._mpl_repr(), series.values, **kwds)[0])
            labels.append(pprint_thing(series.name))

    return lines, labels


def decorate_axes(ax: Axes, freq: BaseOffset) -> None:
    """Initialize axes for time-series plotting"""
    if not hasattr(ax, "_plot_data"):
        # TODO #54485
        ax._plot_data = []  # type: ignore[attr-defined]

    # TODO #54485
    ax.freq = freq  # type: ignore[attr-defined]
    xaxis = ax.get_xaxis()
    # TODO #54485
    xaxis.freq = freq  # type: ignore[attr-defined]


def _get_ax_freq(ax: Axes):
    """
    Get the freq attribute of the ax object if set.
    Also checks shared axes (eg when using secondary yaxis, sharex=True
    or twinx)
    """
    ax_freq = getattr(ax, "freq", None)
    if ax_freq is None:
        # check for left/right ax in case of secondary yaxis
        if hasattr(ax, "left_ax"):
            ax_freq = getattr(ax.left_ax, "freq", None)
        elif hasattr(ax, "right_ax"):
            ax_freq = getattr(ax.right_ax, "freq", None)
    if ax_freq is None:
        # check if a shared ax (sharex/twinx) has already freq set
        shared_axes = ax.get_shared_x_axes().get_siblings(ax)
        if len(shared_axes) > 1:
            for shared_ax in shared_axes:
                ax_freq = getattr(shared_ax, "freq", None)
                if ax_freq is not None:
                    break
    return ax_freq


def _get_period_alias(freq: timedelta | BaseOffset | str) -> str | None:
    if isinstance(freq, BaseOffset):
        freqstr = freq.name
    else:
        freqstr = to_offset(freq, is_period=True).rule_code

    return get_period_alias(freqstr)


def _get_freq(ax: Axes, series: Series):
    # get frequency from data
    freq = getattr(series.index, "freq", None)
    if freq is None:
        freq = getattr(series.index, "inferred_freq", None)
        freq = to_offset(freq, is_period=True)

    ax_freq = _get_ax_freq(ax)

    # use axes freq if no data freq
    if freq is None:
        freq = ax_freq

    # get the period frequency
    freq = _get_period_alias(freq)
    return freq, ax_freq


def use_dynamic_x(ax: Axes, data: DataFrame | Series) -> bool:
    freq = _get_index_freq(data.index)
    ax_freq = _get_ax_freq(ax)

    if freq is None:  # convert irregular if axes has freq info
        freq = ax_freq
    # do not use tsplot if irregular was plotted first
    elif (ax_freq is None) and (len(ax.get_lines()) > 0):
        return False

    if freq is None:
        return False

    freq_str = _get_period_alias(freq)

    if freq_str is None:
        return False

    # FIXME: hack this for 0.10.1, creating more technical debt...sigh
    if isinstance(data.index, ABCDatetimeIndex):
        # error: "BaseOffset" has no attribute "_period_dtype_code"
        freq_str = OFFSET_TO_PERIOD_FREQSTR.get(freq_str, freq_str)
        base = to_offset(
            freq_str, is_period=True
        )._period_dtype_code  # type: ignore[attr-defined]
        x = data.index
        if base <= FreqGroup.FR_DAY.value:
            return x[:1].is_normalized
        period = Period(x[0], freq_str)
        assert isinstance(period, Period)
        return period.to_timestamp().tz_localize(x.tz) == x[0]
    return True


def _get_index_freq(index: Index) -> BaseOffset | None:
    freq = getattr(index, "freq", None)
    if freq is None:
        freq = getattr(index, "inferred_freq", None)
        if freq == "B":
            # error: "Index" has no attribute "dayofweek"
            weekdays = np.unique(index.dayofweek)  # type: ignore[attr-defined]
            if (5 in weekdays) or (6 in weekdays):
                freq = None

    freq = to_offset(freq)
    return freq


def maybe_convert_index(ax: Axes, data: NDFrameT) -> NDFrameT:
    # tsplot converts automatically, but don't want to convert index
    # over and over for DataFrames
    if isinstance(data.index, (ABCDatetimeIndex, ABCPeriodIndex)):
        freq: str | BaseOffset | None = data.index.freq

        if freq is None:
            # We only get here for DatetimeIndex
            data.index = cast("DatetimeIndex", data.index)
            freq = data.index.inferred_freq
            freq = to_offset(freq)

        if freq is None:
            freq = _get_ax_freq(ax)

        if freq is None:
            raise ValueError("Could not get frequency alias for plotting")

        freq_str = _get_period_alias(freq)

        with warnings.catch_warnings():
            # suppress Period[B] deprecation warning
            # TODO: need to find an alternative to this before the deprecation
            #  is enforced!
            warnings.filterwarnings(
                "ignore",
                r"PeriodDtype\[B\] is deprecated",
                category=FutureWarning,
            )

            if isinstance(data.index, ABCDatetimeIndex):
                data = data.tz_localize(None).to_period(freq=freq_str)
            elif isinstance(data.index, ABCPeriodIndex):
                data.index = data.index.asfreq(freq=freq_str)
    return data


# Patch methods for subplot.


def _format_coord(freq, t, y) -> str:
    time_period = Period(ordinal=int(t), freq=freq)
    return f"t = {time_period}  y = {y:8f}"


def format_dateaxis(
    subplot, freq: BaseOffset, index: DatetimeIndex | PeriodIndex
) -> None:
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
    from matplotlib import pylab

    # handle index specific formatting
    # Note: DatetimeIndex does not use this
    # interface. DatetimeIndex uses matplotlib.date directly
    if isinstance(index, ABCPeriodIndex):
        majlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_locator(majlocator)
        subplot.xaxis.set_minor_locator(minlocator)

        majformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_formatter(majformatter)
        subplot.xaxis.set_minor_formatter(minformatter)

        # x and y coord info
        subplot.format_coord = functools.partial(_format_coord, freq)

    elif isinstance(index, ABCTimedeltaIndex):
        subplot.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter())
    else:
        raise TypeError("index type not supported")

    pylab.draw_if_interactive()
<|endoftext|>from __future__ import annotations

import random
from typing import TYPE_CHECKING

from matplotlib import patches
import matplotlib.lines as mlines
import numpy as np

from pandas.core.dtypes.missing import notna

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    do_adjust_figure,
    maybe_adjust_figure,
    set_ticks_props,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax=None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds=None,
    hist_kwds=None,
    range_padding: float = 0.05,
    **kwds,
):
    df = frame._get_numeric_data()
    n = df.columns.size
    naxes = n * n
    fig, axes = create_subplots(naxes=naxes, figsize=figsize, ax=ax, squeeze=False)

    # no gaps between subplots
    maybe_adjust_figure(fig, wspace=0, hspace=0)

    mask = notna(df)

    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # GH 14855
    kwds.setdefault("edgecolors", "none")

    boundaries_list = []
    for a in df.columns:
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    for i, a in enumerate(df.columns):
        for j, b in enumerate(df.columns):
            ax = axes[i, j]

            if i == j:
                values = df[a].values[mask[a].values]

                # Deal with the diagonal by drawing a histogram there.
                if diagonal == "hist":
                    ax.hist(values, **hist_kwds)

                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)

                ax.set_xlim(boundaries_list[i])

            else:
                common = (mask[a] & mask[b]).values

                ax.scatter(
                    df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds
                )

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(b)
            ax.set_ylabel(a)

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)

    if len(df.columns) > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)

    set_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    return axes


def _get_marker_compat(marker):
    if marker not in mlines.lineMarkers:
        return "o"
    return marker


def radviz(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame.drop(class_column, axis=1).apply(normalize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    to_plot: dict[Hashable, list[list]] = {}
    colors = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )

    for kls in classes:
        to_plot[kls] = [[], []]

    m = len(frame.columns) - 1
    s = np.array(
        [(np.cos(t), np.sin(t)) for t in [2 * np.pi * (i / m) for i in range(m)]]
    )

    for i in range(n):
        row = df.iloc[i].values
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        y = (s * row_).sum(axis=0) / row.sum()
        kls = class_col.iat[i]
        to_plot[kls][0].append(y[0])
        to_plot[kls][1].append(y[1])

    for i, kls in enumerate(classes):
        ax.scatter(
            to_plot[kls][0],
            to_plot[kls][1],
            color=colors[i],
            label=pprint_thing(kls),
            **kwds,
        )
    ax.legend()

    ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor="none"))

    for xy, name in zip(s, df.columns):
        ax.add_patch(patches.Circle(xy, radius=0.025, facecolor="gray"))

        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(
                xy[0] - 0.025, xy[1] - 0.025, name, ha="right", va="top", size="small"
            )
        elif xy[0] < 0.0 <= xy[1]:
            ax.text(
                xy[0] - 0.025,
                xy[1] + 0.025,
                name,
                ha="right",
                va="bottom",
                size="small",
            )
        elif xy[1] < 0.0 <= xy[0]:
            ax.text(
                xy[0] + 0.025, xy[1] - 0.025, name, ha="left", va="top", size="small"
            )
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(
                xy[0] + 0.025, xy[1] + 0.025, name, ha="left", va="bottom", size="small"
            )

    ax.axis("equal")
    return ax


def andrews_curves(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    samples: int = 200,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def function(amplitudes):
        def f(t):
            x1 = amplitudes[0]
            result = x1 / np.sqrt(2.0)

            # Take the rest of the coefficients and resize them
            # appropriately. Take a copy of amplitudes as otherwise numpy
            # deletes the element from amplitudes itself.
            coeffs = np.delete(np.copy(amplitudes), 0)
            coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))

            # Generate the harmonics and arguments for the sin and cos
            # functions.
            harmonics = np.arange(0, coeffs.shape[0]) + 1
            trig_args = np.outer(harmonics, t)

            result += np.sum(
                coeffs[:, 0, np.newaxis] * np.sin(trig_args)
                + coeffs[:, 1, np.newaxis] * np.cos(trig_args),
                axis=0,
            )
            return result

        return f

    n = len(frame)
    class_col = frame[class_column]
    classes = frame[class_column].drop_duplicates()
    df = frame.drop(class_column, axis=1)
    t = np.linspace(-np.pi, np.pi, samples)
    used_legends: set[str] = set()

    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    colors = dict(zip(classes, color_values))
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-np.pi, np.pi)
    for i in range(n):
        row = df.iloc[i].values
        f = function(row)
        y = f(t)
        kls = class_col.iat[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(t, y, color=colors[kls], label=label, **kwds)
        else:
            ax.plot(t, y, color=colors[kls], **kwds)

    ax.legend(loc="upper right")
    ax.grid()
    return ax


def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds,
) -> Figure:
    import matplotlib.pyplot as plt

    # TODO: is the failure mentioned below still relevant?
    # random.sample(ndarray, int) fails on python 3.3, sigh
    data = list(series.values)
    samplings = [random.sample(data, size) for _ in range(samples)]

    means = np.array([np.mean(sampling) for sampling in samplings])
    medians = np.array([np.median(sampling) for sampling in samplings])
    midranges = np.array(
        [(min(sampling) + max(sampling)) * 0.5 for sampling in samplings]
    )
    if fig is None:
        fig = plt.figure()
    x = list(range(samples))
    axes = []
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlabel("Sample")
    axes.append(ax1)
    ax1.plot(x, means, **kwds)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlabel("Sample")
    axes.append(ax2)
    ax2.plot(x, medians, **kwds)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlabel("Sample")
    axes.append(ax3)
    ax3.plot(x, midranges, **kwds)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_xlabel("Mean")
    axes.append(ax4)
    ax4.hist(means, **kwds)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlabel("Median")
    axes.append(ax5)
    ax5.hist(medians, **kwds)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlabel("Midrange")
    axes.append(ax6)
    ax6.hist(midranges, **kwds)
    for axis in axes:
        plt.setp(axis.get_xticklabels(), fontsize=8)
        plt.setp(axis.get_yticklabels(), fontsize=8)
    if do_adjust_figure(fig):
        plt.tight_layout()
    return fig


def parallel_coordinates(
    frame: DataFrame,
    class_column,
    cols=None,
    ax: Axes | None = None,
    color=None,
    use_columns: bool = False,
    xticks=None,
    colormap=None,
    axvlines: bool = True,
    axvlines_kwds=None,
    sort_labels: bool = False,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    if axvlines_kwds is None:
        axvlines_kwds = {"linewidth": 1, "color": "black"}

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends: set[str] = set()

    ncols = len(df.columns)

    # determine values to use for xticks
    x: list[int] | Index
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError("Columns must be numeric to be used as xticks")
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError("xticks specified must be numeric")
        if len(xticks) != ncols:
            raise ValueError("Length of xticks must match number of columns")
        x = xticks
    else:
        x = list(range(ncols))

    if ax is None:
        ax = plt.gca()

    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )

    if sort_labels:
        classes = sorted(classes)
        color_values = sorted(color_values)
    colors = dict(zip(classes, color_values))

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(x, y, color=colors[kls], label=label, **kwds)
        else:
            ax.plot(x, y, color=colors[kls], **kwds)

    if axvlines:
        for i in x:
            ax.axvline(i, **axvlines_kwds)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc="upper right")
    ax.grid()
    return ax


def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    # workaround because `c='b'` is hardcoded in matplotlib's scatter method
    import matplotlib.pyplot as plt

    kwds.setdefault("c", plt.rcParams["patch.facecolor"])

    data = series.values
    y1 = data[:-lag]
    y2 = data[lag:]
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("y(t)")
    ax.set_ylabel(f"y(t + {lag})")
    ax.scatter(y1, y2, **kwds)
    return ax


def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    import matplotlib.pyplot as plt

    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if "label" in kwds:
        ax.legend()
    ax.grid()
    return ax


def unpack_single_str_list(keys):
    # GH 42795
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]
    return keys
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.plotting._matplotlib.boxplot import (
    BoxPlot,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
)
from pandas.plotting._matplotlib.converter import (
    deregister,
    register,
)
from pandas.plotting._matplotlib.core import (
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
)
from pandas.plotting._matplotlib.hist import (
    HistPlot,
    KdePlot,
    hist_frame,
    hist_series,
)
from pandas.plotting._matplotlib.misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    scatter_matrix,
)
from pandas.plotting._matplotlib.tools import table

if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot

PLOT_CLASSES: dict[str, type[MPLPlot]] = {
    "line": LinePlot,
    "bar": BarPlot,
    "barh": BarhPlot,
    "box": BoxPlot,
    "hist": HistPlot,
    "kde": KdePlot,
    "area": AreaPlot,
    "pie": PiePlot,
    "scatter": ScatterPlot,
    "hexbin": HexBinPlot,
}


def plot(data, kind, **kwargs):
    # Importing pyplot at the top of the file (before the converters are
    # registered) causes problems in matplotlib 2 (converters seem to not
    # work)
    import matplotlib.pyplot as plt

    if kwargs.pop("reuse_plot", False):
        ax = kwargs.get("ax")
        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            kwargs["ax"] = getattr(ax, "left_ax", ax)
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result


__all__ = [
    "plot",
    "hist_series",
    "hist_frame",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "table",
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "lag_plot",
    "parallel_coordinates",
    "radviz",
    "scatter_matrix",
    "register",
    "deregister",
]
<|endoftext|># being a bit too dynamic
from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
import warnings

from matplotlib import ticker
import matplotlib.table
import numpy as np

from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.table import Table

    from pandas import (
        DataFrame,
        Series,
    )


def do_adjust_figure(fig: Figure) -> bool:
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, "get_constrained_layout"):
        return False
    return not fig.get_constrained_layout()


def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None:
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)


def format_date_labels(ax: Axes, rot) -> None:
    # mini version of autofmt_xdate
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        # should always be a Figure but can technically be None
        maybe_adjust_figure(fig, bottom=0.2)  # type: ignore[arg-type]


def table(
    ax, data: DataFrame | Series, rowLabels=None, colLabels=None, **kwargs
) -> Table:
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError("Input data must be DataFrame or Series")

    if rowLabels is None:
        rowLabels = data.index

    if colLabels is None:
        colLabels = data.columns

    cellText = data.values

    # error: Argument "cellText" to "table" has incompatible type "ndarray[Any,
    # Any]"; expected "Sequence[Sequence[str]] | None"
    return matplotlib.table.table(
        ax,
        cellText=cellText,  # type: ignore[arg-type]
        rowLabels=rowLabels,
        colLabels=colLabels,
        **kwargs,
    )


def _get_layout(
    nplots: int,
    layout: tuple[int, int] | None = None,
    layout_type: str = "box",
) -> tuple[int, int]:
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")

        nrows, ncols = layout

        if nrows == -1 and ncols > 0:
            layout = nrows, ncols = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = nrows, ncols = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = "At least one dimension of layout must be positive"
            raise ValueError(msg)

        if nrows * ncols < nplots:
            raise ValueError(
                f"Layout of {nrows}x{ncols} must be larger than required size {nplots}"
            )

        return layout

    if layout_type == "single":
        return (1, 1)
    elif layout_type == "horizontal":
        return (1, nplots)
    elif layout_type == "vertical":
        return (nplots, 1)

    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while k**2 < nplots:
            k += 1

        if (k - 1) * k >= nplots:
            return k, (k - 1)
        else:
            return k, k


# copied from matplotlib/pyplot.py and modified for pandas.plotting


def create_subplots(
    naxes: int,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw=None,
    ax=None,
    layout=None,
    layout_type: str = "box",
    **fig_kw,
):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    import matplotlib.pyplot as plt

    if subplot_kw is None:
        subplot_kw = {}

    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = flatten_axes(ax)
            if layout is not None:
                warnings.warn(
                    "When passing multiple axes, layout keyword is ignored.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if sharex or sharey:
                warnings.warn(
                    "When passing multiple axes, sharex and sharey "
                    "are ignored. These settings must be specified when creating axes.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return fig, ax
            else:
                raise ValueError(
                    f"The number of passed axes must be {naxes}, the "
                    "same as the output plot"
                )

        fig = ax.get_figure()
        # if ax is passed and a number of subplots is 1, return ax as it is
        if naxes == 1:
            if squeeze:
                return fig, ax
            else:
                return fig, flatten_axes(ax)
        else:
            warnings.warn(
                "To output multiple subplots, the figure containing "
                "the passed axes is being cleared.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            fig.clear()

    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    if sharex:
        subplot_kw["sharex"] = ax0
    if sharey:
        subplot_kw["sharey"] = ax0
    axarr[0] = ax0

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        # Set sharex and sharey to None for blank/dummy axes, these can
        # interfere with proper axis limits on the visible axes if
        # they share axes e.g. issue #7528
        if i >= naxes:
            kwds["sharex"] = None
            kwds["sharey"] = None
        ax = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax

    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        axes = axarr.reshape(nrows, ncols)

    return fig, axes


def _remove_labels_from_axis(axis: Axis) -> None:
    for t in axis.get_majorticklabels():
        t.set_visible(False)

    # set_visible will not be effective if
    # minor axis has NullLocator and NullFormatter (default)
    if isinstance(axis.get_minor_locator(), ticker.NullLocator):
        axis.set_minor_locator(ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), ticker.NullFormatter):
        axis.set_minor_formatter(ticker.FormatStrFormatter(""))
    for t in axis.get_minorticklabels():
        t.set_visible(False)

    axis.get_label().set_visible(False)


def _has_externally_shared_axis(ax1: Axes, compare_axis: str) -> bool:
    """
    Return whether an axis is externally shared.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Axis to query.
    compare_axis : str
        `"x"` or `"y"` according to whether the X-axis or Y-axis is being
        compared.

    Returns
    -------
    bool
        `True` if the axis is externally shared. Otherwise `False`.

    Notes
    -----
    If two axes with different positions are sharing an axis, they can be
    referred to as *externally* sharing the common axis.

    If two axes sharing an axis also have the same position, they can be
    referred to as *internally* sharing the common axis (a.k.a twinning).

    _handle_shared_axes() is only interested in axes externally sharing an
    axis, regardless of whether either of the axes is also internally sharing
    with a third axis.
    """
    if compare_axis == "x":
        axes = ax1.get_shared_x_axes()
    elif compare_axis == "y":
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError(
            "_has_externally_shared_axis() needs 'x' or 'y' as a second parameter"
        )

    axes_siblings = axes.get_siblings(ax1)

    # Retain ax1 and any of its siblings which aren't in the same position as it
    ax1_points = ax1.get_position().get_points()

    for ax2 in axes_siblings:
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True

    return False


def handle_shared_axes(
    axarr: Iterable[Axes],
    nplots: int,
    naxes: int,
    nrows: int,
    ncols: int,
    sharex: bool,
    sharey: bool,
) -> None:
    if nplots > 1:
        row_num = lambda x: x.get_subplotspec().rowspan.start
        col_num = lambda x: x.get_subplotspec().colspan.start

        is_first_col = lambda x: x.get_subplotspec().is_first_col()

        if nrows > 1:
            try:
                # first find out the ax layout,
                # so that we can correctly handle 'gaps"
                layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
                for ax in axarr:
                    layout[row_num(ax), col_num(ax)] = ax.get_visible()

                for ax in axarr:
                    # only the last row of subplots should get x labels -> all
                    # other off layout handles the case that the subplot is
                    # the last in the column, because below is no subplot/gap.
                    if not layout[row_num(ax) + 1, col_num(ax)]:
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

            except IndexError:
                # if gridspec is used, ax.rowNum and ax.colNum may different
                # from layout shape. in this case, use last_row logic
                is_last_row = lambda x: x.get_subplotspec().is_last_row()
                for ax in axarr:
                    if is_last_row(ax):
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

        if ncols > 1:
            for ax in axarr:
                # only the first column should get y labels -> set all other to
                # off as we only have labels in the first column and we always
                # have a subplot there, we can skip the layout test
                if is_first_col(ax):
                    continue
                if sharey or _has_externally_shared_axis(ax, "y"):
                    _remove_labels_from_axis(ax.yaxis)


def flatten_axes(axes: Axes | Sequence[Axes]) -> np.ndarray:
    if not is_list_like(axes):
        return np.array([axes])
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        return np.asarray(axes).ravel()
    return np.array(axes)


def set_ticks_props(
    axes: Axes | Sequence[Axes],
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
):
    import matplotlib.pyplot as plt

    for ax in flatten_axes(axes):
        if xlabelsize is not None:
            plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            plt.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            plt.setp(ax.get_yticklabels(), rotation=yrot)
    return axes


def get_all_lines(ax: Axes) -> list[Line2D]:
    lines = ax.get_lines()

    if hasattr(ax, "right_ax"):
        lines += ax.right_ax.get_lines()

    if hasattr(ax, "left_ax"):
        lines += ax.left_ax.get_lines()

    return lines


def get_xlim(lines: Iterable[Line2D]) -> tuple[float, float]:
    left, right = np.inf, -np.inf
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return left, right
<|endoftext|>from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)
import warnings

import matplotlib as mpl
import numpy as np

from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_bool,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_number,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
    decorate_axes,
    format_dateaxis,
    maybe_convert_index,
    maybe_resample,
    use_dynamic_x,
)
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    format_date_labels,
    get_all_lines,
    get_xlim,
    handle_shared_axes,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure

    from pandas._typing import (
        IndexLabel,
        NDFrameT,
        PlottingOrientation,
        npt,
    )

    from pandas import Series


def _color_in_style(style: str) -> bool:
    """
    Check if there is a color letter in the style string.
    """
    from matplotlib.colors import BASE_COLORS

    return not set(BASE_COLORS).isdisjoint(style)


class MPLPlot(ABC):
    """
    Base class for assembling a pandas plot using matplotlib

    Parameters
    ----------
    data :

    """

    @property
    @abstractmethod
    def _kind(self) -> str:
        """Specify kind str. Must be overridden in child class"""
        raise NotImplementedError

    _layout_type = "vertical"
    _default_rot = 0

    @property
    def orientation(self) -> str | None:
        return None

    data: DataFrame

    def __init__(
        self,
        data,
        kind=None,
        by: IndexLabel | None = None,
        subplots: bool | Sequence[Sequence[str]] = False,
        sharex: bool | None = None,
        sharey: bool = False,
        use_index: bool = True,
        figsize: tuple[float, float] | None = None,
        grid=None,
        legend: bool | str = True,
        rot=None,
        ax=None,
        fig=None,
        title=None,
        xlim=None,
        ylim=None,
        xticks=None,
        yticks=None,
        xlabel: Hashable | None = None,
        ylabel: Hashable | None = None,
        fontsize: int | None = None,
        secondary_y: bool | tuple | list | np.ndarray = False,
        colormap=None,
        table: bool = False,
        layout=None,
        include_bool: bool = False,
        column: IndexLabel | None = None,
        *,
        logx: bool | None | Literal["sym"] = False,
        logy: bool | None | Literal["sym"] = False,
        loglog: bool | None | Literal["sym"] = False,
        mark_right: bool = True,
        stacked: bool = False,
        label: Hashable | None = None,
        style=None,
        **kwds,
    ) -> None:
        import matplotlib.pyplot as plt

        # if users assign an empty list or tuple, raise `ValueError`
        # similar to current `df.box` and `df.hist` APIs.
        if by in ([], ()):
            raise ValueError("No group keys passed!")
        self.by = com.maybe_make_list(by)

        # Assign the rest of columns into self.columns if by is explicitly defined
        # while column is not, only need `columns` in hist/box plot when it's DF
        # TODO: Might deprecate `column` argument in future PR (#28373)
        if isinstance(data, DataFrame):
            if column:
                self.columns = com.maybe_make_list(column)
            elif self.by is None:
                self.columns = [
                    col for col in data.columns if is_numeric_dtype(data[col])
                ]
            else:
                self.columns = [
                    col
                    for col in data.columns
                    if col not in self.by and is_numeric_dtype(data[col])
                ]

        # For `hist` plot, need to get grouped original data before `self.data` is
        # updated later
        if self.by is not None and self._kind == "hist":
            self._grouped = data.groupby(unpack_single_str_list(self.by))

        self.kind = kind

        self.subplots = type(self)._validate_subplots_kwarg(
            subplots, data, kind=self._kind
        )

        self.sharex = type(self)._validate_sharex(sharex, ax, by)
        self.sharey = sharey
        self.figsize = figsize
        self.layout = layout

        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.use_index = use_index
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.fontsize = fontsize

        if rot is not None:
            self.rot = rot
            # need to know for format_date_labels since it's rotated to 30 by
            # default
            self._rot_set = True
        else:
            self._rot_set = False
            self.rot = self._default_rot

        if grid is None:
            grid = False if secondary_y else plt.rcParams["axes.grid"]

        self.grid = grid
        self.legend = legend
        self.legend_handles: list[Artist] = []
        self.legend_labels: list[Hashable] = []

        self.logx = type(self)._validate_log_kwd("logx", logx)
        self.logy = type(self)._validate_log_kwd("logy", logy)
        self.loglog = type(self)._validate_log_kwd("loglog", loglog)
        self.label = label
        self.style = style
        self.mark_right = mark_right
        self.stacked = stacked

        # ax may be an Axes object or (if self.subplots) an ndarray of
        #  Axes objects
        self.ax = ax
        # TODO: deprecate fig keyword as it is ignored, not passed in tests
        #  as of 2023-11-05

        # parse errorbar input if given
        xerr = kwds.pop("xerr", None)
        yerr = kwds.pop("yerr", None)
        nseries = self._get_nseries(data)
        xerr, data = type(self)._parse_errorbars("xerr", xerr, data, nseries)
        yerr, data = type(self)._parse_errorbars("yerr", yerr, data, nseries)
        self.errors = {"xerr": xerr, "yerr": yerr}
        self.data = data

        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y

        # ugly TypeError if user passes matplotlib's `cmap` name.
        # Probably better to accept either.
        if "cmap" in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        if "cmap" in kwds:
            self.colormap = kwds.pop("cmap")
        else:
            self.colormap = colormap

        self.table = table
        self.include_bool = include_bool

        self.kwds = kwds

        color = kwds.pop("color", lib.no_default)
        self.color = self._validate_color_args(color, self.colormap)
        assert "color" not in self.kwds

        self.data = self._ensure_frame(self.data)

    @final
    @staticmethod
    def _validate_sharex(sharex: bool | None, ax, by) -> bool:
        if sharex is None:
            # if by is defined, subplots are used and sharex should be False
            if ax is None and by is None:  # pylint: disable=simplifiable-if-statement
                sharex = True
            else:
                # if we get an axis, the users should do the visibility
                # setting...
                sharex = False
        elif not is_bool(sharex):
            raise TypeError("sharex must be a bool or None")
        return bool(sharex)

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, str) and value == "sym")
        ):
            return value
        raise ValueError(
            f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'"
        )

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: bool | Sequence[Sequence[str]], data: Series | DataFrame, kind: str
    ) -> bool | list[tuple[int, ...]]:
        """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
        - add missing columns in a group of their own
        See comments in code below for more details.

        Parameters
        ----------
        subplots : subplots parameters as passed to PlotAccessor

        Returns
        -------
        validated subplots : a bool or a list of tuples of column indices. Columns
        in the same tuple will be grouped together in the resulting plot.
        """

        if isinstance(subplots, bool):
            return subplots
        elif not isinstance(subplots, Iterable):
            raise ValueError("subplots should be a bool or an iterable")

        supported_kinds = (
            "line",
            "bar",
            "barh",
            "hist",
            "kde",
            "density",
            "area",
            "pie",
        )
        if kind not in supported_kinds:
            raise ValueError(
                "When subplots is an iterable, kind must be "
                f"one of {', '.join(supported_kinds)}. Got {kind}."
            )

        if isinstance(data, ABCSeries):
            raise NotImplementedError(
                "An iterable subplots for a Series is not supported."
            )

        columns = data.columns
        if isinstance(columns, ABCMultiIndex):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with a MultiIndex column "
                "is not supported."
            )

        if columns.nunique() != len(columns):
            raise NotImplementedError(
                "An iterable subplots for a DataFrame with non-unique column "
                "labels is not supported."
            )

        # subplots is a list of tuples where each tuple is a group of
        # columns to be grouped together (one ax per group).
        # we consolidate the subplots list such that:
        # - the tuples contain indices instead of column names
        # - the columns that aren't yet in the list are added in a group
        #   of their own.
        # For example with columns from a to g, and
        # subplots = [(a, c), (b, f, e)],
        # we end up with [(ai, ci), (bi, fi, ei), (di,), (gi,)]
        # This way, we can handle self.subplots in a homogeneous manner
        # later.
        # TODO: also accept indices instead of just names?

        out = []
        seen_columns: set[Hashable] = set()
        for group in subplots:
            if not is_list_like(group):
                raise ValueError(
                    "When subplots is an iterable, each entry "
                    "should be a list/tuple of column names."
                )
            idx_locs = columns.get_indexer_for(group)
            if (idx_locs == -1).any():
                bad_labels = np.extract(idx_locs == -1, group)
                raise ValueError(
                    f"Column label(s) {list(bad_labels)} not found in the DataFrame."
                )
            unique_columns = set(group)
            duplicates = seen_columns.intersection(unique_columns)
            if duplicates:
                raise ValueError(
                    "Each column should be in only one subplot. "
                    f"Columns {duplicates} were found in multiple subplots."
                )
            seen_columns = seen_columns.union(unique_columns)
            out.append(tuple(idx_locs))

        unseen_columns = columns.difference(seen_columns)
        for column in unseen_columns:
            idx_loc = columns.get_loc(column)
            out.append((idx_loc,))
        return out

    def _validate_color_args(self, color, colormap):
        if color is lib.no_default:
            # It was not provided by the user
            if "colors" in self.kwds and colormap is not None:
                warnings.warn(
                    "'color' and 'colormap' cannot be used simultaneously. "
                    "Using 'color'",
                    stacklevel=find_stack_level(),
                )
            return None
        if self.nseries == 1 and color is not None and not is_list_like(color):
            # support series.plot(color='green')
            color = [color]

        if isinstance(color, tuple) and self.nseries == 1 and len(color) in (3, 4):
            # support RGB and RGBA tuples in series plot
            color = [color]

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if self.style is not None:
            if is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            # need only a single match
            for s in styles:
                if _color_in_style(s):
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and "
                        "'color' keyword argument. Please use one or the "
                        "other or pass 'style' without a color symbol"
                    )
        return color

    @final
    @staticmethod
    def _iter_data(
        data: DataFrame | dict[Hashable, Series | DataFrame]
    ) -> Iterator[tuple[Hashable, np.ndarray]]:
        for col, values in data.items():
            # This was originally written to use values.values before EAs
            #  were implemented; adding np.asarray(...) to keep consistent
            #  typing.
            yield col, np.asarray(values.values)

    def _get_nseries(self, data: Series | DataFrame) -> int:
        # When `by` is explicitly assigned, grouped data size will be defined, and
        # this will determine number of subplots to have, aka `self.nseries`
        if data.ndim == 1:
            return 1
        elif self.by is not None and self._kind == "hist":
            return len(self._grouped)
        elif self.by is not None and self._kind == "box":
            return len(self.columns)
        else:
            return data.shape[1]

    @final
    @property
    def nseries(self) -> int:
        return self._get_nseries(self.data)

    @final
    def draw(self) -> None:
        self.plt.draw_if_interactive()

    @final
    def generate(self) -> None:
        self._compute_plot_data()
        fig = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)

        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)

    @final
    @staticmethod
    def _has_plotted_object(ax: Axes) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
        if not self.on_right(axes_num):
            # secondary axes may be passed via ax kw
            return self._get_ax_layer(ax)

        if hasattr(ax, "right_ax"):
            # if it has right_ax property, ``ax`` must be left axes
            return ax.right_ax
        elif hasattr(ax, "left_ax"):
            # if it has left_ax property, ``ax`` must be right axes
            return ax
        else:
            # otherwise, create twin axes
            orig_ax, new_ax = ax, ax.twinx()
            # TODO: use Matplotlib public API when available
            new_ax._get_lines = orig_ax._get_lines  # type: ignore[attr-defined]
            # TODO #54485
            new_ax._get_patches_for_fill = (  # type: ignore[attr-defined]
                orig_ax._get_patches_for_fill  # type: ignore[attr-defined]
            )
            # TODO #54485
            orig_ax.right_ax, new_ax.left_ax = (  # type: ignore[attr-defined]
                new_ax,
                orig_ax,
            )

            if not self._has_plotted_object(orig_ax):  # no data on left y
                orig_ax.get_yaxis().set_visible(False)

            if self.logy is True or self.loglog is True:
                new_ax.set_yscale("log")
            elif self.logy == "sym" or self.loglog == "sym":
                new_ax.set_yscale("symlog")
            return new_ax

    @final
    @cache_readonly
    def fig(self) -> Figure:
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    # TODO: can we annotate this as both a Sequence[Axes] and ndarray[object]?
    def axes(self) -> Sequence[Axes]:
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> tuple[Sequence[Axes], Figure]:
        if self.subplots:
            naxes = (
                self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            )
            fig, axes = create_subplots(
                naxes=naxes,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
        elif self.ax is None:
            fig = self.plt.figure(figsize=self.figsize)
            axes = fig.add_subplot(111)
        else:
            fig = self.ax.get_figure()
            if self.figsize is not None:
                fig.set_size_inches(self.figsize)
            axes = self.ax

        axes = flatten_axes(axes)

        if self.logx is True or self.loglog is True:
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            [a.set_xscale("symlog") for a in axes]

        if self.logy is True or self.loglog is True:
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            [a.set_yscale("symlog") for a in axes]

        axes_seq = cast(Sequence["Axes"], axes)
        return axes_seq, fig

    @property
    def result(self):
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and not is_list_like(self.ax):
                # error: "Sequence[Any]" has no attribute "reshape"
                return self.axes.reshape(*self.layout)  # type: ignore[attr-defined]
            else:
                return self.axes
        else:
            sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
            # error: Argument 1 to "len" has incompatible type "Union[bool,
            # Tuple[Any, ...], List[Any], ndarray[Any, Any]]"; expected "Sized"
            all_sec = (
                is_list_like(self.secondary_y)
                and len(self.secondary_y) == self.nseries  # type: ignore[arg-type]
            )
            if sec_true or all_sec:
                # if all data is plotted on secondary, return right axes
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                return self.axes[0]

    @final
    @staticmethod
    def _convert_to_ndarray(data):
        # GH31357: categorical columns are processed separately
        if isinstance(data.dtype, CategoricalDtype):
            return data

        # GH32073: cast to float if values contain nulled integers
        if (is_integer_dtype(data.dtype) or is_float_dtype(data.dtype)) and isinstance(
            data.dtype, ExtensionDtype
        ):
            return data.to_numpy(dtype="float", na_value=np.nan)

        # GH25587: cast ExtensionArray of pandas (IntegerArray, etc.) to
        # np.ndarray before plot.
        if len(data) > 0:
            return np.asarray(data)

        return data

    @final
    def _ensure_frame(self, data) -> DataFrame:
        if isinstance(data, ABCSeries):
            label = self.label
            if label is None and data.name is None:
                label = ""
            if label is None:
                # We'll end up with columns of [0] instead of [None]
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)
        elif self._kind in ("hist", "box"):
            cols = self.columns if self.by is None else self.columns + self.by
            data = data.loc[:, cols]
        return data

    @final
    def _compute_plot_data(self) -> None:
        data = self.data

        # GH15079 reconstruct data if by is defined
        if self.by is not None:
            self.subplots = True
            data = reconstruct_data_with_by(self.data, by=self.by, cols=self.columns)

        # GH16953, infer_objects is needed as fallback, for ``Series``
        # with ``dtype == object``
        data = data.infer_objects(copy=False)
        include_type = [np.number, "datetime", "datetimetz", "timedelta"]

        # GH23719, allow plotting boolean
        if self.include_bool is True:
            include_type.append(np.bool_)

        # GH22799, exclude datetime-like type for boxplot
        exclude_type = None
        if self._kind == "box":
            # TODO: change after solving issue 27881
            include_type = [np.number]
            exclude_type = ["timedelta"]

        # GH 18755, include object and category type for scatter plot
        if self._kind == "scatter":
            include_type.extend(["object", "category", "string"])

        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)

        is_empty = numeric_data.shape[-1] == 0
        # no non-numeric frames or series allowed
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data.apply(type(self)._convert_to_ndarray)

    def _make_plot(self, fig: Figure) -> None:
        raise AbstractMethodError(self)

    @final
    def _add_table(self) -> None:
        if self.table is False:
            return
        elif self.table is True:
            data = self.data.transpose()
        else:
            data = self.table
        ax = self._get_ax(0)
        tools.table(ax, data)

    @final
    def _post_plot_logic_common(self, ax: Axes) -> None:
        """Common post process for each axes"""
        if self.orientation == "vertical" or self.orientation is None:
            type(self)._apply_axis_properties(
                ax.xaxis, rot=self.rot, fontsize=self.fontsize
            )
            type(self)._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )

        elif self.orientation == "horizontal":
            type(self)._apply_axis_properties(
                ax.yaxis, rot=self.rot, fontsize=self.fontsize
            )
            type(self)._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )
        else:  # pragma no cover
            raise ValueError

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data) -> None:
        """Post process for each axes. Overridden in child classes"""

    @final
    def _adorn_subplots(self, fig: Figure) -> None:
        """Common post process unrelated to data"""
        if len(self.axes) > 0:
            all_axes = self._get_subplots(fig)
            nrows, ncols = self._get_axes_layout(fig)
            handle_shared_axes(
                axarr=all_axes,
                nplots=len(all_axes),
                naxes=nrows * ncols,
                nrows=nrows,
                ncols=ncols,
                sharex=self.sharex,
                sharey=self.sharey,
            )

        for ax in self.axes:
            ax = getattr(ax, "right_ax", ax)
            if self.yticks is not None:
                ax.set_yticks(self.yticks)

            if self.xticks is not None:
                ax.set_xticks(self.xticks)

            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            if self.xlim is not None:
                ax.set_xlim(self.xlim)

            # GH9093, currently Pandas does not show ylabel, so if users provide
            # ylabel will set it as ylabel in the plot.
            if self.ylabel is not None:
                ax.set_ylabel(pprint_thing(self.ylabel))

            ax.grid(self.grid)

        if self.title:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.nseries:
                        raise ValueError(
                            "The length of `title` must equal the number "
                            "of columns if using `title` of type `list` "
                            "and `subplots=True`.\n"
                            f"length of title = {len(self.title)}\n"
                            f"number of columns = {self.nseries}"
                        )

                    for ax, title in zip(self.axes, self.title):
                        ax.set_title(title)
                else:
                    fig.suptitle(self.title)
            else:
                if is_list_like(self.title):
                    msg = (
                        "Using `title` of type `list` is not supported "
                        "unless `subplots=True` is passed"
                    )
                    raise ValueError(msg)
                self.axes[0].set_title(self.title)

    @final
    @staticmethod
    def _apply_axis_properties(
        axis: Axis, rot=None, fontsize: int | None = None
    ) -> None:
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
        if rot is not None or fontsize is not None:
            # rot=0 is a valid setting, hence the explicit None check
            labels = axis.get_majorticklabels() + axis.get_minorticklabels()
            for label in labels:
                if rot is not None:
                    label.set_rotation(rot)
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @final
    @property
    def legend_title(self) -> str | None:
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            stringified = map(pprint_thing, self.data.columns.names)
            return ",".join(stringified)

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    @final
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self) -> None:
        ax, leg = self._get_ax_legend(self.axes[0])

        handles = []
        labels = []
        title = ""

        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                # Replace leg.legend_handles because it misses marker info
                if Version(mpl.__version__) < Version("3.7"):
                    handles = leg.legendHandles
                else:
                    handles = leg.legend_handles
                labels = [x.get_text() for x in leg.get_texts()]

            if self.legend:
                if self.legend == "reverse":
                    handles += reversed(self.legend_handles)
                    labels += reversed(self.legend_labels)
                else:
                    handles += self.legend_handles
                    labels += self.legend_labels

                if self.legend_title is not None:
                    title = self.legend_title

            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", title=title)

        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            "No artists with labels found to put in legend.",
                            UserWarning,
                        )
                        ax.legend(loc="best")

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes):
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()

        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return ax, leg

    @final
    @cache_readonly
    def plt(self):
        import matplotlib.pyplot as plt

        return plt

    _need_to_set_index = False

    @final
    def _get_xticks(self):
        index = self.data.index
        is_datetype = index.inferred_type in ("datetime", "date", "datetime64", "time")

        # TODO: be stricter about x?
        x: list[int] | np.ndarray
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                # test_mixed_freq_irreg_period
                x = index.to_timestamp()._mpl_repr()
                # TODO: why do we need to do to_timestamp() here but not other
                #  places where we call mpl_repr?
            elif is_any_real_numeric_dtype(index.dtype):
                # Matplotlib supports numeric values or datetime objects as
                # xaxis values. Taking LBYL approach here, by the time
                # matplotlib raises exception when using non numeric/datetime
                # values for xaxis, several actions are already taken by plt.
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                x = index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))

        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls, ax: Axes, x, y: np.ndarray, style=None, is_errorbar: bool = False, **kwds
    ):
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)

        if isinstance(x, ABCIndex):
            x = x._mpl_repr()

        if is_errorbar:
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        else:
            # prevent style kwarg from going to errorbar, where it is unsupported
            args = (x, y, style) if style is not None else (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self):
        """Specify whether xlabel/ylabel should be used to override index name"""
        return self.xlabel

    @final
    def _get_index_name(self) -> str | None:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ",".join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)

        # GH 45145, override the default axis label if one is provided.
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)

        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax, primary: bool = True):
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, "left_ax", ax)
        else:
            return getattr(ax, "right_ax", ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
        if isinstance(self.subplots, list):
            # Subplots is a list: some columns will be grouped together in the same ax
            return next(
                group_idx
                for (group_idx, group) in enumerate(self.subplots)
                if col_idx in group
            )
        else:
            # subplots is True: one ax per column
            return col_idx

    @final
    def _get_ax(self, i: int):
        # get the twinx ax if appropriate
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            # error: Unsupported target for indexed assignment ("Sequence[Any]")
            self.axes[i] = ax  # type: ignore[index]
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)

        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int):
        if isinstance(self.secondary_y, bool):
            return self.secondary_y

        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y

    @final
    def _apply_style_colors(
        self, colors, kwds: dict[str, Any], col_num: int, label: str
    ):
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style

        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return style, kwds

    def _get_colors(
        self,
        num_colors: int | None = None,
        color_kwds: str = "color",
    ):
        if num_colors is None:
            num_colors = self.nseries
        if color_kwds == "color":
            color = self.color
        else:
            color = self.kwds.get(color_kwds)
        return get_standard_colors(
            num_colors=num_colors,
            colormap=self.colormap,
            color=color,
        )

    # TODO: tighter typing for first return?
    @final
    @staticmethod
    def _parse_errorbars(
        label: str, err, data: NDFrameT, nseries: int
    ) -> tuple[Any, NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return None, data

        def match_labels(data, e):
            e = e.reindex(data.index)
            return e

        # key-matched DataFrame
        if isinstance(err, ABCDataFrame):
            err = match_labels(data, err)
        # key-matched dict
        elif isinstance(err, dict):
            pass

        # Series of error values
        elif isinstance(err, ABCSeries):
            # broadcast error series across data
            err = match_labels(data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (nseries, 1))

        # errors are a column in the dataframe
        elif isinstance(err, str):
            evalues = data[err].values
            data = data[data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (nseries, 1))

        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                # raw error values
                err = np.atleast_2d(err)

            err_shape = err.shape

            # asymmetrical error bars
            if isinstance(data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(data):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape (2, {len(data)})"
                    )
            elif isinstance(data, ABCDataFrame) and err.ndim == 3:
                if (
                    (err_shape[0] != nseries)
                    or (err_shape[1] != 2)
                    or (err_shape[2] != len(data))
                ):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape ({nseries}, 2, {len(data)})"
                    )

            # broadcast errors to each data series
            if len(err) == 1:
                err = np.tile(err, (nseries, 1))

        elif is_number(err):
            err = np.tile(
                [err],
                (nseries, len(data)),
            )

        else:
            msg = f"No valid {label} detected"
            raise ValueError(msg)

        return err, data

    @final
    def _get_errorbars(
        self, label=None, index=None, xerr: bool = True, yerr: bool = True
    ) -> dict[str, Any]:
        errors = {}

        for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                # user provided label-matched dataframe of errors
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]

                if err is not None:
                    errors[kw] = err
        return errors

    @final
    def _get_subplots(self, fig: Figure):
        if Version(mpl.__version__) < Version("3.8"):
            from matplotlib.axes import Subplot as Klass
        else:
            from matplotlib.axes import Axes as Klass

        return [
            ax
            for ax in fig.get_axes()
            if (isinstance(ax, Klass) and ax.get_subplotspec() is not None)
        ]

    @final
    def _get_axes_layout(self, fig: Figure) -> tuple[int, int]:
        axes = self._get_subplots(fig)
        x_set = set()
        y_set = set()
        for ax in axes:
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))


class PlanePlot(MPLPlot, ABC):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """

    _layout_type = "single"

    def __init__(self, data, x, y, **kwargs) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        if is_integer(x) and not self.data.columns._holds_integer():
            x = self.data.columns[x]
        if is_integer(y) and not self.data.columns._holds_integer():
            y = self.data.columns[y]

        self.x = x
        self.y = y

    @final
    def _get_nseries(self, data: Series | DataFrame) -> int:
        return 1

    @final
    def _post_plot_logic(self, ax: Axes, data) -> None:
        x, y = self.x, self.y
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible
        # type "Hashable"; expected "str"
        ax.set_xlabel(xlabel)  # type: ignore[arg-type]
        ax.set_ylabel(ylabel)  # type: ignore[arg-type]

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds):
        # Addresses issues #10611 and #10678:
        # When plotting scatterplots and hexbinplots in IPython
        # inline backend the colorbar axis height tends not to
        # exactly match the parent axis height.
        # The difference is due to small fractional differences
        # in floating points with similar representation.
        # To deal with this, this method forces the colorbar
        # height to take the height of the parent axes.
        # For a more detailed description of the issue
        # see the following link:
        # https://github.com/ipython/ipython/issues/11215

        # GH33389, if ax is used multiple times, we should always
        # use the last one which contains the latest information
        # about the ax
        img = ax.collections[-1]
        return fig.colorbar(img, ax=ax, **kwds)


class ScatterPlot(PlanePlot):
    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    def __init__(
        self,
        data,
        x,
        y,
        s=None,
        c=None,
        *,
        colorbar: bool | lib.NoDefault = lib.no_default,
        norm=None,
        **kwargs,
    ) -> None:
        if s is None:
            # hide the matplotlib default for size, in case we want to change
            # the handling of this argument later
            s = 20
        elif is_hashable(s) and s in data.columns:
            s = data[s]
        self.s = s

        self.colorbar = colorbar
        self.norm = norm

        super().__init__(data, x, y, **kwargs)
        if is_integer(c) and not self.data.columns._holds_integer():
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self, fig: Figure) -> None:
        x, y, c, data = self.x, self.y, self.c, self.data
        ax = self.axes[0]

        c_is_column = is_hashable(c) and c in self.data.columns

        color_by_categorical = c_is_column and isinstance(
            self.data[c].dtype, CategoricalDtype
        )

        color = self.color
        c_values = self._get_c_values(color, color_by_categorical, c_is_column)
        norm, cmap = self._get_norm_and_cmap(c_values, color_by_categorical)
        cb = self._get_colorbar(c_values, c_is_column)

        if self.legend:
            label = self.label
        else:
            label = None
        scatter = ax.scatter(
            data[x].values,
            data[y].values,
            c=c_values,
            label=label,
            cmap=cmap,
            norm=norm,
            s=self.s,
            **self.kwds,
        )
        if cb:
            cbar_label = c if c_is_column else ""
            cbar = self._plot_colorbar(ax, fig=fig, label=cbar_label)
            if color_by_categorical:
                n_cats = len(self.data[c].cat.categories)
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(self.data[c].cat.categories)

        if label is not None:
            self._append_legend_handles_labels(
                # error: Argument 2 to "_append_legend_handles_labels" of
                # "MPLPlot" has incompatible type "Hashable"; expected "str"
                scatter,
                label,  # type: ignore[arg-type]
            )

        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values, linestyle="none", **err_kwds)

    def _get_c_values(self, color, color_by_categorical: bool, c_is_column: bool):
        c = self.c
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        if c is None and color is None:
            c_values = self.plt.rcParams["patch.facecolor"]
        elif color is not None:
            c_values = color
        elif color_by_categorical:
            c_values = self.data[c].cat.codes
        elif c_is_column:
            c_values = self.data[c].values
        else:
            c_values = c
        return c_values

    def _get_norm_and_cmap(self, c_values, color_by_categorical: bool):
        c = self.c
        if self.colormap is not None:
            cmap = mpl.colormaps.get_cmap(self.colormap)
        # cmap is only used if c_values are integers, otherwise UserWarning.
        # GH-53908: additionally call isinstance() because is_integer_dtype
        # returns True for "b" (meaning "blue" and not int8 in this context)
        elif not isinstance(c_values, str) and is_integer_dtype(c_values):
            # pandas uses colormap, matplotlib uses cmap.
            cmap = mpl.colormaps["Greys"]
        else:
            cmap = None

        if color_by_categorical and cmap is not None:
            from matplotlib import colors

            n_cats = len(self.data[c].cat.categories)
            cmap = colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            bounds = np.linspace(0, n_cats, n_cats + 1)
            norm = colors.BoundaryNorm(bounds, cmap.N)
            # TODO: warn that we are ignoring self.norm if user specified it?
            #  Doesn't happen in any tests 2023-11-09
        else:
            norm = self.norm
        return norm, cmap

    def _get_colorbar(self, c_values, c_is_column: bool) -> bool:
        # plot colorbar if
        # 1. colormap is assigned, and
        # 2.`c` is a column containing only numeric values
        plot_colorbar = self.colormap or c_is_column
        cb = self.colorbar
        if cb is lib.no_default:
            return is_numeric_dtype(c_values) and plot_colorbar
        return cb


class HexBinPlot(PlanePlot):
    @property
    def _kind(self) -> Literal["hexbin"]:
        return "hexbin"

    def __init__(self, data, x, y, C=None, *, colorbar: bool = True, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
        if is_integer(C) and not self.data.columns._holds_integer():
            C = self.data.columns[C]
        self.C = C

        self.colorbar = colorbar

        # Scatter plot allows to plot objects data
        if len(self.data[self.x]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires x column to be numeric")
        if len(self.data[self.y]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires y column to be numeric")

    def _make_plot(self, fig: Figure) -> None:
        x, y, data, C = self.x, self.y, self.data, self.C
        ax = self.axes[0]
        # pandas uses colormap, matplotlib uses cmap.
        cmap = self.colormap or "BuGn"
        cmap = mpl.colormaps.get_cmap(cmap)
        cb = self.colorbar

        if C is None:
            c_values = None
        else:
            c_values = data[C].values

        ax.hexbin(data[x].values, data[y].values, C=c_values, cmap=cmap, **self.kwds)
        if cb:
            self._plot_colorbar(ax, fig=fig)

    def _make_legend(self) -> None:
        pass


class LinePlot(MPLPlot):
    _default_rot = 0

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    @property
    def _kind(self) -> Literal["line", "area", "hist", "kde", "box"]:
        return "line"

    def __init__(self, data, **kwargs) -> None:
        from pandas.plotting import plot_params

        MPLPlot.__init__(self, data, **kwargs)
        if self.stacked:
            self.data = self.data.fillna(value=0)
        self.x_compat = plot_params["x_compat"]
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))

    @final
    def _is_ts_plot(self) -> bool:
        # this is slightly deceptive
        return not self.x_compat and self.use_index and self._use_dynamic_x()

    @final
    def _use_dynamic_x(self) -> bool:
        return use_dynamic_x(self._get_ax(0), self.data)

    def _make_plot(self, fig: Figure) -> None:
        if self._is_ts_plot():
            data = maybe_convert_index(self._get_ax(0), self.data)

            x = data.index  # dummy, not used
            plotf = self._ts_plot
            it = data.items()
        else:
            x = self._get_xticks()
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, Any, Any, Any, Any, Any, KwArg(Any)], Any]", variable has
            # type "Callable[[Any, Any, Any, Any, KwArg(Any)], Any]")
            plotf = self._plot  # type: ignore[assignment]
            # error: Incompatible types in assignment (expression has type
            # "Iterator[tuple[Hashable, ndarray[Any, Any]]]", variable has
            # type "Iterable[tuple[Hashable, Series]]")
            it = self._iter_data(data=self.data)  # type: ignore[assignment]

        stacking_id = self._get_stacking_id()
        is_errorbar = com.any_not_none(*self.errors.values())

        colors = self._get_colors()
        for i, (label, y) in enumerate(it):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color
            style, kwds = self._apply_style_colors(
                colors,
                kwds,
                i,
                # error: Argument 4 to "_apply_style_colors" of "MPLPlot" has
                # incompatible type "Hashable"; expected "str"
                label,  # type: ignore[arg-type]
            )

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            newlines = plotf(
                ax,
                x,
                y,
                style=style,
                column_num=i,
                stacking_id=stacking_id,
                is_errorbar=is_errorbar,
                **kwds,
            )
            self._append_legend_handles_labels(newlines[0], label)

            if self._is_ts_plot():
                # reset of xlim should be used for ts data
                # TODO: GH28021, should find a way to change view limit on xaxis
                lines = get_all_lines(ax)
                left, right = get_xlim(lines)
                ax.set_xlim(left, right)

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        **kwds,
    ):
        # column_num is used to get the target column from plotf in line and
        # area plots
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return lines

    @final
    def _ts_plot(self, ax: Axes, x, data: Series, style=None, **kwds):
        # accept x to be consistent with normal plot func,
        # x is not passed to tsplot as it uses data.index as x coordinate
        # column_num must be in kwds for stacking purpose
        freq, data = maybe_resample(data, ax, kwds)

        # Set ax with freq info
        decorate_axes(ax, freq)
        # digging deeper
        if hasattr(ax, "left_ax"):
            decorate_axes(ax.left_ax, freq)
        if hasattr(ax, "right_ax"):
            decorate_axes(ax.right_ax, freq)
        # TODO #54485
        ax._plot_data.append((data, self._kind, kwds))  # type: ignore[attr-defined]

        lines = self._plot(ax, data.index, np.asarray(data.values), style=style, **kwds)
        # set date formatter, locators and rescale limits
        # TODO #54485
        format_dateaxis(ax, ax.freq, data.index)  # type: ignore[arg-type, attr-defined]
        return lines

    @final
    def _get_stacking_id(self) -> int | None:
        if self.stacked:
            return id(self.data)
        else:
            return None

    @final
    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id, n: int) -> None:
        if stacking_id is None:
            return
        if not hasattr(ax, "_stacker_pos_prior"):
            # TODO #54485
            ax._stacker_pos_prior = {}  # type: ignore[attr-defined]
        if not hasattr(ax, "_stacker_neg_prior"):
            # TODO #54485
            ax._stacker_neg_prior = {}  # type: ignore[attr-defined]
        # TODO #54485
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]
        # TODO #54485
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]

    @final
    @classmethod
    def _get_stacked_values(
        cls, ax: Axes, stacking_id: int | None, values: np.ndarray, label
    ) -> np.ndarray:
        if stacking_id is None:
            return values
        if not hasattr(ax, "_stacker_pos_prior"):
            # stacker may not be initialized for subplots
            cls._initialize_stacker(ax, stacking_id, len(values))

        if (values >= 0).all():
            # TODO #54485
            return (
                ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )
        elif (values <= 0).all():
            # TODO #54485
            return (
                ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )

        raise ValueError(
            "When stacked is True, each column must be either "
            "all positive or all negative. "
            f"Column '{label}' contains both positive and negative values"
        )

    @final
    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id: int | None, values) -> None:
        if stacking_id is None:
            return
        if (values >= 0).all():
            # TODO #54485
            ax._stacker_pos_prior[stacking_id] += values  # type: ignore[attr-defined]
        elif (values <= 0).all():
            # TODO #54485
            ax._stacker_neg_prior[stacking_id] += values  # type: ignore[attr-defined]

    def _post_plot_logic(self, ax: Axes, data) -> None:
        from matplotlib.ticker import FixedLocator

        def get_label(i):
            if is_float(i) and i.is_integer():
                i = int(i)
            try:
                return pprint_thing(data.index[i])
            except Exception:
                return ""

        if self._need_to_set_index:
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            # error: Argument 1 to "FixedLocator" has incompatible type "ndarray[Any,
            # Any]"; expected "Sequence[float]"
            ax.xaxis.set_major_locator(FixedLocator(xticks))  # type: ignore[arg-type]
            ax.set_xticklabels(xticklabels)

        # If the index is an irregular time series, then by default
        # we rotate the tick labels. The exception is if there are
        # subplots which don't share their x-axes, in which we case
        # we don't rotate the ticklabels as by default the subplots
        # would be too close together.
        condition = (
            not self._use_dynamic_x()
            and (data.index._is_all_dates and self.use_index)
            and (not self.subplots or (self.subplots and self.sharex))
        )

        index_name = self._get_index_name()

        if condition:
            # irregular TS rotated 30 deg. by default
            # probably a better place to check / set this.
            if not self._rot_set:
                self.rot = 30
            format_date_labels(ax, rot=self.rot)

        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)


class AreaPlot(LinePlot):
    @property
    def _kind(self) -> Literal["area"]:
        return "area"

    def __init__(self, data, **kwargs) -> None:
        kwargs.setdefault("stacked", True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Downcasting object dtype arrays",
                category=FutureWarning,
            )
            data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)

        if not self.stacked:
            # use smaller alpha to distinguish overlap
            self.kwds.setdefault("alpha", 0.5)

        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar: bool = False,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # need to remove label, because subplots uses mpl legend as it is
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # get data from the line to get coordinates for fill_between
        xdata, y_values = lines[0].get_data(orig=False)

        # unable to use ``_get_stacked_values`` here to get starting point
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            # TODO #54485
            start = ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
        elif (y <= 0).all():
            # TODO #54485
            start = ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
        else:
            start = np.zeros(len(y))

        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot expects list of artists
        res = [rect]
        return res

    def _post_plot_logic(self, ax: Axes, data) -> None:
        LinePlot._post_plot_logic(self, ax, data)

        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        # do not override the default axis behaviour in case of shared y axes
        if self.ylim is None and not is_shared_y:
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)


class BarPlot(MPLPlot):
    @property
    def _kind(self) -> Literal["bar", "barh"]:
        return "bar"

    _default_rot = 90

    @property
    def orientation(self) -> PlottingOrientation:
        return "vertical"

    def __init__(
        self,
        data,
        *,
        align="center",
        bottom=0,
        left=0,
        width=0.5,
        position=0.5,
        log=False,
        **kwargs,
    ) -> None:
        # we have to treat a series differently than a
        # 1-column DataFrame w.r.t. color handling
        self._is_series = isinstance(data, ABCSeries)
        self.bar_width = width
        self._align = align
        self._position = position
        self.tick_pos = np.arange(len(data))

        if is_list_like(bottom):
            bottom = np.array(bottom)
        if is_list_like(left):
            left = np.array(left)
        self.bottom = bottom
        self.left = left

        self.log = log

        MPLPlot.__init__(self, data, **kwargs)

    @cache_readonly
    def ax_pos(self) -> np.ndarray:
        return self.tick_pos - self.tickoffset

    @cache_readonly
    def tickoffset(self):
        if self.stacked or self.subplots:
            return self.bar_width * self._position
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return self.bar_width * (self._position - 0.5) + w * 0.5
        else:
            return self.bar_width * self._position

    @cache_readonly
    def lim_offset(self):
        if self.stacked or self.subplots:
            if self._align == "edge":
                return self.bar_width / 2
            else:
                return 0
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return w * 0.5
        else:
            return 0

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

    @property
    def _start_base(self):
        return self.bottom

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        ncolors = len(colors)

        pos_prior = neg_prior = np.zeros(len(self.data))
        K = self.nseries

        data = self.data.fillna(0)
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)

            if (("yerr" in kwds) or ("xerr" in kwds)) and (kwds.get("ecolor") is None):
                kwds["ecolor"] = mpl.rcParams["xtick.color"]

            start = 0
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base

            kwds["align"] = self._align
            if self.subplots:
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                ax.set_title(label)
            elif self.stacked:
                mask = y > 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            else:
                w = self.bar_width / K
                rect = self._plot(
                    ax,
                    self.ax_pos + (i + 0.5) * w,
                    y,
                    w,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
            self._append_legend_handles_labels(rect, label)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            str_index = [pprint_thing(key) for key in range(data.shape[0])]

        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset

        self._decorate_ticks(ax, self._get_index_name(), str_index, s_edge, e_edge)

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        ax.set_xlim((start_edge, end_edge))

        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)

        if name is not None and self.use_index:
            ax.set_xlabel(name)


class BarhPlot(BarPlot):
    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    _default_rot = 0

    @property
    def orientation(self) -> Literal["horizontal"]:
        return "horizontal"

    @property
    def _start_base(self):
        return self.left

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _get_custom_index_name(self):
        return self.ylabel

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        # horizontal bars
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)
        # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible type
        # "Hashable | None"; expected "str"
        ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]


class PiePlot(MPLPlot):
    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    _layout_type = "horizontal"

    def __init__(self, data, kind=None, **kwargs) -> None:
        data = data.fillna(value=0)
        if (data < 0).any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(
                f"PiePlot ignores the '{kwd}' keyword",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        return False

    def _validate_color_args(self, color, colormap) -> None:
        # TODO: warn if color is passed and ignored?
        return None

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)

        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            if label is not None:
                label = pprint_thing(label)
                ax.set_ylabel(label)

            kwds = self.kwds.copy()

            def blank_labeler(label, value):
                if value == 0:
                    return ""
                else:
                    return label

            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)
            # labels is used for each wedge's labels
            # Blank out labels for values of 0 so they don't overlap
            # with nonzero wedges
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                blabels = None
            results = ax.pie(y, labels=blabels, **kwds)

            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []

            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)

            # leglabels is used for legend labels
            leglabels = labels if labels is not None else idx
            for _patch, _leglabel in zip(patches, leglabels):
                self._append_legend_handles_labels(_patch, _leglabel)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        pass
<|endoftext|>from __future__ import annotations

import contextlib
import datetime as pydt
from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import matplotlib.dates as mdates
from matplotlib.ticker import (
    AutoLocator,
    Formatter,
    Locator,
)
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    Timestamp,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    periods_per_day,
)
from pandas._typing import (
    F,
    npt,
)

from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_nested_list_like,
)

from pandas import (
    Index,
    Series,
    get_option,
)
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas.core.tools.datetimes as tools

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axis import Axis

    from pandas._libs.tslibs.offsets import BaseOffset


_mpl_units = {}  # Cache for units overwritten by us


def get_pairs():
    pairs = [
        (Timestamp, DatetimeConverter),
        (Period, PeriodConverter),
        (pydt.datetime, DatetimeConverter),
        (pydt.date, DatetimeConverter),
        (pydt.time, TimeConverter),
        (np.datetime64, DatetimeConverter),
    ]
    return pairs


def register_pandas_matplotlib_converters(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pandas_converters():
            return func(*args, **kwargs)

    return cast(F, wrapper)


@contextlib.contextmanager
def pandas_converters() -> Generator[None, None, None]:
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
    value = get_option("plotting.matplotlib.register_converters")

    if value:
        # register for True or "auto"
        register()
    try:
        yield
    finally:
        if value == "auto":
            # only deregister for "auto"
            deregister()


def register() -> None:
    pairs = get_pairs()
    for type_, cls in pairs:
        # Cache previous converter if present
        if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
            previous = munits.registry[type_]
            _mpl_units[type_] = previous
        # Replace with pandas converter
        munits.registry[type_] = cls()


def deregister() -> None:
    # Renamed in pandas.plotting.__init__
    for type_, cls in get_pairs():
        # We use type to catch our classes directly, no inheritance
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)

    # restore the old keys
    for unit, formatter in _mpl_units.items():
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            # make it idempotent by excluding ours.
            munits.registry[unit] = formatter


def _to_ordinalf(tm: pydt.time) -> float:
    tot_sec = tm.hour * 3600 + tm.minute * 60 + tm.second + tm.microsecond / 10**6
    return tot_sec


def time2num(d):
    if isinstance(d, str):
        parsed = Timestamp(d)
        return _to_ordinalf(parsed.time())
    if isinstance(d, pydt.time):
        return _to_ordinalf(d)
    return d


class TimeConverter(munits.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        valid_types = (str, pydt.time)
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return time2num(value)
        if isinstance(value, Index):
            return value.map(time2num)
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [time2num(x) for x in value]
        return value

    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None:
        if unit != "time":
            return None

        majloc = AutoLocator()
        majfmt = TimeFormatter(majloc)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label="time")

    @staticmethod
    def default_units(x, axis) -> str:
        return "time"


# time formatter
class TimeFormatter(Formatter):
    def __init__(self, locs) -> None:
        self.locs = locs

    def __call__(self, x, pos: int | None = 0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt = "%H:%M:%S.%f"
        s = int(x)
        msus = round((x - s) * 10**6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime("%H:%M:%S")

        return pydt.time(h, m).strftime("%H:%M")


# Period Conversion


class PeriodConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, units, axis):
        if is_nested_list_like(values):
            values = [PeriodConverter._convert_1d(v, units, axis) for v in values]
        else:
            values = PeriodConverter._convert_1d(values, units, axis)
        return values

    @staticmethod
    def _convert_1d(values, units, axis):
        if not hasattr(axis, "freq"):
            raise TypeError("Axis must have `freq` set to convert to Periods")
        valid_types = (str, datetime, Period, pydt.date, pydt.time, np.datetime64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Period with BDay freq is deprecated", category=FutureWarning
            )
            warnings.filterwarnings(
                "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
            )
            if (
                isinstance(values, valid_types)
                or is_integer(values)
                or is_float(values)
            ):
                return get_datevalue(values, axis.freq)
            elif isinstance(values, PeriodIndex):
                return values.asfreq(axis.freq).asi8
            elif isinstance(values, Index):
                return values.map(lambda x: get_datevalue(x, axis.freq))
            elif lib.infer_dtype(values, skipna=False) == "period":
                # https://github.com/pandas-dev/pandas/issues/24304
                # convert ndarray[period] -> PeriodIndex
                return PeriodIndex(values, freq=axis.freq).asi8
            elif isinstance(values, (list, tuple, np.ndarray, Index)):
                return [get_datevalue(x, axis.freq) for x in values]
        return values


def get_datevalue(date, freq):
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    elif isinstance(date, (str, datetime, pydt.date, pydt.time, np.datetime64)):
        return Period(date, freq).ordinal
    elif (
        is_integer(date)
        or is_float(date)
        or (isinstance(date, (np.ndarray, Index)) and (date.size == 1))
    ):
        return date
    elif date is None:
        return None
    raise ValueError(f"Unrecognizable date '{date}'")


# Datetime Conversion
class DatetimeConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, unit, axis):
        # values might be a 1-d array, or a list-like of arrays.
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def _convert_1d(values, unit, axis):
        def try_parse(values):
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values

        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return try_parse(values)
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            if isinstance(values, Series):
                # https://github.com/matplotlib/matplotlib/issues/11391
                # Series was skipped. Convert to DatetimeIndex to get asi8
                values = Index(values)
            if isinstance(values, Index):
                values = values.values
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)

            if is_integer_dtype(values) or is_float_dtype(values):
                return values

            try:
                values = tools.to_datetime(values)
            except Exception:
                pass

            values = mdates.date2num(values)

        return values

    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        majloc = PandasAutoDateLocator(tz=tz)
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)

        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label="", default_limits=(datemin, datemax)
        )


class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    def __init__(self, locator, tz=None, defaultfmt: str = "%Y-%m-%d") -> None:
        mdates.AutoDateFormatter.__init__(self, locator, tz, defaultfmt)


class PandasAutoDateLocator(mdates.AutoDateLocator):
    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
        tot_sec = (dmax - dmin).total_seconds()

        if abs(tot_sec) < self.minticks:
            self._freq = -1
            locator = MilliSecondLocator(self.tz)
            locator.set_axis(self.axis)

            # error: Item "None" of "Axis | _DummyAxis | _AxisWrapper | None"
            # has no attribute "get_data_interval"
            locator.axis.set_view_interval(  # type: ignore[union-attr]
                *self.axis.get_view_interval()  # type: ignore[union-attr]
            )
            locator.axis.set_data_interval(  # type: ignore[union-attr]
                *self.axis.get_data_interval()  # type: ignore[union-attr]
            )
            return locator

        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)

    def _get_unit(self):
        return MilliSecondLocator.get_unit_generic(self._freq)


class MilliSecondLocator(mdates.DateLocator):
    UNIT = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz) -> None:
        mdates.DateLocator.__init__(self, tz)
        self._interval = 1.0

    def _get_unit(self):
        return self.get_unit_generic(-1)

    @staticmethod
    def get_unit_generic(freq):
        unit = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        # We need to cap at the endpoints of valid datetime
        nmax, nmin = mdates.date2num((dmax, dmin))

        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            # We went through the whole loop without breaking, default to 1
            self._interval = 1000.0

        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())

        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(
                "MillisecondLocator estimated to generate "
                f"{estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS"
                f"* 2 ({self.MAXTICKS * 2:d}) "
            )

        interval = self._get_interval()
        freq = f"{interval}ms"
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmin.replace(tzinfo=None)
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)

        try:
            if len(all_dates) > 0:
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:  # pragma: no cover
            pass

        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self):
        return self._interval

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        # We need to cap at the endpoints of valid datetime
        dmin, dmax = self.datalim_to_dt()

        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)

        return self.nonsingular(vmin, vmax)


def _from_ordinal(x, tz: tzinfo | None = None) -> datetime:
    ix = int(x)
    dt = datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24 * remainder, 1)
    minute, remainder = divmod(60 * remainder, 1)
    second, remainder = divmod(60 * remainder, 1)
    microsecond = int(1_000_000 * remainder)
    if microsecond < 10:
        microsecond = 0  # compensate for rounding errors
    dt = datetime(
        dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond
    )
    if tz is not None:
        dt = dt.astimezone(tz)

    if microsecond > 999990:  # compensate for rounding errors
        dt += timedelta(microseconds=1_000_000 - microsecond)

    return dt


# Fixed frequency dynamic tick locators and formatters

# -------------------------------------------------------------------------
# --- Locators ---
# -------------------------------------------------------------------------


def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        (min_spacing, maj_spacing) = (1, 1)
    elif nyears < 20:
        (min_spacing, maj_spacing) = (1, 2)
    elif nyears < 50:
        (min_spacing, maj_spacing) = (1, 5)
    elif nyears < 100:
        (min_spacing, maj_spacing) = (5, 10)
    elif nyears < 200:
        (min_spacing, maj_spacing) = (5, 25)
    elif nyears < 600:
        (min_spacing, maj_spacing) = (10, 50)
    else:
        factor = nyears // 1000 + 1
        (min_spacing, maj_spacing) = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)


def _period_break(dates: PeriodIndex, period: str) -> npt.NDArray[np.intp]:
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
    mask = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]


def _period_break_mask(dates: PeriodIndex, period: str) -> npt.NDArray[np.bool_]:
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous


def has_level_label(label_flags: npt.NDArray[np.intp], vmin: float) -> bool:
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    if label_flags.size == 0 or (
        label_flags.size == 1 and label_flags[0] == 0 and vmin % 1 > 0.0
    ):
        return False
    else:
        return True


def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]:
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)

    ppd = -1  # placeholder for above-day freqs

    if dtype_code >= FreqGroup.FR_HR.value:
        # error: "BaseOffset" has no attribute "_creso"
        ppd = periods_per_day(freq._creso)  # type: ignore[attr-defined]
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1  # placerholder
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm = -1  # placeholder
        ppy = 1
    else:
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")

    return ppd, ppm, ppy


@functools.cache
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]

    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)

    # save this for later usage
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Period with BDay freq is deprecated", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
        )
        dates_ = period_range(
            start=Period(ordinal=vmin, freq=freq),
            end=Period(ordinal=vmax, freq=freq),
            freq=freq,
        )

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", np.int64), ("maj", bool), ("min", bool), ("fmt", "|S20")]
    )
    info["val"][:] = dates_.asi8
    info["fmt"][:] = ""
    info["maj"][[0, -1]] = True
    # .. and set some shortcuts
    info_maj = info["maj"]
    info_min = info["min"]
    info_fmt = info["fmt"]

    def first_label(label_flags):
        if (label_flags[0] == 0) and (label_flags.size > 1) and ((vmin_orig % 1) > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]

    # Case 1. Less than a month
    if span <= periodspermonth:
        day_start = _period_break(dates_, "day")
        month_start = _period_break(dates_, "month")
        year_start = _period_break(dates_, "year")

        def _hour_finder(label_interval: int, force_year_start: bool) -> None:
            target = dates_.hour
            mask = _period_break_mask(dates_, "hour")
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"
            if force_year_start and not has_level_label(year_start, vmin_orig):
                info_fmt[first_label(day_start)] = "%H:%M\n%d-%b\n%Y"

        def _minute_finder(label_interval: int) -> None:
            target = dates_.minute
            hour_start = _period_break(dates_, "hour")
            mask = _period_break_mask(dates_, "minute")
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"

        def _second_finder(label_interval: int) -> None:
            target = dates_.second
            minute_start = _period_break(dates_, "minute")
            mask = _period_break_mask(dates_, "second")
            info_maj[minute_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M:%S"
            info_fmt[day_start] = "%H:%M:%S\n%d-%b"
            info_fmt[year_start] = "%H:%M:%S\n%d-%b\n%Y"

        if span < periodsperday / 12000:
            _second_finder(1)
        elif span < periodsperday / 6000:
            _second_finder(2)
        elif span < periodsperday / 2400:
            _second_finder(5)
        elif span < periodsperday / 1200:
            _second_finder(10)
        elif span < periodsperday / 800:
            _second_finder(15)
        elif span < periodsperday / 400:
            _second_finder(30)
        elif span < periodsperday / 150:
            _minute_finder(1)
        elif span < periodsperday / 70:
            _minute_finder(2)
        elif span < periodsperday / 24:
            _minute_finder(5)
        elif span < periodsperday / 12:
            _minute_finder(15)
        elif span < periodsperday / 6:
            _minute_finder(30)
        elif span < periodsperday / 2.5:
            _hour_finder(1, False)
        elif span < periodsperday / 1.5:
            _hour_finder(2, False)
        elif span < periodsperday * 1.25:
            _hour_finder(3, False)
        elif span < periodsperday * 2.5:
            _hour_finder(6, True)
        elif span < periodsperday * 4:
            _hour_finder(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            info_fmt[day_start] = "%d"
            info_fmt[month_start] = "%d\n%b"
            info_fmt[year_start] = "%d\n%b\n%Y"
            if not has_level_label(year_start, vmin_orig):
                if not has_level_label(month_start, vmin_orig):
                    info_fmt[first_label(day_start)] = "%d\n%b\n%Y"
                else:
                    info_fmt[first_label(month_start)] = "%d\n%b\n%Y"

    # Case 2. Less than three months
    elif span <= periodsperyear // 4:
        month_start = _period_break(dates_, "month")
        info_maj[month_start] = True
        if dtype_code < FreqGroup.FR_HR.value:
            info["min"] = True
        else:
            day_start = _period_break(dates_, "day")
            info["min"][day_start] = True
        week_start = _period_break(dates_, "week")
        year_start = _period_break(dates_, "year")
        info_fmt[week_start] = "%d"
        info_fmt[month_start] = "\n\n%b"
        info_fmt[year_start] = "\n\n%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = "\n\n%b\n%Y"
            else:
                info_fmt[first_label(month_start)] = "\n\n%b\n%Y"
    # Case 3. Less than 14 months ...............
    elif span <= 1.15 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        week_start = _period_break(dates_, "week")
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = "%b\n%Y"
    # Case 4. Less than 2.5 years ...............
    elif span <= 2.5 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        month_start = _period_break(dates_, "month")
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 4. Less than 4 years .................
    elif span <= 4 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False

        month_break = dates_[month_start].month
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 5. Less than 11 years ................
    elif span <= 11 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = "%Y"
    # Case 6. More than 12 years ................
    else:
        year_start = _period_break(dates_, "year")
        year_break = dates_[year_start].year
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(year_break % maj_anndef == 0)]
        info_maj[major_idx] = True
        minor_idx = year_start[(year_break % min_anndef == 0)]
        info_min[minor_idx] = True
        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)

    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    dates_ = info["val"]
    info["fmt"] = ""
    year_start = (dates_ % 12 == 0).nonzero()[0]
    info_maj = info["maj"]
    info_fmt = info["fmt"]

    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "%b"
        info_fmt[year_start] = "%b\n%Y"

        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "%b\n%Y"

    elif span <= 2.5 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        # TODO: Check the following : is it really info['fmt'] ?
        #  2023-09-15 this is reached in test_finder_monthly
        info["fmt"][quarter_start] = True
        info["min"] = True

        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        jan_or_jul = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 11 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info["min"][quarter_start] = True

        info_fmt[year_start] = "%Y"

    else:
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        years = dates_[year_start] // 12 + 1
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True

        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]
    info_maj = info["maj"]
    info_fmt = info["fmt"]
    year_start = (dates_ % 4 == 0).nonzero()[0]

    if span <= 3.5 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "Q%q"
        info_fmt[year_start] = "Q%q\n%F"
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "Q%q\n%F"

    elif span <= 11 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[year_start] = "%F"

    else:
        # https://github.com/pandas-dev/pandas/pull/47602
        years = dates_[year_start] // 4 + 1970
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True
        info_fmt[major_idx] = "%F"

    return info


@functools.cache
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # Note: small difference here vs other finders in adding 1 to vmax
    (vmin, vmax) = (int(vmin), int(vmax + 1))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]

    (min_anndef, maj_anndef) = _get_default_annual_spacing(span)
    major_idx = dates_ % maj_anndef == 0
    minor_idx = dates_ % min_anndef == 0
    info["maj"][major_idx] = True
    info["min"][minor_idx] = True
    info["fmt"][major_idx] = "%Y"

    return info


def get_finder(freq: BaseOffset):
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    fgroup = FreqGroup.from_period_dtype_code(dtype_code)

    if fgroup == FreqGroup.FR_ANN:
        return _annual_finder
    elif fgroup == FreqGroup.FR_QTR:
        return _quarterly_finder
    elif fgroup == FreqGroup.FR_MTH:
        return _monthly_finder
    elif (dtype_code >= FreqGroup.FR_BUS.value) or fgroup == FreqGroup.FR_WK:
        return _daily_finder
    else:  # pragma: no cover
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")


class TimeSeries_DateLocator(Locator):
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        base: int = 1,
        quarter: int = 1,
        month: int = 1,
        day: int = 1,
        plot_obj=None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.freq = freq
        self.base = base
        (self.quarter, self.month, self.day) = (quarter, month, day)
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _get_default_locs(self, vmin, vmax):
        """Returns the default locations of ticks."""
        locator = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            return np.compress(locator["min"], locator["val"])
        return np.compress(locator["maj"], locator["val"])

    def __call__(self):
        """Return the locations of the ticks."""
        # axis calls Locator.set_axis inside set_m<xxxx>_formatter

        vi = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:  # pragma: no cover
            base = self.base
            (d, m) = divmod(vmin, base)
            vmin = (d + 1) * base
            # error: No overload variant of "range" matches argument types "float",
            # "float", "int"
            locs = list(range(vmin, vmax + 1, base))  # type: ignore[call-overload]
        return locs

    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        # requires matplotlib >= 0.98.0
        (vmin, vmax) = self.axis.get_data_interval()

        locs = self._get_default_locs(vmin, vmax)
        (vmin, vmax) = locs[[0, -1]]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return nonsingular(vmin, vmax)


# -------------------------------------------------------------------------
# --- Formatter ---
# -------------------------------------------------------------------------


class TimeSeries_DateFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        plot_obj=None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.format = None
        self.freq = freq
        self.locs: list[Any] = []  # unused, for matplotlib compat
        self.formatdict: dict[Any, Any] | None = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
        info = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            format = np.compress(info["min"] & np.logical_not(info["maj"]), info)
        else:
            format = np.compress(info["maj"], info)
        self.formatdict = {x: f for (x, _, _, f) in format}
        return self.formatdict

    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
        # don't actually use the locs. This is just needed to work with
        # matplotlib. Force to use vmin, vmax

        self.locs = locs

        (vmin, vmax) = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            (vmin, vmax) = (vmax, vmin)
        self._set_default_format(vmin, vmax)

    def __call__(self, x, pos: int | None = 0) -> str:
        if self.formatdict is None:
            return ""
        else:
            fmt = self.formatdict.pop(x, "")
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode("utf-8")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Period with BDay freq is deprecated",
                    category=FutureWarning,
                )
                period = Period(ordinal=int(x), freq=self.freq)
            assert isinstance(period, Period)
            return period.strftime(fmt)


class TimeSeries_TimedeltaFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """

    axis: Axis

    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        s, ns = divmod(x, 10**9)  # TODO(non-nano): this looks like it assumes ns
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        decimals = int(ns * 10 ** (n_decimals - 9))
        s = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        if n_decimals > 0:
            s += f".{decimals:0{n_decimals}d}"
        if d != 0:
            s = f"{int(d):d} days {s}"
        return s

    def __call__(self, x, pos: int | None = 0) -> str:
        (vmin, vmax) = tuple(self.axis.get_view_interval())
        n_decimals = min(int(np.ceil(np.log10(100 * 10**9 / abs(vmax - vmin)))), 9)
        return self.format_timedelta_ticks(x, pos, n_decimals)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.missing import remove_na_arraylike

from pandas import (
    MultiIndex,
    concat,
)

from pandas.plotting._matplotlib.misc import unpack_single_str_list

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import IndexLabel

    from pandas import (
        DataFrame,
        Series,
    )


def create_iter_data_given_by(
    data: DataFrame, kind: str = "hist"
) -> dict[Hashable, DataFrame | Series]:
    """
    Create data for iteration given `by` is assigned or not, and it is only
    used in both hist and boxplot.

    If `by` is assigned, return a dictionary of DataFrames in which the key of
    dictionary is the values in groups.
    If `by` is not assigned, return input as is, and this preserves current
    status of iter_data.

    Parameters
    ----------
    data : reformatted grouped data from `_compute_plot_data` method.
    kind : str, plot kind. This function is only used for `hist` and `box` plots.

    Returns
    -------
    iter_data : DataFrame or Dictionary of DataFrames

    Examples
    --------
    If `by` is assigned:

    >>> import numpy as np
    >>> tuples = [('h1', 'a'), ('h1', 'b'), ('h2', 'a'), ('h2', 'b')]
    >>> mi = pd.MultiIndex.from_tuples(tuples)
    >>> value = [[1, 3, np.nan, np.nan],
    ...          [3, 4, np.nan, np.nan], [np.nan, np.nan, 5, 6]]
    >>> data = pd.DataFrame(value, columns=mi)
    >>> create_iter_data_given_by(data)
    {'h1':     h1
         a    b
    0  1.0  3.0
    1  3.0  4.0
    2  NaN  NaN, 'h2':     h2
         a    b
    0  NaN  NaN
    1  NaN  NaN
    2  5.0  6.0}
    """

    # For `hist` plot, before transformation, the values in level 0 are values
    # in groups and subplot titles, and later used for column subselection and
    # iteration; For `box` plot, values in level 1 are column names to show,
    # and are used for iteration and as subplots titles.
    if kind == "hist":
        level = 0
    else:
        level = 1

    # Select sub-columns based on the value of level of MI, and if `by` is
    # assigned, data must be a MI DataFrame
    assert isinstance(data.columns, MultiIndex)
    return {
        col: data.loc[:, data.columns.get_level_values(level) == col]
        for col in data.columns.levels[level]
    }


def reconstruct_data_with_by(
    data: DataFrame, by: IndexLabel, cols: IndexLabel
) -> DataFrame:
    """
    Internal function to group data, and reassign multiindex column names onto the
    result in order to let grouped data be used in _compute_plot_data method.

    Parameters
    ----------
    data : Original DataFrame to plot
    by : grouped `by` parameter selected by users
    cols : columns of data set (excluding columns used in `by`)

    Returns
    -------
    Output is the reconstructed DataFrame with MultiIndex columns. The first level
    of MI is unique values of groups, and second level of MI is the columns
    selected by users.

    Examples
    --------
    >>> d = {'h': ['h1', 'h1', 'h2'], 'a': [1, 3, 5], 'b': [3, 4, 6]}
    >>> df = pd.DataFrame(d)
    >>> reconstruct_data_with_by(df, by='h', cols=['a', 'b'])
       h1      h2
       a     b     a     b
    0  1.0   3.0   NaN   NaN
    1  3.0   4.0   NaN   NaN
    2  NaN   NaN   5.0   6.0
    """
    by_modified = unpack_single_str_list(by)
    grouped = data.groupby(by_modified)

    data_list = []
    for key, group in grouped:
        # error: List item 1 has incompatible type "Union[Hashable,
        # Sequence[Hashable]]"; expected "Iterable[Hashable]"
        columns = MultiIndex.from_product([[key], cols])  # type: ignore[list-item]
        sub_group = group[cols]
        sub_group.columns = columns
        data_list.append(sub_group)

    data = concat(data_list, axis=1)
    return data


def reformat_hist_y_given_by(y: np.ndarray, by: IndexLabel | None) -> np.ndarray:
    """Internal function to reformat y given `by` is applied or not for hist plot.

    If by is None, input y is 1-d with NaN removed; and if by is not None, groupby
    will take place and input y is multi-dimensional array.
    """
    if by is not None and len(y.shape) > 1:
        return np.array([remove_na_arraylike(col) for col in y.T]).T
    return remove_na_arraylike(y)
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Collection,
    Iterator,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import matplotlib as mpl
import matplotlib.colors
import numpy as np

from pandas._typing import MatplotlibColor as Color
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = None,
    color_type: str = "default",
    color: dict[str, Color] | Color | Collection[Color] | None = None,
):
    """
    Get standard colors based on `colormap`, `color_type` or `color` inputs.

    Parameters
    ----------
    num_colors : int
        Minimum number of colors to be returned.
        Ignored if `color` is a dictionary.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap` are not None.
    color : dict or str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a dictionary, or a single color (single color string,
        or sequence of floats representing a single color),
        or a sequence of colors.

    Returns
    -------
    dict or list
        Standard colors. Can either be a mapping if `color` was a dictionary,
        or a list of colors with a length of `num_colors` or more.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if isinstance(color, dict):
        return color

    colors = _derive_colors(
        color=color,
        colormap=colormap,
        color_type=color_type,
        num_colors=num_colors,
    )

    return list(_cycle_colors(colors, num_colors=num_colors))


def _derive_colors(
    *,
    color: Color | Collection[Color] | None,
    colormap: str | Colormap | None,
    color_type: str,
    num_colors: int,
) -> list[Color]:
    """
    Derive colors from either `colormap`, `color_type` or `color` inputs.

    Get a list of colors either from `colormap`, or from `color`,
    or from `color_type` (if both `colormap` and `color` are None).

    Parameters
    ----------
    color : str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a single color (single color string, or sequence of floats
        representing a single color), or a sequence of colors.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap`` are not None.
    num_colors : int
        Number of colors to be extracted.

    Returns
    -------
    list
        List of colors extracted.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if color is None and colormap is not None:
        return _get_colors_from_colormap(colormap, num_colors=num_colors)
    elif color is not None:
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )
        return _get_colors_from_color(color)
    else:
        return _get_colors_from_color_type(color_type, num_colors=num_colors)


def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    """Cycle colors until achieving max of `num_colors` or length of `colors`.

    Extra colors will be ignored by matplotlib if there are more colors
    than needed and nothing needs to be done here.
    """
    max_colors = max(num_colors, len(colors))
    yield from itertools.islice(itertools.cycle(colors), max_colors)


def _get_colors_from_colormap(
    colormap: str | Colormap,
    num_colors: int,
) -> list[Color]:
    """Get colors from colormap."""
    cmap = _get_cmap_instance(colormap)
    return [cmap(num) for num in np.linspace(0, 1, num=num_colors)]


def _get_cmap_instance(colormap: str | Colormap) -> Colormap:
    """Get instance of matplotlib colormap."""
    if isinstance(colormap, str):
        cmap = colormap
        colormap = mpl.colormaps[colormap]
        if colormap is None:
            raise ValueError(f"Colormap {cmap} is not recognized")
    return colormap


def _get_colors_from_color(
    color: Color | Collection[Color],
) -> list[Color]:
    """Get colors from user input color."""
    if len(color) == 0:
        raise ValueError(f"Invalid color argument: {color}")

    if _is_single_color(color):
        color = cast(Color, color)
        return [color]

    color = cast(Collection[Color], color)
    return list(_gen_list_of_colors_from_iterable(color))


def _is_single_color(color: Color | Collection[Color]) -> bool:
    """Check if `color` is a single color, not a sequence of colors.

    Single color is of these kinds:
        - Named color "red", "C0", "firebrick"
        - Alias "g"
        - Sequence of floats, such as (0.1, 0.2, 0.3) or (0.1, 0.2, 0.3, 0.4).

    See Also
    --------
    _is_single_string_color
    """
    if isinstance(color, str) and _is_single_string_color(color):
        # GH #36972
        return True

    if _is_floats_color(color):
        return True

    return False


def _gen_list_of_colors_from_iterable(color: Collection[Color]) -> Iterator[Color]:
    """
    Yield colors from string of several letters or from collection of colors.
    """
    for x in color:
        if _is_single_color(x):
            yield x
        else:
            raise ValueError(f"Invalid color {x}")


def _is_floats_color(color: Color | Collection[Color]) -> bool:
    """Check if color comprises a sequence of floats representing color."""
    return bool(
        is_list_like(color)
        and (len(color) == 3 or len(color) == 4)
        and all(isinstance(x, (int, float)) for x in color)
    )


def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    """Get colors from user input color type."""
    if color_type == "default":
        return _get_default_colors(num_colors)
    elif color_type == "random":
        return _get_random_colors(num_colors)
    else:
        raise ValueError("color_type must be either 'default' or 'random'")


def _get_default_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of default colors from matplotlib rc params."""
    import matplotlib.pyplot as plt

    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    return colors[0:num_colors]


def _get_random_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of random colors."""
    return [_random_color(num) for num in range(num_colors)]


def _random_color(column: int) -> list[float]:
    """Get a random color represented as a list of length 3"""
    # GH17525 use common._random_state to avoid resetting the seed
    rs = com.random_state(column)
    return rs.rand(3).tolist()


def _is_single_string_color(color: Color) -> bool:
    """Check if `color` is a single string color.

    Examples of single string colors:
        - 'r'
        - 'g'
        - 'red'
        - 'green'
        - 'C3'
        - 'firebrick'

    Parameters
    ----------
    color : Color
        Color string or sequence of floats.

    Returns
    -------
    bool
        True if `color` looks like a valid color.
        False otherwise.
    """
    conv = matplotlib.colors.ColorConverter()
    try:
        # error: Argument 1 to "to_rgba" of "ColorConverter" has incompatible type
        # "str | Sequence[float]"; expected "tuple[float, float, float] | ..."
        conv.to_rgba(color)  # type: ignore[arg-type]
    except ValueError:
        return False
    else:
        return True
<|endoftext|>"""
Support pre-0.12 series pickle compatibility.
"""
from __future__ import annotations

import contextlib
import copy
import io
import pickle as pkl
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset

from pandas import Index
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.internals import BlockManager

if TYPE_CHECKING:
    from collections.abc import Generator


def load_reduce(self) -> None:
    stack = self.stack
    args = stack.pop()
    func = stack[-1]

    try:
        stack[-1] = func(*args)
        return
    except TypeError as err:
        # If we have a deprecated function,
        # try to replace and try again.

        msg = "_reconstruct: First argument must be a sub-type of ndarray"

        if msg in str(err):
            try:
                cls = args[0]
                stack[-1] = object.__new__(cls)
                return
            except TypeError:
                pass
        elif args and isinstance(args[0], type) and issubclass(args[0], BaseOffset):
            # TypeError: object.__new__(Day) is not safe, use Day.__new__()
            cls = args[0]
            stack[-1] = cls.__new__(*args)
            return
        elif args and issubclass(args[0], PeriodArray):
            cls = args[0]
            stack[-1] = NDArrayBacked.__new__(*args)
            return

        raise


# If classes are moved, provide compat here.
_class_locations_map = {
    ("pandas.core.sparse.array", "SparseArray"): ("pandas.core.arrays", "SparseArray"),
    # 15477
    ("pandas.core.base", "FrozenNDArray"): ("numpy", "ndarray"),
    # Re-routing unpickle block logic to go through _unpickle_block instead
    # for pandas <= 1.3.5
    ("pandas.core.internals.blocks", "new_block"): (
        "pandas._libs.internals",
        "_unpickle_block",
    ),
    ("pandas.core.indexes.frozen", "FrozenNDArray"): ("numpy", "ndarray"),
    ("pandas.core.base", "FrozenList"): ("pandas.core.indexes.frozen", "FrozenList"),
    # 10890
    ("pandas.core.series", "TimeSeries"): ("pandas.core.series", "Series"),
    ("pandas.sparse.series", "SparseTimeSeries"): (
        "pandas.core.sparse.series",
        "SparseSeries",
    ),
    # 12588, extensions moving
    ("pandas._sparse", "BlockIndex"): ("pandas._libs.sparse", "BlockIndex"),
    ("pandas.tslib", "Timestamp"): ("pandas._libs.tslib", "Timestamp"),
    # 18543 moving period
    ("pandas._period", "Period"): ("pandas._libs.tslibs.period", "Period"),
    ("pandas._libs.period", "Period"): ("pandas._libs.tslibs.period", "Period"),
    # 18014 moved __nat_unpickle from _libs.tslib-->_libs.tslibs.nattype
    ("pandas.tslib", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "__nat_unpickle",
    ),
    ("pandas._libs.tslib", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "__nat_unpickle",
    ),
    # 15998 top-level dirs moving
    ("pandas.sparse.array", "SparseArray"): (
        "pandas.core.arrays.sparse",
        "SparseArray",
    ),
    ("pandas.indexes.base", "_new_Index"): ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.indexes.base", "Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",  # updated in 50775
    ),
    ("pandas.indexes.range", "RangeIndex"): ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.indexes.multi", "MultiIndex"): ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.tseries.index", "_new_DatetimeIndex"): (
        "pandas.core.indexes.datetimes",
        "_new_DatetimeIndex",
    ),
    ("pandas.tseries.index", "DatetimeIndex"): (
        "pandas.core.indexes.datetimes",
        "DatetimeIndex",
    ),
    ("pandas.tseries.period", "PeriodIndex"): (
        "pandas.core.indexes.period",
        "PeriodIndex",
    ),
    # 19269, arrays moving
    ("pandas.core.categorical", "Categorical"): ("pandas.core.arrays", "Categorical"),
    # 19939, add timedeltaindex, float64index compat from 15998 move
    ("pandas.tseries.tdi", "TimedeltaIndex"): (
        "pandas.core.indexes.timedeltas",
        "TimedeltaIndex",
    ),
    ("pandas.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",  # updated in 50775
    ),
    # 50775, remove Int64Index, UInt64Index & Float64Index from codabase
    ("pandas.core.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "UInt64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"): (
        "pandas.core.dtypes.dtypes",
        "SparseDtype",
    ),
}


# our Unpickler sub-class to override methods and some dispatcher
# functions for compat and uses a non-public class of the pickle module.


class Unpickler(pkl._Unpickler):
    def find_class(self, module, name):
        # override superclass
        key = (module, name)
        module, name = _class_locations_map.get(key, key)
        return super().find_class(module, name)


Unpickler.dispatch = copy.copy(Unpickler.dispatch)
Unpickler.dispatch[pkl.REDUCE[0]] = load_reduce


def load_newobj(self) -> None:
    args = self.stack.pop()
    cls = self.stack[-1]

    # compat
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    elif issubclass(cls, DatetimeArray) and not args:
        arr = np.array([], dtype="M8[ns]")
        obj = cls.__new__(cls, arr, arr.dtype)
    elif issubclass(cls, TimedeltaArray) and not args:
        arr = np.array([], dtype="m8[ns]")
        obj = cls.__new__(cls, arr, arr.dtype)
    elif cls is BlockManager and not args:
        obj = cls.__new__(cls, (), [], False)
    else:
        obj = cls.__new__(cls, *args)

    self.stack[-1] = obj


Unpickler.dispatch[pkl.NEWOBJ[0]] = load_newobj


def load_newobj_ex(self) -> None:
    kwargs = self.stack.pop()
    args = self.stack.pop()
    cls = self.stack.pop()

    # compat
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    else:
        obj = cls.__new__(cls, *args, **kwargs)
    self.append(obj)


try:
    Unpickler.dispatch[pkl.NEWOBJ_EX[0]] = load_newobj_ex
except (AttributeError, KeyError):
    pass


def load(fh, encoding: str | None = None, is_verbose: bool = False):
    """
    Load a pickle, with a provided encoding,

    Parameters
    ----------
    fh : a filelike object
    encoding : an optional encoding
    is_verbose : show exception output
    """
    try:
        fh.seek(0)
        if encoding is not None:
            up = Unpickler(fh, encoding=encoding)
        else:
            up = Unpickler(fh)
        # "Unpickler" has no attribute "is_verbose"  [attr-defined]
        up.is_verbose = is_verbose  # type: ignore[attr-defined]

        return up.load()
    except (ValueError, TypeError):
        raise


def loads(
    bytes_object: bytes,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
):
    """
    Analogous to pickle._loads.
    """
    fd = io.BytesIO(bytes_object)
    return Unpickler(
