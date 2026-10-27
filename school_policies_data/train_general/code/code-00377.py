        box_aspect = fig_aspect * (position.height / position.width)
        data_ratio = box_aspect / aspect

        y_expander = data_ratio * xsize / ysize - 1
        # If y_expander > 0, the dy/dx viewLim ratio needs to increase
        if abs(y_expander) < 0.005:
            return

        dL = self.dataLim
        x0, x1 = x_trf.transform(dL.intervalx)
        y0, y1 = y_trf.transform(dL.intervaly)
        xr = 1.05 * (x1 - x0)
        yr = 1.05 * (y1 - y0)

        xmarg = xsize - xr
        ymarg = ysize - yr
        Ysize = data_ratio * xsize
        Xsize = ysize / data_ratio
        Xmarg = Xsize - xr
        Ymarg = Ysize - yr
        # Setting these targets to, e.g., 0.05*xr does not seem to help.
        xm = 0
        ym = 0

        shared_x = self in self._shared_axes["x"]
        shared_y = self in self._shared_axes["y"]

        if shared_x and shared_y:
            raise RuntimeError("set_aspect(..., adjustable='datalim') or "
                               "axis('equal') are not allowed when both axes "
                               "are shared.  Try set_aspect(..., "
                               "adjustable='box').")

        # If y is shared, then we are only allowed to change x, etc.
        if shared_y:
            adjust_y = False
        else:
            if xmarg > xm and ymarg > ym:
                adjy = ((Ymarg > 0 and y_expander < 0) or
                        (Xmarg < 0 and y_expander > 0))
            else:
                adjy = y_expander > 0
            adjust_y = shared_x or adjy  # (Ymarg > xmarg)

        if adjust_y:
            yc = 0.5 * (ymin + ymax)
            y0 = yc - Ysize / 2.0
            y1 = yc + Ysize / 2.0
            if not self.get_autoscaley_on():
                _log.warning("Ignoring fixed y limits to fulfill fixed data aspect "
                             "with adjustable data limits.")
            self.set_ybound(y_trf.inverted().transform([y0, y1]))
        else:
            xc = 0.5 * (xmin + xmax)
            x0 = xc - Xsize / 2.0
            x1 = xc + Xsize / 2.0
            if not self.get_autoscalex_on():
                _log.warning("Ignoring fixed x limits to fulfill fixed data aspect "
                             "with adjustable data limits.")
            self.set_xbound(x_trf.inverted().transform([x0, x1]))

    def axis(self, arg=None, /, *, emit=True, **kwargs):
        """
        Convenience method to get or set some axis properties.

        Call signatures::

          xmin, xmax, ymin, ymax = axis()
          xmin, xmax, ymin, ymax = axis([xmin, xmax, ymin, ymax])
          xmin, xmax, ymin, ymax = axis(option)
          xmin, xmax, ymin, ymax = axis(**kwargs)

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float, optional
            The axis limits to be set.  This can also be achieved using ::

                ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        option : bool or str
            If a bool, turns axis lines and labels on or off. If a string,
            possible values are:

            ================ ===========================================================
            Value            Description
            ================ ===========================================================
            'off' or `False` Hide all axis decorations, i.e. axis labels, spines,
                             tick marks, tick labels, and grid lines.
                             This is the same as `~.Axes.set_axis_off()`.
            'on' or `True`   Do not hide all axis decorations, i.e. axis labels, spines,
                             tick marks, tick labels, and grid lines.
                             This is the same as `~.Axes.set_axis_on()`.
            'equal'          Set equal scaling (i.e., make circles circular) by
                             changing the axis limits. This is the same as
                             ``ax.set_aspect('equal', adjustable='datalim')``.
                             Explicit data limits may not be respected in this case.
            'scaled'         Set equal scaling (i.e., make circles circular) by
                             changing dimensions of the plot box. This is the same as
                             ``ax.set_aspect('equal', adjustable='box', anchor='C')``.
                             Additionally, further autoscaling will be disabled.
            'tight'          Set limits just large enough to show all data, then
                             disable further autoscaling.
            'auto'           Automatic scaling (fill plot box with data).
            'image'          'scaled' with axis limits equal to data limits.
            'square'         Square plot; similar to 'scaled', but initially forcing
                             ``xmax-xmin == ymax-ymin``.
            ================ ===========================================================

        emit : bool, default: True
            Whether observers are notified of the axis limit change.
            This option is passed on to `~.Axes.set_xlim` and
            `~.Axes.set_ylim`.

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            The axis limits.

        See Also
        --------
        matplotlib.axes.Axes.set_xlim
        matplotlib.axes.Axes.set_ylim

        Notes
        -----
        For 3D Axes, this method additionally takes *zmin*, *zmax* as
        parameters and likewise returns them.
        """
        if isinstance(arg, (str, bool)):
            if arg is True:
                arg = 'on'
            if arg is False:
                arg = 'off'
            arg = arg.lower()
            if arg == 'on':
                self.set_axis_on()
            elif arg == 'off':
                self.set_axis_off()
            elif arg in [
                    'equal', 'tight', 'scaled', 'auto', 'image', 'square']:
                self.set_autoscale_on(True)
                self.set_aspect('auto')
                self.autoscale_view(tight=False)
                if arg == 'equal':
                    self.set_aspect('equal', adjustable='datalim')
                elif arg == 'scaled':
                    self.set_aspect('equal', adjustable='box', anchor='C')
                    self.set_autoscale_on(False)  # Req. by Mark Bakker
                elif arg == 'tight':
                    self.autoscale_view(tight=True)
                    self.set_autoscale_on(False)
                elif arg == 'image':
                    self.autoscale_view(tight=True)
                    self.set_autoscale_on(False)
                    self.set_aspect('equal', adjustable='box', anchor='C')
                elif arg == 'square':
                    self.set_aspect('equal', adjustable='box', anchor='C')
                    self.set_autoscale_on(False)
                    xlim = self.get_xlim()
                    ylim = self.get_ylim()
                    edge_size = max(np.diff(xlim), np.diff(ylim))[0]
                    self.set_xlim([xlim[0], xlim[0] + edge_size],
                                  emit=emit, auto=False)
                    self.set_ylim([ylim[0], ylim[0] + edge_size],
                                  emit=emit, auto=False)
            else:
                raise ValueError(f"Unrecognized string {arg!r} to axis; "
                                 "try 'on' or 'off'")
        else:
            if arg is not None:
                if len(arg) != 2*len(self._axis_names):
                    raise TypeError(
                        "The first argument to axis() must be an iterable of the form "
                        "[{}]".format(", ".join(
                            f"{name}min, {name}max" for name in self._axis_names)))
                limits = {
                    name: arg[2*i:2*(i+1)]
                    for i, name in enumerate(self._axis_names)
                }
            else:
                limits = {}
                for name in self._axis_names:
                    ax_min = kwargs.pop(f'{name}min', None)
                    ax_max = kwargs.pop(f'{name}max', None)
                    limits[name] = (ax_min, ax_max)
            for name, (ax_min, ax_max) in limits.items():
                ax_auto = (None  # Keep autoscale state as is.
                           if ax_min is None and ax_max is None
                           else False)  # Turn off autoscale.
                set_ax_lim = getattr(self, f'set_{name}lim')
                set_ax_lim(ax_min, ax_max, emit=emit, auto=ax_auto)
        if kwargs:
            raise _api.kwarg_error("axis", kwargs)
        lims = ()
        for name in self._axis_names:
            get_ax_lim = getattr(self, f'get_{name}lim')
            lims += get_ax_lim()
        return lims

    def get_legend(self):
        """Return the `.Legend` instance, or None if no legend is defined."""
        return self.legend_

    def get_images(self):
        r"""Return a list of `.AxesImage`\s contained by the Axes."""
        return cbook.silent_list('AxesImage', self.images)

    def get_lines(self):
        """Return a list of lines contained by the Axes."""
        return cbook.silent_list('Line2D', self.lines)

    def get_xaxis(self):
        """
        [*Discouraged*] Return the XAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute `~.Axes.xaxis`.
        """
        return self.xaxis

    def get_yaxis(self):
        """
        [*Discouraged*] Return the YAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute `~.Axes.yaxis`.
        """
        return self.yaxis

    get_xgridlines = _axis_method_wrapper("xaxis", "get_gridlines")
    get_xticklines = _axis_method_wrapper("xaxis", "get_ticklines")
    get_ygridlines = _axis_method_wrapper("yaxis", "get_gridlines")
    get_yticklines = _axis_method_wrapper("yaxis", "get_ticklines")

    # Adding and tracking artists

    def _sci(self, im):
        """
        Set the current image.

        This image will be the target of colormap functions like
        ``pyplot.viridis``, and other functions such as `~.pyplot.clim`.  The
        current image is an attribute of the current Axes.
        """
        _api.check_isinstance((mcoll.Collection, mimage.AxesImage), im=im)
        if im not in self._children:
            raise ValueError("Argument must be an image or collection in this Axes")
        self._current_image = im

    def _gci(self):
        """Helper for `~matplotlib.pyplot.gci`; do not use elsewhere."""
        return self._current_image

    def has_data(self):
        """
        Return whether any artists have been added to the Axes.

        This should not be used to determine whether the *dataLim*
        need to be updated, and may not actually be useful for
        anything.
        """
        return any(isinstance(a, (mcoll.Collection, mimage.AxesImage,
                                  mlines.Line2D, mpatches.Patch))
                   for a in self._children)

    def add_artist(self, a):
        """
        Add an `.Artist` to the Axes; return the artist.

        Use `add_artist` only for artists for which there is no dedicated
        "add" method; and if necessary, use a method such as `update_datalim`
        to manually update the `~.Axes.dataLim` if the artist is to be included
        in autoscaling.

        If no ``transform`` has been specified when creating the artist (e.g.
        ``artist.get_transform() == None``) then the transform is set to
        ``ax.transData``.
        """
        a.axes = self
        self._children.append(a)
        a._remove_method = self._children.remove
        self._set_artist_props(a)
        if a.get_clip_path() is None:
            a.set_clip_path(self.patch)
        self.stale = True
        return a

    def add_child_axes(self, ax):
        """
        Add an `.Axes` to the Axes' children; return the child Axes.

        This is the lowlevel version.  See `.axes.Axes.inset_axes`.
        """

        # normally Axes have themselves as the Axes, but these need to have
        # their parent...
        # Need to bypass the getter...
        ax._axes = self
        ax.stale_callback = martist._stale_axes_callback

        self.child_axes.append(ax)
        ax._remove_method = functools.partial(
            self.get_figure(root=False)._remove_axes, owners=[self.child_axes])
        self.stale = True
        return ax

    def add_collection(self, collection, autolim=True):
        """
        Add a `.Collection` to the Axes; return the collection.
        """
        _api.check_isinstance(mcoll.Collection, collection=collection)
        if not collection.get_label():
            collection.set_label(f'_child{len(self._children)}')
        self._children.append(collection)
        collection._remove_method = self._children.remove
        self._set_artist_props(collection)

        if collection.get_clip_path() is None:
            collection.set_clip_path(self.patch)

        if autolim:
            # Make sure viewLim is not stale (mostly to match
            # pre-lazy-autoscale behavior, which is not really better).
            self._unstale_viewLim()
            datalim = collection.get_datalim(self.transData)
            points = datalim.get_points()
            if not np.isinf(datalim.minpos).all():
                # By definition, if minpos (minimum positive value) is set
                # (i.e., non-inf), then min(points) <= minpos <= max(points),
                # and minpos would be superfluous. However, we add minpos to
                # the call so that self.dataLim will update its own minpos.
                # This ensures that log scales see the correct minimum.
                points = np.concatenate([points, [datalim.minpos]])
            # only update the dataLim for x/y if the collection uses transData
            # in this direction.
            x_is_data, y_is_data = (collection.get_transform()
                                    .contains_branch_seperately(self.transData))
            ox_is_data, oy_is_data = (collection.get_offset_transform()
                                      .contains_branch_seperately(self.transData))
            self.update_datalim(
                points,
                updatex=x_is_data or ox_is_data,
                updatey=y_is_data or oy_is_data,
            )

        self.stale = True
        return collection

    def add_image(self, image):
        """
        Add an `.AxesImage` to the Axes; return the image.
        """
        _api.check_isinstance(mimage.AxesImage, image=image)
        self._set_artist_props(image)
        if not image.get_label():
            image.set_label(f'_child{len(self._children)}')
        self._children.append(image)
        image._remove_method = self._children.remove
        self.stale = True
        return image

    def _update_image_limits(self, image):
        xmin, xmax, ymin, ymax = image.get_extent()
        self.axes.update_datalim(((xmin, ymin), (xmax, ymax)))

    def add_line(self, line):
        """
        Add a `.Line2D` to the Axes; return the line.
        """
        _api.check_isinstance(mlines.Line2D, line=line)
        self._set_artist_props(line)
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)

        self._update_line_limits(line)
        if not line.get_label():
            line.set_label(f'_child{len(self._children)}')
        self._children.append(line)
        line._remove_method = self._children.remove
        self.stale = True
        return line

    def _add_text(self, txt):
        """
        Add a `.Text` to the Axes; return the text.
        """
        _api.check_isinstance(mtext.Text, txt=txt)
        self._set_artist_props(txt)
        self._children.append(txt)
        txt._remove_method = self._children.remove
        self.stale = True
        return txt

    def _update_line_limits(self, line):
        """
        Figures out the data limit of the given line, updating `.Axes.dataLim`.
        """
        path = line.get_path()
        if path.vertices.size == 0:
            return

        line_trf = line.get_transform()

        if line_trf == self.transData:
            data_path = path
        elif any(line_trf.contains_branch_seperately(self.transData)):
            # Compute the transform from line coordinates to data coordinates.
            trf_to_data = line_trf - self.transData
            # If transData is affine we can use the cached non-affine component
            # of line's path (since the non-affine part of line_trf is
            # entirely encapsulated in trf_to_data).
            if self.transData.is_affine:
                line_trans_path = line._get_transformed_path()
                na_path, _ = line_trans_path.get_transformed_path_and_affine()
                data_path = trf_to_data.transform_path_affine(na_path)
            else:
                data_path = trf_to_data.transform_path(path)
        else:
            # For backwards compatibility we update the dataLim with the
            # coordinate range of the given path, even though the coordinate
            # systems are completely different. This may occur in situations
            # such as when ax.transAxes is passed through for absolute
            # positioning.
            data_path = path

        if not data_path.vertices.size:
            return

        updatex, updatey = line_trf.contains_branch_seperately(self.transData)
        if self.name != "rectilinear":
            # This block is mostly intended to handle axvline in polar plots,
            # for which updatey would otherwise be True.
            if updatex and line_trf == self.get_yaxis_transform():
                updatex = False
            if updatey and line_trf == self.get_xaxis_transform():
                updatey = False
        self.dataLim.update_from_path(data_path,
                                      self.ignore_existing_data_limits,
                                      updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

    def add_patch(self, p):
        """
        Add a `.Patch` to the Axes; return the patch.
        """
        _api.check_isinstance(mpatches.Patch, p=p)
        self._set_artist_props(p)
        if p.get_clip_path() is None:
            p.set_clip_path(self.patch)
        self._update_patch_limits(p)
        self._children.append(p)
        p._remove_method = self._children.remove
        return p

    def _update_patch_limits(self, patch):
        """Update the data limits for the given patch."""
        # hist can add zero height Rectangles, which is useful to keep
        # the bins, counts and patches lined up, but it throws off log
        # scaling.  We'll ignore rects with zero height or width in
        # the auto-scaling

        # cannot check for '==0' since unitized data may not compare to zero
        # issue #2150 - we update the limits if patch has non zero width
        # or height.
        if (isinstance(patch, mpatches.Rectangle) and
                ((not patch.get_width()) and (not patch.get_height()))):
            return
        p = patch.get_path()
        # Get all vertices on the path
        # Loop through each segment to get extrema for Bezier curve sections
        vertices = []
        for curve, code in p.iter_bezier(simplify=False):
            # Get distance along the curve of any extrema
            _, dzeros = curve.axis_aligned_extrema()
            # Calculate vertices of start, end and any extrema in between
            vertices.append(curve([0, *dzeros, 1]))

        if len(vertices):
            vertices = np.vstack(vertices)

        patch_trf = patch.get_transform()
        updatex, updatey = patch_trf.contains_branch_seperately(self.transData)
        if not (updatex or updatey):
            return
        if self.name != "rectilinear":
            # As in _update_line_limits, but for axvspan.
            if updatex and patch_trf == self.get_yaxis_transform():
                updatex = False
            if updatey and patch_trf == self.get_xaxis_transform():
                updatey = False
        trf_to_data = patch_trf - self.transData
        xys = trf_to_data.transform(vertices)
        self.update_datalim(xys, updatex=updatex, updatey=updatey)

    def add_table(self, tab):
        """
        Add a `.Table` to the Axes; return the table.
        """
        _api.check_isinstance(mtable.Table, tab=tab)
        self._set_artist_props(tab)
        self._children.append(tab)
        if tab.get_clip_path() is None:
            tab.set_clip_path(self.patch)
        tab._remove_method = self._children.remove
        return tab

    def add_container(self, container):
        """
        Add a `.Container` to the Axes' containers; return the container.
        """
        label = container.get_label()
        if not label:
            container.set_label('_container%d' % len(self.containers))
        self.containers.append(container)
        container._remove_method = self.containers.remove
        return container

    def _unit_change_handler(self, axis_name, event=None):
        """
        Process axis units changes: requests updates to data and view limits.
        """
        if event is None:  # Allow connecting `self._unit_change_handler(name)`
            return functools.partial(
                self._unit_change_handler, axis_name, event=object())
        _api.check_in_list(self._axis_map, axis_name=axis_name)
        for line in self.lines:
            line.recache_always()
        self.relim()
        self._request_autoscale_view(axis_name)

    def relim(self, visible_only=False):
        """
        Recompute the data limits based on current artists.

        At present, `.Collection` instances are not supported.

        Parameters
        ----------
        visible_only : bool, default: False
            Whether to exclude invisible artists.
        """
        # Collections are deliberately not supported (yet); see
        # the TODO note in artists.py.
        self.dataLim.ignore(True)
        self.dataLim.set_points(mtransforms.Bbox.null().get_points())
        self.ignore_existing_data_limits = True

        for artist in self._children:
            if not visible_only or artist.get_visible():
                if isinstance(artist, mlines.Line2D):
                    self._update_line_limits(artist)
                elif isinstance(artist, mpatches.Patch):
                    self._update_patch_limits(artist)
                elif isinstance(artist, mimage.AxesImage):
                    self._update_image_limits(artist)

    def update_datalim(self, xys, updatex=True, updatey=True):
        """
        Extend the `~.Axes.dataLim` Bbox to include the given points.

        If no data is set currently, the Bbox will ignore its limits and set
        the bound to be the bounds of the xydata (*xys*). Otherwise, it will
        compute the bounds of the union of its current data and the data in
        *xys*.

        Parameters
        ----------
        xys : 2D array-like
            The points to include in the data limits Bbox. This can be either
            a list of (x, y) tuples or a (N, 2) array.

        updatex, updatey : bool, default: True
            Whether to update the x/y limits.
        """
        xys = np.asarray(xys)
        if not np.any(np.isfinite(xys)):
            return
        self.dataLim.update_from_data_xy(xys, self.ignore_existing_data_limits,
                                         updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

    def _process_unit_info(self, datasets=None, kwargs=None, *, convert=True):
        """
        Set axis units based on *datasets* and *kwargs*, and optionally apply
        unit conversions to *datasets*.

        Parameters
        ----------
        datasets : list
            List of (axis_name, dataset) pairs (where the axis name is defined
            as in `._axis_map`).  Individual datasets can also be None
            (which gets passed through).
        kwargs : dict
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar) entries) is popped, if present.  Note that this dict is
            mutated in-place!
        convert : bool, default: True
            Whether to return the original datasets or the converted ones.

        Returns
        -------
        list
            Either the original datasets if *convert* is False, or the
            converted ones if *convert* is True (the default).
        """
        # The API makes datasets a list of pairs rather than an axis_name to
        # dataset mapping because it is sometimes necessary to process multiple
        # datasets for a single axis, and concatenating them may be tricky
        # (e.g. if some are scalars, etc.).
        datasets = datasets or []
        kwargs = kwargs or {}
        axis_map = self._axis_map
        for axis_name, data in datasets:
            try:
                axis = axis_map[axis_name]
            except KeyError:
                raise ValueError(f"Invalid axis name: {axis_name!r}") from None
            # Update from data if axis is already set but no unit is set yet.
            if axis is not None and data is not None and not axis.have_units():
                axis.update_units(data)
        for axis_name, axis in axis_map.items():
            # Return if no axis is set.
            if axis is None:
                continue
            # Check for units in the kwargs, and if present update axis.
            units = kwargs.pop(f"{axis_name}units", axis.units)
            if self.name == "polar":
                # Special case: polar supports "thetaunits"/"runits".
                polar_units = {"x": "thetaunits", "y": "runits"}
                units = kwargs.pop(polar_units[axis_name], units)
            if units != axis.units and units is not None:
                axis.set_units(units)
                # If the units being set imply a different converter,
                # we need to update again.
                for dataset_axis_name, data in datasets:
                    if dataset_axis_name == axis_name and data is not None:
                        axis.update_units(data)
        return [axis_map[axis_name].convert_units(data)
                if convert and data is not None else data
                for axis_name, data in datasets]

    def in_axes(self, mouseevent):
        """
        Return whether the given event (in display coords) is in the Axes.
        """
        return self.patch.contains(mouseevent)[0]

    get_autoscalex_on = _axis_method_wrapper("xaxis", "_get_autoscale_on")
    get_autoscaley_on = _axis_method_wrapper("yaxis", "_get_autoscale_on")
    set_autoscalex_on = _axis_method_wrapper("xaxis", "_set_autoscale_on")
    set_autoscaley_on = _axis_method_wrapper("yaxis", "_set_autoscale_on")

    def get_autoscale_on(self):
        """Return True if each axis is autoscaled, False otherwise."""
        return all(axis._get_autoscale_on()
                   for axis in self._axis_map.values())

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied to each axis on the next draw or
        call to `.Axes.autoscale_view`.

        Parameters
        ----------
        b : bool
        """
        for axis in self._axis_map.values():
            axis._set_autoscale_on(b)

    @property
    def use_sticky_edges(self):
        """
        When autoscaling, whether to obey all `.Artist.sticky_edges`.

        Default is ``True``.

        Setting this to ``False`` ensures that the specified margins
        will be applied, even if the plot includes an image, for
        example, which would otherwise force a view limit to coincide
        with its data limit.

        The changing this property does not change the plot until
        `autoscale` or `autoscale_view` is called.
        """
        return self._use_sticky_edges

    @use_sticky_edges.setter
    def use_sticky_edges(self, b):
        self._use_sticky_edges = bool(b)
        # No effect until next autoscaling, which will mark the Axes as stale.

    def get_xmargin(self):
        """
        Retrieve autoscaling margin of the x-axis.

        .. versionadded:: 3.9

        Returns
        -------
        xmargin : float

        See Also
        --------
        matplotlib.axes.Axes.set_xmargin
        """
        return self._xmargin

    def get_ymargin(self):
        """
        Retrieve autoscaling margin of the y-axis.

        .. versionadded:: 3.9

        Returns
        -------
        ymargin : float

        See Also
        --------
        matplotlib.axes.Axes.set_ymargin
        """
        return self._ymargin

    def set_xmargin(self, m):
        """
        Set padding of X data limits prior to autoscaling.

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
        self._xmargin = m
        self._request_autoscale_view("x")
        self.stale = True

    def set_ymargin(self, m):
        """
        Set padding of Y data limits prior to autoscaling.

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
        self._ymargin = m
        self._request_autoscale_view("y")
        self.stale = True

    def margins(self, *margins, x=None, y=None, tight=True):
        """
        Set or retrieve margins around the data for autoscaling axis limits.

        This allows to configure the padding around the data without having to
        set explicit limits using `~.Axes.set_xlim` / `~.Axes.set_ylim`.

        Autoscaling determines the axis limits by adding *margin* times the
        data interval as padding around the data. See the following illustration:

        .. plot:: _embedded_plots/axes_margins.py

        All input parameters must be floats greater than -0.5. Passing both
        positional and keyword arguments is invalid and will raise a TypeError.
        If no arguments (positional or otherwise) are provided, the current
        margins will remain unchanged and simply be returned.

        The default margins are :rc:`axes.xmargin` and :rc:`axes.ymargin`.

        Parameters
        ----------
        *margins : float, optional
            If a single positional argument is provided, it specifies
            both margins of the x-axis and y-axis limits. If two
            positional arguments are provided, they will be interpreted
            as *xmargin*, *ymargin*. If setting the margin on a single
            axis is desired, use the keyword arguments described below.

        x, y : float, optional
            Specific margin values for the x-axis and y-axis,
            respectively. These cannot be used with positional
            arguments, but can be used individually to alter on e.g.,
            only the y-axis.

        tight : bool or None, default: True
            The *tight* parameter is passed to `~.axes.Axes.autoscale_view`,
            which is executed after a margin is changed; the default
            here is *True*, on the assumption that when margins are
            specified, no additional padding to match tick marks is
            usually desired.  Setting *tight* to *None* preserves
            the previous setting.

        Returns
        -------
        xmargin, ymargin : float

        Notes
        -----
        If a previously used Axes method such as :meth:`pcolor` has set
        `~.Axes.use_sticky_edges` to `True`, only the limits not set by
        the "sticky artists" will be modified. To force all
        margins to be set, set `~.Axes.use_sticky_edges` to `False`
        before calling :meth:`margins`.

        See Also
        --------
        .Axes.set_xmargin, .Axes.set_ymargin
        """

        if margins and (x is not None or y is not None):
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments for x and/or y.')
        elif len(margins) == 1:
            x = y = margins[0]
        elif len(margins) == 2:
            x, y = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all '
                            'margins, or one for each margin (x, y).')

        if x is None and y is None:
            if tight is not True:
                _api.warn_external(f'ignoring tight={tight!r} in get mode')
            return self._xmargin, self._ymargin

        if tight is not None:
            self._tight = tight
        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)

    def set_rasterization_zorder(self, z):
        """
        Set the zorder threshold for rasterization for vector graphics output.

        All artists with a zorder below the given value will be rasterized if
        they support rasterization.

        This setting is ignored for pixel-based output.

        See also :doc:`/gallery/misc/rasterization_demo`.

        Parameters
        ----------
        z : float or None
            The zorder below which artists are rasterized.
            If ``None`` rasterization based on zorder is deactivated.
        """
        self._rasterization_zorder = z
        self.stale = True

    def get_rasterization_zorder(self):
        """Return the zorder value below which artists will be rasterized."""
        return self._rasterization_zorder

    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Autoscale the axis view to the data (toggle).

        Convenience method for simple axis view autoscaling.
        It turns autoscaling on or off, and then,
        if autoscaling for either axis is on, it performs
        the autoscaling on the specified axis or Axes.

        Parameters
        ----------
        enable : bool or None, default: True
            True turns autoscaling on, False turns it off.
            None leaves the autoscaling state unchanged.
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be set
            to 'z', and 'both' refers to all three Axes.)
        tight : bool or None, default: None
            If True, first set the margins to zero.  Then, this argument is
            forwarded to `~.axes.Axes.autoscale_view` (regardless of
            its value); see the description of its behavior there.
        """
        if enable is None:
            scalex = True
            scaley = True
        else:
            if axis in ['x', 'both']:
                self.set_autoscalex_on(bool(enable))
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(bool(enable))
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
        if tight and scalex:
            self._xmargin = 0
        if tight and scaley:
            self._ymargin = 0
        if scalex:
            self._request_autoscale_view("x", tight=tight)
        if scaley:
            self._request_autoscale_view("y", tight=tight)

    def autoscale_view(self, tight=None, scalex=True, scaley=True):
        """
        Autoscale the view limits using the data limits.

        Parameters
        ----------
        tight : bool or None
            If *True*, only expand the axis limits using the margins.  Note
            that unlike for `autoscale`, ``tight=True`` does *not* set the
            margins to zero.

            If *False* and :rc:`axes.autolimit_mode` is 'round_numbers', then
            after expansion by the margins, further expand the axis limits
            using the axis major locator.

            If None (the default), reuse the value set in the previous call to
            `autoscale_view` (the initial value is False, but the default style
            sets :rc:`axes.autolimit_mode` to 'data', in which case this
            behaves like True).

        scalex : bool, default: True
            Whether to autoscale the x-axis.

        scaley : bool, default: True
            Whether to autoscale the y-axis.

        Notes
        -----
        The autoscaling preserves any preexisting axis direction reversal.

        The data limits are not updated automatically when artist data are
        changed after the artist has been added to an Axes instance.  In that
        case, use :meth:`matplotlib.axes.Axes.relim` prior to calling
        autoscale_view.

        If the views of the Axes are fixed, e.g. via `set_xlim`, they will
        not be changed by autoscale_view().
        See :meth:`matplotlib.axes.Axes.autoscale` for an alternative.
        """
        if tight is not None:
            self._tight = bool(tight)

        x_stickies = y_stickies = np.array([])
        if self.use_sticky_edges:
            if self._xmargin and scalex and self.get_autoscalex_on():
                x_stickies = np.sort(np.concatenate([
                    artist.sticky_edges.x
                    for ax in self._shared_axes["x"].get_siblings(self)
                    for artist in ax.get_children()]))
            if self._ymargin and scaley and self.get_autoscaley_on():
                y_stickies = np.sort(np.concatenate([
                    artist.sticky_edges.y
                    for ax in self._shared_axes["y"].get_siblings(self)
                    for artist in ax.get_children()]))
        if self.get_xscale() == 'log':
            x_stickies = x_stickies[x_stickies > 0]
        if self.get_yscale() == 'log':
            y_stickies = y_stickies[y_stickies > 0]

        def handle_single_axis(
                scale, shared_axes, name, axis, margin, stickies, set_bound):

            if not (scale and axis._get_autoscale_on()):
                return  # nothing to do...

            shared = shared_axes.get_siblings(self)
            # Base autoscaling on finite data limits when there is at least one
            # finite data limit among all the shared_axes and intervals.
            values = [val for ax in shared
                      for val in getattr(ax.dataLim, f"interval{name}")
                      if np.isfinite(val)]
            if values:
                x0, x1 = (min(values), max(values))
            elif getattr(self._viewLim, f"mutated{name}")():
                # No data, but explicit viewLims already set:
                # in mutatedx or mutatedy.
                return
            else:
                x0, x1 = (-np.inf, np.inf)
            # If x0 and x1 are nonfinite, get default limits from the locator.
            locator = axis.get_major_locator()
            x0, x1 = locator.nonsingular(x0, x1)
            # Find the minimum minpos for use in the margin calculation.
            minimum_minpos = min(
                getattr(ax.dataLim, f"minpos{name}") for ax in shared)

            # Prevent margin addition from crossing a sticky value.  A small
            # tolerance must be added due to floating point issues with
            # streamplot; it is defined relative to x1-x0 but has
            # no absolute term (e.g. "+1e-8") to avoid issues when working with
            # datasets where all values are tiny (less than 1e-8).
            tol = 1e-5 * abs(x1 - x0)
            # Index of largest element < x0 + tol, if any.
            i0 = stickies.searchsorted(x0 + tol) - 1
            x0bound = stickies[i0] if i0 != -1 else None
            # Index of smallest element > x1 - tol, if any.
            i1 = stickies.searchsorted(x1 - tol)
            x1bound = stickies[i1] if i1 != len(stickies) else None

            # Add the margin in figure space and then transform back, to handle
            # non-linear scales.
            transform = axis.get_transform()
            inverse_trans = transform.inverted()
            x0, x1 = axis._scale.limit_range_for_scale(x0, x1, minimum_minpos)
            x0t, x1t = transform.transform([x0, x1])
            delta = (x1t - x0t) * margin
            if not np.isfinite(delta):
                delta = 0  # If a bound isn't finite, set margin to zero.
            x0, x1 = inverse_trans.transform([x0t - delta, x1t + delta])

            # Apply sticky bounds.
            if x0bound is not None:
                x0 = max(x0, x0bound)
            if x1bound is not None:
                x1 = min(x1, x1bound)

            if not self._tight:
                x0, x1 = locator.view_limits(x0, x1)
            set_bound(x0, x1)
            # End of definition of internal function 'handle_single_axis'.

        handle_single_axis(
            scalex, self._shared_axes["x"], 'x', self.xaxis, self._xmargin,
            x_stickies, self.set_xbound)
        handle_single_axis(
            scaley, self._shared_axes["y"], 'y', self.yaxis, self._ymargin,
            y_stickies, self.set_ybound)

    def _update_title_position(self, renderer):
        """
        Update the title position based on the bounding box enclosing
        all the ticklabels and x-axis spine and xlabel...
        """
        if self._autotitlepos is not None and not self._autotitlepos:
            _log.debug('title position was updated manually, not adjusting')
            return

        titles = (self.title, self._left_title, self._right_title)

        if not any(title.get_text() for title in titles):
            # If the titles are all empty, there is no need to update their positions.
            return

        # Need to check all our twins too, aligned axes, and all the children
        # as well.
        axs = set()
        axs.update(self.child_axes)
        axs.update(self._twinned_axes.get_siblings(self))
        axs.update(
            self.get_figure(root=False)._align_label_groups['title'].get_siblings(self))

        for ax in self.child_axes:  # Child positions must be updated first.
            locator = ax.get_axes_locator()
            ax.apply_aspect(locator(self, renderer) if locator else None)

        top = -np.inf
        for ax in axs:
            bb = None
            if (ax.xaxis.get_ticks_position() in ['top', 'unknown'] or
                    ax.xaxis.get_label_position() == 'top'):
                bb = ax.xaxis.get_tightbbox(renderer)
            if bb is None:
                # Extent of the outline for colorbars, of the axes otherwise.
                bb = ax.spines.get("outline", ax).get_window_extent()
            top = max(top, bb.ymax)

        for title in titles:
            x, _ = title.get_position()
            # need to start again in case of window resizing
            title.set_position((x, 1.0))
            if title.get_text():
                for ax in axs:
                    ax.yaxis.get_tightbbox(renderer)  # update offsetText
                    if ax.yaxis.offsetText.get_text():
                        bb = ax.yaxis.offsetText.get_tightbbox(renderer)
                        if bb.intersection(title.get_tightbbox(renderer), bb):
                            top = bb.ymax
            if top < 0:
                # the top of Axes is not even on the figure, so don't try and
                # automatically place it.
                _log.debug('top of Axes not in the figure, so title not moved')
                return
            if title.get_window_extent(renderer).ymin < top:
                _, y = self.transAxes.inverted().transform((0, top))
                title.set_position((x, y))
                # empirically, this doesn't always get the min to top,
                # so we need to adjust again.
                if title.get_window_extent(renderer).ymin < top:
                    _, y = self.transAxes.inverted().transform(
                        (0., 2 * top - title.get_window_extent(renderer).ymin))
                    title.set_position((x, y))

        ymax = max(title.get_position()[1] for title in titles)
        for title in titles:
            # now line up all the titles at the highest baseline.
            x, _ = title.get_position()
            title.set_position((x, ymax))

    # Drawing
    @martist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if renderer is None:
            raise RuntimeError('No renderer defined')
        if not self.get_visible():
            return
        self._unstale_viewLim()

        renderer.open_group('axes', gid=self.get_gid())

        # prevent triggering call backs during the draw process
        self._stale = True

        # loop over self and child Axes...
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        artists = self.get_children()
        artists.remove(self.patch)

        # the frame draws the edges around the Axes patch -- we
        # decouple these so the patch can be in the background and the
        # frame in the foreground. Do this before drawing the axis
        # objects so that the spine has the opportunity to update them.
        if not (self.axison and self._frameon):
            for spine in self.spines.values():
                artists.remove(spine)

        self._update_title_position(renderer)

        if not self.axison:
            for _axis in self._axis_map.values():
                artists.remove(_axis)

        if not self.get_figure(root=True).canvas.is_saving():
            artists = [
                a for a in artists
                if not a.get_animated() or isinstance(a, mimage.AxesImage)]
        artists = sorted(artists, key=attrgetter('zorder'))

        # rasterize artists with negative zorder
        # if the minimum zorder is negative, start rasterization
        rasterization_zorder = self._rasterization_zorder

        if (rasterization_zorder is not None and
                artists and artists[0].zorder < rasterization_zorder):
            split_index = np.searchsorted(
                [art.zorder for art in artists],
                rasterization_zorder, side='right'
            )
            artists_rasterized = artists[:split_index]
            artists = artists[split_index:]
        else:
            artists_rasterized = []

        if self.axison and self._frameon:
            if artists_rasterized:
                artists_rasterized = [self.patch] + artists_rasterized
            else:
                artists = [self.patch] + artists

        if artists_rasterized:
            _draw_rasterized(self.get_figure(root=True), artists_rasterized, renderer)

        mimage._draw_list_compositing_images(
            renderer, self, artists, self.get_figure(root=True).suppressComposite)

        renderer.close_group('axes')
        self.stale = False

    def draw_artist(self, a):
        """
        Efficiently redraw a single artist.
        """
        a.draw(self.get_figure(root=True).canvas.get_renderer())

    def redraw_in_frame(self):
        """
        Efficiently redraw Axes data, but not axis ticks, labels, etc.
        """
        with ExitStack() as stack:
            for artist in [*self._axis_map.values(),
                           self.title, self._left_title, self._right_title]:
                stack.enter_context(artist._cm_set(visible=False))
            self.draw(self.get_figure(root=True).canvas.get_renderer())

    # Axes rectangle characteristics

    def get_frame_on(self):
        """Get whether the Axes rectangle patch is drawn."""
        return self._frameon

    def set_frame_on(self, b):
        """
        Set whether the Axes rectangle patch is drawn.

        Parameters
        ----------
        b : bool
        """
        self._frameon = b
        self.stale = True

    def get_axisbelow(self):
        """
        Get whether axis ticks and gridlines are above or below most artists.

        Returns
        -------
        bool or 'line'

        See Also
        --------
        set_axisbelow
        """
        return self._axisbelow

    def set_axisbelow(self, b):
        """
        Set whether axis ticks and gridlines are above or below most artists.

        This controls the zorder of the ticks and gridlines. For more
        information on the zorder see :doc:`/gallery/misc/zorder_demo`.

        Parameters
        ----------
        b : bool or 'line'
            Possible values:

            - *True* (zorder = 0.5): Ticks and gridlines are below patches and
              lines, though still above images.
            - 'line' (zorder = 1.5): Ticks and gridlines are above patches
              (e.g. rectangles, with default zorder = 1) but still below lines
              and markers (with their default zorder = 2).
            - *False* (zorder = 2.5): Ticks and gridlines are above patches
              and lines / markers.

        Notes
        -----
        For more control, call the `~.Artist.set_zorder` method of each axis.

        See Also
        --------
        get_axisbelow
        """
        # Check that b is True, False or 'line'
        self._axisbelow = axisbelow = validate_axisbelow(b)
        zorder = {
            True: 0.5,
            'line': 1.5,
            False: 2.5,
        }[axisbelow]
        for axis in self._axis_map.values():
            axis.set_zorder(zorder)
        self.stale = True

    @_docstring.interpd
    def grid(self, visible=None, which='major', axis='both', **kwargs):
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None, optional
            Whether to show the grid lines.  If any *kwargs* are supplied, it
            is assumed you want the grid on and *visible* will be set to True.

            If *visible* is *None* and there are no *kwargs*, this toggles the
            visibility of the lines.

        which : {'major', 'minor', 'both'}, optional
            The grid lines to apply the changes on.

        axis : {'both', 'x', 'y'}, optional
            The axis to apply the changes on.

        **kwargs : `~matplotlib.lines.Line2D` properties
            Define the line properties of the grid, e.g.::

                grid(color='r', linestyle='-', linewidth=2)

            Valid keyword arguments are:

            %(Line2D:kwdoc)s

        Notes
        -----
        The axis is drawn as a unit, so the effective zorder for drawing the
        grid is determined by the zorder of each axis, not by the zorder of the
        `.Line2D` objects comprising the grid.  Therefore, to set grid zorder,
        use `.set_axisbelow` or, for more control, call the
        `~.Artist.set_zorder` method of each axis.
        """
        _api.check_in_list(['x', 'y', 'both'], axis=axis)
        if axis in ['x', 'both']:
            self.xaxis.grid(visible, which=which, **kwargs)
        if axis in ['y', 'both']:
            self.yaxis.grid(visible, which=which, **kwargs)

    def ticklabel_format(self, *, axis='both', style=None, scilimits=None,
                         useOffset=None, useLocale=None, useMathText=None):
        r"""
        Configure the `.ScalarFormatter` used by default for linear Axes.

        If a parameter is not set, the corresponding property of the formatter
        is left unchanged.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to configure.  Only major ticks are affected.

        style : {'sci', 'scientific', 'plain'}
            Whether to use scientific notation.
            The formatter default is to use scientific notation.
            'sci' is equivalent to 'scientific'.

        scilimits : pair of ints (m, n)
            Scientific notation is used only for numbers outside the range
            10\ :sup:`m` to 10\ :sup:`n` (and only if the formatter is
            configured to use scientific notation at all).  Use (0, 0) to
            include all numbers.  Use (m, m) where m != 0 to fix the order of
            magnitude to 10\ :sup:`m`.
            The formatter default is :rc:`axes.formatter.limits`.

        useOffset : bool or float
            If True, the offset is calculated as needed.
            If False, no offset is used.
            If a numeric value, it sets the offset.
            The formatter default is :rc:`axes.formatter.useoffset`.

        useLocale : bool
            Whether to format the number using the current locale or using the
            C (English) locale.  This affects e.g. the decimal separator.  The
            formatter default is :rc:`axes.formatter.use_locale`.

        useMathText : bool
            Render the offset and scientific notation in mathtext.
            The formatter default is :rc:`axes.formatter.use_mathtext`.

        Raises
        ------
        AttributeError
            If the current formatter is not a `.ScalarFormatter`.
        """
        if isinstance(style, str):
            style = style.lower()
        axis = axis.lower()
        if scilimits is not None:
            try:
                m, n = scilimits
                m + n + 1  # check that both are numbers
            except (ValueError, TypeError) as err:
                raise ValueError("scilimits must be a sequence of 2 integers"
                                 ) from err
        STYLES = {'sci': True, 'scientific': True, 'plain': False, '': None, None: None}
        # The '' option is included for backwards-compatibility.
        is_sci_style = _api.check_getitem(STYLES, style=style)
        axis_map = {**{k: [v] for k, v in self._axis_map.items()},
                    'both': list(self._axis_map.values())}
        axises = _api.check_getitem(axis_map, axis=axis)
        try:
            for axis in axises:
                if is_sci_style is not None:
                    axis.major.formatter.set_scientific(is_sci_style)
                if scilimits is not None:
                    axis.major.formatter.set_powerlimits(scilimits)
                if useOffset is not None:
                    axis.major.formatter.set_useOffset(useOffset)
                if useLocale is not None:
                    axis.major.formatter.set_useLocale(useLocale)
                if useMathText is not None:
                    axis.major.formatter.set_useMathText(useMathText)
        except AttributeError as err:
            raise AttributeError(
                "This method only works with the ScalarFormatter") from err

    def locator_params(self, axis='both', tight=None, **kwargs):
        """
        Control behavior of major tick locators.

        Because the locator is involved in autoscaling, `~.Axes.autoscale_view`
        is called automatically after the parameters are changed.

        Parameters
        ----------
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be
            set to 'z', and 'both' refers to all three axes.)
        tight : bool or None, optional
            Parameter passed to `~.Axes.autoscale_view`.
            Default is None, for no change.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed to directly to the
            ``set_params()`` method of the locator. Supported keywords depend
            on the type of the locator. See for example
            `~.ticker.MaxNLocator.set_params` for the `.ticker.MaxNLocator`
            used by default for linear.

        Examples
        --------
        When plotting small subplots, one might want to reduce the maximum
        number of ticks and use tight bounds, for example::

            ax.locator_params(tight=True, nbins=4)

        """
        _api.check_in_list([*self._axis_names, "both"], axis=axis)
        for name in self._axis_names:
            if axis in [name, "both"]:
                loc = self._axis_map[name].get_major_locator()
                loc.set_params(**kwargs)
                self._request_autoscale_view(name, tight=tight)
        self.stale = True

    def tick_params(self, axis='both', **kwargs):
        """
        Change the appearance of ticks, tick labels, and gridlines.

        Tick properties that are not explicitly set using the keyword
        arguments remain unchanged unless *reset* is True. For the current
        style settings, see `.Axis.get_tick_params`.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to which the parameters are applied.
        which : {'major', 'minor', 'both'}, default: 'major'
            The group of ticks to which the parameters are applied.
        reset : bool, default: False
            Whether to reset the ticks to defaults before updating them.

        Other Parameters
        ----------------
        direction : {'in', 'out', 'inout'}
            Puts ticks inside the Axes, outside the Axes, or both.
        length : float
            Tick length in points.
        width : float
            Tick width in points.
        color : :mpltype:`color`
            Tick color.
        pad : float
            Distance in points between tick and label.
        labelsize : float or str
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : :mpltype:`color`
            Tick label color.
        labelfontfamily : str
            Tick label font.
        colors : :mpltype:`color`
            Tick color and label color.
        zorder : float
            Tick and label zorder.
        bottom, top, left, right : bool
            Whether to draw the respective ticks.
        labelbottom, labeltop, labelleft, labelright : bool
            Whether to draw the respective tick labels.
        labelrotation : float
            Tick label rotation
        grid_color : :mpltype:`color`
            Gridline color.
        grid_alpha : float
            Transparency of gridlines: 0 (transparent) to 1 (opaque).
        grid_linewidth : float
            Width of gridlines in points.
        grid_linestyle : str
            Any valid `.Line2D` line style spec.

        Examples
        --------
        ::

            ax.tick_params(direction='out', length=6, width=2, colors='r',
                           grid_color='r', grid_alpha=0.5)

        This will make all major ticks be red, pointing out of the box,
        and with dimensions 6 points by 2 points.  Tick labels will
        also be red.  Gridlines will be red and translucent.

        """
        _api.check_in_list(['x', 'y', 'both'], axis=axis)
        if axis in ['x', 'both']:
            xkw = dict(kwargs)
            xkw.pop('left', None)
            xkw.pop('right', None)
            xkw.pop('labelleft', None)
            xkw.pop('labelright', None)
            self.xaxis.set_tick_params(**xkw)
        if axis in ['y', 'both']:
            ykw = dict(kwargs)
            ykw.pop('top', None)
            ykw.pop('bottom', None)
            ykw.pop('labeltop', None)
            ykw.pop('labelbottom', None)
            self.yaxis.set_tick_params(**ykw)

    def set_axis_off(self):
        """
        Hide all visual components of the x- and y-axis.

        This sets a flag to suppress drawing of all axis decorations, i.e.
        axis labels, axis spines, and the axis tick component (tick markers,
        tick labels, and grid lines). Individual visibility settings of these
        components are ignored as long as `set_axis_off()` is in effect.
        """
        self.axison = False
        self.stale = True

    def set_axis_on(self):
        """
        Do not hide all visual components of the x- and y-axis.

        This reverts the effect of a prior `.set_axis_off()` call. Whether the
        individual axis decorations are drawn is controlled by their respective
        visibility settings.

        This is on by default.
        """
        self.axison = True
        self.stale = True

    # data limits, ticks, tick labels, and formatting

    def get_xlabel(self):
        """
        Get the xlabel text string.
        """
        label = self.xaxis.label
        return label.get_text()

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *,
                   loc=None, **kwargs):
        """
        Set the label for the x-axis.

        Parameters
        ----------
        xlabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'left', 'center', 'right'}, default: :rc:`xaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *x* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
        if labelpad is not None:
            self.xaxis.labelpad = labelpad
        protected_kw = ['x', 'horizontalalignment', 'ha']
        if {*kwargs} & {*protected_kw}:
            if loc is not None:
                raise TypeError(f"Specifying 'loc' is disallowed when any of "
                                f"its corresponding low level keyword "
                                f"arguments ({protected_kw}) are also "
                                f"supplied")

        else:
            loc = (loc if loc is not None
                   else mpl.rcParams['xaxis.labellocation'])
            _api.check_in_list(('left', 'center', 'right'), loc=loc)

            x = {
                'left': 0,
                'center': 0.5,
                'right': 1,
            }[loc]
            kwargs.update(x=x, horizontalalignment=loc)

        return self.xaxis.set_label_text(xlabel, fontdict, **kwargs)

    def invert_xaxis(self):
        """
        Invert the x-axis.

        See Also
        --------
        xaxis_inverted
        get_xlim, set_xlim
        get_xbound, set_xbound
        """
        self.xaxis.set_inverted(not self.xaxis.get_inverted())

    xaxis_inverted = _axis_method_wrapper("xaxis", "get_inverted")

    def get_xbound(self):
        """
        Return the lower and upper x-axis bounds, in increasing order.

        See Also
        --------
        set_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        left, right = self.get_xlim()
        if left < right:
            return left, right
        else:
            return right, left

    def set_xbound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the x-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalex_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

            .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
        if upper is None and np.iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_xbound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        self.set_xlim(sorted((lower, upper),
                             reverse=bool(self.xaxis_inverted())),
                      auto=None)

    def get_xlim(self):
        """
        Return the x-axis view limits.

        Returns
        -------
        left, right : (float, float)
            The current x-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_xlim
        .Axes.set_xbound, .Axes.get_xbound
        .Axes.invert_xaxis, .Axes.xaxis_inverted

        Notes
        -----
        The x-axis may be inverted, in which case the *left* value will
        be greater than the *right* value.
        """
        return tuple(self.viewLim.intervalx)

    def _validate_converted_limits(self, limit, convert):
        """
        Raise ValueError if converted limits are non-finite.

        Note that this function also accepts None as a limit argument.

        Returns
        -------
        The limit value after call to convert(), or None if limit is None.
        """
        if limit is not None:
            converted_limit = convert(limit)
            if isinstance(converted_limit, np.ndarray):
                converted_limit = converted_limit.squeeze()
            if (isinstance(converted_limit, Real)
                    and not np.isfinite(converted_limit)):
                raise ValueError("Axis limits cannot be NaN or Inf")
            return converted_limit

    def set_xlim(self, left=None, right=None, *, emit=True, auto=False,
                 xmin=None, xmax=None):
        """
        Set the x-axis view limits.

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
            Whether to turn on autoscaling of the x-axis. True turns on,
            False turns off, None leaves unchanged.

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
        case the x-axis values will decrease from left to right.

        Examples
        --------
        >>> set_xlim(left, right)
        >>> set_xlim((left, right))
        >>> left, right = set_xlim(left, right)

        One limit may be left unchanged.

        >>> set_xlim(right=right_lim)

        Limits may be passed in reverse order to flip the direction of
        the x-axis. For example, suppose *x* represents the number of
        years before present. The x-axis limits might be set like the
        following so 5000 years ago is on the left of the plot and the
        present is on the right.

        >>> set_xlim(5000, 0)
        """
        if right is None and np.iterable(left):
            left, right = left
        if xmin is not None:
            if left is not None:
                raise TypeError("Cannot pass both 'left' and 'xmin'")
            left = xmin
        if xmax is not None:
            if right is not None:
                raise TypeError("Cannot pass both 'right' and 'xmax'")
            right = xmax
        return self.xaxis._set_lim(left, right, emit=emit, auto=auto)

    get_xscale = _axis_method_wrapper("xaxis", "get_scale")
    set_xscale = _axis_method_wrapper("xaxis", "_set_axes_scale")
    get_xticks = _axis_method_wrapper("xaxis", "get_ticklocs")
    set_xticks = _axis_method_wrapper("xaxis", "set_ticks",
                                      doc_sub={'set_ticks': 'set_xticks'})
    get_xmajorticklabels = _axis_method_wrapper("xaxis", "get_majorticklabels")
    get_xminorticklabels = _axis_method_wrapper("xaxis", "get_minorticklabels")
    get_xticklabels = _axis_method_wrapper("xaxis", "get_ticklabels")
    set_xticklabels = _axis_method_wrapper(
        "xaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes.set_xticks"})

    def get_ylabel(self):
        """
        Get the ylabel text string.
        """
        label = self.yaxis.label
        return label.get_text()

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *,
                   loc=None, **kwargs):
        """
        Set the label for the y-axis.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *y* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
        if labelpad is not None:
            self.yaxis.labelpad = labelpad
        protected_kw = ['y', 'horizontalalignment', 'ha']
        if {*kwargs} & {*protected_kw}:
            if loc is not None:
                raise TypeError(f"Specifying 'loc' is disallowed when any of "
                                f"its corresponding low level keyword "
                                f"arguments ({protected_kw}) are also "
                                f"supplied")

        else:
            loc = (loc if loc is not None
                   else mpl.rcParams['yaxis.labellocation'])
            _api.check_in_list(('bottom', 'center', 'top'), loc=loc)

            y, ha = {
                'bottom': (0, 'left'),
                'center': (0.5, 'center'),
                'top': (1, 'right')
            }[loc]
            kwargs.update(y=y, horizontalalignment=ha)

        return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)

    def invert_yaxis(self):
        """
        Invert the y-axis.

        See Also
        --------
        yaxis_inverted
        get_ylim, set_ylim
        get_ybound, set_ybound
        """
        self.yaxis.set_inverted(not self.yaxis.get_inverted())

    yaxis_inverted = _axis_method_wrapper("yaxis", "get_inverted")

    def get_ybound(self):
        """
        Return the lower and upper y-axis bounds, in increasing order.

        See Also
        --------
        set_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        bottom, top = self.get_ylim()
        if bottom < top:
            return bottom, top
        else:
            return top, bottom

    def set_ybound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the y-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

         .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
        if upper is None and np.iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_ybound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        self.set_ylim(sorted((lower, upper),
                             reverse=bool(self.yaxis_inverted())),
                      auto=None)

    def get_ylim(self):
        """
        Return the y-axis view limits.

        Returns
        -------
        bottom, top : (float, float)
            The current y-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_ylim
        .Axes.set_ybound, .Axes.get_ybound
        .Axes.invert_yaxis, .Axes.yaxis_inverted

        Notes
        -----
        The y-axis may be inverted, in which case the *bottom* value
        will be greater than the *top* value.
        """
        return tuple(self.viewLim.intervaly)

    def set_ylim(self, bottom=None, top=None, *, emit=True, auto=False,
                 ymin=None, ymax=None):
        """
        Set the y-axis view limits.

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
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        if ymin is not None:
            if bottom is not None:
                raise TypeError("Cannot pass both 'bottom' and 'ymin'")
            bottom = ymin
        if ymax is not None:
            if top is not None:
                raise TypeError("Cannot pass both 'top' and 'ymax'")
            top = ymax
        return self.yaxis._set_lim(bottom, top, emit=emit, auto=auto)

    get_yscale = _axis_method_wrapper("yaxis", "get_scale")
    set_yscale = _axis_method_wrapper("yaxis", "_set_axes_scale")
    get_yticks = _axis_method_wrapper("yaxis", "get_ticklocs")
    set_yticks = _axis_method_wrapper("yaxis", "set_ticks",
                                      doc_sub={'set_ticks': 'set_yticks'})
    get_ymajorticklabels = _axis_method_wrapper("yaxis", "get_majorticklabels")
    get_yminorticklabels = _axis_method_wrapper("yaxis", "get_minorticklabels")
    get_yticklabels = _axis_method_wrapper("yaxis", "get_ticklabels")
    set_yticklabels = _axis_method_wrapper(
        "yaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes.set_yticks"})

    xaxis_date = _axis_method_wrapper("xaxis", "axis_date")
    yaxis_date = _axis_method_wrapper("yaxis", "axis_date")

    def format_xdata(self, x):
        """
        Return *x* formatted as an x-value.

        This function will use the `.fmt_xdata` attribute if it is not None,
        else will fall back on the xaxis major formatter.
        """
        return (self.fmt_xdata if self.fmt_xdata is not None
                else self.xaxis.get_major_formatter().format_data_short)(x)

    def format_ydata(self, y):
        """
        Return *y* formatted as a y-value.

        This function will use the `.fmt_ydata` attribute if it is not None,
        else will fall back on the yaxis major formatter.
        """
        return (self.fmt_ydata if self.fmt_ydata is not None
                else self.yaxis.get_major_formatter().format_data_short)(y)

    def format_coord(self, x, y):
        """Return a format string formatting the *x*, *y* coordinates."""
        twins = self._twinned_axes.get_siblings(self)
        if len(twins) == 1:
            return "(x, y) = ({}, {})".format(
                "???" if x is None else self.format_xdata(x),
                "???" if y is None else self.format_ydata(y))
        screen_xy = self.transData.transform((x, y))
        xy_strs = []
        # Retrieve twins in the order of self.figure.axes to sort tied zorders (which is
        # the common case) by the order in which they are added to the figure.
        for ax in sorted(twins, key=attrgetter("zorder")):
            data_x, data_y = ax.transData.inverted().transform(screen_xy)
            xy_strs.append(
                "({}, {})".format(ax.format_xdata(data_x), ax.format_ydata(data_y)))
        return "(x, y) = {}".format(" | ".join(xy_strs))

    def minorticks_on(self):
        """
        Display minor ticks on the Axes.

        Displaying minor ticks may reduce performance; you may turn them off
        using `minorticks_off()` if drawing speed is a problem.
        """
        self.xaxis.minorticks_on()
        self.yaxis.minorticks_on()

    def minorticks_off(self):
        """Remove minor ticks from the Axes."""
        self.xaxis.minorticks_off()
        self.yaxis.minorticks_off()

    # Interactive manipulation

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.
        """
        return True

    def can_pan(self):
        """
        Return whether this Axes supports any pan/zoom button functionality.
        """
        return True

    def get_navigate(self):
        """
        Get whether the Axes responds to navigation commands.
        """
        return self._navigate

    def set_navigate(self, b):
        """
        Set whether the Axes responds to navigation toolbar commands.

        Parameters
        ----------
        b : bool

        See Also
        --------
        matplotlib.axes.Axes.set_forward_navigation_events

        """
        self._navigate = b

    def get_navigate_mode(self):
        """
        Get the navigation toolbar button status: 'PAN', 'ZOOM', or None.
        """
        return self._navigate_mode

    def set_navigate_mode(self, b):
        """
        Set the navigation toolbar button status.

        .. warning::
            This is not a user-API function.

        """
        self._navigate_mode = b

    def _get_view(self):
        """
        Save information required to reproduce the current view.

        This method is called before a view is changed, such as during a pan or zoom
        initiated by the user.  It returns an opaque object that describes the current
        view, in a format compatible with :meth:`_set_view`.

        The default implementation saves the view limits and autoscaling state.
        Subclasses may override this as needed, as long as :meth:`_set_view` is also
        adjusted accordingly.
        """
        return {
            "xlim": self.get_xlim(), "autoscalex_on": self.get_autoscalex_on(),
            "ylim": self.get_ylim(), "autoscaley_on": self.get_autoscaley_on(),
        }

    def _set_view(self, view):
        """
        Apply a previously saved view.

        This method is called when restoring a view (with the return value of
        :meth:`_get_view` as argument), such as with the navigation buttons.

        Subclasses that override :meth:`_get_view` also need to override this method
        accordingly.
        """
        self.set(**view)

    def _prepare_view_from_bbox(self, bbox, direction='in',
                                mode=None, twinx=False, twiny=False):
        """
        Helper function to prepare the new bounds from a bbox.

        This helper function returns the new x and y bounds from the zoom
        bbox. This a convenience method to abstract the bbox logic
        out of the base setter.
        """
        if len(bbox) == 3:
            xp, yp, scl = bbox  # Zooming code
            if scl == 0:  # Should not happen
                scl = 1.
            if scl > 1:
                direction = 'in'
            else:
                direction = 'out'
                scl = 1/scl
            # get the limits of the axes
            (xmin, ymin), (xmax, ymax) = self.transData.transform(
                np.transpose([self.get_xlim(), self.get_ylim()]))
            # set the range
            xwidth = xmax - xmin
            ywidth = ymax - ymin
            xcen = (xmax + xmin)*.5
            ycen = (ymax + ymin)*.5
            xzc = (xp*(scl - 1) + xcen)/scl
            yzc = (yp*(scl - 1) + ycen)/scl
            bbox = [xzc - xwidth/2./scl, yzc - ywidth/2./scl,
                    xzc + xwidth/2./scl, yzc + ywidth/2./scl]
        elif len(bbox) != 4:
            # should be len 3 or 4 but nothing else
            _api.warn_external(
                "Warning in _set_view_from_bbox: bounding box is not a tuple "
                "of length 3 or 4. Ignoring the view change.")
            return

        # Original limits.
        xmin0, xmax0 = self.get_xbound()
        ymin0, ymax0 = self.get_ybound()
        # The zoom box in screen coords.
        startx, starty, stopx, stopy = bbox
        # Convert to data coords.
        (startx, starty), (stopx, stopy) = self.transData.inverted().transform(
            [(startx, starty), (stopx, stopy)])
        # Clip to axes limits.
        xmin, xmax = np.clip(sorted([startx, stopx]), xmin0, xmax0)
        ymin, ymax = np.clip(sorted([starty, stopy]), ymin0, ymax0)
        # Don't double-zoom twinned axes or if zooming only the other axis.
        if twinx or mode == "y":
            xmin, xmax = xmin0, xmax0
        if twiny or mode == "x":
            ymin, ymax = ymin0, ymax0

        if direction == "in":
            new_xbound = xmin, xmax
            new_ybound = ymin, ymax

        elif direction == "out":
            x_trf = self.xaxis.get_transform()
            sxmin0, sxmax0, sxmin, sxmax = x_trf.transform(
                [xmin0, xmax0, xmin, xmax])  # To screen space.
            factor = (sxmax0 - sxmin0) / (sxmax - sxmin)  # Unzoom factor.
            # Move original bounds away by
            # (factor) x (distance between unzoom box and Axes bbox).
            sxmin1 = sxmin0 - factor * (sxmin - sxmin0)
            sxmax1 = sxmax0 + factor * (sxmax0 - sxmax)
            # And back to data space.
            new_xbound = x_trf.inverted().transform([sxmin1, sxmax1])

            y_trf = self.yaxis.get_transform()
            symin0, symax0, symin, symax = y_trf.transform(
                [ymin0, ymax0, ymin, ymax])
            factor = (symax0 - symin0) / (symax - symin)
            symin1 = symin0 - factor * (symin - symin0)
            symax1 = symax0 + factor * (symax0 - symax)
            new_ybound = y_trf.inverted().transform([symin1, symax1])

        return new_xbound, new_ybound

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Update view from a selection bbox.

        .. note::

            Intended to be overridden by new projection types, but if not, the
            default implementation sets the view limits to the bbox directly.

        Parameters
        ----------
        bbox : 4-tuple or 3 tuple
            * If bbox is a 4 tuple, it is the selected bounding box limits,
              in *display* coordinates.
            * If bbox is a 3 tuple, it is an (xp, yp, scl) triple, where
              (xp, yp) is the center of zooming and scl the scale factor to
              zoom by.

        direction : str
            The direction to apply the bounding box.
                * `'in'` - The bounding box describes the view directly, i.e.,
                           it zooms in.
                * `'out'` - The bounding box describes the size to make the
                            existing view, i.e., it zooms out.

        mode : str or None
            The selection mode, whether to apply the bounding box in only the
            `'x'` direction, `'y'` direction or both (`None`).

        twinx : bool
            Whether this axis is twinned in the *x*-direction.

        twiny : bool
            Whether this axis is twinned in the *y*-direction.
        """
        new_xbound, new_ybound = self._prepare_view_from_bbox(
            bbox, direction=direction, mode=mode, twinx=twinx, twiny=twiny)
        if not twinx and mode != "y":
            self.set_xbound(new_xbound)
            self.set_autoscalex_on(False)
        if not twiny and mode != "x":
            self.set_ybound(new_ybound)
            self.set_autoscaley_on(False)

    def start_pan(self, x, y, button):
        """
        Called when a pan operation has started.

        Parameters
        ----------
        x, y : float
            The mouse coordinates in display coords.
        button : `.MouseButton`
            The pressed mouse button.

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
        self._pan_start = types.SimpleNamespace(
            lim=self.viewLim.frozen(),
            trans=self.transData.frozen(),
            trans_inverse=self.transData.inverted().frozen(),
            bbox=self.bbox.frozen(),
            x=x,
            y=y)

    def end_pan(self):
        """
        Called when a pan operation completes (when the mouse button is up.)

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
        del self._pan_start

    def _get_pan_points(self, button, key, x, y):
        """
        Helper function to return the new points after a pan.

        This helper function returns the points on the axis after a pan has
        occurred. This is a convenience method to abstract the pan logic
        out of the base setter.
        """
        def format_deltas(key, dx, dy):
            if key == 'control':
                if abs(dx) > abs(dy):
                    dy = dx
                else:
                    dx = dy
            elif key == 'x':
                dy = 0
            elif key == 'y':
                dx = 0
            elif key == 'shift':
                if 2 * abs(dx) < abs(dy):
                    dx = 0
                elif 2 * abs(dy) < abs(dx):
                    dy = 0
                elif abs(dx) > abs(dy):
                    dy = dy / abs(dy) * abs(dx)
                else:
                    dx = dx / abs(dx) * abs(dy)
            return dx, dy

        p = self._pan_start
        dx = x - p.x
        dy = y - p.y
        if dx == dy == 0:
            return
        if button == 1:
            dx, dy = format_deltas(key, dx, dy)
            result = p.bbox.translated(-dx, -dy).transformed(p.trans_inverse)
        elif button == 3:
            try:
                dx = -dx / self.bbox.width
                dy = -dy / self.bbox.height
                dx, dy = format_deltas(key, dx, dy)
                if self.get_aspect() != 'auto':
                    dx = dy = 0.5 * (dx + dy)
                alpha = np.power(10.0, (dx, dy))
                start = np.array([p.x, p.y])
                oldpoints = p.lim.transformed(p.trans)
                newpoints = start + alpha * (oldpoints - start)
                result = (mtransforms.Bbox(newpoints)
                          .transformed(p.trans_inverse))
            except OverflowError:
                _api.warn_external('Overflow while panning')
                return
        else:
            return

        valid = np.isfinite(result.transformed(p.trans))
        points = result.get_points().astype(object)
        # Just ignore invalid limits (typically, underflow in log-scale).
        points[~valid] = None
        return points

    def drag_pan(self, button, key, x, y):
        """
        Called when the mouse moves during a pan operation.

        Parameters
        ----------
        button : `.MouseButton`
            The pressed mouse button.
        key : str or None
            The pressed key, if any.
        x, y : float
            The mouse coordinates in display coords.

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
        points = self._get_pan_points(button, key, x, y)
        if points is not None:
            self.set_xlim(points[:, 0])
            self.set_ylim(points[:, 1])

    def get_children(self):
        # docstring inherited.
        return [
            *self._children,
            *self.spines.values(),
            *self._axis_map.values(),
            self.title, self._left_title, self._right_title,
            *self.child_axes,
            *([self.legend_] if self.legend_ is not None else []),
            self.patch,
        ]

    def contains(self, mouseevent):
        # docstring inherited.
        return self.patch.contains(mouseevent)

    def contains_point(self, point):
        """
        Return whether *point* (pair of pixel coordinates) is inside the Axes
        patch.
        """
        return self.patch.contains_point(point, radius=1.0)

    def get_default_bbox_extra_artists(self):
        """
        Return a default list of artists that are used for the bounding box
        calculation.

        Artists are excluded either by not being visible or
        ``artist.set_in_layout(False)``.
        """

        artists = self.get_children()

        for axis in self._axis_map.values():
            # axis tight bboxes are calculated separately inside
            # Axes.get_tightbbox() using for_layout_only=True
            artists.remove(axis)
        if not (self.axison and self._frameon):
            # don't do bbox on spines if frame not on.
            for spine in self.spines.values():
                artists.remove(spine)

        artists.remove(self.title)
        artists.remove(self._left_title)
        artists.remove(self._right_title)

        # always include types that do not internally implement clipping
        # to Axes. may have clip_on set to True and clip_box equivalent
        # to ax.bbox but then ignore these properties during draws.
        noclip = (_AxesBase, maxis.Axis,
                  offsetbox.AnnotationBbox, offsetbox.OffsetBox)
        return [a for a in artists if a.get_visible() and a.get_in_layout()
                and (isinstance(a, noclip) or not a._fully_clipped_to_axes())]

    def get_tightbbox(self, renderer=None, *, call_axes_locator=True,
                      bbox_extra_artists=None, for_layout_only=False):
        """
        Return the tight bounding box of the Axes, including axis and their
        decorators (xlabel, title, etc).

        Artists that have ``artist.set_in_layout(False)`` are not included
        in the bbox.

        Parameters
        ----------
        renderer : `.RendererBase` subclass
            renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        bbox_extra_artists : list of `.Artist` or ``None``
            List of artists to include in the tight bounding box.  If
            ``None`` (default), then all artist children of the Axes are
            included in the tight bounding box.

        call_axes_locator : bool, default: True
            If *call_axes_locator* is ``False``, it does not call the
            ``_axes_locator`` attribute, which is necessary to get the correct
            bounding box. ``call_axes_locator=False`` can be used if the
            caller is only interested in the relative size of the tightbbox
            compared to the Axes bbox.

        for_layout_only : default: False
            The bounding box will *not* include the x-extent of the title and
            the xlabel, or the y-extent of the ylabel.

        Returns
        -------
        `.BboxBase`
            Bounding box in figure pixel coordinates.

        See Also
        --------
        matplotlib.axes.Axes.get_window_extent
        matplotlib.axis.Axis.get_tightbbox
        matplotlib.spines.Spine.get_window_extent
        """

        bb = []
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()

        if not self.get_visible():
            return None

        locator = self.get_axes_locator()
        self.apply_aspect(
            locator(self, renderer) if locator and call_axes_locator else None)

        for axis in self._axis_map.values():
            if self.axison and axis.get_visible():
                ba = martist._get_tightbbox_for_layout_only(axis, renderer)
                if ba:
                    bb.append(ba)
        self._update_title_position(renderer)
        axbbox = self.get_window_extent(renderer)
        bb.append(axbbox)

        for title in [self.title, self._left_title, self._right_title]:
            if title.get_visible():
                bt = title.get_window_extent(renderer)
                if for_layout_only and bt.width > 0:
                    # make the title bbox 1 pixel wide so its width
                    # is not accounted for in bbox calculations in
                    # tight/constrained_layout
                    bt.x0 = (bt.x0 + bt.x1) / 2 - 0.5
                    bt.x1 = bt.x0 + 1.0
                bb.append(bt)

        bbox_artists = bbox_extra_artists
        if bbox_artists is None:
            bbox_artists = self.get_default_bbox_extra_artists()

        for a in bbox_artists:
            bbox = a.get_tightbbox(renderer)
            if (bbox is not None
                    and 0 < bbox.width < np.inf
                    and 0 < bbox.height < np.inf):
                bb.append(bbox)
        return mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

    def _make_twin_axes(self, *args, **kwargs):
        """Make a twinx Axes of self. This is used for twinx and twiny."""
        if 'sharex' in kwargs and 'sharey' in kwargs:
            # The following line is added in v2.2 to avoid breaking Seaborn,
            # which currently uses this internal API.
            if kwargs["sharex"] is not self and kwargs["sharey"] is not self:
                raise ValueError("Twinned Axes may share only one axis")
        ss = self.get_subplotspec()
        if ss:
            twin = self.get_figure(root=False).add_subplot(ss, *args, **kwargs)
        else:
            twin = self.get_figure(root=False).add_axes(
                self.get_position(True), *args, **kwargs,
                axes_locator=_TransformedBoundsLocator(
                    [0, 0, 1, 1], self.transAxes))
        self.set_adjustable('datalim')
        twin.set_adjustable('datalim')
        twin.set_zorder(self.zorder)

        self._twinned_axes.join(self, twin)
        return twin

    def twinx(self):
        """
        Create a twin Axes sharing the xaxis.

        Create a new Axes with an invisible x-axis and an independent
        y-axis positioned opposite to the original one (i.e. at right). The
        x-axis autoscale setting will be inherited from the original
        Axes.  To ensure that the tick marks of both y-axes align, see
        `~matplotlib.ticker.LinearLocator`.

        Returns
        -------
        Axes
            The newly created Axes instance

        Notes
        -----
        For those who are 'picking' artists while using twinx, pick
        events are only called for the artists in the top-most Axes.
        """
        ax2 = self._make_twin_axes(sharex=self)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_offset_position('right')
        ax2.set_autoscalex_on(self.get_autoscalex_on())
        self.yaxis.tick_left()
        ax2.xaxis.set_visible(False)
        ax2.patch.set_visible(False)
        ax2.xaxis.units = self.xaxis.units
        return ax2

    def twiny(self):
        """
        Create a twin Axes sharing the yaxis.

        Create a new Axes with an invisible y-axis and an independent
        x-axis positioned opposite to the original one (i.e. at top). The
        y-axis autoscale setting will be inherited from the original Axes.
        To ensure that the tick marks of both x-axes align, see
        `~matplotlib.ticker.LinearLocator`.

        Returns
        -------
        Axes
            The newly created Axes instance

        Notes
        -----
        For those who are 'picking' artists while using twiny, pick
        events are only called for the artists in the top-most Axes.
        """
        ax2 = self._make_twin_axes(sharey=self)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.set_autoscaley_on(self.get_autoscaley_on())
        self.xaxis.tick_bottom()
        ax2.yaxis.set_visible(False)
        ax2.patch.set_visible(False)
        ax2.yaxis.units = self.yaxis.units
        return ax2

    def get_shared_x_axes(self):
        """Return an immutable view on the shared x-axes Grouper."""
        return cbook.GrouperView(self._shared_axes["x"])

    def get_shared_y_axes(self):
        """Return an immutable view on the shared y-axes Grouper."""
        return cbook.GrouperView(self._shared_axes["y"])

    def label_outer(self, remove_inner_ticks=False):
        """
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row (or first row, if
        labels are on the top side); y-labels only for subplots on the first
        column (or last column, if labels are on the right side).

        Parameters
        ----------
        remove_inner_ticks : bool, default: False
            If True, remove the inner ticks as well (not only tick labels).

            .. versionadded:: 3.8
        """
        self._label_outer_xaxis(skip_non_rectangular_axes=False,
                                remove_inner_ticks=remove_inner_ticks)
        self._label_outer_yaxis(skip_non_rectangular_axes=False,
                                remove_inner_ticks=remove_inner_ticks)

    def _get_subplotspec_with_optional_colorbar(self):
        """
        Return the subplotspec for this Axes, except that if this Axes has been
        moved to a subgridspec to make room for a colorbar, then return the
        subplotspec that encloses both this Axes and the colorbar Axes.
        """
        ss = self.get_subplotspec()
        if any(cax.get_subplotspec() for cax in self._colorbars):
            ss = ss.get_gridspec()._subplot_spec
        return ss

    def _label_outer_xaxis(self, *, skip_non_rectangular_axes,
                           remove_inner_ticks=False):
        # see documentation in label_outer.
        if skip_non_rectangular_axes and not isinstance(self.patch,
                                                        mpl.patches.Rectangle):
            return
        ss = self._get_subplotspec_with_optional_colorbar()
        if ss is None:
            return
        label_position = self.xaxis.get_label_position()
        if not ss.is_first_row():  # Remove top label/ticklabels/offsettext.
            if label_position == "top":
                self.set_xlabel("")
            top_kw = {'top': False} if remove_inner_ticks else {}
            self.xaxis.set_tick_params(
                which="both", labeltop=False, **top_kw)
            if self.xaxis.offsetText.get_position()[1] == 1:
                self.xaxis.offsetText.set_visible(False)
        if not ss.is_last_row():  # Remove bottom label/ticklabels/offsettext.
            if label_position == "bottom":
                self.set_xlabel("")
            bottom_kw = {'bottom': False} if remove_inner_ticks else {}
            self.xaxis.set_tick_params(
                which="both", labelbottom=False, **bottom_kw)
            if self.xaxis.offsetText.get_position()[1] == 0:
                self.xaxis.offsetText.set_visible(False)

    def _label_outer_yaxis(self, *, skip_non_rectangular_axes,
                           remove_inner_ticks=False):
        # see documentation in label_outer.
        if skip_non_rectangular_axes and not isinstance(self.patch,
                                                        mpl.patches.Rectangle):
            return
        ss = self._get_subplotspec_with_optional_colorbar()
        if ss is None:
            return
        label_position = self.yaxis.get_label_position()
        if not ss.is_first_col():  # Remove left label/ticklabels/offsettext.
            if label_position == "left":
                self.set_ylabel("")
            left_kw = {'left': False} if remove_inner_ticks else {}
            self.yaxis.set_tick_params(
                which="both", labelleft=False, **left_kw)
            if self.yaxis.offsetText.get_position()[0] == 0:
                self.yaxis.offsetText.set_visible(False)
        if not ss.is_last_col():  # Remove right label/ticklabels/offsettext.
            if label_position == "right":
                self.set_ylabel("")
            right_kw = {'right': False} if remove_inner_ticks else {}
            self.yaxis.set_tick_params(
                which="both", labelright=False, **right_kw)
            if self.yaxis.offsetText.get_position()[0] == 1:
                self.yaxis.offsetText.set_visible(False)

    def set_forward_navigation_events(self, forward):
        """
        Set how pan/zoom events are forwarded to Axes below this one.

        Parameters
        ----------
        forward : bool or "auto"
            Possible values:

            - True: Forward events to other axes with lower or equal zorder.
            - False: Events are only executed on this axes.
            - "auto": Default behaviour (*True* for axes with an invisible
              patch and *False* otherwise)

        See Also
        --------
        matplotlib.axes.Axes.set_navigate

        """
        self._forward_navigation_events = forward

    def get_forward_navigation_events(self):
        """Get how pan/zoom events are forwarded to Axes below this one."""
        return self._forward_navigation_events


def _draw_rasterized(figure, artists, renderer):
    """
    A helper function for rasterizing the list of artists.

    The bookkeeping to track if we are or are not in rasterizing mode
    with the mixed-mode backends is relatively complicated and is now
    handled in the matplotlib.artist.allow_rasterization decorator.

    This helper defines the absolute minimum methods and attributes on a
    shim class to be compatible with that decorator and then uses it to
    rasterize the list of artists.

    This is maybe too-clever, but allows us to reuse the same code that is
    used on normal artists to participate in the "are we rasterizing"
    accounting.

    Please do not use this outside of the "rasterize below a given zorder"
    functionality of Axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure all of the artists belong to (not checked).  We need this
        because we can at the figure level suppress composition and insert each
        rasterized artist as its own image.

    artists : List[matplotlib.artist.Artist]
        The list of Artists to be rasterized.  These are assumed to all
        be in the same Figure.

    renderer : matplotlib.backendbases.RendererBase
        The currently active renderer

    Returns
    -------
    None

    """
    class _MinimalArtist:
        def get_rasterized(self):
            return True

        def get_agg_filter(self):
            return None

        def __init__(self, figure, artists):
            self.figure = figure
            self.artists = artists

        def get_figure(self, root=False):
            if root:
                return self.figure.get_figure(root=True)
            else:
                return self.figure

        @martist.allow_rasterization
        def draw(self, renderer):
            for a in self.artists:
                a.draw(renderer)

    return _MinimalArtist(figure, artists).draw(renderer)
<|endoftext|>"""
Interpolation inside triangular grids.
"""

import numpy as np

from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer

__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')


class TriInterpolator:
    """
    Abstract base class for classes used to interpolate on a triangular grid.

    Derived classes implement the following methods:

    - ``__call__(x, y)``,
      where x, y are array-like point coordinates of the same shape, and
      that returns a masked array of the same shape containing the
      interpolated z-values.

    - ``gradient(x, y)``,
      where x, y are array-like point coordinates of the same
      shape, and that returns a list of 2 masked arrays of the same shape
      containing the 2 derivatives of the interpolator (derivatives of
      interpolated z values with respect to x and y).
    """

    def __init__(self, triangulation, z, trifinder=None):
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

        self._z = np.asarray(z)
        if self._z.shape != self._triangulation.x.shape:
            raise ValueError("z array must have same length as triangulation x"
                             " and y arrays")

        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
        self._trifinder = trifinder or self._triangulation.get_trifinder()

        # Default scaling factors : 1.0 (= no scaling)
        # Scaling may be used for interpolations for which the order of
        # magnitude of x, y has an impact on the interpolant definition.
        # Please refer to :meth:`_interpolate_multikeys` for details.
        self._unit_x = 1.0
        self._unit_y = 1.0

        # Default triangle renumbering: None (= no renumbering)
        # Renumbering may be used to avoid unnecessary computations
        # if complex calculations are done inside the Interpolator.
        # Please refer to :meth:`_interpolate_multikeys` for details.
        self._tri_renum = None

    # __call__ and gradient docstrings are shared by all subclasses
    # (except, if needed, relevant additions).
    # However these methods are only implemented in subclasses to avoid
    # confusion in the documentation.
    _docstring__call__ = """
        Returns a masked array containing interpolated values at the specified
        (x, y) points.

        Parameters
        ----------
        x, y : array-like
            x and y coordinates of the same shape and any number of
            dimensions.

        Returns
        -------
        np.ma.array
            Masked array of the same shape as *x* and *y*; values corresponding
            to (*x*, *y*) points outside of the triangulation are masked out.

        """

    _docstringgradient = r"""
        Returns a list of 2 masked arrays containing interpolated derivatives
        at the specified (x, y) points.

        Parameters
        ----------
        x, y : array-like
            x and y coordinates of the same shape and any number of
            dimensions.

        Returns
        -------
        dzdx, dzdy : np.ma.array
            2 masked arrays of the same shape as *x* and *y*; values
            corresponding to (x, y) points outside of the triangulation
            are masked out.
            The first returned array contains the values of
            :math:`\frac{\partial z}{\partial x}` and the second those of
            :math:`\frac{\partial z}{\partial y}`.

        """

    def _interpolate_multikeys(self, x, y, tri_index=None,
                               return_keys=('z',)):
        """
        Versatile (private) method defined for all TriInterpolators.

        :meth:`_interpolate_multikeys` is a wrapper around method
        :meth:`_interpolate_single_key` (to be defined in the child
        subclasses).
        :meth:`_interpolate_single_key actually performs the interpolation,
        but only for 1-dimensional inputs and at valid locations (inside
        unmasked triangles of the triangulation).

        The purpose of :meth:`_interpolate_multikeys` is to implement the
        following common tasks needed in all subclasses implementations:

        - calculation of containing triangles
        - dealing with more than one interpolation request at the same
          location (e.g., if the 2 derivatives are requested, it is
          unnecessary to compute the containing triangles twice)
        - scaling according to self._unit_x, self._unit_y
        - dealing with points outside of the grid (with fill value np.nan)
        - dealing with multi-dimensional *x*, *y* arrays: flattening for
          :meth:`_interpolate_params` call and final reshaping.

        (Note that np.vectorize could do most of those things very well for
        you, but it does it by function evaluations over successive tuples of
        the input arrays. Therefore, this tends to be more time-consuming than
        using optimized numpy functions - e.g., np.dot - which can be used
        easily on the flattened inputs, in the child-subclass methods
        :meth:`_interpolate_single_key`.)

        It is guaranteed that the calls to :meth:`_interpolate_single_key`
        will be done with flattened (1-d) array-like input parameters *x*, *y*
        and with flattened, valid `tri_index` arrays (no -1 index allowed).

        Parameters
        ----------
        x, y : array-like
            x and y coordinates where interpolated values are requested.
        tri_index : array-like of int, optional
            Array of the containing triangle indices, same shape as
            *x* and *y*. Defaults to None. If None, these indices
            will be computed by a TriFinder instance.
            (Note: For point outside the grid, tri_index[ipt] shall be -1).
        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
            Defines the interpolation arrays to return, and in which order.

        Returns
        -------
        list of arrays
            Each array-like contains the expected interpolated values in the
            order defined by *return_keys* parameter.
        """
        # Flattening and rescaling inputs arrays x, y
        # (initial shape is stored for output)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sh_ret = x.shape
        if x.shape != y.shape:
            raise ValueError("x and y shall have same shapes."
                             f" Given: {x.shape} and {y.shape}")
        x = np.ravel(x)
        y = np.ravel(y)
        x_scaled = x/self._unit_x
        y_scaled = y/self._unit_y
        size_ret = np.size(x_scaled)

        # Computes & ravels the element indexes, extract the valid ones.
        if tri_index is None:
            tri_index = self._trifinder(x, y)
        else:
            if tri_index.shape != sh_ret:
                raise ValueError(
                    "tri_index array is provided and shall"
                    " have same shape as x and y. Given: "
                    f"{tri_index.shape} and {sh_ret}")
            tri_index = np.ravel(tri_index)

        mask_in = (tri_index != -1)
        if self._tri_renum is None:
            valid_tri_index = tri_index[mask_in]
        else:
            valid_tri_index = self._tri_renum[tri_index[mask_in]]
        valid_x = x_scaled[mask_in]
        valid_y = y_scaled[mask_in]

        ret = []
        for return_key in return_keys:
            # Find the return index associated with the key.
            try:
                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
            except KeyError as err:
                raise ValueError("return_keys items shall take values in"
                                 " {'z', 'dzdx', 'dzdy'}") from err

            # Sets the scale factor for f & df components
            scale = [1., 1./self._unit_x, 1./self._unit_y][return_index]

            # Computes the interpolation
            ret_loc = np.empty(size_ret, dtype=np.float64)
            ret_loc[~mask_in] = np.nan
            ret_loc[mask_in] = self._interpolate_single_key(
                return_key, valid_tri_index, valid_x, valid_y) * scale
            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]

        return ret

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        """
        Interpolate at points belonging to the triangulation
        (inside an unmasked triangles).

        Parameters
        ----------
        return_key : {'z', 'dzdx', 'dzdy'}
            The requested values (z or its derivatives).
        tri_index : 1D int array
            Valid triangle index (cannot be -1).
        x, y : 1D arrays, same shape as `tri_index`
            Valid locations where interpolation is requested.

        Returns
        -------
        1-d array
            Returned array of the same size as *tri_index*
        """
        raise NotImplementedError("TriInterpolator subclasses" +
                                  "should implement _interpolate_single_key!")


class LinearTriInterpolator(TriInterpolator):
    """
    Linear interpolator on a triangular grid.

    Each triangle is represented by a plane so that an interpolated value at
    point (x, y) lies on the plane of the triangle containing (x, y).
    Interpolated values are therefore continuous across the triangulation, but
    their first derivatives are discontinuous at edges between triangles.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    trifinder : `~matplotlib.tri.TriFinder`, optional
        If this is not specified, the Triangulation's default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    """
    def __init__(self, triangulation, z, trifinder=None):
        super().__init__(triangulation, z, trifinder)

        # Store plane coefficients for fast interpolation calculations.
        self._plane_coefficients = \
            self._triangulation.calculate_plane_coefficients(self._z)

    def __call__(self, x, y):
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('z',))[0]
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('dzdx', 'dzdy'))
    gradient.__doc__ = TriInterpolator._docstringgradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        if return_key == 'z':
            return (self._plane_coefficients[tri_index, 0]*x +
                    self._plane_coefficients[tri_index, 1]*y +
                    self._plane_coefficients[tri_index, 2])
        elif return_key == 'dzdx':
            return self._plane_coefficients[tri_index, 0]
        else:  # 'dzdy'
            return self._plane_coefficients[tri_index, 1]


class CubicTriInterpolator(TriInterpolator):
    r"""
    Cubic interpolator on a triangular grid.

    In one-dimension - on a segment - a cubic interpolating function is
    defined by the values of the function and its derivative at both ends.
    This is almost the same in 2D inside a triangle, except that the values
    of the function and its 2 derivatives have to be defined at each triangle
    node.

    The CubicTriInterpolator takes the value of the function at each node -
    provided by the user - and internally computes the value of the
    derivatives, resulting in a smooth interpolation.
    (As a special feature, the user can also impose the value of the
    derivatives at each node, but this is not supposed to be the common
    usage.)

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The triangulation to interpolate over.
    z : (npoints,) array-like
        Array of values, defined at grid points, to interpolate between.
    kind : {'min_E', 'geom', 'user'}, optional
        Choice of the smoothing algorithm, in order to compute
        the interpolant derivatives (defaults to 'min_E'):

        - if 'min_E': (default) The derivatives at each node is computed
          to minimize a bending energy.
        - if 'geom': The derivatives at each node is computed as a
          weighted average of relevant triangle normals. To be used for
          speed optimization (large grids).
        - if 'user': The user provides the argument *dz*, no computation
          is hence needed.

    trifinder : `~matplotlib.tri.TriFinder`, optional
        If not specified, the Triangulation's default TriFinder will
        be used by calling `.Triangulation.get_trifinder`.
    dz : tuple of array-likes (dzdx, dzdy), optional
        Used only if  *kind* ='user'. In this case *dz* must be provided as
        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
        are the interpolant first derivatives at the *triangulation* points.

    Methods
    -------
    `__call__` (x, y) : Returns interpolated values at (x, y) points.
    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.

    Notes
    -----
    This note is a bit technical and details how the cubic interpolation is
    computed.

    The interpolation is based on a Clough-Tocher subdivision scheme of
    the *triangulation* mesh (to make it clearer, each triangle of the
    grid will be divided in 3 child-triangles, and on each child triangle
    the interpolated function is a cubic polynomial of the 2 coordinates).
    This technique originates from FEM (Finite Element Method) analysis;
    the element used is a reduced Hsieh-Clough-Tocher (HCT)
    element. Its shape functions are described in [1]_.
    The assembled function is guaranteed to be C1-smooth, i.e. it is
    continuous and its first derivatives are also continuous (this
    is easy to show inside the triangles but is also true when crossing the
    edges).

    In the default case (*kind* ='min_E'), the interpolant minimizes a
    curvature energy on the functional space generated by the HCT element
    shape functions - with imposed values but arbitrary derivatives at each
    node. The minimized functional is the integral of the so-called total
    curvature (implementation based on an algorithm from [2]_ - PCG sparse
    solver):

    .. math::

        E(z) = \frac{1}{2} \int_{\Omega} \left(
            \left( \frac{\partial^2{z}}{\partial{x}^2} \right)^2 +
            \left( \frac{\partial^2{z}}{\partial{y}^2} \right)^2 +
            2\left( \frac{\partial^2{z}}{\partial{y}\partial{x}} \right)^2
        \right) dx\,dy

    If the case *kind* ='geom' is chosen by the user, a simple geometric
    approximation is used (weighted average of the triangle normal
    vectors), which could improve speed on very large grids.

    References
    ----------
    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
        Hsieh-Clough-Tocher triangles, complete or reduced.",
        International Journal for Numerical Methods in Engineering,
        17(5):784 - 789. 2.01.
    .. [2] C.T. Kelley, "Iterative Methods for Optimization".

    """
    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
                 dz=None):
        super().__init__(triangulation, z, trifinder)

        # Loads the underlying c++ _triangulation.
        # (During loading, reordering of triangulation._triangles may occur so
        # that all final triangles are now anti-clockwise)
        self._triangulation.get_cpp_triangulation()

        # To build the stiffness matrix and avoid zero-energy spurious modes
        # we will only store internally the valid (unmasked) triangles and
        # the necessary (used) points coordinates.
        # 2 renumbering tables need to be computed and stored:
        #  - a triangle renum table in order to translate the result from a
        #    TriFinder instance into the internal stored triangle number.
        #  - a node renum table to overwrite the self._z values into the new
        #    (used) node numbering.
        tri_analyzer = TriAnalyzer(self._triangulation)
        (compressed_triangles, compressed_x, compressed_y, tri_renum,
         node_renum) = tri_analyzer._get_compressed_triangulation()
        self._triangles = compressed_triangles
        self._tri_renum = tri_renum
        # Taking into account the node renumbering in self._z:
        valid_node = (node_renum != -1)
        self._z[node_renum[valid_node]] = self._z[valid_node]

        # Computing scale factors
        self._unit_x = np.ptp(compressed_x)
        self._unit_y = np.ptp(compressed_y)
        self._pts = np.column_stack([compressed_x / self._unit_x,
                                     compressed_y / self._unit_y])
        # Computing triangle points
        self._tris_pts = self._pts[self._triangles]
        # Computing eccentricities
        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
        # Computing dof estimations for HCT triangle shape function
        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
        self._dof = self._compute_dof(kind, dz=dz)
        # Loading HCT element
        self._ReferenceElement = _ReducedHCT_Element()

    def __call__(self, x, y):
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('z',))[0]
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('dzdx', 'dzdy'))
    gradient.__doc__ = TriInterpolator._docstringgradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        tris_pts = self._tris_pts[tri_index]
        alpha = self._get_alpha_vec(x, y, tris_pts)
        ecc = self._eccs[tri_index]
        dof = np.expand_dims(self._dof[tri_index], axis=1)
        if return_key == 'z':
            return self._ReferenceElement.get_function_values(
                alpha, ecc, dof)
        else:  # 'dzdx', 'dzdy'
            J = self._get_jacobian(tris_pts)
            dzdx = self._ReferenceElement.get_function_derivatives(
                alpha, J, ecc, dof)
            if return_key == 'dzdx':
                return dzdx[:, 0, 0]
            else:
                return dzdx[:, 1, 0]

    def _compute_dof(self, kind, dz=None):
        """
        Compute and return nodal dofs according to kind.

        Parameters
        ----------
        kind : {'min_E', 'geom', 'user'}
            Choice of the _DOF_estimator subclass to estimate the gradient.
        dz : tuple of array-likes (dzdx, dzdy), optional
            Used only if *kind*=user; in this case passed to the
            :class:`_DOF_estimator_user`.

        Returns
        -------
        array-like, shape (npts, 2)
            Estimation of the gradient at triangulation nodes (stored as
            degree of freedoms of reduced-HCT triangle elements).
        """
        if kind == 'user':
            if dz is None:
                raise ValueError("For a CubicTriInterpolator with "
                                 "*kind*='user', a valid *dz* "
                                 "argument is expected.")
            TE = _DOF_estimator_user(self, dz=dz)
        elif kind == 'geom':
            TE = _DOF_estimator_geom(self)
        else:  # 'min_E', checked in __init__
            TE = _DOF_estimator_min_E(self)
        return TE.compute_dof_from_df()

    @staticmethod
    def _get_alpha_vec(x, y, tris_pts):
        """
        Fast (vectorized) function to compute barycentric coordinates alpha.

        Parameters
        ----------
        x, y : array-like of dim 1 (shape (nx,))
            Coordinates of the points whose points barycentric coordinates are
            requested.
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 2 (shape (nx, 3))
            Barycentric coordinates of the points inside the containing
            triangles.
        """
        ndim = tris_pts.ndim-2

        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        abT = np.stack([a, b], axis=-1)
        ab = _transpose_vectorized(abT)
        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]

        metric = ab @ abT
        # Here we try to deal with the colinear cases.
        # metric_inv is in this case set to the Moore-Penrose pseudo-inverse
        # meaning that we will still return a set of valid barycentric
        # coordinates.
        metric_inv = _pseudo_inv22sym_vectorized(metric)
        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
        ksi = metric_inv @ Covar
        alpha = _to_matrix_vectorized([
            [1-ksi[:, 0, 0]-ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
        return alpha

    @staticmethod
    def _get_jacobian(tris_pts):
        """
        Fast (vectorized) function to compute triangle jacobian matrix.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 3 (shape (nx, 2, 2))
            Barycentric coordinates of the points inside the containing
            triangles.
            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
            itri, so that the following (matrix) relationship holds:
               [dz/dksi] = [J] x [dz/dx]
            with x: global coordinates
                 ksi: element parametric coordinates in triangle first apex
                 local basis.
        """
        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]],
                                   [b[:, 0], b[:, 1]]])
        return J

    @staticmethod
    def _compute_tri_eccentricities(tris_pts):
        """
        Compute triangle eccentricities.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the triangles apexes.

        Returns
        -------
        array like of dim 2 (shape: (nx, 3))
            The so-called eccentricity parameters [1] needed for HCT triangular
            element.
        """
        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
        # Do not use np.squeeze, this is dangerous if only one triangle
        # in the triangulation...
        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
        # Note that this line will raise a warning for dot_a, dot_b or dot_c
        # zeros, but we choose not to support triangles with duplicate points.
        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
                                      [(dot_a-dot_c) / dot_b],
                                      [(dot_b-dot_a) / dot_c]])


# FEM element used for interpolation and for solving minimisation
# problem (Reduced HCT element)
class _ReducedHCT_Element:
    """
    Implementation of reduced HCT triangular element with explicit shape
    functions.

    Computes z, dz, d2z and the element stiffness matrix for bending energy:
    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)

    *** Reference for the shape functions: ***
    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
        reduced.
        Michel Bernadou, Kamal Hassan
        International Journal for Numerical Methods in Engineering.
        17(5):784 - 789.  2.01

    *** Element description: ***
    9 dofs: z and dz given at 3 apex
    C1 (conform)

    """
    # 1) Loads matrices to generate shape functions as a function of
    #    triangle eccentricities - based on [1] p.11 '''
    M = np.array([
        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
    M0 = np.array([
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
    M1 = np.array([
        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
    M2 = np.array([
        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])

    # 2) Loads matrices to rotate components of gradient & Hessian
    #    vectors in the reference basis of triangle first apex (a0)
    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
                          [ 0.,  1.], [-1., -1.],
                          [-1., -1.], [ 1.,  0.]])

    rotate_d2V = np.array([[1., 0., 0.], [0., 1., 0.], [ 0.,  0.,  1.],
                           [0., 1., 0.], [1., 1., 1.], [ 0., -2., -1.],
                           [1., 1., 1.], [1., 0., 0.], [-2.,  0., -1.]])

    # 3) Loads Gauss points & weights on the 3 sub-_triangles for P2
    #    exact integral - 3 points on each subtriangles.
    # NOTE: as the 2nd derivative is discontinuous , we really need those 9
    # points!
    n_gauss = 9
    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
                          [ 4./18., 13./18.,  1./18.],
                          [ 7./18.,  7./18.,  4./18.],
                          [ 1./18., 13./18.,  4./18.],
                          [ 1./18.,  4./18., 13./18.],
                          [ 4./18.,  7./18.,  7./18.],
                          [ 4./18.,  1./18., 13./18.],
                          [13./18.,  1./18.,  4./18.],
                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
    gauss_w = np.ones([9], dtype=np.float64) / 9.

    #  4) Stiffness matrix for curvature energy
    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])

    #  5) Loads the matrix to compute DOF_rot from tri_J at apex 0
    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])

    def get_function_values(self, alpha, ecc, dofs):
        """
        Parameters
        ----------
        alpha : is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates,
        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities,
        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the N-array of interpolated function values.
        """
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        x_sq = x*x
        y_sq = y*y
        z_sq = z*z
        V = _to_matrix_vectorized([
            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
        prod = self.M @ V
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
        s = _roll_vectorized(prod, 3*subtri, axis=0)
        return (dofs @ s)[:, 0, 0]

    def get_function_derivatives(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices of
        barycentric coordinates)
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
        eccentricities)
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
        in global coordinates at locations alpha, as a column-matrices of
        shape (N x 2 x 1).
        """
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        x_sq = x*x
        y_sq = y*y
        z_sq = z*z
        dV = _to_matrix_vectorized([
            [    -3.*x_sq,     -3.*x_sq],
            [     3.*y_sq,           0.],
            [          0.,      3.*z_sq],
            [     -2.*x*z, -2.*x*z+x_sq],
            [-2.*x*y+x_sq,      -2.*x*y],
            [ 2.*x*y-y_sq,        -y_sq],
            [      2.*y*z,         y_sq],
            [        z_sq,       2.*y*z],
            [       -z_sq,  2.*x*z-z_sq],
            [     x*z-y*z,      x*y-y*z]])
        # Puts back dV in first apex basis
        dV = dV @ _extract_submatrices(
            self.rotate_dV, subtri, block_size=2, axis=0)

        prod = self.M @ dV
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
        dfdksi = dofs @ dsdksi
        # In global coordinates:
        # Here we try to deal with the simplest colinear cases, returning a
        # null matrix.
        J_inv = _safe_inv22_vectorized(J)
        dfdx = J_inv @ _transpose_vectorized(dfdksi)
        return dfdx

    def get_function_hessians(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function 2nd-derivatives
        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
        as a column-matrices of shape (N x 3 x 1).
        """
        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
        d2fdksi2 = dofs @ d2sdksi2
        H_rot = self.get_Hrot_from_J(J)
        d2fdx2 = d2fdksi2 @ H_rot
        return _transpose_vectorized(d2fdx2)

    def get_d2Sidksij2(self, alpha, ecc):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
        expressed in covariant coordinates in first apex basis.
        """
        subtri = np.argmin(alpha, axis=1)[:, 0]
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        d2V = _to_matrix_vectorized([
            [     6.*x,      6.*x,      6.*x],
            [     6.*y,        0.,        0.],
            [       0.,      6.*z,        0.],
            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
            [2.*x-4.*y,        0.,     -2.*y],
            [     2.*z,        0.,      2.*y],
            [       0.,      2.*y,      2.*z],
            [       0., 2.*x-4.*z,     -2.*z],
            [    -2.*z,     -2.*y,     x-y-z]])
        # Puts back d2V in first apex basis
        d2V = d2V @ _extract_submatrices(
            self.rotate_d2V, subtri, block_size=3, axis=0)
        prod = self.M @ d2V
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
        return d2sdksi2

    def get_bending_matrices(self, J, ecc):
        """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the element K matrices for bending energy expressed in
        GLOBAL nodal coordinates.
        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
        tri_J is needed to rotate dofs from local basis to global basis
        """
        n = np.size(ecc, 0)

        # 1) matrix to rotate dofs in global coordinates
        J1 = self.J0_to_J1 @ J
        J2 = self.J0_to_J2 @ J
        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
        DOF_rot[:, 0, 0] = 1
        DOF_rot[:, 3, 3] = 1
        DOF_rot[:, 6, 6] = 1
        DOF_rot[:, 1:3, 1:3] = J
        DOF_rot[:, 4:6, 4:6] = J1
        DOF_rot[:, 7:9, 7:9] = J2

        # 2) matrix to rotate Hessian in global coordinates.
        H_rot, area = self.get_Hrot_from_J(J, return_area=True)

        # 3) Computes stiffness matrix
        # Gauss quadrature.
        K = np.zeros([n, 9, 9], dtype=np.float64)
        weights = self.gauss_w
        pts = self.gauss_pts
        for igauss in range(self.n_gauss):
            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
            alpha = np.expand_dims(alpha, 2)
            weight = weights[igauss]
            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
            d2Skdx2 = d2Skdksi2 @ H_rot
            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))

        # 4) With nodal (not elem) dofs
        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot

        # 5) Need the area to compute total element energy
        return _scalar_vectorized(area, K)

    def get_Hrot_from_J(self, J, return_area=False):
        """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)

        Returns
        -------
        Returns H_rot used to rotate Hessian from local basis of first apex,
        to global coordinates.
        if *return_area* is True, returns also the triangle area (0.5*det(J))
        """
        # Here we try to deal with the simplest colinear cases; a null
        # energy and area is imposed.
        J_inv = _safe_inv22_vectorized(J)
        Ji00 = J_inv[:, 0, 0]
        Ji11 = J_inv[:, 1, 1]
        Ji10 = J_inv[:, 1, 0]
        Ji01 = J_inv[:, 0, 1]
        H_rot = _to_matrix_vectorized([
            [Ji00*Ji00, Ji10*Ji10, Ji00*Ji10],
            [Ji01*Ji01, Ji11*Ji11, Ji01*Ji11],
            [2*Ji00*Ji01, 2*Ji11*Ji10, Ji00*Ji11+Ji10*Ji01]])
        if not return_area:
            return H_rot
        else:
            area = 0.5 * (J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0])
            return H_rot, area

    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
        """
        Build K and F for the following elliptic formulation:
        minimization of curvature energy with value of function at node
        imposed and derivatives 'free'.

        Build the global Kff matrix in cco format.
        Build the full Ff vec Ff = - Kfc x Uc.

        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *triangles* is a (N x 3) array of nodes indexes.
        *Uc* is (N x 3) array of imposed displacements at nodes

        Returns
        -------
        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
        (row, col) entries must be summed.
        Ff: force vector - dim npts * 3
        """
        ntri = np.size(ecc, 0)
        vec_range = np.arange(ntri, dtype=np.int32)
        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
        f_dof = [1, 2, 4, 5, 7, 8]
        c_dof = [0, 3, 6]

        # vals, rows and cols indices in global dof numbering
        f_dof_indices = _to_matrix_vectorized([[
            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])

        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
        f_col_indices = expand_indices @ f_dof_indices
        K_elem = self.get_bending_matrices(J, ecc)

        # Extracting sub-matrices
        # Explanation & notations:
        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
        #    (i.e. z at all nodes)
        # * F = [Ff, Fc] is the force vector
        # * U = [Uf, Uc] is the imposed dof vector
        #        [ Kff Kfc ]
        # * K =  [         ]  is the laplacian stiffness matrix
        #        [ Kcf Kff ]
        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc

        # Computing Kff stiffness matrix in sparse coo format
        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])

        # Computing Ff force vector in sparse coo format
        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
        Uc_elem = np.expand_dims(Uc, axis=2)
        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]

        # Extracting Ff force vector in dense format
        # We have to sum duplicate indices -  using bincount
        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
        return Kff_rows, Kff_cols, Kff_vals, Ff


# :class:_DOF_estimator, _DOF_estimator_user, _DOF_estimator_geom,
# _DOF_estimator_min_E
# Private classes used to compute the degree of freedom of each triangular
# element for the TriCubicInterpolator.
class _DOF_estimator:
    """
    Abstract base class for classes used to estimate a function's first
    derivatives, and deduce the dofs for a CubicTriInterpolator using a
    reduced HCT element formulation.

    Derived classes implement ``compute_df(self, **kwargs)``, returning
    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
    gradient coordinates.
    """
    def __init__(self, interpolator, **kwargs):
        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
        self._pts = interpolator._pts
        self._tris_pts = interpolator._tris_pts
        self.z = interpolator._z
        self._triangles = interpolator._triangles
        (self._unit_x, self._unit_y) = (interpolator._unit_x,
                                        interpolator._unit_y)
        self.dz = self.compute_dz(**kwargs)
        self.compute_dof_from_df()

    def compute_dz(self, **kwargs):
        raise NotImplementedError

    def compute_dof_from_df(self):
        """
        Compute reduced-HCT elements degrees of freedom, from the gradient.
        """
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        tri_z = self.z[self._triangles]
        tri_dz = self.dz[self._triangles]
        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
        return tri_dof

    @staticmethod
    def get_dof_vec(tri_z, tri_dz, J):
        """
        Compute the dof vector of a triangle, from the value of f, df and
        of the local Jacobian at each node.

        Parameters
        ----------
        tri_z : shape (3,) array
            f nodal values.
        tri_dz : shape (3, 2) array
            df/dx, df/dy nodal values.
        J
            Jacobian matrix in local basis of apex 0.

        Returns
        -------
        dof : shape (9,) array
            For each apex ``iapex``::

                dof[iapex*3+0] = f(Ai)
                dof[iapex*3+1] = df(Ai).(AiAi+)
                dof[iapex*3+2] = df(Ai).(AiAi-)
        """
        npt = tri_z.shape[0]
        dof = np.zeros([npt, 9], dtype=np.float64)
        J1 = _ReducedHCT_Element.J0_to_J1 @ J
        J2 = _ReducedHCT_Element.J0_to_J2 @ J

        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)

        dfdksi = _to_matrix_vectorized([
            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
        dof[:, 0:7:3] = tri_z
        dof[:, 1:8:3] = dfdksi[:, 0]
        dof[:, 2:9:3] = dfdksi[:, 1]
        return dof


class _DOF_estimator_user(_DOF_estimator):
    """dz is imposed by user; accounts for scaling if any."""

    def compute_dz(self, dz):
        (dzdx, dzdy) = dz
        dzdx = dzdx * self._unit_x
        dzdy = dzdy * self._unit_y
        return np.vstack([dzdx, dzdy]).T


class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""

    def compute_dz(self):
        """
        self.df is computed as weighted average of _triangles sharing a common
        node. On each triangle itri f is first assumed linear (= ~f), which
        allows to compute d~f[itri]
        Then the following approximation of df nodal values is then proposed:
            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
        The weighted coeff. w[itri] are proportional to the angle of the
        triangle itri at apex ipt
        """
        el_geom_w = self.compute_geom_weights()
        el_geom_grad = self.compute_geom_grads()

        # Sum of weights coeffs
        w_node_sum = np.bincount(np.ravel(self._triangles),
                                 weights=np.ravel(el_geom_w))

        # Sum of weighted df = (dfx, dfy)
        dfx_el_w = np.empty_like(el_geom_w)
        dfy_el_w = np.empty_like(el_geom_w)
        for iapex in range(3):
            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
        dfx_node_sum = np.bincount(np.ravel(self._triangles),
                                   weights=np.ravel(dfx_el_w))
        dfy_node_sum = np.bincount(np.ravel(self._triangles),
                                   weights=np.ravel(dfy_el_w))

        # Estimation of df
        dfx_estim = dfx_node_sum/w_node_sum
        dfy_estim = dfy_node_sum/w_node_sum
        return np.vstack([dfx_estim, dfy_estim]).T

    def compute_geom_weights(self):
        """
        Build the (nelems, 3) weights coeffs of _triangles angles,
        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
        """
        weights = np.zeros([np.size(self._triangles, 0), 3])
        tris_pts = self._tris_pts
        for ipt in range(3):
            p0 = tris_pts[:, ipt % 3, :]
            p1 = tris_pts[:, (ipt+1) % 3, :]
            p2 = tris_pts[:, (ipt-1) % 3, :]
            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
            # In the below formula we could take modulo 2. but
            # modulo 1. is safer regarding round-off errors (flat triangles).
            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
            # Weight proportional to angle up np.pi/2; null weight for
            # degenerated cases 0 and np.pi (note that *angle* is normalized
            # by np.pi).
            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
        return weights

    def compute_geom_grads(self):
        """
        Compute the (global) gradient component of f assumed linear (~f).
        returns array df of shape (nelems, 2)
        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
        """
        tris_pts = self._tris_pts
        tris_f = self.z[self._triangles]

        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        dM = np.dstack([dM1, dM2])
        # Here we try to deal with the simplest colinear cases: a null
        # gradient is assumed in this case.
        dM_inv = _safe_inv22_vectorized(dM)

        dZ1 = tris_f[:, 1] - tris_f[:, 0]
        dZ2 = tris_f[:, 2] - tris_f[:, 0]
        dZ = np.vstack([dZ1, dZ2]).T
        df = np.empty_like(dZ)

        # With np.einsum: could be ej,eji -> ej
        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]
        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]
        return df


class _DOF_estimator_min_E(_DOF_estimator_geom):
    """
    The 'smoothest' approximation, df is computed through global minimization
    of the bending energy:
      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
    """
    def __init__(self, Interpolator):
        self._eccs = Interpolator._eccs
        super().__init__(Interpolator)

    def compute_dz(self):
        """
        Elliptic solver for bending energy minimization.
        Uses a dedicated 'toy' sparse Jacobi PCG solver.
        """
        # Initial guess for iterative PCG solver.
        dz_init = super().compute_dz()
        Uf0 = np.ravel(dz_init)

        reference_element = _ReducedHCT_Element()
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        eccs = self._eccs
        triangles = self._triangles
        Uc = self.z[self._triangles]

        # Building stiffness matrix and force vector in coo format
        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
            J, eccs, triangles, Uc)

        # Building sparse matrix and solving minimization problem
        # We could use scipy.sparse direct solver; however to avoid this
        # external dependency an implementation of a simple PCG solver with
        # a simple diagonal Jacobi preconditioner is implemented.
        tol = 1.e-10
        n_dof = Ff.shape[0]
        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
                                     shape=(n_dof, n_dof))
        Kff_coo.compress_csc()
        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
        # If the PCG did not converge, we return the best guess between Uf0
        # and Uf.
        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
        if err0 < err:
            # Maybe a good occasion to raise a warning here ?
            _api.warn_external("In TriCubicInterpolator initialization, "
                               "PCG sparse solver did not converge after "
                               "1000 iterations. `geom` approximation is "
                               "used instead of `min_E`")
            Uf = Uf0

        # Building dz from Uf
        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
        dz[:, 0] = Uf[::2]
        dz[:, 1] = Uf[1::2]
        return dz


# The following private :class:_Sparse_Matrix_coo and :func:_cg provide
# a PCG sparse solver for (symmetric) elliptic problems.
class _Sparse_Matrix_coo:
    def __init__(self, vals, rows, cols, shape):
        """
        Create a sparse matrix in coo format.
        *vals*: arrays of values of non-null entries of the matrix
        *rows*: int arrays of rows of non-null entries of the matrix
        *cols*: int arrays of cols of non-null entries of the matrix
        *shape*: 2-tuple (n, m) of matrix shape
        """
        self.n, self.m = shape
        self.vals = np.asarray(vals, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)

    def dot(self, V):
        """
        Dot product of self by a vector *V* in sparse-dense to dense format
        *V* dense vector of shape (self.m,).
        """
        assert V.shape == (self.m,)
        return np.bincount(self.rows,
                           weights=self.vals*V[self.cols],
                           minlength=self.m)

    def compress_csc(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csc format.
        """
        _, unique, indices = np.unique(
            self.rows + self.n*self.cols,
            return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def compress_csr(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csr format.
        """
        _, unique, indices = np.unique(
            self.m*self.rows + self.cols,
            return_index=True, return_inverse=True)
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def to_dense(self):
        """
        Return a dense matrix representing self, mainly for debugging purposes.
        """
        ret = np.zeros([self.n, self.m], dtype=np.float64)
        nvals = self.vals.size
        for i in range(nvals):
            ret[self.rows[i], self.cols[i]] += self.vals[i]
        return ret

    def __str__(self):
        return self.to_dense().__str__()

    @property
    def diag(self):
        """Return the (dense) vector of the diagonal elements."""
        in_diag = (self.rows == self.cols)
        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
        diag[self.rows[in_diag]] = self.vals[in_diag]
        return diag


def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
    """
    Use Preconditioned Conjugate Gradient iteration to solve A x = b
    A simple Jacobi (diagonal) preconditioner is used.

    Parameters
    ----------
    A : _Sparse_Matrix_coo
        *A* must have been compressed before by compress_csc or
        compress_csr method.
    b : array
        Right hand side of the linear system.
    x0 : array, optional
        Starting guess for the solution. Defaults to the zero vector.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below tol. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after *maxiter*
        steps even if the specified tolerance has not been achieved. Defaults
        to 1000.

    Returns
    -------
    x : array
        The converged solution.
    err : float
        The absolute error np.linalg.norm(A.dot(x) - b)
    """
    n = b.size
    assert A.n == n
    assert A.m == n
    b_norm = np.linalg.norm(b)

    # Jacobi pre-conditioner
    kvec = A.diag
    # For diag elem < 1e-6 we keep 1e-6.
    kvec = np.maximum(kvec, 1e-6)

    # Initial guess
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0

    r = b - A.dot(x)
    w = r/kvec

    p = np.zeros(n)
    beta = 0.0
    rho = np.dot(r, w)
    k = 0

    # Following C. T. Kelley
    while (np.sqrt(abs(rho)) > tol*b_norm) and (k < maxiter):
        p = w + beta*p
        z = A.dot(p)
        alpha = rho/np.dot(p, z)
        r = r - alpha*z
        w = r/kvec
        rhoold = rho
        rho = np.dot(r, w)
        x = x + alpha*p
        beta = rho/rhoold
        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
        k += 1
    err = np.linalg.norm(A.dot(x) - b)
    return x, err


# The following private functions:
#     :func:`_safe_inv22_vectorized`
#     :func:`_pseudo_inv22sym_vectorized`
#     :func:`_scalar_vectorized`
#     :func:`_transpose_vectorized`
#     :func:`_roll_vectorized`
#     :func:`_to_matrix_vectorized`
#     :func:`_extract_submatrices`
# provide fast numpy implementation of some standard operations on arrays of
# matrices - stored as (:, n_rows, n_cols)-shaped np.arrays.

# Development note: Dealing with pathologic 'flat' triangles in the
# CubicTriInterpolator code and impact on (2, 2)-matrix inversion functions
# :func:`_safe_inv22_vectorized` and :func:`_pseudo_inv22sym_vectorized`.
#
# Goals:
# 1) The CubicTriInterpolator should be able to handle flat or almost flat
#    triangles without raising an error,
# 2) These degenerated triangles should have no impact on the automatic dof
#    calculation (associated with null weight for the _DOF_estimator_geom and
#    with null energy for the _DOF_estimator_min_E),
# 3) Linear patch test should be passed exactly on degenerated meshes,
# 4) Interpolation (with :meth:`_interpolate_single_key` or
#    :meth:`_interpolate_multi_key`) shall be correctly handled even *inside*
#    the pathologic triangles, to interact correctly with a TriRefiner class.
#
# Difficulties:
# Flat triangles have rank-deficient *J* (so-called jacobian matrix) and
# *metric* (the metric tensor = J x J.T). Computation of the local
# tangent plane is also problematic.
#
# Implementation:
# Most of the time, when computing the inverse of a rank-deficient matrix it
# is safe to simply return the null matrix (which is the implementation in
# :func:`_safe_inv22_vectorized`). This is because of point 2), itself
# enforced by:
#    - null area hence null energy in :class:`_DOF_estimator_min_E`
#    - angles close or equal to 0 or np.pi hence null weight in
#      :class:`_DOF_estimator_geom`.
#      Note that the function angle -> weight is continuous and maximum for an
#      angle np.pi/2 (refer to :meth:`compute_geom_weights`)
# The exception is the computation of barycentric coordinates, which is done
# by inversion of the *metric* matrix. In this case, we need to compute a set
# of valid coordinates (1 among numerous possibilities), to ensure point 4).
# We benefit here from the symmetry of metric = J x J.T, which makes it easier
# to compute a pseudo-inverse in :func:`_pseudo_inv22sym_vectorized`
def _safe_inv22_vectorized(M):
    """
    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
    matrices.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0]*M[:, 1, 1]
    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]

    # We set delta_inv to 0. in case of a rank deficient matrix; a
    # rank-deficient input matrix *M* will lead to a null matrix in output
    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
    if np.all(rank2):
        # Normal 'optimized' flow.
        delta_inv = 1./delta
    else:
        # 'Pathologic' flow.
        delta_inv = np.zeros(M.shape[0])
        delta_inv[rank2] = 1./delta[rank2]

    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
    return M_inv


def _pseudo_inv22sym_vectorized(M):
    """
    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.

    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
    In case M is of rank 0, we return the null matrix.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0]*M[:, 1, 1]
    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))

    if np.all(rank2):
        # Normal 'optimized' flow.
        M_inv[:, 0, 0] = M[:, 1, 1] / delta
        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
        M_inv[:, 1, 1] = M[:, 0, 0] / delta
    else:
        # 'Pathologic' flow.
        # Here we have to deal with 2 sub-cases
        # 1) First sub-case: matrices of rank 2:
        delta = delta[rank2]
        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
        rank01 = ~rank2
        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
        tr_zeros = (np.abs(tr) < 1.e-8)
        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
        # sq_tr_inv = 1. / tr**2
        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv

    return M_inv


def _scalar_vectorized(scalar, M):
    """
    Scalar product between scalars and matrices.
    """
    return scalar[:, np.newaxis, np.newaxis]*M


def _transpose_vectorized(M):
    """
    Transposition of an array of matrices *M*.
    """
    return np.transpose(M, [0, 2, 1])


def _roll_vectorized(M, roll_indices, axis):
    """
    Roll an array of matrices along *axis* (0: rows, 1: columns) according to
    an array of indices *roll_indices*.
    """
    assert axis in [0, 1]
    ndim = M.ndim
    assert ndim == 3
    ndim_roll = roll_indices.ndim
    assert ndim_roll == 1
    sh = M.shape
    r, c = sh[-2:]
    assert sh[0] == roll_indices.shape[0]
    vec_indices = np.arange(sh[0], dtype=np.int32)

    # Builds the rolled matrix
    M_roll = np.empty_like(M)
    if axis == 0:
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
    else:  # 1
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
    return M_roll


def _to_matrix_vectorized(M):
    """
    Build an array of matrices from individuals np.arrays of identical shapes.

    Parameters
    ----------
    M
        ncols-list of nrows-lists of shape sh.

    Returns
    -------
    M_res : np.array of shape (sh, nrow, ncols)
        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.
    """
    assert isinstance(M, (tuple, list))
    assert all(isinstance(item, (tuple, list)) for item in M)
    c_vec = np.asarray([len(item) for item in M])
    assert np.all(c_vec-c_vec[0] == 0)
    r = len(M)
    c = c_vec[0]
    M00 = np.asarray(M[0][0])
    dt = M00.dtype
    sh = [M00.shape[0], r, c]
    M_ret = np.empty(sh, dtype=dt)
    for irow in range(r):
        for icol in range(c):
            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
    return M_ret


def _extract_submatrices(M, block_indices, block_size, axis):
    """
    Extract selected blocks of a matrices *M* depending on parameters
    *block_indices* and *block_size*.

    Returns the array of extracted matrices *Mres* so that ::

        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]
    """
    assert block_indices.ndim == 1
    assert axis in [0, 1]

    r, c = M.shape
    if axis == 0:
        sh = [block_indices.shape[0], block_size, c]
    else:  # 1
        sh = [block_indices.shape[0], r, block_size]

    dt = M.dtype
    M_res = np.empty(sh, dtype=dt)
    if axis == 0:
        for ir in range(block_size):
            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
    else:  # 1
        for ic in range(block_size):
            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]

    return M_res
<|endoftext|>import numpy as np
from matplotlib.tri._triangulation import Triangulation
import matplotlib.cbook as cbook
import matplotlib.lines as mlines


def triplot(ax, *args, **kwargs):
    """
    Draw an unstructured triangular grid as lines and/or markers.

    Call signatures::

      triplot(triangulation, ...)
      triplot(x, y, [triangles], *, [mask=mask], ...)

    The triangular grid can be specified either by passing a `.Triangulation`
    object as the first parameter, or by passing the points *x*, *y* and
    optionally the *triangles* and a *mask*. If neither of *triangulation* or
    *triangles* are given, the triangulation is calculated on the fly.

    Parameters
    ----------
    triangulation : `.Triangulation`
        An already created triangular grid.
    x, y, triangles, mask
        Parameters defining the triangular grid. See `.Triangulation`.
        This is mutually exclusive with specifying *triangulation*.
    other_parameters
        All other args and kwargs are forwarded to `~.Axes.plot`.

    Returns
    -------
    lines : `~matplotlib.lines.Line2D`
        The drawn triangles edges.
    markers : `~matplotlib.lines.Line2D`
        The drawn marker nodes.
    """
    import matplotlib.axes

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    x, y, edges = (tri.x, tri.y, tri.edges)

    # Decode plot format string, e.g., 'ro-'
    fmt = args[0] if args else ""
    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)

    # Insert plot format string into a copy of kwargs (kwargs values prevail).
    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    for key, val in zip(('linestyle', 'marker', 'color'),
                        (linestyle, marker, color)):
        if val is not None:
            kw.setdefault(key, val)

    # Draw lines without markers.
    # Note 1: If we drew markers here, most markers would be drawn more than
    #         once as they belong to several edges.
    # Note 2: We insert nan values in the flattened edges arrays rather than
    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
    #         as it considerably speeds-up code execution.
    linestyle = kw['linestyle']
    kw_lines = {
        **kw,
        'marker': 'None',  # No marker to draw.
        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
    }
    if linestyle not in [None, 'None', '', ' ']:
        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
                            **kw_lines)
    else:
        tri_lines = ax.plot([], [], **kw_lines)

    # Draw markers separately.
    marker = kw['marker']
    kw_markers = {
        **kw,
        'linestyle': 'None',  # No line to draw.
    }
    kw_markers.pop('label', None)
    if marker not in [None, 'None', '', ' ']:
        tri_markers = ax.plot(x, y, **kw_markers)
    else:
        tri_markers = ax.plot([], [], **kw_markers)

    return tri_lines + tri_markers
<|endoftext|>"""
Unstructured triangular grid functions.
"""

from ._triangulation import Triangulation
from ._tricontour import TriContourSet, tricontour, tricontourf
from ._trifinder import TriFinder, TrapezoidMapTriFinder
from ._triinterpolate import (TriInterpolator, LinearTriInterpolator,
                              CubicTriInterpolator)
from ._tripcolor import tripcolor
from ._triplot import triplot
from ._trirefine import TriRefiner, UniformTriRefiner
from ._tritools import TriAnalyzer


__all__ = ["Triangulation",
           "TriContourSet", "tricontour", "tricontourf",
           "TriFinder", "TrapezoidMapTriFinder",
           "TriInterpolator", "LinearTriInterpolator", "CubicTriInterpolator",
           "tripcolor",
           "triplot",
           "TriRefiner", "UniformTriRefiner",
           "TriAnalyzer"]
<|endoftext|>import sys

import numpy as np

from matplotlib import _api


class Triangulation:
    """
    An unstructured triangular grid consisting of npoints points and
    ntri triangles.  The triangles can either be specified by the user
    or automatically generated using a Delaunay triangulation.

    Parameters
    ----------
    x, y : (npoints,) array-like
        Coordinates of grid points.
    triangles : (ntri, 3) array-like of int, optional
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.  If not
        specified, the Delaunay triangulation is calculated.
    mask : (ntri,) array-like of bool, optional
        Which triangles are masked out.

    Attributes
    ----------
    triangles : (ntri, 3) array of int
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner. If you want to
        take the *mask* into account, use `get_masked_triangles` instead.
    mask : (ntri, 3) array of bool or None
        Masked out triangles.
    is_delaunay : bool
        Whether the Triangulation is a calculated Delaunay
        triangulation (where *triangles* was not specified) or not.

    Notes
    -----
    For a Triangulation to be valid it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.
    """
    def __init__(self, x, y, triangles=None, mask=None):
        from matplotlib import _qhull

        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError("x and y must be equal-length 1D arrays, but "
                             f"found shapes {self.x.shape!r} and "
                             f"{self.y.shape!r}")

        self.mask = None
        self._edges = None
        self._neighbors = None
        self.is_delaunay = False

        if triangles is None:
            # No triangulation specified, so use matplotlib._qhull to obtain
            # Delaunay triangulation.
            self.triangles, self._neighbors = _qhull.delaunay(x, y, sys.flags.verbose)
            self.is_delaunay = True
        else:
            # Triangulation specified. Copy, since we may correct triangle
            # orientation.
            try:
                self.triangles = np.array(triangles, dtype=np.int32, order='C')
            except ValueError as e:
                raise ValueError('triangles must be a (N, 3) int array, not '
                                 f'{triangles!r}') from e
            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
                raise ValueError(
                    'triangles must be a (N, 3) int array, but found shape '
                    f'{self.triangles.shape!r}')
            if self.triangles.max() >= len(self.x):
                raise ValueError(
                    'triangles are indices into the points and must be in the '
                    f'range 0 <= i < {len(self.x)} but found value '
                    f'{self.triangles.max()}')
            if self.triangles.min() < 0:
                raise ValueError(
                    'triangles are indices into the points and must be in the '
                    f'range 0 <= i < {len(self.x)} but found value '
                    f'{self.triangles.min()}')

        # Underlying C++ object is not created until first needed.
        self._cpp_triangulation = None

        # Default TriFinder not created until needed.
        self._trifinder = None

        self.set_mask(mask)

    def calculate_plane_coefficients(self, z):
        """
        Calculate plane equation coefficients for all unmasked triangles from
        the point (x, y) coordinates and specified z-array of shape (npoints).
        The returned array has shape (npoints, 3) and allows z-value at (x, y)
        position in triangle tri to be calculated using
        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
        """
        return self.get_cpp_triangulation().calculate_plane_coefficients(z)

    @property
    def edges(self):
        """
        Return integer array of shape (nedges, 2) containing all edges of
        non-masked triangles.

        Each row defines an edge by its start point index and end point
        index.  Each edge appears only once, i.e. for an edge between points
        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
        """
        if self._edges is None:
            self._edges = self.get_cpp_triangulation().get_edges()
        return self._edges

    def get_cpp_triangulation(self):
        """
        Return the underlying C++ Triangulation object, creating it
        if necessary.
        """
        from matplotlib import _tri
        if self._cpp_triangulation is None:
            self._cpp_triangulation = _tri.Triangulation(
                # For unset arrays use empty tuple which has size of zero.
                self.x, self.y, self.triangles,
                self.mask if self.mask is not None else (),
                self._edges if self._edges is not None else (),
                self._neighbors if self._neighbors is not None else (),
                not self.is_delaunay)
        return self._cpp_triangulation

    def get_masked_triangles(self):
        """
        Return an array of triangles taking the mask into account.
        """
        if self.mask is not None:
            return self.triangles[~self.mask]
        else:
            return self.triangles

    @staticmethod
    def get_from_args_and_kwargs(*args, **kwargs):
        """
        Return a Triangulation object from the args and kwargs, and
        the remaining args and kwargs with the consumed values removed.

        There are two alternatives: either the first argument is a
        Triangulation object, in which case it is returned, or the args
        and kwargs are sufficient to create a new Triangulation to
        return.  In the latter case, see Triangulation.__init__ for
        the possible args and kwargs.
        """
        if isinstance(args[0], Triangulation):
            triangulation, *args = args
            if 'triangles' in kwargs:
                _api.warn_external(
                    "Passing the keyword 'triangles' has no effect when also "
                    "passing a Triangulation")
            if 'mask' in kwargs:
                _api.warn_external(
                    "Passing the keyword 'mask' has no effect when also "
                    "passing a Triangulation")
        else:
            x, y, triangles, mask, args, kwargs = \
                Triangulation._extract_triangulation_params(args, kwargs)
            triangulation = Triangulation(x, y, triangles, mask)
        return triangulation, args, kwargs

    @staticmethod
    def _extract_triangulation_params(args, kwargs):
        x, y, *args = args
        # Check triangles in kwargs then args.
        triangles = kwargs.pop('triangles', None)
        from_args = False
        if triangles is None and args:
            triangles = args[0]
            from_args = True
        if triangles is not None:
            try:
                triangles = np.asarray(triangles, dtype=np.int32)
            except ValueError:
                triangles = None
        if triangles is not None and (triangles.ndim != 2 or
                                      triangles.shape[1] != 3):
            triangles = None
        if triangles is not None and from_args:
            args = args[1:]  # Consumed first item in args.
        # Check for mask in kwargs.
        mask = kwargs.pop('mask', None)
        return x, y, triangles, mask, args, kwargs

    def get_trifinder(self):
        """
        Return the default `matplotlib.tri.TriFinder` of this
        triangulation, creating it if necessary.  This allows the same
        TriFinder object to be easily shared.
        """
        if self._trifinder is None:
            # Default TriFinder class.
            from matplotlib.tri._trifinder import TrapezoidMapTriFinder
            self._trifinder = TrapezoidMapTriFinder(self)
        return self._trifinder

    @property
    def neighbors(self):
        """
        Return integer array of shape (ntri, 3) containing neighbor triangles.

        For each triangle, the indices of the three triangles that
        share the same edges, or -1 if there is no such neighboring
        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
        to the edge from point index ``triangles[i, j]`` to point index
        ``triangles[i, (j+1)%3]``.
        """
        if self._neighbors is None:
            self._neighbors = self.get_cpp_triangulation().get_neighbors()
        return self._neighbors

    def set_mask(self, mask):
        """
        Set or clear the mask array.

        Parameters
        ----------
        mask : None or bool array of length ntri
        """
        if mask is None:
            self.mask = None
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.shape != (self.triangles.shape[0],):
                raise ValueError('mask array must have same length as '
                                 'triangles array')

        # Set mask in C++ Triangulation.
        if self._cpp_triangulation is not None:
            self._cpp_triangulation.set_mask(
                self.mask if self.mask is not None else ())

        # Clear derived fields so they are recalculated when needed.
        self._edges = None
        self._neighbors = None

        # Recalculate TriFinder if it exists.
        if self._trifinder is not None:
            self._trifinder._initialize()
<|endoftext|>"""
Tools for triangular grids.
"""

import numpy as np

from matplotlib import _api
from matplotlib.tri import Triangulation


class TriAnalyzer:
    """
    Define basic tools for triangular mesh analysis and improvement.

    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
    tools for mesh analysis and mesh improvement.

    Attributes
    ----------
    scale_factors

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation to analyze.
    """

    def __init__(self, triangulation):
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

    @property
    def scale_factors(self):
        """
        Factors to rescale the triangulation into a unit square.

        Returns
        -------
        (float, float)
            Scaling factors (kx, ky) so that the triangulation
            ``[triangulation.x * kx, triangulation.y * ky]``
            fits exactly inside a unit square.
        """
        compressed_triangles = self._triangulation.get_masked_triangles()
        node_used = (np.bincount(np.ravel(compressed_triangles),
                                 minlength=self._triangulation.x.size) != 0)
        return (1 / np.ptp(self._triangulation.x[node_used]),
                1 / np.ptp(self._triangulation.y[node_used]))

    def circle_ratios(self, rescale=True):
        """
        Return a measure of the triangulation triangles flatness.

        The ratio of the incircle radius over the circumcircle radius is a
        widely used indicator of a triangle flatness.
        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
        triangles. Circle ratios below 0.01 denote very flat triangles.

        To avoid unduly low values due to a difference of scale between the 2
        axis, the triangular mesh can first be rescaled to fit inside a unit
        square with `scale_factors` (Only if *rescale* is True, which is
        its default value).

        Parameters
        ----------
        rescale : bool, default: True
            If True, internally rescale (based on `scale_factors`), so that the
            (unmasked) triangles fit exactly inside a unit square mesh.

        Returns
        -------
        masked array
            Ratio of the incircle radius over the circumcircle radius, for
            each 'rescaled' triangle of the encapsulated triangulation.
            Values corresponding to masked triangles are masked out.

        """
        # Coords rescaling
        if rescale:
            (kx, ky) = self.scale_factors
        else:
            (kx, ky) = (1.0, 1.0)
        pts = np.vstack([self._triangulation.x*kx,
                         self._triangulation.y*ky]).T
        tri_pts = pts[self._triangulation.triangles]
        # Computes the 3 side lengths
        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
        a = np.hypot(a[:, 0], a[:, 1])
        b = np.hypot(b[:, 0], b[:, 1])
        c = np.hypot(c[:, 0], c[:, 1])
        # circumcircle and incircle radii
        s = (a+b+c)*0.5
        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
        # We have to deal with flat triangles with infinite circum_radius
        bool_flat = (prod == 0.)
        if np.any(bool_flat):
            # Pathologic flow
            ntri = tri_pts.shape[0]
            circum_radius = np.empty(ntri, dtype=np.float64)
            circum_radius[bool_flat] = np.inf
            abc = a*b*c
            circum_radius[~bool_flat] = abc[~bool_flat] / (
                4.0*np.sqrt(prod[~bool_flat]))
        else:
            # Normal optimized flow
            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
        in_radius = (a*b*c) / (4.0*circum_radius*s)
        circle_ratio = in_radius/circum_radius
        mask = self._triangulation.mask
        if mask is None:
            return circle_ratio
        else:
            return np.ma.array(circle_ratio, mask=mask)

    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
        """
        Eliminate excessively flat border triangles from the triangulation.

        Returns a mask *new_mask* which allows to clean the encapsulated
        triangulation from its border-located flat triangles
        (according to their :meth:`circle_ratios`).
        This mask is meant to be subsequently applied to the triangulation
        using `.Triangulation.set_mask`.
        *new_mask* is an extension of the initial triangulation mask
        in the sense that an initially masked triangle will remain masked.

        The *new_mask* array is computed recursively; at each step flat
        triangles are removed only if they share a side with the current mesh
        border. Thus, no new holes in the triangulated domain will be created.

        Parameters
        ----------
        min_circle_ratio : float, default: 0.01
            Border triangles with incircle/circumcircle radii ratio r/R will
            be removed if r/R < *min_circle_ratio*.
        rescale : bool, default: True
            If True, first, internally rescale (based on `scale_factors`) so
            that the (unmasked) triangles fit exactly inside a unit square
            mesh.  This rescaling accounts for the difference of scale which
            might exist between the 2 axis.

        Returns
        -------
        array of bool
            Mask to apply to encapsulated triangulation.
            All the initially masked triangles remain masked in the
            *new_mask*.

        Notes
        -----
        The rationale behind this function is that a Delaunay
        triangulation - of an unstructured set of points - sometimes contains
        almost flat triangles at its border, leading to artifacts in plots
        (especially for high-resolution contouring).
        Masked with computed *new_mask*, the encapsulated
        triangulation would contain no more unmasked border triangles
        with a circle ratio below *min_circle_ratio*, thus improving the
        mesh quality for subsequent plots or interpolation.
        """
        # Recursively computes the mask_current_borders, true if a triangle is
        # at the border of the mesh OR touching the border through a chain of
        # invalid aspect ratio masked_triangles.
        ntri = self._triangulation.triangles.shape[0]
        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio

        current_mask = self._triangulation.mask
        if current_mask is None:
            current_mask = np.zeros(ntri, dtype=bool)
        valid_neighbors = np.copy(self._triangulation.neighbors)
        renum_neighbors = np.arange(ntri, dtype=np.int32)
        nadd = -1
        while nadd != 0:
            # The active wavefront is the triangles from the border (unmasked
            # but with a least 1 neighbor equal to -1
            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
            # The element from the active wavefront will be masked if their
            # circle ratio is bad.
            added_mask = wavefront & mask_bad_ratio
            current_mask = added_mask | current_mask
            nadd = np.sum(added_mask)

            # now we have to update the tables valid_neighbors
            valid_neighbors[added_mask, :] = -1
            renum_neighbors[added_mask] = -1
            valid_neighbors = np.where(valid_neighbors == -1, -1,
                                       renum_neighbors[valid_neighbors])

        return np.ma.filled(current_mask, True)

    def _get_compressed_triangulation(self):
        """
        Compress (if masked) the encapsulated triangulation.

        Returns minimal-length triangles array (*compressed_triangles*) and
        coordinates arrays (*compressed_x*, *compressed_y*) that can still
        describe the unmasked triangles of the encapsulated triangulation.

        Returns
        -------
        compressed_triangles : array-like
            the returned compressed triangulation triangles
        compressed_x : array-like
            the returned compressed triangulation 1st coordinate
        compressed_y : array-like
            the returned compressed triangulation 2nd coordinate
        tri_renum : int array
            renumbering table to translate the triangle numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for masked triangles (deleted from *compressed_triangles*).
        node_renum : int array
            renumbering table to translate the point numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for unused points (i.e. those deleted from *compressed_x* and
            *compressed_y*).

        """
        # Valid triangles and renumbering
        tri_mask = self._triangulation.mask
        compressed_triangles = self._triangulation.get_masked_triangles()
        ntri = self._triangulation.triangles.shape[0]
        if tri_mask is not None:
            tri_renum = self._total_to_compress_renum(~tri_mask)
        else:
            tri_renum = np.arange(ntri, dtype=np.int32)

        # Valid nodes and renumbering
        valid_node = (np.bincount(np.ravel(compressed_triangles),
                                  minlength=self._triangulation.x.size) != 0)
        compressed_x = self._triangulation.x[valid_node]
        compressed_y = self._triangulation.y[valid_node]
        node_renum = self._total_to_compress_renum(valid_node)

        # Now renumbering the valid triangles nodes
        compressed_triangles = node_renum[compressed_triangles]

        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
                node_renum)

    @staticmethod
    def _total_to_compress_renum(valid):
        """
        Parameters
        ----------
        valid : 1D bool array
            Validity mask.

        Returns
        -------
        int array
            Array so that (`valid_array` being a compressed array
            based on a `masked_array` with mask ~*valid*):

            - For all i with valid[i] = True:
              valid_array[renum[i]] = masked_array[i]
            - For all i with valid[i] = False:
              renum[i] = -1 (invalid value)
        """
        renum = np.full(np.size(valid), -1, dtype=np.int32)
        n_valid = np.sum(valid)
        renum[valid] = np.arange(n_valid, dtype=np.int32)
        return renum
<|endoftext|>import numpy as np

from matplotlib import _docstring
from matplotlib.contour import ContourSet
from matplotlib.tri._triangulation import Triangulation


@_docstring.interpd
class TriContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions for
    a triangular grid.

    This class is typically not instantiated directly by the user but by
    `~.Axes.tricontour` and `~.Axes.tricontourf`.

    %(contour_set_attributes)s
    """
    def __init__(self, ax, *args, **kwargs):
        """
        Draw triangular grid contour lines or filled regions,
        depending on whether keyword arg *filled* is False
        (default) or True.

        The first argument of the initializer must be an `~.axes.Axes`
        object.  The remaining arguments and keyword arguments
        are described in the docstring of `~.Axes.tricontour`.
        """
        super().__init__(ax, *args, **kwargs)

    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], TriContourSet):
            C = args[0]._contour_generator
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            from matplotlib import _tri
            tri, z = self._contour_args(args, kwargs)
            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
            self._mins = [tri.x.min(), tri.y.min()]
            self._maxs = [tri.x.max(), tri.y.max()]

        self._contour_generator = C
        return kwargs

    def _contour_args(self, args, kwargs):
        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
                                                                   **kwargs)
        z, *args = args
        z = np.ma.asarray(z)
        if z.shape != tri.x.shape:
            raise ValueError('z array must have same length as triangulation x'
                             ' and y arrays')

        # z values must be finite, only need to check points that are included
        # in the triangulation.
        z_check = z[np.unique(tri.get_masked_triangles())]
        if np.ma.is_masked(z_check):
            raise ValueError('z must not contain masked points within the '
                             'triangulation')
        if not np.isfinite(z_check).all():
            raise ValueError('z array must not contain non-finite values '
                             'within the triangulation')

        z = np.ma.masked_invalid(z, copy=False)
        self.zmax = float(z_check.max())
        self.zmin = float(z_check.min())
        if self.logscale and self.zmin <= 0:
            func = 'contourf' if self.filled else 'contour'
            raise ValueError(f'Cannot {func} log of negative values.')
        self._process_contour_level_args(args, z.dtype)
        return (tri, z)


_docstring.interpd.register(_tricontour_doc="""
Draw contour %%(type)s on an unstructured triangular grid.

Call signatures::

    %%(func)s(triangulation, z, [levels], ...)
    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)

The triangular grid can be specified either by passing a `.Triangulation`
object as the first parameter, or by passing the points *x*, *y* and
optionally the *triangles* and a *mask*. See `.Triangulation` for an
explanation of these parameters. If neither of *triangulation* or
*triangles* are given, the triangulation is calculated on the fly.

It is possible to pass *triangles* positionally, i.e.
``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
clarity, pass *triangles* via keyword argument.

Parameters
----------
triangulation : `.Triangulation`, optional
    An already created triangular grid.

x, y, triangles, mask
    Parameters defining the triangular grid. See `.Triangulation`.
    This is mutually exclusive with specifying *triangulation*.

z : array-like
    The height values over which the contour is drawn.  Color-mapping is
    controlled by *cmap*, *norm*, *vmin*, and *vmax*.

    .. note::
        All values in *z* must be finite. Hence, nan and inf values must
        either be removed or `~.Triangulation.set_mask` be used.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
    automatically choose no more than *n+1* "nice" contour levels between
    between minimum and maximum numeric values of *Z*.

    If array-like, draw contour lines at the specified levels.  The values must
    be in increasing order.

Returns
-------
`~matplotlib.tri.TriContourSet`

Other Parameters
----------------
colors : :mpltype:`color` or list of :mpltype:`color`, optional
    The colors of the levels, i.e., the contour %%(type)s.

    The sequence is cycled for the levels in ascending order. If the sequence
    is shorter than the number of levels, it is repeated.

    As a shortcut, single color strings may be used in place of one-element
    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
    same color. This shortcut does only work for color strings, not for other
    ways of specifying colors.

    By default (value *None*), the colormap specified by *cmap* will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

%(cmap_doc)s

    This parameter is ignored if *colors* is set.

%(norm_doc)s

    This parameter is ignored if *colors* is set.

%(vmin_vmax_doc)s

    If *vmin* or *vmax* are not given, the default color scaling is based on
    *levels*.

    This parameter is ignored if *colors* is set.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *z* by specifying the
    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.

    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
    - 'image': Use the value from :rc:`image.origin`.

extent : (x0, x1, y0, y1), optional
    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
    gives the outer pixel boundaries. In this case, the position of z[0, 0] is
    the center of the pixel, not a corner. If *origin* is *None*, then
    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position
    of z[-1, -1].

    This argument is ignored if *X* and *Y* are specified in the call to
    contour.

locator : ticker.Locator subclass, optional
    The locator is used to determine the contour levels if they are not given
    explicitly via *levels*.
    Defaults to `~.ticker.MaxNLocator`.

