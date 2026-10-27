                                           axlim_clip=axlim_clip, **kwargs)
            # average over the three points of each triangle
            avg_z = verts[:, :, 2].mean(axis=1)
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            polyc = art3d.Poly3DCollection(
                verts, *args, shade=shade, lightsource=lightsource,
                facecolors=color, axlim_clip=axlim_clip, **kwargs)

        self.add_collection(polyc)
        self.auto_scale_xyz(tri.x, tri.y, z, had_data)

        return polyc

    def _3d_extend_contour(self, cset, stride=5):
        """
        Extend a contour in 3D by creating
        """

        dz = (cset.levels[1] - cset.levels[0]) / 2
        polyverts = []
        colors = []
        for idx, level in enumerate(cset.levels):
            path = cset.get_paths()[idx]
            subpaths = [*path._iter_connected_components()]
            color = cset.get_edgecolor()[idx]
            top = art3d._paths_to_3d_segments(subpaths, level - dz)
            bot = art3d._paths_to_3d_segments(subpaths, level + dz)
            if not len(top[0]):
                continue
            nsteps = max(round(len(top[0]) / stride), 2)
            stepsize = (len(top[0]) - 1) / (nsteps - 1)
            polyverts.extend([
                (top[0][round(i * stepsize)], top[0][round((i + 1) * stepsize)],
                 bot[0][round((i + 1) * stepsize)], bot[0][round(i * stepsize)])
                for i in range(round(nsteps) - 1)])
            colors.extend([color] * (round(nsteps) - 1))
        self.add_collection3d(art3d.Poly3DCollection(
            np.array(polyverts),  # All polygons have 4 vertices, so vectorize.
            facecolors=colors, edgecolors=colors, shade=True))
        cset.remove()

    def add_contour_set(
            self, cset, extend3d=False, stride=5, zdir='z', offset=None,
            axlim_clip=False):
        zdir = '-' + zdir
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            art3d.collection_2d_to_3d(
                cset, zs=offset if offset is not None else cset.levels, zdir=zdir,
                axlim_clip=axlim_clip)

    def add_contourf_set(self, cset, zdir='z', offset=None, *, axlim_clip=False):
        self._add_contourf_set(cset, zdir=zdir, offset=offset,
                               axlim_clip=axlim_clip)

    def _add_contourf_set(self, cset, zdir='z', offset=None, axlim_clip=False):
        """
        Returns
        -------
        levels : `numpy.ndarray`
            Levels at which the filled contours are added.
        """
        zdir = '-' + zdir

        midpoints = cset.levels[:-1] + np.diff(cset.levels) / 2
        # Linearly interpolate to get levels for any extensions
        if cset._extend_min:
            min_level = cset.levels[0] - np.diff(cset.levels[:2]) / 2
            midpoints = np.insert(midpoints, 0, min_level)
        if cset._extend_max:
            max_level = cset.levels[-1] + np.diff(cset.levels[-2:]) / 2
            midpoints = np.append(midpoints, max_level)

        art3d.collection_2d_to_3d(
            cset, zs=offset if offset is not None else midpoints, zdir=zdir,
            axlim_clip=axlim_clip)
        return midpoints

    @_preprocess_data()
    def contour(self, X, Y, Z, *args,
                extend3d=False, stride=5, zdir='z', offset=None, axlim_clip=False,
                **kwargs):
        """
        Create a 3D contour plot.

        Parameters
        ----------
        X, Y, Z : array-like,
            Input data. See `.Axes.contour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int, default: 5
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        axlim_clip : bool, default: False
            Whether to hide lines with a vertex outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contour`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contour(jX, jY, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset, axlim_clip)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contour3D = contour

    @_preprocess_data()
    def tricontour(self, *args,
                   extend3d=False, stride=5, zdir='z', offset=None, axlim_clip=False,
                   **kwargs):
        """
        Create a 3D contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int, default: 5
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        axlim_clip : bool, default: False
            Whether to hide lines with a vertex outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.

        Returns
        -------
        matplotlib.tri._tricontour.TriContourSet
        """
        had_data = self.has_data()

        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            # We do this so Z doesn't get passed as an arg to Axes.tricontour
            Z, *args = args

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)

        cset = super().tricontour(tri, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset, axlim_clip)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    def _auto_scale_contourf(self, X, Y, Z, zdir, levels, had_data):
        # Autoscale in the zdir based on the levels added, which are
        # different from data range if any contour extensions are present
        dim_vals = {'x': X, 'y': Y, 'z': Z, zdir: levels}
        # Input data and levels have different sizes, but auto_scale_xyz
        # expected same-size input, so manually take min/max limits
        limits = [(np.nanmin(dim_vals[dim]), np.nanmax(dim_vals[dim]))
                  for dim in ['x', 'y', 'z']]
        self.auto_scale_xyz(*limits, had_data)

    @_preprocess_data()
    def contourf(self, X, Y, Z, *args,
                 zdir='z', offset=None, axlim_clip=False, **kwargs):
        """
        Create a 3D filled contour plot.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.contourf` for supported data shapes.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        axlim_clip : bool, default: False
            Whether to hide lines with a vertex outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contourf`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contourf(jX, jY, jZ, *args, **kwargs)
        levels = self._add_contourf_set(cset, zdir, offset, axlim_clip)

        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        return cset

    contourf3D = contourf

    @_preprocess_data()
    def tricontourf(self, *args, zdir='z', offset=None, axlim_clip=False, **kwargs):
        """
        Create a 3D filled contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontourf` for supported data shapes.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir.
        axlim_clip : bool, default: False
            Whether to hide lines with a vertex outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to
            `matplotlib.axes.Axes.tricontourf`.

        Returns
        -------
        matplotlib.tri._tricontour.TriContourSet
        """
        had_data = self.has_data()

        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            # We do this so Z doesn't get passed as an arg to Axes.tricontourf
            Z, *args = args

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)

        cset = super().tricontourf(tri, jZ, *args, **kwargs)
        levels = self._add_contourf_set(cset, zdir, offset, axlim_clip)

        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        return cset

    def add_collection3d(self, col, zs=0, zdir='z', autolim=True, *,
                         axlim_clip=False):
        """
        Add a 3D collection object to the plot.

        2D collection types are converted to a 3D version by
        modifying the object and adding z coordinate information,
        *zs* and *zdir*.

        Supported 2D collection types are:

        - `.PolyCollection`
        - `.LineCollection`
        - `.PatchCollection` (currently not supporting *autolim*)

        Parameters
        ----------
        col : `.Collection`
            A 2D collection object.
        zs : float or array-like, default: 0
            The z-positions to be used for the 2D objects.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use for the z-positions.
        autolim : bool, default: True
            Whether to update the data limits.
        axlim_clip : bool, default: False
            Whether to hide the scatter points outside the axes view limits.

            .. versionadded:: 3.10
        """
        had_data = self.has_data()

        zvals = np.atleast_1d(zs)
        zsortval = (np.min(zvals) if zvals.size
                    else 0)  # FIXME: arbitrary default

        # FIXME: use issubclass() (although, then a 3D collection
        #       object would also pass.)  Maybe have a collection3d
        #       abstract class to test for and exclude?
        if type(col) is mcoll.PolyCollection:
            art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir,
                                           axlim_clip=axlim_clip)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.LineCollection:
            art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir,
                                           axlim_clip=axlim_clip)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.PatchCollection:
            art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir,
                                            axlim_clip=axlim_clip)
            col.set_sort_zpos(zsortval)

        if autolim:
            if isinstance(col, art3d.Line3DCollection):
                self.auto_scale_xyz(*np.array(col._segments3d).transpose(),
                                    had_data=had_data)
            elif isinstance(col, art3d.Poly3DCollection):
                self.auto_scale_xyz(*col._vec[:-1], had_data=had_data)
            elif isinstance(col, art3d.Patch3DCollection):
                pass
                # FIXME: Implement auto-scaling function for Patch3DCollection
                # Currently unable to do so due to issues with Patch3DCollection
                # See https://github.com/matplotlib/matplotlib/issues/14298 for details

        collection = super().add_collection(col)
        return collection

    @_preprocess_data(replace_names=["xs", "ys", "zs", "s",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"])
    def scatter(self, xs, ys,
                zs=0, zdir='z', s=20, c=None, depthshade=True, *args,
                axlim_clip=False, **kwargs):
        """
        Create a scatter plot.

        Parameters
        ----------
        xs, ys : array-like
            The data positions.
        zs : float or array-like, default: 0
            The z-positions. Either an array of the same length as *xs* and
            *ys* or a single value to place all points in the same plane.
        zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, default: 'z'
            The axis direction for the *zs*. This is useful when plotting 2D
            data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting
            *zdir* to 'y' then plots the data to the x-z-plane.

            See also :doc:`/gallery/mplot3d/2dcollections3d`.

        s : float or array-like, default: 20
            The marker size in points**2. Either an array of the same length
            as *xs* and *ys* or a single value to make all markers the same
            size.
        c : :mpltype:`color`, sequence, or sequence of colors, optional
            The marker color. Possible values:

            - A single color format string.
            - A sequence of colors of length n.
            - A sequence of n numbers to be mapped to colors using *cmap* and
              *norm*.
            - A 2D array in which the rows are RGB or RGBA.

            For more details see the *c* argument of `~.axes.Axes.scatter`.
        depthshade : bool, default: True
            Whether to shade the scatter markers to give the appearance of
            depth. Each call to ``scatter()`` will perform its depthshading
            independently.
        axlim_clip : bool, default: False
            Whether to hide the scatter points outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            All other keyword arguments are passed on to `~.axes.Axes.scatter`.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`
        """

        had_data = self.has_data()
        zs_orig = zs

        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)
        s = np.ma.ravel(s)  # This doesn't have to match x, y in size.

        xs, ys, zs, s, c, color = cbook.delete_masked_points(
            xs, ys, zs, s, c, kwargs.get('color', None)
            )
        if kwargs.get("color") is not None:
            kwargs['color'] = color

        # For xs and ys, 2D scatter() will do the copying.
        if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.
            zs = zs.copy()

        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir,
                                        depthshade=depthshade,
                                        axlim_clip=axlim_clip)

        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)

        self.auto_scale_xyz(xs, ys, zs, had_data)

        return patches

    scatter3D = scatter

    @_preprocess_data()
    def bar(self, left, height, zs=0, zdir='z', *args,
            axlim_clip=False, **kwargs):
        """
        Add 2D bar(s).

        Parameters
        ----------
        left : 1D array-like
            The x coordinates of the left sides of the bars.
        height : 1D array-like
            The height of the bars.
        zs : float or 1D array-like, default: 0
            Z coordinate of bars; if a single value is specified, it will be
            used for all bars.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
        axlim_clip : bool, default: False
            Whether to hide bars with points outside the axes view limits.

            .. versionadded:: 3.10
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            Other keyword arguments are forwarded to
            `matplotlib.axes.Axes.bar`.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Patch3DCollection
        """
        had_data = self.has_data()

        patches = super().bar(left, height, *args, **kwargs)

        zs = np.broadcast_to(zs, len(left), subok=True)

        verts = []
        verts_zs = []
        for p, z in zip(patches, zs):
            vs = art3d._get_patch_verts(p)
            verts += vs.tolist()
            verts_zs += [z] * len(vs)
            art3d.patch_2d_to_3d(p, z, zdir, axlim_clip)
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])

        if len(verts) > 0:
            # the following has to be skipped if verts is empty
            # NOTE: Bugs could still occur if len(verts) > 0,
            #       but the "2nd dimension" is empty.
            xs, ys = zip(*verts)
        else:
            xs, ys = [], []

        xs, ys, verts_zs = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)

        return patches

    @_preprocess_data()
    def bar3d(self, x, y, z, dx, dy, dz, color=None,
              zsort='average', shade=True, lightsource=None, *args,
              axlim_clip=False, **kwargs):
        """
        Generate a 3D barplot.

        This method creates three-dimensional barplot where the width,
        depth, height, and color of the bars can all be uniquely set.

        Parameters
        ----------
        x, y, z : array-like
            The coordinates of the anchor point of the bars.

        dx, dy, dz : float or array-like
            The width, depth, and height of the bars, respectively.

        color : sequence of colors, optional
            The color of the bars can be specified globally or
            individually. This parameter can be:

            - A single color, to color all bars the same color.
            - An array of colors of length N bars, to color each bar
              independently.
            - An array of colors of length 6, to color the faces of the
              bars similarly.
            - An array of colors of length 6 * N bars, to color each face
              independently.

            When coloring the faces of the boxes specifically, this is
            the order of the coloring:

            1. -Z (bottom of box)
            2. +Z (top of box)
            3. -Y
            4. +Y
            5. -X
            6. +X

        zsort : {'average', 'min', 'max'}, default: 'average'
            The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`

        shade : bool, default: True
            When true, this shades the dark sides of the bars (relative
            to the plot's source of light).

        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.

        axlim_clip : bool, default: False
            Whether to hide the bars with points outside the axes view limits.

            .. versionadded:: 3.10

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Any additional keyword arguments are passed onto
            `~.art3d.Poly3DCollection`.

        Returns
        -------
        collection : `~.art3d.Poly3DCollection`
            A collection of three-dimensional polygons representing the bars.
        """

        had_data = self.has_data()

        x, y, z, dx, dy, dz = np.broadcast_arrays(
            np.atleast_1d(x), y, z, dx, dy, dz)
        minx = np.min(x)
        maxx = np.max(x + dx)
        miny = np.min(y)
        maxy = np.max(y + dy)
        minz = np.min(z)
        maxz = np.max(z + dz)

        # shape (6, 4, 3)
        # All faces are oriented facing outwards - when viewed from the
        # outside, their vertices are in a counterclockwise ordering.
        cuboid = np.array([
            # -z
            (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ),
            # +z
            (
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            # -y
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
            ),
            # +y
            (
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (1, 1, 0),
            ),
            # -x
            (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
            ),
            # +x
            (
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
            ),
        ])

        # indexed by [bar, face, vertex, coord]
        polys = np.empty(x.shape + cuboid.shape)

        # handle each coordinate separately
        for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
            p = p[..., np.newaxis, np.newaxis]
            dp = dp[..., np.newaxis, np.newaxis]
            polys[..., i] = p + dp * cuboid[..., i]

        # collapse the first two axes
        polys = polys.reshape((-1,) + polys.shape[2:])

        facecolors = []
        if color is None:
            color = [self._get_patches_for_fill.get_next_color()]

        color = list(mcolors.to_rgba_array(color))

        if len(color) == len(x):
            # bar colors specified, need to expand to number of faces
            for c in color:
                facecolors.extend([c] * 6)
        else:
            # a single color specified, or face colors specified explicitly
            facecolors = color
            if len(facecolors) < len(x):
                facecolors *= (6 * len(x))

        col = art3d.Poly3DCollection(polys,
                                     zsort=zsort,
                                     facecolors=facecolors,
                                     shade=shade,
                                     lightsource=lightsource,
                                     axlim_clip=axlim_clip,
                                     *args, **kwargs)
        self.add_collection(col)

        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

        return col

    def set_title(self, label, fontdict=None, loc='center', **kwargs):
        # docstring inherited
        ret = super().set_title(label, fontdict=fontdict, loc=loc, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)
        return ret

    @_preprocess_data()
    def quiver(self, X, Y, Z, U, V, W, *,
               length=1, arrow_length_ratio=.3, pivot='tail', normalize=False,
               axlim_clip=False, **kwargs):
        """
        Plot a 3D field of arrows.

        The arguments can be array-like or scalars, so long as they can be
        broadcast together. The arguments can also be masked arrays. If an
        element in any of argument is masked, then that corresponding quiver
        element will not be plotted.

        Parameters
        ----------
        X, Y, Z : array-like
            The x, y and z coordinates of the arrow locations (default is
            tail of arrow; see *pivot* kwarg).

        U, V, W : array-like
            The x, y and z components of the arrow vectors.

        length : float, default: 1
            The length of each quiver.

        arrow_length_ratio : float, default: 0.3
            The ratio of the arrow head with respect to the quiver.

        pivot : {'tail', 'middle', 'tip'}, default: 'tail'
            The part of the arrow that is at the grid point; the arrow
            rotates about this point, hence the name *pivot*.

        normalize : bool, default: False
            Whether all arrows are normalized to have the same length, or keep
            the lengths defined by *u*, *v*, and *w*.

        axlim_clip : bool, default: False
            Whether to hide arrows with points outside the axes view limits.

            .. versionadded:: 3.10

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Any additional keyword arguments are delegated to
            :class:`.Line3DCollection`
        """

        def calc_arrows(UVW):
            # get unit direction vector perpendicular to (u, v, w)
            x = UVW[:, 0]
            y = UVW[:, 1]
            norm = np.linalg.norm(UVW[:, :2], axis=1)
            x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
            y_p = np.divide(-x,  norm, where=norm != 0, out=np.ones_like(x))
            # compute the two arrowhead direction unit vectors
            rangle = math.radians(15)
            c = math.cos(rangle)
            s = math.sin(rangle)
            # construct the rotation matrices of shape (3, 3, n)
            r13 = y_p * s
            r32 = x_p * s
            r12 = x_p * y_p * (1 - c)
            Rpos = np.array(
                [[c + (x_p ** 2) * (1 - c), r12, r13],
                 [r12, c + (y_p ** 2) * (1 - c), -r32],
                 [-r13, r32, np.full_like(x_p, c)]])
            # opposite rotation negates all the sin terms
            Rneg = Rpos.copy()
            Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
            # Batch n (3, 3) x (3) matrix multiplications ((3, 3, n) x (n, 3)).
            Rpos_vecs = np.einsum("ij...,...j->...i", Rpos, UVW)
            Rneg_vecs = np.einsum("ij...,...j->...i", Rneg, UVW)
            # Stack into (n, 2, 3) result.
            return np.stack([Rpos_vecs, Rneg_vecs], axis=1)

        had_data = self.has_data()

        input_args = cbook._broadcast_with_masks(X, Y, Z, U, V, W,
                                                 compress=True)

        if any(len(v) == 0 for v in input_args):
            # No quivers, so just make an empty collection and return early
            linec = art3d.Line3DCollection([], **kwargs)
            self.add_collection(linec)
            return linec

        shaft_dt = np.array([0., length], dtype=float)
        arrow_dt = shaft_dt * arrow_length_ratio

        _api.check_in_list(['tail', 'middle', 'tip'], pivot=pivot)
        if pivot == 'tail':
            shaft_dt -= length
        elif pivot == 'middle':
            shaft_dt -= length / 2

        XYZ = np.column_stack(input_args[:3])
        UVW = np.column_stack(input_args[3:]).astype(float)

        # Normalize rows of UVW
        if normalize:
            norm = np.linalg.norm(UVW, axis=1)
            norm[norm == 0] = 1
            UVW = UVW / norm.reshape((-1, 1))

        if len(XYZ) > 0:
            # compute the shaft lines all at once with an outer product
            shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
            # compute head direction vectors, n heads x 2 sides x 3 dimensions
            head_dirs = calc_arrows(UVW)
            # compute all head lines at once, starting from the shaft ends
            heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
            # stack left and right head lines together
            heads = heads.reshape((len(arrow_dt), -1, 3))
            # transpose to get a list of lines
            heads = heads.swapaxes(0, 1)

            lines = [*shafts, *heads[::2], *heads[1::2]]
        else:
            lines = []

        linec = art3d.Line3DCollection(lines, axlim_clip=axlim_clip, **kwargs)
        self.add_collection(linec)

        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)

        return linec

    quiver3D = quiver

    def voxels(self, *args, facecolors=None, edgecolors=None, shade=True,
               lightsource=None, axlim_clip=False, **kwargs):
        """
        ax.voxels([x, y, z,] /, filled, facecolors=None, edgecolors=None, \
**kwargs)

        Plot a set of filled voxels

        All voxels are plotted as 1x1x1 cubes on the axis, with
        ``filled[0, 0, 0]`` placed with its lower corner at the origin.
        Occluded faces are not plotted.

        Parameters
        ----------
        filled : 3D np.array of bool
            A 3D array of values, with truthy values indicating which voxels
            to fill

        x, y, z : 3D np.array, optional
            The coordinates of the corners of the voxels. This should broadcast
            to a shape one larger in every dimension than the shape of
            *filled*.  These can be used to plot non-cubic voxels.

            If not specified, defaults to increasing integers along each axis,
            like those returned by :func:`~numpy.indices`.
            As indicated by the ``/`` in the function signature, these
            arguments can only be passed positionally.

        facecolors, edgecolors : array-like, optional
            The color to draw the faces and edges of the voxels. Can only be
            passed as keyword arguments.
            These parameters can be:

            - A single color value, to color all voxels the same color. This
              can be either a string, or a 1D RGB/RGBA array
            - ``None``, the default, to use a single color for the faces, and
              the style default for the edges.
            - A 3D `~numpy.ndarray` of color names, with each item the color
              for the corresponding voxel. The size must match the voxels.
            - A 4D `~numpy.ndarray` of RGB/RGBA data, with the components
              along the last axis.

        shade : bool, default: True
            Whether to shade the facecolors.

        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.

        axlim_clip : bool, default: False
            Whether to hide voxels with points outside the axes view limits.

            .. versionadded:: 3.10

        **kwargs
            Additional keyword arguments to pass onto
            `~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

        Returns
        -------
        faces : dict
            A dictionary indexed by coordinate, where ``faces[i, j, k]`` is a
            `.Poly3DCollection` of the faces drawn for the voxel
            ``filled[i, j, k]``. If no faces were drawn for a given voxel,
            either because it was not asked to be drawn, or it is fully
            occluded, then ``(i, j, k) not in faces``.

        Examples
        --------
        .. plot:: gallery/mplot3d/voxels.py
        .. plot:: gallery/mplot3d/voxels_rgb.py
        .. plot:: gallery/mplot3d/voxels_torus.py
        .. plot:: gallery/mplot3d/voxels_numpy_logo.py
        """

        # work out which signature we should be using, and use it to parse
        # the arguments. Name must be voxels for the correct error message
        if len(args) >= 3:
            # underscores indicate position only
            def voxels(__x, __y, __z, filled, **kwargs):
                return (__x, __y, __z), filled, kwargs
        else:
            def voxels(filled, **kwargs):
                return None, filled, kwargs

        xyz, filled, kwargs = voxels(*args, **kwargs)

        # check dimensions
        if filled.ndim != 3:
            raise ValueError("Argument filled must be 3-dimensional")
        size = np.array(filled.shape, dtype=np.intp)

        # check xyz coordinates, which are one larger than the filled shape
        coord_shape = tuple(size + 1)
        if xyz is None:
            x, y, z = np.indices(coord_shape)
        else:
            x, y, z = (np.broadcast_to(c, coord_shape) for c in xyz)

        def _broadcast_color_arg(color, name):
            if np.ndim(color) in (0, 1):
                # single color, like "red" or [1, 0, 0]
                return np.broadcast_to(color, filled.shape + np.shape(color))
            elif np.ndim(color) in (3, 4):
                # 3D array of strings, or 4D array with last axis rgb
                if np.shape(color)[:3] != filled.shape:
                    raise ValueError(
                        f"When multidimensional, {name} must match the shape "
                        "of filled")
                return color
            else:
                raise ValueError(f"Invalid {name} argument")

        # broadcast and default on facecolors
        if facecolors is None:
            facecolors = self._get_patches_for_fill.get_next_color()
        facecolors = _broadcast_color_arg(facecolors, 'facecolors')

        # broadcast but no default on edgecolors
        edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

        # scale to the full array, even if the data is only in the center
        self.auto_scale_xyz(x, y, z)

        # points lying on corners of a square
        square = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.intp)

        voxel_faces = defaultdict(list)

        def permutation_matrices(n):
            """Generate cyclic permutation matrices."""
            mat = np.eye(n, dtype=np.intp)
            for i in range(n):
                yield mat
                mat = np.roll(mat, 1, axis=0)

        # iterate over each of the YZ, ZX, and XY orientations, finding faces
        # to render
        for permute in permutation_matrices(3):
            # find the set of ranges to iterate over
            pc, qc, rc = permute.T.dot(size)
            pinds = np.arange(pc)
            qinds = np.arange(qc)
            rinds = np.arange(rc)

            square_rot_pos = square.dot(permute.T)
            square_rot_neg = square_rot_pos[::-1]

            # iterate within the current plane
            for p in pinds:
                for q in qinds:
                    # iterate perpendicularly to the current plane, handling
                    # boundaries. We only draw faces between a voxel and an
                    # empty space, to avoid drawing internal faces.

                    # draw lower faces
                    p0 = permute.dot([p, q, 0])
                    i0 = tuple(p0)
                    if filled[i0]:
                        voxel_faces[i0].append(p0 + square_rot_neg)

                    # draw middle faces
                    for r1, r2 in itertools.pairwise(rinds):
                        p1 = permute.dot([p, q, r1])
                        p2 = permute.dot([p, q, r2])

                        i1 = tuple(p1)
                        i2 = tuple(p2)

                        if filled[i1] and not filled[i2]:
                            voxel_faces[i1].append(p2 + square_rot_pos)
                        elif not filled[i1] and filled[i2]:
                            voxel_faces[i2].append(p2 + square_rot_neg)

                    # draw upper faces
                    pk = permute.dot([p, q, rc-1])
                    pk2 = permute.dot([p, q, rc])
                    ik = tuple(pk)
                    if filled[ik]:
                        voxel_faces[ik].append(pk2 + square_rot_pos)

        # iterate over the faces, and generate a Poly3DCollection for each
        # voxel
        polygons = {}
        for coord, faces_inds in voxel_faces.items():
            # convert indices into 3D positions
            if xyz is None:
                faces = faces_inds
            else:
                faces = []
                for face_inds in faces_inds:
                    ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                    face = np.empty(face_inds.shape)
                    face[:, 0] = x[ind]
                    face[:, 1] = y[ind]
                    face[:, 2] = z[ind]
                    faces.append(face)

            # shade the faces
            facecolor = facecolors[coord]
            edgecolor = edgecolors[coord]

            poly = art3d.Poly3DCollection(
                faces, facecolors=facecolor, edgecolors=edgecolor,
                shade=shade, lightsource=lightsource, axlim_clip=axlim_clip,
                **kwargs)
            self.add_collection3d(poly)
            polygons[coord] = poly

        return polygons

    @_preprocess_data(replace_names=["x", "y", "z", "xerr", "yerr", "zerr"])
    def errorbar(self, x, y, z, zerr=None, yerr=None, xerr=None, fmt='',
                 barsabove=False, errorevery=1, ecolor=None, elinewidth=None,
                 capsize=None, capthick=None, xlolims=False, xuplims=False,
                 ylolims=False, yuplims=False, zlolims=False, zuplims=False,
                 axlim_clip=False,
                 **kwargs):
        """
        Plot lines and/or markers with errorbars around them.

        *x*/*y*/*z* define the data locations, and *xerr*/*yerr*/*zerr* define
        the errorbar sizes. By default, this draws the data markers/lines as
        well the errorbars. Use fmt='none' to draw errorbars only.

        Parameters
        ----------
        x, y, z : float or array-like
            The data positions.

        xerr, yerr, zerr : float or array-like, shape (N,) or (2, N), optional
            The errorbar sizes:

            - scalar: Symmetric +/- values for all data points.
            - shape(N,): Symmetric +/-values for each data point.
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar.

            Note that all error arrays should have *positive* values.

        fmt : str, default: ''
            The format for the data points / data lines. See `.plot` for
            details.

            Use 'none' (case-insensitive) to plot errorbars without any data
            markers.

        ecolor : :mpltype:`color`, default: None
            The color of the errorbar lines.  If None, use the color of the
            line connecting the markers.

        elinewidth : float, default: None
            The linewidth of the errorbar lines. If None, the linewidth of
            the current style is used.

        capsize : float, default: :rc:`errorbar.capsize`
            The length of the error bar caps in points.

        capthick : float, default: None
            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
            This setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*. This may change in future
            releases.

        barsabove : bool, default: False
            If True, will plot the errorbars above the plot
            symbols. Default is below.

        xlolims, ylolims, zlolims : bool, default: False
            These arguments can be used to indicate that a value gives only
            lower limits. In that case a caret symbol is used to indicate
            this. *lims*-arguments may be scalars, or array-likes of the same
            length as the errors. To use limits with inverted axes,
            `~.set_xlim`, `~.set_ylim`, or `~.set_zlim` must be
            called before `errorbar`. Note the tricky parameter names: setting
            e.g. *ylolims* to True means that the y-value is a *lower* limit of
            the True value, so, only an *upward*-pointing arrow will be drawn!

        xuplims, yuplims, zuplims : bool, default: False
            Same as above, but for controlling the upper limits.

        errorevery : int or (int, int), default: 1
            draws error bars on a subset of the data. *errorevery* =N draws
            error bars on the points (x[::N], y[::N], z[::N]).
            *errorevery* =(start, N) draws error bars on the points
            (x[start::N], y[start::N], z[start::N]). e.g. *errorevery* =(6, 3)
            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
            Used to avoid overlapping error bars when two series share x-axis
            values.

        axlim_clip : bool, default: False
            Whether to hide error bars that are outside the axes limits.

            .. versionadded:: 3.10

        Returns
        -------
        errlines : list
            List of `~mpl_toolkits.mplot3d.art3d.Line3DCollection` instances
            each containing an errorbar line.
        caplines : list
            List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each
            containing a capline object.
        limmarks : list
            List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each
            containing a marker with an upper or lower limit.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments for styling errorbar lines are passed
            `~mpl_toolkits.mplot3d.art3d.Line3DCollection`.

        Examples
        --------
        .. plot:: gallery/mplot3d/errorbar3d.py
        """
        had_data = self.has_data()

        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        # Drop anything that comes in as None to use the default instead.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)

        self._process_unit_info([("x", x), ("y", y), ("z", z)], kwargs,
                                convert=False)

        # make sure all the args are iterable; use lists not arrays to
        # preserve units
        x = x if np.iterable(x) else [x]
        y = y if np.iterable(y) else [y]
        z = z if np.iterable(z) else [z]

        if not len(x) == len(y) == len(z):
            raise ValueError("'x', 'y', and 'z' must have the same size")

        everymask = self._errorevery_to_mask(x, errorevery)

        label = kwargs.pop("label", None)
        kwargs['label'] = '_nolegend_'

        # Create the main line and determine overall kwargs for child artists.
        # We avoid calling self.plot() directly, or self._get_lines(), because
        # that would call self._process_unit_info again, and do other indirect
        # data processing.
        (data_line, base_style), = self._get_lines._plot_args(
            self, (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
        art3d.line_2d_to_3d(data_line, zs=z, axlim_clip=axlim_clip)

        # Do this after creating `data_line` to avoid modifying `base_style`.
        if barsabove:
            data_line.set_zorder(kwargs['zorder'] - .1)
        else:
            data_line.set_zorder(kwargs['zorder'] + .1)

        # Add line to plot, or throw it away and use it to determine kwargs.
        if fmt.lower() != 'none':
            self.add_line(data_line)
        else:
            data_line = None
            # Remove alpha=0 color that _process_plot_format returns.
            base_style.pop('color')

        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']

        # Eject any line-specific information from format string, as it's not
        # needed for bars or caps.
        for key in ['marker', 'markersize', 'markerfacecolor',
                    'markeredgewidth', 'markeredgecolor', 'markevery',
                    'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle',
                    'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle']:
            base_style.pop(key, None)

        # Make the style dict for the line collections (the bars).
        eb_lines_style = {**base_style, 'color': ecolor}

        if elinewidth:
            eb_lines_style['linewidth'] = elinewidth
        elif 'linewidth' in kwargs:
            eb_lines_style['linewidth'] = kwargs['linewidth']

        for key in ('transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_lines_style[key] = kwargs[key]

        # Make the style dict for caps (the "hats").
        eb_cap_style = {**base_style, 'linestyle': 'None'}
        if capsize is None:
            capsize = mpl.rcParams["errorbar.capsize"]
        if capsize > 0:
            eb_cap_style['markersize'] = 2. * capsize
        if capthick is not None:
            eb_cap_style['markeredgewidth'] = capthick
        eb_cap_style['color'] = ecolor

        def _apply_mask(arrays, mask):
            # Return, for each array in *arrays*, the elements for which *mask*
            # is True, without using fancy indexing.
            return [[*itertools.compress(array, mask)] for array in arrays]

        def _extract_errs(err, data, lomask, himask):
            # For separate +/- error values we need to unpack err
            if len(err.shape) == 2:
                low_err, high_err = err
            else:
                low_err, high_err = err, err

            lows = np.where(lomask | ~everymask, data, data - low_err)
            highs = np.where(himask | ~everymask, data, data + high_err)

            return lows, highs

        # collect drawn items while looping over the three coordinates
        errlines, caplines, limmarks = [], [], []

        # list of endpoint coordinates, used for auto-scaling
        coorderrs = []

        # define the markers used for errorbar caps and limits below
        # the dictionary key is mapped by the `i_xyz` helper dictionary
        capmarker = {0: '|', 1: '|', 2: '_'}
        i_xyz = {'x': 0, 'y': 1, 'z': 2}

        # Calculate marker size from points to quiver length. Because these are
        # not markers, and 3D Axes do not use the normal transform stack, this
        # is a bit involved. Since the quiver arrows will change size as the
        # scene is rotated, they are given a standard size based on viewing
        # them directly in planar form.
        quiversize = eb_cap_style.get('markersize',
                                      mpl.rcParams['lines.markersize']) ** 2
        quiversize *= self.get_figure(root=True).dpi / 72
        quiversize = self.transAxes.inverted().transform([
            (0, 0), (quiversize, quiversize)])
        quiversize = np.mean(np.diff(quiversize, axis=0))
        # quiversize is now in Axes coordinates, and to convert back to data
        # coordinates, we need to run it through the inverse 3D transform. For
        # consistency, this uses a fixed elevation, azimuth, and roll.
        with cbook._setattr_cm(self, elev=0, azim=0, roll=0):
            invM = np.linalg.inv(self.get_proj())
        # elev=azim=roll=0 produces the Y-Z plane, so quiversize in 2D 'x' is
        # 'y' in 3D, hence the 1 index.
        quiversize = np.dot(invM, [quiversize, 0, 0, 0])[1]
        # Quivers use a fixed 15-degree arrow head, so scale up the length so
        # that the size corresponds to the base. In other words, this constant
        # corresponds to the equation tan(15) = (base / 2) / (arrow length).
        quiversize *= 1.8660254037844388
        eb_quiver_style = {**eb_cap_style,
                           'length': quiversize, 'arrow_length_ratio': 1}
        eb_quiver_style.pop('markersize', None)

        # loop over x-, y-, and z-direction and draw relevant elements
        for zdir, data, err, lolims, uplims in zip(
                ['x', 'y', 'z'], [x, y, z], [xerr, yerr, zerr],
                [xlolims, ylolims, zlolims], [xuplims, yuplims, zuplims]):

            dir_vector = art3d.get_dir_vector(zdir)
            i_zdir = i_xyz[zdir]

            if err is None:
                continue

            if not np.iterable(err):
                err = [err] * len(data)

            err = np.atleast_1d(err)

            # arrays fine here, they are booleans and hence not units
            lolims = np.broadcast_to(lolims, len(data)).astype(bool)
            uplims = np.broadcast_to(uplims, len(data)).astype(bool)

            # a nested list structure that expands to (xl,xh),(yl,yh),(zl,zh),
            # where x/y/z and l/h correspond to dimensions and low/high
            # positions of errorbars in a dimension we're looping over
            coorderr = [
                _extract_errs(err * dir_vector[i], coord, lolims, uplims)
                for i, coord in enumerate([x, y, z])]
            (xl, xh), (yl, yh), (zl, zh) = coorderr

            # draws capmarkers - flat caps orthogonal to the error bars
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                lo_caps_xyz = _apply_mask([xl, yl, zl], nolims & everymask)
                hi_caps_xyz = _apply_mask([xh, yh, zh], nolims & everymask)

                # setting '_' for z-caps and '|' for x- and y-caps;
                # these markers will rotate as the viewing angle changes
                cap_lo = art3d.Line3D(*lo_caps_xyz, ls='',
                                      marker=capmarker[i_zdir],
                                      axlim_clip=axlim_clip,
                                      **eb_cap_style)
                cap_hi = art3d.Line3D(*hi_caps_xyz, ls='',
                                      marker=capmarker[i_zdir],
                                      axlim_clip=axlim_clip,
                                      **eb_cap_style)
                self.add_line(cap_lo)
                self.add_line(cap_hi)
                caplines.append(cap_lo)
                caplines.append(cap_hi)

            if lolims.any():
                xh0, yh0, zh0 = _apply_mask([xh, yh, zh], lolims & everymask)
                self.quiver(xh0, yh0, zh0, *dir_vector, **eb_quiver_style)
            if uplims.any():
                xl0, yl0, zl0 = _apply_mask([xl, yl, zl], uplims & everymask)
                self.quiver(xl0, yl0, zl0, *-dir_vector, **eb_quiver_style)

            errline = art3d.Line3DCollection(np.array(coorderr).T,
                                             axlim_clip=axlim_clip,
                                             **eb_lines_style)
            self.add_collection(errline)
            errlines.append(errline)
            coorderrs.append(coorderr)

        coorderrs = np.array(coorderrs)

        def _digout_minmax(err_arr, coord_label):
            return (np.nanmin(err_arr[:, i_xyz[coord_label], :, :]),
                    np.nanmax(err_arr[:, i_xyz[coord_label], :, :]))

        minx, maxx = _digout_minmax(coorderrs, 'x')
        miny, maxy = _digout_minmax(coorderrs, 'y')
        minz, maxz = _digout_minmax(coorderrs, 'z')
        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

        # Adapting errorbar containers for 3d case, assuming z-axis points "up"
        errorbar_container = mcontainer.ErrorbarContainer(
            (data_line, tuple(caplines), tuple(errlines)),
            has_xerr=(xerr is not None or yerr is not None),
            has_yerr=(zerr is not None),
            label=label)
        self.containers.append(errorbar_container)

        return errlines, caplines, limmarks

    def get_tightbbox(self, renderer=None, *, call_axes_locator=True,
                      bbox_extra_artists=None, for_layout_only=False):
        ret = super().get_tightbbox(renderer,
                                    call_axes_locator=call_axes_locator,
                                    bbox_extra_artists=bbox_extra_artists,
                                    for_layout_only=for_layout_only)
        batch = [ret]
        if self._axis3don:
            for axis in self._axis_map.values():
                if axis.get_visible():
                    axis_bb = martist._get_tightbbox_for_layout_only(
                        axis, renderer)
                    if axis_bb:
                        batch.append(axis_bb)
        return mtransforms.Bbox.union(batch)

    @_preprocess_data()
    def stem(self, x, y, z, *, linefmt='C0-', markerfmt='C0o', basefmt='C3-',
             bottom=0, label=None, orientation='z', axlim_clip=False):
        """
        Create a 3D stem plot.

        A stem plot draws lines perpendicular to a baseline, and places markers
        at the heads. By default, the baseline is defined by *x* and *y*, and
        stems are drawn vertically from *bottom* to *z*.

        Parameters
        ----------
        x, y, z : array-like
            The positions of the heads of the stems. The stems are drawn along
            the *orientation*-direction from the baseline at *bottom* (in the
            *orientation*-coordinate) to the heads. By default, the *x* and *y*
            positions are used for the baseline and *z* for the head position,
            but this can be changed by *orientation*.

        linefmt : str, default: 'C0-'
            A string defining the properties of the vertical lines. Usually,
            this will be a color or a color and a linestyle:

            =========  =============
            Character  Line Style
            =========  =============
            ``'-'``    solid line
            ``'--'``   dashed line
            ``'-.'``   dash-dot line
            ``':'``    dotted line
            =========  =============

            Note: While it is technically possible to specify valid formats
            other than color or color and linestyle (e.g. 'rx' or '-.'), this
            is beyond the intention of the method and will most likely not
            result in a reasonable plot.

        markerfmt : str, default: 'C0o'
            A string defining the properties of the markers at the stem heads.

        basefmt : str, default: 'C3-'
            A format string defining the properties of the baseline.

        bottom : float, default: 0
            The position of the baseline, in *orientation*-coordinates.

        label : str, optional
            The label to use for the stems in legends.

        orientation : {'x', 'y', 'z'}, default: 'z'
            The direction along which stems are drawn.

        axlim_clip : bool, default: False
            Whether to hide stems that are outside the axes limits.

            .. versionadded:: 3.10

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        Returns
        -------
        `.StemContainer`
            The container may be treated like a tuple
            (*markerline*, *stemlines*, *baseline*)

        Examples
        --------
        .. plot:: gallery/mplot3d/stem3d_demo.py
        """

        from matplotlib.container import StemContainer

        had_data = self.has_data()

        _api.check_in_list(['x', 'y', 'z'], orientation=orientation)

        xlim = (np.min(x), np.max(x))
        ylim = (np.min(y), np.max(y))
        zlim = (np.min(z), np.max(z))

        # Determine the appropriate plane for the baseline and the direction of
        # stemlines based on the value of orientation.
        if orientation == 'x':
            basex, basexlim = y, ylim
            basey, baseylim = z, zlim
            lines = [[(bottom, thisy, thisz), (thisx, thisy, thisz)]
                     for thisx, thisy, thisz in zip(x, y, z)]
        elif orientation == 'y':
            basex, basexlim = x, xlim
            basey, baseylim = z, zlim
            lines = [[(thisx, bottom, thisz), (thisx, thisy, thisz)]
                     for thisx, thisy, thisz in zip(x, y, z)]
        else:
            basex, basexlim = x, xlim
            basey, baseylim = y, ylim
            lines = [[(thisx, thisy, bottom), (thisx, thisy, thisz)]
                     for thisx, thisy, thisz in zip(x, y, z)]

        # Determine style for stem lines.
        linestyle, linemarker, linecolor = _process_plot_format(linefmt)
        if linestyle is None:
            linestyle = mpl.rcParams['lines.linestyle']

        # Plot everything in required order.
        baseline, = self.plot(basex, basey, basefmt, zs=bottom,
                              zdir=orientation, label='_nolegend_')
        stemlines = art3d.Line3DCollection(
            lines, linestyles=linestyle, colors=linecolor, label='_nolegend_',
            axlim_clip=axlim_clip)
        self.add_collection(stemlines)
        markerline, = self.plot(x, y, z, markerfmt, label='_nolegend_')

        stem_container = StemContainer((markerline, stemlines, baseline),
                                       label=label)
        self.add_container(stem_container)

        jx, jy, jz = art3d.juggle_axes(basexlim, baseylim, [bottom, bottom],
                                       orientation)
        self.auto_scale_xyz([*jx, *xlim], [*jy, *ylim], [*jz, *zlim], had_data)

        return stem_container

    stem3D = stem


def get_test_data(delta=0.05):
    """Return a tuple X, Y, Z with a test data set."""
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z


class _Quaternion:
    """
    Quaternions
    consisting of scalar, along 1, and vector, with components along i, j, k
    """

    def __init__(self, scalar, vector):
        self.scalar = scalar
        self.vector = np.array(vector)

    def __neg__(self):
        return self.__class__(-self.scalar, -self.vector)

    def __mul__(self, other):
        """
        Product of two quaternions
        i*i = j*j = k*k = i*j*k = -1
        Quaternion multiplication can be expressed concisely
        using scalar and vector parts,
        see <https://en.wikipedia.org/wiki/Quaternion#Scalar_and_vector_parts>
        """
        return self.__class__(
            self.scalar*other.scalar - np.dot(self.vector, other.vector),
            self.scalar*other.vector + self.vector*other.scalar
            + np.cross(self.vector, other.vector))

    def conjugate(self):
        """The conjugate quaternion -(1/2)*(q+i*q*i+j*q*j+k*q*k)"""
        return self.__class__(self.scalar, -self.vector)

    @property
    def norm(self):
        """The 2-norm, q*q', a scalar"""
        return self.scalar*self.scalar + np.dot(self.vector, self.vector)

    def normalize(self):
        """Scaling such that norm equals 1"""
        n = np.sqrt(self.norm)
        return self.__class__(self.scalar/n, self.vector/n)

    def reciprocal(self):
        """The reciprocal, 1/q = q'/(q*q') = q' / norm(q)"""
        n = self.norm
        return self.__class__(self.scalar/n, -self.vector/n)

    def __div__(self, other):
        return self*other.reciprocal()

    __truediv__ = __div__

    def rotate(self, v):
        # Rotate the vector v by the quaternion q, i.e.,
        # calculate (the vector part of) q*v/q
        v = self.__class__(0, v)
        v = self*v/self
        return v.vector

    def __eq__(self, other):
        return (self.scalar == other.scalar) and (self.vector == other.vector).all

    def __repr__(self):
        return "_Quaternion({}, {})".format(repr(self.scalar), repr(self.vector))

    @classmethod
    def rotate_from_to(cls, r1, r2):
        """
        The quaternion for the shortest rotation from vector r1 to vector r2
        i.e., q = sqrt(r2*r1'), normalized.
        If r1 and r2 are antiparallel, then the result is ambiguous;
        a normal vector will be returned, and a warning will be issued.
        """
        k = np.cross(r1, r2)
        nk = np.linalg.norm(k)
        th = np.arctan2(nk, np.dot(r1, r2))
        th /= 2
        if nk == 0:  # r1 and r2 are parallel or anti-parallel
            if np.dot(r1, r2) < 0:
                warnings.warn("Rotation defined by anti-parallel vectors is ambiguous")
                k = np.zeros(3)
                k[np.argmin(r1*r1)] = 1  # basis vector most perpendicular to r1-r2
                k = np.cross(r1, k)
                k = k / np.linalg.norm(k)  # unit vector normal to r1-r2
                q = cls(0, k)
            else:
                q = cls(1, [0, 0, 0])  # = 1, no rotation
        else:
            q = cls(np.cos(th), k*np.sin(th)/nk)
        return q

    @classmethod
    def from_cardan_angles(cls, elev, azim, roll):
        """
        Converts the angles to a quaternion
            q = exp((roll/2)*e_x)*exp((elev/2)*e_y)*exp((-azim/2)*e_z)
        i.e., the angles are a kind of Tait-Bryan angles, -z,y',x".
        The angles should be given in radians, not degrees.
        """
        ca, sa = np.cos(azim/2), np.sin(azim/2)
        ce, se = np.cos(elev/2), np.sin(elev/2)
        cr, sr = np.cos(roll/2), np.sin(roll/2)

        qw = ca*ce*cr + sa*se*sr
        qx = ca*ce*sr - sa*se*cr
        qy = ca*se*cr + sa*ce*sr
        qz = ca*se*sr - sa*ce*cr
        return cls(qw, [qx, qy, qz])

    def as_cardan_angles(self):
        """
        The inverse of `from_cardan_angles()`.
        Note that the angles returned are in radians, not degrees.
        The angles are not sensitive to the quaternion's norm().
        """
        qw = self.scalar
        qx, qy, qz = self.vector[..., :]
        azim = np.arctan2(2*(-qw*qz+qx*qy), qw*qw+qx*qx-qy*qy-qz*qz)
        elev = np.arcsin(np.clip(2*(qw*qy+qz*qx)/(qw*qw+qx*qx+qy*qy+qz*qz), -1, 1))
        roll = np.arctan2(2*(qw*qx-qy*qz), qw*qw-qx*qx-qy*qy+qz*qz)
        return elev, azim, roll
<|endoftext|>from numbers import Number
import functools
from types import MethodType

import numpy as np

from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec

from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist


class CbarAxesBase:
    def __init__(self, *args, orientation, **kwargs):
        self.orientation = orientation
        super().__init__(*args, **kwargs)

    def colorbar(self, mappable, **kwargs):
        return self.get_figure(root=False).colorbar(
            mappable, cax=self, location=self.orientation, **kwargs)


_cbaraxes_class_factory = cbook._make_class_factory(CbarAxesBase, "Cbar{}")


class Grid:
    """
    A grid of Axes.

    In Matplotlib, the Axes location (and size) is specified in normalized
    figure coordinates. This may not be ideal for images that needs to be
    displayed with a given aspect ratio; for example, it is difficult to
    display multiple images of a same size with some fixed padding between
    them.  AxesGrid can be used in such case.

    Attributes
    ----------
    axes_all : list of Axes
        A flat list of Axes. Note that you can also access this directly
        from the grid. The following is equivalent ::

            grid[i] == grid.axes_all[i]
            len(grid) == len(grid.axes_all)

    axes_column : list of list of Axes
        A 2D list of Axes where the first index is the column. This results
        in the usage pattern ``grid.axes_column[col][row]``.
    axes_row : list of list of Axes
        A 2D list of Axes where the first index is the row. This results
        in the usage pattern ``grid.axes_row[row][col]``.
    axes_llc : Axes
        The Axes in the lower left corner.
    ngrids : int
        Number of Axes in the grid.
    """

    _defaultAxesClass = Axes

    def __init__(self, fig,
                 rect,
                 nrows_ncols,
                 ngrids=None,
                 direction="row",
                 axes_pad=0.02,
                 *,
                 share_all=False,
                 share_x=True,
                 share_y=True,
                 label_mode="L",
                 axes_class=None,
                 aspect=False,
                 ):
        """
        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float), (int, int, int), int, or \
    `~.SubplotSpec`
            The axes position, as a ``(left, bottom, width, height)`` tuple,
            as a three-digit subplot position code (e.g., ``(1, 2, 1)`` or
            ``121``), or as a `~.SubplotSpec`.
        nrows_ncols : (int, int)
            Number of rows and columns in the grid.
        ngrids : int or None, default: None
            If not None, only the first *ngrids* axes in the grid are created.
        direction : {"row", "column"}, default: "row"
            Whether axes are created in row-major ("row by row") or
            column-major order ("column by column").  This also affects the
            order in which axes are accessed using indexing (``grid[index]``).
        axes_pad : float or (float, float), default: 0.02
            Padding or (horizontal padding, vertical padding) between axes, in
            inches.
        share_all : bool, default: False
            Whether all axes share their x- and y-axis.  Overrides *share_x*
            and *share_y*.
        share_x : bool, default: True
            Whether all axes of a column share their x-axis.
        share_y : bool, default: True
            Whether all axes of a row share their y-axis.
        label_mode : {"L", "1", "all", "keep"}, default: "L"
            Determines which axes will get tick labels:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": All axes are labelled.
            - "keep": Do not do anything.

        axes_class : subclass of `matplotlib.axes.Axes`, default: `.mpl_axes.Axes`
            The type of Axes to create.
        aspect : bool, default: False
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits.
        """
        self._nrows, self._ncols = nrows_ncols

        if ngrids is None:
            ngrids = self._nrows * self._ncols
        else:
            if not 0 < ngrids <= self._nrows * self._ncols:
                raise ValueError(
                    "ngrids must be positive and not larger than nrows*ncols")

        self.ngrids = ngrids

        self._horiz_pad_size, self._vert_pad_size = map(
            Size.Fixed, np.broadcast_to(axes_pad, 2))

        _api.check_in_list(["column", "row"], direction=direction)
        self._direction = direction

        if axes_class is None:
            axes_class = self._defaultAxesClass
        elif isinstance(axes_class, (list, tuple)):
            cls, kwargs = axes_class
            axes_class = functools.partial(cls, **kwargs)

        kw = dict(horizontal=[], vertical=[], aspect=aspect)
        if isinstance(rect, (Number, SubplotSpec)):
            self._divider = SubplotDivider(fig, rect, **kw)
        elif len(rect) == 3:
            self._divider = SubplotDivider(fig, *rect, **kw)
        elif len(rect) == 4:
            self._divider = Divider(fig, rect, **kw)
        else:
            raise TypeError("Incorrect rect format")

        rect = self._divider.get_position()

        axes_array = np.full((self._nrows, self._ncols), None, dtype=object)
        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            if share_all:
                sharex = sharey = axes_array[0, 0]
            else:
                sharex = axes_array[0, col] if share_x else None
                sharey = axes_array[row, 0] if share_y else None
            axes_array[row, col] = axes_class(
                fig, rect, sharex=sharex, sharey=sharey)
        self.axes_all = axes_array.ravel(
            order="C" if self._direction == "row" else "F").tolist()
        self.axes_column = axes_array.T.tolist()
        self.axes_row = axes_array.tolist()
        self.axes_llc = self.axes_column[0][-1]

        self._init_locators()

        for ax in self.axes_all:
            fig.add_axes(ax)

        self.set_label_mode(label_mode)

    def _init_locators(self):
        self._divider.set_horizontal(
            [Size.Scaled(1), self._horiz_pad_size] * (self._ncols-1) + [Size.Scaled(1)])
        self._divider.set_vertical(
            [Size.Scaled(1), self._vert_pad_size] * (self._nrows-1) + [Size.Scaled(1)])
        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            self.axes_all[i].set_axes_locator(
                self._divider.new_locator(nx=2 * col, ny=2 * (self._nrows - 1 - row)))

    def _get_col_row(self, n):
        if self._direction == "column":
            col, row = divmod(n, self._nrows)
        else:
            row, col = divmod(n, self._ncols)

        return col, row

    # Good to propagate __len__ if we have __getitem__
    def __len__(self):
        return len(self.axes_all)

    def __getitem__(self, i):
        return self.axes_all[i]

    def get_geometry(self):
        """
        Return the number of rows and columns of the grid as (nrows, ncols).
        """
        return self._nrows, self._ncols

    def set_axes_pad(self, axes_pad):
        """
        Set the padding between the axes.

        Parameters
        ----------
        axes_pad : (float, float)
            The padding (horizontal pad, vertical pad) in inches.
        """
        self._horiz_pad_size.fixed_size = axes_pad[0]
        self._vert_pad_size.fixed_size = axes_pad[1]

    def get_axes_pad(self):
        """
        Return the axes padding.

        Returns
        -------
        hpad, vpad
            Padding (horizontal pad, vertical pad) in inches.
        """
        return (self._horiz_pad_size.fixed_size,
                self._vert_pad_size.fixed_size)

    def set_aspect(self, aspect):
        """Set the aspect of the SubplotDivider."""
        self._divider.set_aspect(aspect)

    def get_aspect(self):
        """Return the aspect of the SubplotDivider."""
        return self._divider.get_aspect()

    def set_label_mode(self, mode):
        """
        Define which axes have tick labels.

        Parameters
        ----------
        mode : {"L", "1", "all", "keep"}
            The label mode:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": All axes are labelled.
            - "keep": Do not do anything.
        """
        _api.check_in_list(["all", "L", "1", "keep"], mode=mode)
        is_last_row, is_first_col = (
            np.mgrid[:self._nrows, :self._ncols] == [[[self._nrows - 1]], [[0]]])
        if mode == "all":
            bottom = left = np.full((self._nrows, self._ncols), True)
        elif mode == "L":
            bottom = is_last_row
            left = is_first_col
        elif mode == "1":
            bottom = left = is_last_row & is_first_col
        else:
            return
        for i in range(self._nrows):
            for j in range(self._ncols):
                ax = self.axes_row[i][j]
                if isinstance(ax.axis, MethodType):
                    bottom_axis = SimpleAxisArtist(ax.xaxis, 1, ax.spines["bottom"])
                    left_axis = SimpleAxisArtist(ax.yaxis, 1, ax.spines["left"])
                else:
                    bottom_axis = ax.axis["bottom"]
                    left_axis = ax.axis["left"]
                bottom_axis.toggle(ticklabels=bottom[i, j], label=bottom[i, j])
                left_axis.toggle(ticklabels=left[i, j], label=left[i, j])

    def get_divider(self):
        return self._divider

    def set_axes_locator(self, locator):
        self._divider.set_locator(locator)

    def get_axes_locator(self):
        return self._divider.get_locator()


class ImageGrid(Grid):
    """
    A grid of Axes for Image display.

    This class is a specialization of `~.axes_grid1.axes_grid.Grid` for displaying a
    grid of images.  In particular, it forces all axes in a column to share their x-axis
    and all axes in a row to share their y-axis.  It further provides helpers to add
    colorbars to some or all axes.
    """

    def __init__(self, fig,
                 rect,
                 nrows_ncols,
                 ngrids=None,
                 direction="row",
                 axes_pad=0.02,
                 *,
                 share_all=False,
                 aspect=True,
                 label_mode="L",
                 cbar_mode=None,
                 cbar_location="right",
                 cbar_pad=None,
                 cbar_size="5%",
                 cbar_set_cax=True,
                 axes_class=None,
                 ):
        """
        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float) or int
            The axes position, as a ``(left, bottom, width, height)`` tuple or
            as a three-digit subplot position code (e.g., "121").
        nrows_ncols : (int, int)
            Number of rows and columns in the grid.
        ngrids : int or None, default: None
            If not None, only the first *ngrids* axes in the grid are created.
        direction : {"row", "column"}, default: "row"
            Whether axes are created in row-major ("row by row") or
            column-major order ("column by column").  This also affects the
            order in which axes are accessed using indexing (``grid[index]``).
        axes_pad : float or (float, float), default: 0.02in
            Padding or (horizontal padding, vertical padding) between axes, in
            inches.
        share_all : bool, default: False
            Whether all axes share their x- and y-axis.  Note that in any case,
            all axes in a column share their x-axis and all axes in a row share
            their y-axis.
        aspect : bool, default: True
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits.
        label_mode : {"L", "1", "all"}, default: "L"
            Determines which axes will get tick labels:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": all axes are labelled.

        cbar_mode : {"each", "single", "edge", None}, default: None
            Whether to create a colorbar for "each" axes, a "single" colorbar
            for the entire grid, colorbars only for axes on the "edge"
            determined by *cbar_location*, or no colorbars.  The colorbars are
            stored in the :attr:`cbar_axes` attribute.
        cbar_location : {"left", "right", "bottom", "top"}, default: "right"
        cbar_pad : float, default: None
            Padding between the image axes and the colorbar axes.

            .. versionchanged:: 3.10
                ``cbar_mode="single"`` no longer adds *axes_pad* between the axes
                and the colorbar if the *cbar_location* is "left" or "bottom".

        cbar_size : size specification (see `.Size.from_any`), default: "5%"
            Colorbar size.
        cbar_set_cax : bool, default: True
            If True, each axes in the grid has a *cax* attribute that is bound
            to associated *cbar_axes*.
        axes_class : subclass of `matplotlib.axes.Axes`, default: None
        """
        _api.check_in_list(["each", "single", "edge", None],
                           cbar_mode=cbar_mode)
        _api.check_in_list(["left", "right", "bottom", "top"],
                           cbar_location=cbar_location)
        self._colorbar_mode = cbar_mode
        self._colorbar_location = cbar_location
        self._colorbar_pad = cbar_pad
        self._colorbar_size = cbar_size
        # The colorbar axes are created in _init_locators().

        super().__init__(
            fig, rect, nrows_ncols, ngrids,
            direction=direction, axes_pad=axes_pad,
            share_all=share_all, share_x=True, share_y=True, aspect=aspect,
            label_mode=label_mode, axes_class=axes_class)

        for ax in self.cbar_axes:
            fig.add_axes(ax)

        if cbar_set_cax:
            if self._colorbar_mode == "single":
                for ax in self.axes_all:
                    ax.cax = self.cbar_axes[0]
            elif self._colorbar_mode == "edge":
                for index, ax in enumerate(self.axes_all):
                    col, row = self._get_col_row(index)
                    if self._colorbar_location in ("left", "right"):
                        ax.cax = self.cbar_axes[row]
                    else:
                        ax.cax = self.cbar_axes[col]
            else:
                for ax, cax in zip(self.axes_all, self.cbar_axes):
                    ax.cax = cax

    def _init_locators(self):
        # Slightly abusing this method to inject colorbar creation into init.

        if self._colorbar_pad is None:
            # horizontal or vertical arrangement?
            if self._colorbar_location in ("left", "right"):
                self._colorbar_pad = self._horiz_pad_size.fixed_size
            else:
                self._colorbar_pad = self._vert_pad_size.fixed_size
        self.cbar_axes = [
            _cbaraxes_class_factory(self._defaultAxesClass)(
                self.axes_all[0].get_figure(root=False), self._divider.get_position(),
                orientation=self._colorbar_location)
            for _ in range(self.ngrids)]

        cb_mode = self._colorbar_mode
        cb_location = self._colorbar_location

        h = []
        v = []

        h_ax_pos = []
        h_cb_pos = []
        if cb_mode == "single" and cb_location in ("left", "bottom"):
            if cb_location == "left":
                sz = self._nrows * Size.AxesX(self.axes_llc)
                h.append(Size.from_any(self._colorbar_size, sz))
                h.append(Size.from_any(self._colorbar_pad, sz))
                locator = self._divider.new_locator(nx=0, ny=0, ny1=-1)
            elif cb_location == "bottom":
                sz = self._ncols * Size.AxesY(self.axes_llc)
                v.append(Size.from_any(self._colorbar_size, sz))
                v.append(Size.from_any(self._colorbar_pad, sz))
                locator = self._divider.new_locator(nx=0, nx1=-1, ny=0)
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
            self.cbar_axes[0].set_axes_locator(locator)
            self.cbar_axes[0].set_visible(True)

        for col, ax in enumerate(self.axes_row[0]):
            if col != 0:
                h.append(self._horiz_pad_size)

            if ax:
                sz = Size.AxesX(ax, aspect="axes", ref_ax=self.axes_all[0])
            else:
                sz = Size.AxesX(self.axes_all[0],
                                aspect="axes", ref_ax=self.axes_all[0])

            if (cb_location == "left"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and col == 0))):
                h_cb_pos.append(len(h))
                h.append(Size.from_any(self._colorbar_size, sz))
                h.append(Size.from_any(self._colorbar_pad, sz))

            h_ax_pos.append(len(h))
            h.append(sz)

            if (cb_location == "right"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and col == self._ncols - 1))):
                h.append(Size.from_any(self._colorbar_pad, sz))
                h_cb_pos.append(len(h))
                h.append(Size.from_any(self._colorbar_size, sz))

        v_ax_pos = []
        v_cb_pos = []
        for row, ax in enumerate(self.axes_column[0][::-1]):
            if row != 0:
                v.append(self._vert_pad_size)

            if ax:
                sz = Size.AxesY(ax, aspect="axes", ref_ax=self.axes_all[0])
            else:
                sz = Size.AxesY(self.axes_all[0],
                                aspect="axes", ref_ax=self.axes_all[0])

            if (cb_location == "bottom"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and row == 0))):
                v_cb_pos.append(len(v))
                v.append(Size.from_any(self._colorbar_size, sz))
                v.append(Size.from_any(self._colorbar_pad, sz))

            v_ax_pos.append(len(v))
            v.append(sz)

            if (cb_location == "top"
                    and (cb_mode == "each"
                         or (cb_mode == "edge" and row == self._nrows - 1))):
                v.append(Size.from_any(self._colorbar_pad, sz))
                v_cb_pos.append(len(v))
                v.append(Size.from_any(self._colorbar_size, sz))

        for i in range(self.ngrids):
            col, row = self._get_col_row(i)
            locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                ny=v_ax_pos[self._nrows-1-row])
            self.axes_all[i].set_axes_locator(locator)

            if cb_mode == "each":
                if cb_location in ("right", "left"):
                    locator = self._divider.new_locator(
                        nx=h_cb_pos[col], ny=v_ax_pos[self._nrows - 1 - row])

                elif cb_location in ("top", "bottom"):
                    locator = self._divider.new_locator(
                        nx=h_ax_pos[col], ny=v_cb_pos[self._nrows - 1 - row])

                self.cbar_axes[i].set_axes_locator(locator)
            elif cb_mode == "edge":
                if (cb_location == "left" and col == 0
                        or cb_location == "right" and col == self._ncols - 1):
                    locator = self._divider.new_locator(
                        nx=h_cb_pos[0], ny=v_ax_pos[self._nrows - 1 - row])
                    self.cbar_axes[row].set_axes_locator(locator)
                elif (cb_location == "bottom" and row == self._nrows - 1
                      or cb_location == "top" and row == 0):
                    locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                        ny=v_cb_pos[0])
                    self.cbar_axes[col].set_axes_locator(locator)

        if cb_mode == "single":
            if cb_location == "right":
                sz = self._nrows * Size.AxesX(self.axes_llc)
                h.append(Size.from_any(self._colorbar_pad, sz))
                h.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=-2, ny=0, ny1=-1)
            elif cb_location == "top":
                sz = self._ncols * Size.AxesY(self.axes_llc)
                v.append(Size.from_any(self._colorbar_pad, sz))
                v.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=0, nx1=-1, ny=-2)
            if cb_location in ("right", "top"):
                for i in range(self.ngrids):
                    self.cbar_axes[i].set_visible(False)
                self.cbar_axes[0].set_axes_locator(locator)
                self.cbar_axes[0].set_visible(True)
        elif cb_mode == "each":
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(True)
        elif cb_mode == "edge":
            if cb_location in ("right", "left"):
                count = self._nrows
            else:
                count = self._ncols
            for i in range(count):
                self.cbar_axes[i].set_visible(True)
            for j in range(i + 1, self.ngrids):
                self.cbar_axes[j].set_visible(False)
        else:
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
                self.cbar_axes[i].set_position([1., 1., 0.001, 0.001],
                                               which="active")

        self._divider.set_horizontal(h)
        self._divider.set_vertical(v)


AxesGrid = ImageGrid
<|endoftext|>from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes


class ParasiteAxesBase:

    def __init__(self, parent_axes, aux_transform=None,
                 *, viewlim_mode=None, **kwargs):
        self._parent_axes = parent_axes
        self.transAux = aux_transform
        self.set_viewlim_mode(viewlim_mode)
        kwargs["frameon"] = False
        super().__init__(parent_axes.get_figure(root=False),
                         parent_axes._position, **kwargs)

    def clear(self):
        super().clear()
        martist.setp(self.get_children(), visible=False)
        self._get_lines = self._parent_axes._get_lines
        self._parent_axes.callbacks._connect_picklable(
            "xlim_changed", self._sync_lims)
        self._parent_axes.callbacks._connect_picklable(
            "ylim_changed", self._sync_lims)

    def pick(self, mouseevent):
        # This most likely goes to Artist.pick (depending on axes_class given
        # to the factory), which only handles pick events registered on the
        # axes associated with each child:
        super().pick(mouseevent)
        # But parasite axes are additionally given pick events from their host
        # axes (cf. HostAxesBase.pick), which we handle here:
        for a in self.get_children():
            if (hasattr(mouseevent.inaxes, "parasites")
                    and self in mouseevent.inaxes.parasites):
                a.pick(mouseevent)

    # aux_transform support

    def _set_lim_and_transforms(self):
        if self.transAux is not None:
            self.transAxes = self._parent_axes.transAxes
            self.transData = self.transAux + self._parent_axes.transData
            self._xaxis_transform = mtransforms.blended_transform_factory(
                self.transData, self.transAxes)
            self._yaxis_transform = mtransforms.blended_transform_factory(
                self.transAxes, self.transData)
        else:
            super()._set_lim_and_transforms()

    def set_viewlim_mode(self, mode):
        _api.check_in_list([None, "equal", "transform"], mode=mode)
        self._viewlim_mode = mode

    def get_viewlim_mode(self):
        return self._viewlim_mode

    def _sync_lims(self, parent):
        viewlim = parent.viewLim.frozen()
        mode = self.get_viewlim_mode()
        if mode is None:
            pass
        elif mode == "equal":
            self.viewLim.set(viewlim)
        elif mode == "transform":
            self.viewLim.set(viewlim.transformed(self.transAux.inverted()))
        else:
            _api.check_in_list([None, "equal", "transform"], mode=mode)

    # end of aux_transform support


parasite_axes_class_factory = cbook._make_class_factory(
    ParasiteAxesBase, "{}Parasite")
ParasiteAxes = parasite_axes_class_factory(Axes)


class HostAxesBase:
    def __init__(self, *args, **kwargs):
        self.parasites = []
        super().__init__(*args, **kwargs)

    def get_aux_axes(
            self, tr=None, viewlim_mode="equal", axes_class=None, **kwargs):
        """
        Add a parasite axes to this host.

        Despite this method's name, this should actually be thought of as an
        ``add_parasite_axes`` method.

        .. versionchanged:: 3.7
           Defaults to same base axes class as host axes.

        Parameters
        ----------
        tr : `~matplotlib.transforms.Transform` or None, default: None
            If a `.Transform`, the following relation will hold:
            ``parasite.transData = tr + host.transData``.
            If None, the parasite's and the host's ``transData`` are unrelated.
        viewlim_mode : {"equal", "transform", None}, default: "equal"
            How the parasite's view limits are set: directly equal to the
            parent axes ("equal"), equal after application of *tr*
            ("transform"), or independently (None).
        axes_class : subclass type of `~matplotlib.axes.Axes`, optional
            The `~.axes.Axes` subclass that is instantiated.  If None, the base
            class of the host axes is used.
        **kwargs
            Other parameters are forwarded to the parasite axes constructor.
        """
        if axes_class is None:
            axes_class = self._base_axes_class
        parasite_axes_class = parasite_axes_class_factory(axes_class)
        ax2 = parasite_axes_class(
            self, tr, viewlim_mode=viewlim_mode, **kwargs)
        # note that ax2.transData == tr + ax1.transData
        # Anything you draw in ax2 will match the ticks and grids of ax1.
        self.parasites.append(ax2)
        ax2._remove_method = self.parasites.remove
        return ax2

    def draw(self, renderer):
        orig_children_len = len(self._children)

        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.set_position(pos, which="active")
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        rect = self.get_position()
        for ax in self.parasites:
            ax.apply_aspect(rect)
            self._children.extend(ax.get_children())

        super().draw(renderer)
        del self._children[orig_children_len:]

    def clear(self):
        super().clear()
        for ax in self.parasites:
            ax.clear()

    def pick(self, mouseevent):
        super().pick(mouseevent)
        # Also pass pick events on to parasite axes and, in turn, their
        # children (cf. ParasiteAxesBase.pick)
        for a in self.parasites:
            a.pick(mouseevent)

    def twinx(self, axes_class=None):
        """
        Create a twin of Axes with a shared x-axis but independent y-axis.

        The y-axis of self will have ticks on the left and the returned axes
        will have ticks on the right.
        """
        ax = self._add_twin_axes(axes_class, sharex=self)
        self.axis["right"].set_visible(False)
        ax.axis["right"].set_visible(True)
        ax.axis["left", "top", "bottom"].set_visible(False)
        return ax

    def twiny(self, axes_class=None):
        """
        Create a twin of Axes with a shared y-axis but independent x-axis.

        The x-axis of self will have ticks on the bottom and the returned axes
        will have ticks on the top.
        """
        ax = self._add_twin_axes(axes_class, sharey=self)
        self.axis["top"].set_visible(False)
        ax.axis["top"].set_visible(True)
        ax.axis["left", "right", "bottom"].set_visible(False)
        return ax

    def twin(self, aux_trans=None, axes_class=None):
        """
        Create a twin of Axes with no shared axis.

        While self will have ticks on the left and bottom axis, the returned
        axes will have ticks on the top and right axis.
        """
        if aux_trans is None:
            aux_trans = mtransforms.IdentityTransform()
        ax = self._add_twin_axes(
            axes_class, aux_transform=aux_trans, viewlim_mode="transform")
        self.axis["top", "right"].set_visible(False)
        ax.axis["top", "right"].set_visible(True)
        ax.axis["left", "bottom"].set_visible(False)
        return ax

    def _add_twin_axes(self, axes_class, **kwargs):
        """
        Helper for `.twinx`/`.twiny`/`.twin`.

        *kwargs* are forwarded to the parasite axes constructor.
        """
        if axes_class is None:
            axes_class = self._base_axes_class
        ax = parasite_axes_class_factory(axes_class)(self, **kwargs)
        self.parasites.append(ax)
        ax._remove_method = self._remove_any_twin
        return ax

    def _remove_any_twin(self, ax):
        self.parasites.remove(ax)
        restore = ["top", "right"]
        if ax._sharex:
            restore.remove("top")
        if ax._sharey:
            restore.remove("right")
        self.axis[tuple(restore)].set_visible(True)
        self.axis[tuple(restore)].toggle(ticklabels=False, label=False)

    def get_tightbbox(self, renderer=None, *, call_axes_locator=True,
                      bbox_extra_artists=None):
        bbs = [
            *[ax.get_tightbbox(renderer, call_axes_locator=call_axes_locator)
              for ax in self.parasites],
            super().get_tightbbox(renderer,
                                  call_axes_locator=call_axes_locator,
                                  bbox_extra_artists=bbox_extra_artists)]
        return Bbox.union([b for b in bbs if b.width != 0 or b.height != 0])


host_axes_class_factory = host_subplot_class_factory = \
    cbook._make_class_factory(HostAxesBase, "{}HostAxes", "_base_axes_class")
HostAxes = SubplotHost = host_axes_class_factory(Axes)


def host_axes(*args, axes_class=Axes, figure=None, **kwargs):
    """
    Create axes that can act as a hosts to parasitic axes.

    Parameters
    ----------
    figure : `~matplotlib.figure.Figure`
        Figure to which the axes will be added. Defaults to the current figure
        `.pyplot.gcf()`.

    *args, **kwargs
        Will be passed on to the underlying `~.axes.Axes` object creation.
    """
    import matplotlib.pyplot as plt
    host_axes_class = host_axes_class_factory(axes_class)
    if figure is None:
        figure = plt.gcf()
    ax = host_axes_class(figure, *args, **kwargs)
    figure.add_axes(ax)
    return ax


host_subplot = host_axes
<|endoftext|>import matplotlib.axes as maxes
from matplotlib.artist import Artist
from matplotlib.axis import XAxis, YAxis


class SimpleChainedObjects:
    def __init__(self, objects):
        self._objects = objects

    def __getattr__(self, k):
        _a = SimpleChainedObjects([getattr(a, k) for a in self._objects])
        return _a

    def __call__(self, *args, **kwargs):
        for m in self._objects:
            m(*args, **kwargs)


class Axes(maxes.Axes):

    class AxisDict(dict):
        def __init__(self, axes):
            self.axes = axes
            super().__init__()

        def __getitem__(self, k):
            if isinstance(k, tuple):
                r = SimpleChainedObjects(
                    # super() within a list comprehension needs explicit args.
                    [super(Axes.AxisDict, self).__getitem__(k1) for k1 in k])
                return r
            elif isinstance(k, slice):
                if k.start is None and k.stop is None and k.step is None:
                    return SimpleChainedObjects(list(self.values()))
                else:
                    raise ValueError("Unsupported slice")
            else:
                return dict.__getitem__(self, k)

        def __call__(self, *v, **kwargs):
            return maxes.Axes.axis(self.axes, *v, **kwargs)

    @property
    def axis(self):
        return self._axislines

    def clear(self):
        # docstring inherited
        super().clear()
        # Init axis artists.
        self._axislines = self.AxisDict(self)
        self._axislines.update(
            bottom=SimpleAxisArtist(self.xaxis, 1, self.spines["bottom"]),
            top=SimpleAxisArtist(self.xaxis, 2, self.spines["top"]),
            left=SimpleAxisArtist(self.yaxis, 1, self.spines["left"]),
            right=SimpleAxisArtist(self.yaxis, 2, self.spines["right"]))


class SimpleAxisArtist(Artist):
    def __init__(self, axis, axisnum, spine):
        self._axis = axis
        self._axisnum = axisnum
        self.line = spine

        if isinstance(axis, XAxis):
            self._axis_direction = ["bottom", "top"][axisnum-1]
        elif isinstance(axis, YAxis):
            self._axis_direction = ["left", "right"][axisnum-1]
        else:
            raise ValueError(
                f"axis must be instance of XAxis or YAxis, but got {axis}")
        super().__init__()

    @property
    def major_ticks(self):
        tickline = "tick%dline" % self._axisnum
        return SimpleChainedObjects([getattr(tick, tickline)
                                     for tick in self._axis.get_major_ticks()])

    @property
    def major_ticklabels(self):
        label = "label%d" % self._axisnum
        return SimpleChainedObjects([getattr(tick, label)
                                     for tick in self._axis.get_major_ticks()])

    @property
    def label(self):
        return self._axis.label

    def set_visible(self, b):
        self.toggle(all=b)
        self.line.set_visible(b)
        self._axis.set_visible(True)
        super().set_visible(b)

    def set_label(self, txt):
        self._axis.set_label_text(txt)

    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):

        if all:
            _ticks, _ticklabels, _label = True, True, True
        elif all is not None:
            _ticks, _ticklabels, _label = False, False, False
        else:
            _ticks, _ticklabels, _label = None, None, None

        if ticks is not None:
            _ticks = ticks
        if ticklabels is not None:
            _ticklabels = ticklabels
        if label is not None:
            _label = label

        if _ticks is not None:
            tickparam = {f"tick{self._axisnum}On": _ticks}
            self._axis.set_tick_params(**tickparam)
        if _ticklabels is not None:
            tickparam = {f"label{self._axisnum}On": _ticklabels}
            self._axis.set_tick_params(**tickparam)

        if _label is not None:
            pos = self._axis.get_label_position()
            if (pos == self._axis_direction) and not _label:
                self._axis.label.set_visible(False)
            elif _label:
                self._axis.label.set_visible(True)
                self._axis.set_label_position(self._axis_direction)
<|endoftext|>from . import axes_size as Size
from .axes_divider import Divider, SubplotDivider, make_axes_locatable
from .axes_grid import AxesGrid, Grid, ImageGrid

from .parasite_axes import host_subplot, host_axes

__all__ = ["Size",
           "Divider", "SubplotDivider", "make_axes_locatable",
           "AxesGrid", "Grid", "ImageGrid",
           "host_subplot", "host_axes"]
<|endoftext|>from matplotlib import transforms
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  DrawingArea, TextArea, VPacker)
from matplotlib.patches import (Rectangle, ArrowStyle,
                                FancyArrowPatch, PathPatch)
from matplotlib.text import TextPath

__all__ = ['AnchoredDrawingArea', 'AnchoredAuxTransformBox',
           'AnchoredSizeBar', 'AnchoredDirectionArrows']


class AnchoredDrawingArea(AnchoredOffsetbox):
    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, prop=None, frameon=True,
                 **kwargs):
        """
        An anchored container with a fixed size and fillable `.DrawingArea`.

        Artists added to the *drawing_area* will have their coordinates
        interpreted as pixels. Any transformations set on the artists will be
        overridden.

        Parameters
        ----------
        width, height : float
            Width and height of the container, in pixels.
        xdescent, ydescent : float
            Descent of the container in the x- and y- direction, in pixels.
        loc : str
            Location of this artist.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child objects, in fraction of the font size.
        borderpad : float, default: 0.5
            Border padding, in fraction of the font size.
        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property used as a reference for paddings.
        frameon : bool, default: True
            If True, draw a box around this artist.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        drawing_area : `~matplotlib.offsetbox.DrawingArea`
            A container for artists to display.

        Examples
        --------
        To display blue and red circles of different sizes in the upper right
        of an Axes *ax*:

        >>> ada = AnchoredDrawingArea(20, 20, 0, 0,
        ...                           loc='upper right', frameon=False)
        >>> ada.drawing_area.add_artist(Circle((10, 10), 10, fc="b"))
        >>> ada.drawing_area.add_artist(Circle((30, 10), 5, fc="r"))
        >>> ax.add_artist(ada)
        """
        self.da = DrawingArea(width, height, xdescent, ydescent)
        self.drawing_area = self.da

        super().__init__(
            loc, pad=pad, borderpad=borderpad, child=self.da, prop=None,
            frameon=frameon, **kwargs
        )


class AnchoredAuxTransformBox(AnchoredOffsetbox):
    def __init__(self, transform, loc,
                 pad=0.4, borderpad=0.5, prop=None, frameon=True, **kwargs):
        """
        An anchored container with transformed coordinates.

        Artists added to the *drawing_area* are scaled according to the
        coordinates of the transformation used. The dimensions of this artist
        will scale to contain the artists added.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transData`.
        loc : str
            Location of this artist.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child objects, in fraction of the font size.
        borderpad : float, default: 0.5
            Border padding, in fraction of the font size.
        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property used as a reference for paddings.
        frameon : bool, default: True
            If True, draw a box around this artist.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        drawing_area : `~matplotlib.offsetbox.AuxTransformBox`
            A container for artists to display.

        Examples
        --------
        To display an ellipse in the upper left, with a width of 0.1 and
        height of 0.4 in data coordinates:

        >>> box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
        >>> el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)
        >>> box.drawing_area.add_artist(el)
        >>> ax.add_artist(box)
        """
        self.drawing_area = AuxTransformBox(transform)

        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self.drawing_area, prop=prop, frameon=frameon,
                         **kwargs)


class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2,
                 frameon=True, size_vertical=0, color='black',
                 label_top=False, fontproperties=None, fill_bar=None,
                 **kwargs):
        """
        Draw a horizontal scale bar with a center-aligned label underneath.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transData`.
        size : float
            Horizontal length of the size bar, given in coordinates of
            *transform*.
        label : str
            Label to display.
        loc : str
            Location of the size bar.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.1
            Padding around the label and size bar, in fraction of the font
            size.
        borderpad : float, default: 0.1
            Border padding, in fraction of the font size.
        sep : float, default: 2
            Separation between the label and the size bar, in points.
        frameon : bool, default: True
            If True, draw a box around the horizontal bar and label.
        size_vertical : float, default: 0
            Vertical length of the size bar, given in coordinates of
            *transform*.
        color : str, default: 'black'
            Color for the size bar and label.
        label_top : bool, default: False
            If True, the label will be over the size bar.
        fontproperties : `~matplotlib.font_manager.FontProperties`, optional
            Font properties for the label text.
        fill_bar : bool, optional
            If True and if *size_vertical* is nonzero, the size bar will
            be filled in with the color specified by the size bar.
            Defaults to True if *size_vertical* is greater than
            zero and False otherwise.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        size_bar : `~matplotlib.offsetbox.AuxTransformBox`
            Container for the size bar.
        txt_label : `~matplotlib.offsetbox.TextArea`
            Container for the label of the size bar.

        Notes
        -----
        If *prop* is passed as a keyword argument, but *fontproperties* is
        not, then *prop* is assumed to be the intended *fontproperties*.
        Using both *prop* and *fontproperties* is not supported.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mpl_toolkits.axes_grid1.anchored_artists import (
        ...     AnchoredSizeBar)
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.random.random((10, 10)))
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 data units', 4)
        >>> ax.add_artist(bar)
        >>> fig.show()

        Using all the optional parameters

        >>> import matplotlib.font_manager as fm
        >>> fontprops = fm.FontProperties(size=14, family='monospace')
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 units', 4, pad=0.5,
        ...                       sep=5, borderpad=0.5, frameon=False,
        ...                       size_vertical=0.5, color='white',
        ...                       fontproperties=fontprops)
        """
        if fill_bar is None:
            fill_bar = size_vertical > 0

        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, size_vertical,
                                           fill=fill_bar, facecolor=color,
                                           edgecolor=color))

        if fontproperties is None and 'prop' in kwargs:
            fontproperties = kwargs.pop('prop')

        if fontproperties is None:
            textprops = {'color': color}
        else:
            textprops = {'color': color, 'fontproperties': fontproperties}

        self.txt_label = TextArea(label, textprops=textprops)

        if label_top:
            _box_children = [self.txt_label, self.size_bar]
        else:
            _box_children = [self.size_bar, self.txt_label]

        self._box = VPacker(children=_box_children,
                            align="center",
                            pad=0, sep=sep)

        super().__init__(loc, pad=pad, borderpad=borderpad, child=self._box,
                         prop=fontproperties, frameon=frameon, **kwargs)


class AnchoredDirectionArrows(AnchoredOffsetbox):
    def __init__(self, transform, label_x, label_y, length=0.15,
                 fontsize=0.08, loc='upper left', angle=0, aspect_ratio=1,
                 pad=0.4, borderpad=0.4, frameon=False, color='w', alpha=1,
                 sep_x=0.01, sep_y=0, fontproperties=None, back_length=0.15,
                 head_width=10, head_length=15, tail_width=2,
                 text_props=None, arrow_props=None,
                 **kwargs):
        """
        Draw two perpendicular arrows to indicate directions.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transAxes`.
        label_x, label_y : str
            Label text for the x and y arrows
        length : float, default: 0.15
            Length of the arrow, given in coordinates of *transform*.
        fontsize : float, default: 0.08
            Size of label strings, given in coordinates of *transform*.
        loc : str, default: 'upper left'
            Location of the arrow.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        angle : float, default: 0
            The angle of the arrows in degrees.
        aspect_ratio : float, default: 1
            The ratio of the length of arrow_x and arrow_y.
            Negative numbers can be used to change the direction.
        pad : float, default: 0.4
            Padding around the labels and arrows, in fraction of the font size.
        borderpad : float, default: 0.4
            Border padding, in fraction of the font size.
        frameon : bool, default: False
            If True, draw a box around the arrows and labels.
        color : str, default: 'white'
            Color for the arrows and labels.
        alpha : float, default: 1
            Alpha values of the arrows and labels
        sep_x, sep_y : float, default: 0.01 and 0 respectively
            Separation between the arrows and labels in coordinates of
            *transform*.
        fontproperties : `~matplotlib.font_manager.FontProperties`, optional
            Font properties for the label text.
        back_length : float, default: 0.15
            Fraction of the arrow behind the arrow crossing.
        head_width : float, default: 10
            Width of arrow head, sent to `.ArrowStyle`.
        head_length : float, default: 15
            Length of arrow head, sent to `.ArrowStyle`.
        tail_width : float, default: 2
            Width of arrow tail, sent to `.ArrowStyle`.
        text_props, arrow_props : dict
            Properties of the text and arrows, passed to `.TextPath` and
            `.FancyArrowPatch`.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        arrow_x, arrow_y : `~matplotlib.patches.FancyArrowPatch`
            Arrow x and y
        text_path_x, text_path_y : `~matplotlib.text.TextPath`
            Path for arrow labels
        p_x, p_y : `~matplotlib.patches.PathPatch`
            Patch for arrow labels
        box : `~matplotlib.offsetbox.AuxTransformBox`
            Container for the arrows and labels.

        Notes
        -----
        If *prop* is passed as a keyword argument, but *fontproperties* is
        not, then *prop* is assumed to be the intended *fontproperties*.
        Using both *prop* and *fontproperties* is not supported.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mpl_toolkits.axes_grid1.anchored_artists import (
        ...     AnchoredDirectionArrows)
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.random.random((10, 10)))
        >>> arrows = AnchoredDirectionArrows(ax.transAxes, '111', '110')
        >>> ax.add_artist(arrows)
        >>> fig.show()

        Using several of the optional parameters, creating downward pointing
        arrow and high contrast text labels.

        >>> import matplotlib.font_manager as fm
        >>> fontprops = fm.FontProperties(family='monospace')
        >>> arrows = AnchoredDirectionArrows(ax.transAxes, 'East', 'South',
        ...                                  loc='lower left', color='k',
        ...                                  aspect_ratio=-1, sep_x=0.02,
        ...                                  sep_y=-0.01,
        ...                                  text_props={'ec':'w', 'fc':'k'},
        ...                                  fontproperties=fontprops)
        """
        if arrow_props is None:
            arrow_props = {}

        if text_props is None:
            text_props = {}

        arrowstyle = ArrowStyle("Simple",
                                head_width=head_width,
                                head_length=head_length,
                                tail_width=tail_width)

        if fontproperties is None and 'prop' in kwargs:
            fontproperties = kwargs.pop('prop')

        if 'color' not in arrow_props:
            arrow_props['color'] = color

        if 'alpha' not in arrow_props:
            arrow_props['alpha'] = alpha

        if 'color' not in text_props:
            text_props['color'] = color

        if 'alpha' not in text_props:
            text_props['alpha'] = alpha

        t_start = transform
        t_end = t_start + transforms.Affine2D().rotate_deg(angle)

        self.box = AuxTransformBox(t_end)

        length_x = length
        length_y = length*aspect_ratio

        self.arrow_x = FancyArrowPatch(
                (0, back_length*length_y),
                (length_x, back_length*length_y),
                arrowstyle=arrowstyle,
                shrinkA=0.0,
                shrinkB=0.0,
                **arrow_props)

        self.arrow_y = FancyArrowPatch(
                (back_length*length_x, 0),
                (back_length*length_x, length_y),
                arrowstyle=arrowstyle,
                shrinkA=0.0,
                shrinkB=0.0,
                **arrow_props)

        self.box.add_artist(self.arrow_x)
        self.box.add_artist(self.arrow_y)

        text_path_x = TextPath((
            length_x+sep_x, back_length*length_y+sep_y), label_x,
            size=fontsize, prop=fontproperties)
        self.p_x = PathPatch(text_path_x, transform=t_start, **text_props)
        self.box.add_artist(self.p_x)

        text_path_y = TextPath((
            length_x*back_length+sep_x, length_y*(1-back_length)+sep_y),
            label_y, size=fontsize, prop=fontproperties)
        self.p_y = PathPatch(text_path_y, **text_props)
        self.box.add_artist(self.p_y)

        super().__init__(loc, pad=pad, borderpad=borderpad, child=self.box,
                         frameon=frameon, **kwargs)
<|endoftext|>from types import MethodType

import numpy as np

from .axes_divider import make_axes_locatable, Size
from .mpl_axes import Axes, SimpleAxisArtist


def make_rgb_axes(ax, pad=0.01, axes_class=None, **kwargs):
    """
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes instance to create the RGB Axes in.
    pad : float, optional
        Fraction of the Axes height to pad.
    axes_class : `matplotlib.axes.Axes` or None, optional
        Axes class to use for the R, G, and B Axes. If None, use
        the same class as *ax*.
    **kwargs
        Forwarded to *axes_class* init for the R, G, and B Axes.
    """

    divider = make_axes_locatable(ax)

    pad_size = pad * Size.AxesY(ax)

    xsize = ((1-2*pad)/3) * Size.AxesX(ax)
    ysize = ((1-2*pad)/3) * Size.AxesY(ax)

    divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])
    divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])

    ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))

    ax_rgb = []
    if axes_class is None:
        axes_class = type(ax)

    for ny in [4, 2, 0]:
        ax1 = axes_class(ax.get_figure(), ax.get_position(original=True),
                         sharex=ax, sharey=ax, **kwargs)
        locator = divider.new_locator(nx=2, ny=ny)
        ax1.set_axes_locator(locator)
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)
        try:
            for axis in ax1.axis.values():
                axis.major_ticklabels.set_visible(False)
        except AttributeError:
            pass

        ax_rgb.append(ax1)

    fig = ax.get_figure()
    for ax1 in ax_rgb:
        fig.add_axes(ax1)

    return ax_rgb


class RGBAxes:
    """
    4-panel `~.Axes.imshow` (RGB, R, G, B).

    Layout::

        ┌───────────────┬─────┐
        │               │  R  │
        │               ├─────┤
        │      RGB      │  G  │
        │               ├─────┤
        │               │  B  │
        └───────────────┴─────┘

    Subclasses can override the ``_defaultAxesClass`` attribute.
    By default RGBAxes uses `.mpl_axes.Axes`.

    Attributes
    ----------
    RGB : ``_defaultAxesClass``
        The Axes object for the three-channel `~.Axes.imshow`.
    R : ``_defaultAxesClass``
        The Axes object for the red channel `~.Axes.imshow`.
    G : ``_defaultAxesClass``
        The Axes object for the green channel `~.Axes.imshow`.
    B : ``_defaultAxesClass``
        The Axes object for the blue channel `~.Axes.imshow`.
    """

    _defaultAxesClass = Axes

    def __init__(self, *args, pad=0, **kwargs):
        """
        Parameters
        ----------
        pad : float, default: 0
            Fraction of the Axes height to put as padding.
        axes_class : `~matplotlib.axes.Axes`
            Axes class to use. If not provided, ``_defaultAxesClass`` is used.
        *args
            Forwarded to *axes_class* init for the RGB Axes
        **kwargs
            Forwarded to *axes_class* init for the RGB, R, G, and B Axes
        """
        axes_class = kwargs.pop("axes_class", self._defaultAxesClass)
        self.RGB = ax = axes_class(*args, **kwargs)
        ax.get_figure().add_axes(ax)
        self.R, self.G, self.B = make_rgb_axes(
            ax, pad=pad, axes_class=axes_class, **kwargs)
        # Set the line color and ticks for the axes.
        for ax1 in [self.RGB, self.R, self.G, self.B]:
            if isinstance(ax1.axis, MethodType):
                ad = Axes.AxisDict(self)
                ad.update(
                    bottom=SimpleAxisArtist(ax1.xaxis, 1, ax1.spines["bottom"]),
                    top=SimpleAxisArtist(ax1.xaxis, 2, ax1.spines["top"]),
                    left=SimpleAxisArtist(ax1.yaxis, 1, ax1.spines["left"]),
                    right=SimpleAxisArtist(ax1.yaxis, 2, ax1.spines["right"]))
            else:
                ad = ax1.axis
            ad[:].line.set_color("w")
            ad[:].major_ticks.set_markeredgecolor("w")

    def imshow_rgb(self, r, g, b, **kwargs):
        """
        Create the four images {rgb, r, g, b}.

        Parameters
        ----------
        r, g, b : array-like
            The red, green, and blue arrays.
        **kwargs
            Forwarded to `~.Axes.imshow` calls for the four images.

        Returns
        -------
        rgb : `~matplotlib.image.AxesImage`
        r : `~matplotlib.image.AxesImage`
        g : `~matplotlib.image.AxesImage`
        b : `~matplotlib.image.AxesImage`
        """
        if not (r.shape == g.shape == b.shape):
            raise ValueError(
                f'Input shapes ({r.shape}, {g.shape}, {b.shape}) do not match')
        RGB = np.dstack([r, g, b])
        R = np.zeros_like(RGB)
        R[:, :, 0] = r
        G = np.zeros_like(RGB)
        G[:, :, 1] = g
        B = np.zeros_like(RGB)
        B[:, :, 2] = b
        im_rgb = self.RGB.imshow(RGB, **kwargs)
        im_r = self.R.imshow(R, **kwargs)
        im_g = self.G.imshow(G, **kwargs)
        im_b = self.B.imshow(B, **kwargs)
        return im_rgb, im_r, im_g, im_b
<|endoftext|>"""
A collection of functions and objects for creating or placing inset axes.
"""

from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.transforms import IdentityTransform, TransformedBbox

from . import axes_size as Size
from .parasite_axes import HostAxes


class AnchoredLocatorBase(AnchoredOffsetbox):
    def __init__(self, bbox_to_anchor, offsetbox, loc,
                 borderpad=0.5, bbox_transform=None):
        super().__init__(
            loc, pad=0., child=None, borderpad=borderpad,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform
        )

    def draw(self, renderer):
        raise RuntimeError("No draw method should be called")

    def __call__(self, ax, renderer):
        fig = ax.get_figure(root=False)
        if renderer is None:
            renderer = fig._get_renderer()
        self.axes = ax
        bbox = self.get_window_extent(renderer)
        px, py = self.get_offset(bbox.width, bbox.height, 0, 0, renderer)
        bbox_canvas = Bbox.from_bounds(px, py, bbox.width, bbox.height)
        tr = fig.transSubfigure.inverted()
        return TransformedBbox(bbox_canvas, tr)


class AnchoredSizeLocator(AnchoredLocatorBase):
    def __init__(self, bbox_to_anchor, x_size, y_size, loc,
                 borderpad=0.5, bbox_transform=None):
        super().__init__(
            bbox_to_anchor, None, loc,
            borderpad=borderpad, bbox_transform=bbox_transform
        )

        self.x_size = Size.from_any(x_size)
        self.y_size = Size.from_any(y_size)

    def get_bbox(self, renderer):
        bbox = self.get_bbox_to_anchor()
        dpi = renderer.points_to_pixels(72.)

        r, a = self.x_size.get_size(renderer)
        width = bbox.width * r + a * dpi
        r, a = self.y_size.get_size(renderer)
        height = bbox.height * r + a * dpi

        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize

        return Bbox.from_bounds(0, 0, width, height).padded(pad)


class AnchoredZoomLocator(AnchoredLocatorBase):
    def __init__(self, parent_axes, zoom, loc,
                 borderpad=0.5,
                 bbox_to_anchor=None,
                 bbox_transform=None):
        self.parent_axes = parent_axes
        self.zoom = zoom
        if bbox_to_anchor is None:
            bbox_to_anchor = parent_axes.bbox
        super().__init__(
            bbox_to_anchor, None, loc, borderpad=borderpad,
            bbox_transform=bbox_transform)

    def get_bbox(self, renderer):
        bb = self.parent_axes.transData.transform_bbox(self.axes.viewLim)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return (
            Bbox.from_bounds(
                0, 0, abs(bb.width * self.zoom), abs(bb.height * self.zoom))
            .padded(pad))


class BboxPatch(Patch):
    @_docstring.interpd
    def __init__(self, bbox, **kwargs):
        """
        Patch showing the shape bounded by a Bbox.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.Bbox`
            Bbox to use for the extents of this patch.

        **kwargs
            Patch properties. Valid arguments include:

            %(Patch:kwdoc)s
        """
        if "transform" in kwargs:
            raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        super().__init__(**kwargs)
        self.bbox = bbox

    def get_path(self):
        # docstring inherited
        x0, y0, x1, y1 = self.bbox.extents
        return Path._create_closed([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class BboxConnector(Patch):
    @staticmethod
    def get_bbox_edge_pos(bbox, loc):
        """
        Return the ``(x, y)`` coordinates of corner *loc* of *bbox*; parameters
        behave as documented for the `.BboxConnector` constructor.
        """
        x0, y0, x1, y1 = bbox.extents
        if loc == 1:
            return x1, y1
        elif loc == 2:
            return x0, y1
        elif loc == 3:
            return x0, y0
        elif loc == 4:
            return x1, y0

    @staticmethod
    def connect_bbox(bbox1, bbox2, loc1, loc2=None):
        """
        Construct a `.Path` connecting corner *loc1* of *bbox1* to corner
        *loc2* of *bbox2*, where parameters behave as documented as for the
        `.BboxConnector` constructor.
        """
        if isinstance(bbox1, Rectangle):
            bbox1 = TransformedBbox(Bbox.unit(), bbox1.get_transform())
        if isinstance(bbox2, Rectangle):
            bbox2 = TransformedBbox(Bbox.unit(), bbox2.get_transform())
        if loc2 is None:
            loc2 = loc1
        x1, y1 = BboxConnector.get_bbox_edge_pos(bbox1, loc1)
        x2, y2 = BboxConnector.get_bbox_edge_pos(bbox2, loc2)
        return Path([[x1, y1], [x2, y2]])

    @_docstring.interpd
    def __init__(self, bbox1, bbox2, loc1, loc2=None, **kwargs):
        """
        Connect two bboxes with a straight line.

        Parameters
        ----------
        bbox1, bbox2 : `~matplotlib.transforms.Bbox`
            Bounding boxes to connect.

        loc1, loc2 : {1, 2, 3, 4}
            Corner of *bbox1* and *bbox2* to draw the line. Valid values are::

                'upper right'  : 1,
                'upper left'   : 2,
                'lower left'   : 3,
                'lower right'  : 4

            *loc2* is optional and defaults to *loc1*.

        **kwargs
            Patch properties for the line drawn. Valid arguments include:

            %(Patch:kwdoc)s
        """
        if "transform" in kwargs:
            raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        kwargs.setdefault(
            "fill", bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
        super().__init__(**kwargs)
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.loc1 = loc1
        self.loc2 = loc2

    def get_path(self):
        # docstring inherited
        return self.connect_bbox(self.bbox1, self.bbox2,
                                 self.loc1, self.loc2)


class BboxConnectorPatch(BboxConnector):
    @_docstring.interpd
    def __init__(self, bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, **kwargs):
        """
        Connect two bboxes with a quadrilateral.

        The quadrilateral is specified by two lines that start and end at
        corners of the bboxes. The four sides of the quadrilateral are defined
        by the two lines given, the line between the two corners specified in
        *bbox1* and the line between the two corners specified in *bbox2*.

        Parameters
        ----------
        bbox1, bbox2 : `~matplotlib.transforms.Bbox`
            Bounding boxes to connect.

        loc1a, loc2a, loc1b, loc2b : {1, 2, 3, 4}
            The first line connects corners *loc1a* of *bbox1* and *loc2a* of
            *bbox2*; the second line connects corners *loc1b* of *bbox1* and
            *loc2b* of *bbox2*.  Valid values are::

                'upper right'  : 1,
                'upper left'   : 2,
                'lower left'   : 3,
                'lower right'  : 4

        **kwargs
            Patch properties for the line drawn:

            %(Patch:kwdoc)s
        """
        if "transform" in kwargs:
            raise ValueError("transform should not be set")
        super().__init__(bbox1, bbox2, loc1a, loc2a, **kwargs)
        self.loc1b = loc1b
        self.loc2b = loc2b

    def get_path(self):
        # docstring inherited
        path1 = self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2)
        path2 = self.connect_bbox(self.bbox2, self.bbox1,
                                  self.loc2b, self.loc1b)
        path_merged = [*path1.vertices, *path2.vertices, path1.vertices[0]]
        return Path(path_merged)


def _add_inset_axes(parent_axes, axes_class, axes_kwargs, axes_locator):
    """Helper function to add an inset axes and disable navigation in it."""
    if axes_class is None:
        axes_class = HostAxes
    if axes_kwargs is None:
        axes_kwargs = {}
    fig = parent_axes.get_figure(root=False)
    inset_axes = axes_class(
        fig, parent_axes.get_position(),
        **{"navigate": False, **axes_kwargs, "axes_locator": axes_locator})
    return fig.add_axes(inset_axes)


@_docstring.interpd
def inset_axes(parent_axes, width, height, loc='upper right',
               bbox_to_anchor=None, bbox_transform=None,
               axes_class=None, axes_kwargs=None,
               borderpad=0.5):
    """
    Create an inset axes with a given width and height.

    Both sizes used can be specified either in inches or percentage.
    For example,::

        inset_axes(parent_axes, width='40%%', height='30%%', loc='lower left')

    creates in inset axes in the lower left corner of *parent_axes* which spans
    over 30%% in height and 40%% in width of the *parent_axes*. Since the usage
    of `.inset_axes` may become slightly tricky when exceeding such standard
    cases, it is recommended to read :doc:`the examples
    </gallery/axes_grid1/inset_locator_demo>`.

    Notes
    -----
    The meaning of *bbox_to_anchor* and *bbox_to_transform* is interpreted
    differently from that of legend. The value of bbox_to_anchor
    (or the return value of its get_points method; the default is
    *parent_axes.bbox*) is transformed by the bbox_transform (the default
    is Identity transform) and then interpreted as points in the pixel
    coordinate (which is dpi dependent).

    Thus, following three calls are identical and creates an inset axes
    with respect to the *parent_axes*::

       axins = inset_axes(parent_axes, "30%%", "40%%")
       axins = inset_axes(parent_axes, "30%%", "40%%",
                          bbox_to_anchor=parent_axes.bbox)
       axins = inset_axes(parent_axes, "30%%", "40%%",
                          bbox_to_anchor=(0, 0, 1, 1),
                          bbox_transform=parent_axes.transAxes)

    Parameters
    ----------
    parent_axes : `matplotlib.axes.Axes`
        Axes to place the inset axes.

    width, height : float or str
        Size of the inset axes to create. If a float is provided, it is
        the size in inches, e.g. *width=1.3*. If a string is provided, it is
        the size in relative units, e.g. *width='40%%'*. By default, i.e. if
        neither *bbox_to_anchor* nor *bbox_transform* are specified, those
        are relative to the parent_axes. Otherwise, they are to be understood
        relative to the bounding box provided via *bbox_to_anchor*.

    loc : str, default: 'upper right'
        Location to place the inset axes.  Valid locations are
        'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center', 'lower right'.
        For backward compatibility, numeric values are accepted as well.
        See the parameter *loc* of `.Legend` for details.

    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional
        Bbox that the inset axes will be anchored to. If None,
        a tuple of (0, 0, 1, 1) is used if *bbox_transform* is set
        to *parent_axes.transAxes* or *parent_axes.figure.transFigure*.
        Otherwise, *parent_axes.bbox* is used. If a tuple, can be either
        [left, bottom, width, height], or [left, bottom].
        If the kwargs *width* and/or *height* are specified in relative units,
        the 2-tuple [left, bottom] cannot be used. Note that,
        unless *bbox_transform* is set, the units of the bounding box
        are interpreted in the pixel coordinate. When using *bbox_to_anchor*
        with tuple, it almost always makes sense to also specify
        a *bbox_transform*. This might often be the axes transform
        *parent_axes.transAxes*.

    bbox_transform : `~matplotlib.transforms.Transform`, optional
        Transformation for the bbox that contains the inset axes.
        If None, a `.transforms.IdentityTransform` is used. The value
        of *bbox_to_anchor* (or the return value of its get_points method)
        is transformed by the *bbox_transform* and then interpreted
        as points in the pixel coordinate (which is dpi dependent).
        You may provide *bbox_to_anchor* in some normalized coordinate,
        and give an appropriate transform (e.g., *parent_axes.transAxes*).

    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`
        The type of the newly created inset axes.

    axes_kwargs : dict, optional
        Keyword arguments to pass to the constructor of the inset axes.
        Valid arguments include:

        %(Axes:kwdoc)s

    borderpad : float, default: 0.5
        Padding between inset axes and the bbox_to_anchor.
        The units are axes font size, i.e. for a default font size of 10 points
        *borderpad = 0.5* is equivalent to a padding of 5 points.

    Returns
    -------
    inset_axes : *axes_class*
        Inset axes object created.
    """

    if (bbox_transform in [parent_axes.transAxes,
                           parent_axes.get_figure(root=False).transFigure]
            and bbox_to_anchor is None):
        _api.warn_external("Using the axes or figure transform requires a "
                           "bounding box in the respective coordinates. "
                           "Using bbox_to_anchor=(0, 0, 1, 1) now.")
        bbox_to_anchor = (0, 0, 1, 1)
    if bbox_to_anchor is None:
        bbox_to_anchor = parent_axes.bbox
    if (isinstance(bbox_to_anchor, tuple) and
            (isinstance(width, str) or isinstance(height, str))):
        if len(bbox_to_anchor) != 4:
            raise ValueError("Using relative units for width or height "
                             "requires to provide a 4-tuple or a "
                             "`Bbox` instance to `bbox_to_anchor.")
    return _add_inset_axes(
        parent_axes, axes_class, axes_kwargs,
        AnchoredSizeLocator(
            bbox_to_anchor, width, height, loc=loc,
            bbox_transform=bbox_transform, borderpad=borderpad))


@_docstring.interpd
def zoomed_inset_axes(parent_axes, zoom, loc='upper right',
                      bbox_to_anchor=None, bbox_transform=None,
                      axes_class=None, axes_kwargs=None,
                      borderpad=0.5):
    """
    Create an anchored inset axes by scaling a parent axes. For usage, also see
    :doc:`the examples </gallery/axes_grid1/inset_locator_demo2>`.

    Parameters
    ----------
    parent_axes : `~matplotlib.axes.Axes`
        Axes to place the inset axes.

    zoom : float
        Scaling factor of the data axes. *zoom* > 1 will enlarge the
        coordinates (i.e., "zoomed in"), while *zoom* < 1 will shrink the
        coordinates (i.e., "zoomed out").

    loc : str, default: 'upper right'
        Location to place the inset axes.  Valid locations are
        'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center', 'lower right'.
        For backward compatibility, numeric values are accepted as well.
        See the parameter *loc* of `.Legend` for details.

    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional
        Bbox that the inset axes will be anchored to. If None,
        *parent_axes.bbox* is used. If a tuple, can be either
        [left, bottom, width, height], or [left, bottom].
        If the kwargs *width* and/or *height* are specified in relative units,
        the 2-tuple [left, bottom] cannot be used. Note that
        the units of the bounding box are determined through the transform
        in use. When using *bbox_to_anchor* it almost always makes sense to
        also specify a *bbox_transform*. This might often be the axes transform
        *parent_axes.transAxes*.

    bbox_transform : `~matplotlib.transforms.Transform`, optional
        Transformation for the bbox that contains the inset axes.
        If None, a `.transforms.IdentityTransform` is used (i.e. pixel
        coordinates). This is useful when not providing any argument to
        *bbox_to_anchor*. When using *bbox_to_anchor* it almost always makes
        sense to also specify a *bbox_transform*. This might often be the
        axes transform *parent_axes.transAxes*. Inversely, when specifying
        the axes- or figure-transform here, be aware that not specifying
        *bbox_to_anchor* will use *parent_axes.bbox*, the units of which are
        in display (pixel) coordinates.

    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`
        The type of the newly created inset axes.

    axes_kwargs : dict, optional
        Keyword arguments to pass to the constructor of the inset axes.
        Valid arguments include:

        %(Axes:kwdoc)s

    borderpad : float, default: 0.5
        Padding between inset axes and the bbox_to_anchor.
        The units are axes font size, i.e. for a default font size of 10 points
        *borderpad = 0.5* is equivalent to a padding of 5 points.

    Returns
    -------
    inset_axes : *axes_class*
        Inset axes object created.
    """

    return _add_inset_axes(
        parent_axes, axes_class, axes_kwargs,
        AnchoredZoomLocator(
            parent_axes, zoom=zoom, loc=loc,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform,
            borderpad=borderpad))


class _TransformedBboxWithCallback(TransformedBbox):
    """
    Variant of `.TransformBbox` which calls *callback* before returning points.

    Used by `.mark_inset` to unstale the parent axes' viewlim as needed.
    """

    def __init__(self, *args, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback

    def get_points(self):
        self._callback()
        return super().get_points()


@_docstring.interpd
def mark_inset(parent_axes, inset_axes, loc1, loc2, **kwargs):
    """
    Draw a box to mark the location of an area represented by an inset axes.

    This function draws a box in *parent_axes* at the bounding box of
    *inset_axes*, and shows a connection with the inset axes by drawing lines
    at the corners, giving a "zoomed in" effect.

    Parameters
    ----------
    parent_axes : `~matplotlib.axes.Axes`
        Axes which contains the area of the inset axes.

    inset_axes : `~matplotlib.axes.Axes`
        The inset axes.

    loc1, loc2 : {1, 2, 3, 4}
        Corners to use for connecting the inset axes and the area in the
        parent axes.

    **kwargs
        Patch properties for the lines and box drawn:

        %(Patch:kwdoc)s

    Returns
    -------
    pp : `~matplotlib.patches.Patch`
        The patch drawn to represent the area of the inset axes.

    p1, p2 : `~matplotlib.patches.Patch`
        The patches connecting two corners of the inset axes and its area.
    """
    rect = _TransformedBboxWithCallback(
        inset_axes.viewLim, parent_axes.transData,
        callback=parent_axes._unstale_viewLim)

    kwargs.setdefault("fill", bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
    pp = BboxPatch(rect, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2
<|endoftext|>"""
Provides classes of simple units that will be used with `.AxesDivider`
class (or others) to determine the size of each Axes. The unit
classes define `get_size` method that returns a tuple of two floats,
meaning relative and absolute sizes, respectively.

Note that this class is nothing more than a simple tuple of two
floats. Take a look at the Divider class to see how these two
values are used.

Once created, the unit classes can be modified by simple arithmetic
operations: addition /subtraction with another unit type or a real number and scaling
(multiplication or division) by a real number.
"""

from numbers import Real

from matplotlib import _api
from matplotlib.axes import Axes


class _Base:
    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Fraction(other, self)

    def __div__(self, other):
        return (1 / other) * self

    def __add__(self, other):
        if isinstance(other, _Base):
            return Add(self, other)
        else:
            return Add(self, Fixed(other))

    def __neg__(self):
        return -1 * self

    def __radd__(self, other):
        # other cannot be a _Base instance, because A + B would trigger
        # A.__add__(B) first.
        return Add(self, Fixed(other))

    def __sub__(self, other):
        return self + (-other)

    def get_size(self, renderer):
        """
        Return two-float tuple with relative and absolute sizes.
        """
        raise NotImplementedError("Subclasses must implement")


class Add(_Base):
    """
    Sum of two sizes.
    """

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def get_size(self, renderer):
        a_rel_size, a_abs_size = self._a.get_size(renderer)
        b_rel_size, b_abs_size = self._b.get_size(renderer)
        return a_rel_size + b_rel_size, a_abs_size + b_abs_size


class Fixed(_Base):
    """
    Simple fixed size with absolute part = *fixed_size* and relative part = 0.
    """

    def __init__(self, fixed_size):
        _api.check_isinstance(Real, fixed_size=fixed_size)
        self.fixed_size = fixed_size

    def get_size(self, renderer):
        rel_size = 0.
        abs_size = self.fixed_size
        return rel_size, abs_size


class Scaled(_Base):
    """
    Simple scaled(?) size with absolute part = 0 and
    relative part = *scalable_size*.
    """

    def __init__(self, scalable_size):
        self._scalable_size = scalable_size

    def get_size(self, renderer):
        rel_size = self._scalable_size
        abs_size = 0.
        return rel_size, abs_size

Scalable = Scaled


def _get_axes_aspect(ax):
    aspect = ax.get_aspect()
    if aspect == "auto":
        aspect = 1.
    return aspect


class AxesX(_Base):
    """
    Scaled size whose relative part corresponds to the data width
    of the *axes* multiplied by the *aspect*.
    """

    def __init__(self, axes, aspect=1., ref_ax=None):
        self._axes = axes
        self._aspect = aspect
        if aspect == "axes" and ref_ax is None:
            raise ValueError("ref_ax must be set when aspect='axes'")
        self._ref_ax = ref_ax

    def get_size(self, renderer):
        l1, l2 = self._axes.get_xlim()
        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)
            aspect = ref_aspect / _get_axes_aspect(self._axes)
        else:
            aspect = self._aspect

        rel_size = abs(l2-l1)*aspect
        abs_size = 0.
        return rel_size, abs_size


class AxesY(_Base):
    """
    Scaled size whose relative part corresponds to the data height
    of the *axes* multiplied by the *aspect*.
    """

    def __init__(self, axes, aspect=1., ref_ax=None):
        self._axes = axes
        self._aspect = aspect
        if aspect == "axes" and ref_ax is None:
            raise ValueError("ref_ax must be set when aspect='axes'")
        self._ref_ax = ref_ax

    def get_size(self, renderer):
        l1, l2 = self._axes.get_ylim()

        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)
            aspect = _get_axes_aspect(self._axes)
        else:
            aspect = self._aspect

        rel_size = abs(l2-l1)*aspect
        abs_size = 0.
        return rel_size, abs_size


class MaxExtent(_Base):
    """
    Size whose absolute part is either the largest width or the largest height
    of the given *artist_list*.
    """

    def __init__(self, artist_list, w_or_h):
        self._artist_list = artist_list
        _api.check_in_list(["width", "height"], w_or_h=w_or_h)
        self._w_or_h = w_or_h

    def add_artist(self, a):
        self._artist_list.append(a)

    def get_size(self, renderer):
        rel_size = 0.
        extent_list = [
            getattr(a.get_window_extent(renderer), self._w_or_h) / a.figure.dpi
            for a in self._artist_list]
        abs_size = max(extent_list, default=0)
        return rel_size, abs_size


class MaxWidth(MaxExtent):
    """
    Size whose absolute part is the largest width of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, "width")


class MaxHeight(MaxExtent):
    """
    Size whose absolute part is the largest height of the given *artist_list*.
    """

    def __init__(self, artist_list):
        super().__init__(artist_list, "height")


class Fraction(_Base):
    """
    An instance whose size is a *fraction* of the *ref_size*.

    >>> s = Fraction(0.3, AxesX(ax))
    """

    def __init__(self, fraction, ref_size):
        _api.check_isinstance(Real, fraction=fraction)
        self._fraction_ref = ref_size
        self._fraction = fraction

    def get_size(self, renderer):
        if self._fraction_ref is None:
            return self._fraction, 0.
        else:
            r, a = self._fraction_ref.get_size(renderer)
            rel_size = r*self._fraction
            abs_size = a*self._fraction
            return rel_size, abs_size


def from_any(size, fraction_ref=None):
    """
    Create a Fixed unit when the first argument is a float, or a
    Fraction unit if that is a string that ends with %. The second
    argument is only meaningful when Fraction unit is created.

    >>> from mpl_toolkits.axes_grid1.axes_size import from_any
    >>> a = from_any(1.2) # => Fixed(1.2)
    >>> from_any("50%", a) # => Fraction(0.5, a)
    """
    if isinstance(size, Real):
        return Fixed(size)
    elif isinstance(size, str):
        if size[-1] == "%":
            return Fraction(float(size[:-1]) / 100, fraction_ref)
    raise ValueError("Unknown format")


class _AxesDecorationsSize(_Base):
    """
    Fixed size, corresponding to the size of decorations on a given Axes side.
    """

    _get_size_map = {
        "left":   lambda tight_bb, axes_bb: axes_bb.xmin - tight_bb.xmin,
        "right":  lambda tight_bb, axes_bb: tight_bb.xmax - axes_bb.xmax,
        "bottom": lambda tight_bb, axes_bb: axes_bb.ymin - tight_bb.ymin,
        "top":    lambda tight_bb, axes_bb: tight_bb.ymax - axes_bb.ymax,
    }

    def __init__(self, ax, direction):
        _api.check_in_list(self._get_size_map, direction=direction)
        self._direction = direction
        self._ax_list = [ax] if isinstance(ax, Axes) else ax

    def get_size(self, renderer):
        sz = max([
            self._get_size_map[self._direction](
                ax.get_tightbbox(renderer, call_axes_locator=False), ax.bbox)
            for ax in self._ax_list])
        dpi = renderer.points_to_pixels(72)
        abs_size = sz / dpi
        rel_size = 0
        return rel_size, abs_size
<|endoftext|>"""
Helper classes to adjust the positions of multiple axes at drawing time.
"""

import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size


class Divider:
    """
    An Axes positioning class.

    The divider is initialized with lists of horizontal and vertical sizes
    (:mod:`mpl_toolkits.axes_grid1.axes_size`) based on which a given
    rectangular area will be divided.

    The `new_locator` method then creates a callable object
    that can be used as the *axes_locator* of the axes.
    """

    def __init__(self, fig, pos, horizontal, vertical,
                 aspect=None, anchor="C"):
        """
        Parameters
        ----------
        fig : Figure
        pos : tuple of 4 floats
            Position of the rectangle that will be divided.
        horizontal : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            Sizes for horizontal division.
        vertical : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            Sizes for vertical division.
        aspect : bool, optional
            Whether overall rectangular area is reduced so that the relative
            part of the horizontal and vertical scales have the same scale.
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
'NW', 'W'}, default: 'C'
            Placement of the reduced rectangle, when *aspect* is True.
        """

        self._fig = fig
        self._pos = pos
        self._horizontal = horizontal
        self._vertical = vertical
        self._anchor = anchor
        self.set_anchor(anchor)
        self._aspect = aspect
        self._xrefindex = 0
        self._yrefindex = 0
        self._locator = None

    def get_horizontal_sizes(self, renderer):
        return np.array([s.get_size(renderer) for s in self.get_horizontal()])

    def get_vertical_sizes(self, renderer):
        return np.array([s.get_size(renderer) for s in self.get_vertical()])

    def set_position(self, pos):
        """
        Set the position of the rectangle.

        Parameters
        ----------
        pos : tuple of 4 floats
            position of the rectangle that will be divided
        """
        self._pos = pos

    def get_position(self):
        """Return the position of the rectangle."""
        return self._pos

    def set_anchor(self, anchor):
        """
        Parameters
        ----------
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
'NW', 'W'}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).

        See Also
        --------
        .Axes.set_anchor
        """
        if isinstance(anchor, str):
            _api.check_in_list(mtransforms.Bbox.coefs, anchor=anchor)
        elif not isinstance(anchor, (tuple, list)) or len(anchor) != 2:
            raise TypeError("anchor must be str or 2-tuple")
        self._anchor = anchor

    def get_anchor(self):
        """Return the anchor."""
        return self._anchor

    def get_subplotspec(self):
        return None

    def set_horizontal(self, h):
        """
        Parameters
        ----------
        h : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            sizes for horizontal division
        """
        self._horizontal = h

    def get_horizontal(self):
        """Return horizontal sizes."""
        return self._horizontal

    def set_vertical(self, v):
        """
        Parameters
        ----------
        v : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`
            sizes for vertical division
        """
        self._vertical = v

    def get_vertical(self):
        """Return vertical sizes."""
        return self._vertical

    def set_aspect(self, aspect=False):
        """
        Parameters
        ----------
        aspect : bool
        """
        self._aspect = aspect

    def get_aspect(self):
        """Return aspect."""
        return self._aspect

    def set_locator(self, _locator):
        self._locator = _locator

    def get_locator(self):
        return self._locator

    def get_position_runtime(self, ax, renderer):
        if self._locator is None:
            return self.get_position()
        else:
            return self._locator(ax, renderer).bounds

    @staticmethod
    def _calc_k(sizes, total):
        # sizes is a (n, 2) array of (rel_size, abs_size); this method finds
        # the k factor such that sum(rel_size * k + abs_size) == total.
        rel_sum, abs_sum = sizes.sum(0)
        return (total - abs_sum) / rel_sum if rel_sum else 0

    @staticmethod
    def _calc_offsets(sizes, k):
        # Apply k factors to (n, 2) sizes array of (rel_size, abs_size); return
        # the resulting cumulative offset positions.
        return np.cumsum([0, *(sizes @ [k, 1])])

    def new_locator(self, nx, ny, nx1=None, ny1=None):
        """
        Return an axes locator callable for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        """
        if nx1 is None:
            nx1 = nx + 1
        if ny1 is None:
            ny1 = ny + 1
        # append_size("left") adds a new size at the beginning of the
        # horizontal size lists; this shift transforms e.g.
        # new_locator(nx=2, ...) into effectively new_locator(nx=3, ...).  To
        # take that into account, instead of recording nx, we record
        # nx-self._xrefindex, where _xrefindex is shifted by 1 by each
        # append_size("left"), and re-add self._xrefindex back to nx in
        # _locate, when the actual axes position is computed.  Ditto for y.
        xref = self._xrefindex
        yref = self._yrefindex
        locator = functools.partial(
            self._locate, nx - xref, ny - yref, nx1 - xref, ny1 - yref)
        locator.get_subplotspec = self.get_subplotspec
        return locator

    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        """
        Implementation of ``divider.new_locator().__call__``.

        The axes locator callable returned by ``new_locator()`` is created as
        a `functools.partial` of this method with *nx*, *ny*, *nx1*, and *ny1*
        specifying the requested cell.
        """
        nx += self._xrefindex
        nx1 += self._xrefindex
        ny += self._yrefindex
        ny1 += self._yrefindex

        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)

        hsizes = self.get_horizontal_sizes(renderer)
        vsizes = self.get_vertical_sizes(renderer)
        k_h = self._calc_k(hsizes, fig_w * w)
        k_v = self._calc_k(vsizes, fig_h * h)

        if self.get_aspect():
            k = min(k_h, k_v)
            ox = self._calc_offsets(hsizes, k)
            oy = self._calc_offsets(vsizes, k)

            ww = (ox[-1] - ox[0]) / fig_w
            hh = (oy[-1] - oy[0]) / fig_h
            pb = mtransforms.Bbox.from_bounds(x, y, w, h)
            pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
            x0, y0 = pb1.anchored(self.get_anchor(), pb).p0

        else:
            ox = self._calc_offsets(hsizes, k_h)
            oy = self._calc_offsets(vsizes, k_v)
            x0, y0 = x, y

        if nx1 is None:
            nx1 = -1
        if ny1 is None:
            ny1 = -1

        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h

        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)

    def append_size(self, position, size):
        _api.check_in_list(["left", "right", "bottom", "top"],
                           position=position)
        if position == "left":
            self._horizontal.insert(0, size)
            self._xrefindex += 1
        elif position == "right":
            self._horizontal.append(size)
        elif position == "bottom":
            self._vertical.insert(0, size)
            self._yrefindex += 1
        else:  # 'top'
            self._vertical.append(size)

    def add_auto_adjustable_area(self, use_axes, pad=0.1, adjust_dirs=None):
        """
        Add auto-adjustable padding around *use_axes* to take their decorations
        (title, labels, ticks, ticklabels) into account during layout.

        Parameters
        ----------
        use_axes : `~matplotlib.axes.Axes` or list of `~matplotlib.axes.Axes`
            The Axes whose decorations are taken into account.
        pad : float, default: 0.1
            Additional padding in inches.
        adjust_dirs : list of {"left", "right", "bottom", "top"}, optional
            The sides where padding is added; defaults to all four sides.
        """
        if adjust_dirs is None:
            adjust_dirs = ["left", "right", "bottom", "top"]
        for d in adjust_dirs:
            self.append_size(d, Size._AxesDecorationsSize(use_axes, d) + pad)


class SubplotDivider(Divider):
    """
    The Divider class whose rectangle area is specified as a subplot geometry.
    """

    def __init__(self, fig, *args, horizontal=None, vertical=None,
                 aspect=None, anchor='C'):
        """
        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`

        *args : tuple (*nrows*, *ncols*, *index*) or int
            The array of subplots in the figure has dimensions ``(nrows,
            ncols)``, and *index* is the index of the subplot being created.
            *index* starts at 1 in the upper left corner and increases to the
            right.

            If *nrows*, *ncols*, and *index* are all single digit numbers, then
            *args* can be passed as a single 3-digit number (e.g. 234 for
            (2, 3, 4)).
        horizontal : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`, optional
            Sizes for horizontal division.
        vertical : list of :mod:`~mpl_toolkits.axes_grid1.axes_size`, optional
            Sizes for vertical division.
        aspect : bool, optional
            Whether overall rectangular area is reduced so that the relative
            part of the horizontal and vertical scales have the same scale.
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', 'N', \
'NW', 'W'}, default: 'C'
            Placement of the reduced rectangle, when *aspect* is True.
        """
        self.figure = fig
        super().__init__(fig, [0, 0, 1, 1],
                         horizontal=horizontal or [], vertical=vertical or [],
                         aspect=aspect, anchor=anchor)
        self.set_subplotspec(SubplotSpec._from_subplot_args(fig, args))

    def get_position(self):
        """Return the bounds of the subplot box."""
        return self.get_subplotspec().get_position(self.figure).bounds

    def get_subplotspec(self):
        """Get the SubplotSpec instance."""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the SubplotSpec instance."""
        self._subplotspec = subplotspec
        self.set_position(subplotspec.get_position(self.figure))


class AxesDivider(Divider):
    """
    Divider based on the preexisting axes.
    """

    def __init__(self, axes, xref=None, yref=None):
        """
        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
        xref
        yref
        """
        self._axes = axes
        if xref is None:
            self._xref = Size.AxesX(axes)
        else:
            self._xref = xref
        if yref is None:
            self._yref = Size.AxesY(axes)
        else:
            self._yref = yref

        super().__init__(fig=axes.get_figure(), pos=None,
                         horizontal=[self._xref], vertical=[self._yref],
                         aspect=None, anchor="C")

    def _get_new_axes(self, *, axes_class=None, **kwargs):
        axes = self._axes
        if axes_class is None:
            axes_class = type(axes)
        return axes_class(axes.get_figure(), axes.get_position(original=True),
                          **kwargs)

    def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method for ``append_axes("left")`` and ``append_axes("right")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
        if pad is None:
            pad = mpl.rcParams["figure.subplot.wspace"] * self._xref
        pos = "left" if pack_start else "right"
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._xref)
            self.append_size(pos, pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._xref)
        self.append_size(pos, size)
        locator = self.new_locator(
            nx=0 if pack_start else len(self._horizontal) - 1,
            ny=self._yrefindex)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method for ``append_axes("top")`` and ``append_axes("bottom")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
        if pad is None:
            pad = mpl.rcParams["figure.subplot.hspace"] * self._yref
        pos = "bottom" if pack_start else "top"
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._yref)
            self.append_size(pos, pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._yref)
        self.append_size(pos, size)
        locator = self.new_locator(
            nx=self._xrefindex,
            ny=0 if pack_start else len(self._vertical) - 1)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    def append_axes(self, position, size, pad=None, *, axes_class=None,
                    **kwargs):
        """
        Add a new axes on a given side of the main axes.

        Parameters
        ----------
        position : {"left", "right", "bottom", "top"}
            Where the new axes is positioned relative to the main axes.
        size : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            The axes width or height.  float or str arguments are interpreted
            as ``axes_size.from_any(size, AxesX(<main_axes>))`` for left or
            right axes, and likewise with ``AxesY`` for bottom or top axes.
        pad : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            Padding between the axes.  float or str arguments are interpreted
            as for *size*.  Defaults to :rc:`figure.subplot.wspace` times the
            main Axes width (left or right axes) or :rc:`figure.subplot.hspace`
            times the main Axes height (bottom or top axes).
        axes_class : subclass type of `~.axes.Axes`, optional
            The type of the new axes.  Defaults to the type of the main axes.
        **kwargs
            All extra keywords arguments are passed to the created axes.
        """
        create_axes, pack_start = _api.check_getitem({
            "left": (self.new_horizontal, True),
            "right": (self.new_horizontal, False),
            "bottom": (self.new_vertical, True),
            "top": (self.new_vertical, False),
        }, position=position)
        ax = create_axes(
            size, pad, pack_start=pack_start, axes_class=axes_class, **kwargs)
        self._fig.add_axes(ax)
        return ax

    def get_aspect(self):
        if self._aspect is None:
            aspect = self._axes.get_aspect()
            if aspect == "auto":
                return False
            else:
                return True
        else:
            return self._aspect

    def get_position(self):
        if self._pos is None:
            bbox = self._axes.get_position(original=True)
            return bbox.bounds
        else:
            return self._pos

    def get_anchor(self):
        if self._anchor is None:
            return self._axes.get_anchor()
        else:
            return self._anchor

    def get_subplotspec(self):
        return self._axes.get_subplotspec()


# Helper for HBoxDivider/VBoxDivider.
# The variable names are written for a horizontal layout, but the calculations
# work identically for vertical layouts.
def _locate(x, y, w, h, summed_widths, equal_heights, fig_w, fig_h, anchor):

    total_width = fig_w * w
    max_height = fig_h * h

    # Determine the k factors.
    n = len(equal_heights)
    eq_rels, eq_abss = equal_heights.T
    sm_rels, sm_abss = summed_widths.T
    A = np.diag([*eq_rels, 0])
    A[:n, -1] = -1
    A[-1, :-1] = sm_rels
    B = [*(-eq_abss), total_width - sm_abss.sum()]
    # A @ K = B: This finds factors {k_0, ..., k_{N-1}, H} so that
    #   eq_rel_i * k_i + eq_abs_i = H for all i: all axes have the same height
    #   sum(sm_rel_i * k_i + sm_abs_i) = total_width: fixed total width
    # (foo_rel_i * k_i + foo_abs_i will end up being the size of foo.)
    *karray, height = np.linalg.solve(A, B)
    if height > max_height:  # Additionally, upper-bound the height.
        karray = (max_height - eq_abss) / eq_rels

    # Compute the offsets corresponding to these factors.
    ox = np.cumsum([0, *(sm_rels * karray + sm_abss)])
    ww = (ox[-1] - ox[0]) / fig_w
    h0_rel, h0_abs = equal_heights[0]
    hh = (karray[0]*h0_rel + h0_abs) / fig_h
    pb = mtransforms.Bbox.from_bounds(x, y, w, h)
    pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
    x0, y0 = pb1.anchored(anchor, pb).p0

    return x0, y0, ox, hh


class HBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes horizontally, while ensuring that
    they have equal heights.

    Examples
    --------
    .. plot:: gallery/axes_grid1/demo_axes_hbox_divider.py
    """

    def new_locator(self, nx, nx1=None):
        """
        Create an axes locator callable for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        """
        return super().new_locator(nx, 0, nx1, 0)

    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        # docstring inherited
        nx += self._xrefindex
        nx1 += self._xrefindex
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_ws = self.get_horizontal_sizes(renderer)
        equal_hs = self.get_vertical_sizes(renderer)
        x0, y0, ox, hh = _locate(
            x, y, w, h, summed_ws, equal_hs, fig_w, fig_h, self.get_anchor())
        if nx1 is None:
            nx1 = -1
        x1, w1 = x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w
        y1, h1 = y0, hh
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


class VBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes vertically, while ensuring that
    they have equal widths.
    """

    def new_locator(self, ny, ny1=None):
        """
        Create an axes locator callable for the specified cell.

        Parameters
        ----------
        ny, ny1 : int
            Integers specifying the row-position of the
            cell. When *ny1* is None, a single *ny*-th row is
            specified. Otherwise, location of rows spanning between *ny*
            to *ny1* (but excluding *ny1*-th row) is specified.
        """
        return super().new_locator(0, ny, 0, ny1)

    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        # docstring inherited
        ny += self._yrefindex
        ny1 += self._yrefindex
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_hs = self.get_vertical_sizes(renderer)
        equal_ws = self.get_horizontal_sizes(renderer)
        y0, x0, oy, ww = _locate(
            y, x, h, w, summed_hs, equal_ws, fig_h, fig_w, self.get_anchor())
        if ny1 is None:
            ny1 = -1
        x1, w1 = x0, ww
        y1, h1 = y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


def make_axes_locatable(axes):
    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    return divider


def make_axes_area_auto_adjustable(
        ax, use_axes=None, pad=0.1, adjust_dirs=None):
    """
    Add auto-adjustable padding around *ax* to take its decorations (title,
    labels, ticks, ticklabels) into account during layout, using
    `.Divider.add_auto_adjustable_area`.

    By default, padding is determined from the decorations of *ax*.
    Pass *use_axes* to consider the decorations of other Axes instead.
    """
    if adjust_dirs is None:
        adjust_dirs = ["left", "right", "bottom", "top"]
    divider = make_axes_locatable(ax)
    if use_axes is None:
        use_axes = ax
    divider.add_auto_adjustable_area(use_axes=use_axes, pad=pad,
                                     adjust_dirs=adjust_dirs)
<|endoftext|>"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible for drawing the axis spine,
ticks, ticklabels and axis labels are separated out from Matplotlib's Axis
class. Originally, this change was motivated to support curvilinear
grid. Here are a few reasons that I came up with a new axes class:

* "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
  different ticks (tick locations and labels). This is not possible
  with the current Matplotlib, although some twin axes trick can help.

* Curvilinear grid.

* angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and has the following
children artists which will draw ticks, labels, etc.

* line
* major_ticks, major_ticklabels
* minor_ticks, minor_ticklabels
* offsetText
* label

Note that these are separate artists from `matplotlib.axis.Axis`, thus most
tick-related functions in Matplotlib won't work. For example, color and
markerwidth of the ``ax.axis["bottom"].major_ticks`` will follow those of
Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection


class _AxisArtistHelperBase:
    """
    Base class for axis helper.

    Subclasses should define the methods listed below.  The *axes*
    argument will be the ``.axes`` attribute of the caller artist. ::

        # Construct the spine.

        def get_line_transform(self, axes):
            return transform

        def get_line(self, axes):
            return path

        # Construct the label.

        def get_axislabel_transform(self, axes):
            return transform

        def get_axislabel_pos_angle(self, axes):
            return (x, y), angle

        # Construct the ticks.

        def get_tick_transform(self, axes):
            return transform

        def get_tick_iterators(self, axes):
            # A pair of iterables (one for major ticks, one for minor ticks)
            # that yield (tick_position, tick_angle, tick_label).
            return iter_major, iter_minor
    """

    def __init__(self, nth_coord):
        self.nth_coord = nth_coord

    def update_lim(self, axes):
        pass

    def get_nth_coord(self):
        return self.nth_coord

    def _to_xy(self, values, const):
        """
        Create a (*values.shape, 2)-shape array representing (x, y) pairs.

        The other coordinate is filled with the constant *const*.

        Example::

            >>> self.nth_coord = 0
            >>> self._to_xy([1, 2, 3], const=0)
            array([[1, 0],
                   [2, 0],
                   [3, 0]])
        """
        if self.nth_coord == 0:
            return np.stack(np.broadcast_arrays(values, const), axis=-1)
        elif self.nth_coord == 1:
            return np.stack(np.broadcast_arrays(const, values), axis=-1)
        else:
            raise ValueError("Unexpected nth_coord")


class _FixedAxisArtistHelperBase(_AxisArtistHelperBase):
    """Helper class for a fixed (in the axes coordinate) axis."""

    @_api.delete_parameter("3.9", "nth_coord")
    def __init__(self, loc, nth_coord=None):
        """``nth_coord = 0``: x-axis; ``nth_coord = 1``: y-axis."""
        super().__init__(_api.check_getitem(
            {"bottom": 0, "top": 0, "left": 1, "right": 1}, loc=loc))
        self._loc = loc
        self._pos = {"bottom": 0, "top": 1, "left": 0, "right": 1}[loc]
        # axis line in transAxes
        self._path = Path(self._to_xy((0, 1), const=self._pos))

    # LINE

    def get_line(self, axes):
        return self._path

    def get_line_transform(self, axes):
        return axes.transAxes

    # LABEL

    def get_axislabel_transform(self, axes):
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        """
        Return the label reference position in transAxes.

        get_label_transform() returns a transform of (transAxes+offset)
        """
        return dict(left=((0., 0.5), 90),  # (position, angle_tangent)
                    right=((1., 0.5), 90),
                    bottom=((0.5, 0.), 0),
                    top=((0.5, 1.), 0))[self._loc]

    # TICK

    def get_tick_transform(self, axes):
        return [axes.get_xaxis_transform(), axes.get_yaxis_transform()][self.nth_coord]


class _FloatingAxisArtistHelperBase(_AxisArtistHelperBase):
    def __init__(self, nth_coord, value):
        self._value = value
        super().__init__(nth_coord)

    def get_line(self, axes):
        raise RuntimeError("get_line method should be defined by the derived class")


class FixedAxisArtistHelperRectilinear(_FixedAxisArtistHelperBase):

    @_api.delete_parameter("3.9", "nth_coord")
    def __init__(self, axes, loc, nth_coord=None):
        """
        nth_coord = along which coordinate value varies
        in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        super().__init__(loc)
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    # TICK

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        angle_normal, angle_tangent = {0: (90, 0), 1: (0, 90)}[self.nth_coord]

        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)

        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)

        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes

        def _f(locs, labels):
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._pos)
                # check if the tick point is inside axes
                c2 = tick_to_axes.transform(c)
                if mpl.transforms._interval_contains_close((0, 1), c2[self.nth_coord]):
                    yield c, angle_normal, angle_tangent, label

        return _f(major_locs, major_labels), _f(minor_locs, minor_labels)


class FloatingAxisArtistHelperRectilinear(_FloatingAxisArtistHelperBase):

    def __init__(self, axes, nth_coord,
                 passingthrough_point, axis_direction="bottom"):
        super().__init__(nth_coord, passingthrough_point)
        self._axis_direction = axis_direction
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    def get_line(self, axes):
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        return Path(self._to_xy((0, 1), const=p[fixed_coord]))

    def get_line_transform(self, axes):
        return axes.transAxes

    def get_axislabel_transform(self, axes):
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        """
        Return the label reference position in transAxes.

        get_label_transform() returns a transform of (transAxes+offset)
        """
        angle = [0, 90][self.nth_coord]
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        verts = self._to_xy(0.5, const=p[fixed_coord])
        return (verts, angle) if 0 <= verts[fixed_coord] <= 1 else (None, None)

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        angle_normal, angle_tangent = {0: (90, 0), 1: (0, 90)}[self.nth_coord]

        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)

        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)

        data_to_axes = axes.transData - axes.transAxes

        def _f(locs, labels):
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._value)
                c1, c2 = data_to_axes.transform(c)
                if 0 <= c1 <= 1 and 0 <= c2 <= 1:
                    yield c, angle_normal, angle_tangent, label

        return _f(major_locs, major_labels), _f(minor_locs, minor_labels)


class AxisArtistHelper:  # Backcompat.
    Fixed = _FixedAxisArtistHelperBase
    Floating = _FloatingAxisArtistHelperBase


class AxisArtistHelperRectlinear:  # Backcompat.
    Fixed = FixedAxisArtistHelperRectilinear
    Floating = FloatingAxisArtistHelperRectilinear


class GridHelperBase:

    def __init__(self):
        self._old_limits = None
        super().__init__()

    def update_lim(self, axes):
        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        if self._old_limits != (x1, x2, y1, y2):
            self._update_grid(x1, y1, x2, y2)
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        """Cache relevant computations when the axes limits have changed."""

    def get_gridlines(self, which, axis):
        """
        Return list of grid lines as a list of paths (list of points).

        Parameters
        ----------
        which : {"both", "major", "minor"}
        axis : {"both", "x", "y"}
        """
        return []


class GridHelperRectlinear(GridHelperBase):

    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    @_api.delete_parameter(
        "3.9", "nth_coord", addendum="'nth_coord' is now inferred from 'loc'.")
    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        if axes is None:
            _api.warn_external(
                "'new_fixed_axis' explicitly requires the axes keyword.")
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        return AxisArtist(axes, FixedAxisArtistHelperRectilinear(axes, loc),
                          offset=offset, axis_direction=axis_direction)

    def new_floating_axis(self, nth_coord, value, axis_direction="bottom", axes=None):
        if axes is None:
            _api.warn_external(
                "'new_floating_axis' explicitly requires the axes keyword.")
            axes = self.axes
        helper = FloatingAxisArtistHelperRectilinear(
            axes, nth_coord, value, axis_direction)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    def get_gridlines(self, which="major", axis="both"):
        """
        Return list of gridline coordinates in data coordinates.

        Parameters
        ----------
        which : {"both", "major", "minor"}
        axis : {"both", "x", "y"}
        """
        _api.check_in_list(["both", "major", "minor"], which=which)
        _api.check_in_list(["both", "x", "y"], axis=axis)
        gridlines = []

        if axis in ("both", "x"):
            locs = []
            y1, y2 = self.axes.get_ylim()
            if which in ("both", "major"):
                locs.extend(self.axes.xaxis.major.locator())
            if which in ("both", "minor"):
                locs.extend(self.axes.xaxis.minor.locator())
            gridlines.extend([[x, x], [y1, y2]] for x in locs)

        if axis in ("both", "y"):
            x1, x2 = self.axes.get_xlim()
            locs = []
            if self.axes.yaxis._major_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.major.locator())
            if self.axes.yaxis._minor_tick_kw["gridOn"]:
                locs.extend(self.axes.yaxis.minor.locator())
            gridlines.extend([[x1, x2], [y, y]] for y in locs)

        return gridlines


class Axes(maxes.Axes):

    def __init__(self, *args, grid_helper=None, **kwargs):
        self._axisline_on = True
        self._grid_helper = grid_helper if grid_helper else GridHelperRectlinear(self)
        super().__init__(*args, **kwargs)
        self.toggle_axisline(True)

    def toggle_axisline(self, b=None):
        if b is None:
            b = not self._axisline_on
        if b:
            self._axisline_on = True
            self.spines[:].set_visible(False)
            self.xaxis.set_visible(False)
            self.yaxis.set_visible(False)
        else:
            self._axisline_on = False
            self.spines[:].set_visible(True)
            self.xaxis.set_visible(True)
            self.yaxis.set_visible(True)

    @property
    def axis(self):
        return self._axislines

    def clear(self):
        # docstring inherited

        # Init gridlines before clear() as clear() calls grid().
        self.gridlines = gridlines = GridlinesCollection(
            [],
            colors=mpl.rcParams['grid.color'],
            linestyles=mpl.rcParams['grid.linestyle'],
            linewidths=mpl.rcParams['grid.linewidth'])
        self._set_artist_props(gridlines)
        gridlines.set_grid_helper(self.get_grid_helper())

        super().clear()

        # clip_path is set after Axes.clear(): that's when a patch is created.
        gridlines.set_clip_path(self.axes.patch)

        # Init axis artists.
        self._axislines = mpl_axes.Axes.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        self._axislines.update({
            loc: new_fixed_axis(loc=loc, axes=self, axis_direction=loc)
            for loc in ["bottom", "top", "left", "right"]})
        for axisline in [self._axislines["top"], self._axislines["right"]]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    def get_grid_helper(self):
        return self._grid_helper

    def grid(self, visible=None, which='major', axis="both", **kwargs):
        """
        Toggle the gridlines, and optionally set the properties of the lines.
        """
        # There are some discrepancies in the behavior of grid() between
        # axes_grid and Matplotlib, because axes_grid explicitly sets the
        # visibility of the gridlines.
        super().grid(visible, which=which, axis=axis, **kwargs)
        if not self._axisline_on:
            return
        if visible is None:
            visible = (self.axes.xaxis._minor_tick_kw["gridOn"]
                       or self.axes.xaxis._major_tick_kw["gridOn"]
                       or self.axes.yaxis._minor_tick_kw["gridOn"]
                       or self.axes.yaxis._major_tick_kw["gridOn"])
        self.gridlines.set(which=which, axis=axis, visible=visible)
        self.gridlines.set(**kwargs)

    def get_children(self):
        if self._axisline_on:
            children = [*self._axislines.values(), self.gridlines]
        else:
            children = []
        children.extend(super().get_children())
        return children

    def new_fixed_axis(self, loc, offset=None):
        return self.get_grid_helper().new_fixed_axis(loc, offset=offset, axes=self)

    def new_floating_axis(self, nth_coord, value, axis_direction="bottom"):
        return self.get_grid_helper().new_floating_axis(
            nth_coord, value, axis_direction=axis_direction, axes=self)


class AxesZero(Axes):

    def clear(self):
        super().clear()
        new_floating_axis = self.get_grid_helper().new_floating_axis
        self._axislines.update(
            xzero=new_floating_axis(
                nth_coord=0, value=0., axis_direction="bottom", axes=self),
            yzero=new_floating_axis(
                nth_coord=1, value=0., axis_direction="left", axes=self),
        )
        for k in ["xzero", "yzero"]:
            self._axislines[k].line.set_clip_path(self.patch)
            self._axislines[k].set_visible(False)


Subplot = Axes
SubplotZero = AxesZero
<|endoftext|>"""
Provides classes to style the axis lines.
"""
import math

import numpy as np

import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform


class _FancyAxislineStyle:
    class SimpleArrow(FancyArrowPatch):
        """The artist class that will be returned for SimpleArrow style."""
        _ARROW_STYLE = "->"

        def __init__(self, axis_artist, line_path, transform,
                     line_mutation_scale):
            self._axis_artist = axis_artist
            self._line_transform = transform
            self._line_path = line_path
            self._line_mutation_scale = line_mutation_scale

            FancyArrowPatch.__init__(self,
                                     path=self._line_path,
                                     arrowstyle=self._ARROW_STYLE,
                                     patchA=None,
                                     patchB=None,
                                     shrinkA=0.,
                                     shrinkB=0.,
                                     mutation_scale=line_mutation_scale,
                                     mutation_aspect=None,
                                     transform=IdentityTransform(),
                                     )

        def set_line_mutation_scale(self, scale):
            self.set_mutation_scale(scale*self._line_mutation_scale)

        def _extend_path(self, path, mutation_size=10):
            """
            Extend the path to make a room for drawing arrow.
            """
            (x0, y0), (x1, y1) = path.vertices[-2:]
            theta = math.atan2(y1 - y0, x1 - x0)
            x2 = x1 + math.cos(theta) * mutation_size
            y2 = y1 + math.sin(theta) * mutation_size
            if path.codes is None:
                return Path(np.concatenate([path.vertices, [[x2, y2]]]))
            else:
                return Path(np.concatenate([path.vertices, [[x2, y2]]]),
                            np.concatenate([path.codes, [Path.LINETO]]))

        def set_path(self, path):
            self._line_path = path

        def draw(self, renderer):
            """
            Draw the axis line.
             1) Transform the path to the display coordinate.
             2) Extend the path to make a room for arrow.
             3) Update the path of the FancyArrowPatch.
             4) Draw.
            """
            path_in_disp = self._line_transform.transform_path(self._line_path)
            mutation_size = self.get_mutation_scale()  # line_mutation_scale()
            extended_path = self._extend_path(path_in_disp,
                                              mutation_size=mutation_size)
            self._path_original = extended_path
            FancyArrowPatch.draw(self, renderer)

        def get_window_extent(self, renderer=None):

            path_in_disp = self._line_transform.transform_path(self._line_path)
            mutation_size = self.get_mutation_scale()  # line_mutation_scale()
            extended_path = self._extend_path(path_in_disp,
                                              mutation_size=mutation_size)
            self._path_original = extended_path
            return FancyArrowPatch.get_window_extent(self, renderer)

    class FilledArrow(SimpleArrow):
        """The artist class that will be returned for FilledArrow style."""
        _ARROW_STYLE = "-|>"

        def __init__(self, axis_artist, line_path, transform,
                     line_mutation_scale, facecolor):
            super().__init__(axis_artist, line_path, transform,
                             line_mutation_scale)
            self.set_facecolor(facecolor)


class AxislineStyle(_Style):
    """
    A container class which defines style classes for AxisArtists.

    An instance of any axisline style class is a callable object,
    whose call signature is ::

       __call__(self, axis_artist, path, transform)

    When called, this should return an `.Artist` with the following methods::

      def set_path(self, path):
          # set the path for axisline.

      def set_line_mutation_scale(self, scale):
          # set the scale

      def draw(self, renderer):
          # draw
    """

    _style_list = {}

    class _Base:
        # The derived classes are required to be able to be initialized
        # w/o arguments, i.e., all its argument (except self) must have
        # the default values.

        def __init__(self):
            """
            initialization.
            """
            super().__init__()

        def __call__(self, axis_artist, transform):
            """
            Given the AxisArtist instance, and transform for the path (set_path
            method), return the Matplotlib artist for drawing the axis line.
            """
            return self.new_line(axis_artist, transform)

    class SimpleArrow(_Base):
        """
        A simple arrow.
        """

        ArrowAxisClass = _FancyAxislineStyle.SimpleArrow

        def __init__(self, size=1):
            """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            """

            self.size = size
            super().__init__()

        def new_line(self, axis_artist, transform):

            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform,
                                           line_mutation_scale=self.size)
            return axisline

    _style_list["->"] = SimpleArrow

    class FilledArrow(SimpleArrow):
        """
        An arrow with a filled head.
        """

        ArrowAxisClass = _FancyAxislineStyle.FilledArrow

        def __init__(self, size=1, facecolor=None):
            """
            Parameters
            ----------
            size : float
                Size of the arrow as a fraction of the ticklabel size.
            facecolor : :mpltype:`color`, default: :rc:`axes.edgecolor`
                Fill color.

                .. versionadded:: 3.7
            """

            if facecolor is None:
                facecolor = mpl.rcParams['axes.edgecolor']
            self.size = size
            self._facecolor = facecolor
            super().__init__(size=size)

        def new_line(self, axis_artist, transform):
            linepath = Path([(0, 0), (0, 1)])
            axisline = self.ArrowAxisClass(axis_artist, linepath, transform,
                                           line_mutation_scale=self.size,
                                           facecolor=self._facecolor)
            return axisline

    _style_list["-|>"] = FilledArrow
<|endoftext|>import numpy as np
import math

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple


def select_step_degree(dv):

    degree_limits_ = [1.5, 3, 7, 13, 20, 40, 70, 120, 270, 520]
    degree_steps_  = [1,   2, 5, 10, 15, 30, 45,  90, 180, 360]
    degree_factors = [1.] * len(degree_steps_)

    minsec_limits_ = [1.5, 2.5, 3.5, 8, 11, 18, 25, 45]
    minsec_steps_  = [1,   2,   3,   5, 10, 15, 20, 30]

    minute_limits_ = np.array(minsec_limits_) / 60
    minute_factors = [60.] * len(minute_limits_)

    second_limits_ = np.array(minsec_limits_) / 3600
    second_factors = [3600.] * len(second_limits_)

    degree_limits = [*second_limits_, *minute_limits_, *degree_limits_]
    degree_steps = [*minsec_steps_, *minsec_steps_, *degree_steps_]
    degree_factors = [*second_factors, *minute_factors, *degree_factors]

    n = np.searchsorted(degree_limits, dv)
    step = degree_steps[n]
    factor = degree_factors[n]

    return step, factor


def select_step_hour(dv):

    hour_limits_ = [1.5, 2.5, 3.5, 5, 7, 10, 15, 21, 36]
    hour_steps_  = [1,   2,   3,   4, 6,  8, 12, 18, 24]
    hour_factors = [1.] * len(hour_steps_)

    minsec_limits_ = [1.5, 2.5, 3.5, 4.5, 5.5, 8, 11, 14, 18, 25, 45]
    minsec_steps_  = [1,   2,   3,   4,   5,   6, 10, 12, 15, 20, 30]

    minute_limits_ = np.array(minsec_limits_) / 60
    minute_factors = [60.] * len(minute_limits_)

    second_limits_ = np.array(minsec_limits_) / 3600
    second_factors = [3600.] * len(second_limits_)

    hour_limits = [*second_limits_, *minute_limits_, *hour_limits_]
    hour_steps = [*minsec_steps_, *minsec_steps_, *hour_steps_]
    hour_factors = [*second_factors, *minute_factors, *hour_factors]

    n = np.searchsorted(hour_limits, dv)
    step = hour_steps[n]
    factor = hour_factors[n]

    return step, factor


def select_step_sub(dv):

    # subarcsec or degree
    tmp = 10.**(int(math.log10(dv))-1.)

    factor = 1./tmp

    if 1.5*tmp >= dv:
        step = 1
    elif 3.*tmp >= dv:
        step = 2
    elif 7.*tmp >= dv:
        step = 5
    else:
        step = 1
        factor = 0.1*factor

    return step, factor


def select_step(v1, v2, nv, hour=False, include_last=True,
                threshold_factor=3600.):

    if v1 > v2:
        v1, v2 = v2, v1

    dv = (v2 - v1) / nv

    if hour:
        _select_step = select_step_hour
        cycle = 24.
    else:
        _select_step = select_step_degree
        cycle = 360.

    # for degree
    if dv > 1 / threshold_factor:
        step, factor = _select_step(dv)
    else:
        step, factor = select_step_sub(dv*threshold_factor)

        factor = factor * threshold_factor

    levs = np.arange(np.floor(v1 * factor / step),
                     np.ceil(v2 * factor / step) + 0.5,
                     dtype=int) * step

    # n : number of valid levels. If there is a cycle, e.g., [0, 90, 180,
    # 270, 360], the grid line needs to be extended from 0 to 360, so
    # we need to return the whole array. However, the last level (360)
    # needs to be ignored often. In this case, so we return n=4.

    n = len(levs)

    # we need to check the range of values
    # for example, -90 to 90, 0 to 360,

    if factor == 1. and levs[-1] >= levs[0] + cycle:  # check for cycle
        nv = int(cycle / step)
        if include_last:
            levs = levs[0] + np.arange(0, nv+1, 1) * step
        else:
            levs = levs[0] + np.arange(0, nv, 1) * step

        n = len(levs)

    return np.array(levs), n, factor


def select_step24(v1, v2, nv, include_last=True, threshold_factor=3600):
    v1, v2 = v1 / 15, v2 / 15
    levs, n, factor = select_step(v1, v2, nv, hour=True,
                                  include_last=include_last,
                                  threshold_factor=threshold_factor)
    return levs * 15, n, factor


def select_step360(v1, v2, nv, include_last=True, threshold_factor=3600):
    return select_step(v1, v2, nv, hour=False,
                       include_last=include_last,
                       threshold_factor=threshold_factor)


class LocatorBase:
    def __init__(self, nbins, include_last=True):
        self.nbins = nbins
        self._include_last = include_last

    def set_params(self, nbins=None):
        if nbins is not None:
            self.nbins = int(nbins)


class LocatorHMS(LocatorBase):
    def __call__(self, v1, v2):
        return select_step24(v1, v2, self.nbins, self._include_last)


class LocatorHM(LocatorBase):
    def __call__(self, v1, v2):
        return select_step24(v1, v2, self.nbins, self._include_last,
                             threshold_factor=60)


class LocatorH(LocatorBase):
    def __call__(self, v1, v2):
        return select_step24(v1, v2, self.nbins, self._include_last,
                             threshold_factor=1)


class LocatorDMS(LocatorBase):
    def __call__(self, v1, v2):
        return select_step360(v1, v2, self.nbins, self._include_last)


class LocatorDM(LocatorBase):
    def __call__(self, v1, v2):
        return select_step360(v1, v2, self.nbins, self._include_last,
                              threshold_factor=60)


class LocatorD(LocatorBase):
    def __call__(self, v1, v2):
        return select_step360(v1, v2, self.nbins, self._include_last,
                              threshold_factor=1)


class FormatterDMS:
    deg_mark = r"^{\circ}"
    min_mark = r"^{\prime}"
    sec_mark = r"^{\prime\prime}"

    fmt_d = "$%d" + deg_mark + "$"
    fmt_ds = r"$%d.%s" + deg_mark + "$"

    # %s for sign
    fmt_d_m = r"$%s%d" + deg_mark + r"\,%02d" + min_mark + "$"
