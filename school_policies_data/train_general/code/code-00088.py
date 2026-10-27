            obj.set_table_styles(
                [{"selector": "position_float", "props": f":{position_float}"}],
                overwrite=False,
            )

        hrules = get_option("styler.latex.hrules") if hrules is None else hrules
        if hrules:
            obj.set_table_styles(
                [
                    {"selector": "toprule", "props": ":toprule"},
                    {"selector": "midrule", "props": ":midrule"},
                    {"selector": "bottomrule", "props": ":bottomrule"},
                ],
                overwrite=False,
            )

        if label:
            obj.set_table_styles(
                [{"selector": "label", "props": f":{{{label.replace(':', '§')}}}"}],
                overwrite=False,
            )

        if caption:
            obj.set_caption(caption)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        environment = environment or get_option("styler.latex.environment")
        multicol_align = multicol_align or get_option("styler.latex.multicol_align")
        multirow_align = multirow_align or get_option("styler.latex.multirow_align")
        latex = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            environment=environment,
            convert_css=convert_css,
            siunitx=siunitx,
            clines=clines,
        )

        encoding = (
            (encoding or get_option("styler.render.encoding"))
            if isinstance(buf, str)  # i.e. a filepath
            else encoding
        )
        return save_to_buffer(latex, buf=buf, encoding=encoding)

    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> None:
        ...

    @overload
    def to_html(
        self,
        buf: None = ...,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        bold_headers: bool = ...,
        caption: str | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...,
        **kwargs,
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs,
    ) -> str | None:
        """
        Write Styler to a file, buffer or string in HTML-CSS format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(buf)s
        table_uuid : str, optional
            Id attribute assigned to the <table> HTML element in the format:

            ``<table id="T_<table_uuid>" ..>``

            If not given uses Styler's initially assigned value.
        table_attributes : str, optional
            Attributes to assign within the `<table>` HTML element in the format:

            ``<table .. <table_attributes> >``

            If not given defaults to Styler's preexisting value.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.

            .. versionadded:: 1.4.0
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.

            .. versionadded:: 1.4.0
        bold_headers : bool, optional
            Adds "font-weight: bold;" as a CSS property to table style header cells.

            .. versionadded:: 1.4.0
        caption : str, optional
            Set, or overwrite, the caption on Styler before rendering.

            .. versionadded:: 1.4.0
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows/max_columns``.

            .. versionadded:: 1.4.0
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).

            .. versionadded:: 1.4.0
        %(encoding)s
        doctype_html : bool, default False
            Whether to output a fully structured HTML file including all
            HTML elements, or just the core ``<style>`` and ``<table>`` elements.
        exclude_styles : bool, default False
            Whether to include the ``<style>`` element and all associated element
            ``class`` and ``id`` identifiers, or solely the ``<table>`` element without
            styling identifiers.
        **kwargs
            Any additional keyword arguments are passed through to the jinja2
            ``self.template.render`` process. This is useful when you need to provide
            additional variables for a custom template.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_html: Write a DataFrame to a file, buffer or string in HTML format.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> print(df.style.to_html())  # doctest: +SKIP
        <style type="text/css">
        </style>
        <table id="T_1e78e">
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T_1e78e_level0_col0" class="col_heading level0 col0" >A</th>
              <th id="T_1e78e_level0_col1" class="col_heading level0 col1" >B</th>
            </tr>
        ...
        """
        obj = self._copy(deepcopy=True)  # manipulate table_styles on obj, not self

        if table_uuid:
            obj.set_uuid(table_uuid)

        if table_attributes:
            obj.set_table_attributes(table_attributes)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        if bold_headers:
            obj.set_table_styles(
                [{"selector": "th", "props": "font-weight: bold;"}], overwrite=False
            )

        if caption is not None:
            obj.set_caption(caption)

        # Build HTML string..
        html = obj._render_html(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            max_rows=max_rows,
            max_cols=max_columns,
            exclude_styles=exclude_styles,
            encoding=encoding or get_option("styler.render.encoding"),
            doctype_html=doctype_html,
            **kwargs,
        )

        return save_to_buffer(
            html, buf=buf, encoding=(encoding if buf is not None else None)
        )

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> None:
        ...

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        encoding: str | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        max_rows: int | None = ...,
        max_columns: int | None = ...,
        delimiter: str = ...,
    ) -> str:
        ...

    @Substitution(buf=buffering_args, encoding=encoding_args)
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = " ",
    ) -> str | None:
        """
        Write Styler to a file, buffer or string in text format.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        %(buf)s
        %(encoding)s
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).
        delimiter : str, default single space
            The separator between data elements.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df.style.to_string()
        ' A B\\n0 1 3\\n1 2 4\\n'
        """
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_string(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
            delimiter=delimiter,
        )
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )

    def set_td_classes(self, classes: DataFrame) -> Styler:
        """
        Set the ``class`` attribute of ``<td>`` HTML elements.

        Parameters
        ----------
        classes : DataFrame
            DataFrame containing strings that will be translated to CSS classes,
            mapped by identical column and index key values that must exist on the
            underlying Styler data. None, NaN values, and empty strings will
            be ignored and not affect the rendered HTML.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        Can be used in combination with ``Styler.set_table_styles`` to define an
        internal CSS solution without reference to external CSS files.

        Examples
        --------
        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> classes = pd.DataFrame([
        ...     ["min-val red", "", "blue"],
        ...     ["red", None, "blue max-val"]
        ... ], index=df.index, columns=df.columns)
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the
        underlying,

        >>> df = pd.DataFrame([[1,2],[3,4]], index=["a", "b"],
        ...     columns=[["level0", "level0"], ["level1a", "level1b"]])
        >>> classes = pd.DataFrame(["min-val"], index=["a"],
        ...     columns=[["level0"],["level1a"]])
        >>> df.style.set_td_classes(classes)  # doctest: +SKIP

        Form of the output with new additional css classes,

        >>> from pandas.io.formats.style import Styler
        >>> df = pd.DataFrame([[1]])
        >>> css = pd.DataFrame([["other-class"]])
        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)
        >>> s.hide(axis=0).to_html()  # doctest: +SKIP
        '<style type="text/css"></style>'
        '<table id="T__">'
        '  <thead>'
        '    <tr><th class="col_heading level0 col0" >0</th></tr>'
        '  </thead>'
        '  <tbody>'
        '    <tr><td class="data row0 col0 other-class" >1</td></tr>'
        '  </tbody>'
        '</table>'
        """
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError(
                "Classes render only if `classes` has unique index and columns."
            )
        classes = classes.reindex_like(self.data)

        for r, row_tup in enumerate(classes.itertuples()):
            for c, value in enumerate(row_tup[1:]):
                if not (pd.isna(value) or value == ""):
                    self.cell_context[(r, c)] = str(value)

        return self

    def _update_ctx(self, attrs: DataFrame) -> None:
        """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError(
                "`Styler.apply` and `.map` are not compatible "
                "with non-unique index or columns."
            )

        for cn in attrs.columns:
            j = self.columns.get_loc(cn)
            ser = attrs[cn]
            for rn, c in ser.items():
                if not c or pd.isna(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                i = self.index.get_loc(rn)
                self.ctx[(i, j)].extend(css_list)

    def _update_ctx_header(self, attrs: DataFrame, axis: AxisInt) -> None:
        """
        Update the state of the ``Styler`` for header cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : Series
            Should contain strings of '<property>: <value>;<prop2>: <val2>', and an
            integer index.
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        axis : int
            Identifies whether the ctx object being updated is the index or columns
        """
        for j in attrs.columns:
            ser = attrs[j]
            for i, c in ser.items():
                if not c or pd.isna(c):
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                if axis == 0:
                    self.ctx_index[(i, j)].extend(css_list)
                else:
                    self.ctx_columns[(j, i)].extend(css_list)

    def _copy(self, deepcopy: bool = False) -> Styler:
        """
        Copies a Styler, allowing for deepcopy or shallow copy

        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.

        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption
          - concatenated stylers

        Non-data dependent attributes [copied and exported]:
          - css
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)

        """
        # GH 40675, 52728
        styler = type(self)(
            self.data,  # populates attributes 'data', 'columns', 'index' as shallow
        )
        shallow = [  # simple string or boolean immutables
            "hide_index_",
            "hide_columns_",
            "hide_column_names",
            "hide_index_names",
            "table_attributes",
            "cell_ids",
            "caption",
            "uuid",
            "uuid_len",
            "template_latex",  # also copy templates if these have been customised
            "template_html_style",
            "template_html_table",
            "template_html",
        ]
        deep = [  # nested lists or dicts
            "css",
            "concatenated",
            "_display_funcs",
            "_display_funcs_index",
            "_display_funcs_columns",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "ctx_index",
            "ctx_columns",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
        ]

        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        return styler

    def __copy__(self) -> Styler:
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo) -> Styler:
        return self._copy(deepcopy=True)

    def clear(self) -> None:
        """
        Reset the ``Styler``, removing any previously applied styles.

        Returns None.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, np.nan]})

        After any added style:

        >>> df.style.highlight_null(color='yellow')  # doctest: +SKIP

        Remove it with:

        >>> df.style.clear()  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        # create default GH 40675
        clean_copy = Styler(self.data, uuid=self.uuid)
        clean_attrs = [a for a in clean_copy.__dict__ if not callable(a)]
        self_attrs = [a for a in self.__dict__ if not callable(a)]  # maybe more attrs
        for attr in clean_attrs:
            setattr(self, attr, getattr(clean_copy, attr))
        for attr in set(self_attrs).difference(clean_attrs):
            delattr(self, attr)

    def _apply(
        self,
        func: Callable,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if data.empty:
            result = DataFrame()
        elif axis is None:
            result = func(data, **kwargs)
            if not isinstance(result, DataFrame):
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function {repr(func)} must return a DataFrame or ndarray "
                        f"when passed to `Styler.apply` with axis=None"
                    )
                if data.shape != result.shape:
                    raise ValueError(
                        f"Function {repr(func)} returned ndarray with wrong shape.\n"
                        f"Result has shape: {result.shape}\n"
                        f"Expected shape: {data.shape}"
                    )
                result = DataFrame(result, index=data.index, columns=data.columns)
        else:
            axis = self.data._get_axis_number(axis)
            if axis == 0:
                result = data.apply(func, axis=0, **kwargs)
            else:
                result = data.T.apply(func, axis=0, **kwargs).T  # see GH 42005

        if isinstance(result, Series):
            raise ValueError(
                f"Function {repr(func)} resulted in the apply method collapsing to a "
                f"Series.\nUsually, this is the result of the function returning a "
                f"single value, instead of list-like."
            )
        msg = (
            f"Function {repr(func)} created invalid {{0}} labels.\nUsually, this is "
            f"the result of the function returning a "
            f"{'Series' if axis is not None else 'DataFrame'} which contains invalid "
            f"labels, or returning an incorrectly shaped, list-like object which "
            f"cannot be mapped to labels, possibly due to applying the function along "
            f"the wrong axis.\n"
            f"Result {{0}} has shape: {{1}}\n"
            f"Expected {{0}} shape:   {{2}}"
        )
        if not all(result.index.isin(data.index)):
            raise ValueError(msg.format("index", result.index.shape, data.index.shape))
        if not all(result.columns.isin(data.columns)):
            raise ValueError(
                msg.format("columns", result.columns.shape, data.columns.shape)
            )
        self._update_ctx(result)
        return self

    @Substitution(subset=subset_args)
    def apply(
        self,
        func: Callable,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a Series if ``axis`` in [0,1] and return a list-like
            object of same length, or a Series, not necessarily of same length, with
            valid index labels considering ``subset``.
            ``func`` should take a DataFrame if ``axis`` is ``None`` and return either
            an ndarray with the same shape or a DataFrame, not necessarily of the same
            shape, with valid index and columns labels considering ``subset``.

            .. versionchanged:: 1.3.0

            .. versionchanged:: 1.4.0

        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        This is similar to ``DataFrame.apply``, except that ``axis=None``
        applies the function to the entire DataFrame at once,
        rather than column-wise or row-wise.

        Examples
        --------
        >>> def highlight_max(x, color):
        ...     return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", None)
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.apply(highlight_max, color='red')  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color='blue', axis=1)  # doctest: +SKIP
        >>> df.style.apply(highlight_max, color='green', axis=None)  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.apply(highlight_max, color='red', subset="A")
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color='red', subset=["A", "B"])
        ... # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.apply(highlight_max, color='red', subset=([0, 1, 2], slice(None)))
        ... # doctest: +SKIP
        >>> df.style.apply(highlight_max, color='red', subset=(slice(0, 5, 2), "A"))
        ... # doctest: +SKIP

        Using a function which returns a Series / DataFrame of unequal length but
        containing valid index labels

        >>> df = pd.DataFrame([[1, 2], [3, 4], [4, 6]], index=["A1", "A2", "Total"])
        >>> total_style = pd.Series("font-weight: bold;", index=["Total"])
        >>> df.style.apply(lambda s: total_style)  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append(
            (lambda instance: getattr(instance, "_apply"), (func, axis, subset), kwargs)
        )
        return self

    def _apply_index(
        self,
        func: Callable,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        method: str = "apply",
        **kwargs,
    ) -> Styler:
        axis = self.data._get_axis_number(axis)
        obj = self.index if axis == 0 else self.columns

        levels_ = refactor_levels(level, obj)
        data = DataFrame(obj.to_list()).loc[:, levels_]

        if method == "apply":
            result = data.apply(func, axis=0, **kwargs)
        elif method == "map":
            result = data.map(func, **kwargs)

        self._update_ctx_header(result, axis)
        return self

    @doc(
        this="apply",
        wise="level-wise",
        alt="map",
        altwise="elementwise",
        func="take a Series and return a string array of the same length",
        input_note="the index as a Series, if an Index, or a level of a MultiIndex",
        output_note="an identically sized array of CSS styles as strings",
        var="s",
        ret='np.where(s == "B", "background-color: yellow;", "")',
        ret2='["background-color: yellow;" if "x" in v else "" for v in s]',
    )
    def apply_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function to the index or column headers, {wise}.

        Updates the HTML representation with the result.

        .. versionadded:: 1.4.0

        .. versionadded:: 2.1.0
           Styler.applymap_index was deprecated and renamed to Styler.map_index.

        Parameters
        ----------
        func : function
            ``func`` should {func}.
        axis : {{0, 1, "index", "columns"}}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.{alt}_index: Apply a CSS-styling function to headers {altwise}.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.
        Styler.map: Apply a CSS-styling function elementwise.

        Notes
        -----
        Each input to ``func`` will be {input_note}. The output of ``func`` should be
        {output_note}, in the format 'attribute: value; attribute2: value2; ...'
        or, if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        Basic usage to conditionally highlight values in the index.

        >>> df = pd.DataFrame([[1,2], [3,4]], index=["A", "B"])
        >>> def color_b(s):
        ...     return {ret}
        >>> df.style.{this}_index(color_b)  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead1.png

        Selectively applying to specific levels of MultiIndex columns.

        >>> midx = pd.MultiIndex.from_product([['ix', 'jy'], [0, 1], ['x3', 'z4']])
        >>> df = pd.DataFrame([np.arange(8)], columns=midx)
        >>> def highlight_x({var}):
        ...     return {ret2}
        >>> df.style.{this}_index(highlight_x, axis="columns", level=[0, 2])
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/appmaphead2.png
        """
        self._todo.append(
            (
                lambda instance: getattr(instance, "_apply_index"),
                (func, axis, level, "apply"),
                kwargs,
            )
        )
        return self

    @doc(
        apply_index,
        this="map",
        wise="elementwise",
        alt="apply",
        altwise="level-wise",
        func="take a scalar and return a string",
        input_note="an index value, if an Index, or a level value of a MultiIndex",
        output_note="CSS styles as a string",
        var="v",
        ret='"background-color: yellow;" if v == "B" else None',
        ret2='"background-color: yellow;" if "x" in v else None',
    )
    def map_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        self._todo.append(
            (
                lambda instance: getattr(instance, "_apply_index"),
                (func, axis, level, "map"),
                kwargs,
            )
        )
        return self

    def applymap_index(
        self,
        func: Callable,
        axis: AxisInt | str = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function to the index or column headers, elementwise.

        .. deprecated:: 2.1.0

           Styler.applymap_index has been deprecated. Use Styler.map_index instead.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        axis : {{0, 1, "index", "columns"}}
            The headers over which to apply the function.
        level : int, str, list, optional
            If index is MultiIndex the level(s) over which to apply the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
        """
        warnings.warn(
            "Styler.applymap_index has been deprecated. Use Styler.map_index instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.map_index(func, axis, level, **kwargs)

    def _map(self, func: Callable, subset: Subset | None = None, **kwargs) -> Styler:
        func = partial(func, **kwargs)  # map doesn't take kwargs?
        if subset is None:
            subset = IndexSlice[:]
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].map(func)
        self._update_ctx(result)
        return self

    @Substitution(subset=subset_args)
    def map(self, func: Callable, subset: Subset | None = None, **kwargs) -> Styler:
        """
        Apply a CSS-styling function elementwise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.map_index: Apply a CSS-styling function to headers elementwise.
        Styler.apply_index: Apply a CSS-styling function to headers level-wise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        >>> def color_negative(v, color):
        ...     return f"color: {color};" if v < 0 else None
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.map(color_negative, color='red')  # doctest: +SKIP

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.map(color_negative, color='red', subset="A")
        ...  # doctest: +SKIP
        >>> df.style.map(color_negative, color='red', subset=["A", "B"])
        ...  # doctest: +SKIP

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.map(color_negative, color='red',
        ...  subset=([0,1,2], slice(None)))  # doctest: +SKIP
        >>> df.style.map(color_negative, color='red', subset=(slice(0,5,2), "A"))
        ...  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        self._todo.append(
            (lambda instance: getattr(instance, "_map"), (func, subset), kwargs)
        )
        return self

    @Substitution(subset=subset_args)
    def applymap(
        self, func: Callable, subset: Subset | None = None, **kwargs
    ) -> Styler:
        """
        Apply a CSS-styling function elementwise.

        .. deprecated:: 2.1.0

           Styler.applymap has been deprecated. Use Styler.map instead.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a string.
        %(subset)s
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        Styler
        """
        warnings.warn(
            "Styler.applymap has been deprecated. Use Styler.map instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.map(func, subset, **kwargs)

    def set_table_attributes(self, attributes: str) -> Styler:
        """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str

        Returns
        -------
        Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_table_attributes('class="pure-table"')  # doctest: +SKIP
        # ... <table class="pure-table"> ...
        """
        self.table_attributes = attributes
        return self

    def export(self) -> dict[str, Any]:
        """
        Export the styles applied to the current Styler.

        Can be applied to a second Styler with ``Styler.use``.

        Returns
        -------
        dict

        See Also
        --------
        Styler.use: Set the styles on the current Styler.
        Styler.copy: Create a copy of the current Styler.

        Notes
        -----
        This method is designed to copy non-data dependent attributes of
        one Styler to another. It differs from ``Styler.copy`` where data and
        data dependent attributes are also copied.

        The following items are exported since they are not generally data dependent:

          - Styling functions added by the ``apply`` and ``map``
          - Whether axes and names are hidden from the display, if unambiguous.
          - Table attributes
          - Table styles

        The following attributes are considered data dependent and therefore not
        exported:

          - Caption
          - UUID
          - Tooltips
          - Any hidden rows or columns identified by Index labels
          - Any formatting applied using ``Styler.format``
          - Any CSS classes added using ``Styler.set_td_classes``

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        return {
            "apply": copy.copy(self._todo),
            "table_attributes": self.table_attributes,
            "table_styles": copy.copy(self.table_styles),
            "hide_index": all(self.hide_index_),
            "hide_columns": all(self.hide_columns_),
            "hide_index_names": self.hide_index_names,
            "hide_column_names": self.hide_column_names,
            "css": copy.copy(self.css),
        }

    def use(self, styles: dict[str, Any]) -> Styler:
        """
        Set the styles on the current Styler.

        Possibly uses styles from ``Styler.export``.

        Parameters
        ----------
        styles : dict(str, Any)
            List of attributes to add to Styler. Dict keys should contain only:
              - "apply": list of styler functions, typically added with ``apply`` or
                ``map``.
              - "table_attributes": HTML attributes, typically added with
                ``set_table_attributes``.
              - "table_styles": CSS selectors and properties, typically added with
                ``set_table_styles``.
              - "hide_index":  whether the index is hidden, typically added with
                ``hide_index``, or a boolean list for hidden levels.
              - "hide_columns": whether column headers are hidden, typically added with
                ``hide_columns``, or a boolean list for hidden levels.
              - "hide_index_names": whether index names are hidden.
              - "hide_column_names": whether column header names are hidden.
              - "css": the css class names used.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.export : Export the non data dependent attributes to the current Styler.

        Examples
        --------

        >>> styler = pd.DataFrame([[1, 2], [3, 4]]).style
        >>> styler2 = pd.DataFrame([[9, 9, 9]]).style
        >>> styler.hide(axis=0).highlight_max(axis=1)  # doctest: +SKIP
        >>> export = styler.export()
        >>> styler2.use(export)  # doctest: +SKIP
        """
        self._todo.extend(styles.get("apply", []))
        table_attributes: str = self.table_attributes or ""
        obj_table_atts: str = (
            ""
            if styles.get("table_attributes") is None
            else str(styles.get("table_attributes"))
        )
        self.set_table_attributes((table_attributes + " " + obj_table_atts).strip())
        if styles.get("table_styles"):
            self.set_table_styles(styles.get("table_styles"), overwrite=False)

        for obj in ["index", "columns"]:
            hide_obj = styles.get("hide_" + obj)
            if hide_obj is not None:
                if isinstance(hide_obj, bool):
                    n = getattr(self, obj).nlevels
                    setattr(self, "hide_" + obj + "_", [hide_obj] * n)
                else:
                    setattr(self, "hide_" + obj + "_", hide_obj)

        self.hide_index_names = styles.get("hide_index_names", False)
        self.hide_column_names = styles.get("hide_column_names", False)
        if styles.get("css"):
            self.css = styles.get("css")  # type: ignore[assignment]
        return self

    def set_uuid(self, uuid: str) -> Styler:
        """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str

        Returns
        -------
        Styler

        Notes
        -----
        Almost all HTML elements within the table, and including the ``<table>`` element
        are assigned ``id`` attributes. The format is ``T_uuid_<extra>`` where
        ``<extra>`` is typically a more specific identifier, such as ``row1_col2``.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['A', 'B'], columns=['c1', 'c2'])

        You can get the `id` attributes with the following:

        >>> print((df).style.to_html())  # doctest: +SKIP

        To add a title to column `c1`, its `id` is T_20a7d_level0_col0:

        >>> df.style.set_uuid("T_20a7d_level0_col0")
        ... .set_caption("Test")  # doctest: +SKIP

        Please see:
        `Table visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        self.uuid = uuid
        return self

    def set_caption(self, caption: str | tuple | list) -> Styler:
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple, list
            For HTML output either the string input is used or the first element of the
            tuple. For LaTeX the string input provides a caption and the additional
            tuple input allows for full captions and short captions, in that order.

        Returns
        -------
        Styler

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df.style.set_caption("test")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        msg = "`caption` must be either a string or 2-tuple of strings."
        if isinstance(caption, (list, tuple)):
            if (
                len(caption) != 2
                or not isinstance(caption[0], str)
                or not isinstance(caption[1], str)
            ):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: Level | list[Level] | None = None,
    ) -> Styler:
        """
        Add CSS to permanently display the index or column headers in a scrolling frame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to make the index or column headers sticky.
        pixel_size : int, optional
            Required to configure the width of index cells or the height of column
            header cells when sticking a MultiIndex (or with a named Index).
            Defaults to 75 and 25 respectively.
        levels : int, str, list, optional
            If ``axis`` is a MultiIndex the specific levels to stick. If ``None`` will
            stick all levels.

        Returns
        -------
        Styler

        Notes
        -----
        This method uses the CSS 'position: sticky;' property to display. It is
        designed to work with visible axes, therefore both:

          - `styler.set_sticky(axis="index").hide(axis="index")`
          - `styler.set_sticky(axis="columns").hide(axis="columns")`

        may produce strange behaviour due to CSS controls with missing elements.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df.style.set_sticky(axis="index")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        axis = self.data._get_axis_number(axis)
        obj = self.data.index if axis == 0 else self.data.columns
        pixel_size = (75 if axis == 0 else 25) if not pixel_size else pixel_size

        props = "position:sticky; background-color:inherit;"
        if not isinstance(obj, pd.MultiIndex):
            # handling MultiIndexes requires different CSS

            if axis == 1:
                # stick the first <tr> of <head> and, if index names, the second <tr>
                # if self._hide_columns then no <thead><tr> here will exist: no conflict
                styles: CSSStyles = [
                    {
                        "selector": "thead tr:nth-child(1) th",
                        "props": props + "top:0px; z-index:2;",
                    }
                ]
                if self.index.names[0] is not None:
                    styles[0]["props"] = (
                        props + f"top:0px; z-index:2; height:{pixel_size}px;"
                    )
                    styles.append(
                        {
                            "selector": "thead tr:nth-child(2) th",
                            "props": props
                            + f"top:{pixel_size}px; z-index:2; height:{pixel_size}px; ",
                        }
                    )
            else:
                # stick the first <th> of each <tr> in both <thead> and <tbody>
                # if self._hide_index then no <th> will exist in <tbody>: no conflict
                # but <th> will exist in <thead>: conflict with initial element
                styles = [
                    {
                        "selector": "thead tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:3 !important;",
                    },
                    {
                        "selector": "tbody tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:1;",
                    },
                ]

        else:
            # handle the MultiIndex case
            range_idx = list(range(obj.nlevels))
            levels_: list[int] = refactor_levels(levels, obj) if levels else range_idx
            levels_ = sorted(levels_)

            if axis == 1:
                styles = []
                for i, level in enumerate(levels_):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({level+1}) th",
                            "props": props
                            + (
                                f"top:{i * pixel_size}px; height:{pixel_size}px; "
                                "z-index:2;"
                            ),
                        }
                    )
                if not all(name is None for name in self.index.names):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({obj.nlevels+1}) th",
                            "props": props
                            + (
                                f"top:{(len(levels_)) * pixel_size}px; "
                                f"height:{pixel_size}px; z-index:2;"
                            ),
                        }
                    )

            else:
                styles = []
                for i, level in enumerate(levels_):
                    props_ = props + (
                        f"left:{i * pixel_size}px; "
                        f"min-width:{pixel_size}px; "
                        f"max-width:{pixel_size}px; "
                    )
                    styles.extend(
                        [
                            {
                                "selector": f"thead tr th:nth-child({level+1})",
                                "props": props_ + "z-index:3 !important;",
                            },
                            {
                                "selector": f"tbody tr th.level{level}",
                                "props": props_ + "z-index:1;",
                            },
                        ]
                    )

        return self.set_table_styles(styles, overwrite=False)

    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles | None = None,
        axis: AxisInt = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ) -> Styler:
        """
        Set the table styles included within the ``<style>`` HTML element.

        This function can be used to style the entire table, columns, rows or
        specific HTML selectors.

        Parameters
        ----------
        table_styles : list or dict
            If supplying a list, each individual table_style should be a
            dictionary with ``selector`` and ``props`` keys. ``selector``
            should be a CSS selector that the style will be applied to
            (automatically prefixed by the table's UUID) and ``props``
            should be a list of tuples with ``(attribute, value)``.
            If supplying a dict, the dict keys should correspond to
            column names or index values, depending upon the specified
            `axis` argument. These will be mapped to row or col CSS
            selectors. MultiIndex values as dict keys should be
            in their respective tuple form. The dict values should be
            a list as specified in the form with CSS selectors and
            props that will be applied to the specified row or column.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``). Only used if `table_styles` is
            dict.
        overwrite : bool, default True
            Styles are replaced if `True`, or extended if `False`. CSS
            rules are preserved so most recent styles set will dominate
            if selectors intersect.
        css_class_names : dict, optional
            A dict of strings used to replace the default CSS classes described below.

            .. versionadded:: 1.4.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        The default CSS classes dict, whose values can be replaced is as follows:

        .. code-block:: python

            css_class_names = {"row_heading": "row_heading",
                               "col_heading": "col_heading",
                               "index_name": "index_name",
                               "col": "col",
                               "row": "row",
                               "col_trim": "col_trim",
                               "row_trim": "row_trim",
                               "level": "level",
                               "data": "data",
                               "blank": "blank",
                               "foot": "foot"}

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4),
        ...                   columns=['A', 'B', 'C', 'D'])
        >>> df.style.set_table_styles(
        ...     [{'selector': 'tr:hover',
        ...       'props': [('background-color', 'yellow')]}]
        ... )  # doctest: +SKIP

        Or with CSS strings

        >>> df.style.set_table_styles(
        ...     [{'selector': 'tr:hover',
        ...       'props': 'background-color: yellow; font-size: 1em;'}]
        ... )  # doctest: +SKIP

        Adding column styling by name

        >>> df.style.set_table_styles({
        ...     'A': [{'selector': '',
        ...            'props': [('color', 'red')]}],
        ...     'B': [{'selector': 'td',
        ...            'props': 'color: blue;'}]
        ... }, overwrite=False)  # doctest: +SKIP

        Adding row styling

        >>> df.style.set_table_styles({
        ...     0: [{'selector': 'td:hover',
        ...          'props': [('font-size', '25px')]}]
        ... }, axis=1, overwrite=False)  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        if css_class_names is not None:
            self.css = {**self.css, **css_class_names}

        if table_styles is None:
            return self
        elif isinstance(table_styles, dict):
            axis = self.data._get_axis_number(axis)
            obj = self.data.index if axis == 1 else self.data.columns
            idf = f".{self.css['row']}" if axis == 1 else f".{self.css['col']}"

            table_styles = [
                {
                    "selector": str(s["selector"]) + idf + str(idx),
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for key, styles in table_styles.items()
                for idx in obj.get_indexer_for([key])
                for s in format_table_styles(styles)
            ]
        else:
            table_styles = [
                {
                    "selector": s["selector"],
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for s in table_styles
            ]

        if not overwrite and self.table_styles is not None:
            self.table_styles.extend(table_styles)
        else:
            self.table_styles = table_styles
        return self

    def hide(
        self,
        subset: Subset | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        names: bool = False,
    ) -> Styler:
        """
        Hide the entire index / column headers, or specific rows / columns from display.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 1d input or single key along the axis within
            `DataFrame.loc[<subset>, :]` or `DataFrame.loc[:, <subset>]` depending
            upon ``axis``, to limit ``data`` to select hidden rows / columns.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list
            The level(s) to hide in a MultiIndex if hiding the entire index / column
            headers. Cannot be used simultaneously with ``subset``.
        names : bool
            Whether to hide the level name(s) of the index / columns headers in the case
            it (or at least one the levels) remains visible.

        Returns
        -------
        Styler

        Notes
        -----
        .. warning::
           This method only works with the output methods ``to_html``, ``to_string``
           and ``to_latex``.

           Other output methods, including ``to_excel``, ignore this hiding method
           and will display all data.

        This method has multiple functionality depending upon the combination
        of the ``subset``, ``level`` and ``names`` arguments (see examples). The
        ``axis`` argument is used only to control whether the method is applied to row
        or column headers:

        .. list-table:: Argument combinations
           :widths: 10 20 10 60
           :header-rows: 1

           * - ``subset``
             - ``level``
             - ``names``
             - Effect
           * - None
             - None
             - False
             - The axis-Index is hidden entirely.
           * - None
             - None
             - True
             - Only the axis-Index names are hidden.
           * - None
             - Int, Str, List
             - False
             - Specified axis-MultiIndex levels are hidden entirely.
           * - None
             - Int, Str, List
             - True
             - Specified axis-MultiIndex levels are hidden entirely and the names of
               remaining axis-MultiIndex levels.
           * - Subset
             - None
             - False
             - The specified data rows/columns are hidden, but the axis-Index itself,
               and names, remain unchanged.
           * - Subset
             - None
             - True
             - The specified data rows/columns and axis-Index names are hidden, but
               the axis-Index itself remains unchanged.
           * - Subset
             - Int, Str, List
             - Boolean
             - ValueError: cannot supply ``subset`` and ``level`` simultaneously.

        Note this method only hides the identified elements so can be chained to hide
        multiple elements in sequence.

        Examples
        --------
        Simple application hiding specific rows:

        >>> df = pd.DataFrame([[1,2], [3,4], [5,6]], index=["a", "b", "c"])
        >>> df.style.hide(["a", "b"])  # doctest: +SKIP
             0    1
        c    5    6

        Hide the index and retain the data values:

        >>> midx = pd.MultiIndex.from_product([["x", "y"], ["a", "b", "c"]])
        >>> df = pd.DataFrame(np.random.randn(6,6), index=midx, columns=midx)
        >>> df.style.format("{:.1f}").hide()  # doctest: +SKIP
                         x                    y
           a      b      c      a      b      c
         0.1    0.0    0.4    1.3    0.6   -1.4
         0.7    1.0    1.3    1.5   -0.0   -0.2
         1.4   -0.8    1.6   -0.2   -0.4   -0.3
         0.4    1.0   -0.2   -0.8   -1.2    1.1
        -0.6    1.2    1.8    1.9    0.3    0.3
         0.8    0.5   -0.3    1.2    2.2   -0.8

        Hide specific rows in a MultiIndex but retain the index:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"]))
        ...   # doctest: +SKIP
                                 x                    y
                   a      b      c      a      b      c
        x   b    0.7    1.0    1.3    1.5   -0.0   -0.2
        y   b   -0.6    1.2    1.8    1.9    0.3    0.3

        Hide specific rows and the index through chaining:

        >>> df.style.format("{:.1f}").hide(subset=(slice(None), ["a", "c"])).hide()
        ...   # doctest: +SKIP
                         x                    y
           a      b      c      a      b      c
         0.7    1.0    1.3    1.5   -0.0   -0.2
        -0.6    1.2    1.8    1.9    0.3    0.3

        Hide a specific level:

        >>> df.style.format("{:,.1f}").hide(level=1)  # doctest: +SKIP
                             x                    y
               a      b      c      a      b      c
        x    0.1    0.0    0.4    1.3    0.6   -1.4
             0.7    1.0    1.3    1.5   -0.0   -0.2
             1.4   -0.8    1.6   -0.2   -0.4   -0.3
        y    0.4    1.0   -0.2   -0.8   -1.2    1.1
            -0.6    1.2    1.8    1.9    0.3    0.3
             0.8    0.5   -0.3    1.2    2.2   -0.8

        Hiding just the index level names:

        >>> df.index.names = ["lev0", "lev1"]
        >>> df.style.format("{:,.1f}").hide(names=True)  # doctest: +SKIP
                                 x                    y
                   a      b      c      a      b      c
        x   a    0.1    0.0    0.4    1.3    0.6   -1.4
            b    0.7    1.0    1.3    1.5   -0.0   -0.2
            c    1.4   -0.8    1.6   -0.2   -0.4   -0.3
        y   a    0.4    1.0   -0.2   -0.8   -1.2    1.1
            b   -0.6    1.2    1.8    1.9    0.3    0.3
            c    0.8    0.5   -0.3    1.2    2.2   -0.8

        Examples all produce equivalently transposed effects with ``axis="columns"``.
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            obj, objs, alt = "index", "index", "rows"
        else:
            obj, objs, alt = "column", "columns", "columns"

        if level is not None and subset is not None:
            raise ValueError("`subset` and `level` cannot be passed simultaneously")

        if subset is None:
            if level is None and names:
                # this combination implies user shows the index and hides just names
                setattr(self, f"hide_{obj}_names", True)
                return self

            levels_ = refactor_levels(level, getattr(self, objs))
            setattr(
                self,
                f"hide_{objs}_",
                [lev in levels_ for lev in range(getattr(self, objs).nlevels)],
            )
        else:
            if axis == 0:
                subset_ = IndexSlice[subset, :]  # new var so mypy reads not Optional
            else:
                subset_ = IndexSlice[:, subset]  # new var so mypy reads not Optional
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            h_els = getattr(self, objs).get_indexer_for(getattr(hide, objs))
            setattr(self, f"hidden_{alt}", h_els)

        if names:
            setattr(self, f"hide_{obj}_names", True)
        return self

    # -----------------------------------------------------------------------
    # A collection of "builtin" styles
    # -----------------------------------------------------------------------

    def _get_numeric_subset_default(self):
        # Returns a boolean mask indicating where `self.data` has numerical columns.
        # Choosing a mask as opposed to the column names also works for
        # boolean column labels (GH47838).
        return self.data.columns.isin(self.data.select_dtypes(include=np.number))

    @doc(
        name="background",
        alt="text",
        image_prefix="bg",
        text_threshold="""text_color_threshold : float or int\n
            Luminance threshold for determining text color in [0, 1]. Facilitates text\n
            visibility across varying background colors. All text is dark if 0, and\n
            light if 1, defaults to 0.408.""",
    )
    @Substitution(subset=subset_args)
    def background_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        """
        Color the {name} in a gradient style.

        The {name} color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {{0, 1, "index", "columns", None}}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(subset)s
        {text_threshold}
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.
        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.
        gmap : array-like, optional
            Gradient map for determining the {name} colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.{alt}_gradient: Color the {alt} in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...                   data=[["Stockholm", 21.6, 5.0, 3.2],
        ...                         ["Oslo", 22.4, 13.3, 3.1],
        ...                         ["Copenhagen", 24.5, 0.0, 6.7]])

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.{name}_gradient(axis=0)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.{name}_gradient(axis=None)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.{name}_gradient(axis=None, low=0.75, high=1.0)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.{name}_gradient(axis=None, vmin=6.7, vmax=21.6)  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.{name}_gradient(axis=0, gmap=df['Temp (c)'], cmap='YlOrRd')
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1,2,3], [2,3,4], [3,4,5]])
        >>> df.style.{name}_gradient(axis=None, gmap=gmap,
        ...     cmap='YlOrRd', subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/{image_prefix}_axNone_gmap.png
        """
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
        )
        return self

    @doc(
        background_gradient,
        name="text",
        alt="background",
        image_prefix="tg",
        text_threshold="",
    )
    def text_gradient(
        self,
        cmap: str | Colormap = "PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        if subset is None and gmap is None:
            subset = self._get_numeric_subset_default()

        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    @Substitution(subset=subset_args)
    def set_properties(self, subset: Subset | None = None, **kwargs) -> Styler:
        """
        Set defined CSS-properties to each ``<td>`` HTML element for the given subset.

        Parameters
        ----------
        %(subset)s
        **kwargs : dict
            A dictionary of property, value pairs to be set for each cell.

        Returns
        -------
        Styler

        Notes
        -----
        This is a convenience methods which wraps the :meth:`Styler.map` calling a
        function returning the CSS-properties independently of the data.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_properties(color="white", align="right")  # doctest: +SKIP
        >>> df.style.set_properties(**{'background-color': 'yellow'})  # doctest: +SKIP

        See `Table Visualization <../../user_guide/style.ipynb>`_ user guide for
        more details.
        """
        values = "".join([f"{p}: {v};" for p, v in kwargs.items()])
        return self.map(lambda x: values, subset=subset)

    @Substitution(subset=subset_args)
    def bar(  # pylint: disable=disallowed-name
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        *,
        color: str | list | tuple | None = None,
        cmap: Any | None = None,
        width: float = 100,
        height: float = 100,
        align: str | float | Callable = "mid",
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = "width: 10em;",
    ) -> Styler:
        """
        Draw bar chart in the cell backgrounds.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        %(subset)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        color : str or 2-tuple/list
            If a str is passed, the color is the same for both
            negative and positive numbers. If 2-tuple/list is used, the
            first element is the color_negative and the second is the
            color_positive (eg: ['#d65f5f', '#5fba7d']).
        cmap : str, matplotlib.cm.ColorMap
            A string name of a matplotlib Colormap, or a Colormap object. Cannot be
            used together with ``color``.

            .. versionadded:: 1.4.0
        width : float, default 100
            The percentage of the cell, measured from the left, in which to draw the
            bars, in [0, 100].
        height : float, default 100
            The percentage height of the bar in the cell, centrally aligned, in [0,100].

            .. versionadded:: 1.4.0
        align : str, int, float, callable, default 'mid'
            How to align the bars within the cells relative to a width adjusted center.
            If string must be one of:

            - 'left' : bars are drawn rightwards from the minimum data value.
            - 'right' : bars are drawn leftwards from the maximum data value.
            - 'zero' : a value of zero is located at the center of the cell.
            - 'mid' : a value of (max-min)/2 is located at the center of the cell,
              or if all values are negative (positive) the zero is
              aligned at the right (left) of the cell.
            - 'mean' : the mean value of the data is located at the center of the cell.

            If a float or integer is given this will indicate the center of the cell.

            If a callable should take a 1d or 2d array and return a scalar.

            .. versionchanged:: 1.4.0

        vmin : float, optional
            Minimum bar value, defining the left hand limit
            of the bar drawing range, lower values are clipped to `vmin`.
            When None (default): the minimum value of the data will be used.
        vmax : float, optional
            Maximum bar value, defining the right hand limit
            of the bar drawing range, higher values are clipped to `vmax`.
            When None (default): the maximum value of the data will be used.
        props : str, optional
            The base CSS of the cell that is extended to add the bar chart. Defaults to
            `"width: 10em;"`.

            .. versionadded:: 1.4.0

        Returns
        -------
        Styler

        Notes
        -----
        This section of the user guide:
        `Table Visualization <../../user_guide/style.ipynb>`_ gives
        a number of examples for different settings and color coordination.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        >>> df.style.bar(subset=['A'], color='gray')  # doctest: +SKIP
        """
        if color is None and cmap is None:
            color = "#d65f5f"
        elif color is not None and cmap is not None:
            raise ValueError("`color` and `cmap` cannot both be given")
        elif color is not None:
            if (isinstance(color, (list, tuple)) and len(color) > 2) or not isinstance(
                color, (str, list, tuple)
            ):
                raise ValueError(
                    "`color` must be string or list or tuple of 2 strings,"
                    "(eg: color=['#d65f5f', '#5fba7d'])"
                )

        if not 0 <= width <= 100:
            raise ValueError(f"`width` must be a value in [0, 100], got {width}")
        if not 0 <= height <= 100:
            raise ValueError(f"`height` must be a value in [0, 100], got {height}")

        if subset is None:
            subset = self._get_numeric_subset_default()

        self.apply(
            _bar,
            subset=subset,
            axis=axis,
            align=align,
            colors=color,
            cmap=cmap,
            width=width / 100,
            height=height / 100,
            vmin=vmin,
            vmax=vmax,
            base_css=props,
        )

        return self

    @Substitution(
        subset=subset_args,
        props=properties_args,
        color=coloring_args.format(default="red"),
    )
    def highlight_null(
        self,
        color: str = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight missing values with a style.

        Parameters
        ----------
        %(color)s

            .. versionadded:: 1.5.0

        %(subset)s

        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, np.nan]})
        >>> df.style.highlight_null(color='yellow')  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        def f(data: DataFrame, props: str) -> np.ndarray:
            return np.where(pd.isna(data).to_numpy(), props, "")

        if props is None:
            props = f"background-color: {color};"
        return self.apply(f, axis=None, subset=subset, props=props)

    @Substitution(
        subset=subset_args,
        color=coloring_args.format(default="yellow"),
        props=properties_args,
    )
    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight the maximum with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [2, 1], 'B': [3, 4]})
        >>> df.style.highlight_max(color='yellow')  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="max"),
            axis=axis,
            subset=subset,
            props=props,
        )

    @Substitution(
        subset=subset_args,
        color=coloring_args.format(default="yellow"),
        props=properties_args,
    )
    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight the minimum with a style.

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        %(props)s

            .. versionadded:: 1.3.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [2, 1], 'B': [3, 4]})
        >>> df.style.highlight_min(color='yellow')  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            partial(_highlight_value, op="min"),
            axis=axis,
            subset=subset,
            props=props,
        )

    @Substitution(
        subset=subset_args,
        color=coloring_args.format(default="yellow"),
        props=properties_args,
    )
    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        left: Scalar | Sequence | None = None,
        right: Scalar | Sequence | None = None,
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler:
        """
        Highlight a defined range with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            If ``left`` or ``right`` given as sequence, axis along which to apply those
            boundaries. See examples.
        left : scalar or datetime-like, or sequence or array-like, default None
            Left bound for defining the range.
        right : scalar or datetime-like, or sequence or array-like, default None
            Right bound for defining the range.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether bounds are closed or open.
        %(props)s

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Notes
        -----
        If ``left`` is ``None`` only the right bound is applied.
        If ``right`` is ``None`` only the left bound is applied. If both are ``None``
        all values are highlighted.

        ``axis`` is only needed if ``left`` or ``right`` are provided as a sequence or
        an array-like object for aligning the shapes. If ``left`` and ``right`` are
        both scalars then all ``axis`` inputs will give the same result.

        This function only works with compatible ``dtypes``. For example a datetime-like
        region can only use equivalent datetime-like ``left`` and ``right`` arguments.
        Use ``subset`` to control regions which have multiple ``dtypes``.

        Examples
        --------
        Basic usage

        >>> df = pd.DataFrame({
        ...     'One': [1.2, 1.6, 1.5],
        ...     'Two': [2.9, 2.1, 2.5],
        ...     'Three': [3.1, 3.2, 3.8],
        ... })
        >>> df.style.highlight_between(left=2.1, right=2.9)  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_basic.png

        Using a range input sequence along an ``axis``, in this case setting a ``left``
        and ``right`` for each column individually

        >>> df.style.highlight_between(left=[1.4, 2.4, 3.4], right=[1.6, 2.6, 3.6],
        ...     axis=1, color="#fffd75")  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_seq.png

        Using ``axis=None`` and providing the ``left`` argument as an array that
        matches the input DataFrame, with a constant ``right``

        >>> df.style.highlight_between(left=[[2,2,3],[2,2,3],[3,3,3]], right=3.5,
        ...     axis=None, color="#fffd75")  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_axNone.png

        Using ``props`` instead of default background coloring

        >>> df.style.highlight_between(left=1.5, right=3.5,
        ...     props='font-weight:bold;color:#e83e8c')  # doctest: +SKIP

        .. figure:: ../../_static/style/hbetw_props.png
        """
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis,
            subset=subset,
            props=props,
            left=left,
            right=right,
            inclusive=inclusive,
        )

    @Substitution(
        subset=subset_args,
        color=coloring_args.format(default="yellow"),
        props=properties_args,
    )
    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: QuantileInterpolation = "linear",
        inclusive: IntervalClosedType = "both",
        props: str | None = None,
    ) -> Styler:
        """
        Highlight values defined by a quantile with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        %(subset)s
        %(color)s
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis along which to determine and highlight quantiles. If ``None`` quantiles
            are measured over the entire DataFrame. See examples.
        q_left : float, default 0
            Left bound, in [0, q_right), for the target quantile range.
        q_right : float, default 1
            Right bound, in (q_left, 1], for the target quantile range.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Argument passed to ``Series.quantile`` or ``DataFrame.quantile`` for
            quantile estimation.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether quantile bounds are closed or open.
        %(props)s

        Returns
        -------
        Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.

        Notes
        -----
        This function does not work with ``str`` dtypes.

        Examples
        --------
        Using ``axis=None`` and apply a quantile to all collective data

        >>> df = pd.DataFrame(np.arange(10).reshape(2,5) + 1)
        >>> df.style.highlight_quantile(axis=None, q_left=0.8, color="#fffd75")
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/hq_axNone.png

        Or highlight quantiles row-wise or column-wise, in this case by row-wise

        >>> df.style.highlight_quantile(axis=1, q_left=0.8, color="#fffd75")
        ...  # doctest: +SKIP

        .. figure:: ../../_static/style/hq_ax1.png

        Use ``props`` instead of default background coloring

        >>> df.style.highlight_quantile(axis=None, q_left=0.2, q_right=0.8,
        ...     props='font-weight:bold;color:#e83e8c')  # doctest: +SKIP

        .. figure:: ../../_static/style/hq_props.png
        """
        subset_ = slice(None) if subset is None else subset
        subset_ = non_reducing_slice(subset_)
        data = self.data.loc[subset_]

        # after quantile is found along axis, e.g. along rows,
        # applying the calculated quantile to alternate axis, e.g. to each column
        quantiles = [q_left, q_right]
        if axis is None:
            q = Series(data.to_numpy().ravel()).quantile(
                q=quantiles, interpolation=interpolation
            )
            axis_apply: int | None = None
        else:
            axis = self.data._get_axis_number(axis)
            q = data.quantile(
                axis=axis, numeric_only=False, q=quantiles, interpolation=interpolation
            )
            axis_apply = 1 - axis

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,
            axis=axis_apply,
            subset=subset,
            props=props,
            left=q.iloc[0],
            right=q.iloc[1],
            inclusive=inclusive,
        )

    @classmethod
    def from_custom_template(
        cls,
        searchpath: Sequence[str],
        html_table: str | None = None,
        html_style: str | None = None,
    ) -> type[Styler]:
        """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.

            .. versionadded:: 1.3.0

        html_style : str
            Name of your custom template to replace the html_style template.

            .. versionadded:: 1.3.0

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``,``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.

        Examples
        --------
        >>> from pandas.io.formats.style import Styler
        >>> EasyStyler = Styler.from_custom_template("path/to/template",
        ...                                          "template.tpl",
        ...                                          )  # doctest: +SKIP
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> EasyStyler(df)  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        # mypy doesn't like dynamically-defined classes
        # error: Variable "cls" is not valid as a type
        # error: Invalid base class "cls"
        class MyStyler(cls):  # type: ignore[valid-type,misc]
            env = jinja2.Environment(loader=loader)
            if html_table:
                template_html_table = env.get_template(html_table)
            if html_style:
                template_html_style = env.get_template(html_style)

        return MyStyler

    def pipe(self, func: Callable, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``, and return the result.

        Parameters
        ----------
        func : function
            Function to apply to the Styler.  Alternatively, a
            ``(callable, keyword)`` tuple where ``keyword`` is a string
            indicating the keyword of ``callable`` that expects the Styler.
        *args : optional
            Arguments passed to `func`.
        **kwargs : optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object :
            The value returned by ``func``.

        See Also
        --------
        DataFrame.pipe : Analogous method for DataFrame.
        Styler.apply : Apply a CSS-styling function column-wise, row-wise, or
            table-wise.

        Notes
        -----
        Like :meth:`DataFrame.pipe`, this method can simplify the
        application of several user-defined functions to a styler.  Instead
        of writing:

        .. code-block:: python

            f(g(df.style.format(precision=3), arg1=a), arg2=b, arg3=c)

        users can write:

        .. code-block:: python

            (df.style.format(precision=3)
               .pipe(g, arg1=a)
               .pipe(f, arg2=b, arg3=c))

        In particular, this allows users to define functions that take a
        styler object, along with other parameters, and return the styler after
        making styling changes (such as calling :meth:`Styler.apply` or
        :meth:`Styler.set_properties`).

        Examples
        --------

        **Common Use**

        A common usage pattern is to pre-define styling operations which
        can be easily applied to a generic styler in a single ``pipe`` call.

        >>> def some_highlights(styler, min_color="red", max_color="blue"):
        ...      styler.highlight_min(color=min_color, axis=None)
        ...      styler.highlight_max(color=max_color, axis=None)
        ...      styler.highlight_null()
        ...      return styler
        >>> df = pd.DataFrame([[1, 2, 3, pd.NA], [pd.NA, 4, 5, 6]], dtype="Int64")
        >>> df.style.pipe(some_highlights, min_color="green")  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl.png

        Since the method returns a ``Styler`` object it can be chained with other
        methods as if applying the underlying highlighters directly.

        >>> (df.style.format("{:.1f}")
        ...         .pipe(some_highlights, min_color="green")
        ...         .highlight_between(left=2, right=5))  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_hl2.png

        **Advanced Use**

        Sometimes it may be necessary to pre-define styling functions, but in the case
        where those functions rely on the styler, data or context. Since
        ``Styler.use`` and ``Styler.export`` are designed to be non-data dependent,
        they cannot be used for this purpose. Additionally the ``Styler.apply``
        and ``Styler.format`` type methods are not context aware, so a solution
        is to use ``pipe`` to dynamically wrap this functionality.

        Suppose we want to code a generic styling function that highlights the final
        level of a MultiIndex. The number of levels in the Index is dynamic so we
        need the ``Styler`` context to define the level.

        >>> def highlight_last_level(styler):
        ...     return styler.apply_index(
        ...         lambda v: "background-color: pink; color: yellow", axis="columns",
        ...         level=styler.columns.nlevels-1
        ...     )  # doctest: +SKIP
        >>> df.columns = pd.MultiIndex.from_product([["A", "B"], ["X", "Y"]])
        >>> df.style.pipe(highlight_last_level)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applymap.png

        Additionally suppose we want to highlight a column header if there is any
        missing data in that column.
        In this case we need the data object itself to determine the effect on the
        column headers.

        >>> def highlight_header_missing(styler, level):
        ...     def dynamic_highlight(s):
        ...         return np.where(
        ...             styler.data.isna().any(), "background-color: red;", ""
        ...         )
        ...     return styler.apply_index(dynamic_highlight, axis=1, level=level)
        >>> df.style.pipe(highlight_header_missing, level=1)  # doctest: +SKIP

        .. figure:: ../../_static/style/df_pipe_applydata.png
        """
        return com.pipe(self, func, *args, **kwargs)


def _validate_apply_axis_arg(
    arg: NDFrame | Sequence | np.ndarray,
    arg_name: str,
    dtype: Any | None,
    data: NDFrame,
) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype = {"dtype": dtype} if dtype else {}
    # raise if input is wrong for axis:
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations "
            f"is a DataFrame since 'axis=None'"
        )
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for "
            f"operations is a Series with 'axis in [0,1]'"
        )
    if isinstance(arg, (Series, DataFrame)):  # align indx / cols to data
        arg = arg.reindex_like(data, method=None).to_numpy(**dtype)
    else:
        arg = np.asarray(arg, **dtype)
        assert isinstance(arg, np.ndarray)  # mypy requirement
        if arg.shape != data.shape:  # check valid input
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over "
                f"selected 'axis': got {arg.shape}, "
                f"expected {data.shape}"
            )
    return arg


def _background_gradient(
    data,
    cmap: str | Colormap = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: float | None = None,
    vmax: float | None = None,
    gmap: Sequence | np.ndarray | DataFrame | Series | None = None,
    text_only: bool = False,
):
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:  # the data is used the gmap
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:  # else validate gmap against the underlying data
        gmap = _validate_apply_axis_arg(gmap, "gmap", float, data)

    with _mpl(Styler.background_gradient) as (_, _matplotlib):
        smin = np.nanmin(gmap) if vmin is None else vmin
        smax = np.nanmax(gmap) if vmax is None else vmax
        rng = smax - smin
        # extend lower / upper bounds, compresses color range
        norm = _matplotlib.colors.Normalize(smin - (rng * low), smax + (rng * high))

        if cmap is None:
            rgbas = _matplotlib.colormaps[_matplotlib.rcParams["image.cmap"]](
                norm(gmap)
            )
        else:
            rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

        def relative_luminance(rgba) -> float:
            """
            Calculate relative luminance of a color.

            The calculation adheres to the W3C standards
            (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

            Parameters
            ----------
            color : rgb or rgba tuple

            Returns
            -------
            float
                The relative luminance as a value from 0 to 1
            """
            r, g, b = (
                x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
                for x in rgba[:3]
            )
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def css(rgba, text_only) -> str:
            if not text_only:
                dark = relative_luminance(rgba) < text_color_threshold
                text_color = "#f1f1f1" if dark else "#000000"
                return (
                    f"background-color: {_matplotlib.colors.rgb2hex(rgba)};"
                    f"color: {text_color};"
                )
            else:
                return f"color: {_matplotlib.colors.rgb2hex(rgba)};"

        if data.ndim == 1:
            return [css(rgba, text_only) for rgba in rgbas]
        else:
            return DataFrame(
                [[css(rgba, text_only) for rgba in row] for row in rgbas],
                index=data.index,
                columns=data.columns,
            )


def _highlight_between(
    data: NDFrame,
    props: str,
    left: Scalar | Sequence | np.ndarray | NDFrame | None = None,
    right: Scalar | Sequence | np.ndarray | NDFrame | None = None,
    inclusive: bool | str = True,
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and not isinstance(left, str):
        left = _validate_apply_axis_arg(left, "left", None, data)

    if np.iterable(right) and not isinstance(right, str):
        right = _validate_apply_axis_arg(right, "right", None, data)

    # get ops with correct boundary attribution
    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither' "
            f"got {inclusive}"
        )

    g_left = (
        # error: Argument 2 to "ge" has incompatible type "Union[str, float,
        # Period, Timedelta, Interval[Any], datetime64, timedelta64, datetime,
        # Sequence[Any], ndarray[Any, Any], NDFrame]"; expected "Union
        # [SupportsDunderLE, SupportsDunderGE, SupportsDunderGT, SupportsDunderLT]"
        ops[0](data, left)  # type: ignore[arg-type]
        if left is not None
        else np.full(data.shape, True, dtype=bool)
    )
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = (
        # error: Argument 2 to "le" has incompatible type "Union[str, float,
        # Period, Timedelta, Interval[Any], datetime64, timedelta64, datetime,
        # Sequence[Any], ndarray[Any, Any], NDFrame]"; expected "Union
        # [SupportsDunderLE, SupportsDunderGE, SupportsDunderGT, SupportsDunderLT]"
        ops[1](data, right)  # type: ignore[arg-type]
        if right is not None
        else np.full(data.shape, True, dtype=bool)
    )
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, "")


def _highlight_value(data: DataFrame | Series, op: str, props: str) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):  # min/max must be done twice to return scalar
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, "")


def _bar(
    data: NDFrame,
    align: str | float | Callable,
    colors: str | list | tuple,
    cmap: Any,
    width: float,
    height: float,
    vmin: float | None,
    vmax: float | None,
    base_css: str,
):
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underling subset of Styler data on which operations are performed.
    align : str in {"left", "right", "mid", "zero", "mean"}, int, float, callable
        Method for how bars are structured or scalar value of centre point.
    colors : list-like of str
        Two listed colors as string in valid CSS.
    width : float in [0,1]
        The percentage of the cell, measured from left, where drawn bars will reside.
    height : float in [0,1]
        The percentage of the cell's height where drawn bars will reside, centrally
        aligned.
    vmin : float, optional
        Overwrite the minimum value of the window.
    vmax : float, optional
        Overwrite the maximum value of the window.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.
    """

    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.

        Parameters
        ----------
        start : float
            Relative positional start of bar coloring in [0,1]
        end : float
            Relative positional end of the bar coloring in [0,1]
        color : str
            CSS valid color to apply.

        Returns
        -------
        str : The CSS applicable to the cell.

        Notes
        -----
        Uses ``base_css`` from outer scope.
        """
        cell_css = base_css
        if end > start:
            cell_css += "background: linear-gradient(90deg,"
            if start > 0:
                cell_css += f" transparent {start*100:.1f}%, {color} {start*100:.1f}%,"
            cell_css += f" {color} {end*100:.1f}%, transparent {end*100:.1f}%)"
        return cell_css

    def css_calc(x, left: float, right: float, align: str, color: str | list | tuple):
        """
        Return the correct CSS for bar placement based on calculated values.

        Parameters
        ----------
        x : float
            Value which determines the bar placement.
        left : float
            Value marking the left side of calculation, usually minimum of data.
        right : float
            Value marking the right side of the calculation, usually maximum of data
            (left < right).
        align : {"left", "right", "zero", "mid"}
            How the bars will be positioned.
            "left", "right", "zero" can be used with any values for ``left``, ``right``.
            "mid" can only be used where ``left <= 0`` and ``right >= 0``.
            "zero" is used to specify a center when all values ``x``, ``left``,
            ``right`` are translated, e.g. by say a mean or median.

        Returns
        -------
        str : Resultant CSS with linear gradient.

        Notes
        -----
        Uses ``colors``, ``width`` and ``height`` from outer scope.
        """
        if pd.isna(x):
            return base_css

        if isinstance(color, (list, tuple)):
            color = color[0] if x < 0 else color[1]
        assert isinstance(color, str)  # mypy redefinition

        x = left if x < left else x
        x = right if x > right else x  # trim data if outside of the window

        start: float = 0
        end: float = 1

        if align == "left":
            # all proportions are measured from the left side between left and right
            end = (x - left) / (right - left)

        elif align == "right":
            # all proportions are measured from the right side between left and right
            start = (x - left) / (right - left)

        else:
            z_frac: float = 0.5  # location of zero based on the left-right range
            if align == "zero":
                # all proportions are measured from the center at zero
                limit: float = max(abs(left), abs(right))
                left, right = -limit, limit
            elif align == "mid":
                # bars drawn from zero either leftwards or rightwards with center at mid
                mid: float = (left + right) / 2
                z_frac = (
                    -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)
                )

            if x < 0:
                start, end = (x - left) / (right - left), z_frac
            else:
                start, end = z_frac, (x - left) / (right - left)

        ret = css_bar(start * width, end * width, color)
        if height < 1 and "background: linear-gradient(" in ret:
            return (
                ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
            )
        else:
            return ret

    values = data.to_numpy()
    # A tricky way to address the issue where np.nanmin/np.nanmax fail to handle pd.NA.
    left = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0  # adjustment to translate data

    if align == "mid":
        if left >= 0:  # "mid" is documented to act as "left" if all values positive
            align, left = "left", 0 if vmin is None else vmin
        elif right <= 0:  # "mid" is documented to act as "right" if all values negative
            align, right = "right", 0 if vmax is None else vmax
    elif align == "mean":
        z, align = np.nanmean(values), "zero"
    elif callable(align):
        z, align = align(values), "zero"
    elif isinstance(align, (float, int)):
        z, align = float(align), "zero"
    elif align not in ("left", "right", "zero"):
        raise ValueError(
            "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a "
            "value defining the center line or a callable that returns a float"
        )

    rgbas = None
    if cmap is not None:
        # use the matplotlib colormap input
        with _mpl(Styler.bar) as (_, _matplotlib):
            cmap = (
                _matplotlib.colormaps[cmap]
                if isinstance(cmap, str)
                else cmap  # assumed to be a Colormap instance as documented
            )
            norm = _matplotlib.colors.Normalize(left, right)
            rgbas = cmap(norm(values))
            if data.ndim == 1:
                rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgbas]
            else:
                rgbas = [
                    [_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgbas
                ]

    assert isinstance(align, str)  # mypy: should now be in [left, right, mid, zero]
    if data.ndim == 1:
        return [
            css_calc(
                x - z, left - z, right - z, align, colors if rgbas is None else rgbas[i]
            )
            for i, x in enumerate(values)
        ]
    else:
        return np.array(
            [
                [
                    css_calc(
                        x - z,
                        left - z,
                        right - z,
                        align,
                        colors if rgbas is None else rgbas[i][j],
                    )
                    for j, x in enumerate(row)
                ]
                for i, row in enumerate(values)
            ]
        )
<|endoftext|>from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from pandas._config import get_option

from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        Dtype,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


frame_max_cols_sub = dedent(
    """\
    max_cols : int, optional
        When to switch from the verbose to the truncated output. If the
        DataFrame has more than `max_cols` columns, the truncated output
        is used. By default, the setting in
        ``pandas.options.display.max_info_columns`` is used."""
)


show_counts_sub = dedent(
    """\
    show_counts : bool, optional
        Whether to show the non-null counts. By default, this is shown
        only if the DataFrame is smaller than
        ``pandas.options.display.max_info_rows`` and
        ``pandas.options.display.max_info_columns``. A value of True always
        shows the counts, and False never shows the counts."""
)


frame_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
    ...                   "float_col": float_values})
    >>> df
        int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta       0.25
    2        3    gamma       0.50
    3        4    delta       0.75
    4        5  epsilon       1.00

    Prints information of all columns:

    >>> df.info(verbose=True)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Prints a summary of columns count and its dtypes but not per column
    information:

    >>> df.info(verbose=False)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Columns: 3 entries, int_col to float_col
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Pipe output of DataFrame.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> df.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big DataFrames and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> df = pd.DataFrame({
    ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
    ... })
    >>> df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 22.9+ MB

    >>> df.info(memory_usage='deep')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 165.9 MB"""
)


frame_see_also_sub = dedent(
    """\
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns.
    DataFrame.memory_usage: Memory usage of DataFrame columns."""
)


frame_sub_kwargs = {
    "klass": "DataFrame",
    "type_sub": " and columns",
    "max_cols_sub": frame_max_cols_sub,
    "show_counts_sub": show_counts_sub,
    "examples_sub": frame_examples_sub,
    "see_also_sub": frame_see_also_sub,
    "version_added_sub": "",
}


series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> s = pd.Series(text_values, index=int_values)
    >>> s.info()
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    5 non-null      object
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Prints a summary excluding information about its values:

    >>> s.info(verbose=False)
    <class 'pandas.core.series.Series'>
    Index: 5 entries, 1 to 5
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Pipe output of Series.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> s.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big Series and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> s = pd.Series(np.random.choice(['a', 'b', 'c'], 10 ** 6))
    >>> s.info()
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 7.6+ MB

    >>> s.info(memory_usage='deep')
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 55.3 MB"""
)


series_see_also_sub = dedent(
    """\
    Series.describe: Generate descriptive statistics of Series.
    Series.memory_usage: Memory usage of Series."""
)


series_sub_kwargs = {
    "klass": "Series",
    "type_sub": "",
    "max_cols_sub": "",
    "show_counts_sub": show_counts_sub,
    "examples_sub": series_examples_sub,
    "see_also_sub": series_see_also_sub,
    "version_added_sub": "\n.. versionadded:: 1.4.0\n",
}


INFO_DOCSTRING = dedent(
    """
    Print a concise summary of a {klass}.

    This method prints information about a {klass} including
    the index dtype{type_sub}, non-null values and memory usage.
    {version_added_sub}\

    Parameters
    ----------
    verbose : bool, optional
        Whether to print the full summary. By default, the setting in
        ``pandas.options.display.max_info_columns`` is followed.
    buf : writable buffer, defaults to sys.stdout
        Where to send the output. By default, the output is printed to
        sys.stdout. Pass a writable buffer if you need to further process
        the output.
    {max_cols_sub}
    memory_usage : bool, str, optional
        Specifies whether total memory usage of the {klass}
        elements (including the index) should be displayed. By default,
        this follows the ``pandas.options.display.memory_usage`` setting.

        True always show memory usage. False never shows memory usage.
        A value of 'deep' is equivalent to "True with deep introspection".
        Memory usage is shown in human-readable units (base-2
        representation). Without deep introspection a memory estimation is
        made based in column dtype and number of rows assuming values
        consume the same memory amount for corresponding dtypes. With deep
        memory introspection, a real memory usage calculation is performed
        at the cost of computational resources. See the
        :ref:`Frequently Asked Questions <df-memory-usage>` for more
        details.
    {show_counts_sub}

    Returns
    -------
    None
        This method prints a summary of a {klass} and returns None.

    See Also
    --------
    {see_also_sub}

    Examples
    --------
    {examples_sub}
    """
)


def _put_str(s: str | Dtype, space: int) -> str:
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, '')
    '22.5 KB'

    >>> _sizeof_fmt(23028, '+')
    '22.5+ KB'
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    return f"{num:3.1f}{size_qualifier} PB"


def _initialize_memory_usage(
    memory_usage: bool | str | None = None,
) -> bool | str:
    """Get memory usage based on inputs and display options."""
    if memory_usage is None:
        memory_usage = get_option("display.memory_usage")
    return memory_usage


class _BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    data: DataFrame | Series
    memory_usage: bool | str

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        return f"{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n"

    @property
    def size_qualifier(self) -> str:
        size_qualifier = ""
        if self.memory_usage:
            if self.memory_usage != "deep":
                # size_qualifier is just a best effort; not guaranteed to catch
                # all cases (e.g., it misses categorical data even with object
                # categories)
                if (
                    "object" in self.dtype_counts
                    or self.data.index._is_memory_usage_qualified()
                ):
                    size_qualifier = "+"
        return size_qualifier

    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        pass


class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self,
        data: DataFrame,
        memory_usage: bool | str | None = None,
    ) -> None:
        self.data: DataFrame = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        printer = _DataFrameInfoPrinter(
            info=self,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)


class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(
        self,
        data: Series,
        memory_usage: bool | str | None = None,
    ) -> None:
        self.data: Series = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        if max_cols is not None:
            raise ValueError(
                "Argument `max_cols` can only be passed "
                "in DataFrame.info, not Series.info"
            )
        printer = _SeriesInfoPrinter(
            info=self,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)

    @property
    def non_null_counts(self) -> Sequence[int]:
        return [self.data.count()]

    @property
    def dtypes(self) -> Iterable[Dtype]:
        return [self.data.dtypes]

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        from pandas.core.frame import DataFrame

        return _get_dataframe_dtype_counts(DataFrame(self.data))

    @property
    def memory_usage_bytes(self) -> int:
        """Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep)


class _InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe info into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:  # pragma: no cover
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""


class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return get_option("display.max_info_rows", len(self.data) + 1)

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        if max_cols is None:
            return get_option("display.max_info_columns", self.col_count + 1)
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return bool(not self.exceeds_info_cols and not self.exceeds_info_rows)
        else:
            return show_counts

    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        elif self.verbose is False:  # specifically set to False, not necessarily None
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )


class _SeriesInfoPrinter(_InfoPrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: SeriesInfo,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.show_counts = self._initialize_show_counts(show_counts)

    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            return _SeriesTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        else:
            return _SeriesTableBuilderNonVerbose(info=self.info)

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return True
        else:
            return show_counts


class _TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    _lines: list[str]
    info: _BaseInfo

    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    def non_null_counts(self) -> Sequence[int]:
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [
            f"{key}({val:d})" for key, val in sorted(self.dtype_counts.items())
        ]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")


class _DataFrameTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo) -> None:
        self.info: DataFrameInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self._lines.append(f"Empty {type(self.data).__name__}\n")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        return self.info.ids

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")


class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        self._lines.append(self.ids._summary(name="Columns"))


class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [
            max(*widths)
            for widths in zip(self.header_column_widths, body_column_widths)
        ]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max(len(x) for x in col) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths
                )
            ]
        )
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)


class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: DataFrameInfo,
        with_counts: bool,
    ) -> None:
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [" # ", "Column", "Non-Null Count", "Dtype"]
        return [" # ", "Column", "Dtype"]

    def add_columns_summary_line(self) -> None:
        self._lines.append(f"Data columns (total {self.col_count} columns):")

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_dtypes(),
        )

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )

    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f" {i}"

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


class _SeriesTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """

    def __init__(self, *, info: SeriesInfo) -> None:
        self.info: SeriesInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        self._fill_non_empty_info()
        return self._lines

    @property
    def data(self) -> Series:
        """Series."""
        return self.info.data

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""


class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    """
    Series info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()


class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: SeriesInfo,
        with_counts: bool,
    ) -> None:
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_series_name_line(self) -> None:
        self._lines.append(f"Series name: {self.data.name}")

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return ["Non-Null Count", "Dtype"]
        return ["Dtype"]

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from self._gen_dtypes()

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )


def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
    # groupby dtype.name to collect e.g. Categorical columns
    return df.dtypes.value_counts().groupby(lambda x: x.name).sum()
<|endoftext|>from __future__ import annotations

import mmap
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import (
    BaseExcelReader,
    ExcelWriter,
)
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from openpyxl import Workbook
    from openpyxl.descriptors.serialisable import Serialisable

    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
        WriteExcelBuffer,
    )


class OpenpyxlWriter(ExcelWriter):
    _engine = "openpyxl"
    _supported_extensions = (".xlsx", ".xlsm")

    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # Use the openpyxl module as the Excel writer.
        from openpyxl.workbook import Workbook

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        super().__init__(
            path,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        # ExcelWriter replaced "a" by "r+" to allow us to first read the excel file from
        # the file and later write to it
        if "r+" in self._mode:  # Load from existing workbook
            from openpyxl import load_workbook

            try:
                self._book = load_workbook(self._handles.handle, **engine_kwargs)
            except TypeError:
                self._handles.handle.close()
                raise
            self._handles.handle.seek(0)
        else:
            # Create workbook object with default optimized_write=True.
            try:
                self._book = Workbook(**engine_kwargs)
            except TypeError:
                self._handles.handle.close()
                raise

            if self.book.worksheets:
                self.book.remove(self.book.worksheets[0])

    @property
    def book(self) -> Workbook:
        """
        Book instance of class openpyxl.workbook.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        result = {name: self.book[name] for name in self.book.sheetnames}
        return result

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        self.book.save(self._handles.handle)
        if "r+" in self._mode and not isinstance(self._handles.handle, mmap.mmap):
            # truncate file to the written content
            self._handles.handle.truncate()

    @classmethod
    def _convert_to_style_kwargs(cls, style_dict: dict) -> dict[str, Serialisable]:
        """
        Convert a style_dict to a set of kwargs suitable for initializing
        or updating-on-copy an openpyxl v2 style object.

        Parameters
        ----------
        style_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'font'
                'fill'
                'border' ('borders')
                'alignment'
                'number_format'
                'protection'

        Returns
        -------
        style_kwargs : dict
            A dict with the same, normalized keys as ``style_dict`` but each
            value has been replaced with a native openpyxl style object of the
            appropriate class.
        """
        _style_key_map = {"borders": "border"}

        style_kwargs: dict[str, Serialisable] = {}
        for k, v in style_dict.items():
            k = _style_key_map.get(k, k)
            _conv_to_x = getattr(cls, f"_convert_to_{k}", lambda x: None)
            new_v = _conv_to_x(v)
            if new_v:
                style_kwargs[k] = new_v

        return style_kwargs

    @classmethod
    def _convert_to_color(cls, color_spec):
        """
        Convert ``color_spec`` to an openpyxl v2 Color object.

        Parameters
        ----------
        color_spec : str, dict
            A 32-bit ARGB hex string, or a dict with zero or more of the
            following keys.
                'rgb'
                'indexed'
                'auto'
                'theme'
                'tint'
                'index'
                'type'

        Returns
        -------
        color : openpyxl.styles.Color
        """
        from openpyxl.styles import Color

        if isinstance(color_spec, str):
            return Color(color_spec)
        else:
            return Color(**color_spec)

    @classmethod
    def _convert_to_font(cls, font_dict):
        """
        Convert ``font_dict`` to an openpyxl v2 Font object.

        Parameters
        ----------
        font_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'name'
                'size' ('sz')
                'bold' ('b')
                'italic' ('i')
                'underline' ('u')
                'strikethrough' ('strike')
                'color'
                'vertAlign' ('vertalign')
                'charset'
                'scheme'
                'family'
                'outline'
                'shadow'
                'condense'

        Returns
        -------
        font : openpyxl.styles.Font
        """
        from openpyxl.styles import Font

        _font_key_map = {
            "sz": "size",
            "b": "bold",
            "i": "italic",
            "u": "underline",
            "strike": "strikethrough",
            "vertalign": "vertAlign",
        }

        font_kwargs = {}
        for k, v in font_dict.items():
            k = _font_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            font_kwargs[k] = v

        return Font(**font_kwargs)

    @classmethod
    def _convert_to_stop(cls, stop_seq):
        """
        Convert ``stop_seq`` to a list of openpyxl v2 Color objects,
        suitable for initializing the ``GradientFill`` ``stop`` parameter.

        Parameters
        ----------
        stop_seq : iterable
            An iterable that yields objects suitable for consumption by
            ``_convert_to_color``.

        Returns
        -------
        stop : list of openpyxl.styles.Color
        """
        return map(cls._convert_to_color, stop_seq)

    @classmethod
    def _convert_to_fill(cls, fill_dict: dict[str, Any]):
        """
        Convert ``fill_dict`` to an openpyxl v2 Fill object.

        Parameters
        ----------
        fill_dict : dict
            A dict with one or more of the following keys (or their synonyms),
                'fill_type' ('patternType', 'patterntype')
                'start_color' ('fgColor', 'fgcolor')
                'end_color' ('bgColor', 'bgcolor')
            or one or more of the following keys (or their synonyms).
                'type' ('fill_type')
                'degree'
                'left'
                'right'
                'top'
                'bottom'
                'stop'

        Returns
        -------
        fill : openpyxl.styles.Fill
        """
        from openpyxl.styles import (
            GradientFill,
            PatternFill,
        )

        _pattern_fill_key_map = {
            "patternType": "fill_type",
            "patterntype": "fill_type",
            "fgColor": "start_color",
            "fgcolor": "start_color",
            "bgColor": "end_color",
            "bgcolor": "end_color",
        }

        _gradient_fill_key_map = {"fill_type": "type"}

        pfill_kwargs = {}
        gfill_kwargs = {}
        for k, v in fill_dict.items():
            pk = _pattern_fill_key_map.get(k)
            gk = _gradient_fill_key_map.get(k)
            if pk in ["start_color", "end_color"]:
                v = cls._convert_to_color(v)
            if gk == "stop":
                v = cls._convert_to_stop(v)
            if pk:
                pfill_kwargs[pk] = v
            elif gk:
                gfill_kwargs[gk] = v
            else:
                pfill_kwargs[k] = v
                gfill_kwargs[k] = v

        try:
            return PatternFill(**pfill_kwargs)
        except TypeError:
            return GradientFill(**gfill_kwargs)

    @classmethod
    def _convert_to_side(cls, side_spec):
        """
        Convert ``side_spec`` to an openpyxl v2 Side object.

        Parameters
        ----------
        side_spec : str, dict
            A string specifying the border style, or a dict with zero or more
            of the following keys (or their synonyms).
                'style' ('border_style')
                'color'

        Returns
        -------
        side : openpyxl.styles.Side
        """
        from openpyxl.styles import Side

        _side_key_map = {"border_style": "style"}

        if isinstance(side_spec, str):
            return Side(style=side_spec)

        side_kwargs = {}
        for k, v in side_spec.items():
            k = _side_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            side_kwargs[k] = v

        return Side(**side_kwargs)

    @classmethod
    def _convert_to_border(cls, border_dict):
        """
        Convert ``border_dict`` to an openpyxl v2 Border object.

        Parameters
        ----------
        border_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'left'
                'right'
                'top'
                'bottom'
                'diagonal'
                'diagonal_direction'
                'vertical'
                'horizontal'
                'diagonalUp' ('diagonalup')
                'diagonalDown' ('diagonaldown')
                'outline'

        Returns
        -------
        border : openpyxl.styles.Border
        """
        from openpyxl.styles import Border

        _border_key_map = {"diagonalup": "diagonalUp", "diagonaldown": "diagonalDown"}

        border_kwargs = {}
        for k, v in border_dict.items():
            k = _border_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            if k in ["left", "right", "top", "bottom", "diagonal"]:
                v = cls._convert_to_side(v)
            border_kwargs[k] = v

        return Border(**border_kwargs)

    @classmethod
    def _convert_to_alignment(cls, alignment_dict):
        """
        Convert ``alignment_dict`` to an openpyxl v2 Alignment object.

        Parameters
        ----------
        alignment_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'horizontal'
                'vertical'
                'text_rotation'
                'wrap_text'
                'shrink_to_fit'
                'indent'
        Returns
        -------
        alignment : openpyxl.styles.Alignment
        """
        from openpyxl.styles import Alignment

        return Alignment(**alignment_dict)

    @classmethod
    def _convert_to_number_format(cls, number_format_dict):
        """
        Convert ``number_format_dict`` to an openpyxl v2.1.0 number format
        initializer.

        Parameters
        ----------
        number_format_dict : dict
            A dict with zero or more of the following keys.
                'format_code' : str

        Returns
        -------
        number_format : str
        """
        return number_format_dict["format_code"]

    @classmethod
    def _convert_to_protection(cls, protection_dict):
        """
        Convert ``protection_dict`` to an openpyxl v2 Protection object.

        Parameters
        ----------
        protection_dict : dict
            A dict with zero or more of the following keys.
                'locked'
                'hidden'

        Returns
        -------
        """
        from openpyxl.styles import Protection

        return Protection(**protection_dict)

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        # Write the frame cells using openpyxl.
        sheet_name = self._get_sheet_name(sheet_name)

        _style_cache: dict[str, dict[str, Serialisable]] = {}

        if sheet_name in self.sheets and self._if_sheet_exists != "new":
            if "r+" in self._mode:
                if self._if_sheet_exists == "replace":
                    old_wks = self.sheets[sheet_name]
                    target_index = self.book.index(old_wks)
                    del self.book[sheet_name]
                    wks = self.book.create_sheet(sheet_name, target_index)
                elif self._if_sheet_exists == "error":
                    raise ValueError(
                        f"Sheet '{sheet_name}' already exists and "
                        f"if_sheet_exists is set to 'error'."
                    )
                elif self._if_sheet_exists == "overlay":
                    wks = self.sheets[sheet_name]
                else:
                    raise ValueError(
                        f"'{self._if_sheet_exists}' is not valid for if_sheet_exists. "
                        "Valid options are 'error', 'new', 'replace' and 'overlay'."
                    )
            else:
                wks = self.sheets[sheet_name]
        else:
            wks = self.book.create_sheet()
            wks.title = sheet_name

        if validate_freeze_panes(freeze_panes):
            freeze_panes = cast(tuple[int, int], freeze_panes)
            wks.freeze_panes = wks.cell(
                row=freeze_panes[0] + 1, column=freeze_panes[1] + 1
            )

        for cell in cells:
            xcell = wks.cell(
                row=startrow + cell.row + 1, column=startcol + cell.col + 1
            )
            xcell.value, fmt = self._value_with_fmt(cell.val)
            if fmt:
                xcell.number_format = fmt

            style_kwargs: dict[str, Serialisable] | None = {}
            if cell.style:
                key = str(cell.style)
                style_kwargs = _style_cache.get(key)
                if style_kwargs is None:
                    style_kwargs = self._convert_to_style_kwargs(cell.style)
                    _style_cache[key] = style_kwargs

            if style_kwargs:
                for k, v in style_kwargs.items():
                    setattr(xcell, k, v)

            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_cells(
                    start_row=startrow + cell.row + 1,
                    start_column=startcol + cell.col + 1,
                    end_column=startcol + cell.mergeend + 1,
                    end_row=startrow + cell.mergestart + 1,
                )

                # When cells are merged only the top-left cell is preserved
                # The behaviour of the other cells in a merged range is
                # undefined
                if style_kwargs:
                    first_row = startrow + cell.row + 1
                    last_row = startrow + cell.mergestart + 1
                    first_col = startcol + cell.col + 1
                    last_col = startcol + cell.mergeend + 1

                    for row in range(first_row, last_row + 1):
                        for col in range(first_col, last_col + 1):
                            if row == first_row and col == first_col:
                                # Ignore first cell. It is already handled.
                                continue
                            xcell = wks.cell(column=col, row=row)
                            for k, v in style_kwargs.items():
                                setattr(xcell, k, v)


class OpenpyxlReader(BaseExcelReader["Workbook"]):
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Reader using openpyxl engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("openpyxl")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[Workbook]:
        from openpyxl import Workbook

        return Workbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> Workbook:
        from openpyxl import load_workbook

        default_kwargs = {"read_only": True, "data_only": True, "keep_links": False}

        return load_workbook(
            filepath_or_buffer,
            **(default_kwargs | engine_kwargs),
        )

    @property
    def sheet_names(self) -> list[str]:
        return [sheet.title for sheet in self.book.worksheets]

    def get_sheet_by_name(self, name: str):
        self.raise_if_bad_sheet_by_name(name)
        return self.book[name]

    def get_sheet_by_index(self, index: int):
        self.raise_if_bad_sheet_by_index(index)
        return self.book.worksheets[index]

    def _convert_cell(self, cell) -> Scalar:
        from openpyxl.cell.cell import (
            TYPE_ERROR,
            TYPE_NUMERIC,
        )

        if cell.value is None:
            return ""  # compat with xlrd
        elif cell.data_type == TYPE_ERROR:
            return np.nan
        elif cell.data_type == TYPE_NUMERIC:
            val = int(cell.value)
            if val == cell.value:
                return val
            return float(cell.value)

        return cell.value

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar]]:
        if self.book.read_only:
            sheet.reset_dimensions()

        data: list[list[Scalar]] = []
        last_row_with_data = -1
        for row_number, row in enumerate(sheet.rows):
            converted_row = [self._convert_cell(cell) for cell in row]
            while converted_row and converted_row[-1] == "":
                # trim trailing empty elements
                converted_row.pop()
            if converted_row:
                last_row_with_data = row_number
            data.append(converted_row)
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break

        # Trim trailing empty rows
        data = data[: last_row_with_data + 1]

        if len(data) > 0:
            # extend rows to max width
            max_width = max(len(data_row) for data_row in data)
            if min(len(data_row) for data_row in data) < max_width:
                empty_cell: list[Scalar] = [""]
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]

        return data
<|endoftext|>from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

import pandas as pd
from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from python_calamine import (
        CalamineSheet,
        CalamineWorkbook,
    )

    from pandas._typing import (
        FilePath,
        NaTType,
        ReadBuffer,
        Scalar,
        StorageOptions,
    )

_CellValue = Union[int, float, str, bool, time, date, datetime, timedelta]


class CalamineReader(BaseExcelReader["CalamineWorkbook"]):
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Reader using calamine engine (xlsx/xls/xlsb/ods).

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("python_calamine")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[CalamineWorkbook]:
        from python_calamine import CalamineWorkbook

        return CalamineWorkbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any
    ) -> CalamineWorkbook:
        from python_calamine import load_workbook

        return load_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self) -> list[str]:
        from python_calamine import SheetTypeEnum

        return [
            sheet.name
            for sheet in self.book.sheets_metadata
            if sheet.typ == SheetTypeEnum.WorkSheet
        ]

    def get_sheet_by_name(self, name: str) -> CalamineSheet:
        self.raise_if_bad_sheet_by_name(name)
        return self.book.get_sheet_by_name(name)

    def get_sheet_by_index(self, index: int) -> CalamineSheet:
        self.raise_if_bad_sheet_by_index(index)
        return self.book.get_sheet_by_index(index)

    def get_sheet_data(
        self, sheet: CalamineSheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar | NaTType | time]]:
        def _convert_cell(value: _CellValue) -> Scalar | NaTType | time:
            if isinstance(value, float):
                val = int(value)
                if val == value:
                    return val
                else:
                    return value
            elif isinstance(value, date):
                return pd.Timestamp(value)
            elif isinstance(value, timedelta):
                return pd.Timedelta(value)
            elif isinstance(value, time):
                return value

            return value

        rows: list[list[_CellValue]] = sheet.to_python(
            skip_empty_area=False, nrows=file_rows_needed
        )
        data = [[_convert_cell(cell) for cell in row] for row in rows]

        return data
<|endoftext|># pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from pyxlsb import Workbook

    from pandas._typing import (
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
    )


class PyxlsbReader(BaseExcelReader["Workbook"]):
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Reader using pyxlsb engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object, or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("pyxlsb")
        # This will call load_workbook on the filepath or buffer
        # And set the result to the book-attribute
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[Workbook]:
        from pyxlsb import Workbook

        return Workbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> Workbook:
        from pyxlsb import open_workbook

        # TODO: hack in buffer capability
        # This might need some modifications to the Pyxlsb library
        # Actual work for opening it is in xlsbpackage.py, line 20-ish

        return open_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self) -> list[str]:
        return self.book.sheets

    def get_sheet_by_name(self, name: str):
        self.raise_if_bad_sheet_by_name(name)
        return self.book.get_sheet(name)

    def get_sheet_by_index(self, index: int):
        self.raise_if_bad_sheet_by_index(index)
        # pyxlsb sheets are indexed from 1 onwards
        # There's a fix for this in the source, but the pypi package doesn't have it
        return self.book.get_sheet(index + 1)

    def _convert_cell(self, cell) -> Scalar:
        # TODO: there is no way to distinguish between floats and datetimes in pyxlsb
        # This means that there is no way to read datetime types from an xlsb file yet
        if cell.v is None:
            return ""  # Prevents non-named columns from not showing up as Unnamed: i
        if isinstance(cell.v, float):
            val = int(cell.v)
            if val == cell.v:
                return val
            else:
                return float(cell.v)

        return cell.v

    def get_sheet_data(
        self,
        sheet,
        file_rows_needed: int | None = None,
    ) -> list[list[Scalar]]:
        data: list[list[Scalar]] = []
        previous_row_number = -1
        # When sparse=True the rows can have different lengths and empty rows are
        # not returned. The cells are namedtuples of row, col, value (r, c, v).
        for row in sheet.rows(sparse=True):
            row_number = row[0].r
            converted_row = [self._convert_cell(cell) for cell in row]
            while converted_row and converted_row[-1] == "":
                # trim trailing empty elements
                converted_row.pop()
            if converted_row:
                data.extend([[]] * (row_number - previous_row_number - 1))
                data.append(converted_row)
                previous_row_number = row_number
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break
        if data:
            # extend rows to max_width
            max_width = max(len(data_row) for data_row in data)
            if min(len(data_row) for data_row in data) < max_width:
                empty_cell: list[Scalar] = [""]
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]
        return data
<|endoftext|>from __future__ import annotations

from datetime import time
import math
from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from xlrd import Book

    from pandas._typing import (
        Scalar,
        StorageOptions,
    )


class XlrdReader(BaseExcelReader["Book"]):
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        err_msg = "Install xlrd >= 2.0.1 for xls Excel support"
        import_optional_dependency("xlrd", extra=err_msg)
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[Book]:
        from xlrd import Book

        return Book

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> Book:
        from xlrd import open_workbook

        if hasattr(filepath_or_buffer, "read"):
            data = filepath_or_buffer.read()
            return open_workbook(file_contents=data, **engine_kwargs)
        else:
            return open_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self):
        return self.book.sheet_names()

    def get_sheet_by_name(self, name):
        self.raise_if_bad_sheet_by_name(name)
        return self.book.sheet_by_name(name)

    def get_sheet_by_index(self, index):
        self.raise_if_bad_sheet_by_index(index)
        return self.book.sheet_by_index(index)

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar]]:
        from xlrd import (
            XL_CELL_BOOLEAN,
            XL_CELL_DATE,
            XL_CELL_ERROR,
            XL_CELL_NUMBER,
            xldate,
        )

        epoch1904 = self.book.datemode

        def _parse_cell(cell_contents, cell_typ):
            """
            converts the contents of the cell into a pandas appropriate object
            """
            if cell_typ == XL_CELL_DATE:
                # Use the newer xlrd datetime handling.
                try:
                    cell_contents = xldate.xldate_as_datetime(cell_contents, epoch1904)
                except OverflowError:
                    return cell_contents

                # Excel doesn't distinguish between dates and time,
                # so we treat dates on the epoch as times only.
                # Also, Excel supports 1900 and 1904 epochs.
                year = (cell_contents.timetuple())[0:3]
                if (not epoch1904 and year == (1899, 12, 31)) or (
                    epoch1904 and year == (1904, 1, 1)
                ):
                    cell_contents = time(
                        cell_contents.hour,
                        cell_contents.minute,
                        cell_contents.second,
                        cell_contents.microsecond,
                    )

            elif cell_typ == XL_CELL_ERROR:
                cell_contents = np.nan
            elif cell_typ == XL_CELL_BOOLEAN:
                cell_contents = bool(cell_contents)
            elif cell_typ == XL_CELL_NUMBER:
                # GH5394 - Excel 'numbers' are always floats
                # it's a minimal perf hit and less surprising
                if math.isfinite(cell_contents):
                    # GH54564 - don't attempt to convert NaN/Inf
                    val = int(cell_contents)
                    if val == cell_contents:
                        cell_contents = val
            return cell_contents

        data = []

        nrows = sheet.nrows
        if file_rows_needed is not None:
            nrows = min(nrows, file_rows_needed)
        for i in range(nrows):
            row = [
                _parse_cell(value, typ)
                for value, typ in zip(sheet.row_values(i), sheet.row_types(i))
            ]
            data.append(row)

        return data
<|endoftext|>from pandas.io.excel._base import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.excel._odswriter import ODSWriter as _ODSWriter
from pandas.io.excel._openpyxl import OpenpyxlWriter as _OpenpyxlWriter
from pandas.io.excel._util import register_writer
from pandas.io.excel._xlsxwriter import XlsxWriter as _XlsxWriter

__all__ = ["read_excel", "ExcelWriter", "ExcelFile"]


register_writer(_OpenpyxlWriter)

register_writer(_XlsxWriter)


register_writer(_ODSWriter)
<|endoftext|>from __future__ import annotations

from collections import defaultdict
import datetime
import json
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    cast,
    overload,
)

from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        StorageOptions,
        WriteExcelBuffer,
    )

    from pandas.io.formats.excel import ExcelCell


class ODSWriter(ExcelWriter):
    _engine = "odf"
    _supported_extensions = (".ods",)

    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format=None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        from odf.opendocument import OpenDocumentSpreadsheet

        if mode == "a":
            raise ValueError("Append mode is not supported with odf!")

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)
        self._book = OpenDocumentSpreadsheet(**engine_kwargs)

        super().__init__(
            path,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        self._style_dict: dict[str, str] = {}

    @property
    def book(self):
        """
        Book instance of class odf.opendocument.OpenDocumentSpreadsheet.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        from odf.table import Table

        result = {
            sheet.getAttribute("name"): sheet
            for sheet in self.book.getElementsByType(Table)
        }
        return result

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        for sheet in self.sheets.values():
            self.book.spreadsheet.addElement(sheet)
        self.book.save(self._handles.handle)

    def _write_cells(
        self,
        cells: list[ExcelCell],
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        """
        Write the frame cells using odf
        """
        from odf.table import (
            Table,
            TableCell,
            TableRow,
        )
        from odf.text import P

        sheet_name = self._get_sheet_name(sheet_name)
        assert sheet_name is not None

        if sheet_name in self.sheets:
            wks = self.sheets[sheet_name]
        else:
            wks = Table(name=sheet_name)
            self.book.spreadsheet.addElement(wks)

        if validate_freeze_panes(freeze_panes):
            freeze_panes = cast(tuple[int, int], freeze_panes)
            self._create_freeze_panes(sheet_name, freeze_panes)

        for _ in range(startrow):
            wks.addElement(TableRow())

        rows: DefaultDict = defaultdict(TableRow)
        col_count: DefaultDict = defaultdict(int)

        for cell in sorted(cells, key=lambda cell: (cell.row, cell.col)):
            # only add empty cells if the row is still empty
            if not col_count[cell.row]:
                for _ in range(startcol):
                    rows[cell.row].addElement(TableCell())

            # fill with empty cells if needed
            for _ in range(cell.col - col_count[cell.row]):
                rows[cell.row].addElement(TableCell())
                col_count[cell.row] += 1

            pvalue, tc = self._make_table_cell(cell)
            rows[cell.row].addElement(tc)
            col_count[cell.row] += 1
            p = P(text=pvalue)
            tc.addElement(p)

        # add all rows to the sheet
        if len(rows) > 0:
            for row_nr in range(max(rows.keys()) + 1):
                wks.addElement(rows[row_nr])

    def _make_table_cell_attributes(self, cell) -> dict[str, int | str]:
        """Convert cell attributes to OpenDocument attributes

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        attributes : Dict[str, Union[int, str]]
            Dictionary with attributes and attribute values
        """
        attributes: dict[str, int | str] = {}
        style_name = self._process_style(cell.style)
        if style_name is not None:
            attributes["stylename"] = style_name
        if cell.mergestart is not None and cell.mergeend is not None:
            attributes["numberrowsspanned"] = max(1, cell.mergestart)
            attributes["numbercolumnsspanned"] = cell.mergeend
        return attributes

    def _make_table_cell(self, cell) -> tuple[object, Any]:
        """Convert cell data to an OpenDocument spreadsheet cell

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        pvalue, cell : Tuple[str, TableCell]
            Display value, Cell value
        """
        from odf.table import TableCell

        attributes = self._make_table_cell_attributes(cell)
        val, fmt = self._value_with_fmt(cell.val)
        pvalue = value = val
        if isinstance(val, bool):
            value = str(val).lower()
            pvalue = str(val).upper()
            return (
                pvalue,
                TableCell(
                    valuetype="boolean",
                    booleanvalue=value,
                    attributes=attributes,
                ),
            )
        elif isinstance(val, datetime.datetime):
            # Fast formatting
            value = val.isoformat()
            # Slow but locale-dependent
            pvalue = val.strftime("%c")
            return (
                pvalue,
                TableCell(valuetype="date", datevalue=value, attributes=attributes),
            )
        elif isinstance(val, datetime.date):
            # Fast formatting
            value = f"{val.year}-{val.month:02d}-{val.day:02d}"
            # Slow but locale-dependent
            pvalue = val.strftime("%x")
            return (
                pvalue,
                TableCell(valuetype="date", datevalue=value, attributes=attributes),
            )
        elif isinstance(val, str):
            return (
                pvalue,
                TableCell(
                    valuetype="string",
                    stringvalue=value,
                    attributes=attributes,
                ),
            )
        else:
            return (
                pvalue,
                TableCell(
                    valuetype="float",
                    value=value,
                    attributes=attributes,
                ),
            )

    @overload
    def _process_style(self, style: dict[str, Any]) -> str:
        ...

    @overload
    def _process_style(self, style: None) -> None:
        ...

    def _process_style(self, style: dict[str, Any] | None) -> str | None:
        """Convert a style dictionary to a OpenDocument style sheet

        Parameters
        ----------
        style : Dict
            Style dictionary

        Returns
        -------
        style_key : str
            Unique style key for later reference in sheet
        """
        from odf.style import (
            ParagraphProperties,
            Style,
            TableCellProperties,
            TextProperties,
        )

        if style is None:
            return None
        style_key = json.dumps(style)
        if style_key in self._style_dict:
            return self._style_dict[style_key]
        name = f"pd{len(self._style_dict)+1}"
        self._style_dict[style_key] = name
        odf_style = Style(name=name, family="table-cell")
        if "font" in style:
            font = style["font"]
            if font.get("bold", False):
                odf_style.addElement(TextProperties(fontweight="bold"))
        if "borders" in style:
            borders = style["borders"]
            for side, thickness in borders.items():
                thickness_translation = {"thin": "0.75pt solid #000000"}
                odf_style.addElement(
                    TableCellProperties(
                        attributes={f"border{side}": thickness_translation[thickness]}
                    )
                )
        if "alignment" in style:
            alignment = style["alignment"]
            horizontal = alignment.get("horizontal")
            if horizontal:
                odf_style.addElement(ParagraphProperties(textalign=horizontal))
            vertical = alignment.get("vertical")
            if vertical:
                odf_style.addElement(TableCellProperties(verticalalign=vertical))
        self.book.styles.addElement(odf_style)
        return name

    def _create_freeze_panes(
        self, sheet_name: str, freeze_panes: tuple[int, int]
    ) -> None:
        """
        Create freeze panes in the sheet.

        Parameters
        ----------
        sheet_name : str
            Name of the spreadsheet
        freeze_panes : tuple of (int, int)
            Freeze pane location x and y
        """
        from odf.config import (
            ConfigItem,
            ConfigItemMapEntry,
            ConfigItemMapIndexed,
            ConfigItemMapNamed,
            ConfigItemSet,
        )

        config_item_set = ConfigItemSet(name="ooo:view-settings")
        self.book.settings.addElement(config_item_set)

        config_item_map_indexed = ConfigItemMapIndexed(name="Views")
        config_item_set.addElement(config_item_map_indexed)

        config_item_map_entry = ConfigItemMapEntry()
        config_item_map_indexed.addElement(config_item_map_entry)

        config_item_map_named = ConfigItemMapNamed(name="Tables")
        config_item_map_entry.addElement(config_item_map_named)

        config_item_map_entry = ConfigItemMapEntry(name=sheet_name)
        config_item_map_named.addElement(config_item_map_entry)

        config_item_map_entry.addElement(
            ConfigItem(name="HorizontalSplitMode", type="short", text="2")
        )
        config_item_map_entry.addElement(
            ConfigItem(name="VerticalSplitMode", type="short", text="2")
        )
        config_item_map_entry.addElement(
            ConfigItem(
                name="HorizontalSplitPosition", type="int", text=str(freeze_panes[0])
            )
        )
        config_item_map_entry.addElement(
            ConfigItem(
                name="VerticalSplitPosition", type="int", text=str(freeze_panes[1])
            )
        )
        config_item_map_entry.addElement(
            ConfigItem(name="PositionRight", type="int", text=str(freeze_panes[0]))
        )
        config_item_map_entry.addElement(
            ConfigItem(name="PositionBottom", type="int", text=str(freeze_panes[1]))
        )
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    MutableMapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeVar,
    overload,
)

from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)

if TYPE_CHECKING:
    from pandas.io.excel._base import ExcelWriter

    ExcelWriter_t = type[ExcelWriter]
    usecols_func = TypeVar("usecols_func", bound=Callable[[Hashable], object])

_writers: MutableMapping[str, ExcelWriter_t] = {}


def register_writer(klass: ExcelWriter_t) -> None:
    """
    Add engine to the excel writer registry.io.excel.

    You must use this method to integrate with ``to_excel``.

    Parameters
    ----------
    klass : ExcelWriter
    """
    if not callable(klass):
        raise ValueError("Can only register callables as engines")
    engine_name = klass._engine
    _writers[engine_name] = klass


def get_default_engine(ext: str, mode: Literal["reader", "writer"] = "reader") -> str:
    """
    Return the default reader/writer for the given extension.

    Parameters
    ----------
    ext : str
        The excel file extension for which to get the default engine.
    mode : str {'reader', 'writer'}
        Whether to get the default engine for reading or writing.
        Either 'reader' or 'writer'

    Returns
    -------
    str
        The default engine for the extension.
    """
    _default_readers = {
        "xlsx": "openpyxl",
        "xlsm": "openpyxl",
        "xlsb": "pyxlsb",
        "xls": "xlrd",
        "ods": "odf",
    }
    _default_writers = {
        "xlsx": "openpyxl",
        "xlsm": "openpyxl",
        "xlsb": "pyxlsb",
        "ods": "odf",
    }
    assert mode in ["reader", "writer"]
    if mode == "writer":
        # Prefer xlsxwriter over openpyxl if installed
        xlsxwriter = import_optional_dependency("xlsxwriter", errors="warn")
        if xlsxwriter:
            _default_writers["xlsx"] = "xlsxwriter"
        return _default_writers[ext]
    else:
        return _default_readers[ext]


def get_writer(engine_name: str) -> ExcelWriter_t:
    try:
        return _writers[engine_name]
    except KeyError as err:
        raise ValueError(f"No Excel writer '{engine_name}'") from err


def _excel2num(x: str) -> int:
    """
    Convert Excel column name like 'AB' to 0-based column index.

    Parameters
    ----------
    x : str
        The Excel column name to convert to a 0-based column index.

    Returns
    -------
    num : int
        The column index corresponding to the name.

    Raises
    ------
    ValueError
        Part of the Excel column name was invalid.
    """
    index = 0

    for c in x.upper().strip():
        cp = ord(c)

        if cp < ord("A") or cp > ord("Z"):
            raise ValueError(f"Invalid column name: {x}")

        index = index * 26 + cp - ord("A") + 1

    return index - 1


def _range2cols(areas: str) -> list[int]:
    """
    Convert comma separated list of column names and ranges to indices.

    Parameters
    ----------
    areas : str
        A string containing a sequence of column ranges (or areas).

    Returns
    -------
    cols : list
        A list of 0-based column indices.

    Examples
    --------
    >>> _range2cols('A:E')
    [0, 1, 2, 3, 4]
    >>> _range2cols('A,C,Z:AB')
    [0, 2, 25, 26, 27]
    """
    cols: list[int] = []

    for rng in areas.split(","):
        if ":" in rng:
            rngs = rng.split(":")
            cols.extend(range(_excel2num(rngs[0]), _excel2num(rngs[1]) + 1))
        else:
            cols.append(_excel2num(rng))

    return cols


@overload
def maybe_convert_usecols(usecols: str | list[int]) -> list[int]:
    ...


@overload
def maybe_convert_usecols(usecols: list[str]) -> list[str]:
    ...


@overload
def maybe_convert_usecols(usecols: usecols_func) -> usecols_func:
    ...


@overload
def maybe_convert_usecols(usecols: None) -> None:
    ...


def maybe_convert_usecols(
    usecols: str | list[int] | list[str] | usecols_func | None,
) -> None | list[int] | list[str] | usecols_func:
    """
    Convert `usecols` into a compatible format for parsing in `parsers.py`.

    Parameters
    ----------
    usecols : object
        The use-columns object to potentially convert.

    Returns
    -------
    converted : object
        The compatible format of `usecols`.
    """
    if usecols is None:
        return usecols

    if is_integer(usecols):
        raise ValueError(
            "Passing an integer for `usecols` is no longer supported.  "
            "Please pass in a list of int from 0 to `usecols` inclusive instead."
        )

    if isinstance(usecols, str):
        return _range2cols(usecols)

    return usecols


@overload
def validate_freeze_panes(freeze_panes: tuple[int, int]) -> Literal[True]:
    ...


@overload
def validate_freeze_panes(freeze_panes: None) -> Literal[False]:
    ...


def validate_freeze_panes(freeze_panes: tuple[int, int] | None) -> bool:
    if freeze_panes is not None:
        if len(freeze_panes) == 2 and all(
            isinstance(item, int) for item in freeze_panes
        ):
            return True

        raise ValueError(
            "freeze_panes must be of form (row, column) "
            "where row and column are integers"
        )

    # freeze_panes wasn't specified, return False so it won't be applied
    # to output sheet
    return False


def fill_mi_header(
    row: list[Hashable], control_row: list[bool]
) -> tuple[list[Hashable], list[bool]]:
    """
    Forward fill blank entries in row but only inside the same parent index.

    Used for creating headers in Multiindex.

    Parameters
    ----------
    row : list
        List of items in a single row.
    control_row : list of bool
        Helps to determine if particular column is in same parent index as the
        previous value. Used to stop propagation of empty cells between
        different indexes.

    Returns
    -------
    Returns changed row and control_row
    """
    last = row[0]
    for i in range(1, len(row)):
        if not control_row[i]:
            last = row[i]

        if row[i] == "" or row[i] is None:
            row[i] = last
        else:
            control_row[i] = False
            last = row[i]

    return row, control_row


def pop_header_name(
    row: list[Hashable], index_col: int | Sequence[int]
) -> tuple[Hashable | None, list[Hashable]]:
    """
    Pop the header name for MultiIndex parsing.

    Parameters
    ----------
    row : list
        The data row to parse for the header name.
    index_col : int, list
        The index columns for our data. Assumed to be non-null.

    Returns
    -------
    header_name : str
        The extracted header name.
    trimmed_row : list
        The original data row with the header name removed.
    """
    # Pop out header name and fill w/blank.
    if is_list_like(index_col):
        assert isinstance(index_col, Iterable)
        i = max(index_col)
    else:
        assert not isinstance(index_col, Iterable)
        i = index_col

    header_name = row[i]
    header_name = None if header_name == "" else header_name

    return header_name, row[:i] + [""] + row[i + 1 :]


def combine_kwargs(engine_kwargs: dict[str, Any] | None, kwargs: dict) -> dict:
    """
    Used to combine two sources of kwargs for the backend engine.

    Use of kwargs is deprecated, this function is solely for use in 1.3 and should
    be removed in 1.4/2.0. Also _base.ExcelWriter.__new__ ensures either engine_kwargs
    or kwargs must be None or empty respectively.

    Parameters
    ----------
    engine_kwargs: dict
        kwargs to be passed through to the engine.
    kwargs: dict
        kwargs to be psased through to the engine (deprecated)

    Returns
    -------
    engine_kwargs combined with kwargs
    """
    if engine_kwargs is None:
        result = {}
    else:
        result = engine_kwargs.copy()
    result.update(kwargs)
    return result
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings
import zipfile

from pandas._config import config

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
    Appender,
    doc,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_bool,
    is_float,
    is_integer,
    is_list_like,
)

from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.excel._util import (
    fill_mi_header,
    get_default_engine,
    get_writer,
    maybe_convert_usecols,
    pop_header_name,
)
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        ExcelWriterIfSheetExists,
        FilePath,
        IntStrT,
        ReadBuffer,
        Self,
        SequenceNotStr,
        StorageOptions,
        WriteExcelBuffer,
    )
_read_excel_doc = (
    """
Read an Excel file into a ``pandas`` ``DataFrame``.

Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions
read from a local filesystem or URL. Supports an option to read
a single sheet or a list of sheets.

Parameters
----------
io : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.

    .. deprecated:: 2.1.0
        Passing byte strings is deprecated. To read from a
        byte string, wrap it in a ``BytesIO`` object.
sheet_name : str, int, list, or None, default 0
    Strings are used for sheet names. Integers are used in zero-indexed
    sheet positions (chart sheets do not count as a sheet position).
    Lists of strings/integers are used to request multiple sheets.
    Specify ``None`` to get all worksheets.

    Available cases:

    * Defaults to ``0``: 1st sheet as a `DataFrame`
    * ``1``: 2nd sheet as a `DataFrame`
    * ``"Sheet1"``: Load sheet with name "Sheet1"
    * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
      as a dict of `DataFrame`
    * ``None``: All worksheets.

header : int, list of int, default 0
    Row (0-indexed) to use for the column labels of the parsed
    DataFrame. If a list of integers is passed those row positions will
    be combined into a ``MultiIndex``. Use None if there is no header.
names : array-like, default None
    List of column names to use. If file contains no header row,
    then you should explicitly pass header=None.
index_col : int, str, list of int, default None
    Column (0-indexed) to use as the row labels of the DataFrame.
    Pass None if there is no such column.  If a list is passed,
    those columns will be combined into a ``MultiIndex``.  If a
    subset of data is selected with ``usecols``, index_col
    is based on the subset.

    Missing values will be forward filled to allow roundtripping with
    ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
    missing values use ``set_index`` after reading the data instead of
    ``index_col``.
usecols : str, list-like, or callable, default None
    * If None, then parse all columns.
    * If str, then indicates comma separated list of Excel column letters
      and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
      both sides.
    * If list of int, then indicates list of column numbers to be parsed
      (0-indexed).
    * If list of string, then indicates list of column names to be parsed.
    * If callable, then evaluate each column name against it and parse the
      column if the callable returns ``True``.

    Returns a subset of the columns according to behavior above.
dtype : Type name or dict of column -> type, default None
    Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32}}
    Use ``object`` to preserve data as stored in Excel and not interpret dtype,
    which will necessarily result in ``object`` dtype.
    If converters are specified, they will be applied INSTEAD
    of dtype conversion.
    If you use ``None``, it will infer the dtype of each column based on the data.
engine : {{'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}}, default None
    If io is not a buffer or path, this must be set to identify io.
    Engine compatibility :

    - ``openpyxl`` supports newer Excel file formats.
    - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
      and OpenDocument (.ods) file formats.
    - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
    - ``pyxlsb`` supports Binary Excel files.
    - ``xlrd`` supports old-style Excel files (.xls).

