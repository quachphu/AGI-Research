    When ``engine=None``, the following logic will be used to determine the engine:

    - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
      then `odf <https://pypi.org/project/odfpy/>`_ will be used.
    - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.
    - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.
    - Otherwise ``openpyxl`` will be used.
converters : dict, default None
    Dict of functions for converting values in certain columns. Keys can
    either be integers or column labels, values are functions that take one
    input argument, the Excel cell content, and return the transformed
    content.
true_values : list, default None
    Values to consider as True.
false_values : list, default None
    Values to consider as False.
skiprows : list-like, int, or callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (int) at the
    start of the file. If callable, the callable function will be evaluated
    against the row indices, returning True if the row should be skipped and
    False otherwise. An example of a valid callable argument would be ``lambda
    x: x in [0, 2]``.
nrows : int, default None
    Number of rows to parse.
na_values : scalar, str, list-like, or dict, default None
    Additional strings to recognize as NA/NaN. If dict passed, specific
    per-column NA values. By default the following values are interpreted
    as NaN: '"""
    + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    ")
    + """'.
keep_default_na : bool, default True
    Whether or not to include the default NaN values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is True, and ``na_values`` are specified,
      ``na_values`` is appended to the default NaN values used for parsing.
    * If ``keep_default_na`` is True, and ``na_values`` are not specified, only
      the default NaN values are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are specified, only
      the NaN values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are not specified, no
      strings will be parsed as NaN.

    Note that if `na_filter` is passed in as False, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of na_values). In
    data without any NAs, passing ``na_filter=False`` can improve the
    performance of reading a large file.
verbose : bool, default False
    Indicate number of NA values placed in non-numeric columns.
parse_dates : bool, list-like, or dict, default False
    The behavior is as follows:

    * ``bool``. If True -> try parsing the index.
    * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
      each as a separate date column.
    * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
      a single date column.
    * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
      result 'foo'

    If a column or index contains an unparsable date, the entire column or
    index will be returned unaltered as an object data type. If you don`t want to
    parse some cells as date just change their type in Excel to "Text".
    For non-standard datetime parsing, use ``pd.to_datetime`` after ``pd.read_excel``.

    Note: A fast-path exists for iso8601-formatted dates.
date_parser : function, optional
    Function to use for converting a sequence of string columns to an array of
    datetime instances. The default uses ``dateutil.parser.parser`` to do the
    conversion. Pandas will try to call `date_parser` in three different ways,
    advancing to the next if an exception occurs: 1) Pass one or more arrays
    (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the
    string values from the columns defined by `parse_dates` into a single array
    and pass that; and 3) call `date_parser` once for each row using one or
    more strings (corresponding to the columns defined by `parse_dates`) as
    arguments.

    .. deprecated:: 2.0.0
       Use ``date_format`` instead, or read in as ``object`` and then apply
       :func:`to_datetime` as-needed.
date_format : str or dict of column -> format, default ``None``
   If used in conjunction with ``parse_dates``, will parse dates according to this
   format. For anything more complex,
   please read in as ``object`` and then apply :func:`to_datetime` as-needed.

   .. versionadded:: 2.0.0
thousands : str, default None
    Thousands separator for parsing string columns to numeric.  Note that
    this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.
decimal : str, default '.'
    Character to recognize as decimal point for parsing string columns to numeric.
    Note that this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.(e.g. use ',' for European data).

    .. versionadded:: 1.4.0

comment : str, default None
    Comments out remainder of line. Pass a character or characters to this
    argument to indicate comments in the input file. Any data between the
    comment string and the end of the current line is ignored.
skipfooter : int, default 0
    Rows at the end to skip (0-indexed).
{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). Behaviour is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
      (default).
    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
      DataFrame.

    .. versionadded:: 2.0

engine_kwargs : dict, optional
    Arbitrary keyword arguments passed to excel engine.

Returns
-------
DataFrame or dict of DataFrames
    DataFrame from the passed in Excel file. See notes in sheet_name
    argument for more information on when a dict of DataFrames is returned.

See Also
--------
DataFrame.to_excel : Write DataFrame to an Excel file.
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
read_csv : Read a comma-separated values (csv) file into DataFrame.
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Notes
-----
For specific information on the methods used for each Excel engine, refer to the pandas
:ref:`user guide <io.excel_reader>`

Examples
--------
The file can be read using the file name as string or an open file object:

>>> pd.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP
       Name  Value
0   string1      1
1   string2      2
2  #Comment      3

>>> pd.read_excel(open('tmp.xlsx', 'rb'),
...               sheet_name='Sheet3')  # doctest: +SKIP
   Unnamed: 0      Name  Value
0           0   string1      1
1           1   string2      2
2           2  #Comment      3

Index and header can be specified via the `index_col` and `header` arguments

>>> pd.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP
     0         1      2
0  NaN      Name  Value
1  0.0   string1      1
2  1.0   string2      2
3  2.0  #Comment      3

Column types are inferred but can be explicitly specified

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               dtype={{'Name': str, 'Value': float}})  # doctest: +SKIP
       Name  Value
0   string1    1.0
1   string2    2.0
2  #Comment    3.0

True, False, and NA values, and thousands separators have defaults,
but can be explicitly specified, too. Supply the values you would like
as strings or lists of strings!

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               na_values=['string1', 'string2'])  # doctest: +SKIP
       Name  Value
0       NaN      1
1       NaN      2
2  #Comment      3

Comment lines in the excel input file can be skipped using the
``comment`` kwarg.

>>> pd.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP
      Name  Value
0  string1    1.0
1  string2    2.0
2     None    NaN
"""
)


@overload
def read_excel(
    io,
    # sheet name is str or int -> DataFrame
    sheet_name: str | int = ...,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_excel(
    io,
    # sheet name is list or None -> dict[IntStrT, DataFrame]
    sheet_name: list[IntStrT] | None,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> dict[IntStrT, DataFrame]:
    ...


@doc(storage_options=_shared_docs["storage_options"])
@Appender(_read_excel_doc)
def read_excel(
    io,
    sheet_name: str | int | list[IntStrT] | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: SequenceNotStr[Hashable] | range | None = None,
    index_col: int | str | Sequence[int] | None = None,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = None,
    dtype: DtypeArg | None = None,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,
    converters: dict[str, Callable] | dict[int, Callable] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
    nrows: int | None = None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: list | dict | bool = False,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine_kwargs: dict | None = None,
) -> DataFrame | dict[IntStrT, DataFrame]:
    check_dtype_backend(dtype_backend)
    should_close = False
    if engine_kwargs is None:
        engine_kwargs = {}

    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(
            io,
            storage_options=storage_options,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    elif engine and engine != io.engine:
        raise ValueError(
            "Engine should not be specified when passing "
            "an ExcelFile - ExcelFile already has the engine set"
        )

    try:
        data = io.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            date_format=date_format,
            thousands=thousands,
            decimal=decimal,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
        )
    finally:
        # make sure to close opened file handles
        if should_close:
            io.close()
    return data


_WorkbookT = TypeVar("_WorkbookT")


class BaseExcelReader(Generic[_WorkbookT]):
    book: _WorkbookT

    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}

        # First argument can also be bytes, so create a buffer
        if isinstance(filepath_or_buffer, bytes):
            filepath_or_buffer = BytesIO(filepath_or_buffer)

        self.handles = IOHandles(
            handle=filepath_or_buffer, compression={"method": None}
        )
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )

        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        elif hasattr(self.handles.handle, "read"):
            # N.B. xlrd.Book has a read attribute too
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError(
                "Must explicitly set engine if not passing in buffer or path for io."
            )

    @property
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                # pyxlsb: opens a TemporaryFile
                # openpyxl: https://stackoverflow.com/questions/31416842/
                #     openpyxl-does-not-close-excel-workbook-in-read-only-mode
                self.book.close()
            elif hasattr(self.book, "release_resources"):
                # xlrd
                # https://github.com/python-excel/xlrd/blob/2.0.1/xlrd/book.py#L548
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str):
        raise NotImplementedError

    def get_sheet_by_index(self, index: int):
        raise NotImplementedError

    def get_sheet_data(self, sheet, rows: int | None = None):
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(
        self,
        skiprows: Callable,
        rows_to_use: int,
    ) -> int:
        """
        Determine how many file rows are required to obtain `nrows` data
        rows when `skiprows` is a function.

        Parameters
        ----------
        skiprows : function
            The function passed to read_excel by the user.
        rows_to_use : int
            The number of rows that will be needed for the header and
            the data.

        Returns
        -------
        int
        """
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(
        self,
        header: int | Sequence[int] | None,
        index_col: int | Sequence[int] | None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None,
        nrows: int | None,
    ) -> int | None:
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.


        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]
        # If there is a MultiIndex header and an index then there is also
        # a row containing just the index name(s)
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def f(skiprows: Sequence, x: int) -> bool:
                return x in skiprows

            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(
                skiprows,
                header_rows + nrows,
            )
        # else unexpected skiprows type: read_excel will not optimize
        # the number of rows read from file
        return None

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        dtype: DtypeArg | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        verbose: bool = False,
        parse_dates: list | dict | bool = False,
        date_parser: Callable | lib.NoDefault = lib.no_default,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ):
        validate_header_arg(header)
        validate_integer("nrows", nrows)

        ret_dict = False

        # Keep sheetname to maintain backwards compatibility.
        sheets: list[int] | list[str]
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]

        # handle same-type duplicates.
        sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))

        output = {}

        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")

            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:  # assume an integer if not a string
                sheet = self.get_sheet_by_index(asheetname)

            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                # pyxlsb opens two TemporaryFiles
                sheet.close()
            usecols = maybe_convert_usecols(usecols)

            if not data:
                output[asheetname] = DataFrame()
                continue

            is_list_header = False
            is_len_one_list_header = False
            if is_list_like(header):
                assert isinstance(header, Sequence)
                is_list_header = True
                if len(header) == 1:
                    is_len_one_list_header = True

            if is_len_one_list_header:
                header = cast(Sequence[int], header)[0]

            # forward fill and pull out names for MultiIndex column
            header_names = None
            if header is not None and is_list_like(header):
                assert isinstance(header, Sequence)

                header_names = []
                control_row = [True] * len(data[0])

                for row in header:
                    if is_integer(skiprows):
                        assert isinstance(skiprows, int)
                        row += skiprows

                    if row > len(data) - 1:
                        raise ValueError(
                            f"header index {row} exceeds maximum index "
                            f"{len(data) - 1} of data.",
                        )

                    data[row], control_row = fill_mi_header(data[row], control_row)

                    if index_col is not None:
                        header_name, _ = pop_header_name(data[row], index_col)
                        header_names.append(header_name)

            # If there is a MultiIndex header and an index then there is also
            # a row containing just the index name(s)
            has_index_names = False
            if is_list_header and not is_len_one_list_header and index_col is not None:
                index_col_list: Sequence[int]
                if isinstance(index_col, int):
                    index_col_list = [index_col]
                else:
                    assert isinstance(index_col, Sequence)
                    index_col_list = index_col

                # We have to handle mi without names. If any of the entries in the data
                # columns are not empty, this is a regular row
                assert isinstance(header, Sequence)
                if len(header) < len(data):
                    potential_index_names = data[len(header)]
                    potential_data = [
                        x
                        for i, x in enumerate(potential_index_names)
                        if not control_row[i] and i not in index_col_list
                    ]
                    has_index_names = all(x == "" or x is None for x in potential_data)

            if is_list_like(index_col):
                # Forward fill values for MultiIndex index.
                if header is None:
                    offset = 0
                elif isinstance(header, int):
                    offset = 1 + header
                else:
                    offset = 1 + max(header)

                # GH34673: if MultiIndex names present and not defined in the header,
                # offset needs to be incremented so that forward filling starts
                # from the first MI value instead of the name
                if has_index_names:
                    offset += 1

                # Check if we have an empty dataset
                # before trying to collect data.
                if offset < len(data):
                    assert isinstance(index_col, Sequence)

                    for col in index_col:
                        last = data[offset][col]

                        for row in range(offset + 1, len(data)):
                            if data[row][col] == "" or data[row][col] is None:
                                data[row][col] = last
                            else:
                                last = data[row][col]

            # GH 12292 : error when read one empty column from excel file
            try:
                parser = TextParser(
                    data,
                    names=names,
                    header=header,
                    index_col=index_col,
                    has_index_names=has_index_names,
                    dtype=dtype,
                    true_values=true_values,
                    false_values=false_values,
                    skiprows=skiprows,
                    nrows=nrows,
                    na_values=na_values,
                    skip_blank_lines=False,  # GH 39808
                    parse_dates=parse_dates,
                    date_parser=date_parser,
                    date_format=date_format,
                    thousands=thousands,
                    decimal=decimal,
                    comment=comment,
                    skipfooter=skipfooter,
                    usecols=usecols,
                    dtype_backend=dtype_backend,
                    **kwds,
                )

                output[asheetname] = parser.read(nrows=nrows)

                if header_names:
                    output[asheetname].columns = output[asheetname].columns.set_names(
                        header_names
                    )

            except EmptyDataError:
                # No Data, return an empty DataFrame
                output[asheetname] = DataFrame()

            except Exception as err:
                err.args = (f"{err.args[0]} (sheet: {asheetname})", *err.args[1:])
                raise err

        if last_sheetname is None:
            raise ValueError("Sheet name is an empty list")

        if ret_dict:
            return output
        else:
            return output[last_sheetname]


@doc(storage_options=_shared_docs["storage_options"])
class ExcelWriter(Generic[_WorkbookT]):
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use:

    * `xlsxwriter <https://pypi.org/project/XlsxWriter/>`__ for xlsx files if xlsxwriter
      is installed otherwise `openpyxl <https://pypi.org/project/openpyxl/>`__
    * `odswriter <https://pypi.org/project/odswriter/>`__ for ods files

    See ``DataFrame.to_excel`` for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.
    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {{'w', 'a'}}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    {storage_options}

    if_sheet_exists : {{'error', 'new', 'replace', 'overlay'}}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.
        * overlay: Write contents to the existing sheet without first removing,
          but possibly over top of, the existing contents.

        .. versionadded:: 1.3.0

        .. versionchanged:: 1.4.0

           Added ``overlay`` option

    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine. These will be passed to
        the following functions of the respective engines:

        * xlsxwriter: ``xlsxwriter.Workbook(file, **engine_kwargs)``
        * openpyxl (write mode): ``openpyxl.Workbook(**engine_kwargs)``
        * openpyxl (append mode): ``openpyxl.load_workbook(file, **engine_kwargs)``
        * odswriter: ``odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)``

        .. versionadded:: 1.3.0

    Notes
    -----
    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    To write to separate sheets in a single file:

    >>> df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])  # doctest: +SKIP
    >>> df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP
    ...     df2.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP

    You can set the date format or datetime format:

    >>> from datetime import date, datetime  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     [
    ...         [date(2014, 1, 31), date(1999, 9, 24)],
    ...         [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
    ...     ],
    ...     index=["Date", "Datetime"],
    ...     columns=["X", "Y"],
    ... )  # doctest: +SKIP
    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     date_format="YYYY-MM-DD",
    ...     datetime_format="YYYY-MM-DD HH:MM:SS"
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    You can also append to an existing Excel file:

    >>> with pd.ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
    ...     df.to_excel(writer, sheet_name="Sheet3")  # doctest: +SKIP

    Here, the `if_sheet_exists` parameter can be set to replace a sheet if it
    already exists:

    >>> with ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="replace",
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP

    You can also write multiple DataFrames to a single sheet. Note that the
    ``if_sheet_exists`` parameter needs to be set to ``overlay``:

    >>> with ExcelWriter("path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="overlay",
    ... ) as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")
    ...     df2.to_excel(writer, sheet_name="Sheet1", startcol=3)  # doctest: +SKIP

    You can store Excel file in RAM:

    >>> import io
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> buffer = io.BytesIO()
    >>> with pd.ExcelWriter(buffer) as writer:
    ...     df.to_excel(writer)

    You can pack Excel file into zip archive:

    >>> import zipfile  # doctest: +SKIP
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with zipfile.ZipFile("path_to_file.zip", "w") as zf:
    ...     with zf.open("filename.xlsx", "w") as buffer:
    ...         with pd.ExcelWriter(buffer) as writer:
    ...             df.to_excel(writer)  # doctest: +SKIP

    You can specify additional arguments to the underlying engine:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="xlsxwriter",
    ...     engine_kwargs={{"options": {{"nan_inf_to_errors": True}}}}
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    In append mode, ``engine_kwargs`` are passed through to
    openpyxl's ``load_workbook``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="openpyxl",
    ...     mode="a",
    ...     engine_kwargs={{"keep_vba": True}}
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP
    """

    # Defining an ExcelWriter implementation (see abstract methods for more...)

    # - Mandatory
    #   - ``write_cells(self, cells, sheet_name=None, startrow=0, startcol=0)``
    #     --> called to write additional DataFrames to disk
    #   - ``_supported_extensions`` (tuple of supported extensions), used to
    #      check that engine supports the given extension.
    #   - ``_engine`` - string that gives the engine name. Necessary to
    #     instantiate class directly and bypass ``ExcelWriterMeta`` engine
    #     lookup.
    #   - ``save(self)`` --> called to save file to disk
    # - Mostly mandatory (i.e. should at least exist)
    #   - book, cur_sheet, path

    # - Optional:
    #   - ``__init__(self, path, engine=None, **kwargs)`` --> always called
    #     with path as first argument.

    # You also need to register the class with ``register_writer()``.
    # Technically, ExcelWriter implementations don't need to subclass
    # ExcelWriter.

    _engine: str
    _supported_extensions: tuple[str, ...]

    def __new__(
        cls,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict | None = None,
    ) -> Self:
        # only switch class if generic(ExcelWriter)
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == "auto"):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = "xlsx"

                try:
                    engine = config.get_option(f"io.excel.{ext}.writer", silent=True)
                    if engine == "auto":
                        engine = get_default_engine(ext, mode="writer")
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err

            # for mypy
            assert engine is not None
            #  error: Incompatible types in assignment (expression has type
            #  "type[ExcelWriter[Any]]", variable has type "type[Self]")
            cls = get_writer(engine)  # type: ignore[assignment]

        return object.__new__(cls)

    # declare external properties you can count on
    _path = None

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Extensions that writer engine supports."""
        return self._supported_extensions

    @property
    def engine(self) -> str:
        """Name of engine."""
        return self._engine

    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        raise NotImplementedError

    @property
    def book(self) -> _WorkbookT:
        """
        Book instance. Class type will depend on the engine used.

        This attribute can be used to access engine-specific features.
        """
        raise NotImplementedError

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        """
        raise NotImplementedError

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        raise NotImplementedError

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
    ) -> None:
        # validate that this engine can handle the extension
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)

        # use mode to open the file
        if "b" not in mode:
            mode += "b"
        # use "a" for the user to append data to excel but internally use "r+" to let
        # the excel backend first read the existing file and then write any data to it
        mode = mode.replace("a", "r+")

        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(
                f"'{if_sheet_exists}' is not valid for if_sheet_exists. "
                "Valid options are 'error', 'new', 'replace' and 'overlay'."
            )
        if if_sheet_exists and "r+" not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        self._if_sheet_exists = if_sheet_exists

        # cast ExcelWriter to avoid adding 'if self._handles is not None'
        self._handles = IOHandles(
            cast(IO[bytes], path), compression={"compression": None}
        )
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(
                path, mode, storage_options=storage_options, is_text=False
            )
        self._cur_sheet = None

        if date_format is None:
            self._date_format = "YYYY-MM-DD"
        else:
            self._date_format = date_format
        if datetime_format is None:
            self._datetime_format = "YYYY-MM-DD HH:MM:SS"
        else:
            self._datetime_format = datetime_format

        self._mode = mode

    @property
    def date_format(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._date_format

    @property
    def datetime_format(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> str:
        """
        How to behave when writing to a sheet that already exists in append mode.
        """
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        return getattr(self._handles.handle, "name", "")

    def _get_sheet_name(self, sheet_name: str | None) -> str:
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:  # pragma: no cover
            raise ValueError("Must pass explicit sheet_name or set _cur_sheet property")
        return sheet_name

    def _value_with_fmt(
        self, val
    ) -> tuple[
        int | float | bool | str | datetime.datetime | datetime.date, str | None
    ]:
        """
        Convert numpy types to Python types for the Excel writers.

        Parameters
        ----------
        val : object
            Value to be written into cells

        Returns
        -------
        Tuple with the first element being the converted value and the second
            being an optional format
        """
        fmt = None

        if is_integer(val):
            val = int(val)
        elif is_float(val):
            val = float(val)
        elif is_bool(val):
            val = bool(val)
        elif isinstance(val, datetime.datetime):
            fmt = self._datetime_format
        elif isinstance(val, datetime.date):
            fmt = self._date_format
        elif isinstance(val, datetime.timedelta):
            val = val.total_seconds() / 86400
            fmt = "0"
        else:
            val = str(val)

        return val, fmt

    @classmethod
    def check_extension(cls, ext: str) -> Literal[True]:
        """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
        if ext.startswith("."):
            ext = ext[1:]
        if not any(ext in extension for extension in cls._supported_extensions):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        return True

    # Allow use as a contextmanager
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """synonym for save, to make it more file-like"""
        self._save()
        self._handles.close()


XLS_SIGNATURES = (
    b"\x09\x00\x04\x00\x07\x00\x10\x00",  # BIFF2
    b"\x09\x02\x06\x00\x00\x00\x10\x00",  # BIFF3
    b"\x09\x04\x06\x00\x00\x00\x10\x00",  # BIFF4
    b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1",  # Compound File Binary
)
ZIP_SIGNATURE = b"PK\x03\x04"
PEEK_SIZE = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))


@doc(storage_options=_shared_docs["storage_options"])
def inspect_excel_format(
    content_or_path: FilePath | ReadBuffer[bytes],
    storage_options: StorageOptions | None = None,
) -> str | None:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    if isinstance(content_or_path, bytes):
        content_or_path = BytesIO(content_or_path)

    with get_handle(
        content_or_path, "rb", storage_options=storage_options, is_text=False
    ) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError("stream is empty")
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)

        if any(peek.startswith(sig) for sig in XLS_SIGNATURES):
            return "xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None

        with zipfile.ZipFile(stream) as zf:
            # Workaround for some third party files that use forward slashes and
            # lower case names.
            component_names = [
                name.replace("\\", "/").lower() for name in zf.namelist()
            ]

        if "xl/workbook.xml" in component_names:
            return "xlsx"
        if "xl/workbook.bin" in component_names:
            return "xlsb"
        if "content.xml" in component_names:
            return "ods"
        return "zip"


class ExcelFile:
    """
    Class for parsing tabular Excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, bytes, path object (pathlib.Path or py._path.local.LocalPath),
        A file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``, ``calamine``
        Engine compatibility :

        - ``xlrd`` supports old-style Excel files (.xls).
        - ``openpyxl`` supports newer Excel file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
          and OpenDocument (.ods) file formats.

        .. versionchanged:: 1.2.0

           The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
           now only supports old-style ``.xls`` files.
           When ``engine=None``, the following logic will be
           used to determine the engine:

           - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
             then `odf <https://pypi.org/project/odfpy/>`_ will be used.
           - Otherwise if ``path_or_buffer`` is an xls format,
             ``xlrd`` will be used.
           - Otherwise if ``path_or_buffer`` is in xlsb format,
             `pyxlsb <https://pypi.org/project/pyxlsb/>`_ will be used.

           .. versionadded:: 1.3.0

           - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,
             then ``openpyxl`` will be used.
           - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.

           .. warning::

            Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.
            This is not supported, switch to using ``openpyxl`` instead.
    engine_kwargs : dict, optional
        Arbitrary keyword arguments passed to excel engine.

    Examples
    --------
    >>> file = pd.ExcelFile('myfile.xlsx')  # doctest: +SKIP
    >>> with pd.ExcelFile("myfile.xls") as xls:  # doctest: +SKIP
    ...     df1 = pd.read_excel(xls, "Sheet1")  # doctest: +SKIP
    """

    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader

    _engines: Mapping[str, Any] = {
        "xlrd": XlrdReader,
        "openpyxl": OpenpyxlReader,
        "odf": ODFReader,
        "pyxlsb": PyxlsbReader,
        "calamine": CalamineReader,
    }

    def __init__(
        self,
        path_or_buffer,
        engine: str | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}

        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")

        # First argument can also be bytes, so create a buffer
        if isinstance(path_or_buffer, bytes):
            path_or_buffer = BytesIO(path_or_buffer)
            warnings.warn(
                "Passing bytes to 'read_excel' is deprecated and "
                "will be removed in a future version. To read from a "
                "byte string, wrap it in a `BytesIO` object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        # Could be a str, ExcelFile, Book, etc.
        self.io = path_or_buffer
        # Always a string
        self._io = stringify_path(path_or_buffer)

        # Determine xlrd version if installed
        if import_optional_dependency("xlrd", errors="ignore") is None:
            xlrd_version = None
        else:
            import xlrd

            xlrd_version = Version(get_version(xlrd))

        if engine is None:
            # Only determine ext if it is needed
            ext: str | None
            if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                ext = "xls"
            else:
                ext = inspect_excel_format(
                    content_or_path=path_or_buffer, storage_options=storage_options
                )
                if ext is None:
                    raise ValueError(
                        "Excel file format cannot be determined, you must specify "
                        "an engine manually."
                    )

            engine = config.get_option(f"io.excel.{ext}.reader", silent=True)
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")

        assert engine is not None
        self.engine = engine
        self.storage_options = storage_options

        self._reader = self._engines[engine](
            self._io,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    def __fspath__(self):
        return self._io

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        converters=None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        parse_dates: list | dict | bool = False,
        date_parser: Callable | lib.NoDefault = lib.no_default,
        date_format: str | dict[Hashable, str] | None = None,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ) -> DataFrame | dict[str, DataFrame] | dict[int, DataFrame]:
        """
        Parse specified sheet(s) into a DataFrame.

        Equivalent to read_excel(ExcelFile, ...)  See the read_excel
        docstring for more info on accepted parameters.

        Returns
        -------
        DataFrame or dict of DataFrames
            DataFrame from the passed in Excel file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
        >>> df.to_excel('myfile.xlsx')  # doctest: +SKIP
        >>> file = pd.ExcelFile('myfile.xlsx')  # doctest: +SKIP
        >>> file.parse()  # doctest: +SKIP
        """
        return self._reader.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            date_parser=date_parser,
            date_format=date_format,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
            **kwds,
        )

    @property
    def book(self):
        return self._reader.book

    @property
    def sheet_names(self):
        return self._reader.sheet_names

    def close(self) -> None:
        """close io if necessary"""
        self._reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._typing import (
    FilePath,
    ReadBuffer,
    Scalar,
    StorageOptions,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

import pandas as pd
from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from odf.opendocument import OpenDocument

    from pandas._libs.tslibs.nattype import NaTType


@doc(storage_options=_shared_docs["storage_options"])
class ODFReader(BaseExcelReader["OpenDocument"]):
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Read tables out of OpenDocument formatted files.

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("odf")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[OpenDocument]:
        from odf.opendocument import OpenDocument

        return OpenDocument

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> OpenDocument:
        from odf.opendocument import load

        return load(filepath_or_buffer, **engine_kwargs)

    @property
    def empty_value(self) -> str:
        """Property for compat with other readers."""
        return ""

    @property
    def sheet_names(self) -> list[str]:
        """Return a list of sheet names present in the document"""
        from odf.table import Table

        tables = self.book.getElementsByType(Table)
        return [t.getAttribute("name") for t in tables]

    def get_sheet_by_index(self, index: int):
        from odf.table import Table

        self.raise_if_bad_sheet_by_index(index)
        tables = self.book.getElementsByType(Table)
        return tables[index]

    def get_sheet_by_name(self, name: str):
        from odf.table import Table

        self.raise_if_bad_sheet_by_name(name)
        tables = self.book.getElementsByType(Table)

        for table in tables:
            if table.getAttribute("name") == name:
                return table

        self.close()
        raise ValueError(f"sheet {name} not found")

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar | NaTType]]:
        """
        Parse an ODF Table into a list of lists
        """
        from odf.table import (
            CoveredTableCell,
            TableCell,
            TableRow,
        )

        covered_cell_name = CoveredTableCell().qname
        table_cell_name = TableCell().qname
        cell_names = {covered_cell_name, table_cell_name}

        sheet_rows = sheet.getElementsByType(TableRow)
        empty_rows = 0
        max_row_len = 0

        table: list[list[Scalar | NaTType]] = []

        for sheet_row in sheet_rows:
            sheet_cells = [
                x
                for x in sheet_row.childNodes
                if hasattr(x, "qname") and x.qname in cell_names
            ]
            empty_cells = 0
            table_row: list[Scalar | NaTType] = []

            for sheet_cell in sheet_cells:
                if sheet_cell.qname == table_cell_name:
                    value = self._get_cell_value(sheet_cell)
                else:
                    value = self.empty_value

                column_repeat = self._get_column_repeat(sheet_cell)

                # Queue up empty values, writing only if content succeeds them
                if value == self.empty_value:
                    empty_cells += column_repeat
                else:
                    table_row.extend([self.empty_value] * empty_cells)
                    empty_cells = 0
                    table_row.extend([value] * column_repeat)

            if max_row_len < len(table_row):
                max_row_len = len(table_row)

            row_repeat = self._get_row_repeat(sheet_row)
            if len(table_row) == 0:
                empty_rows += row_repeat
            else:
                # add blank rows to our table
                table.extend([[self.empty_value]] * empty_rows)
                empty_rows = 0
                table.extend(table_row for _ in range(row_repeat))
            if file_rows_needed is not None and len(table) >= file_rows_needed:
                break

        # Make our table square
        for row in table:
            if len(row) < max_row_len:
                row.extend([self.empty_value] * (max_row_len - len(row)))

        return table

    def _get_row_repeat(self, row) -> int:
        """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
        from odf.namespaces import TABLENS

        return int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))

    def _get_column_repeat(self, cell) -> int:
        from odf.namespaces import TABLENS

        return int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))

    def _get_cell_value(self, cell) -> Scalar | NaTType:
        from odf.namespaces import OFFICENS

        if str(cell) == "#N/A":
            return np.nan

        cell_type = cell.attributes.get((OFFICENS, "value-type"))
        if cell_type == "boolean":
            if str(cell) == "TRUE":
                return True
            return False
        if cell_type is None:
            return self.empty_value
        elif cell_type == "float":
            # GH5394
            cell_value = float(cell.attributes.get((OFFICENS, "value")))
            val = int(cell_value)
            if val == cell_value:
                return val
            return cell_value
        elif cell_type == "percentage":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "string":
            return self._get_cell_string_value(cell)
        elif cell_type == "currency":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "date":
            cell_value = cell.attributes.get((OFFICENS, "date-value"))
            return pd.Timestamp(cell_value)
        elif cell_type == "time":
            stamp = pd.Timestamp(str(cell))
            # cast needed here because Scalar doesn't include datetime.time
            return cast(Scalar, stamp.time())
        else:
            self.close()
            raise ValueError(f"Unrecognized type {cell_type}")

    def _get_cell_string_value(self, cell) -> str:
        """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.office import Annotation
        from odf.text import S

        office_annotation = Annotation().qname
        text_s = S().qname

        value = []

        for fragment in cell.childNodes:
            if isinstance(fragment, Element):
                if fragment.qname == text_s:
                    spaces = int(fragment.attributes.get((TEXTNS, "c"), 1))
                    value.append(" " * spaces)
                elif fragment.qname == office_annotation:
                    continue
                else:
                    # recursive impl needed in case of nested fragments
                    # with multiple spaces
                    # https://github.com/pandas-dev/pandas/pull/36175#discussion_r484639704
                    value.append(self._get_cell_string_value(fragment))
            else:
                value.append(str(fragment).strip("\n"))
        return "".join(value)
<|endoftext|>from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
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


class _XlsxStyler:
    # Map from openpyxl-oriented styles to flatter xlsxwriter representation
    # Ordering necessary for both determinism and because some are keyed by
    # prefixes of others.
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]] = {
        "font": [
            (("name",), "font_name"),
            (("sz",), "font_size"),
            (("size",), "font_size"),
            (("color", "rgb"), "font_color"),
            (("color",), "font_color"),
            (("b",), "bold"),
            (("bold",), "bold"),
            (("i",), "italic"),
            (("italic",), "italic"),
            (("u",), "underline"),
            (("underline",), "underline"),
            (("strike",), "font_strikeout"),
            (("vertAlign",), "font_script"),
            (("vertalign",), "font_script"),
        ],
        "number_format": [(("format_code",), "num_format"), ((), "num_format")],
        "protection": [(("locked",), "locked"), (("hidden",), "hidden")],
        "alignment": [
            (("horizontal",), "align"),
            (("vertical",), "valign"),
            (("text_rotation",), "rotation"),
            (("wrap_text",), "text_wrap"),
            (("indent",), "indent"),
            (("shrink_to_fit",), "shrink"),
        ],
        "fill": [
            (("patternType",), "pattern"),
            (("patterntype",), "pattern"),
            (("fill_type",), "pattern"),
            (("start_color", "rgb"), "fg_color"),
            (("fgColor", "rgb"), "fg_color"),
            (("fgcolor", "rgb"), "fg_color"),
            (("start_color",), "fg_color"),
            (("fgColor",), "fg_color"),
            (("fgcolor",), "fg_color"),
            (("end_color", "rgb"), "bg_color"),
            (("bgColor", "rgb"), "bg_color"),
            (("bgcolor", "rgb"), "bg_color"),
            (("end_color",), "bg_color"),
            (("bgColor",), "bg_color"),
            (("bgcolor",), "bg_color"),
        ],
        "border": [
            (("color", "rgb"), "border_color"),
            (("color",), "border_color"),
            (("style",), "border"),
            (("top", "color", "rgb"), "top_color"),
            (("top", "color"), "top_color"),
            (("top", "style"), "top"),
            (("top",), "top"),
            (("right", "color", "rgb"), "right_color"),
            (("right", "color"), "right_color"),
            (("right", "style"), "right"),
            (("right",), "right"),
            (("bottom", "color", "rgb"), "bottom_color"),
            (("bottom", "color"), "bottom_color"),
            (("bottom", "style"), "bottom"),
            (("bottom",), "bottom"),
            (("left", "color", "rgb"), "left_color"),
            (("left", "color"), "left_color"),
            (("left", "style"), "left"),
            (("left",), "left"),
        ],
    }

    @classmethod
    def convert(cls, style_dict, num_format_str=None):
        """
        converts a style_dict to an xlsxwriter format dict

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """
        # Create a XlsxWriter format object.
        props = {}

        if num_format_str is not None:
            props["num_format"] = num_format_str

        if style_dict is None:
            return props

        if "borders" in style_dict:
            style_dict = style_dict.copy()
            style_dict["border"] = style_dict.pop("borders")

        for style_group_key, style_group in style_dict.items():
            for src, dst in cls.STYLE_MAPPING.get(style_group_key, []):
                # src is a sequence of keys into a nested dict
                # dst is a flat key
                if dst in props:
                    continue
                v = style_group
                for k in src:
                    try:
                        v = v[k]
                    except (KeyError, TypeError):
                        break
                else:
                    props[dst] = v

        if isinstance(props.get("pattern"), str):
            # TODO: support other fill patterns
            props["pattern"] = 0 if props["pattern"] == "none" else 1

        for k in ["border", "top", "right", "bottom", "left"]:
            if isinstance(props.get(k), str):
                try:
                    props[k] = [
                        "none",
                        "thin",
                        "medium",
                        "dashed",
                        "dotted",
                        "thick",
                        "double",
                        "hair",
                        "mediumDashed",
                        "dashDot",
                        "mediumDashDot",
                        "dashDotDot",
                        "mediumDashDotDot",
                        "slantDashDot",
                    ].index(props[k])
                except ValueError:
                    props[k] = 2

        if isinstance(props.get("font_script"), str):
            props["font_script"] = ["baseline", "superscript", "subscript"].index(
                props["font_script"]
            )

        if isinstance(props.get("underline"), str):
            props["underline"] = {
                "none": 0,
                "single": 1,
                "double": 2,
                "singleAccounting": 33,
                "doubleAccounting": 34,
            }[props["underline"]]

        # GH 30107 - xlsxwriter uses different name
        if props.get("valign") == "center":
            props["valign"] = "vcenter"

        return props


class XlsxWriter(ExcelWriter):
    _engine = "xlsxwriter"
    _supported_extensions = (".xlsx",)

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
        # Use the xlsxwriter module as the Excel writer.
        from xlsxwriter import Workbook

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        if mode == "a":
            raise ValueError("Append mode is not supported with xlsxwriter!")

        super().__init__(
            path,
            engine=engine,
            date_format=date_format,
            datetime_format=datetime_format,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        try:
            self._book = Workbook(self._handles.handle, **engine_kwargs)  # type: ignore[arg-type]
        except TypeError:
            self._handles.handle.close()
            raise

    @property
    def book(self):
        """
        Book instance of class xlsxwriter.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        result = self.book.sheetnames
        return result

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        self.book.close()

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        # Write the frame cells using xlsxwriter.
        sheet_name = self._get_sheet_name(sheet_name)

        wks = self.book.get_worksheet_by_name(sheet_name)
        if wks is None:
            wks = self.book.add_worksheet(sheet_name)

        style_dict = {"null": None}

        if validate_freeze_panes(freeze_panes):
            wks.freeze_panes(*(freeze_panes))

        for cell in cells:
            val, fmt = self._value_with_fmt(cell.val)

            stylekey = json.dumps(cell.style)
            if fmt:
                stylekey += fmt

            if stylekey in style_dict:
                style = style_dict[stylekey]
            else:
                style = self.book.add_format(_XlsxStyler.convert(cell.style, fmt))
                style_dict[stylekey] = style

            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_range(
                    startrow + cell.row,
                    startcol + cell.col,
                    startrow + cell.mergestart,
                    startcol + cell.mergeend,
                    val,
                    style,
                )
            else:
                wks.write(startrow + cell.row, startcol + cell.col, val, style)
<|endoftext|>"""
Table Schema builders

https://specs.frictionlessdata.io/table-schema/
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

from pandas import DataFrame
import pandas.core.common as com

from pandas.tseries.frequencies import to_offset

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        JSONSerializable,
    )

    from pandas import Series
    from pandas.core.indexes.multi import MultiIndex


TABLE_SCHEMA_VERSION = "1.4.0"


def as_json_table_type(x: DtypeObj) -> str:
    """
    Convert a NumPy / pandas type to its corresponding json_table.

    Parameters
    ----------
    x : np.dtype or ExtensionDtype

    Returns
    -------
    str
        the Table Schema data types

    Notes
    -----
    This table shows the relationship between NumPy / pandas dtypes,
    and Table Schema dtypes.

    ==============  =================
    Pandas type     Table Schema type
    ==============  =================
    int64           integer
    float64         number
    bool            boolean
    datetime64[ns]  datetime
    timedelta64[ns] duration
    object          str
    categorical     any
    =============== =================
    """
    if is_integer_dtype(x):
        return "integer"
    elif is_bool_dtype(x):
        return "boolean"
    elif is_numeric_dtype(x):
        return "number"
    elif lib.is_np_dtype(x, "M") or isinstance(x, (DatetimeTZDtype, PeriodDtype)):
        return "datetime"
    elif lib.is_np_dtype(x, "m"):
        return "duration"
    elif is_string_dtype(x):
        return "string"
    else:
        return "any"


def set_default_names(data):
    """Sets index names to 'index' for regular, or 'level_x' for Multi"""
    if com.all_not_none(*data.index.names):
        nms = data.index.names
        if len(nms) == 1 and data.index.name == "index":
            warnings.warn(
                "Index name of 'index' is not round-trippable.",
                stacklevel=find_stack_level(),
            )
        elif len(nms) > 1 and any(x.startswith("level_") for x in nms):
            warnings.warn(
                "Index names beginning with 'level_' are not round-trippable.",
                stacklevel=find_stack_level(),
            )
        return data

    data = data.copy()
    if data.index.nlevels > 1:
        data.index.names = com.fill_missing_names(data.index.names)
    else:
        data.index.name = data.index.name or "index"
    return data


def convert_pandas_type_to_json_field(arr) -> dict[str, JSONSerializable]:
    dtype = arr.dtype
    name: JSONSerializable
    if arr.name is None:
        name = "values"
    else:
        name = arr.name
    field: dict[str, JSONSerializable] = {
        "name": name,
        "type": as_json_table_type(dtype),
    }

    if isinstance(dtype, CategoricalDtype):
        cats = dtype.categories
        ordered = dtype.ordered

        field["constraints"] = {"enum": list(cats)}
        field["ordered"] = ordered
    elif isinstance(dtype, PeriodDtype):
        field["freq"] = dtype.freq.freqstr
    elif isinstance(dtype, DatetimeTZDtype):
        if timezones.is_utc(dtype.tz):
            # timezone.utc has no "zone" attr
            field["tz"] = "UTC"
        else:
            # error: "tzinfo" has no attribute "zone"
            field["tz"] = dtype.tz.zone  # type: ignore[attr-defined]
    elif isinstance(dtype, ExtensionDtype):
        field["extDtype"] = dtype.name
    return field


def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
    """
    Converts a JSON field descriptor into its corresponding NumPy / pandas type

    Parameters
    ----------
    field
        A JSON field descriptor

    Returns
    -------
    dtype

    Raises
    ------
    ValueError
        If the type of the provided field is unknown or currently unsupported

    Examples
    --------
    >>> convert_json_field_to_pandas_type({"name": "an_int", "type": "integer"})
    'int64'

    >>> convert_json_field_to_pandas_type(
    ...     {
    ...         "name": "a_categorical",
    ...         "type": "any",
    ...         "constraints": {"enum": ["a", "b", "c"]},
    ...         "ordered": True,
    ...     }
    ... )
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=True, categories_dtype=object)

    >>> convert_json_field_to_pandas_type({"name": "a_datetime", "type": "datetime"})
    'datetime64[ns]'

    >>> convert_json_field_to_pandas_type(
    ...     {"name": "a_datetime_with_tz", "type": "datetime", "tz": "US/Central"}
    ... )
    'datetime64[ns, US/Central]'
    """
    typ = field["type"]
    if typ == "string":
        return field.get("extDtype", None)
    elif typ == "integer":
        return field.get("extDtype", "int64")
    elif typ == "number":
        return field.get("extDtype", "float64")
    elif typ == "boolean":
        return field.get("extDtype", "bool")
    elif typ == "duration":
        return "timedelta64"
    elif typ == "datetime":
        if field.get("tz"):
            return f"datetime64[ns, {field['tz']}]"
        elif field.get("freq"):
            # GH#9586 rename frequency M to ME for offsets
            offset = to_offset(field["freq"])
            freq_n, freq_name = offset.n, offset.name
            freq = freq_to_period_freqstr(freq_n, freq_name)
            # GH#47747 using datetime over period to minimize the change surface
            return f"period[{freq}]"
        else:
            return "datetime64[ns]"
    elif typ == "any":
        if "constraints" in field and "ordered" in field:
            return CategoricalDtype(
                categories=field["constraints"]["enum"], ordered=field["ordered"]
            )
        elif "extDtype" in field:
            return registry.find(field["extDtype"])
        else:
            return "object"

    raise ValueError(f"Unsupported or invalid field type: {typ}")


def build_table_schema(
    data: DataFrame | Series,
    index: bool = True,
    primary_key: bool | None = None,
    version: bool = True,
) -> dict[str, JSONSerializable]:
    """
    Create a Table schema from ``data``.

    Parameters
    ----------
    data : Series, DataFrame
    index : bool, default True
        Whether to include ``data.index`` in the schema.
    primary_key : bool or None, default True
        Column names to designate as the primary key.
        The default `None` will set `'primaryKey'` to the index
        level or levels if the index is unique.
    version : bool, default True
        Whether to include a field `pandas_version` with the version
        of pandas that last revised the table schema. This version
        can be different from the installed pandas version.

    Returns
    -------
    dict

    Notes
    -----
    See `Table Schema
    <https://pandas.pydata.org/docs/user_guide/io.html#table-schema>`__ for
    conversion types.
    Timedeltas as converted to ISO8601 duration format with
    9 decimal places after the seconds field for nanosecond precision.

    Categoricals are converted to the `any` dtype, and use the `enum` field
    constraint to list the allowed values. The `ordered` attribute is included
    in an `ordered` field.

    Examples
    --------
    >>> from pandas.io.json._table_schema import build_table_schema
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='d', periods=3),
    ...     }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': \
[{'name': 'idx', 'type': 'integer'}, \
{'name': 'A', 'type': 'integer'}, \
{'name': 'B', 'type': 'string'}, \
{'name': 'C', 'type': 'datetime'}], \
'primaryKey': ['idx'], \
'pandas_version': '1.4.0'}
    """
    if index is True:
        data = set_default_names(data)

    schema: dict[str, Any] = {}
    fields = []

    if index:
        if data.index.nlevels > 1:
            data.index = cast("MultiIndex", data.index)
            for level, name in zip(data.index.levels, data.index.names):
                new_field = convert_pandas_type_to_json_field(level)
                new_field["name"] = name
                fields.append(new_field)
        else:
            fields.append(convert_pandas_type_to_json_field(data.index))

    if data.ndim > 1:
        for column, s in data.items():
            fields.append(convert_pandas_type_to_json_field(s))
    else:
        fields.append(convert_pandas_type_to_json_field(data))

    schema["fields"] = fields
    if index and data.index.is_unique and primary_key is None:
        if data.index.nlevels == 1:
            schema["primaryKey"] = [data.index.name]
        else:
            schema["primaryKey"] = data.index.names
    elif primary_key is not None:
        schema["primaryKey"] = primary_key

    if version:
        schema["pandas_version"] = TABLE_SCHEMA_VERSION
    return schema


def parse_table_schema(json, precise_float: bool) -> DataFrame:
    """
    Builds a DataFrame from a given schema

    Parameters
    ----------
    json :
        A JSON table schema
    precise_float : bool
        Flag controlling precision when decoding string to double values, as
        dictated by ``read_json``

    Returns
    -------
    df : DataFrame

    Raises
    ------
    NotImplementedError
        If the JSON table schema contains either timezone or timedelta data

    Notes
    -----
        Because :func:`DataFrame.to_json` uses the string 'index' to denote a
        name-less :class:`Index`, this function sets the name of the returned
        :class:`DataFrame` to ``None`` when said string is encountered with a
        normal :class:`Index`. For a :class:`MultiIndex`, the same limitation
        applies to any strings beginning with 'level_'. Therefore, an
        :class:`Index` name of 'index'  and :class:`MultiIndex` names starting
        with 'level_' are not supported.

    See Also
    --------
    build_table_schema : Inverse function.
    pandas.read_json
    """
    table = ujson_loads(json, precise_float=precise_float)
    col_order = [field["name"] for field in table["schema"]["fields"]]
    df = DataFrame(table["data"], columns=col_order)[col_order]

    dtypes = {
        field["name"]: convert_json_field_to_pandas_type(field)
        for field in table["schema"]["fields"]
    }

    # No ISO constructor for Timedelta as of yet, so need to raise
    if "timedelta64" in dtypes.values():
        raise NotImplementedError(
            'table="orient" can not yet read ISO-formatted Timedelta data'
        )

    df = df.astype(dtypes)

    if "primaryKey" in table["schema"]:
        df = df.set_index(table["schema"]["primaryKey"])
        if len(df.index.names) == 1:
            if df.index.name == "index":
                df.index.name = None
        else:
            df.index.names = [
                None if x.startswith("level_") else x for x in df.index.names
            ]

    return df
<|endoftext|>from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import abc
from io import StringIO
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.json import (
    ujson_dumps,
    ujson_loads,
)
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    ensure_str,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
    to_datetime,
)
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import (
    IOHandles,
    dedup_names,
    extension_to_compression,
    file_exists,
    get_handle,
    is_fsspec_url,
    is_potential_multi_index,
    is_url,
    stringify_path,
)
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import (
    build_table_schema,
    parse_table_schema,
)
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        JSONEngine,
        JSONSerializable,
        ReadBuffer,
        Self,
        StorageOptions,
        WriteBuffer,
    )

    from pandas.core.generic import NDFrame

FrameSeriesStrT = TypeVar("FrameSeriesStrT", bound=Literal["frame", "series"])


# interface to/from
@overload
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> None:
    ...


@overload
def to_json(
    path_or_buf: None,
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> str:
    ...


def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] | None,
    obj: NDFrame,
    orient: str | None = None,
    date_format: str = "epoch",
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = "ms",
    default_handler: Callable[[Any], JSONSerializable] | None = None,
    lines: bool = False,
    compression: CompressionOptions = "infer",
    index: bool | None = None,
    indent: int = 0,
    storage_options: StorageOptions | None = None,
    mode: Literal["a", "w"] = "w",
) -> str | None:
    if orient in ["records", "values"] and index is True:
        raise ValueError(
            "'index=True' is only valid when 'orient' is 'split', 'table', "
            "'index', or 'columns'."
        )
    elif orient in ["index", "columns"] and index is False:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split', 'table', "
            "'records', or 'values'."
        )
    elif index is None:
        # will be ignored for orient='records' and 'values'
        index = True

    if lines and orient != "records":
        raise ValueError("'lines' keyword only valid when 'orient' is records")

    if mode not in ["a", "w"]:
        msg = (
            f"mode={mode} is not a valid option."
            "Only 'w' and 'a' are currently supported."
        )
        raise ValueError(msg)

    if mode == "a" and (not lines or orient != "records"):
        msg = (
            "mode='a' (append) is only supported when "
            "lines is True and orient is 'records'"
        )
        raise ValueError(msg)

    if orient == "table" and isinstance(obj, Series):
        obj = obj.to_frame(name=obj.name or "values")

    writer: type[Writer]
    if orient == "table" and isinstance(obj, DataFrame):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")

    s = writer(
        obj,
        orient=orient,
        date_format=date_format,
        double_precision=double_precision,
        ensure_ascii=force_ascii,
        date_unit=date_unit,
        default_handler=default_handler,
        index=index,
        indent=indent,
    ).write()

    if lines:
        s = convert_to_line_delimits(s)

    if path_or_buf is not None:
        # apply compression and byte/text conversion
        with get_handle(
            path_or_buf, mode, compression=compression, storage_options=storage_options
        ) as handles:
            handles.handle.write(s)
    else:
        return s
    return None


class Writer(ABC):
    _default_orient: str

    def __init__(
        self,
        obj: NDFrame,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        self.obj = obj

        if orient is None:
            orient = self._default_orient

        self.orient = orient
        self.date_format = date_format
        self.double_precision = double_precision
        self.ensure_ascii = ensure_ascii
        self.date_unit = date_unit
        self.default_handler = default_handler
        self.index = index
        self.indent = indent

        self.is_copy = None
        self._format_axes()

    def _format_axes(self) -> None:
        raise AbstractMethodError(self)

    def write(self) -> str:
        iso_dates = self.date_format == "iso"
        return ujson_dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent,
        )

    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        """Object to write in JSON format."""


class SeriesWriter(Writer):
    _default_orient = "index"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        if not self.index and self.orient == "split":
            return {"name": self.obj.name, "data": self.obj.values}
        else:
            return self.obj

    def _format_axes(self) -> None:
        if not self.obj.index.is_unique and self.orient == "index":
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")


class FrameWriter(Writer):
    _default_orient = "columns"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        if not self.index and self.orient == "split":
            obj_to_write = self.obj.to_dict(orient="split")
            del obj_to_write["index"]
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        """
        Try to format axes if they are datelike.
        """
        if not self.obj.index.is_unique and self.orient in ("index", "columns"):
            raise ValueError(
                f"DataFrame index must be unique for orient='{self.orient}'."
            )
        if not self.obj.columns.is_unique and self.orient in (
            "index",
            "columns",
            "records",
        ):
            raise ValueError(
                f"DataFrame columns must be unique for orient='{self.orient}'."
            )


class JSONTableWriter(FrameWriter):
    _default_orient = "records"

    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super().__init__(
            obj,
            orient,
            date_format,
            double_precision,
            ensure_ascii,
            date_unit,
            index,
            default_handler=default_handler,
            indent=indent,
        )

        if date_format != "iso":
            msg = (
                "Trying to write with `orient='table'` and "
                f"`date_format='{date_format}'`. Table Schema requires dates "
                "to be formatted with `date_format='iso'`"
            )
            raise ValueError(msg)

        self.schema = build_table_schema(obj, index=self.index)

        # NotImplemented on a column MultiIndex
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError(
                "orient='table' is not supported for MultiIndex columns"
            )

        # TODO: Do this timedelta properly in objToJSON.c See GH #15137
        if (
            (obj.ndim == 1)
            and (obj.name in set(obj.index.names))
            or len(obj.columns.intersection(obj.index.names))
        ):
            msg = "Overlapping names between the index and columns"
            raise ValueError(msg)

        obj = obj.copy()
        timedeltas = obj.select_dtypes(include=["timedelta"]).columns
        if len(timedeltas):
            obj[timedeltas] = obj[timedeltas].map(lambda x: x.isoformat())
        # Convert PeriodIndex to datetimes before serializing
        if isinstance(obj.index.dtype, PeriodDtype):
            obj.index = obj.index.to_timestamp()

        # exclude index from obj if index=False
        if not self.index:
            self.obj = obj.reset_index(drop=True)
        else:
            self.obj = obj.reset_index(drop=False)
        self.date_format = "iso"
        self.orient = "records"
        self.index = index

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        return {"schema": self.schema, "data": self.obj}


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["frame"]]:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["series"]]:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> Series:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> DataFrame:
    ...


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buf",
)
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = None,
    typ: Literal["frame", "series"] = "frame",
    dtype: DtypeArg | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    lines: bool = False,
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine: JSONEngine = "ujson",
) -> DataFrame | Series | JsonReader:
    """
    Convert a JSON string to pandas object.

    Parameters
    ----------
    path_or_buf : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

        .. deprecated:: 2.1.0
            Passing json literal strings is deprecated.

    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like
          ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{{'schema': {{schema}}, 'data': {{data}}}}``

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{{'split','records','index',
            'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

        .. versionadded:: 1.3.0

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.
    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.
        If this is None, all the rows will be returned.

    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    engine : {{"ujson", "pyarrow"}}, default "ujson"
        Parser engine to use. The ``"pyarrow"`` engine is only available when
        ``lines=True``.

        .. versionadded:: 2.0

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        A JsonReader is returned when ``chunksize`` is not ``0`` or ``None``.
        Otherwise, the type returned depends on the value of ``typ``.

    See Also
    --------
    DataFrame.to_json : Convert a DataFrame to a JSON string.
    Series.to_json : Convert a Series to a JSON string.
    json_normalize : Normalize semi-structured JSON data into a flat table.

    Notes
    -----
    Specific to ``orient='table'``, if a :class:`DataFrame` with a literal
    :class:`Index` name of `index` gets written with :func:`to_json`, the
    subsequent read operation will incorrectly set the :class:`Index` name to
    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`
    to denote a missing :class:`Index` name, and the subsequent
    :func:`read_json` operation cannot distinguish between the two. The same
    limitation is encountered with a :class:`MultiIndex` and any names
    beginning with ``'level_'``.

    Examples
    --------
    >>> from io import StringIO
    >>> df = pd.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])

    Encoding/decoding a Dataframe using ``'split'`` formatted JSON:

    >>> df.to_json(orient='split')
        '\
{{\
"columns":["col 1","col 2"],\
"index":["row 1","row 2"],\
"data":[["a","b"],["c","d"]]\
}}\
'
    >>> pd.read_json(StringIO(_), orient='split')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

    >>> df.to_json(orient='index')
    '{{"row 1":{{"col 1":"a","col 2":"b"}},"row 2":{{"col 1":"c","col 2":"d"}}}}'

    >>> pd.read_json(StringIO(_), orient='index')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
    Note that index labels are not preserved with this encoding.

    >>> df.to_json(orient='records')
    '[{{"col 1":"a","col 2":"b"}},{{"col 1":"c","col 2":"d"}}]'
    >>> pd.read_json(StringIO(_), orient='records')
      col 1 col 2
    0     a     b
    1     c     d

    Encoding with Table Schema

    >>> df.to_json(orient='table')
        '\
{{"schema":{{"fields":[\
{{"name":"index","type":"string"}},\
{{"name":"col 1","type":"string"}},\
{{"name":"col 2","type":"string"}}],\
"primaryKey":["index"],\
"pandas_version":"1.4.0"}},\
"data":[\
{{"index":"row 1","col 1":"a","col 2":"b"}},\
{{"index":"row 2","col 1":"c","col 2":"d"}}]\
}}\
'

    The following example uses ``dtype_backend="numpy_nullable"``

    >>> data = '''{{"index": {{"0": 0, "1": 1}},
    ...        "a": {{"0": 1, "1": null}},
    ...        "b": {{"0": 2.5, "1": 4.5}},
    ...        "c": {{"0": true, "1": false}},
    ...        "d": {{"0": "a", "1": "b"}},
    ...        "e": {{"0": 1577.2, "1": 1577.1}}}}'''
    >>> pd.read_json(StringIO(data), dtype_backend="numpy_nullable")
       index     a    b      c  d       e
    0      0     1  2.5   True  a  1577.2
    1      1  <NA>  4.5  False  b  1577.1
    """
    if orient == "table" and dtype:
        raise ValueError("cannot pass both dtype and orient='table'")
    if orient == "table" and convert_axes:
        raise ValueError("cannot pass both convert_axes and orient='table'")

    check_dtype_backend(dtype_backend)

    if dtype is None and orient != "table":
        # error: Incompatible types in assignment (expression has type "bool", variable
        # has type "Union[ExtensionDtype, str, dtype[Any], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object], Dict[Hashable,
        # Union[ExtensionDtype, Union[str, dtype[Any]], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object]]], None]")
        dtype = True  # type: ignore[assignment]
    if convert_axes is None and orient != "table":
        convert_axes = True

    json_reader = JsonReader(
        path_or_buf,
        orient=orient,
        typ=typ,
        dtype=dtype,
        convert_axes=convert_axes,
        convert_dates=convert_dates,
        keep_default_dates=keep_default_dates,
        precise_float=precise_float,
        date_unit=date_unit,
        encoding=encoding,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        nrows=nrows,
        storage_options=storage_options,
        encoding_errors=encoding_errors,
        dtype_backend=dtype_backend,
        engine=engine,
    )

    if chunksize:
        return json_reader
    else:
        return json_reader.read()


class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """

    def __init__(
        self,
        filepath_or_buffer,
        orient,
        typ: FrameSeriesStrT,
        dtype,
        convert_axes: bool | None,
        convert_dates,
        keep_default_dates: bool,
        precise_float: bool,
        date_unit,
        encoding,
        lines: bool,
        chunksize: int | None,
        compression: CompressionOptions,
        nrows: int | None,
        storage_options: StorageOptions | None = None,
        encoding_errors: str | None = "strict",
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        engine: JSONEngine = "ujson",
    ) -> None:
        self.orient = orient
        self.typ = typ
        self.dtype = dtype
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.keep_default_dates = keep_default_dates
        self.precise_float = precise_float
        self.date_unit = date_unit
        self.encoding = encoding
        self.engine = engine
        self.compression = compression
        self.storage_options = storage_options
        self.lines = lines
        self.chunksize = chunksize
        self.nrows_seen = 0
        self.nrows = nrows
        self.encoding_errors = encoding_errors
        self.handles: IOHandles[str] | None = None
        self.dtype_backend = dtype_backend

        if self.engine not in {"pyarrow", "ujson"}:
            raise ValueError(
                f"The engine type {self.engine} is currently not supported."
            )
        if self.chunksize is not None:
            self.chunksize = validate_integer("chunksize", self.chunksize, 1)
            if not self.lines:
                raise ValueError("chunksize can only be passed if lines=True")
            if self.engine == "pyarrow":
                raise ValueError(
                    "currently pyarrow engine doesn't support chunksize parameter"
                )
        if self.nrows is not None:
            self.nrows = validate_integer("nrows", self.nrows, 0)
            if not self.lines:
                raise ValueError("nrows can only be passed if lines=True")
        if (
            isinstance(filepath_or_buffer, str)
            and not self.lines
            and "\n" in filepath_or_buffer
        ):
            warnings.warn(
                "Passing literal json to 'read_json' is deprecated and "
                "will be removed in a future version. To read from a "
                "literal string, wrap it in a 'StringIO' object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if self.engine == "pyarrow":
            if not self.lines:
                raise ValueError(
                    "currently pyarrow engine only supports "
                    "the line-delimited JSON format"
                )
            self.data = filepath_or_buffer
        elif self.engine == "ujson":
            data = self._get_data_from_filepath(filepath_or_buffer)
            self.data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        """
        At this point, the data either has a `read` attribute (e.g. a file
        object or a StringIO) or is a string that is a JSON document.

        If self.chunksize, we prepare the data for the `__next__` method.
        Otherwise, we read it into memory for the `read` method.
        """
        if hasattr(data, "read") and not (self.chunksize or self.nrows):
            with self:
                data = data.read()
        if not hasattr(data, "read") and (self.chunksize or self.nrows):
            data = StringIO(data)

        return data

    def _get_data_from_filepath(self, filepath_or_buffer):
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
            3. JSON string

        This method turns (1) into (2) to simplify the rest of the processing.
        It returns input types (2) and (3) unchanged.

        It raises FileNotFoundError if the input is a string ending in
        one of .json, .json.gz, .json.bz2, etc. but no such file exists.
        """
        # if it is a string but the file does not exist, it might be a JSON string
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if (
            not isinstance(filepath_or_buffer, str)
            or is_url(filepath_or_buffer)
            or is_fsspec_url(filepath_or_buffer)
            or file_exists(filepath_or_buffer)
        ):
            self.handles = get_handle(
                filepath_or_buffer,
                "r",
                encoding=self.encoding,
                compression=self.compression,
                storage_options=self.storage_options,
                errors=self.encoding_errors,
            )
            filepath_or_buffer = self.handles.handle
        elif (
            isinstance(filepath_or_buffer, str)
            and filepath_or_buffer.lower().endswith(
                (".json",) + tuple(f".json{c}" for c in extension_to_compression)
            )
            and not file_exists(filepath_or_buffer)
        ):
            raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
        else:
            warnings.warn(
                "Passing literal json to 'read_json' is deprecated and "
                "will be removed in a future version. To read from a "
                "literal string, wrap it in a 'StringIO' object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return filepath_or_buffer

    def _combine_lines(self, lines) -> str:
        """
        Combines a list of JSON objects into one JSON object.
        """
        return (
            f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
        )

    @overload
    def read(self: JsonReader[Literal["frame"]]) -> DataFrame:
        ...

    @overload
    def read(self: JsonReader[Literal["series"]]) -> Series:
        ...

    @overload
    def read(self: JsonReader[Literal["frame", "series"]]) -> DataFrame | Series:
        ...

    def read(self) -> DataFrame | Series:
        """
        Read the whole JSON input into a pandas object.
        """
        obj: DataFrame | Series
        with self:
            if self.engine == "pyarrow":
                pyarrow_json = import_optional_dependency("pyarrow.json")
                pa_table = pyarrow_json.read_json(self.data)
                return arrow_table_to_pandas(pa_table, dtype_backend=self.dtype_backend)
            elif self.engine == "ujson":
                if self.lines:
                    if self.chunksize:
                        obj = concat(self)
                    elif self.nrows:
                        lines = list(islice(self.data, self.nrows))
                        lines_json = self._combine_lines(lines)
                        obj = self._get_object_parser(lines_json)
                    else:
                        data = ensure_str(self.data)
                        data_lines = data.split("\n")
                        obj = self._get_object_parser(self._combine_lines(data_lines))
                else:
                    obj = self._get_object_parser(self.data)
                if self.dtype_backend is not lib.no_default:
                    return obj.convert_dtypes(
                        infer_objects=False, dtype_backend=self.dtype_backend
                    )
                else:
                    return obj

    def _get_object_parser(self, json) -> DataFrame | Series:
        """
        Parses a json document into a pandas object.
        """
        typ = self.typ
        dtype = self.dtype
        kwargs = {
            "orient": self.orient,
            "dtype": self.dtype,
            "convert_axes": self.convert_axes,
            "convert_dates": self.convert_dates,
            "keep_default_dates": self.keep_default_dates,
            "precise_float": self.precise_float,
            "date_unit": self.date_unit,
            "dtype_backend": self.dtype_backend,
        }
        obj = None
        if typ == "frame":
            obj = FrameParser(json, **kwargs).parse()

        if typ == "series" or obj is None:
            if not isinstance(dtype, bool):
                kwargs["dtype"] = dtype
            obj = SeriesParser(json, **kwargs).parse()

        return obj

    def close(self) -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        if self.handles is not None:
            self.handles.close()

    def __iter__(self) -> Self:
        return self

    @overload
    def __next__(self: JsonReader[Literal["frame"]]) -> DataFrame:
        ...

    @overload
    def __next__(self: JsonReader[Literal["series"]]) -> Series:
        ...

    @overload
    def __next__(self: JsonReader[Literal["frame", "series"]]) -> DataFrame | Series:
        ...

    def __next__(self) -> DataFrame | Series:
        if self.nrows and self.nrows_seen >= self.nrows:
            self.close()
            raise StopIteration

        lines = list(islice(self.data, self.chunksize))
        if not lines:
            self.close()
            raise StopIteration

        try:
            lines_json = self._combine_lines(lines)
            obj = self._get_object_parser(lines_json)

            # Make sure that the returned objects have the right index.
            obj.index = range(self.nrows_seen, self.nrows_seen + len(obj))
            self.nrows_seen += len(obj)
        except Exception as ex:
            self.close()
            raise ex

        if self.dtype_backend is not lib.no_default:
            return obj.convert_dtypes(
                infer_objects=False, dtype_backend=self.dtype_backend
            )
        else:
            return obj

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Parser:
    _split_keys: tuple[str, ...]
    _default_orient: str

    _STAMP_UNITS = ("s", "ms", "us", "ns")
    _MIN_STAMPS = {
        "s": 31536000,
        "ms": 31536000000,
        "us": 31536000000000,
        "ns": 31536000000000000,
    }
    json: str

    def __init__(
        self,
        json: str,
        orient,
        dtype: DtypeArg | None = None,
        convert_axes: bool = True,
        convert_dates: bool | list[str] = True,
        keep_default_dates: bool = False,
        precise_float: bool = False,
        date_unit=None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    ) -> None:
        self.json = json

        if orient is None:
            orient = self._default_orient

        self.orient = orient

        self.dtype = dtype

        if date_unit is not None:
            date_unit = date_unit.lower()
            if date_unit not in self._STAMP_UNITS:
                raise ValueError(f"date_unit must be one of {self._STAMP_UNITS}")
            self.min_stamp = self._MIN_STAMPS[date_unit]
        else:
            self.min_stamp = self._MIN_STAMPS["s"]

        self.precise_float = precise_float
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.date_unit = date_unit
        self.keep_default_dates = keep_default_dates
        self.obj: DataFrame | Series | None = None
        self.dtype_backend = dtype_backend

    @final
    def check_keys_split(self, decoded: dict) -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))
        if bad_keys:
            bad_keys_joined = ", ".join(bad_keys)
            raise ValueError(f"JSON data had unexpected key(s): {bad_keys_joined}")

    @final
    def parse(self):
        self._parse()

        if self.obj is None:
            return None
        if self.convert_axes:
            self._convert_axes()
        self._try_convert_types()
        return self.obj

    def _parse(self) -> None:
        raise AbstractMethodError(self)

    @final
    def _convert_axes(self) -> None:
        """
        Try to convert axes.
        """
        obj = self.obj
        assert obj is not None  # for mypy
        for axis_name in obj._AXIS_ORDERS:
            ax = obj._get_axis(axis_name)
            ser = Series(ax, dtype=ax.dtype, copy=False)
            new_ser, result = self._try_convert_data(
                name=axis_name,
                data=ser,
                use_dtypes=False,
                convert_dates=True,
                is_axis=True,
            )
            if result:
                new_axis = Index(new_ser, dtype=new_ser.dtype, copy=False)
                setattr(self.obj, axis_name, new_axis)

    def _try_convert_types(self) -> None:
        raise AbstractMethodError(self)

    @final
    def _try_convert_data(
        self,
        name: Hashable,
        data: Series,
        use_dtypes: bool = True,
        convert_dates: bool | list[str] = True,
        is_axis: bool = False,
    ) -> tuple[Series, bool]:
        """
        Try to parse a Series into a column by inferring dtype.
        """
        # don't try to coerce, unless a force conversion
        if use_dtypes:
            if not self.dtype:
                if all(notna(data)):
                    return data, False

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "Downcasting object dtype arrays",
                        category=FutureWarning,
                    )
                    filled = data.fillna(np.nan)

                return filled, True

            elif self.dtype is True:
                pass
            else:
                # dtype to force
                dtype = (
                    self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
                )
                if dtype is not None:
                    try:
                        return data.astype(dtype), True
                    except (TypeError, ValueError):
                        return data, False

        if convert_dates:
            new_data, result = self._try_convert_to_date(data)
            if result:
                return new_data, True

        converted = False
        if self.dtype_backend is not lib.no_default and not is_axis:
            # Fall through for conversion later on
            return data, True
        elif is_string_dtype(data.dtype):
            # try float
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        if data.dtype.kind == "f" and data.dtype != "float64":
            # coerce floats to 64
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        # don't coerce 0-len data
        if len(data) and data.dtype in ("float", "object"):
            # coerce ints if we can
            try:
                new_data = data.astype("int64")
                if (new_data == data).all():
                    data = new_data
                    converted = True
            except (TypeError, ValueError, OverflowError):
                pass

        if data.dtype == "int" and data.dtype != "int64":
            # coerce ints to 64
            try:
                data = data.astype("int64")
                converted = True
            except (TypeError, ValueError):
                pass

        # if we have an index, we want to preserve dtypes
        if name == "index" and len(data):
            if self.orient == "split":
                return data, False

        return data, converted

    @final
    def _try_convert_to_date(self, data: Series) -> tuple[Series, bool]:
        """
        Try to parse a ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats. Return a boolean if parsing was successful.
        """
        # no conversion on empty
        if not len(data):
            return data, False

        new_data = data

        if new_data.dtype == "string":
            new_data = new_data.astype(object)

        if new_data.dtype == "object":
            try:
                new_data = data.astype("int64")
            except OverflowError:
                return data, False
            except (TypeError, ValueError):
                pass

        # ignore numbers that are out of range
        if issubclass(new_data.dtype.type, np.number):
            in_range = (
                isna(new_data._values)
                | (new_data > self.min_stamp)
                | (new_data._values == iNaT)
            )
            if not in_range.all():
                return data, False

        date_units = (self.date_unit,) if self.date_unit else self._STAMP_UNITS
        for date_unit in date_units:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time "
                        "zones will raise an error",
                        category=FutureWarning,
                    )
                    new_data = to_datetime(new_data, errors="raise", unit=date_unit)
            except (ValueError, OverflowError, TypeError):
                continue
            return new_data, True
        return data, False


class SeriesParser(Parser):
    _default_orient = "index"
    _split_keys = ("name", "index", "data")
    obj: Series | None

    def _parse(self) -> None:
        data = ujson_loads(self.json, precise_float=self.precise_float)

        if self.orient == "split":
            decoded = {str(k): v for k, v in data.items()}
            self.check_keys_split(decoded)
            self.obj = Series(**decoded)
        else:
            self.obj = Series(data)

    def _try_convert_types(self) -> None:
        if self.obj is None:
            return
        obj, result = self._try_convert_data(
            "data", self.obj, convert_dates=self.convert_dates
        )
        if result:
            self.obj = obj


class FrameParser(Parser):
    _default_orient = "columns"
    _split_keys = ("columns", "index", "data")
    obj: DataFrame | None

    def _parse(self) -> None:
        json = self.json
        orient = self.orient

        if orient == "columns":
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )
        elif orient == "split":
            decoded = {
                str(k): v
                for k, v in ujson_loads(json, precise_float=self.precise_float).items()
            }
            self.check_keys_split(decoded)
            orig_names = [
                (tuple(col) if isinstance(col, list) else col)
                for col in decoded["columns"]
            ]
            decoded["columns"] = dedup_names(
                orig_names,
                is_potential_multi_index(orig_names, None),
            )
            self.obj = DataFrame(dtype=None, **decoded)
        elif orient == "index":
            self.obj = DataFrame.from_dict(
                ujson_loads(json, precise_float=self.precise_float),
                dtype=None,
                orient="index",
            )
        elif orient == "table":
            self.obj = parse_table_schema(json, precise_float=self.precise_float)
        else:
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )

    def _process_converter(
        self,
        f: Callable[[Hashable, Series], tuple[Series, bool]],
        filt: Callable[[Hashable], bool] | None = None,
    ) -> None:
        """
        Take a conversion function and possibly recreate the frame.
        """
        if filt is None:
            filt = lambda col: True

        obj = self.obj
        assert obj is not None  # for mypy

        needs_new_obj = False
        new_obj = {}
        for i, (col, c) in enumerate(obj.items()):
            if filt(col):
                new_data, result = f(col, c)
                if result:
                    c = new_data
                    needs_new_obj = True
            new_obj[i] = c

        if needs_new_obj:
            # possibly handle dup columns
            new_frame = DataFrame(new_obj, index=obj.index)
            new_frame.columns = obj.columns
            self.obj = new_frame

    def _try_convert_types(self) -> None:
        if self.obj is None:
            return
        if self.convert_dates:
            self._try_convert_dates()

        self._process_converter(
            lambda col, c: self._try_convert_data(col, c, convert_dates=False)
        )

    def _try_convert_dates(self) -> None:
        if self.obj is None:
            return

        # our columns to parse
        convert_dates_list_bool = self.convert_dates
        if isinstance(convert_dates_list_bool, bool):
            convert_dates_list_bool = []
        convert_dates = set(convert_dates_list_bool)

        def is_ok(col) -> bool:
            """
            Return if this col is ok to try for a date parse.
            """
            if col in convert_dates:
                return True
            if not self.keep_default_dates:
                return False
            if not isinstance(col, str):
                return False

            col_lower = col.lower()
            if (
                col_lower.endswith(("_at", "_time"))
                or col_lower == "modified"
                or col_lower == "date"
                or col_lower == "datetime"
                or col_lower.startswith("timestamp")
            ):
                return True
            return False

        self._process_converter(lambda col, c: self._try_convert_to_date(c), filt=is_ok)
<|endoftext|># ---------------------------------------------------------------------
# JSON normalization routines
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
)

import numpy as np

from pandas._libs.writers import convert_json_to_lines

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas._typing import (
        IgnoreRaise,
        Scalar,
    )


def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    # Determine we have a JSON list to turn to lines otherwise just return the
    # json object, only lists can
    if not s[0] == "[" and s[-1] == "]":
        return s
    s = s[1:-1]

    return convert_json_to_lines(s)


def nested_to_record(
    ds,
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
):
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    level: int, optional, default: 0
        The number of levels in the json string.

    max_level: int, optional, default: None
        The max depth to normalize.

    Returns
    -------
    d - dict or list of dicts, matching `ds`

    Examples
    --------
    >>> nested_to_record(
    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
            # each key gets renamed with prefix
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k

            # flatten if type is dict and
            # current dict level  < maximum level provided and
            # only dicts gets recurse-flattened
            # only at level>1 do we rename the rest of the keys
            if not isinstance(v, dict) or (
                max_level is not None and level >= max_level
            ):
                if level != 0:  # so we skip copying for top level, common case
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue

            v = new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)

    if singleton:
        return new_ds[0]
    return new_ds


def _normalise_json(
    data: Any,
    key_string: str,
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{key_string}{separator}{key}"

            if not key_string:
                new_key = new_key.removeprefix(separator)

            _normalise_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalised_json_object`
    """
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    return {**top_dict_, **nested_dict_}


def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    """
    A optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalised_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}

    """
    normalised_json_object = {}
    # expect a dictionary, as most jsons are. However, lists are perfectly valid
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalised_json_list
    return normalised_json_object


def json_normalize(
    data: dict | list[dict],
    record_path: str | list | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.

    Parameters
    ----------
    data : dict or list of dicts
        Unserialized JSON objects.
    record_path : str or list of str, default None
        Path in each object to list of records. If not passed, data will be
        assumed to be an array of records.
    meta : list of paths (str or list of str), default None
        Fields to use as metadata for each record in resulting table.
    meta_prefix : str, default None
        If True, prefix records with dotted (?) path, e.g. foo.bar.field if
        meta is ['foo', 'bar'].
    record_prefix : str, default None
        If True, prefix records with dotted (?) path, e.g. foo.bar.field if
        path to records is ['foo', 'bar'].
    errors : {'raise', 'ignore'}, default 'raise'
        Configures error handling.

        * 'ignore' : will ignore KeyError if keys listed in meta are not
          always present.
        * 'raise' : will raise KeyError if keys listed in meta are not
          always present.
    sep : str, default '.'
        Nested records will generate names separated by sep.
        e.g., for sep='.', {'foo': {'bar': 0}} -> foo.bar.
    max_level : int, default None
        Max number of levels(depth of dict) to normalize.
        if None, normalizes all levels.

    Returns
    -------
    frame : DataFrame
    Normalize semi-structured JSON data into a flat table.

    Examples
    --------
    >>> data = [
    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    ...     {"name": {"given": "Mark", "family": "Regner"}},
    ...     {"id": 2, "name": "Faye Raker"},
    ... ]
    >>> pd.json_normalize(data)
        id name.first name.last name.given name.family        name
    0  1.0     Coleen      Volk        NaN         NaN         NaN
    1  NaN        NaN       NaN       Mark      Regner         NaN
    2  2.0        NaN       NaN        NaN         NaN  Faye Raker

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=0)
        id        name                        fitness
    0  1.0   Cole Volk  {'height': 130, 'weight': 60}
    1  NaN    Mark Reg  {'height': 130, 'weight': 60}
    2  2.0  Faye Raker  {'height': 130, 'weight': 60}

    Normalizes nested data up to level 1.

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=1)
        id        name  fitness.height  fitness.weight
    0  1.0   Cole Volk             130              60
    1  NaN    Mark Reg             130              60
    2  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "state": "Florida",
    ...         "shortname": "FL",
    ...         "info": {"governor": "Rick Scott"},
    ...         "counties": [
    ...             {"name": "Dade", "population": 12345},
    ...             {"name": "Broward", "population": 40000},
    ...             {"name": "Palm Beach", "population": 60000},
    ...         ],
    ...     },
    ...     {
    ...         "state": "Ohio",
    ...         "shortname": "OH",
    ...         "info": {"governor": "John Kasich"},
    ...         "counties": [
    ...             {"name": "Summit", "population": 1234},
    ...             {"name": "Cuyahoga", "population": 1337},
    ...         ],
    ...     },
    ... ]
    >>> result = pd.json_normalize(
    ...     data, "counties", ["state", "shortname", ["info", "governor"]]
    ... )
    >>> result
             name  population    state shortname info.governor
    0        Dade       12345   Florida    FL    Rick Scott
    1     Broward       40000   Florida    FL    Rick Scott
    2  Palm Beach       60000   Florida    FL    Rick Scott
    3      Summit        1234   Ohio       OH    John Kasich
    4    Cuyahoga        1337   Ohio       OH    John Kasich

    >>> data = {"A": [1, 2]}
    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")
        Prefix.0
    0          1
    1          2

    Returns normalized data with columns prefixed with the given string.
    """

    def _pull_field(
        js: dict[str, Any], spec: list | str, extract_record: bool = False
    ) -> Scalar | Iterable:
        """Internal function to pull field"""
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return np.nan
            else:
                raise KeyError(
                    f"Key {e} not found. To replace missing values of {e} with "
                    f"np.nan, pass in errors='ignore'"
                ) from e

        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list:
        """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
        result = _pull_field(js, spec, extract_record=True)

        # GH 31507 GH 30145, GH 26284 if result is not list, raise TypeError if not
        # null, otherwise return an empty list
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(
                    f"{js} has non list value {result} for path {spec}. "
                    "Must be list or null."
                )
        return result

    if isinstance(data, list) and not data:
        return DataFrame()
    elif isinstance(data, dict):
        # A bit of a hackjob
        data = [data]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        data = list(data)
    else:
        raise NotImplementedError

    # check to see if a simple recursive function is possible to
    # improve performance (see #15621) but only for cases such
    # as pd.Dataframe(data) or pd.Dataframe(data, sep)
    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep))

    if record_path is None:
        if any([isinstance(x, dict) for x in y.values()] for y in data):
            # naive normalization, this is idempotent for flat records
            # and potentially will inflate the data considerably for
            # deeply nested structures:
            #  {VeryLong: { b: 1,c:2}} -> {VeryLong.b:1 ,VeryLong.c:@}
            #
            # TODO: handle record value which are lists, at least error
            #       reasonably
            data = nested_to_record(data, sep=sep, max_level=max_level)
        return DataFrame(data)
    elif not isinstance(record_path, list):
        record_path = [record_path]

    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]

    _meta = [m if isinstance(m, list) else [m] for m in meta]

    # Disastrously inefficient for now
    records: list = []
    lengths = []

    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    def _recursive_extract(data, path, seen_meta, level: int = 0) -> None:
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for val, key in zip(_meta, meta_keys):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])

                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                recs = _pull_records(obj, path[0])
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level)
                    if isinstance(r, dict)
                    else r
                    for r in recs
                ]

                # For repeating the metadata later
                lengths.append(len(recs))
                for val, key in zip(_meta, meta_keys):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                records.extend(recs)

    _recursive_extract(data, record_path, {}, level=0)

    result = DataFrame(records)

    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")

    # Data types, a problem
    for k, v in meta_vals.items():
        if meta_prefix is not None:
            k = meta_prefix + k

        if k in result:
            raise ValueError(
                f"Conflicting metadata name {k}, need distinguishing prefix "
            )
        # GH 37782

        values = np.array(v, dtype=object)

        if values.ndim > 1:
            # GH 37782
            values = np.empty((len(v),), dtype=object)
            for i, v in enumerate(v):
                values[i] = v

        result[k] = values.repeat(lengths)
    return result
<|endoftext|>from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    Sequence,
)
import csv
from io import StringIO
import re
from typing import (
    IO,
    TYPE_CHECKING,
    DefaultDict,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_numeric_dtype,
)
from pandas.core.dtypes.inference import is_dict_like

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    parser_defaults,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        ReadCsvBuffer,
        Scalar,
    )

    from pandas import (
        Index,
        MultiIndex,
    )

# BOM character (byte order mark)
# This exists at the beginning of a file to indicate endianness
# of a file (stream). Unfortunately, this marker screws up parsing,
# so we need to remove it if we see it.
_BOM = "\ufeff"


class PythonParser(ParserBase):
    _no_thousands_columns: set[int]

    def __init__(self, f: ReadCsvBuffer[str] | list, **kwds) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
        super().__init__(kwds)

        self.data: Iterator[str] | None = None
        self.buf: list = []
        self.pos = 0
        self.line_pos = 0

        self.skiprows = kwds["skiprows"]

        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows

        self.skipfooter = _validate_skipfooter_arg(kwds["skipfooter"])
        self.delimiter = kwds["delimiter"]

        self.quotechar = kwds["quotechar"]
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)

        self.escapechar = kwds["escapechar"]
        self.doublequote = kwds["doublequote"]
        self.skipinitialspace = kwds["skipinitialspace"]
        self.lineterminator = kwds["lineterminator"]
        self.quoting = kwds["quoting"]
        self.skip_blank_lines = kwds["skip_blank_lines"]

        self.has_index_names = False
        if "has_index_names" in kwds:
            self.has_index_names = kwds["has_index_names"]

        self.verbose = kwds["verbose"]

        self.thousands = kwds["thousands"]
        self.decimal = kwds["decimal"]

        self.comment = kwds["comment"]

        # Set self.data to something that can read lines.
        if isinstance(f, list):
            # read_excel: f is a list
            self.data = cast(Iterator[str], f)
        else:
            assert hasattr(f, "readline")
            self.data = self._make_reader(f)

        # Get columns in two steps: infer from data, then
        # infer column indices from self.usecols if it is specified.
        self._col_indices: list[int] | None = None
        columns: list[list[Scalar | None]]
        (
            columns,
            self.num_original_columns,
            self.unnamed_cols,
        ) = self._infer_columns()

        # Now self.columns has the set of columns that we will process.
        # The original set is stored in self.original_columns.
        # error: Cannot determine type of 'index_names'
        (
            self.columns,
            self.index_names,
            self.col_names,
            _,
        ) = self._extract_multi_indexer_columns(
            columns,
            self.index_names,  # type: ignore[has-type]
        )

        # get popped off for index
        self.orig_names: list[Hashable] = list(self.columns)

        # needs to be cleaned/refactored
        # multiple date column thing turning into a real spaghetti factory

        if not self._has_complex_date_col:
            (index_names, self.orig_names, self.columns) = self._get_index_name()
            self._name_processed = True
            if self.index_names is None:
                self.index_names = index_names

        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))

        self._parse_date_cols = self._validate_parse_dates_presence(self.columns)
        self._no_thousands_columns = self._set_no_thousand_columns()

        if len(self.decimal) != 1:
            raise ValueError("Only length-1 decimal markers supported")

    @cache_readonly
    def num(self) -> re.Pattern:
        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = rf"^[\-\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\-?[0-9]+)?$"
        else:
            thousands = re.escape(self.thousands)
            regex = (
                rf"^[\-\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?"
                rf"([0-9]?(E|e)\-?[0-9]+)?$"
            )
        return re.compile(regex)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]):
        sep = self.delimiter

        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError(
                    "Custom line terminators not supported in python parser (yet)"
                )

            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = "\n"

            dia = MyDialect

            if sep is not None:
                dia.delimiter = sep
            else:
                # attempt to sniff the delimiter from the first valid line,
                # i.e. no comment line and not in skiprows
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                lines_str = cast(list[str], lines)

                # since `line` was a string, lines will be a list containing
                # only a single string
                line = lines_str[0]

                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter

                # Note: encoding is irrelevant here
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))

            # Note: encoding is irrelevant here
            reader = csv.reader(f, dialect=dia, strict=True)

        else:

            def _read():
                line = f.readline()
                pat = re.compile(sep)

                yield pat.split(line.strip())

                for line in f:
                    yield pat.split(line.strip())

            reader = _read()

        return reader

    def read(
        self, rows: int | None = None
    ) -> tuple[
        Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]
    ]:
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise

        # done with first read, next time raise StopIteration
        self._first_chunk = False

        columns: Sequence[Hashable] = list(self.orig_names)
        if not len(content):  # pragma: no cover
            # DataFrame with the right metadata, even though it's length 0
            # error: Cannot determine type of 'index_col'
            names = dedup_names(
                self.orig_names,
                is_potential_multi_index(
                    self.orig_names,
                    self.index_col,  # type: ignore[has-type]
                ),
            )
            index, columns, col_dict = self._get_empty_meta(
                names,
                self.dtype,
            )
            conv_columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return index, conv_columns, col_dict

        # handle new style for names in index
        count_empty_content_vals = count_empty_vals(content[0])
        indexnamerow = None
        if self.has_index_names and count_empty_content_vals == len(columns):
            indexnamerow = content[0]
            content = content[1:]

        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)

        conv_data = self._convert_data(data)
        columns, conv_data = self._do_date_conversions(columns, conv_data)

        index, result_columns = self._make_index(
            conv_data, alldata, columns, indexnamerow
        )

        return index, result_columns, conv_data

    def _exclude_implicit_index(
        self,
        alldata: list[np.ndarray],
    ) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]:
        # error: Cannot determine type of 'index_col'
        names = dedup_names(
            self.orig_names,
            is_potential_multi_index(
                self.orig_names,
                self.index_col,  # type: ignore[has-type]
            ),
        )

        offset = 0
        if self._implicit_index:
            # error: Cannot determine type of 'index_col'
            offset = len(self.index_col)  # type: ignore[has-type]

        len_alldata = len(alldata)
        self._check_data_length(names, alldata)

        return {
            name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata
        }, names

    # legacy
    def get_chunk(
        self, size: int | None = None
    ) -> tuple[
        Index | None, Sequence[Hashable] | MultiIndex, Mapping[Hashable, ArrayLike]
    ]:
        if size is None:
            # error: "PythonParser" has no attribute "chunksize"
            size = self.chunksize  # type: ignore[attr-defined]
        return self.read(rows=size)

    def _convert_data(
        self,
        data: Mapping[Hashable, np.ndarray],
    ) -> Mapping[Hashable, ArrayLike]:
        # apply converters
        clean_conv = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)

        # Apply NA values.
        clean_na_values = {}
        clean_na_fvalues = {}

        if isinstance(self.na_values, dict):
            for col in self.na_values:
                na_value = self.na_values[col]
                na_fvalue = self.na_fvalues[col]

                if isinstance(col, int) and col not in self.orig_names:
                    col = self.orig_names[col]

                clean_na_values[col] = na_value
                clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues

        return self._convert_to_ndarrays(
            data,
            clean_na_values,
            clean_na_fvalues,
            self.verbose,
            clean_conv,
            clean_dtypes,
        )

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        if self.header is None:
            return False

        header = self.header
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(
        self,
    ) -> tuple[list[list[Scalar | None]], int, set[Scalar | None]]:
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols: set[Scalar | None] = set()

        if self.header is not None:
            header = self.header
            have_mi_columns = self._have_mi_columns

            if isinstance(header, (list, tuple, np.ndarray)):
                # we have a mi columns, so read an extra line
                if have_mi_columns:
                    header = list(header) + [header[-1] + 1]
            else:
                header = [header]

            columns: list[list[Scalar | None]] = []
            for level, hr in enumerate(header):
                try:
                    line = self._buffered_line()

                    while self.line_pos <= hr:
                        line = self._next_line()

                except StopIteration as err:
                    if 0 < self.line_pos <= hr and (
                        not have_mi_columns or hr != header[-1]
                    ):
                        # If no rows we want to raise a different message and if
                        # we have mi columns, the last line is not part of the header
                        joi = list(map(str, header[:-1] if have_mi_columns else header))
                        msg = f"[{','.join(joi)}], len of {len(joi)}, "
                        raise ValueError(
                            f"Passed header={msg}"
                            f"but only {self.line_pos} lines in file"
                        ) from err

                    # We have an empty file, so check
                    # if columns are provided. That will
                    # serve as the 'line' for parsing
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self._clear_buffer()
                        columns.append([None] * len(columns[-1]))
                        return columns, num_original_columns, unnamed_cols

                    if not self.names:
                        raise EmptyDataError("No columns to parse from file") from err

                    line = self.names[:]

                this_columns: list[Scalar | None] = []
                this_unnamed_cols = []

                for i, c in enumerate(line):
                    if c == "":
                        if have_mi_columns:
                            col_name = f"Unnamed: {i}_level_{level}"
                        else:
                            col_name = f"Unnamed: {i}"

                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)

                if not have_mi_columns:
                    counts: DefaultDict = defaultdict(int)
                    # Ensure that regular columns are used before unnamed ones
                    # to keep given names and mangle unnamed columns
                    col_loop_order = [
                        i
                        for i in range(len(this_columns))
                        if i not in this_unnamed_cols
                    ] + this_unnamed_cols

                    # TODO: Use pandas.io.common.dedup_names instead (see #50371)
                    for i in col_loop_order:
                        col = this_columns[i]
                        old_col = col
                        cur_count = counts[col]

                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col = f"{old_col}.{cur_count}"
                                if col in this_columns:
                                    cur_count += 1
                                else:
                                    cur_count = counts[col]

                            if (
                                self.dtype is not None
                                and is_dict_like(self.dtype)
                                and self.dtype.get(old_col) is not None
                                and self.dtype.get(col) is None
                            ):
                                self.dtype.update({col: self.dtype.get(old_col)})
                        this_columns[i] = col
                        counts[col] = cur_count + 1
                elif have_mi_columns:
                    # if we have grabbed an extra line, but its not in our
                    # format so save in the buffer, and create an blank extra
                    # line for the rest of the parsing code
                    if hr == header[-1]:
                        lc = len(this_columns)
                        # error: Cannot determine type of 'index_col'
                        sic = self.index_col  # type: ignore[has-type]
                        ic = len(sic) if sic is not None else 0
                        unnamed_count = len(this_unnamed_cols)

                        # if wrong number of blanks or no index, not our format
                        if (lc != unnamed_count and lc - ic > unnamed_count) or ic == 0:
                            clear_buffer = False
                            this_columns = [None] * lc
                            self.buf = [self.buf[-1]]

                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})

                if len(columns) == 1:
                    num_original_columns = len(this_columns)

            if clear_buffer:
                self._clear_buffer()

            first_line: list[Scalar] | None
            if names is not None:
                # Read first row after header to check if data are longer
                try:
                    first_line = self._next_line()
                except StopIteration:
                    first_line = None

                len_first_data_row = 0 if first_line is None else len(first_line)

                if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                    raise ValueError(
                        "Number of passed names did not match "
                        "number of header fields in the file"
                    )
                if len(columns) > 1:
                    raise TypeError("Cannot pass names with multi-index columns")

                if self.usecols is not None:
                    # Set _use_cols. We don't store columns because they are
                    # overwritten.
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns = len(names)
                if self._col_indices is not None and len(names) != len(
                    self._col_indices
                ):
                    columns = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns = [names]
            else:
                columns = self._handle_usecols(
                    columns, columns[0], num_original_columns
                )
        else:
            ncols = len(self._header_line)
            num_original_columns = ncols

            if not names:
                columns = [list(range(ncols))]
                columns = self._handle_usecols(columns, columns[0], ncols)
            elif self.usecols is None or len(names) >= ncols:
                columns = self._handle_usecols([names], names, ncols)
                num_original_columns = len(names)
            elif not callable(self.usecols) and len(names) != len(self.usecols):
                raise ValueError(
                    "Number of passed names did not match number of "
                    "header fields in the file"
                )
            else:
                # Ignore output but set used columns.
                columns = [names]
                self._handle_usecols(columns, columns[0], ncols)

        return columns, num_original_columns, unnamed_cols

    @cache_readonly
    def _header_line(self):
        # Store line for reuse in _get_index_name
        if self.header is not None:
            return None

        try:
            line = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError("No columns to parse from file") from err

            line = self.names[:]
        return line

    def _handle_usecols(
        self,
        columns: list[list[Scalar | None]],
        usecols_key: list[Scalar | None],
        num_original_columns: int,
    ) -> list[list[Scalar | None]]:
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
        col_indices: set[int] | list[int]
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices = self._evaluate_usecols(self.usecols, usecols_key)
            elif any(isinstance(u, str) for u in self.usecols):
                if len(columns) > 1:
                    raise ValueError(
                        "If using multiple headers, usecols must be integers."
                    )
                col_indices = []

                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols = [
                    col for col in self.usecols if col >= num_original_columns
                ]
                if missing_usecols:
                    raise ParserError(
                        "Defining usecols with out-of-bounds indices is not allowed. "
                        f"{missing_usecols} are out-of-bounds.",
                    )
                col_indices = self.usecols

            columns = [
                [n for i, n in enumerate(column) if i in col_indices]
                for column in columns
            ]
            self._col_indices = sorted(col_indices)
        return columns

    def _buffered_line(self) -> list[Scalar]:
        """
        Return a line from buffer, filling buffer if required.
        """
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        # first_row will be a list, so we need to check
        # that that list is not empty before proceeding.
        if not first_row:
            return first_row

        # The first element of this row is the one that could have the
        # BOM that we want to remove. Check that the first element is a
        # string before proceeding.
        if not isinstance(first_row[0], str):
            return first_row

        # Check that the string is not empty, as that would
        # obviously not have a BOM at the start of it.
        if not first_row[0]:
            return first_row

        # Since the string is non-empty, check that it does
        # in fact begin with a BOM.
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row

        first_row_bom = first_row[0]
        new_row: str

        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2

            # Extract the data between the quotation marks
            new_row = first_row_bom[start:end]

            # Extract any remaining data after the second
            # quotation mark.
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1 :]

        else:
            # No quotation so just remove BOM from first element
            new_row = first_row_bom[1:]

        new_row_list: list[Scalar] = [new_row]
        return new_row_list + first_row[1:]

    def _is_line_empty(self, line: list[Scalar]) -> bool:
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all(not x for x in line)

    def _next_line(self) -> list[Scalar]:
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                if self.pos >= len(self.data):
                    break
                self.pos += 1

            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    # either uncommented or blank to begin with
                    if not self.skip_blank_lines and (
                        self._is_line_empty(self.data[self.pos - 1]) or line
                    ):
                        break
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError:
                    raise StopIteration
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                # assert for mypy, data is Iterator[str] or None, would error in next
                assert self.data is not None
                next(self.data)

            while True:
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1

                if orig_line is not None:
                    line = self._check_comments([orig_line])[0]

                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])

                        if ret:
                            line = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break

        # This was the first line of the file,
        # which could contain the BOM at the
        # beginning of it.
        if self.pos == 1:
            line = self._check_for_bom(line)

        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg: str, row_num: int) -> None:
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        if self.on_bad_lines == self.BadLineHandleMethod.WARN:
            warnings.warn(
                f"Skipping line {row_num}: {msg}\n",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
        try:
            # assert for mypy, data is Iterator[str] or None, would error in next
            assert self.data is not None
            line = next(self.data)
            # for mypy
            assert isinstance(line, list)
            return line
        except csv.Error as e:
            if self.on_bad_lines in (
                self.BadLineHandleMethod.ERROR,
                self.BadLineHandleMethod.WARN,
            ):
                msg = str(e)

                if "NULL byte" in msg or "line contains NUL" in msg:
                    msg = (
                        "NULL byte detected. This byte "
                        "cannot be processed in Python's "
                        "native csv library at the moment, "
                        "so please pass in engine='c' instead"
                    )

                if self.skipfooter > 0:
                    reason = (
                        "Error could possibly be due to "
                        "parsing errors in the skipped footer rows "
                        "(the skipfooter keyword is only applied "
                        "after Python's csv library has parsed "
                        "all rows)."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if (
                    not isinstance(x, str)
                    or self.comment not in x
                    or x in self.na_values
                ):
                    rl.append(x)
                else:
                    x = x[: x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        """
        # Remove empty lines and lines with only one whitespace value
        ret = [
            line
            for line in lines
            if (
                len(line) > 1
                or len(line) == 1
                and (not isinstance(line[0], str) or line[0].strip())
            )
        ]
        return ret

    def _check_thousands(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.thousands is None:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.thousands, replace=""
        )

    def _search_replace_num_columns(
        self, lines: list[list[Scalar]], search: str, replace: str
    ) -> list[list[Scalar]]:
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                if (
                    not isinstance(x, str)
                    or search not in x
                    or i in self._no_thousands_columns
                    or not self.num.search(x.strip())
                ):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.decimal == parser_defaults["decimal"]:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.decimal, replace="."
        )

    def _clear_buffer(self) -> None:
        self.buf = []

    def _get_index_name(
        self,
    ) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
        columns: Sequence[Hashable] = self.orig_names
        orig_names = list(columns)
        columns = list(columns)

        line: list[Scalar] | None
        if self._header_line is not None:
            line = self._header_line
        else:
            try:
                line = self._next_line()
            except StopIteration:
                line = None

        next_line: list[Scalar] | None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None

        # implicitly index_col=0 b/c 1 fewer column names
        implicit_first_cols = 0
        if line is not None:
            # leave it 0, #2442
            # Case 1
            # error: Cannot determine type of 'index_col'
            index_col = self.index_col  # type: ignore[has-type]
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns

            # Case 0
            if (
                next_line is not None
                and self.header is not None
                and index_col is not False
            ):
                if len(next_line) == len(line) + self.num_original_columns:
                    # column and index names on diff rows
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]

                    for c in reversed(line):
                        columns.insert(0, c)

                    # Update list of original names to include all indices.
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return line, orig_names, columns

        if implicit_first_cols > 0:
            # Case 1
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))

            index_name = None

        else:
            # Case 2
            (index_name, _, self.index_col) = self._clean_index_names(
                columns, self.index_col
            )

        return index_name, orig_names, columns

    def _rows_to_cols(self, content: list[list[Scalar]]) -> list[np.ndarray]:
        col_len = self.num_original_columns

        if self._implicit_index:
            col_len += len(self.index_col)

        max_len = max(len(row) for row in content)

        # Check that there are no rows with too many
        # elements in their row (rows with too few
        # elements are padded with NaN).
        # error: Non-overlapping identity check (left operand type: "List[int]",
        # right operand type: "Literal[False]")
        if (
            max_len > col_len
            and self.index_col is not False  # type: ignore[comparison-overlap]
            and self.usecols is None
        ):
            footers = self.skipfooter if self.skipfooter else 0
            bad_lines = []

            iter_content = enumerate(content)
            content_len = len(content)
            content = []

            for i, _content in iter_content:
                actual_len = len(_content)

                if actual_len > col_len:
                    if callable(self.on_bad_lines):
                        new_l = self.on_bad_lines(_content)
                        if new_l is not None:
                            content.append(new_l)
                    elif self.on_bad_lines in (
                        self.BadLineHandleMethod.ERROR,
                        self.BadLineHandleMethod.WARN,
                    ):
                        row_num = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len))

                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(_content)

            for row_num, actual_len in bad_lines:
                msg = (
                    f"Expected {col_len} fields in line {row_num + 1}, saw "
                    f"{actual_len}"
                )
                if (
                    self.delimiter
                    and len(self.delimiter) > 1
                    and self.quoting != csv.QUOTE_NONE
                ):
                    # see gh-13374
                    reason = (
                        "Error could possibly be due to quotes being "
                        "ignored when a multi-char delimiter is used."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num + 1)

        # see gh-13320
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)

        if self.usecols:
            assert self._col_indices is not None
            col_indices = self._col_indices

            if self._implicit_index:
                zipped_content = [
                    a
                    for i, a in enumerate(zipped_content)
                    if (
                        i < len(self.index_col)
                        or i - len(self.index_col) in col_indices
                    )
                ]
            else:
                zipped_content = [
                    a for i, a in enumerate(zipped_content) if i in col_indices
                ]
        return zipped_content

    def _get_lines(self, rows: int | None = None) -> list[list[Scalar]]:
        lines = self.buf
        new_rows = None

        # already fetched some number
        if rows is not None:
            # we already have the lines in the buffer
            if len(self.buf) >= rows:
                new_rows, self.buf = self.buf[:rows], self.buf[rows:]

            # need some lines
            else:
                rows -= len(self.buf)

        if new_rows is None:
            if isinstance(self.data, list):
                if self.pos > len(self.data):
                    raise StopIteration
                if rows is None:
                    new_rows = self.data[self.pos :]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos : self.pos + rows]
                    new_pos = self.pos + rows

                new_rows = self._remove_skipped_rows(new_rows)
                lines.extend(new_rows)
                self.pos = new_pos

            else:
                new_rows = []
                try:
                    if rows is not None:
                        row_index = 0
                        row_ct = 0
                        offset = self.pos if self.pos is not None else 0
                        while row_ct < rows:
                            # assert for mypy, data is Iterator[str] or None, would
                            # error in next
                            assert self.data is not None
                            new_row = next(self.data)
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            new_rows.append(new_row)

                        len_new_rows = len(new_rows)
                        new_rows = self._remove_skipped_rows(new_rows)
                        lines.extend(new_rows)
                    else:
                        rows = 0

                        while True:
                            next_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1

                            if next_row is not None:
                                new_rows.append(next_row)
                        len_new_rows = len(new_rows)

                except StopIteration:
                    len_new_rows = len(new_rows)
                    new_rows = self._remove_skipped_rows(new_rows)
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len_new_rows

            self.buf = []
        else:
            lines = new_rows

        if self.skipfooter:
            lines = lines[: -self.skipfooter]

        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)

    def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]:
        if self.skiprows:
            return [
                row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)
            ]
        return new_rows

    def _set_no_thousand_columns(self) -> set[int]:
        no_thousands_columns: set[int] = set()
        if self.columns and self.parse_dates:
            assert self._col_indices is not None
            no_thousands_columns = self._set_noconvert_dtype_columns(
                self._col_indices, self.columns
            )
        if self.columns and self.dtype:
            assert self._col_indices is not None
            for i, col in zip(self._col_indices, self.columns):
                if not isinstance(self.dtype, dict) and not is_numeric_dtype(
                    self.dtype
                ):
                    no_thousands_columns.add(i)
                if (
                    isinstance(self.dtype, dict)
                    and col in self.dtype
                    and (
                        not is_numeric_dtype(self.dtype[col])
                        or is_bool_dtype(self.dtype[col])
                    )
                ):
                    no_thousands_columns.add(i)
        return no_thousands_columns


class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(
        self,
        f: IO[str] | ReadCsvBuffer[str],
        colspecs: list[tuple[int, int]] | Literal["infer"],
        delimiter: str | None,
        comment: str | None,
        skiprows: set[int] | None = None,
        infer_nrows: int = 100,
    ) -> None:
        self.f = f
        self.buffer: Iterator | None = None
        self.delimiter = "\r\n" + delimiter if delimiter else "\n\r\t "
        self.comment = comment
        if colspecs == "infer":
            self.colspecs = self.detect_colspecs(
                infer_nrows=infer_nrows, skiprows=skiprows
            )
        else:
            self.colspecs = colspecs

        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(
                "column specifications must be a list or tuple, "
                f"input was a {type(colspecs).__name__}"
            )

        for colspec in self.colspecs:
            if not (
                isinstance(colspec, (tuple, list))
                and len(colspec) == 2
                and isinstance(colspec[0], (int, np.integer, type(None)))
                and isinstance(colspec[1], (int, np.integer, type(None)))
            ):
                raise TypeError(
                    "Each column specification must be "
                    "2 element tuple or list of integers"
                )

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None = None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(
        self, infer_nrows: int = 100, skiprows: set[int] | None = None
    ) -> list[tuple[int, int]]:
        # Regex escape the delimiters
        delimiters = "".join([rf"\{x}" for x in self.delimiter])
        pattern = re.compile(f"([^{delimiters}]+)")
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError("No rows from which to infer column width")
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start() : m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where((mask ^ shifted) == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self) -> list[str]:
        # Argument 1 to "next" has incompatible type "Union[IO[str],
        # ReadCsvBuffer[str]]"; expected "SupportsNext[str]"
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)  # type: ignore[arg-type]
        else:
            line = next(self.f)  # type: ignore[arg-type]
        # Note: 'colspecs' is a sequence of half-open intervals.
        return [line[from_:to].strip(self.delimiter) for (from_, to) in self.colspecs]


class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None:
        # Support iterators, convert to a list.
        self.colspecs = kwds.pop("colspecs")
        self.infer_nrows = kwds.pop("infer_nrows")
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader:
        return FixedWidthReader(
            f,
            self.colspecs,
            self.delimiter,
            self.comment,
            self.skiprows,
            self.infer_nrows,
        )

    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [
            line
            for line in lines
            if any(not isinstance(e, str) or e.strip() for e in line)
        ]


def count_empty_vals(vals) -> int:
    return sum(1 for v in vals if v == "" or v is None)


def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError("skipfooter must be an integer")

    if skipfooter < 0:
        raise ValueError("skipfooter cannot be negative")

    # Incompatible return value type (got "Union[int, integer[Any]]", expected "int")
    return skipfooter  # type: ignore[return-value]
<|endoftext|>from __future__ import annotations

from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import isna

from pandas import (
    ArrowDtype,
    DataFrame,
    DatetimeIndex,
    StringDtype,
    concat,
)
from pandas.core import algorithms
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    BooleanArray,
    Categorical,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
)
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
    ensure_index_from_sequences,
)
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools

from pandas.io.common import is_potential_multi_index

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        DtypeObj,
        Scalar,
    )


class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    keep_date_col: bool
    usecols_dtype: str | None

    def __init__(self, kwds) -> None:
        self._implicit_index = False

        self.names = kwds.get("names")
        self.orig_names: Sequence[Hashable] | None = None

        self.index_col = kwds.get("index_col", None)
        self.unnamed_cols: set = set()
        self.index_names: Sequence[Hashable] | None = None
        self.col_names: Sequence[Hashable] | None = None

        self.parse_dates = _validate_parse_dates_arg(kwds.pop("parse_dates", False))
        self._parse_date_cols: Iterable = []
        self.date_parser = kwds.pop("date_parser", lib.no_default)
        self.date_format = kwds.pop("date_format", None)
        self.dayfirst = kwds.pop("dayfirst", False)
        self.keep_date_col = kwds.pop("keep_date_col", False)

        self.na_values = kwds.get("na_values")
        self.na_fvalues = kwds.get("na_fvalues")
        self.na_filter = kwds.get("na_filter", False)
        self.keep_default_na = kwds.get("keep_default_na", True)

        self.dtype = copy(kwds.get("dtype", None))
        self.converters = kwds.get("converters")
        self.dtype_backend = kwds.get("dtype_backend")

        self.true_values = kwds.get("true_values")
        self.false_values = kwds.get("false_values")
        self.cache_dates = kwds.pop("cache_dates", True)

        self._date_conv = _make_date_converter(
            date_parser=self.date_parser,
            date_format=self.date_format,
